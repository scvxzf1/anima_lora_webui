# 块交换基线测量（标准参考）

状态：基线参考 / 已固化
适用版本：当前 main
日期：2026-08-06

本文是块交换（block swap）优化工作的**基准参照基线**。后续所有改动（copy stream、slab、int8、prepare 同步）都应以本表为对照判断收益。测量脚本已固化进仓库，可在任意 GPU 上重跑复现。注：预取深度 K（方向 1）经实测撤回，见「方向落地状态」。

## 测量环境

| 项 | RTX 3080（主测） | CMP 90HX（对照） |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3080 | NVIDIA CMP 90HX |
| 显存 | 10 GB | 10 GB |
| PCIe | **Gen3 x16（约 10 GB/s 实测）** | Gen1 x16（约 0.8 GB/s 实测） |
| 角色 | 本次优化目标卡 | 极端低带宽对照 |

复现：

```bash
# 综合「计算 vs 传输」探针（主指标）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_baseline_probe.py

# 分项微基准
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/fp8_blockswap_h2d_microbench.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_copy_plan_microbench.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_restore_path_microbench.py
```

原始产物：`/tmp/anima-blockswap-baseline/{baseline_probe,h2d_microbench,copy_plan,restore_path}_rtx3080.json`。

## 核心基线数据（RTX 3080）

单个 DiT block 的 frozen 权重约 **132 MiB**（20 个 tensor）。前向计算与传输的相对量级决定了预取策略是否有效：

| 指标 | RTX 3080 | CMP 90HX |
| --- | ---: | ---: |
| 单块前向计算（seq=4096，无 checkpoint） | **11.8 ms** | — |
| 单块 H2D bf16（132 MiB） | **13.4 ms** | 170.8 ms |
| 单块 H2D int8/fp8（66 MiB） | **6.7 ms** | 85.5 ms |
| `overlap_ratio` = 计算 / bf16 传输 | **0.878** | — |
| `overlap_ratio` = 计算 / int8 传输 | **1.755** | — |

**`overlap_ratio < 1` 表示传输超过单块计算、藏不住；`>= 1` 表示可被单块计算隐藏。**

## 拷贝路径对比（RTX 3080，132 MiB / 块）

来自 `blockswap_copy_plan_microbench.py`（纯 DMA）与 `blockswap_restore_path_microbench.py`（含 host issue 成本）：

| 路径 | 纯 DMA p50 | host_issue p50 | ready p50 |
| --- | ---: | ---: | ---: |
| loop_copy（逐 tensor） | 14.12 ms | 0.381 ms | 14.19 ms |
| foreach_copy（当前默认） | 14.26 ms | 0.346 ms | 14.23 ms |
| slab_copy（整段一次） | 13.70 ms | 0.163 ms | 13.51 ms |

- **slab 比 foreach 快约 0.7 ms / 块（约 5%）**，且 host issue 减半。在高带宽卡上 slab 有温和但真实的收益。
- 在 CMP 90HX（Gen1）上三者几乎相同（均约 170ms）：拷贝已被带宽饱和，小拷贝次数无关紧要。

## 关键结论（驱动后续优化方向）

1. **3080 上 bf16 传输（13.4ms）略大于单块计算（11.8ms）**：领先量 K=1 的预取**藏不住传输**，每块约空转 2ms。~~方向 1（预取深度 K）~~ **经实测撤回**：K≥2 与该设计根本冲突（见「方向落地状态」方向1），不能用加深 lead 解决；该缺口改由方向 3（slab）与方向 4（int8 减传输量）弥补。
2. **int8 传输（6.7ms）远小于单块计算（11.8ms）**：压缩后 `overlap_ratio=1.755`，传输可被轻松隐藏。→ **方向 4（int8）在 3080 上传输侧有收益**（但端到端被 restore 调度抵消，见下「方向 4 验证结论」）。
3. **host issue 成本极小（<0.4ms）**：瓶颈在 PCIe 传输与调度窗口，不在 Python 侧。方向 2（多 copy stream）原本为配合方向 1 的多级流水线；方向 1 撤回后，K=1 每步只有一个 H2D job 在飞，per-slot copy stream 不再构成并行收益，仅保留为无害的实现细节。
4. **slab（方向 3）** 在 3080 上每块省约 0.7ms，叠加 12 块约 8ms/step，值得在无非 int8 路径默认化。
5. CMP 90HX 这类 Gen1 卡上，唯一有意义的手段是**减少传输量（int8）与尽量重叠**，拷贝路径优化无意义。

## 方向落地状态（2026-08-06）

| 方向 | 状态 | 落点 | 说明 |
| --- | --- | --- | --- |
| 1 预取深度 K | **已撤回（钳 1）** | `library/runtime/offloading.py::submit_move_blocks`、`library/runtime/block_swap_config.py::_block_swap_prefetch_depth` | K≥2 经实测是真实训练回归（fwd+bwd 必崩 `mat2 is on cpu`），且与该设计根本冲突（见下「`ANIMA_BLOCK_SWAP_PREFETCH_DEPTH`」）。训练/推理 prefetch lead 统一钳 1；env 仅作兼容旋钮（`>1` 被忽略）。 |
| 2 多 copy stream | **已落地（作用收窄）** | `library/runtime/offloading.py::_get_copy_stream_for_slot` | 每个 swap slot 一条 copy stream。原为配合方向 1 让多个 H2D 并行；方向 1 撤回后 K=1 每步单 job 在飞，不再构成并行收益，仅保留为无害实现细节。 |
| 3 slab 默认 | **已落地** | `library/training/cli_args.py`、`configs/base.toml`、WebUI `defaults.js`/`app-constants.js` | `--block_swap_restore_mode` 默认 `slab`；int8 或混合 dtype 时自动回退 foreach/copy。 |
| 4 int8 端到端 | **已验证，不建议默认开** | 见下「方向 4 验证结论」 | 传输侧收益真实，但 restore 调度/显存开销抵消，端到端不占优。 |
| 5 prepare 去 empty_cache | **已落地** | `library/training/unet_prepare.py`（`free_cache=False`） | 去掉每步 prepare 末尾的 `empty_cache`，消除 5060 Ti 上每步 ~1GB 的显存摆动。 |

### 组合叠加矩阵（真实 Block×28 + ModelOffloader，RTX 3080）

`scripts/experiments/blockswap_combo_ab_probe.py`，blocks_to_swap=12、seq=4096、checkpoint 开，每模式独立子进程测 fwd+bwd step time。它同时是方向 1 回归的复现与修复验证。

**修复前（方向 1 落地 K=2 默认时）**：所有 K=2 模式 fwd+bwd 必崩 `mat2 is on cpu`——

| 模式 | K | restore | dtype | step_ms | 结果 |
| --- | --- | --- | --- | ---: | --- |
| base | 1 | foreach | bf16 | 1451.15 | OK |
| k2 | 2 | foreach | bf16 | — | **ERROR mat2 is on cpu** |
| slab | 1 | slab | bf16 | 1419.02 | OK（1.023×） |
| k2slab | 2 | slab | bf16 | — | **ERROR** |
| int8 | 1 | foreach | int8 | 1354.09 | OK（1.072×） |
| int8_k2 | 2 | foreach | int8 | — | **ERROR** |
| all | 2 | slab | int8 | — | **ERROR** |

**修复后（K 钳 1，提交 `ac13590b`）**：7 模式全过、零崩溃；K=2 模式退化为 K=1，step 与对应 K=1 模式一致（run 间噪声内）——

| 模式 | K（请求） | restore | dtype | step_ms | speedup vs base |
| --- | --- | --- | --- | ---: | ---: |
| base | 1 | foreach | bf16 | 1408.41 | 1.000× |
| k2 | 2→钳1 | foreach | bf16 | 1402.27 | 1.004×（≈base，证明钳制生效） |
| slab | 1 | slab | bf16 | 1407.32 | 1.001× |
| k2slab | 2→钳1 | slab | bf16 | 1403.75 | 1.003×（≈slab） |
| int8 | 1 | foreach | int8 | 1348.12 | 1.045× |
| int8_k2 | 2→钳1 | foreach | int8 | 1339.76 | 1.051×（≈int8） |
| all | 2→钳1 | slab→foreach | int8 | 1343.69 | 1.048×（int8 时 slab 自动回退） |

**读法**：① 所有 K=2 请求被钳到 1，不再崩且与同 restore/dtype 的 K=1 模式几乎同速——钳制无回归。② K=1 各模式间差异（slab 1.001×、int8 1.045×）在合成矩阵的 run 间噪声内：这里传输（13.4ms/块）已被单块 fwd+bwd+checkpoint（≈50ms）完全覆盖，叠加增益本就接近噪声。③ int8 仍一致略快（传输量减半），但端到端 audit（wait/显存）结论不变——不建议默认开。④ 叠加矩阵结论：方向 3×4 互斥（int8 禁 slab），方向 5 与一切正交，方向 1/2 已撤回。

### `ANIMA_BLOCK_SWAP_PREFETCH_DEPTH`

**已停用（钳 1），仅作兼容旋钮。** 2026-08-07 复核发现，K≥2 的预取深度与块交换的核心设计**根本冲突**，不是边界条件：

- **冲突根源**：这套机制不做 D2H 拷贝，resident GPU slot 数恒等于 `num_blocks - blocks_to_swap`，且每个退役块的 storage 恰好是「`to_cuda = num_blocks - blocks_to_swap + to_cpu`」这一块传入块的 H2D 目标。预取想提前，就只能「提前 retire 未来的块」——但未来块还没跑、不能 retire。
- **实测崩溃**：K=2 时 `submit_move_blocks` 的 `step=1` job 把「尚未运行的块 `block_idx+1`」当退役块，在 worker 线程把它 park 到 CPU（`offloading.py:1210`/`:1232`），主线程紧接着对它跑 forward → `RuntimeError: mat2 is on cpu`。且 `step=1` 与下一步的 `step=0` 还指向同一个 `to_cuda`，同 slot 双 job 竞争。真实训练路径（`_run_blocks` + backward hook + `loss.backward()`）与合成探针驱动序列一致，**fwd+bwd 必崩，K=1 正常**（均已实测）。
- **先前注记有误**：本节曾写「真实 Anima 模型不触发、仅 toy 模型可复现」，经真实 `Block×28 + ModelOffloader` 组合探针证伪——**真实训练稳定复现**。据此撤回。
- **为何不回退到「K≥2 + 独立 staging buffer」**：正确做法是给超前 job 配独立 GPU 暂存 buffer，代价是 +1 块常驻显存，恰好抵消 block swap 省显存的目的，不划算。

**结论**：`submit_move_blocks` 已把训练/推理的 prefetch lead 统一钳到 `1`；`ANIMA_BLOCK_SWAP_PREFETCH_DEPTH` 保留仅为兼容旧 env 设置（`>1` 被忽略，默认 `1`）。3080 上 K=1 的 ~2ms/块传输缺口由方向 3（slab）与方向 4（int8 减传输量）弥补，而非加深 lead。

### 方向 4 验证结论（int8 端到端）

数值等价与 H2D 收益均已在 RTX 3080 上验证（`scripts/experiments/int8_blockswap_h2d_probe.py`，本仓固化；端到端 loss/grad 见 `docs/findings/anima_int8_base_linear_audit.md`）：

- **数值等价 PASS**：int8 为对称 per-row 量化（`scale=amax/127`，反量化 `q*scale`）。copy/direct_bind 两模式反量化**逐位一致**；reuse_storage 因在 bf16 下乘 scale 引入 ~ULP 级舍入，但总相对误差与 copy 同量级（reuse 0.00915 ≈ copy 0.00899）。端到端三种模式 loss delta ≤0.059%、adapter grad delta ≤0.06%，全 PASS。
- **传输侧收益真实**：纯传输+反量化下 int8 为 bf16 的 0.501× 字节、约 1.56× 快（本探针），overlap_ratio 由 1.82 → 2.84。
- **但端到端不建议默认开**：audit 文档（dim=1024 大矩阵、三模式）显示 int8 的端到端 H2D 仅 bf16 的 0.71~0.88×，且 **wait 时间全面变差（4~15×）**、**peak reserved 显存更高（1.19~1.64×）**。瓶颈在 restore 调度、反量化临时 tensor 与额外 GPU copy，不在量化本身。
- **取舍**：显存极度紧张、追求最小传输量时可手动开（`--block_swap_transfer_dtype int8`）；追求 step time 时保持默认 bf16+slab。int8 scope 可用 `--block_swap_int8_scope`（如 `mlp`）收窄只量化 MLP。`reuse_storage` 的 `chunk_rows` 保持默认 0（分行使 wait 更差）。

复现：

```bash
# 传输+反量化+三模式数值等价（本仓固化探针）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/int8_blockswap_h2d_probe.py

# 端到端 loss/grad 等价（真实 Anima 结构，合成权重）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/int8_linear_equivalence_probe.py --model-kind anima --device cuda:0 --scope all
```

## 验收口径（供后续改动对照）

任一方向的改动合入前，应在 RTX 3080 上重跑 `blockswap_baseline_probe.py` 与相关微基准，并满足：

- 数值等价：`tests/test_compile_checkpoint_block_swap_hot.py` 全绿。
- `overlap_ratio_bf16` 改善，或 `gpu_wait_ms` / `forward_wait` p95 在 profile 中下降。
- step time 在 `blocks_to_swap=12`（balanced_16g 档）有可测收益。
