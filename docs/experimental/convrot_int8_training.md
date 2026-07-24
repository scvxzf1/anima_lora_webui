# ConvRot int8 训练探索（W8A16 / W8A8）

状态：实验 / **可运行（默认关闭）**
适用版本：当前 main（2026-07-24 起接入）
日期：2026-07-24（研究） / 2026-07-24（M1–M5 实现接入）
入口命令：

```bash
# 单元测试
timeout 60 .venv/bin/python -m pytest tests/test_convrot_*.py -q

# 多 seed toy probe（CI 友好）
.venv/bin/python scripts/experiments/convrot_equivalence_probe.py --mode w8a16 --seeds 0,1,2,3,4

# 训练（显式开启；默认 bf16 不变）
python tasks.py lora ... --base_compute w8a16_convrot
# 可选：--convrot_group_size 256 --convrot_scope mlp
# W8A8（真 int8 GEMM 优先，形状不支持时回退 float）：
python tasks.py lora ... --base_compute w8a8_convrot
# 可选：
#   ANIMA_CONVROT_INT8_GEMM=auto|int_mm|float
#   ANIMA_CONVROT_FUSED=1|0              # 默认 1：单 autograd.Function 融合 RHT+GEMM
#   ANIMA_CONVROT_RHT=dense|fwht         # 默认 dense（本机 3080 更快；fwht 仅 sylvester）
#   ANIMA_CONVROT_HADAMARD=sylvester|regular  # 默认 sylvester；regular=论文 Kronecker H_4^k
#   ANIMA_CONVROT_W8A16_KERNEL=dequant|int8pack  # 默认 dequant（int8pack 本机更慢）

# P0-A step profiler（bf16 / w8a16 / w8a8 单 step 饼图）
.venv/bin/python scripts/experiments/convrot_step_profile_probe.py \
  --cases bf16,w8a16_free,w8a8_auto \
  --json-out output/tests/convrot_step_profile.json

# P0-C：导出 / 加载 prequant 权重（去掉 apply 期 online weight RHT+quant）
.venv/bin/python scripts/experiments/convrot_export_prequant.py \
  --scope mlp --group-size 256 \
  --out output/tests/convrot_prequant_mlp_g256.safetensors
python tasks.py lora ... --base_compute w8a16_convrot \
  --convrot_weight_source prequant_checkpoint \
  --convrot_prequant_path output/tests/convrot_prequant_mlp_g256.safetensors
```

相关代码：

- `library/runtime/convrot/`（RHT / quant / W8A16 / W8A8 / apply / free_base / gemm / fused / prequant / checks / metadata）
- `library/training/bootstrap.py::maybe_apply_convrot_base`（compile 前薄钩子）
- `library/training/cli_args.py`（`base_compute` / `convrot_*`）
- 探针：`scripts/experiments/convrot_{equivalence,checkpoint,short_train,mem_speed,step_profile,export_prequant}_probe.py`（export 脚本名见上）
- 非 ConvRot 对照：`library/runtime/int8_linear.py`、`block_swap_transfer_dtype=int8`

相关 findings：[`../findings/anima_int8_base_linear_audit.md`](../findings/anima_int8_base_linear_audit.md)、[`../methods/channel_scaling.md`](../methods/channel_scaling.md)、`bench/channel_stats/`
落地规格：[`../proposal/convrot_w8a_training_plan.md`](../proposal/convrot_w8a_training_plan.md)
后续优化：[`../proposal/convrot_w8a_optimization_roadmap.md`](../proposal/convrot_w8a_optimization_roadmap.md)

> **一句话：** ConvRot 训练路径已按战略 C 自建接入；默认 `base_compute=bf16`。
> **不要**把 `int8_linear` 裸 rowwise 存储或 block-swap int8 传输当成 ConvRot。
> **不要**与 `--block_swap_transfer_dtype int8` 同时开启。

---

## 实现状态（2026-07-24）

| 里程碑 | 状态 | 说明 |
| --- | --- | --- |
| M1 算法核 | ✅ | `rht` / `quant` / `linear_w8a16` + unit tests |
| M2 训练 apply | ✅ | `apply_convrot_to_lora_network` + bootstrap 薄钩 + CLI + 与 block-swap int8 互斥 |
| M3 质量 gate | ✅ | toy + full-checkpoint + 20-step short-train（W8A16/W8A8 均已跑） |
| M4 W8A8 | ✅ | 真 int8 GEMM（`torch._int_mm`，形状不支持时 float 回退）+ STE |
| M5 产品化钩子 | ✅ 最小集 + WebUI MVP | metadata stamp、merge 拒绝、DoRA 拒绝、skip 日志；WebUI：`base_compute` / `convrot_group_size` / `convrot_scope=mlp`（配置页 → 显存与速度优化） |
| 显存优化 | ✅ | quant 后默认释放 base `Linear.weight`（meta 占位，去掉双份 bf16） |
| 融合路径 | ✅ | 单 `autograd.Function` 融合 RHT+quant+GEMM；默认 **dense RHT + dequant**（见 §F） |
| P0-A step profile | ✅ | `torch.profiler` 三 case 饼图（见 §G）；**不**满足 Triton 门槛 |
| P0-A2 bf16 计算 | ✅ | W8A16 dequant linear 用 bf16 TC（见 §G.4） |
| P0-C prequant 加载 | ✅ | 原生 `anima_lora_convrot_prequant_v1` + `weight_scale` 别名（见 §H） |
| P0-D regular Hadamard | ✅ | Kronecker `H_{4^k}` 实现 + multi-seed 对照（见 §I）；**默认仍 sylvester** |

### 验证矩阵总览（本机，2026-07-24 / 07-25）

| 层级 | 脚本 | W8A16 | W8A8 |
| --- | --- | --- | --- |
| Unit / apply / free / gemm / fused / profile / prequant / rht | `tests/test_convrot_*.py` | PASS（66） | PASS |
| Toy multi-seed | `scripts/experiments/convrot_equivalence_probe.py` | PASS（5 seed） | PASS（3 seed） |
| Full DiT + LoRA 1-step | `scripts/experiments/convrot_checkpoint_probe.py` | 2/3 严格 gate | 2/3 严格 gate |
| 20-step short-train + sample | `scripts/experiments/convrot_short_train_probe.py` | loss rel ≪1% | loss rel ≪1% |
| Mem/speed（free-base 后） | `scripts/experiments/convrot_mem_speed_probe.py` | peak **低于** bf16 | peak **低于** bf16 |
| Step profile 饼图 | `scripts/experiments/convrot_step_profile_probe.py` | 1.77× 更慢 | 1.46× 更慢 |

共同设置（checkpoint / short-train）：

- DiT：`models/diffusion_models/anima-preview3-base.safetensors`
- 数据：`post_image_dataset/rokkotsu_goddess` cached latent/TE
- adapter：plain LoRA r=4 / α=4，`scope=mlp`，`group=256`，`patched=56`
- 严格 smoke：`out_rel≤3%`、`loss_rel≤5%`、`grad_rel≤5%`（仅 1-step probe）
- 产物目录：`output/tests/convrot_ckpt_*.json`、`output/tests/convrot_short_train/`

### A. Toy probe

```text
W8A16 group=16 seeds=0..4
  max_out_rel≈0.009  max_grad_rel≈0.007  → gates PASS
W8A8  group=16 seeds=0..2
  max_out_rel≈0.013  max_grad_rel≈0.009  → gates PASS
```

### B. Full checkpoint 1-step probe（bf16 LoRA vs ConvRot base）

JSON：`output/tests/convrot_ckpt_w8a16_seeds012.json`、`output/tests/convrot_ckpt_w8a8_seeds012.json`

| mode | seed0 out/loss/grad | seed1 | seed2 | 严格 gate | 汇总 |
| --- | --- | --- | --- | --- | --- |
| **W8A16** | 0.0062 / 0.0004 / **0.0594 FAIL** | 0.0049 / 0.0003 / 0.0010 PASS | 0.0155 / 0.0001 / 0.0126 PASS | **2/3** | out_max=0.0155；grad_max=0.0594；mean_grad≈0.024；peak≈6.2 GB |
| **W8A8** | 0.0077 / 0.0003 / **0.1290 FAIL** | 0.0063 / 0.0009 / 0.0005 PASS | 0.0215 / 0.0004 / 0.0034 PASS | **2/3** | out_max=0.0215；grad_max=0.1290；mean_grad≈0.044；peak≈6.7 GB |

结论：

- **输出 / loss**：两模式全 seed 都远低于阈值。
- **adapter grad**：多数 seed 稳定；seed0 对两种模式都是最差 seed。W8A16 略超 5%（~5.9%），W8A8 更松（~12.9%），符合「act quant 额外噪声」预期。
- **不发散**：失败 seed 仍是有限相对误差，不是 NaN / 量级爆炸。

### C. Short-train sample 对照（20 steps，seed=0）

JSON：`output/tests/convrot_short_train/short_train_w8a16_seed0.json`、`short_train_w8a8_seed0.json`
图：`sample_bf16_seed0.png`、`sample_w8a16_seed0.png`、`sample_w8a8_seed0.png`
并排：`sample_bf16_vs_w8a16_seed0.png`、`sample_bf16_vs_w8a8_seed0.png`、`sample_bf16_w8a16_w8a8_seed0.png`

| 路径 | first → last loss | last_loss rel vs bf16 | per-step mean rel | sample pixel rel vs bf16 | peak GB（优化前） |
| --- | --- | --- | --- | --- | --- |
| bf16 | 0.9520 → 0.1353 | — | — | — | ≈5.0 |
| **W8A16** | 0.9521 → 0.1355 | **≈0.09%** | ≈0.036% | ≈1.6%（MAE≈1.1/255） | ≈6.2（双份权重） |
| **W8A8** | 0.9523 → 0.1354 | **≈0.09%** | ≈0.060% | ≈2.2%（MAE≈1.6/255） | ≈6.7（双份权重） |

结论：20 步内两条 ConvRot 路径的 loss 轨迹与 bf16 几乎重合；sample 像素差很小。上表 peak 为 **free-base 优化前**；优化后 peak 见 §E.3。

### D. 总判定（对照 proposal §4.1）

| 标准 | W8A16 | W8A8 |
| --- | --- | --- |
| adapter grad 多 seed 稳定 | **大体通过**（1/3 seed 略超 5% smoke） | **可用但更松**（1/3 seed grad 12.9%） |
| 短训 sample 接近同超参 bf16 | **通过** | **通过** |
| 默认 bf16 零回归 | 保持（默认关闭） | 保持（默认关闭） |
| Phase 1 优先推荐 | **是** | 实验可用；真 int8 GEMM 已接 |

### E. 任务 1+2：释放 base 权重 + 真 int8 GEMM（2026-07-24 续）

#### E.1 释放 GPU 上的 bf16 base（任务 1）

实现：`library/runtime/convrot/free_base.py`，`apply_convrot_to_lora_network(..., free_base_weights=True)` 默认开启。

- quant 成功后把 patched 的 `nn.Linear.weight` 换成 **meta** 占位（形状/dtype 保留，**0 字节存储**）
- 仅替换 `org_forward` 的约定不变；误走原 `Linear.forward` 会因 meta 立刻失败
- dry_run / `free_base_weights=False` 不释放
- mlp scope 实测 **freed ≈ 1792 MiB**（56 个 Linear）

#### E.2 真 int8 GEMM（任务 2）

实现：`library/runtime/convrot/gemm.py` + `linear_w8a8.py`。

- 前向：act per-token absmax → int8；`int8×int8 → int32 → ×scales → float`
- CUDA 优先 `torch._int_mm`（`M>16` 且 K/N 合理）；否则 float 模拟
- 反传：STE（对 act quant 视为恒等；base 冻结无 W 梯度）
- 环境变量：`ANIMA_CONVROT_INT8_GEMM=auto|int_mm|float`（默认 `auto`）

#### E.3 Mem/speed 对照（free-base 后，8/6 step microbench）

脚本：`scripts/experiments/convrot_mem_speed_probe.py`
JSON：`output/tests/convrot_mem_speed.json`

| case | peak GB | alloc after apply GB | sec/step | meta bases | freed MiB |
| --- | --- | --- | --- | --- | --- |
| **bf16** | **5.00** | 3.96 | **1.84** | 0 | 0 |
| **w8a16_free** | **4.43** | 3.10 | 2.94 | 56 | 1792 |
| w8a16_nofree（对照） | 7.07 | 5.74 | 2.96 | 0 | 0 |
| **w8a8_auto**（int_mm） | **4.52** | 3.10 | 2.54 | 56 | 1792 |
| w8a8_float | 4.52 | 3.10 | 3.23 | 56 | 1792 |

解读：

1. **显存：** 释放 base 后 W8A16/W8A8 peak **低于** bf16（约 −0.5 GB）；不释放时 peak 反而到 ~7 GB（双份权重）。
2. **速度：** 端到端仍慢于 bf16（online RHT + quant 开销主导）。W8A8 `auto` 快于 `float`（约 2.54 vs 3.23 s/step），但仍慢于 bf16。
3. **内核微基准**（单次大 GEMM `64×2048 @ 8192`）：本机 3080 上 `_int_mm` 与 float 模拟同量级（甚至 float 略快）——端到端收益主要来自 **少一次完整 dequant 权重物化**，不是单 kernel 碾压 bf16 Tensor Core。
4. **质量：** 同 loss 量级；free-base 不改变 quant 数值路径。

### F. 任务 3：融合 RHT + quant + GEMM（2026-07-24 续）

实现：`library/runtime/convrot/fused.py` + `group_fwht`（`rht.py`）。

融合内容（训练安全、无 Triton 依赖）：

1. **单 `autograd.Function`**：W8A16 / W8A8 前向把 RHT →（act quant）→ GEMM 收成一次 forward/backward，避免 Python 层中间图。
2. **RHT 后端**（`ANIMA_CONVROT_RHT`）：
   - `dense`（**默认**）：缓存 `H/√G` + `matmul`（cuBLAS）
   - `fwht`：O(D log G) butterfly（与 dense Sylvester 数值等价）
3. **W8A16 权重 GEMM**（`ANIMA_CONVROT_W8A16_KERNEL`）：
   - `dequant`（**默认 / auto**）：`dequant(w_q) → F.linear`
   - `int8pack`：`torch._weight_int8pack_mm`（opt-in）
4. **W8A8**：RHT → absmax act quant → `int8_mm_scaled`（`_int_mm` / float 回退）
5. **开关**：`ANIMA_CONVROT_FUSED=1`（默认）/ `0` 回退非融合路径

#### F.1 内核微基准（RTX 3080，本机）

| 路径 | 相对结果（同 shape 量级） |
| --- | --- |
| dense RHT vs FWHT | dense **约 10× 更快**（例 FWHT ~6.3 ms vs dense ~0.6 ms） |
| dequant+`F.linear` vs `_weight_int8pack_mm` | dequant **显著更快**（例 ~3 ms vs ~97 ms @ 512×2048@8192） |

#### F.2 端到端 mem/speed（融合默认 vs 错误默认）

JSON：

- 推荐默认（fused + dense RHT + dequant）：`output/tests/convrot_mem_speed_fused_dense.json`
- free-base 初版对照：`output/tests/convrot_mem_speed.json`（§E.3）
- 曾误默认 FWHT/int8pack 的 fused 对照：`output/tests/convrot_mem_speed_fused.json`

| case | peak GB | sec/step | 备注 |
| --- | --- | --- | --- |
| bf16 | 5.00 | **1.70** | 基线 |
| w8a16 free + fused dense/dequant（**当前默认**） | **4.37** | **2.98** | 可用 |
| w8a8 free + fused auto int_mm（**当前默认**） | **4.49** | **2.55** | 可用 |
| w8a16 free + fused 旧默认（FWHT/int8pack） | 4.55 | **47.6** | 不可用；已废弃为默认 |
| w8a8 free + fused 旧默认 | 4.55 | 3.61 | 仍慢于 dense 路径 |

解读：

1. **融合接口已落地**，但「融合 ≠ 更快」。本机 3080 上 **dense RHT + dequant** 才是可用默认。
2. FWHT / `_weight_int8pack_mm` 保留为 **opt-in**（研究 / 其它 GPU 再 profile）。
3. 真正的吞吐跃迁仍需 **Triton 真融合 kernel** 或 prequant checkpoint 去掉 online rotate 热路径。
4. 数值：dense 与 FWHT 相对误差 <1e-6；fused vs legacy dequant 路径 <1e-4（unit tests）。

**尚未做：** 完整 ComfyUI MixedPrecisionOps INT8-ConvRot 布局自动适配（仅 best-effort 键名）；Triton 真融合 kernel；完整 `train.py` 长训；WebUI 高级项（prequant / Hadamard / kernel env）；Hydra 等变体 `org_module_ref` 全覆盖。
**WebUI MVP：** 配置页 → 优化 →「显存与速度优化」：`base_compute` / `convrot_group_size` / `convrot_scope=mlp`（默认 bf16）。

### G. P0-A：单 step CUDA profiler（2026-07-24）

脚本：`scripts/experiments/convrot_step_profile_probe.py`
JSON：`output/tests/convrot_step_profile.json`
环境：RTX 3080 / torch 2.12.0+cu130；默认 dense RHT + dequant fused；warmup=2；capture 1 train step + 3-step wall microbench。

#### G.1 端到端

| case | sec/step（无 profiler） | peak GB | CUDA self 合计 | vs bf16 wall |
| --- | --- | --- | --- | --- |
| bf16 | **1.73 s** | 4.99 | 3.91 s | 1.0× |
| w8a16_free | **3.06 s** | **4.37** | 7.86 s | **1.77×** |
| w8a8_auto | **2.52 s** | **4.49** | 6.48 s | **1.46×** |

#### G.2 CUDA self-time 粗桶占比（%）

| bucket | bf16 | w8a16_free | w8a8_auto |
| --- | --- | --- | --- |
| gemm_generic | 51.7 | 52.6 | 37.8 |
| convrot_gemm（标记） | — | **12.0** | **11.7** |
| convrot_rht | — | **1.4** | **1.7** |
| convrot_dequant | — | 0.6 | — |
| convrot_act_quant | — | — | 2.6 |
| attention | 10.0 | 5.4 | 6.3 |
| memcpy_cast | 6.0 | 6.1 | 11.0 |
| other | 30.4 | 21.0 | 27.6 |

显式标记事件（top）：

- W8A16：`convrot::gemm_dequant_linear` ≈ **941 ms**（单 step self）；RHT 仅 ~1.4%
- W8A8：`convrot::gemm_int8` / `_int_mm` 可见；act quant ~2.6%；RHT 仍小

说明：profiler 的 `self` 时间会把部分 matmul 记在 `aten::mm`（gemm_generic）下；`convrot::*` 是注入的 `record_function` 边界，用于归因 ConvRot 路径，不是与 wall 秒严格一一对应。

#### G.3 决策（路线图分支）

自动启发式（基于 w8a16_free）：

```text
branch = fix_w8a16_keep_bf16_compute
reason = W8A16 dequant linear ~12% 而 RHT 仅 ~1.4%：
         更像 fp32 F.linear 相对 bf16 Tensor Core 的税，而非 RHT 内存链
convrot_tax ≈ 14%  << 50%  →  **不** 开 P2-K Triton
wall_ratio ≈ 1.77×
```

解读：

1. **Triton 真融合门槛未满足**（ConvRot 链 <50% CUDA self）。
2. **W8A16 慢的主因** 更像 **dequant 后走 fp32 `F.linear`**（top kernel 出现 `ampere_sgemm_*` / cutlass sgemm），而 bf16 路径吃 `bf16` Tensor Core gemm。
3. **RHT 本身不是主税**（~1–2%）；继续优化 FWHT/融合 RHT ROI 低。
4. **下一刀建议（按 ROI）**：
   - **已做（P0-A2）**：W8A16 dequant/`F.linear` 改用 bf16/fp16 计算 dtype（见 §G.4）
   - **已做（P0-C）**：prequant 加载/导出（见 §H；去掉 apply 期 online weight quant）
   - 仍 **不要** 默认 int8pack / FWHT；Triton 门槛仍未达

#### G.4 P0-A2 落地：W8A16 保持 bf16 计算（同日续）

代码：`fused.py` / `linear_w8a16.py` — RHT 仍 float32；dequant linear 用 act 的 bf16/fp16（CUDA 上 fp32 输入回退 bf16 TC）。

Mem/speed 复测 JSON：`output/tests/convrot_mem_speed_bf16_compute.json`（6 steps）

| case | peak GB | sec/step | vs bf16 | 备注 |
| --- | --- | --- | --- | --- |
| bf16 | 5.00 | **1.62** | 1.0× | 基线 |
| w8a16_free（A2 后） | **4.34** | **1.99** | **1.23×** | 此前 ~3.0 s / 1.77× |
| w8a8_auto | 4.49 | 2.55 | 1.57× | 与 A2 前同量级 |

结论：A2 把 W8A16 端到端从 **~1.77× 慢收到 ~1.23×**，且 peak 仍低于 bf16。剩余差距主要来自 RHT + dequant 额外链与其它开销，**不是**「再写一个 Python fusion」能抹平；Triton 仍非当前最高 ROI。

### H. P0-C：prequant_checkpoint 加载（2026-07-25）

实现：`library/runtime/convrot/prequant.py` + `apply.py` 接线；导出：`scripts/experiments/convrot_export_prequant.py`。

#### H.1 原生格式 `anima_lora_convrot_prequant_v1`

| 项 | 内容 |
| --- | --- |
| 张量 | `{original_name}.weight` int8 `[out,in]`（**已在旋转域**） |
| scale | `{original_name}.scale` float32 `[out]`（per-out absmax） |
| metadata | `format=anima_lora_convrot_prequant_v1`、`group_size`、`rht=sylvester`、可选 `mode` |
| 别名 | 亦接受 `{name}.weight_scale`（Comfy 风格键名 best-effort） |
| 名称前缀 | 自动尝试 `model.` / `diffusion_model.` 去前缀/加前缀匹配 |

**明确不做：** 完整解析 Comfy `MixedPrecisionOps` / 任意社区 INT8-ConvRot 布局；不保证 obsxrver 权重开箱即用（键名兼容了，但 RHT 定义/group 必须一致）。

#### H.2 行为

- `--convrot_weight_source prequant_checkpoint --convrot_prequant_path <file>`
- 文件 `group_size` 与 CLI 不一致时 **strict raise**（可用 apply 参数 `prequant_group_size_strict=False` 警告并用文件值）
- 缺层 / shape 不符：该层 skip；若最终 patched=0 则 raise
- 仍默认 **free base**（meta）；merge 仍拒绝
- **只去掉 apply 期 online weight RHT+quant**；每步 **act RHT 仍在**

#### H.3 测试

| 测试 | 结果 |
| --- | --- |
| `tests/test_convrot_prequant.py` | save/load 往返、weight_scale 别名、apply 与 online 数值一致、free base、缺层/gs 错/缺 path |
| 全量 `tests/test_convrot_*.py` | **58 passed** |

#### H.4 用法

```bash
# 从本机 bf16 DiT 导出 mlp scope
.venv/bin/python scripts/experiments/convrot_export_prequant.py \
  --scope mlp --group-size 256 \
  --out output/tests/convrot_prequant_mlp_g256.safetensors

# 训练时加载（示例）
python tasks.py lora ... --base_compute w8a16_convrot \
  --convrot_weight_source prequant_checkpoint \
  --convrot_prequant_path output/tests/convrot_prequant_mlp_g256.safetensors \
  --convrot_group_size 256 --convrot_scope mlp

# mem/speed 对照（含 prequant cases）
.venv/bin/python scripts/experiments/convrot_mem_speed_probe.py \
  --steps 6 \
  --cases bf16,w8a16_free,w8a16_prequant,w8a8_auto,w8a8_prequant \
  --prequant-path output/tests/convrot_prequant_mlp_g256.safetensors \
  --json-out output/tests/convrot_mem_speed_prequant.json
```

#### H.5 Mem/speed 热测（2026-07-25，RTX 3080，6 steps）

JSON：`output/tests/convrot_mem_speed_prequant.json`
prequant 文件：`output/tests/convrot_prequant_mlp_g256.safetensors`（56 layers, g=256）

| case | peak GB | sec/step | apply wall | vs bf16 step | 备注 |
| --- | --- | --- | --- | --- | --- |
| bf16 | 5.00 | **1.65** | — | 1.0× | 基线 |
| w8a16_free（online） | **4.34** | **2.03** | 0.18 s | 1.23× | A2 后 |
| **w8a16_prequant** | **4.34** | **2.05** | 0.39 s | 1.24× | step ≈ online |
| w8a8_auto（online） | 4.49 | 2.49 | 0.11 s | 1.51× | |
| **w8a8_prequant** | 4.49 | 2.50 | 0.35 s | 1.51× | step ≈ online |

解读：

1. **稳态 step time：prequant ≈ online**（W8A16 2.05 vs 2.03；W8A8 2.50 vs 2.49）——符合「act RHT 仍在、权重 quant 只发生一次」的设计。
2. **apply 时间：prequant 更慢**（读 safetensors + 设备拷贝 ≈0.35–0.39 s vs online RHT+quant ≈0.11–0.18 s）；只影响启动，不影响每 step。
3. **peak 显存：与 free-base online 相同**（仍低于 bf16）。
4. **因此 P0-C 的 KPI 是「可复现加载 / 去掉 apply 期 quant / 与社区权重接口」**，不是 step 加速；当初验收以 unit 为主是因为路线图已写明「不承诺 step 大加速」。

### I. P0-D：regular Hadamard 质量对齐（2026-07-25）

实现：`library/runtime/convrot/rht.py`

| 构造 | 阶 | 列 discrepancy \(\|H^\top 1\|_\infty\) | 入口 |
| --- | --- | --- | --- |
| **sylvester**（默认） | \(2^k\) | \(=n\)（全 1 列） | `ANIMA_CONVROT_HADAMARD=sylvester` |
| **regular**（论文） | \(4^k\)（4/16/64/256/1024…） | \(=\sqrt{n}\)（最小） | `ANIMA_CONVROT_HADAMARD=regular` |

- regular：`H_4` 基 + Kronecker `H_{4^{k+1}}=H_{4^k}⊗H_4`（ConvRot Thm 3.3）
- FWHT **仅**等价 sylvester；regular 强制 dense matmul
- **默认不改**：仍 sylvester + group=256（兼容已有 prequant / 探针基线）

#### I.1 Checkpoint multi-seed（W8A16，seeds 0–2，strict grad≤5%）

JSON：`output/tests/convrot_ckpt_w8a16_{syl_g256,reg_g64,reg_g256,reg_g1024}.json`

| 配置 | seed0 out/loss/**grad** | seed1 | seed2 | 严格 pass | grad_max |
| --- | --- | --- | --- | --- | --- |
| sylvester g=256 | 0.0063 / 0.0009 / **0.0875 FAIL** | PASS | PASS | **2/3** | 0.0875 |
| **regular g=64** | 0.0062 / 0.0000 / **0.0475 PASS** | PASS | 0.0161 / 0.0002 / **0.0509 FAIL** | **2/3** | **0.0509** |
| regular g=256 | 0.0063 / 0.0001 / **0.0841 FAIL** | PASS | PASS | **2/3** | 0.0841 |
| regular g=1024 | 0.0065 / 0.0005 / **0.1487 FAIL** | PASS | PASS | **2/3** | 0.1487 |

W8A8 regular g=256：seed0 **grad_rel≈4.1**（异常大，seed1/2 正常）→ **不推荐** 默认切 regular@256 给 W8A8。

#### I.2 解读与默认策略

1. **输出 / loss**：所有配置全 seed 远低于阈值。
2. **seed0 grad** 仍是最难 seed；regular **不保证**全面碾压 sylvester。
3. **regular g=64** 在本矩阵里 **seed0 首次过 5%**，grad_max 也最低（~5.1%），但 seed2 略超；值得作为 **质量 opt-in**。
4. regular g 越大（256→1024）seed0 grad **变差**（本机）——与「更大旋转窗更好」的朴素预期相反，需更多 seed 才下强结论。
5. **默认保持 sylvester + g=256**；实验可：
   `ANIMA_CONVROT_HADAMARD=regular` + `--convrot_group_size 64`。
6. 若已导出 prequant，切换 hadamard/group **必须重导出**（旋转域不同）。

### Phase 1 支持面

- plain LoRA + `scope=mlp`（默认）
- `online_from_bf16` 权重量化
- `prequant_checkpoint` 加载（原生 v1 + weight_scale 别名）
- quant 后默认释放 base bf16 权重（meta）
- W8A8：真 int8 GEMM（可回退 float）
- 融合 autograd 路径（默认 dense RHT + dequant；FWHT/int8pack opt-in）
- Hadamard：`sylvester`（默认）/ `regular`（opt-in，阶须 \(4^k\)）
- step profiler 探针（P0-A）
- DoRA / 无 `org_module_ref` 变体：跳过或拒绝（禁止 0 patch 静默成功）
- merge / fuse：默认拒绝 ConvRot base

---

## 一句话结论

**ConvRot 是修正「DiT 上裸 rowwise int8 效果差」的正确方向**（group-wise Hadamard 压 outlier）。本仓已自建 W8A16 / W8A8；free-base 后显存可低于 bf16；A2 后 W8A16 ~1.23× bf16；prequant 不加速 step。**P0-D 已实现论文 regular Hadamard**；本机 multi-seed 下 **regular@64 改善 seed0 grad**，但默认仍 **sylvester@256** 以保兼容。Triton 门槛未达。

---
## 1. ConvRot 是什么

| 项 | 内容 |
| --- | --- |
| 论文 | [ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers](https://arxiv.org/abs/2512.03673)（Huang et al., arXiv:2512.03673, 2025-12） |
| 本质 | **Group-wise Regular Hadamard Transform (RHT)**；名称里的 Conv **不是**“卷积层量化” |
| 论文目标 | DiT **免重训 PTQ**；主推 **W4A4**；生态扩展出 INT8 ConvRot / W8A8 推理 |
| 运行时 | 权重：旋转后量化存盘；激活：在线旋转 + 动态量化再 GEMM；bias / norm / embedding 等常保留高精度 |
| 模块概念 | `ConvLinear4bit`：旋转 → 量化 → GEMM → 反量化，作为可替换 Linear |

复杂度宣称：全维旋转 \(\mathcal{O}(K^2)\) → group-wise \(\mathcal{O}(K)\)。组大小论文/社区常见 `64 / 256 / 1024`（须能整除层维度）。

### 与 rowwise int8 失败的关系

| 机制 | 问题 |
| --- | --- |
| 裸 rowwise / per-channel absmax | DiT 有强 row-wise + column-wise outlier；少数通道支配 scale，其余通道被压扁 |
| QuaRot 类 full FWHT | 论文指出 Sylvester/FWHT 含全 1 列，可能**放大** DiT row-wise outlier |
| AdaLN | 破坏 LLM 那套“旋转融进相邻 Linear、省 online rotate”的路径 |
| ConvRot group RHT | 组内旋转摊平 outlier；更适配 DiT 局部结构 |

本仓已有旁证：

- `bench/channel_stats/`：Anima frozen base 存在 20–100× DC-bias outlier（LLM.int8 类 outlier feature，不是 attention sink）。
- [`../methods/channel_scaling.md`](../methods/channel_scaling.md)：SmoothQuant 风格 **adapter 侧** 预缩放，**不是** base int8 GEMM。
- 社区 latent 基准（[ComfyUI-INT8-Fast Metrics](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/Metrics.md)，含 Anima）：

```text
GGUF Q8 > INT8 ConvRot > MXFP8 > FP8 ≥ INT8 Row > INT8 Tensorwise
```

社区定义：

> INT8 ConvRot = row-wise INT8，但在量化前对 weight/activation 做 ConvRot 旋转。

因此：**以前 int8 训练效果差，很大概率不是“int8 本身不行”，而是“裸 rowwise 未处理 DiT outlier”。**

---

## 2. W8A16 / W8A8 语义（训练目标）

| 模式 | Weight | Activation | 生态现状 | 训练含义 |
| --- | --- | --- | --- | --- |
| **W8A16** | int8（旋转后） | bf16 / fp16 | 几乎无人单独作为发布格式 | **第一期训练优先**：权重量化 + 高精度 act |
| **W8A8** | int8 | **动态** int8 | 推理有（Whole W8A8 / native int8 ConvRot） | 真 int8×int8 GEMM；act 每步旋转+量化 |
| **W4A4** | int4 packed | 动态 int4 | 论文主推 + ComfyUI 布局 | 推理向；训练更难 |

必须区分的三条路径：

| 路径 | 实际在算什么 | 是否等于目标 |
| --- | --- | --- |
| **QLoRA** | 存低比特 → **dequant 到 bf16 再 matmul**；通常不量化激活 | 结构可参考，**不是** W8A8 训练 GEMM |
| **本仓 `int8_linear` / block-swap int8** | int8 存储或 CPU master；执行时反量化回执行 dtype 再 `F.linear` / 高精度算子 | **存储/传输压缩**，不是 ConvRot，也不是 int8 GEMM 训练 |
| **真 ConvRot W8A8** | 存旋转 int8 权重；forward 对 act 做 RHT+quant，再 **int8 GEMM** | 目标形态 |

---

## 3. GEMM「融合」指什么

论文与社区说的 fusion 主要是 **推理算子链**：

```text
x (bf16)
  → group RHT
  → quant
  → int8 / int4 GEMM(W_q)
  → dequant
  → y
```

公开生态要点（时间敏感，约 2026-06～07）：

- ComfyUI **v0.27.0**：native **int8 ConvRot** 加载；含 int8 上 apply LoRA / requant 修复（**服务推理**，不是 `train.py`）。
- ComfyUI PR **#14859**：`convrot_w4a4` layout，`convrot_groupsize` 默认 256。
- Anima 推理加速示例：[ComfyUI-AnimaTurbo](https://github.com/TheLegendOfKitty/ComfyUI-AnimaTurbo)（shared rotated act cache、warp-FHT quantize、fuse QKV）。
- 转换工具方向：[`convert_to_quant`](https://github.com/silveroxides/convert_to_quant) 一类
  `--int8 --scaling_mode row --convrot --convrot-group-size {64,256,1024} --comfy_quant`。
- 公开 Anima 权重示例：[obsxrver/ComfyUI-Native-INT8_ConvRot](https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot) 中的 `anima-preview3-base-int8-ConvRot`。

**没有**公开的「冻结 base ConvRot + LoRA 训练」fused kernel。
训练第一期可用 **可微 / STE 的 fake path** 验数值；速度第二期再接 Triton / comfy-kitchen 类 kernel。

---

## 4. 相关方法对照

| 方法 | 训练？ | 机制 | 对 Anima 训练的启示 |
| --- | --- | --- | --- |
| 裸 rowwise int8 | 可做但 Anima 差 | 无旋转 | 已踩过的失败模式 |
| SmoothQuant | 多为 PTQ | scale 迁移 outlier | 本仓 channel_scaling 是 adapter 侧近亲 |
| LLM.int8 / bitsandbytes | 推理 / 优化器 | outlier 通道混精；AdamW8bit | 优化器已有；**不是** ConvRot |
| QLoRA | 训练 | 存储 4bit + dequant bf16 GEMM | 冻结 base + 高精度 adapter 结构可参考 |
| QuaRot | PTQ | full Hadamard | DiT row-wise 可能更差；Anima 基准弱于 ConvRot |
| SpinQuant | PTQ 学旋转 | Cayley SGD | 推理向；非本仓训练栈 |
| **ConvRot 论文** | **仅 PTQ** | group RHT | 算法源 |
| **Comfy INT8 ConvRot** | 推理 + LoRA apply | 原生 loader | 有 Anima 量化权重 |
| **Ostris ARA** | 实验训练 | 冻结 quant base + 小 LoRA 补误差；另有 QAT 向实验 | **最接近「训练」的公开方向** |
| HSWQ | PTQ | 敏感层保护 + 余下 ConvRot | 打包推理用 |

### 不要当既定事实写入设计的说法

对抗校验中被驳回或不可靠的表述（勿写入默认承诺）：

- 某社交帖对 FLUX Whole W8A8 的具体 VRAM 数字（如固定 16.30 GiB / −32.3%）
- HSWQ 营销向 SSIM / 体积数字（未过校验）
- SpinQuant「相对 QuaRot 45.1%」一类原句营销表述（未过校验）

---

## 5. 训练支持：已证实 vs 缺口

### 已证实（高置信）

1. ConvRot 是 DiT 向 rotation **PTQ**；主证据在推理 W4A4 / W8A8。
2. 运行时：旋转量化权重存盘 + 激活在线旋转动态量化。
3. Comfy 生态成熟于 **load / forward / serialize**，不是 training loop。
4. QLoRA ≠ 训练期 W8A8 GEMM。
5. Anima **推理** INT8 ConvRot 质量显著优于裸 INT8 Row（社区 latent 基准）。
6. Anima `model_channels=2048`、`mlp_ratio=4 → 8192`，**64 / 256 / 1024 均可整除**，group size 约束友好。

### 未证实 / 工程缺口

1. **没有主源**给出 ConvRot vs rowwise 在 **LoRA 训练 loss / 画质** 上的对照实验。
2. **W8A16 训练**不是生态标准选项，需本仓定义。
3. LoRA 与 **group 旋转边界**、**requant 后权重域** 在训练侧未标准化。
4. Anima 特有约束需单独验证：5D latent、max-padded TE sink、FEI / Hydra router、`compile_blocks` after `apply_to`。
5. AdaLN / modulation / final layer 应排除在 int8 候选外（论文与社区 arch filter 一致）。

### 与本仓现状对照

| 已有 | 缺 |
| --- | --- |
| rowwise / per-channel 存 int8（`library/runtime/int8_linear.py`） | group RHT |
| block-swap CPU master `transfer_dtype=int8`（实验） | 在线 act 旋转 + quant |
| AdamW8bit 等 bnb 优化器 | int8 GEMM 训练 wrapper |
| channel_scaling（adapter 侧） | W8A16 / W8A8 配置面、WebUI、metadata |
| int8 probe / gate 习惯 | 与 LoRA `org_forward` 正确域残差相加 |

---

## 6. 建议的产品语义（若落地）

不要把「加载 Comfy int8-ConvRot 权重做推理」和「训练期 W8A*」混成一个开关。

### 分期

**Phase 0 — 对齐（可选）**

- 用 `convert_to_quant` 或公开 `anima-*-int8-ConvRot` 做推理数值对齐。
- 确认 group size、layer 排除（AdaLN / final）与 Comfy 布局一致。

**Phase 1 — W8A16 训练（优先）**

- 冻结 base 候选 Linear：`RHT + int8 weight`。
- 激活保持 bf16 → W8A16 GEMM（或 dequant 权重 × bf16 act 作为数值 baseline）。
- LoRA 在 **原空间 / dequant 后** 残差相加：

```text
y = base_w8a16(x) + lora(x)
```

- STE 或 stop-grad 穿过 frozen quant 权重；adapter 正常反传。
- scope：先 `mlp`，再 attention projection；排除 AdaLN / modulation / final。

**Phase 2 — W8A8 训练**

- 同 group RHT 后对 act **动态 quant**。
- int8×int8 GEMM + dequant。
- 独立 gate：output L2、adapter grad、短训 sample。
- 默认 **off**，仅实验开关。

### 第一期明确不做

- 把现有 `Int8FrozenLinear` 贴牌成 ConvRot。
- 只改 `block_swap_transfer_dtype=int8` 当训练质量方案。
- 默认写入 `configs/base.toml`。
- 全层含 AdaLN 一起 int8。
- 把 Comfy 推理 quant 权重格式直接当训练图。

### 配置草图（尚未实现）

```toml
# 建议语义，当前 main 不存在这些键
base_compute = "bf16" | "w8a16_convrot" | "w8a8_convrot"  # 默认 bf16
convrot_group_size = 256   # 64 | 256 | 1024
convrot_scope = "mlp"      # mlp | attention | all（仍排除 AdaLN）
convrot_weight_source = "online_from_bf16" | "prequant_checkpoint"
```

### 集成触点（实现时）

1. `network.apply_to` + `load_weights` **之后**，`compile_blocks` **之前** patch base / `org_forward`。
2. 不可 `replace Linear` 破坏 LoRA monkey-patch 链（对齐现有 `patch_lora_frozen_base_forwards_with_int8` 的思路，但换成 ConvRot 算子）。
3. 保存：默认只存 adapter；base quant 状态写 metadata（例如 `ss_base_compute=...`）；merge 默认拒绝或需先 dequant。
4. 测试：RHT 正交/可逆、shape、STE 梯度、toy bf16 vs W8A16、可选加载公开 Anima INT8-ConvRot 做推理对齐。
5. 评估必须以 **adapter grad + 短训 sample** 为准，不能只看权重反量化 L2（旧 int8_linear audit 不够）。

### 三条战略（按投入）

| 路线 | 内容 | 收益 | 风险 |
| --- | --- | --- | --- |
| A. 仅推理/服务 | 文档 + 可选加载 int8-ConvRot 推理 | 低成本对接社区 | 不解决 int8 **训练** |
| B. ARA / 冻结 quant base + LoRA | 预量化 base 上训 adapter | 有社区先例；可降显存 | 可能是补 quant 误差，而非真低精 GEMM 训练 |
| C. 真 W8A16→W8A8 训练路径 | 自研 ConvRot Linear + 训练 hook | 直接打 rowwise 失败根因 + 可冲 GEMM | Tier 1.5/2；kernel / compile / 数值工作量大 |

目标若是 **W8A16 + W8A8 训练支持**，应走 **C**；B 可作对照实验。

---

## 7. 开放问题

1. 第一期是否只实现 W8A16，接口预留 W8A8？（建议：接口齐、实现先 W8A16）
2. 权重来源：在线从 bf16 做 RHT+quant，还是加载社区 `*-int8-ConvRot.safetensors`？
3. 成功标准：短训 sample 接近 bf16 LoRA，还是显存 / 速度 KPI？
4. 范围：仅 LoRA family，还是 Hydra / T-LoRA / FEI 全开？（router 与 online act rotate 共存需单独设计）
5. 是否存在开源 **训练期** W8A16/W8A8 GEMM + STE 穿过 ConvRot 冻结权重的实现，及其在 V100/A100 级上的实测收益？

---

## 8. 实现顺序（历史清单；核心路径已落地）

1. [x] 确认 Phase 1 范围与权重来源。
2. [x] 模块：`library/runtime/convrot/`（RHT、quantize、W8A16/W8A8 Linear wrapper）。
3. [x] 训练钩子：`apply_to` 之后 patch frozen base / `org_forward`。
4. [x] 配置 / CLI / WebUI MVP 实验项（默认 `bf16`，UI 标明实验）。
5. [x] 测试：unit + probe gate；公开权重推理对齐仍为 best-effort。
6. [~] 文档：本页为可运行实验说明；稳定后再考虑迁 `docs/methods/`。

**当前默认不做**：改训练默认配置、长训、下载大模型、推送。

---

## 9. 主要来源

- [arXiv:2512.03673 ConvRot](https://arxiv.org/abs/2512.03673)
- [FLUX.1-dev-ConvRot 模型卡](https://huggingface.co/SearchingMan/FLUX.1-dev-ConvRot)
- [ComfyUI v0.27.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.27.0) / [PR #14859](https://github.com/Comfy-Org/ComfyUI/pull/14859)
- [Anima INT8 ConvRot 权重](https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot)
- [INT8 质量基准（含 Anima）](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/Metrics.md)
- [convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- [ComfyUI-AnimaTurbo](https://github.com/TheLegendOfKitty/ComfyUI-AnimaTurbo)
- [QLoRA](https://arxiv.org/abs/2305.14314) / [SmoothQuant](https://arxiv.org/abs/2211.10438) / [QuaRot](https://arxiv.org/abs/2404.00456) / [SpinQuant](https://arxiv.org/abs/2405.16406)
- 本仓：[`../findings/anima_int8_base_linear_audit.md`](../findings/anima_int8_base_linear_audit.md)、`library/runtime/int8_linear.py`、`bench/channel_stats/`
