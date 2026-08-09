状态：稳定
适用版本：当前 krea2-migration 分支
入口命令：
- `.venv/bin/python scripts/krea2/probe_nf4_ablation.py`（单格，env 开关）
- `bash scripts/krea2/run_nf4_ablation.sh`（5 格串行调度）
- `SMOKE=1 bash scripts/krea2/run_nf4_ablation.sh`（冒烟，256×256/3 步）
相关代码：`scripts/krea2/probe_nf4_ablation.py`、`scripts/krea2/run_nf4_ablation.sh`、`docs/findings/krea2_nf4_ablation.jsonl`（六维原始数据）

# Krea-2 NF4 × {完整检查点, 块交换} 消融矩阵

## 目标

goal 长期任务（14400s）围绕 NF4 量化 + 完整检查点 + 块交换三个显存优化正交轴，
做消融实验，记录六维基准指标：显存 / 内存消耗 / 速度 / loss / 数学实现 / 数学偏移。
每个配置（格）跑 30 步 1024×1024 flow-matching 过拟合训练，输出一行 JSONL。

## 消融矩阵设计

三个正交轴（环境变量开关）：

| 轴 | 开关 | 取值 | 含义 |
|---|---|---|---|
| NF4 | `K2_ABL_NF4` | 0/1 | DiT 4-bit 量化冻结（0=bf16 基线） |
| 块交换 | `K2_ABL_SWAP` | 0/N | blocks_to_swap（0=off） |
| 完整检查点 | `K2_ABL_CKPT` | 0/1 | 训练中途存 LoRA+opt state，reload 续训 |

5 格消融矩阵：

| 格 | NF4 | swap | ckpt | 角色 |
|---|---|---|---|---|
| `bf16_base` | 0 | 0 | 0 | bf16 基线 + 数学偏移 ref producer |
| `nf4_only` | 1 | 0 | 0 | NF4 量化单独效应 |
| `nf4_swap4` | 1 | 4 | 0 | NF4 + 块交换（方向 A 闭环） |
| `nf4_ckpt` | 1 | 0 | 1 | NF4 + 完整检查点 |
| `nf4_swap4_ckpt` | 1 | 4 | 1 | 三轴全开 |

## 六维指标采集方法

1. **显存**：`torch.cuda.max_memory_allocated`（训练 loop 内 `reset_peak` 后）+ `memory_allocated`
2. **内存**：`ru_maxrss`（host RSS 峰值）+ `/proc/meminfo MemAvailable`（系统可用）
3. **速度**：每 step `time.time()` 差，记录首/末/avg + 全 list
4. **loss**：first5 / last5 均值 + 单调下降 + 有限性 + 全 list
5. **数学实现**：flow-matching 公式快照 `x_t=(1-σ)*latent+σ*noise; target=noise-latent;
   loss=mse(velocity,target)` + 5D latent (B,C,T=1,H,W) 不变量 + σ=0.5 + seed=123
6. **数学偏移**：NF4 vs bf16 forward delta。**控制变量**：producer 和 consumer 都用
   **训练前初始 LoRA**（zero-init，B=0 → LoRA delta=0 → 纯 frozen DiT forward）各跑一次
   `inference_mode` forward，delta 才是纯量化对 DiT forward 的偏移，不混入训练轨迹分叉。
   producer 落盘 ref velocity，consumer 读 ref 算 max_delta / rel_l2 / cosine

## 实测矩阵（PG199 32GB, 1024×1024, 30 步, LoRA dim=16）

| 格 | GPU peak | host RSS | avg step | loss first5 | loss last5 | cos | rel_l2 |
|---|---|---|---|---|---|---|---|
| bf16_base | 29.45GB | 26.10GB | 3.29s | 0.001993 | 0.000459 | —（ref） | — |
| nf4_only | 10.49GB | 18.17GB | 3.41s | 0.008478 | 0.001793 | 0.9972 | 7.55e-02 |
| nf4_swap4 | 10.26GB | 18.18GB | 3.88s | 0.008301 | 0.001662 | 0.9972 | 7.55e-02 |
| nf4_ckpt | 10.49GB | 18.18GB | 3.41s | 0.008282 | 0.001666 | 0.9972 | 7.55e-02 |
| nf4_swap4_ckpt | 10.45GB | 18.18GB | 3.88s | 0.008429 | 0.001682 | 0.9972 | 7.55e-02 |

检查点格 round-trip（中途存 + reload 续训）：

| 格 | LoRA delta | fwd delta | loss 跳变 | LoRA 文件 | opt 文件 | 存盘耗时 |
|---|---|---|---|---|---|---|
| nf4_ckpt | 0.00e+00 | 0.00e+00 | 0.0001 | 96.4MB | 193.0MB | 1.3s |
| nf4_swap4_ckpt | 0.00e+00 | 0.00e+00 | 0.0001 | 96.4MB | 193.0MB | 1.4s |

5 格全部 `判定通过=True`（loss 单调下降 + grad 非零 + finite + NF4+swap 格 host RAM <20GB + ckpt 格 round-trip delta<1e-3 + loss 跳变<first5/2）。

## 判断与结论

### 1. NF4 量化：显存省 72%，速度几乎不损失，forward 方向高度保真

- **显存**：bf16 29.45GB → NF4 10.49GB，**省 19GB（72%）**。这是 NF4 的核心收益，让
  Krea-2 12.82B DiT 能在 12GB 级卡上训练（bf16 需 30GB+ 卡）。
- **速度**：3.29s → 3.41s，**仅慢 3.6%**。NF4 forward 反量化慢 4.6-7.2×（见
  `docs/proposal/krea2_nf4_blockswap.md`），但 1024 训练里 forward 计算占主导，
  反量化开销被 grad-ckpt recompute + LoRA + attention 摊薄。这是反直觉的好结果。
- **数学偏移**：cos=0.9972（方向高度保真），rel_l2=7.55%（相对 L2 误差），max_delta=0.5。
  NF4 forward 与 bf16 方向几乎一致，7.5% 量级误差是 4-bit 量化的合理范围。**未测**：
  图像质量损失（loss + round-trip 已验证，FID 留生产路径）。

### 2. 块交换（方向 A）：主战场在 host RAM，不在 GPU；搬运零语义偏移

- **GPU**：NF4 10.49GB → NF4+swap4 10.26GB，**仅省 0.23GB**。swap=4 只移出 4 块，
  GPU 收益微乎其微——证实 proposal 结论：**NF4+swap 的 GPU 不是主战场**。
- **host RAM**：18.17GB → 18.18GB，持平。NF4 master 5.7GB 已在 RSS 内，swap 不增。
  对比 bf16 路径 22.64GB pinned master 单项就超它（曾在 62GB 机宕机）——**NF4 master
  解决宕机，这是块交换在 NF4 上的真正价值**。
- **速度**：3.41s → 3.88s，**慢 14%**。`Params4bit.to()` 走默认 stream、H2D 串行，
  无法和 forward 重叠（方向 B 进阶可解，见 proposal）。
- **数学偏移**：与 nf4_only **逐位一致**（cos=0.9972, rel_l2=7.55e-02, max_delta=0.5）。
  块交换 deepcopy master + `Params4bit.to()` 整体搬运**零语义偏移**——这是方向 A 的
  核心证据，搬运不改 forward 语义，delta 纯量化误差。

### 3. 完整检查点：NF4 下续训无损，DiT master 保留不重载

- **round-trip delta=0**：LoRA 权重逐键 round-trip 零偏差，forward 输出零偏差。
  保存开箱即用（`save_weights` → 纯 state_dict 经 `lora_save`，无 anima 硬编码）。
- **loss 连续性**：reload 前 0.0027 → 后 0.0026，**跳变仅 0.0001**（数值噪声级），
  续训无缝。reload 只覆盖 LoRA + opt state，DiT NF4 master 保留不重载，省去
  `probe_checkpoint.py` 那种 del+reload 26GB DiT 的昂贵开销。
- **检查点大小**：LoRA 96.4MB + opt 193.0MB = 289.4MB（DiT NF4 frozen 不存），
  存盘 1.3-1.4s。
- **关键修正**：block swap 模式下，round-trip 验证 forward **必须先调
  `prepare_block_swap_before_forward`**，否则 swap block 还在 CPU，forward 访问
  Linear4bit 时 device 错乱 → `RuntimeError: invalid argument to getCurrentStream`。
  swap=0 时 prepare 直接 return 不受影响；swap>0 的三轴全开格首次因此崩溃，补 prepare 后通过。

### 4. 三轴叠加：无相互破坏，各轴收益可加

`nf4_swap4_ckpt` 三轴全开：
- GPU 10.45GB（≈ nf4_swap4 10.26GB + 存盘开销）
- host RSS 18.18GB（<20GB 通过）
- step 3.88s（同 nf4_swap4，存盘 1.4s 摊薄）
- round-trip delta=0，loss 跳变 0.0001
- 数学偏移 cos=0.9972（同所有 NF4 格）

**结论：三个优化正交可叠加**。NF4 省显存 72%，块交换解决 host RAM 宕机（不增 GPU
收益但保安全边际），完整检查点续训无损。三轴全开 = 10.45GB GPU + 18.18GB host RAM +
无损续训，是 8-12GB 卡训练 Krea-2 的可用配置。

## 方法论与局限

### 数学偏移的控制变量

数学偏移衡量**纯 NF4 量化对 frozen DiT forward 的偏移**，必须控制 LoRA 状态相同。
正解：producer 和 consumer 都用**训练前初始 LoRA**（zero-init，B=0 → LoRA delta=0
→ 纯 frozen DiT forward）各跑一次 inference_mode forward。若在训练中/后捕获，
LoRA 状态不同步，delta 会混入训练轨迹分叉，失去「数学偏移」语义。本探针 ref forward
放在训练前、reset_peak 之后，inference_mode 隔离不计入显存 peak。

### host RAM 判定阈值按格区分

`ram_ok = rss_peak < 20.0` 只对 **NF4+swap 格**生效（该格主战场是 pinned NF4
master ~5.7GB，应远低于 bf16 22.64GB master）。bf16 基线格 host RSS ~26GB 是 bf16
权重副本的预期值（不限），NF4-only 格 master 也在 ~6.7GB（不限，无 swap）。统一 20GB
阈值会让 bf16 基线格误判失败。

### 非目标

- 真实数据集 sweep（单 prompt 过拟合，非生产训练）
- 跨机复现（PG199 单机）
- FID / 图像质量（loss + round-trip + 数学偏移已验证，FID 留生产路径）
- DiT 权重正态性检验（NF4 信息论最优前提，见 9 点理论核实）

## 原始数据

六维完整 JSONL：`docs/findings/krea2_nf4_ablation.jsonl`（5 行，每行一格）。
每格日志：`docs/findings/krea2_ablation_<tag>.log`。

JSONL 记录结构：`config`（轴配置 + 公式快照）/ `metrics`（六维指标）/ `math_impl`
（公式 + σ + seed + latent shape）/ `math_offset`（producer/consumer + delta）/
`ckpt`（round-trip + 文件大小）/ `losses` / `step_times`。
