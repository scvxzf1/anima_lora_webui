# Krea-2-Raw 上的 Block Swap 与自动水位（CMP 50HX 真机）

状态：实验（跨族真机验收，第 10 轮）
适用版本：`codex/block-swap-int8`
相关代码：`library/runtime/block_swap_auto.py`、`library/models/krea2_raw/dit.py`、`bench/block_swap_pipeline/family_bytes.py`
运行目录：`output/bench/block_swap_pipeline/krea2-auto-20261019/`（`train512b.log` 为通过的那次）

## 目的

P1 的进程内水位调优（`--block_swap_auto`）此前只在 Anima 上做过真机。本轮在**第二个模型族**
（Krea-2-Raw）上真机验收，检查三件事：换出块的**字节口径**是否与头部推算一致、
每步 DMA/带宽是否复现 Anima 上的链路结论、以及无余量时调优器的决策是否正确。

## 环境与配置

| 项 | 值 |
| --- | --- |
| 卡 | CMP 50HX，10240 MiB（可用 9.56 GiB），Turing sm_75 |
| DiT | `models/diffusion_models/krea2_raw_bf16.safetensors`（26.28 GB 文件，bf16 全量） |
| TE / VAE | `models/text_encoders/qwen3vl_4b_bf16.safetensors` / `models/vae/qwen_image_vae.safetensors` |
| 注意力 | **只能用 `attn_mode="torch"`**：flash 在 Turing 不可用；`sdpa` 直接报 `RuntimeError: No available kernel` |
| 数据 | 2 张真实图；原生 896×1200 / 768×1344 → 降到 512×704 / 384×672 才跑得动 |
| LoRA | plain LoRA dim16/alpha8，bf16，gradient checkpointing，`torch_compile=false` |
| swap | `blocks_to_swap=26`（= `num_blocks-2`）、`block_swap_auto=true`、`block_swap_vram_margin_mib=1200` |

## 结果 1：字节口径与头部推算逐字一致

训练日志：

```text
Block swap frozen CPU masters prepared: 22.64 GiB across 28 blocks (transfer_dtype=bf16)
```

profile 里的 `frozen_weight_master_bytes = 24,310,185,984` B（22.64 GiB），
`family_bytes.py` 从 safetensors 头部算出的 Krea-2-Raw 总量是 **24,312,952,832** B
（22.64 GiB / 28 块 / 每块 828.1 MiB）。两者相差 **2.64 MiB（0.011%）** —— 头部口径会算上
块内不参与换出的少量张量（Offloader 只把冻结的 linear master 计入换出计划）。
所以结论应表述为「两口径在 0.01% 内一致」，而不是逐字相等；用来推算其它族的量级与
每步 DMA 是可靠的（见结果 3 的 −0.013%）。

## 结果 2：无余量时的决策正确（本轮核心）

```text
block_swap_auto: keeping blocks_to_swap=26
(free=0.67 GiB, margin=1.17 GiB, block=828.0 MiB, no headroom above margin)
```

- `free=0.67 GiB` 是第一步之后的实际空闲（该步峰值 reserved 8.50 GiB / 可用 9.56 GiB），
  低于 1.17 GiB 余量 → 命中 `no headroom above margin`，**不晋升**。这是设计行为，不是失败。
- 更硬的结论：即使把余量降到 0，**一块 Krea-2 载荷（828 MiB）本身就大于当时的空闲 686 MiB**，
  所以在这张卡上 Krea-2 的自动晋升是**结构性不可行**的 —— 晋升粒度（1 块）大于卡的余量。
  调优器此时唯一正确的动作就是保持配置值，它做到了。

## 结果 3：每步 DMA / 带宽复现 Anima 的链路结论

4 步训练（通过的那次），profile 按步统计：

| 指标 | 实测 | 预测（按 3.14 GB/s 链路 + 头部字节） | 偏差 |
| --- | --- | --- | --- |
| 任务/步 | 52 | 2 × 26 = 52 | 0 |
| H2D 字节/步 | 45147 MB | 2 × 26 × 868.3 MB = 45153 MB | −0.013% |
| DMA/步 | 14.41 ~ 14.63 s | 45.153 GB / 3.14 GB/s = 14.38 s | +0.2% ~ +1.7% |
| 有效带宽 | **3.114 GB/s** | 天花板 3.14 GB/s | 99.2% |

步时 15.96 / 15.01 / 14.99 / 14.98 s（`recent_s_per_step` 14.05 → 14.94），即 **DMA 占步时约 96%** ——
Krea-2-Raw 在 50HX 上就是一条被 PCIe 卡住的流水线，与分辨率无关（分辨率只影响那 4% 的计算）。
其余口径：`exposed_wait_ms` 6.96/步（无暴露等待，链路饱和）、`worker_queue_ms` 165 ms、
`forward_tail_idle_ms` 1609 ms。

## 结果 4：跨族真机的四个坑（都已实测解决）

1. **原生分辨率跑不动**：896×1200/768×1344 下 `attn_mode="torch"` 在 swap=20 与 swap=26 都 OOM
   （单次申请 4.23 GiB，发生在 `run_krea2_attention`）。降 token 数是唯一有效杠杆：
   512×704/384×672 一次通过，峰值 reserved 8.50 GiB。
   也就是说 **Krea-2 在 10 GB 卡上的边界是注意力内存，不是 block swap**。
2. **旧缓存 identity 过期**：`output/runs/auto-block-swap-hot-20260908/` 里现成的 latent/TE 缓存
   是更早的版本建的，直接复用会被 `is_latents_cache_complete()` 判为不完整；
   用当前代码重建即可（2 张图十几秒）：
   `scripts.preprocess.cache_latents --model_family krea2_raw --vae …` +
   `scripts.krea2.preprocess_te_cache --qwen3 …`。
3. **latent 缓存尺寸只在 `image_dir` 里找**：`library/datasets/dreambooth.py:198` 从 `subset.image_dir`
   glob `*{cache_suffix}`，拿到的尺寸直接决定 `get_latents_npz_path`。缓存放在单独的 `cache_dir` 时
   日志是 `set image size from cache files: 0/2` 并在后面报 “Latent cache is incomplete”；
   把 `image_dir` 与 `cache_dir` 指向同一目录后变成 **2/2** 并通过。
4. **`save_every_n_steps = 0` 会零除**：`library/training/checkpoints.py:861` 做取模，
   第一次保存检查即 `ZeroDivisionError`。该键应当省略而不是写 0。

## 边界与未证明项

- 只在 512 档跑通；原生 896×1200/768×1344 未跑通（原因见结果 4.1），因此**本文件不是**
  “Krea-2 在 50HX 上可用”的完整结论，只是 swap 机制的跨族验收。
- 未测 Z-Image 与 Qwen-Image 2.1 真机。
- 4 步短跑，无收敛性结论；最后一次进度行出现 `avr_loss=nan`，未排查（2 张图的玩具集 + 2e-4 lr）。
- 本轮没有观察到一次**正向晋升**（因为该卡上没有余量），Krea-2 上的晋升正路径仍未实测。

## 复现

```bash
# 1) 把图缩到 512 档并把缓存与图放同一目录，然后重建缓存
.venv/bin/python -m scripts.preprocess.cache_latents --dir <workspace> --cache_dir <workspace> \
  --batch_size 1 --dtype bfloat16 --vae models/vae/qwen_image_vae.safetensors \
  --model_family krea2_raw --require_cache_identity --overwrite
.venv/bin/python -m scripts.krea2.preprocess_te_cache --dir <workspace> --cache_dir <workspace> \
  --qwen3 models/text_encoders/qwen3vl_4b_bf16.safetensors --batch_size 1 --dtype bfloat16 --overwrite
# 2) 训练（runtime 见运行目录的 runtime512.toml）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --config_file output/bench/block_swap_pipeline/krea2-auto-20261019/runtime512.toml \
  --dataset_config output/bench/block_swap_pipeline/krea2-auto-20261019/dataset512.toml \
  --require_cache_identity
```