# Z-Image 上的 Block Swap 与自动水位（CMP 50HX 真机）

状态：实验（跨族真机验收 + 探针族预设，第 11 轮）
适用版本：`codex/block-swap-int8`
相关代码：`bench/block_swap_pipeline/run_probe.py`（`--family` 预设）、`library/runtime/block_swap_auto.py`
运行目录：`output/bench/block_swap_pipeline/20261019-092632-family-zimage2`（固定 swap=28）、
`.../20261019-094034-family-zimage-auto`（`--block_swap_auto`）

## 目的

两件事：给探针加**族预设**（`--family`），让跨族真机不再靠手改命令；并在 Z-Image 上验收
`--block_swap_auto`。上一轮在 Krea-2 上只观察到「不晋升」，**非 Anima 族的正向晋升路径仍未实测**
——本轮把它补上。

## 探针族预设（本轮实现）

`run_probe.py --family {anima,krea2_raw,z_image,qwen_image_2_1}` 预设各族：dit/te/vae 路径、
sidecar 后缀（`_anima.npz` / `_z_image.npz` / …）、运行时键（`model_family`、
`cache_llm_adapter_outputs`、Z-Image 的 `timestep_sampling`/`discrete_flow_shift`），
并按族选择 TE 缓存入口与参数名。Anima 沿用历史工作区 `workspace`（缓存可继续复用），
其它族使用 `workspace-<family>[-<side>]`。

实现中踩到并修掉的三个坑（都在这一轮实测暴露）：

1. **各族 TE 脚本的参数名**：`scripts/z_image|qwen_image_2_1|krea2/preprocess_te_cache.py`
   都是 `--qwen3`；最初用正则提取时被数字截断成 `--qwen`，会静默走错分支。
2. **Z-Image 必须用官方 Diffusers 根**（`/home/scv/nvme0n1p1/Z-Image`）：
   单文件 safetensors 无法实例化 tokenizer（`ValueError: Couldn't instantiate the backend tokenizer`），
   TE 缓存必须从带 `tokenizer/` 的根加载。探针对**目录型组件**改为原地引用而不是 `copy2`。
3. **旧文件副本会污染 cache identity**：第一次运行留下的 `workspace-z_image/vae.safetensors`
   （单文件 335 MB）被写进了 latent 缓存的 identity 戳；改用目录根之后训练侧判定为 `stale`，
   报 “Latent cache is incomplete”。探针现在会在组件源是目录时删掉这类遗留副本。
   教训：**换组件来源后必须重建 latent 缓存**，否则报错信息完全指向别处。

另：非 Anima 族的 latent 缓存必须和图片放同一目录 —— `library/datasets/dreambooth.py:198`
只在 `image_dir` 里找缓存文件名来推断图片尺寸，分离目录会让尺寸解析为 0/2 并最终报
“Latent cache is incomplete”（krea2_raw 与 z_image 都实测如此，Anima 容忍分离布局）。

## 结果 1：字节口径再次与头部推算一致

固定 swap=28 与 auto 两次运行的日志均为：

```text
Block swap frozen CPU masters prepared: 10.11 GiB across 30 blocks (transfer_dtype=bf16)
```

`family_bytes.py` 从头部算出 Z-Image 是 **10.11 GiB / 30 块 / 每块 345.1 MiB**（promotion 日志报
`block=345.0 MiB`）。这是继 Anima（132.0 MiB）与 Krea-2-Raw（828.1 MiB / 22.64 GiB）之后
第三个族的独立核对。

## 结果 2：固定 swap=28（4 步）

| 指标 | 实测 | 预测（头部字节 + 3.14 GB/s） | 偏差 |
| --- | --- | --- | --- |
| 任务/步 | 56 | 2 × 28 = 56 | 0 |
| H2D 字节/步 | 20260 MB | 2 × 28 × 361.8 MB = 20262 MB | −0.009% |
| DMA/步 | 6473 ~ 6572 ms | 20.262 GB / 3.14 GB/s = 6452 ms | +0.3% ~ +1.9% |
| 有效带宽 | **3.117 GB/s** | 天花板 3.14 GB/s | 99.3% |

步时 7.62 / 7.19 / 7.06 / 6.98 / 7.04 s → **DMA 占步时约 93%**，与 Krea-2（96%）同构：
Z-Image 在 50HX 上同样是被 PCIe 卡住的流水线。

## 结果 3：自动晋升在非 Anima 族实测成功（本轮关键）

`--blocks-to-swap 28 --block-swap-auto --margin-mib 800`：

```text
block_swap_auto: promoting (free=6.15 GiB, margin=0.78 GiB, block=345.0 MiB)
```

| 阶段 | `swap` | 任务/步 | H2D 字节/步 | DMA/步 | 步时 |
| --- | --- | --- | --- | --- | --- |
| 晋升前（step 1~2） | 28 | 56 | 20260 MB | 6473 ~ 6572 ms | 7.69 / 8.12 s |
| 过渡（step 3） | 28 → 13 | 26 | 9407 MB | 8700 ms | 6.58 s |
| 晋升后（step 4~5） | 13 | 26 | 9407 MB | 3011 / 3013 ms | 5.30 / 5.38 s |

自洽性核对（全部逐项吻合）：

- 晋升块数：`(6.15 − 0.78) GiB / 345.0 MiB = 15.9 → 15`，与日志一致（另受「保留 2 块」上限 26 约束，未触发）
  → `28 − 15 = 13` ✓
- 每步任务 2 × 13 = **26** ✓；每步字节 2 × 13 × 361.8 MB = **9407 MB** ✓
- 晋升后 DMA 9407 MB / 3.14 GB/s = 2996 ms，实测 3011 ~ 3013 ms（+0.5%）
- 过渡步的 8700 ms 是一次性代价：把 15 块（约 5.05 GiB ≈ 1.6 s）搬进显存，**一步内即回本**

收益：步时 7.69/8.12 s → 5.30/5.38 s，**每步省约 2.4 s（−30%）**，每步少搬 10853 MB。
这也印证了第 9 轮的判断：非 Anima 族「省一块」的价值是 Anima 的 2.6~6.3 倍（Z-Image 2.6 倍）。

## 边界与未证明项

- 仍是 512×512、2 张程序生成图、5 步以内的短跑；**没有**收敛性与长跑稳定性结论。
- Qwen-Image 2.1 未验收：`--family qwen_image_2_1` 的源路径目前是占位值（该族 latent 是
  `F16C64_P1`，与其它族的 stride/通道都不同，需要单独确认）。
- Z-Image 用的是官方 Diffusers 根；ComfyUI 的单文件 Z-Image 组件在本轮**未能**用于 TE 缓存
  （tokenizer 无法实例化），本文件不代表单文件路径可用。
- 晋升决策只测了 `margin=800 MiB` 一档；`stop_ratio` 在 Z-Image 上未测。

## 复现

```bash
# 固定水位
.venv/bin/python -m bench.block_swap_pipeline.run_probe --family z_image \
    --gpu 0 --steps 4 --blocks-to-swap 28 --label family-zimage
# 自动水位
.venv/bin/python -m bench.block_swap_pipeline.run_probe --family z_image \
    --gpu 0 --steps 5 --blocks-to-swap 28 --block-swap-auto --margin-mib 800
```