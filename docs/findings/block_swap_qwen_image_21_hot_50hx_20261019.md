# Qwen-Image 2.1 上的 Block Swap 与自动水位（CMP 50HX 第 12 轮）

状态：实验
适用版本：当前 main
入口命令：.venv/bin/python bench/block_swap_pipeline/run_probe.py --family qwen_image_2_1
相关代码：bench/block_swap_pipeline/run_probe.py, library/runtime/block_swap_auto.py,
library/models/qwen_image_2_1/weights.py, library/models/qwen_image_2_1/strategy.py

本轮是 P1（`blocks_to_swap="auto"` 动态显存调优）第四族真机验收，也是四族收口的一轮。
同时**修正上一轮 P0' 的一个结论**（见末节）。

## 1. 结论摘要

在 CMP 50HX 10240 MiB 上以 512×512、bf16、`attn_mode="torch"`、
`blocks_to_swap=30`（该族上界）跑通 Qwen-Image 2.1 训练，并实测自动晋升：

| 指标 | 固定 swap=30 | 自动晋升 30→14 | 变化 |
| --- | --- | --- | --- |
| 稳态任务数/步 | 60（=2S） | 28（=2S'） | −53.3% |
| 稳态搬运量/步 | 24.38 GiB | 11.38 GiB | **−53.3%** |
| DMA 忙/步 | 8362 ms | 8542 ms | +2.2% |
| 步跨度/步 | 7848 ms | 3877 ms | **−50.6%** |
| DMA 占步跨 | 106.6% | 220.3% | 搬运已完全被计算掩盖 |
| 暴露等待/步 | 4.51 ms | 2.12 ms | −53% |

晋升日志（`train.log` 原文）：

```text
block_swap_auto: promoting 16 block(s) to resident, blocks_to_swap 30 -> 14
(free=7.00 GiB, margin=0.39 GiB, block=416.0 MiB)
```

驱动力：block 只有 416.0 MiB，free 7.00 GiB 远高于 0.39 GiB 余量，可晋升 16 块，
但受 `plan_promotion` 的 offloader 契约限制为 `current - 2 = 28`，取 min 后为 16。

## 2. 与只读头部推算的一致性

`family_bytes.py` 的头部推算预测该族为 32 块、每块 416.0 MiB、合计 13.00 GiB。
实测独立验证同一数字：

- 每步 60 任务 × 每任务字节数 = 26172487680 B/步 → 每块 **416.0 MiB**（=2×416.0 MiB/job-set），
  与头部推算逐字节一致。
- 训练日志 `frozen_weight_master_bytes` 与历史 90HX 记录同为 13.00 GiB 量级。

即四族的 block payload 头部推算现已全部有真机交叉验证（anima 132.0 / krea2 828.1 /
z_image 345.1 / qwen_image_2_1 416.0 MiB）。

## 3. 一个必须记录的方法学修正：`duplicate_restores` 不是缺陷证据

`run_probe.py` 把「同一步内同一个 block_idx 出现两次」记为 `duplicate_restores`。
本轮数据显示该指标在 baseline 上报 28 个块号，看起来像「折返点重复搬运」。
逐事件核对后确认**这是误报，不是缺陷**：

```text
step 1: jobs 60 (fwd 30/bwd 30) fwd_unique 30 bwd_unique 30 overlap 28
        block 2 forward_prefetch  queued ...325.2374
        block 2 backward_prefetch queued ...332.7800
```

forward 相位与 backward 相位**各自覆盖全部 30 个被换出的块**，两个相位的交集自然非空。
这是块交换的正常契约（前向用完即退、反向再取回），不是重复恢复。真正的重复恢复应当是
**同一相位内同一块出现两次**；那两个相位的 unique 数都等于任务数（30/30），重复数均为 0。

因此该字段的口径需要收敛为「同相位内同块重复」。本轮不改探针行为（改动会影响历史可比性），
但后续读数不应再把非零值直接解释为折返点证据。在 30→14 的两个稳态步里 overlap 降为 0
只是因为 S'=14 时两个相位覆盖集合不再重叠，同样不能解释为「问题被修好」。

## 4. 对上一轮 P0' 结论的修正

第 1 轮 P0 曾据「`exposed_wait ≈ 1.6 ms/步`」判定「没有折返点重复搬运、P0 前提不成立」。
本轮更高分辨率的读数显示该结论**方向正确但依据不完整**：

- 正确：确实不存在同相位重复恢复，暴露等待确实极小（本轮 2.12–4.51 ms）。
- 不完整：`exposed_wait` 小并**不等于**搬运不拖慢训练。本轮 baseline 的 DMA 忙时间
  8362 ms > 步跨度 7848 ms（106.6%），说明搬运线程整体被计算掩盖，而**步时本身被
  「搬运 + 计算」共同拉长**。把块晋升到常驻后，搬运量腰斩、步跨度随之腰斩（−50.6%），
  证明此前步时里有约一半是**被搬运量间接决定的**，只是它表现为「计算等待数据」而不是
  「训练线程显式等待搬运完成」。

结论：P0 对「U-Turn 重复搬运」这一具体机制的否定仍然成立，P0' 仍应保持关闭；
但不应把 `exposed_wait` 当作「搬运不构成瓶颈」的充分证据。瓶颈的正确读法是
**搬运量/步**与 **DMA 占步跨比**，这两者本轮都被 P1 显著改善。

## 5. 实测坑（本轮新增）

1. **Qwen3-VL tokenizer 不在 TE 单文件旁边**。
   `strategy.py::resolve_tokenizer_path` 从 TE 路径向上找 `tokenizer/`（含
   `tokenizer_config.json`）或 ComfyUI 的 `comfy/text_encoders/qwen25_tokenizer`；
   ComfyUI 单文件 TE 两者都没有，报
   `FileNotFoundError: Qwen Image 2.1 needs a local Qwen3-VL tokenizer`。
   探针改为在 workspace 的 TE 副本旁按 preset 的 `tokenizer` 键就地铺设一份 tokenizer。
   `qwen25_tokenizer` 虽可加载（pad 151643 合法），但其词表对齐 Qwen2.5 而非 Qwen3-VL；
   本轮选用本机 HF 缓存的 Qwen3-VL-4B snapshot（`<|im_start|>`=151644、pad=151643、
   prefix 6 token），与族内断言一致。
2. **`timestep_sampling` 会从 `configs/base.toml` 漏进来**。
   base 默认 `sigmoid`，而该族契约（`library/training/compat_matrix.py:223`）只放行
   `uniform`，报
   `ValueError: Qwen Image 2.1 T2I adapter uses uniformly sampled dynamic-shift sigmas`。
   探针的 `runtime_extra` 原为空串导致该项未被覆盖；现按 Z-Image 的做法显式
   `timestep_sampling = "uniform"`。注意方法配置 `configs/methods/qwen_image_2_1_lora.toml`
   自带该键，所以走合并链的训练不受影响——只有探针这类手写 runtime TOML 会踩到。
3. **ComfyUI 2511/Edit 的 60 块 checkpoint 不是本族**。
   `.../diffusion_models/qwen/qwenimage2512-fp8_e4m3fn.safetensors` 会被
   `_validate_header` 直接拒绝；本族正确权重在
   `/home/scv/nvme0n1p1/ComfyUI/local_models/{diffusion_models,text_encoders,vae}/Qwen-Image-2.1/`，
   已核对为 32 块 + packed `img_mlp.gate_up`。

## 6. 复现命令

```bash
# baseline：固定 30 块换出
.venv/bin/python bench/block_swap_pipeline/run_probe.py \
  --family qwen_image_2_1 --side 512 --steps 4 --blocks-to-swap 30 --gpu 0

# 自动水位
.venv/bin/python bench/block_swap_pipeline/run_probe.py \
  --family qwen_image_2_1 --side 512 --steps 4 --blocks-to-swap 30 --block-swap-auto --gpu 0
```

产物：`output/bench/block_swap_pipeline/20261019-103352-swap30/`（baseline）、
`output/bench/block_swap_pipeline/20261019-103643-swap30/`（自动）。

## 7. 未验证边界

- 只跑了 4 步、512×512、单卡、LoRA dim 8；未做长跑稳态与热降频观察。
- 未测 compile 路径（本族 `torch_compile` 默认 true，探针显式关闭）。
- 未测 Edit 分支、未测 1024 原生分辨率（block 416 MiB 下 1024 的激活显存风险未评估）。
- 未测 NF4 transformer 变体（本机另有 3.67 GB 的 `qwen_image_2.1_nf4.safetensors`）。
- 晋升阈值仍是单点观测；`margin_mib` 敏感性未做扫描。
