# Qwen Image 2.1 CMP 90HX 热测

日期：2026-09-26
状态：单次 60-step 热测通过；非收敛或画质验收。
结论：指定配置的 GPU 训练与检查点保存短测通过；不是收敛或画质验收。

## 测试配置

- GPU：NVIDIA CMP 90HX，SM 8.6；训练进程内 `cuda:0` 对应本机 90HX（物理 GPU index 1）。
- 模型：Qwen Image 2.1 T2I，BF16，plain LoRA rank 16，batch size 1，fused AdamW。
- 内存策略：full gradient checkpointing + `blocks_to_swap=24/32`。
- 编译：LoRA apply/load 后逐 block 编译，32/32 blocks；Inductor，`dynamic=True`。
- 数据：同一张 `672x1536` resized 图片重复 60 次，单 epoch、60 optimizer steps；没有 regularization images。
- 运行产物：`output/runs/qwen21-hot-90hx-20260926-60step/`，包含 progress、训练日志、遥测和检查点。

## 结果

- 60/60 步完成；`progress.jsonl` 最终状态为 `ok`，`final_step=60`，进程退出码 0。
- 热态 `recent_s_per_step` 约 7.95 秒；训练进度条耗时约 8 分 29 秒，包含首步编译/预热成本。
- PyTorch peak allocated/reserved：5.44/7.80 GiB；设备级遥测显存最高 8286 MiB。
- 60 个、约每 10 秒一次的遥测样本：温度 42–56°C、平均 51.5°C；采样功耗最高 238.78 W、平均 174.61 W。
- 成功保存 `qwen21_cmp90hx_swap24_60step.safetensors`（32.1 MB）。训练日志未见 OOM、CUDA error 或 traceback。

## 验收边界

这证明 CMP 90HX 上 full checkpoint、swap24 和逐 block compile 的组合可以完成一次 60-step 单图训练并保存 LoRA。单图重复不代表多图吞吐或训练质量；本轮没有无 swap/无 compile 对照，没有加载已保存检查点做 round-trip/续训，也没有评估收敛或样图。因此 swap24 仅是当前硬件上的可运行测试点，不据此设定跨硬件生产推荐。
