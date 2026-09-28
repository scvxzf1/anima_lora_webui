# Qwen Image 2.1 INT8 Packed Block-Swap CMP 90HX 热测

日期：2026-09-27
状态：INT8 packed swap 与 BF16 对照各完成一次 60-step 热测；检查点保存成功。当前只支持“本机、本配置下有约 3.7% 单轮步时改善”的初步结论，不足以推荐或改为默认。

## 测试配置

- GPU：NVIDIA CMP 90HX，物理 GPU 1；训练进程通过 `CUDA_VISIBLE_DEVICES=1` 使用。
- Qwen Image 2.1 T2I，BF16 base compute、plain LoRA rank 16、batch size 1、fused AdamW。
- 两轮共同设置：seed 1234、full gradient checkpointing、swap 24/32、torch attention、compile 开启、60 steps。snapshot 中模型族、精度、checkpoint、swap 数、attention、compile、seed 和训练步数一致；有意对照差异是 `block_swap_transfer_dtype`（INT8 / BF16）。INT8 专用路径使用 `int8_scope=all`、`int8_restore_mode=copy`。
- INT8 运行：`output/runs/qwen21-int8-packed-90hx-60step-20260927/`。
- BF16 对照：`output/runs/qwen21-bf16-swap24-90hx-60step-20260927/`。

## 结果

步时口径为 `progress.jsonl` 最近 20 条 step 事件的 `recent_s_per_step` 均值；进度文件每两个 global step 记录一次，因此不是逐步原始计时。

| 配置 | 后 20 条事件均值 | 全部事件范围 | Peak allocated / reserved | 最终平均 loss | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 swap | 7.9908 s/step | 7.9850–7.9930 s/step | 5.4438 / 7.8047 GiB | 0.362560 | 60/60，`status=ok` |
| INT8 packed swap | 7.6955 s/step | 7.5894–7.7388 s/step | 5.4450 / 7.9531 GiB | 0.361698 | 60/60，`status=ok` |

INT8 单轮步时比 BF16 低约 **3.69%**。两轮均成功保存 LoRA 检查点，训练记录未见 OOM、NaN 或 CUDA 错误。INT8 检查点为 `checkpoints/qwen21_int8_packed_swap24_60step.safetensors`（33,608,872 bytes）。

INT8 swap profile 记录 192 个量化张量；每块传输 payload 为 268,599,808 bytes，对应 BF16 的 436,208,128 bytes，减少约 38.4%。profile 的 `relative_l2` 范围约 0.00986–0.01461，`max_abs_error` 约 0.00307–0.00655。这些是权重反量化误差，不等于训练质量或 adapter 输出质量验收。

## 遥测与口径限制

- 两轮是顺序运行，不是交替 A/B/A。对 `gpu_telemetry.csv` 中利用率至少 90% 的样本统计，INT8 为 168 个样本、平均温度 53.5°C、平均功耗 221.8 W、平均核心时钟 1707.9 MHz；BF16 为 244 个样本、51.3°C、181.6 W、1674.4 MHz。利用率都接近 100%，但两轮频率和功耗明显不同，3.69% 差异不能全归因于 INT8 传输。
- 两份 `block_swap.jsonl` 都有 2,880 条事件，事件字段 `step` 均为 1；只能作为内部 block 事件流，不能据此按 60 个 optimizer step 分桶。INT8 的 `h2d_ms` 包含 packed 传输及恢复/反量化工作，不应当作 PCIe 纯传输时间；`gpu_wait_ms=0` 是未启用该计时的占位值。
- 这是各一轮的固定 seed 对照，没有温频匹配的交替重复，也没有相对 BF16 的多提示词学习曲线/图像质量、检查点 reload 或续训验证。保存成功不等于这些验收已通过。

## 验证与结论

定向测试：

```text
rtk test timeout 300 .venv/bin/python -m pytest tests/test_block_swapping.py tests/test_qwen_image_2_1_block_swap.py tests/test_int8_linear_runtime.py tests/test_int8_blockswap_equivalence_probe.py -q
99 passed, 16 warnings in 29.09s
```

结论：方案 1 的 packed INT8 CPU master、块 slab 传输和 BF16 slot 分块恢复路径已在 90HX 完成首次 60-step 验收；观察到小幅步时收益，但低于结构性优化审阅中的约 5% 投入门槛，且 A/B 遥测存在时钟差异。**保持 INT8 显式 opt-in，不改默认值。** 推荐前仍需至少三轮温频受控的交替对照，并补充 checkpoint reload/续训及多提示词质量对照；若收益不稳定或质量不通过，则不继续推广。
