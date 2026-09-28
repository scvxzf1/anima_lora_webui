# Qwen Image 2.1 编辑采样多参考图热测（2026-09-28）

状态：实验验收
适用范围：`qwen_image_2_1` 训练过程中的编辑采样；不代表独立推理或最终画质验收。

## 结论

在 CMP 90HX 10 GiB（CUDA 可见设备使用 GPU UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`）上，基于
`configs/imported/9-27-50x-qwen21-edit.toml` 及其数据集缓存，完成了独立 3 步 BF16 编辑训练热测。训练保留
`blocks_to_swap=24`、full checkpoint、`torch_compile=true`，并在训练开始和第 2 步各生成 4 条样张：

- 单参考图：通过；
- 双参考图：通过；
- 四参考图：通过；
- 双参考图 + 显式空负面提示词 + CFG=3：通过。

共生成 8 张 PNG 及 8 个编辑对比图，所有 latent 均成功解码并从待处理目录移除。训练进程退出码为 0，采样后继续完成优化步骤并保存 LoRA。

## 运行与遥测

运行目录：`output/runs/qwen21-sampling-hot-20260928-125157`。

短测使用 50 对数据的既有 `dataset.runtime.toml` 和缓存，不重写原始数据集；输出、样张 JSON、参考图快照、编译缓存和日志均位于独立运行目录。样张目标为 512×512、每条 4 Euler steps，训练步数为 3。

训练 TensorBoard 标量均为有限值：

- loss/current：0.5544、0.2242、0.1837；
- loss/average：0.5544、0.3893、0.3208；
- sample step：0、2 均有 4 条输出；
- 采样后第 3 步仍完成，最终保存成功。

监控进程观测到 GPU 显存峰值 9106 MiB，最高温度 55°C；这是 2 秒采样的非瞬时观测值，不能替代 allocator 的精确峰值。训练记录的 CUDA max allocated 为 8.45 GiB，max reserved 为 8.60 GiB。

## 关键修复

1. 训练采样的 Qwen3-VL 条件缓存复用已加载编码器的 CPU-offload 上下文，不再把 8B 编码器整体搬到 10 GiB GPU；异常路径会拆除 Accelerate hooks 并清理元数据。
2. 缓存设备跟随 `Accelerator` 的实际 indexed CUDA device，避免多卡进程默认争用 `cuda:0`。
3. 参考图读取对没有 `n_frames` 属性的 JPEG 按单帧处理，同时继续拒绝动画图片。
4. 多参考图顺序、参考 latent、VAE image slot、CFG 正/负条件和对比图元数据均保序。

## 验证边界

- 本次验证的是采样链路和采样后恢复训练，不是 28 步或更高步数的画质评审。
- 没有把 3 步 loss 当作训练质量结论；已有 50 步编辑训练报告仍是训练链路验收依据。
- 未验证多卡分布式真实采样、其他 GPU 架构、长时间采样热稳定性或独立推理入口。
- 首次误用 GPU 数字编号时命中了 GTX 1050（当前 PyTorch 不含该 SM 架构），随后改用 90HX UUID 重跑；这次失败未改动训练数据。

## 回归测试

- `.venv/bin/python -m pytest -q tests/test_qwen_text_encoder_runtime.py tests/test_qwen_cache_policy.py`：20 passed；
- `.venv/bin/python -m pytest -q tests/test_training_preview_spec.py tests/test_sample_references.py`：45 passed；
- Ruff 对本次修改文件通过；`git diff --check` 通过。
