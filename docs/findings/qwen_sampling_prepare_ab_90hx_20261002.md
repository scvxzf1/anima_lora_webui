# Qwen 编辑采样块交换初始化 A/B（2026-10-02）

状态：定向 GPU 验收通过
适用范围：Qwen Image 2.1 训练预览；CMP 90HX、BF16、plain LoRA、swap24/slab、all-block compile。

## 结果

删除每个去噪预测前的块交换初始化后，两张编辑样张的热态批次耗时平均从
**480.568 秒降到 262.233 秒**，减少 **45.43%**，吞吐约为原版的 **1.83 倍**。
固定权重、条件和种子时，完整 28 步的 latent 逐元素相同，解码后的 PNG 像素也相同。

| 顺序 | 路径 | 竖图生成 / 秒 | 横图生成 / 秒 | 批次总耗时 / 秒 | 去噪循环内 prepare 次数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | A：原版 | 236.334 | 236.551 | 480.663 | 56 |
| 2 | B：新版 | 127.224 | 127.229 | 262.225 | 0 |
| 3 | B：新版 | 127.251 | 127.218 | 262.242 | 0 |
| 4 | A：原版 | 236.301 | 236.391 | 480.473 | 56 |

每个批次进入推理和恢复训练仍各初始化一次，因此 prepare 总次数为 A=58、B=2。
总耗时包含布局切换、两张图生成、latent 保存和测试事件落盘；不包含启动条件缓存、模型加载、
独立预热、后处理 PNG 解码。每张图和批次边界执行 CUDA 同步，避免仅测异步任务入队时间。

正式 A/B 前另跑了独立原版热态基线：236.432 / 236.338 秒，两张及切换合计 480.543 秒。
首次编译与每图两步的 A/B 预热不纳入上述性能统计。
正式轮次最大 allocated 为 A≈3.955 GiB、B≈3.953 GiB，reserved 均≈4.193 GiB；
本次不宣称有实质显存收益。该加速比例仅适用于本硬件和已测条件。

## 改动和验证

- `library/models/qwen_image_2_1/sampling.py`：删除 `predict()` 内的重复 prepare，
  明确 caller 负责批次前切入 inference、批次后恢复 training。
- `training_preview._sample_local()` 已实现上述批次生命周期，其代码无需改动。
  共享 offloader 的环形预取、训练初始化和异常恢复保持原有逻辑。
- CPU 生命周期测试覆盖两个 prompt、28 步、CFG 双分支，以及第二个 prompt 异常、RNG 和模式恢复。
- 真实 GPU CFG=3、显式空负面提示词的两步 A/B：latent 完全相同，最大绝对误差与相对 L2 均为 0。
- 正式两张 28 步样张及两轮复跑：所有 A/B 与独立 A 基线对照的 latent 完全相同；
  8 张解码 PNG 的对应像素数组也完全相同。
- GPU 第二个 prompt 第一次预测前注入异常：待传输任务排空，推理/训练各 prepare 一次，
  正确恢复 training 模式。
- 异常恢复后在同一模型上执行真实 full-checkpoint forward、backward、AdamW step：
  loss=0.750989，adapter 梯度有限且非零，更新 L2=0.0368453，基座保持冻结。
  此恢复烟测的目标 latent 为 `[1,64,8,8]`（128×128），保留完整真实参考条件；
  它不代表 1024 分辨率长训练复测，也不是恢复原任务的 optimizer/global step。

定向测试：

```text
timeout 60 .venv/bin/python -m pytest -q tests/test_qwen_image_2_1_preview.py tests/test_qwen_image_2_1_block_swap.py
35 passed

timeout 60 .venv/bin/python -m pytest -q tests/test_block_swapping.py -k 'slab_slot_ownership_tracks_forward_only_wraparound or submit_move_blocks_prefetch_depth_default_one_in_forward_only'
2 passed

CUDA_VISIBLE_DEVICES=GPU-806debd9-d254-da57-d515-ccae15b4ac6a timeout 60 .venv/bin/python -m pytest -q tests/test_block_swapping.py -k bf16_slab_forward_only_wraparound_matches_baseline
1 passed
```

最后一项是真实 BF16 CUDA/slab 非整除槽位测试：30 blocks、swap26、4 slots，
一次 prepare 后连续三次 forward 与不交换权重的基线逐元素相同。

## 输入与复现

独立运行目录：`output/runs/qwen-sampling-prepare-ab-20261002-r4`。
该目录的 `summary.json`、`events.jsonl` 保存逐阶段计时、准备次数、显存峰值、
源代码/输入哈希和数值对照；`baseline_sampling.py` 保留原版采样器，`conditions.pt` 保留实际条件。
运行退出码为 0、summary 状态为 `ok`，完成后 GPU 释放。

- 使用原热测第 400 步 LoRA；严格加载所有键匹配，128 个 attention LoRA 模块。
- CMP 90HX UUID 与 CUDA 映射核验；只有目标 GPU 可见，启动前没有其他 compute 进程。
- 与源运行一致：BF16、torch/SDPA attention、24/32 block swap、slab restore、
  full checkpoint、32/32 inner-block compile、dynamic sequence。
- 参考图 `a-2/134205076_p0.jpg`：384×512、Euler 28 步、seed42、guidance1。
- 参考图 `a-2/144305387_p0.png`：512×352、Euler 28 步、seed43、guidance1。
- 文本与参考 latent 在加载 DiT 前缓存并冻结，两条路径复用同一模型、adapter 和条件。
  没有接入 K/V 缓存或调整采样步数、尺寸、attention、权重精度。

使用新的输出目录复现（既有输出不得覆盖）：

```bash
CUDA_VISIBLE_DEVICES=GPU-806debd9-d254-da57-d515-ccae15b4ac6a .venv/bin/python -u \
  scripts/qwen_image_2_1/sampling_prepare_ab.py \
  --config output/runs/9-27-50x-qwen21-edit-20261002-073957/config.runtime.toml \
  --checkpoint output/runs/9-27-50x-qwen21-edit-20261002-073957/training_output/9-27-50x-qwen21-edit-hot1000-20261002-step00000400.safetensors \
  --baseline-sampling output/runs/qwen-sampling-prepare-ab-20261002-r4/baseline_sampling.py \
  --output output/runs/qwen-sampling-prepare-ab-recheck \
  --gpu-uuid GPU-806debd9-d254-da57-d515-ccae15b4ac6a \
  --baseline-repeats 1 --order A B B A
```

该脚本是显式运行的单机验收工具，会占用目标 GPU，不能与训练并行执行。
早期 r1–r3 测试夹具启动检查失败记录保留，未进入性能统计；r4 是完整验收结果。
当前支持边界不外推到其他 GPU、低比特交换、其他 adapter、多卡或所有多参考图组合。
