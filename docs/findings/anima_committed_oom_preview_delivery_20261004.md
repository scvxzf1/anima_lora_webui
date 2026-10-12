# Anima 已提交 OOM 恢复与独立预览回退验收

日期：2026-10-04。状态：受限实验实现；Anima 原生 FP16 短步恢复与 CPU 协议独立验收通过。
默认行为不变，中途恢复需显式开启。此报告区分状态恢复、真实 CUDA 控制流程、浮点数值与训练质量。

## 实现结果

参考 [SimpleTuner 固定源码调研](public_oom_fallback_survey_20261004.md#simpletuner) 中
sampler 保存和恢复后重新施加设备放置的思路，本项目补齐自身的数据与预取契约，未复制外部训练 loop。

`adaptive_oom_retry_resume=true` 在完整提交点保存 adapter、AdamW、scheduler、scaler、
训练计数、Python/NumPy/Torch CPU/CUDA RNG、隔离的数据 RNG、bucket 实际排列和
stateful sampler cursor。快照原子发布、内容哈希绑定，只加载 supervisor 所有的已提交目录。
新进程恢复后重新安排 swap 设备布局，校验 trainables 和 optimizer 的设备/dtype 及精度 guard。
图片 symlink 身份绑定到真实目标；恢复快照本身禁止 symlink 和路径穿越。

默认 `adaptive_oom_retry_resume=false`；开启后仍保持精度、有效 batch 与基座身份，
只增加 swap。forward/backward 容量 OOM 可以回到提交点；optimizer/checkpoint OOM、
CPU OOM、非有限值、非法访问、超时、取消、快照损坏及精度漂移保持停止。

预览单独使用 `sample_vae_decode=auto`：原设备解码 → GPU tiled → 预算允许的 CPU tiled。
只认明确 CUDA OOM。CPU 预算尊重 cgroup/RAM，不把 swap 当 RAM；拒绝时保留 latent。
事件写入 `sample/vae-decode-events.jsonl`，不会修改训练精度/batch 或发送训练 OOM 重试请求。
退出恢复 VAE 设备、dtype、tiling 和 tile 参数；恢复失败停止，普通失败保留 latent。

## 独立 CPU 验收

收尾审计：`output/tests/recovery-final-audit-20261004/result.json`。

```bash
CUDA_VISIBLE_DEVICES='' timeout 240 .venv/bin/python -m pytest \
  tests/test_training_recovery_checkpoint.py \
  tests/test_training_recovery_cursor.py \
  tests/test_training_recovery_integration.py \
  tests/test_training_recovery_subprocess.py \
  tests/test_training_recovery_startup_rng.py \
  tests/test_adaptive_training_supervisor.py \
  tests/test_native_startup_runtime.py \
  tests/test_native_startup_oom.py \
  tests/test_training_precision_report.py \
  tests/test_preview_decode_fallback.py -q
```

209 passed，87.87 秒，exit 0。定向 Ruff / Pyright 均 exit 0，独立审核 Critical=0、Warning=0。
34 条测试 warning 来自 torchdata 上游弃用接口，无共享 Accelerate 状态冲突。
本轮 8 篇修改文档的链接/锚点、索引、生命周期与总入口可达性定向检查通过。
全仓 `tests/test_documentation_integrity.py` 为 5 passed / 3 failed：失败项是已有 proposal
索引/可达性缺口及其他旧文档状态标签，未列出本轮修改文档；未扩展本次工作去改写这些材料。
对应记录为 `output/tests/recovery-final-audit-20261004/documentation.json`。
真实 CPU Accelerate/AdamW/scheduler 验证了新 PID、预取后部分参数/moments 污染的
精确回滚、下一输入、scaler/RNG 以及 epoch 耗尽/跨 epoch 游标。CPU subprocess 使用
故障注入，并非真实 GPU 分配证据。

## 真实 Anima CUDA OOM / 连续训练对照

固定现有 Anima 基座、初始 adapter、缓存数据集、seed=20261003；native AMP FP16，
FP32 trainables / 保护，scale=1024，rank=4，AdamW，workers=0，accumulation=1，
torch attention，完整 activation checkpoint。无采样/验证 schedule，最大 3 步。
唯一目标为 CMP 90HX UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，未使用桌面 GTX1050。

运行目录：

- `output/runs/anima-oom-resume-90hx-20261004-hot4/`
- `output/runs/anima-oom-resume-90hx-20261004-control2/`

hot4 在第 1 步提交后，下一个 forward 使用 `torch.empty(..., uint8, cuda)` 请求
20,714,225,664 字节，实际产生 typed CUDA OutOfMemoryError。这是容量故障注入，
不是自然训练峰值溢出。首 worker 退出后，新 worker 增加 swap 并恢复第 1 步状态。
control2 使用相同配置与初始状态，禁止注入并连续运行 3 步。

| 证据 | OOM 恢复 hot4 | 连续 control2 |
| --- | --- | --- |
| 退出/超时 | exit 0 / 否 | exit 0 / 否 |
| 实际训练 worker PID | 309279 → 310454 | 单 worker |
| swap | 20 → 24 | 20 |
| 最终有效更新 / skip | 3 / 0 | 3 / 0 |
| 全流程耗时 | 175.20 秒 | 102.09 秒 |
| 整卡显存采样峰值 | 3460 MiB | 3482 MiB |
| 进程树 RSS 采样峰值 | 11.20 GiB | 11.11 GiB |

耗时包含权重加载、失败尝试、新进程加载和快照 IO，不代表稳定训练每步速度。
监控为每秒采样，RSS 求和可能重复计算共享页，也可能遗漏短暂峰值。

三个提交点的 `cursor.pt`、`strict_rng.pt`、`scaler.pt`、`scheduler.bin` 共 12 项
逐字段完全一致。即时恢复的 cursor/RNG 观测均匹配，scaler=1024。
独立加载六份快照重新核对全部 inventory 哈希、精度 manifest、native guard、train_state，
每份 392 组 AdamW moments 均为有限 FP32、step=1/2/3；最终 840 个 adapter 张量全有限。
最终 adapter 另按配置保存为 BF16，恢复快照 trainables 保留 FP32。

原始证据与可复核脚本：hot4 下 `summary.json`、`monitor.jsonl`、`training.log`、
`hot-observations.jsonl`、`audit/result.json` / `light_compare.py`，各 attempt 的 worker 日志与快照。
独立审核脚本和输出另在 `output/tests/recovery-final-audit-20261004/`。

## 定位过程中保留的失败现场

| 目录后缀 | 结果与修复 |
| --- | --- |
| hot1 | 数据 identity 拒绝合法图片 symlink，尚未训练；改为绑定 resolved target 和内容 |
| hot2 | 完成首步并触发真实 OOM，恢复被 `cuda` / `cuda:0` 别名校验拒绝；规范化设备索引 |
| hot3 | 已恢复并完成 3 步，提交游标一致；Python RNG 多消费 23 次，严格控制状态审计 FAIL |
| hot4 / control2 | 两项修复及下述日志/缓存修复后，12 项控制状态逐字段一致，独立审计 PASS |

hot3 的 Python 差异来自恢复后重复的三条 Rich 启动日志。日志源码超链接在
`rich/style.py` 创建/copy Style 时调用 `randint`；从真实提交 RNG CPU 重放精确得到
275→283→291→298，额外 23 次。`resume_runtime.startup_log` 只隔离恢复时这三条日志；
初始启动不变，scheduler/数据/epoch 的实际随机调用不被屏蔽。测试撤回日志修复后可失败。

hot3 即时 cursor 摘要 false 来自 Accelerate 的 `dl_state_dict` 旧缓存，而提交 cursor
一直一致。`restore_loader` 现在 materialize base lazy restore 后刷新缓存，并覆盖了
中途与耗尽边界。旧 false 观察、旧 FAIL 与旧输出全部保留，没有改写。

hot3 连续对照在第 1 步、OOM 发生之前就存在浮点差异（权重最大绝对差 1.8326e-4）。
因此不把后续所有数值差异归因于恢复，也不宣称 GPU 权重/梯度逐位等价。
本轮没有新增训练质量评估，不能覆盖旧严格数学矩阵或 200 步质量观察的各自结论。

## 真实预览 CPU 与解码回退

使用现有 Anima Qwen VAE、CPU BF16、固定随机 latent `[1,16,1,8,8]`，
输出 64×64。真实调用共享 `decode_pending_samples`，未加载 DiT 或运行训练。
首轮目录为 `output/tests/preview-cpu-fallback-90hx-20261004/`，硬件路径均完成：
CPU 78.57 秒 / 1 callback；auto 77.99 秒 / 3 callbacks。
auto 的两个 GPU callback 各请求 2×GPU total memory，产生真实 CUDA OOM，随后
真实 RAM 预算 accepted，CPU 解码得到有限像素、PNG 保存、latent 删除。
两策略训练 args 未变，VAE CPU/BF16、tiling 与全部 tile 属性恢复。

首轮脚本最终 exit 1：测试脚本宽泛检查包含 `adaptive` 的模块名，误匹配 PyTorch 的
`torch.nn.modules.adaptive`。真实解码断言已经通过；原脚本/结果和 stderr 保留。
更正为精确项目模块名后，在新目录 `output/tests/preview-cpu-fallback-90hx-20261004-r2/`
完整复跑；最终结果见该目录 `result.json`，不覆盖首轮记录。

完整复跑已 exit 0、status=passed、无超时，共 177.91 秒。CPU / auto 分别
87.89 / 80.95 秒，callback 为 1 / 3；两张 PNG 保存、latent 删除、有限像素、
训练 args 未变和 VAE 状态恢复均通过。项目 adaptive runtime 模块未加载，
训练恢复日志通知列表为空。每秒采样 GPU 峰值 480 MiB、进程树 RSS 1,283,600,384
字节（约 1.20 GiB），退出后 GPU 为 17 MiB / 0%。CPU 小图也较慢，属于保底功能验收，
不代表大尺寸 CPU 吞吐已经可用。没有改精度、形状或 mock 主机预算以获得通过。

## 开放范围与后续边界

配置见 [中途恢复说明](../experimental/adaptive-training-recovery.md) 与
[预览解码策略](../configuration/sample-preview-memory.md)。

中途恢复保持单进程、workers=0、accumulation=1、整数 seed、标准磁盘缓存数据集的
实验合同。当前拒绝训练采样/验证 schedule，未完成两个功能在完整训练采样后 backward
与中途恢复联合开启的验收。Krea native 自动 retry 不开放；其他模型/适配器/多 worker
不从本次 Anima 证据外推。自然高分辨率预览 OOM、Turing、长窗口和质量未验收。

每步快照约 60 MiB（本次 rank=4），保存与数据身份哈希有 IO 成本；保留所有旧快照，
长训练前需要规划磁盘。运行摘要继续 `production_ready=false`；GPU 流程通过与完整状态
CPU 精确回滚，不构成长期质量或 GPU 浮点逐位一致承诺。
