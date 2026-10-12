# 训练中途 OOM 恢复

状态：可运行实验，默认关闭。2026-10-04 已完成 CPU 恢复协议和 Anima 原生 FP16 的
CMP 90HX 短步真实 CUDA OOM 验收；尚未作长期稳定性、质量或其他硬件认证。

显式启用后，训练前向/反向遇到明确 CUDA OOM，会退出 worker，增加块交换并在新进程
恢复最近完整提交点。精度、有效 batch、基座身份与 AMP 策略保持不变。
未开启此开关时，原生 FP16 仍只允许首次 optimizer 开始前的启动重试。

## 配置

在符合原生 FP16 合同的 Anima 训练配置中增加：

```toml
adaptive_precision = "native_fp16"
mixed_precision = "fp16"
adaptive_oom_retry = true
adaptive_oom_retry_resume = true
seed = 20261003
gradient_accumulation_steps = 1
max_data_loader_n_workers = 0
cache_latents_to_disk = true
cache_text_encoder_outputs_to_disk = true
```

普通 LoRA、完整 checkpoint、torch attention、BF16 基座加载及既有传输策略等要求
继续遵守 [Anima 原生 FP16 合同](anima-native-fp16.md)。块交换初值和上限按模型、GPU
与主机内存预算设置，不应照搬某次实验的 swap=20/24。

使用现有 `train.py --config_file <配置>` 或 task 训练入口启动；CLI 同名开关为
`--adaptive_oom_retry_resume`。需要 torchdata 0.11.0，已列入项目依赖。
每次运行使用新的输出目录；已有 `.adaptive-recovery/` 时拒绝复用，保留旧记录。

## 支持范围

- 单进程、workers=0、accumulation=1、显式整数 seed。
- 标准 `DatasetGroup` / `DreamBoothDataset`，使用磁盘 latent 与 text cache。
  图片或缓存的 symlink 可以使用，但身份绑定到实际文件路径和内容。
- Anima native AMP 已有真实 GPU 短测；现有 FP16/FP32 islands 接线有 CPU 协议验证，
  不据此宣称所有模型族均已真机通过。Krea-2 native 的自动 OOM retry 仍拒绝。
- 拒绝公共 `resume`、阶段切换、自定义数据集、动态网络修改、训练采样/验证 schedule、
  首次采样/验证和自动块交换等未支持组合。普通手动续训不由此开关接管。

## 提交与恢复

每个完整更新边界保存 adapter、optimizer、scheduler、可选 scaler、训练计数、
主 Python/NumPy/Torch CPU/CUDA RNG、隔离的数据 RNG、实际 bucket 排列及 stateful
sampler cursor。AMP 跳步单独记录，global step 不代表有效更新数。
Accelerate 在 workers=0 时也会预取一批；提交游标锚定已交付的批次，恢复会重取未提交批次。

快照先写 `.recovery-*.partial`，完整性校验并 fsync 后原子发布为 `recovery-*`。
只允许加载 supervisor 自有目录中的已提交快照；拒绝路径穿越、快照 symlink、
缺失或损坏文件，以及数据/精度身份漂移。模型与 optimizer 的失败中间修改通过完整加载回滚。
快照不包含冻结 DiT 的完整副本，依赖冻结基座身份检查和重新加载。

恢复 Accelerator 状态与游标/RNG 后，重新执行 offloader 设备放置和 swap prepare，
再检查 FP32 trainables 与 optimizer 状态的设备/dtype、native guard 和计数。
恢复启动日志隔离 Rich 超链接 ID 对 Python RNG 的消耗；训练、数据和 scheduler 的
实际随机调用不在隔离范围内。耗尽游标的 lazy restore 在下一 epoch 切换前完成。

只对明确 CUDA 容量错误增加 swap。optimizer/checkpoint 阶段 OOM、CPU/host OOM、
非有限值、非法访问、精度漂移、超时、取消、损坏快照和预算不足均停止。
达到最大尝试数或交换上限也停止；更多交换块不保证解决任意 OOM。

## 输出与成本

`output_dir/.adaptive-recovery/summary.json` 记录各次尝试、swap 和停止原因；
`attempt-*/worker.log`、`result.json` 与 `recovery-*` 保留 worker 日志和完整快照。
原生精度报告继续写入 `precision_report.json`，中途恢复范围单独标识。
运行报告仍标记 `production_ready=false`，不能把流程成功当作质量认证。

当前每个完整更新边界保存一次，不自动删除旧快照。GPU 短测的 rank=4 adapter
单个快照约 60 MiB；200 个同等快照约 12 GiB，rank 和 optimizer 不同会改变大小。
数据身份哈希、逐步保存、重新加载基座和重建 swap 都有 IO/时间成本；本轮不承诺稳态加速。

## 验收

CPU 使用真实 Accelerate、AdamW、scheduler，验证 fresh PID、失败后部分权重/moments
污染回滚、预取、epoch 边界、scaler/RNG 和下一输入的精确一致。定向收尾 209 项通过，
独立 Ruff/Pyright 与真实快照审计通过。

Anima 90HX 两组各 3 步对照：第 1 步提交后真实 CUDA 分配 OOM，新 worker 将 swap
20→24，完成有效更新 3、跳步 0。三个提交点的 cursor、全 RNG、scaler、scheduler
共 12 项逐字段一致，AdamW moments 和最终 adapter 全有限。
GPU 浮点权重不承诺逐位等价，也未验证长期质量和 Turing。
完整证据及保留的失败现场见 [交付验收](../findings/anima_committed_oom_preview_delivery_20261004.md)。

## 预览回退

预览解码使用独立的 `sample_vae_decode` 策略和事件日志，不改变训练精度/batch，
也不请求 supervisor 重启。`auto` 解码阶段依次尝试目标设备、分块 GPU、预算允许的
分块 CPU；详情见 [预览内存策略](../configuration/sample-preview-memory.md)。
当前 committed recovery 范围拒绝训练采样 schedule，因此两个功能尚未作为训练中
联合开启的完整采样/恢复组合验收。
