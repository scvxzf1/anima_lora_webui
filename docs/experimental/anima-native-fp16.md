# Anima 原生 FP16 训练入口

状态：普通入口已完成短训；OOM 回退默认关闭，2026-10-04 已补受限中途恢复的真实 CUDA 短测。
范围为 Anima 普通 LoRA；90HX 的
[200步 FP16/BF16 对照](../findings/anima_fp16_bf16_ab200_90hx_20261002.md)
支持继续使用原生 AMP，但不构成 Turing 或长期质量认证。

## 使用

在现有可用训练 TOML 中设置：

```toml
adaptive_precision = "native_fp16"
mixed_precision = "fp16"
adaptive_loss_scale = 1024.0
adaptive_fp32_modules = []
adaptive_oom_retry = false
attn_mode = "torch"
gradient_checkpointing = true
selective_checkpoint = "off"
torch_compile = false
full_fp16 = false
full_bf16 = false
```

通过原训练入口运行：

```bash
python train.py --config_file /path/to/training.toml
```

基础模型使用 FP16，保留原生 autocast、FP32 LoRA 参数和 Anima 的 FP32
残差/最终投影保护。`adaptive_loss_scale` 只配置新 scaler 的初始值；
Accelerate 续训时加载保存的 scaler，不重新设置初始 scale。
支持既有采样、保存和 `resume` 入口。首版拒绝 NF4、非普通 LoRA、
`dim_from_weights`、compile、非完整 checkpoint、多进程。
从普通 LoRA 文件加载现有权重无需打开 `dim_from_weights`。

`adaptive_precision = "auto"` 按实际 GPU 能力选择：安培及以上优先 BF16，
FP16 候选在 Anima 上使用 `native_fp16` 并选择 torch attention，较旧的
FP32 候选使用 FP32。硬件选择不会执行任务质量校准；旧卡实际 kernel、容量
和速度仍需真机验证。其他模型族保持原精度路由。
已冻结的旧 `fp16_fp32` 决策保留身份；显式 `fp16_fp32` 继续是独立实验模式。
默认 `off` 保持既有精度和 scaler 行为。

## 启动 OOM 回退

启动合同通过 CPU 夹具和真实 CPU 子进程验证；后续已完成
[中途恢复的真实 CUDA 短测](../findings/anima_committed_oom_preview_delivery_20261004.md)。
本节描述默认 `adaptive_oom_retry_resume=false` 的启动重试。
在上述配置基础上，使用已有完整磁盘缓存、新结果目录和固定 seed：

```toml
adaptive_oom_retry = true
seed = 42
gradient_accumulation_steps = 1
max_data_loader_n_workers = 0
use_vae_cache = true
use_text_cache = true
block_swap_transfer_dtype = "bf16"
block_swap_restore_mode = "foreach"
auto_block_swap = false
sample_at_first = false
adaptive_oom_retry_max_attempts = 4
adaptive_oom_retry_swap_increment = 2
adaptive_oom_retry_max_swap = 0
```

父进程冻结精度配置、设备/后端和本地基座/初始 LoRA 文件的 SHA256。
worker 启动及加载完成后核验来源内容，检查实际 dtype、AMP、残差保护和完整
checkpoint。来源核验有读取成本，不放在每步训练路径。

仅明确 CUDA OOM、阶段为 `model_load`/`forward`/`backward`、且首次
optimizer 尚未开始时，才启动新的 worker 并增加交换块。首次 optimizer
开始后，即使没有完成更新或发生 scaler 跳步，也不重试、不加载诊断快照。
CPU OOM、普通错误、未知状态、精度漂移、超时、取消和主机预算不足均停止。

交换块受真实结构 `N-2`、用户预算和加载后主机预算共同约束；增加交换块
不是一定能解决 OOM 的保证。默认指数递增，最后一次可直接尝试有效上限；
`max_attempts` 包含首次，`max_swap=0` 表示结构上限，不表示关闭交换。
传输设置中的 `bf16` 是既有原始 dtype 保留策略，FP16 权重仍保持 FP16。

首版拒绝输入 `resume`、初始步数偏移、`checkpointing_epochs` 自动续训、
首次采样、自动交换、基座额外合并、NF4、compile、非普通 LoRA及累积大于 1。
正常训练后的周期采样、验证、日志、adapter 和训练 state 保存保持可用。
不启用 OOM 回退时，普通原生 FP16 保存/手动续训不受此额外合同限制。

每次尝试的日志与结构化结果位于 `output_dir/.adaptive-recovery/attempt-*`，
总表为 `.adaptive-recovery/summary.json`。为保护历史，已有该目录时拒绝
复用；下一次请用新结果目录。精度报告另记录尝试、最终交换块及停止理由。
完整静态验收见[启动 OOM 报告](../findings/anima_native_fp16_startup_oom_static_20261002.md)。

另加 `adaptive_oom_retry_resume=true` 可启用已提交更新点的实验恢复，完整状态包含
sampler/bucket/预取、optimizer/scaler/RNG，并重新施加 swap 设备布局。
该模式要求磁盘缓存且拒绝训练采样/验证 schedule；详细范围和逐步快照成本见
[中途恢复说明](adaptive-training-recovery.md)。

## 训练报告

训练输出目录自动生成 `precision_report.json`，记录选择理由、实际参数 dtype、
AMP/scaler、FP32 保护、继承的 global step、本次尝试/有效更新/跳步、
显存和进程 RSS 观测及完成/拒绝/失败/取消状态。
只有最终保存后的完成事件才标记 `completed`；空数据或未创建网络为
`rejected`，数据调试为 `debug_completed`，其他正常提前返回为 `incomplete`。
同目录已有报告时，新运行生成 `precision_report.<run_id>.json`，保留历史。
仅主进程写入，文件原子发布；报告写入失败会告警，不改变训练异常。

global step 可能包含跳步边界，不能替代有效更新数。CUDA 峰值来自进程共享
统计器，RSS 是进程生命周期峰值；报告不重置统计器或逐步复制参数。
SIGKILL、进程崩溃和现有 profiler 硬退出可能留下 `running` 状态。
解析配置之前的 CLI 错误无法可靠确定输出目录，不保证生成训练报告。

旧严格数值矩阵仍为 0/12 通过，和本次实际训练观察分别记录。
当前入口不提供模型自动敏感层搜索或质量认证；更新后 OOM 恢复仅在上述受限实验开关开启时可用。
