# 实际混合精度训练与数据游标原型：2026-09-22

状态：实验 / 分阶段验证，非质量或长期性能认证
相关计划：[阶段开发计划](../proposal/adaptive_training_roadmap.md)
当前入口：[实验说明](../experimental/adaptive-runtime.md)
前序证据：[2026-09-21 实测与失败记录](adaptive_runtime_20260921.md)

## 本轮范围

用户恢复开发并授权使用 T10 16GB、RTX 3080 20GB。两卡均已可见。
T10 空闲，3080 有 Blender 进程及持续 GPU 负载；没有终止该进程或更改显卡设置。
初始整模 smoke 在 T10 执行，3080 先运行短小的 CUDA scaler/clip 状态恢复测试。
随后按用户补充请求，3080 又完成普通非量化 Krea BF16 三步实际训练，详见下节。
未下载模型、重建用户缓存、修改原数据集配置或启动长训练。

本轮交付两项分层结果：

1. 非量化 Krea 实际 `train.py` 路径完成 FP16/FP32 三次更新和原子状态保存。
2. worker=0 数据游标原型通过跨 bucket、跨 epoch、新进程续步对照。

补充交付：RTX 3080 20GB 的普通 BF16 实际训练基线，仍只属于链路 smoke。

没有把这两项拼接为“实际训练中途 OOM 已自动恢复”。生产训练 worker 仍然
`data_cursor_resume_supported=false`、`committed_checkpoint=null`。

## 真实训练输入

原真实缓存：`output/adaptive-runtime-20260921/krea-real-inputs.safetensors`，
SHA256 `e058dd82a7cf3c668a8898b2fdefd6265fefa88b15484999109f8007f83126a2`。
原缓存含真实图像的 Qwen VAE latent 和 Qwen3-VL 条件，详见相邻 JSON 来源记录。

新增 `bench/adaptive_runtime/training_fixture.py`：检查缓存、原图和 caption hash；
将缓存转换为训练 loader 的 `_0256x0256_anima.npz` 和 `_krea2_te.safetensors`，
通过生产缓存策略验证后输出独立 dataset TOML。Krea 的 latent 使用共享 VAE 缓存格式，
不是把 Anima text cache 改名冒充 Krea。hidden 必须可无损回转 BF16，否则拒绝转换。

图像副本使用与 `prepare_krea_inputs.encode` 相同的 EXIF transpose、RGB、
`ImageOps.fit(..., LANCZOS)` 后保存无损 PNG。已有 latent 不重采样，噪声由实际训练 loop
重新生成，不使用固定输入探针的 noise。目标目录必须全新，源图/文本/缓存只读。

```bash
.venv/bin/python -m bench.adaptive_runtime.training_fixture \
  --source output/adaptive-runtime-20260921/krea-real-inputs.safetensors \
  --destination output/adaptive-runtime-20260922/krea-fixture \
  --training-output output/adaptive-runtime-20260922/t10-training-smoke \
  --weights models/diffusion_models/krea2_raw_bf16.safetensors
```

本轮是一张真实图片、256x256、重复 3 次的链路 smoke，**不是**多输入、双 token-family、
多 bucket 或质量校准集。该范围不足以验收阶段计划 P1 的全部要求。

## T10 实际训练结果

```bash
CUDA_VISIBLE_DEVICES=GPU-82301866-a770-fb04-0535-9a40fa990311 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  timeout -k 15 650 .venv/bin/python train.py \
  --config_file output/adaptive-runtime-20260922/krea-fixture/train.toml

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
  .venv/bin/python -m bench.adaptive_runtime.audit_training \
  output/adaptive-runtime-20260922/t10-training-smoke
```

| 项目 | 结果 |
| --- | --- |
| 训练入口 | CLI supervisor -> fresh worker -> AnimaTrainer 实际 Krea session/loop |
| 精度 | 264 个普通 Linear FP16；非 Linear/残差 FP32；LoRA 参数 FP32 |
| FP32 Linear 名单 | 空，仅用于最小链路 smoke，不是推荐数值计划 |
| LoRA | plain、rank=4、alpha=4、196 个模块 |
| 训练配置 | AdamW、lr=1e-4、constant scheduler、full checkpoint、torch attention |
| 内存配置 | swap26、BF16 CPU masters、foreach restore |
| 有效更新 | 3，worker status=ok |
| CUDA 峰值 allocated | 4.134973 GiB |
| CUDA 峰值 reserved | 约 4.31 GiB |
| CPU masters | 22.64 GiB / 28 blocks |
| Worker 总耗时 | 335.48 秒，含加载和初始化 |
| 日志训练段 | 约 28 秒；不足以当作稳定吞吐结论 |
| OOM | 未发生，没有实际重试或 checkpoint resume |

初始化阶段有明显主机内存压力与 swap 活动，不能用训练段时间掩盖端到端成本。
主机可用内存保护保持原值；未使用磁盘 swap 充当可用 RAM 预算。

状态审计：

- 最终 adapter 与 `state-00000003/model.safetensors` 的 588 个张量逐位相同且全部有限。
- 392 组 AdamW 状态均为 step=3，动量张量有限。
- scaler scale=1024、growth tracker=3，scheduler last_epoch=3。
- 快照约 48.2MB adapter + 96.7MB optimizer，未把整份冻结 DiT 保存进去。
- 只保留最近的 `state-00000003`；完整文件清单与精度名单保存在快照内。
- `training-audit.json` 明确 `precision_calibrated=false`、`data_cursor_resume_supported=false`。

证据根：`output/adaptive-runtime-20260922/t10-training-smoke/`。
需保留其隐藏目录 `.adaptive-recovery/` 才能重新执行状态审计。

## 补充：3080 20GB BF16 实际训练

用户明确要求补测后执行，未停止持续占用 GPU 的 Blender，未改功耗、频率或风扇。
启动前 3080 约占用 2GB 显存、95% GPU 利用率，因此此次是**并发负载下的可运行性基线**，
不是空闲卡性能基准。按 UUID 选择 GPU，设备报告 SM 8.6、原生 BF16 支持。

复用上述同一图片/缓存、256x256、3 repeats、seed=20260922、rank/alpha=4、AdamW、
lr=1e-4、constant scheduler、full checkpoint、swap26、torch attention、不开 compile。
普通 `train.py` 入口设 `mixed_precision=bf16`、`adaptive_precision=off`、
`adaptive_oom_retry=false`、`full_bf16=false`。冻结 DiT 参数为 BF16，LoRA 参数保持 FP32；
这不是所有算子/状态一律 BF16，也不是 NF4 实验。普通 BF16 不使用 FP16 loss scaler。

新增 bench 包装器只负责隔离进程、600 秒预算、复用原有主机保护和输出审计，未改生产训练代码。
它通过 `runpy` 执行实际 `train.py` CLI，保存普通 Accelerate 状态；不使用要求 FP16 scaler 的
adaptive worker，也不放宽该 worker 的契约。输出目录必须全新，源配置与缓存不修改。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -m bench.adaptive_runtime.bf16_training_smoke \
  --source-config output/adaptive-runtime-20260922/krea-fixture/train.toml \
  --output output/adaptive-runtime-20260922/3080-bf16-training-smoke \
  --gpu GPU-8d47b1e2-27f2-8f05-f3d6-9e81562e1086 --timeout 600

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
  .venv/bin/python -m bench.adaptive_runtime.bf16_training_smoke \
  --audit output/adaptive-runtime-20260922/3080-bf16-training-smoke
```

| 项目 | 实测 |
| --- | --- |
| 训练结果 | 3 次有效更新，progress `run_end=ok`，未出现 OOM |
| 冻结 DiT 参数 | 12,820,073,036 个，BF16 |
| 可训练 LoRA 参数 | 12,042,240 个，FP32；196 个模块 |
| CUDA peak allocated / reserved | 3.667349 / 3.775391 GiB，本训练进程 PyTorch 分配器统计 |
| CPU masters | 22.64 GiB / 28 blocks，BF16，见 worker.log 的 offloader 初始化记录 |
| 主机 MemAvailable 最低值 / 保护线 | 约 25.46 / 6.26 GiB；未触发保护 |
| 初始化至 `loop_ready` | 约 177 秒，memory probe 时钟 |
| Worker 总耗时 | 222.00 秒，含初始化、训练、保存和首次审计失败/退出 |
| 后两段 step 事件间隔 | 9.125 / 8.538 秒，含日志等开销，不是纯计算或热稳态 |
| `avr_loss` | 0.197026 / 0.172936 / 0.169428；累计平均，不是逐步原始 loss |
| 状态审计 | 588 个 adapter 张量全部有限，最终文件与状态快照逐位一致 |
| 更新进度 | 392 组 AdamW 均 step=3，scheduler last_epoch=3，train_state current_step=3 |
| Scaler / RNG | 无 scaler 文件符合 BF16 预期；普通 RNG 文件存在，未做恢复重放 |

### 审计脚本修复与证据边界

第一次训练本身成功，但包装器的首版审计错误假定无 tracker 的 progress 也有
`loss/current`，结束后抛 `KeyError`，因此原始 `summary.json` 和 supervisor 保留 `status=error`。
源码核验发现该路径实际输出 `avr_loss`。修复字段读取与指标标注，补齐真实日志形状回归后，
对**同一批原始产物**独立审计通过；没有重训、修改权重或覆盖最初失败记录。
`training-audit.json` 的 `status=ok` 表示事后状态审计成功，同时显式带有
`execution_summary_status=error` 和原始异常。不要把这一轮描述为包装器首次端到端零错误。

主机初始化伴有 swap 活动，计数是机器级而非本进程独占；磁盘 swap 从未计入可用 RAM。
与 T10 混合 smoke 的差异除了硬件、计算精度和负载，还有保存频率：T10 每步提交，
本次只在训练结束保存普通状态。不能用两组时间推导精度加速比，也不能从三步平均 loss
下降推导收敛或质量。当前**未完成同卡 BF16 / FP16+FP32 / FP32 的实际训练 A/B**，
本轮没有 OOM 注入、自动重试或检查点恢复测试。

证据根：`output/adaptive-runtime-20260922/3080-bf16-training-smoke/`，包含 `train.toml`、
`device.json`、`progress.jsonl`、`memory.jsonl`、`summary.json`、`training-audit.json`、
`bf16-smoke.safetensors`、`bf16-smoke-state/` 和 `.benchmark/attempt-000/worker.log`。
设备原生支持、实际参数 dtype 与保存张量分别核验，不能只凭配置字符串认定 BF16 已运行。

补测代码的最终定向回归：**46 passed、1 skipped、14 个既有 torch JIT 弃用警告，18.36 秒**。
覆盖 `test_adaptive_bf16_training_smoke.py`、训练 fixture/audit、隔离进程生命周期及
`test_adaptive_training_precision.py`；该批显式禁用 CUDA，跳过的是小模型 CUDA 条件测试，
不影响上面的实际 3080 训练证据。XML：`output/adaptive-runtime-20260922/bf16-smoke-tests.xml`。
新增两文件 Ruff 通过。训练进程已退出，Blender 和其它用户进程未终止。

## 数据恢复原型

### 依赖与范围

当前 Accelerate 1.13.0 的 `DataLoaderConfiguration` 和 `DataLoaderShard` 已支持
`use_stateful_dataloader`。本轮在现有 `.venv` 单独安装 `torchdata==0.11.0`，未升级其它包，
未改 pyproject 或 lock。它当前仅供 bench 原型测试，缺依赖时相关测试明确 skip。

```bash
uv pip install --python .venv/bin/python --no-deps torchdata==0.11.0
```

原型留在 `bench/adaptive_runtime/`，未导入生产训练路径：

- `bucket_state.py` 保存实际 DatasetGroup/BaseDataset 所需的桶顺序、batch indices、
  largest bucket、epoch/step；恢复前校验结构与样本重复次数，全部验证后才修改各 dataset。
- `stateful_dataset.py` 使用 torchdata 自带 stateful RandomSampler/BatchSampler，
  将数据读取的 Python、NumPy、torch CPU RNG 与训练 RNG 隔离。
- 数据 view 在 getitem 前设置 epoch/step，避免沿用旧 collator 在 getitem 后才设置的时序。
- loader 状态由 Accelerate 的真实单批 look-ahead 路径捕获，不是多 worker 队列预取。

### 实际发现和修正

最初只打开 stateful loader 但保留普通 PyTorch RandomSampler，恢复后样本顺序不同。
这证明只有 loader 开关不够，sampler 本身也必须能恢复游标。

切换为 torchdata 的状态化 sampler 后，另发现 epoch 末尾 lazy restore 会在设置新 epoch
之后覆盖 dataset 的新 shuffle。原型先物化 torchdata 的延迟状态恢复，再协调 Accelerate
iteration；末批已消费的状态将进入下一 epoch，非末批则继续当前 epoch。

测试直接断言：用户已消费 K 批时，保存的 dataset fetched=K，而活跃 dataset 通常
fetched=K+1，证明覆盖了 Accelerate 单批前瞻，不只是在普通 loader 上跳过样本。

### 验证

`tests/test_adaptive_data_cursor.py` 使用实际 BaseDataset epoch/shuffle、BucketManager
和 DatasetGroup 方法，构建两个 dataset、每个两个 bucket、包含重复样本的 CPU 小 fixture。
getitem 的 payload 是用于确定性校验的随机张量，不是完整 Krea 真实缓存数据。

- 第 1、5、11、12 批保存，跨 epoch 恢复后的 sample ID、bucket、step 与随机 payload 逐位一致。
- 在第 5 或 12 次更新保存，**新 Python 进程**恢复训练到第 18 次更新。
- 与不中断对照相比，后续输入、target、loss、模型、AdamW、scheduler 均逐位一致。
- 桶成员/重复次数变化、损坏 schema 等拒绝，不在部分修改后才发现另一 dataset 不兼容。

这推进了 P2 的选型与最小协议，但不代表 P2 完整验收。仍缺：真实缓存内容/底模身份、
完整训练恢复接线、scaler overflow 与有效更新计数协调、真实多 bucket 训练对照。
worker>0、分布式、CUDA 数据变换、动态数据集、stage schedule 均不支持。

## CUDA 与回归证据

本轮暂停续跑首先完成 44 项训练 supervisor/loop 回归，覆盖此前最后的 0600 文件创建改动。
新 fixture 转换、审计与数据游标定向测试共 15 项通过。

T10 和 3080 分别运行：

```bash
CUDA_VISIBLE_DEVICES=<GPU UUID> OMP_NUM_THREADS=4 timeout 60 \
  .venv/bin/python -m pytest \
  tests/test_adaptive_training_precision.py::test_real_accelerate_scaler_clip_and_resume -q
```

两卡各 1 passed，分别约 6.40 / 6.89 秒。验证小型生产 LoRAModule 与 Accelerate scaler、
unscaled clip、保存后重放下一步，不外推为 3080 整模型训练完成。

XML 位于 `output/adaptive-runtime-20260922/`：
`p0-restart-tests.xml`、`data-cursor-tests.xml`、`t10-scaler-tests.xml`、`3080-scaler-tests.xml`。

最终扩展 CPU 回归：**464 passed、2 skipped、25 warnings，44.25 秒，退出码 0**。
范围为 `tests/test_adaptive_*.py` 加训练 loop/bootstrap、兼容矩阵、Web preflight、
混合精度 resolver、Krea attention、配置和分布式梯度同步；XML 为
`training-regression.xml`。14 个警告来自既有 torch JIT 弃用，11 个来自 torchdata
对当前 PyTorch `set_vital` 弃用接口的调用。两项 CUDA 条件测试在此 CPU 批次跳过，
不等于本轮所有 CUDA 测试都已双卡重跑；本轮双卡独立执行范围仅为上述 scaler 测试。
新增文件 Ruff 与文档 diff 检查通过，本任务测试和训练进程均已结束。

## 下一批次

1. 在独立真实多 bucket 缓存集上接数据 view，核验 sample/caption/latent 与 timestep/noise 摘要。
2. 建立底模、缓存和有效配置身份清单；通过后再把数据游标状态纳入原子训练 checkpoint。
3. 加上实际 CUDA scaler 状态，比较完整实际 loop 的不中断与新进程恢复。
4. 保持精度名单不变，才开放更新后 OOM 从已提交点恢复，并用真实分配失败验证。
5. 同步推进显式敏感 Linear 对照及多输入数值验证；本轮未解决历史梯度误差和无加速结论。
