# Anima MLP2 / GELU 与投影拆分探测

状态：2026-09-30，Luna 完成两批固定状态、sigma=0.5 的 CMP 90HX 单步探测。
前置：[局部 VJP 报告](anima_local_vjp_90hx_20260930.md)。生产默认、门槛及 OOM 策略不变。

## 结论

在本次三个采样 block 中，固定参考 cotangent、只改变 GELU 输入产生的 VJP 偏差
约 0.01665%--0.05003%；替换为候选 cotangent 时，早期 block 偏差约 147%。
因此，这些样本不支持把 GELU 输入的小幅舍入作为大梯度漂移的主要解释。
但传入 cotangent 为什么改变，仍未形成完整因果定位。

整网投影干预结果仍全部拒绝低精度候选：

| 使用 FP16 的普通 Linear | FP16 / FP32 数量 | Prediction rel-L2 | Input grad rel-L2 | LoRA grad rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| blocks MLP1 + MLP2 | 56 / 321 | 0.020736% | 110.704% | 40.379% |
| 仅 blocks MLP1 | 28 / 349 | 0.015922% | 232.619% | 85.482% |
| 仅 blocks MLP2 | 28 / 349 | 0.016014% | 41.813% | 15.323% |
| 全 FP32 控制组 | 0 / 377 | 0% | 0.0001894% | 0.0001599% |

误差均为百分数（原始比值乘 100）。MLP1-only 比两者同时 FP16 偏差更大，
是本固定样本的非单调结果，不支持逐层误差相加、按 FP32 数量二分或直接推荐 MLP2 FP16。
全 FP32 控制组通过，仅作为诊断控制，不是低精度方案通过。

## 固定条件

同一 BF16 来源 checkpoint（685 个 tensor）、同一真实单图缓存，GPU
`GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，CMP 90HX 10 GiB、SM86。
两批各自新录 FP32 reference；候选按 reference pre-step adapter/input/RNG 重放。
LoRA rank/alpha 4/4，初始 up/B 为零；AdamW lr=1e-4，microbatch1、full checkpoint、
swap20、torch attention、无 compile/TF32、无预览或生产 checkpoint。
所有候选 scale1024，reference scale1；单步 sigma 固定为 0.5。

输入 latent/noise `[1,16,1,150,112]`、prompt `[1,512,1024]`；保持文本 full padding，
latent mask 全零。两个 reference 的 snapshot/state/input/RNG SHA、初始 adapter SHA 相同；
跨批 prediction 逐位相同，但 191 个 tensor 存在数值差异，input gradient relative-L2
约 `1.496e-6`、adapter aggregate 约 `1.488e-6`（原始比值）。差异远小于低精度误差，
但不能声称跨进程逐位确定性。每批内部比较均核验捕获、身份和逐步 replay 证据。

## GELU 路径

新增 `--local-vjp-profile mlp-path` 捕获 blocks 0/5/27 的 `mlp.layer1/layer2`，
包括 base x/y、base 输出 cotangent g，以及真实输入 cotangent gx。
使用同一 64-token row-major 索引，保留 eager 5D 原 shape。
profile 必须确认目标 adapter up 为零、activation 是受支持的 GELU；
当前真实模型为 `approximate="none"`。

由初始 up=0 的实际合同，layer1 base y 对应 GELU 输入 z，layer2 x 对应其输出 a，
layer2 gx 对应 GELU 输出 cotangent gy，layer1 g 对应 GELU 输入 cotangent gz。
参考/候选两侧 `GELU(z)==a` 和 FP32 autograd VJP 与真实 gz 的采样 fidelity
均为零误差，支持本次捕获边界对应关系。

| Block | z drift | gy drift | gz drift | 固定 gy，只换 z 的 VJP 误差 | 固定 z，只换 gy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.02691% | 147.960% | 147.032% | 0.01665% | 147.034% |
| 5 | 0.02683% | 148.464% | 146.532% | 0.03724% | 146.531% |
| 27 | 0.04051% | 0.07102% | 0.08805% | 0.05003% | 0.07267% |

block0/5 gy cosine 分别约 -0.7974/-0.7210，说明不仅量级不同，方向也明显改变。
四组 VJP（reference / z-only / gy-only / both）是单方向采样对照，不能相减成
可相加的整网因果贡献，也不是完整 Jacobian / 最坏方向上界。

固定参考 x/W/g 的 MLP2 实际 FP16 autograd dx：

| Block | Scale1 rel-L2 | Scale64 rel-L2 | Scale1024 rel-L2 | Scale1024 非零 g 转零 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.743% | 0.1379% | 0.02760% | 826 |
| 5 | 3.163% | 0.05493% | 0.02524% | 65 |
| 27 | 13.94% | 0.3299% | 0.02602% | 5,636 |

同批 MLP1 scale1024 dx 误差为 0.03080% / 0.03004% / 0.06951%。六层、所有
测试 scale 无 overflow，仍有转零及非零次正规数。抽样只有 64 行，改变 GEMM shape；
局部误差小不能证明完整 block 反传或整网安全。尚缺 sampled 实际 gx 对照原形
FP16 重算 fidelity 的直接报告，不能将固定参考 VJP 直接视为候选真实反传。

## 投影拆分

`diagnose_anima_precision` 新增显式 case `mlp-layer1-only-fp16`、
`mlp-layer2-only-fp16` 和 `fp32-control`，以及仅一步可用的 `--single-sigma`。
原默认六组及最大八个 case 的预算不变；使用真实 Linear 名单补集生成 FP32 名称，
所有 case 均执行，不因控制组通过而输出生产计划。

L1-only 的 input/adapter cosine 为 -0.3788 / 0.6850；L2-only 为 0.9160 / 0.9883，
均不通过现有门槛。392 个 adapter 梯度清单相同，loss/gradient norm 有限，
scaler 保持 1024，每组完成一个有效 AdamW update，adapter delta 非零。
L1/L2-only 前八个 block 的 adapter 误差平方和贡献约 74.70% / 74.89%，
后八个约 0.000140% / 0.002388%；这仍只是误差位置，不证明源头在早期 block。

## 资源与产物

Luna 所有 worker 均走 target GPU UUID 的有界 supervisor，监控显存、RSS tree、
host available/reserve、swap、温度/时钟等，每 worker timeout150秒。

| 批次 / Worker | GPU used 峰值 | RSS tree 峰值（GB） | 最低 host available（GB） |
| --- | ---: | ---: | ---: |
| MLP-path reference | 5236 MiB | 19.71 | 88.69 |
| MLP-path candidate | 4714 MiB | 11.00 | 95.75 |
| MLP-path analysis | 990 MiB | 1.79 | 106.89 |
| split reference | 5236 MiB | 19.68 | 86.94 |
| split L1-only | 4972 MiB | 11.92 | 93.51 |
| split L2-only | 4972 MiB | 11.88 | 93.61 |
| split FP32-control | 5236 MiB | 19.70 | 87.08 |

无 OOM、遥测查询错误或保护停止。split 批次全机 swap 从 2,177,368,064 B 增至
2,236,874,752 B，主要增长出现在 reference/control；不能将全机统计归因于本任务
或单个投影。GPU 温度峰不超过 54C。结束 90HX 为 17 MiB、0%，无本任务残留。
捕获/IO 会影响步时，不据此宣传速度或显存优化收益。

成功目录：

- `output/runs/anima-adaptive-precision-90hx-20260930-mlp-path/`
- `output/runs/anima-adaptive-precision-90hx-20260930-mlp-split-retest/`

MLP-path 每侧六个 local-VJP safetensors 共 232,789,584 B，低于 512 MiB；
分析仅保留 GELU 所需采样，不将全量六层权重常驻路径列表。
`environment.json`、`summary.json`、capture/replay、local manifest、comparison、
gradient-diagnostics、worker/supervisor 日志及 telemetry 均保留。
另有 `...-mlp-split/` 在 worker 启动前因执行代理预建输出目录被 FileExistsError 拒绝，
只留资源记录，无有效训练；不能计作矩阵成功。复测使用全新目录，没有覆盖旧数据。

## 身份与验证

| 身份 | SHA-256 |
| --- | --- |
| weights | `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e` |
| inputs | `0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78` |
| probe | `4a1f022787cf994a66334129b4f5771f56f896558c2a4deab8b0859494aafc9a` |
| local helper | `d4eb9217a94c5d45ad01ca45f9469a378c666b926d0fcbff6e8cdc5cfad132c1` |
| MLP helper | `bdee14bdc87776a034e2f00968c0ceeb8ce439feff9fc87ce37f01ab2ec8485e` |

独立 Sol CPU 回归 646 passed、2 skipped、3 xfailed；修改文件 Ruff 通过。
MLP-profile 的真实 LoRA/non-reentrant checkpoint/train_steps record/replay/compare/analyze
六层三路径闭环通过。投影拆分入口独立定向 50 passed，Ruff 通过。
保留既有共享参数 xfail，不视为通过。独立审计确认两批精度分配、捕获身份及数值边界。
最终主代理 CPU 全 adaptive 回归为 655 passed、2 skipped、3 xfailed，退出码 0；
九个相关 Python 文件 Ruff 通过。热测后仅补文档，源码 SHA 与实测版本一致。

## 运行与下一步

```bash
.venv/bin/python -m bench.adaptive_runtime.diagnose_anima_local_vjp \
  --weights /path/to/anima-bf16.safetensors --inputs /path/to/inputs.safetensors \
  --output-dir output/runs/new-mlp-path --gpu-uuid GPU-... \
  --swap 20 --worker-timeout 150 --local-vjp-profile mlp-path

.venv/bin/python -m bench.adaptive_runtime.diagnose_anima_precision \
  --weights /path/to/anima-bf16.safetensors --inputs /path/to/inputs.safetensors \
  --output-dir output/runs/new-mlp-split --gpu-uuid GPU-... --swap 20 \
  --steps 1 --single-sigma 0.5 --worker-timeout 150 --max-cases 3 \
  --case mlp-layer1-only-fp16 --case mlp-layer2-only-fp16 --case fp32-control
```

CLI exit0 仅说明诊断完成，不能等同于 FP16 获准。下一步需要受控反事实区分
前向轨迹扰动与反向精度边界（保持相同前向、仅替换冻结基座反传计算作为诊断），
并评估误差与 optimizer 更新的关系；通过后再增加 sigma/bucket/短训质量覆盖。
反事实梯度不是原低精度图的生产梯度，不能未经独立验证接入训练默认。
