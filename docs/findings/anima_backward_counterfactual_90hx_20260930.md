# Anima 同前向、FP32 base-dx 反事实

状态：2026-09-30，Luna 完成 CMP 90HX 三组有界单步热测，Sol 独立审计实际 tensor。
前置：[MLP2/GELU 与投影拆分](anima_mlp_path_90hx_20260930.md)。生产精度默认、门槛和 OOM 策略不变。

## 结论

原生 MLP FP16 与反事实的 prediction tensor 逐位相同，但将 MLP base-dx 改为
FP32 计算并未改善相对 FP32 reference 的大梯度漂移。仅在本固定样本、当前权重、
FP16 forward 不变的干预下，MLP base-dx 的 FP16 反传计算误差不是大漂移的主要解释。
这不排除前向舍入改变的激活/Jacobian、上游 cotangent 或其他反向路径，也不是完整根因结论。

| 对比 | Prediction rel-L2 | Input grad rel-L2 | Adapter grad rel-L2 |
| --- | ---: | ---: | ---: |
| FP32 reference vs 原生 MLP FP16 | 0.020736% | 110.7059% | 40.3654% |
| FP32 reference vs FP32 base-dx 反事实 | 0.020736% | 110.8807% | 40.4323% |
| 原生 MLP FP16 vs 反事实 | 0%（逐位相同） | 0.234584% | 0.086557% |

表中原始比值乘 100。两种 FP16 候选相对 FP32 reference 均不达实验梯度门槛
（rel-L2 <= 2%、cosine >= .999）；它们之间接近不代表低精度合格。
三组 finite、各完成一次有效 optimizer update，adapter max delta 约 1e-4；scale 未回退。

## 实验边界与实现

新增 bench-only `anima_backward_counterfactual.py`、`diagnose_anima_backward.py`。
custom autograd 接收原 FP32 x，内部执行与原 island 相同的
`F.linear(x.half(), current_fp16_weight, None).float()`；backward 以
`grad.float() @ current_fp16_weight.float()` 返回 FP32 dx。这里不恢复 BF16 来源信息，
也不直接改 LoRA 分支。上游 LoRA 梯度可以随 base-dx 的改变自然变化。

helper 安装于 precision islands 后、adapter apply 前；不保存 weight tensor，
backward 核验参数身份/shape/dtype/device，并读取换入后的当前权重。
保留 full non-reentrant checkpoint 与 swap20。CPU 测试覆盖真实 LoRA、scale、
checkpoint、模拟 weight.data 重绑与非法模式/身份拒绝。
三侧共享 helper SHA/suite；独立 driver 校验 mode、精度分配、输入/源码和 replay 身份，
先要求两种 FP16 prediction 逐位相同。没有放宽共享 comparator/replay 默认合同。

固定 BF16 来源 Anima checkpoint、同一真实缓存图，初始 LoRA up 为零，rank/alpha4，
AdamW lr1e-4、torch attention、无 compile/TF32。单步 sigma=.5；reference scale1，
两候选 scale1024，精确 56 个 MLP Linear FP16 / 321 FP32。候选重放 reference 的
pre-step adapter/input/RNG；不是独立优化轨迹或训练质量测试。

结果：`output/runs/anima-adaptive-precision-90hx-20260930-backward-cf/`。
权重 SHA `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e`，
输入 SHA `0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78`。
当前源码 SHA 与 environment/三侧签名匹配；实际 tensor 独立加载确认 prediction 相等。

## 资源与验收

| Worker | Peak GPU MiB | Peak RSS tree GB | Host available min GB | Whole-host swap delta MiB | 温度 C |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 5236 | 19.69 | 88.06 | +192 | 41--52 |
| Baseline | 4714 | 10.97 | 94.92 | 0 | 41--50 |
| Counterfactual | 4714 | 10.94 | 95.16 | 0 | 42--51 |

GB 为十进制，swap 是整机变化，不能完全归因于 worker。内置 supervisor 各采样45--51次，
telemetry failures=0，无 OOM，三个 worker 正常退出；结束目标卡17MiB、0%利用率，
显示卡未触碰。输出目录保留日志、capture、replay、对比及资源记录。

定向独立 CPU 验收32 passed；全 adaptive 回归666 passed、2 skipped、3 xfailed，exit0。
已知 xfail 不计作通过。scope 始终为诊断，`production_ready=false`，没有生成推荐方案。
下一步应沿 block 边界定位 cotangent 首次放大处，再做单因素干预，而非扩大训练或放宽门槛。

后续[29个残差流边界对照](anima_block_boundaries_90hx_20260930.md)已完成，
反向首次采样梯度超2%在boundary14，仍须单block固定输入/cotangent对照才能继续归因。
