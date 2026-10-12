# Anima 残差流边界 cotangent 定位

状态：2026-09-30，Luna 完成三组有界单步热测，Sol 独立加载实际 tensor 并用 FP64 范数复算。
前置：[同前向 FP32 base-dx 反事实](anima_backward_counterfactual_90hx_20260930.md)。
仅增加 bench 诊断；生产默认、数值门槛和 OOM 回退不变。

## 结论

按反向传播顺序 boundary28 → boundary0，候选采样梯度偏差从 0.071% 逐步增大，
boundary14 首次超过 2%，随后在 boundary11 → 10 和 9 → 8 明显放大。
FP32 重放控制组最大采样偏差仅 0.000350%，远小于候选漂移。
这提供了下一轮单 block 对照的范围，但没有确定误差源头或安全 FP16 层名单。

boundary0 是 block0 输入；boundary i（1..27）是 block(i-1) 输出、block i 输入；
boundary28 是 block27 输出。因此 boundary15 → 14 穿过 block14 的反传，
boundary11 → 10 穿过 block10，boundary9 → 8 穿过 block8。
相邻 relative-L2 的分母和方向均会改变，不能把它们的差视为该 block 的因果贡献。
超门槛处也不等于偏差最初产生处。

| Boundary（反向顺序节选） | 候选梯度 rel-L2 | 候选 cosine | 激活 rel-L2 | 控制组梯度 rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| 28 | 0.070952% | 0.99999975 | 0.014449% | 0% |
| 20 | 0.382012% | 0.99999270 | 0.047371% | 0.000248% |
| 17 | 0.709087% | 0.99997490 | 0.040880% | 0.000251% |
| 16 | 0.839057% | 0.99996482 | 0.038253% | 0.000257% |
| 15 | 1.075988% | 0.99994211 | 0.041667% | 0.000265% |
| 14 | 2.166390% | 0.99976608 | 0.036358% | 0.000267% |
| 13 | 2.884494% | 0.99959491 | 0.036656% | 0.000281% |
| 12 | 3.810555% | 0.99928455 | 0.023855% | 0.000300% |
| 11 | 6.989916% | 0.99755416 | 0.023268% | 0.000310% |
| 10 | 35.402514% | 0.93644161 | 0.023464% | 0.000325% |
| 9 | 50.831860% | 0.86719720 | 0.023602% | 0.000350% |
| 8 | 94.766918% | 0.49432112 | 0.023356% | 0.000337% |
| 7 | 104.435768% | 0.36910589 | 0.023380% | 0.000328% |
| 2（最大偏差） | 122.335643% | 0.08483223 | 0.023533% | 0.000281% |
| 0 | 115.896514% | 0.20284571 | 0% | 0.000287% |

百分数为原始比值乘 100。全 29 边界表在 `boundary-analysis.json`；观测梯度是固定
512 token 样本，不是全部 token 的整网数值上界、完整 Jacobian 或训练质量证书。
boundary0 的激活相同而梯度不同，说明不同后续轨迹的反传可以回到相同入口并产生大漂移。

整网既有 comparator：

| 对比 FP32 reference | Prediction rel-L2 | Input grad rel-L2 | Adapter grad rel-L2 | 实验门槛 |
| --- | ---: | ---: | ---: | --- |
| MLP-only FP16 | 0.020736% | 110.931% | 40.455% | 拒绝 |
| 全 FP32 重放控制 | 0% | 0.000276% | 0.000177% | 通过 |

## 捕获合同与固定条件

新增 `anima_block_boundaries.py` 和三 worker `diagnose_anima_boundaries.py`；
probe 通过显式 `--capture-block-boundaries` 开启，限制单步 sigma=.5。
block 的外层 forward hook 捕获入口及每块输出，tensor hook 只复制采样并返回原梯度。
full non-reentrant checkpoint 直接重算 `_forward`，不会重复捕获外层 module 边界。
独立 CPU 测试包含 checkpoint 与 full-backward hooks，输出和输入梯度保持不变。

block swap 的 full-backward hook 引入 view，相邻张量可能不是同一个 Python 对象。
已改为同 storage pointer/offset、shape、stride、device、dtype、version 的严格别名检查，
拒绝 clone/copy，未修改 offloader。此修正不改变数值路径。

实际 eager shape 全为 `[1,1,75,56,2048]`，4200 token；每边界按相同 row-major
linspace 取512 token，保存 FP32 activation、除 loss scale 后的梯度及索引。
每侧29边界、87张量、tensor预算243,388,416 B（约232.11 MiB），
safetensors文件243,396,408 B，低于独立512 MiB上限。不保存基座权重或重算小形状 GEMM。
manifest、文件 SHA、布局、索引、finite、完整性及三侧 source/replay/初始adapter身份均核验。

同一 BF16 来源 Anima checkpoint、同一真实缓存图、GPU
`GPU-806debd9-d254-da57-d515-ccae15b4ac6a`（CMP90HX 10GiB）。
LoRA rank/alpha4、初始up为零，AdamW lr1e-4、full checkpoint、swap20、torch attention，
无 compile/TF32。Reference录 pre-step adapter/input/RNG，scale1；候选与控制重放，
scale1024，候选精确56个MLP Linear FP16/321 FP32，控制377 FP32。
三组各一步有效更新、loss/梯度 finite、scale稳定、adapter max delta约1e-4。
不保存生产 checkpoint/adapter，不生成 selected plan；`production_ready=false`。

结果目录：`output/runs/anima-adaptive-precision-90hx-20260930-boundaries-retest/`。
权重 SHA `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e`，
输入 SHA `0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78`；
当前 helper/probe SHA 与 environment、三侧签名匹配。控制组通过不代表 FP16 候选通过。

## 资源与失败记录

| Worker | Peak GPU MiB | Peak RSS tree GB | Host available min GB |
| --- | ---: | ---: | ---: |
| Reference | 5236 | 19.80 | 100.4 |
| Candidate | 4714 | 11.07 | 110.5 |
| Control | 5236 | 19.80 | 99.7 |

GB为十进制。三组累计148次内置遥测，telemetry failures=0，无OOM，整机swap使用为0，
最高温度53C；约3分07秒完成，三组returncode0。自有workers已退出，目标卡恢复17MiB/0%，
GTX1050显示卡未触碰。

首轮 `output/runs/anima-adaptive-precision-90hx-20260930-boundaries/` 保留失败证据：
reference forward因过强对象身份检查报 `Nonadjacent block boundary 1`，没有完成更新；
候选/控制未启动，不能计作通过或有效边界对照。修正后使用新目录与新reference重测。
首轮峰值GPU4094MiB、RSS19.65GB、遥测失败0；整机swap增加约302MiB，不能全归因于worker。

定向CPU回归44 passed，独立边界验收23 passed；Ruff与空白检查通过。
全adaptive回归689 passed、2 skipped、3 xfailed，exit0；已知xfail不计作通过。
下一步优先在 block14/10/8 固定同一输入和上游 cotangent 做单 block VJP 对照，
拆分激活轨迹改变与局部计算改变；此处只提出定位范围，不外推生产配置。

后续先完成[block14完整形状VJP](anima_block14_vjp_90hx_20260930.md)，
输入/cotangent/policy八组对照更指向传入cotangent差异，仍未构成源头归因。
