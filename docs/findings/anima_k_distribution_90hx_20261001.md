# Anima block17 K插值梯度的head/token分布

状态：2026-10-01。复用[K插值十格证据](anima_k_interpolation_90hx_20261001.md)，
Sol开发、Luna执行首个成功CPU分析，主代理修正来源校验并执行最终版本，
另一Sol独立直接NumPy复算。没有新增训练步数或GPU计算，不改变生产精度/OOM策略。

## 结果

固定参考Q/du，比较各V背景下K方向插值lambda0到lambda1的已保存返回梯度。
以下份额均为平方误差能量份额，不是元素数、参考梯度能量份额或因果贡献。
4200个token的top1%为按各自误差排序的42个token；两种背景的集合不保证相同。

| V背景 | fused rel-L2 | head13误差份额 | top4 heads份额 | top1% tokens份额 | top5% tokens份额 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 参考R | 0.945133% | 77.414965% | 98.074700% | 99.148729% | 99.622810% |
| 候选C | 0.737217% | 82.713856% | 97.555411% | 98.680987% | 99.399460% |

两背景fused top4 heads均为零基索引`[13,8,9,10]`，q/k/v各自最大head也为13。
fused/k/v的最大token均为3343，对应75×56 token网格的零基`(59,39)`；
q最大token为2623，即`(46,47)`。这些是token网格坐标，不是原图像素坐标。

参考V下q/k/v rel-L2分别0.353254%/1.255131%/0.853358%；
候选V下分别0.261064%/0.948070%/0.853358%。
fused绝对平方误差为3.50071047545e-12/2.13043034906e-12，
参考平方范数为3.91895441347e-8/3.91991586540e-8；
绝对量依赖本次捕获cotangent的尺度，不当作通用数值门槛。

这一保存梯度的变化高度集中，支持下一阶段缩小干预范围。它没有测量原始K激活
差异的空间分布，不能证明head13/token3343是最初FP16误差源头，也不能推荐精度名单。
下一步先用CPU比较实际post-RoPE K激活漂移与该梯度分布是否重合，再设计固定Q/du、
按head/token局部替换K的有界GPU干预；每个干预仍须重算真实VJP并保留整网验收边界。
后续[CPU激活重合检查](anima_k_activation_distribution_90hx_20261001.md)已完成：
最大token一致，激活top42覆盖约93.1% fused梯度误差；仍不证明最初误差源头。

## 实现与验收

新增`bench/adaptive_runtime/anima_k_distribution.py`及定向测试。
来源summary中的`shape=[1,1,75,56,2048]`是attention输出u；
落盘c是`[1,1,75,56,6144]`，按q/k/v→16heads×128拆解，不能混用两个shape。
对8组baseline、8组adjacent和5组V对照用NumPy FP64、128token分块计算。
输出绝对误差/参考平方范数、全部head/token份额、排名和浓度；零误差份额未定义时
输出null，近零分母的相对误差也显式置null。最多两格梯度同时驻留，不保存N²矩阵。

核验summary/result/environment、已落盘audit-cpu与summary SHA、统计数值一致性、
审计端点门槛及norm ratio/cosine自洽、现时代码SHA、每格manifest/SHA/shape/dtype/finite。
该工具依赖已有审计，不重新执行其源端点VJP或完整递归审计；报告显式保留此边界。
输出只允许fresh且与来源目录互不包含，要求`CUDA_VISIBLE_DEVICES=''`。

首个Luna运行exit1，因误将summary要求为6144而拒绝输入，未生成输出。
独立审查发现审计只比键名和端点指标可不自洽，均已修正并补测试。
最终定向interpolation/QKV/distribution回归13 passed，Ruff通过。
最终Sol不调用实现的统计helper，直接NumPy FP64复算两背景端点的q/k/v/fused能量、
全部份额、排名和浓度，rtol1e-11全部一致；源码/四份来源/四个端点文件SHA匹配。
独立定向7 passed、Ruff通过，未发现剩余P1/P2。相邻/非端点统计未独立逐格复算；
本轮不独立重算attention backward，不外推其他模型/数据/训练步。

## 运行与资源

最终目录：`output/runs/anima-adaptive-precision-90hx-20261001-k-distribution-r2/`，
`distribution.json` status analyzed、production_ready false，命令exit0。
耗时22.609秒，user CPU26.445秒、system CPU6.660秒，进程峰值RSS1,134,336KiB
（约1.08GiB）；4个Torch线程、CUDA设备为空。此处RSS为进程getrusage高水位，
不是主机总占用，也未新增GPU遥测。首个成功版本目录保留，不覆盖运行数据。

```bash
CUDA_VISIBLE_DEVICES='' timeout 120 .venv/bin/python -m bench.adaptive_runtime.anima_k_distribution \
  --run output/runs/anima-adaptive-precision-90hx-20261001-k-interpolation \
  --output-dir output/runs/<fresh-k-distribution>
```
