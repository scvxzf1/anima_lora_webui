# Anima block17 K激活漂移与梯度分布重合

状态：2026-10-01。前置：[已保存梯度分布](anima_k_distribution_90hx_20261001.md)。
复用既有core QKV激活、真实norm参数与捕获RoPE，在CPU FP32重建post-RoPE K，
用NumPy FP64统计误差分布。没有新增训练步数、GPU计算或attention backward。
生产精度默认、门槛和OOM策略保持原有行为。

## 结果

参考/候选K激活rel-L2为0.546329%，误差平方能量381.316556571，
参考平方能量12,775,456.7402。绝对量依赖本次捕获，不作为通用门槛。
head13激活误差份额10.2414%，仍为最大head，但head间较分散；
其后head3/6/0/14份额约7.91%/7.16%/7.09%/7.01%。
这不同于返回梯度误差在head13占77%--84%的集中性，不能将两者的份额混为一谈。

token3343为最大激活误差位置，承载77.7518%激活误差能量，与K/fused梯度最大位置相同。
它是75×56 token网格的零基`(59,39)`，不是原图像素坐标。
其次token2857占9.4390%；激活top1%/5%/10% tokens分别占95.9954%/98.1681%/98.6498%。
top1%各取42个token，按各自误差独立排序；以下为集合交集与交叉能量覆盖：

| V背景/返回梯度 | top1%交集数 | 激活top42内的梯度误差份额 | 梯度top42内的激活误差份额 |
| --- | ---: | ---: | ---: |
| 参考V / K | 15/42 | 98.5479% | 91.6743% |
| 候选V / K | 15/42 | 97.8494% | 91.7557% |
| 参考V / fused | 13/42 | 93.1056% | 91.6503% |
| 候选V / fused | 13/42 | 93.1344% | 91.6822% |

交叉覆盖的分母分别是梯度和激活各自的总平方误差，不是共同分母。
集合并不完全重合，但主误差能量显著重合；head13也是四种比较的最大梯度误差head。
这些结果支持下一阶段用head13/token3343及top42集合设计局部K替换，
并加入同规模低误差集合对照。尚不能区分最初舍入来源、attention敏感性或cotangent
分布的作用；空间关联不能代替实际干预，更不能给出FP16放行或安全层名单。
后续[局部K替换](anima_k_local_90hx_20261001.md)已完成：top42真实VJP响应接近full，
但方向残差仍约12%--15%，低误差集合响应小；依然不等于整网精度认证。

## 实现和验证

新增`bench/adaptive_runtime/anima_k_activation_distribution.py`和定向测试。
复用完整证据链验证core p、真实RMSNorm参数、conditioning/RoPE、权重/代码SHA；
只做norm/RoPE CPU前向，保留完整4200tokens/padding，K形状`[4200,16,128]`。
读取现有插值端点c，分别计算K分量和q/k/v平方能量求和的fused head/token分布。
按head/token的top1/5/10%集合输出交集及双向覆盖；16heads中top1%和top5%均取1head，
top10%取2heads，不把这些不同百分数误认为不同大小的前三组集合。

来源与现有distribution报告SHA绑定，端点能量、份额、排名、浓度、零误差状态逐项复算。
输出须fresh、与来源目录互不包含，要求`CUDA_VISIBLE_DEVICES=''`，记录资源和代码SHA。
任一分布零总能量时排名交集未定义，返回null，不用默认索引伪造重合。

首版全尺寸CPU运行exit0，但独立审查指出缺fused比较、零能量排名及部分报告字段核验不足。
修正后关联测试16 passed、Ruff通过；另一Sol独立代码审查无P1/P2，
联合activation/distribution定向13 passed。首版报告保留，其代码SHA不能替代最终版本验收。

最终独立Sol核验真实权重、5份来源文件、5级chain summary、core p/RoPE及4个端点的manifest/SHA。
不调用本工具统计helper，NumPy独立实现RMSNorm eps1e-6及非交错rotate_half/RoPE。
独立FP32 K漂移能量381.316557691，与报告相对差2.94e-9；参考能量相对差3.23e-10。
FP64版本漂移能量381.316544089，FP32/FP64最大逐元素差1.73e-6。
完整head排序一致；完整token排序有38个位置受合法FP32算式舍入影响，首处在rank390。
top1%的42个token排序与集合完全一致，不能宣称完整排序逐位重现。
两V背景K/fused梯度能量一致，四组top1%集合交集和双向覆盖吻合，覆盖最大偏差5.22e-10。
本次不独立重算attention梯度，不验证其他步/模型/数据的因果或生产适用性。

## 运行证据

最终目录：`output/runs/anima-adaptive-precision-90hx-20261001-k-activation-distribution-r2/`。
`activation-distribution.json` status analyzed、production_ready false，命令exit0，未触发timeout420。
耗时164.012秒，user CPU236.245秒、system CPU77.213秒；进程getrusage峰值RSS
4,297,540KiB（约4.10GiB），4个Torch线程，CUDA设备为空。没有新增GPU遥测。

```bash
CUDA_VISIBLE_DEVICES='' timeout 420 .venv/bin/python -m bench.adaptive_runtime.anima_k_activation_distribution \
  --run output/runs/anima-adaptive-precision-90hx-20261001-k-interpolation \
  --distribution-run output/runs/anima-adaptive-precision-90hx-20261001-k-distribution-r2 \
  --weights /path/to/anima-base-v1.0.safetensors \
  --output-dir output/runs/<fresh-k-activation-distribution>
```
