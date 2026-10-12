# Anima block17 局部post-RoPE K替换

状态：2026-10-01。前置：[K激活与梯度分布](anima_k_activation_distribution_90hx_20261001.md)。
Sol实现、其他Sol独立审查、Luna执行指定CMP90HX受监控重放，主代理修复并集成，
另一Sol核验最终产物。复用已有真实捕获，不启动训练，不改变生产精度/OOM策略。

## 结果和解释

固定参考Q/du和FP32策略，分别使用参考/候选V。从参考K出发，仅在选定token/head
位置替换为候选post-RoPE K；baseline/full直接选择原端点。每V共9条件，完整4200tokens。
以下距离均为返回fused梯度c的L2，除以同V下baseline到full的距离。
两个距离比不能相加解释为贡献份额；接近full也不表示候选精度更准确。

| 替换范围 | 包含K漂移平方能量 | R距baseline/full响应 | R距full/full响应 | C距baseline/full响应 | C距full/full响应 |
| --- | ---: | ---: | ---: | ---: | ---: |
| head13 | 10.2414% | 87.9858% | 47.5237% | 90.9472% | 41.5766% |
| token3343 | 77.7518% | 98.3078% | 27.5137% | 95.9026% | 36.2736% |
| 激活误差top42 tokens | 95.9954% | 99.9360% | 11.7888% | 99.8750% | 15.1400% |
| 低误差42 tokens | 0.00293055% | 0.586636% | 99.9987% | 0.752114% | 99.9980% |
| 最低误差单token | 0.0000107879% | 0.065679% | 约100% | 0.084175% | 约100% |
| 最低误差head | 3.71774% | 2.19533% | 99.9759% | 2.60058% | 99.9662% |
| head13∩token3343 | 8.34446% | 86.9362% | 52.5280% | 88.1790% | 51.2757% |

baseline距baseline为0、距full为完整响应；full反之。完整响应绝对L2分别为
1.87101861e-6/1.45959941e-6；baseline梯度范数约1.97963e-4/1.97988e-4。
报告保留q/k/v/fused各分量绝对距离及分母，不只输出上表归一值。

在固定捕获下，局部K变化确实改变真实attention VJP；高误差token集合与同规模
低误差集合响应明显不同。head13只有约10%激活漂移能量，却产生较大的梯度响应，
说明不能只按激活误差大小推断响应；该对照未控制head间漂移幅度相同，不能证明
head13具有跨输入的固有敏感性。top42的响应幅度接近full，但距full仍有约12%--15%，
不能忽略方向差异、非线性或V背景作用。

下一步优先在候选K背景做反向恢复：恢复head13/token3343/top42及低误差对照为参考K，
检查是否仍接近参考响应；这与本轮从参考背景加入漂移不等价。随后才考虑局部上游
计算/精度干预及整网梯度验收。尚未定位最初FP16舍入来源，不给出生产安全名单。

## 实现和验收边界

新增`anima_k_local.py`、`diagnose_anima_k_local.py`和定向测试。
mask取最终activation-r2的stable rank，绑定report/source/code/weights SHA；
重建实际GPU K漂移后核对总量与逐head/token份额，并检查排名逆序的舍入容差。
份额rtol1e-5/atol1e-10、排名份额逆序atol1e-10；仅用于CPU/CUDA FP32重建差异，
没有放宽既有梯度fidelity门槛。记录实际mask索引/SHA、包含漂移份额与输入tensor SHA。
FP32 torch dispatch、TF32关闭，每条件两次forward逐位一致；重复VJP最大rel-L2
9.07026e-8。baseline/full对旧matrix和interpolation端点、参考历史c校验最大9.84560e-8，
均低于既有1e-4；参考core前向逐位一致。

首轮独立审查发现mask排名未绑定实际逐轴漂移、坏输入创建输出目录过早，已修正。
首轮热测exit1，因输入hash误用fused6144 reshape helper，在case执行前失败；
改为普通CPU tensor helper并补小形状测试，失败目录原样保留。
最终关联18 passed；修正独立审查相关25 passed，Ruff通过。

最终产物审计核对summary/result/supervisor一致、9×2条件、18replay和9endpoint完整；
源码/driver/6份上游summary/activation和distribution SHA一致。
独立NumPy从落盘rank重建mask，索引/数量/SHA全部吻合；实际漂移平方总量
381.316541839与CPU报告381.316556571一致于舍入容差。
qkv/fused平方分区、baseline/full零距、三角不等式及有限值检查通过。
独立定向9 passed、Ruff通过，未发现P1/P2。

没有保存新的c张量，只存梯度SHA和统计，因此本轮审计不能独立复算局部attention
backward；重复VJP不等于独立数学审计。本次产物审计未重新哈希大模型，依赖已通过
worker验证的权重SHA链。结论只适用于本次block17固定Q/du、两V背景和已有输入。

## 运行和资源

最终目录：`output/runs/anima-adaptive-precision-90hx-20261001-k-local-r2/`。
外层exit0，worker/supervisor ok、returncode0，summary completed、analysis analyzed。
worker监控233.241秒、213samples；GPU峰值1130MiB、进程树RSS峰值5,123,088,384B，
主机available最低100,273,999,872B，swap0、telemetry错误0。
结束后指定GPU17MiB/0%、43°C，无owned worker或该卡compute进程残留。
只使用`GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，显示GPU和功耗/风扇未操作。

```bash
timeout 540 .venv/bin/python -m bench.adaptive_runtime.diagnose_anima_k_local \
  --matrix-run output/runs/anima-adaptive-precision-90hx-20261001-qkv-matrix \
  --qknorm-run output/runs/anima-adaptive-precision-90hx-20261001-qknorm-r3 \
  --core-run output/runs/anima-adaptive-precision-90hx-20261001-attention-core \
  --source-run output/runs/anima-adaptive-precision-90hx-20260930-upstream-vjp \
  --stage-run output/runs/anima-adaptive-precision-90hx-20261001-block17-paths-r2 \
  --run output/runs/anima-adaptive-precision-90hx-20261001-k-interpolation \
  --distribution-run output/runs/anima-adaptive-precision-90hx-20261001-k-distribution-r2 \
  --activation-run output/runs/anima-adaptive-precision-90hx-20261001-k-activation-distribution-r2 \
  --weights /path/to/anima-base-v1.0.safetensors --output-dir output/runs/<fresh-k-local> \
  --gpu-uuid GPU-806debd9-d254-da57-d515-ccae15b4ac6a --analysis-timeout 480
```
