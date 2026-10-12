# Anima block17 K差异方向插值与V背景

状态：2026-10-01。Sol实现、另一Sol独立代码验收、Luna执行GPU分析；
主代理接手CPU审计。前置：[Q/K/V八格对照](anima_qkv_matrix_90hx_20261001.md)。
复用已有真实捕获，不启动新训练，不改变生产精度默认、数值门槛或OOM策略。

## 结果和边界

固定参考post-RoPE Q、参考du/conditioning，以
`K(lambda)=K_R+lambda*(K_C-K_R)`，lambda取0/0.25/0.5/0.75/1，
分别搭配真实参考V和候选V。端点直接选原tensor，插值点显式FP32计算。
插值在post-RoPE空间进行，不代表完整模型实际产生的中间轨迹，也不保证仍在
RMSNorm输出流形上；用途是检查固定方向的attention梯度响应，不是新精度模式。

下表是完整返回梯度c相对各自lambda=0的rel-L2百分数：

| K lambda | V参考 | V候选 |
| --- | ---: | ---: |
| 0.25 | 0.365907% | 0.283853% |
| 0.5 | 0.625016% | 0.485028% |
| 0.75 | 0.810166% | 0.629851% |
| 1 | 0.945133% | 0.737217% |

相邻四段差异：V参考0.365907%、0.259998%、0.187945%、0.140854%；
V候选0.283853%、0.202169%、0.148044%、0.114155%。
五个采样点相对起点的误差递增、相邻响应减小，未见采样点间明显跳变；
这不证明整区间连续或给出全局稳定性界限。
中点梯度相对端点梯度FP64平均的偏差为0.160322%/0.125748%，
说明这组方向上的梯度响应不能直接线性插值。

同lambda只替换V，fused梯度差异依次为
0.437939%、0.369628%、0.329638%、0.307761%、0.296749%。
返回c_V逐位不变，变化在c_Q/c_K，符合固定Q/K/du下attention对V线性结构。
V背景继续影响K方向响应大小，不能把K/V对照相加成因果贡献。
候选V响应较小不等于候选更准确，分母/方向均有限定。

这一结果支持继续检查K差异的局部结构及其如何改变attention后向，而不是把现有
偏差解释成纯重放噪声或某一端点偶然跳变。尚未确定最初FP16误差源头，
也不能推荐生产安全层名单。下一步可按head/token定位K差异和返回梯度变化的分布，
先使用已有落盘证据作CPU分析，再决定是否需要新的GPU干预。

## 实现和验收

新增 `anima_k_interpolation.py`、`diagnose_anima_k_interpolation.py`、
`audit_anima_k_interpolation.py`、`tests/test_adaptive_anima_k_interpolation.py`。
所有格使用原torch dispatch、完整4200tokens/padding、FP32、无TF32。
每格两次前向逐位相同、重复VJP最大误差9.76824e-8，低于既有1e-4门槛。
Q固定R，端点绑定上一轮RRR/RRC/RCR/RCC实际c，只有lambda0/V参考对原core u逐位检查。
不将lambda1/V候选错误地当作CCC。

主代理在GPU前发现并修正前向u检查调用了fused-QKV宽度helper的问题，
改为普通CPU tensor helper后按原SHAPE恢复；定向3项再次通过。
独立Sol未发现阻断P1/P2，完整adaptive回归741 passed、2 skipped、3 xfailed；Ruff通过。
覆盖缺口：现有CPU测试未调用analyze完整编排，建议补真实形状映射/端点的编排fixture；
不声称此项已覆盖。

文件/统计/来源验收包括原matrix八格实际tensor及递归qknorm/core/stage/upstream/
weights/实时代码SHA、FP32/shape/finite、端点fidelity和统计复算。
十格文件各103,219,288B，合计1,032,192,880B，小于1GiB；单文件小于512MiB。
不保存N²矩阵。CPU审计使用NumPy FP64复算baseline/adjacent/V对照和midpoint，
及端点与源矩阵梯度fidelity；明确不独立重算插值格的attention backward。

## 运行证据与资源

目录：`output/runs/anima-adaptive-precision-90hx-20261001-k-interpolation/`。
analysis worker returncode0，supervisor status ok；父进程summary status completed。
外层`timeout 540`实际退出124，虽然已输出completed，不记为外层成功。
Luna因累计耗时超过审计启动预算未继续审计，主代理随后在CPU接手。
CPU独立审计exit0、status audited；四端点与源矩阵c误差6.29e-8--1.04e-7，
参考历史c误差7.50e-8。baseline/adjacent/V对照/midpoint和分量平方误差分区均通过。
外层热测退出124的事实仍保留，不以独立审计成功改写该退出状态。

worker监控296.43秒、249次遥测；GPU峰值1030MiB、RSS tree峰值5,626,359,808B，
最低主机available 101,503,397,888B，峰值48°C、swap0、遥测错误0，无OOM。
最终目标GPU17MiB/0%，owned worker退出。仅CMP90HX UUID
`GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，不操作显示GPU/功耗/风扇。
来源重验与统计CPU开销已接近外层预算，后续优先复用现有证据CPU定位，
避免把新增分析范围继续堆入同一个300秒worker。

```bash
timeout 540 .venv/bin/python -m bench.adaptive_runtime.diagnose_anima_k_interpolation \
  --matrix-run output/runs/anima-adaptive-precision-90hx-20261001-qkv-matrix \
  --qknorm-run output/runs/anima-adaptive-precision-90hx-20261001-qknorm-r3 \
  --core-run output/runs/anima-adaptive-precision-90hx-20261001-attention-core \
  --source-run output/runs/anima-adaptive-precision-90hx-20260930-upstream-vjp \
  --stage-run output/runs/anima-adaptive-precision-90hx-20261001-block17-paths-r2 \
  --weights /path/to/anima-base-v1.0.safetensors --output-dir output/runs/<fresh-k-interpolation> \
  --gpu-uuid GPU-806debd9-d254-da57-d515-ccae15b4ac6a --analysis-timeout 300
CUDA_VISIBLE_DEVICES='' timeout 300 .venv/bin/python -m bench.adaptive_runtime.audit_anima_k_interpolation \
  --run output/runs/<completed-k-interpolation> --weights /path/to/anima-base-v1.0.safetensors
```
