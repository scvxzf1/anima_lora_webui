# Anima block17 post-RoPE Q/K/V 激活八格对照

状态：2026-10-01。Sol实现、另一Sol独立验收、Luna热测与NumPy审计完成。
前置：[隔离QK-norm四格](anima_qknorm_90hx_20261001.md)。无新训练，生产精度默认/
数值门槛/OOM策略不变；本固定样本诊断不是FP16安全层或训练质量认证。

## 结论与比较轴

从原core p和真实checkpoint Q/K RMSNorm权重重建post-RoPE Q/K与原V，分别选
参考R/候选C轨迹，形成RRR..CCC八格。标签三位顺序为输入Q/K/V；所有格使用
参考FP32、参考conditioning与固定du_R。只替换真实轨迹，不改变算子精度。

每个输入轴有四对，另外两个输入固定。下表为返回梯度c的rel-L2百分数：

| 切换输入轴 | fused c | 返回c_Q | 返回c_K | 返回c_V |
| --- | ---: | ---: | ---: | ---: |
| Q | 0.172339%--0.180741% | 0.166205%--0.168555% | 0.187115%--0.198196% | 0.119480%--0.139662% |
| K | **0.737217%--0.946813%** | 0.260393%--0.353564% | 0.948070%--1.258126% | 0.850389%--0.853358% |
| V | 0.296749%--0.439113% | 0.203008%--0.210763% | 0.381786%--0.603589% | **0%，逐位不变** |

K输入切换产生最大的fused对照，绝对L2范围1.459599e-6--1.874392e-6；
Q为3.412111e-7--3.578207e-7，V为5.874880e-7--8.693051e-7。
K输入的影响随V选取变化：V参考时fused约0.945%--0.947%，V候选时约0.737%--0.739%。
V输入切换影响Q/K返回梯度，而V返回梯度不随V变化，符合固定Q/K/du下attention对V
线性的结构；不能把“输入轴V”与“返回分量V”混淆。

主代理核对实际audit JSON后修正了Luna文字摘要中反向的V分量描述；表以实际文件为准。
范围来自不同固定背景/分母，不是可相加的全网误差贡献。输入激活本身来自已有轨迹差异，
不能把K轴对照最大解释成K投影首先产生误差，也不能证明其他通道可以安全FP16。
下一步优先固定背景细查K轨迹及其与V的交互，或对K差异方向做有限幅度插值；
须继续保持du/conditioning与端点fidelity，不直接形成生产精度名单。

## 实现与验收

新增 `anima_qkv_matrix.py`、`diagnose_anima_qkv_matrix.py`、
`audit_anima_qkv_matrix.py`、`tests/test_adaptive_anima_qkv_matrix.py`。
生产RMSNorm与bshd RoPE重建激活，使用原torch dispatch，完整4200tokens/padding保留、
FP32/无TF32。八格逐格计算，每格重复dispatch/VJP两次：前向逐位相同，梯度
重放误差最大1.08687e-7，低于既有1e-4门槛。

RRR/CCC前向与原core u逐位一致；历史post-RoPE c误差8.63e-8/1.106e-7。
端点c经RoPE转置与对应norm VJP重建原dp，CPU FP64解析验收误差9.77e-8/1.190e-7。
来源逐层绑定qknorm/core/stage/upstream的result/environment、weights/code SHA、实际
manifest和张量；旧源签名与实时实现一致。

八格仅保存c，每格103,219,288B，合计825,754,304B；各512MiB、整批1GiB预算通过。
不保存N²矩阵。父进程核验实际SHA/inventory/FP32/finite/shape和12对输入轴对比。
NumPy FP64审计复算12对fused及36个QKV分量指标、平方绝对误差分区、端点链式关系。
审计明确 `mixed_backward_independently_recomputed=false`：没有CPU独立重算混合attention
backward，混合梯度来自实际GPU实现与重复计算控制，不能扩大独立验收范围。

独立Sol未发现阻断P1/P2。相关CPU测试8 passed（矩阵+QKnorm），完整adaptive回归
738 passed、2 skipped、3 xfailed；四新增文件Ruff通过，diff whitespace检查通过。
完整audit来源/端点篡改fixture可继续补充，不作为当前已覆盖项。

## 资源和产物

目录：`output/runs/anima-adaptive-precision-90hx-20261001-qkv-matrix/`。
热测外层exit0/completed，审计exit0/audited，所有指标finite。
worker 141.06秒、130次遥测，GPU峰值1502MiB、RSS tree峰值4,646,076,416B，
最低主机available 102,035,668,992B，遥测错误0、无OOM。owned worker退出，
最终指定GPU17MiB、0%、44°C。仅CMP90HX UUID
`GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，不操作显示GPU/功耗/风扇。

```bash
timeout 540 .venv/bin/python -m bench.adaptive_runtime.diagnose_anima_qkv_matrix \
  --qknorm-run output/runs/anima-adaptive-precision-90hx-20261001-qknorm-r3 \
  --core-run output/runs/anima-adaptive-precision-90hx-20261001-attention-core \
  --source-run output/runs/anima-adaptive-precision-90hx-20260930-upstream-vjp \
  --stage-run output/runs/anima-adaptive-precision-90hx-20261001-block17-paths-r2 \
  --weights /path/to/anima-base-v1.0.safetensors --output-dir output/runs/<fresh-qkv-matrix> \
  --gpu-uuid GPU-806debd9-d254-da57-d515-ccae15b4ac6a --analysis-timeout 300
CUDA_VISIBLE_DEVICES='' timeout 180 .venv/bin/python -m bench.adaptive_runtime.audit_anima_qkv_matrix \
  --run output/runs/<completed-qkv-matrix> --weights /path/to/anima-base-v1.0.safetensors
```

后续：[K差异方向插值与V背景](anima_k_interpolation_90hx_20261001.md)已完成。
