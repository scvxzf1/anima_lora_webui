# Anima block17 隔离 QK-norm 与 attention cotangent 四格定位

状态：2026-10-01。Sol 实现、另一 Sol 独立代码验收，Luna 执行热测与 CPU NumPy 审计。
前置：[自注意力两段与投影/核心拆分](anima_attention_split_90hx_20261001.md)。
仅复用已有捕获的 analysis-only 实验，无新训练。生产精度默认、数值门槛和 OOM 策略不变。

## 结果

实际 block17 core 已可隔离重放：释放整块模型和 LoRA 后，只保留真实 Q/K RMSNorm、
RoPE、attention params，使用原 fused-QKV 激活 p 与固定 du_R。
隔离 attention 输出 u 与原捕获逐位一致，完整 dp fidelity rel-L2 为
7.68e-8 / 8.48e-8，低于既有1e-4门槛。GPU峰值由上一轮9504MiB降为1500MiB。

四格 `M_ij = J_N(p_i)^T a_j` 中，N是Q/K norm和V Identity；
`c_j = J_dispatch(r_j)^T du_R` 是attention返回的梯度，
`a_j = J_RoPE^T c_j` 是进入norm输出的梯度。i/j分别为真实参考/候选轨迹，
两侧均使用参考FP32 policy、同一conditioning、实际4200 tokens，候选不是FP16重放。

| 对照（百分数为rel-L2乘100） | fused QKV | Q分量 | K分量 | V分量 |
| --- | ---: | ---: | ---: | ---: |
| 固定a_R，只换p | 0.374426% | 0.111342% | 0.742535% | 0%，逐位相同 |
| 固定a_C，只换p | 0.116466% | 0.117584% | 0.185624% | 0%，逐位相同 |
| 固定p_R，只换a | 1.160737% | 0.457067% | 2.003955% | 0.855451% |
| 固定p_C，只换a | 1.319098% | 0.469786% | 2.365133% | 0.855451% |

本样本方向上，改变attention传回cotangent的对照大于norm Jacobian输入变化对照；
K分量仍最明显。固定a时norm也有非零响应，不能排除其影响。
attention输出梯度c、进入norm梯度a的两侧差异均约1.094590%；
norm后Q/K激活与RoPE后Q/K激活差异均约0.547675%。
RoPE采用同一cos/sin，是固定线性映射；本轮两侧RoPE转置梯度独立复算误差为0。
这些百分数分母和固定方向不同，不相加成因果贡献，不推导Jacobian全局上界，
不说明最初FP16舍入误差在哪里产生，也不能推荐安全层名单。

下一步优先在隔离路径中做实际Q/K/V激活的受控替换，观察attention后向对不同通道
轨迹的响应。仍需固定du、保持conditioning/布局，并首先通过原对角dp重现；
不会直接据K分量大就认定K投影或K norm是根因。

## 实现与独立验收

新增 `anima_qknorm_paths.py` / `diagnose_anima_qknorm.py`：逐侧释放计算图，
使用生产 `apply_rotary_pos_emb_qk` 和 `dispatch_attention`，保留真实bshd布局、
16 heads×128、RMSNorm eps1e-6、torch attention、FP32、无TF32、全部padding。
原 block 的 base/adapter state digest核验后，才提取norm并释放其余模型。
源core/stage/upstream、weights、实际conditioning、文件及实时代码SHA均绑定。
GPU worker的timeout仍最多300秒。

证据仅保存两侧c/a和四格M：每侧206,438,568B，矩阵412,877,224B，
合计825,754,360B。单文件512MiB、聚合1GiB门槛均通过。
不复制源p/u/du/dp，不保存N² attention矩阵；norm/RoPE激活只保留摘要，
未声称独立审计重新验证了未落盘激活摘要。

独立验收发现首版交叉格只核对统计与对角，未独立重算交叉格数学关系；
随后新增 `anima_qknorm_math.py`，用实际checkpoint Q/K权重（与模型相同的
BF16加载后FP32转换）及FP64解析式重算全部四格：
`r = (mean(x²)+eps)^(-1/2)`，`g = a*w`，
`dx = r*g - x*r³*mean(g*x)`。
worker、父进程和独立CPU审计都执行四格Q/K门槛与V Identity逐位检查。
解析重算最大rel-L2 **1.14663e-7**，低于既有1e-4门槛。
篡改交叉格并同步重签manifest SHA/统计的CPU测试也会被拒绝。

`audit_anima_qknorm.py` 使用NumPy FP64复算实际指标，要求提供真实weights，
重验result/environment、实时代码/weights、完整core/stage/upstream来源。
RoPE转置与V Identity通过；对角矩阵与原dp的CPU误差约7.51e-8--8.32e-8。
最终独立Sol复核未发现阻断P1/P2，735 passed、2 skipped、3 xfailed，Ruff通过。
定向5项通过。尚可补更完整的audit来源篡改fixture，不作为本轮通过证据。

## 热测、失败记录与资源

最终成功目录：`output/runs/anima-adaptive-precision-90hx-20261001-qknorm-r3/`。
包含完整environment/result/summary/supervisor、worker.log/telemetry、三个evidence/
manifest与audit-cpu.json。热测外层exit0、status completed，CPU审计exit0、audited。

- analysis worker 99.09秒、91次遥测，峰值GPU1500MiB、RSS tree 10,698,153,984B。
- 最低主机available 96,719,646,720B；峰值44°C，swap used/in/out为0，遥测错误0。
- 没有OOM、非有限值或分析错误；最终目标GPU17MiB、0%，owned worker退出。
- 仅 CMP90HX UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，没有操作显示GPU、功耗或风扇。

首轮 `...-qknorm/` 保留加固前完成的结果，不能作为最终四格解析验收。
加固后第二轮 `...-qknorm-r2/` 的analysis worker成功，但外层360秒超时退出143，
summary缺失，独立审计因此失败；没有补造summary或标记completed。
后续CPU分段重验source约57.8秒、files约3.5秒、stats约15.5秒，全部通过，
未重现当次超时。第三轮仅外层改540秒，worker仍300秒、代码和门槛不变。
热测总耗时约147秒、CPU审计约126秒；前两目录完整保留。

```bash
timeout 540 .venv/bin/python -m bench.adaptive_runtime.diagnose_anima_qknorm \
  --core-run output/runs/anima-adaptive-precision-90hx-20261001-attention-core \
  --source-run output/runs/anima-adaptive-precision-90hx-20260930-upstream-vjp \
  --stage-run output/runs/anima-adaptive-precision-90hx-20261001-block17-paths-r2 \
  --weights /path/to/anima-base-v1.0.safetensors --output-dir output/runs/<fresh-qknorm> \
  --gpu-uuid GPU-806debd9-d254-da57-d515-ccae15b4ac6a --analysis-timeout 300
CUDA_VISIBLE_DEVICES='' timeout 180 .venv/bin/python -m bench.adaptive_runtime.audit_anima_qknorm \
  --run output/runs/<completed-qknorm> --weights /path/to/anima-base-v1.0.safetensors
```

后续：[post-RoPE Q/K/V八格受控替换](anima_qkv_matrix_90hx_20261001.md)已完成。
