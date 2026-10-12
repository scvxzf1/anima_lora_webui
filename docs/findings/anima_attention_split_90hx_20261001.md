# Anima block17 自注意力归一化与投影/核心拆分

状态：2026-10-01。四个推进环节完成：两段式热测、独立CPU复算、投影/核心细化热测、
验收与文档整理。主代理执行短诊断（代理创建线程上限），没有新训练。
前置：[三个残差阶段定位](anima_block17_paths_90hx_20261001.md)。
生产精度默认、数值门槛和OOM策略未改，没有推荐安全层名单。

## 两段拆分结论

写第一条 gated branch 为 `A(N(x))`。N是实际LayerNorm/AdaLN调制，A是实际self_attn
模块，包含QKV、Q/K normalization、RoPE、attention、输出投影。
参考和候选输入来自原始捕获，两侧重放都使用同一reference FP32 block和conditioning。
固定参考阶段输出cotangent与gate乘积 `v = g_R * gate_R`，得到
`h_R = J_A(z_R)^T v` 和 `h_C = J_A(z_C)^T v`；再保存四格 `J_N(x_i)^T h_j`。

| 对照轴 | 两组 rel-L2 差异 |
| --- | ---: |
| 固定 h，只换归一化输入 x | 0.021864%--0.039692% |
| 固定 x，只换中间 cotangent h | 1.271337%--1.296043% |

原始残差输入x差异0.127923%，调制后z差异0.606057%，attention分支输出b差异0.257263%。
固定v的中间h差异1.213417%，branch输入梯度差异1.263987%。
本方向上，归一化Jacobian的输入变化对照较小，传回h变化的对照较大。
这并不否认N改变前向激活轨迹：z/x的相对误差分母不同，不能用其比值作为Lipschitz上界。
四格差异不能相加成因果贡献，也不能据此定位最初FP16舍入误差来源。

## 投影与核心子链结论

在同样固定v的参考FP32重放中，记录实际fused `qkv_proj` 输出p与 `output_proj` 输入u。
模块hook保留已经应用的LoRA forward，未重写投影或attention实现。
参考 `du_R = J_output(u_R)^T v` 固定后，两侧计算 `dp_i = J_core(p_i)^T du_R`。
core包含Q/K normalization、RoPE和attention；随后检查
`J_qkv(z_i)^T dp_i` 与实际h的链式一致性。
另固定 `dp_R`，比较QKV投影两侧输入VJP。

| 张量/对照 | rel-L2 差异 |
| --- | ---: |
| fused-QKV激活 p | 0.528999% |
| attention输出、投影前 u | 0.435656% |
| 固定v的输出投影输入cotangent du | **0%，逐位相同** |
| 固定du_R的fused-QKV输入cotangent dp | **1.130019%** |
| dp的Q分量 | 0.480999% |
| dp的K分量 | **1.922555%** |
| dp的V分量 | 0.855451% |
| self_attn模块输入cotangent h | 1.213417% |
| 固定dp_R、改变z的QKV输入VJP | **0%，逐位相同** |

同一FP32策略下，本样本的梯度变化集中在core子链对不同p的响应；固定cotangent的
两个投影VJP不随输入变化。K分量的相对与绝对差异最大（absolute L2 1.016362e-6，
Q 3.109566e-7，V 5.665768e-7）。Q/K/V各自的平方绝对误差相加等于fused dp平方误差，
这是张量坐标分区恒等式，**不是三个算子的因果贡献**。

尚未区分K normalization、RoPE与attention。输入p本身来自前向轨迹变化，不能宣称
core制造了全部误差，不能据此关闭QKV/输出投影的生产精度保护。
下一步在固定真实p/du下拆查Q/K normalization与attention后向；需先避免继续保留完整
block图造成显存超预算，优先隔离core重放并验证其与现有dp的fidelity。

## 实现和证据合同

`anima_self_attention_trace.py`、`anima_attention_core_trace.py`是checkpoint-aware引用记录hook；
重算不重复记录，结束及异常恢复原方法和hook。CPU测试含实际仿LoRA monkeypatch、
非线性token耦合attention、交叉cotangent、多次VJP、链式梯度与破损文件/统计拒绝。

`anima_self_attention_paths.py`、`anima_attention_core_paths.py`复用已有4200-token capture与
reference模型加载/LoRA replay。两模式共享 `diagnose_anima_self_attention` 监控入口，
新增 `--component normalization`（默认）/`projection-core`，新增训练步数0。
无TF32、torch attention、full checkpoint、全部padding保留；不落盘N² attention矩阵。

两模式核验原weights/capture/replay/conditioning/helper/model源码SHA、实际stage文件、
逐位前向、完整梯度与原branch梯度fidelity。既有梯度门槛1e-4不变，实际fidelity约
1.2e-7--2.3e-7。core中固定du/实际du逐位一致；QKV链式梯度通过。
父进程再次验源输入/gate/terminal、文件SHA/shape/dtype/finite、实际指标和文件预算。
core的QKV固定dp对照只作为测量报告，不强制任意非线性adapter也必须得到零。

独立 `audit_anima_self_attention` / `audit_anima_attention_core` 使用实际SHA核验后的
NumPy数组，以FP64复算全部对照指标，和结果JSON差异小于1e-12；core另核验Q/K/V
平方误差分区。审计由主代理完成，未取得另一代理的独立代码验收。

首轮 `...-self-attention/` 失败记录保留：额外要求的重复参考梯度bitwise equality失败。
原有前向、完整梯度、branch和chain fidelity均通过。修正为前向继续bitwise、重复梯度
使用既有1e-4门槛，额外落盘reference_replay供独立复算。实测h重复rel-L2 1.938059e-7、
branch重复2.224577e-7；未放宽既有fidelity或覆盖失败目录。

## 资源与验证

产物位于 `output/runs/`，每次独立目录包含environment/result/summary/worker.log/
telemetry/supervisor、evidence/manifest和audit-cpu.json：

| 成功目录后缀（前缀anima-adaptive-precision-90hx-20261001-） | worker秒数 | 显存峰值MiB | RSS tree峰值B | evidence文件合计B |
| --- | ---: | ---: | ---: | ---: |
| self-attention-r2 | 49.30 | 8668 | 9,790,492,672 | 688,154,576 |
| attention-core | 51.55 | 9504 | 9,779,609,600 | 860,178,000 |

两批每文件小于512MiB，合计各小于1GiB；温度峰值48°C、swap used0、telemetry错误0，
无OOM或非有限值。固定CMP90HX UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，
没有操作显示GPU、功耗或风扇。最终GPU17MiB、利用率0%，owned worker均退出。
9504MiB的峰值说明当前完整图的观察余量较小，不外推下一轮能容纳更多边界。

相关定向测试10 passed；全adaptive回归730 passed、2 skipped、3 xfailed。
增加CPU审计断言后定向10项再次通过，Ruff和diff whitespace检查通过。

```bash
timeout 360 .venv/bin/python -m bench.adaptive_runtime.diagnose_anima_self_attention \
  --component projection-core \
  --source-run output/runs/anima-adaptive-precision-90hx-20260930-upstream-vjp \
  --stage-run output/runs/anima-adaptive-precision-90hx-20261001-block17-paths-r2 \
  --weights /path/to/anima-base-v1.0.safetensors \
  --output-dir output/runs/<fresh-analysis> \
  --gpu-uuid GPU-806debd9-d254-da57-d515-ccae15b4ac6a --analysis-timeout 300
CUDA_VISIBLE_DEVICES='' timeout 60 .venv/bin/python -m bench.adaptive_runtime.audit_anima_attention_core \
  --run output/runs/<completed-core-analysis>
```

归一化四格使用 `--component normalization`，独立复算入口为
`bench.adaptive_runtime.audit_anima_self_attention`。

后续：[隔离QK-norm与attention cotangent四格](anima_qknorm_90hx_20261001.md)已完成。
