# Anima 上游 block15/17/27 完整 VJP 定位

状态：2026-09-30，Luna 完成一次参考/候选多块捕获与三个离线分析；
主代理完成实际24个矩阵张量的CPU独立FP64归约。前置：[block14完整VJP](anima_block14_vjp_90hx_20260930.md)。
仅为诊断，生产精度默认、门槛与OOM策略不变。

## 结论

block27 的完整形状局部对照较小；block17 中，固定cotangent和policy只换激活输入，
产生约0.673%--0.698%的VJP差异，与只换cotangent约0.660%--0.679%接近。
到block15，只换cotangent约1.321%--1.325%，明显大于输入变化对照。
三块固定输入/cotangent、只换policy的差异均约0.026%--0.042%。

本固定样本中，漂移沿上游反传传播，同时block17已经表现出对激活轨迹变化的敏感性。
不能只追踪上游g，也不支持将当前块的FP16反向计算作为主要解释。
没有锁定最初误差源头；不把三种对照相加成因果贡献，不推荐安全层名单。

| Block（按反向顺序） | 固定policy/g换输入x | 固定policy/x换cotangent g | 固定x/g换policy |
| --- | ---: | ---: | ---: |
| 27 | 0.040528%--0.048801% | 0.064636%--0.070894% | 0.025542%--0.025598% |
| 17 | 0.673015%--0.697892% | 0.660451%--0.679428% | 0.039739%--0.040172% |
| 15 | 0.202554%--0.382616% | 1.321460%--1.325204% | 0.041340%--0.041572% |

表中百分数均为原始比值乘100；每项范围含该轴的四组对照，完整36项在`audit-cpu.json`。
JSON轴分别是`activation_fixed_policy_and_cotangent`、`cotangent_fixed_policy_and_activation`、
`policy_fixed_activation_and_cotangent`。policy含权重舍入、算子dtype、residual与scale，
不是纯权重干预。比较分母和方向不同，不是可相加的全网误差分解。

同一批参考/候选完整捕获差异：

| Block | 输入x rel-L2 | 输出y rel-L2 | 输出cotangent g rel-L2 | 输入cotangent gx rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| 27 | 0.039738% | 0.015675% | 0.069101% | 0.079984% |
| 17 | 0.127923% | 0.111112% | 0.621195% | 0.974068% |
| 15 | 0.109826% | 0.119600% | 1.196863% | 1.324470% |

block17的相对输入变化小，但对照gx变化更大，值得继续拆开内部路径。
这些只是一组输入及cotangent方向，不能从百分数比值推导完整Jacobian/Lipschitz上界。
blocks18..26、16未在本轮做完整局部矩阵，不宣称已连续追踪或排除了这些位置。
此前512-token边界指标与本次完整4200-token指标也不能混比。

下一步优先拆查block17的self-attention、cross-attention、MLP与其归一化/残差边界，
在同一FP32策略下固定同一cotangent，仅切换真实激活，确认哪条路径对轨迹扰动敏感。
这比直接增加FP32层更有区分力；本报告仍不是根因结论或训练质量证书。

后续：[2026-10-01 block17内部阶段对照](anima_block17_paths_90hx_20261001.md)已完成，
固定参考阶段cotangent时自注意力路径的完整VJP对照最大，继续向其内部细分。

## 实现和实验合同

捕获端新增`MultiBlockVJPCapture`，一次训练同时观测15/17/27；
`--capture-block-vjp --block-vjp-indices 15,17,27`使用有序身份签名，默认单14模式保持。
支持的实验范围只为单14或固定三块，非法、重复或错序索引在GPU前拒绝。
保留真实full non-reentrant checkpoint和block swap hooks；梯度副本clone后除scale，
原梯度返回不变。每个捕获保存完整x/y/g/gx与实际conditioning、base/LoRA状态摘要。

两训练worker各自共享硬512MiB预算：forward为激活、conditioning及随后两个梯度预留，
close再次核验全部实际tensor与文件总量；不是把三个各自512MiB上限相加。
每侧实际432,168,960B（约412.15MiB）tensor，子文件均逐项SHA校验，缺失块不能视为完成。
模型源码与生产runtime未改。

`diagnose_anima_upstream_vjp.py`先运行fresh FP32 reference，再运行重放候选，
只做两次真实单步训练，随后逐块串行离线分析。分析器显式block身份、对应training/replay根目录，
原CPU模型加载/精度安装/LoRA恢复与各块状态摘要保持一致，仅目标块与其LoRA上GPU。
每块八组真实4200-token VJP，完整padding、torch attention、full checkpoint、无TF32，
两侧conditioning元数据和tensor逐位一致。

固定GPU `GPU-806debd9-d254-da57-d515-ccae15b4ac6a` CMP90HX10GiB，同一BF16来源Anima、
真实缓存图、初始LoRA up为零、rank/alpha4、AdamW lr1e-4、swap20、单步sigma=.5。
Reference scale1 record；候选scale1024 replay，精确56个MLP FP16/321 FP32。
每块同侧前向重现逐位相同，gx重现原始relative-L2约3.3e-8--1.4e-7，
低于forward1e-5/gx1e-4的既有fidelity门槛。没有放宽门槛或共享comparator/replay合同。

矩阵每块8个FP32完整gx，shape`[1,1,75,56,2048]`，单矩阵262.5MiB；
三矩阵payload787.5MiB，实际文件合计825,756,240B，另执行硬1GiB整批矩阵上限。
父进程逐块核验索引、来源/状态/replay、矩阵文件SHA、完整格数/shape/dtype/finite，
三块全部通过才标记`completed`。没有selected plan，始终`production_ready=false`。

整网候选仍拒绝：prediction rel-L2约0.020736%，input gradient约110.957%，
adapter gradient约40.465%。两组各一次finite有效更新，scale1/1024保持不变，
adapter max delta约1e-4；局部定位不等于整体低精度通过。

## 证据、资源与验证

产物：`output/runs/anima-adaptive-precision-90hx-20260930-upstream-vjp/`，包含
两侧`block-vjp/block-15|17|27`、三个`analysis-block-*`、完整矩阵与`audit-cpu.json`。
权重SHA `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e`，
输入SHA `0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78`；
本次helper/probe/analysis/两个driver的当前SHA均与environment一致。
CPU重新加载24个实际gx，对36项FP64 rel-L2/cosine与记录逐项核验，差均小于1e-12，
同时核验完整捕获、conditioning及同侧gx重现；不是只信summary。

| Worker | Peak GPU MiB | Peak RSS tree GB | Host available min GB | 最高温度C |
| --- | ---: | ---: | ---: | ---: |
| Reference | 5236 | 19.87 | 89.61 | 54 |
| Candidate | 4714 | 11.12 | 99.01 | 50 |
| Analysis15 | 1846 | 9.79 | 101.55 | 46 |
| Analysis17 | 1846 | 9.78 | 104.62 | 48 |
| Analysis27 | 1846 | 9.78 | 104.11 | 47 |

GB为十进制。累计280次内置遥测，telemetry failures=0、无OOM，五worker整机swap起止均0。
自有workers均退出，目标卡恢复17MiB/0%，GTX1050未触碰。
训练worker上限150秒、每个分析上限300秒；整批worker限额和不超过1200秒，
不把全部加载/父进程验证耗时当训练step时间。

定向CPU回归47 passed；全adaptive715 passed、2 skipped、3已知xfailed，exit0。
Ruff和空白检查通过。捕获由Sol实现、主代理集成分析驱动与数值验收；
新的独立代理复审派生被工具会话上限拒绝，本轮未获得独立代理对整批驱动的再次审查。
已知xfail不计作通过，CPU数值验收不外推生产安全性。

```text
.venv/bin/python -m bench.adaptive_runtime.diagnose_anima_upstream_vjp \
  --weights <BF16-source.safetensors> --inputs <prepared-inputs.safetensors> \
  --output-dir <new-directory> --gpu-uuid <UUID> --swap20 \
  --worker-timeout 150 --analysis-timeout 300
```

输出目录必须不存在。exit0表示定位批次完成，不表示FP16候选获准。
