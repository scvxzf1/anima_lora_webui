# Anima block14 完整形状受控 VJP

状态：2026-09-30，Luna 完成新 reference/candidate 捕获与八组离线 VJP，
随后仅重跑分析以导出八组完整梯度；Sol 审查与 CPU 独立 FP64 复算完成。
前置：[残差流边界定位](anima_block_boundaries_90hx_20260930.md)。仅为 bench 诊断，生产策略不变。

## 结论

固定 block14 的同一精度策略和输入，切换参考/候选上游 cotangent，输出输入梯度
产生约 2.711%--2.880% 差异；固定同一策略和 cotangent，只切换输入，差异约
0.182%--0.235%；固定同一输入/cotangent，只切换策略，差异约 0.042%。
本完整形状局部实验更指向传入 cotangent 的差异，而不是 block14 的局部策略切换。
这不证明更后面某个 block 是源头，也不能将三个对照相加成因果贡献。

初次热测摘要曾错配三个标签，独立审计按源码和实际 JSON 更正；下表与八个实际 gx
张量的 CPU FP64 复算一致。原始比值均乘100，百分数与比值不可混用。

| 改变因素 | 固定因素 | 四组对照 gx rel-L2 范围 |
| --- | --- | ---: |
| 输入 x | policy、上游 g | 0.182241%--0.235359% |
| 上游 g | policy、输入 x | 2.711200%--2.879835% |
| policy | 输入 x、上游 g | 0.042007%--0.042057% |

policy 同时包含基座权重舍入、算子 dtype、FP32 residual 和 loss scale，不能当作
纯权重效应。conditioning 的完整元数据与 tensor 均逐位一致，未混入第四组 conditioning。
各对照是不同参考方向/分母，不是对全网误差的可相加分解或最坏方向界。

完整捕获 tensor 的参考/候选相对偏差：

| Tensor | rel-L2 |
| --- | ---: |
| block14 输入 x | 0.108246% |
| block14 输出 y | 0.109826% |
| 输出 cotangent g | 1.327276% |
| 输入 cotangent gx | 2.804862% |

此前 boundary14 的2.166%是512-token采样指标；此处2.805%是完整4200-token指标。
不能把两个指标直接混比，也不能据其差异宣称实验不稳定。
后续优先沿传入 g 的来源，即更后面的 block15..27 路径定位，并固定同一 cotangent
检验局部放大；不据此推荐将 block14 升为 FP32 或认证任何 FP16 层名单。

## 重现门槛与实现

`anima_block_vjp_capture.py` 保存单个 block 的完整 x/y/g/gx、逐块 t embedding、
完整 padded cross-attention embedding、RoPE、AdaLN 和 AttentionParams 的全部实际字段。
输入/输出 tensor hooks 返回原梯度，副本显式 clone 后除 scale，避免 CPU alias 被原地缩放。
外层 module hook 不重复捕获 non-reentrant checkpoint 的 `_forward` 重算；测试包含
full-backward hooks。要求真实 fp32_residual/checkpoint 状态、LoRA up 为零、完整/finite/预算校验。

离线分析使用原 loader 在 CPU 重建 BF16 来源整模，再按参考/候选策略转换，恢复
reference pre-step LoRA 快照。逐 tensor base/adapter 状态摘要必须匹配捕获，
之后只把 block14 与该块 LoRA 上卡。全模型没有放到 GPU；没有手写 checkpoint key 融合。
保持真实全4200-token形状、torch attention、full non-reentrant checkpoint、训练模式、
无 autocast/TF32；g乘各自scale后反传，再除scale。未裁剪文本padding或做小形状GEMM。

每个policy运行 x_R/x_C × g_R/g_C 四组，共八组。先要求同侧重现捕获：
forward rel-L2 <=1e-5，gx rel-L2 <=1e-4（均原始比值，不是生产容差）。

| 同侧 fidelity | Forward rel-L2 | gx rel-L2（原始比值） |
| --- | ---: | ---: |
| Reference（完整矩阵重跑） | 0，逐位相同 | 1.5458e-7 |
| Candidate（完整矩阵重跑） | 0，逐位相同 | 1.4805e-7 |

两侧均通过，八格 finite；交叉对照本身不按同侧 fidelity 门槛判失败。
八个完整 gx 保存为独立 safetensors，轴明确为 `[policy, activation_source, cotangent_source]`。
父进程成功验收增加实际文件SHA、完整八格、shape/dtype/finite/预算及capture/source/replay绑定；
缺失或损坏不能只凭子进程成功标记完成。

## 固定条件与产物

GPU `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，CMP90HX10GiB；同一BF16来源Anima与真实缓存图。
rank/alpha4、初始LoRA up为零、AdamW lr1e-4，单步sigma=.5，full checkpoint、swap20、
torch attention、无compile/TF32。Reference scale1记录pre-step adapter/input/RNG；
candidate scale1024重放，精确56个MLP Linear FP16/321 FP32。两侧同一adapter状态摘要，
各自基座摘要按真实dtype验证，不将候选FP16升FP32误称恢复BF16来源信息。

整网既有 comparator 仍拒绝候选：prediction误差0.020736%，input gradient110.853604%，
adapter gradient40.419729%。两组finite、一次有效optimizer update、scale稳定、
adapter max delta约1e-4。此局部诊断未让整网低精度通过。

捕获及首轮分析目录：`output/runs/anima-adaptive-precision-90hx-20260930-block14-vjp/`。
完整矩阵重跑与独立审计：`output/runs/anima-adaptive-precision-90hx-20260930-block14-vjp-matrix/`，
包括 `matrix-evidence/manifest.json`、`matrix.safetensors`、`audit-cpu.json`、监控和日志。
捕获shape为 `[1,1,75,56,2048]`，每侧tensor含conditioning共144,056,320 B，
八格gx payload275,251,200 B（262.5MiB）、文件275,252,080 B，分别低于独立512MiB预算。
矩阵SHA `b6fb384182e9ef80277cf5b5fb549e8f9f464c04a26426a8cf164da50517d6e4`。

权重SHA `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e`，
输入SHA `0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78`。
捕获helper/probe未变，矩阵重跑只更新analysis/driver；各run保存当时源码SHA。
后续父级文件验收补丁由CPU实际产物验证，旧run的driver SHA保留原值，不声称与新源码相同。
CPU独立加载八格，12项交叉对照的FP64 rel-L2/cosine与落盘值差均小于1e-12。

复现入口：

```text
.venv/bin/python -m bench.adaptive_runtime.diagnose_anima_block_vjp \
  --weights <BF16-source.safetensors> --inputs <prepared-inputs.safetensors> \
  --output-dir <new-directory> --gpu-uuid <UUID> --swap20 \
  --worker-timeout 150 --analysis-timeout 300
```

输出目录必须不存在。exit0只表示诊断矩阵完成；始终 `production_ready=false`，没有推荐方案。

## 资源与验证

| Worker | Peak GPU MiB | Peak RSS tree GB | Host available min GB |
| --- | ---: | ---: | ---: |
| Reference捕获 | 5236 | 19.72 | 98.4 |
| Candidate捕获 | 4714 | 10.99 | 106.5 |
| 首轮分析 | 1846 | 9.78 | 109.9 |
| 完整矩阵分析重跑 | 1846 | 9.79 | 108.90 |

GB为十进制，swap使用0、无OOM或遥测失败，完整矩阵分析49.32秒/45次采样，最高49C。
所有自有worker退出，目标卡恢复17MiB/0%；GTX1050未触碰。
定向CPU回归31 passed；全adaptive回归702 passed、2 skipped、3已知xfailed，exit0，
Ruff和空白检查通过。已知xfail不计作通过。

后续[15/17/27上游完整VJP](anima_upstream_vjp_90hx_20260930.md)已完成，
block17的激活轨迹对照已不可忽略，需继续拆查其内部路径而非只追踪传入g。
