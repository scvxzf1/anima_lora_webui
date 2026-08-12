# Krea-2 NF4 下游目标低秩回补 90HX 验证

状态：数学与端到端 velocity 误差验证完成，尚未生产化  
日期：2026-08-12

## 问题

旧探针已经证明，对每个 NF4 Linear 独立拟合固定 rank-16 回补

\[
Q(W)x + BAx
\]

可消除约 `75%` 的层级 \((W-Q(W))X\) 误差能量，却只使完整 DiT
velocity rel-L2 改善 `2.3%-3.6%`。本轮验证下游敏感度是否是缺失因素：

1. 保持完全相同的 196 层、rank-16、96.4MB 参数预算。
2. 以旧的局部激活加权 SVD 因子为初始值。
3. 仅在校准阶段更新回补因子，直接最小化最终 BF16 velocity 误差。
4. 拟合后冻结因子，在未参与优化的真实样本上评估。

直接优化最终 velocity 的梯度通过链式法则包含后续 Jacobian；同时因为
所有回补层联合更新，它也包含跨层误差项和补偿后的真实运行轨迹。

## 实验设计

- GPU：NVIDIA CMP 90HX 10GB（物理 GPU 1）。
- 数据：`post_image_dataset/resized` 与 `post_image_dataset/lora`。
- 数量：60 张真实画风数据中确定性打乱后取 48 张。
- 划分：32 calibration + 16 held-out，seed `20260812`。
- 每个样本独立 noise；σ 按训练默认 `sigmoid(randn)` 分布确定性抽样。
- teacher：BF16 Krea-2-Raw，Flash attention，swap26，精确 BF16 权重传输。
- student：v1 磁盘 NF4 overlay，BF16 compute，Flash attention，swap20。
- 补偿：196 层统一 rank-16，优化前后都是 `96.382MB`。
- 优化：AdamW，lr `2e-5`，weight decay 0，2 epochs，64 steps。
- 运行：full gradient checkpoint + fixed resident compile。
- 指标：单步最终 velocity 相对 L2、MSE、cosine、max delta。

探针：`scripts/krea2/probe_nf4_downstream_correction.py`。

## 结果

### 最终 velocity rel-L2

| 方法 | Calibration mean | Held-out mean | 相对原始 NF4 改善 |
|---|---:|---:|---:|
| 原始 NF4 | 8.8095% | 8.2347% | - |
| 局部激活加权 SVD | 8.6102% | 8.0536% | 2.20% |
| 最终 velocity 目标联合优化 | **6.5204%** | **6.2108%** | **24.58%** |

在相同 96.4MB 预算下，下游目标优化相对局部 SVD 在 held-out 上再改善
`22.88%`。原始 NF4 到下游补偿的 held-out rel-L2 绝对下降为
`2.0239` 个百分点。

### 误差能量

Held-out mean MSE：

| 方法 | MSE | 相对原始 NF4 改善 |
|---|---:|---:|
| 原始 NF4 | 0.0083222 | - |
| 局部 SVD | 0.0079667 | 4.27% |
| 下游目标 | **0.0047267** | **43.20%** |

`rel-L2 mean` 是先对每个样本计算比值再平均，因此不应将 MSE 改善直接开方
与上表数值强行等同。

### Held-out 配对统计

- 下游目标相对原始 NF4：`16/16` 样本改善。
- 下游目标相对局部 SVD：`16/16` 样本改善。
- 局部 SVD 相对原始 NF4：`16/16` 样本改善。
- 下游目标相对原始 NF4 的逐样本改善均值：`24.01%`。
- 逐样本改善中位数：`25.20%`。
- 20,000 次 paired bootstrap 95% CI：`[20.84%, 27.08%]`。
- 逐样本改善范围：`[10.67%, 32.94%]`。

改善并非由少数异常样本拉动。

### Compile 控制

为排除 baseline/local 在 compile 前评估，downstream 在 compile 后评估带来的数值差异，
额外运行了“原始局部因子 + resident compile + 零更新”控制：

| 方法 | Held-out rel-L2 |
|---|---:|
| 局部 SVD，compile 前 | 8.05360% |
| 局部 SVD，compile 后 | 8.04986% |

compile 只带来 `0.046%` 相对变化，远小于下游优化相对局部 SVD 的
`22.88%`，因此不能解释主结果。

### 资源

| 阶段 | 结果 |
|---|---:|
| BF16 teacher peak GPU | 4.035GB |
| BF16 teacher 48 样本 | 5471s |
| NF4 下游优化 peak GPU | 6.274GB |
| 优化热态 median step | 11.511s |
| 64 步优化阶段（含首次 compile） | 1462s |
| 冻结回补文件 | 96.382MB |

## 结论

1. **下游敏感度是局部收益无法转化为全局收益的主要缺失因素。**
   相同参数预算下，从局部目标切换到最终 velocity 目标，held-out 收益从
   `2.20%` 增加到 `24.58%`。
2. **局部误差的欧氏大小不是正确的 rank 预算分配目标。**
   最终损失的梯度会自动将回补预算投向经后续 Jacobian 放大的误差方向，并考虑
   跨层抵消。
3. **收益仍未达到层级 L2 减半的理论上界。**
   目前仅回补 196/264 个 Linear，rank 仍统一，校准只有 32 个状态。
4. **这一结果足以推翻“低秩回补端到端收益必然只有 2%-4%”的判断，但还不足以直接生产接入。**
   需要继续验证完整 denoising rollout、跨风格 prompt 与最终图像质量。

## 限制

- 评估是单步 velocity teacher matching，不是 FID、CMMD 或人工图像评审。
- held-out 样本来自同一画风数据集，尚未验证跨风格泛化。
- 每张图只抽了一个 noise/σ 状态；虽然整体遵循训练分布，仍未覆盖同图多状态方差。
- 局部 SVD 对照因子来自旧的单样本 1024²/σ=0.5 拟合，没有用本轮 32 样本重新做
  多样本局部 SVD。因此 `2.20%` 不是代表性局部 SVD 能达到的严格上界，但不影响
  “同一组因子经最终目标更新后大幅改善”的配对结论。

## 产物与复现

- 完整对照：`output/tests/krea2_nf4_downstream_90hx/report_optimize.json`
- compile 控制：`output/tests/krea2_nf4_downstream_90hx/report_control.json`
- teacher 报告：`output/tests/krea2_nf4_downstream_90hx/report_teacher.json`
- 冻结因子：`output/tests/krea2_nf4_downstream_90hx/downstream_correction.safetensors`
- 样本与随机状态：`output/tests/krea2_nf4_downstream_90hx/manifest.json`

```bash
# BF16 teacher reference（可逐样本续跑）
K2_DS_GPU=1 K2_DS_PHASE=teacher \
K2_DS_CALIBRATION=32 K2_DS_HELDOUT=16 \
.venv/bin/python scripts/krea2/probe_nf4_downstream_correction.py

# NF4 对照与下游目标优化
K2_DS_GPU=1 K2_DS_PHASE=optimize \
K2_DS_CALIBRATION=32 K2_DS_HELDOUT=16 K2_DS_EPOCHS=2 \
K2_DS_NF4=models/diffusion_models/krea2_raw_nf4.safetensors \
.venv/bin/python scripts/krea2/probe_nf4_downstream_correction.py

# compile 零更新控制
K2_DS_GPU=1 K2_DS_PHASE=control \
K2_DS_CALIBRATION=32 K2_DS_HELDOUT=16 \
K2_DS_NF4=models/diffusion_models/krea2_raw_nf4.safetensors \
.venv/bin/python scripts/krea2/probe_nf4_downstream_correction.py
```
