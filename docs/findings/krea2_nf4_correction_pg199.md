# Krea-2 NF4 激活加权精度回补 PG199 实测

状态：已完成，结论为暂不生产化
日期：2026-08-09

## 问题

验证冻结 NF4 DiT 的每个 LoRA 目标 Linear 是否可以增加一个低显存的固定
rank-16 分支，对量化输出误差

\[
E X=(W-Q_{NF4}(W))X
\]

进行精度回补，并核对数学收益、训练显存和训练速度是否符合前期推导。

## 环境与控制变量

- GPU：NVIDIA DRIVE-PG199-PROD 32GB（物理 GPU 1）。
- 分辨率：1024×1024；单样本 4608 token。
- DiT：Krea-2-Raw BF16 / 磁盘预量化 NF4。
- 目标：28 blocks × 7 Linear = 196 层。
- 回补：每层 rank 16，输入协方差 PCA rank 128，BF16 固定权重。
- 训练：现有 task LoRA rank 16，完整 gradient checkpoint，无 block swap。
- 计时：2 warmup + 5 measured steps，固定 latent、text、noise 和 sigma。
- 校准：sigma=0.5、noise seed=123。
- 留出：同 sigma 新 noise，以及 sigma=0.2/0.8、noise seed=456。

探针：`scripts/krea2/probe_nf4_correction.py`。

## 拟合方法

对层输入矩阵 \(X\) 做截断 SVD：

\[
X\approx U_x S_x V_x^T
\]

在保留的输入协方差子空间构造：

\[
M=(W-Q(W))V_xS_x/\sqrt{N}
\]

对 \(M\) 取 rank-16 SVD，并解析还原固定 \(B,A\)，使

\[
Q(W)X+BAX
\]

逼近 BF16 输出。拟合在 BF16 前向的真实层输入上进行；NF4 验证阶段不同时持有
BF16 DiT。回补通过 forward hook 叠加，不修改 NF4 权重和生产训练路径。

## 结果

### 层级数学收益

| 指标 | 结果 |
|---|---:|
| 196 层合计误差能量覆盖 | 75.171% |
| 剩余层级 L2 因子 | 0.4983 |
| 单层覆盖率中位数 | 70.210% |
| 固定回补权重文件 | 96.382MB |

实际因子覆盖率与独立奇异谱探针预测的 75.238% 基本一致，证明激活加权
低秩推导和实现正确；在校准激活上，层级 \(EX\) L2 确实约减半。

### 端到端 BF16 velocity 误差

| Case | NF4 rel-L2 | 回补后 rel-L2 | 相对改善 |
|---|---:|---:|---:|
| calibration | 7.5490% | 7.3696% | 2.38% |
| heldout noise | 7.7350% | 7.4570% | 3.59% |
| sigma=0.2 | 8.5106% | 8.3133% | 2.32% |
| sigma=0.8 | 6.6384% | 6.3995% | 3.60% |

四个 case 均改善，说明固定回补没有只记忆校准 noise；但层级误差减半只转化为
2.3%-3.6% 的端到端 rel-L2 改善。逐层独立最小化 \(EX\) 没有考虑后续 Jacobian、
非线性和跨层误差抵消，因此不能把 0.4983 直接乘到最终 velocity 误差上。

### 训练资源

| 指标 | NF4+LoRA16 | +固定回补16 | 增量 |
|---|---:|---:|---:|
| peak GPU | 10.4914GB | 10.5872GB | +0.0958GB |
| mean step | 3.3877s | 3.5766s | +5.58% |
| median step | 3.3870s | 3.5770s | +5.61% |

显存实测与 96MB 参数推导完全一致，额外激活在完整 checkpoint 下没有形成明显峰值。
速度则没有达到仅按 FLOPs 推导的约 0.43%：196 个独立 down/up 分支产生大量 skinny
GEMM 和 kernel launch，实际慢 5.58%。

## 结论

1. **数学容量假设成立**：rank-16 在校准分布上可消除约 75% 的层级 NF4 误差能量。
2. **显存假设成立**：实际只增加 95.8MB，NF4 训练仍保持在 10.6GB 内。
3. **速度低估**：独立 hook 实现慢 5.58%，不能用 0.43% FLOPs 直接代替 wall time。
4. **端到端收益有限**：velocity rel-L2 只改善 2.3%-3.6%，远小于层级 L2 改善。
5. **当前不建议生产接入**：收益/复杂度比尚不足。后续若继续，应优先做下游
   Jacobian/Hessian 加权的层选择和非均匀 rank，再把固定分支与 task LoRA 融合以减少
   kernel launch；不应直接给全部 196 层统一加独立 rank-16 hook。

## 复现

```bash
K2_CORR_GPU=1 \
K2_CORR_IMG=1024 \
K2_CORR_INPUT_RANK=128 \
K2_CORR_WARMUP=2 \
K2_CORR_STEPS=5 \
.venv/bin/python scripts/krea2/probe_nf4_correction.py
```

原始结果：`output/tests/krea2_nf4_correction_rank16_pg199_1024.json`。

回补权重：`output/tests/krea2_nf4_correction_rank16_pg199_1024.safetensors`。
