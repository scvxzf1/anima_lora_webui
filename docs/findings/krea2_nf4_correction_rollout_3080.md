# Krea-2 NF4 低秩回补完整 Rollout 验证

状态：3080 完整 denoising rollout 验证已完成，统一 rank-16 冻结回补不建议生产化  
日期：2026-08-12

## 目标

90HX 单步验证中，相同 `96.382MB` 回补预算下，直接优化最终
velocity 相对原始 NF4 将 held-out rel-L2 降低了 `24.58%`。本轮检验
这个 teacher-forced 单步收益能否转化为 free-running 完整采样收益。

## 实验设计

- GPU：NVIDIA GeForce RTX 3080 10GB。
- 分辨率：`512x512`；Euler 步数：`12`；CFG：`4.5`。
- seed：`20260812`。
- prompt：3 个跨风格合成 prompt，分别覆盖写实摄影、动漫插画、水彩。
- 方法：BF16 teacher、原始 NF4、局部激活加权 SVD、下游 velocity 目标回补。
- 回补预算：196 层统一 rank-16，`96.382MB`；全部冻结。
- 控制变量：同 prompt、seed、初始 latent、Euler 时间网格、CFG 和 BF16 VAE。
- 后端：Flash attention + fixed resident compile。
- 块交换：BF16 teacher swap26，NF4 及两种回补 swap20。
- 指标：每步 latent rel-L2/MSE/cosine，最终 pixel rel-L2/PSNR/SSIM，
  全部以同 prompt 的 BF16 teacher 为参照。

探针：`scripts/krea2/probe_nf4_correction_rollout.py`。

## 结果

### 最终 latent

| 方法 | rel-L2 mean | 相对原始 NF4 改善 |
|---|---:|---:|
| 原始 NF4 | **0.582148** | - |
| 局部 SVD | 0.583870 | **-0.296%** |
| 下游目标 | 0.591728 | **-1.646%** |

这里的负改善表示误差变大。下游目标回补相对局部 SVD 也差
`1.346%`。

### 逐 prompt 配对

| Prompt | 原始 NF4 | 局部 SVD | 下游目标 | Local vs NF4 | Downstream vs NF4 |
|---|---:|---:|---:|---:|---:|
| 写实玻璃天文台 | 0.789551 | 0.792957 | 0.798640 | -0.43% | -1.15% |
| 动漫天文学家 | 0.397439 | 0.404689 | 0.411077 | -1.82% | -3.43% |
| 水彩雨中电车 | 0.559453 | **0.553964** | 0.565467 | +0.98% | -1.08% |

- 下游目标回补相对 NF4：`0/3` 改善，`3/3` 变差。
- 局部 SVD 相对 NF4：`1/3` 改善，`2/3` 变差。
- 样本只有 3 个 prompt，配对方向足以否定本轮的大收益外推，但不足以
  给出通用生成质量的置信区间。

### 逐步 latent 漂移

| Euler 步 | 原始 NF4 | 局部 SVD | 下游目标 |
|---:|---:|---:|---:|
| 1 | 0.019295 | 0.019742 | 0.019699 |
| 2 | 0.037725 | 0.038128 | 0.038448 |
| 3 | 0.068852 | 0.069191 | 0.070115 |
| 4 | 0.096828 | 0.097534 | 0.098866 |
| 5 | 0.134513 | 0.135417 | 0.137366 |
| 6 | 0.183535 | 0.184814 | 0.187013 |
| 7 | 0.249656 | 0.251397 | 0.253830 |
| 8 | 0.329572 | 0.331597 | 0.334662 |
| 9 | 0.416211 | 0.417735 | 0.421504 |
| 10 | 0.495971 | 0.497564 | 0.502677 |
| 11 | 0.554403 | 0.555767 | 0.562402 |
| 12 | **0.582148** | 0.583870 | 0.591728 |

表中是 3 个 prompt 的每步 rel-L2 均值。下游回补从第 1 步起就高于
原始 NF4，并且到第 12 步一直没有反超。差距不是只在 VAE 解码时出现，
而是从第一次 Euler 更新就开始。

### 最终像素距离

| 方法 | Pixel rel-L2 | PSNR | SSIM |
|---|---:|---:|---:|
| 原始 NF4 | **0.424486** | **12.4766** | **0.479962** |
| 局部 SVD | 0.430885 | 12.3611 | 0.476818 |
| 下游目标 | 0.440716 | 12.1413 | 0.476807 |

像素指标与 latent 指标方向一致。但它们只衡量相对 BF16 teacher 的数值
保真度，**不等于主观画质或语义质量评分**。本轮没有 LPIPS、CLIP、
DINO、FID 或人工盲评。

## 为什么单步收益没有转化

### 1. Teacher forcing 与 free-running 的状态分布不同

单步目标在 teacher/数据状态 \(x_t\) 上拟合

\[
f_{Q+R}(x_t,t,c) \approx f_W(x_t,t,c).
\]

完整采样中，student 第一步后访问的是自己产生的
\(x_t^S=x_t^T+\delta x_t\)。即使回补降低了 teacher 状态上的速度误差，
在 student 状态上仍然有一阶项

\[
f_S(x_t^S)-f_T(x_t^T)
\approx e(x_t^T)+J_S(x_t^T)\delta x_t.
\]

单步损失只直接压低第一项，没有约束已经偏离轨迹后的 Jacobian 传播。
这是局部/单步收益高而全局收益低的核心原因。

### 2. CFG 会组合两条误差

本轮 CFG 速度是

\[
v_{cfg}=(1+w)v_c-wv_u,
\]

因此误差是

\[
e_{cfg}=(1+w)e_c-we_u.
\]

当 `w=4.5` 时，cond/uncond 误差不会自动抵消，两者的方向差异可以被
放大。90HX 校准目标不是本轮这种多步、cond/uncond 成对的 CFG rollout。

### 3. 校准的 sigma、输入与 prompt 分布不匹配

90HX 优化使用 32 个真实训练样本，每个样本只抽一个
`sigmoid(randn)` sigma，held-out 也来自同一画风数据集。本轮却沿确定的
Euler sigma 网格循环 12 步，并使用跨风格 prompt。固定低秩矩阵对校准分布
的改善不具备轨迹外的一致性保证。

## 资源与速度

| 方法 | Peak GPU | 热态 CFG Euler 步时 |
|---|---:|---:|
| BF16 teacher | 4.010GB | 约 19.1-22.6s |
| 原始 NF4 | 2.801GB | 约 4.53-4.60s |
| 局部/下游回补 | 2.967GB | 约 4.56-4.75s |

该 peak 受块交换影响，不包含 CPU 侧模型存储，也不是不启用 swap 时的完整模型
显存占用。

## 结论与决策

1. **单步 `24.58%` 收益不能外推到完整生成。** 本轮完整 rollout 中，
   downstream 最终 latent rel-L2 相对原始 NF4 反而变差 `1.646%`。
2. **局部 SVD 也没有稳定全局收益。** 它只在 `1/3` prompt 改善，平均变差
   `0.296%`。
3. **当前统一 rank-16 冻结回补不应接入生产路径。** 它增加 `96.382MB`
   常驻权重和少量步时，却没有提高完整轨迹对 BF16 teacher 的保真度。
4. 如果继续研究，优化目标必须改为 rollout-aware distillation：在 student 自身
   轨迹上采集多 sigma 状态，成对覆盖 CFG cond/uncond，直接优化 next-state
   或短 unroll 损失，必要时使用 truncated BPTT。

## 限制

- 只有 3 个 prompt、1 个 seed、`512x512` 和 12 步。
- 评估目标是复现 BF16 teacher，不是绝对生成美学质量。
- 自实现 SSIM 只用于本轮内部配对，不与其他工具的 SSIM 数值横向比较。
- 局部 SVD 因子沿用旧探针的单样本 `1024x1024`/sigma=0.5 拟合，不代表
  更强的多状态局部校准上界。

## 产物

- 汇总报告：`output/tests/krea2_nf4_correction_rollout_3080/report_report.json`
- 轨迹：`output/tests/krea2_nf4_correction_rollout_3080/trajectories/`
- 图像：`output/tests/krea2_nf4_correction_rollout_3080/images/`
- 文本条件缓存：`output/tests/krea2_nf4_correction_rollout_3080/text/`
- 运行配置：`output/tests/krea2_nf4_correction_rollout_3080/manifest.json`

产物完整性已点验：12 个 trajectory、12 张 PNG、3 个文本条件文件，
无缺失项。
