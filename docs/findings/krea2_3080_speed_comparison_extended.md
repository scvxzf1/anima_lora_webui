状态：扩展速度对比（阶段 1-12 数据汇总）
日期：2026-08-09
主摘要：[krea2_3080_speed_final.md](krea2_3080_speed_final.md)

# Krea-2 PG199 / RTX 3080 扩展速度对比

## 对比口径

除非特别标注，训练级数据均为 1024²、batch 1、BF16 compute、NF4 DiT、rank16
LoRA 和 full gradient checkpoint。`it/min=60/step_s`；“耗时变化”比较单步延迟，
“吞吐变化”比较单位时间迭代数。两者不能混写，例如 `3.370→2.726s` 是耗时下降
19.1%，但吞吐提升 23.6%。

冷态短测、热稳态、不同 token family、不同分辨率和算子微基准分别列出，不直接互相
外推。

## PG199 1024² 训练矩阵

基准为 NF4、swap0、28/28 full checkpoint、eager 的 `3.370s/it`。

| 配置 | 稳态 step | it/min | 耗时 vs 基准 | 吞吐 vs 基准 | GPU peak | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full ckpt + eager | 3.370s | 17.80 | 基准 | 基准 | 10.49GB | 可用基准 |
| last2 不 ckpt + eager | 3.299s | 18.19 | -2.1% | +2.2% | 11.83GB | 收益不足 |
| every-other ckpt + eager | 2.901s | 20.68 | -13.9% | +16.2% | 28.46GB | 32GB opt-in |
| full ckpt + cuDNN compile | 2.726s | 22.01 | -19.1% | +23.6% | 11.06GB | 生产安全档 |
| 16/28 ckpt + cuDNN compile | 2.408s | 24.92 | -28.5% | +40.0% | 31.55GB | 无余量实验档 |
| full ckpt + Flash varlen compile | 2.421s | 24.78 | -28.2% | +39.2% | 10.87GB | 显式 opt-in |

Flash varlen 相对生产 cuDNN compile 进一步降低 11.2% 单步耗时、提高 12.6% 吞吐，
但只多使用 eager 基准约 0.38GB。它以 10.87GB 峰值达到接近 31.55GB selective
实验档的速度；当前差距仅 `0.013s/it`。Flash 生产 backend 契约现已完成，
但仍保持显式 opt-in 而非默认。

Flash 的 50 步复核通常保持 `2.417-2.439s`，末步 `2.429s`，因此表中不是短窗口
峰值。compile 首步受 Inductor cache 影响较大：历史记录为 `15.303-28.644s`，不应
用单个首步推导固定回本步数。

## PG199 精度、rank 与 bucket

### BF16 / NF4 独立消融

这组来自早期 30 步五格矩阵，计时与上表不是同一轮，适合比较相对代价：

| base 权重 | step | it/min | GPU peak | 相对结论 |
| --- | ---: | ---: | ---: | --- |
| BF16 | 3.29s | 18.24 | 29.45GB | 参考 |
| NF4 | 3.41s | 17.60 | 10.49GB | 慢 3.6%，省 72% 显存 |

NF4 不是 RTX 3080 比 PG199 慢 3.6 倍的主因；它在单卡内只带来约 3.6% 整步代价。

### LoRA rank

| rank | 参数量 | step | GPU peak | 结论 |
| ---: | ---: | ---: | ---: | --- |
| 16 | 48.17M | 2.726s | 11.057GB | 参考 |
| 8 | 24.08M | 2.728-2.737s | 10.912GB | 速度持平，仅省 145MB |

### 两个 token family

| family | cuDNN compile | Flash varlen | 耗时变化 | 吞吐变化 | Flash peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4032 image tokens | 2.731s | 2.388s | -12.6% | +14.4% | 10.965GB |
| 4200 image tokens | 2.956s | 2.569s | -13.1% | +15.1% | 11.154GB |

24 个生产 buckets 最终只形成 padded length 4608/4864 两张图。同 family 不同宽高比
可以复用；4200 family 本身比 4032 family 多约 8% 步时。

## RTX 3080 1024² 训练矩阵

RTX 3080 会在 20 步内出现约 5% 热漂移，所以短窗口与热稳态必须分开：

| 配置/窗口 | step | it/min | GPU peak | 解读 |
| --- | ---: | ---: | ---: | --- |
| swap20 eager，历史 10 步 | 12.14s | 4.94 | 7.65GB | 冷/短窗口边界点 |
| swap20 resident compile，4 步 | 11.744s | 5.11 | 约 6.15GB | 冷态，不能外推 |
| swap20 resident compile，20 步末段 | 12.50-12.65s | 4.74-4.80 | 6.153GB | cuDNN 热稳态 |
| swap20 Flash varlen compile，20 步末五步 | 12.145s | 4.94 | 6.094GB | opt-in 热稳态 |

Flash 相对同口径 cuDNN 长窗口 `12.65s` 降低约 4.0% 延迟、提高约 4.2% 吞吐。
它只是把热稳态恢复到早期 eager 冷态约 `12.14s` 的水平，并未消除大矩阵瓶颈。
resident compile 在 3080 上的生产价值仍主要是把 eager 的首个 backward OOM 边界变成
可稳定运行，而不是已证明的持续提速。

## RTX 3080 swap 曲线

以下均为历史 10 步短窗口，适合选择显存工作点，不与 20 步热稳态直接相减：

| swap | step | it/min | GPU peak | host RSS | 相对 swap20 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 12.14s | 4.94 | 7.65GB | 20.47GB | 基准/临界点 |
| 22 | 12.27s | 4.89 | 6.78GB | 21.07GB | 慢 1.1% |
| 24 | 12.37s | 4.85 | 5.91GB | 21.66GB | 慢 1.9% |
| 26 | 12.48s | 4.81 | 5.04GB | 22.25GB | 慢 2.8% |

每增加 2 个 swap 大约省 0.87GB GPU、增加 0.6GB host RSS。生产建议 swap24：相对
swap20 只损失约 1.9% 延迟，却增加约 1.74GB GPU 余量。

## 算子级速度解释

| 算子，BF16 | PG199 | RTX 3080 | 3080/PG199 |
| --- | ---: | ---: | ---: |
| Linear 6144×6144，4608 tokens | 1.523ms | 7.844ms | 5.15x |
| NF4 Linear 6144×6144 | 1.908ms | 8.870ms | 4.65x |
| Linear 16384×6144 | 4.025ms | 17.831ms | 4.43x |
| NF4 Linear 16384×6144 | 4.976ms | 20.354ms | 4.09x |
| cuDNN GQA + dense mask | 4.952ms | 14.015ms | 2.83x |

完整训练 `12.14/3.37=3.60x` 正好落在 attention 的 2.83x 与大 Linear 的 4.1-5.2x
之间。PG199 compiled profile 中 GEMM + attention 又占约 89%，因此卡间差距主要是硬件
吞吐，而不是 Python、H2D 或 LoRA rank。

Flash varlen 的独立 forward+backward attention 微基准为 PG199 `27.93→14.80ms`
（-47%）和 RTX 3080 `47.12→30.94ms`（-34%）。全模型收益较小，是因为 Flash 不会
加速占比更大的 GEMM、NF4 反量化和 block-swap 路径。

## 无法作为生产速度档的结果

| 候选 | 测得速度信号 | 拒绝原因 |
| --- | --- | --- |
| FP16 NF4 | 3080 Linear 快 13.7%；NF4 backward 27.071→10.018ms | 输入梯度 rel-L2 35.65% |
| every-other checkpoint on 3080 | 理论可省重算 | 放开单 block 即 OOM |
| reduce-overhead | 理论减少 launch | CUDA Graph 覆盖 checkpoint 输出，运行时报错 |
| rank16→8 | 参数减半 | 整步速度持平 |
| padding 裁剪 | 总长度 4608→4107 | PG199 3.370→3.366s，噪声级；3080 无收益 |
| 每步 prepare/H2D overlap | 可见约 196ms 探针开销 | 生产稳态本来不每步 prepare；H2D 实测约 0% |

## 选择建议

- PG199 生产安全档：full checkpoint + fixed cuDNN resident compile，约 `2.726s/it`。
- PG199 显存换速度档：every-other eager，约 `2.901s/it / 28.46GB`。
- PG199 极限研究档：16/28 checkpoint + compile，`2.408s/it / 31.55GB`，不适合长训。
- RTX 3080 生产档：NF4 + full checkpoint + resident compile + swap24；预期热稳态仍约
  `12.5-13s/it` 量级，但显存余量比 swap20 更可靠。
- Flash varlen：PG199 `2.421s/it`、3080 热稳态 `12.145s/it`，现为受支持的
  `attn_mode="flash"` 显式 opt-in；生产默认仍是 `attn_mode="torch"`。
