# Anima block17 内部残差阶段 VJP 定位

状态：2026-10-01，主代理执行 analysis-only 热测，随后从落盘张量使用 NumPy
独立 FP64 CPU 归约核验。前置：[上游15/17/27完整VJP](anima_upstream_vjp_90hx_20260930.md)。
本轮没有新训练，不修改生产精度默认、数值门槛或 OOM 策略。

## 结果

固定 reference FP32 policy、conditioning 与 block 末端参考 cotangent，分别重放
真实参考输入 x_R 和候选输入 x_C。各局部阶段再固定其对应参考阶段输出 cotangent，
比较完整残差 VJP 与仅 gated branch VJP。候选侧只是输入来自原 FP16 候选，
本轮两个重放都使用同一个 FP32 block。

| 阶段 | 阶段输入差异 rel-L2 | 完整残差 VJP 差异 | gated branch VJP 差异 | 完整 VJP 绝对 L2 差异 |
| --- | ---: | ---: | ---: | ---: |
| self-attention | 0.127923% | 0.676043% | 1.263987% | 1.363614e-7 |
| cross-attention | 0.111326% | 0.080592% | 0.604278% | 1.466318e-8 |
| MLP | 0.103095% | 0.163601% | 0.224073% | 2.939929e-8 |

表中百分数是原始比值乘100，分母分别为对应参考张量的 L2 norm。
同末端 cotangent 的整块输入 VJP 差异为 **0.696389%**。
自注意力阶段在本样本/方向上的完整梯度对照最大，且绝对差异最大，
下一步优先拆查其前置 normalization/modulation、QKV、Q/K normalization、attention
与输出投影。阶段包含整个分支路径，尚不能归因到 attention 算子或 normalization。

branch 的分母不同，百分数不能直接与 complete 比较成放大倍数；三个局部阶段固定的
cotangent 也不同，不能把其误差相加成整块因果贡献。没有修改候选方案，仍不能
推荐 FP16 安全层名单、宣称最初误差源头或认证训练质量。

## 实现和验收

新增 `anima_block_stages.py` 仅在 bench 隔离 block 中包装三次 `_gated_residual_add`，
捕获四个激活/梯度边界和三组 gate；非重入 checkpoint 的记录/重算 context 可重复进入，
重算不重复记 stage。结束时恢复原方法，不修改模型源码。
`anima_block17_paths.py` 复用原 CPU 模型加载与 LoRA replay，保留真实4200-token
`[1,1,75,56,2048]` 输入、torch attention、full checkpoint、无 TF32。

来源校验包含原 training captures、八格矩阵、实际 conditioning 逐位一致、weights/
capture/replay/helper/model-source SHA。新 forward 与未包装 baseline 逐位一致；
整块梯度与原矩阵 fidelity 约1.20e-7--1.35e-7，低于既有1e-4门槛。
六组 `complete = reference_stage_g + branch` 恒等式通过，最大 rel-L2 约1.30e-7。
阶段参考 VJP 也在 worker 内与对应完整链 cotangent 校验。

两个 trace 和一个 stage-matrix 共12个局部 VJP 张量，总文件 **963,431,664 B**，
各文件限512MiB、聚合限1GiB。父进程复核 inventory、FP32 dtype、shape、finite、SHA
和 fidelity。CPU 独立核验实际张量的三组输入/六组VJP/整块对比，指标与结果JSON差异
小于1e-12；另验六组残差恒等式，不调用 GPU 分析器的 metric helper。

## 热测与资源

成功产物：`output/runs/anima-adaptive-precision-90hx-20261001-block17-paths-r2/`，
包括 environment、summary、result、worker.log、telemetry、三个 evidence/manifest 和
`audit-cpu.json` / `audit-cpu.py`。固定 CMP90HX UUID
`GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，未操作显示GPU、功耗或风扇。

- analysis worker exit0，48.13秒、45次遥测；新增训练步数0。
- 峰值 GPU used 8136MiB，RSS tree 9,785,765,888B，温度49°C。
- 最低主机 available 102,029,299,712B；swap used0、遥测错误0，无OOM或非有限值。
- 最终 GPU used17MiB、utilization0%，owned worker退出。

首轮目录 `output/runs/anima-adaptive-precision-90hx-20261001-block17-paths/` 保留失败证据。
来源 comparison 在 GPU 模型加载前被拒绝：历史4线程 FP64 归约与新1线程的结果有
末位差异（例如prediction reference L2相差约1.1e-13）。入口改用相同4线程后比较JSON
完全一致；没有改来源文件、放宽 equality/fidelity 或覆盖失败目录。

```bash
timeout 360 .venv/bin/python -m bench.adaptive_runtime.diagnose_anima_block17_paths \
  --source-run output/runs/anima-adaptive-precision-90hx-20260930-upstream-vjp \
  --weights /path/to/anima-base-v1.0.safetensors \
  --output-dir output/runs/<fresh-block17-stage-analysis> \
  --gpu-uuid GPU-806debd9-d254-da57-d515-ccae15b4ac6a \
  --analysis-timeout 300
```

CPU 定向测试5 passed；新增模块完成后全 adaptive 回归720 passed、2 skipped、3 xfailed。
线程修正后定向5项再次通过。Ruff 与 diff whitespace 检查通过。
本轮独立计算核验由主代理执行；未完成另一代理的独立代码验收（新代理创建遇到线程上限）。

后续：[自注意力归一化与投影/核心拆分](anima_attention_split_90hx_20261001.md)已完成。
