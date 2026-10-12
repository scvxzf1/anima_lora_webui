# Anima FP16 梯度误差定位：分支与 scale 对照

状态：2026-09-30，Luna 执行两批有界 CMP 90HX 热测；全部候选仍未通过数值门槛。
这是[首阶段探针](anima_adaptive_precision_90hx_20260930.md)的后续定位，不是生产训练认证。

## 结论

- 单独将 MLP、attention 或 conditioning Linear 置为 FP16，均能完成更新，
  但输入梯度及 LoRA 梯度仍显著偏离 FP32 参考。不是只有一个分支敏感。
- 前 8 个 block 占逐步 adapter-gradient 误差平方和的 72.3%--92.4%；
  最后 8 个 block 占比不超过 0.12%。这是误差出现的位置，不能证明源头就在这些 block。
- 同一精度计划下 scale 从 1 提升至 64，最大 LoRA 梯度 relative-L2 从 56.90%
  降至 14.37%；1024 为 14.14%，仍远超 2% 门槛，且改善不逐项单调。
  缩放有帮助，但不能把 scale 不足当作唯一原因，也不能继续只提高 scale。
- 没有放宽比较门槛、生成推荐名单、修改生产默认或启动 OOM 重试。
  CLI 的 exit 0 只表示诊断矩阵完成，六个候选均为 numerical_rejection。

## 固定条件与证据

GPU 为 `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，CMP 90HX 10 GiB、SM86。
每批分别新录制 FP32 reference，再将本批各候选重放至相同 pre-step adapter、
输入和 RNG 状态。每个 worker 三步，sigma 依次为 0.2/0.5/0.8；
swap20、full checkpoint、torch attention、无 compile/TF32、LoRA rank/alpha 4/4、
AdamW lr=1e-4。两个 reference 分别记录，不能声称跨批逐位一致。

源 checkpoint 有 685 个 BF16 tensor，FP32 参考是这些已存数值的提升，不是原始
FP32 预训练权重。latent/noise `[1,16,1,150,112]`、prompt `[1,512,1024]`，
文本保持完整 padding，latent mask 全零，输入 replay 为 FP32。

| 身份 | SHA-256 |
| --- | --- |
| checkpoint | `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e` |
| inputs | `0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78` |
| probe source | `d17e94eef679d3df8e694b6a86498311acb121c6719f6ba6ceeebe2e39dec2b0` |

数据仍仅单图缓存、单 bucket。sigma 与 step/adapter 状态同时变化，不能把第二步
误差更大解释为 sigma=0.5 普遍最敏感。记录未保存 Adam moments；固定状态梯度
对比不证明独立 optimizer 轨迹等价。

## 分支对照

下表为三步中的最大 relative-L2，已经乘 100 转为百分数；各列最大值未必同一步。
FP16 数量仅统计 Linear，不代表算力占比。

| 仅这些 Linear 使用 FP16 | FP16 / FP32 Linear | Prediction | Input grad | LoRA grad |
| --- | ---: | ---: | ---: | ---: |
| blocks MLP | 56 / 321 | 0.0327% | 53.742% | 14.167% |
| blocks self/cross attention | 140 / 237 | 0.0259% | 127.692% | 35.027% |
| conditioning | 118 / 259 | 0.0677% | 92.063% | 30.112% |

conditioning 包括 `t_embedder.*`、`blocks.*.adaln_*`、`final_layer.*` 和
`x_embedder.proj.1`。仅指定分支低精度时，探针按实际普通 Linear 名单生成其余
FP32 精确名称；各 case 的 manifest 保存完整实际分配。

误差贡献分组是受影响的 LoRA 参数，不是精度干预所在分支。
例如 attention-only 的第二步，MLP / self-attn / cross-attn adapter-gradient
误差平方和贡献为 44.96% / 54.58% / 0.45%，虽然 MLP 基座此时全 FP32。
说明低精度分支的影响会传播到其它分支梯度；不能按贡献直接推导必须提升的层。

## Loss Scale 对照

这三个 case 的实际计划完全一致：119 个 FP16、258 个 FP32 Linear。
conditioning 与 blocks attention 为 FP32；其余主要含 blocks MLP 和
`llm_adapter` 投影，因此不等同于上一表的 MLP-only 组。

| Scale | 最大 Prediction rel-L2 | 最大 Input grad rel-L2 | 最大 LoRA grad rel-L2 |
| --- | ---: | ---: | ---: |
| 1 | 0.0327% | 137.529% | 56.904% |
| 64 | 0.0327% | 53.487% | 14.367% |
| 1024 | 0.0327% | 53.619% | 14.139% |

scale1024 各步 LoRA relative-L2 为 14.139% / 11.827% / 5.710%，cosine 为
0.99004 / 0.99299 / 0.99839。三种 scale 的 loss 与 prediction 一致；提高 scale
只明显改变反向误差，仍不能判定剩余偏差来自 activation 舍入、局部 VJP 或累计传播中的哪项。

捕获清单每步 392 个梯度 tensor。第一步 reference/candidate 均有 2,179,072 个零
元素，主要是合法的初始 LoRA down/A 零梯度；后两步两侧均无零元素。
三种 scale 的 reference-zero/candidate-nonzero 与反向转移均为零。
这些是最终捕获梯度的观测，不排除中间反向边界的下溢，也不是硬件 FTZ 认证。

全部 loss、捕获梯度和梯度范数有限，scaler 保持各自初始值，AdamW step 递增核验通过，
adapter 更新非零。未保存生产 adapter 或训练 checkpoint，没有长训或图像质量验收。

## 资源与产物

Luna 依次运行两批，每批 reference 加三个候选，总计 8 个三步 worker。
两批峰值 GPU used 均 5236 MiB（含 reference）；峰值 RSS 分别约 18.34 / 18.33 GiB。
最低 host available 分别约 80.34 / 81.20 GiB，swap 约 1.956 GiB 基本不变
（后一批末尾减少约 256 KiB）。无 telemetry 查询错误、资源停止或 OOM。
这不是稳态吞吐/显存优化结论。结束时 90HX 为 17 MiB、0%，无本任务残留 worker。

结果目录：

- `output/runs/anima-adaptive-precision-90hx-20260930-diagnostic-branches/`
- `output/runs/anima-adaptive-precision-90hx-20260930-diagnostic-scales/`

每批 `environment.json` / `summary.json`，reference 的 capture/replay，
每 case 的 `result.json`、`supervisor.json`、`telemetry.jsonl`、`worker.log`、
`comparison.json`、`gradient-diagnostics.json` 均保留。分组报告使用 FP64 规约，
含逐 step/block/family/down-up 的参考范数、绝对/相对误差、cosine、误差贡献及零计数。

实验 CLI：

```bash
.venv/bin/python -m bench.adaptive_runtime.diagnose_anima_precision \
  --weights /path/to/anima-bf16.safetensors --inputs /path/to/inputs.safetensors \
  --output-dir output/runs/new-anima-diagnostics --gpu-uuid GPU-... \
  --swap 20 --steps 3 --worker-timeout 150 --max-cases 3 \
  --case conditioning-attention-scale1 --case conditioning-attention-scale64 \
  --case conditioning-attention-scale1024
```

case 选择还支持 `mlp-only-fp16`、`attention-only-fp16`、`conditioning-only-fp16`。
仅数值拒绝可继续后续 case；timeout、host/GPU/telemetry、cleanup 和普通运行异常停止矩阵。
GPU 空闲检查依然不是独占锁，检查与启动有竞态。

## 验证与下一步

后续[固定状态局部 VJP 与 cotangent 实测](anima_local_vjp_90hx_20260930.md)已完成，
区分出局部 cast/kernel 误差与整网轨迹偏移，仍未认证 FP16 候选。

热测前独立 Sol 核验：adaptive CPU 回归 631 passed、2 skipped、3 xfailed，
Ruff 通过；真实 CPU train_steps record/replay/compare/diagnose 两步集成通过。
修复聚合 gradient norm 非有限值的数值分类，以及 cleanup 覆盖 nonfinite 的停止分类。
后续补齐通过候选不早停、runtime/cleanup 停止的测试，不改变热测算法。
最终主代理复跑 `CUDA_VISIBLE_DEVICES='' timeout 120 .venv/bin/python -m pytest -q tests/test_adaptive*.py`
为 634 passed、2 skipped、3 xfailed（退出码 0），修改 Python 文件 Ruff 通过。
三项 xfail 是既有共享参数边界缺口，不计作通过。

下一步应固定同一 adapter 状态与 sigma，做 block 边界 cotangent 及局部 VJP 对照，
分离 FP16 forward activation 舍入和 backward 边界误差；必要时进一步区分 self/cross
attention、MLP layer1/layer2 与 `llm_adapter`。不能直接按早期 block 名单增加 FP32，
也不能依据此三步判断长期收敛或 Turing 可用性。
