# Optimizations 文档索引

一句话：这里汇总 compile、kernel、显存和训练性能优化相关文档。

状态：稳定
适用版本：当前 main
相关入口：`docs/README.md`

## 当前文档

| 文档 | 说明 |
| --- | --- |
| [for_compile.md](for_compile.md) | 为 torch.compile / dynamo 做过的结构调整 |
| [fa4.md](fa4.md) | Flash Attention 4 评估和移除原因 |
| [adamw_fused.md](adamw_fused.md) | AdamW8bit 切换到 fused AdamW 的原因 |
| [hydra_analysis.md](hydra_analysis.md) | HydraLoRA + ReFT nsys 优化记录 |
| [training_profiling.md](training_profiling.md) | 训练性能 profiling 落地流程 |

## 维护规则

新增性能、compile、kernel、显存或 profiling 文档时，优先放在本目录，并同步更新本索引和上级 [docs/README.md](../README.md)。
