# Experimental 索引

一句话：这里放可运行但仍在实验、调参或验证阶段的方法说明。

状态：实验
适用版本：当前 main
相关入口：`docs/README.md`

## 当前文档

| 文档 | 说明 |
| --- | --- |
| [anima_tagger.md](anima_tagger.md) | Anima Tagger，多标签 tagger 与 DirectEdit 文本入口 |
| [byg.md](byg.md) | BYG unpaired instruction-editing：训练可用，专用推理仍是占位 |
| [chimera-hydra.md](chimera-hydra.md) | ChimeraHydra 双池 MoE，配合 [../structure/chimera-hydra.md](../structure/chimera-hydra.md) 阅读 |
| [convrot_int8_training.md](convrot_int8_training.md) | ConvRot int8 训练（W8A16/W8A8）；**可运行实验、默认关闭**；规格见 [../proposal/convrot_w8a_training_plan.md](../proposal/convrot_w8a_training_plan.md)；后续优化见 [../proposal/convrot_w8a_optimization_roadmap.md](../proposal/convrot_w8a_optimization_roadmap.md) |
| [directedit_editing_v3.md](directedit_editing_v3.md) | DirectEdit v3，flow-inversion 图像编辑 |
| [dpdmd.md](dpdmd.md) | DP-DMD / Turbo Anima 少步蒸馏，配合 [../structure/dpdmd.md](../structure/dpdmd.md) 阅读 |
| [easycontrol.md](easycontrol.md) | EasyControl 图像条件控制 |
| [fera.md](fera.md) | FeRA / FEI 路由实验 |
| [ip-adapter.md](ip-adapter.md) | IP-Adapter 图像 cross-attention 条件 |
| [postfix.md](postfix.md) | Postfix 兼容入口，当前用户入口见训练参考 |
| [soft_tokens.md](soft_tokens.md) | Soft Tokens / SoftREPA 风格 per-layer token bank |
| [spd.md](spd.md) | SPD：Spectral Progressive Diffusion 推理实验 |
| [vera_ablation.md](vera_ablation.md) | VeRA 短期消融计划 |
| [vr_loss.md](vr_loss.md) | Variance reduction loss 实验 |

## 维护规则

- 文档顶部或正文要说清楚实验边界、占位能力和当前可运行入口。
- 能力稳定并进入主路径后，迁到 `docs/methods/`，并同步更新两边索引。
- 原理、数学和架构细节放到 `docs/structure/`。
- 新增文档后，同步更新本索引和 [../README.md](../README.md)。
