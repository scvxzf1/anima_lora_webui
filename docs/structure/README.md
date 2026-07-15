# Structure 索引

一句话：这里解释原理、数学和实现结构，配合 methods / experimental 一起读。

状态：稳定
适用版本：当前 main
相关入口：`docs/README.md`

## 当前文档

| 文档 | 说明 |
| --- | --- |
| [anima.md](anima.md) | Anima 模型结构、文本条件、VAE latent 和训练步 |
| [anima-optimizations.md](anima-optimizations.md) | Anima 性能与 compile 优化结构说明 |
| [lora.md](lora.md) | Plain LoRA 在 Anima 中的接入方式 |
| [ortholora.md](ortholora.md) | OrthoLoRA 正交基和 Cayley 参数化 |
| [timestep-mask.md](timestep-mask.md) | T-LoRA rank schedule 与 mask 应用 |
| [hydralora.md](hydralora.md) | HydraLoRA layer-local MoE 原理 |
| [reft.md](reft.md) | ReFT residual-stream intervention 原理 |
| [modulation.md](modulation.md) | pooled-text modulation 与 mod-guidance hook 点 |
| [spectrum.md](spectrum.md) | Spectrum Chebyshev feature forecasting 原理 |
| [chimera-hydra.md](chimera-hydra.md) | ChimeraHydra 双池 additive MoE 原理 |
| [dpdmd.md](dpdmd.md) | DP-DMD 多角色 LoRA 蒸馏结构 |

相关图片在 [../structure_images/](../structure_images/) 和 [../structure_images_korean/](../structure_images_korean/)。

## 维护规则

- 本目录写“为什么这样设计、结构怎么串”，不写完整用户操作手册。
- 使用说明放到 `docs/methods/` 或 `docs/experimental/`，两边互相链接。
- 新增结构文档后，同步更新本索引和 [../README.md](../README.md)。
