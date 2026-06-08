# Anima LoRA 文档索引

这里是 Anima LoRA WebUI 的项目文档入口。当前整理原则是：

- **用户入口中文优先**：新用户先看中文指南、Linux 部署和训练/推理参考。
- **方法文档分层**：`methods/` 写稳定或已接入能力的使用方式，`experimental/` 写可运行但仍在实验阶段的能力，`structure/` 写原理、数学和架构图解。
- **实验记录保留上下文**：`findings/` 放结论、失败路径、调研报告和调参记录。
- **归档优先**：过期或缺失上下文的历史提案优先放到 `_archive/docs/`；仍在代码注释里出现但本文档树没有对应文件的旧提案，不代表当前可直接执行。

## 推荐阅读路径

| 目标 | 建议入口 |
|------|----------|
| 从零安装和跑 WebUI | [guidelines/指南书.md](guidelines/指南书.md)、[guidelines/linux-deployment.zh.md](guidelines/linux-deployment.zh.md) |
| 了解训练参数和方法选择 | [guidelines/training.md](guidelines/training.md) |
| 了解推理参数、DCW、Spectrum | [guidelines/inference.md](guidelines/inference.md) |
| 维护或扩展训练方法 | 先读 [structure/anima.md](structure/anima.md)，再读对应 `methods/` 或 `experimental/` 文档 |
| 复盘实验和技术取舍 | [findings/](findings/) 与 [optimizations/](optimizations/) |

## Guidelines

面向用户和维护者的操作指南。

| 文档 | 说明 |
|------|------|
| [guidelines/指南书.md](guidelines/指南书.md) | 中文综合指南，覆盖安装、数据集、WebUI、训练、推理和 ComfyUI 部署 |
| [guidelines/linux-deployment.zh.md](guidelines/linux-deployment.zh.md) | Linux 部署与启动指南 |
| [guidelines/guidebook.md](guidelines/guidebook.md) | 英文综合指南 |
| [guidelines/ガイドブック.md](guidelines/ガイドブック.md) | 日文综合指南 |
| [guidelines/가이드북.md](guidelines/가이드북.md) | 韩文综合指南 |
| [guidelines/training.md](guidelines/training.md) | 训练参考：LoRA 变体、caption shuffle、masked loss、数据集配置 |
| [guidelines/inference.md](guidelines/inference.md) | 推理参考：推理命令、P-GRAFT、DCW、Spectrum、prompt 文件 |
| [guidelines/difference_between_comfy.md](guidelines/difference_between_comfy.md) | anima_lora 与 ComfyUI 核心实现差异 |

## Methods

稳定或已接入的训练/推理能力。偏“怎么用、怎么配置、运行时行为是什么”。

| 文档 | 说明 |
|------|------|
| [methods/hydra-lora.md](methods/hydra-lora.md) | HydraLoRA 多专家路由，配合 [structure/hydralora.md](structure/hydralora.md) 阅读 |
| [methods/psoft-integrated-ortholora.md](methods/psoft-integrated-ortholora.md) | OrthoLoRA / Cayley 正交参数化，配合 [structure/ortholora.md](structure/ortholora.md) 阅读 |
| [methods/timestep_mask.md](methods/timestep_mask.md) | T-LoRA 时间步 rank mask，配合 [structure/timestep-mask.md](structure/timestep-mask.md) 阅读 |
| [methods/reft.md](methods/reft.md) | ReFT 残差流表示编辑，配合 [structure/reft.md](structure/reft.md) 阅读 |
| [methods/mod-guidance.md](methods/mod-guidance.md) | Modulation guidance，基于 pooled-text AdaLN steering |
| [methods/invert.md](methods/invert.md) | Embedding inversion 与 K-slot reference inversion |
| [methods/spectrum.md](methods/spectrum.md) | Spectrum 推理加速，配合 [structure/spectrum.md](structure/spectrum.md) 阅读 |
| [methods/dcw.md](methods/dcw.md) | DCW：post-step SNR-t bias correction |
| [methods/smc_cfg.md](methods/smc_cfg.md) | SMC-CFG / CFG-Ctrl 风格控制器 |
| [methods/cns.md](methods/cns.md) | Colored Noise Sampling |
| [methods/channel_scaling.md](methods/channel_scaling.md) | 通道缩放相关方法记录 |

Postfix 当前入口在 [guidelines/training.md#postfix](guidelines/training.md#postfix)；[experimental/postfix.md](experimental/postfix.md) 只保留兼容跳转，不再恢复旧的独立深文档。

## Experimental

可运行但仍在实验、调参或验证阶段的方法。

| 文档 | 说明 |
|------|------|
| [experimental/anima_tagger.md](experimental/anima_tagger.md) | Anima Tagger，多标签 tagger 与 DirectEdit 文本入口 |
| [experimental/chimera-hydra.md](experimental/chimera-hydra.md) | ChimeraHydra 双池 MoE，配合 [structure/chimera-hydra.md](structure/chimera-hydra.md) 阅读 |
| [experimental/directedit_editing_v3.md](experimental/directedit_editing_v3.md) | DirectEdit v3，flow-inversion 图像编辑 |
| [experimental/dpdmd.md](experimental/dpdmd.md) | DP-DMD / Turbo Anima 少步蒸馏，配合 [structure/dpdmd.md](structure/dpdmd.md) 阅读 |
| [experimental/easycontrol.md](experimental/easycontrol.md) | EasyControl 图像条件控制 |
| [experimental/fera.md](experimental/fera.md) | FeRA / FEI 路由实验 |
| [experimental/ip-adapter.md](experimental/ip-adapter.md) | IP-Adapter 图像 cross-attention 条件 |
| [experimental/postfix.md](experimental/postfix.md) | Postfix 兼容入口，当前用户入口见训练参考 |
| [experimental/soft_tokens.md](experimental/soft_tokens.md) | Soft Tokens / SoftREPA 风格 per-layer token bank |
| [experimental/spd.md](experimental/spd.md) | SPD：Spectral Progressive Diffusion 推理实验 |
| [experimental/vera_ablation.md](experimental/vera_ablation.md) | VeRA 短期消融计划 |
| [experimental/vr_loss.md](experimental/vr_loss.md) | Variance reduction loss 实验 |

## Structure

原理、数学、架构图和实现结构说明。通常与 `methods/` 或 `experimental/` 的使用文档成对阅读。

| 文档 | 说明 |
|------|------|
| [structure/anima.md](structure/anima.md) | Anima 模型结构、文本条件、VAE latent 和训练步 |
| [structure/anima-optimizations.md](structure/anima-optimizations.md) | Anima 性能与 compile 优化结构说明 |
| [structure/lora.md](structure/lora.md) | Plain LoRA 在 Anima 中的接入方式 |
| [structure/ortholora.md](structure/ortholora.md) | OrthoLoRA 正交基和 Cayley 参数化 |
| [structure/timestep-mask.md](structure/timestep-mask.md) | T-LoRA rank schedule 与 mask 应用 |
| [structure/hydralora.md](structure/hydralora.md) | HydraLoRA layer-local MoE 原理 |
| [structure/reft.md](structure/reft.md) | ReFT residual-stream intervention 原理 |
| [structure/modulation.md](structure/modulation.md) | pooled-text modulation 与 mod-guidance hook 点 |
| [structure/spectrum.md](structure/spectrum.md) | Spectrum Chebyshev feature forecasting 原理 |
| [structure/chimera-hydra.md](structure/chimera-hydra.md) | ChimeraHydra 双池 additive MoE 原理 |
| [structure/dpdmd.md](structure/dpdmd.md) | DP-DMD 多角色 LoRA 蒸馏结构 |

相关图像在 [structure_images/](structure_images/)；韩文图像在 [structure_images_korean/](structure_images_korean/)。

## Findings

实验结论、失败路径、调研和运行报告。

| 文档 | 说明 |
|------|------|
| [findings/webui_frontend_visual_audit_20260530.md](findings/webui_frontend_visual_audit_20260530.md) | WebUI 视觉和交互审计 |
| [findings/webui_god_files_refactor_20260607.md](findings/webui_god_files_refactor_20260607.md) | WebUI 上帝文件治理合并记录 |
| [findings/selfflow.md](findings/selfflow.md) | Self-Flow rep-loss 在冻结 Anima backbone 上的否定结果 |
| [findings/mod_guidance_quality_tag_axis.md](findings/mod_guidance_quality_tag_axis.md) | Mod-guidance quality tag 轴分析 |
| [findings/channel_stats_content_independence.md](findings/channel_stats_content_independence.md) | channel stats 与 content independence 分析 |
| [findings/asymflow_parameterization.md](findings/asymflow_parameterization.md) | Anima velocity / sigma 参数化记录 |
| [findings/l2p_pixel_transfer.md](findings/l2p_pixel_transfer.md) | L2P pixel transfer 调研 |
| [findings/fasterdit_signal_densification_plan.md](findings/fasterdit_signal_densification_plan.md) | FasterDiT signal densification 计划 |
| [findings/anima_lokr_blockswap_oom_report.md](findings/anima_lokr_blockswap_oom_report.md) | LoKr 16G block swap OOM 报告，含 next-goal 收口 |
| [findings/lokr_anima_shaojianV1_run_report.md](findings/lokr_anima_shaojianV1_run_report.md) | lokr-anima-shaojianV1 运行报告 |
| [findings/anima_balanced_16g_blockswap_ablation_plan.md](findings/anima_balanced_16g_blockswap_ablation_plan.md) | Balanced 16G block swap 消融 |
| [findings/anima_fp8_blockswap_transfer_report.md](findings/anima_fp8_blockswap_transfer_report.md) | FP8 block swap transfer 最终报告，含原消融计划口径 |

配套截图和图表在 [findings/assets/](findings/assets/)。

## Optimizations

编译、kernel、显存和训练性能相关文档。

| 文档 | 说明 |
|------|------|
| [optimizations/for_compile.md](optimizations/for_compile.md) | 为 torch.compile / dynamo 做过的结构调整 |
| [optimizations/fa4.md](optimizations/fa4.md) | Flash Attention 4 评估和移除原因 |
| [optimizations/adamw_fused.md](optimizations/adamw_fused.md) | AdamW8bit 切换到 fused AdamW 的原因 |
| [optimizations/hydra_analysis.md](optimizations/hydra_analysis.md) | HydraLoRA + ReFT nsys 优化记录 |

## Proposals

仍保留在 `docs/proposal/` 的活跃或半活跃提案。缺失上下文的历史提案优先归档到 `_archive/docs/proposal/`。

| 文档 | 说明 |
|------|------|
| [proposal/turbo_anima_dmd_lora.md](proposal/turbo_anima_dmd_lora.md) | Turbo Anima / DMD LoRA 蒸馏提案 |
| [proposal/soft_tokens_contrastive.md](proposal/soft_tokens_contrastive.md) | Soft Tokens contrastive 兼容入口 |
| [proposal/soft_tokens_softrank.md](proposal/soft_tokens_softrank.md) | Soft Tokens soft-rank 兼容入口 |
| [proposal/soft_tokens_agsm.md](proposal/soft_tokens_agsm.md) | Soft Tokens AGSM 提案 |
| [proposal/prior_preservation_from_synth_pool.md](proposal/prior_preservation_from_synth_pool.md) | synth pool prior preservation 提案 |
| [proposal/postfix_residual_for_directedit.md](proposal/postfix_residual_for_directedit.md) | DirectEdit image-conditional postfix residual 提案 |
| [proposal/postfix_residual_per_image_inversion.md](proposal/postfix_residual_per_image_inversion.md) | per-image postfix tail inversion probe 提案 |

## Repo-Level Notes

| 文档 | 说明 |
|------|------|
| [multi_model_support.md](multi_model_support.md) | 多模型支持的仓库级架构草案 |
| [separation_plan.md](separation_plan.md) | 训练/推理/文档分离计划记录 |

`side_by_side/` 保存 LoRA / OrthoLoRA / T-LoRA 等结果对比图；它不是文字文档入口。
