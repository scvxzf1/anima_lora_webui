# Anima LoRA 文档导航

这里是 Anima LoRA WebUI 的唯一文档总入口，负责把读者带到正确的分区。

## 先看哪里

不同读者先看不同入口，避免一上来掉进实验报告堆里。

| 你想做什么 | 先看 |
| --- | --- |
| 从零安装并启动 WebUI | [../README.md](../README.md#linux-部署启动)、[guidelines/linux-deployment.zh.md](guidelines/linux-deployment.zh.md) |
| 使用 WebUI 与命令行全流程 | [../README.md](../README.md#启动后怎么用)、[guidelines/training.md](guidelines/training.md) |
| 学会训练和选方法 | [guidelines/training.md](guidelines/training.md) |
| 学会推理、DCW、Spectrum | [guidelines/inference.md](guidelines/inference.md) |
| 理解模型和训练结构 | [structure/anima.md](structure/anima.md)、[structure/anima-optimizations.md](structure/anima-optimizations.md) |
| 查配置、外置配置和路径规则 | [configuration/README.md](configuration/README.md) |
| 查 WebUI 独立功能 | [features/README.md](features/README.md) |
| 快速审核当前分支前端健康度 | [features/frontend-health-scorecard.md](features/frontend-health-scorecard.md)、[superpowers/specs/2026-07-11-five-round-auto-iteration-protocol.md](superpowers/specs/2026-07-11-five-round-auto-iteration-protocol.md) |
| 查实验结论、审计和运行报告 | [findings/README.md](findings/README.md) |
| 查仍可能推进的提案 | [proposal/README.md](proposal/README.md) |
| 查看当前贡献优先事项 | [contribution-priorities.md](contribution-priorities.md)、[../CONTRIBUTING.md](../CONTRIBUTING.md) |
| 查历史或已完成提案 | [archive-index.md](archive-index.md) |

## 维护规则

新增、移动或归档文档时，先按下面规则放。

| 分区 | 放什么 | 规则 |
| --- | --- | --- |
| `guidelines/` | 用户操作指南 | 中文主路径优先；WebUI 以根 README/Linux 部署指南为准；旧多语种《指南书》仅作历史参考 |
| `methods/` | 稳定或已接入能力 | 偏“怎么用、怎么配置、运行时行为是什么” |
| `experimental/` | 可运行但仍实验的能力 | 顶部或正文要说清楚实验边界和占位能力 |
| `structure/` | 原理、数学、架构 | 和 `methods/` / `experimental/` 配套阅读 |
| `configuration/` | 配置、路径、环境变量 | 配置字段和路径规则变更时同步更新 |
| `features/` | UI 或产品功能 | WebUI 独立功能说明放这里 |
| `findings/` | 审计、实验结论、失败路径 | 不作为新用户主路径，用分区索引归档上下文 |
| `optimizations/` | compile、kernel、显存、性能 | 只记录当前仍有维护价值的优化说明 |
| `proposal/` | 活跃或半活跃提案 | 完成、过期或只服务旧合并工作的提案移到 `_archive/docs/proposal/` |
| `superpowers/` | 当前迭代的 spec / plan 施工区 | 不作为用户主路径，完成后迁入 findings/正式文档或归档 |

所有 `docs/` 下新增 Markdown 必须从本页或一个分区索引可达。

文档结构、内部链接、章节锚点、代码围栏、分区索引、生命周期标记和部分
“当前配置事实”由以下定向测试守护：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q
```

## Guidelines

这一组是用户和维护者最常用的操作说明。

| 文档 | 说明 |
| --- | --- |
| [guidelines/README.md](guidelines/README.md) | Guidelines 分区索引 |
| [guidelines/指南书.md](guidelines/指南书.md) | 历史中文 GUI/CLI 综合指南；桌面 GUI 命令已移除，当前入口以根 README 为准 |
| [guidelines/linux-deployment.zh.md](guidelines/linux-deployment.zh.md) | Linux 部署与启动指南 |
| [guidelines/git-sync-policy.md](guidelines/git-sync-policy.md) | 本地 `main` 与线上 `webui/main` 的同步规则 |
| [guidelines/training.md](guidelines/training.md) | 训练参考：LoRA 变体、caption shuffle、masked loss、数据集配置 |
| [guidelines/inference.md](guidelines/inference.md) | 推理参考：推理命令、DCW、Spectrum、prompt 文件 |
| [guidelines/difference_between_comfy.md](guidelines/difference_between_comfy.md) | anima_lora 与 ComfyUI 核心实现差异 |
| [guidelines/guidebook.md](guidelines/guidebook.md) | 历史英文 GUI/CLI 综合指南 |
| [guidelines/ガイドブック.md](guidelines/ガイドブック.md) | 历史日文 GUI/CLI 综合指南 |
| [guidelines/가이드북.md](guidelines/가이드북.md) | 历史韩文 GUI/CLI 综合指南 |

## Methods

这一组是稳定或已接入能力的使用说明。

分区索引：[methods/README.md](methods/README.md)。

| 文档 | 说明 |
| --- | --- |
| [methods/lokr.md](methods/lokr.md) | LoKr Kronecker 分解、全因子模式与旧哨兵迁移 |
| [methods/loha.md](methods/loha.md) | LoHa Hadamard 积分解：兼容可用、非主力，PEFT/LyCORIS 键布局 |
| [methods/hydra-lora.md](methods/hydra-lora.md) | HydraLoRA 多专家路由，配合 [structure/hydralora.md](structure/hydralora.md) 阅读 |
| [methods/psoft-integrated-ortholora.md](methods/psoft-integrated-ortholora.md) | OrthoLoRA / Cayley 正交参数化，配合 [structure/ortholora.md](structure/ortholora.md) 阅读 |
| [methods/timestep_mask.md](methods/timestep_mask.md) | T-LoRA 时间步 rank mask，配合 [structure/timestep-mask.md](structure/timestep-mask.md) 阅读 |
| [methods/reft.md](methods/reft.md) | ReFT 残差流表示编辑，配合 [structure/reft.md](structure/reft.md) 阅读 |
| [methods/mod-guidance.md](methods/mod-guidance.md) | Modulation guidance，基于 pooled-text AdaLN steering |
| [methods/invert.md](methods/invert.md) | 历史 inversion 说明；当前可运行入口是 `exp-invert-directedit` 探针 |
| [methods/spectrum.md](methods/spectrum.md) | Spectrum 推理加速，配合 [structure/spectrum.md](structure/spectrum.md) 阅读 |
| [methods/dcw.md](methods/dcw.md) | DCW：post-step SNR-t bias correction |
| [methods/smc_cfg.md](methods/smc_cfg.md) | SMC-CFG / CFG-Ctrl 风格控制器 |
| [methods/cns.md](methods/cns.md) | Colored Noise Sampling |
| [methods/channel_scaling.md](methods/channel_scaling.md) | 通道缩放相关方法记录 |

Postfix 当前用户入口在 [guidelines/training.md#postfix](guidelines/training.md#postfix)，[experimental/postfix.md](experimental/postfix.md) 只保留兼容跳转。

## Experimental

这一组是可运行但仍在实验、调参或验证阶段的方法。

分区索引：[experimental/README.md](experimental/README.md)。

| 文档 | 说明 |
| --- | --- |
| [experimental/anima_tagger.md](experimental/anima_tagger.md) | Anima Tagger，多标签 tagger 与 DirectEdit 文本入口 |
| [experimental/byg.md](experimental/byg.md) | BYG unpaired instruction-editing：训练可用，专用推理仍是占位 |
| [experimental/chimera-hydra.md](experimental/chimera-hydra.md) | ChimeraHydra 双池 MoE，配合 [structure/chimera-hydra.md](structure/chimera-hydra.md) 阅读 |
| [experimental/convrot_int8_training.md](experimental/convrot_int8_training.md) | ConvRot int8 训练（W8A16/W8A8）；可运行实验、默认关闭；后续优化见 [proposal/convrot_w8a_optimization_roadmap.md](proposal/convrot_w8a_optimization_roadmap.md) |
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

这一组解释原理、数学和实现结构。

分区索引：[structure/README.md](structure/README.md)。

| 文档 | 说明 |
| --- | --- |
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

相关图片在 [structure_images/](structure_images/) 和 [structure_images_korean/](structure_images_korean/)。

## Configuration And Features

这一组承接配置和独立功能说明。

| 文档 | 说明 |
| --- | --- |
| [configuration/README.md](configuration/README.md) | 配置文档分区索引 |
| [configuration/external-configs.md](configuration/external-configs.md) | `ANIMA_CONFIGS_ROOT` 和 WebUI 外置配置根目录说明 |
| [features/README.md](features/README.md) | 功能文档分区索引 |
| [features/config-workbench.md](features/config-workbench.md) | 配置工作台：预设、表单、启动与续接 |
| [features/dataset-editor.md](features/dataset-editor.md) | 数据集蓝图编辑器 |
| [features/training-queue.md](features/training-queue.md) | 训练队列管理 |
| [features/history-collections.md](features/history-collections.md) | 历史任务与集合 |
| [features/preview.md](features/preview.md) | 训练/推理预览与权重列表 |
| [features/global-settings.md](features/global-settings.md) | 全局输出、模型、配置根与界面设置 |
| [features/ui-scale.md](features/ui-scale.md) | WebUI UI 缩放（默认与分页面） |
| [features/frontend-health-scorecard.md](features/frontend-health-scorecard.md) | 前端健康度评分卡（维护用） |

## Findings And Proposals

这一组是报告和提案，不放在新手主路径里。

| 分区 | 说明 |
| --- | --- |
| [findings/README.md](findings/README.md) | 审计、实验结论、失败路径、运行报告索引 |
| [proposal/README.md](proposal/README.md) | 活跃或半活跃提案索引 |
| [superpowers/README.md](superpowers/README.md) | 当前迭代规格、执行计划和迭代日志索引 |
| [archive-index.md](archive-index.md) | 已归档历史文档索引 |

## Optimizations

这一组是性能、显存、compile 和训练优化说明。

| 文档 | 说明 |
| --- | --- |
| [optimizations/README.md](optimizations/README.md) | 优化文档分区索引 |
| [optimizations/for_compile.md](optimizations/for_compile.md) | 为 torch.compile / dynamo 做过的结构调整 |
| [optimizations/fa4.md](optimizations/fa4.md) | Flash Attention 4 评估和移除原因 |
| [optimizations/adamw_fused.md](optimizations/adamw_fused.md) | AdamW8bit 切换到 fused AdamW 的原因 |
| [optimizations/hydra_analysis.md](optimizations/hydra_analysis.md) | HydraLoRA + ReFT nsys 优化记录 |
| [optimizations/training_profiling.md](optimizations/training_profiling.md) | 训练性能 profiling 落地流程 |
| [optimization-configs-current.md](optimization-configs-current.md) | 当前优化配置事实清单 |
| [optimization-roadmap.md](optimization-roadmap.md) | 优化路线图 |

## Repo-Level Notes

这一组是仓库级架构草案或拆分计划。

| 文档 | 说明 |
| --- | --- |
| [multi_model_support.md](multi_model_support.md) | 多模型支持的仓库级架构草案 |
| [separation_plan.md](separation_plan.md) | 训练/推理/文档分离计划记录 |

`side_by_side/` 保存 LoRA / OrthoLoRA / T-LoRA 等结果对比图，不是文字文档入口。
