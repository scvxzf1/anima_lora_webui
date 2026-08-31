# Proposal 索引

状态：索引
适用版本：当前文档树

这里只保留活跃、半活跃或刚完成待归档的提案。已经完成、过期或只服务旧合并工作的文档放到 [_archive/docs/proposal/](../../_archive/docs/proposal/)。

## 活跃或半活跃提案

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [auto_vram_v1.md](auto_vram_v1.md) | 提案 / 未实现 | AutoVram v1 自动显存档位搜索与可审计协议 |
| [krea2_raw_migration.md](krea2_raw_migration.md) | 核心已落地 / 待归档 | Anima → Krea-2-Raw 历史迁移计划；当前事实见 [多模型说明](../multi_model_support.md)，配套风险快照见 [krea2_raw_migration_notes.md](krea2_raw_migration_notes.md) |
| [dcgen_anima_f32c32.md](dcgen_anima_f32c32.md) | POC / 阶段 1-2 | DC-Gen Anima f32c32 latent space、双缓存和 patch 对齐探针记录 |
| [krea2_raw_gradient_checkpointing.md](krea2_raw_gradient_checkpointing.md) | 已实现 / 设计背景 | Krea-2-Raw 梯度检查点落地方案（阶段 4 子设计；已移植 Anima block 级 `use_reentrant=False` 机制并完成 1024² 训练验证） |
| [krea2_nf4_blockswap.md](krea2_nf4_blockswap.md) | 提案 / 已完成 | NF4 × block swap 落地方案（方向 A deepcopy 已落地 + 落盘/小卡链路 + compat_matrix 放开 + train.py 接线 + 5 格消融矩阵六维基准，见 [krea2_nf4_ablation_findings.md](../findings/krea2_nf4_ablation_findings.md)；方向 B slab+手动重建作为速度进阶存储） |
| [adapter-aware-checkpoint.md](adapter-aware-checkpoint.md) | 半活跃 | Adapter-aware activation checkpoint 可行性探索 |
| [turbo_anima_dmd_lora.md](turbo_anima_dmd_lora.md) | 半活跃 | Turbo Anima / DMD LoRA 蒸馏提案 |
| [prior_preservation_from_synth_pool.md](prior_preservation_from_synth_pool.md) | 半活跃 | synth pool prior preservation 提案 |
| [postfix_residual_for_directedit.md](postfix_residual_for_directedit.md) | 半活跃 | DirectEdit image-conditional postfix residual 提案 |
| [postfix_residual_per_image_inversion.md](postfix_residual_per_image_inversion.md) | 半活跃 | per-image postfix tail inversion probe 提案 |
| [soft_tokens_agsm.md](soft_tokens_agsm.md) | 半活跃 | Soft Tokens AGSM 提案 |
| [soft_tokens_contrastive.md](soft_tokens_contrastive.md) | 兼容入口 | Soft Tokens contrastive 方向记录 |
| [soft_tokens_softrank.md](soft_tokens_softrank.md) | 兼容入口 | Soft Tokens soft-rank 方向记录 |
| [personalization-region-curriculum.md](personalization-region-curriculum.md) | 半活跃 | 区域→整图课程、先验保持与 APT 风格自适应正则化的可行性及实施计划 |
| [convrot_w8a_training_plan.md](convrot_w8a_training_plan.md) | 提案 / 核心已实现（实验默认关闭） | ConvRot 战略 C：W8A16→W8A8 训练路径规格 + 落地状态 |
| [convrot_w8a_optimization_roadmap.md](convrot_w8a_optimization_roadmap.md) | 提案 / 半活跃（P0-A/A2/C/D 完成） | ConvRot 优化路线图；regular Hadamard opt-in；默认仍 sylvester；Triton 未达门槛 |

## 归档规则

- 已经落地并有正式说明的提案，移到 `_archive/docs/proposal/`。
- 服务一次性合并、审计或旧路线图的文档，移到 `_archive/docs/proposal/`。
- 算法仍可能继续推进、但尚未成为稳定方法的文档，继续留在本目录。
- 归档时同步更新 [../archive-index.md](../archive-index.md) 和 [_archive/docs/proposal/README.md](../../_archive/docs/proposal/README.md)。

已完成的 `anima-app` legacy bridge 收尾记录已归档到
[anima-app-legacy-bridge-cleanup.md](../../_archive/docs/proposal/anima-app-legacy-bridge-cleanup.md)。
