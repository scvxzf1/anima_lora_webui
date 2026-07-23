# Proposal 索引

状态：索引
适用版本：当前文档树

这里只保留活跃、半活跃或刚完成待归档的提案。已经完成、过期或只服务旧合并工作的文档放到 [_archive/docs/proposal/](../../_archive/docs/proposal/)。

## 活跃或半活跃提案

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [auto_vram_v1.md](auto_vram_v1.md) | 提案 / 未实现 | AutoVram v1 自动显存档位搜索与可审计协议 |
| [adapter-aware-checkpoint.md](adapter-aware-checkpoint.md) | 半活跃 | Adapter-aware activation checkpoint 可行性探索 |
| [turbo_anima_dmd_lora.md](turbo_anima_dmd_lora.md) | 半活跃 | Turbo Anima / DMD LoRA 蒸馏提案 |
| [prior_preservation_from_synth_pool.md](prior_preservation_from_synth_pool.md) | 半活跃 | synth pool prior preservation 提案 |
| [postfix_residual_for_directedit.md](postfix_residual_for_directedit.md) | 半活跃 | DirectEdit image-conditional postfix residual 提案 |
| [postfix_residual_per_image_inversion.md](postfix_residual_per_image_inversion.md) | 半活跃 | per-image postfix tail inversion probe 提案 |
| [soft_tokens_agsm.md](soft_tokens_agsm.md) | 半活跃 | Soft Tokens AGSM 提案 |
| [soft_tokens_contrastive.md](soft_tokens_contrastive.md) | 兼容入口 | Soft Tokens contrastive 方向记录 |
| [soft_tokens_softrank.md](soft_tokens_softrank.md) | 兼容入口 | Soft Tokens soft-rank 方向记录 |
| [personalization-region-curriculum.md](personalization-region-curriculum.md) | 半活跃 | 区域→整图课程、先验保持与 APT 风格自适应正则化的可行性及实施计划 |

## 归档规则

- 已经落地并有正式说明的提案，移到 `_archive/docs/proposal/`。
- 服务一次性合并、审计或旧路线图的文档，移到 `_archive/docs/proposal/`。
- 算法仍可能继续推进、但尚未成为稳定方法的文档，继续留在本目录。
- 归档时同步更新 [../archive-index.md](../archive-index.md) 和 [_archive/docs/proposal/README.md](../../_archive/docs/proposal/README.md)。

已完成的 `anima-app` legacy bridge 收尾记录已归档到
[anima-app-legacy-bridge-cleanup.md](../../_archive/docs/proposal/anima-app-legacy-bridge-cleanup.md)。
