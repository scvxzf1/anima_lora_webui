# 文档库合并整理报告 - 2026-07-06

本文记录本轮文档库整理的盘点、合并、归档和验证结果，作为后续维护的基线。

## 目标

本轮目标是完成五个阶段：

1. 盘点、分类、坏链、未索引清单，明确问题。
2. 重建 `docs/README.md`，让导航成型。
3. 更新根 `README.md` 和根级维护约束，让入口和规则成型。
4. 合并或归档重复文档，降低噪音。
5. 跑验证并留下整理报告，让结果可交付。

## 盘点结果

整理前，文档库主要由用户指南、方法说明、实验说明、结构说明、审计报告、优化记录和提案组成。

| 范围 | Markdown 数量 | 说明 |
| --- | ---: | --- |
| `docs/` | 104 | 主文档库 |
| `_archive/docs/` | 1 | 历史文档归档区，整理前几乎为空 |
| `bench/`、`custom_nodes/`、`examples/` | 22 | 配套说明，不进入主文档入口全文索引 |
| 合计 | 127 | 仓库内可见 Markdown 文档 |

## 分类结果

本轮按读者路径重新确认了文档分类。

| 分类 | 目录 | 状态 | 处理方式 |
| --- | --- | --- | --- |
| 总入口 | `docs/README.md` | 需扩展 | 重建为总导航，只放入口和规则 |
| 用户指南 | `docs/guidelines/` | 保留 | 继续作为安装、训练、推理主路径 |
| 方法说明 | `docs/methods/` | 保留 | 稳定或已接入能力放这里 |
| 实验能力 | `docs/experimental/` | 保留 | 可运行但未稳定的能力放这里 |
| 架构原理 | `docs/structure/` | 保留 | 原理、数学和实现结构放这里 |
| 配置说明 | `docs/configuration/` | 需补索引 | 新增分区索引 |
| 功能说明 | `docs/features/` | 需补索引 | 新增分区索引 |
| 审计报告 | `docs/findings/` | 需补索引 | 新增分区索引，避免总入口过长 |
| 优化记录 | `docs/optimizations/` 与根级优化文档 | 保留 | 从总入口统一挂载 |
| 活跃提案 | `docs/proposal/` | 需降噪 | 新增分区索引，只保留活跃或半活跃提案 |
| 历史提案 | `_archive/docs/proposal/` | 需填充 | 完成或过期计划移动到这里 |

## 坏链结果

整理前使用严格 Markdown 链接扫描，排除代码块和行内代码后得到：

| 项目 | 结果 |
| --- | ---: |
| 扫描 Markdown | 106 |
| 本地链接 | 132 |
| 外部链接 | 37 |
| 真实坏链 | 0 |

结论：文档链接基础健康，本轮重点不是修坏链，而是补入口、补索引和归档历史噪音。

## 未索引清单

整理前 `docs/README.md` 直接索引约 73 个 Markdown，约 30 个文档没有直接挂到总入口。

主要未索引来源：

| 目录 | 数量 | 处理 |
| --- | ---: | --- |
| `docs/findings/` | 11 | 新增 `docs/findings/README.md` |
| `docs/findings/agent_audit_20260622/` | 6 | 通过 findings 分区索引挂载 |
| `docs/proposal/` | 8 | 新增 `docs/proposal/README.md`，并归档历史项 |
| `docs/configuration/` | 2 | 新增 `docs/configuration/README.md` |
| `docs/features/` | 1 | 新增 `docs/features/README.md` |
| `docs/` 根级零散文档 | 2 | 从总入口的仓库级说明区挂载 |

## 合并和归档决策

本轮不删除历史资料，只把明显完成或只服务于旧合并工作的材料从活跃提案区移到归档区。

| 文档 | 原位置 | 新位置 | 原因 |
| --- | --- | --- | --- |
| `upstream_high_value_merge_roadmap_2026-06-24.md` | `docs/proposal/` | `_archive/docs/proposal/` | 上游合并路线图已执行并形成完成报告 |
| `upstream_merge_completion_report_2026-06-24.md` | `docs/proposal/` | `_archive/docs/proposal/` | 历史完成报告，不应占用活跃提案区 |
| `upstream_merge_completion_report_2026-06-24_audit.md` | `docs/proposal/` | `_archive/docs/proposal/` | 历史审核报告，与完成报告一起归档 |
| `upstream_merge_completion_report_fixes_summary.md` | `docs/proposal/` | `_archive/docs/proposal/` | 历史修复摘要，与完成报告一起归档 |
| `upstream_preprocess_robustness_analysis.md` | `docs/proposal/` | `_archive/docs/proposal/` | 服务于旧上游合并，当前实现事实看源码和测试 |
| `configs_external_data_root_plan_2026-06-24.md` | `docs/proposal/` | `_archive/docs/proposal/` | 配置外置已落地，当前说明由 `docs/configuration/` 承接 |
| `compile_safety_patches_analysis.md` | `docs/proposal/` | `_archive/docs/proposal/` | 服务于旧上游合并，当前实现事实看 runtime 文档和测试 |

保留在 `docs/proposal/` 的文档仍视为活跃或半活跃：

- `adapter-aware-checkpoint.md`
- `postfix_residual_for_directedit.md`
- `postfix_residual_per_image_inversion.md`
- `prior_preservation_from_synth_pool.md`
- `soft_tokens_agsm.md`
- `soft_tokens_contrastive.md`
- `soft_tokens_softrank.md`
- `turbo_anima_dmd_lora.md`

## 新导航策略

总入口只负责告诉读者“先看哪里”，分区索引负责列全量文档。

```text
README.md
  -> docs/README.md
      -> guidelines/
      -> methods/
      -> experimental/
      -> structure/
      -> configuration/README.md
      -> features/README.md
      -> findings/README.md
      -> proposal/README.md
      -> archive-index.md
```

## 验收标准

整理完成需要满足：

- 根 `README.md` 明确指向 `docs/README.md`。
- `docs/README.md` 能把读者导向所有主要文档分区。
- `docs/findings/`、`docs/proposal/`、`docs/configuration/`、`docs/features/` 有分区索引。
- 明显历史性 proposal 已归档。
- 真实坏链为 0。
- `git diff --check` 通过。
- 文档维护约束写入根级 `AGENTS.md`。

## 后续维护建议

- 新增文档时，必须同时更新总入口或分区索引。
- 完成的计划和上游合并报告默认进入 `_archive/docs/`。
- 方法状态变化时，同步更新 `docs/README.md`、对应方法文档和测试建议。
- 多语言指南按中文主指南为源，翻译滞后时要标注，不要让翻译文档成为唯一事实来源。

## 最终验证

整理后执行了文档验证。

| 检查 | 结果 |
| --- | --- |
| `docs/` Markdown | 103 |
| `_archive/docs/` Markdown | 8 |
| 本地链接扫描 | `scanned=113 local_links=193 external_links=37 broken=0` |
| `docs/README.md` 可达性 | `docs_md=103 reachable_from_docs_readme=103 missing=0` |
| 空白检查 | `git diff --check -- README.md AGENTS.md docs _archive/docs` 通过 |
| 命令入口加载 | `timeout 60 .venv/bin/python tasks.py --help` 通过 |

结论：五个阶段均已落地，当前主文档库没有真实坏链，且所有 `docs/` 下 Markdown 都能从总入口触达。
