# 文档归档索引

这里说明哪些文档已经从主文档路径退下去，以及为什么退下去。

## 归档原则

- 归档不是删除，历史上下文仍保留。
- 归档文档默认不是当前实现说明。
- 当前实现以 `docs/README.md` 指向的 `guidelines/`、`methods/`、`experimental/`、`structure/`、`configuration/`、`features/`、`findings/` 和 `optimizations/` 为准。

## Proposal 归档

历史提案和一次性合并材料放在 [_archive/docs/proposal/](../_archive/docs/proposal/)。

| 文档 | 原因 |
| --- | --- |
| [anima-app-deglobalization.md](../_archive/docs/proposal/anima-app-deglobalization.md) | 阶段 0-3 已落地（85 轮，`globalThis` 从 1075 降到 270），剩余收尾由 `anima-app-legacy-bridge-cleanup.md` 接续 |
| [anima-app-legacy-bridge-cleanup.md](../_archive/docs/proposal/anima-app-legacy-bridge-cleanup.md) | `anima-app` legacy bridge 收尾已完成，仅保留实施上下文 |
| [lora-network-decomposition.md](../_archive/docs/proposal/lora-network-decomposition.md) | `LoRANetwork` 分层拆分核心代码已落地，当前仅保留实施计划和历史上下文 |
| [upstream_high_value_merge_roadmap_2026-06-24.md](../_archive/docs/proposal/upstream_high_value_merge_roadmap_2026-06-24.md) | 上游合并路线图已执行，当前仅保留历史上下文 |
| [upstream_merge_completion_report_2026-06-24.md](../_archive/docs/proposal/upstream_merge_completion_report_2026-06-24.md) | 上游合并完成报告，已经不是活跃提案 |
| [upstream_merge_completion_report_2026-06-24_audit.md](../_archive/docs/proposal/upstream_merge_completion_report_2026-06-24_audit.md) | 上游合并完成报告审核记录 |
| [upstream_merge_completion_report_fixes_summary.md](../_archive/docs/proposal/upstream_merge_completion_report_fixes_summary.md) | 上游合并报告修复摘要 |
| [upstream_preprocess_robustness_analysis.md](../_archive/docs/proposal/upstream_preprocess_robustness_analysis.md) | 上游预处理健壮性分析，服务旧合并工作 |
| [configs_external_data_root_plan_2026-06-24.md](../_archive/docs/proposal/configs_external_data_root_plan_2026-06-24.md) | 配置外置已落地，当前说明由 `docs/configuration/` 承接 |
| [compile_safety_patches_analysis.md](../_archive/docs/proposal/compile_safety_patches_analysis.md) | Compile safety 分析已转为实现和测试事实 |
| [anima-app-runtime-migration.md](../_archive/docs/proposal/anima-app-runtime-migration.md) | WebUI `anima-app` runtime 迁移阶段计划已执行，当前仅保留历史上下文 |

## Findings 归档

历史实验结论与当前使用说明分开保存：

| 文档 | 说明 |
| --- | --- |
| [findings/README.md](../_archive/docs/findings/README.md) | 历史 findings 归档索引 |
| [dcw-bias-findings-202605.md](../_archive/docs/findings/dcw-bias-findings-202605.md) | DCW CFG=1 和早期 band-mask 实验结论 |
| [smc-cfg-analysis-and-proposal-202605.md](../_archive/docs/findings/smc-cfg-analysis-and-proposal-202605.md) | SMC-CFG α-adaptive 控制器的历史分析 |
| [vr-loss-headroom/README.md](../_archive/docs/findings/vr-loss-headroom/README.md) | Variance-reduction loss headroom 历史实验结论 |
| [vr-loss-headroom/proposal.md](../_archive/docs/findings/vr-loss-headroom/proposal.md) | Variance-reduction loss 历史接入提案 |

## Configuration 归档

| 文档 | 说明 |
| --- | --- |
| [configuration/README.md](../_archive/docs/configuration/README.md) | 历史配置实施报告索引 |
| [implementation-report.md](../_archive/docs/configuration/implementation-report.md) | 2026-06-24 配置根外置的本机实施快照，已脱敏 |
