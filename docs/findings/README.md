# Findings 索引

状态：索引
适用版本：当前文档树

这里放审计、实验结论、失败路径、性能记录和阶段性整理报告。

## 维护和整理

| 文档 | 说明 |
| --- | --- |
| [documentation_consolidation_20260706.md](documentation_consolidation_20260706.md) | 2026-07-06 文档库合并整理报告 |
| [project_cleanup_checkpoint_20260705.md](project_cleanup_checkpoint_20260705.md) | 项目清理检查点 |
| [project_cleanup_long_running_goal_20260705.md](project_cleanup_long_running_goal_20260705.md) | 跨系统长期清理目标 |
| [project_cleanup_next_stage_goal_20260705.md](project_cleanup_next_stage_goal_20260705.md) | 下一阶段清理目标 |
| [project_cleanup_sustained_goal_20260705.md](project_cleanup_sustained_goal_20260705.md) | 持续清理目标记录 |
| [project_cleanup_sustained_goal_20260706.md](project_cleanup_sustained_goal_20260706.md) | 持续清理目标最新记录 |

## WebUI 和配置

| 文档 | 说明 |
| --- | --- |
| [webui_frontend_p0_p2_fix_20260726.md](webui_frontend_p0_p2_fix_20260726.md) | 2026-07-26 WebUI 前端 P0–P2 审核修复记录（no-undef / 搜索 debounce / WS / dashboard） |
| [webui_dataset_cross_group_drag_fix_20260727.md](webui_dataset_cross_group_drag_fix_20260727.md) | 2026-07-27 数据集/文件分组跨组拖动静默失败修复 |
| [webui_frontend_visual_audit_20260530.md](webui_frontend_visual_audit_20260530.md) | WebUI 视觉和交互审计 |
| [webui_god_files_refactor_20260607.md](webui_god_files_refactor_20260607.md) | WebUI 上帝文件治理合并记录 |
| [training_history_detail_performance.md](training_history_detail_performance.md) | 训练历史详情性能记录 |
| [ui_scale_independent_settings.md](ui_scale_independent_settings.md) | UI 缩放独立设置结论 |

## Runtime 和能力边界

| 文档 | 说明 |
| --- | --- |
| [runtime_support_matrix_20260704.md](runtime_support_matrix_20260704.md) | compile / checkpoint / block swap 组合矩阵审计 |
| [adapter_registry_capabilities_audit_20260704.md](adapter_registry_capabilities_audit_20260704.md) | Adapter registry、merge、推理加载和续训能力边界审计 |
| [anima_int8_base_linear_audit.md](anima_int8_base_linear_audit.md) | Anima int8 base linear 审计（存储/传输 int8；非 ConvRot） |
| [../experimental/convrot_int8_training.md](../experimental/convrot_int8_training.md) | ConvRot int8 训练探索（W8A16/W8A8）；与上条区分 |

## 性能、显存和训练报告

| 文档 | 说明 |
| --- | --- |
| [loha_hot_test_20260725.md](loha_hot_test_20260725.md) | LoHa 在 RTX 3080 10GB 上的 12-step 热测与检查点验证 |
| [training_profiling_hot_test_20260629.md](training_profiling_hot_test_20260629.md) | 训练 profiling 热测记录 |
| [mfu_plain_lora_vs_lokr_blockswap_20260629_022138.md](mfu_plain_lora_vs_lokr_blockswap_20260629_022138.md) | Plain LoRA 与 LoKr block swap MFU 对比 |
| [anima_lokr_blockswap_oom_report.md](anima_lokr_blockswap_oom_report.md) | LoKr 16G block swap OOM 报告 |
| [anima_lokr_16g_next_goal.md](anima_lokr_16g_next_goal.md) | LoKr 16G 下一步目标 |
| [lokr_anima_shaojianV1_run_report.md](lokr_anima_shaojianV1_run_report.md) | lokr-anima-shaojianV1 运行报告 |
| [anima_balanced_16g_blockswap_ablation_plan.md](anima_balanced_16g_blockswap_ablation_plan.md) | Balanced 16G block swap 消融 |
| [anima_fp8_blockswap_transfer_ablation_plan.md](anima_fp8_blockswap_transfer_ablation_plan.md) | FP8 block swap transfer 消融计划 |
| [anima_fp8_blockswap_transfer_report.md](anima_fp8_blockswap_transfer_report.md) | FP8 block swap transfer 最终报告 |

## 方法和研究结论

| 文档 | 说明 |
| --- | --- |
| [selfflow.md](selfflow.md) | Self-Flow rep-loss 在冻结 Anima backbone 上的否定结果 |
| [mod_guidance_quality_tag_axis.md](mod_guidance_quality_tag_axis.md) | Mod-guidance quality tag 轴分析 |
| [channel_stats_content_independence.md](channel_stats_content_independence.md) | channel stats 与 content independence 分析 |
| [asymflow_parameterization.md](asymflow_parameterization.md) | Anima velocity / sigma 参数化记录 |
| [l2p_pixel_transfer.md](l2p_pixel_transfer.md) | L2P pixel transfer 调研 |
| [fasterdit_signal_densification_plan.md](fasterdit_signal_densification_plan.md) | FasterDiT signal densification 计划 |

## Agent Audit 2026-06-22

| 文档 | 说明 |
| --- | --- |
| [agent_audit_20260622/00_INDEX.md](agent_audit_20260622/00_INDEX.md) | Agent audit 总索引 |
| [agent_audit_20260622/01_architecture_map.md](agent_audit_20260622/01_architecture_map.md) | 架构地图 |
| [agent_audit_20260622/02_invariants_risk_audit.md](agent_audit_20260622/02_invariants_risk_audit.md) | 不变量和风险审计 |
| [agent_audit_20260622/03_test_coverage_map.md](agent_audit_20260622/03_test_coverage_map.md) | 测试覆盖地图 |
| [agent_audit_20260622/04_webui_maintenance_ux_audit.md](agent_audit_20260622/04_webui_maintenance_ux_audit.md) | WebUI 维护和 UX 审计 |
| [agent_audit_20260622/05_config_method_matrix.md](agent_audit_20260622/05_config_method_matrix.md) | 配置和方法矩阵 |

配套截图和图表在 [assets/](assets/)。
