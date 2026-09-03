# Findings 索引

状态：索引
适用版本：当前文档树

这里放审计、实验结论、失败路径、性能记录和阶段性整理报告。

## 维护和整理

| 文档 | 说明 |
| --- | --- |
| [test_suite_audit_20260831.md](test_suite_audit_20260831.md) | 2026-08-31 测试集规模、分层、重复、source-probe 与默认门禁审计 |
| [test_script_audit_20260809.md](test_script_audit_20260809.md) | 2026-08-09 测试脚本审计、精简结果与 probe 保护边界 |
| [documentation_consolidation_20260706.md](documentation_consolidation_20260706.md) | 2026-07-06 文档库合并整理报告 |
| [project_cleanup_checkpoint_20260705.md](project_cleanup_checkpoint_20260705.md) | 项目清理检查点 |
| [project_cleanup_long_running_goal_20260705.md](project_cleanup_long_running_goal_20260705.md) | 跨系统长期清理目标 |
| [project_cleanup_next_stage_goal_20260705.md](project_cleanup_next_stage_goal_20260705.md) | 下一阶段清理目标 |
| [project_cleanup_sustained_goal_20260705.md](project_cleanup_sustained_goal_20260705.md) | 持续清理目标记录 |
| [project_cleanup_sustained_goal_20260706.md](project_cleanup_sustained_goal_20260706.md) | 持续清理目标最新记录 |

## WebUI 和配置

| 文档 | 说明 |
| --- | --- |
| [captioning_uiux_external_review_20260829.md](captioning_uiux_external_review_20260829.md) | Captioning 13 界面外部视觉评审 API 探针、失败边界与后续裁决要求 |
| [captioning_uiux_iteration_20260829.md](captioning_uiux_iteration_20260829.md) | Captioning/打标工作台 13 界面两轮 UI/UX 优化与验收记录 |
| [dragon_frontend_performance_20260826.md](dragon_frontend_performance_20260826.md) | Dragon 五个核心路由的按需加载与运行时优化历史基线（不含后续 tagging 工作台） |
| [dragon_frontend_uiux_review_20260829.md](dragon_frontend_uiux_review_20260829.md) | Dragon 13 页面双视口截图评审、外部视觉模型意见与实施顺序 |
| [dragon_responsive_hardcoding_audit_20260824.md](dragon_responsive_hardcoding_audit_20260824.md) | Dragon 视口硬编码解耦、浏览器降级和响应式验证记录 |
| [dragon_classic_feature_parity_audit_20260815.md](dragon_classic_feature_parity_audit_20260815.md) | Classic、现有 Dragon 与 React 新前端的逐功能对照审计及迁移证据 |
| [dragon_dataset_subset_field_audit_20260815.md](dragon_dataset_subset_field_audit_20260815.md) | Dragon 数据集 subset 字段与后端契约对照审计 |
| [dragon_next_audit_20260819.md](dragon_next_audit_20260819.md) | Dragon React 前端对 live backend 的阶段性审计 |
| [dragon_next_stage1_queue_monitor_history_20260819.md](dragon_next_stage1_queue_monitor_history_20260819.md) | Dragon React 队列、监控和历史页面阶段 1 集成记录 |
| [dragon_ui_pr1_integration_20260814.md](dragon_ui_pr1_integration_20260814.md) | Dragon UI PR #1 的分阶段集成、测试门禁、调试与发布记录 |
| [webui_backend_audit_20260815.md](webui_backend_audit_20260815.md) | WebUI 后端安全、并发、状态机和测试覆盖审计 |
| [webui_frontend_p0_p2_fix_20260726.md](webui_frontend_p0_p2_fix_20260726.md) | 2026-07-26 WebUI 前端 P0–P2 审核修复记录（no-undef / 搜索 debounce / WS / dashboard） |
| [webui_dataset_cross_group_drag_fix_20260727.md](webui_dataset_cross_group_drag_fix_20260727.md) | 2026-07-27 数据集/文件分组跨组拖动静默失败修复 |
| [webui_frontend_visual_audit_20260530.md](webui_frontend_visual_audit_20260530.md) | WebUI 视觉和交互审计 |
| [webui_god_files_refactor_20260607.md](webui_god_files_refactor_20260607.md) | WebUI 上帝文件治理合并记录 |
| [training_history_detail_performance.md](training_history_detail_performance.md) | 训练历史详情性能记录 |
| [ui_scale_independent_settings.md](ui_scale_independent_settings.md) | UI 缩放独立设置结论 |
| [uncommitted_remote_review_20260810.md](uncommitted_remote_review_20260810.md) | 2026-08-10 未提交改动与远端同步审计 |

## Runtime 和能力边界

| 文档 | 说明 |
| --- | --- |
| [backend_multi_model_audit_20260810.md](backend_multi_model_audit_20260810.md) | 2026-08-10 后端多模型兼容审计：Anima/Krea-2 主链、P1 风险、测试缺口与 registry 路线 |
| [runtime_support_matrix_20260704.md](runtime_support_matrix_20260704.md) | compile / checkpoint / block swap 组合矩阵审计 |
| [v100_flash_attention_support.md](v100_flash_attention_support.md) | V100 FlashAttention 目标仓库对照、移植状态与生产边界 |
| [adapter_registry_capabilities_audit_20260704.md](adapter_registry_capabilities_audit_20260704.md) | Adapter registry、merge、推理加载和续训能力边界审计 |
| [anima_int8_base_linear_audit.md](anima_int8_base_linear_audit.md) | Anima int8 base linear 审计（存储/传输 int8；非 ConvRot） |
| [../experimental/convrot_int8_training.md](../experimental/convrot_int8_training.md) | ConvRot int8 训练探索（W8A16/W8A8）；与上条区分 |

## 性能、显存和训练报告

| 文档 | 说明 |
| --- | --- |
| [anima_dual_gpu_parallel_probe_20260904.md](anima_dual_gpu_parallel_probe_20260904.md) | Anima 异构双卡 PP2、TP2、TP2 INT8 通信的 BS=1 性能、显存、数值与同参数图片对比 |
| [lycoris_4_fused_kernel_audit_20260902.md](lycoris_4_fused_kernel_audit_20260902.md) | LyCORIS 4.0.0 Triton/TileLang fused kernel 发布、性能、精度与 release 宣传审计，以及对本项目 LoKr/LoHa 的借鉴优先级 |
| [lokr_fused_backward_stage1_20260902.md](lokr_fused_backward_stage1_20260902.md) | 基于 LyCORIS 审计方向独立实现 LoKr `grad_w1` Triton reduction；RTX 3080 组件验证与 CMP 170HX 3-seed x 50-step 端到端热测、Nsight、resume/compile/swap gate |
| [convrot_longrun_bf16_w8a8_w8a16_20260727.md](convrot_longrun_bf16_w8a8_w8a16_20260727.md) | RTX 3080 上 BF16/W8A8/W8A16 三组 1710-step 长训审计（速度、显存、loss、样图及最终保存回归修复） |
| [loha_hot_test_20260725.md](loha_hot_test_20260725.md) | LoHa 在 RTX 3080 10GB 上的 12-step 热测与检查点验证 |
| [training_profiling_hot_test_20260629.md](training_profiling_hot_test_20260629.md) | 训练 profiling 热测记录 |
| [anima_lokr_blockswap_oom_report.md](anima_lokr_blockswap_oom_report.md) | LoKr 16G block swap OOM 报告 |
| [anima_lokr_16g_next_goal.md](anima_lokr_16g_next_goal.md) | LoKr 16G 下一步目标 |
| [lokr_anima_shaojianV1_run_report.md](lokr_anima_shaojianV1_run_report.md) | lokr-anima-shaojianV1 运行报告 |
| [anima_balanced_16g_blockswap_ablation_plan.md](anima_balanced_16g_blockswap_ablation_plan.md) | Balanced 16G block swap 消融 |
| [anima_fp8_blockswap_transfer_ablation_plan.md](anima_fp8_blockswap_transfer_ablation_plan.md) | FP8 block swap transfer 消融计划 |
| [anima_fp8_blockswap_transfer_report.md](anima_fp8_blockswap_transfer_report.md) | FP8 block swap transfer 最终报告 |
| [blockswap_baseline_20260806.md](blockswap_baseline_20260806.md) | 块交换优化基线测量（计算 vs 传输，RTX 3080 / CMP 90HX，标准参考） |
| [krea2_90hx_nsight_tiles.md](krea2_90hx_nsight_tiles.md) | Krea-2 CMP 90HX Nsight tile/cache 验证：尾 tile 主因 REJECT、CUTLASS/cuDNN/FA2 实际 tile、NCU 权限边界 |
| [krea2_nf4_ablation_findings.md](krea2_nf4_ablation_findings.md) | Krea-2 NF4 × {完整检查点, 块交换} 消融矩阵（5 格六维指标：显存/内存/速度/loss/数学实现/数学偏移，PG199 1024×1024 30 步） |
| [krea2_nf4_self_contained.md](krea2_nf4_self_contained.md) | Krea-2 自包含 NF4 v2：无重新量化构建、版本化严格加载、v1 兼容与 PG199/RTX 3080 训练验证 |
| [krea2_nf4_correction_pg199.md](krea2_nf4_correction_pg199.md) | Krea-2 NF4 激活加权 rank-16 回补实测：层级误差下降但端到端收益不足，暂不生产化 |
| [krea2_nf4_downstream_correction_90hx.md](krea2_nf4_downstream_correction_90hx.md) | Krea-2 NF4 下游 velocity 目标低秩回补的 90HX 校准与 held-out 验证 |
| [krea2_nf4_correction_rollout_3080.md](krea2_nf4_correction_rollout_3080.md) | Krea-2 NF4 rank-16 回补的 RTX 3080 完整 rollout 否定结果 |
| [krea2_nf4_h2d_bottleneck_findings.md](krea2_nf4_h2d_bottleneck_findings.md) | Krea-2 方向 B 前置诊断：NF4+swap H2D 搬运占比双口径实测 0% + 理论上界 2% → NOT_WORTH 归档（含口径缺陷诚实记录 + 与消融矩阵矛盾核验） |
| [krea2_3080_speed_stage1.md](krea2_3080_speed_stage1.md) | Krea-2 RTX 3080 12s/it 阶段 1：同机双卡 Linear/attention 消融、满功耗核验、prepare 口径修正、padding 尾裁剪 NOT_WORTH |
| [krea2_3080_speed_stage2.md](krea2_3080_speed_stage2.md) | Krea-2 速度阶段 2：PG199 every-other checkpoint 快 13.9%；RTX 3080 放开单 block 仍 OOM，NOT_FEASIBLE |
| [krea2_3080_speed_stage3.md](krea2_3080_speed_stage3.md) | Krea-2 速度阶段 3：per-block compile 在 PG199 快 19.1%；RTX 3080 的 3.3% 仅为冷态短窗口，后由阶段 5 修正 |
| [krea2_3080_speed_stage4.md](krea2_3080_speed_stage4.md) | Krea-2 速度阶段 4：PG199 compile+16/28 checkpoint 20 步稳态 2.408s/it（-28.5%），但 31.55GB 仅实验用 |
| [krea2_3080_speed_stage5.md](krea2_3080_speed_stage5.md) | Krea-2 速度阶段 5：RTX 3080 compile 20 步 12.06→12.65s 热漂移；纯 GEMM 84°C 复现，compile 主价值修正为显存余量 |
| [krea2_3080_speed_stage6.md](krea2_3080_speed_stage6.md) | Krea-2 速度阶段 6：FP16 NF4 backward 虽快，但输入梯度 rel-L2 35.6%，REJECT，保留 BF16 强制契约 |
| [krea2_3080_speed_stage7.md](krea2_3080_speed_stage7.md) | Krea-2 速度阶段 7：reduce-overhead CUDA Graph 与 checkpoint recompute 冲突，限制为 default Inductor mode |
| [krea2_3080_speed_stage8.md](krea2_3080_speed_stage8.md) | Krea-2 速度阶段 8：rank16→8 可训参数减半但步时持平、仅省 145MB，不作为速度建议 |
| [krea2_3080_speed_stage9.md](krea2_3080_speed_stage9.md) | Krea-2 速度阶段 9：24 buckets 折叠为 4608/4864 两张可复用 compile 图，默认开启 fixed resident compile |
| [krea2_3080_speed_stage10.md](krea2_3080_speed_stage10.md) | Krea-2 速度阶段 10：compile 与 LoRA+optimizer checkpoint round-trip delta=0，reload 后 2.73s 无重编译续训 |
| [krea2_3080_speed_stage11.md](krea2_3080_speed_stage11.md) | Krea-2 速度阶段 11：compile 融合 mul/copy/add 并减少 dequant；GEMM+attention 不变且占 compiled 约 89% |
| [krea2_3080_speed_stage12.md](krea2_3080_speed_stage12.md) | Krea-2 速度阶段 12：packed varlen FlashAttention 在 PG199 快 11-13%，3080 长稳态快 4%；现为显式 opt-in 生产后端 |
| [krea2_3080_speed_final.md](krea2_3080_speed_final.md) | Krea-2 RTX 3080 速度研究最终审计：根因、生产建议、可选后端、否决路径与证据边界总表 |
| [krea2_3080_speed_comparison_extended.md](krea2_3080_speed_comparison_extended.md) | Krea-2 PG199/RTX 3080 扩展速度矩阵：step、it/min、显存、冷/热稳态、swap、checkpoint、compile 与 Flash 对比 |
| [anima_perband_dynamic_seq_20260830.md](anima_perband_dynamic_seq_20260830.md) | Anima per-band dynamic-seq 移植、64GB CMP 170HX 60-step union/per-band A/B 与默认关闭结论 |

## 方法和研究结论

| 文档 | 说明 |
| --- | --- |
| [selfflow.md](selfflow.md) | Self-Flow rep-loss 在冻结 Anima backbone 上的否定结果 |
| [mod_guidance_quality_tag_axis.md](mod_guidance_quality_tag_axis.md) | Mod-guidance quality tag 轴分析 |
| [channel_stats_content_independence.md](channel_stats_content_independence.md) | channel stats 与 content independence 分析 |
| [asymflow_parameterization.md](asymflow_parameterization.md) | Anima velocity / sigma 参数化记录 |
| [l2p_pixel_transfer.md](l2p_pixel_transfer.md) | L2P pixel transfer 调研 |
| [fasterdit_signal_densification_plan.md](fasterdit_signal_densification_plan.md) | FasterDiT signal densification 计划 |
| [krea2_raw_migration_stage0_findings.md](krea2_raw_migration_stage0_findings.md) | Krea-2-Raw 迁移阶段 0：R1/R2/R4/R8 定论 + VAE 互逆基准 + DiT key 清单 |
| [krea2_raw_migration_stage1_findings.md](krea2_raw_migration_stage1_findings.md) | Krea-2-Raw 迁移阶段 1：Qwen3-VL 文本链路 + 12 层 MFA + R1 padding 契约 (mask 屏蔽非 zero-sink) |
| [krea2_raw_migration_stage2_findings.md](krea2_raw_migration_stage2_findings.md) | Krea-2-Raw 迁移阶段 2：DiT 本体移植 + 加载器 + 单 latent forward 基准 |
| [krea2_raw_migration_stage3_findings.md](krea2_raw_migration_stage3_findings.md) | Krea-2-Raw 迁移阶段 3：LoRA 注入点 spec + family-aware target + attach+forward 真火测试 |
| [krea2_raw_migration_stage4_findings.md](krea2_raw_migration_stage4_findings.md) | Krea-2-Raw 迁移阶段 4：训练串通 + forward_for_loss 承重接口 + 单 prompt 过拟合 loss 下降 |
| [krea2_raw_migration_stage5_findings.md](krea2_raw_migration_stage5_findings.md) | Krea-2-Raw 迁移阶段 5：推理串通 + flow-matching Euler ODE + mu shift + CFG 采样 + VAE decode 出图 |
| [krea2_raw_migration_stage6_findings.md](krea2_raw_migration_stage6_findings.md) | Krea-2-Raw 迁移阶段 6：块交换 (ModelOffloader 复用) + 检查点 save/load round-trip + 256×256 净负发现 |

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
