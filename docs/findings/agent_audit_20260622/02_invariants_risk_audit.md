# R2 — 不变量与高风险区审计

状态：历史审计快照
适用版本：2026-06-22 审计时点；不作为当前 main 操作说明

## 15 项（摘要表）
| # | 不变量 | 关键路径 | 测试 | 等级 |
|---|--------|----------|------|------|
| 1 | TE max-padding | library/anima/strategy.py:452 | test_ensure_text_strategies | 🔴 |
| 2 | buckets+compile顺序 | buckets.py:27; harness.py:9-11 | constant_token_buckets, native_flatten, runtime_harness_cli | 🔴 |
| 3 | lazy TE→VAE→DiT | training/bootstrap.py | test_training_bootstrap | 🟡 |
| 4 | 5D latent dim2=T | anima/models.py | 分散 | 🔴 |
| 5 | 三轴 metadata | lora_anima/config.py:605+ | test_network_cfg, factory_metadata | 🔴 |
| 6 | set_fei | router_conditioning.py; inference/adapters.py | global_router, router_compute | 🔴 |
| 7 | attn_fuse | networks/attn_fuse.py | per_channel_scaling_roundtrip | 🔴 |
| 8 | attention layout | attention_dispatch.py:45 | lora_custom_autograd | 🟡 |
| 9 | T-LoRA mask buffer | lora_anima/factory.py | lora_custom_autograd | 🟡 |
| 10 | merge 拒绝 | merge_to_dit.py:12-46 | 弱 | 🟡 |
| 11 | vendor-sync | tasks.py:230 | chimera_node_loader | 🟡 |
| 12 | output_root | settings_service.py:62-148 | preview_service, training_queue | 🔴 |
| 13 | jsonl auto | memory_probe.py:161; launcher.py:431 | 缺口 | 🟡 |
| 14 | DCW bucket 顺序 | buckets.py:64-70 | constant_token_buckets | 🔴 |
| 15 | daemon pidfile | daemon/__main__.py:66 | test_daemon | 🟢 |

## Top 15 易炸文件对
lora_save↔loading; attn_fuse↔anima weights; launcher↔cli_args; catalog↔presets; io.py↔config_service; generation↔training forward; _vendor↔library; gui-methods↔methods lora; merge↔plugins; buckets↔compile_blocks; index.html↔chunks; web-ui-settings↔settings_service; config.py↔lora.toml; harness↔train bootstrap; docs↔exp-*.

## 立即可做 / 需改代码 / 不做什么
- 立即可做: .venv pytest 上表 🔴 项
- 需改代码: 动 DCW/buckets 前版本化 fusion
- 不做: 恢复 ss_use_hydra fallback
