# R5 — 配置/方法能力矩阵（2026-06-22 快照）

状态：历史审计快照
适用版本：2026-06-22 审计时点；不作为当前 main 操作说明

## Preset 速查（configs/presets.toml）
| preset | blocks_to_swap | gradient_checkpointing | torch_compile | 备注 |
|--------|----------------|------------------------|---------------|------|
| default | 0 | — | — | |
| low_vram | — | true | — | unsloth_offload_checkpointing true |
| low_vram_blockswap | 8 | true | false | |
| balanced_16g | 12 | false | true | block_swap_profile_jsonl=auto |
| graft | 20 | — | — | |
| half/quarter/tenth | — | — | — | sample_ratio 0.5/0.25/0.1 |
| debug | — | — | — | sample_ratio 0.001 |

## configs/methods（9）
| 文件 | 训练命令 | network_module | merge | 推理/实验 | 备注 |
|------|----------|----------------|-------|-----------|------|
| lora.toml | tasks.py lora | 默认 lora_anima | 部分可 merge | test | 注释块切换三轴 |
| chimera.toml | exp-chimera | chimera 模块 | 拒绝 moe | exp | |
| byg.toml | exp-byg | plain lora + BYG adapter | 部分 | exp | 需 byg tuples |
| easycontrol.toml | exp-easycontrol | networks.methods.easycontrol | 拒绝 | exp-test-easycontrol | |
| ip_adapter.toml | exp-ip-adapter | networks.methods.ip_adapter | 拒绝 | exp-test-ip | PE cache |
| soft_tokens.toml | exp-soft-tokens | networks.methods.soft_tokens | 拒绝 | exp-test-soft | |
| colorize.toml | exp-easycontrol EASYADAPTER=colorize | easycontrol | 拒绝 | exp | dataset colorize |
| spd.toml | exp-spd / distill | N/A train.py | N/A | exp-test-spd | 非 train merge |
| turbo.toml | exp-turbo / distill | N/A train.py | N/A | exp-test-turbo | 非 train merge |

## configs/gui-methods（18，目录全量）
| 文件 | 典型命令 | 关键标志 | Web 暴露 | merge |
|------|----------|----------|----------|-------|
| chimera_hydra.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| easycontrol.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| glora.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| hydralora-8gb.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| hydralora.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| ip_adapter.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| loha.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| lokr.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| lora-8gb.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| lora.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| lora_signal_probe.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| ortholora.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| reft.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| soft_tokens.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| tlora-8gb.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| tlora.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| tlora_ortho_reft.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |
| vera.toml | lora-gui / Web methods_subdir=gui-methods | 见文件内 use_* / network_module | 是 | 依变体（Hydra/ReFT 等见 merge_to_dit） |

**network_module 显式:** easycontrol, ip_adapter, soft_tokens（其余默认 LoRA family 插件标志 use_loha/use_lokr/use_vera/add_reft/use_moe 等）。

## 立即可做 / 需改代码 / 不做什么
- 立即可做: `ls configs/gui-methods`; `python tasks.py print-config METHOD=lora PRESET=balanced_16g`（venv）
- 需改代码: 新 gui-methods 变体须自包含键，不用注释切换块
- 不做: 在文档硬编码变体列表而不写「以目录为准」
