# Agent Audit Index — 2026-06-22

**仓库:** `/home/scv/nvme0n1p1/训练器相关/anima_lora` | **快照:** 2026-06-22 | **只读审计**

## 报告
| ID | 文件 |
|----|------|
| R1 | [01_architecture_map.md](01_architecture_map.md) |
| R2 | [02_invariants_risk_audit.md](02_invariants_risk_audit.md) |
| R3 | [03_test_coverage_map.md](03_test_coverage_map.md) |
| R4 | [04_webui_maintenance_ux_audit.md](04_webui_maintenance_ux_audit.md) |
| R5 | [05_config_method_matrix.md](05_config_method_matrix.md) |

## Top 30 行动项
| # | P | 项 | 报告 |
|---|-----|-----|------|
| 1 | P0 | 配置以 `rg --files configs/gui-methods configs/methods` 为准 | R5 |
| 2 | P0 | DCW_ASPECT_BUCKETS 顺序不可乱 (`library/datasets/buckets.py:64-70`) | R2 |
| 3 | P0 | compile_blocks 在 apply+load 后 (`library/runtime/harness.py:9-11`) | R2 |
| 4 | P0 | TE max-padding，勿 mask padding (`library/anima/strategy.py:452`) | R2 |
| 5 | P0 | Web 删除/预览限 `resolve_output_root` | R2 |
| 6 | P0 | memory_probe/block_swap auto → 任务目录 (`launcher.py:431-450`) | R2 |
| 7 | P1 | `tasks.py lora` → `scripts/tasks/training.py:18` → `_common.py:646` → `train.py` | R1 |
| 8 | P1 | `lora-gui` 读 `configs/gui-methods/<variant>.toml` | R1/R5 |
| 9 | P1 | merge 拒绝 ReFT/Hydra/postfix (`scripts/merge_to_dit.py`) | R2 |
| 10 | P1 | 三轴 metadata，旧 ss_use_hydra 不加载 | R2 |
| 11 | P1 | set_fei 训练/推理对称 | R2 |
| 12 | P1 | Comfy 改 live 后 vendor-sync | R2 |
| 13 | P1 | daemon 端口写 pidfile，客户端重解析 | R2 |
| 14 | P1 | 禁恢复 legacy-app.js | R4 |
| 15 | P1 | 改 ES import 同步 ?v= cache token | R4 |
| 16 | P1 | turbo.toml/spd.toml 非 train.py 合并链 | R5 |
| 17 | P2 | block_swap/peak_probe auto 缺集成测 | R3 |
| 18 | P2 | DirectEdit+DCW 推理组合缺口 | R3 |
| 19 | P2 | LoKr 16G 按钮与 balanced_16g 字段 | R4 |
| 20 | P2 | methods/lora.toml 注释块 vs gui 自包含 | R5 |
| 21 | P2 | 历史 UI 仅 collections | R4 |
| 22 | P2 | sample-prompts 分叉 | R4 |
| 23 | P2 | 预处理 sidecar 命名契约 | R1 |
| 24 | P2 | exp-* 在 tasks.py 易变 | R1 |
| 25 | P2 | pytest 用 .venv/bin/python | R3 |
| 26 | P2 | 根 CLAUDE.md 已删，AGENTS.md 自包含 | R1 |
| 27 | P2 | docs 仍写 CLAUDE.md | R1 |
| 28 | P2 | weight_analysis 限 output_root | R4 |
| 29 | P3 | bench 用 harness compile 顺序 | R2 |
| 30 | P3 | Tier2 需 bench+merge story | CONTRIBUTING |

## 建议 5 PR 切片
1. docs: CLAUDE→AGENTS 2. test: launcher auto jsonl 3. web: LoKr/preset 4. networks: metadata 5. findings 互链

## 立即可做 / 需改代码 / 不做什么
见 R1–R5 文末。不做长训练、vendor-sync、用户数据目录清理。
