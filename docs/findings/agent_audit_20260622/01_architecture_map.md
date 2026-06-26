# R1 — 项目真实架构地图

## 1. 五条端到端链路

### 1.1 训练默认路径
- **触发:** `python tasks.py lora`（`tasks.py:44-48`）→ `scripts/tasks/training.py:18` `cmd_lora` → `scripts/tasks/_common.py:646` `train()`
- **启动:** `build_launch_cmd`（`_common.py:555`）→ `train.py`
- **核心:** `train.py:263` `AnimaTrainer`, `train()` @1904；`library/training/bootstrap.py`；checkpoint → `output/ckpt/` / Web `output/runs/`
- **GUI:** `lora-gui` → `configs/gui-methods/{variant}.toml`（`training.py:22-47`）
- **队列:** `--queue` → daemon（`_common.py` `_queue_submit`）

### 1.2 配置合并
`configs/base.toml` → `presets.toml` → `methods|gui-methods` → CLI → `library/config/io.py:355` `load_method_preset` / `:569` `apply_config_to_args`
**例外:** `turbo.toml`、`spd.toml` 由 distill 脚本读，非 train merge。

### 1.3 预处理与缓存
`tasks.py preprocess*` → `scripts/tasks/preprocess.py` + `library/preprocess/`
Sidecar: `{stem}_{WxH}_anima.npz`, `{stem}_anima_te.safetensors`, PE `{stem}_anima_pe.safetensors` under `post_image_dataset/`.

### 1.4 推理
`inference.py` + `library/inference/generation.py`；`scripts/tasks/inference.py` 包装 test/test-dcw/test-merge 等。

### 1.5 WebUI 队列
`web/static/app.js` → `js/features/anima-app/` → `web/routes/training.py` → `web/services/training/*` → 子进程 `train.py` + `configs/web-training-queue/`.

## 2. 模块归属（节选）
| 目录 | 入口 | 误改后果 |
|------|------|----------|
| library/training | train.py | 训练失败/OOM |
| library/config/io.py | 全链路配置 | 静默错参 |
| library/runtime/harness.py | compile 顺序 | 无效 compile |
| networks/ | resolve_network_spec | 加载失败 |
| web/services/training/launcher.py | Web 启动 | 路径/CLI 错误 |

## 3. 文档 vs 源码（12 条）
1. docs 引用根 CLAUDE.md — 已删，AGENTS.md 为准 🔴维护
2. lora.toml 提 fera gui 变体 — 目录无 fera.toml 🟡
3. legacy-app 为主债务 — anima-app 已替代 🟢表述旧
4. print-config turbo — turbo 非 train schema 🟡
5. Web 历史 config/flat — 仅 collections 🟡
6. DCW bucket 顺序 — buckets.py:64-70 与 checkpoint 绑定 🔴
7. 系统 python pytest — 需 .venv 🟡
8–12. 见 AGENTS 与 harness/FEI 文档与源码一致项

## 立即可做 / 需改代码 / 不做什么
- 立即可做: tasks.py --help; rg load_method_preset library/config/io.py
- 需改代码: 新 CLI 走 cli_args + schema + Web catalog
- 不做: 手写绕过合并链的配置注入
