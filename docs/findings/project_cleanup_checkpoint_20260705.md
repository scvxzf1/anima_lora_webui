# 项目清理当前检查点与后续推进计划

一句话：这份文档替代旧的超长施工日志，只保留当前真实状态、剩余风险和下一步动作。

日期：2026-07-05
范围：`anima_lora` 主仓工程整理，不包含真实训练、模型下载、队列清理或用户数据清理。
来源：由上一版超长施工日志的最终状态表和阶段记录整理而来。
线上基线：`webui/main` 已推送到 `01bd55bc refactor: advance cleanup task checkpoint`。

---

## ✅ 1. 使用方式

一句话：后续 agent 先读这份文档，不再从旧的逐轮日志里翻状态。

- 先看 **第 2 节当前状态总表**，不要重复做已经阶段收口的任务。
- 再看 **第 3 节隔离和禁止事项**，确认哪些目录不能碰。
- 如果继续写代码，优先按 **第 5 节下一步建议** 小步推进。
- 不能把 `TASK-01` 到 `TASK-10` 统一说成“全部完成”。

---

## 📌 2. 当前状态总表

一句话：当前阶段已经完成一次大收口，但仍有几个任务只能算阶段完成。

| TASK | 当前状态 | 当前证据 | 剩余风险 / 下一步 |
|---|---|---|---|
| `TASK-01` CLI / 文档 / 配置事实对账 | 阶段收口 | LoRA 默认口径已修；`download-tagger`、Postfix / FeRA 等已按阶段记录复核；配置测试通过 | 未做全文档重扫；后续只在发现具体误导时小修 |
| `TASK-02` pytest 分层和 bench 超时 | 阶段收口 | MFU bench、GPU theoretical、dry-run / timeout / 输出根隔离已落地并有测试 | 未跑真实 MFU benchmark；不要默认启动真实训练 |
| `TASK-03` WebUI DOM 契约和安全绑定 | 完成当前边界 | DOM contract、required / optional 节点、安全事件绑定和结构测试已覆盖 | 未做额外 Playwright 全页面交互；当前不是阻塞 |
| `TASK-04` WebUI 真 feature 拆分 | 部分完成 | `live-training/index.js` 纯 helper 已抽出；Node 行为测试、前端结构测试、Chrome headless 模块 smoke 通过 | 还不是完整实时训练 UI 集成验证；`updateStatus()` / `updateProgress()` DOM 联动未做浏览器模拟 |
| `TASK-05` CSS 功能收口 | 阶段收口 | `21-history-panels.css` 已做维护分区注释；CSS diff check 通过 | 未继续拆 CSS 文件；不改变视觉 |
| `TASK-06` Runtime offloading 纯工具拆分 | 完成当前边界 | block swap config / CPU master / profiler helper 已拆出并合并；runtime 测试链通过 | CUDA stream / Event、swap plan、thread pool、hook 调度继续留在 `offloading.py`，除缺陷外不扩大 |
| `TASK-07` LoRA targeting / builder 拆分 | 阶段收口 | `targeting.py` 已抽出候选发现；LoRA construction / network cfg 测试通过 | builder / router / load / save 仍在 `network.py`；继续前必须只读评估和 characterization test |
| `TASK-08` Training forward canonical home | 阶段收口 | prior-preservation forward canonical home 已落地，旧 shim 保留；prior-preservation 测试通过 | `train.py` 其它方法 hook 化未继续推进 |
| `TASK-09` Config service 去 legacy | 持续推进，未完成 | `merge`、`output_runs`、`estimation`、`preflight`、`datasets`、`file_groups` 已多轮 direct-import-safe / shim / 显式依赖推进；Web config 拆分测试通过 | `_legacy.py` 旧函数体仍存在；datasets / file_groups 仍保留 facade lazy sync；不能标为完全去 legacy |
| `TASK-10` 类型检查分目录收紧 | 试点完成 | `config_compat.py` / `config_explain.py` 类型友好试点和测试已落地；ruff / py_compile / pytest 通过 | `.venv` 内无 `pyright` / `basedpyright` / `mypy`；正式类型门禁未建立 |

---

## 🛡️ 3. 隔离和禁止事项

一句话：后续继续整理时，保护用户数据比清理代码更重要。

禁止清理、覆盖或默认纳入提交的目录：

- `.venv/`
- `.worktrees/`
- `models/`
- `output/`
- `logs/`
- `post_image_dataset/`
- `configs/imported/`
- `configs/web-training-history/`
- `configs/web-training-queue/`
- `web/configs/`
- `bench/mfu/assets/`
- `tmp/`

禁止默认执行的操作：

- 不启动真实训练。
- 不下载模型。
- 不删除、移动或清理用户数据目录。
- 不做 `git add -A`。
- 不做 `git reset --hard`、`git checkout -- <path>`、强推或批量删除。

---

## 🔒 4. 高风险锁文件

一句话：这些入口多人最容易冲突，同一轮只能一个 worker 写。

| 锁名 | 文件 / 区域 | 说明 |
|---|---|---|
| `LOCK_TASKS` | `tasks.py`、`scripts/tasks/utilities.py` | 命令注册和测试入口 |
| `LOCK_TRAIN_MAIN` | `train.py` | 训练生命周期入口 |
| `LOCK_WEB_BOOT` | `web/static/app.js`、`web/static/js/features/anima-app/index.js`、`imports.js` | 前端模块加载顺序和 cache token |
| `LOCK_WEB_STATE` | `web/static/js/features/anima-app/chunks/01-scope-state.js` | `globalThis` 状态池 |
| `LOCK_WEB_EVENTS` | `web/static/js/features/anima-app/chunks/36-setup-event-listeners.js` | DOM 事件绑定中心 |
| `LOCK_WEB_DOM` | `web/static/index.html` | DOM id 跨模块契约 |
| `LOCK_WEB_CSS_ROOT` | `web/static/style.css`、`web/static/css/90-responsive.css` | CSS import 顺序和响应式兜底 |
| `LOCK_FRONTEND_TEST` | `tests/test_training_frontend_state.py` | 前端结构守门测试 |
| `LOCK_CONFIG_DOCS` | `docs/guidelines/training.md`、`docs/guidelines/inference.md`、`docs/README.md` | 用户入口文档 |
| `LOCK_RUNTIME_CORE` | `library/runtime/offloading.py` | block swap 调度和显存逻辑 |
| `LOCK_LORA_CORE` | `networks/lora_anima/network.py` | LoRA family 核心对象 |

---

## 🧭 5. 下一步建议

一句话：下一步不要继续大拆，优先处理最小、可验证、低冲突的缺口。

推荐顺序：

1. 继续 `TASK-09`：只读评估 `_legacy.py` 的 dataset 公开入口 lazy shim，不要直接删旧函数体。
2. 或者补 `TASK-04`：做完整 DOM fixture 或浏览器端模拟方案，再验证 `updateStatus()` / `updateProgress()` 联动。
3. 暂缓继续扩大 `TASK-07`：LoRA builder / router / load / save 深拆前，先补 characterization test。
4. 暂缓继续扩大 `TASK-06`：runtime block swap 当前边界已收口，除缺陷外不拆 CUDA stream / swap plan / hook 调度。
5. `TASK-10` 若要继续，需要先决定是否引入 `pyright` 或 `basedpyright`，否则只保持脚本级试点。

---

## 🧪 6. 当前阶段验证摘要

一句话：上一检查点已经跑过轻量综合验证，新的代码推进仍要按改动范围补验证。

已确认通过的检查包括：

- `git diff --cached --check`
- `git diff --check`
- 禁入路径扫描
- TASK-02 / TASK-07 / TASK-08 / TASK-10 小范围 pytest：`56 passed`
- TASK-09 Web config smoke：`7 passed`
- TASK-04 前端结构测试：`59 passed`
- 相关 ruff 检查：`All checks passed!`

后续最小验证建议：

| 方向 | 建议命令 |
|---|---|
| Web config | `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "<相关关键词>"` |
| 前端结构 | `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py` |
| LoRA targeting | `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py tests/test_network_cfg.py` |
| config 脚本 | `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_config_explain.py tests/test_config_compat.py tests/test_training_compat_matrix.py` |
| 格式和空白 | `git diff --check` |

---

## 📌 7. 当前结论

一句话：旧的逐轮施工日志已经完成使命，现在以这份检查点文档作为后续入口。

可以对外说：

- `TASK-03`、`TASK-06` 完成当前边界。
- `TASK-01`、`TASK-02`、`TASK-05`、`TASK-07`、`TASK-08`、`TASK-10` 阶段收口。
- `TASK-04`、`TASK-09` 继续部分完成，有明确下一步。

不能对外说：

- 不能说 `TASK-01` 到 `TASK-10` 全部完全完成。
- 不能说 Web config 已完全去 legacy。
- 不能说已经建立正式类型检查门禁。
- 不能说已经跑过真实 MFU benchmark 或真实训练。
