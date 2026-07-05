# 项目清理当前检查点与后续推进计划

一句话：这份文档替代旧的超长施工日志，只保留当前真实状态、剩余风险和下一步动作。

日期：2026-07-05
范围：`anima_lora` 主仓工程整理，不包含真实训练、模型下载、队列清理或用户数据清理。
来源：由上一版超长施工日志的最终状态表和阶段记录整理而来。
线上基线：本检查点已推送到 `webui/main`，以远端最新 HEAD 为准。
Git 同步口径：本地 `main` 只和 `webui/main` 沟通；`private/main` 不再作为默认同步或发布目标。

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
| `TASK-04` WebUI 真 feature 拆分 | 阶段收口 | `live-training/index.js` 纯 helper 已抽出；Node 行为测试、前端结构测试、Chrome headless 模块 smoke 通过；2026-07-05 已补 `updateStatus()` / `updateProgress()` DOM fixture | 还不是启动 WebUI 后的真实浏览器全页面交互；当前剩余风险不阻塞继续整理 |
| `TASK-05` CSS 功能收口 | 阶段收口 | `21-history-panels.css` 已做维护分区注释；CSS diff check 通过 | 未继续拆 CSS 文件；不改变视觉 |
| `TASK-06` Runtime offloading 纯工具拆分 | 完成当前边界 | block swap config / CPU master / profiler helper 已拆出并合并；runtime 测试链通过 | CUDA stream / Event、swap plan、thread pool、hook 调度继续留在 `offloading.py`，除缺陷外不扩大 |
| `TASK-07` LoRA targeting / builder 拆分 | 阶段收口 | `targeting.py` 已抽出候选发现；LoRA construction / network cfg 测试通过 | builder / router / load / save 仍在 `network.py`；继续前必须只读评估和 characterization test |
| `TASK-08` Training forward canonical home | 阶段收口 | prior-preservation forward canonical home 已落地，旧 shim 保留；prior-preservation 测试通过 | `train.py` 其它方法 hook 化未继续推进 |
| `TASK-09` Config service 去 legacy | 完成当前边界 | `merge`、`output_runs`、`estimation`、`preflight`、`datasets`、`file_groups` 已多轮 direct-import-safe / shim / 显式依赖推进；2026-07-05 已补齐 `_legacy.py` dataset、file group、preflight 公开入口 shim，覆盖对应 split module 的 `__all__`；preflight shim 已补 facade 状态恢复，raw_files 同步后会恢复 file group shim；datasets / raw_files legacy-private 额外 shim 已有测试保护；preflight、merge、output_runs、estimation、raw_files 公开入口旧函数体已收薄为转发桩；dataset user-facing 入口旧函数体已收薄为转发桩；`save_dataset_editor` 已稳定捕获 facade 注入的 raw writer，避免嵌套 helper 同步覆盖测试/运行时写入器；sample_prompts 4 个入口旧函数体已收薄为转发桩，split module 已补 ruff 友好的默认路径依赖；file_groups 全部 `__all__` 入口旧函数体已收薄为转发桩；dataset helper 入口旧函数体已收薄为转发桩；split module 同步规则已避免覆盖 legacy raw_files / file_groups shim；当前 8 个已拆 config split module 的 `__all__` 在 `_legacy.py` 内已无旧函数体残留；merge 私有 helper（variant metadata / custom variants）已改为转发 `merge.py`；output run 私有 helper（summary / config path / save-as path / mtime / time format）已改为转发 `output_runs.py`；file group 分组识别 / 归一化 / 归档命名 helper 首批 5 个已改为转发 `file_groups.py`；file group 分组构建 / fallback / 排序 / 权限判断 helper 21 个已改为转发 `file_groups.py`；file group id / label / 系统预设 / 备份路径 / 列表解析 helper 11 个已改为转发 `file_groups.py`；dataset summary/grouping helper 4 个旧函数体已改为转发 `datasets.py`；dataset 路径/default/row settings helper 17 个已改为转发 `datasets.py`；dataset 图片预览 / caption / nl-tag-mix helper 18 个已改为转发 `datasets.py`；dataset/raw legacy shim 已补 facade 状态恢复，避免污染 `config_service.list_config_file_groups`；preflight 私有 helper 12 个旧函数体已改为转发 `preflight.py`；公共 path / coercion / `_load` helper 已抽到 `common.py`，legacy 只保留转发；当前 `_legacy.py` 非转发函数降到 10 个且均为 shim 调度 / 恢复函数 | `_legacy.py` 仍作为兼容 facade 存在；若未来要彻底删除文件，需要先迁移所有外部 import surface 和第三方兼容入口 |
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

1. 当前检查点已提交并推送到 `webui/main`；不建议继续扩大重构。
2. 若未来继续 `TASK-09`，下一步是让 split modules 逐步复用 `common.py`，或制定删除 `_legacy.py` facade 的外部 import 迁移计划。
3. 暂缓继续扩大 `TASK-07`：LoRA builder / router / load / save 深拆前，先补 characterization test。
4. 暂缓继续扩大 `TASK-06`：runtime block swap 当前边界已收口，除缺陷外不拆 CUDA stream / swap plan / hook 调度。
5. `TASK-10` 若要继续，需要先决定是否引入 `pyright` 或 `basedpyright`，否则只保持脚本级试点。
6. 如果要补更强 UI 证据，再单独启动 WebUI 做真实浏览器全页面交互，不和代码拆分混在同轮。

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

2026-07-05 追加验证：

- `_legacy.py` dataset 公开入口 shim：`7 passed`
- `_legacy.py` / `tests/test_web_config_service.py` ruff：多轮 `All checks passed!`
- `updateStatus()` / `updateProgress()` DOM fixture：`3 passed`
- 前端结构完整守门：`60 passed`
- 前端测试 ruff：`All checks passed!`
- `_legacy.py` file group 公开入口 shim：`8 passed`
- `_legacy.py` legacy shim 矩阵（preflight / dataset / file_group / raw_files / merge / output_runs / estimation）：`15 passed`
- Web config preflight 定向：`30 passed`
- preflight 公开入口旧函数体收薄后，metadata 兼容导出和 preflight 定向：`31 passed`
- merge 公开入口旧函数体收薄后，merge / import / metadata 定向：`8 passed`
- output_runs 公开入口旧函数体收薄后，output run 定向：`8 passed`
- estimation 公开入口旧函数体收薄后，estimate_training_steps 定向：`1 passed`
- raw_files 公开入口旧函数体收薄后，metadata 兼容导出和 raw file 定向：`2 passed`
- datasets / raw_files legacy-private shim 归属测试：`3 passed`
- dataset preset 公开入口旧函数体收薄后，dataset legacy 定向：`5 passed`
- dataset preset 公开入口旧函数体收薄后，legacy shim 矩阵：`15 passed`
- dataset preset 公开入口旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- apply dataset preset 和 dataset preview 公开入口旧函数体收薄后，dataset preset 加宽验证：`28 passed`
- apply dataset preset 和 dataset preview 公开入口旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- dataset editor 公开入口旧函数体收薄后，editor 定向验证：`8 passed`
- dataset editor 公开入口旧函数体收薄后，dataset preset 加宽验证：`28 passed`
- dataset editor 公开入口旧函数体收薄后，legacy shim 矩阵：`15 passed`
- dataset editor 公开入口旧函数体收薄后，dataset 公开入口 AST 检查：全部为 shim
- dataset editor 公开入口旧函数体收薄后，`_legacy.py` / `datasets.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- sample_prompts 入口旧函数体收薄后，sample prompt 定向验证：`6 passed`
- sample_prompts 入口旧函数体收薄后，legacy shim 矩阵：`16 passed`
- sample_prompts 入口旧函数体收薄后，sample prompt AST 检查：全部为 shim
- sample_prompts 入口旧函数体收薄后，`_legacy.py` / `sample_prompts.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- file_groups 用户锁入口旧函数体收薄后，file group / sync 定向验证：`8 passed`
- file_groups 用户锁入口旧函数体收薄后，legacy shim 矩阵：`16 passed`
- file_groups 用户锁入口旧函数体收薄后，AST 检查：`set_user_file_lock` / `set_user_group_lock` 均为 shim
- split module 同步规则加固后，`_legacy.py` / config split modules / `tests/test_web_config_service.py` ruff：`All checks passed!`
- file_groups 分组创建 / 重命名 / 删除入口旧函数体收薄后，file group 定向验证：`5 passed`
- file_groups 分组创建 / 重命名 / 删除入口旧函数体收薄后，legacy shim 矩阵：`16 passed`
- file_groups 分组创建 / 重命名 / 删除入口旧函数体收薄后，AST 检查：3 个入口均为 shim
- file_groups 分组创建 / 重命名 / 删除入口旧函数体收薄后，`_legacy.py` / `file_groups.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- file_groups 排序 / 移动入口旧函数体收薄后，file group 定向验证：`5 passed`
- file_groups 排序 / 移动入口旧函数体收薄后，legacy shim 矩阵：`16 passed`
- file_groups 排序 / 移动入口旧函数体收薄后，AST 检查：5 个入口均为 shim
- file_groups 排序 / 移动入口旧函数体收薄后，`_legacy.py` / `file_groups.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- file_groups 列表 / metadata / 导出 / restore 入口旧函数体收薄后，file group 定向验证：`6 passed`
- file_groups 列表 / metadata / 导出 / restore 入口旧函数体收薄后，legacy shim 矩阵：`16 passed`
- file_groups 列表 / metadata / 导出 / restore 入口旧函数体收薄后，AST 检查：5 个入口均为 shim
- file_groups 列表 / metadata / 导出 / restore 入口旧函数体收薄后，`_legacy.py` / `file_groups.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- file_groups helper 入口旧函数体收薄后，file group helper 定向验证：`8 passed`
- file_groups helper 入口旧函数体收薄后，legacy shim 矩阵：`16 passed`
- file_groups helper 入口旧函数体收薄后，AST 检查：12 个 helper 均为 shim
- file_groups helper 入口旧函数体收薄后，`_legacy.py` / `file_groups.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- dataset helper 入口旧函数体收薄后，dataset helper 定向验证：`6 passed`
- dataset helper 入口旧函数体收薄后，dataset preset 加宽验证：`28 passed`
- dataset helper 入口旧函数体收薄后，legacy shim 矩阵：`16 passed`
- dataset helper 入口旧函数体收薄后，已拆 config split module `__all__` AST 对照：`remaining=0`
- dataset helper 入口旧函数体收薄后，`_legacy.py` / `datasets.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- merge 私有 helper 旧函数体收薄后，merge 定向验证：`3 passed`
- merge 私有 helper 旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- merge 私有 helper 旧函数体收薄后，AST 检查：4 个 helper 均调用 `_call_merge_impl`
- output run 私有 helper 旧函数体收薄后，output run 定向验证：`7 passed`
- output run 私有 helper 旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- output run 私有 helper 旧函数体收薄后，AST 检查：5 个 helper 均调用 `_call_output_runs_impl`
- file group 分组识别 / 归一化 / 归档命名 helper 旧函数体收薄后，file group 定向验证：`10 passed`
- file group 分组识别 / 归一化 / 归档命名 helper 旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- file group 分组识别 / 归一化 / 归档命名 helper 旧函数体收薄后，AST 检查：5 个 helper 均调用 `_call_file_groups_impl`
- file group 分组构建 / fallback / 排序 / 权限判断 helper 旧函数体收薄后，file group 定向验证：`11 passed`
- file group 分组构建 / fallback / 排序 / 权限判断 helper 旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- file group 分组构建 / fallback / 排序 / 权限判断 helper 旧函数体收薄后，AST 检查：21 个 helper 均调用 `_call_file_groups_impl`
- file group id / label / 系统预设 / 备份路径 / 列表解析 helper 旧函数体收薄后，file group 定向验证：`12 passed`
- file group id / label / 系统预设 / 备份路径 / 列表解析 helper 旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- file group id / label / 系统预设 / 备份路径 / 列表解析 helper 旧函数体收薄后，AST 检查：11 个 helper 均调用 `_call_file_groups_impl`
- dataset summary/grouping helper 旧函数体收薄和 dataset/raw facade 状态恢复后，dataset 宽筛选验证：`53 passed`
- dataset/raw facade 状态污染回归链验证：`11 passed`
- dataset/raw facade 状态恢复后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- dataset summary/grouping helper 旧函数体收薄后，AST 检查：4 个 helper 均调用 `_call_dataset_impl`
- dataset 路径/default/row settings helper 旧函数体收薄后，dataset 宽筛选验证：`54 passed`
- dataset 路径/default/row settings helper 旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- dataset 路径/default/row settings helper 旧函数体收薄后，AST 检查：17 个 helper 均调用 `_call_dataset_impl`
- dataset 图片预览 / caption / nl-tag-mix helper 旧函数体收薄后，预览和 preflight 定向验证：`14 passed`
- dataset 图片预览 / caption / nl-tag-mix helper 旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- dataset 图片预览 / caption / nl-tag-mix helper 旧函数体收薄后，AST 检查：18 个 helper 均调用 `_call_dataset_impl`
- preflight 私有 helper 旧函数体收薄后，preflight 加宽验证：`31 passed`
- preflight 私有 helper 旧函数体收薄后，`_legacy.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- preflight 私有 helper 旧函数体收薄后，AST 检查：11 个 helper 均调用 `_call_preflight_impl`
- `_legacy.py` 非转发旧函数体盘点：剩余 `24` 个
- `_legacy.py` 剩余旧体只读归类：`_inspect_network_weight` 需单独处理；数值转换、路径解析和 `_load` 属于公共 helper 候选；`_call_*_impl` 属于 shim 调度器，不应作为业务旧体统计。
- `common.py` 公共 helper 抽出后，preflight / common / legacy 定向验证：`38 passed`
- `common.py` 公共 helper 抽出后，`_legacy.py` / `common.py` / `preflight.py` / `tests/test_web_config_service.py` ruff：`All checks passed!`
- `common.py` 公共 helper 抽出后，AST 检查：`_legacy.py` 剩余 10 个非转发函数均为 shim 调度 / 恢复函数
- 当前检查点提交 / 推送后，`HEAD...webui/main` 核验为 `0 0`
- 相关路径空白检查：通过
- `tests/test_web_config_service.py` 全文件验证：60 秒上限内未跑完，未计为完整通过

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
- `TASK-04` 已完成 DOM fixture 验证这一阶段，可标为阶段收口。
- `TASK-09` 已完成 dataset、file group、preflight 公开入口 shim，datasets / raw_files legacy-private shim 归属测试，以及 preflight / merge / output_runs / estimation / raw_files / dataset / sample_prompts / file_groups 全部已拆 config split module `__all__` 入口旧函数体收薄；merge、output run、file group 私有 helper、dataset 私有 helper、preflight 私有 helper 和公共 helper 均已转发到 split/common module；dataset/raw shim 已补 facade 状态恢复；`_legacy.py` 当前只剩兼容 shim 调度 / 恢复函数。

不能对外说：

- 不能说 `TASK-01` 到 `TASK-10` 全部完全完成。
- 不能说 `_legacy.py` 文件已经删除，或 config_service 兼容 facade 已经不需要保留。
- 不能说已经建立正式类型检查门禁。
- 不能说已经跑过真实 MFU benchmark 或真实训练。
