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
- 本轮收口目标入口是 `docs/findings/project_cleanup_next_stage_goal_20260705.md`，完成记录见第 13 节。
- `docs/findings/project_cleanup_long_running_goal_20260705.md` 已完成归档，不要再作为活跃目标重复执行。
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
| `TASK-07` LoRA targeting / builder 拆分 | 阶段收口 | `targeting.py` 已抽出候选发现；已补 mixed Hydra/plain from-weights 恢复和 global-router save metadata characterization tests；checkpoint key 扫描已抽成私有 helper；LoRA/router 相关测试通过 | builder / router / load / save 主流程仍未深拆；后续只适合继续小步拆纯检测、metadata 组装或保存分流，不要改 checkpoint key 格式 |
| `TASK-08` Training forward canonical home | 阶段收口 | prior-preservation forward canonical home 已落地，旧 shim 保留；prior-preservation 测试通过 | `train.py` 其它方法 hook 化未继续推进 |
| `TASK-09` Config service 去 legacy | 完成当前边界 | `merge`、`output_runs`、`estimation`、`preflight`、`datasets`、`file_groups` 已多轮 direct-import-safe / shim / 显式依赖推进；2026-07-05 已补齐 `_legacy.py` dataset、file group、preflight 公开入口 shim，覆盖对应 split module 的 `__all__`；preflight shim 已补 facade 状态恢复，raw_files 同步后会恢复 file group shim；datasets / raw_files legacy-private 额外 shim 已有测试保护；preflight、merge、output_runs、estimation、raw_files 公开入口旧函数体已收薄为转发桩；dataset user-facing 入口旧函数体已收薄为转发桩；`save_dataset_editor` 已稳定捕获 facade 注入的 raw writer，避免嵌套 helper 同步覆盖测试/运行时写入器；sample_prompts 4 个入口旧函数体已收薄为转发桩，split module 已补 ruff 友好的默认路径依赖；file_groups 全部 `__all__` 入口旧函数体已收薄为转发桩；dataset helper 入口旧函数体已收薄为转发桩；split module 同步规则已避免覆盖 legacy raw_files / file_groups shim；当前 8 个已拆 config split module 的 `__all__` 在 `_legacy.py` 内已无旧函数体残留；merge 私有 helper（variant metadata / custom variants）已改为转发 `merge.py`；output run 私有 helper（summary / config path / save-as path / mtime / time format）已改为转发 `output_runs.py`；file group 分组识别 / 归一化 / 归档命名 helper 首批 5 个已改为转发 `file_groups.py`；file group 分组构建 / fallback / 排序 / 权限判断 helper 21 个已改为转发 `file_groups.py`；file group id / label / 系统预设 / 备份路径 / 列表解析 helper 11 个已改为转发 `file_groups.py`；dataset summary/grouping helper 4 个旧函数体已改为转发 `datasets.py`；dataset 路径/default/row settings helper 17 个已改为转发 `datasets.py`；dataset 图片预览 / caption / nl-tag-mix helper 18 个已改为转发 `datasets.py`；dataset/raw legacy shim 已补 facade 状态恢复，避免污染 `config_service.list_config_file_groups`；preflight 私有 helper 12 个旧函数体已改为转发 `preflight.py`；公共 path / coercion / `_load` helper 已抽到 `common.py`，legacy 只保留转发；当前 `_legacy.py` 非转发函数降到 10 个且均为 shim 调度 / 恢复函数 | `_legacy.py` 仍作为兼容 facade 存在；若未来要彻底删除文件，需要先迁移所有外部 import surface 和第三方兼容入口 |
| `TASK-10` 类型检查分目录收紧 | 试点门禁建立 | `pyright` 已加入 dev 依赖；`python tasks.py type-check` 默认检查 `scripts/config_compat.py` / `scripts/config_explain.py` 并通过；ruff / py_compile / pytest 已有阶段验证 | 这不是全仓类型门禁；后续扩大范围前要按目录逐步收紧 |

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

1. 本轮已按 `docs/findings/project_cleanup_next_stage_goal_20260705.md` 完成 `N0` 到 `N6`，不要重复执行同一目标书。
2. 当前检查点已提交并推送到 `webui/main`；不建议继续扩大重构。
3. 若未来继续 `TASK-09`，下一步优先迁移低风险内部 import surface，例如 `ROOT`、`_resolve_project_path`、`estimate_training_steps`。
4. 若继续 `TASK-07`，下一步只适合继续加 characterization tests 或拆更小的纯 helper；不要改保存/加载格式。
5. 暂缓继续扩大 `TASK-06`：runtime block swap 当前边界已收口，除缺陷外不拆 CUDA stream / swap plan / hook 调度。
6. `TASK-10` 已选择 `pyright` 并建立 config 脚本试点门禁；若继续，只按目录逐步扩大范围，不要一次性切全仓。
7. 如果要补更强 UI 证据，再单独启动 WebUI 做真实浏览器全页面交互，不和代码拆分混在同轮。

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
- 不能说已经建立全仓类型检查门禁。
- 不能说已经跑过真实 MFU benchmark 或真实训练。

---

## 📌 8. 2026-07-05 五项补缺推进记录

一句话：这轮补的是前一版 checkpoint 明确留下的证据缺口，不扩大高风险重构。

已推进：

- WebUI 真实浏览器 smoke：当前仓库服务启动在 `127.0.0.1:20104`，Chrome headless 通过 CDP 打开页面，7 个主 Tab 和训练页 3 个子视图均可切换；未发现 console、运行时异常或资源加载失败。
- `TASK-09` 最小收口：`environment_check_service.py` 不再直接 import `_legacy.PREPROCESS_ENV_REQUIRED_FILES`，改从 `metadata.py` 读取；`datasets.py` 复用 `common.py` 的纯标量 helper，不碰路径同步和 shim 调度。
- `TASK-07` 拆分前保护：新增 `test_create_network_global_fei_shared_a_cell_uses_real_builder_path`，用真实 `create_network(...)` 覆盖 `shared_A + route_per_layer=false + router_source=fei + router_targets` 的混合 Hydra/plain LoRA builder 路径。
- `TASK-10` 类型检查入口：新增 `python tasks.py type-check`，选择现有 `[tool.pyright]` 作为试点入口；当时 `.venv` 未安装 pyright，因此命令会明确提示并返回 `2`，没有改依赖锁文件。
- 文档 / CLI 漂移：修正 `test-dcw-v4` 默认 No-LoRA 语义，移除当前 `inference.py` 不存在的 `--postfix_weight` 入口说明，并把 DirectEdit help 从旧 `wd-swinv2-tagger-v3` 改为 Anima Tagger v1。

本轮已验证：

- WebUI Chrome CDP smoke：通过。
- `tests/test_web_config_service.py -k "common_config_helpers_import_without_facade_cycle or legacy_common_private_helpers_forward_to_common_module or config_module_facade_sync_preserves_legacy_raw_file_shims"`：`6 passed`。
- `tests/test_environment_check_service.py`：`8 passed`。
- `tests/test_lora_network_construction.py::test_create_network_global_fei_shared_a_cell_uses_real_builder_path`：`1 passed`。
- `tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py tests/test_network_cfg.py tests/test_router_compute.py`：`60 passed`。
- `python tasks.py type-check`：当时按预期提示 pyright 未安装并返回 `2`。
- `python tasks.py exp-test-directedit --help` 和 `python tasks.py test-dcw-v4 --help`：文字与当前行为一致。

仍不能对外说：

- 不能说已经建立全仓类型检查门禁；当时只是 pyright 入口和缺依赖提示。
- 不能说 `_legacy.py` 已经可以删除；它仍是兼容 facade。
- 不能说 LoRA builder/router/load/save 已经完成深拆；目前只是补了保护测试和一个只读扫描 helper。

---

## 📌 9. 2026-07-05 TASK-10 正式试点门禁

一句话：`TASK-10` 已从“入口存在但缺工具”推进到“pyright 试点门禁可执行并通过”。

已推进：

- `pyright>=1.1.411` 已加入 `[dependency-groups].dev`，`uv.lock` 已记录 `pyright` 和 `nodeenv`。
- `python tasks.py type-check` 默认检查 `scripts/config_compat.py` 和 `scripts/config_explain.py`。
- 仍支持显式传参扩大或缩小范围，例如 `python tasks.py type-check scripts/config_compat.py`。
- `[tool.pyright]` 增加 `reportMissingModuleSource = "none"`，避免第三方无源码包产生非行动性噪声。

当前验证：

- `uv add --dev pyright`：完成，安装 `pyright 1.1.411`。
- `.venv/bin/python -m pyright --version`：`pyright 1.1.411`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。

仍不能对外说：

- 不能说已经建立全仓类型检查门禁。
- 不能说所有目录都已类型友好；当前只覆盖 config 脚本试点范围。

---

## 📌 10. 2026-07-05 TASK-07 二阶段小步推进

一句话：这轮给 LoRA builder/router/load/save 继续拆分前补了保护测试，并只拆了一个只读检测 helper。

已推进：

- 新增 `test_create_network_from_weights_restores_mixed_hydra_plain_router_names`，覆盖 `create_network_from_weights(...)` 从混合 Hydra/plain checkpoint 恢复 `hydra_router_names`，确认 routed leg 建成 `HydraLoRAModule`，plain fallback leg 建成 `LoRAModule`。
- 新增 `test_save_weights_stamps_three_axis_metadata_for_shared_a_global_fei`，覆盖 `shared_A + route_per_layer=false + router_source=fei` 的保存行为，确认 `*_moe.safetensors` 写入 three-axis metadata、FEI metadata 和顶层 `global_router.*` key，且不写 `_routing_weights`。
- `networks/lora_anima/factory.py` 新增私有 `_CheckpointKeyScan` / `_scan_lora_checkpoint_keys()`，只抽离 checkpoint key 只读扫描和 dim / alpha / flag / name-set 收集逻辑。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py::test_create_network_from_weights_restores_mixed_hydra_plain_router_names tests/test_lora_network_construction.py::test_save_weights_stamps_three_axis_metadata_for_shared_a_global_fei`：`2 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py`：`4 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py tests/test_network_cfg.py tests/test_router_compute.py`：`62 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说 LoRA builder/router/load/save 已经完成深拆；这轮只做了拆分前保护和一个低风险只读扫描 helper。
- 不能说保存格式有变化；本轮没有改 public API、checkpoint key 格式或三轴路由语义。
- 不能说跑过真实训练、模型下载或真实 MFU benchmark。

---

## 📌 11. 2026-07-05 长目标继续推进：TASK-07 / TASK-09 小阶段

一句话：本轮按长目标执行书继续小步推进，完成了 LoRA from-weights 保护、router feature helper 拆分、Web config legacy 审计和 merge/common helper 复用。

阶段完成：

- `P0` 基线确认：当前在 `main`，跟踪 `webui/main`；执行前 `HEAD` 与 `webui/main` 一致；工作区只有本轮目标执行书未跟踪。
- `P1` TASK-07 保护测试：新增 `test_create_network_from_weights_recovers_fei_router_names_from_metadata_widths`，覆盖 `ss_router_source="fei"`、`ss_fei_feature_dim` 和 `router.weight` 宽度推断。
- `P2` TASK-07 helper 拆分：`factory.py` 新增私有 `_RouterFeatureScan` / `_scan_router_feature_metadata_and_names()`，只抽出 router feature dim 与 `sigma_router_names` 推断，不改 public API、checkpoint key 或三轴路由语义。
- `P5` TASK-09 只读审计：`_legacy.py` 当前未发现业务旧函数体，剩余真函数体属于 `_call_*_impl`、`_make_*_shim`、`_restore_raw_files_shims` 等兼容调度 / 恢复壳。
- `P6` TASK-09 split module 复用：`merge.py` 的路径 / 数据目录 helper 改为同步 `common.py` 状态后复用公共 helper，保留同名 wrapper，避免 direct import 和 monkeypatch 场景绕回真实配置根。
- Web config facade 补缺：`_legacy.py` 统一从 `metadata.py` 重导出 caption source 和配置标签常量，修复 `config_service` 兼容表面缺 `CAPTION_SOURCE_TXT` 等 metadata 常量的问题。
- `P7` 文档收口：本检查点记录本轮改动；长目标执行书 `project_cleanup_long_running_goal_20260705.md` 作为可追踪目标文档纳入本轮提交。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py::test_create_network_from_weights_recovers_fei_router_names_from_metadata_widths tests/test_lora_network_construction.py::test_create_network_from_weights_restores_mixed_hydra_plain_router_names`：`2 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py::test_create_network_from_weights_recovers_fei_router_names_from_metadata_widths tests/test_lora_network_construction.py::test_create_network_from_weights_restores_mixed_hydra_plain_router_names tests/test_factory_metadata_flow.py`：`6 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "merge_common_path_helpers_forward_to_common_module or merge_helpers_remain_available_from_legacy_module or legacy_merge_private_helpers_forward_to_split_module or merge_module_imports_without_facade_cycle or common_config_helpers"`：`5 passed, 150 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile web/services/config/merge.py tests/test_web_config_service.py networks/lora_anima/factory.py tests/test_lora_network_construction.py`：通过。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py tests/test_network_cfg.py tests/test_router_compute.py`：`63 passed, 2 warnings`。warning 来自本机 GTX 960 与当前 PyTorch CUDA 架构不匹配，不影响本轮 CPU 侧测试结论。
- `tests/test_web_config_service.py -k "legacy or dataset or file_group or preflight or merge or output_run"`：单次 60 秒内未完整跑完；修复 metadata facade 失败后，已拆块验证覆盖同一方向。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py::test_config_metadata_exports_remain_available_from_legacy_facade`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or merge or common_config_helpers"`：`9 passed, 146 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "dataset and not runtime_preflight"`：`54 passed, 101 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_group or output_run"`：`23 passed, 132 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "preflight"`：`31 passed, 124 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说 LoRA builder/router/load/save 已经完成深拆；本轮只是继续拆出一个低风险只读检测 helper。
- 不能说 `_legacy.py` 可以删除；它仍是兼容 facade。
- 不能说全仓类型检查门禁已经建立；当前仍是既有试点范围。
- 不能说跑过真实训练、模型下载、真实 MFU benchmark 或全量 WebUI 浏览器交互。

---

## 📌 12. 2026-07-05 文档清理与下一阶段入口

一句话：已完成的长目标文档已降级为历史记录，当时新的下一阶段目标文档也已经创建。

已清理：

- `docs/findings/project_cleanup_long_running_goal_20260705.md` 已标记为“已完成归档”。
- 旧文档中的可复制 Prompt 已改为历史记录，避免后续重复执行 `P0/P1/P2/P5/P6/P7`。
- 旧文档保留为审计材料，没有删除历史证据。

新入口：

- `docs/findings/project_cleanup_next_stage_goal_20260705.md`
- 下一阶段从 `N0` 到 `N6`，主线是：
  - `TASK-07` save variant characterization。
  - `TASK-07` save metadata helper 小拆分。
  - `TASK-07` loading split/refuse helper 边界测试。
  - `TASK-09` config import surface 审计。
  - `TASK-09` facade 保留清单和风险文档。

使用方式：

```text
请按 docs/findings/project_cleanup_next_stage_goal_20260705.md 连续推进下一阶段项目清理目标。
```

仍不能对外说：

- 不能说已完成文档被删除；只是完成归档。
- 不能说第 12 节写入时下一阶段已经执行；当时只是创建了入口，当前完成记录见第 13 节。

---

## 📌 13. 2026-07-05 下一阶段目标执行收口

一句话：本轮按下一阶段目标书完成了 `N0` 到 `N6`，重点是 LoRA save/load 边界保护和 Web config facade 保留清单。

阶段完成：

- `N0` 基线复核：当前在普通 `main` checkout，跟踪 `webui/main`；旧目标 `project_cleanup_long_running_goal_20260705.md` 已标记完成归档；本目标书已作为本轮入口并在收口时改为完成归档。
- `N1` save variant characterization：新增 `tests/test_lora_save_pipeline.py`，覆盖 `lora_save.save_network_weights()` 的 Hydra、StackedExperts、Chimera 保存分流，确认 `*_moe.safetensors` / `*_chimera.safetensors`、metadata 透传和关键 key 形态不变。
- `N2` save metadata helper 小拆分：`networks/lora_anima/network.py` 新增私有 `_stamp_lora_save_metadata(...)`，只移动 `LoRANetwork.save_weights()` 的 metadata 组装；`metadata={}` 时仍不回填 `ss_network_spec`，由测试继续保护。
- `N3` loading split/refuse helper 边界测试：新增 `tests/test_lora_loading_keys.py`，覆盖 `_stack_lora_ups` 排序 stack、malformed expert index 报错、Hydra q/k/v refuse 且不误收 plain LoRA leg、Chimera content/freq 双池 q/k/v refuse。
- `N4` config import surface 审计：只读确认 `_legacy.py` 仍是 `config_service` 兼容 facade 的源头，`config_service.py` 仍通过 `_legacy.__dict__` 初始化历史导入面。
- `N5` facade 保留清单：明确 `_legacy.py` 当前不能删除；删除前必须先迁移外部 import surface 并补 facade 兼容测试。
- `N6` 文档和验证收口：本检查点、旧目标书、新目标书、LoRA helper 和新增测试已准备显式提交推送；未使用 `git add -A`。

`TASK-09` facade 保留清单：

| 类别 | 当前必须保留 |
|---|---|
| facade 文件 | `web.services.config_service` |
| legacy 文件 | `web.services.config._legacy` |
| 路由 API | `list_methods`、`list_variants`、`list_presets`、`load_merged_config`、`estimate_training_steps`、`preflight_training_config`、raw file、dataset preset、file group、output run、sample prompts 等入口 |
| 路径 / 运行时 | `ROOT`、`CONFIGS_DIR`、`DATASET_PRESETS_DIR`、`_resolve_project_path`、`_config_file_path`、`is_web_runtime_config` |
| metadata facade | `CAPTION_SOURCE_*`、`CONFIG_FILE_LABELS_ZH`、`PREPROCESS_ENV_REQUIRED_FILES`、`SUPPORTED_TRAINING_SAMPLE_SAMPLERS`、`get_field_help`、`get_groups` |
| legacy shim | `_call_*_impl`、`_make_*_shim`、`_restore_raw_files_shims` |

后续可迁移候选：

- `environment_check_service.py` 和 `project_python.py` 可考虑从轻量 project root helper 读取 `ROOT`。
- `image_test_service.py` 可考虑从 `web.services.config.common` 读取 `_resolve_project_path`，但要先确认 monkeypatch 同步场景。
- `training/history.py` 可考虑改走 `web.services.config.estimation.estimate_training_steps`。
- `web/routes/config.py` 依赖面最宽，建议最后迁。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py tests/test_network_cfg.py tests/test_router_compute.py`：`70 passed, 2 warnings`。warning 来自本机 GTX 960 与当前 PyTorch CUDA 架构不匹配，不影响本轮 CPU 侧测试结论。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy or dataset or file_group or preflight or merge or output_run or common_config_helpers"`：60 秒内未完整跑完，已拆块验证。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy or merge or common_config_helpers"`：`38 passed, 117 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "dataset and not runtime_preflight"`：`54 passed, 101 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_group or output_run"`：`23 passed, 132 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "preflight"`：`31 passed, 124 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说 `_legacy.py` 可以删除；它仍是兼容 facade。
- 不能说 LoRA save/load 已经完成深拆；本轮只是补保护测试并抽一个 metadata helper。
- 不能说改过 checkpoint key、public API 或三轴路由语义；本轮刻意没有改。
- 不能说已经建立全仓类型检查门禁；当前仍是既有试点范围。
- 不能说跑过真实训练、模型下载、真实 MFU benchmark 或全量 WebUI 浏览器交互。
