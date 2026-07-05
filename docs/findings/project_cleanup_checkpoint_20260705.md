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
- 当前活跃强制长跑目标入口是 `docs/findings/project_cleanup_sustained_goal_20260706.md`。
- `docs/findings/project_cleanup_sustained_goal_20260705.md` 已完成归档，提交 `bd591b83`，不要再作为活跃目标重复执行。
- 本轮收口目标入口 `docs/findings/project_cleanup_next_stage_goal_20260705.md` 已完成归档，完成记录见第 13 节。
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
| `TASK-07` LoRA targeting / builder 拆分 | 阶段收口 | `targeting.py` 已抽出候选发现；已补 mixed Hydra/plain from-weights 恢复、global-router save metadata、split q/k/v、missing alpha、旧 router / sigma_mlp 拒绝等 characterization tests；checkpoint key 扫描已抽成私有 helper；LoRA/router 相关测试通过 | builder / router / load / save 主流程仍未深拆；新增测试多为 synthetic state_dict 级边界保护，不代表真实 checkpoint 全覆盖或加载格式改变；后续只适合继续小步拆纯检测、metadata 组装或保存分流，不要改 checkpoint key 格式 |
| `TASK-08` Training forward canonical home | 阶段收口 | prior-preservation forward canonical home 已落地，旧 shim 保留；prior-preservation 测试通过 | `train.py` 其它方法 hook 化未继续推进 |
| `TASK-09` Config service 去 legacy | 完成当前边界 | `merge`、`output_runs`、`estimation`、`preflight`、`datasets`、`file_groups` 已多轮 direct-import-safe / shim / 显式依赖推进；2026-07-05 已补齐 `_legacy.py` dataset、file group、preflight 公开入口 shim，覆盖对应 split module 的 `__all__`；preflight shim 已补 facade 状态恢复，raw_files 同步后会恢复 file group shim；datasets / raw_files legacy-private 额外 shim 已有测试保护；preflight、merge、output_runs、estimation、raw_files 公开入口旧函数体已收薄为转发桩；dataset user-facing 入口旧函数体已收薄为转发桩；`save_dataset_editor` 已稳定捕获 facade 注入的 raw writer，避免嵌套 helper 同步覆盖测试/运行时写入器；sample_prompts 4 个入口旧函数体已收薄为转发桩，split module 已补 ruff 友好的默认路径依赖；file_groups 全部 `__all__` 入口旧函数体已收薄为转发桩；dataset helper 入口旧函数体已收薄为转发桩；split module 同步规则已避免覆盖 legacy raw_files / file_groups shim；当前 8 个已拆 config split module 的 `__all__` 在 `_legacy.py` 内已无旧函数体残留；merge 私有 helper（variant metadata / custom variants）已改为转发 `merge.py`；output run 私有 helper（summary / config path / save-as path / mtime / time format）已改为转发 `output_runs.py`；file group 分组识别 / 归一化 / 归档命名 helper 首批 5 个已改为转发 `file_groups.py`；file group 分组构建 / fallback / 排序 / 权限判断 helper 21 个已改为转发 `file_groups.py`；file group id / label / 系统预设 / 备份路径 / 列表解析 helper 11 个已改为转发 `file_groups.py`；dataset summary/grouping helper 4 个旧函数体已改为转发 `datasets.py`；dataset 路径/default/row settings helper 17 个已改为转发 `datasets.py`；dataset 图片预览 / caption / nl-tag-mix helper 18 个已改为转发 `datasets.py`；dataset/raw legacy shim 已补 facade 状态恢复，避免污染 `config_service.list_config_file_groups`；preflight 私有 helper 12 个旧函数体已改为转发 `preflight.py`；公共 path / coercion / `_load` helper 已抽到 `common.py`，legacy 只保留转发；当前 `_legacy.py` 非转发函数降到 10 个且均为 shim 调度 / 恢复函数 | `_legacy.py` 仍作为兼容 facade 存在；若未来要彻底删除文件，需要先迁移所有外部 import surface 和第三方兼容入口 |
| `TASK-10` 类型检查分目录收紧 | 试点门禁建立并扩到白名单边界 | `pyright` 已加入 dev 依赖；`python tasks.py type-check` 当前默认检查 `library/config`、config 脚本、`scripts/tasks/_common.py` / `utilities.py`、以及 10 个选定 Web config split module；默认 pyright 白名单由测试固定，针对本轮改动文件和护栏测试跑过 ruff / py_compile / pytest | 这不是全仓类型门禁，也不是整个 `web/services/config/` 门禁；`datasets.py` / `_legacy.py` 暂缓加入，后续新增 Web config 文件必须有单独理由和窄测试 |

当前 `type-check` 白名单边界：

- 已纳入：`library/config`、`scripts/config_compat.py`、`scripts/config_explain.py`、`scripts/tasks/_common.py`、`scripts/tasks/utilities.py`。
- 已纳入 Web config 选定 split module：`common.py`、`estimation.py`、`file_groups.py`、`merge.py`、`metadata.py`、`output_runs.py`、`paths.py`、`preflight.py`、`raw_files.py`、`sample_prompts.py`。
- 暂缓纳入：`web/services/config/datasets.py`、`web/services/config/_legacy.py` 和整个 `web/services/config/` 目录。
- 新增 Web config 文件进默认门禁前，必须先有单独理由、direct-import 测试或窄行为测试；不能只因为 pyright 单跑通过就自动加入。

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

1. 下一轮优先按 `docs/findings/project_cleanup_sustained_goal_20260706.md` 执行跨子系统强制长跑目标，不要重复执行已完成的 `project_cleanup_sustained_goal_20260705.md`。
2. 当前检查点已提交并推送到 `webui/main`；不建议继续扩大重构。
3. 若未来继续 `TASK-09`，下一步优先迁移低风险内部 import surface，例如 `ROOT`、`_resolve_project_path`、`estimate_training_steps`。
4. 若继续 `TASK-07`，下一步只适合继续加 characterization tests 或拆更小的纯 helper；不要改保存/加载格式。
5. 暂缓继续扩大 `TASK-06`：runtime block swap 当前边界已收口，除缺陷外不拆 CUDA stream / swap plan / hook 调度。
6. `TASK-10` 已选择 `pyright` 并建立当前白名单门禁；Web config 扩围已接近边界，后续先暂停继续加 Web config 文件，不要一次性切全仓或整目录。
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

当时新入口：

- `docs/findings/project_cleanup_next_stage_goal_20260705.md`
- 下一阶段从 `N0` 到 `N6`，主线是：
  - `TASK-07` save variant characterization。
  - `TASK-07` save metadata helper 小拆分。
  - `TASK-07` loading split/refuse helper 边界测试。
  - `TASK-09` config import surface 审计。
  - `TASK-09` facade 保留清单和风险文档。

使用方式：

```text
请按 docs/findings/project_cleanup_sustained_goal_20260706.md 执行跨子系统强制长跑项目清理目标。
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

---

## 📌 14. 2026-07-05 强制长跑目标书创建

一句话：为避免后续目标再次二十分钟收口，已创建带硬性时间和阶段门槛的新任务书。

新入口：

- `docs/findings/project_cleanup_sustained_goal_20260705.md`

关键变化：

- 不再使用“建议总时长”作为软约束。
- 完成条件必须同时满足：
  - `goal.timeUsedSeconds >= 7200`。
  - 至少完成 `12` 个可验收小阶段。
  - 至少完成 `3` 个推进轮。
  - 总验证通过，或 60 秒超时项拆分验证并记录。
  - checkpoint 更新、提交并推送到 `webui/main`。
- 如果基础轮提前完成但未达时间或阶段数，必须进入 `EXT` 扩展阶段池继续推进。
- 明确禁止用 `sleep`、空等、无意义轮询凑时间。

建议启动语：

```text
请按 docs/findings/project_cleanup_sustained_goal_20260706.md 执行跨子系统强制长跑项目清理目标。
```

仍不能对外说：

- 第 14 节只记录目标书创建；实际执行进度见第 15 节。
- 不能说强制长跑目标已经完成；完成仍受第 15 节记录和 `goal.timeUsedSeconds >= 7200` 约束。

---

## 📌 15. 2026-07-05 强制长跑执行记录

一句话：本节记录 `project_cleanup_sustained_goal_20260705.md` 的实际执行进度，避免和第 14 节的“目标书创建”混在一起。

### R0 基线和目标启动

一句话：R0 只做事实确认、活跃目标确认和可回滚记录，不做源码扩张。

阶段完成：

- `S00` 基线确认：当前在普通 `main` checkout，跟踪 `webui/main`；`HEAD` 为 `7c5c277c test: cover next cleanup boundaries`；`HEAD...webui/main` 为 `0 0`；本轮记录开始 epoch 为 `1783266658`。
- `S01` 已完成目标归档确认：`project_cleanup_long_running_goal_20260705.md` 和 `project_cleanup_next_stage_goal_20260705.md` 均标记为已完成归档；当前活跃入口是 `docs/findings/project_cleanup_sustained_goal_20260705.md`。

启动时工作区状态：

- 已存在未提交文档改动：`docs/findings/project_cleanup_checkpoint_20260705.md`、`docs/findings/project_cleanup_long_running_goal_20260705.md`、`docs/findings/project_cleanup_next_stage_goal_20260705.md`。
- 已存在未跟踪新目标书：`docs/findings/project_cleanup_sustained_goal_20260705.md`。
- 这些改动属于当前清理目标文档链，后续提交前仍会显式列文件 stage，不使用 `git add -A`。

并行审计安排：

- `S-AUDIT-LORA`：只读审计 LoRA save/load 后续保护点。
- `S-AUDIT-WEB-CONFIG`：只读审计 Web config facade/import surface。
- `S-AUDIT-TYPECHECK`：只读审计 type-check 试点扩大候选。

### R1 LoRA save/load 保护加宽

一句话：R1 只补保存/加载边界测试和一个私有 helper，不改 checkpoint key、文件名、public API 或三轴路由语义。

阶段完成：

- `S02` save metadata 行为保护：新增 `test_save_weights_preserves_non_empty_metadata_and_stamps_network_spec`，确认非空 metadata 会保留原字段并写入当前 spec / 三轴 stamp；既有空 dict 行为仍由 `test_save_weights_stamps_three_axis_metadata_for_shared_a_global_fei` 保护。
- `S03` `lora_save` fallback 分流保护：新增 `test_save_network_weights_hydra_fallback_writes_moe_sibling`，确认 `save_variant=""` 但 state dict 含 `.lora_up_weight` 时仍写 `*_moe.safetensors`，不生成普通输出。
- `S04` loading malformed / incomplete key 边界：新增非连续 expert 编号报错测试、Chimera pool 非连续报错测试、Hydra 缺组件不融合测试、Chimera 缺 pool 不融合测试。
- `S05` loading helper 只读审计：并行审计确认后续候选只适合继续拆 `_stack_lora_ups` / `_stack_chimera_lora_ups` 周边的纯检测函数；`_refuse_unfused_attn_lora_keys` 和保存/加载格式不适合本轮扩大。
- `S06` 小范围 helper 拆分：`networks/lora_anima/loading.py` 新增私有 `_stack_contiguous_experts(...)`，只统一 expert 编号连续性校验和 `torch.stack`；合法 key 的输出形态不变。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py`：`4 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`8 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_network_construction.py -k "metadata or save_weights"`：`3 passed, 8 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_lora_network_construction.py -k "metadata or save_weights or stack_lora_ups or stack_chimera_lora_ups or fallback or missing_component or missing_pool"`：`11 passed, 12 deselected`。

仍不能对外说：

- 不能说 LoRA save/load 已经深拆完成；本轮只补保护和一个私有小 helper。
- 不能说缺 q/k/v 或缺 Chimera pool 会硬报错；当前保护的是“保留原 key、不融合”的现有行为。

### R2 Web config facade/import surface 收口

一句话：R2 继续保护 facade 兼容面，补 direct-import smoke，并只做一个 `common.py` 低风险复用。

阶段完成：

- `S07` import surface grep 审计：确认 `web.services.config_service` 仍被 routes、training runtime/history、image test、GUI variant 测试、training resume 等依赖；`web.services.config._legacy` 仍是 facade 导出源，不能删除。
- `S08` facade metadata 导出保护：`test_config_metadata_exports_remain_available_from_legacy_facade` 改为遍历 `web.services.config.metadata.__all__`，避免新增 metadata 常量后 facade / legacy 漏导出。
- `S09` split module direct-import smoke：加强 `test_sample_prompts_module_imports_without_facade_cycle`，确认直接导入 `sample_prompts` 本身不拉起 `config_service` 或 `_legacy`。
- `S10` `common.py` 复用候选审计：`file_groups.py` 的 `_load` / `_safe_resolve` / `_display_path` 与 `common.py` 重复，且已有 file group / legacy 测试可兜住；`datasets.py` 和 `preflight.py` 依赖面更宽，本轮不碰。
- `S11` 一个低风险 `common.py` 复用：`web/services/config/file_groups.py` 的 `_load` / `_safe_resolve` / `_display_path` 改为转发 `common.py`，并在调用前同步本模块的 `ROOT` / `CONFIGS_DIR`；新增 `test_file_group_common_path_helpers_forward_to_common_module` 保护。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or sample_prompts_module_imports_without_facade_cycle or file_group_common_path_helpers_forward_to_common_module"`：`3 passed, 153 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy or file_group_common_path_helpers_forward_to_common_module"`：`34 passed, 122 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile web/services/config/file_groups.py tests/test_web_config_service.py`：通过。

仍不能对外说：

- 不能说 `_legacy.py` 可以删除；当前 import surface 仍明确依赖兼容 facade。
- 不能说所有 split module 都已彻底复用 `common.py`；本轮只处理 `file_groups.py` 的三个私有 helper。

### R3 type-check 试点小范围扩大

一句话：R3 只把一个低依赖配置 provenance 模块加入默认 pyright 试点，不做全仓类型检查切换。

阶段完成：

- `S12` 候选选择审计：并行审计和本地复核都指向 `library/config/provenance.py`，它依赖面窄、与当前 config 收口主线相关；`library/config` 整目录和 `scripts/tasks` 整目录虽可作为后续候选，但本轮范围偏大。
- `S13` 候选单独 pyright：`timeout 60 .venv/bin/python tasks.py type-check library/config/provenance.py` 通过，结果为 `0 errors, 0 warnings, 0 informations`。
- `S14` 纳入默认 type-check：`scripts/tasks/utilities.py::TYPE_CHECK_TARGETS` 新增 `library/config/provenance.py`。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config/provenance.py scripts/config_compat.py scripts/config_explain.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py library/config/provenance.py`：通过。

仍不能对外说：

- 不能说已经建立全仓 type-check；当前仍是小范围 pyright 试点。
- 不能说 `library/config` 整目录已经纳入默认门禁；本轮只纳入 `provenance.py`。

### EXT 第一组低风险扩展阶段

一句话：硬性时间门槛未满足时，继续做有验证价值的小阶段，不靠空等凑时间。

阶段完成：

- `E01` standard save metadata 透传测试：新增 `test_save_network_weights_standard_preserves_metadata_and_adds_hashes`，确认普通 `.safetensors` 保存分支保留用户 metadata，并添加 `sshs_model_hash` / `sshs_legacy_hash`。
- `E02` loading malformed key 错误信息加宽：新增 `test_stack_lora_downs_non_contiguous_expert_indices_raise_value_error`，覆盖 StackedExperts `lora_downs` 非连续 expert 编号报错。
- `E08` split module direct import smoke：新增 `test_paths_module_imports_without_facade_cycle`，确认 `web.services.config.paths` 作为路径底座可直接导入，不拉起 `config_service` / `_legacy`。
- `E08` common helper 快照测试：新增 `test_common_config_path_helpers_work_without_facade_snapshot`，用 `tmp_path` 覆盖 `common.py` 的 `_load` / `_safe_resolve` / `_safe_config_subdir` / `_resolve_project_path` / `_display_path`。
- `E10` type-check 候选单独验证：`library/config/io.py`、`library/config/schema.py`、`web/services/config/paths.py web/services/config/common.py` 单独 pyright 均通过。
- `E10` 默认 type-check 小步加宽：`scripts/tasks/utilities.py::TYPE_CHECK_TARGETS` 继续纳入 `library/config/io.py` 和 `library/config/schema.py`，仍不纳入整个 `library/config/` 目录。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "paths_module_imports_without_facade_cycle"`：`1 passed, 156 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "imports_without_facade_cycle"`：`7 passed, 150 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check library/config/io.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check library/config/schema.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/paths.py web/services/config/common.py`：`0 errors, 0 warnings, 0 informations`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "common_config_path_helpers_work_without_facade_snapshot or common_config_helpers_import_without_facade_cycle"`：`2 passed, 156 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config/io.py library/config/provenance.py library/config/schema.py scripts/config_compat.py scripts/config_explain.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py`：`14 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "common_config_path_helpers_work_without_facade_snapshot or paths_module_imports_without_facade_cycle"`：`2 passed, 156 deselected`。

仍不能对外说：

- 不能说默认 type-check 已覆盖整个 `library/config/`；当前只是纳入 `io.py` / `provenance.py` / `schema.py` 三个文件。
- 不能说 Web config facade 已可删除；direct-import smoke 只是证明 split module 可独立导入。

### EXT 第二组低风险扩展阶段

一句话：第二组 EXT 继续补 LoRA metadata 保护和 config helper 直接使用证据，仍不扩大到高风险重构。

阶段完成：

- `E03` no-metadata actionable error 变体：新增 `test_bare_stacked_experts_weights_sd_raises_actionable_error`，确认 StackedExperts / FeRA 形态的裸 `weights_sd` 缺 safetensors metadata 时，同样报出包含 `three-axis` / `load_file` / `metadata=` 的可操作错误。
- `E10` type-check 默认试点再加宽：在已验证通过后，默认门禁纳入 `library/config/io.py` 和 `library/config/schema.py`；仍不纳入整个 `library/config/`。
- `E08` common helper 直接使用证据：`test_common_config_path_helpers_work_without_facade_snapshot` 已通过，确认直接 monkeypatch `common.py` 的 `ROOT` / `CONFIGS_DIR` 后，路径 helper 行为稳定。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py`：`5 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py`：`19 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_factory_metadata_flow.py`：通过。

仍不能对外说：

- 不能说旧无 metadata MoE checkpoint 可自动恢复；当前行为仍是明确报错，要求传 `file=` 或 `metadata=`。

### EXT 第三组低风险扩展阶段

一句话：第三组 EXT 继续补 Web config facade 和 output run helper 的兼容保护，仍只加测试。

阶段完成：

- `E06` facade common wrapper 同步测试：新增 `test_config_service_common_private_wrappers_sync_facade_state_to_legacy`，确认 `config_service._load` / `_safe_config_subdir` / `_display_path` 调用前会把 facade 上的 `ROOT` / `CONFIGS_DIR` 同步给 `_legacy`。
- `E07` output_runs 直接私有 helper smoke：新增 `test_output_runs_direct_private_helpers_work_without_facade_cycle`，确认 `_normalize_output_run_name.__wrapped__`、`_output_run_config_path`、`_normalize_output_run_save_as_path` 可在 split module 直接导入后使用，不拉起 `config_service` / `_legacy`。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "config_service_common_private_wrappers_sync_facade_state_to_legacy or output_runs_direct_private_helpers_work_without_facade_cycle"`：`2 passed, 158 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "imports_without_facade_cycle or direct_private_helpers or common_private_wrappers"`：`9 passed, 151 deselected`。

仍不能对外说：

- 不能说 output run 全链路做了浏览器或真实运行验证；本轮只是 split module 私有 helper 和 facade 同步保护。

### EXT 第四组低风险扩展阶段

一句话：第四组 EXT 继续小步扩大 type-check，并修正旧目标书里容易被复制误用的入口文案。

阶段完成：

- `E09` type-check 扩围候选审计：单独验证 `scripts/tasks/utilities.py`、`web/services/config/paths.py`、`web/services/config/common.py`，均为 `0 errors, 0 warnings, 0 informations`。
- `E10` 默认 type-check 小步加宽：`TYPE_CHECK_TARGETS` 纳入 `scripts/tasks/utilities.py`、`web/services/config/common.py`、`web/services/config/paths.py`；默认门禁仍是小文件试点，不是整目录或全仓。
- `E13` 旧目标书引用检查：扫描 `project_cleanup_*20260705.md`，发现旧 `next_stage` 可复制 Prompt 和 checkpoint 旧使用方式会误导后续 agent，已改为 sustained 入口。
- `E14` docs diff check：针对 checkpoint 和 next-stage goal 文档执行 `git diff --check`，通过。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check scripts/tasks/utilities.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/paths.py web/services/config/common.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check scripts/tasks/utilities.py web/services/config/paths.py web/services/config/common.py library/config/io.py library/config/provenance.py library/config/schema.py scripts/config_compat.py scripts/config_explain.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 8 个小文件。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py web/services/config/common.py web/services/config/paths.py`：通过。
- `rg -n "请按 docs/findings/project_cleanup_next_stage_goal_20260705.md|可直接复制给 Codex 的下一阶段 Prompt" docs/findings/project_cleanup_*20260705.md`：无匹配。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md docs/findings/project_cleanup_next_stage_goal_20260705.md`：通过。

仍不能对外说：

- 不能说旧目标书历史记录被删除；只修正了后续入口文案。

### EXT 第五组低风险扩展阶段

一句话：第五组 EXT 给 StackedExperts 加载重组边界补直接测试，避免只靠 global-router 间接覆盖。

阶段完成：

- `E02` loading malformed / incomplete key 继续加宽：新增 `_refuse_split_stacked_experts_keys` 的 q/k/v 重组测试，确认 independent-A 的 `lora_up_weight` 会按 out 维拼接，`lora_down_weight` 取首个组件。
- `E02` incomplete key 行为保护：新增 StackedExperts 缺组件不融合测试，确认缺 k 组件时不生成 fused key，原 split key 留给后续 load_state_dict 暴露。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`11 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_global_router.py -k "stacked_experts_save_load_round_trip or stacked_experts_module_consumes_routing_weights"`：`2 passed, 15 deselected`。

仍不能对外说：

- 不能说 StackedExperts 加载链路已完整深拆；这里只是补 split/refuse 边界保护。

### EXT 阶段中期综合验证

一句话：多组 EXT 后跑一轮中等宽度验证，确认 LoRA、Web config 和 type-check 没被局部改动拖坏。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py tests/test_global_router.py tests/test_network_cfg.py tests/test_router_compute.py`：`81 passed, 2 warnings`。warning 仍是本机 GTX 960 与当前 PyTorch CUDA 架构不匹配，不影响本轮 CPU 侧结论。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_exports or legacy or direct_private_helpers or imports_without_facade_cycle or common_config_helpers or file_group_common_path_helpers"`：`44 passed, 116 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。

仍不能对外说：

- 不能说已经跑完全量测试；本轮是与当前改动相关的中等宽度验证。

### EXT Web config 复用后加宽验证

一句话：`file_groups.py` 复用 `common.py` 后，再补一轮更贴近业务的 Web config 筛选验证。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_group or output_run"`：`25 passed, 135 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "dataset and not runtime_preflight"`：`54 passed, 106 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile networks/lora_anima/loading.py scripts/tasks/utilities.py web/services/config/file_groups.py tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py tests/test_web_config_service.py`：通过。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py`：60 秒上限内未完整跑完，未计为全文件通过；本轮继续以拆分筛选验证覆盖相关风险。

仍不能对外说：

- 不能说 `tests/test_web_config_service.py` 全文件已经在 60 秒内完整通过；本轮仍是拆分筛选验证。

### EXT 第六组低风险扩展阶段

一句话：第六组 EXT 继续保护加载阶段遇到形状不一致 split key 时的保守跳过行为。

阶段完成：

- `E02` Hydra shape mismatch 边界：新增 `test_refuse_split_hydra_keys_inconsistent_up_shape_leaves_keys_untouched`，确认 expert rank 不一致时不生成 fused key，原 split key 保留。
- `E02` StackedExperts shape mismatch 边界：新增 `test_refuse_split_stacked_experts_keys_inconsistent_down_shape_leaves_keys_untouched`，确认 independent-A down rank 不一致时不融合。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`13 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_lora_network_construction.py tests/test_factory_metadata_flow.py`：`34 passed`。

仍不能对外说：

- 不能说 malformed shape 会被自动修复；当前保护的是“不融合，保留原 key”。

### EXT 第七组低风险扩展阶段

一句话：第七组 EXT 继续把类型检查试点扩大到更小的纯 helper 文件，但仍避开高依赖 loader。

阶段完成：

- `E09` type-check 候选审计：确认 `library/config/normalize.py` 和空 `library/config/__init__.py` 依赖极轻，适合纳入默认门禁。
- `E10` 默认 type-check 小步加宽：默认门禁纳入 `library/config/__init__.py` 和 `library/config/normalize.py`；仍不纳入 `library/config/loader.py` 或 `library/config/cli_args.py`。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check library/config/normalize.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check library/config/__init__.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 10 个小文件。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile library/config/__init__.py library/config/normalize.py scripts/tasks/utilities.py`：通过。

仍不能对外说：

- 不能说 `library/config` 整目录已纳入默认 type-check；`loader.py` / `cli_args.py` 仍未纳入。

### EXT 第八组低风险扩展阶段

一句话：第八组 EXT 在单文件都通过后，把 type-check 试点从零散 config 文件收口成 `library/config` 小目录。

阶段完成：

- `E09` 重依赖候选审计：单独验证 `library/config/loader.py`、`library/config/cli_args.py` 和整个 `library/config` 目录，均为 `0 errors, 0 warnings, 0 informations`。
- `E10` 默认 type-check 小目录化：`TYPE_CHECK_TARGETS` 从多个 `library/config/*.py` 文件收口为 `library/config` 目录；仍只扩大一个小目录，不做全仓 type-check。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check library/config/loader.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check library/config/cli_args.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check library/config`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/paths.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py library/config/__init__.py library/config/io.py library/config/normalize.py library/config/provenance.py library/config/schema.py library/config/loader.py library/config/cli_args.py`：通过。

当前可对外说：

- 可以说默认 type-check 已覆盖 `library/config` 小目录。

仍不能对外说：

- 不能说已经建立全仓 type-check；当前仍是小目录 / 小文件试点。

### EXT 第九组低风险扩展阶段

一句话：第九组 EXT 把本轮新增的 `file_groups.py -> common.py` 同步逻辑再收成一个私有小 helper。

阶段完成：

- `E11` common 复用小收口：`web/services/config/file_groups.py` 新增私有 `_sync_common_paths()`，统一同步 `common.py` 的 `ROOT` / `CONFIGS_DIR`，避免 `_load` / `_safe_resolve` / `_display_path` 三处重复赋值。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_group_common_path_helpers_forward_to_common_module or file_group or output_run"`：`25 passed, 135 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile web/services/config/file_groups.py tests/test_web_config_service.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check networks/lora_anima/loading.py scripts/tasks/utilities.py web/services/config/file_groups.py tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py tests/test_web_config_service.py`：`All checks passed!`。

仍不能对外说：

- 不能说 Web config 公共路径逻辑已经全量统一；本轮只收口 `file_groups.py` 的三处 helper。

### EXT 第十组低风险扩展阶段

一句话：第十组 EXT 继续钉住 standard LoRA 保存分支的 metadata/hash 边界。

阶段完成：

- `E01` standard save 空 metadata 边界：新增 `test_save_network_weights_standard_empty_metadata_still_adds_hashes`，确认标准 `.safetensors` 保存传入 `metadata={}` 时仍写入 `sshs_model_hash` / `sshs_legacy_hash`，且不额外写 `ss_network_spec`。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py`：`6 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_network_construction.py -k "metadata or save_weights or fallback or standard"`：`7 passed, 10 deselected`。

仍不能对外说：

- 不能说 standard LoRA 保存会写 `ss_network_spec`；空 metadata 下仍不写该字段。

### EXT 第十一组低风险扩展阶段

一句话：第十一组 EXT 把本轮实际触碰或审计过的 Web config split modules 纳入默认 type-check 试点。

阶段完成：

- `E10` Web config type-check 候选验证：单独验证 `web/services/config/file_groups.py`、`output_runs.py`、`sample_prompts.py`，均为 `0 errors, 0 warnings, 0 informations`。
- `E10` 默认 type-check 小步加宽：默认门禁纳入 `web/services/config/file_groups.py`、`output_runs.py`、`sample_prompts.py`；加上已有的 `common.py` / `paths.py`，当前覆盖 5 个 Web config split/helper 模块，但仍不纳入整个 `web/services/config/`。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/file_groups.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/output_runs.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/sample_prompts.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/file_groups.py web/services/config/output_runs.py web/services/config/paths.py web/services/config/sample_prompts.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py web/services/config/file_groups.py web/services/config/output_runs.py web/services/config/sample_prompts.py`：通过。

仍不能对外说：

- 不能说整个 `web/services/config/` 已纳入默认 type-check；当前只覆盖 5 个 split/helper 模块。

### EXT 第十二组低风险扩展阶段

一句话：第十二组 EXT 给默认 type-check 试点范围补结构测试，防止后续误扩成全仓。

阶段完成：

- `E10` type-check 试点清单保护：新增 `tests/test_type_check_targets.py`，固定当前默认 pyright 目标为 `library/config`、两个 config 脚本、`scripts/tasks/utilities.py`、5 个 Web config split/helper 模块，并明确禁止默认目标为 `.`、`tests` 或整个 `web/services/config`。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`1 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_type_check_targets.py scripts/tasks/utilities.py`：通过。

仍不能对外说：

- 不能说 type-check 目标不会再变；只是现在有测试保护，变更需要显式更新测试。

### EXT config 目录验证

一句话：`library/config` 纳入默认 type-check 后，补跑对应 config 单测，确认静态门禁和现有行为一致。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_config.py tests/test_config_normalize.py tests/test_config_provenance.py tests/test_config_compat.py tests/test_config_explain.py`：`41 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`1 passed`。

仍不能对外说：

- 不能说训练配置全路径都做了集成训练验证；这里是 config 单测和静态检查。

### EXT 第十三组低风险扩展阶段

一句话：第十三组 EXT 给 `tasks.py type-check` 命令壳补轻量行为测试，不真正执行 pyright 子进程。

阶段完成：

- `E10` type-check 命令行为保护：`tests/test_type_check_targets.py` 新增 `cmd_type_check` 默认目标测试，确认无显式参数时使用 `TYPE_CHECK_TARGETS`。
- `E10` 显式目标分隔符保护：新增 `--` 分隔符测试，确认 `python tasks.py type-check -- library/config` 会把显式路径传给 pyright。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`3 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_type_check_targets.py scripts/tasks/utilities.py`：`All checks passed!`。

仍不能对外说：

- 不能说 `cmd_type_check` 测试启动了真实 pyright 子进程；命令壳测试用 monkeypatch 截获 `run()`。

### EXT 第十四组低风险扩展阶段

一句话：第十四组 EXT 同步更新 type-check 用户可见文案，避免仍说它只检查 config scripts。

阶段完成：

- `E10` type-check 文案同步：`scripts/tasks/utilities.py::cmd_type_check` docstring 从 “typed config-script pilot” 更新为 “config/WebUI pilot surface”。
- `E10` tasks help 同步：`tasks.py` 中 `type-check` 命令说明更新为通用 “pyright pilot gate”。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tasks.py scripts/tasks/utilities.py tests/test_type_check_targets.py`：通过。
- `timeout 60 .venv/bin/python tasks.py --help | rg -n "type-check|pyright pilot"`：确认 help 输出 `type-check  Run the configured pyright pilot gate.`。

仍不能对外说：

- 不能说命令默认覆盖全仓；文案只是去掉过窄的 “config scripts” 表述。

### EXT 第十五组低风险扩展阶段

一句话：第十五组 EXT 补齐 `cmd_type_check` 的缺依赖错误路径测试。

阶段完成：

- `E10` pyright 缺失提示保护：`tests/test_type_check_targets.py` 新增 `test_cmd_type_check_exits_when_pyright_is_missing`，确认 pyright 未安装时命令以 `SystemExit(2)` 退出并输出明确提示。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`4 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_type_check_targets.py scripts/tasks/utilities.py`：`All checks passed!`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。

仍不能对外说：

- 不能说本机缺 pyright；本机当前 pyright 可用，这只是命令错误路径保护。

### EXT 第十六组低风险扩展阶段

一句话：第十六组 EXT 给 Web config 路由层补两个只读/越界保护测试，避免只测 service 层。

阶段完成：

- `E12` sample prompts 路由越界保护：新增 `test_sample_prompts_route_rejects_prompt_file_outside_configs`，确认 `/api/config/sample-prompts?file=../outside.txt` 返回 400，不落到真实文件读写。
- `E12` raw patch preview 路由只预览保护：新增 `test_raw_patch_preview_route_does_not_write_config_file`，确认 `/api/config/raw/patch-preview` 返回预览内容但不写回原 TOML。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "sample_prompts_route_rejects_prompt_file_outside_configs or raw_patch_preview_route_does_not_write_config_file"`：`2 passed, 160 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_web_config_service.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py`：`All checks passed!`。

仍不能对外说：

- 不能说 `tests/test_web_config_service.py` 全文件已经在 60 秒内完整通过；这里只是新增路由保护的窄验证。

### EXT 第十七组低风险扩展阶段

一句话：第十七组 EXT 继续给 LoRA 加载时 q/k/v 回灌补隐性不变量测试。

阶段完成：

- `E02` pre-fused round-trip rank 保护：新增 `test_refuse_unfused_attn_lora_keys_prefused_roundtrip_keeps_rank`，确认 q/k/v 拆分自同一个 fused down 且 alpha 相同时，不走 block-diagonal 膨胀路径。
- `E02` split DoRA magnitude 回灌保护：新增 `test_refuse_unfused_attn_lora_keys_fuses_split_dora_magnitudes`，确认 `.dora_scale` / `.dora_magnitude` / `.magnitude` 都回收到 fused `.magnitude`，并清掉 split 残留 key。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "prefused_roundtrip_keeps_rank or fuses_split_dora_magnitudes"`：`2 passed, 13 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`15 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_lora_loading_keys.py`：通过。

仍不能对外说：

- 不能说改了 LoRA checkpoint key 或加载格式；本阶段只补 characterization tests，生产代码未因这两例改变。

### EXT 第十八组低风险扩展阶段

一句话：第十八组 EXT 补齐 `type-check` 命令壳对显式 pyright flags 的转发保护。

阶段完成：

- `E10` 显式 pyright flags 保护：`tests/test_type_check_targets.py` 新增 `test_cmd_type_check_preserves_explicit_pyright_flags`，确认 `python tasks.py type-check -- --warnings web/services/config/common.py` 不混入默认目标。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`5 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_type_check_targets.py scripts/tasks/utilities.py`：`All checks passed!`。

仍不能对外说：

- 不能说这个测试执行了真实 pyright；它仍然是 monkeypatch 截获 `run()` 的命令壳测试。

### EXT 第十九组低风险扩展阶段

一句话：第十九组 EXT 把一个更轻的 Web config 静态元数据模块纳入默认 type-check 试点。

阶段完成：

- `E10` metadata 候选验证：先单独验证 `web/services/config/metadata.py`，再模拟加入默认目标组合，均为 `0 errors, 0 warnings, 0 informations`。
- `E10` 默认 type-check 小步加宽：`TYPE_CHECK_TARGETS` 新增 `web/services/config/metadata.py`，并同步更新 `tests/test_type_check_targets.py` 固定清单。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/metadata.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/file_groups.py web/services/config/output_runs.py web/services/config/paths.py web/services/config/sample_prompts.py web/services/config/metadata.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/file_groups.py web/services/config/metadata.py web/services/config/output_runs.py web/services/config/paths.py web/services/config/sample_prompts.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`5 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py tests/test_type_check_targets.py web/services/config/metadata.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check scripts/tasks/utilities.py tests/test_type_check_targets.py web/services/config/metadata.py`：`All checks passed!`。

仍不能对外说：

- 不能说整个 `web/services/config/` 已纳入默认 type-check；当前只是 6 个 split/helper/metadata 模块加 `library/config` 小目录。

### EXT 第二十组低风险扩展阶段

一句话：第二十组 EXT 给 `metadata.py` 补直接导入测试，和默认 type-check 覆盖形成配套。

阶段完成：

- `E10` metadata 直接导入保护：新增 `test_metadata_module_imports_without_facade_cycle`，确认 `web.services.config.metadata` 可单独导入，并且不会拉起 `web.services.config_service` 或 `_legacy`。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_module_imports_without_facade_cycle or config_metadata_exports"`：`2 passed, 161 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py`：`All checks passed!`。

仍不能对外说：

- 不能说所有 Web config split modules 都具备直接导入测试；本阶段只补 `metadata.py`。

### EXT 第二十一组低风险扩展阶段

一句话：第二十一组 EXT 把训练步数估算模块作为下一个小文件纳入默认 type-check 试点。

阶段完成：

- `E10` estimation 候选验证：单独验证 `web/services/config/estimation.py`，并复跑已有直接导入测试。
- `E10` 默认 type-check 小步加宽：`TYPE_CHECK_TARGETS` 新增 `web/services/config/estimation.py`，并同步更新 `tests/test_type_check_targets.py` 固定清单。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/estimation.py`：`0 errors, 0 warnings, 0 informations`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "estimation_module_imports"`：`1 passed, 162 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/estimation.py web/services/config/file_groups.py web/services/config/metadata.py web/services/config/output_runs.py web/services/config/paths.py web/services/config/sample_prompts.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`5 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py tests/test_type_check_targets.py web/services/config/estimation.py`：通过。

仍不能对外说：

- 不能说整个 `web/services/config/` 已纳入默认 type-check；本阶段仍是单文件扩围。

### EXT 第二十二组低风险扩展阶段

一句话：第二十二组 EXT 继续保护 Hydra split q/k/v 回灌时的可选路由辅助权重。

阶段完成：

- `E02` Hydra sigma/inv_scale 回灌保护：新增 `test_refuse_split_hydra_keys_rehomes_sigma_mlp_and_inv_scale`，确认 split q/k/v 的 `inv_scale` 和 `sigma_mlp.*` 从第一份组件回收到 fused `qkv_proj` 前缀，并清掉 split 残留。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "sigma_mlp_and_inv_scale or hydra"`：`4 passed, 12 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`16 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。

仍不能对外说：

- 不能说旧 legacy `sigma_mlp.*` checkpoint 可继续加载；这里只保护 split save/load 回灌路径，后续 factory 仍会拒绝真正旧格式。

### EXT 第二十三组低风险扩展阶段

一句话：第二十三组 EXT 修复 `independent_A` checkpoint 通过 metadata 加载时专家数被默认成 4 的真实小 bug。

问题确认：

- 用 `_stacked_experts_state_dict()` 加 `metadata={"ss_use_moe_style": "independent_A", "ss_route_per_layer": "False", "ss_router_source": "fei", "ss_fei_feature_dim": "2"}` 构建网络时，修复前 `cfg.num_experts` 和 `GlobalRouter.num_experts` 会变成默认 `4`，而 checkpoint 扫描出的专家数是 `3`。

阶段完成：

- `E02` independent-A 专家数恢复修复：`LoRANetworkCfg.from_weights()` 在 `is_stacked_experts=True` 时也使用 checkpoint 扫描出的 `hydra_num_experts`，不再只给 Hydra/OrthoHydra 用。
- `E02` independent-A 成功流测试：新增 `test_independent_a_metadata_kwarg_lands_three_axes_and_expert_count`，确认三轴 metadata、FEI 维度、专家数和 `GlobalRouter` 专家数都保留。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py -k independent_a_metadata_kwarg_lands_three_axes_and_expert_count`：`1 passed, 5 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py`：`6 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile networks/lora_anima/config.py tests/test_factory_metadata_flow.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check networks/lora_anima/config.py tests/test_factory_metadata_flow.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py tests/test_factory_metadata_flow.py tests/test_lora_loading_keys.py`：`43 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check networks/lora_anima/config.py`：`0 errors, 0 warnings, 0 informations`。

仍不能对外说：

- 不能说跑过真实 FeRA 训练或真实 checkpoint 推理；这是小 fixture 级加载恢复测试。

### EXT 第二十四组低风险扩展阶段

一句话：第二十四组 EXT 把 `merge.py` 作为单文件加入默认 type-check，并补一个不拉 facade 的直接行为断言。

阶段完成：

- `E10` merge 候选验证：只把 `web/services/config/merge.py` 加入 `TYPE_CHECK_TARGETS`，不把整个 `web/services/config/` 目录加入默认门禁。
- `E10` merge 直接行为保护：扩展 `test_merge_module_imports_without_facade_cycle`，确认 `merge.list_methods.__wrapped__()` 可直接返回包含 `lora` 的方法列表，且不拉起 facade / legacy。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/estimation.py web/services/config/file_groups.py web/services/config/merge.py web/services/config/metadata.py web/services/config/output_runs.py web/services/config/paths.py web/services/config/sample_prompts.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`5 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py::test_merge_module_imports_without_facade_cycle`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py tests/test_type_check_targets.py tests/test_web_config_service.py web/services/config/merge.py`：通过。

仍不能对外说：

- 不能说默认 type-check 已覆盖所有 Web config 拆分模块；当前仍是小文件白名单。

### EXT 当前硬门槛盘点

一句话：阶段数已远超最低要求，但耗时硬门槛仍未达到，所以继续 EXT，不进入 R4 提交推送。

当前事实：

- `goal.timeUsedSeconds = 3321`，仍小于 `7200`。
- `git diff --check`：tracked diff 通过；未跟踪文件仍需在最终显式 stage 后纳入提交前检查。
- `git status --short --branch`：仍在 `main...webui/main`，存在当前目标相关未提交改动和未跟踪目标书 / type-check 测试文件。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成；R4 必须等 `goal.timeUsedSeconds >= 7200` 后执行。

### EXT 第二十五组低风险扩展阶段

一句话：第二十五组 EXT 补普通 q/k/v per-component split 的 block-diag 数学语义测试。

阶段完成：

- `E02` split block-diag alpha 保护：新增 `test_refuse_unfused_attn_lora_keys_split_components_block_diag_and_alpha_scales`，确认普通 split q/k/v 的 `down` 拼接后 rank 变 `n*r`，`up` 只填本组件 block，off-block 为 0，并按各自 `alpha/r` 缩放。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "block_diag_and_alpha_scales or prefused_roundtrip_keeps_rank"`：`2 passed, 15 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`17 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_lora_loading_keys.py`：通过。

仍不能对外说：

- 不能说改了加载数学；这里只是把现有 block-diag 行为钉成测试。

### EXT 第二十六组低风险扩展阶段

一句话：第二十六组 EXT 把 raw TOML helper 单文件纳入默认 type-check，但不宣称 raw 行为全量通过。

阶段完成：

- `E10` raw_files 候选验证：单独验证 `web/services/config/raw_files.py` 为 `0 errors, 0 warnings, 0 informations`。
- `E10` 默认 type-check 小步加宽：`TYPE_CHECK_TARGETS` 新增 `web/services/config/raw_files.py`，并同步更新 `tests/test_type_check_targets.py` 固定清单。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/raw_files.py`：`0 errors, 0 warnings, 0 informations`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "raw_files_module_imports or raw_patch_preview_route_does_not_write_config_file or raw_patch_ignores_dataset_picker_ui_field"`：`3 passed, 160 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/estimation.py web/services/config/file_groups.py web/services/config/merge.py web/services/config/metadata.py web/services/config/output_runs.py web/services/config/paths.py web/services/config/raw_files.py web/services/config/sample_prompts.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`5 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py tests/test_type_check_targets.py web/services/config/raw_files.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check scripts/tasks/utilities.py tests/test_type_check_targets.py web/services/config/raw_files.py`：`All checks passed!`。

仍不能对外说：

- 不能说 raw-file 全部宽筛选都通过；只记录本轮跑过的 3 个相关窄测试。

### EXT 第二十七组低风险扩展阶段

一句话：第二十七组 EXT 给非 safetensors 标准保存分支补直接测试。

阶段完成：

- `E01` torch.save 分支保护：新增 `test_save_network_weights_standard_torch_save_branch_writes_loadable_pt`，确认标准 `.pt` 保存走 `torch.save`，`torch.load(weights_only=True)` 可读，并且 dtype 转换生效。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py -k torch_save_branch`：`1 passed, 6 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py`：`7 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_save_pipeline.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_lora_save_pipeline.py`：通过。

仍不能对外说：

- 不能说 `.pt` 分支写入 safetensors metadata/hash；该分支不依赖 safetensors metadata。

### EXT 第二十八组低风险扩展阶段

一句话：第二十八组 EXT 把 preflight 模块作为单文件加入默认 type-check，并用已有 preflight 切片验证。

阶段完成：

- `E10` preflight 候选验证：单独验证 `web/services/config/preflight.py` 为 `0 errors, 0 warnings, 0 informations`，并跑 preflight 相关窄测试。
- `E10` 默认 type-check 小步加宽：`TYPE_CHECK_TARGETS` 新增 `web/services/config/preflight.py`，并同步更新 `tests/test_type_check_targets.py` 固定清单。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check web/services/config/preflight.py`：`0 errors, 0 warnings, 0 informations`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "preflight_module_imports or preflight"`：`31 passed, 132 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/estimation.py web/services/config/file_groups.py web/services/config/merge.py web/services/config/metadata.py web/services/config/output_runs.py web/services/config/paths.py web/services/config/preflight.py web/services/config/raw_files.py web/services/config/sample_prompts.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`5 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/utilities.py tests/test_type_check_targets.py web/services/config/preflight.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check scripts/tasks/utilities.py tests/test_type_check_targets.py web/services/config/preflight.py`：`All checks passed!`。

仍不能对外说：

- 不能说 preflight 做过真实训练环境检查；这里只是 Web config service 层测试和静态检查。

### EXT 当前硬门槛盘点 2

一句话：本轮继续有真实测试和小修产物，但耗时仍未达标，继续扩展阶段。

当前事实：

- `goal.timeUsedSeconds = 3663`，仍小于 `7200`。
- `git diff --check`：tracked diff 通过；未跟踪文件仍需在最终显式 stage 后纳入提交前检查。
- `git status --short --branch`：仍在 `main...webui/main`，存在当前目标相关未提交改动和未跟踪目标书 / type-check 测试文件。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

---

## 🔚 16. 2026-07-06 最新入口与下一阶段目标书

一句话：`20260705` 的 sustained 长跑目标已经完成，当前新的活跃入口是 `20260706` 跨子系统长跑目标书。

当前最终事实：

- `project_cleanup_sustained_goal_20260705.md` 已完成归档，最终提交为 `bd591b83 test: extend sustained cleanup coverage`。
- 用户验收记录显示该目标 `goal.timeUsedSeconds = 7889`，约 `2 小时 11 分钟`，并已推送到 `webui/main`。
- 历史 EXT 小节中的“未满 7200 秒 / 不能 commit / 不能 push”是执行中间快照，不再代表当前状态。
- 当前活跃目标书是 `docs/findings/project_cleanup_sustained_goal_20260706.md`。

本轮文档维护完成：

- 新增 `project_cleanup_sustained_goal_20260706.md`，把硬门槛提高为 `10800` 秒、`20` 个阶段、`5` 个推进轮和至少 `4` 个子系统覆盖。
- 旧 `project_cleanup_sustained_goal_20260705.md` 改为已完成归档。
- `project_cleanup_long_running_goal_20260705.md` 和 `project_cleanup_next_stage_goal_20260705.md` 的后续入口改为 `20260706` 新目标书。
- 修正 blocked/complete 文案漏洞：阻塞只能标记 `blocked`，不能作为提前 `complete` 的例外。
- 修正 checkpoint 旧尾部状态冲突：R4 正式收尾和 `bd591b83` 推送事实优先于执行中间快照。

下一轮可直接复制：

```text
请按 docs/findings/project_cleanup_sustained_goal_20260706.md 执行跨子系统强制长跑项目清理目标。
```

不能夸大的边界：

- 不能说全仓技术债已清完；这是下一阶段长跑任务书，不是最终清债证明。
- 不能说 `20260706` 目标已经执行；本节只记录目标书创建和入口切换。
- 不能说旧目标文档被删除；它们只是归档并改为历史入口。

### R4 正式收尾验证

一句话：R4 已满足长跑目标的时间、阶段、轮次和验证门槛，可以进入提交推送。

硬门槛状态：

- `goal.timeUsedSeconds = 7758`，已超过 `7200` 秒最低耗时门槛。
- 已完成阶段数远超 `12` 个，推进轮数远超 `3` 轮。
- 最后一轮已执行总验证；`tests/test_web_config_service.py` 全文件在 60 秒内未跑完，因此按目标书要求拆成更窄命令验证。

R4 最终验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py tests/test_factory_metadata_flow.py tests/test_lora_loading_keys.py tests/test_lora_save_pipeline.py tests/test_lora_network_construction.py tests/test_tasks_runner.py tests/test_type_check_targets.py`：`96 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "raw_patch or raw_put_route or raw_save_as_route or sample_prompts"`：`24 passed, 145 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "module_imports_without_facade_cycle"`：`8 passed, 161 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "facade_sync or direct_path_helpers or glob_uses_synced_common_paths or common_private_wrappers or common_config_helpers or common_path_helpers or legacy_common_private_helpers or raw_file_helpers or legacy_raw_file_shim"`：`14 passed, 155 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_group"`：`17 passed, 152 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python -m ruff check networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py tasks.py web/services/config/file_groups.py tests/test_factory_metadata_flow.py tests/test_lora_loading_keys.py tests/test_lora_network_construction.py tests/test_lora_save_pipeline.py tests/test_web_config_service.py tests/test_tasks_runner.py tests/test_type_check_targets.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py tasks.py web/services/config/file_groups.py tests/test_factory_metadata_flow.py tests/test_lora_loading_keys.py tests/test_lora_network_construction.py tests/test_lora_save_pipeline.py tests/test_web_config_service.py tests/test_tasks_runner.py tests/test_type_check_targets.py`：通过。
- `git diff --check`：通过。

R4 修复补记：

- 修复 `tests/test_web_config_service.py` 的 raw file 路径隔离 helper：测试现在同时同步 `config_service`、`legacy_config` 和 `raw_files` 的路径状态，避免 raw patch 写回项目默认 `configs/` 而不是测试临时目录。

当前允许收口：

- 可以显式 stage 本目标相关文件。
- 可以 commit。
- 可以 `git push webui main:main`。
- 推送成功后可以 `update_goal complete`。

---

## 🔚 尾部最新状态索引

一句话：因为本文件前面有补写顺序错位，后续接手时先看这里判断当前准入状态。

当前最新事实：

- `project_cleanup_sustained_goal_20260705.md` 已完成归档，并已提交推送到 `webui/main`。
- 最终提交：`bd591b83 test: extend sustained cleanup coverage`。
- 用户验收记录显示 `goal.timeUsedSeconds = 7889`，约 `2 小时 11 分钟`，已超过该目标书要求的 `7200` 秒。
- 该目标已完成正式 R4、显式 stage、commit、push 和 `update_goal complete`。
- 本节后面的旧 EXT 硬门槛盘点是历史补写记录，不能覆盖这里的已完成事实。

下一步入口：

- 下一轮请执行 `docs/findings/project_cleanup_sustained_goal_20260706.md`。
- 旧 `20260705` 三份目标书都只作为历史记录保留，不要重复执行。
- 新目标要求至少 `10800` 秒、`20` 个阶段、`5` 个推进轮、覆盖至少 `4` 个子系统。

### 尾部预审补充

一句话：测试代码里出现的用户数据样例路径均通过临时目录或字符串 fixture 使用，不表示触碰真实用户数据。

补充事实：

- `tests/test_web_config_service.py` 里新增 route/file_groups 用例通过 `_write_minimal_config_tree(tmp_path)`、`_patch_config_service_paths(...)` 或子进程临时目录设置路径。
- 本轮禁入路径扫描针对实际 git diff / untracked 文件结果为 `forbidden=[]`。
- 后续正式 stage 仍必须按文件清单逐个加入，不能用 `git add -A`。

### EXT 第六十八组低风险扩展阶段

一句话：第六十八组 EXT 做最终 R4 前的核心组合验证，但因耗时未达标仍只计 EXT。

阶段完成：

- `R4-prep` 核心组合验证：LoRA / config / tasks/type-check 组合在 60 秒内通过。
- `R4-prep` Web config 拆分验证：仍按 direct-import、file_groups/common、route 三组拆跑并通过。
- `R4-prep` type-check 和 diff check：均通过。

本轮已验证：

- `git diff --check`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py tests/test_network_cfg.py tests/test_config.py tests/test_config_normalize.py tests/test_config_provenance.py tests/test_config_compat.py tests/test_config_explain.py tests/test_tasks_runner.py tests/test_type_check_targets.py`：`137 passed`。
- Web config direct-import 拆分：`8 passed, 161 deselected`。
- Web config file_groups/common 拆分：`5 passed, 164 deselected`。
- Web config route 拆分：`6 passed, 163 deselected`。

仍不能对外说：

- 不能说这是正式 R4 总验证；正式 R4 必须在 `goal.timeUsedSeconds >= 7200` 后再确认并记录。

### EXT 当前硬门槛盘点 18

一句话：最终前验证已基本完成，但当前还差约数分钟才到 7200 秒。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6830`，仍小于 `7200`。
- 当前仍不能提交、推送或 complete。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第六十七组低风险扩展阶段

一句话：第六十七组 EXT 做禁入路径扫描和最终 stage 文件清单预审。

阶段完成：

- `R4-prep` 空白检查：`git diff --check` 通过。
- `R4-prep` 禁入路径扫描：tracked diff + untracked 文件共 16 个，未命中禁入用户数据目录。
- `R4-prep` stage 清单预审：最终需要显式 stage 的文件包括 13 个 tracked 改动文件和 3 个未跟踪目标相关文件。

本轮已验证：

- `git diff --check`：通过。
- 禁入路径扫描脚本输出：`files= 16`，`forbidden= []`。
- `git diff --name-only && git ls-files --others --exclude-standard`：确认 16 个目标相关文件清单。

仍不能对外说：

- 不能说已经 stage；本阶段只是 stage 前清单和禁入路径预审。

### EXT 当前硬门槛盘点 17

一句话：最终 stage 清单已准备好，但耗时仍没到 7200 秒。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6724`，仍小于 `7200`。
- 当前 checkpoint 本身较长，最终提交前仍需跑 `git diff --check` 和 R4 总验证。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第六十六组低风险扩展阶段

一句话：第六十六组 EXT 做远端状态预检，为最终 R4 做准备但不提交。

阶段完成：

- `R4-prep` HEAD 预检：确认当前本地 HEAD 仍为 `7c5c277c`，并且本地 `main` 与 `webui/main` 在提交层面 `0 0` 对齐。
- `R4-prep` fetch 预检：执行 `git fetch webui --prune`，远端无新变化。
- `R4-prep` 工作区预检：确认当前仍只有本目标相关未提交/未跟踪改动，尚未 stage。

本轮已验证：

- `git log -1 --oneline --decorate`：`7c5c277c (HEAD -> main, webui/main, webui/HEAD) test: cover next cleanup boundaries`。
- `git rev-list --left-right --count HEAD...webui/main`：`0 0`。
- `git fetch webui --prune`：通过。
- `git diff --check`：通过。
- `git status --short --branch`：仍有本轮改动和 3 个未跟踪目标相关文件。

仍不能对外说：

- 不能说已经提交或推送；本阶段只是 R4 前只读/网络预检。

### EXT 当前硬门槛盘点 16

一句话：远端预检已完成，但实时耗时仍小于 7200 秒。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6677`，仍小于 `7200`。
- 本地 HEAD 与 `webui/main` 提交层面仍 `0 0` 对齐，当前改动尚未提交。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第六十四组低风险扩展阶段

一句话：第六十四组 EXT 做接近最终验证的相关测试组合，并记录 Web config 超时拆分。

阶段完成：

- `E02/E09/E10` 相关组合验证：LoRA/network、config、tasks/type-check 分别通过。
- `E09` Web config 长命令拆分：一次合并 `-k` 在 60 秒内未完成，按目标书要求拆成 direct-import、file_groups/common、route 三组窄命令，并逐组通过。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py tests/test_network_cfg.py`：`79 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_config.py tests/test_config_normalize.py tests/test_config_provenance.py tests/test_config_compat.py tests/test_config_explain.py`：`41 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`17 passed`。
- Web config 合并 `-k` 命令：超过 60 秒被 timeout，未计为通过。
- Web config direct-import 拆分：`8 passed, 161 deselected`。
- Web config file_groups/common 拆分：`5 passed, 164 deselected`。
- Web config route 拆分：`6 passed, 163 deselected`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说 `tests/test_web_config_service.py` 全文件通过；本阶段明确是拆分后的相关切片通过。

### EXT 第六十五组低风险扩展阶段

一句话：第六十五组 EXT 对本轮 touched 文件做统一 ruff 和 py_compile。

阶段完成：

- `E00/R4-prep` ruff 统一验证：覆盖本轮 touched 的生产代码和测试文件。
- `E00/R4-prep` py_compile 统一验证：覆盖同一组 Python 文件。

本轮已验证：

- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py tests/test_tasks_runner.py tests/test_type_check_targets.py tests/test_web_config_service.py networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py tasks.py web/services/config/file_groups.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py tests/test_tasks_runner.py tests/test_type_check_targets.py tests/test_web_config_service.py networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py tasks.py web/services/config/file_groups.py`：通过。

仍不能对外说：

- 不能说已经完成 R4；这些是 EXT 阶段的最终前静态验证。

### EXT 当前硬门槛盘点 15

一句话：最终前验证越来越完整，但耗时仍小于 7200 秒。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6625`，仍小于 `7200`。
- 仍不能提交、推送或 complete。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第六十三组低风险扩展阶段

一句话：第六十三组 EXT 做最终 stage 前的只读文件范围预审，但不执行 stage。

阶段完成：

- `E00/R4-prep` 文件范围预审：核对 tracked diff 与未跟踪文件，确认未跟踪文件为当前目标书和两个新测试文件。
- `E00/R4-prep` 禁入目录预审：当前 diff 文件列表未包含 `.venv/`、`models/`、`output/`、`logs/`、`configs/imported/`、`configs/web-training-history/`、`configs/web-training-queue/` 等禁入用户数据目录。
- `E00/R4-prep` 文档空白检查：对本轮 findings 文档相关路径跑 `git diff --check`。

本轮已验证：

- `git diff --name-only && git ls-files --others --exclude-standard`：列出 13 个 tracked 改动文件和 3 个未跟踪目标相关文件。
- `git status --short --branch`：仍在 `main...webui/main`，尚未 stage。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md docs/findings/project_cleanup_long_running_goal_20260705.md docs/findings/project_cleanup_next_stage_goal_20260705.md docs/findings/project_cleanup_sustained_goal_20260705.md`：通过。

仍不能对外说：

- 不能说已经显式 stage；本阶段只是预审，最终 stage 必须等 `goal.timeUsedSeconds >= 7200` 和 R4 总验证后再做。

### EXT 当前硬门槛盘点 14

一句话：文件范围预审已完成，但耗时硬门槛仍未满足。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6467`，仍小于 `7200`。
- 仍未执行 R4 总验证、显式 stage、commit 和 push。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第六十二组低风险扩展阶段

一句话：第六十二组 EXT 对 tasks/type-check 命令壳和 pilot gate 做组合验证。

阶段完成：

- `E10` tasks/type-check pytest 验证：覆盖 `tasks.py` 命令壳、inline env、help、type-check 白名单和显式目标转发。
- `E10` tasks/type-check 静态验证：对 `tasks.py`、`scripts/tasks/utilities.py` 和相关测试跑 ruff / py_compile。
- `E10` pyright pilot gate 验证：跑实际 `tasks.py type-check`，确认当前默认白名单仍为 0 errors。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`17 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_tasks_runner.py tests/test_type_check_targets.py tasks.py scripts/tasks/utilities.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_tasks_runner.py tests/test_type_check_targets.py tasks.py scripts/tasks/utilities.py`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说已经建立全仓 type-check；当前仍是明确白名单 pilot gate。

### EXT 当前硬门槛盘点 13

一句话：type-check 侧验证已经加宽，但仍不能收口。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6428`，仍小于 `7200`。
- `git diff --check` 最近一次通过。
- 仍未执行 R4 总验证、显式 stage、commit 和 push。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第六十一组低风险扩展阶段

一句话：第六十一组 EXT 对本轮 Web config route/file_groups/common 相关切片做加宽验证。

阶段完成：

- `E09` Web config route/file_groups 切片验证：覆盖 common path helper、file_groups direct path / glob、sample prompts route、raw preview/save-as/put/patch route 等本轮 touched 行为。
- `E09` Web config 静态验证：对 `tests/test_web_config_service.py` 和 `web/services/config/file_groups.py` 跑 ruff 与 py_compile。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "common_path_helpers or file_groups_direct_path_helpers or file_groups_glob or sample_prompts_route or sample_prompts_put_route or raw_patch_preview_route or raw_save_as_route or raw_put_route or raw_patch_route"`：`11 passed, 158 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py web/services/config/file_groups.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_web_config_service.py web/services/config/file_groups.py`：通过。
- `git diff --check`：通过。

仍不能对外说：

- 不能说 `tests/test_web_config_service.py` 全文件已在 60 秒内通过；这里仍是按长跑规则拆窄的相关切片。

### EXT 当前硬门槛盘点 12

一句话：Web config 验证继续增加，但耗时硬门槛仍未满足。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6379`，仍小于 `7200`。
- 当前仍不能进入 R4 总验证、提交和推送。
- `git diff --check` 最近一次通过。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第六十组低风险扩展阶段

一句话：第六十组 EXT 对本轮 LoRA save/load/factory 相关测试做文件级回归验证。

阶段完成：

- `E02` LoRA loading 文件级验证：跑完整 `tests/test_lora_loading_keys.py`，覆盖本轮新增的 stack/refuse、DoRA alias、missing/shape mismatch、Chimera dual-pool 等边界。
- `E02` LoRA save 文件级验证：跑完整 `tests/test_lora_save_pipeline.py`，覆盖 Hydra / StackedExperts / Chimera 保存形态和 dtype 读回测试。
- `E02` factory/network construction 文件组合验证：跑 `tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py`，覆盖 metadata flow 和网络构造边界。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`28 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py`：`10 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py`：`20 passed`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说跑过真实训练或真实大 checkpoint；这些仍是单元级 synthetic tensor / small module 测试。

### EXT 当前硬门槛盘点 11

一句话：LoRA 相关验证更完整了，但仍未达到 7200 秒。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6311`，仍小于 `7200`。
- 阶段和验证数量均已满足最低要求，但耗时、R4 总验证、显式 stage、提交和推送仍未满足。
- `git diff --check` 最近一次通过。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第五十七组低风险扩展阶段

一句话：第五十七组 EXT 给 `_stack_lora_ups()` 的 independent-A downs 排序补测试。

阶段完成：

- `E02` independent-A down stack 保护：`tests/test_lora_loading_keys.py` 新增 `test_stack_lora_ups_stacks_sorted_independent_a_downs`，确认保存态 `lora_downs.N.weight` 会按 expert index 排序堆成 `lora_down_weight`，并清掉原 per-expert keys。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "stacks_sorted_independent_a_downs or stack_lora_ups_stacks_sorted_experts"`：`2 passed, 23 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说所有 independent-A 保存态都已真实加载；这是 stack helper 的排序和清理边界测试。

### EXT 第五十八组低风险扩展阶段

一句话：第五十八组 EXT 给普通 split LoRA up/down shape mismatch 的保守跳过行为补测试。

阶段完成：

- `E02` 普通 split up shape mismatch 保护：新增 `test_refuse_unfused_attn_lora_keys_inconsistent_up_shape_leaves_keys_untouched`。
- `E02` 普通 split down shape mismatch 保护：新增 `test_refuse_unfused_attn_lora_keys_inconsistent_down_shape_leaves_keys_untouched`。
- 两个测试都确认形状不一致时不会生成 fused qkv key，也不会删除 split 残留。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "inconsistent_up_shape_leaves or inconsistent_down_shape_leaves or missing_component_leaves"`：`8 passed, 19 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说形状损坏 checkpoint 可以自动修复；本阶段只确认不会半融合。

### EXT 第五十九组低风险扩展阶段

一句话：第五十九组 EXT 给 Chimera dual-pool stack 的乱序 expert 排序补测试。

阶段完成：

- `E02` Chimera dual-pool stack 保护：新增 `test_stack_chimera_lora_ups_stacks_sorted_dual_pools_independently`，确认 `lora_ups_c.N.weight` 和 `lora_ups_f.N.weight` 两个池分别按 expert index 排序堆叠，并清掉原 per-expert keys。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "stack_chimera_lora_ups"`：`2 passed, 26 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说真实 ChimeraHydra checkpoint 全加载路径都覆盖；这是 dual-pool stack helper 的排序测试。

### EXT 当前硬门槛盘点 10

一句话：LoRA helper 边界继续补齐，但实时耗时仍没有达到 7200 秒。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6255`，仍小于 `7200`。
- 最近中宽验证通过：LoRA `54 passed`，tasks/type-check `17 passed`，Web config route/file_groups 切片 `7 passed`，type-check `0 errors`。
- `git diff --check` 最近一次通过。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第五十二组低风险扩展阶段

一句话：第五十二组 EXT 给 `tasks.py` inline env 的等号值和未知命令路径补测试。

阶段完成：

- `E10` inline env value 保护：`tests/test_tasks_runner.py` 新增 `test_tasks_main_preserves_equals_inside_inline_env_value`，确认 `KEY=a=b=c` 只按第一个 `=` 拆分，值里的等号完整保留。
- `E10` unknown command env 保护：新增 `test_tasks_main_unknown_command_does_not_apply_inline_env`，确认未知命令在报错前不会写入尾随 `KEY=value` 到 `os.environ`。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`17 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_tasks_runner.py tests/test_type_check_targets.py`：`All checks passed!`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说所有 shell quoting 场景都覆盖；这里是 Python 层 token 解析测试。

### EXT 第五十三组低风险扩展阶段

一句话：第五十三组 EXT 给 DoRA 旧别名加载重命名 helper 补直接测试。

阶段完成：

- `E02` DoRA alias rename 保护：`tests/test_lora_loading_keys.py` 新增 `test_rename_dora_scale_for_load_maps_export_scale_keys_to_magnitude`，确认 `.dora_scale` 和 `.dora_magnitude` 都会归一到 `.magnitude`，普通 LoRA key 不受影响。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "rename_dora_scale"`：`1 passed, 22 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。

仍不能对外说：

- 不能说 DoRA 所有历史 checkpoint 都完整覆盖；这里锁定的是加载前 key alias 归一行为。

### EXT 第五十四组低风险扩展阶段

一句话：第五十四组 EXT 给普通 split q/k/v LoRA 缺组件时的保守跳过行为补测试。

阶段完成：

- `E02` 普通 split missing component 保护：新增 `test_refuse_unfused_attn_lora_keys_missing_component_leaves_keys_untouched`，确认 q/k/v 缺少一个组件时不会生成半残 fused qkv key，也不会删除已有 split key。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "rename_dora_scale or missing_component_leaves"`：`4 passed, 20 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。

仍不能对外说：

- 不能说损坏的 split LoRA checkpoint 可恢复；本阶段确认的是保守不半融合。

### EXT 第五十五组低风险扩展阶段

一句话：第五十五组 EXT 给 raw PUT 路由的坏 TOML 错误路径补测试。

阶段完成：

- `E09` raw PUT invalid TOML 保护：`tests/test_web_config_service.py` 新增 `test_raw_put_route_rejects_invalid_toml_without_creating_file`，确认 `handle_raw_put()` 遇到坏 TOML 时返回 400、错误包含 `TOML 语法错误`，且目标文件不会被创建。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "raw_put_route_rejects_invalid_toml_without_creating_file or raw_patch_route_rejects_non_object_values"`：`2 passed, 167 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py tests/test_lora_loading_keys.py tests/test_tasks_runner.py`：`All checks passed!`。

仍不能对外说：

- 不能说 raw route 全错误路径都覆盖；这里只补坏 TOML 保存失败路径。

### EXT 第五十六组低风险扩展阶段

一句话：第五十六组 EXT 给 raw PATCH 路由的非对象 values 错误路径补测试。

阶段完成：

- `E09` raw PATCH values 类型保护：新增 `test_raw_patch_route_rejects_non_object_values`，确认 `values` 不是 dict 时返回 400、错误包含 `字段补丁格式不合法`，且原 TOML 文件内容不变。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "raw_put_route_rejects_invalid_toml_without_creating_file or raw_patch_route_rejects_non_object_values"`：`2 passed, 167 deselected`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说 raw PATCH 所有字段语义都覆盖；这里只补 route handler 的错误返回和不落盘保护。

### EXT 当前硬门槛盘点 9

一句话：继续有真实测试产物，但离 7200 秒还有距离。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 6016`，仍小于 `7200`。
- 当前仍处于 EXT 扩展阶段，不能进入 R4。
- `git diff --check` 最近一次通过。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第五十组低风险扩展阶段

一句话：第五十组 EXT 给 file_groups glob 在外置 configs root 下的路径稳定性补测试。

阶段完成：

- `E09` glob 外置配置根保护：`tests/test_web_config_service.py` 新增 `test_file_groups_glob_uses_synced_common_paths_under_external_configs_root`，确认 `_glob_config_files("configs/imported/*.toml")` 在 facade 和 split module 同步到外置 `CONFIGS_DIR` 后仍返回稳定的 `configs/imported/external.toml`，不会退化为绝对路径或裸 `imported/...`。
- `E09` 越界 pattern 保护：同一测试确认 `../*.toml` 这类越界 pattern 返回空列表。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_groups_direct_path_helpers_support_external_configs_root or file_groups_glob_uses_synced_common_paths"`：`2 passed, 165 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py tests/test_lora_save_pipeline.py`：`All checks passed!`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说所有 file group 列表和前端显示路径都全量覆盖；这里只保护 glob helper 的外置配置根路径行为。

### EXT 第五十一组低风险扩展阶段

一句话：第五十一组 EXT 给 ChimeraHydra 保存分支补 content/freq pool tensor 形态保护。

阶段完成：

- `E02` Chimera 保存 tensor 保护：`tests/test_lora_save_pipeline.py` 新增 `test_save_network_weights_chimera_writes_typed_pool_tensors`，读取 `_chimera.safetensors` 中 q/k/v content/freq pool tensors，确认 pool expert index、q/k/v chunk 位置和 `torch.float16` dtype 都正确。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py -k "chimera and typed"`：`1 passed, 9 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py tests/test_lora_save_pipeline.py`：`All checks passed!`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说真实 ChimeraHydra checkpoint 全格式都已覆盖；这是 synthetic state_dict 级保存形态测试。

### EXT 当前硬门槛盘点 8

一句话：阶段继续增加，但实时耗时仍小于 7200 秒，不能进入 R4。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 5800`，仍小于 `7200`。
- 中宽验证最近通过：LoRA `51 passed`，tasks/type-check `15 passed`，Web config 切片 `4 passed`。
- `git diff --check` 最近一次通过。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第四十六组低风险扩展阶段

一句话：第四十六组 EXT 给 StackedExperts 保存分支补 tensor 形态和 dtype 保护。

阶段完成：

- `E02` StackedExperts 保存 tensor 保护：`tests/test_lora_save_pipeline.py` 新增 `test_save_network_weights_stacked_experts_writes_typed_expert_tensors`，读取 `_moe.safetensors` 里的 q/k/v per-expert tensors，确认 expert index、chunk 位置和 `torch.float16` dtype 都正确。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py -k "stacked_experts and typed"`：`1 passed, 8 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_save_pipeline.py`：`All checks passed!`。

仍不能对外说：

- 不能说真实 StackedExperts checkpoint 全格式都已验证；这是 synthetic state_dict 级保存形态测试。

### EXT 第四十七组低风险扩展阶段

一句话：第四十七组 EXT 给 `tasks.py` 子命令帮助的无 docstring 路径补测试。

阶段完成：

- `E10` help fallback 保护：`tests/test_tasks_runner.py` 新增 `test_tasks_main_prints_subcommand_help_without_docstring`，确认子命令函数没有 docstring 时，`python tasks.py fake --help` 仍输出 `(no detailed help available)` 并正常退出。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`15 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_tasks_runner.py tests/test_type_check_targets.py`：`All checks passed!`。

仍不能对外说：

- 不能说所有真实任务命令都跑过；这里只测命令壳帮助输出。

### EXT 第四十八组低风险扩展阶段

一句话：第四十八组 EXT 给 `tasks.py type-check -- ...` 的完整入口分隔符处理补测试。

阶段完成：

- `E10` type-check 分隔符保护：`tests/test_tasks_runner.py` 新增 `test_tasks_main_type_check_strips_separator_before_pyright`，确认 `tasks.main()` 接收 `type-check -- --warnings path` 后，最终 `cmd_type_check()` 调 pyright 前会去掉分隔符，只转发 pyright flag 和目标路径。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`15 passed`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说默认 type-check 已扩大到全仓；这只是命令入口的参数转发保护。

### EXT 第四十九组低风险扩展阶段

一句话：第四十九组 EXT 给 file_groups 复用 common 后的外置 configs root 路径行为补测试。

阶段完成：

- `E09` 外置配置根路径保护：`tests/test_web_config_service.py` 新增 `test_file_groups_direct_path_helpers_support_external_configs_root_without_facade_snapshot`，确认 `file_groups._load/_safe_resolve/_display_path` 在 `CONFIGS_DIR` 不位于项目根时仍能解析 `configs/imported/external.toml`、显示为稳定的 `configs/...` 路径，并拒绝 `../outside.toml`。
- `E09` direct-import 保护：同一测试确认不会拉入 `web.services.config_service` 或 `_legacy` facade。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_groups_direct_path_helpers"`：`2 passed, 164 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py tests/test_tasks_runner.py tests/test_lora_save_pipeline.py`：`All checks passed!`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说整个 Web config 外置配置根流程都已全量覆盖；这里保护的是 file_groups 的 direct helper 路径行为。

### EXT 当前硬门槛盘点 7

一句话：本轮继续新增测试和验证，但实时耗时仍未达 7200 秒。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 5345`，仍小于 `7200`。
- 阶段数量和推进轮数量已满足最低要求，但耗时、R4 总验证、显式 stage、提交和推送仍未满足。
- `git diff --check` 最近一次通过。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第四十二组低风险扩展阶段

一句话：第四十二组 EXT 给 StackedExperts split q/k/v 的 inconsistent up shape 补保守跳过测试。

阶段完成：

- `E02` StackedExperts inconsistent up 保护：`tests/test_lora_loading_keys.py` 新增 `test_refuse_split_stacked_experts_keys_inconsistent_up_shape_leaves_keys_untouched`，确认 q/k/v 组件的 `lora_up_weight` rank 不一致时不会生成 fused qkv key，原 split key 保持给后续加载错误路径处理。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "stacked_experts_keys_inconsistent"`：`2 passed, 19 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。

仍不能对外说：

- 不能说 StackedExperts 所有损坏 checkpoint 都可恢复；本阶段只确认形状不一致时保守不回灌。

### EXT 第四十三组低风险扩展阶段

一句话：第四十三组 EXT 补保存态 StackedExperts stack + refuse 的最小加载链路测试。

阶段完成：

- `E02` stack + refuse 链路保护：新增 `test_stack_lora_ups_then_refuse_split_stacked_experts_keys_refuses_qkv`，用保存态 `lora_ups.N.weight` / `lora_downs.N.weight` 先跑 `_stack_lora_ups()`，再跑 `_refuse_split_stacked_experts_keys()`，确认最终生成 fused `qkv_proj.lora_up_weight` / `lora_down_weight`。
- `E02` 专家顺序保护：同一测试断言 expert index 和 q/k/v chunk 的代表性值，避免只看 key 存在。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "stack_lora_ups_then_refuse or stacked_experts_keys_inconsistent"`：`3 passed, 19 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py tests/test_web_config_service.py`：`All checks passed!`。

仍不能对外说：

- 不能说真实 StackedExperts checkpoint 全覆盖；这里是最小 synthetic state_dict 级链路测试。

### EXT 第四十四组低风险扩展阶段

一句话：第四十四组 EXT 给 sample prompts PUT 路由补训练配置分叉保存测试。

阶段完成：

- `E09` sample prompts PUT route 保护：`tests/test_web_config_service.py` 新增 `test_sample_prompts_put_route_forks_to_training_config_specific_file`，确认 `handle_sample_prompts_put()` 会把 `train_config_file` 传到服务层，并把默认提示词文件分叉到 `configs/sample-prompts/imported/lora.txt`。
- `E09` route 层 prompt 解析保护：同一测试确认注释和空行保留在文件内容里，但返回的 `prompts` 列表只包含有效 prompt。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "sample_prompts_put_route_forks"`：`1 passed, 163 deselected`。

仍不能对外说：

- 不能说 `tests/test_web_config_service.py` 全文件在 60 秒内通过；这里只跑 route 级小片段。

### EXT 第四十五组低风险扩展阶段

一句话：第四十五组 EXT 给 raw save-as 路由补“不覆盖已有配置”保护。

阶段完成：

- `E09` raw save-as route 覆盖保护：新增 `test_raw_save_as_route_never_overwrites_existing_config`，确认 `handle_raw_save_as()` 对已有 `configs/imported/lora.toml` 返回 400，并通过 `overwrite=False` 保持原文件内容不变。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "sample_prompts_put_route_forks or raw_save_as_route_never_overwrites"`：`2 passed, 163 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py`：`All checks passed!`。
- `git diff --check`：通过。

仍不能对外说：

- 不能说 Web config route 全链路都已覆盖；本阶段只补两个可直接调用的 route handler 小用例。

### EXT 当前硬门槛盘点 6

一句话：本轮继续新增 LoRA 和 Web route 测试，但还没到 7200 秒硬门槛。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 5044`，仍小于 `7200`。
- LoRA loading、Web config route 和 type-check 命令壳都继续有小测试产物。
- `git diff --check` 最近一次通过。
- 未跟踪文件仍需最终显式 stage；不能用 `git add -A`。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 阅读顺序纠偏

一句话：EXT 第二十九到第三十七组是按补写顺序落入文档，不能按文件位置判断时间先后。

当前纠偏：

- 本文前面的 EXT 段落存在补写错位，尤其是第二十九到第三十七组的文件位置不等于真实推进顺序。
- 判断 `20260705` sustained 目标是否完成时，以 R4 正式收尾验证、`bd591b83` 推送事实和尾部最新状态索引为准；旧 EXT 硬门槛盘点都是历史快照。
- 阶段数量已经满足最低要求；推进轮数量按记录看已满足最低要求，但不替代耗时、R4 总验证、显式 stage、提交和推送门槛。
- S15/S16 总验证尚未执行；EXT 中宽验证不等同于 R4 总验证。
- 当前仍不能 `commit`、不能 `push webui main:main`、不能 `update_goal complete`，也不能说强制长跑目标已完成。

### EXT 第三十八组低风险扩展阶段

一句话：第三十八组 EXT 给 factory metadata 冲突优先级补回归测试。

阶段完成：

- `E02` metadata 显式优先保护：`tests/test_factory_metadata_flow.py` 新增 `test_explicit_metadata_overrides_file_metadata`，确认 safetensors 文件 metadata 与调用方 `metadata=` 冲突时，显式 `metadata=` 优先。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py -k explicit_metadata_overrides_file_metadata`：`1 passed, 8 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py`：`9 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py tests/test_factory_metadata_flow.py tests/test_lora_loading_keys.py`：`49 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_factory_metadata_flow.py`：`All checks passed!`。

仍不能对外说：

- 不能说所有外部调用方都会传入正确 metadata；本阶段只确认显式 `metadata=` 的优先级。

### EXT 第三十九组低风险扩展阶段

一句话：第三十九组 EXT 给普通 split LoRA 的 partial scaling / DoRA 降级路径补测试。

阶段完成：

- `E02` partial scaling 降级保护：`tests/test_lora_loading_keys.py` 新增 `test_refuse_unfused_attn_lora_keys_partial_scaling_metadata_is_dropped`，确认 q/k/v 只有部分组件带 `inv_scale` 或 DoRA magnitude 时，不生成半残的 fused `inv_scale` / `magnitude`，并清掉 split 残留。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "partial_scaling or dora"`：`2 passed, 18 deselected`。

仍不能对外说：

- 不能说 partial channel-scaling checkpoint 被兼容融合；当前行为是保守丢弃无法完整表达的 fused 辅助信息。

### EXT 第四十组低风险扩展阶段

一句话：第四十组 EXT 给 Hydra fallback 保存分支补 empty metadata 和 tensor 形态保护。

阶段完成：

- `E02` Hydra fallback empty metadata 保护：`tests/test_lora_save_pipeline.py` 新增 `test_save_network_weights_hydra_fallback_empty_metadata_writes_typed_moe`，确认 `save_variant=""` 且 state_dict 含 `.lora_up_weight` 时仍写 `_moe.safetensors`。
- `E02` 保存 tensor 形态保护：同一测试读取 `_moe.safetensors` 里的 q/k/v per-expert tensor，确认专家索引完整、q/k/v chunk 位置正确、`dtype=torch.float16` 生效，且空 metadata 不会补写 `ss_network_spec`。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py -k "fallback or empty_metadata"`：`3 passed, 5 deselected`。

仍不能对外说：

- 不能说真实 Hydra checkpoint 全格式都已覆盖；这里仍是 synthetic state_dict 级保存分支测试。

### EXT 第四十一组低风险扩展阶段

一句话：第四十一组 EXT 给 `type-check` 命令壳补空分隔符和入口转发保护。

阶段完成：

- `E10` 空 `--` 保护：`tests/test_type_check_targets.py` 新增 `test_cmd_type_check_empty_separator_uses_default_targets`，确认 `cmd_type_check(["--"])` 回落默认 pilot 白名单，而不是把 pyright 变成空目标。
- `E10` tasks 入口转发保护：`tests/test_tasks_runner.py` 新增 `test_tasks_main_forwards_type_check_separator_and_targets`，确认 `python tasks.py type-check -- --warnings web/services/config/common.py` 的分隔符和 pyright flags 会原样交给子命令。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`13 passed`。

仍不能对外说：

- 不能说 `tasks.py` 已纳入默认 pyright；它仍由命令壳测试保护。
- 不能说默认 type-check 已扩大到全仓；白名单仍明确排除 `datasets.py`、`_legacy.py` 和整个 `web/services/config/` 目录。

### EXT 当前硬门槛盘点 5

一句话：本轮继续有真实测试产物，但耗时仍未到 7200 秒，所以必须继续 EXT。

当前事实：

- 最新实时 `get_goal` 显示 `goal.timeUsedSeconds = 4788`，仍小于 `7200`。
- 阶段数量已满足最低要求，但耗时、R4 总验证、显式 stage、提交和推送门槛仍未满足。
- `git diff --check` 最近一次通过；本轮新增测试和 checkpoint 追加后仍需再跑一次。
- 未跟踪文件仍需在最终显式 stage 时逐个列出，不能用 `git add -A`。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第三十六组低风险扩展阶段

一句话：第三十六组 EXT 继续钉住 split q/k/v LoRA 在 alpha 混合缺失时的缩放和拼接顺序。

阶段完成：

- `E02` q/k/v down 顺序保护：在 block-diag 测试中补断言，确认 fused `lora_down.weight` 按 q、k、v 顺序拼接。
- `E02` mixed missing alpha 保护：新增 `test_refuse_unfused_attn_lora_keys_mixed_missing_alpha_uses_unit_scale`，确认部分组件缺 `.alpha` 时只对缺失项用 `scale=1`，已有 alpha 的组件仍按 `alpha/rank` 缩放。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "mixed_missing_alpha or block_diag_and_alpha_scales"`：`2 passed, 17 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`19 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_lora_loading_keys.py`：通过。

仍不能对外说：

- 不能说所有真实 checkpoint 的 alpha 组合都验证过；这是 synthetic state_dict 级边界测试。

### EXT 第三十七组低风险扩展阶段

一句话：第三十七组 EXT 给 `tasks.py` inline env 解析补非法 token 转发测试。

阶段完成：

- `E10` inline env 边界保护：`tests/test_tasks_runner.py` 新增 `test_tasks_main_forwards_invalid_inline_env_tokens`，确认 `BAD-NAME=value` 这类非法环境变量名不会写入 `os.environ`，而是作为普通参数转发给子命令。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py`：`6 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`11 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_tasks_runner.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_tasks_runner.py`：通过。

仍不能对外说：

- 不能说所有 shell quoting / Windows CMD 场景都覆盖；这里只测 Python 层 token 解析。

### EXT 当前硬门槛盘点 4

一句话：继续有小阶段产物，但还没到 7200 秒，必须继续推进。

当前事实：

- `goal.timeUsedSeconds = 4271`，仍小于 `7200`。
- `git diff --check`：tracked diff 通过；未跟踪文件仍需在最终显式 stage 后纳入提交前检查。
- `git status --short --branch`：仍在 `main...webui/main`，存在当前目标相关未提交改动和未跟踪目标书 / 新测试文件。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第三十四组低风险扩展阶段

一句话：第三十四组 EXT 继续给旧 Hydra 全局 router 格式补明确拒绝测试。

阶段完成：

- `E02` old global Hydra router 拒绝保护：新增 `test_old_global_hydra_router_keys_are_rejected`，确认 `_hydra_router.*` 老格式会在 factory key scan 阶段明确报错，不能进入当前 per-module / three-axis 路由加载。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py -k old_global_hydra_router`：`1 passed, 7 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py`：`8 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py tests/test_factory_metadata_flow.py tests/test_lora_loading_keys.py`：`47 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_factory_metadata_flow.py`：`All checks passed!`。

仍不能对外说：

- 不能说旧 `_hydra_router.*` checkpoint 可迁移；当前行为是明确拒绝并要求重训。

### EXT 第三十五组低风险扩展阶段

一句话：第三十五组 EXT 给 `tasks.py` 无参数入口补全局帮助测试。

阶段完成：

- `E10` tasks 无参数帮助保护：`tests/test_tasks_runner.py` 新增 `test_tasks_main_prints_global_help_without_command`，确认 `python tasks.py` 无命令时打印全局帮助并以 `SystemExit(0)` 退出。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py`：`5 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`10 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_tasks_runner.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_tasks_runner.py`：通过。

仍不能对外说：

- 不能说所有 `tasks.py` 子命令都实际运行过；这是分发层和帮助输出测试。

### EXT 中宽验证 2

一句话：第三十五组后跑一轮中宽验证，确认 LoRA、任务入口、type-check 和 Web config direct-import 切片仍稳。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_network_construction.py`：`44 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`10 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "metadata_module_imports_without_facade_cycle or merge_module_imports_without_facade_cycle or estimation_module_imports or preflight_module_imports or raw_files_module_imports or sample_prompts_module_imports or output_runs_module_imports or paths_module_imports"`：`8 passed, 155 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。

仍不能对外说：

- 不能说 `tests/test_web_config_service.py` 全文件在 60 秒内通过；这里仍是 direct-import 相关切片。

### EXT 最新硬门槛状态

一句话：本文前面的盘点按补写顺序可能不是时间顺序，判断能否收口时以本小节和实时 `get_goal` 为准。

当前事实：

- 最新已记录 `goal.timeUsedSeconds = 4271`，仍小于 `7200`。
- 已完成阶段数远超 12，已完成推进轮数远超 3，但耗时硬门槛仍未达成。
- 当前不能进入 R4 提交推送；R4 必须在 `goal.timeUsedSeconds >= 7200` 后再跑最终验证。
- `git diff --check` 最近一次通过；未跟踪文件仍需在最终显式 stage 后纳入提交前检查。

当前禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

### EXT 第二十九组低风险扩展阶段

一句话：第二十九组 EXT 给 `tasks.py` 用户入口补轻量命令壳测试，不把它纳入默认 pyright。

阶段完成：

- `E10` tasks 命令壳保护：新增 `tests/test_tasks_runner.py`，覆盖全局 `--help`、未知命令、子命令 `--help`、Make 风格 `KEY=value` 环境变量剥离并转发剩余参数。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`9 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_tasks_runner.py tasks.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check tests/test_tasks_runner.py tasks.py`：`All checks passed!`。

仍不能对外说：

- 不能说 `tasks.py` 已纳入默认 type-check；本阶段是命令分发层测试。

### EXT 第三十组低风险扩展阶段

一句话：第三十组 EXT 给普通 split LoRA 缺失 alpha 的兼容默认值补测试。

阶段完成：

- `E02` missing alpha 默认 scale 保护：新增 `test_refuse_unfused_attn_lora_keys_missing_alpha_uses_unit_scale`，确认 q/k/v split 组件缺 `.alpha` 时按 `scale=1`，fused alpha 仍设为 `n*r`。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k "missing_alpha or block_diag"`：`2 passed, 16 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`18 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py`：`All checks passed!`。

仍不能对外说：

- 不能说所有混合 alpha 情况都有真实 checkpoint 覆盖；本阶段是 synthetic state_dict 级行为测试。

### EXT 第三十一组低风险扩展阶段

一句话：第三十一组 EXT 把任务公共 helper 纳入默认 type-check，并保持 `tasks.py` 在命令壳测试层。

阶段完成：

- `E10` `_common.py` 候选验证：单独验证 `scripts/tasks/_common.py` 为 `0 errors, 0 warnings, 0 informations`。
- `E10` 默认 type-check 小步加宽：`TYPE_CHECK_TARGETS` 新增 `scripts/tasks/_common.py`，并同步更新 `tests/test_type_check_targets.py`。

本轮已验证：

- `timeout 60 .venv/bin/python tasks.py type-check scripts/tasks/_common.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，实际检查 `library/config scripts/config_compat.py scripts/config_explain.py scripts/tasks/_common.py scripts/tasks/utilities.py web/services/config/common.py web/services/config/estimation.py web/services/config/file_groups.py web/services/config/merge.py web/services/config/metadata.py web/services/config/output_runs.py web/services/config/paths.py web/services/config/preflight.py web/services/config/raw_files.py web/services/config/sample_prompts.py`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`5 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile scripts/tasks/_common.py scripts/tasks/utilities.py tests/test_type_check_targets.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check scripts/tasks/_common.py scripts/tasks/utilities.py tests/test_type_check_targets.py`：`All checks passed!`。

仍不能对外说：

- 不能说任务系统所有命令模块都纳入 type-check；当前只加了 `_common.py` 和 `utilities.py`。

### EXT 第三十二组低风险扩展阶段

一句话：第三十二组 EXT 给 factory 入口补 legacy `sigma_mlp.*` 拒绝链路测试。

阶段完成：

- `E02` split legacy sigma_mlp 拒绝保护：新增 `test_split_hydra_sigma_mlp_keys_are_rejected_after_refuse`，确认 split q/k/v Hydra 里的旧 `sigma_mlp.*` 先经 refuser 回收到 fused prefix 后，factory 入口仍 loud fail，不会误当成新三轴路由加载。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py -k sigma_mlp`：`1 passed, 6 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_factory_metadata_flow.py`：`7 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py tests/test_factory_metadata_flow.py tests/test_lora_loading_keys.py`：`46 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_factory_metadata_flow.py tests/test_lora_loading_keys.py networks/lora_anima/config.py`：`All checks passed!`。

仍不能对外说：

- 不能说 legacy `sigma_mlp.*` checkpoint 被兼容加载；本阶段确认的是旧格式必须明确拒绝。

### EXT 第三十三组低风险扩展阶段

一句话：第三十三组 EXT 收紧 type-check 白名单文档边界，避免 Web config 继续靠近整目录。

阶段完成：

- `E10` checkpoint 边界摘要更新：第 2 节 `TASK-10` 当前证据已改为当前真实白名单：`library/config`、config 脚本、`scripts/tasks/_common.py` / `utilities.py`、10 个选定 Web config split module。
- `E10` Web config 暂缓规则：checkpoint 明确 `datasets.py`、`_legacy.py` 和整个 `web/services/config/` 目录暂缓纳入；新增 Web config 文件进入默认门禁前必须有单独理由、direct-import 测试或窄行为测试。
- `E10` 清单护栏加固：`tests/test_type_check_targets.py` 额外禁止 `web/services/config/datasets.py` 和 `web/services/config/_legacy.py` 出现在默认目标里。

本轮已验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`5 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md tests/test_type_check_targets.py scripts/tasks/utilities.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check tests/test_type_check_targets.py`：`All checks passed!`。

仍不能对外说：

- 不能说 Web config type-check 会继续扩大；当前决定是暂停继续扩 Web config 默认白名单。

### EXT 当前硬门槛盘点 3（历史快照）

一句话：这是 `20260705` sustained 目标执行中的历史快照，不代表当前最终状态。

当时事实：

- `goal.timeUsedSeconds = 3991`，仍小于 `7200`。
- `git diff --check`：tracked diff 通过；未跟踪文件仍需在最终显式 stage 后纳入提交前检查。
- `git status --short --branch`：仍在 `main...webui/main`，存在当前目标相关未提交改动和未跟踪目标书 / 新测试文件。
- `git diff --stat` 不包含未跟踪文件，最终 stage 前不能只看 stat 估算完整改动。

当时禁止收口：

- 不能 `commit`。
- 不能 `push webui main:main`。
- 不能 `update_goal complete`。
- 不能说 R4 总验证已完成。

当前说明：

- 后续 R4 已正式完成，并已在 `bd591b83` 提交推送。
- 判断当前状态时，以本文最后的 `2026-07-06 文件末尾最新状态` 为准。

---

## 🔚 17. 2026-07-06 文件末尾最新状态

一句话：这是当前文件的最终阅读入口，优先级高于上面所有执行中间快照。

当前状态：

- `project_cleanup_sustained_goal_20260705.md` 已完成归档，最终提交为 `bd591b83 test: extend sustained cleanup coverage`。
- 用户验收记录显示该目标 `goal.timeUsedSeconds = 7889`，约 `2 小时 11 分钟`，并已推送到 `webui/main`。
- 当前新的活跃目标书是 `docs/findings/project_cleanup_sustained_goal_20260706.md`。
- 下一轮不要重复执行 `20260705` 的 long_running、next_stage 或 sustained 目标书。

新目标书强度：

- 最低耗时：`10800` 秒。
- 最低阶段：`20` 个。
- 最低推进轮：`5` 个。
- 最低子系统覆盖：`4` 类。
- 阻塞只能标记 `blocked`，不能作为提前 `complete` 的例外。

下一轮可直接复制：

```text
请按 docs/findings/project_cleanup_sustained_goal_20260706.md 执行跨子系统强制长跑项目清理目标。
```
