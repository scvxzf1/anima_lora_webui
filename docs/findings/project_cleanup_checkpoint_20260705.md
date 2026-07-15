# 项目清理当前检查点与后续推进计划

状态：研究记录 / 阶段快照
适用版本：以文中日期、提交和运行环境为准；不作为当前 main 操作说明

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

---

## 🔚 19. 20260706 跨子系统长跑目标最终收口

一句话：这是 `project_cleanup_sustained_goal_20260706.md` 的最终收口摘要，真正的文件末尾入口见尾部第 17 节。

最终完成事实：

- 活跃目标：`docs/findings/project_cleanup_sustained_goal_20260706.md`。
- 收口前最新 `get_goal`：`goal.timeUsedSeconds = 10889`，已满足 `>=10800` 秒硬门槛。
- 推进轮数：已完成 `R0` 到 `R6`，并在 EXT 池持续推进到第七十八组以上。
- 阶段数：远超 20 个可验收小阶段，每组均绑定测试、源码护栏、只读审计、文档索引或验证证据。
- 子系统覆盖：WebUI frontend、WebUI backend / queue / preview、runtime / launch / config path、training bootstrap、LoRA/config/type-check、docs/archive。
- 远端口径：本地 `main` 发布到 `webui/main`。

R6 最终验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py`：`74 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "output_runs"`：`5 passed, 167 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "file_groups or raw_files or sample_prompts or direct_import"`：`15 passed, 157 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py tests/test_preview_service.py tests/test_weight_analysis_service.py`：`80 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_launch_config.py tests/test_training_bootstrap.py tests/test_runtime_harness_cli.py`：`68 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_lora_save_pipeline.py tests/test_network_cfg.py tests/test_type_check_targets.py tests/test_tasks_runner.py tests/test_docs_archive_indexes.py`：`111 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check`：通过。
- docs 可达性复扫：`docs_md=104 reachable_from_docs_readme=104 missing=0`。
- docs 本地链接复扫：`scanned=112 local_links=194 external_links=27 skipped_non_path=59 broken=0`。
- `git fetch webui --prune && git rev-list --left-right --count HEAD...webui/main`：`0 0`。

中断和拆分说明：

- 一条并行 Web config 切片曾因用户中断未计为通过；之后已重新运行更明确的 `file_groups or raw_files or sample_prompts or direct_import` 切片，并得到 `15 passed, 157 deselected`。
- 早前较宽 Web config 命令若接近或超过 60 秒，均按目标书要求拆成较窄切片记录，不把超时或中断命令当成通过。

最终修改范围：

- docs：整理 `docs/README.md`、分区 README、归档索引、proposal 归档副本和目标/checkpoint 记录。
- WebUI frontend：补 DOM、主题、GPU picker、tab、queue renderer 等静态/Node fixture 测试。
- WebUI backend：补 output root、preview、queue、history、weight analysis 相关边界测试和小护栏。
- runtime / config path / launch：补 `.env`、外置 configs root、显式 env、launch command builder、type-check 目标护栏。
- training：补 bootstrap / compile-after-apply 顺序相关 monkeypatch 测试。
- LoRA/config：补 loading/save/config characterization tests，继续保护 checkpoint key、public API 和三轴路由语义。

明确未做事项：

- 没有跑真实训练。
- 没有下载模型。
- 没有删除、移动或清理用户数据目录。
- 没有删除 `_legacy.py`。
- 没有改 LoRA checkpoint key、public API 或三轴路由语义。
- 没有建立全仓 type-check；当前仍是明确白名单门禁。

最终 stage 风险处理：

- 归档 proposal 文件在 Git 中表现为 `docs/proposal/*.md` 删除 + `_archive/docs/proposal/*.md` 新增，最终必须两边一起显式 stage。
- 不使用 `git add -A`。
- stage 后必须再跑 `git diff --cached --check`。

---

## 🧭 18. 20260706 跨子系统长跑目标执行记录

一句话：本节开始记录 `project_cleanup_sustained_goal_20260706.md` 的真实推进，不再重复执行 20260705 旧目标。

### R0 启动和旧目标归档确认

一句话：R0 已确认本地远端同步、旧目标归档关系和当前工作区风险。

启动事实：

- 当前活跃目标：`docs/findings/project_cleanup_sustained_goal_20260706.md`。
- `goal.timeUsedSeconds = 74` 时启动读取；本轮后续盘点为 `532`，仍远小于 `10800` 秒硬门槛。
- `git log -1 --oneline --decorate`：`cbad09af (HEAD -> main, webui/main, webui/HEAD) docs: add cross-system sustained cleanup goal`。
- `git rev-list --left-right --count HEAD...webui/main`：`0 0`，本地 `main` 与 `webui/main` 同步。
- `date +%s`：`1783312354`。

旧目标归档扫描：

- `rg -n "状态：活跃|project_cleanup_sustained_goal_20260705.md|project_cleanup_sustained_goal_20260706.md" docs/findings/project_cleanup_*20260705.md docs/findings/project_cleanup_checkpoint_20260705.md docs/findings/project_cleanup_sustained_goal_20260706.md` 已执行。
- 旧目标文件顶部已指向 `project_cleanup_sustained_goal_20260706.md` 作为后续入口。
- `project_cleanup_sustained_goal_20260705.md` 在 checkpoint 末尾记录为已完成归档，最终提交 `bd591b83`。
- checkpoint 内部仍有历史执行段提到 `20260705` 活跃入口，这是历史记录，不作为当前入口。

当前工作区风险：

- 启动时已有未提交文档整理改动：`README.md`、`AGENTS.md`、`docs/README.md`、分区索引、7 个 proposal 从 `docs/proposal/` 移到 `_archive/docs/proposal/`。
- 子代理只读审计确认：这些改动与 20260706 长跑目标中的 docs / CLI / type-check 门禁方向一致；未发现用户数据风险。
- 提交前必须显式 stage 原 `D` 文件和 `_archive/docs/proposal/` 新文件，不能只看 `git diff --stat`，否则会把“搬家”误当成删除。

已完成阶段：

- `A00` 基线确认：完成。
- `A01` 旧目标归档扫描：完成。
- `DOC-R0` 文档整理现状审计：完成，作为 docs 子系统证据计入。

### R1 WebUI 前端 / DOM / 静态模块证据

一句话：R1 已补两个不启动服务的 WebUI 前端测试，一个守 DOM id 契约，一个跑队列渲染 DOM fixture。

只读审计：

- 主线运行了 `rg -n "from './|from \"./|globalThis|querySelector|getElementById|addEventListener" web/static/js/features/anima-app web/static/js/features/live-training web/static/js/features/queue web/static/js/features/history-detail`。
- 子代理 `A02-FE-AUDIT` 只读建议：队列管理渲染缺少真实 DOM fixture，推荐在 `tests/test_training_frontend_state.py` 补 `createQueueRenderer` + `updateQueueStateFromPayload` 的 Node fixture。

新增测试：

- `test_queue_and_history_detail_literal_dom_ids_match_index_html`
  - 扫描 `queue/render.js`、`queue/actions.js`、`history-detail/dialog.js` 中 literal `document.getElementById('...')`。
  - 对照 `web/static/index.html`，防止 JS 引用已不存在的 DOM id。
- `test_training_queue_renderer_updates_dom_fixture`
  - 通过 Node 直接 import `createQueueRenderer`、`createQueueState`、`updateQueueStateFromPayload`。
  - 用 fake DOM 渲染 running / queued / error / done 队列。
  - 验证 summary、badge、manager status、stats、filter title、running progress、危险操作按钮 disabled / aria 状态。

本轮验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "queue_renderer_updates_dom_fixture"`：`1 passed, 66 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "dom or selector or queue or history"`：`13 passed, 54 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_frontend_state.py`：通过。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py`：`All checks passed!`。
- `timeout 60 git diff --check -- README.md AGENTS.md docs _archive/docs tests/test_training_frontend_state.py`：通过。
- `timeout 60 .venv/bin/python tasks.py --help >/tmp/anima_tasks_help_check.txt && wc -c /tmp/anima_tasks_help_check.txt`：通过，输出大小 `10534`。

已完成阶段：

- `A02` 前端模块依赖图审计：完成。
- `A03` DOM id 契约补测：完成。
- `A05` queue/history 前端入口保护：完成一部分，新增 queue renderer DOM fixture；history-detail 本轮只覆盖 literal id 契约。
- `A06` CSS / 文档空白守门：本轮跑了相关 `git diff --check`，未做 CSS 结构变更。

当前覆盖和门槛盘点：

- 已推进轮次：`R0` 完成，`R1` 已完成部分可验收阶段；距离最低 `5` 轮还不足。
- 已完成阶段：当前可计 `7` 个左右，距离最低 `20` 个还不足。
- 已覆盖子系统：docs / CLI 文档门禁、WebUI frontend，当前约 `2` 类，距离最低 `4` 类还不足。
- 非纯文档验证：已有前端 pytest、py_compile、ruff、diff check、tasks help；最终仍需按目标书跑至少 `6` 组并覆盖实际改动。
- `goal.timeUsedSeconds = 532`，远小于 `10800`，禁止提交推送和禁止 `update_goal complete`。

仍不能对外说：

- 不能说 20260706 长跑目标已完成。
- 不能说 WebUI 做过真实浏览器全链路验证；本轮只做静态测试和 Node fake DOM fixture。
- 不能说训练、队列或 daemon 被真实启动；本轮没有启动真实训练。
- 不能说文档整理已提交；当前仍是未提交工作区状态。
- 不能说 proposal 文件已删除；当前意图是归档搬家，提交前必须同时 stage 删除和新增归档文件。

### R2 WebUI 后端 / output root / preview / queue 安全

一句话：R2 已补两条路径边界测试，并复跑 preview/env 与 queue runtime 窄验证。

只读审计：

- 主线运行 `rg -n "resolve_output_root|output_root|HISTORY_DIR|QUEUE_DIR|training-history|training-queue|runtime_config|metadata" web/services tests/test_preview_service.py tests/test_training_queue.py tests/test_env_config_paths.py`。
- 审计发现 `tests/test_preview_service.py` 已覆盖全局 output root 正常保存、绝对路径、preview 图片越界、training queue runtime 删除边界等大量场景。
- 子代理 `R2-BACKEND-AUDIT` 建议补 `save_preview_settings()` 的 `training_dir` 项目内限制测试；主线另补 `save_global_settings()` 的 `output_root` 禁止 `..` 测试。

新增测试：

- `test_global_settings_reject_output_root_parent_traversal`
  - 覆盖 `settings_service.save_global_settings({"output_root": "../outside"})`。
  - 确认抛出 `ValueError`，且原 `web-ui-settings.toml` 的安全 `output_root` 不被污染。
- `test_preview_settings_reject_training_dir_outside_project`
  - 覆盖 `preview_service.save_preview_settings()` 保存项目外绝对 `training_dir`。
  - 确认训练预览目录必须在项目目录内，且失败后原 preview 设置不被覆盖。

本轮验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py -k "global_settings"`：`2 passed, 22 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_env_config_paths.py`：`31 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "runtime or metadata or output_root or launch"`：`18 passed, 22 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py -k "preview_settings_reject_training_dir_outside_project"`：`1 passed, 24 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py -k "global_settings or preview_settings"`：`5 passed, 20 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_preview_service.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_preview_service.py`：通过，ruff 为 `All checks passed!`。

已完成阶段：

- `A07` output root 边界审计：完成。
- `A08` preview / settings 路径测试：完成，两条新测试覆盖 settings output root 与 preview training_dir。
- `A09` training queue runtime config 保护：本轮跑了 queue runtime / metadata / output_root / launch 窄验证，未新增 queue 测试。
- `A11` service 层不能越界清单：部分完成，已记录本轮未启动训练、未碰真实队列、未清理用户数据。

当前覆盖和门槛盘点：

- 已推进轮次：`R0`、`R1`、`R2` 已有产物，距离最低 `5` 轮仍不足。
- 已完成阶段：当前约 `11` 个，距离最低 `20` 个仍不足。
- 已覆盖子系统：docs / CLI 文档门禁、WebUI frontend、WebUI backend / preview / queue，当前约 `3` 类，距离最低 `4` 类仍不足。
- 非纯文档验证：已超过 `6` 组，但最终仍需按实际改动做 R6 总验证。
- 当前仍未满足 `goal.timeUsedSeconds >= 10800`，禁止提交推送和禁止 `update_goal complete`。

仍不能对外说：

- 不能说队列真实启动或真实训练已验证；本轮只有单元测试和 monkeypatch / tmp_path 验证。
- 不能说 output root 全链路都无风险；本轮只补了两个窄边界并跑相关测试。
- 不能说 WebUI 后端清理完成；下一轮仍需 runtime / launch / training bootstrap 等子系统覆盖。

### R3 runtime / launch / config path 护栏

一句话：R3 已补一个 launch command builder 行为测试，并跑 config path 与 runtime harness 轻量验证。

只读审计：

- 主线运行 `rg -n "build_launch_cmd|accelerate_training_command_prefix|ANIMA_ACCELERATE_LAUNCH|PROFILE_STEPS|python_exe|runtime_config|compile_blocks_for_training" scripts/tasks/_common.py library/runtime/launch.py web/services/training/runtime_config.py tests/test_launch_config.py tests/test_tasks_runner.py tests/test_env_config_paths.py tests/test_runtime_harness_cli.py`。
- 审计确认 `library/runtime/launch.py` 的 command prefix 可独立测试，不需要启动训练或子进程。
- `scripts/tasks/_common.py::build_launch_cmd` 仍通过 `accelerate_training_command_prefix()` 组合命令，本轮只补底层 runtime launch helper 测试，不拆 task runner。

新增测试：

- `test_direct_training_command_ignores_accelerate_detail_env_when_launch_disabled`
  - 覆盖 `ANIMA_ACCELERATE_NUM_PROCESSES` / `ANIMA_ACCELERATE_MIXED_PRECISION` 已设置，但 `ANIMA_ACCELERATE_LAUNCH` 未启用时，命令仍保持直接 `python train.py`。
  - 保护默认单进程入口，不让细节 env 意外改变普通训练命令构造。

本轮验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py`：`12 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py tests/test_tasks_runner.py`：`23 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_launch_config.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_launch_config.py`：通过，ruff 为 `All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_config.py -k "configs_root or env or path"`：`7 passed, 30 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_runtime_harness_cli.py tests/test_native_flatten.py`：`20 passed, 2 warnings`；警告为本机 GTX 960 CUDA capability 与当前 PyTorch 构建不匹配。

已完成阶段：

- `A12` launch command builder 审计：完成。
- `A13` launch helper 补测：完成。
- `A14` config root 外置路径护栏：本轮跑了现有 path/env/configs_root 窄验证，未新增 config path 测试。
- `A15` runtime harness 低风险验证：完成现有测试验证。
- `A16` block swap 不扩大审计：尚未完成；未拆 CUDA stream / Event / thread pool / hook 调度。

当前覆盖和门槛盘点：

- 已推进轮次：`R0`、`R1`、`R2`、`R3` 已有产物，距离最低 `5` 轮还差至少 `1` 轮。
- 已完成阶段：当前约 `15` 个，距离最低 `20` 个仍不足。
- 已覆盖子系统：docs / CLI 文档门禁、WebUI frontend、WebUI backend / preview / queue、runtime / launch / config path，当前约 `4` 类，已达到子系统覆盖最低数量，但仍需完成时间、阶段和轮次门槛。
- 当前仍未满足 `goal.timeUsedSeconds >= 10800`，禁止提交推送和禁止 `update_goal complete`。

仍不能对外说：

- 不能说训练启动全链路被验证；本轮只测 command builder 和 runtime/config path 单元测试。
- 不能说 block swap 复杂调度已拆清；本轮未碰 offloading CUDA stream / Event / thread pool / hook 调度。
- 不能说 runtime harness 覆盖真实大模型；本轮是模型无关测试。

### R3 追加补记：config root 父级跳转护栏

一句话：R3 后续补上了 config root 相对路径 `..` 拒绝逻辑，避免外置配置根目录静默跳出项目。

追加源码护栏：

- `library/env.py` 新增 `_resolve_project_relative_override()`：
  - 统一处理 WebUI 本机设置文件和环境变量里的相对路径。
  - 相对路径中只要包含 `..` 就抛出 `ValueError`。
  - 绝对路径保持原行为，普通相对路径仍相对项目根解析。
- `get_configs_root()`：
  - WebUI `.anima-webui-settings.toml` 读取分支不再用宽 `except Exception` 吞掉路径校验错误。
  - `ANIMA_CONFIGS_ROOT="../outside"` 会明确失败。
- `get_training_history_root()` 和 `get_training_queue_root()`：
  - 同步复用 helper，防止 history / queue root 通过相对 `..` 跳出项目根。

追加测试：

- `test_get_configs_root_rejects_parent_traversal`
  - 覆盖 `ANIMA_CONFIGS_ROOT="../outside"`。
  - 验证抛错信息包含 `ANIMA_CONFIGS_ROOT`。
- `test_get_configs_root_settings_file_rejects_parent_traversal`
  - 覆盖 `.anima-webui-settings.toml [paths].configs_root = "../outside"`。
  - 验证 WebUI 本机配置文件里的越界路径不会被吞掉。

追加验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py`：`9 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_config.py -k "configs_root or env or path"`：`9 passed, 30 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_runtime_harness_cli.py tests/test_native_flatten.py`：`20 passed, 2 warnings`；警告为本机 GTX 960 与 PyTorch CUDA 构建不匹配。
- `timeout 60 .venv/bin/python -m ruff check library/env.py tests/test_env_config_paths.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile library/env.py tests/test_env_config_paths.py`：通过。

仍不能对外说：

- 不能说所有项目路径入口都已统一用这个 helper；本阶段只覆盖 config root、training history root 和 training queue root。
- 不能说绝对路径被禁止；绝对路径仍按项目既有外置配置约定允许。

### R4 training bootstrap / compile order 保护

一句话：R4 补上训练启动路径里“compile 必须最后”的顺序测试，不加载真实模型、不启动训练。

只读审计：

- 子代理 `R4-TRAINING-AUDIT` 已回收，建议优先保护 `TrainingBootstrap.create_and_apply_network()` 的调用顺序。
- 审计确认关键顺序是 `apply_to -> load_weights -> gradient_checkpointing -> compile_blocks_for_training`。
- 风险点：如果以后把 `torch.compile` 提前，可能 trace 到还没被 adapter monkey-patch 的 forward。

新增测试：

- `test_bootstrap_compiles_after_apply_load_and_gradient_checkpointing`
  - 使用 fake `network_module.create_network()`，不加载真实权重。
  - fake network 记录 `apply_to`、`load_weights`、`enable_gradient_checkpointing`。
  - fake unet 记录 `enable_gradient_checkpointing`。
  - monkeypatch `library.runtime.harness.compile_blocks_for_training()` 记录 `compile`。
  - 只断言相对顺序：`compile` 晚于 `apply_to`、`load_weights`、`unet_grad_ckpt`、`network_grad_ckpt`，不锁死完整事件列表。

R4 验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py::test_bootstrap_compiles_after_apply_load_and_gradient_checkpointing`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py tests/test_runtime_harness_cli.py -k "compile or adapter or apply"`：`7 passed, 12 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_bootstrap.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_bootstrap.py library/training/bootstrap.py`：通过。
- `git diff --check -- tests/test_training_bootstrap.py library/env.py tests/test_env_config_paths.py tests/test_launch_config.py tests/test_preview_service.py tests/test_training_frontend_state.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前覆盖和门槛盘点：

- 已推进轮次：`R0`、`R1`、`R2`、`R3`、`R4` 均已有真实产物，达到最低 `5` 轮。
- 已覆盖子系统：docs / CLI 文档门禁、WebUI frontend、WebUI backend / preview / queue、runtime / launch / config path、training bootstrap / compile order，已超过最低 `4` 类。
- 已完成阶段：按 checkpoint 可验收产物粗略可计 `20+`，但最终仍需继续用后续 EXT / R5 / R6 做收口验证。
- `goal.timeUsedSeconds = 1317`，仍远小于 `10800` 秒硬门槛。

仍不能对外说：

- 不能说训练启动全链路已验证；本轮是 monkeypatch 顺序测试，没有真实训练、没有真实模型加载。
- 不能说可以提交或推送；耗时硬门槛、R5/R6 和最终总验证都未满足。
- 不能 `update_goal complete`；当前目标仍必须继续推进。

### R5 LoRA / config / type-check 残余保护

一句话：R5 补了 ChimeraHydra 配置层专家数派生保护，并把 `tasks.py` 纳入默认 type-check 试点。

只读审计：

- 子代理 `R5-LORA-AUDIT` 只读确认：LoRA 三轴 metadata、FEI 保存、MoE metadata flow、split key refusion 已有较多覆盖；最小缺口是 ChimeraHydra 的 `num_experts` 是否由 `num_experts_content + num_experts_freq` 派生。
- 子代理 `R5-TYPECHECK-DOCS-AUDIT` 只读确认：`tasks.py` 已有 CLI helper 行为测试，但还没进入默认 `TYPE_CHECK_TARGETS`；本轮只建议加 `tasks.py`，不建议把整个 `scripts/tasks/` 目录塞进默认门禁。

源码修复：

- `networks/lora_anima/config.py`
  - `from_kwargs()`：把 `num_experts = num_experts_content + num_experts_freq` 从不可达的 `raise` 后面移回 `use_chimera_hydra` 分支。
  - `from_weights()`：新增 `resolved_num_experts`，ChimeraHydra 用 `K_c + K_f`，普通 Hydra / StackedExperts 仍使用原来的 `hydra_num_experts`，非 MoE 仍回落默认 `4`。
- `scripts/tasks/utilities.py`
  - `TYPE_CHECK_TARGETS` 增加 `tasks.py`。
  - 不扩大到全仓，也不扩大到整个 `scripts/tasks/` 目录。

新增测试：

- `test_chimera_from_kwargs_derives_total_num_experts_from_pool_split`
  - 先失败后修复，失败表现为 `cfg.num_experts == 4`，预期为 `2 + 5 = 7`。
  - 同时确认 Chimera 三轴仍固定为 `shared_A / True / input`。
- `test_chimera_from_weights_derives_total_num_experts_from_stamped_pool_split`
  - 先失败后修复，覆盖 checkpoint/config load 直调路径不应回落默认 `4`。
  - 不改 checkpoint key，不改 public API，不改三轴语义。
- `tests/test_type_check_targets.py`
  - 精确白名单同步增加 `tasks.py`，继续禁止 `.`、`tests`、整个 `web/services/config`、`datasets.py`、`_legacy.py` 误入默认目标。

R5 验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "chimera_from_kwargs_derives_total_num_experts or chimera_from_weights_derives_total_num_experts"`：`2 passed, 21 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py`：`23 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py tests/test_network_cfg.py`：`70 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check tasks.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`，当前默认目标为 `library/config tasks.py scripts/config_compat.py scripts/config_explain.py scripts/tasks/_common.py scripts/tasks/utilities.py` 和 10 个选定 Web config split module。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`6 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`17 passed`。
- `timeout 60 .venv/bin/python -m ruff check networks/lora_anima/config.py tests/test_network_cfg.py scripts/tasks/utilities.py tests/test_type_check_targets.py tasks.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile networks/lora_anima/config.py tests/test_network_cfg.py scripts/tasks/utilities.py tests/test_type_check_targets.py tasks.py`：通过。
- `git diff --check -- networks/lora_anima/config.py tests/test_network_cfg.py scripts/tasks/utilities.py tests/test_type_check_targets.py`：通过。

当前覆盖和门槛盘点：

- 已推进轮次：`R0` 到 `R5` 均已有真实产物。
- 已覆盖子系统：docs / CLI 文档门禁、WebUI frontend、WebUI backend / preview / queue、runtime / launch / config path、training bootstrap / compile order、LoRA/config/type-check，当前约 `6` 类。
- 已完成阶段：已达到并超过最低 `20` 个可验收阶段。
- `goal.timeUsedSeconds = 1765`，仍远小于 `10800` 秒硬门槛。

仍不能对外说：

- 不能说 LoRA save/load/builder/router 已彻底拆完；本轮只修 ChimeraHydra 配置层专家数派生。
- 不能说建立了全仓 type-check；本轮只是把 `tasks.py` 加入默认 pyright pilot gate。
- 不能提交、不能 push、不能 `update_goal complete`；耗时和最终 R6 收口仍未满足。

### EXT 第一组：docs 优化索引 + WebUI live fallback 行为测试

一句话：因为耗时硬门槛仍未满足，本组继续做两个低风险扩展阶段，一个补文档可达性，一个补 WebUI live fallback 行为证据。

docs 扩展：

- 新增 `docs/optimizations/README.md`：
  - 索引 `for_compile.md`、`fa4.md`、`adamw_fused.md`、`hydra_analysis.md`、`training_profiling.md`。
  - 写明本目录用于 compile、kernel、显存和训练性能优化说明。
- 更新 `docs/README.md`：
  - Optimizations 分区新增 `optimizations/README.md` 入口。
  - 确保新增分区 README 从总索引可达。

WebUI frontend 扩展：

- 子代理 `EXT-WEB-LIVE-AUDIT` 只读建议：现有 `test_live_training_rest_fallbacks_are_wired` 主要是字符串连线检查，缺少 `applyStatusSnapshotFallbacks()` 的行为级断言。
- 新增 `test_live_training_status_snapshot_fallbacks_replay_latest_payloads`：
  - Node 小夹具只 import `web/static/js/features/anima-app/chunks/26a-status-polling.js`。
  - running 状态下，非空 `latest_progress` / `latest_metric` / `latest_system` 分别调用 `updateProgress` / `updateMetrics` / `updateSystem`，且参数带 `{ replay: true }`。
  - idle 状态不回放。
  - 空对象、`null`、`undefined` 不回放。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py::test_live_training_status_snapshot_fallbacks_replay_latest_payloads`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "live or progress or status"`：`10 passed, 58 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_frontend_state.py`：通过。
- `git diff --check -- docs/README.md docs/optimizations/README.md tests/test_training_frontend_state.py`：通过。
- `rg -n "optimizations/README.md|Optimizations 文档索引|for_compile.md|training_profiling.md" docs/README.md docs/optimizations/README.md`：确认新索引和上级入口存在。

仍不能对外说：

- 不能说做过真实浏览器或真实 WebSocket 断线验证；本阶段是 Node 小夹具行为测试。
- 不能说文档链接全仓检查已完成；本阶段只确认新增优化索引可达和 diff 空白干净。
- 当前仍不能提交、push 或标记目标完成。

### EXT 第二组：history / queue root 越界护栏补测

一句话：本组把 R3 新 helper 的剩余两个调用点补上直接测试，避免只测 configs root 而漏掉 history / queue。

新增测试：

- `test_get_training_history_root_rejects_parent_traversal`
  - 覆盖 `ANIMA_TRAINING_HISTORY_ROOT="../outside-history"`。
  - 验证 `get_training_history_root()` 抛出 `ValueError`，错误信息包含环境变量名。
- `test_get_training_queue_root_rejects_parent_traversal`
  - 覆盖 `ANIMA_TRAINING_QUEUE_ROOT="../outside-queue"`。
  - 验证 `get_training_queue_root()` 抛出 `ValueError`，错误信息包含环境变量名。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py -k "parent_traversal or training_history_root or training_queue_root"`：`7 passed, 4 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py`：`11 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_env_config_paths.py`：`36 passed`。
- `timeout 60 .venv/bin/python -m ruff check library/env.py tests/test_env_config_paths.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile library/env.py tests/test_env_config_paths.py`：通过。
- `git diff --check -- tests/test_env_config_paths.py library/env.py`：通过。

仍不能对外说：

- 不能说所有路径入口都已经统一越界拒绝；本组只覆盖 config root、history root、queue root。
- 不能说真实队列目录或真实历史目录被读写；全部测试都在 `tmp_path` 和 monkeypatch 环境中完成。

### EXT 第三组：accelerate launch command 形态保护

一句话：本组给训练 launch helper 补完整命令形态测试，仍然不启动训练、不起子进程。

新增测试：

- `test_accelerate_launch_command_wraps_train_script_with_safe_defaults`
  - 覆盖 `ANIMA_ACCELERATE_LAUNCH=1`、`ANIMA_ACCELERATE_NUM_PROCESSES=2`、`ANIMA_ACCELERATE_MIXED_PRECISION=fp16`。
  - 断言返回命令完整形态为 `python -m accelerate.commands.accelerate_cli launch ... train.py`。
  - 同时锁住安全默认项：`--num_machines 1`、`--dynamo_backend no`、`--num_cpu_threads_per_process 3`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py`：`13 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py tests/test_tasks_runner.py`：`24 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_launch_config.py library/runtime/launch.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_launch_config.py library/runtime/launch.py`：通过。
- `git diff --check -- tests/test_launch_config.py library/runtime/launch.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说 accelerate 真实分布式训练启动过；本阶段只测试命令列表构造。
- 不能说 launch/runtime 全链路完成；仍未触碰真实训练、GPU 或 accelerate 子进程。

### EXT 第四组：config facade 与 training resume 窄验证

一句话：本组不新增代码，只补目标书 A10 / A23 要求的低风险验证证据。

验证范围：

- Web config facade / `_legacy`：
  - 确认 metadata re-export、split module direct import、facade cycle 相关测试仍通过。
  - 继续确认 `_legacy.py` 仍是兼容 facade 的一部分，不能在本目标里删除。
- Training resume / history：
  - 只跑 `history`、`meta`、`output_root`、`runtime` 相关筛选。
  - 不启动真实训练，不写真实历史目录。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "legacy or module_imports_without_facade_cycle or metadata"`：`43 passed, 126 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_resume.py -k "history or meta or output_root or runtime"`：`62 passed, 64 deselected`。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md tests/test_launch_config.py tests/test_env_config_paths.py`：通过。

仍不能对外说：

- 不能说 `_legacy.py` 已能删除；当前验证结果反而支持继续保留 facade 兼容入口。
- 不能说 training resume 做过真实训练恢复；本阶段只是现有单元测试窄验证。

### EXT 第五组：runtime harness cond-stream compile 保护

一句话：本组给 `compile_blocks_for_training()` 的 adapter cond-stream 分支补模型无关测试。

新增测试：

- `test_compile_blocks_for_training_compiles_adapter_cond_stream`
  - 使用 fake `unet.compile_blocks()` 和 fake `network.compile_cond_stream()`。
  - 确认 `compile_cond_stream()` 会收到同一组关键 compile 参数：`backend`、`mode`、`n_token_families`、`dynamic_seq`、`seq_range`。
  - 不加载 DiT，不加载 adapter，不触发真实 `torch.compile`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_runtime_harness_cli.py::test_compile_blocks_for_training_compiles_adapter_cond_stream`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_runtime_harness_cli.py tests/test_native_flatten.py`：`21 passed, 2 warnings`；warning 仍是本机 GTX 960 与当前 PyTorch CUDA 构建不匹配。
- `timeout 60 .venv/bin/python -m ruff check tests/test_runtime_harness_cli.py library/runtime/harness.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_runtime_harness_cli.py library/runtime/harness.py`：通过。
- `git diff --check -- tests/test_runtime_harness_cli.py library/runtime/harness.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说 EasyControl 真实模型或真实 cond stream 编译跑过；本阶段只验证 harness 调用契约。
- 不能说 GPU compile 全链路验证完成；本阶段完全模型无关。

### EXT 第六组：live-training rate parser 边界补测

一句话：本组给 live-training 进度速度解析 helper 补两个小边界，避免日志速度格式轻微变化就丢 ETA。

新增断言：

- `parseProgressRateSeconds('3s/step') == 3`
- `parseProgressRateSeconds('4 IT/S') == 0.25`

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py::test_live_training_progress_helpers_parse_runtime_text`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "live or progress or status"`：`10 passed, 58 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_frontend_state.py`：通过，ruff 为 `All checks passed!`。

仍不能对外说：

- 不能说前端所有日志格式都覆盖了；本阶段只补 `s/step` 和大小写/空格 `it/s` 两个边界。

### EXT 第七组：ChimeraHydra 非正池大小拒绝测试

一句话：本组继续收紧 R5 的 ChimeraHydra 配置层保护，确保 content / freq 任一池大小非正时直接失败。

新增测试：

- `test_chimera_from_kwargs_rejects_non_positive_pool_sizes`
  - 覆盖 `num_experts_content` 或 `num_experts_freq` 为 `0` / `-1`。
  - 验证 `LoRANetworkCfg.from_kwargs()` 抛出 `ValueError`，错误信息指向 `num_experts_content > 0` / `num_experts_freq > 0`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "chimera_from_kwargs"`：`5 passed, 22 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py`：`27 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py tests/test_network_cfg.py`：`74 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_network_cfg.py networks/lora_anima/config.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_network_cfg.py networks/lora_anima/config.py`：通过，ruff 为 `All checks passed!`。
- `git diff --check -- tests/test_network_cfg.py networks/lora_anima/config.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说 ChimeraHydra 训练或推理全链路已覆盖；本阶段仍是配置层 characterization test。
- 不能说改了 checkpoint key、public API 或三轴路由语义；这些都没有改。

### EXT 第八组：preprocess runtime placeholder + archive-index 小修

一句话：本组补一个 preprocess runtime 路径占位符测试，并修正归档索引对当前文档分区范围的描述。

preprocess 新增测试：

- `test_preprocess_dataset_rows_expands_runtime_path_placeholders`
  - 构造 `tmp_path/runs/demo/dataset.runtime.toml`。
  - 在 `image_dir`、`cache_dir`、`custom_attributes.source_dir` 中使用 `{output_dir}` 和 `{source_image_dir}`。
  - monkeypatch `preprocess.run` 只收集命令，不执行真实 resize / VAE cache / TE cache。
  - 断言 resize / vae / te 命令里路径都已展开，且不残留 `{output_dir}`、`{source_image_dir}` 或默认 `post_image_dataset`。

docs 小修：

- `docs/archive-index.md` 的归档原则补齐当前实现说明范围：
  - 原来只列 `guidelines/`、`methods/`、`experimental/`、`structure/`、`configuration/`、`findings/`。
  - 现在同步补入 `features/` 和 `optimizations/`，与 `docs/README.md` 当前分区一致。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py -k "preprocess_dataset_rows_expands_runtime_path_placeholders"`：`1 passed, 29 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py -k "dataset_config or runtime or caption or cache_dir or path"`：`30 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_preprocess_paths.py scripts/tasks/preprocess.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_preprocess_paths.py scripts/tasks/preprocess.py`：通过，ruff 为 `All checks passed!`。
- `git diff --check -- docs/archive-index.md tests/test_preprocess_paths.py scripts/tasks/preprocess.py`：通过。

仍不能对外说：

- 不能说真实 preprocess 已执行；本阶段只收集命令列表。
- 不能说真实 `post_image_dataset/`、`output/` 或 `models/` 被读写；本阶段没有碰这些用户数据目录。
- 不能说文档链接全量检查已完成；本阶段只修归档索引描述并做 diff 空白检查。

### EXT 第九组：env root 空白值拒绝护栏

一句话：本组补齐路径 root 环境变量的空白值拒绝，避免纯空白被误解析成项目根。

源码护栏：

- `library/env.py::_resolve_project_relative_override()`：
  - `expand_env_vars(value).strip()` 后如果为空，直接抛出 `ValueError`。
  - 覆盖 `ANIMA_CONFIGS_ROOT`、`ANIMA_TRAINING_HISTORY_ROOT`、`ANIMA_TRAINING_QUEUE_ROOT` 三条复用路径。

新增测试：

- `test_path_root_overrides_reject_blank_values`
  - 用 `tmp_path/project` 隔离项目根。
  - 分别设置三个 root env 为纯空白。
  - 确认对应 getter 都抛出带 env 名的 `ValueError`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py -k "blank_values or parent_traversal or training_history_root or training_queue_root"`：`8 passed, 4 deselected`。
- `timeout 60 .venv/bin/python -m ruff check library/env.py tests/test_env_config_paths.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile library/env.py tests/test_env_config_paths.py`：通过。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_preview_service.py`：`37 passed`。

仍不能对外说：

- 不能说所有路径入口都已有空白值拒绝；本组只覆盖复用 `_resolve_project_relative_override()` 的 root 类入口。
- 不能说绝对路径被禁止；绝对路径外置配置仍按当前项目约定允许。

### EXT 第十组：LoRA from_weights 三轴 stamp 拒绝护栏

一句话：本组让 checkpoint metadata 路径复用训练配置的三轴路由不变量，坏 stamp 会明确失败。

源码护栏：

- `networks/lora_anima/config.py` 新增 `_validate_three_axis_routing()`：
  - `use_moe_style=False` 时不能开启任何 router。
  - `router_source="input"` 必须是 per-layer router。
  - `router_source="crossattn_emb"` 必须是 network-level router。
- `from_kwargs()` 和 `from_weights()` 共同调用该 helper。
- `from_weights()` 的 `new_route_per_layer` 改走 `_as_bool()`，让字符串 stamp 不会被 `bool("false")` 误判。

新增测试：

- `test_from_weights_rejects_invalid_three_axis_stamp_combinations`
  - 覆盖 `route_per_layer=False + router_source="input"`。
  - 覆盖 `route_per_layer=True + router_source="crossattn_emb"`。
  - 只拒绝 malformed metadata，不改 checkpoint key、不改 public API、不改三轴语义。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "invalid_three_axis_stamp or chimera_from_weights or warm_start_shape"`：`4 passed, 25 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py`：`29 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py tests/test_network_cfg.py`：`76 passed`。
- `timeout 60 .venv/bin/python -m ruff check networks/lora_anima/config.py tests/test_network_cfg.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile networks/lora_anima/config.py tests/test_network_cfg.py`：通过。

仍不能对外说：

- 不能说兼容加载非法 three-axis checkpoint；当前行为是明确拒绝。
- 不能说 LoRA checkpoint key 或 public API 有改动；本组只加校验和测试。

### EXT 第十一组：queue state 错误快照保留测试

一句话：本组用纯 Node 小夹具保护队列状态层，后端错误 payload 不应清空上一份有效快照。

只读审计：

- 子代理 `EXT09-FE-AUDIT` 建议优先补 `updateQueueStateFromPayload()` 的错误快照保留行为。
- 主线采用该建议，避免重复已完成的 DOM id、queue renderer、live fallback、rate parser 测试。

新增测试：

- `test_queue_state_preserves_snapshot_on_error_payloads`
  - import `createQueueState()`、`updateQueueStateFromPayload()`、`queueSummaryCounts()`、`queueManagerSections()`。
  - 先写入 running / queued / done / canceled 的有效 payload。
  - 再写入 `ok:false` 且无 `items` 的错误 payload。
  - 确认旧 items、paused、failure policy、status、current item 都保留，只更新 error。
  - 确认没有 summary 时 `queueSummaryCounts()` 能从 items 回退统计。
  - 确认 `done` / `canceled` 筛选会展开终态分组。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py::test_queue_state_preserves_snapshot_on_error_payloads`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "queue_state or queue_renderer or queue_and_history"`：`3 passed, 66 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "queue or live or progress or status"`：`16 passed, 53 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_frontend_state.py`：通过。
- `git diff --check -- library/env.py tests/test_env_config_paths.py networks/lora_anima/config.py tests/test_network_cfg.py tests/test_training_frontend_state.py`：通过。

仍不能对外说：

- 不能说启动过 WebUI、队列 daemon 或真实训练；本组只是 ES module 小夹具。
- 不能说覆盖了所有队列错误恢复路径；本组只保护错误状态保留旧快照这一条。

### EXT 当前硬门槛盘点 1

一句话：阶段数、轮次和子系统覆盖已经超过最低线，但 3 小时时间门槛还远没到，必须继续 EXT。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 3257`，仍小于 `10800`。
- 已完成 `R0` 到 `R5`，并继续完成 EXT 1 到 EXT 11。
- 子系统覆盖包括 docs / WebUI frontend / WebUI backend / runtime path / launch / training bootstrap / LoRA config / preprocess。
- 非纯文档验证已经远超 `6` 组，但最终仍要在 R6 重新跑总验证。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第十二组：preprocess runtime 顶层路径 fallback 测试

一句话：本组确认没有可用 dataset_config 时，preprocess 仍使用 runtime/top-level 路径，不回退到真实默认用户目录。

只读审计：

- 子代理 `EXT09-PREPROCESS-RUNTIME-AUDIT` 建议补 `_preprocess_rows()` 的 fallback 命令构造测试。
- 主线采用该建议，只捕获命令列表，不执行真实 resize、VAE cache 或 TE cache。

新增测试：

- `test_preprocess_rows_fallback_uses_runtime_top_level_paths`
  - `_PATH_OVERRIDES_CACHE` 中提供缺失的 `dataset_config` 和 runtime 顶层路径。
  - monkeypatch `preprocess.run` 收集三条命令。
  - monkeypatch caption backup 和 caption index，避免读写真实数据。
  - 断言 resize / vae / te 命令使用 `output/runs/demo/...` 路径。
  - 断言命令里不包含默认 `post_image_dataset` 或 `image_dataset`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py::test_preprocess_rows_fallback_uses_runtime_top_level_paths`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py -k "preprocess_rows or runtime_path_placeholders or dataset_config"`：`5 passed, 26 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py -k "dataset_config or runtime or caption or cache_dir or path"`：`31 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_preprocess_paths.py scripts/tasks/preprocess.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_preprocess_paths.py scripts/tasks/preprocess.py`：通过。

仍不能对外说：

- 不能说真实 preprocess 已执行；本组只验证命令构造。
- 不能说真实 `post_image_dataset/`、`output/` 或 `models/` 被读写；本组没有碰这些目录。

### EXT 第十三组：history curve data 纯函数测试

一句话：本组用 Node 小夹具保护 history 曲线数据 helper，避免学习率别名、过滤、平滑和降采样行为悄悄变。

新增测试：

- `test_history_curve_data_helpers_normalize_filter_and_downsample`
  - import `historyCurveNormalizePoint()` / `historyCurveNormalizeRawMetricPoint()` / `historyCurveFilteredPoints()` / `historyCurveSmoothPoints()` / `historyCurveDisplayPoints()`。
  - 覆盖 `lr`、`learningRate`、`learning_rate` 三种学习率字段归一。
  - 覆盖无效 metric 点过滤。
  - 覆盖 custom step 范围过滤。
  - 覆盖平滑窗口输出。
  - 覆盖 `HISTORY_CURVE_RENDER_POINT_LIMIT = 1600` 下的降采样数量、首尾和唯一索引。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py::test_history_curve_data_helpers_normalize_filter_and_downsample`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "history_curve_data_helpers or queue_state or queue_renderer or live or progress or status"`：`13 passed, 57 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "history_curve or queue or live or progress or status"`：`17 passed, 53 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_frontend_state.py`：通过。
- `git diff --check -- tests/test_preprocess_paths.py tests/test_training_frontend_state.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说真实历史任务文件被读取；本组只跑前端 ES module 纯函数。
- 不能说浏览器图表渲染全链路已验证；本组不涉及 DOM / SVG 渲染。

### EXT 当前硬门槛盘点 2

一句话：本轮继续增加 preprocess 和 WebUI 前端证据，但耗时仍没到 3 小时，所以继续 EXT。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 3471`，仍小于 `10800`。
- 已完成 EXT 1 到 EXT 13；所有新增阶段都有测试、源码护栏或文档索引证据。
- 最近组合验证包括 preprocess 路径切片 `31 passed`、前端 history/queue/live 切片 `17 passed`。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第十四组：preprocess resize 同源同目标跳过测试

一句话：本组保护 resize 步骤的自读自写边界，同一 source/destination 时默认跳过。

新增测试：

- `test_preprocess_resize_skips_same_source_and_destination_without_path_override`
  - 直接调用 `_run_preprocess_resize()`，不执行真实 resize。
  - 当 `source_image_dir` 与 `resized_image_dir` 规范化后相同时，确认不调用 `run()`。
  - 确认输出包含 `skip resize`。
  - 当用户显式传入 `--src` / `--dst` override 时，确认命令仍被转交。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py::test_preprocess_resize_skips_same_source_and_destination_without_path_override`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py -k "preprocess_resize_skips or preprocess_rows or runtime_path_placeholders or dataset_config"`：`6 passed, 26 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py -k "preprocess_resize_skips or preprocess_rows or runtime_path_placeholders or dataset_config or path"`：`32 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_preprocess_paths.py scripts/tasks/preprocess.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_preprocess_paths.py scripts/tasks/preprocess.py`：通过。

仍不能对外说：

- 不能说真实 resize 已执行；本组只验证 helper 分支和命令转交。
- 不能说所有 preprocess 写路径都已统一做同源跳过；本组只覆盖 resize 步骤。

### EXT 第十五组：LoRA register/router scalar 解析测试

一句话：本组给 LoRA 配置层补字符串 scalar 解析测试，锁住现状但不改生产逻辑。

新增测试：

- `test_from_kwargs_parses_register_and_router_scalar_knobs`
  - 覆盖 `lora_fp32_compute` 字符串布尔值。
  - 覆盖 `down_init="weight_svd"`。
  - 覆盖 network-level FEI router 的 `router_hidden` alias、`router_tau`、`fei_feature_dim`。
  - 覆盖 `num_registers`、`register_insert_block`、`register_lr_scale`、`register_init_std`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "register_and_router_scalar or invalid_three_axis_stamp"`：`3 passed, 27 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py`：`30 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py tests/test_network_cfg.py`：`77 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_network_cfg.py networks/lora_anima/config.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_network_cfg.py networks/lora_anima/config.py`：通过。
- `git diff --check -- tests/test_preprocess_paths.py tests/test_network_cfg.py networks/lora_anima/config.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说 LoRA 配置层所有字段都完成 characterization；本组只覆盖 register/router scalar 和 down init 等小字段。
- 不能说改动了 LoRA public API、checkpoint key 或三轴语义；本组没有改这些。

### EXT 当前硬门槛盘点 3

一句话：阶段继续增加，验证仍干净，但耗时只有约一小时，必须继续 EXT。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 3641`，仍小于 `10800`。
- 最近组合验证包括 preprocess 路径切片 `32 passed`、LoRA 组合 `77 passed`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第十六组：WebUI configs_root settings 优先级测试

一句话：本组确认 WebUI 本机 settings 文件里的 `configs_root` 优先于环境变量，并且 history/queue 默认跟随它。

新增测试：

- `test_training_roots_follow_webui_configs_root_settings`
  - 在 `tmp_path/project` 写入 `.anima-webui-settings.toml [paths].configs_root = "local-configs"`。
  - 同时设置 `ANIMA_CONFIGS_ROOT` 指向另一个目录。
  - 清空 `ANIMA_TRAINING_HISTORY_ROOT` 和 `ANIMA_TRAINING_QUEUE_ROOT`。
  - 确认 `get_configs_root()` 使用 settings 文件路径。
  - 确认 history / queue 默认分别落到该 configs root 下的 `web-training-history` / `web-training-queue`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py -k "training_roots_follow_webui_configs_root_settings or blank_values or parent_traversal"`：`6 passed, 7 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_preview_service.py`：`38 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_env_config_paths.py tests/test_training_frontend_state.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_env_config_paths.py tests/test_training_frontend_state.py`：通过。

仍不能对外说：

- 不能说 WebUI settings UI 实际保存链路被浏览器验证；本组只测 env/helper 层。
- 不能说 history/queue 专用 env 被禁止；专用 env 仍高于 fallback。

### EXT 第十七组：history detail state alias / resume 文案测试

一句话：本组用纯 Node 小夹具保护 history detail 状态别名、reset 行为和续训剩余步数文案。

新增测试：

- `test_history_detail_state_aliases_and_resume_labels`
  - 覆盖 `resume -> overview`、`chart -> analysis`、`samples -> preview`、`paths -> config_files`。
  - 覆盖未知 tab 回 `overview`。
  - 覆盖 `setHistoryDetailTab()` 的别名结果。
  - 覆盖 `resetHistoryDetailViewState()` 清掉 payload、return state、main task return 和 hover step。
  - 覆盖 `resumeCheckpointProgressText()` 和 `resumeCheckpointRemainingText()` 的正常剩余步数与估算失败文案。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py::test_history_detail_state_aliases_and_resume_labels`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "history_detail_state_aliases or history_curve or queue or live or progress or status"`：`18 passed, 53 deselected`。
- `git diff --check -- tests/test_env_config_paths.py tests/test_training_frontend_state.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说真实 history detail 弹窗或浏览器交互已验证；本组只测前端纯状态 helper。
- 不能说 resume 真实训练恢复已验证；本组只验证显示文案和状态清理。

### EXT 当前硬门槛盘点 4

一句话：当前推进已经很宽，但 3 小时时间门槛仍未达到。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 3775`，仍小于 `10800`。
- 最近组合验证包括 env/preview `38 passed`、前端 history/queue/live 切片 `18 passed`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第十八组：accelerate launch falsey env 修复

一句话：本组修复 `ANIMA_ACCELERATE_LAUNCH="0"` 仍会启用 accelerate 的小 bug，让常见 falsey env 保持直启训练命令。

源码修复：

- `library/runtime/launch.py`
  - 新增 `_env_flag_enabled()`。
  - `accelerate_training_command_prefix()` 只在 `1/true/yes/on` 时启用 accelerate launch。
  - `0/false/no/off/空字符串` 都保持 `[python, train.py]` 直启命令。

新增测试：

- `test_falsey_accelerate_launch_env_keeps_direct_training_command`
  - 覆盖 `0`、`false`、`False`、`no`、`off`、空字符串。
  - 即使同时设置 num processes 和 mixed precision，也不启用 accelerate。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py -k "falsey_accelerate or accelerate_launch_command or direct_training"`：`8 passed, 11 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py tests/test_tasks_runner.py`：`30 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py tests/test_runtime_harness_cli.py -k "launch or compile or adapter or apply"`：`25 passed, 6 deselected`。
- `timeout 60 .venv/bin/python -m ruff check library/runtime/launch.py tests/test_launch_config.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile library/runtime/launch.py tests/test_launch_config.py`：通过。
- `git diff --check -- library/runtime/launch.py tests/test_launch_config.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说真实训练或 accelerate 子进程已启动；本组只验证命令列表构造。
- 不能说 launch/runtime 全链路已完成；本组只修 falsey env 语义。

### EXT 当前硬门槛盘点 5

一句话：launch 小 bug 已修，但当前耗时仍远不到 10800 秒。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 3873`，仍小于 `10800`。
- 最近 launch/runtime 组合验证 `25 passed, 6 deselected`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第十九组：accelerate launch truthy env 回归测试

一句话：本组补齐 launch env 修复的另一半，确认常见 truthy 值仍会启用 accelerate。

新增测试：

- `test_truthy_accelerate_launch_env_enables_launch_command`
  - 覆盖 `1`、`true`、`TRUE`、`yes`、`on`。
  - 确认命令前缀仍是 `python -m accelerate.commands.accelerate_cli launch`。
  - 确认 train script 仍在命令末尾。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py -k "truthy_accelerate or falsey_accelerate or accelerate_launch_command"`：`12 passed, 12 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py`：`24 passed`。
- `timeout 60 .venv/bin/python -m ruff check library/runtime/launch.py tests/test_launch_config.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile library/runtime/launch.py tests/test_launch_config.py`：通过，ruff 为 `All checks passed!`。

仍不能对外说：

- 不能说 accelerate 真实启动过；本组仍是 command builder 单元测试。

### EXT 第二十组：LoRA from_weights 字符串 route_per_layer stamp 测试

一句话：本组保护 checkpoint metadata 字符串解析，`"false"` 不能被 Python `bool("false")` 误当成 true。

新增测试：

- `test_from_weights_parses_string_route_per_layer_stamp`
  - 使用 `new_route_per_layer="false"` 和 `new_router_source="crossattn_emb"`。
  - 确认 `cfg.route_per_layer is False`。
  - 确认合法 network-level `crossattn_emb` metadata 不被三轴校验误拒绝。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "string_route_per_layer or invalid_three_axis_stamp or crossattn_emb"`：`6 passed, 25 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py`：`31 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py tests/test_network_cfg.py`：`78 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_network_cfg.py networks/lora_anima/config.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_network_cfg.py networks/lora_anima/config.py`：通过，ruff 为 `All checks passed!`。
- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- library/runtime/launch.py tests/test_launch_config.py networks/lora_anima/config.py tests/test_network_cfg.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说支持 legacy metadata fallback；本组只保护新三轴 stamp 的字符串解析。
- 不能说 LoRA checkpoint key 或 public API 有变化；本组没有改。

### EXT 当前硬门槛盘点 6

一句话：阶段数继续增加，但当前耗时仍只有约 3978 秒，不能进入 R6。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 3978`，仍小于 `10800`。
- `timeout 60 .venv/bin/python tasks.py type-check` 通过，输出 `0 errors, 0 warnings, 0 informations`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第二十一组：accelerate launch env trim 测试

一句话：本组确认 launch 相关 env 值会先 trim / normalize，再参与命令构造。

新增测试：

- `test_accelerate_launch_env_values_are_stripped_before_parsing`
  - 覆盖 `ANIMA_ACCELERATE_LAUNCH=" true "`。
  - 覆盖 `ANIMA_ACCELERATE_NUM_PROCESSES=" 3 "`。
  - 覆盖 `ANIMA_ACCELERATE_MIXED_PRECISION=" FP16 "`。
  - 确认命令里 `--num_processes 3`、`--mixed_precision fp16`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py -k "stripped_before_parsing or truthy_accelerate or falsey_accelerate"`：`12 passed, 13 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_launch_config.py tests/test_type_check_targets.py library/runtime/launch.py scripts/tasks/utilities.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_launch_config.py tests/test_type_check_targets.py library/runtime/launch.py scripts/tasks/utilities.py`：通过。

仍不能对外说：

- 不能说真实 accelerate 已执行；仍是 launch helper 命令构造测试。

### EXT 第二十二组：type-check 显式目标命令壳测试

一句话：本组确认 `tasks.py type-check` 不加 `--` 也能接显式目标，避免用户文档和维护命令不一致。

新增测试：

- `test_cmd_type_check_accepts_explicit_targets_without_separator`
  - monkeypatch `find_spec("pyright")` 和 `run()`。
  - 调用 `utilities.cmd_type_check(["library/runtime/launch.py"])`。
  - 确认命令为 `python -m pyright library/runtime/launch.py`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py -k "explicit_targets_without_separator or separator or default_targets"`：`4 passed, 3 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check library/runtime/launch.py scripts/tasks/utilities.py`：`0 errors, 0 warnings, 0 informations`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py tests/test_tasks_runner.py tests/test_type_check_targets.py`：`43 passed`。
- `git diff --check -- library/runtime/launch.py tests/test_launch_config.py scripts/tasks/utilities.py tests/test_type_check_targets.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说全仓 pyright 已开启；仍是显式目标和默认 pilot gate。
- 不能说所有 task 命令模块都纳入默认 type-check；默认范围仍是小白名单。

### EXT 当前硬门槛盘点 7

一句话：CLI/launch 门禁继续加宽，但耗时仍没到硬门槛。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4069`，仍小于 `10800`。
- 最近验证包括显式 type-check `0 errors, 0 warnings, 0 informations` 和 CLI 组合 `43 passed`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第二十三组：training bootstrap V100 fp32 compute 判断测试

一句话：本组用 monkeypatch 保护 V100 fp16 下自动启用 `lora_fp32_compute` 的判断逻辑，不碰真实 GPU。

新增测试：

- `test_bootstrap_auto_enables_lora_fp32_compute_on_v100_fp16`
  - monkeypatch `torch.cuda.is_available()` 返回 true。
  - monkeypatch `torch.cuda.get_device_capability()` 返回 `(7, 0)`。
  - 确认 `mixed_precision="fp16"` 且未显式设置 `lora_fp32_compute` 时返回 true。
  - 确认 CUDA capability 查询接收 accelerator device。
- `test_bootstrap_does_not_auto_enable_lora_fp32_compute_when_user_set`
  - 显式传入 `{"lora_fp32_compute": "false"}`。
  - 确认不探测 CUDA，并返回 false。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py -k "auto_enables_lora_fp32_compute or does_not_auto_enable_lora_fp32"`：`2 passed, 8 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py -k "lora_fp32 or register or compile"`：`6 passed, 4 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_bootstrap.py library/training/bootstrap.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_bootstrap.py library/training/bootstrap.py`：通过。

仍不能对外说：

- 不能说真实 GPU 或真实训练已验证；本组只 monkeypatch CUDA capability。
- 不能说自动 fp32 compute 在所有 GPU 上都覆盖；本组只保护 V100/sm70 分支和显式用户设置优先级。

### EXT 第二十四组：training bootstrap/resume/optimizer 窄组合验证

一句话：本组不新增代码，只确认刚补的 bootstrap 判断没有误伤训练基础测试面。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py tests/test_training_optimizers.py tests/test_training_resume.py -k "lora_fp32 or register or compile or bootstrap or resume or optimizer"`：`140 passed, 3 skipped`。
- `git diff --check -- tests/test_training_bootstrap.py library/training/bootstrap.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说真实训练启动或恢复已执行；本组仍是单元测试和 monkeypatch 验证。
- 不能说训练子系统全量测试完成；这里只是和本轮改动相关的窄组合。

### EXT 当前硬门槛盘点 8

一句话：训练 bootstrap 继续有新证据，但 3 小时硬门槛仍未达到。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4203`，仍小于 `10800`。
- 最近训练组合验证 `140 passed, 3 skipped`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第二十五组：training bootstrap CUDA capability 失败兜底测试

一句话：本组确认 V100 fp32 compute 自动判断在 CUDA capability 读取失败时保守关闭，并记录 warning。

新增测试：

- `test_bootstrap_auto_lora_fp32_compute_fails_closed_on_capability_error`
  - monkeypatch `torch.cuda.is_available()` 返回 true。
  - monkeypatch `torch.cuda.get_device_capability()` 抛出 `RuntimeError`。
  - 确认 `should_auto_enable_lora_fp32_compute()` 返回 false。
  - 确认日志包含 `could not read GPU compute capability`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py -k "lora_fp32_compute"`：`3 passed, 8 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py`：`11 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_bootstrap.py library/training/bootstrap.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_bootstrap.py library/training/bootstrap.py`：通过。

仍不能对外说：

- 不能说真实 CUDA capability 探测已执行；本组是 monkeypatch 兜底测试。

### EXT 第二十六组：前端状态全文件 + training/runtime 切片验证

一句话：本组不新增代码，只跑较宽验证，确认新增前端 Node 小夹具和 training bootstrap 测试没有破坏既有测试面。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py`：`71 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_bootstrap.py tests/test_runtime_harness_cli.py -k "lora_fp32 or register or compile or launch"`：`13 passed, 10 deselected`。
- `git diff --check -- tests/test_training_bootstrap.py tests/test_training_frontend_state.py library/training/bootstrap.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说做过浏览器端真实交互；前端验证仍是静态/Node/pytest。
- 不能说 runtime harness 覆盖真实大模型；本组仍是模型无关测试。

### EXT 当前硬门槛盘点 9

一句话：宽验证继续通过，但耗时还不到硬门槛的一半。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4306`，仍小于 `10800`。
- 最近验证包括前端状态全文件 `71 passed`、training/runtime 切片 `13 passed, 10 deselected`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第二十七组：proposal 归档搬家内容一致性验证

一句话：本组验证 7 个 proposal 归档搬家不是内容删除，归档副本与 HEAD 旧文件内容一致。

验证内容：

- 用 `git show HEAD:docs/proposal/<file>` 读取旧版本内容。
- 与 `_archive/docs/proposal/<file>` 当前未跟踪归档副本逐字节比较。
- 覆盖 7 个文件：
  - `compile_safety_patches_analysis.md`
  - `configs_external_data_root_plan_2026-06-24.md`
  - `upstream_high_value_merge_roadmap_2026-06-24.md`
  - `upstream_merge_completion_report_2026-06-24.md`
  - `upstream_merge_completion_report_2026-06-24_audit.md`
  - `upstream_merge_completion_report_fixes_summary.md`
  - `upstream_preprocess_robustness_analysis.md`

本组验证：

- Python 内容比较脚本输出：`archive proposal content matches HEAD for 7 files`。
- `git diff --check -- README.md AGENTS.md docs _archive/docs`：通过。
- `rg -n "archive-index.md|configuration/README.md|features/README.md|findings/README.md|optimizations/README.md|proposal/README.md|documentation_consolidation_20260706.md" README.md docs/README.md docs/archive-index.md docs/findings/README.md docs/proposal/README.md _archive/docs/proposal/README.md`：确认上级入口和分区索引引用存在。
- `find docs -maxdepth 2 -type f -name 'README.md' | sort`：确认当前 docs 分区 README 包括 `configuration`、`features`、`findings`、`optimizations`、`proposal`。

仍不能对外说：

- 不能说这些 proposal 被简单删除；当前目标是归档搬家。
- 不能说最终 stage 已完成；归档新增和原路径删除仍需最后显式 stage。

### EXT 当前硬门槛盘点 10

一句话：文档归档验证已补，但时间门槛仍远未达到。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4357`，仍小于 `10800`。
- proposal 归档内容一致性已验证，文档 diff 空白检查通过。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第二十八组：ChimeraHydra from_weights 非正池大小拒绝

一句话：本组让 ChimeraHydra checkpoint metadata 路径和训练配置路径保持一致，content/freq 池大小必须是正数。

源码护栏：

- `networks/lora_anima/config.py::LoRANetworkCfg.from_weights()`
  - `is_chimera_hydra=True` 时，`num_experts_content` / `num_experts_freq` 不仅必须存在，还必须都大于 0。
  - 非正值直接抛出 `RuntimeError`，提示 checkpoint metadata malformed。

新增测试：

- `test_chimera_from_weights_rejects_non_positive_pool_sizes`
  - 覆盖 `(0, 3)`、`(3, 0)`、`(-1, 3)`、`(3, -1)`。
  - 确认错误信息包含 `requires positive`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "chimera_from_weights"`：`5 passed, 30 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py`：`35 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py tests/test_network_cfg.py`：`82 passed`。
- `timeout 60 .venv/bin/python -m ruff check networks/lora_anima/config.py tests/test_network_cfg.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile networks/lora_anima/config.py tests/test_network_cfg.py`：通过。
- `git diff --check -- networks/lora_anima/config.py tests/test_network_cfg.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说会迁移 malformed Chimera checkpoint；当前行为是明确拒绝坏 metadata。
- 不能说 checkpoint key 或 public API 有变化；本组只加校验。

### EXT 当前硬门槛盘点 11

一句话：LoRA metadata 护栏继续加固，但耗时仍不到 10800 秒。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4445`，仍小于 `10800`。
- 最近 LoRA 组合验证 `82 passed`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第二十九组：ChimeraHydra from_weights 字符串 pool size metadata 测试

一句话：本组让 `from_weights()` 自己能稳妥处理字符串形式的 Chimera pool size metadata，避免直接调用时字符串拼接误伤 mismatch 检查。

源码护栏：

- `networks/lora_anima/config.py::LoRANetworkCfg.from_weights()`
  - 在 ChimeraHydra 分支先把 `num_experts_content` / `num_experts_freq` 转为 `int`。
  - 后续正数检查、`K_c + K_f` mismatch 检查和 `resolved_num_experts` 都使用 int 值。

新增测试：

- `test_chimera_from_weights_accepts_string_pool_size_metadata`
  - 传入 `num_experts_content="2"`、`num_experts_freq="5"` 和 `hydra_num_experts=7`。
  - 确认 `cfg.num_experts == 7`，两个 pool size 在 cfg 中是 int。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "chimera_from_weights"`：`6 passed, 30 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py`：`36 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_factory_metadata_flow.py tests/test_lora_save_pipeline.py tests/test_network_cfg.py`：`83 passed`。
- `timeout 60 .venv/bin/python -m ruff check networks/lora_anima/config.py tests/test_network_cfg.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile networks/lora_anima/config.py tests/test_network_cfg.py`：通过，ruff 为 `All checks passed!`。
- `git diff --check -- networks/lora_anima/config.py tests/test_network_cfg.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说 factory metadata parser 有问题；factory 入口本来通常已转 int，本组是加固 `from_weights()` 自身边界。
- 不能说 checkpoint key 或 public API 有变化；本组没有改。

### EXT 当前硬门槛盘点 12

一句话：工作区仍未 stage，耗时仍未达到 3 小时。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4540`，仍小于 `10800`。
- `git status --short --branch` 确认当前仍未 stage，未跟踪文件包括目标相关新索引和 proposal 归档副本。
- 当前仍不能进入 R6。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第三十组：path root 绝对路径 `..` 拒绝护栏

一句话：本组把 root 路径覆盖规则收紧为“任何路径中包含 `..` 都拒绝”，和项目路径协议保持一致。

源码护栏：

- `library/env.py::_resolve_project_relative_override()`
  - 以前只拒绝相对路径中的 `..`。
  - 现在绝对路径和相对路径只要 `Path.parts` 中包含 `..` 都抛出 `ValueError`。
  - 普通绝对路径仍允许。

新增测试：

- `test_get_configs_root_rejects_absolute_parent_traversal`
  - 设置 `ANIMA_CONFIGS_ROOT` 为绝对路径形式的 `safe/../outside`。
  - 确认 `get_configs_root()` 抛出带环境变量名的 `ValueError`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py -k "absolute_parent_traversal or env_absolute or parent_traversal or blank_values or training_roots"`：`8 passed, 6 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_preview_service.py`：`39 passed`。
- `timeout 60 .venv/bin/python -m ruff check library/env.py tests/test_env_config_paths.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile library/env.py tests/test_env_config_paths.py`：通过，ruff 为 `All checks passed!`。
- `git diff --check -- library/env.py tests/test_env_config_paths.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说绝对外置配置路径被禁止；普通绝对路径仍允许。
- 不能说所有项目路径入口都已统一；本组只覆盖使用 `_resolve_project_relative_override()` 的 root 覆盖入口。

### EXT 第三十一组：preprocess paths 全文件验证

一句话：本组不新增代码，只跑 preprocess 路径测试全文件，确认本轮 placeholder、fallback 和 resize skip 补测整体稳定。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py`：`32 passed`。

仍不能对外说：

- 不能说真实 preprocess 被执行；测试仍通过 monkeypatch、小图片 fixture 或 helper 层验证。

### EXT 当前硬门槛盘点 13

一句话：path/preprocess 验证继续通过，但耗时仍不足 3 小时。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4605`，仍小于 `10800`。
- 最近验证包括 env/preview `39 passed` 和 preprocess paths 全文件 `32 passed`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第三十二组：WebUI output_root 绝对路径 `..` 拒绝护栏

一句话：本组把 WebUI 全局 `output_root` 的 `..` 拒绝规则从相对路径扩展到绝对路径。

源码护栏：

- `web/services/settings_service.py::_normalize_output_root()`
  - 先检查 `Path(clean).parts` 是否包含 `..`。
  - 绝对路径和相对路径只要包含 `..` 都抛出 `ValueError("输出文件夹不能包含 ..")`。
  - 普通绝对路径仍允许保存和解析。

新增测试：

- `test_global_settings_reject_absolute_output_root_parent_traversal`
  - 设置已有安全 `output_root = "safe/runs"`。
  - 尝试保存绝对路径形式的 `safe/../outside`。
  - 确认抛错，并且原配置不被污染。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py -k "absolute_output_root_parent_traversal or output_root_parent_traversal or global_settings"`：`3 passed, 23 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_env_config_paths.py`：`40 passed`。
- `timeout 60 .venv/bin/python -m ruff check web/services/settings_service.py tests/test_preview_service.py library/env.py tests/test_env_config_paths.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile web/services/settings_service.py tests/test_preview_service.py library/env.py tests/test_env_config_paths.py`：通过，ruff 为 `All checks passed!`。
- `git diff --check -- web/services/settings_service.py tests/test_preview_service.py library/env.py tests/test_env_config_paths.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说普通绝对 output_root 被禁止；仍允许不含 `..` 的绝对路径。
- 不能说真实 WebUI 设置页面被浏览器验证；本组是服务层单元测试。

### EXT 当前硬门槛盘点 14

一句话：WebUI 路径安全继续收紧，但当前耗时仍只有约 4738 秒。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4738`，仍小于 `10800`。
- 最近 preview/env 组合验证 `40 passed`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第三十三组：preview 目录绝对路径 `..` 拒绝护栏

一句话：本组把 preview 的 training/inference/custom 目录规范化也收紧为路径文本包含 `..` 就拒绝。

源码护栏：

- `web/services/preview_service.py`
  - `_normalize_preview_dir()` 在绝对路径分支前检查 `..`。
  - `_normalize_project_file()` 在绝对路径和相对路径分支前统一检查 `..`。
  - 普通绝对 inference/custom 目录仍允许。

新增测试：

- `test_preview_settings_reject_absolute_preview_dir_parent_traversal`
  - 已有 preview settings 包含安全 training/inference/custom。
  - 保存绝对 inference 路径 `inference/../outside` 时抛错。
  - 保存绝对 custom 路径 `custom/../outside` 时抛错。
  - 失败后确认旧 settings 没被污染。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py -k "absolute_preview_dir_parent_traversal or preview_settings_reject or allow_absolute"`：`3 passed, 24 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_env_config_paths.py`：第一次 `41 passed`，修正 ruff 后复跑仍为 `41 passed`。

仍不能对外说：

- 不能说普通绝对 inference/custom 目录被禁止；仍允许不含 `..` 的绝对路径。
- 不能说浏览器设置页已验证；本组是服务层测试。

### EXT 第三十四组：preview_service ruff 暴露无用变量清理

一句话：本组在对 touched 文件跑 ruff 时发现并清理 `_weight_sort_key()` 里未使用的 `epoch` 变量。

源码清理：

- `web/services/preview_service.py::_weight_sort_key()`
  - 删除未使用局部变量 `epoch`。
  - 排序 tuple 不变，行为不变。

本组验证：

- `timeout 60 .venv/bin/python -m ruff check web/services/preview_service.py tests/test_preview_service.py web/services/settings_service.py library/env.py tests/test_env_config_paths.py`：修复后 `All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile web/services/preview_service.py tests/test_preview_service.py web/services/settings_service.py library/env.py tests/test_env_config_paths.py`：通过。
- `git diff --check -- web/services/preview_service.py tests/test_preview_service.py web/services/settings_service.py library/env.py tests/test_env_config_paths.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说 preview_service 做了行为重构；这里只是删除未使用变量。

### EXT 当前硬门槛盘点 15

一句话：WebUI preview 路径安全继续加固，但耗时仍不到 10800 秒。

当前事实：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4923`，仍小于 `10800`。
- 最近 preview/env 组合验证 `41 passed`，ruff 和 py_compile 通过。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

### EXT 第三十五组：weight analysis 权重路径 `..` 拒绝护栏

一句话：本组把权重分析服务的路径解析收紧为绝对/相对路径只要包含 `..` 都拒绝。

源码护栏：

- `web/services/weight_analysis_service.py::resolve_analysis_weight()`
  - 在绝对路径和相对路径分支前统一检查 `Path(clean).parts`。
  - 即使 `resolve()` 后会落到允许目录，只要用户输入路径文本含 `..` 就抛出 `ValueError("权重路径不能包含 ..")`。

新增测试：

- `test_invalid_missing_and_escaped_paths_are_rejected`
  - 追加一个真实合法 output root 下的 `safe.safetensors`。
  - 使用 `nested/../safe.safetensors` 形式调用 `inspect_weight()`。
  - 确认因为输入路径包含 `..` 被拒绝。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_weight_analysis_service.py -k "invalid_missing_and_escaped_paths"`：`1 passed, 6 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_weight_analysis_service.py`：`7 passed`。
- `timeout 60 .venv/bin/python -m ruff check web/services/weight_analysis_service.py tests/test_weight_analysis_service.py && PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile web/services/weight_analysis_service.py tests/test_weight_analysis_service.py`：通过，ruff 为 `All checks passed!`。
- `git diff --check -- web/services/weight_analysis_service.py tests/test_weight_analysis_service.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说真实模型或推理已执行；weight analysis 仍只读 safetensors 静态权重。
- 不能说所有上传路径边界都变更；拖入上传字节流仍走独立 `uploaded://` 临时分析路径。

### EXT 第三十六组：WebUI 后端路径组合验证

一句话：本组不新增代码，只把 preview/env/weight-analysis 路径安全相关测试合起来验证，并跑显式 type-check。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_env_config_paths.py tests/test_weight_analysis_service.py`：`48 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/weight_analysis_service.py web/services/preview_service.py web/services/settings_service.py library/env.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- web/services/weight_analysis_service.py tests/test_weight_analysis_service.py web/services/preview_service.py tests/test_preview_service.py web/services/settings_service.py library/env.py tests/test_env_config_paths.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

仍不能对外说：

- 不能说 WebUI 后端全量测试完成；本组是路径安全相关组合。

### EXT 第三十七组：WebUI output-runs 保存路径护栏

一句话：本组只补 `output-runs/save-as` 的路径逃逸 characterization test，不改业务行为。

本组新增：

- 在 `tests/test_web_config_service.py` 新增
  `test_output_run_save_as_rejects_paths_outside_imported_configs`。
- 覆盖 `../escape`、`configs/imported/../escape`、`configs/other/escape`、
  项目内非 imported 绝对路径、项目外绝对路径。
- 验证这些输入不会写入 `configs/imported/escape.toml`、`configs/other/escape.toml`
  或项目外 `escape.toml`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py::test_output_run_save_as_rejects_paths_outside_imported_configs`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "output_run_save_as"`：`3 passed, 167 deselected`。
- `git diff --check -- tests/test_web_config_service.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 5339`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 output-runs 全量行为完成；本组只保护 save-as 目标路径边界。

### EXT 第三十八组：WebUI 队列设置恢复调度保护

一句话：本组只补队列从暂停恢复到继续时的内存级行为测试，不启动训练进程。

本组新增：

- 在 `tests/test_training_queue.py` 新增
  `test_set_queue_settings_unpauses_and_dispatches_waiting_item`。
- 覆盖 `set_queue_settings(paused=False, failure_policy="continue")` 会：
  - 更新 snapshot 的 `paused=False`。
  - 归一化并保存 `failure_policy="continue"`。
  - 对已有 queued item 触发一次 `_schedule_queue_dispatch()`。
  - 写入 `queue.json`，保留队列设置。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py::test_set_queue_settings_unpauses_and_dispatches_waiting_item`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "set_queue_settings"`：`1 passed, 40 deselected`。
- `git diff --check -- tests/test_training_queue.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 5467`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实训练调度已执行；本组只 monkeypatch 观察调度钩子调用。

### EXT 第三十九组：WebUI 队列脏状态归一化保护

一句话：本组补 `get_queue_snapshot()` 对脏队列状态的 characterization test，不改队列实现。

本组新增：

- 在 `tests/test_training_queue.py` 新增
  `test_get_queue_snapshot_normalizes_dirty_queue_state`。
- 覆盖内存队列里存在：
  - 非法 `failure_policy`。
  - 非 dict item。
  - `attempt = 0`。
  - `attempt = "3"`。
  - 缺失 `retry_of`。
- 验证 snapshot 会过滤非 dict item，summary 只统计有效 item，
  `failure_policy` 回退为 `pause`，`attempt` 拉正为正整数，并回写归一化后的
  `svc._queue["items"]`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py::test_get_queue_snapshot_normalizes_dirty_queue_state`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "normalize_queue or get_queue_snapshot"`：`1 passed, 41 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "set_queue_settings or get_queue_snapshot_normalizes_dirty_queue_state"`：`2 passed, 40 deselected`。
- `git diff --check -- tests/test_training_queue.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 5608`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说队列全量回放或真实调度已覆盖；本组只覆盖 snapshot 归一化边界。

### EXT 第四十组：文档归档口径一致性小修

一句话：本组按只读审计结果修正文档归档口径，不移动文件、不改归档内容。

本组修改：

- `docs/README.md` 的 `proposal/` 维护规则从 `_archive/docs/` 精确改为
  `_archive/docs/proposal/`。
- `_archive/docs/proposal/README.md` 的当前实现说明补齐 `features/` 和
  `optimizations/`，与 `docs/archive-index.md` 口径一致。

本组验证：

- `git diff --check -- README.md docs _archive/docs`：通过。
- 逐个比较 7 个归档 proposal 与 `HEAD:docs/proposal/<file>`：无 `DIFF` 输出。
- `rg -n '_archive/docs/proposal/' docs/README.md _archive/docs/proposal/README.md docs/archive-index.md`：
  命中 `docs/README.md` 和 `docs/archive-index.md` 的归档路径。
- `rg -n 'features|optimizations' _archive/docs/proposal/README.md docs/archive-index.md`：
  两处当前实现口径均包含 `features` 和 `optimizations`。

过程说明：

- 本组有一条最初的 `rg` 命令因 shell 反引号转义错误失败，之后已改用上面两条简单
  `rg` 查询复核；失败原因是命令写法，不是文档内容问题。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 5646`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说文档全量链接检查由本组重新完成；本组只做归档口径和内容一致性验证。

### EXT 第四十一组：WebUI output-runs limit 语义保护

一句话：本组补 `list_output_runs(limit=...)` 的当前排序和截断行为断言，不改实现。

本组新增：

- 在 `tests/test_web_config_service.py::test_output_runs_list_reads_direct_run_dirs_sorted`
  中补充：
  - `limit=1` 只返回 mtime 最新的 run。
  - `limit=0` 按当前实现走默认上限，仍返回本测试里的两个 run。

过程说明：

- 初版误以为 `limit=0` 会被 `max(1, limit)` 归为 1，导致断言失败。
- 复读实现后确认 `limit=0` 会先被 `limit or 200` 当成默认值，测试已改成记录当前真实行为。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py::test_output_runs_list_reads_direct_run_dirs_sorted`：
  初版失败 1 次，修正后 `1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "output_runs_list or output_run_save_as"`：
  初版失败 1 次，修正后 `4 passed, 166 deselected`。
- `git diff --check -- tests/test_web_config_service.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 5737`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 output-runs API 行为已重新设计；本组只锁定现有 limit 行为。

### EXT 第四十二组：launch 禁用 accelerate 时忽略坏细节环境变量

一句话：本组补 `ANIMA_ACCELERATE_LAUNCH=0` 时的直接训练命令护栏，不启动训练。

本组新增：

- 在 `tests/test_launch_config.py` 新增
  `test_direct_training_command_ignores_invalid_accelerate_detail_env_when_launch_disabled`。
- 覆盖当 `ANIMA_ACCELERATE_LAUNCH="0"` 且
  `ANIMA_ACCELERATE_NUM_PROCESSES="many"`、`ANIMA_ACCELERATE_MIXED_PRECISION="fp4"` 时，
  `accelerate_training_command_prefix()` 仍返回 `["python", "train.py"]`。
- 这个测试锁定当前语义：只有显式 truthy 的 `ANIMA_ACCELERATE_LAUNCH` 才会解析 accelerate
  细节参数。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py::test_direct_training_command_ignores_invalid_accelerate_detail_env_when_launch_disabled`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py -k "direct_training_command or falsey_accelerate_launch"`：`8 passed, 18 deselected`。
- `git diff --check -- tests/test_launch_config.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 5737`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说训练已启动或 accelerate 可用性已验证；本组只验证命令前缀构造。

### EXT 第四十三组：runtime home 相对路径锚点保护

一句话：本组补 `resolve_under_home()` 的相对路径锚点测试，保护从任意 cwd 调用时的路径语义。

本组新增：

- 在 `tests/test_env_config_paths.py` 新增
  `test_resolve_under_home_uses_anima_home_for_relative_paths`。
- 覆盖 `ANIMA_HOME` 指向临时 repo home、当前 cwd 指向另一个目录时：
  - `resolve_under_home("output/runs")` 解析到 `ANIMA_HOME/output/runs`。
  - 绝对路径输入保持绝对路径不被重新锚定。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py::test_resolve_under_home_uses_anima_home_for_relative_paths`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py -k "resolve_under_home or configs_root or training_queue_root"`：`12 passed, 3 deselected`。
- `git diff --check -- tests/test_env_config_paths.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6055`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说所有运行时路径都重新验证；本组只覆盖 `resolve_under_home()` 相对/绝对边界。

### EXT 第四十四组：ChimeraHydra from_weights metadata 报错保护

一句话：本组补 ChimeraHydra checkpoint metadata 缺失和 K_c/K_f 不一致的报错测试。

本组新增：

- 在 `tests/test_network_cfg.py` 新增
  `test_chimera_from_weights_rejects_missing_or_mismatched_pool_metadata`。
- 覆盖 `is_chimera_hydra=True` 时：
  - 缺 `num_experts_content` 会报 `missing ss_num_experts_content`。
  - `num_experts_content + num_experts_freq` 与 `hydra_num_experts` 不一致会报
    `K_c + K_f mismatch`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py::test_chimera_from_weights_rejects_missing_or_mismatched_pool_metadata`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "chimera_from_weights"`：`7 passed, 30 deselected`。
- `git diff --check -- tests/test_network_cfg.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6139`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说改了 LoRA checkpoint key 或加载语义；本组只锁定已有 metadata 报错边界。

### EXT 第四十五组：LoRA 三轴 stamp 未知值拒绝保护

一句话：本组补 MoE checkpoint 三轴 metadata 的未知值拒绝测试，不改变三轴语义。

本组新增：

- 在 `tests/test_network_cfg.py` 新增
  `test_from_weights_rejects_unknown_three_axis_stamp_values`。
- 覆盖：
  - `new_use_moe_style="true"` 会因未知 `use_moe_style` 报错。
  - `new_router_source="crossattn"` 会因未知 `router_source` 报错。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py::test_from_weights_rejects_unknown_three_axis_stamp_values`：`2 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "three_axis"`：`4 passed, 35 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py -k "chimera_from_weights or three_axis"`：`11 passed, 28 deselected`。
- `git diff --check -- tests/test_network_cfg.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6216`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说旧 checkpoint 兼容面扩大；本组只确认坏 metadata 不会静默改语义。

### EXT 第四十六组：docs 本地链接和可达性门禁复跑

一句话：本组不改代码，只复跑 docs 本地链接、可达性和归档索引门禁。

本组验证：

- `timeout 60 .venv/bin/python - <<'PY' ...` 本地 Markdown 链接扫描：
  `scanned=113 local_links=280 external_links=41 broken=0`。
- `timeout 60 .venv/bin/python - <<'PY' ...` docs 可达性和归档索引检查：
  `docs_md=104 reachable_from_docs_readme=104 missing=0`，
  `archive_proposals=7 missing_in_docs_archive_index=0 missing_in_archive_readme=0`。
- `timeout 60 git diff --check -- README.md docs _archive/docs`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6291`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说外部链接已验证；本组只检查本地 Markdown 链接和可达性。

### EXT 第四十七组：type-check 显式 flag 无分隔符转发保护

一句话：本组补 `tasks.py type-check` 命令壳对无 `--` 显式 pyright flag 和多目标的转发测试。

本组新增：

- 在 `tests/test_type_check_targets.py` 新增
  `test_cmd_type_check_accepts_explicit_flags_without_separator`。
- 覆盖 `cmd_type_check(["--warnings", "library/runtime/launch.py", "library/env.py"])`
  会转成 `python -m pyright --warnings library/runtime/launch.py library/env.py`，
  不混入默认白名单。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py::test_cmd_type_check_accepts_explicit_flags_without_separator`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：`8 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check -- --warnings library/runtime/launch.py library/env.py`：
  `0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_type_check_targets.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6408`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说默认 type-check 扩到全仓；本组只验证显式参数转发。

### EXT 第四十八组：launch train_script Path 字符串化保护

一句话：本组补 `accelerate_training_command_prefix()` 对 `Path` 类型训练脚本参数的行为测试。

本组新增：

- 在 `tests/test_launch_config.py` 新增
  `test_accelerate_command_stringifies_path_train_script`。
- 覆盖：
  - direct 模式下 `Path("train.py")` 输出为 `"train.py"`。
  - accelerate launch 模式下命令最后一项也是 `"train.py"`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py::test_accelerate_command_stringifies_path_train_script`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py`：`27 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check library/runtime/launch.py`：
  `0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_launch_config.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6477`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实训练启动已覆盖；本组只验证命令列表构造。

### EXT 第四十九组：runtime / LoRA / type-check 中宽验证

一句话：本组不新增代码，只复跑 runtime、launch、LoRA config 和 type-check 相关中宽验证。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_launch_config.py tests/test_network_cfg.py tests/test_type_check_targets.py`：`90 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check library/env.py library/runtime/launch.py networks/lora_anima/config.py scripts/tasks/utilities.py`：`0 errors, 0 warnings, 0 informations`。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6887`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说这是 R6 最终总验证；本组只是 EXT 中宽验证。
- 不能说目标已完成；耗时硬门槛仍未满足。

### EXT 第五十组：`.env` loader 基础解析保护

一句话：本组给项目最底层 `.env` loader 补 characterization test，确认已有环境变量不会被文件覆盖。

本组新增：

- 在 `tests/test_env_config_paths.py` 新增
  `test_load_dotenv_preserves_existing_env_and_parses_simple_file`。
- 覆盖：
  - 注释、空行、无 `=` 的坏行会被忽略。
  - 单引号和双引号包裹的值会去掉外层引号。
  - `KEY = value` 会 trim 空白。
  - 已在 `os.environ` 里的 key 不会被 `.env` 覆盖。
  - `load_dotenv()` 返回值只包含本次真正新增的 key。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py::test_load_dotenv_preserves_existing_env_and_parses_simple_file`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py -k "load_dotenv or resolve_under_home or configs_root or training_queue_root"`：`13 passed, 3 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_env_config_paths.py tests/test_web_config_service.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_env_config_paths.py tests/test_web_config_service.py`：通过。
- `git diff --check -- tests/test_env_config_paths.py tests/test_web_config_service.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6887`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 `.env` 支持 shell 插值；当前 loader 明确是字面值解析。
- 不能说所有环境变量入口都已全量覆盖；本组只保护 `.env` 基础读取行为。

### EXT 第五十一组：WebUI output-runs 输出根目录边界保护

一句话：本组补 `list_output_runs()` 在输出根目录缺失或是文件时的当前行为测试。

本组新增：

- 在 `tests/test_web_config_service.py` 新增
  `test_output_runs_list_handles_missing_or_file_output_root`。
- 覆盖：
  - `output_root` 不存在时返回 `ok=True`、`runs=[]`，并保留稳定的 `output_root` 和
    `output_root_abs`。
  - `output_root` 是普通文件时抛出 `ValueError("输出文件夹不是目录")`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py::test_output_runs_list_handles_missing_or_file_output_root`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "output_runs_list or output_run_save_as"`：`5 passed, 166 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_env_config_paths.py tests/test_web_config_service.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_env_config_paths.py tests/test_web_config_service.py`：通过。
- `git diff --check -- tests/test_env_config_paths.py tests/test_web_config_service.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 6887`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 output-runs API 行为已重新设计；本组只锁定现有列表边界。
- 不能说真实输出目录或用户运行结果被读写；本组只使用 `tmp_path`。

### EXT 第五十二组：WebUI 主题切换 DOM 契约保护

一句话：本组给 WebUI app shell 的主题切换按钮补静态契约测试，不启动浏览器。

本组新增：

- 在 `tests/test_training_frontend_state.py` 新增
  `test_app_shell_theme_toggle_contract_matches_index_html`。
- 覆盖：
  - `index.html` 里存在 `theme-toggle` 和 `theme-toggle-text`。
  - 主题按钮初始 `type="button"`、`aria-pressed="false"`。
  - `theme.js` 会设置 `root.dataset.theme`、按钮 `aria-pressed`、按钮 `title`、label 文案。
  - `initThemeToggle()` 绑定 click，并继续调用 localStorage 和 loss chart `setTheme` 回调。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py::test_app_shell_theme_toggle_contract_matches_index_html`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "theme or dom or selector or queue or history"`：`18 passed, 54 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_frontend_state.py`：通过。
- `git diff --check -- tests/test_training_frontend_state.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 7046`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实浏览器主题切换已验证；本组是静态源码/DOM 契约测试。
- 不能说 WebUI 前端全量完成；这里只保护 app shell 主题按钮契约。

### EXT 第五十三组：launch 显式 env 映射隔离保护

一句话：本组补训练 launch helper 的显式 env 隔离测试，确认不会误读真实进程环境。

本组新增：

- 在 `tests/test_launch_config.py` 新增
  `test_explicit_env_mapping_isolated_from_process_env`。
- 覆盖真实 `os.environ` 已设置：
  - `ANIMA_ACCELERATE_LAUNCH=1`
  - `ANIMA_ACCELERATE_NUM_PROCESSES=8`
  - `ANIMA_ACCELERATE_MIXED_PRECISION=fp16`
- 当调用方显式传入 `env={}` 时，`accelerate_training_command_prefix()` 仍返回
  `["python", "train.py"]`，不读取外部进程环境。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py::test_explicit_env_mapping_isolated_from_process_env`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py`：`28 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_launch_config.py library/runtime/launch.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_launch_config.py library/runtime/launch.py`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check library/runtime/launch.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_launch_config.py library/runtime/launch.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 7187`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实 accelerate 或训练进程启动过；本组只测试命令列表构造。
- 不能说 launch 全链路覆盖真实 GPU；本组完全不碰 GPU 和子进程。

### EXT 第五十四组：WebUI 队列失败 continue 策略保护

一句话：本组补队列运行项失败时 `failure_policy="continue"` 不暂停队列的测试。

本组新增：

- 在 `tests/test_training_queue.py` 新增
  `test_queue_process_error_continues_when_failure_policy_continue`。
- 复用 fake process，不启动真实训练。
- 覆盖：
  - 当前运行项 `q1` 的 fake process 返回 `7`。
  - `q1` 进入 `error`。
  - 等待项 `q2` 仍保持 `queued`。
  - 队列 `paused` 保持 `False`。
  - 服务状态回到 `idle`。
  - `_schedule_queue_dispatch()` 被调用一次，允许后续调度继续推进。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py::test_queue_process_error_continues_when_failure_policy_continue`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "queue_process_error or set_queue_settings or get_queue_snapshot_normalizes_dirty_queue_state"`：`4 passed, 39 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py`：`43 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_queue.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_queue.py web/services/training/live_monitor.py web/services/training/queue.py`：通过。
- `git diff --check -- tests/test_training_queue.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 7400`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实训练调度已执行；本组只用 fake process 和 monkeypatch。
- 不能说所有队列失败策略都全链路覆盖；本组只锁定 continue 分支的失败后行为。

### EXT 第五十五组：LoRA ReFT layer spec 解析保护

一句话：本组给 LoRA loading helper 的 `_parse_reft_layers()` 补纯函数测试。

本组新增：

- 在 `tests/test_lora_loading_keys.py` 新增：
  - `test_parse_reft_layers_supported_specs`
  - `test_parse_reft_layers_rejects_invalid_stride`
  - `test_parse_reft_layers_rejects_out_of_range_indices`
- 覆盖：
  - `None` / `"all"` / `""` 返回全部 block。
  - `"last_2"` / `"first_2"` / `"stride_2"` 返回对应 block 列表。
  - `"3,1,3"` 和 `[2, 0]` 会去重并排序。
  - `"stride_0"` 和越界索引会明确抛 `ValueError`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py -k parse_reft_layers`：`10 passed, 28 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py`：`38 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_loading_keys.py tests/test_network_cfg.py -k "parse_reft_layers or three_axis or chimera_from_weights"`：`21 passed, 56 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_loading_keys.py networks/lora_anima/loading.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_lora_loading_keys.py networks/lora_anima/loading.py`：通过。
- `git diff --check -- tests/test_lora_loading_keys.py networks/lora_anima/loading.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 7530`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 ReFT 训练或推理全链路已验证；本组只测 loading helper 纯函数。
- 不能说 LoRA checkpoint key 或 public API 有变更；本组只新增测试。

### EXT 第五十六组：type-check 目标清单完整性保护

一句话：本组给默认 pyright pilot gate 的目标列表补清单完整性测试。

本组新增：

- 在 `tests/test_type_check_targets.py` 新增
  `test_type_check_targets_are_unique_existing_relative_paths`。
- 覆盖：
  - `TYPE_CHECK_TARGETS` 没有重复项。
  - 每个目标都是仓库相对路径。
  - 目标路径不包含 `..`。
  - 每个目标在当前仓库真实存在。

过程说明：

- 初版错误假定 `scripts.tasks.utilities` 暴露 `ROOT`，测试失败 1 次，已改为从
  `tests/test_type_check_targets.py` 的 `__file__` 计算仓库根。
- 第二版把拼接后的绝对路径拿去断言“不是绝对路径”，测试失败 1 次，已改为检查
  `Path(target).is_absolute()`。
- 两次失败都是新增测试写法问题，不是 `TYPE_CHECK_TARGETS` 行为问题。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py::test_type_check_targets_are_unique_existing_relative_paths`：修正后 `1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_type_check_targets.py`：修正后 `9 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_type_check_targets.py scripts/tasks/utilities.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_type_check_targets.py scripts/tasks/utilities.py`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check scripts/tasks/utilities.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_type_check_targets.py scripts/tasks/utilities.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 7639`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说默认 type-check 已扩大到全仓；本组只保护现有白名单质量。
- 不能说这等同于全项目静态类型验证；本组只跑了目标清单测试和 `scripts/tasks/utilities.py`。

### EXT 第五十七组：跨子系统中宽验证

一句话：本组不新增代码，只把最近新增的 runtime/env、LoRA loading、queue、前端和 type-check 切片合起来验证。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_launch_config.py tests/test_env_config_paths.py tests/test_type_check_targets.py tests/test_lora_loading_keys.py`：`91 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py tests/test_training_frontend_state.py -k "queue_process_error or set_queue_settings or get_queue_snapshot_normalizes_dirty_queue_state or theme or dom or selector or queue or history"`：`61 passed, 54 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check library/env.py library/runtime/launch.py networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_launch_config.py tests/test_env_config_paths.py tests/test_type_check_targets.py tests/test_lora_loading_keys.py tests/test_training_queue.py tests/test_training_frontend_state.py tests/test_web_config_service.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。
- `git status --short --branch`：仍在 `main...webui/main`，有目标相关未提交改动，未 stage。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 7709`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说这是 R6 最终总验证；本组只是 EXT 中宽验证。
- 不能说全仓测试已跑；这里只跑了目标相关切片。

### EXT 第五十八组：WebUI GPU picker DOM 契约保护

一句话：本组给 WebUI app shell 的 GPU 选择器补静态 DOM / aria / API 契约测试。

本组新增：

- 在 `tests/test_training_frontend_state.py` 新增
  `test_app_shell_gpu_picker_contract_matches_index_html`。
- 覆盖：
  - `index.html` 里存在 `gpu-picker`、`gpu-picker-toggle`、`gpu-picker-panel`、
    `gpu-all-checkbox`、`gpu-option-list`、`gpu-picker-note`。
  - toggle 初始 `type="button"`、`aria-expanded="false"`。
  - `gpu-picker.js` 读取这些 DOM id。
  - `gpu-picker.js` 仍调用 `/api/training/gpus`，并保留 `file:` 静态打开兜底。
  - 打开/关闭 panel 时会维护 `aria-expanded`。
  - CSS 仍包含 `.gpu-picker-toggle[aria-expanded="true"]` 样式入口。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py::test_app_shell_gpu_picker_contract_matches_index_html`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "gpu_picker or theme or dom or selector or queue or history"`：`19 passed, 54 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_frontend_state.py`：通过。
- `git diff --check -- tests/test_training_frontend_state.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 7979`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实浏览器 GPU picker 已交互验证；本组是静态契约测试。
- 不能说真实训练 GPU 白名单已执行；本组不启动训练。

### EXT 第五十九组：WebUI 队列 compact 保护和静态声明补齐

一句话：本组补队列裁剪保护测试，并修复 ruff 暴露的 `_int_or_none` 静态绑定漏项。

本组新增测试：

- 在 `tests/test_training_queue.py` 新增
  `test_compact_queue_preserves_waiting_and_running_over_limit`。
- 覆盖：
  - 临时把 `MAX_QUEUE_ITEMS` 调小到 `2`。
  - 队列里有 3 个受保护项：`queued` / `running` / `queued`。
  - `_compact_queue()` 不会为了满足上限丢掉 waiting/running 项。
  - 终态 `done` / `error` 项会被裁掉。

本组源码小修：

- `web/services/training/queue.py` 的 `TYPE_CHECKING` import 列表补入 `_int_or_none`。
- 这是静态声明补齐；运行时仍由 `_bind_legacy()` 绑定 facade helper，不改运行逻辑。

过程说明：

- 初次运行 `ruff check tests/test_training_queue.py web/services/training/queue.py` 时发现
  `web/services/training/queue.py:309` 的 `_int_or_none` 静态未定义。
- 该问题不是本组测试引入，但属于当前队列/type-check 清理范围，已用最小改动修复。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py::test_compact_queue_preserves_waiting_and_running_over_limit`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "compact_queue or queue_process_error or get_queue_snapshot_normalizes_dirty_queue_state"`：`4 passed, 40 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py`：`44 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_queue.py web/services/training/queue.py`：修复后 `All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_queue.py web/services/training/queue.py`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/training/queue.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_training_queue.py web/services/training/queue.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 8451`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实队列文件或真实训练任务被清理；本组只用内存队列和 `tmp_path`。
- 不能说队列系统全量行为已完成；本组只保护 compact 边界和一个静态声明漏项。

### EXT 第六十组：tasks.py inline env 非法形态转发保护

一句话：本组补 CLI 命令壳对非大写内联环境变量 token 的转发测试。

本组新增：

- 在 `tests/test_tasks_runner.py` 新增
  `test_tasks_main_forwards_non_uppercase_inline_env_tokens`。
- 覆盖：
  - `method=lora` 小写 key 不写入 `os.environ`。
  - `Mixed_KEY=value` 混合大小写 key 不写入 `os.environ`。
  - `1BAD=value` 数字开头 key 不写入 `os.environ`。
  - 以上 token 都原样转发给子命令。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py::test_tasks_main_forwards_non_uppercase_inline_env_tokens`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_tasks_runner.py tests/test_type_check_targets.py`：`21 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_tasks_runner.py tests/test_type_check_targets.py tasks.py scripts/tasks/utilities.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_tasks_runner.py tests/test_type_check_targets.py tasks.py scripts/tasks/utilities.py`：通过。
- `timeout 60 .venv/bin/python tasks.py --help >/tmp/anima_tasks_help_check_ext60.txt && wc -c /tmp/anima_tasks_help_check_ext60.txt`：通过，输出大小 `10534`。
- `git diff --check -- tests/test_tasks_runner.py tests/test_type_check_targets.py tasks.py scripts/tasks/utilities.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 8546`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 CLI inline env 规则改了；本组只锁定现有规则。
- 不能说所有任务命令都执行过；本组只跑命令壳测试和 help smoke。

### EXT 第六十一组：LoRA from_weights plugin_args 顶层复制保护

一句话：本组给 `LoRANetworkCfg.from_weights()` 的 `plugin_args` 顶层复制行为补测试。

本组新增：

- 在 `tests/test_network_cfg.py` 新增
  `test_from_weights_copies_plugin_args_without_top_level_aliasing`。
- 覆盖：
  - `from_weights(..., plugin_args=plugin_args)` 会保留传入键值。
  - `cfg.plugin_args` 不是原始 dict 对象。
  - 调用后修改原始 dict 的顶层值，不会污染 `cfg.plugin_args`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py::test_from_weights_copies_plugin_args_without_top_level_aliasing`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py`：`40 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_network_cfg.py tests/test_lora_loading_keys.py -k "plugin_args or parse_reft_layers or three_axis or chimera_from_weights"`：`22 passed, 56 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_network_cfg.py networks/lora_anima/config.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_network_cfg.py networks/lora_anima/config.py`：通过。
- `git diff --check -- tests/test_network_cfg.py networks/lora_anima/config.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 8702`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 `plugin_args` 做了深拷贝；本组只确认顶层 dict 不共用对象。
- 不能说 LoRA checkpoint key 或 public API 有变更；本组只新增测试。

### EXT 第六十二组：queue / LoRA / CLI 相关组合验证

一句话：本组不新增代码，只把队列、LoRA loading/config、CLI/type-check 相关测试组合起来验证。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py tests/test_network_cfg.py tests/test_lora_loading_keys.py tests/test_tasks_runner.py tests/test_type_check_targets.py`：`143 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/training/queue.py networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md tests/test_training_queue.py web/services/training/queue.py tests/test_network_cfg.py tests/test_lora_loading_keys.py tests/test_tasks_runner.py tests/test_type_check_targets.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 8788`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说这是 R6 最终总验证；本组只是 EXT 相关组合验证。
- 不能说全仓 pytest 已完成；本组只跑了当前改动相关测试文件。

### EXT 第六十三组：归档 proposal 索引不漏项测试

一句话：本组给归档 proposal 文档补索引完整性测试，防止归档文件只搬家但不进索引。

本组新增：

- 新增 `tests/test_docs_archive_indexes.py`。
- 新增测试 `test_archived_proposals_are_listed_in_archive_indexes`。
- 覆盖：
  - `_archive/docs/proposal/*.md` 中除 `README.md` 外的归档文档列表非空。
  - 每个归档 proposal 文件名都出现在 `docs/archive-index.md`。
  - 每个归档 proposal 文件名都出现在 `_archive/docs/proposal/README.md`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_docs_archive_indexes.py`：`1 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_docs_archive_indexes.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_docs_archive_indexes.py`：通过。
- `git diff --check -- README.md docs _archive/docs tests/test_docs_archive_indexes.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 8864`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说外部链接已验证；本组只测本地归档索引完整性。
- 不能说活跃 `docs/proposal/` 已清空；该目录仍保留半活跃提案。

### EXT 第六十四组：env 嵌套结构变量展开保护

一句话：本组给 `expand_env_vars_in_obj()` 补嵌套 dict/list/tuple 递归展开测试。

本组新增：

- 在 `tests/test_env_config_paths.py` 新增
  `test_expand_env_vars_in_obj_recurses_nested_structures`。
- 覆盖：
  - dict 内 `$ANIMA_NESTED_ROOT/data` 会展开。
  - list 内 `~/cache` 会按 `HOME` 展开。
  - tuple 内 `$ANIMA_NESTED_ROOT/a` 会展开。
  - 普通字符串和非字符串值保持不变。
  - tuple 递归后仍保持 tuple 类型。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py::test_expand_env_vars_in_obj_recurses_nested_structures`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py`：`17 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_env_config_paths.py library/env.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_env_config_paths.py library/env.py`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check library/env.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_env_config_paths.py library/env.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 8955`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说所有配置对象路径语义都已全量覆盖；本组只测试 env expansion helper。
- 不能说 `.env` 支持 shell 插值；当前仍是字面值 env 展开。

### EXT 第六十五组：WebUI output-run 缺失固定配置错误路径

一句话：本组给 output-run 读取 runtime/dataset 固定文件缺失时的错误路径补测试。

本组新增：

- 在 `tests/test_web_config_service.py` 新增
  `test_output_run_read_reports_missing_fixed_config_file`。
- 覆盖：
  - 运行目录存在且有 `config.original.toml`。
  - 缺少 `config.runtime.toml` 时，读取 `runtime` 抛 `FileNotFoundError("运行配置不存在")`。
  - 缺少 `dataset.runtime.toml` 时，读取 `dataset` 抛同类错误。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py::test_output_run_read_reports_missing_fixed_config_file`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "output_run_read or output_runs_list or output_run_save_as"`：`7 passed, 165 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "output_run or file_groups_direct_path_helpers or file_groups_glob"`：`16 passed, 156 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_web_config_service.py web/services/config/output_runs.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_web_config_service.py web/services/config/output_runs.py`：通过。
- `git diff --check -- tests/test_web_config_service.py web/services/config/output_runs.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9118`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 output-runs API 行为已重设计；本组只锁定当前缺失文件错误路径。
- 不能说真实 output/runs 用户目录被读取；本组只用 `tmp_path`。

### EXT 第六十六组：WebUI 队列非法清理状态拒绝保护

一句话：本组补队列批量清理接口对非法状态集合的拒绝测试。

本组新增：

- 在 `tests/test_training_queue.py` 新增
  `test_clear_queue_items_by_state_rejects_invalid_states_without_saving`。
- 覆盖：
  - `clear_queue_items_by_state({"error"})` 抛 `ValueError("只能清理已完成或已取消")`。
  - `clear_queue_items_by_state(set())` 同样抛错。
  - 失败后内存队列列表不变。
  - 失败后不会写入临时 `queue.json`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py::test_clear_queue_items_by_state_rejects_invalid_states_without_saving`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "clear_queue_items_by_state or clear_finished or clear_completed or clear_canceled"`：`2 passed, 43 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py`：`45 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_queue.py web/services/training/queue.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_queue.py web/services/training/queue.py`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/training/queue.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_training_queue.py web/services/training/queue.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9225`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实队列目录被修改；本组只在 `tmp_path` 下确认失败不落盘。
- 不能说 error 队列项现在可清理；当前规则仍只允许 done/canceled。

### EXT 第六十七组：WebUI 队列 history task id 规范化保护

一句话：本组补 `_attach_history_task_to_queue_item()` 对 history task id 的去重和 list 规范化测试。

本组新增：

- 在 `tests/test_training_queue.py` 新增
  `test_attach_history_task_to_queue_item_deduplicates_and_normalizes_ids`。
- 覆盖：
  - 非 list 的 `history_task_ids` 会被规范成 list。
  - 同一 history task id 连续 attach 不会重复追加。
  - 已包含该 history task id 的队列项不重复追加。
  - 缺失 item id 或空 task id 不会产生额外记录。
  - 成功 attach 后 `message` 更新为 `正在运行`。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py::test_attach_history_task_to_queue_item_deduplicates_and_normalizes_ids`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py -k "attach_history_task or compact_queue or clear_queue_items_by_state"`：`3 passed, 43 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py`：`46 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_queue.py web/services/training/queue.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_queue.py web/services/training/queue.py`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/training/queue.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- tests/test_training_queue.py web/services/training/queue.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9337`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 `_attach_history_task_to_queue_item()` 自己会落盘；当前实现只改内存，保存发生在外层流程。
- 不能说真实 history 任务被创建或修改；本组只操作内存队列。

### EXT 第六十八组：LoRA legacy Ortho save key 转换保护

一句话：本组给 legacy sig-type OrthoLoRA 到标准 LoRA key 的转换 helper 补形态测试。

本组新增：

- 在 `tests/test_lora_save_pipeline.py` 新增
  `test_convert_legacy_ortho_to_lora_replaces_sig_layout_with_standard_keys`。
- 覆盖：
  - legacy `p_layer.weight` / `q_layer.weight` / `lambda_layer` / `base_*` key 被移除。
  - 新增标准 `lora_up.weight` 和 `lora_down.weight`。
  - 输出 shape 分别为 `(out, rank)` 和 `(rank, in)`。
  - 指定 `dtype=torch.float16` 时输出 tensor dtype 为 `torch.float16`。
  - `alpha` 被保留。
- 测试中 monkeypatch `torch.cuda.is_available()` 为 `False`，避免本机 CUDA 能力差异影响 SVD 路径。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py::test_convert_legacy_ortho_to_lora_replaces_sig_layout_with_standard_keys`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py`：`11 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_network_cfg.py -k "legacy_ortho or stacked_experts or chimera or parse_reft_layers or plugin_args"`：`38 passed, 51 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_lora_save_pipeline.py networks/lora_save.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_lora_save_pipeline.py networks/lora_save.py`：通过。
- `git diff --check -- tests/test_lora_save_pipeline.py networks/lora_save.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9467`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说 legacy Ortho 数值重建精度已验证；本组只测 key/shape/dtype/alpha 形态。
- 不能说真实 checkpoint 文件被转换；本组只直接调用 helper 操作内存 state_dict。

### EXT 第六十九组：跨 WebUI / LoRA / CLI / docs 宽组合验证

一句话：本组不新增代码，只把近期新增测试面做一次更宽的组合验证。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_network_cfg.py tests/test_training_queue.py tests/test_env_config_paths.py tests/test_tasks_runner.py tests/test_type_check_targets.py tests/test_docs_archive_indexes.py`：`174 passed`。
- `timeout 60 .venv/bin/python tasks.py type-check library/env.py web/services/training/queue.py networks/lora_save.py networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_network_cfg.py tests/test_training_queue.py tests/test_env_config_paths.py tests/test_tasks_runner.py tests/test_type_check_targets.py tests/test_docs_archive_indexes.py web/services/training/queue.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9524`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说这是 R6 最终总验证；本组仍是 EXT 中宽/宽组合验证。
- 不能说全仓 pytest 已完成；本组只覆盖当前新增和相关子系统测试。

### EXT 第七十组：WebUI 顶层 tab 按钮和内容区契约保护

一句话：本组给 WebUI 顶层 tab 的 `data-tab` 按钮和对应 `tab-*` 内容区补静态测试。

本组新增：

- 在 `tests/test_training_frontend_state.py` 新增
  `test_top_level_tab_buttons_have_matching_content_sections`。
- 覆盖：
  - 顶层 tab 按钮仍是 `config`、`datasets`、`training`、`weight-analysis`、
    `settings`、`environment`、`image-test`。
  - 每个顶层 `data-tab` 都有对应 `id="tab-..."` section。
  - `preview` 不作为顶层按钮出现。
  - `tabs.js` 仍保留 training/config fallback，并会把隐藏 `tab-preview` 移出 active 状态。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py::test_top_level_tab_buttons_have_matching_content_sections`：`1 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py -k "top_level_tab or gpu_picker or theme or dom or selector or queue or history"`：`20 passed, 54 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_training_frontend_state.py`：通过。
- `git diff --check -- tests/test_training_frontend_state.py docs/findings/project_cleanup_checkpoint_20260705.md`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9618`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实浏览器 tab 交互已验证；本组是静态 DOM/source 契约测试。
- 不能说前端全量页面都做了视觉验证；本组只锁顶层 tab 对应关系。

### EXT 第七十一组：WebUI 前端 / output-runs / docs 组合验证

一句话：本组不新增代码，只把 WebUI 前端静态全文件、output-runs 切片和 docs 归档索引组合验证。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py`：`74 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "output_run or file_groups_direct_path_helpers or file_groups_glob"`：`16 passed, 156 deselected`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_docs_archive_indexes.py`：`1 passed`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_frontend_state.py tests/test_web_config_service.py tests/test_docs_archive_indexes.py`：`All checks passed!`。
- `git diff --check -- README.md docs _archive/docs tests/test_docs_archive_indexes.py tests/test_web_config_service.py tests/test_training_frontend_state.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9723`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实浏览器自动化已跑；前端验证仍是静态/Node 夹具。
- 不能说全量 Web config 测试已跑；本组只跑 output-runs/file-groups 切片。

### EXT 第七十二组：docs 本地链接和可达性复跑

一句话：本组不新增代码，只复跑 docs 本地链接、docs 可达性和归档 proposal 索引检查。

过程说明：

- 第一版临时链接脚本误把 `docs/guidelines/difference_between_comfy.md` 中的代码片段
  `self.t_embedder[0](timesteps_B_T` 当成 Markdown 链接，出现 1 个误报。
- 随后改用更保守的路径式链接规则复跑，确认真实本地链接无坏链。

本组验证：

- 保守版本地 Markdown 链接扫描：
  `scanned=114 local_links=262 external_links=41 skipped_non_path=1 broken=0`。
- docs 可达性和归档 proposal 索引检查：
  `docs_md=104 reachable_from_docs_readme=104 missing=0`；
  `archive_proposals=7 missing_in_docs_archive_index=0 missing_in_archive_readme=0`。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9821`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说外部链接已检查；本组只检查本地 Markdown 链接。
- 不能说这是 R6 最终总验证；本组仍是 EXT docs 验证。

### EXT 第七十三组：当前改动相关测试分批宽验证

一句话：本组不新增代码，只把当前改动相关测试按 60 秒限制拆成两批验证，并跑相关 type-check。

本组验证：

- 后端 / LoRA / CLI / docs 批次：
  `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_env_config_paths.py tests/test_launch_config.py tests/test_network_cfg.py tests/test_lora_loading_keys.py tests/test_lora_save_pipeline.py tests/test_training_queue.py tests/test_type_check_targets.py tests/test_tasks_runner.py tests/test_docs_archive_indexes.py`：`202 passed`。
- 前端 / Web config 批次：
  `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py tests/test_web_config_service.py -k "output_run or file_groups_direct_path_helpers or file_groups_glob or top_level_tab or gpu_picker or theme or dom or selector or queue or history"`：`37 passed, 209 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check library/env.py library/runtime/launch.py web/services/training/queue.py networks/lora_save.py networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py web/services/config/output_runs.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md tests/test_env_config_paths.py tests/test_launch_config.py tests/test_network_cfg.py tests/test_lora_loading_keys.py tests/test_lora_save_pipeline.py tests/test_training_queue.py tests/test_training_frontend_state.py tests/test_type_check_targets.py tests/test_tasks_runner.py tests/test_docs_archive_indexes.py tests/test_web_config_service.py web/services/training/queue.py`：通过。
- `git status --short --branch`：仍在 `main...webui/main`，有目标相关未提交改动，未 stage。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 9940`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说全仓 pytest 已完成；本组只跑当前改动相关批次。
- 不能说可以进入 R6；耗时硬门槛仍未满足。

### EXT 第七十四组：路径服务 / preprocess / runtime 相关切片验证

一句话：本组不新增代码，只复跑本轮已改过或关联较强的路径、preview、weight-analysis、preprocess、runtime 和 bootstrap 切片。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_weight_analysis_service.py tests/test_env_config_paths.py`：`51 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preprocess_paths.py tests/test_runtime_harness_cli.py tests/test_training_bootstrap.py -k "runtime or path or placeholder or compile or adapter or apply or preprocess_dataset_rows_expands"`：`46 passed, 9 deselected`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_preview_service.py tests/test_weight_analysis_service.py tests/test_preprocess_paths.py tests/test_runtime_harness_cli.py tests/test_training_bootstrap.py web/services/preview_service.py web/services/weight_analysis_service.py library/env.py`：`All checks passed!`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m py_compile tests/test_preview_service.py tests/test_weight_analysis_service.py tests/test_preprocess_paths.py tests/test_runtime_harness_cli.py tests/test_training_bootstrap.py web/services/preview_service.py web/services/weight_analysis_service.py library/env.py`：通过。
- `timeout 60 .venv/bin/python tasks.py type-check web/services/preview_service.py web/services/weight_analysis_service.py library/env.py`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md tests/test_preview_service.py tests/test_weight_analysis_service.py tests/test_preprocess_paths.py tests/test_runtime_harness_cli.py tests/test_training_bootstrap.py web/services/preview_service.py web/services/weight_analysis_service.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 10021`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实 preprocess 或真实训练被执行；本组仍是测试夹具和 monkeypatch。
- 不能说路径服务全量边界都已证明；本组只覆盖已改路径相关切片。

### EXT 第七十五组：未收口 type-check / diff / 工作区预检

一句话：本组不新增代码，只在未达到耗时门槛前做一次默认 type-check、全量 diff check 和工作区清单预检。

本组验证：

- `timeout 60 .venv/bin/python tasks.py type-check`：`0 errors, 0 warnings, 0 informations`。
- `git diff --check`：通过。
- `git diff --name-only`：输出目标相关改动清单，包括 docs 归档、path/runtime/LoRA/queue/frontend/type-check/tests 等文件。
- `git status --short --branch`：仍在 `main...webui/main`，有目标相关未提交改动，未 stage。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 10079`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说已经 stage；本组只是预检。
- 不能说已经进入 R6；耗时门槛仍未满足。

### EXT 第七十六组：Web config / 前端最终前分块复跑

一句话：本组不新增代码，只复跑 Web config route/output-runs/file-groups 切片和前端静态全文件。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_frontend_state.py`：`74 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_web_config_service.py -k "output_run or raw_put_route_rejects_invalid_toml_without_creating_file or raw_patch_route_rejects_non_object_values or sample_prompts or file_groups"`：`29 passed, 143 deselected`。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md tests/test_web_config_service.py tests/test_training_frontend_state.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 10198`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说全量 Web config 测试已完成；本组仍是切片。
- 不能说浏览器端交互已跑；前端仍是静态/Node 夹具。

### EXT 第七十七组：LoRA / queue / runtime / preprocess 分批验证

一句话：本组不新增代码，只复跑 LoRA、queue、runtime、preprocess、preview、weight-analysis 和 training bootstrap 相关验证。

本组验证：

- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_training_queue.py tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_network_cfg.py`：`135 passed`。
- `PYTHONDONTWRITEBYTECODE=1 timeout 60 .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_preview_service.py tests/test_weight_analysis_service.py tests/test_preprocess_paths.py tests/test_runtime_harness_cli.py tests/test_training_bootstrap.py -k "runtime or path or placeholder or compile or adapter or apply or preview or invalid_missing_and_escaped_paths"`：`76 passed, 13 deselected`。
- `timeout 60 .venv/bin/python tasks.py type-check library/env.py library/runtime/launch.py web/services/training/queue.py web/services/preview_service.py web/services/weight_analysis_service.py networks/lora_save.py networks/lora_anima/config.py networks/lora_anima/loading.py scripts/tasks/utilities.py`：`0 errors, 0 warnings, 0 informations`。
- `timeout 60 .venv/bin/python -m ruff check tests/test_training_queue.py tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_network_cfg.py tests/test_preview_service.py tests/test_weight_analysis_service.py tests/test_preprocess_paths.py tests/test_runtime_harness_cli.py tests/test_training_bootstrap.py`：`All checks passed!`。
- `git diff --check -- docs/findings/project_cleanup_checkpoint_20260705.md tests/test_training_queue.py tests/test_lora_save_pipeline.py tests/test_lora_loading_keys.py tests/test_network_cfg.py tests/test_preview_service.py tests/test_weight_analysis_service.py tests/test_preprocess_paths.py tests/test_runtime_harness_cli.py tests/test_training_bootstrap.py`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 10295`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说真实训练、真实 preprocess 或真实 checkpoint 转换已执行；本组仍是测试夹具。
- 不能说 R6 总验证已经完成；本组仍是 EXT 验证。

### EXT 第七十八组：远端同步和 diff 空白预检

一句话：本组不新增代码，只做发布前的只读远端预检和 diff 空白检查。

本组验证：

- `git fetch webui --prune`：通过。
- `git log --oneline --decorate --max-count=5 webui/main..HEAD`：无输出，本地没有领先 `webui/main` 的已提交内容。
- `git rev-list --left-right --count HEAD...webui/main`：`0 0`，本地 `HEAD` 与 `webui/main` 提交同步。
- `git diff --check`：通过。

当前硬门槛：

- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 10350`，仍小于 `10800`。
- 当前仍不能 `git add`、不能 commit、不能 push、不能 `update_goal complete`。

仍不能对外说：

- 不能说已经发布；本组没有 stage、commit 或 push。
- 不能说可以进入 R6；耗时硬门槛仍未满足。

### EXT 历史硬门槛盘点 16（goal paused 快照）

一句话：这一段是 EXT36 后的历史快照，后续最新状态以上方 EXT37 和之后记录为准。

当前事实：

- 实时 `get_goal` 显示 `goal.status = paused`。
- 实时 `get_goal` 显示 `goal.timeUsedSeconds = 4976`，仍小于 `10800`。
- 最近 WebUI 后端路径组合验证 `48 passed`，显式 type-check `0 errors, 0 warnings, 0 informations`。
- 当前仍没有 stage、commit 或 push。

当前禁止收口：

- 不能 `git add` / `commit` / `push`。
- 不能 `update_goal complete`。
- 不能说 R6 总验证已经完成。

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

---

## 🔚 20. 20260706 目标完成状态补记

一句话：这是 20260706 执行记录中的完成状态补记，真正的最终阅读入口以后续文件末尾状态为准。

当前最终状态：

- `docs/findings/project_cleanup_sustained_goal_20260706.md` 已改为完成归档。
- 本目标的详细阶段、验证和边界记录见第 18 节和第 19 节。
- R6 总验证已通过；其中一次被中断的 Web config 切片已重跑通过，未计入失败/中断结果。
- 最终 Git 收口按本节之后的显式 stage、commit、push 执行，目标远端仍是 `webui/main`。

不要重复执行：

- 不要再把 `project_cleanup_sustained_goal_20260706.md` 当成活跃目标入口。
- 不要重新执行 20260705 的 long-running / next-stage / sustained 旧目标书。
- 后续如果继续清理，应另开新的目标书或使用新的用户指令。

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

## 🔚 17. 2026-07-06 文件末尾最终状态

一句话：这是当前文件真正的最后阅读入口，20260706 跨子系统长跑目标已完成，不要再重复执行旧目标书。

当前最终状态：

- `project_cleanup_sustained_goal_20260705.md` 已完成归档，最终提交为 `bd591b83 test: extend sustained cleanup coverage`。
- `project_cleanup_sustained_goal_20260706.md` 已完成归档，收口前最新 `goal.timeUsedSeconds = 10889`，已满足 `>=10800` 秒硬门槛。
- 20260706 目标已完成 `R0` 到 `R6`，并通过 EXT 扩展到第七十八组以上；阶段数、推进轮数、子系统覆盖和验证密度均满足目标书硬条件。
- R6 总验证、docs 可达性、本地链接、归档索引、type-check、`git diff --check` 和远端同步预检均已通过；详细命令见第 19 节。
- 本目标没有跑真实训练，没有下载模型，没有删除或移动用户数据目录，没有删除 `_legacy.py`，没有改 LoRA checkpoint key、public API 或三轴路由语义。
- 最终 Git 收口按本节后的显式 stage、commit、push 执行，目标远端仍是 `webui/main`。

后续提醒：

- 不要再把 `project_cleanup_sustained_goal_20260706.md` 当成活跃目标入口。
- 不要重新执行 20260705 的 long-running / next-stage / sustained 旧目标书。
- 后续若继续项目清理，应另开新的目标书或等待新的用户指令。
