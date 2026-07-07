# Anima App 剩余全局桥收尾提案

状态：已完成（待归档）
适用版本：当前 main
入口命令：无，本文是 `web/static/js/features/anima-app/` 的收尾重构计划
基线日期：2026-07-07
前序计划：`_archive/docs/proposal/anima-app-deglobalization.md`（阶段 0-3，85 轮）

## 核心结论

一句话：前一批 history / toml-manager / sample-prompts 热点 chunk、runtime globals，以及 history collections workbench 这条 state-heavy 链路已完成显式 bucket 化，`globalThis.` 总量保持 0；`installLegacyStateGlobals(runtime)` 已从生产入口退场，`legacy-globals.js` 兼容桥也已物理删除。

## 当前基线

| 指标 | 数量 |
| --- | --- |
| `globalThis.` 总出现次数 | 0 次 |
| `legacy-globals.js` 文件状态 | 已删除 |
| `legacy-globals.js` 生产入口可达性 | 不可达（`index.js` 已移除 import/call，文件已删除） |
| `legacy-globals.js` repo 内源码 consumer | 0 个（代码路径） |
| `chunks/` 总文件数 | 41 个 |
| `helpers/` 总文件数 | 52 个 |
| `runtime/` 总文件数 | 4 个 |
| `imports.js` | 已删除 |
| 生产代码 `Object.assign(globalThis, ...)` | 0 处 |
| 前端模块 cache token | `module-bootstrap-20260707-93` |

## 当前桥接面

| 文件 | 直接写入 | 职责 |
| --- | ---: | --- |
| `helpers/*-state-bridge.js` | 0 | 显式承接 app shell / config / dataset / history / toml state bucket，已接管生产路径的状态读写 |
| `helpers/runtime-bridge.js` | 0 | 显式承接 `api / datasetPresetApi / val / populateSelect`，替代旧 runtime globals |
## 阶段 9 进度

- `27-render-history-collections-workbench.js` 已切到 `historyState`，history collections workbench 的选择态、搜索态、filters 和 visible task ids 不再依赖 state proxy。
- `29-start-history-config-group-pointer-drag.js`、`30-start-history-collection-pointer-drag.js` 已切到 `historyState` / `trainingState`，拖拽、drop feedback、popover、pending 状态不再直接走全局 state proxy。
- `31-create-history-collection-workbench-card.js`、`32-history-task-collection-label.js`、`33-create-history-task-item.js` 已切到 `historyState`，历史分组卡片、批量选择、collection settings、history item active/resume 状态已从裸状态读写迁到显式 bucket。
- `24-show-preflight-pending-dialog.js`、`25-update-progress.js`、`37-config-training-source.js` 已完成 bridge 对齐并做过定向导入 / pytest 校验；训练前预检、实时进度、training source 审查链当前按 `tomlState` / `trainingState` / `historyState` 读取关键运行态。
- `21-update-toml-selection-ui.js` 已切到 `configState` / `datasetState` / `tomlState` / `trainingState`，当前训练配置文件、未保存变更计算、sample prompts 编辑模式、dataset override、共享切换弹窗 busy/timer 等选择态已不再依赖 state proxy。
- `22-update-toml-action-state.js`、`23-move-current-toml-to-group.js` 已切到 `tomlState` / `trainingState` / `datasetState`，TOML 锁定/删除确认、状态提示计时器、训练入口按钮启停、当前选择清理、删除后 fallback、输出快照模式判断等控制态不再直接走 state proxy。
- `20-can-drop-toml-file-to-group.js` 已切到 `tomlState` / `trainingState`，分组拖拽、当前选择刷新、批量分组操作 busy、当前训练标签等拖拽工作台状态已改成显式 bucket 读取。
- `19-current-sample-prompt-text.js` 已切到 `configState` / `tomlState` / `trainingState`，sample prompts 模式切换、载入序号、training source fallback、dataset apply 后的 TOML 回写都不再依赖 state proxy。
- `06-stronger-selective-checkpoint-value.js` 已切到 `appShellState` / `configState` / `datasetState` / `historyState` / `tomlState` / `trainingState`，config dataset picker、continue-lora 选择弹窗、全局模型路径覆盖、训练来源兼容性提示等 UI 状态不再直接吃 state proxy。
- `13-update-dataset-editor-rows-setting-value.js` 已切到 `appShellState` / `configState` / `datasetState` / `trainingState`，dataset editor scope 选择、current training source、choice guide hint 序号、selection snapshot、兼容字段同步等共享状态已迁到显式 bucket。
- `14-lora-adapter-kind-from-config.js` 已切到 `configState`，LoRA adapter kind 草稿态、precision preference 派生态、sample prompts 表单模式判断、live form choice guide 所需的配置快照都不再直接走 state proxy。
- `15-append-sample-prompt-row.js`、`16-load-output-run-config.js` 已切到 `datasetState` / `tomlState` / `trainingState`，output-run manager、TOML manager 模式切换、快照保存回项目配置的关键状态已从裸引用迁到显式 bucket。
- `04-create-config-group-entry.js` 已切到 `configState`，配置分类目录、sticky actions、group hint 序号、stage resolution state，以及 field help 搜索兜底都不再直接依赖 state proxy。
- `15-append-sample-prompt-row.js` 已补上 `configState.fieldHelp`，表单帮助信息的远端 schema fallback 不再直接读取旧 state proxy。
- `07-render-config-dataset-picker-dialog.js` 已切到 `configState` / `datasetState`，config dataset picker 的搜索词、选中 preset、preview request seq、preview payload、summary、dirty 提示，以及 file-group drag source payload / drop target registry 都不再直接依赖 state proxy。
- `08-origin-closest.js` 已切到 `datasetState`，file-group pointer drag、drop preview、active drop indicator、drop target registry 和 drop payload 读取已从裸 state proxy 迁到显式 bucket。
- `09-setup-config-group-drop-target.js` 已切到 `configState` / `datasetState`，dataset preset groups、dataset editor 默认行、active dataset label、drag target payload 读取，以及 dataset preset / editor state holder 都不再直接依赖 state proxy。
- `10a-dataset-inline-help.js` 已切到 `datasetState`，dataset experimental open key、open state map，以及 preset/config 上下文选择不再直接依赖 `datasetPresetState / datasetEditorState / datasetExperimentalOpenStates` 这组旧 state proxy 入口。
- `11-create-dataset-editor-row.js` 已切到 `datasetState`，dataset preset tab 下预览按钮的启用/禁用判断现在通过显式 bucket 读取 selected file / dirty 状态。
- `12-create-dataset-row-caption-source-mode-editor.js` 已切到 `datasetState`，caption source help seq、dataset preview seq / payload / dialog 标题、preview 所需 preset file，以及 defaults / datasets 回写都已从裸 state proxy 改为显式 bucket 读写。
- `03-parse-network-arg-entry.js` 已切到 `configState` / `datasetState` / `trainingState`，step estimate、dataset editor / preset manager、live chart 统计面板所需的关键序号与状态对象已从 state proxy 迁到显式 bucket。
- `17-apply-selected-dataset-preset-to-current-config.js` 已切到 `configState` / `datasetState` / `tomlState`，数据集预设应用、保存/导入/删除后的选中态、编辑器回写与 summary 刷新不再直接吃 state proxy。
- `18-delete-dataset-preset-group.js` 已切到 `configState` / `datasetState` / `tomlState` / `trainingState`，dataset editor 保存、form patch、sample prompts 持久化路径、训练源 fallback 等共享表单状态已改为显式 bucket 读取。
- `02-ensure-history-detail-feature.js` 已切到 `configState` / `datasetState` / `tomlState` / `trainingState`，config bootstrap、merged config 加载序号、currentConfig / form draft、dataset preset summary、sample prompts 模式/路径/载入序号、当前训练 TOML 对齐，以及 live chart holder 都不再直接依赖 state proxy。
- `34-show-history-collection-select-dialog.js`、`35-render-config-group-timeline.js`、`36-setup-event-listeners.js` 已切到 `trainingState`，历史详情 loss chart 和 live chart 控制开关的 holder 读写已从裸 `lossChart / liveChartState` 改为显式 bucket。
- `index.js` 已移除 `installLegacyStateGlobals(runtime)` 的 import/call，生产入口不再加载 state proxy；随后 `legacy-globals.js` 已物理删除。
- 已补 repo 级源码扫描断言，确认仓内生产/源码路径不再 import `legacy-globals.js`，也不再调用 `installLegacyStateGlobals`。
- 已用与生产入口一致的 query specifier 做过一次 Node 动态导入校验，确认 bridge 配置后的热点 chunk 可以正常加载；最终收口轮再做整份 `tests/test_training_frontend_state.py` 的分半验收。

## 实施边界

一句话：旧 6 个 chunk 已清零，本提案已完成收口；以下边界保留为本轮落地记录。

允许触碰：

- `web/static/js/features/anima-app/chunks/*.js`（仅限剩余 global bridge cleanup 相关调用点）
- `web/static/js/features/anima-app/helpers/*.js`（新增或扩展现有 bridge）
- `web/static/js/features/anima-app/index.js`（启动链路调整）
- `tests/test_training_frontend_state.py`（基线调整）
- 本文档

不触碰：

- `runtime/`、`state/`、已清零的 chunk
- 后端 API、DOM id 契约、CSS
- 训练核心代码

## 迁移不变量

- `createAnimaApp(ctx)` 仍是入口。
- 每轮只改一个 chunk，不改业务行为。
- 旧函数名短期通过 bridge 保持可用，迁完调用点后再删 bridge。
- 每轮 cache token bump，测试基线下调。
- 新增模块不得写 `globalThis`，也不得恢复 runtime globals 别名。

## 阶段计划

| 阶段 | 目标 | 主要 chunk |
| --- | --- | --- |
| 1-6 | 已完成 | `26 / 02 / 21 / 19 / 24 / 20` 已分别迁到对应 bridge |
| 7 | 已完成 | `37-config-training-source.js` 已切到 training state / toml action bridge |
| 8 | 已完成 | `01a-image-test-feature.js`、image-test/status polling 已切到显式 bridge |
| 9 | 已完成 | 生产入口已移除 `installLegacyStateGlobals(runtime)`，repo 级源码扫描已确认无真实 consumer |
| 10 | 已完成 | `globalThis` 只剩浏览器原生必要使用，`legacy-globals.js` 已删除 |

## 验收命令

```bash
git diff --check -- web/static docs/proposal tests/test_training_frontend_state.py
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py
rg -o "globalThis\." web/static/js/features/anima-app | wc -l
```

最终验收追加：

```bash
rg "installLegacyRuntimeGlobals|globalThis\\.api|globalThis\\.datasetPresetApi|globalThis\\.val|globalThis\\.populateSelect" \
  web/static/js/features/anima-app \
  tests/test_training_frontend_state.py
rg "installLegacyStateGlobals" web/static/js/features/anima-app/index.js
rg "legacy-globals\\.js|installLegacyStateGlobals" \
  web gui library scripts anima_lora train.py inference.py tasks.py
[ ! -e web/static/js/features/anima-app/legacy-globals.js ]
```

## 完成定义

- 6 个旧 chunk 直接全局写入全部归零。
- `37-config-training-source.js`、`01a-image-test-feature.js` 已完成收口。
- `appShell / config / dataset / history / toml` 至少具备一个显式 state bridge，并开始承接真实调用点。
- `01-scope-state.js`、`26-load-global-settings.js`、`26a-status-polling.js` 不再依赖隐式 holder / poll target；改由 state bridge 驱动。
- `helpers/runtime-bridge.js` 承接 runtime helper，旧 runtime globals 不再需要 compat shim。
- `legacy-globals.js` 已删除。
- repo 内生产/源码路径不再消费 `legacy-globals.js`。
- `globalThis.` 总量保持为 0（anima-app 目录）。
- `tests/test_training_frontend_state.py` 全部通过。
