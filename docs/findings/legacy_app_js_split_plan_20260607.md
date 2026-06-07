# legacy-app.js 拆分执行计划

日期：2026-06-07
目标文件：`web/static/js/features/legacy-app.js`
执行方式：分批拆分、每批完成后必须审核修改链路；如审核发现问题，先返修同一批，再进入下一批。

## 总原则

这次拆分目标是降低 `legacy-app.js` 的长期维护风险，不是顺手改 UI、重做交互、改 DOM 结构或重命名 class。

每一批都按“搬家型重构”处理：

- 行为不变。
- DOM id / class 不变。
- CSS 不同步大改。
- API 路径不变。
- 文案不主动改。
- import cache token 保持一致。
- 已拆出的 feature 不再回填到 `legacy-app.js`。

允许新增 `web/static/js/features/<feature>/`、`web/static/js/shared/` 下的模块。`legacy-app.js` 在过渡期可以继续作为 feature 编排层，但每一批完成后它的净行数必须下降，除非本批只是先搭基础设施。

## 每批固定工作流

每个拆分批次必须完整走一遍：

1. **定位链路**
   - 搜索本批函数、状态变量、DOM id、事件绑定、测试断言。
   - 明确入口、调用者、被调用者和跨模块依赖。

2. **制定本批边界**
   - 写清楚要移动哪些函数和状态。
   - 写清楚哪些函数暂时留在 `legacy-app.js`。
   - 不顺手拆下一个领域。

3. **执行搬移**
   - 新建 feature 目录，优先 `index.js` 聚合。
   - 新模块导出 `createXFeature(ctx, deps)` 或纯函数。
   - `legacy-app.js` 只保留创建 feature、传依赖、调用入口。

4. **同步测试**
   - 如果测试原来读取 `legacy-app.js` 文本，改成读取新 feature 模块文本。
   - 保持测试验证同一行为/钩子，不因拆分降低断言强度。

5. **静态审核**
   - 检查模块图可达。
   - 检查 import cache token。
   - 检查没有把已拆逻辑复制两份。
   - 检查 `legacy-app.js` 是否没有新增无关逻辑。

6. **运行验证**
   - 至少运行：
     ```bash
     timeout 60 python -m pytest tests/test_training_frontend_state.py
     ```
   - 涉及 preview/queue/history 时，补跑相关后端服务测试，见各批说明。

7. **审核修改链路**
   - 用 `git diff -- web/static/js tests/test_training_frontend_state.py` 看本批全链路。
   - 审核“入口 -> 状态 -> API -> DOM -> 事件 -> 测试”是否闭合。
   - 发现问题必须返修并重新跑本批验证，不允许带病进入下一批。

## 共同工程规则

### 模块结构

新 feature 推荐结构：

```text
web/static/js/features/<feature>/
  index.js      # createXFeature(ctx, deps)
  state.js      # feature 私有状态工厂
  api.js        # feature API 包装
  render.js     # DOM 渲染
  actions.js    # 用户动作、保存/删除/启动等
```

小 feature 可以只建 `index.js`。纯工具进入 `web/static/js/shared/`，但必须无业务状态。

### `ctx` 和 `deps`

`ctx` 只放通用依赖：`api`、`dom`、`download`、`format`、`catalog`、`MetricsChart`。
feature 间依赖用 `deps` 明确传入，不要在模块里反向 import `legacy-app.js`。

推荐模式：

```js
export function createGpuPickerFeature(ctx) {
    const state = { available: [], selected: [] };
    return {
        load,
        selectedPayload,
        bindEvents,
    };
}
```

### 缓存 token

当前入口 token 是 `module-bootstrap-20260604-11`。新增 import 必须使用同一个 token，直到专门做 cache token bump：

```js
import { createXFeature } from './x/index.js?v=module-bootstrap-20260604-11';
```

如果需要 bump token，必须同步：

- `web/static/index.html`
- `web/static/app.js`
- 所有被改 JS import
- `tests/test_training_frontend_state.py` 相关断言

### 测试迁移规则

`tests/test_training_frontend_state.py` 当前会读取 `legacy-app.js` 并做大量字符串断言。拆分时要把对应断言迁移到新模块，例如：

```python
legacy_source = _frontend_module_text("js/features/legacy-app.js")
gpu_source = _frontend_module_text("js/features/gpu-picker/index.js")
```

不要简单删除断言。测试应该从“某字符串存在 legacy”改成“某字符串存在新 feature，legacy 只保留接入调用”。

### 禁止事项

- 禁止同时重构 `style.css`。
- 禁止改 DOM id / class，除非本批计划明确列出并同步所有 selector/test。
- 禁止把多个业务域揉进一个新模块。
- 禁止复制旧函数后不删除旧实现。
- 禁止新 feature 直接读写其他 feature 的内部状态。
- 禁止把 `window` 全局变量当作 feature 通信方式。

## 当前 legacy 分区概览

按现有注释和函数数量粗分：

| 分区 | 约函数数 | 说明 |
|---|---:|---|
| 状态/feature ensure | 6 | preview、queue、history-detail、weight-analysis 接入 |
| 初始化 | 21 | theme、GPU picker、教程入口等 |
| Tab 切换 | 2 | 顶层 tab 和懒加载触发 |
| 加载初始数据 | 5 | methods/presets/help/config 初始化 |
| 配置表单渲染 | 325 | 最大块，实际包含配置表单、数据集、sample prompts、拖拽、选择 guide、字段输入等 |
| TOML 编辑器 | 188 | TOML manager、output run、dataset preset 保存、分组、锁定、导入导出 |
| 训练控制 | 23 | start/preflight/preprocess/stop/queue request |
| WebSocket | 40 | WS、日志、指标、runtime、dashboard、健康提示 |
| 全局设置 | 8 | settings API 和表单 |
| 预览图 | 25 | 主要是 preview feature 兼容代理 |
| 训练队列 | 6 | 主要是 queue feature 兼容代理 |
| 状态轮询/历史 | 260 | pollStatus、历史列表、集合、拖拽、详情兼容、resume 代理 |
| 事件绑定 | 2 | 超大事件绑定和 tooltip |
| 工具函数 | 4 | api、datasetPresetApi、val、populateSelect |

拆分优先级：先拆边界清楚的小块，再拆大块；先拆已模块化 feature 的兼容代理，再拆复杂表单和历史。

## 批次 0：建立拆分护栏

### 目标

让后续拆分有统一工具和审核标准。

### 建议修改

- 新增或补强前端测试辅助，不改变生产代码行为。
- 可在 `docs/findings/legacy_app_js_split_plan_20260607.md` 记录每批执行结果。
- 检查 `test_legacy_app_is_transition_glue_not_new_feature_home` 的行数阈值，后续每批应下降。

### 审核重点

- 没有业务 JS 改动。
- 没有 CSS 改动。
- 现有测试通过。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py
```

## 批次 1：拆 GPU picker

### 为什么先拆

GPU picker 边界小，依赖少，能验证拆分模式：私有 state + DOM render + localStorage + API。

### 迁移范围

从 `legacy-app.js` 移动：

- `loadStoredGpuWhitelist`
- `saveGpuWhitelist`
- `loadGpuOptions`
- `sanitizeGpuWhitelist`
- `renderGpuPicker`
- `gpuPickerSummary`
- `gpuPickerTitle`
- `updateGpuPickerNote`
- `setGpuWhitelist`
- `toggleGpuSelection`
- `selectedGpuPayload`
- `closeGpuPickerPanel`
- `initGpuPickerEvents`

相关状态：

- `GPU_WHITELIST_STORAGE_KEY`
- `availableGpus`
- `selectedGpuWhitelist`

建议新文件：

```text
web/static/js/features/gpu-picker/index.js
```

### 对外接口

```js
const gpuPicker = createGpuPickerFeature(ctx);
await gpuPicker.loadOptions();
gpuPicker.bindEvents();
gpuPicker.selectedPayload();
```

`legacy-app.js` 中替换：

- 初始化阶段调用 `gpuPicker.bindEvents()`。
- `loadInitialData()` 中调用 `gpuPicker.loadOptions()`。
- 训练启动/预处理/队列请求里把 `selectedGpuPayload()` 改成 `gpuPicker.selectedPayload()`。

### 测试调整

在 `tests/test_training_frontend_state.py` 增加/修改断言：

- 新模块可达。
- `localStorage` key 仍存在。
- `/api/training/gpus` 仍存在。
- `selectedPayload` 或同等函数存在。
- `legacy-app.js` 不再包含 `let availableGpus`、`function renderGpuPicker`。

### 审核链路

- 入口：`createLegacyApp -> createGpuPickerFeature`。
- 事件：`gpu-picker-toggle`、`gpu-all-checkbox`、GPU checkbox。
- 数据：localStorage -> state -> render -> selected payload -> training request。
- 失败返修：如果训练请求拿不到 GPU 白名单，优先检查 `selectedPayload` 是否通过 deps 传到 start/preprocess/queue。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_training_gpu_selection.py
```

## 批次 2：拆 theme、tab、教程入口

### 迁移范围

拆出小型 shell feature：

- `currentTheme`
- `storedTheme`
- `saveTheme`
- `applyTheme`
- `initThemeToggle`
- `normalizeTopLevelTabState`
- `setupTabs`
- `openTutorialDialog`

建议新文件：

```text
web/static/js/features/app-shell/index.js
```

### 对外接口

```js
const appShell = createAppShellFeature(ctx, {
    onTabChange({ previousTab, nextTab }) {},
});
appShell.applyInitialTheme();
appShell.bindEvents();
appShell.setupTabs();
```

`onTabChange` 继续由 `legacy-app.js` 提供，用于触发 preview、history、weight-analysis、settings 等懒加载。

### 审核链路

- 主题：localStorage -> `document.documentElement.dataset.theme` -> 按钮文案。
- tab：`.tab-btn` -> `.tab-content` -> lazy load callbacks。
- 不允许在 `app-shell` 内直接 import preview/history/queue。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py
```

## 批次 3：拆全局设置

### 迁移范围

移动：

- `loadGlobalSettings`
- `saveGlobalSettings`
- `resetGlobalSettings`
- `setGlobalSettingsStatus`
- `applyGlobalSettingsToInputs`
- `collectGlobalSettingsPayload`
- `getGlobalModelPathOverrides`
- `toggleGlobalSettingHelp`

状态：

- `globalSettings`

建议新文件：

```text
web/static/js/features/global-settings/index.js
```

### 对外接口

```js
const globalSettingsFeature = createGlobalSettingsFeature(ctx);
await globalSettingsFeature.load();
globalSettingsFeature.getModelPathOverrides();
globalSettingsFeature.bindEvents();
```

`legacy-app.js` 中 `fillGlobalModelPathsIntoConfigForm()` 仍可通过接口读取全局模型路径。

### 审核链路

- API：`/api/settings/global`。
- DOM：`global-output-root`、`global-pretrained-model-path`、`global-qwen3-path`、`global-vae-path`。
- 依赖：配置表单只读接口，不直接拿 feature 内部 state。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_preview_service.py
```

## 批次 4：拆 preview 兼容代理

### 背景

preview 主体已经在 `web/static/js/features/preview/`。`legacy-app.js` 中剩下的是一组兼容代理函数。

### 迁移范围

把这些代理调用改成直接通过 `previewFeature` 暴露，不再在 legacy 保留同名函数，或只保留极少数被跨 feature 依赖的桥接函数：

- `loadPreviewSettings`
- `savePreviewSettings`
- `resetPreviewSettings`
- `loadPreviewImages`
- `loadPreviewWeights`
- `setPreviewSource`
- `openTrainingPreview`
- `openCurrentTrainingPreview`
- `openHistoryConfigGroupPreview`
- `normalizePreviewGroup`
- `renderPreviewTaskSelect`
- `changePreviewTask`
- `togglePreviewWeightSort`
- `openPreviewDialog`
- `closePreviewImageDialog`
- `openPreviewPanel`
- `closePreviewPanel`
- `restorePreviewWorkspaceAfterPanelClose`
- `setPreviewStatus`
- `createPreviewDetailRow`
- `createPreviewDetailBlock`
- `renderDatasetImageDialogDetails`
- `formatTotalPixels`
- `copyText`
- `formatBytes`

优先方案：扩充 preview feature 的返回对象，让 legacy 用 `ensurePreviewFeature().method()` 调用。

### 审核链路

- 页面预览、弹窗预览、历史详情挂载仍是同一个 `#preview-workspace`。
- `openTrainingPreview({ taskId })` 和 `openTrainingPreview({ group })` 行为不变。
- dataset preview 如果复用 preview 的 detail helper，要明确通过 deps 注入，不要反向 import。

### 测试调整

已有 preview 测试要改为断言 preview 模块内存在实现，legacy 只存在 `ensurePreviewFeature()` 接入。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_preview_service.py
```

## 批次 5：拆 queue 兼容代理

### 背景

queue 主体已经在 `web/static/js/features/queue/`。legacy 仍有代理和训练控制交叉。

### 迁移范围

处理：

- `loadTrainingQueue`
- `updateTrainingQueueFromPayload`
- `renderTrainingQueue`
- `refreshQueueRunningProgressViews`
- `showTrainingView`
- `renderTrainingViewMode`
- `queueCurrentTrainingFromConfig`
- `enqueueTrainingFromConfig`
- `enqueueTrainingQueueRequest`
- `queueResumeTrainingFromCheckpoint`
- `enqueueTomlGroupToQueue` 相关调用

注意：`startTraining()` 和 `stopTraining()` 暂不拆入 queue。queue 只接收“冻结后的训练请求 payload”。

### 对外接口

queue feature 应明确提供：

```js
queueFeature.loadQueue();
queueFeature.updateFromPayload(payload);
queueFeature.render();
queueFeature.showTrainingView(mode);
queueFeature.enqueueTrainingFromConfig(payload);
queueFeature.queueResumeTrainingFromCheckpoint(payload);
```

### 审核链路

- 当前配置 -> frozen config file -> queue API。
- TOML group -> 批量 enqueue -> queue state refresh。
- resume -> checkpoint -> queue resume API。
- WS `type: queue` -> queue feature update。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_training_queue.py
```

## 批次 6：拆训练控制和 preflight

### 迁移范围

建议新模块：

```text
web/static/js/features/training-control/index.js
web/static/js/features/training-control/preflight.js
```

移动：

- `startTraining`
- `runPreflight`
- `isCliOnlySpdSource`
- `currentTrainingConfigIsRuntime`
- `chooseTrainingLaunchMode`
- `confirmTrainingLaunch`
- `startTrainingUnchecked`
- `enterLiveTrainingForNewRun`
- `showPreflightDialog`
- `showPreflightPendingDialog`
- `renderPreflightPending`
- `showPreflightRequestError`
- `isPreflightDialogOpen`
- `waitForPreflightDialogClose`
- `renderPreflightResult`
- `preflightCanStartPreprocess`
- `startPreprocessFromPreflight`
- `currentTrainingConfigFile`
- `preflightPlainText`
- `stopTraining`

### 依赖注入

训练控制需要从 legacy/其他 feature 获得：

- 当前 variant/preset/methods_subdir。
- 当前 TOML 文件。
- `collectChangedFormValues` / 保存表单 patch。
- `continueTrainingRequestPayload`。
- GPU whitelist payload。
- queue enqueue 接口。
- live training dashboard 更新接口。

这些必须通过 `deps` 明确传入。

### 审核链路

- 开始训练：按钮 -> preflight -> save pending config -> start/preprocess/queue。
- 停止训练：按钮 -> `/api/training/stop` -> 状态更新。
- continue LoRA：继续训练来源进入 start payload。
- GPU payload 仍生效。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_training_queue.py
```

## 批次 7：拆 WebSocket、日志、实时 dashboard

### 迁移范围

建议新模块：

```text
web/static/js/features/training-live/index.js
web/static/js/features/training-live/logs.js
web/static/js/features/training-live/metrics.js
web/static/js/features/training-live/dashboard.js
```

移动：

- `connectWebSocket`
- `handleWsMessage`
- `appendLog`
- `appendLogRecord`
- `renderLogOutputLines`
- `currentLogOutputLines`
- `logLineTone`
- `scheduleLogFlush`
- `flushLogBuffer`
- `replayTrainingLogs`
- `replayMetricsHistory`
- `replayMetricsFromLogRecord`
- `setLogStatus`
- `updateLogStatusText`
- `setTrainingHealthNotice`
- `recoverLiveTrainingState`
- `updateProgress`
- `updateMetrics`
- `updateStatus`
- `liveStatusState`
- `terminalStatusMessage`
- `resetLiveSystemPeaks`
- `clearRuntimeInfo`
- `applyRuntimeInfoToState`
- `renderCurrentRuntimePaths`
- `currentRuntimeTaskInfo`
- `updateSystem`
- `formatRuntimeVram`
- `renderLiveTrainingDashboard`
- `trainingEtaMetricInfo`
- `parseProgressRateSeconds`
- `formatEtaClock`
- `isSameDate`
- `markTrainingActivity`
- `refreshTrainingHealth`
- `parseMetricsFromProgressLine`
- `lastValue`
- `readConfigNumber`
- `formatLr`
- `formatDuration`

状态：

- `ws`
- `lossChart`
- `stepCounter`
- `trainingStatusPollFailures`
- `liveChartState`
- `trainingRuntime`
- live system peak 相关字段

### 审核链路

- WS reconnect。
- `status/progress/metrics/system/queue` 消息分发。
- 日志 buffer flush。
- live chart 仍能绘制。
- health notice 仍按输出活动更新。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_progress_sink.py
```

## 批次 8：拆 status polling 与历史列表基础

### 迁移范围

建议新模块：

```text
web/static/js/features/history-list/index.js
web/static/js/features/history-list/api.js
web/static/js/features/history-list/render.js
web/static/js/features/history-list/state.js
```

先移动低耦合历史列表/筛选，不碰集合拖拽：

- `pollStatus`
- `applyStatusSnapshotFallbacks`
- `hasStatusPayload`
- `loadTrainingHistoryList`
- `loadHistoryCollectionSettings`
- `saveHistoryCollectionSettings`
- `normalizeHistoryCollectionSettings`
- `uniqueStringList`
- `normalizeHistoryConfigGroupOrder`
- `renderTrainingHistoryList`
- `recentTrainingSidebarTasks`
- `renderHistoryManager`
- `renderHistoryManagerItems`
- `renderHistoryManagerStats`
- `applyHistoryStatFilter`
- `historyStatFilterIsActive`
- `historyManagerFilteredTasks`
- `historyManagerBaseFilteredTasks`
- `historyManagerVisibleTasks`
- `uniqueHistoryTasks`
- `historyTaskMatchesSourceFilter`
- `historyTaskSearchText`
- search/sort/filter 相关纯函数

### 审核链路

- status poll 仍能更新当前任务。
- 历史刷新按钮 -> API -> state -> 侧边栏/管理台。
- 归档显示、筛选、搜索、排序不变。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_training_resume.py
```

## 批次 9：拆历史集合与拖拽

### 迁移范围

建议新模块：

```text
web/static/js/features/history-collections/index.js
web/static/js/features/history-collections/drag.js
web/static/js/features/history-collections/render.js
```

移动集合和拖拽相关：

- collection workbench 渲染。
- history config group sort。
- history collection sort。
- task drop to collection。
- drop popover。
- bulk collection actions。

这批风险高，必须小步拆。建议拆成 2-3 个子批：

1. 纯函数和排序：`moveItemNearList`、collection/group order helpers。
2. render：collection card、config group card。
3. drag：pointer/mouse/touch/HTML5 drag 事件。

### 审核链路

- 集合管理大界面打开。
- 配置分组排序。
- 集合排序。
- 历史任务拖入集合。
- 批量分组/清空/重命名。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_training_resume.py
```

建议另做浏览器手验：拖拽排序、拖任务到集合、集合重命名。

## 批次 10：拆 history detail 兼容代理

### 背景

history detail 主体已经模块化，但 legacy 仍保留大量兼容函数和路径工具。

### 迁移范围

优先把以下代理和路径 helper 下沉或接入已有 `history-detail/`：

- `loadHistoryTask`
- `refreshHistoryView`
- `loadConfigGroupTimeline`
- `renderHistoryTask`
- `renderConfigGroupTimeline`
- `returnToLiveTraining`
- `loadResumeOptionsForTask`
- `resumeTrainingFromCheckpoint`
- `queueResumeTrainingFromCheckpoint`
- `renderHistoryPaths`
- `runtimePathItems`
- `historyAbsolutePath`
- `historyProjectRoot`
- `historyArtifactUrl`
- detail dialog open/close/keydown/preview activate 相关代理

### 审核链路

- 历史任务详情弹窗。
- tabs 切换。
- 预览工作区挂载和恢复。
- resume panel。
- config group timeline。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_training_resume.py tests/test_preview_service.py
```

## 批次 11：拆配置表单基础

### 背景

配置表单是最大块，必须后拆。前面批次完成后，legacy 里的跨域依赖会少很多。

### 目标结构

```text
web/static/js/features/config-form/
  index.js
  state.js
  render.js
  fields.js
  network-args.js
  choice-guide.js
  sample-prompts.js
  stage-resolution.js
  resource-presets.js
```

### 第一子批：纯状态和 draft

移动：

- `resetConfigFormDraft`
- `syncConfigDraftFromForm`
- `updateConfigDraftFromInput`
- `originalConfigFieldValue`
- `displayConfigFieldValue`
- `configDraftValueChanged`

状态：

- `configFormState`
- `currentConfig` 仍可暂时在 legacy，由接口读取/写入。

### 第二子批：字段输入和 network args

移动：

- `createFieldInput`
- `createSelectInput`
- number/bool/array parse helpers。
- `applyNetworkArgFields`
- `activeNetworkArgSpecs`
- `parseNetworkArgMap`
- `collectNetworkArgsFromForm`

### 第三子批：渲染 layout

移动：

- `renderConfigForm`
- `createGroup`
- `appendFieldRows`
- `createConfigFormControls`
- category/search/sticky directory 相关。

### 第四子批：sample prompts editor

移动 sample prompts 表格/文本编辑器相关函数。

### 第五子批：choice guide、resource presets、stage resolution

分别拆独立小模块。

### 审核链路

- method/variant/preset 切换 -> loadConfig -> renderConfigForm。
- 字段修改 -> draft -> dirty badge -> save patch。
- network_args roundtrip。
- sample prompts 保存路径和内容。
- sticky category 搜索/折叠。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_web_config_service.py
```

## 批次 12：拆数据集编辑器与数据集预设 UI

### 目标结构

```text
web/static/js/features/dataset-presets/
  index.js
  state.js
  api.js
  render-list.js
  render-editor.js
  preview.js
  drag.js
```

### 迁移范围

- `loadDatasetEditor`
- `loadDatasetPresets`
- `loadDatasetPreset`
- dataset preset list/group/search。
- dataset editor rows/defaults/settings。
- dataset preview dialog。
- nl/tag mix、trigger clone 编辑器。
- dataset preset import/export/copy/rename/delete/save。
- dataset editor drag。

### 建议分三子批

1. API + state + list render。
2. editor rows/defaults/settings。
3. preview + drag + import/export 操作。

### 审核链路

- 配置页 dataset picker 与数据集页管理状态互不打架。
- 保存 dataset preset 后，配置页选择能刷新。
- preview image path 不逃逸。
- path_pattern、caption_source_mode、nl_tag_mix、trigger_clone roundtrip。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_web_config_service.py tests/test_preprocess_paths.py
```

## 批次 13：拆 TOML manager

### 目标结构

```text
web/static/js/features/toml-manager/
  index.js
  state.js
  api.js
  render-groups.js
  editor.js
  output-runs.js
  save-as.js
  locks.js
  drag.js
```

### 迁移范围

- TOML 文件列表。
- 项目/输出 run mode 切换。
- raw TOML load/save/patch。
- output run config read/copy/export/save-as。
- TOML group render/lock/move/rename/delete/export/queue。
- dirty state、badges、unsaved switch dialog。
- import/export/download helpers。

### 建议分四子批

1. `output-runs`：边界比较独立。
2. TOML 文件列表和选择。
3. save/patch/dirty/lock。
4. group/drag/export/queue。

### 审核链路

- 右侧 TOML 选择 -> 左侧配置同步。
- 左侧字段保存 -> TOML patch。
- 直接编辑器保存二次确认。
- 系统锁/用户锁/分组锁不被破坏。
- output run 只读复制为项目预设。
- group 加入队列仍走 queue feature。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_web_config_service.py tests/test_training_queue.py
```

## 批次 14：拆事件绑定和 tooltip

### 背景

当前 `setupEventListeners()` 是大型中心函数。前面 feature 拆完后，它应该变成每个 feature 自己 bind。

### 迁移范围

- 每个 feature 自己暴露 `bindEvents()`。
- `legacy-app.js` 只调用：
  ```js
  appShell.bindEvents();
  gpuPicker.bindEvents();
  configForm.bindEvents();
  tomlManager.bindEvents();
  trainingControl.bindEvents();
  trainingLive.bindEvents();
  historyList.bindEvents();
  ```
- `installBeginnerTooltips` 拆到：
  ```text
  web/static/js/features/tooltips/index.js
  ```

### 审核链路

- 所有按钮仍能触发。
- tooltip title 仍存在。
- 测试不再从 legacy 截取 listener_section，而是检查各 feature 的 bind 函数。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py
```

建议浏览器手验全页面主要按钮。

## 批次 15：清理 legacy-app.js

### 目标

让 `legacy-app.js` 只保留：

- import feature factory。
- 创建共享 ctx/deps。
- `DOMContentLoaded` 初始化顺序。
- 少量跨 feature 编排函数。
- 过渡说明。

目标行数可设为 500-1200 行。不要强行追求 0 行，因为它仍可以作为 WebUI app composer。

### 清理项

- 删除已迁移状态变量。
- 删除重复 helper。
- 删除已无用 feature ensure。
- 删除旧 section 注释。
- 更新 `test_legacy_app_is_transition_glue_not_new_feature_home` 行数阈值。

### 验证

```bash
timeout 60 python -m pytest tests/test_training_frontend_state.py tests/test_web_config_service.py tests/test_training_queue.py tests/test_training_resume.py tests/test_preview_service.py
```

## 审核清单模板

每批结束后在提交说明或任务记录里填写：

```text
批次：
移动的函数/状态：
新增文件：
修改入口：
测试迁移：
运行测试：
legacy-app.js 行数变化：

入口链路：
- app.js -> legacy-app.js -> feature:
- DOM 事件 -> action:
- API -> state -> render:

风险检查：
- DOM id/class 是否保持：
- CSS 是否未改：
- import token 是否一致：
- 旧实现是否删除：
- 测试是否仍验证同一行为：

发现问题：
返修动作：
复测结果：
```

## 失败处理规则

- **模块不可达**：先检查 import specifier 和 token，再检查 `app.js` 图。
- **测试找不到旧字符串**：不要硬塞字符串回 legacy；把断言迁移到新 feature。
- **运行时报 undefined**：检查 deps 是否漏传，不要用全局变量补洞。
- **按钮失效**：检查本 feature `bindEvents()` 是否被初始化调用，以及 DOM 是否在调用时已存在。
- **状态不同步**：优先把状态收归 feature，并暴露 getter/setter；不要两个模块各维护一份。
- **CSS 异常**：本计划不改 CSS，先确认是否误改 DOM class/id。
- **队列/训练 payload 异常**：检查 config file、continue info、GPU whitelist、extra args 是否完整穿透。

## 推荐执行顺序总览

```text
0. 建立护栏
1. GPU picker
2. app shell/theme/tab
3. global settings
4. preview 兼容代理
5. queue 兼容代理
6. training control/preflight
7. websocket/log/live dashboard
8. status polling/history list
9. history collections/drag
10. history-detail 兼容代理
11. config form
12. dataset presets/editor
13. toml manager
14. event binding/tooltips
15. legacy cleanup
```

如果执行中发现某批依赖比预期复杂，允许拆成子批，但不建议跨批合并。宁可多跑几次测试，也不要一次移动几千行后再定位问题。
