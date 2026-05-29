from __future__ import annotations

from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "web" / "static" / "app.js"
CHART_JS = Path(__file__).resolve().parents[1] / "web" / "static" / "chart.js"
INDEX_HTML = Path(__file__).resolve().parents[1] / "web" / "static" / "index.html"
STYLE_CSS = Path(__file__).resolve().parents[1] / "web" / "static" / "style.css"


def _section(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_new_training_launch_enters_live_monitoring() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    helper = _section(source, "function enterLiveTrainingForNewRun()", "function showPreflightDialog")
    tab_setup = _section(source, "function setupTabs()", "// ── 加载初始数据 ──")

    assert "returnToLiveTraining({ refresh: false });" in helper
    assert 'document.querySelector(\'[data-tab="training"]\')?.click();' in helper
    assert "pollStatus();" in helper
    assert "replayTrainingLogs();" in helper
    assert "previousTab === 'training' && nextTab !== 'training'" in tab_setup
    assert "resetTrainingExpandedStateOnLeave();" in tab_setup

    start_path = _section(source, "async function startTrainingUnchecked", "function enterLiveTrainingForNewRun")
    preprocess_path = _section(source, "async function startPreprocessFromPreflight", "function currentTrainingConfigFile")

    assert "enterLiveTrainingForNewRun();" in start_path
    assert "enterLiveTrainingForNewRun();" in preprocess_path
    assert 'document.querySelector(\'[data-tab="training"]\').click();' not in start_path
    assert 'document.querySelector(\'[data-tab="training"]\').click();' not in preprocess_path


def test_return_to_live_training_clears_runtime_cursor() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    body = _section(source, "function returnToLiveTraining", "async function loadResumeOptionsForTask")

    for snippet in (
        "viewingHistoryTaskId = '';",
        "historyViewMode = 'live';",
        "trainingRuntime.lastLogId = 0;",
        "trainingRuntime.logLineCount = 0;",
        "stepCounter = 0;",
        "lossChart?.clear();",
    ):
        assert snippet in body


def test_training_queue_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    queue_section = _section(source, "async function loadTrainingQueue()", "// ── 状态轮询 ──")
    assert "function renderTrainingQueue()" in queue_section
    assert "function renderTrainingQueueManager()" in queue_section
    assert "function queueManagerSections()" in queue_section
    assert "function createTrainingQueueSection" in queue_section
    assert "function createTrainingQueueItem" in queue_section
    assert "function createTrainingQueueManagerItem" in queue_section
    assert "let trainingQueueFilter = 'actionable';" in source
    assert "async function toggleTrainingQueuePause()" in queue_section
    assert "deleteQueueItem" in queue_section
    assert "删除队列记录和缓存" in queue_section
    assert "delete_runtime: true" in queue_section
    assert "queueDeleteRuntimeMessage" in queue_section
    assert "queueRuntimeDirLabel" in queue_section
    assert "新任务已加入队列" in queue_section
    assert "删除原记录和缓存" in queue_section
    assert "完成和已取消记录" in queue_section
    assert "retryQueueItem" in queue_section
    assert "cancelWaitingQueueItems" in queue_section
    assert "clearFinishedQueueItems" in queue_section
    assert "trainingQueueFilter" in queue_section
    assert "/api/training/queue" in queue_section
    assert "/api/training/queue/settings" in queue_section
    assert "/api/training/queue/cancel-waiting" in queue_section
    assert "/api/training/queue/clear" in queue_section

    assert "training-queue-manager" in html
    assert "btn-training-queue-view" in html
    assert "training-queue-failure-policy" in html
    assert 'data-queue-filter="actionable">待处理' in html
    assert "training-queue-more-menu" in html

    summary_panel = _section(html, '<section class="panel training-queue-panel"', '<section class="panel task-history-panel">')
    manager_panel = _section(html, '<section id="training-queue-manager"', '<section id="training-history-placeholder"')
    assert "btn-cancel-waiting-queue" not in summary_panel
    assert "btn-clear-finished-queue" not in summary_panel
    assert "btn-cancel-waiting-queue" in manager_panel
    assert "btn-clear-finished-queue" in manager_panel

    ws_section = _section(source, "function handleWsMessage", "function appendLog")
    assert "case 'queue':" in ws_section
    assert "updateTrainingQueueFromPayload(msg);" in ws_section

    start_section = _section(source, "async function startTrainingUnchecked", "function enterLiveTrainingForNewRun")
    assert "enqueueTrainingFromConfig" in start_section
    assert "chooseTrainingLaunchMode" in source


def test_resume_queue_button_is_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    resume_section = _section(source, "function renderResumePanelState", "function optionNode")
    assert "btn-queue-resume-training" in resume_section
    assert "queueBtn.disabled" in resume_section

    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")
    assert "queueResumeTrainingFromCheckpoint" in source
    assert "btn-queue-resume-training" in listener_section


def test_sample_prompts_save_uses_current_training_config_context() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    body = _section(source, "async function saveSamplePrompts", "async function importTomlFile")
    prepare_body = _section(source, "async function prepareFormPatchValues", "function shouldSkipUiDefaultField")

    assert "train_config_file: currentTrainingSource.file || currentTomlFile || ''" in body
    assert "await saveSamplePrompts('');" not in prepare_body


def test_history_list_marks_queue_tasks() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    queue_label = _section(source, "function historyQueueLabel", "function historyContinueLabel")
    task_item = _section(source, "function createHistoryTaskItem", "function createHistoryActionButton")

    assert "来自队列" in queue_label
    assert "queue_attempt" in queue_label
    assert "historyQueueLabel(task)" in task_item


def test_history_manager_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    chart_source = CHART_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    history_section = _section(source, "async function loadTrainingHistoryList()", "function groupHistoryTasks")
    detail_section = _section(source, "function renderHistoryManagerDetail", "async function loadHistoryTask")
    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")
    preview_open_section = _section(source, "async function openTrainingPreview", "function normalizePreviewGroup")
    tab_setup_section = _section(source, "function setupTabs()", "// ── 加载初始数据 ──")
    sidebar_history_section = _section(source, "function renderTrainingHistoryList()", "function recentTrainingSidebarTasks")
    recent_sidebar_section = _section(source, "function recentTrainingSidebarTasks()", "function renderHistoryManager")
    log_append_section = _section(source, "function appendLogRecord", "async function replayTrainingLogs")
    history_review_mode_section = _section(source, "function isHistoryReviewMode()", "function openTutorialDialog")
    sidebar_task_item_section = _section(source, "function createHistoryTaskItem", "function createHistoryActionButton")
    manager_row_section = _section(source, "function createHistoryManagerRow", "function selectedHistoryConfigGroups")
    load_task_section = _section(source, "async function loadHistoryTask", "async function refreshHistoryView")

    assert "training-history-manager" in html
    assert "history-manager-search" in html
    assert "history-collection-search" in html
    assert "history-config-group-search" in html
    assert "集合搜索" in html
    assert "配置组搜索" in html
    assert "history-group-mode" in html
    assert "集合分组" in html
    assert "集合管理" in html
    assert '<option value="config">配置分组</option>' not in html
    assert '<option value="flat">平铺列表</option>' not in html
    assert "btn-history-collections-workbench" in html
    assert "btn-preview-training-results" in html
    assert "预览结果" in html
    assert 'data-tab="preview"' not in html
    assert 'class="preview-workspace-host" hidden aria-hidden="true"' in html
    assert "preview-page-mount" in html
    assert "preview-workspace" in html
    assert "preview-panel-dialog" in html
    assert "preview-dialog-mount" in html
    assert "btn-close-preview-panel" in html
    assert "training-dashboard" in html
    assert "training-run-state" in html
    assert "training-run-summary" in html
    assert "metric-vram-peak" in html
    assert "metric-gpu-peak" in html
    assert "metric-temp" in html
    assert "metric-temp-peak" in html
    assert "最近训练" in html
    assert "未归档 · 最新 20 个训练任务" in html
    assert "chart.js?v=lr-overlay" in html
    assert "app.js?v=lr-persist" in html
    assert "history-bulk-bar" in html
    assert "btn-history-bulk-delete" in html
    assert "设置集合" in html
    assert 'id="history-detail-panel"' not in html
    assert 'id="history-detail-dialog"' in html
    assert "history-detail-dialog-shell" in html
    assert "btn-close-history-detail" in html
    assert "history-detail-tabs" in html
    assert "data-history-detail-tab=\"overview\"" in html
    assert "data-history-detail-tab=\"system\"" in html
    assert "data-history-detail-tab=\"config\"" in html

    assert "renderHistoryManager()" in history_section
    assert "recentTrainingSidebarTasks" in history_section
    assert "groupHistoryTasks(" not in sidebar_history_section
    assert "task.job === 'training' && !historyTaskIsArchived(task)" in recent_sidebar_section
    assert ".slice(0, 20)" in recent_sidebar_section
    assert "historyManagerFilteredTasks" in history_section
    assert "function historyManagerBaseFilteredTasks" in history_section
    assert "function historyManagerVisibleTasks" in history_section
    assert "normalizeHistoryGroupMode" in history_section
    assert "historyManagerVisibleTasks(historyManagerBaseFilteredTasks())" in history_section
    assert "mode === 'config'" not in history_section
    assert "mode === 'flat'" not in history_section
    assert "historyConfigGroupVisibleForSearch" not in source
    assert "uniqueHistoryTasks" in history_section
    assert "createHistoryManagerRow" in history_section
    assert "renderHistoryManagerGrouped" in history_section
    assert "resetTrainingExpandedStateOnLeave" in history_section
    assert "collapseVisibleHistoryManagerGroups" in history_section
    assert "collapsedHistoryCollections.add(collection.key)" in history_section
    assert "collapsedHistoryConfigGroups.add(historyConfigGroupCollapseKey(group, collection.key))" in history_section
    assert "expandHistoryCollectionConfigGroups" not in source
    assert "historyStatFilterIsActive" in history_section
    assert "archived: 'all'" in history_section
    assert "next.kind = state" in history_section
    assert "createHistoryManagerCollectionSection" in history_section
    assert "createHistoryManagerConfigGroupSection" in source
    assert "createHistoryManagerCollectionSection(collection, splitCollections, collections)" in history_section
    assert "collection: selectedCollection" in history_section
    assert "groups: collection.groups" in source
    assert "renderHistoryCollectionsWorkbench" in history_section
    assert "createHistoryCollectionWorkbenchCard" in history_section
    assert "createHistoryConfigGroupWorkbenchCard" in history_section
    assert "historyCollectionWorkbenchTarget" in source
    assert "historyCollectionSettings" in source
    assert "historyCollectionSearch" in source
    assert "historyConfigGroupSearch" in source
    assert "historyTaskMatchesCollectionSearch" in source
    assert "historyTaskMatchesConfigGroupSearch" not in source
    assert "historyCollectionSearchText(collection).includes(collectionSearch)" in source
    assert "historyConfigGroupSearchText(group).includes(configSearch)" in source
    assert "document.getElementById('history-collection-search').addEventListener('input'" in source
    assert "document.getElementById('history-config-group-search').addEventListener('input'" in source
    assert "selectedHistoryCollectionKey" in source
    assert "/api/training/history/collections/settings" in source
    assert "loadHistoryCollectionSettings" in source
    assert "saveHistoryCollectionSettings" in source
    assert "collection_order" in source
    assert "config_group_order" in source
    assert "moveHistoryCollection" in source
    assert "moveHistoryConfigGroup" in source
    assert "moveHistoryCollection(collection, 'top', allCollections)" in source
    assert "moveHistoryConfigGroup(group, 'top', options.groups, options.collection)" in source
    assert "applySelectedHistoryTasksToCollection" in source
    assert "groupHistoryTasks(scopedTasks)" in history_section
    assert "task?.group" in source
    assert "设置集合" in source
    assert "清除集合" in source
    assert "搜索或新建集合" in source
    assert "未分配集合" in source
    assert "selectedHistoryCollectionKey === collection.key ? '' : collection.key" in source
    assert "加入已选" in source
    assert "加入目标" in source
    assert "查看集合" in source
    assert "取消查看" in source
    assert "选择分组" in source
    assert "置顶" in source
    assert "置底" in source
    assert "合并查看" in source
    assert "createHistoryConfigGroupMergeButton" in source
    assert "createHistoryConfigGroupPreviewButton" in source
    assert "loadConfigGroupTimeline(group, { skipSelectionDialog: true })" in source
    assert "查看这个自动配置分组内的全部训练结果" in source
    assert "openTrainingPreview({ group })" in source
    assert "openTrainingPreview({ taskId: task.id })" in source
    assert "canPreviewHistoryConfigGroup" in source
    assert "normalizePreviewGroup" in source
    assert "selectedPreviewGroup = normalizePreviewGroup(options.group)" in source
    assert "openPreviewPanel" in source
    assert "closePreviewPanel" in source
    assert "mountPreviewWorkspaceInDialog" in source
    assert "mountPreviewWorkspaceInPage" in source
    assert "openPreviewPanel();" in preview_open_section
    assert "document.querySelector('[data-tab=\"preview\"]')?.click()" not in preview_open_section
    assert "if (nextTab === 'preview')" not in tab_setup_section
    assert "mountPreviewWorkspaceInPage();" not in tab_setup_section
    assert "btn-preview-training-results" in listener_section
    assert "openCurrentTrainingPreview" in source
    assert "event?.preventDefault?.()" in source
    assert "event?.stopPropagation?.()" in source
    assert "addEventListener('click', openCurrentTrainingPreview)" in listener_section
    assert "chooseTimelineTasksForMerge" not in source
    assert "showTimelineTaskSelectionDialog" not in source
    assert "选择要合并查看的训练分组" not in source
    assert "选择合并查看" not in source
    assert "该配置组分布在" in source
    assert "selectedHistoryTaskIds" in source
    assert "applyHistoryBatchAction" in source
    assert "deleteHistoryTasksThorough" in source
    assert "confirm_text: confirmText" in source
    assert "彻底删除" in source
    assert "/api/training/history/batch" in source
    assert "openHistoryDetailDialog" in source
    assert "closeHistoryDetailDialog" in source
    assert "return historyViewMode !== 'live';" in history_review_mode_section
    assert "Boolean(viewingHistoryTaskId)" not in history_review_mode_section
    assert "main.addEventListener('click', () => loadHistoryTask(task.id))" in sidebar_task_item_section
    assert "createHistoryActionButton('查看', () => loadHistoryTask(task.id))" in sidebar_task_item_section
    assert "main.addEventListener('click', () => loadHistoryTask(task.id))" in manager_row_section
    assert "createHistoryActionButton('查看', () => loadHistoryTask(task.id))" in manager_row_section
    assert "showTrainingView('history')" not in load_task_section
    assert "renderHistoryTask(payload)" not in load_task_section
    assert "currentHistoryTaskForResume = payload.task || null;" in load_task_section
    assert "renderHistoryManagerDetail(payload, { open: true })" in load_task_section

    assert "renderHistoryDetailDialog" in detail_section
    assert "renderHistoryDetailOverview" in detail_section
    assert "renderHistoryDetailResume" in detail_section
    assert "renderHistoryDetailChart" in detail_section
    assert "renderHistoryDetailLogs" in detail_section
    assert "renderHistoryDetailSystem" in detail_section
    assert "renderHistoryDetailConfig" in detail_section
    assert "renderHistoryDetailPaths" in detail_section
    assert "historyCurveState" in source
    assert "renderHistoryCurveStats" in detail_section
    assert "renderHistoryCurveToolbar" in detail_section
    assert "renderHistoryCurveMainChart" in detail_section
    assert "createHistoryCurveSvg" in detail_section
    assert "renderHistoryCurveInspector" in detail_section
    assert "renderHistoryCurveSegments" in detail_section
    assert "historyCurveSmoothPoints" in detail_section
    assert "historyCurveFilteredPoints" in detail_section
    assert "historyCurveDisplayPoints" in detail_section
    assert "HISTORY_CURVE_RENDER_POINT_LIMIT" in source
    assert "绘图已降采样" in detail_section
    assert "stageBreakBefore" in detail_section
    assert "display_step" in detail_section
    assert "平滑窗口" in detail_section
    assert "最近100点" in detail_section
    assert "最近25%" in detail_section
    assert "自定义 Step" in detail_section
    assert "box.appendChild(createHistorySparkline(lossPoints));" not in source
    assert "historySystemSummary" in detail_section
    assert "historySystemRecords" in detail_section
    assert "HISTORY_SYSTEM_TABLE_RENDER_LIMIT" in source
    assert "historyDetailLimitNotice" in detail_section
    assert "仅显示最近" in detail_section
    assert "最后 VRAM" in detail_section
    assert "峰值 GPU" in detail_section
    assert "无系统采样记录" in detail_section
    assert "system.jsonl" in detail_section
    assert "history-detail-metrics-body" in detail_section
    assert "historyDetailSection('任务信息'" not in source
    assert "history-detail-section info" not in source
    assert "loadHistoryResumeWeights" in source
    assert "/api/preview/weights?task_id=" in source
    assert "diagnostic" in source
    assert "权重热启动" in source
    assert "optimizer、scheduler 和已完成步数" in source

    assert "btn-history-manager-refresh" in listener_section
    assert "btn-history-collections-workbench" in listener_section
    assert "btn-history-bulk-archive" in listener_section
    assert "btn-live-training" in listener_section
    assert "returnToLiveTraining" in listener_section
    assert "history-filter-kind" in listener_section
    assert "history-group-mode" in listener_section
    assert "key === 'groupMode' ? normalizeHistoryGroupMode(value) : value" in listener_section
    assert "history-detail-tab" in listener_section
    assert "btn-close-history-detail" in listener_section
    assert "logBuffer" in source
    assert "scheduleLogFlush" in log_append_section
    assert "requestAnimationFrame" in log_append_section
    assert "MAX_LOG_LINES" in log_append_section
    assert "lastLrText" not in source
    assert "recordLearningRateChange" not in source
    assert "announceLr" not in source
    assert "updatePointMetadata" in source
    assert "['学习率', formatLr(point.lr)]" in source
    assert "lr: item.lr" in source
    assert "peakVramUsedGb" in source
    assert "peakGpuUtil" in source
    assert "peakGpuTemp" in source
    assert "renderLiveTrainingDashboard" in source
    assert "resetLiveSystemPeaks" in source

    assert ".training-workspace.history-wide" in css
    assert ".training-dashboard-head" in css
    assert ".training-run-state" in css
    assert ".training-run-summary" in css
    assert ".training-panels.training-dashboard" in css
    assert ".metrics-panel,\n.chart-panel" in css
    assert "grid-column: 1 / -1;" in css
    assert ".history-manager-grid" in css
    assert ".history-manager-row" in css
    assert ".history-manager-collection" in css
    assert ".history-manager-config-group" in css
    assert "--history-manager-bg" in css
    assert "--history-collection-bg" in css
    assert "--history-config-bg" in css
    assert "--history-row-bg" in css
    assert "--history-row-hover-bg" in css
    assert "--history-level-border" in css
    assert "--history-collection-accent" in css
    assert "--history-config-accent" in css
    assert "--history-ungrouped-accent" in css
    assert ".history-manager-collection.ungrouped" in css
    assert "border-left: 5px solid var(--history-collection-accent)" in css
    assert "border-left: 3px solid var(--history-config-accent)" in css
    assert "background: var(--history-row-bg)" in css
    assert "background: var(--history-row-hover-bg)" in css
    assert ".history-row-state.done" in css
    assert ".history-row-state.running" in css
    assert ".history-row-state.queued" in css
    assert ".history-row-state.interrupted" in css
    assert ".history-collections-workbench" in css
    assert ".history-collections-body" in css
    assert ".history-collection-card" in css
    assert ".history-collection-card.active" in css
    assert ".history-collection-select-dialog" in css
    assert ".history-collection-select-list" in css
    assert ".history-config-group-card" in css
    assert ".history-config-group-task-list" in css
    assert ".history-manager-group-head" in css
    assert ".history-manager-group-actions" in css
    assert ".history-manager-stat.active" in css
    assert ".history-detail-dialog" in css
    assert ".history-detail-overview-dashboard" in css
    assert ".history-detail-progress" in css
    assert ".history-detail-metrics-body" in css
    assert ".history-curve-workbench" in css
    assert ".history-curve-toolbar" in css
    assert ".history-curve-svg" in css
    assert ".history-curve-inspector" in css
    assert ".history-curve-segment-line" in css
    assert ".history-detail-limit-note" in css
    assert ".history-system-trends" in css
    assert ".history-system-table" in css
    assert ".history-detail-section.info" not in css
    assert ".preview-panel-dialog" in css
    assert ".preview-panel-body" in css

    assert "this.lrColor" in chart_source
    assert "_drawLrLine" in chart_source
    assert "updatePointMetadata" in chart_source
    assert "LR:" in chart_source
    assert "_formatLr" in chart_source


def test_output_scope_group_does_not_expose_unwired_stage_resolution_dialog() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    section = _section(source, "title: '输出格式与训练范围'", "title: '方法内部与实验架构'")
    create_group = _section(source, "function createGroup", "function createOpenStageResolutionDialogButton")

    assert "className: 'config-group-output-scope'" in section
    assert "header.appendChild(createOpenStageResolutionDialogButton());" not in create_group
    assert 'id="stage-resolution-dialog"' not in html
    assert 'class="preview-dialog stage-resolution-dialog"' not in html
    assert "stage-resolution-dialog-body" not in html
    assert "btn-open-stage-resolution-dialog" not in _section(source, "function installBeginnerTooltips", "// ── 工具函数 ──")


def test_config_form_hides_retired_and_unread_fields() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    method_section = _section(source, "title: '方法内部与实验架构'", "title: 'Soft Tokens 参数'")
    assert "'use_hydra'," not in method_section
    assert "'use_sigma_router'," not in method_section
    assert "'use_fei_router'," in source
    assert "const RETIRED_CONFIG_FORM_FIELDS = new Set([" in source
    assert "if (RETIRED_CONFIG_FORM_FIELDS.has(key)) return true;" in source
    assert "['weight_decay', new Set(['spd'])]" in source
    assert "const SOFT_TOKENS_UI_DEFAULT_FIELDS = new Set([]);" in source


def test_config_page_hides_unimplemented_dataset_placeholder() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    dataset_picker = _section(source, "function renderConfigDatasetPicker", "function createConfigDatasetCurrentSummary")
    assert "btn-open-unnamed-dataset-dialog" not in dataset_picker
    assert "待命名" not in dataset_picker
    assert 'id="unnamed-dataset-dialog"' not in html


def test_sample_prompts_editor_preserves_raw_text_when_needed() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    render_body = _section(source, "function renderSamplePromptRows", "function switchSamplePromptsEditorToTextMode")
    serialize_body = _section(source, "function serializeSamplePromptsEditor", "function samplePromptRowFromElement")

    assert "samplePromptsContentNeedsTextMode(content)" in render_body
    assert "sample-prompts-textarea" in render_body
    assert "return editor.querySelector('.sample-prompts-textarea')?.value || '';" in serialize_body
    assert "function createSamplePromptTextModeButton" in source


def test_dataset_json_caption_switch_ui_is_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    defaults_editor = _section(source, "function createDatasetDefaultsEditor", "function createDatasetConfigInput")
    item_factory = _section(source, "function createDatasetEditorItem", "function createDatasetEditorRow")
    row_factory = _section(source, "function createDatasetEditorRow", "function createDatasetExperimentalFeaturesEditor")
    experimental_factory = _section(source, "function createDatasetExperimentalFeaturesEditor", "function createDatasetRowSettingsEditor")
    caption_extension_factory = _section(source, "function createDatasetCaptionExtensionEditor", "function createDatasetNlTagMixEditor")
    mix_factory = _section(source, "function createDatasetNlTagMixEditor", "function normalizeCaptionSourceMode")
    caption_source_factory = _section(source, "function createDatasetRowCaptionSourceModeEditor", "function createDatasetRowSettingInput")
    normalize_factory = _section(source, "function normalizeNlTagMix", "function updateDatasetDefault")
    row_update_factory = _section(source, "function updateDatasetEditorRowSetting", "function updateDatasetEditorRowNlTagMix")

    assert "通用标注设置" in defaults_editor
    assert "这里只保留 keep_tokens" in defaults_editor
    assert "文本标注扩展名等兼容项在每组数据集的高级区配置" in defaults_editor
    assert "['caption_extension', 'text']" not in defaults_editor
    assert "['keep_tokens', 'number']" in defaults_editor
    assert "['prefer_json_caption', 'switch', 'switch']" not in defaults_editor

    assert "createDatasetEditorItem(row, index)" in source
    assert "dataset-editor-item" in item_factory
    assert "createDatasetEditorRow(row, index)" in item_factory
    assert "createDatasetExperimentalFeaturesEditor(row, index)" in item_factory
    assert "createDatasetExperimentalFeaturesEditor(row, index)" not in row_factory
    assert "createDatasetRowCaptionSourceModeEditor(settings, index)" in row_factory
    assert "createDatasetNlTagMixEditor(row, index)" in row_factory
    assert "实验性/高级/旧功能" in experimental_factory
    assert "dataset-experimental-features" in experimental_factory
    assert "createDatasetExperimentalScopePicker(index)" in experimental_factory
    assert "createDatasetCaptionExtensionEditor(row, index)" in experimental_factory
    assert "createDatasetNlTagMixEditor(row, index)" not in experimental_factory
    assert "createDatasetRowCaptionSourceModeEditor(settings, index)" not in experimental_factory
    assert "对应第 ${index + 1} 组数据集" in experimental_factory
    assert "这些选项按当前这组数据集单独保存" in experimental_factory
    assert "生效范围 / 对多数据集负责" in source
    assert "全选数据集" in source
    assert "datasetExperimentalScopeIndices" in source
    assert "setDatasetExperimentalScopeIndices" in source
    assert "datasetValidTargetIndices" in source

    assert ".dataset-editor-item" in css
    assert ".dataset-experimental-features" in css
    assert ".dataset-experimental-body" in css
    assert ".dataset-experimental-scope" in css
    assert ".dataset-scope-chip" in css
    assert ".dataset-caption-source" in css
    assert ".dataset-caption-source-options" in css
    assert ".dataset-caption-source-option.selected" in css
    assert ".dataset-caption-source-title-row" in css
    assert ".dataset-caption-source-help-toggle" in css
    assert ".dataset-caption-extension-advanced" in css
    assert ".dataset-caption-extension-input" in css
    assert ".dataset-caption-extension-help" in css

    assert "文本标注扩展名 / caption_extension" in caption_extension_factory
    assert "高级兼容项：仅在 txt 来源或 auto 回退到文本 sidecar 时使用。" in caption_extension_factory
    assert "updateDatasetEditorRowsSettingValue(" in caption_extension_factory
    assert "datasetExperimentalScopeIndices(index)" in caption_extension_factory
    assert "'caption_extension'" in caption_extension_factory
    assert "createHelpContent('caption_extension'" in caption_extension_factory

    assert "normalizeNlTagMix(row.nl_tag_mix)" in row_factory
    assert "nlTagMixSummary(mix)" in row_factory
    assert "captions格式nl/tag权重调整" in mix_factory
    assert "自动识别 nl/tag" not in mix_factory
    assert "面向 DiffPipeForge captions.json 的多标注数据集优化" in mix_factory
    assert "按短标签串和自然语言句子判断 tag/nl" in mix_factory
    assert "按比例抽样重建运行时 captions.json" in mix_factory
    assert "写入 results.json" in mix_factory
    assert "从同一父目录下的 tag/ 与 nl/ 固定抽样。" not in mix_factory
    assert "tag 占比" in mix_factory
    assert "ratioInput.type = 'range';" in mix_factory
    assert "updateDatasetEditorRowNlTagMix(index" in mix_factory
    assert "datasetExperimentalScopeIndices(index)" not in mix_factory

    assert "caption_source_mode" in caption_source_factory
    assert "默认 auto 自动识别" in caption_source_factory
    assert "datasetCaptionSourceHelpSeq" in source
    assert "dataset-caption-source-help-toggle" in caption_source_factory
    assert "helpBtn.textContent = '?'" in caption_source_factory
    assert "helpBtn.setAttribute('aria-expanded', 'false')" in caption_source_factory
    assert "notes.hidden = true" in caption_source_factory
    assert "helpBtn.classList.toggle('active', nextVisible)" in caption_source_factory
    assert "sd-scripts" in caption_source_factory
    assert "AnimaLoraToolkit" in caption_source_factory
    assert "DiffPipeForge" in caption_source_factory
    assert '"1.png+1.txt"*n = sd-scripts格式标注' in caption_source_factory
    assert '"1.png+1.json"*n = AnimaLoraToolkit格式标注' in caption_source_factory
    assert '"png*n"+captions.json = DiffPipeForge格式标注' in caption_source_factory
    assert "caption_extension 仅影响 txt 来源或 auto 回退到文本标注" in caption_source_factory
    assert "json / captions.json 模式会忽略它" in caption_source_factory
    assert "updateDatasetEditorRowsSettingValue(" in caption_source_factory
    assert "[index]" in caption_source_factory
    assert "datasetExperimentalScopeIndices(index)" not in caption_source_factory
    assert "input.type === 'checkbox'" in row_update_factory
    assert "DEFAULT_NL_TAG_MIX" in normalize_factory
    assert "nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix)" in normalize_factory
    assert ".dataset-nl-tag-mix" in css
    assert ".dataset-nl-tag-summary" in css
