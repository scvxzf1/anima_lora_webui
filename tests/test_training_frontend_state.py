from __future__ import annotations

from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "web" / "static" / "app.js"
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
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    history_section = _section(source, "async function loadTrainingHistoryList()", "function groupHistoryTasks")
    detail_section = _section(source, "function renderHistoryManagerDetail", "async function loadHistoryTask")
    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")

    assert "training-history-manager" in html
    assert "history-manager-search" in html
    assert "history-collection-search" in html
    assert "history-config-group-search" in html
    assert "集合搜索" in html
    assert "配置组搜索" in html
    assert "history-group-mode" in html
    assert "集合分组" in html
    assert "集合管理" in html
    assert "配置分组" in html
    assert "平铺列表" in html
    assert "btn-history-collections-workbench" in html
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
    assert "historyManagerFilteredTasks" in history_section
    assert "function historyManagerBaseFilteredTasks" in history_section
    assert "function historyManagerVisibleTasks" in history_section
    assert "historyManagerVisibleTasks(historyManagerBaseFilteredTasks())" in history_section
    assert "historyConfigGroupVisibleForSearch" in history_section
    assert "uniqueHistoryTasks" in history_section
    assert "createHistoryManagerRow" in history_section
    assert "renderHistoryManagerGrouped" in history_section
    assert "resetTrainingExpandedStateOnLeave" in history_section
    assert "collapseVisibleHistoryManagerGroups" in history_section
    assert "collapsedHistoryCollections.add(collection.key)" in history_section
    assert "collapsedHistoryConfigGroups.add(historyConfigGroupCollapseKey(group, collection.key))" in history_section
    assert "historyStatFilterIsActive" in history_section
    assert "archived: 'all'" in history_section
    assert "next.kind = state" in history_section
    assert "createHistoryManagerCollectionSection" in history_section
    assert "createHistoryManagerConfigGroupSection" in history_section
    assert "createHistoryManagerCollectionSection(collection, splitCollections, collections)" in history_section
    assert "collection: '__all__'" in history_section
    assert "groups: collection.groups" in source
    assert "renderHistoryCollectionsWorkbench" in history_section
    assert "createHistoryCollectionWorkbenchCard" in history_section
    assert "createHistoryConfigGroupWorkbenchCard" in history_section
    assert "historyCollectionWorkbenchTarget" in source
    assert "historyCollectionSettings" in source
    assert "historyCollectionSearch" in source
    assert "historyConfigGroupSearch" in source
    assert "historyTaskMatchesCollectionSearch" in source
    assert "historyTaskMatchesConfigGroupSearch" in source
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
    assert "groupHistoryTasks(visible)" in history_section
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
    assert "该配置组分布在" in source
    assert "selectedHistoryTaskIds" in source
    assert "applyHistoryBatchAction" in source
    assert "deleteHistoryTasksThorough" in source
    assert "confirm_text: confirmText" in source
    assert "彻底删除" in source
    assert "/api/training/history/batch" in source
    assert "openHistoryDetailDialog" in source
    assert "closeHistoryDetailDialog" in source

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
    assert "history-filter-kind" in listener_section
    assert "history-group-mode" in listener_section
    assert "history-detail-tab" in listener_section
    assert "btn-close-history-detail" in listener_section

    assert ".training-workspace.history-wide" in css
    assert ".history-manager-grid" in css
    assert ".history-manager-row" in css
    assert ".history-manager-collection" in css
    assert ".history-manager-config-group" in css
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


def test_output_scope_group_opens_stage_resolution_dialog() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    section = _section(source, "title: '输出格式与训练范围'", "title: '方法内部与实验架构'")
    create_group = _section(source, "function createGroup", "function createOpenStageResolutionDialogButton")
    button_factory = _section(source, "function createOpenStageResolutionDialogButton", "function createFillGlobalModelPathsButton")

    assert "className: 'config-group-output-scope'" in section
    assert "if (extraClass === 'config-group-output-scope')" in create_group
    assert source.count("header.appendChild(createOpenStageResolutionDialogButton());") == 1

    assert "btn-open-stage-resolution-dialog" in button_factory
    assert "btn.textContent = '阶段调度';" in button_factory
    assert "btn.addEventListener('click', openStageResolutionDialog);" in button_factory
    assert "function openStageResolutionDialog()" in button_factory
    assert "stage-resolution-dialog" in button_factory
    assert "showModal" in button_factory
    assert "enabled: false" in source
    assert "stage-resolution-enable-toggle" in button_factory
    assert "function setStageResolutionEnabled(enabled)" in button_factory
    assert "stageResolutionState.enabled = Boolean(enabled);" in button_factory
    assert "启用阶段调度" in button_factory

    assert 'id="stage-resolution-dialog"' in html
    assert 'class="preview-dialog stage-resolution-dialog"' in html
    assert "<h2>阶段分辨率调度</h2>" in html
    assert "stage-resolution-dialog-body" in html

    for snippet in (
        "function renderStageResolutionDialog()",
        "function drawStageResolutionChart()",
        "function createStageResolutionEditor",
        "function createStageResolutionTable",
        "stageResolutionState",
        "STAGE_RESOLUTION_STEPS_PER_EPOCH",
    ):
        assert snippet in source


def test_dataset_json_caption_switch_ui_is_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    defaults_editor = _section(source, "function createDatasetDefaultsEditor", "function createDatasetConfigInput")
    input_factory = _section(source, "function createDatasetConfigInput", "function createDatasetConfigSwitch")
    switch_factory = _section(source, "function createDatasetConfigSwitch", "function datasetConfigLabel")

    assert "['prefer_json_caption', 'switch', 'switch']" in defaults_editor
    assert "layout === 'switch' ? 'switch' : ''" in defaults_editor
    assert "return createDatasetConfigSwitch(key, defaults);" in input_factory
    assert "input.type = 'checkbox';" in switch_factory
    assert "dataset-json-switch" in switch_factory
    assert "已启用" in switch_factory
    assert "已关闭" in switch_factory
    assert "updateDatasetConfigValue(key, input);" in switch_factory

    assert ".dataset-config-field.switch" in css
    assert ".dataset-json-switch" in css
    assert ".dataset-json-switch.enabled" in css
