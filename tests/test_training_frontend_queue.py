# Split from test_training_frontend_state.py (queue)

from __future__ import annotations

from tests.frontend_test_support import *  # noqa: F403


import tests.frontend_test_support as _frontend_support
for _k, _v in vars(_frontend_support).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

def test_training_queue_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    legacy_source = _anima_app_container_text()
    index_source = _frontend_module_text("js/features/anima-app/index.js")
    preflight_source = _frontend_module_text("js/features/preflight-dialog/index.js")
    live_log_source = _frontend_module_text("js/features/live-log/index.js")
    queue_index = _frontend_module_text("js/features/queue/index.js")
    queue_state = _frontend_module_text("js/features/queue/state.js")
    queue_api = _frontend_module_text("js/features/queue/api.js")
    queue_render = _frontend_module_text("js/features/queue/render.js")
    queue_actions = _frontend_module_text("js/features/queue/actions.js")
    queue_enqueue = _frontend_module_text("js/features/queue/enqueue.js")
    feature_ensurers = _frontend_module_text("js/features/anima-app/helpers/feature-ensurers.js")
    training_state_source = _frontend_module_text("js/features/anima-app/state/training-state.js")
    queue_feature_source = "\n".join([
        queue_index,
        queue_state,
        queue_api,
        queue_render,
        queue_actions,
        queue_enqueue,
    ])
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    queue_section = queue_feature_source
    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")
    group_actions_impl = _frontend_feature_text(
        "js/features/toml-manager/drag-render.js",
        "js/features/toml-manager/drag-actions.js",
    )
    group_actions = group_actions_impl
    current_queue = _section(queue_enqueue, "async function queueCurrentTrainingFromConfig", "async function enqueueTrainingFromConfig")
    enqueue_section = _section(queue_enqueue, "async function enqueueTrainingFromConfig", "async function enqueueTrainingQueueRequest")
    queue_view_source = _frontend_module_text("js/features/queue/view-mode.js")
    status_polling_source = _frontend_module_text("js/features/live-training/status-polling.js")
    view_section = queue_view_source
    stop_section = _section(live_log_source, "async function stopTraining()", "export function connectWebSocket")
    poll_section = _section(status_polling_source, "async function pollStatus", "function applyStatusSnapshotFallbacks")
    assert "configureQueueFeatureEnsurer(ctx, appShellState, {" in legacy_source
    assert "createQueueFeature(ctx, deps)" in feature_ensurers
    assert "globalThis.ensureQueueFeature" not in legacy_source
    assert "ensureQueueFeature().bindQueueEvents();" in listener_section
    for name in (
        "loadTrainingQueue",
        "renderTrainingQueue",
        "updateTrainingQueueFromPayload",
        "updateRunningQueueProgress",
        "queueCurrentTrainingFromConfig",
        "enqueueTrainingFromConfig",
        "enqueueTrainingQueueRequest",
        "enqueueTrainingQueueBatchRequest",
        "queueResumeTrainingFromCheckpoint",
        "bindQueueEvents",
    ):
        assert name in queue_index
    assert "function renderTrainingQueue()" in queue_section
    assert "function renderTrainingQueueManager()" in queue_section
    assert "function queueManagerSections(state)" in queue_state
    assert "const isErrorOnly = payload.ok === false && !hasItems" in queue_state
    assert "status: payload.status === undefined" in queue_state
    assert "isErrorOnly ? (previous.items || []) : []" in queue_state
    assert "isErrorOnly ? (previous.summary || {}) : {}" in queue_state
    assert "function createTrainingQueueSection" in queue_section
    assert "function createTrainingQueueItem" in queue_section
    assert "function createTrainingQueueManagerItem" in queue_section
    assert "filter: 'actionable'" in queue_state
    assert "let trainingQueueState" not in legacy_source
    assert "let trainingQueueFilter" not in legacy_source
    assert "async function toggleTrainingQueuePause()" in queue_section
    assert "cancelAllQueueItems" in queue_section
    assert "removeQueueItemFromList" in queue_section
    assert "移除列表" in queue_section
    assert "只会将这条记录从队列界面移除" in queue_section
    assert "event.preventDefault()" in queue_section
    assert "event.stopPropagation()" in queue_section
    assert "HTMLDetailsElement" in queue_section
    assert "delete_runtime: true" not in queue_section
    assert "queueDeleteRuntimeMessage" not in queue_section
    assert "queueRuntimeDirLabel" not in queue_section
    assert "新任务已加入队列" in queue_section
    assert "移除原记录" in queue_section
    assert "feedback:" in queue_state
    assert "setQueueFeedback" in queue_state
    assert "beginQueueFeedback" in queue_actions
    assert "finishQueueFeedback" in queue_actions
    assert "queueMoveDirectionLabel" in queue_actions
    assert "正在刷新队列状态" in queue_actions
    assert "清理已完成记录" in queue_section
    assert "清理已取消记录" in queue_section
    assert "retryQueueItem" in queue_section
    assert "cancelWaitingQueueItems" in queue_section
    assert "clearCompletedQueueItems" in queue_section
    assert "clearCanceledQueueItems" in queue_section
    assert "focusQueueFilterAfterTerminalClear" in queue_section
    assert "已完成记录已保留" in queue_section
    assert "清理已取消不会影响这里" in queue_section
    assert "清理已完成不会影响这里" in queue_section
    assert "缓存或任何实际文件" in queue_section
    assert "btn-clear-completed-queue" in html
    assert "btn-clear-canceled-queue" in html
    assert ".training-queue-item-more[open]" in css
    assert "z-index: 130" in css
    assert "state.filter" in queue_section
    assert "renderQueueManagerOverview" in queue_section
    assert "queueFilterLabel" in queue_section
    assert "createQueueFactRow" in queue_section
    assert "queueShortId" in queue_section
    assert "updateQueueFilterButton" in queue_section
    assert "queueFilterCount" in queue_section
    assert "queueEmptyStateText" in queue_section
    assert "updateQueueActionHints" in queue_section
    assert "renderQueueFeedback" in queue_section
    assert "queueFeedbackBusyAction" in queue_section
    assert "queueFeedbackItemState" in queue_section
    assert "queue-action-busy" in queue_section
    assert "aria-pressed" in queue_section
    assert "aria-disabled" in queue_section
    assert "aria-busy" in queue_section
    assert "queueDetailsSummaryText" in queue_section
    assert ".training-queue-overview-item" in css
    assert ".training-queue-feedback" in css
    assert ".training-queue-manager-item.queue-feedback-pending" in css
    assert ".task-history-action.queue-action-busy" in css
    assert ".training-queue-facts" in css
    assert ".training-queue-filter b" in css
    assert "grid-template-columns: repeat(5, minmax(0, 1fr))" in css
    assert "/api/training/queue" in queue_section
    assert "/api/training/queue/settings" in queue_section
    assert "/api/training/queue/cancel-all" in queue_section
    assert "/api/training/queue/abort-after-current" in queue_section
    assert "/api/training/queue/force-abort" in queue_section
    assert "/api/training/queue/cancel-waiting" in queue_section
    assert "/api/training/queue/clear-completed" in queue_section
    assert "/api/training/queue/clear-canceled" in queue_section

    assert "training-queue-manager" in html
    assert "training-queue-manager-overview" in html
    assert "training-queue-feedback" in html
    assert 'role="status" aria-live="polite"' in html
    assert "btn-training-queue-view" in html
    assert "training-queue-failure-policy" in html
    assert 'data-queue-filter="actionable">待处理' in html
    assert "training-queue-more-menu" in html
    assert "btn-queue-from-config" in html
    assert "btn-open-history-manager" in html
    assert "未归档 · 最新 6 个训练任务" in html
    assert "queueCurrentTrainingFromConfig" in listener_section
    _assert_imports_from(legacy_source, "../helpers/queue-view-bridge.js", ("showTrainingView",))
    assert "on('btn-open-history-manager', 'click', () => showTrainingView('history'))" in listener_section
    assert "const mainWide = isQueue || isHistory;" in view_section
    assert "workspace.classList.toggle('main-wide', mainWide)" in view_section
    assert "trainingRoot.classList.toggle('history-mode', isHistory)" in view_section
    assert "workspace.classList.toggle('history-mode', isHistory)" in view_section
    assert "history-wide" not in view_section
    assert ".slice(0, 6)" in source

    summary_panel = _section(html, '<section class="panel training-queue-panel"', '<section class="panel task-history-panel">')
    manager_panel = _section(html, '<section id="training-queue-manager"', '<section id="training-history-placeholder"')
    assert "btn-cancel-all-queue" not in summary_panel
    assert "btn-abort-queue-after-current" not in summary_panel
    assert "btn-force-abort-queue" not in summary_panel
    assert "btn-cancel-waiting-queue" not in summary_panel
    assert "btn-clear-finished-queue" not in summary_panel
    assert "btn-clear-completed-queue" not in summary_panel
    assert "btn-clear-canceled-queue" not in summary_panel
    assert "btn-cancel-all-queue" in manager_panel
    assert "btn-abort-queue-after-current" in manager_panel
    assert "btn-force-abort-queue" in manager_panel
    assert "中止后续队列" in manager_panel
    assert "强制中止队列" in manager_panel
    assert "btn-cancel-waiting-queue" in manager_panel
    assert "btn-clear-finished-queue" not in manager_panel
    assert "btn-clear-completed-queue" in manager_panel
    assert "btn-clear-canceled-queue" in manager_panel

    ws_section = _section(live_log_source, "function handleWsMessage", "function appendLog")
    assert "case 'queue':" in ws_section
    assert "updateTrainingQueueFromPayload(msg);" in ws_section

    start_section = _section(source, "async function startTrainingUnchecked", "function enterLiveTrainingForNewRun")
    assert "enqueueTrainingFromConfig" in start_section
    assert "chooseTrainingLaunchMode" in source
    assert "await enqueueTrainingFromConfig(variant, preset, methodsSubdir" in current_queue
    assert "options.configFile || deps.currentTrainingConfigFile()" in enqueue_section
    assert "const startPaused = options.startPaused !== false" in enqueue_section
    assert "队列会保存独立运行配置并保持暂停" in enqueue_section
    assert "includeContinueSource === false" in enqueue_section
    assert "const wasDisabled = Boolean(stopBtn?.disabled)" in stop_section
    assert "await pollStatus();" in stop_section
    assert "await loadTrainingQueue();" in stop_section
    assert "setTrainingHealthNotice(message, 'error')" in stop_section
    assert "trainingStatusPollFailures: 0" in training_state_source
    assert "trainingStatusPollTimer: null" in training_state_source
    assert "trainingStatusPollPromise: null" in training_state_source
    assert "trainingStatusPollForceReplayMetrics: false" in training_state_source
    assert "installLegacyStateGlobals(runtime)" not in index_source
    assert "import { installLegacyStateGlobals }" not in index_source
    assert "if (status.ok === false) throw new Error(status.error || '读取训练状态失败')" in poll_section
    assert "if (target.trainingStatusPollPromise) return target.trainingStatusPollPromise;" in poll_section
    assert "target.trainingStatusPollFailures < 3" in poll_section
    assert "训练状态轮询连续失败" in poll_section
    assert "setTrainingHealthNotice(message, 'error')" in poll_section
    assert "async function enqueueTrainingQueueRequest" in queue_enqueue
    assert "async function enqueueTrainingQueueBatchRequest" in queue_enqueue
    assert "queueBatchApiUnsupported(res)" in queue_enqueue
    assert "enqueueTrainingQueueBatchCompat(requestOptions, res)" in queue_enqueue
    assert "enqueueTrainingQueueBatchRootCompat(options, aliasRes || unsupported)" in queue_enqueue
    assert "enqueueTrainingQueueBatchFallback(options, rootRes || unsupported)" in queue_enqueue
    assert "method not allowed" in queue_enqueue
    assert "status_code" in queue_enqueue
    assert "if (!queueBatchApiUnsupported(unsupported)) throw e;" in queue_enqueue
    assert "async function abortQueueAfterCurrent" in queue_section
    assert "async function forceAbortQueue" in queue_section
    assert "showAppConfirmDialog" in queue_section
    assert "当前正在运行的任务会继续执行到完成" in queue_section
    assert "会立即停止当前正在运行的训练/预处理进程" in queue_section
    assert "on('btn-abort-queue-after-current', 'click', abortQueueAfterCurrent)" in queue_section
    assert "on('btn-force-abort-queue', 'click', forceAbortQueue)" in queue_section
    assert "abortTrainingQueueAfterCurrent(ctx)" in queue_section
    assert "forceAbortTrainingQueue(ctx)" in queue_section
    assert "const abortAfterCurrentBtn = document.getElementById('btn-abort-queue-after-current')" in queue_section
    assert "const forceAbortBtn = document.getElementById('btn-force-abort-queue')" in queue_section
    assert "function queueBackendRunning()" in queue_section
    assert "state.queue.status === 'running'" in queue_section
    assert "deps.getTrainingRuntime()?.state === 'running'" in queue_section
    assert "counts.queued <= 0" in queue_section
    assert "counts.queued + counts.running" in queue_section
    assert "createTomlGroupActionButton('加入队列', () => enqueueTomlGroupToQueue(group)" in group_actions
    assert "queueableTomlGroupFiles(group)" in group_actions
    assert "async function enqueueTomlGroupToQueue" in source
    assert "tomlItemQueueEntry(item, preset)" in source
    assert "tomlGroupQueueFailureLabel(item, failure, failedIndex)" in source
    assert "label: label === '未命名配置文件' ? '' : label" in source
    assert "failure.label || failure.filename" in source
    assert "第 ${fallbackIndex} 个配置" in source
    assert "showTomlGroupQueueConfirmDialog(group, files)" in source
    assert "队列会保持暂停，等待你手动继续" in group_actions
    assert "startPaused: true" in group_actions
    assert "/api/training/queue/batch/start" in queue_api
    assert "/api/training/queue/batch-start" in queue_api
    assert "/api/training/queue', options" in queue_api
    assert "start_paused: Boolean(options.startPaused)" in queue_api
    assert "if (!Object.prototype.hasOwnProperty.call(data, 'status_code')) data.status_code = res.status;" in source
    assert "item?.config_file" in source
    assert ".toml-group-action-btn-queue" in css
    assert "导出单个" in html
    assert "createTomlGroupActionButton('导出分组', () => exportTomlGroup(group)" in group_actions
    assert "exportableTomlGroupFiles(group)" in group_actions
    assert "async function exportTomlGroup" in source
    assert "createTomlZipBlob(entries)" in source
    assert "downloadBlob(blob, filename)" in source
    assert "/api/config/raw?file=${encodeURIComponent(path)}" in source
    assert "uniqueZipEntryName" in source
    assert "ZIP_CRC_TABLE" in source
    assert "内含 ${files.length} 个独立 TOML 文件" in source
    assert ".toml-group-action-btn-export" in css


def test_resume_queue_button_is_wired() -> None:
    legacy_source = _anima_app_container_text()
    resume_source = _frontend_module_text("js/features/history-detail/resume/panel.js")

    resume_section = _section(resume_source, "function renderResumePanelState", "return { renderResumePanelState")
    history_detail_deps = _section(legacy_source, "function ensureHistoryDetailFeature", "// ── 初始化 ──")
    assert "btn-queue-resume-training" in resume_section
    assert "queueBtn.disabled" in resume_section
    assert "selected.resume_available !== false" in resume_section
    assert "resumeCheckpointRemainingText(selected)" in resume_section
    assert "deps.shouldRenderInlineResumePanel?.() !== true" in resume_section
    assert "resetInlineResumePanel(panel, select, btn, queueBtn, summary, status);" in resume_section
    assert "syncHistoryDetailResumeContent();" in resume_section
    assert "shouldRenderInlineResumePanel" in history_detail_deps

    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")
    assert "queueResumeTrainingFromCheckpoint" in legacy_source
    assert "btn-queue-resume-training" in listener_section


def test_training_queue_renderer_updates_dom_fixture() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for queue DOM fixture checks")
    script = r"""
import { createQueueRenderer } from './web/static/js/features/queue/render.js';
import { createQueueState, updateQueueStateFromPayload } from './web/static/js/features/queue/state.js';

const nodes = new Map();
let createdSeq = 0;

function makeStyle() {
    const values = {};
    return {
        values,
        setProperty(name, value) { values[name] = String(value); },
        removeProperty(name) { delete values[name]; },
    };
}

function makeClassList(node) {
    const values = new Set(String(node.className || '').split(/\s+/).filter(Boolean));
    return {
        add: (...names) => names.forEach((name) => values.add(name)),
        remove: (...names) => names.forEach((name) => values.delete(name)),
        toggle: (name, force) => {
            const enabled = force === undefined ? !values.has(name) : Boolean(force);
            if (enabled) values.add(name);
            else values.delete(name);
            return enabled;
        },
        contains: (name) => values.has(name),
        values: () => [...values].sort(),
    };
}

function matches(node, selector) {
    if (selector.startsWith('.')) return String(node.className || '').split(/\s+/).includes(selector.slice(1));
    return node.tagName === selector.toUpperCase();
}

function findFirst(node, selector) {
    for (const child of node.children || []) {
        if (matches(child, selector)) return child;
        const nested = findFirst(child, selector);
        if (nested) return nested;
    }
    return null;
}

function makeNode(id, tagName = 'div') {
    const item = {
        id,
        tagName: tagName.toUpperCase(),
        style: makeStyle(),
        dataset: {},
        hidden: false,
        disabled: false,
        className: '',
        textContent: '',
        title: '',
        value: '',
        type: '',
        children: [],
        parentNode: null,
        attrs: {},
        set innerHTML(value) {
            this._innerHTML = String(value);
            this.children = [];
        },
        get innerHTML() {
            return this._innerHTML || '';
        },
        setAttribute(name, value) { this.attrs[name] = String(value); this[name] = String(value); },
        removeAttribute(name) { delete this.attrs[name]; delete this[name]; },
        append(...children) { children.forEach((child) => this.appendChild(child)); },
        appendChild(child) { child.parentNode = this; this.children.push(child); return child; },
        insertBefore(child, before) {
            child.parentNode = this;
            const index = this.children.indexOf(before);
            if (index < 0) this.children.push(child);
            else this.children.splice(index, 0, child);
            return child;
        },
        replaceChildren(...children) {
            this.children = [];
            children.forEach((child) => this.appendChild(child));
        },
        addEventListener() {},
        querySelector(selector) { return findFirst(this, selector); },
        closest() { return null; },
    };
    item.classList = makeClassList(item);
    return item;
}

function node(id) {
    if (!nodes.has(id)) nodes.set(id, makeNode(id));
    return nodes.get(id);
}

function createElement(tag) {
    return makeNode(`created-${tag}-${createdSeq++}`, tag);
}

const filterKeys = ['actionable', 'all', 'queued', 'running', 'error', 'done', 'canceled'];
const filterNodes = filterKeys.map((key) => {
    const item = makeNode(`filter-${key}`, 'button');
    item.className = 'training-queue-filter';
    item.classList = makeClassList(item);
    item.dataset.queueFilter = key;
    return item;
});

globalThis.document = {
    getElementById: (id) => node(id),
    createElement,
    querySelectorAll: (selector) => selector === '.training-queue-filter' ? filterNodes : [],
};

const state = createQueueState();
const runtime = {
    state: 'running',
    progressCurrent: 4,
    progressTotal: 10,
    progressLabel: 'train',
    progressRate: '2it/s',
};
const renderer = createQueueRenderer({
    state,
    deps: {
        renderTrainingViewMode: () => { node('view-mode').dataset.rendered = 'true'; },
        getTrainingRuntime: () => runtime,
        runLabelFromPath: (value) => String(value || '').split('/').pop() || '',
    },
    actions: {
        moveQueueItem() {},
        cancelQueueItem() {},
        retryQueueItem() {},
        removeQueueItemFromList() {},
    },
});

updateQueueStateFromPayload(state, {
    ok: true,
    paused: false,
    failure_policy: 'pause',
    status: 'running',
    items: [
        {
            id: 'run-1',
            state: 'running',
            runtime_config_file: 'output/runs/run-1/config.runtime.toml',
            source_config_file: 'configs/imported/lora.toml',
            variant: 'lora',
            preset: 'default',
            methods_subdir: 'gui-methods',
            created_at_text: '10:00',
            started_at_text: '10:01',
        },
        { id: 'queue-2', state: 'queued', variant: 'loha', preset: 'low_vram', created_at_text: '10:02' },
        { id: 'error-3', state: 'error', variant: 'lokr', message: 'boom', created_at_text: '10:03' },
        { id: 'done-4', state: 'done', variant: 'vera', created_at_text: '10:04', finished_at_text: '10:05' },
    ],
});
renderer.renderTrainingQueue();

const summary = node('training-queue-summary');
const badge = node('training-queue-tab-badge');
const managerStatus = node('training-queue-manager-status');
const stats = node('training-queue-stats');
const managerList = node('training-queue-manager-list');
const firstManagerCard = managerList.querySelector('.training-queue-manager-item');
const progress = firstManagerCard?.querySelector('.training-queue-running-progress');
const runningFilter = filterNodes.find((item) => item.dataset.queueFilter === 'running');
const allFilter = filterNodes.find((item) => item.dataset.queueFilter === 'all');

console.log(JSON.stringify({
    summaryText: summary.textContent,
    summaryClass: summary.className,
    badgeHidden: badge.hidden,
    badgeText: badge.textContent,
    managerStatus: managerStatus.textContent,
    statsCount: stats.children.length,
    managerSections: managerList.children.length,
    progressText: progress?.querySelector('span')?.textContent || '',
    progressStyle: firstManagerCard?.style.values['--queue-progress'] || '',
    runningFilterPressed: runningFilter?.attrs['aria-pressed'],
    runningFilterTitle: runningFilter?.title,
    allFilterTitle: allFilter?.title,
    abortDisabled: node('btn-abort-queue-after-current').disabled,
    forceAbortDisabled: node('btn-force-abort-queue').disabled,
    clearCompletedDisabled: node('btn-clear-completed-queue').disabled,
    viewRendered: node('view-mode').dataset.rendered,
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "summaryText": "正在运行：lora.toml · 等待 1 个",
        "summaryClass": "training-queue-summary running error",
        "badgeHidden": False,
        "badgeText": "2",
        "managerStatus": "队列运行中 · 等待 1 个 · 异常 1 个 · 失败后暂停队列",
        "statsCount": 6,
        "managerSections": 3,
        "progressText": "train: 4/10 · 40.0% · 2it/s",
        "progressStyle": "40.0%",
        "runningFilterPressed": "false",
        "runningFilterTitle": "运行：1 项",
        "allFilterTitle": "全部：4 项",
        "abortDisabled": False,
        "forceAbortDisabled": False,
        "clearCompletedDisabled": False,
        "viewRendered": "true",
    }


def test_queue_state_preserves_snapshot_on_error_payloads() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for queue state checks")
    script = r"""
import {
    createQueueState,
    queueManagerSections,
    queueSummaryCounts,
    setQueueFilter,
    updateQueueStateFromPayload,
} from './web/static/js/features/queue/state.js';

const state = createQueueState();
updateQueueStateFromPayload(state, {
    ok: true,
    paused: true,
    failure_policy: 'continue',
    status: 'running',
    current_item_id: 'run-1',
    items: [
        { id: 'run-1', state: 'running' },
        { id: 'queue-2', state: 'queued' },
        { id: 'done-3', state: 'done' },
        { id: 'cancel-4', state: 'canceled' },
    ],
});

const fallbackCounts = queueSummaryCounts(state);
updateQueueStateFromPayload(state, { ok: false, error: 'backend offline' });
const errorSnapshot = {
    paused: state.queue.paused,
    failurePolicy: state.queue.failurePolicy,
    status: state.queue.status,
    currentItemId: state.queue.currentItemId,
    error: state.queue.error,
    itemIds: state.queue.items.map((item) => item.id),
    summary: state.queue.summary,
};
setQueueFilter(state, 'done');
const doneSections = queueManagerSections(state).map((section) => ({
    key: section.key,
    collapsed: Boolean(section.collapsed),
    itemIds: section.items.map((item) => item.id),
}));
setQueueFilter(state, 'canceled');
const canceledSections = queueManagerSections(state).map((section) => ({
    key: section.key,
    collapsed: Boolean(section.collapsed),
    itemIds: section.items.map((item) => item.id),
}));

console.log(JSON.stringify({ fallbackCounts, errorSnapshot, doneSections, canceledSections }));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "fallbackCounts": {
            "total": 4,
            "queued": 1,
            "running": 1,
            "done": 1,
            "error": 0,
            "canceled": 1,
        },
        "errorSnapshot": {
            "paused": True,
            "failurePolicy": "continue",
            "status": "running",
            "currentItemId": "run-1",
            "error": "backend offline",
            "itemIds": ["run-1", "queue-2", "done-3", "cancel-4"],
            "summary": {},
        },
        "doneSections": [
            {"key": "done", "collapsed": False, "itemIds": ["done-3"]},
        ],
        "canceledSections": [
            {"key": "canceled", "collapsed": False, "itemIds": ["cancel-4"]},
        ],
    }

