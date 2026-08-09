# Split from test_training_frontend_state.py (live)

from __future__ import annotations

from tests.frontend_test_support import *  # noqa: F403


import tests.frontend_test_support as _frontend_support
for _k, _v in vars(_frontend_support).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

def test_live_training_eta_metric_helper_computes_display_states() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for live-training ES module behavior checks")
    script = r"""
import { calculateTrainingEtaMetricInfo } from './web/static/js/features/live-training/index.js';

const nowMs = Date.UTC(2026, 0, 1, 12, 0, 0);
const formatDuration = (seconds) => `${seconds}s`;
const cases = [
    calculateTrainingEtaMetricInfo({ isRunning: false }),
    calculateTrainingEtaMetricInfo({ isRunning: true, current: 1, total: 0 }),
    calculateTrainingEtaMetricInfo({
        isRunning: true,
        current: 5,
        total: 5,
        progressSecondsPerStep: 2,
        nowMs,
        formatDuration,
    }),
    calculateTrainingEtaMetricInfo({
        isRunning: true,
        current: 1,
        total: 5,
        progressRate: '',
        nowMs,
        formatDuration,
    }),
    calculateTrainingEtaMetricInfo({
        isRunning: true,
        current: 5,
        total: 10,
        progressSecondsPerStep: 2,
        nowMs,
        formatDuration,
    }),
    calculateTrainingEtaMetricInfo({
        isRunning: true,
        current: 5,
        total: 10,
        progressRate: '2it/s',
        nowMs,
        formatDuration,
    }),
];
console.log(JSON.stringify(cases));
"""
    env = {**os.environ, "TZ": "UTC"}
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    cases = json.loads(result.stdout)
    assert cases == [
        {"text": "待计算", "empty": True, "title": "训练开始并收到进度后显示预计完成时间。"},
        {"text": "待计算", "empty": True, "title": "等待进度总数。"},
        {"text": "即将完成", "empty": False, "title": "当前进度已到达总步数。"},
        {"text": "待计算", "empty": True, "title": "等待速度数据后计算预计完成时间。"},
        {"text": "1/1 12:00", "empty": False, "title": "按当前速度估算，剩余约 10s。"},
        {"text": "1/1 12:00", "empty": False, "title": "按当前速度估算，剩余约 3s。"},
    ]


def test_live_training_progress_helpers_parse_runtime_text() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for live-training ES module behavior checks")
    script = r"""
import {
    formatLr,
    isLiveRunningState,
    lastValue,
    parseMetricsFromProgressLine,
    parseProgressRateSeconds,
    readConfigNumber,
} from './web/static/js/features/live-training/index.js';

const results = {
    rates: [
        parseProgressRateSeconds('1.5s/it'),
        parseProgressRateSeconds('500ms/it'),
        parseProgressRateSeconds('2it/s'),
        parseProgressRateSeconds('3s/step'),
        parseProgressRateSeconds('4 IT/S'),
        parseProgressRateSeconds('bad'),
    ],
    metrics: [
        parseMetricsFromProgressLine('| 12/100 [00:10<01:00, 1.25s/it, loss=0.1234, lr=1e-4]'),
        parseMetricsFromProgressLine('| 7/100 [00:10<01:00, 19.83s/it, recent_s_per_step=1.92, avr_loss=0.151]'),
        parseMetricsFromProgressLine('step: 8 recent_s_per_step=1.92 avr_loss=0.150'),
        parseMetricsFromProgressLine('step: 7 avr_loss: nan learning_rate: inf'),
        parseMetricsFromProgressLine('nothing useful here'),
    ],
    lastValue: lastValue([{ loss: '' }, { loss: null }, { loss: 0 }], 'loss'),
    configNumbers: [
        readConfigNumber('max_train_steps = 1200\nlr = "0.0001"\nx.y = 5\n', 'max_train_steps'),
        readConfigNumber('max_train_steps = 1200\nlr = "0.0001"\nx.y = 5\n', 'lr'),
        readConfigNumber('max_train_steps = 1200\nlr = "0.0001"\nx.y = 5\n', 'x.y'),
        readConfigNumber('max_train_steps = 1200\n', 'missing'),
    ],
    learningRates: [
        formatLr(0.0001),
        formatLr('bad'),
        formatLr(null),
    ],
    liveStates: [
        isLiveRunningState('running'),
        isLiveRunningState('compiling'),
        isLiveRunningState('idle'),
        isLiveRunningState(''),
    ],
};
console.log(JSON.stringify(results));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    results = json.loads(result.stdout)
    assert results == {
        "rates": [1.5, 0.5, 0.5, 3, 0.25, None],
        "metrics": [
            {"step": 12, "loss": "0.1234", "lr": 0.0001, "rate": "1.25s/it"},
            {"step": 7, "loss": "0.151", "rate": "19.83s/it"},
            {"step": 8, "loss": "0.150"},
            {"step": 7, "loss": "nan"},
            None,
        ],
        "lastValue": 0,
        "configNumbers": [1200, 0.0001, 5, None],
        "learningRates": ["1.00e-4", "-", "-"],
        "liveStates": [True, True, False, False],
    }


def test_live_training_status_and_progress_update_dom_fixture() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for live-training DOM fixture checks")
    script = r"""
const nodes = new Map();

function makeClassList(node) {
    const values = new Set();
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

function node(id) {
    if (!nodes.has(id)) {
        const item = {
            id,
            style: {},
            dataset: {},
            hidden: false,
            disabled: false,
            className: '',
            textContent: '',
            innerHTML: '',
            title: '',
            children: [],
            setAttribute(name, value) { this[name] = String(value); },
            removeAttribute(name) { delete this[name]; },
            append(...children) { this.children.push(...children); },
            appendChild(child) { this.children.push(child); return child; },
            addEventListener() {},
            querySelector() { return null; },
            closest() { return null; },
        };
        item.classList = makeClassList(item);
        nodes.set(id, item);
    }
    return nodes.get(id);
}

globalThis.window = { setTimeout: () => 0 };
globalThis.ctx = {
    format: {
        formatDuration: (seconds) => `${seconds}s`,
    },
};
globalThis.document = {
    getElementById: (id) => node(id),
    querySelector: (selector) => selector === '.dot' ? node('dot') : node(selector),
    createElement: (tag) => node(`created-${tag}-${nodes.size}`),
};

const logLines = [];
const previewUpdates = [];
globalThis.isHistoryReviewMode = () => false;
globalThis.logLineTone = (line) => String(line || '').includes('ERROR') ? 'error' : 'info';
globalThis.runtimePathItems = (task) => Object.entries(task).filter(([, value]) => value);
globalThis.renderHistoryPaths = (task) => { node('history-paths').dataset.rendered = String(Boolean(task.output_dir)); };
globalThis.refreshQueueRunningProgressViews = () => { node('queue-progress').dataset.refreshed = 'true'; };
globalThis.appendLog = (line) => logLines.push(line);
globalThis.previewFeature = {
    updateRuntimeSampleState: (payload) => previewUpdates.push(payload),
};
globalThis.formatDuration = (seconds) => `${seconds}s`;
    globalThis.copyText = async () => {};

    const { configureAppShellStateBridge } = await import('./web/static/js/features/anima-app/helpers/app-shell-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
    const { configureAppContextBridge } = await import('./web/static/js/features/anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260809-nf4-v2');
    const { configureConfigStateBridge } = await import('./web/static/js/features/anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
    const { configureDatasetStateBridge } = await import('./web/static/js/features/anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
    const { configureHistoryStateBridge } = await import('./web/static/js/features/anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
    const { configureRuntimeBridge } = await import('./web/static/js/features/anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2');
    const { configureTomlStateBridge } = await import('./web/static/js/features/anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
    const { configureTrainingStateBridge } = await import('./web/static/js/features/anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
    const { createAppShellState } = await import('./web/static/js/features/anima-app/state/app-shell-state.js?v=module-bootstrap-20260809-nf4-v2');
    const { createConfigState } = await import('./web/static/js/features/anima-app/state/config-state.js?v=module-bootstrap-20260809-nf4-v2');
    const { createDatasetState } = await import('./web/static/js/features/anima-app/state/dataset-state.js?v=module-bootstrap-20260809-nf4-v2');
    const { createHistoryState } = await import('./web/static/js/features/anima-app/state/history-state.js?v=module-bootstrap-20260809-nf4-v2');
    const { createTomlState } = await import('./web/static/js/features/anima-app/state/toml-state.js?v=module-bootstrap-20260809-nf4-v2');
    const { createTrainingState } = await import('./web/static/js/features/anima-app/state/training-state.js?v=module-bootstrap-20260809-nf4-v2');
    const appShellState = createAppShellState();
    const configState = createConfigState();
    const datasetState = createDatasetState();
    const historyState = createHistoryState();
    const tomlState = createTomlState();
    const trainingState = createTrainingState();
    Object.assign(trainingState.trainingRuntime, {
    state: 'idle',
    job: '',
    variant: '',
    preset: '',
    methodsSubdir: '',
    progressCurrent: 0,
    progressTotal: 0,
    progressLabel: '',
    progressRate: '',
    progressSecondsPerStep: null,
    progressUpdatedAt: 0,
    lastOutputAt: 0,
    lastUiActivityAt: 0,
    quietHintShown: false,
    lastTerminalMessage: '',
    lastTerminalHint: '',
    lastAnomalyMessage: '',
    runDir: '',
    runtimeConfigFile: '',
    originalConfigFile: '',
    datasetConfigFile: '',
    modelCacheDir: '',
    datasetCacheDir: '',
    trainingOutputDir: '',
    logsDir: '',
    outputDir: '',
    sampleDir: '',
    sampleConfig: null,
    });
    globalThis.trainingRuntime = trainingState.trainingRuntime;
    configureAppContextBridge(globalThis.ctx);
const { configureQueueViewBridge } = await import('./web/static/js/features/anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureQueueViewBridge({
    refreshQueueRunningProgressViews: () => {},
    loadTrainingQueue: async () => {},
    updateTrainingQueueFromPayload: () => {},
    renderTrainingQueue: () => {},
    showTrainingView: () => {},
    trainingViewTabs: () => [],
    focusTrainingViewTab: () => {},
    activateTrainingViewTabButton: () => {},
    moveTrainingViewTabFocus: () => {},
    bindTrainingViewTabKeyboard: () => {},
    renderTrainingViewMode: () => {},
    resetTrainingExpandedStateOnLeave: () => {},
});

    configureAppShellStateBridge(appShellState);
    configureConfigStateBridge(configState);
    configureDatasetStateBridge(datasetState);
    configureHistoryStateBridge(historyState);
    configureRuntimeBridge({
        api: Object.assign(() => ({ ok: true }), {
            datasetPresetApi: () => ({ ok: true, images: [] }),
        }),
        dom: {
            val: () => '',
            populateSelect: () => {},
        },
    });
    configureTomlStateBridge(tomlState);
    configureTrainingStateBridge(trainingState);

const featureEnsurers = await import('./web/static/js/features/anima-app/helpers/feature-ensurers.js?v=module-bootstrap-20260809-nf4-v2');
featureEnsurers.configurePreviewFeatureEnsurer(globalThis.ctx, globalThis, {});

const { configureHistoryDetailBridge } = await import('./web/static/js/features/anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureHistoryDetailBridge({
    isHistoryReviewMode: () => false,
    ensureHistoryDetailFeature: () => ({}),
    openHistoryDetailDialog: () => {},
});
const { configureLiveStatusBridge } = await import('./web/static/js/features/anima-app/helpers/live-status-bridge.js?v=module-bootstrap-20260809-nf4-v2');
// liveStatus module will overwrite these after import if needed; provide safe no-ops for early calls
configureLiveStatusBridge({
    updateProgress: () => {},
    updateStatus: () => {},
    updateDashboard: () => {},
    clearProgress: () => {},
    setStatusIndicator: () => {},
});


const { configureLiveLogBridge } = await import('./web/static/js/features/anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureLiveLogBridge({
    logLineTone: () => '',
    appendLog: () => {},
    clearLogs: () => {},
    renderLogs: () => {},
});


const { configureHistoryTimelineBridge } = await import('./web/static/js/features/anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureHistoryTimelineBridge({
    runtimePathItems: () => [],
    renderHistoryPaths: () => {},
    historyAbsolutePath: (v) => String(v || ''),
    historyStateLabel: (v) => String(v || ''),
    returnToLiveTraining: () => {},
});

const liveStatus = await import('./web/static/js/features/anima-app/chunks/25-update-progress.js?dom-fixture');
configureLiveStatusBridge({
    updateProgress: (...args) => liveStatus.updateProgress?.(...args),
    updateStatus: (...args) => liveStatus.updateStatus?.(...args),
    updateDashboard: (...args) => liveStatus.updateDashboard?.(...args),
});


liveStatus.updateStatus({
    state: 'running',
    job: 'train',
    variant: 'lora',
    preset: 'default',
    methods_subdir: 'gui-methods',
    output_dir: 'output/runs/job',
    sample_dir: 'output/runs/job/sample',
    sample_config: { prompt: 'demo' },
    run_dir: 'output/runs/job',
    runtime_config_file: 'output/runs/job/config.runtime.toml',
});
liveStatus.updateProgress({
    label: 'train',
    current: 4,
    total: 10,
    rate: '2it/s',
    ts: 1,
});

console.log(JSON.stringify({
    statusClass: node('dot').className,
    statusText: node('status-text').textContent,
    stopDisabled: node('btn-stop-training').disabled,
    stopClasses: node('btn-stop-training').classList.values(),
    progressWidth: node('progress-bar').style.width,
    progressText: node('progress-text').textContent,
    metricStep: node('metric-step').textContent,
    metricRate: node('metric-rate').textContent,
    trainVariant: node('train-variant').textContent,
    trainPreset: node('train-preset').textContent,
    runtimeState: trainingRuntime.state,
    runtimeProgressCurrent: trainingRuntime.progressCurrent,
    runtimeSecondsPerStep: trainingRuntime.progressSecondsPerStep,
    dashboardState: node('training-run-state').textContent,
    runMeta: node('training-run-meta').textContent,
    previewUpdates,
    logLines,
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
    assert json.loads(result.stdout) == {
        "statusClass": "dot running",
        "statusText": "训练中",
        "stopDisabled": False,
        "stopClasses": ["is-emergency"],
        "progressWidth": "40.0%",
        "progressText": "train: 4/10 (40.0%) — 2it/s",
        "metricStep": "4",
        "metricRate": "2it/s",
        "trainVariant": "lora",
        "trainPreset": "default",
        "runtimeState": "running",
        "runtimeProgressCurrent": 4,
        "runtimeSecondsPerStep": 0.5,
        "dashboardState": "训练中",
        "runMeta": "方法目录 gui-methods · 配置 lora · 预设 default",
        "previewUpdates": [
            {"sampleDir": "output/runs/job/sample"},
            {"sampleConfig": {"prompt": "demo"}},
        ],
        "logLines": [],
    }


def test_live_training_idle_status_clears_stale_runtime_snapshot() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for live-training DOM fixture checks")
    script = r"""
const nodes = new Map();

function makeClassList(node) {
    const values = new Set();
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

function node(id) {
    if (!nodes.has(id)) {
        const item = {
            id,
            style: {},
            dataset: {},
            hidden: false,
            disabled: false,
            className: '',
            textContent: '',
            innerHTML: '',
            title: '',
            children: [],
            setAttribute(name, value) { this[name] = String(value); },
            removeAttribute(name) { delete this[name]; },
            append(...children) { this.children.push(...children); },
            appendChild(child) { this.children.push(child); return child; },
            addEventListener() {},
            querySelector() { return null; },
            closest() { return null; },
        };
        item.classList = makeClassList(item);
        nodes.set(id, item);
    }
    return nodes.get(id);
}

globalThis.window = { setTimeout: () => 0 };
globalThis.ctx = {
    format: {
        formatDuration: (seconds) => `${seconds}s`,
    },
};
globalThis.document = {
    getElementById: (id) => node(id),
    querySelector: (selector) => selector === '.dot' ? node('dot') : node(selector),
    createElement: (tag) => node(`created-${tag}-${nodes.size}`),
};

const previewUpdates = [];
globalThis.isHistoryReviewMode = () => false;
globalThis.logLineTone = (line) => String(line || '').includes('ERROR') ? 'error' : 'info';
globalThis.runtimePathItems = (task) => Object.entries(task).filter(([, value]) => value);
globalThis.renderHistoryPaths = (task) => { node('history-paths').dataset.rendered = String(Boolean(task.output_dir)); };
globalThis.refreshQueueRunningProgressViews = () => { node('queue-progress').dataset.refreshed = 'true'; };
globalThis.appendLog = () => {};
globalThis.previewFeature = {
    updateRuntimeSampleState: (payload) => previewUpdates.push(payload),
};
globalThis.formatDuration = (seconds) => `${seconds}s`;
globalThis.copyText = async () => {};

const { configureAppShellStateBridge } = await import('./web/static/js/features/anima-app/helpers/app-shell-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureAppContextBridge } = await import('./web/static/js/features/anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureConfigStateBridge } = await import('./web/static/js/features/anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureDatasetStateBridge } = await import('./web/static/js/features/anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureHistoryStateBridge } = await import('./web/static/js/features/anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureRuntimeBridge } = await import('./web/static/js/features/anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureTomlStateBridge } = await import('./web/static/js/features/anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureTrainingStateBridge } = await import('./web/static/js/features/anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { createAppShellState } = await import('./web/static/js/features/anima-app/state/app-shell-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createConfigState } = await import('./web/static/js/features/anima-app/state/config-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createDatasetState } = await import('./web/static/js/features/anima-app/state/dataset-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createHistoryState } = await import('./web/static/js/features/anima-app/state/history-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createTomlState } = await import('./web/static/js/features/anima-app/state/toml-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createTrainingState } = await import('./web/static/js/features/anima-app/state/training-state.js?v=module-bootstrap-20260809-nf4-v2');
const appShellState = createAppShellState();
const configState = createConfigState();
const datasetState = createDatasetState();
const historyState = createHistoryState();
const tomlState = createTomlState();
const trainingState = createTrainingState();
let chartCleared = 0;
trainingState.lossChart = {
    data: [{ step: 1, loss: 0.12 }],
    clear() { chartCleared += 1; this.data = []; },
    setXLabel() {},
    setScaleMode() {},
    push() {},
};
Object.assign(trainingState.trainingRuntime, {
    state: 'idle',
    job: '',
    variant: '',
    preset: '',
    methodsSubdir: '',
    progressCurrent: 0,
    progressTotal: 0,
    progressLabel: '',
    progressRate: '',
    progressSecondsPerStep: null,
    progressUpdatedAt: 0,
    lastOutputAt: 0,
    lastUiActivityAt: 0,
    quietHintShown: false,
    lastTerminalMessage: '',
    lastTerminalHint: '',
    lastAnomalyMessage: '',
    runDir: '',
    runtimeConfigFile: '',
    originalConfigFile: '',
    datasetConfigFile: '',
    modelCacheDir: '',
    datasetCacheDir: '',
    trainingOutputDir: '',
    logsDir: '',
    outputDir: '',
    sampleDir: '',
    sampleConfig: null,
});
globalThis.trainingRuntime = trainingState.trainingRuntime;
configureAppContextBridge(globalThis.ctx);
const { configureQueueViewBridge } = await import('./web/static/js/features/anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureQueueViewBridge({
    refreshQueueRunningProgressViews: () => {},
    loadTrainingQueue: async () => {},
    updateTrainingQueueFromPayload: () => {},
    renderTrainingQueue: () => {},
    showTrainingView: () => {},
    trainingViewTabs: () => [],
    focusTrainingViewTab: () => {},
    activateTrainingViewTabButton: () => {},
    moveTrainingViewTabFocus: () => {},
    bindTrainingViewTabKeyboard: () => {},
    renderTrainingViewMode: () => {},
    resetTrainingExpandedStateOnLeave: () => {},
});

configureAppShellStateBridge(appShellState);
configureConfigStateBridge(configState);
configureDatasetStateBridge(datasetState);
configureHistoryStateBridge(historyState);
configureRuntimeBridge({
    api: Object.assign(() => ({ ok: true }), {
        datasetPresetApi: () => ({ ok: true, images: [] }),
    }),
    dom: {
        val: () => '',
        populateSelect: () => {},
    },
});
configureTomlStateBridge(tomlState);
configureTrainingStateBridge(trainingState);

const featureEnsurers = await import('./web/static/js/features/anima-app/helpers/feature-ensurers.js?v=module-bootstrap-20260809-nf4-v2');
featureEnsurers.configurePreviewFeatureEnsurer(globalThis.ctx, globalThis, {});

const { configureHistoryDetailBridge } = await import('./web/static/js/features/anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureHistoryDetailBridge({
    isHistoryReviewMode: () => false,
    ensureHistoryDetailFeature: () => ({}),
    openHistoryDetailDialog: () => {},
});
const { configureLiveStatusBridge } = await import('./web/static/js/features/anima-app/helpers/live-status-bridge.js?v=module-bootstrap-20260809-nf4-v2');
// liveStatus module will overwrite these after import if needed; provide safe no-ops for early calls
configureLiveStatusBridge({
    updateProgress: () => {},
    updateStatus: () => {},
    updateDashboard: () => {},
    clearProgress: () => {},
    setStatusIndicator: () => {},
});


const { configureLiveLogBridge } = await import('./web/static/js/features/anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureLiveLogBridge({
    logLineTone: () => '',
    appendLog: () => {},
    clearLogs: () => {},
    renderLogs: () => {},
});


const { configureHistoryTimelineBridge } = await import('./web/static/js/features/anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureHistoryTimelineBridge({
    runtimePathItems: () => [],
    renderHistoryPaths: () => {},
    historyAbsolutePath: (v) => String(v || ''),
    historyStateLabel: (v) => String(v || ''),
    returnToLiveTraining: () => {},
});

const liveStatus = await import('./web/static/js/features/anima-app/chunks/25-update-progress.js?idle-dom-fixture');
configureLiveStatusBridge({
    updateProgress: (...args) => liveStatus.updateProgress?.(...args),
    updateStatus: (...args) => liveStatus.updateStatus?.(...args),
    updateDashboard: (...args) => liveStatus.updateDashboard?.(...args),
});


liveStatus.updateStatus({
    state: 'running',
    job: 'train',
    variant: 'lora',
    preset: 'default',
    methods_subdir: 'gui-methods',
    output_dir: 'output/runs/job/training_output',
    sample_dir: 'output/runs/job/training_output/sample',
    sample_config: { prompt: 'demo' },
    run_dir: 'output/runs/job',
    runtime_config_file: 'output/runs/job/config.runtime.toml',
    original_config_file: 'output/runs/job/config.original.toml',
});
liveStatus.updateMetrics({
    step: 12,
    loss: 0.12345,
    lr: 0.0001,
});
liveStatus.updateStatus({
    state: 'idle',
    job: 'train',
    message: '训练完成',
    output_dir: 'output/runs/job/training_output',
    sample_dir: 'output/runs/job/training_output/sample',
    sample_config: { prompt: 'demo' },
    run_dir: 'output/runs/job',
    runtime_config_file: 'output/runs/job/config.runtime.toml',
    original_config_file: 'output/runs/job/config.original.toml',
});

console.log(JSON.stringify({
    metricLoss: node('metric-loss').textContent,
    metricLr: node('metric-lr').textContent,
    metricStep: node('metric-step').textContent,
    progressText: node('progress-text').textContent,
    trainVariant: node('train-variant').textContent,
    trainPreset: node('train-preset').textContent,
    dashboardState: node('training-run-state').textContent,
    runMeta: node('training-run-meta').textContent,
    outputDir: trainingRuntime.outputDir,
    runDir: trainingRuntime.runDir,
    sampleDir: trainingRuntime.sampleDir,
    historyConfigHidden: node('history-config-panel').hidden,
    chartCleared,
    previewUpdates,
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
    assert json.loads(result.stdout) == {
        "metricLoss": "N/A",
        "metricLr": "N/A",
        "metricStep": "N/A",
        "progressText": "暂无正在运行的任务目录...",
        "trainVariant": "-",
        "trainPreset": "-",
        "dashboardState": "空闲",
        "runMeta": "等待训练任务启动。",
        "outputDir": "",
        "runDir": "",
        "sampleDir": "",
        "historyConfigHidden": True,
        "chartCleared": 1,
        "previewUpdates": [
            {"sampleDir": "output/runs/job/training_output/sample"},
            {"sampleConfig": {"prompt": "demo"}},
            {"sampleDir": "", "sampleConfig": None},
        ],
    }


def test_return_to_live_training_clears_runtime_cursor() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    timeline_impl = _frontend_module_text("js/features/anima-app/chunks/35-render-config-group-timeline.js")
    body = _section(timeline_impl, "function returnToLiveTraining", "async function loadResumeOptionsForTask")

    for snippet in (
        "historyState.viewingHistoryTaskId = '';",
        "historyState.historyViewMode = 'live';",
        "trainingState.trainingRuntime.lastLogId = 0;",
        "trainingState.trainingRuntime.logLineCount = 0;",
        "trainingState.stepCounter = 0;",
        "trainingState.lossChart?.clear();",
        "recoverLiveTrainingState();",
    ):
        assert snippet in body


def test_live_training_rest_fallbacks_are_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    progress_source = _frontend_module_text("js/features/anima-app/chunks/25-update-progress.js")
    live_training_source = _frontend_module_text("js/features/live-training/index.js")
    poll_delay_section = _section(source, "function trainingStatusPollDelayMs", "async function pollStatus")
    poll_section = _section(source, "async function pollStatus", "function applyStatusSnapshotFallbacks")
    update_status = _section(progress_source, "function updateStatus", "function resetLiveSystemPeaks")
    health_section = _section(progress_source, "function refreshTrainingHealth", "function formatDuration")
    parse_metrics_section = _section(live_training_source, "function parseMetricsFromProgressLine", "function lastValue")
    recovery_section = _section(source, "async function recoverLiveTrainingState", "function updateProgress")
    ready_section = _section(source, "function startAnimaApp", "function chartTheme")

    assert "function isLiveRunningState" in live_training_source
    assert "globalThis.isLiveRunningState" not in source
    assert "import { isLiveRunningState }" in _frontend_module_text("js/features/live-training/status-polling.js")
    assert "function trainingStatusPollDelayMs" in source
    assert "function scheduleStatusPoll(options = {})" in source
    assert "target.trainingStatusPollTimer = window.setTimeout" in source
    assert "window.clearTimeout(target.trainingStatusPollTimer);" in source
    assert "if (!visible) return wsOpen ? (running ? 15000 : 120000) : 60000;" in poll_delay_section
    assert "if (!wsOpen) return running ? 2000 : 15000;" in poll_delay_section
    assert "return running ? 4000 : 60000;" in poll_delay_section
    assert "last_log_line: status.last_log_line" in poll_section
    assert "error_hint: status.error_hint" in poll_section
    assert "anomaly_message: status.anomaly_message || ''" in poll_section
    assert "if (options.forceReplayMetrics) {" in poll_section
    assert "target.trainingStatusPollForceReplayMetrics = true;" in poll_section
    assert "if (target.trainingStatusPollPromise) return target.trainingStatusPollPromise;" in poll_section
    assert "const forceReplayMetrics = target.trainingStatusPollForceReplayMetrics;" in poll_section
    assert "target.trainingStatusPollForceReplayMetrics = false;" in poll_section
    assert "target.trainingStatusPollPromise = null;" in poll_section
    assert "scheduleStatusPoll();" in poll_section
    assert "applyStatusSnapshotFallbacks(status);" in poll_section
    assert "const shouldReplayRecoveredArtifacts = shouldReplayRecoveredLiveArtifacts(status);" in poll_section
    assert "shouldReplayRecoveredArtifacts && (status.last_log_id || 0) > (target.trainingRuntime?.lastLogId || 0)" in poll_section
    assert "shouldReplayRecoveredArtifacts && (forceReplayMetrics || isLiveRunningState(target.trainingRuntime?.state))" in poll_section
    assert "function shouldReplayRecoveredLiveArtifacts(status = {})" in source
    assert "return isLiveRunningState(state) || state === 'error';" in source
    assert "forceReplayMetrics || isLiveRunningState() || hasStatusPayload(status.latest_metric)" not in poll_section
    assert "forceReplayMetrics || isLiveRunningState()" not in poll_section
    assert "function applyStatusSnapshotFallbacks(status = {})" in source
    assert "updateProgress(status.latest_progress, { replay: true });" in source
    assert "updateMetrics(status.latest_metric, { replay: true });" in source
    assert "updateSystem(status.latest_system, { replay: true });" in source
    assert "function hasStatusPayload(value)" in source

    assert "const state = liveStatusState(msg);" in update_status
    assert "const terminalMessage = terminalStatusMessage(msg);" in update_status
    assert "trainingRuntime.lastTerminalMessage = state === 'error' ? terminalMessage : '';" in update_status
    assert "const canStop = isLiveRunningState(state);" in update_status
    assert "stopBtn.disabled = !canStop;" in update_status
    assert "Object.prototype.hasOwnProperty.call(msg, 'anomaly_message')" in update_status
    assert "trainingRuntime.lastAnomalyMessage = String(msg.anomaly_message || '').trim();" in update_status
    assert "state === 'running' || (state === 'idle' && !terminalMessage)" in update_status
    assert "if (state === 'idle') {" in update_status
    assert "resetLiveRuntimeSnapshot();" in update_status
    assert "function resetLiveRuntimeSnapshot()" in progress_source
    assert "function liveStatusState(msg = {})" in update_status
    assert "if (state === 'idle' && terminalStatusMessage(msg)) return 'error';" in update_status
    assert "function terminalStatusMessage(msg = {})" in update_status
    assert "const state = String(msg.state || '');" in update_status
    assert "const lineIsError = logLineTone(line) === 'error';" in update_status
    assert "if (state !== 'error' && !lineIsError) return '';" in update_status
    assert "return line.includes(hint) ? line : `${line}；${hint}`;" in update_status

    assert "trainingRuntime.lastAnomalyMessage" in health_section
    assert "el.title = trainingRuntime.lastAnomalyMessage;" in health_section
    assert "el.removeAttribute('title');" in health_section
    assert "trainingRuntime.state === 'error' && trainingRuntime.lastTerminalMessage" in health_section
    assert "最近任务异常" in health_section
    assert "const metricNumberToken = '([+\\\\-]?" in parse_metrics_section
    assert "if (lossMatch) out.loss = lossMatch[1];" in parse_metrics_section
    assert "if (out.loss !== undefined && !Number.isFinite(out.loss)) delete out.loss;" not in parse_metrics_section
    assert "pollStatus({ forceReplayMetrics: true });" in recovery_section
    assert "replayTrainingLogs({ includeMetrics: false });" not in recovery_section
    assert "replayMetricsHistory();" not in recovery_section
    assert "scheduleStatusPoll();" in ready_section
    assert "document.addEventListener('visibilitychange'" in ready_section
    assert "scheduleStatusPoll({ immediate: !document.hidden });" in ready_section
    assert "scheduleStatusPoll({ immediate: true });" in ready_section
    assert "window.addEventListener('online', () => {" in ready_section
    assert "recoverLiveTrainingState();" in ready_section


def test_status_poll_incrementally_updates_live_history_and_full_refreshes_only_at_boundaries() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for status polling policy checks")
    script = r"""
globalThis.location = { protocol: 'http:' };
globalThis.document = { hidden: false };
globalThis.window = { setTimeout: () => 0, clearTimeout: () => {} };
globalThis.WebSocket = { OPEN: 1 };
let now = 100000;
Date.now = () => now;

const token = 'module-bootstrap-20260809-nf4-v2';
const { configureHistoryStateBridge } = await import(`./web/static/js/features/anima-app/helpers/history-state-bridge.js?v=${token}`);
const { configureHistoryListBridge } = await import(`./web/static/js/features/anima-app/helpers/history-list-bridge.js?v=${token}`);
const { configureQueueViewBridge } = await import(`./web/static/js/features/anima-app/helpers/queue-view-bridge.js?v=${token}`);
const { configureHistoryDetailBridge } = await import(`./web/static/js/features/anima-app/helpers/history-detail-bridge.js?v=${token}`);
const { configureLiveLogBridge } = await import(`./web/static/js/features/anima-app/helpers/live-log-bridge.js?v=${token}`);

const calls = { history: 0, queue: 0, merge: 0 };
configureHistoryStateBridge({ historyTasks: [{ id: 'task-1', state: 'running' }] });
configureHistoryListBridge({
    loadTrainingHistoryList: async () => { calls.history += 1; },
    mergeLiveTrainingHistoryTask: () => { calls.merge += 1; return true; },
});
configureQueueViewBridge({ loadTrainingQueue: async () => { calls.queue += 1; } });
configureHistoryDetailBridge({ isHistoryReviewMode: () => false });
configureLiveLogBridge({ appendLog: () => {} });

const { createStatusPollingBridge } = await import(
    `./web/static/js/features/live-training/status-polling.js?v=${token}`
);
const bridge = createStatusPollingBridge({ ws: { readyState: 1 } });
await bridge.refreshTrainingSidebarSummariesFromPoll({ task_id: 'task-1', status: 'running' });
await bridge.refreshTrainingSidebarSummariesFromPoll({ task_id: 'task-1', status: 'running' });
now += 16000;
await bridge.refreshTrainingSidebarSummariesFromPoll({ task_id: 'task-1', status: 'running' });
await bridge.refreshTrainingSidebarSummariesFromPoll({ task_id: 'task-1', status: 'idle' });
console.log(JSON.stringify(calls));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {"history": 2, "queue": 3, "merge": 3}


def test_live_training_status_snapshot_fallbacks_replay_latest_payloads() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for live-training status snapshot checks")
    script = r"""
const calls = [];

globalThis.ctx = {};
globalThis.document = { hidden: false };
globalThis.window = { setTimeout: () => 0, clearTimeout: () => {} };
globalThis.location = { protocol: 'http:' };
globalThis.ws = { readyState: 0 };
globalThis.WebSocket = { OPEN: 1 };
globalThis.trainingStatusPollTimer = null;
globalThis.trainingStatusPollPromise = null;
globalThis.trainingStatusPollForceReplayMetrics = false;
globalThis.trainingStatusPollFailures = 0;
globalThis.historyTasks = [];
globalThis.trainingRuntime = {};
// status-polling imports update* from live-status-bridge, not globalThis.
globalThis.updateProgress = () => {};
globalThis.updateMetrics = () => {};
globalThis.updateSystem = () => {};

const { configureAppContextBridge: configureAppContextBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureAppShellStateBridge: configureAppShellStateBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/app-shell-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureConfigStateBridge: configureConfigStateBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureDatasetStateBridge: configureDatasetStateBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureHistoryStateBridge: configureHistoryStateBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureTomlStateBridge: configureTomlStateBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureTrainingStateBridge: configureTrainingStateBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { configureRuntimeBridge: configureRuntimeBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const { createAppShellState: createAppShellStateForSnapshot } = await import('./web/static/js/features/anima-app/state/app-shell-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createConfigState: createConfigStateForSnapshot } = await import('./web/static/js/features/anima-app/state/config-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createDatasetState: createDatasetStateForSnapshot } = await import('./web/static/js/features/anima-app/state/dataset-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createHistoryState: createHistoryStateForSnapshot } = await import('./web/static/js/features/anima-app/state/history-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createTomlState: createTomlStateForSnapshot } = await import('./web/static/js/features/anima-app/state/toml-state.js?v=module-bootstrap-20260809-nf4-v2');
const { createTrainingState: createTrainingStateForSnapshot } = await import('./web/static/js/features/anima-app/state/training-state.js?v=module-bootstrap-20260809-nf4-v2');
configureAppContextBridgeForSnapshot(globalThis.ctx || {});
configureAppShellStateBridgeForSnapshot(createAppShellStateForSnapshot());
configureConfigStateBridgeForSnapshot(createConfigStateForSnapshot());
configureDatasetStateBridgeForSnapshot(createDatasetStateForSnapshot());
configureHistoryStateBridgeForSnapshot(createHistoryStateForSnapshot());
configureTomlStateBridgeForSnapshot(createTomlStateForSnapshot());
configureTrainingStateBridgeForSnapshot(createTrainingStateForSnapshot());
configureRuntimeBridgeForSnapshot({ api: globalThis.ctx?.api, timers: {} });
const { configureLiveLogBridge: configureLiveLogBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureLiveLogBridgeForSnapshot({ logLineTone: () => '', appendLog: () => {}, clearLogs: () => {}, renderLogs: () => {} });
const { configureHistoryDetailBridge: configureHistoryDetailBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureHistoryDetailBridgeForSnapshot({ isHistoryReviewMode: () => false, ensureHistoryDetailFeature: () => ({}), openHistoryDetailDialog: () => {} });
const { configureQueueViewBridge: configureQueueViewBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260809-nf4-v2');
configureQueueViewBridgeForSnapshot({ refreshQueueRunningProgressViews: () => {} });
const { configureLiveStatusBridge: configureLiveStatusBridgeForSnapshot } = await import('./web/static/js/features/anima-app/helpers/live-status-bridge.js?v=module-bootstrap-20260809-nf4-v2');
const recordLiveStatus = (kind) => (payload, options) => {
    calls.push({ kind, payload, options });
};
configureLiveStatusBridgeForSnapshot({
    updateProgress: recordLiveStatus('progress'),
    updateStatus: () => {},
    updateDashboard: () => {},
    updateMetrics: recordLiveStatus('metric'),
    updateSystem: recordLiveStatus('system'),
    liveStatusState: () => ({}),
    terminalStatusMessage: () => '',
    resetLiveSystemPeaks: () => {},
    clearRuntimeInfo: () => {},
    applyRuntimeInfoToState: () => {},
    renderCurrentRuntimePaths: () => {},
    currentRuntimeTaskInfo: () => ({}),
    formatRuntimeVram: () => '',
    renderTrainingRunSummary: () => {},
    renderLiveTrainingDashboard: () => {},
    trainingEtaMetricInfo: () => ({}),
    markTrainingActivity: () => {},
    refreshTrainingHealth: () => {},
    formatDuration: () => '',
    clearProgress: () => {},
    setStatusIndicator: () => {},
});
const statusPollingModule = await import('./web/static/js/features/anima-app/chunks/26a-status-polling.js?snapshot-fixture');
Object.assign(globalThis, statusPollingModule.createStatusPollingBridge(globalThis));

globalThis.applyStatusSnapshotFallbacks({
    status: 'running',
    latest_progress: { current: 4, total: 10 },
    latest_metric: { loss: 0.12 },
    latest_system: { gpu_util: 80 },
});
globalThis.applyStatusSnapshotFallbacks({
    status: 'idle',
    latest_progress: { current: 9, total: 10 },
    latest_metric: { loss: 0.01 },
    latest_system: { gpu_util: 10 },
});
globalThis.applyStatusSnapshotFallbacks({
    status: 'running',
    latest_progress: {},
    latest_metric: null,
    latest_system: undefined,
});

console.log(JSON.stringify({
    calls,
    hasEmptyPayload: globalThis.hasStatusPayload({}),
    hasProgressPayload: globalThis.hasStatusPayload({ current: 1 }),
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
    assert json.loads(result.stdout) == {
        "calls": [
            {
                "kind": "progress",
                "payload": {"current": 4, "total": 10},
                "options": {"replay": True},
            },
            {
                "kind": "metric",
                "payload": {"loss": 0.12},
                "options": {"replay": True},
            },
            {
                "kind": "system",
                "payload": {"gpu_util": 80},
                "options": {"replay": True},
            },
        ],
        "hasEmptyPayload": False,
        "hasProgressPayload": True,
    }
