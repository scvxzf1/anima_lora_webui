/**
 * Live training dashboard metric / chart UI helpers.
 * Extracted from anima-app chunk 03.
 */
import { formatLossValue } from '../history-detail/curve/data.js?v=module-bootstrap-20260714-stage-dataset5';
import { numberOrNull } from '../history-detail/ui.js?v=module-bootstrap-20260714-stage-dataset5';
import { formatLr } from './index.js?v=module-bootstrap-20260714-stage-dataset5';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

const trainingState = getTrainingState();

export function setText(id, text) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (el.classList.contains('metric-value')) {
        const empty = metricValueIsEmpty(text);
        el.classList.toggle('metric-empty', empty);
        el.title = empty ? '' : String(text);
        el.closest('.metric-item')?.classList.toggle('is-empty', empty);
    }
}

export function metricValueIsEmpty(value) {
    const text = String(value ?? '').trim();
    return !text || text === '-' || text.toUpperCase() === 'N/A';
}

export function setMetricText(id, value) {
    const text = metricValueIsEmpty(value) ? 'N/A' : String(value);
    setText(id, text);
}

export function setEtaMetricText(info = {}) {
    const el = document.getElementById('metric-eta');
    if (!el) return;
    const text = String(info.text || '').trim() || '待计算';
    el.textContent = text;
    el.title = info.title || '';
    const empty = info.empty !== undefined ? Boolean(info.empty) : (text === '待计算' || metricValueIsEmpty(text));
    el.classList.toggle('metric-empty', empty);
    el.closest('.metric-item')?.classList.toggle('is-empty', empty);
}

export function resetLiveMetricPlaceholders(options = {}) {
    const includePrimary = options.primary !== false;
    const ids = [
        ...(includePrimary ? ['metric-loss', 'metric-lr', 'metric-step', 'metric-rate'] : ['metric-rate']),
        'metric-vram',
        'metric-vram-peak',
        'metric-gpu',
        'metric-gpu-peak',
        'metric-temp',
        'metric-temp-peak',
        'metric-log-age',
    ];
    ids.forEach((id) => setMetricText(id, 'N/A'));
    setEtaMetricText({ text: '待计算', empty: true, title: '需要进度总数和速度后计算预计完成时间。' });
}

export function updateDashboardProgressIdleState(active = null) {
    const trainingRuntime = trainingState.trainingRuntime;
    const wrap = document.querySelector('#tab-training .training-dashboard-progress');
    const head = document.querySelector('#tab-training .training-dashboard-head');
    const text = document.getElementById('progress-text');
    if (!wrap) return;
    const hasProgress = active !== null
        ? Boolean(active)
        : Number(trainingRuntime.progressTotal || 0) > 0;
    wrap.classList.toggle('is-idle', !hasProgress);
    head?.classList.toggle('is-idle', !hasProgress);
    if (!hasProgress && text) {
        text.textContent = '暂无正在运行的任务目录...';
    }
}

export function setTrainingDashboardHeadState(state = 'idle') {
    const head = document.querySelector('#tab-training .training-dashboard-head');
    if (!head) return;
    head.classList.remove('is-idle', 'is-running', 'is-compiling', 'is-error', 'is-history');
    head.classList.add(`is-${state || 'idle'}`);
}

export function syncLossChartEmptyState() {
    const lossChart = trainingState.lossChart;
    const shell = document.getElementById('loss-chart-shell');
    if (!shell) return;
    const pointCount = Array.isArray(lossChart?.data) ? lossChart.data.length : 0;
    shell.classList.toggle('is-empty', pointCount < 2);
    renderLiveChartPanel();
}

export function syncLiveChartControls() {
    const liveChartState = trainingState.liveChartState;
    const lrToggle = document.getElementById('live-chart-toggle-lr');
    if (lrToggle) lrToggle.checked = liveChartState.showLr;
    const rangeSelect = document.getElementById('live-chart-range');
    if (rangeSelect) rangeSelect.value = liveChartState.rangeMode;
}

function liveChartVisiblePoints(points = []) {
    const liveChartState = trainingState.liveChartState;
    const all = Array.isArray(points) ? points : [];
    const match = String(liveChartState.rangeMode || 'all').match(/^last(\d+)$/);
    if (!match) return all;
    const count = Number(match[1]);
    return Number.isFinite(count) && count > 0 ? all.slice(-count) : all;
}

export function renderLiveChartPanel() {
    const lossChart = trainingState.lossChart;
    const liveChartState = trainingState.liveChartState;
    const points = Array.isArray(lossChart?.data) ? lossChart.data : [];
    lossChart?.setDisplayOptions?.({
        showLr: liveChartState.showLr,
        rangeMode: liveChartState.rangeMode,
    });
    const visible = liveChartVisiblePoints(points);
    const latest = visible[visible.length - 1] || null;
    const latestLr = [...visible].reverse().find((point) => numberOrNull(point.lr) !== null) || null;
    setLiveChartStat('live-chart-stat-loss', latest ? formatLossValue(latest.value) : 'N/A');
    setLiveChartStat('live-chart-stat-lr', latestLr ? formatLr(latestLr.lr) : 'N/A');
    setLiveChartStat('live-chart-stat-points', visible.length ? `${visible.length}/${points.length}` : '0', !visible.length);
    setLiveChartStat('live-chart-stat-range', liveChartStepRangeText(visible), !visible.length);
    const lrLegend = document.getElementById('live-chart-lr-legend');
    if (lrLegend) {
        lrLegend.classList.toggle('muted', !liveChartState.showLr || !latestLr);
    }
}

function setLiveChartStat(id, value, empty = null) {
    const el = document.getElementById(id);
    if (!el) return;
    const text = metricValueIsEmpty(value) ? 'N/A' : String(value);
    el.textContent = text;
    const isEmpty = empty === null ? metricValueIsEmpty(text) : Boolean(empty);
    el.closest('.live-chart-stat')?.classList.toggle('is-empty', isEmpty);
}

function liveChartStepRangeText(points = []) {
    if (!points.length) return 'N/A';
    const first = points[0]?.step;
    const last = points[points.length - 1]?.step;
    return `${formatStepLabel(first)} - ${formatStepLabel(last)}`;
}

function formatStepLabel(value) {
    const number = Number(value);
    return Number.isFinite(number) ? String(Math.round(number)) : '-';
}

export function updateTrainingToolbarState(state, label) {
    const safeState = state || 'idle';
    const stateEl = document.getElementById('training-toolbar-state');
    const textEl = document.getElementById('training-toolbar-state-text');
    if (stateEl) stateEl.className = `training-toolbar-state ${safeState}`;
    if (textEl) textEl.textContent = label || '空闲';
}
