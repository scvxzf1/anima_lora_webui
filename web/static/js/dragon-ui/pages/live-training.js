/* Live training page: loss chart, step progress, system metrics.
 * Polls /api/training/status and /api/training/metrics, with WebSocket updates.
 */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { connectWebSocket, disconnectWebSocket, onClose, onMessage, onOpen } from '../ws.js?v=dragon-ui-20260812v35';
import { bindLiveLogTools, renderLogView, updateLogSummary } from './live-training-log-tools.js?v=dragon-ui-20260814v43';
import {
    applyProgress,
    applySystem,
    connectionState,
    createLiveModel,
    formatConfigLabel,
    isRunningState,
    mergeStatusSnapshot,
    stateText,
    visualState,
    visibleLogs,
} from './live-training-state.js?v=dragon-ui-20260814v43';
import {
    formatEta,
    formatLoss,
    formatLr,
    formatPercent,
    formatRate,
    formatTemperature,
    formatVram,
    renderLiveTrainingPage,
} from './live-training-view.js?v=dragon-ui-20260814v43';

const api = createApiClient();

export async function loadLiveTraining() {
    const model = await loadLiveModel();
    let cleanup = null;
    return {
        html: renderLiveTraining(model),
        onMount: (root) => { cleanup = mountLiveTraining(root, model); },
        onUnmount: () => cleanup?.(),
    };
}

async function loadLiveModel() {
    const snapshot = await readLiveSnapshot();
    if (snapshot.ok) return createLiveModel(snapshot.status, snapshot.metrics, snapshot.logs);
    const model = createLiveModel({ status: 'unavailable' }, [], { records: [] });
    model.apiConnected = false;
    model.apiError = snapshot.error;
    model.lastActivity = snapshot.error;
    return model;
}

async function readLiveSnapshot() {
    const results = await Promise.allSettled([
        readApi('/api/training/status'),
        readApi('/api/training/metrics'),
        readApi('/api/training/logs?limit=300'),
    ]);
    const rejected = results.find((result) => result.status === 'rejected');
    if (rejected) return { ok: false, error: rejected.reason?.message || '读取训练监控数据失败' };
    return {
        ok: true,
        status: results[0].value,
        metrics: results[1].value,
        logs: results[2].value,
    };
}

async function readApi(url) {
    try {
        const result = await api(url);
        if (!result || result.ok === false || Number(result.status) >= 400 || typeof result.error === 'string') {
            throw new Error(result?.error || `读取 ${url} 失败`);
        }
        return result;
    } catch (error) {
        throw new Error(error?.message || `读取 ${url} 失败`);
    }
}

function renderLiveTraining(model) {
    return renderLiveTrainingPage(model, renderLossChart);
}

function mountLiveTraining(root, model) {
    const subscriptions = [
        onMessage('progress', (message) => {
            model.state = 'running';
            applyProgress(model, message);
            renderLiveState(root, model);
        }),
        onMessage('metrics', (message) => {
            if (Array.isArray(message.metrics)) model.metrics = message.metrics;
            else model.metrics = [...model.metrics, message].slice(-500);
            if (message.loss != null) model.loss = message.loss;
            if (message.lr != null) model.lr = message.lr;
            if (message.epoch != null) model.epoch = message.epoch;
            if (message.rate != null) model.rate = message.rate;
            renderLiveState(root, model, { chart: true });
        }),
        onMessage('system', (message) => {
            applySystem(model, message);
            renderLiveState(root, model, { chart: false });
        }),
        onMessage('log', (message) => {
            model.logs = [...model.logs, message].slice(-300);
            model.lastLogId = Math.max(model.lastLogId || 0, Number(message.id || 0));
            model.lastActivity = message.line || message.message || model.lastActivity;
            renderLiveState(root, model, { chart: false, logs: true, forceLogBottom: model.autoScroll });
        }),
        onMessage('status', (message) => {
            const previousRunDir = model.runDir;
            if (message.status || message.state) model.state = message.status || message.state;
            if (message.variant || message.preset || message.history_source_config_file || message.runtime_config_file) {
                model.configLabel = formatConfigLabel(message);
            }
            model.runDir = message.run_dir || message.output_dir || model.runDir;
            if (previousRunDir !== model.runDir && (message.run_dir || message.output_dir)) resetLivePeaks(model);
            model.lastActivity = message.message || message.last_log_line || model.lastActivity;
            renderLiveState(root, model, { chart: false });
        }),
        onOpen(() => {
            model.wsState = 'open';
            model.wsError = '';
            renderLiveState(root, model, { chart: false, logs: false });
        }),
        onClose((event = {}) => {
            if (event.intentional) return;
            model.wsState = 'closed';
            model.wsError = '实时推送已断开，正在自动重连；页面仍会定时刷新。';
            renderLiveState(root, model, { chart: false, logs: false });
        }),
    ].filter(Boolean);

    connectWebSocket();
    bindLiveActions(root, model);
    bindLiveLogTools(root, model, (options = {}) => renderLiveState(root, model, options));
    renderLiveState(root, model, { chart: false, logs: false });

    const pollTimer = window.setInterval(async () => {
        const snapshot = await readLiveSnapshot();
        if (snapshot.ok) {
            mergeStatusSnapshot(model, snapshot.status, snapshot.metrics, snapshot.logs);
            model.apiConnected = true;
            model.apiError = '';
            renderLiveState(root, model, { chart: true, logs: true });
            return;
        }
        model.apiConnected = false;
        model.apiError = `${snapshot.error}。请检查 WebUI 服务后重试。`;
        model.lastActivity = model.apiError;
        renderLiveState(root, model, { chart: false, logs: false });
    }, 5000);

    return () => {
        window.clearInterval(pollTimer);
        subscriptions.forEach((unsubscribe) => unsubscribe());
        disconnectWebSocket();
    };
}

function renderLiveState(root, model, options = {}) {
    if (!document.contains(root)) return;
    setText(root, '[data-live-state-text]', stateText(model.state));
    setText(root, '[data-live-progress-text]', model.total > 0 ? `${model.progressPct.toFixed(1)}%` : '等待训练');
    setText(root, '[data-live-step-text]', `${model.step} / ${model.total} 步`);
    setText(root, '[data-live-epoch-text]', `第 ${model.epoch ?? '-'} 轮`);
    setText(root, '[data-live-metric="live-loss"] strong', formatLoss(model.loss));
    setText(root, '[data-live-metric="live-lr"] strong', formatLr(model.lr));
    setText(root, '[data-live-metric="live-rate"] strong', formatRate(model.rate));
    setText(root, '[data-live-metric="live-eta"] strong', formatEta(model));
    setText(root, '[data-live-metric="live-vram"] strong', formatVram(model.vram, model.vramTotal));
    setText(root, '[data-live-metric="live-vram-peak"] strong', formatVram(model.peakVram, model.vramTotal));
    setText(root, '[data-live-metric="live-gpu-util"] strong', formatPercent(model.gpuUtil));
    setText(root, '[data-live-metric="live-gpu-util"] small', peakText(formatPercent(model.peakGpuUtil)));
    setText(root, '[data-live-metric="live-gpu-temp"] strong', formatTemperature(model.gpuTemp));
    setText(root, '[data-live-metric="live-gpu-temp"] small', peakText(formatTemperature(model.peakGpuTemp)));
    setText(root, '[data-live-context="config"]', model.configLabel);
    setText(root, '[data-live-context="run-dir"]', model.runDir);
    setText(root, '[data-live-context="activity"]', model.lastActivity);
    setText(root, '[data-live-chart-count]', `最近 ${model.metrics.length} 条记录`);
    renderConnectionState(root, model);
    if (options.chart !== false) {
        const chart = root.querySelector('[data-live-chart]');
        if (chart) chart.innerHTML = renderLossChart(model.metrics);
    }
    if (options.logs) renderLogView(root, model, { forceBottom: options.forceLogBottom });
    else updateLogSummary(root, model, visibleLogs(model).length);
    const badge = root.querySelector('[data-live-state]');
    if (badge) badge.dataset.state = visualState(model.state);
    const dot = root.querySelector('[data-live-state] .dragon-nav-status-dot');
    if (dot) dot.dataset.state = visualState(model.state);
    const fill = root.querySelector('[data-live-progress-fill]');
    if (fill) fill.style.width = `${model.progressPct}%`;
    const progress = root.querySelector('[data-live-progress]');
    if (progress) progress.setAttribute('aria-valuenow', String(Math.round(model.progressPct)));
    const stop = root.querySelector('[data-tool-action="stop"]');
    if (stop) {
        stop.hidden = !isRunningState(model.state);
        stop.disabled = !isRunningState(model.state);
    }
}

export function renderLossChart(metrics) {
    if (!metrics.length) {
        return '<div class="dragon-empty-state"><p>暂无训练数据</p></div>';
    }

    const width = 800;
    const height = 240;
    const padding = { top: 20, right: 20, bottom: 30, left: 50 };
    const innerW = width - padding.left - padding.right;
    const innerH = height - padding.top - padding.bottom;

    const losses = metrics.map((m) => Number(m.loss)).filter((n) => !Number.isNaN(n));
    if (!losses.length) return '<div class="dragon-empty-state"><p>暂无损失数据</p></div>';

    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const range = maxLoss - minLoss || 1;

    const points = losses.map((loss, i) => {
        const x = padding.left + (i / (losses.length - 1 || 1)) * innerW;
        const y = padding.top + (1 - (loss - minLoss) / range) * innerH;
        return `${x},${y}`;
    }).join(' ');

    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
        const y = padding.top + t * innerH;
        const val = (maxLoss - t * range).toFixed(3);
        return `<line x1="${padding.left}" y1="${y}" x2="${padding.left + innerW}" y2="${y}" stroke="var(--dragon-divider)" stroke-width="0.5"/>
                 <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" fill="var(--dragon-text-quaternary)" font-size="11">${val}</text>`;
    }).join('');

    return `
        <svg class="dragon-loss-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby="dragon-live-loss-title">
            <title id="dragon-live-loss-title">最近 ${losses.length} 个训练损失值的变化曲线</title>
            ${yTicks}
            <polyline points="${points}" fill="none" stroke="var(--dragon-accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
        </svg>
    `;
}

function setText(root, selector, value) {
    const node = root.querySelector(selector);
    if (node) node.textContent = value;
}

function bindLiveActions(root, model) {
    root.querySelector('[data-tool-action="queue"]')?.addEventListener('click', () => { window.location.hash = '#queue'; });
    root.querySelector('[data-tool-action="stop"]')?.addEventListener('click', async (event) => {
        if (!window.confirm('确认停止当前训练吗？已生成的日志、样张和权重会保留。')) return;
        const button = event.currentTarget;
        button.disabled = true;
        showFeedback(root, '正在发送停止请求…', 'info');
        try {
            const payload = await api('/api/training/stop', { method: 'POST' });
            if (payload.ok === false) throw new Error(payload.error || '停止训练失败');
            model.state = 'stopped';
            model.lastActivity = payload.message || '停止请求已发送';
            showFeedback(root, model.lastActivity, 'success');
            renderLiveState(root, model);
        } catch (error) {
            showFeedback(root, error.message || '停止训练失败', 'error');
            button.disabled = false;
        }
    });
}

function showFeedback(root, message, tone) {
    const feedback = root.querySelector('[data-live-feedback]');
    if (!feedback) return;
    feedback.textContent = message;
    feedback.dataset.tone = tone;
    feedback.classList.add('dragon-config-feedback-visible');
}

function renderConnectionState(root, model) {
    const connection = connectionState(model);
    const container = root.querySelector('[data-live-connection]');
    if (container) container.dataset.tone = connection.tone;
    setText(root, '[data-live-connection-label]', connection.label);
    setText(root, '[data-live-connection-detail]', connection.detail);
}

function peakText(value) { return value === '-' ? '峰值等待采样' : `峰值 ${value}`; }

function resetLivePeaks(model) {
    model.vram = undefined;
    model.gpuUtil = undefined;
    model.gpuTemp = undefined;
    model.peakVram = undefined;
    model.peakGpuUtil = undefined;
    model.peakGpuTemp = undefined;
}
