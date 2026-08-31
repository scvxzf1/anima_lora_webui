/* Live training page: loss chart, step progress, system metrics.
 * Polls /api/training/status and /api/training/metrics, with WebSocket updates.
 */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { connectWebSocket, disconnectWebSocket, onClose, onMessage, onOpen } from '../ws.js?v=dragon-ui-20260812v35';
import { bindLiveLogTools, createLiveLogBindings, renderLogView, updateLogSummary } from './live-training-log-tools.js?v=dragon-ui-20260826v47';
import {
    applyProgress,
    applySystem,
    connectionState,
    createLiveModel,
    formatConfigLabel,
    isRunningState,
    mergeStatusSnapshot,
    formatCurrentTaskLabel,
    stateText,
    visualState,
    visibleLogs,
} from './live-training-state.js?v=dragon-ui-20260825v46';
import {
    formatEta,
    formatLoss,
    formatLr,
    formatPercent,
    formatRate,
    formatTemperature,
    formatVram,
    renderLiveTrainingPage,
} from './live-training-view.js?v=dragon-ui-20260825v46';
import { areaSvgPath, emaValues, smoothSvgPath } from './trend-utils.js?v=dragon-ui-20260825v1';
import {
    applyWorkspaceSnapshot,
    hardwarePercent,
    liveWorkspaceMode,
    lossDelta,
    renderLiveSidebarBody,
} from './live-training-workspace.js?v=dragon-ui-20260825v46';
import { createVisibilityPoller } from '../visibility-poller.js?v=dragon-ui-20260826v2';
import {
    createLiveDomBindings,
    setLiveAttribute,
    setLiveDataset,
    setLiveProperty,
    setLiveText,
    setLiveWidth,
} from './live-training-dom.js?v=dragon-ui-20260826v2';
import { createFrameScheduler } from '../frame-scheduler.js?v=dragon-ui-20260826v1';

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
    if (snapshot.ok) return applyWorkspaceSnapshot(
        createLiveModel(snapshot.status, snapshot.metrics, snapshot.logs),
        snapshot.queue,
        snapshot.history,
    );
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
        readApi('/api/training/queue'),
        readApi('/api/training/history?limit=5'),
    ]);
    const rejected = results.slice(0, 3).find((result) => result.status === 'rejected');
    if (rejected) return { ok: false, error: rejected.reason?.message || '读取训练监控数据失败' };
    return {
        ok: true,
        status: results[0].value,
        metrics: results[1].value,
        logs: results[2].value,
        queue: results[3].status === 'fulfilled' ? results[3].value : null,
        history: results[4].status === 'fulfilled' ? results[4].value : null,
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
    model.chartSmoothing = Number.isFinite(Number(model.chartSmoothing)) ? Number(model.chartSmoothing) : .2;
    const dom = createLiveDomBindings(root);
    const logDom = createLiveLogBindings(root);
    const renderScheduler = createLiveRenderScheduler((options) => renderLiveState(dom, logDom, model, options));
    const subscriptions = subscribeToLiveEvents(model, renderScheduler);

    connectWebSocket();
    const unbindActions = bindLiveActions(root, dom, logDom, model, async () => {
        const snapshot = await readLiveSnapshot();
        if (!snapshot.ok) return false;
        mergeStatusSnapshot(model, snapshot.status, snapshot.metrics, snapshot.logs);
        applyWorkspaceSnapshot(model, snapshot.queue, snapshot.history);
        model.apiConnected = true;
        model.apiError = '';
        renderScheduler.flush({ chart: true, logs: true, sidebar: true });
        return true;
    });
    const unbindLogTools = bindLiveLogTools(logDom, model, (options = {}) => renderLiveState(dom, logDom, model, options));
    const smoothingInput = root.querySelector('[data-live-chart-smoothing]');
    const smoothingRender = createFrameScheduler(
        () => renderLiveState(dom, logDom, model, { chart: true, logs: false }),
        root.ownerDocument?.defaultView,
    );
    const onSmoothingInput = () => {
        model.chartSmoothing = Math.min(.99, Math.max(0, Number(smoothingInput?.value || 0) / 100));
        setLiveText(dom, 'chartSmoothing', `${Math.round(model.chartSmoothing * 100)}%`);
        smoothingRender.schedule();
    };
    smoothingInput?.addEventListener('input', onSmoothingInput);
    renderLiveState(dom, logDom, model, { chart: false, logs: false });

    const poller = createVisibilityPoller({ poll: async () => {
        const snapshot = await readLiveSnapshot();
        if (snapshot.ok) {
            mergeStatusSnapshot(model, snapshot.status, snapshot.metrics, snapshot.logs);
            applyWorkspaceSnapshot(model, snapshot.queue, snapshot.history);
            model.apiConnected = true;
            model.apiError = '';
            renderScheduler.flush({ chart: true, logs: true, sidebar: true });
            return;
        }
        model.apiConnected = false;
        model.apiError = `${snapshot.error}。请检查 WebUI 服务后重试。`;
        model.lastActivity = model.apiError;
        renderScheduler.flush();
    }, delay: 5000 });
    poller.start();

    return () => {
        poller.stop();
        subscriptions.forEach((unsubscribe) => unsubscribe());
        smoothingInput?.removeEventListener('input', onSmoothingInput);
        smoothingRender.cancel();
        renderScheduler.cancel();
        unbindActions?.();
        unbindLogTools?.();
        disconnectWebSocket();
    };
}

function subscribeToLiveEvents(model, renderScheduler) {
    return [
        onMessage('progress', (message) => {
            model.state = 'running';
            applyProgress(model, message);
            renderScheduler.schedule({ chart: false });
        }),
        onMessage('metrics', (message) => {
            if (Array.isArray(message.metrics)) model.metrics = message.metrics;
            else model.metrics = [...model.metrics, message].slice(-500);
            if (message.loss != null) model.loss = message.loss;
            if (message.lr != null) model.lr = message.lr;
            if (message.epoch != null) model.epoch = message.epoch;
            if (message.rate != null) model.rate = message.rate;
            renderScheduler.schedule({ chart: true });
        }),
        onMessage('system', (message) => {
            applySystem(model, message);
            renderScheduler.schedule({ chart: false });
        }),
        onMessage('log', (message) => {
            model.logs = [...model.logs, message].slice(-300);
            model.lastLogId = Math.max(model.lastLogId || 0, Number(message.id || 0));
            model.lastActivity = message.line || message.message || model.lastActivity;
            renderScheduler.schedule({ chart: false, logs: true, forceLogBottom: model.autoScroll });
        }),
        onMessage('status', (message) => {
            const previousRunDir = model.runDir;
            if (message.status || message.state) model.state = message.status || message.state;
            if (message.variant || message.preset || message.history_source_config_file || message.runtime_config_file) {
                model.configLabel = formatConfigLabel(message);
            }
            if (message.task_id || message.queue_item_id || message.job || message.status || message.state) {
                model.currentTask = formatCurrentTaskLabel({ ...message, variant: message.variant || undefined, preset: message.preset || undefined });
            }
            model.runDir = message.run_dir || message.output_dir || model.runDir;
            if (previousRunDir !== model.runDir && (message.run_dir || message.output_dir)) resetLivePeaks(model);
            model.lastActivity = message.message || message.last_log_line || model.lastActivity;
            renderScheduler.flush();
        }),
        onOpen(() => {
            model.wsState = 'open';
            model.wsError = '';
            renderScheduler.flush();
        }),
        onClose((event = {}) => {
            if (event.intentional) return;
            model.wsState = 'closed';
            model.wsError = '实时推送已断开，正在自动重连；页面仍会定时刷新。';
            renderScheduler.flush();
        }),
    ].filter(Boolean);
}

function renderLiveState(dom, logDom, model, options = {}) {
    const { root } = dom;
    if (!document.contains(root)) return;
    const mode = liveWorkspaceMode(model.state);
    const delta = lossDelta(model.metrics);
    const visual = visualState(model.state);
    const running = isRunningState(model.state);
    setLiveDataset(root, 'liveMode', mode);
    dom.sections.forEach((section) => setLiveProperty(section, 'hidden', section.dataset.liveSection !== mode));
    setLiveText(dom, 'state', stateText(model.state));
    setLiveText(dom, 'headerTitle', liveHeaderTitle(model, mode));
    setLiveText(dom, 'headerMeta', liveHeaderMeta(model, mode));
    setLiveText(dom, 'errorMessage', model.lastActivity || '训练已异常中断，请检查日志或历史任务。');
    setLiveText(dom, 'progressText', model.total > 0 ? `${model.progressPct.toFixed(1)}%` : '-');
    setLiveText(dom, 'stepText', `${model.step} / ${model.total} 步`);
    setLiveText(dom, 'epochText', `第 ${model.epoch ?? '-'} 轮`);
    setLiveText(dom, 'lossValue', formatLoss(model.loss));
    setLiveText(dom, 'lossDetail', delta.text);
    const lossDetail = dom.text.lossDetail;
    setLiveDataset(lossDetail, 'tone', delta.tone);
    setLiveText(dom, 'lrValue', formatLr(model.lr));
    setLiveText(dom, 'rateValue', formatRate(model.rate));
    setLiveText(dom, 'eta', formatEta(model));
    setLiveText(dom, 'vramValue', formatVram(model.vram, null));
    setLiveText(dom, 'vramDetail', `Max ${formatVram(model.vramTotal, null)} · ${peakMetricText(formatVram(model.peakVram, null))}`);
    setLiveText(dom, 'gpuValue', formatPercent(model.gpuUtil));
    setLiveText(dom, 'gpuDetail', peakMetricText(formatPercent(model.peakGpuUtil)));
    setLiveText(dom, 'temperatureValue', formatTemperature(model.gpuTemp));
    setLiveText(dom, 'temperatureDetail', Number(model.gpuTemp) >= 80 ? `高温预警 · 峰值 ${formatTemperature(model.peakGpuTemp)}` : `峰值 ${formatTemperature(model.peakGpuTemp)}`);
    setLiveWidth(dom.meters.vram, hardwarePercent(model.vram, model.vramTotal));
    setLiveWidth(dom.meters.gpu, Number(model.gpuUtil) || 0);
    setLiveWidth(dom.meters.temperature, Number(model.gpuTemp) || 0);
    setLiveDataset(dom.temperatureCard, 'tone', Number(model.gpuTemp) >= 80 ? 'warning' : 'normal');
    setLiveText(dom, 'chartCount', `最近 ${model.metrics.length} 条记录`);
    if (options.sidebar) {
        if (dom.sidebar) dom.sidebar.innerHTML = renderLiveSidebarBody(model);
    }
    renderConnectionState(dom, model);
    if (options.chart !== false) {
        if (dom.chart) dom.chart.innerHTML = renderLossChart(model.metrics, model.chartSmoothing);
    }
    if (options.logs) renderLogView(logDom, model, { forceBottom: options.forceLogBottom });
    else updateLogSummary(
        logDom,
        model,
        Number.isFinite(model.visibleLogCount) ? model.visibleLogCount : visibleLogs(model).length,
    );
    setLiveDataset(dom.stateBadge, 'state', visual);
    setLiveDataset(dom.stateDot, 'state', visual);
    setLiveWidth(dom.progressFill, model.progressPct);
    setLiveAttribute(dom.progress, 'aria-valuenow', Math.round(model.progressPct));
    setLiveProperty(dom.stop, 'hidden', !running);
    setLiveProperty(dom.stop, 'disabled', !running);
}

function createLiveRenderScheduler(render, delay = 150) {
    let timer = null;
    let pending = { chart: false, logs: false, sidebar: false, forceLogBottom: false };
    const flush = (options = {}) => {
        if (timer) window.clearTimeout(timer);
        timer = null;
        const next = { ...pending, ...options };
        pending = { chart: false, logs: false, sidebar: false, forceLogBottom: false };
        render(next);
    };
    return {
        schedule(options = {}) {
            pending.chart ||= options.chart === true;
            pending.logs ||= options.logs === true;
            pending.sidebar ||= options.sidebar === true;
            pending.forceLogBottom ||= options.forceLogBottom === true;
            if (!timer) timer = window.setTimeout(() => flush(), delay);
        },
        flush,
        cancel() { if (timer) window.clearTimeout(timer); timer = null; },
    };
}

export function renderLossChart(metrics, smoothing = .2) {
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

    const displayLosses = emaValues(losses, smoothing);
    const rawMin = Math.min(...displayLosses);
    const rawMax = Math.max(...displayLosses);
    const rawRange = rawMax - rawMin;
    const axisPadding = Math.max(rawRange * .08, Math.abs(rawMax || 1) * .015, .0001);
    const minLoss = rawMin - axisPadding;
    const maxLoss = rawMax + axisPadding;
    const range = maxLoss - minLoss;
    const pointPairs = displayLosses.map((loss, i) => {
        const x = padding.left + (i / (losses.length - 1 || 1)) * innerW;
        const y = padding.top + (1 - (loss - minLoss) / range) * innerH;
        return [x, y];
    });
    const linePath = smoothSvgPath(pointPairs, .2);
    const areaPath = areaSvgPath(linePath, pointPairs, padding.top + innerH);

    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
        const y = padding.top + t * innerH;
        const val = (maxLoss - t * range).toFixed(3);
        return `<line x1="${padding.left}" y1="${y}" x2="${padding.left + innerW}" y2="${y}" stroke="var(--dragon-divider)" stroke-width="0.5"/>
                 <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" fill="var(--dragon-text-quaternary)" font-size="11">${val}</text>`;
    }).join('');

    return `
        <svg class="dragon-loss-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby="dragon-live-loss-title">
            <title id="dragon-live-loss-title">最近 ${losses.length} 个训练损失值的变化曲线</title>
            <defs><linearGradient id="dragon-live-loss-area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--dragon-accent)" stop-opacity=".15"/><stop offset="1" stop-color="var(--dragon-accent)" stop-opacity="0"/></linearGradient></defs>
            ${yTicks}
            <path d="${areaPath}" fill="url(#dragon-live-loss-area)"/>
            <path d="${linePath}" fill="none" stroke="var(--dragon-accent)" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" class="dragon-live-loss-line"/>
        </svg>
    `;
}

function bindLiveActions(root, dom, logDom, model, refresh) {
    const queue = root.querySelector('[data-tool-action="queue"]');
    const stop = root.querySelector('[data-tool-action="stop"]');
    const retry = root.querySelector('[data-tool-action="retry"]');
    const confirmStop = root.querySelector('[data-live-stop-confirm]');
    const dialog = root.querySelector('[data-live-stop-dialog]');
    const openQueue = () => { window.location.hash = '#queue'; };
    const openStopDialog = () => { if (!dialog?.open) dialog?.showModal?.(); };
    const retryConnection = async (event) => {
        event.currentTarget.disabled = true;
        showFeedback(root, '正在重新连接监控…', 'info');
        const ok = await refresh();
        showFeedback(root, ok ? '监控数据已刷新。' : '重新连接失败，请检查 WebUI 服务。', ok ? 'success' : 'error');
        event.currentTarget.disabled = false;
    };
    const stopTraining = async () => {
        confirmStop.disabled = true;
        showFeedback(root, '正在发送停止请求…', 'info');
        try {
            const payload = await api('/api/training/stop', { method: 'POST' });
            if (payload.ok === false) throw new Error(payload.error || '停止训练失败');
            model.state = 'stopped';
            model.lastActivity = payload.message || '停止请求已发送';
            dialog?.close('confirmed');
            showFeedback(root, model.lastActivity, 'success');
            renderLiveState(dom, logDom, model, { chart: false, logs: false });
        } catch (error) {
            showFeedback(root, error.message || '停止训练失败', 'error');
        } finally {
            confirmStop.disabled = false;
        }
    };
    queue?.addEventListener('click', openQueue);
    stop?.addEventListener('click', openStopDialog);
    retry?.addEventListener('click', retryConnection);
    confirmStop?.addEventListener('click', stopTraining);
    return () => {
        queue?.removeEventListener('click', openQueue);
        stop?.removeEventListener('click', openStopDialog);
        retry?.removeEventListener('click', retryConnection);
        confirmStop?.removeEventListener('click', stopTraining);
    };
}

function showFeedback(root, message, tone) {
    const feedback = root.querySelector('[data-live-feedback]');
    if (!feedback) return;
    feedback.textContent = message;
    feedback.dataset.tone = tone;
    feedback.classList.add('dragon-config-feedback-visible');
}

function renderConnectionState(dom, model) {
    const connection = connectionState(model);
    const container = dom.connection;
    setLiveDataset(container, 'tone', connection.tone);
    setLiveText(dom, 'connectionLabel', connection.label);
    setLiveText(dom, 'connectionDetail', connection.detail);
}

function liveHeaderTitle(model, mode) { return mode === 'running' ? `正在训练：${model.currentTask}` : (mode === 'error' ? '训练异常' : '训练监控工作台'); }
function liveHeaderMeta(model, mode) { return mode === 'running' ? `Step ${model.step} / ${model.total}` : (mode === 'error' ? model.lastActivity : '等待训练任务'); }
function peakMetricText(value) { return value === '-' ? '峰值等待采样' : `峰值 ${value}`; }

function resetLivePeaks(model) {
    model.vram = undefined;
    model.gpuUtil = undefined;
    model.gpuTemp = undefined;
    model.peakVram = undefined;
    model.peakGpuUtil = undefined;
    model.peakGpuTemp = undefined;
}
