/* Live training page: loss chart, step progress, system metrics.
 * Polls /api/training/status and /api/training/metrics, with WebSocket updates.
 */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { connectWebSocket, disconnectWebSocket, onMessage } from '../ws.js?v=dragon-ui-20260812v35';

const api = createApiClient();

export async function loadLiveTraining() {
    const model = await loadLiveModel();
    return {
        html: renderLiveTraining(model),
        onMount: (root) => mountLiveTraining(root, model),
    };
}

async function loadLiveModel() {
    const [status, metricsPayload] = await Promise.all([
        readApi('/api/training/status', {}),
        readApi('/api/training/metrics', {}),
    ]);
    return createLiveModel(status, metricsPayload);
}

async function readApi(url, fallback) {
    try {
        const result = await api(url);
        if (!result || result.ok === false || Number(result.status) >= 400 || typeof result.error === 'string') {
            return fallback;
        }
        return result ?? fallback;
    } catch {
        return fallback;
    }
}

function createLiveModel(status = {}, metricsPayload = {}) {
    const progress = status?.latest_progress || {};
    const systemStats = status?.latest_system || {};
    const metrics = Array.isArray(metricsPayload?.metrics)
        ? metricsPayload.metrics
        : (Array.isArray(metricsPayload) ? metricsPayload : []);
    const step = progress.step || 0;
    const total = progress.total_steps || progress.total || 0;
    const progressPct = total > 0 ? Math.min(100, (step / total) * 100) : 0;

    return {
        state: status?.status || 'idle',
        step,
        total,
        progressPct,
        loss: progress.loss,
        lr: progress.lr,
        epoch: progress.epoch || 0,
        vram: systemStats.gpu_memory_used ?? systemStats.vram_used,
        gpuTemp: systemStats.gpu_temp,
        gpuUtil: systemStats.gpu_util,
        metrics,
    };
}

function renderLiveTraining(model) {
    return `
        <div class="dragon-page dragon-page-wide" data-live-training-root>
            <div class="dragon-page-hero dragon-reveal">
                <div class="dragon-status-badge" data-live-state data-state="${model.state}">
                    <span class="dragon-nav-status-dot" data-state="${model.state}"></span>
                    <span data-live-state-text>${stateText(model.state)}</span>
                </div>
                <h1>实时训练</h1>
                <p data-live-subtitle>${model.total > 0 ? `步数 ${model.step} / ${model.total}（${model.progressPct.toFixed(1)}%）` : '训练未运行'}</p>
            </div>

            <div class="dragon-progress-bar-wrapper dragon-reveal" data-stagger="1">
                <div class="dragon-progress-bar">
                    <div class="dragon-progress-bar-fill" data-live-progress-fill style="width: ${model.progressPct}%"></div>
                </div>
                <div class="dragon-progress-bar-meta">
                    <span data-live-step-text>${model.step} / ${model.total} 步</span>
                    <span data-live-epoch-text>轮数 ${model.epoch}</span>
                </div>
            </div>

            <div class="dragon-metrics-grid dragon-reveal" data-stagger="2">
                ${metricTile('损失值', formatLoss(model.loss), 'live-loss')}
                ${metricTile('学习率', formatLr(model.lr), 'live-lr')}
                ${metricTile('显存', formatVram(model.vram), 'live-vram')}
                ${metricTile('GPU 利用率', model.gpuUtil != null ? `${model.gpuUtil}%` : '-', 'live-gpu-util')}
                ${metricTile('GPU 温度', model.gpuTemp != null ? `${model.gpuTemp}\u00b0C` : '-', 'live-gpu-temp')}
                ${metricTile('轮数', model.epoch || '-', 'live-epoch')}
            </div>

            <div class="dragon-section dragon-reveal" data-stagger="3">
                <h2 class="dragon-section-title">损失曲线</h2>
                <p class="dragon-section-desc" data-live-chart-count>最近 ${model.metrics.length} 条记录。</p>
                <div class="dragon-chart-container" data-live-chart>${renderLossChart(model.metrics)}</div>
            </div>
        </div>
    `;
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
            else if (message.metric) model.metrics = [...model.metrics, message.metric].slice(-500);
            renderLiveState(root, model);
        }),
        onMessage('system', (message) => {
            applySystem(model, message);
            renderLiveState(root, model);
        }),
        onMessage('status', (message) => {
            if (message.status) model.state = message.status;
            renderLiveState(root, model);
        }),
    ].filter(Boolean);

    connectWebSocket();

    const pollTimer = window.setInterval(async () => {
        const next = await loadLiveModel();
        Object.assign(model, next);
        renderLiveState(root, model);
    }, 5000);

    const observer = new MutationObserver(() => {
        if (!document.contains(root)) {
            window.clearInterval(pollTimer);
            subscriptions.forEach((unsubscribe) => unsubscribe());
            disconnectWebSocket();
            observer.disconnect();
        }
    });
    observer.observe(root.parentElement || document.body, { childList: true, subtree: true });
}

function applyProgress(model, message) {
    const progress = message.progress || message;
    if (progress.step != null) model.step = progress.step;
    if (progress.total_steps != null) model.total = progress.total_steps;
    else if (progress.total != null) model.total = progress.total;
    if (progress.loss != null) model.loss = progress.loss;
    if (progress.lr != null) model.lr = progress.lr;
    if (progress.epoch != null) model.epoch = progress.epoch;
    model.progressPct = model.total > 0 ? Math.min(100, (model.step / model.total) * 100) : 0;
}

function applySystem(model, message) {
    const system = message.system || message;
    if (system.gpu_memory_used != null) model.vram = system.gpu_memory_used;
    else if (system.vram_used != null) model.vram = system.vram_used;
    if (system.gpu_temp != null) model.gpuTemp = system.gpu_temp;
    if (system.gpu_util != null) model.gpuUtil = system.gpu_util;
}

function renderLiveState(root, model) {
    if (!document.contains(root)) return;
    setText(root, '[data-live-state-text]', stateText(model.state));
    setText(root, '[data-live-subtitle]', model.total > 0 ? `步数 ${model.step} / ${model.total}（${model.progressPct.toFixed(1)}%）` : '训练未运行');
    setText(root, '[data-live-step-text]', `${model.step} / ${model.total} 步`);
    setText(root, '[data-live-epoch-text]', `轮数 ${model.epoch}`);
    setText(root, '[data-live-loss] .dragon-metric-value', formatLoss(model.loss));
    setText(root, '[data-live-lr] .dragon-metric-value', formatLr(model.lr));
    setText(root, '[data-live-vram] .dragon-metric-value', formatVram(model.vram));
    setText(root, '[data-live-gpu-util] .dragon-metric-value', model.gpuUtil != null ? `${model.gpuUtil}%` : '-');
    setText(root, '[data-live-gpu-temp] .dragon-metric-value', model.gpuTemp != null ? `${model.gpuTemp}\u00b0C` : '-');
    setText(root, '[data-live-epoch] .dragon-metric-value', model.epoch || '-');
    setText(root, '[data-live-chart-count]', `最近 ${model.metrics.length} 条记录。`);
    const chart = root.querySelector('[data-live-chart]');
    if (chart) chart.innerHTML = renderLossChart(model.metrics);
    const badge = root.querySelector('[data-live-state]');
    if (badge) badge.dataset.state = model.state;
    const dot = root.querySelector('[data-live-state] .dragon-nav-status-dot');
    if (dot) dot.dataset.state = model.state;
    const fill = root.querySelector('[data-live-progress-fill]');
    if (fill) fill.style.width = `${model.progressPct}%`;
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
        <svg class="dragon-loss-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
            ${yTicks}
            <polyline points="${points}" fill="none" stroke="var(--dragon-accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
        </svg>
    `;
}

function metricTile(label, value, key) {
    return `
        <div class="dragon-metric-tile" data-live-metric="${key}">
            <div class="dragon-metric-value">${value}</div>
            <div class="dragon-metric-label">${label}</div>
        </div>
    `;
}

function setText(root, selector, value) {
    const node = root.querySelector(selector);
    if (node) node.textContent = value;
}

function formatLoss(value) {
    return value != null ? Number(value).toFixed(4) : '-';
}

function formatLr(value) {
    return value != null ? Number(value).toExponential(3) : '-';
}

function formatVram(value) {
    return value != null ? `${Number(value).toFixed(1)} GB` : '-';
}

function stateText(state) {
    const map = {
        idle: '空闲', running: '训练中', training: '训练中',
        queued: '排队中', completed: '已完成', error: '错误',
        stopped: '已停止', unknown: '未知',
    };
    return map[state] || state;
}
