/* Training dashboard homepage.
 * Keeps the page hierarchy while bringing the classic live monitor into the
 * first workspace: progress, metrics, loss curve, run context, and logs.
 */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { calculateTrainingEtaMetricInfo, formatLr } from '../../features/live-training/index.js?v=dragon-ui-20260812v35';
import { renderLossChart } from './live-training.js?v=dragon-ui-20260812v35';
import { appendLogLine, connectWebSocket, disconnectWebSocket, onMessage } from '../ws.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

const api = createApiClient();

export async function loadDashboard() {
    const data = await loadDashboardData();
    return {
        html: renderDashboard(data),
        onMount: (wrapper) => mountDashboard(wrapper, data),
    };
}

async function loadDashboardData() {
    const [status, history, metrics, logs] = await Promise.all([
        readApi('/api/training/status', {}),
        readApi('/api/training/history?limit=8', {}),
        readApi('/api/training/metrics', []),
        readApi('/api/training/logs?limit=120', { records: [] }),
    ]);
    return {
        status: status && typeof status === 'object' ? status : {},
        history: Array.isArray(history?.tasks) ? history.tasks.slice(0, 8) : [],
        metrics: Array.isArray(metrics) ? metrics : [],
        logs: Array.isArray(logs?.records) ? logs.records : [],
    };
}

async function readApi(url, fallback, options = {}) {
    try {
        const result = await api(url, options);
        if (!result || result.ok === false || Number(result.status) >= 400 || typeof result.error === 'string') {
            return fallback;
        }
        return result ?? fallback;
    } catch {
        return fallback;
    }
}

function renderDashboard(data) {
    const model = createDashboardModel(data);
    const historyHtml = model.history.length
        ? `<ul class="dragon-history-list">${model.history.map(renderHistoryItem).join('')}</ul>`
        : '<div class="dragon-empty-state"><p>暂无训练记录</p></div>';
    const logsHtml = model.logs.length
        ? model.logs.map(renderLogRecord).join('')
        : '<div class="dragon-empty-state"><p>训练开始后显示实时日志</p></div>';

    return `
        <div class="dragon-page dragon-page-wide dragon-dashboard-page" data-dashboard-root>
            <div class="dragon-dashboard-hero dragon-reveal">
                <div class="dragon-dashboard-hero-copy">
                    <div class="dragon-dashboard-hero-meta">
                        <span class="dragon-eyebrow">训练工作台</span>
                        <div class="dragon-status-badge" data-dashboard-state data-state="${model.state}">
                            <span class="dragon-nav-status-dot" data-state="${model.state}"></span>
                            <span data-dashboard-state-text>${stateText(model.state)}</span>
                        </div>
                    </div>
                    <h1 data-dashboard-title>${model.running ? '训练进行中' : '训练中心'}</h1>
                    <p data-dashboard-subtitle>${escapeHtml(model.subtitle)}</p>
                    <div class="dragon-dashboard-command-row">
                        ${commandButton('activity', '查看实时训练', 'live', 'dragon-btn-primary')}
                        ${commandButton('settings', '调整训练配置', 'config', 'dragon-btn-secondary')}
                        ${commandButton('stop', '停止训练', 'stop', 'dragon-btn-secondary', !model.running)}
                    </div>
                </div>
                <div class="dragon-dashboard-progress-visual" data-dashboard-ring style="--dashboard-progress: ${model.progressPercent}%">
                    <div class="dragon-dashboard-progress-ring">
                        <div class="dragon-dashboard-progress-ring-center">
                            <strong data-dashboard-progress-percent>${Math.round(model.progressPercent)}%</strong>
                            <span>训练进度</span>
                        </div>
                    </div>
                    <div class="dragon-dashboard-ring-details">
                        <div><span>当前损失</span><strong data-dashboard-ring-loss>${escapeHtml(model.loss)}</strong></div>
                        <div><span>训练速度</span><strong data-dashboard-ring-rate>${escapeHtml(model.rate)}</strong></div>
                    </div>
                </div>
            </div>

            <section class="dragon-dashboard-runband dragon-reveal" data-stagger="1">
                ${contextItem('layers', '当前配置', model.configLabel, 'config')}
                ${contextItem('folder', '运行目录', model.runLabel, 'run')}
                ${contextItem('activity', '最新状态', model.health, 'health')}
            </section>

            <section class="dragon-dashboard-progress-section dragon-reveal" data-stagger="2">
                ${sectionHeading('gauge', '训练进度', '当前训练量', `<strong data-dashboard-progress-text>${escapeHtml(model.progressText)}</strong>`)}
                <div class="dragon-dashboard-progress-track" aria-label="训练进度">
                    <div class="dragon-dashboard-progress-fill" data-dashboard-progress-fill style="width: ${model.progressPercent}%"></div>
                </div>
            </section>

            <section class="dragon-dashboard-overview dragon-reveal" data-stagger="3" aria-label="训练关键概览">
                ${overviewItem(model.lossTrend.icon, '损失趋势', model.lossTrend.value, model.lossTrend.detail, 'loss-trend', model.lossTrend.tone)}
                ${overviewItem('clock', '预计完成', model.eta, `当前速度 ${model.rate}`, 'eta')}
                ${overviewItem('memory', '显存使用', model.vram, `峰值 ${model.vramPeak}`, 'vram')}
                ${overviewItem('cpu', '计算负载', model.gpu, `当前温度 ${model.temp}`, 'gpu')}
            </section>

            <section class="dragon-dashboard-monitor-section dragon-reveal" data-stagger="3">
                ${sectionHeading('activity', '实时指标', '训练参数变化', '<span class="dragon-dashboard-section-note">当前训练输出</span>')}
                <div class="dragon-dashboard-metric-grid">
                    ${metricCell('loss', '损失值', model.loss)}
                    ${metricCell('lr', '学习率', model.lr)}
                    ${metricCell('step', '步数', model.stepText)}
                    ${metricCell('rate', '速度', model.rate)}
                    ${metricCell('eta', '预计完成', model.eta)}
                </div>
                <div class="dragon-dashboard-resource-grid">
                    ${metricCell('epoch', '训练轮数', model.epoch)}
                    ${metricCell('vram', '最后显存', model.vram)}
                    ${metricCell('vram-peak', '峰值显存', model.vramPeak)}
                    ${metricCell('gpu', 'GPU 占用', model.gpu)}
                    ${metricCell('temp', '温度', model.temp)}
                    ${metricCell('log-age', '日志活动', model.logAge)}
                </div>
            </section>

            <section class="dragon-dashboard-chart-section dragon-reveal" data-stagger="4">
                ${sectionHeading('chart', '趋势', '损失值 / 学习率曲线', `<span class="dragon-dashboard-section-note" data-dashboard-chart-count>最近 ${model.metrics.length} 个点</span>`)}
                <div class="dragon-dashboard-chart" data-dashboard-chart>${renderLossChart(model.metrics)}</div>
            </section>

            <div class="dragon-dashboard-lower-grid">
                <section class="dragon-section dragon-dashboard-history-section dragon-reveal" data-stagger="5">
                    ${sectionHeading('history', '记录', '最近训练', `<button class="dragon-inline-action" type="button" data-dashboard-action="history">查看全部</button>`)}
                    ${historyHtml}
                </section>
                <section class="dragon-section dragon-dashboard-log-section dragon-reveal" data-stagger="6">
                    ${sectionHeading('terminal', '输出', '实时日志', `<span class="dragon-dashboard-section-note" data-dashboard-log-status>${model.running ? '实时连接中' : '等待训练'}</span>`)}
                    <div class="dragon-log-stream dragon-dashboard-log-stream" data-dashboard-log>${logsHtml}</div>
                </section>
            </div>
        </div>
    `;
}

function commandButton(icon, label, action, variant, hidden = false) {
    return `<button class="dragon-btn ${variant}" type="button" data-dashboard-action="${action}" ${hidden ? 'hidden' : ''}>${renderIcon(icon, 'dragon-btn-icon')}<span>${label}</span></button>`;
}

function contextItem(icon, label, value, key) {
    return `
        <div class="dragon-dashboard-context-item">
            <span class="dragon-dashboard-context-icon">${renderIcon(icon)}</span>
            <div>
                <span class="dragon-dashboard-kicker">${label}</span>
                <strong data-dashboard-${key}>${escapeHtml(value)}</strong>
            </div>
        </div>
    `;
}

function sectionHeading(icon, eyebrow, title, trailing = '') {
    return `
        <div class="dragon-dashboard-section-head">
            <div class="dragon-dashboard-heading-copy">
                <span class="dragon-dashboard-heading-icon">${renderIcon(icon)}</span>
                <div><span class="dragon-eyebrow">${eyebrow}</span><h2>${title}</h2></div>
            </div>
            ${trailing}
        </div>
    `;
}

function overviewItem(icon, label, value, detail, key, tone = 'neutral') {
    return `
        <div class="dragon-dashboard-overview-item" data-tone="${tone}">
            <span class="dragon-dashboard-overview-icon" data-dashboard-summary-icon="${key}">${renderIcon(icon)}</span>
            <div>
                <span class="dragon-dashboard-overview-label">${label}</span>
                <strong data-dashboard-summary="${key}">${escapeHtml(value)}</strong>
                <small data-dashboard-summary-detail="${key}">${escapeHtml(detail)}</small>
            </div>
        </div>
    `;
}

function metricCell(key, label, value) {
    const icons = {
        loss: 'chart', lr: 'zap', step: 'hash', rate: 'gauge', eta: 'clock',
        epoch: 'layers', vram: 'memory', 'vram-peak': 'memory', gpu: 'cpu',
        temp: 'thermometer', 'log-age': 'terminal',
    };
    return `
        <div class="dragon-dashboard-metric-cell" data-dashboard-metric="${key}">
            <span class="dragon-dashboard-metric-label">${renderIcon(icons[key] || 'activity')}<span>${label}</span></span>
            <strong data-dashboard-value="${key}">${escapeHtml(value)}</strong>
        </div>
    `;
}

function createDashboardModel(data) {
    const status = data.status || {};
    const latest = latestMetric(data.metrics, status.latest_metric);
    const progress = status.latest_progress || {};
    const system = status.latest_system || {};
    const state = normalizeState(status.status);
    const current = numberOrUndefined(progress.current ?? progress.step ?? latest.step);
    const total = numberOrUndefined(progress.total ?? progress.total_steps ?? latest.total);
    const rate = latest.rate || progress.rate || '-';
    const epoch = latest.epoch ?? progress.epoch ?? '-';
    const eta = calculateTrainingEtaMetricInfo({
        isRunning: state === 'running' || state === 'training',
        current,
        total,
        progressRate: rate,
    }).text;
    const runLabel = status.output_dir || status.run_dir || status.current_output_dir || '尚未启动训练';
    const vramPeak = formatVramPeak({
        peakVram: data.peakVram ?? system.vram_used_gb,
        status,
    });
    return {
        ...data,
        state,
        running: state === 'running' || state === 'training' || state === 'compiling',
        subtitle: state === 'running' || state === 'training'
            ? `当前进度 ${formatStep(current)} / ${formatStep(total)}`
            : '尚未启动训练。开始训练后，这里会显示真实的 loss、参数和运行日志。',
        configLabel: formatConfigLabel(status),
        runLabel,
        health: formatHealth(status, state, current, total, latest),
        progressText: formatProgressText(progress, current, total),
        progressPercent: progressPercent(current, total),
        lossTrend: formatLossTrend(data.metrics, latest.loss),
        loss: formatLoss(latest.loss),
        lr: formatLr(latest.lr),
        stepText: formatStep(current),
        rate,
        eta,
        epoch: formatValue(epoch),
        vram: formatVram(system),
        vramPeak,
        gpu: formatPercent(system.gpu_util),
        temp: formatTemperature(system.gpu_temp),
        logAge: formatLogAge(status.last_output_at),
    };
}

function mountDashboard(wrapper, initialData) {
    const model = {
        status: initialData.status || {},
        metrics: [...(initialData.metrics || [])],
        peakVram: numberOrUndefined(initialData.status?.latest_system?.vram_used_gb),
        peakGpu: numberOrUndefined(initialData.status?.latest_system?.gpu_util),
        peakTemp: numberOrUndefined(initialData.status?.latest_system?.gpu_temp),
    };
    const subscriptions = [
        onMessage('progress', (message) => {
            model.status.latest_progress = { ...(model.status.latest_progress || {}), ...message };
            renderDashboardState(wrapper, model);
        }),
        onMessage('metrics', (message) => {
            model.metrics.push(message);
            model.metrics = model.metrics.slice(-500);
            model.status.latest_metric = message;
            renderDashboardState(wrapper, model);
        }),
        onMessage('system', (message) => {
            model.status.latest_system = message;
            model.peakVram = Math.max(model.peakVram || 0, numberOrUndefined(message.vram_used_gb) || 0) || model.peakVram;
            model.peakGpu = Math.max(model.peakGpu || 0, numberOrUndefined(message.gpu_util) || 0) || model.peakGpu;
            model.peakTemp = Math.max(model.peakTemp || 0, numberOrUndefined(message.gpu_temp) || 0) || model.peakTemp;
            renderDashboardState(wrapper, model);
        }),
        onMessage('status', (message) => {
            model.status = { ...model.status, ...message, status: message.state || model.status.status };
            appendStatusLog(wrapper, message);
            renderDashboardState(wrapper, model);
        }),
        onMessage('log', (message) => {
            appendDashboardLog(wrapper, message.message || message.text || message.line || '', message.level);
        }),
    ].filter(Boolean);

    connectWebSocket();
    bindDashboardActions(wrapper, model);

    const pollTimer = window.setInterval(async () => {
        const [status, metrics] = await Promise.all([
            readApi('/api/training/status', {}),
            readApi('/api/training/metrics', []),
        ]);
        const live = {
            status: status && typeof status === 'object' ? status : {},
            metrics: Array.isArray(metrics) ? metrics : [],
        };
        model.status = live.status;
        model.metrics = live.metrics;
        renderDashboardState(wrapper, model);
    }, 5000);

    const observer = new MutationObserver(() => {
        if (!document.contains(wrapper)) {
            window.clearInterval(pollTimer);
            subscriptions.forEach((unsubscribe) => unsubscribe());
            disconnectWebSocket();
            observer.disconnect();
        }
    });
    observer.observe(wrapper.parentElement || document.body, { childList: true, subtree: true });
}

function bindDashboardActions(wrapper, model) {
    wrapper.querySelectorAll('[data-dashboard-action]').forEach((button) => {
        button.addEventListener('click', async () => {
            const action = button.dataset.dashboardAction;
            if (action === 'live') window.location.hash = '#live-training';
            if (action === 'config') window.location.hash = '#config/training-config/base-models';
            if (action === 'history') window.location.hash = '#history';
            if (action === 'stop') {
                button.disabled = true;
                button.innerHTML = `${renderIcon('stop', 'dragon-btn-icon')}<span>正在停止</span>`;
                await readApi('/api/training/stop', { ok: false }, { method: 'POST' });
                model.status.status = 'idle';
                renderDashboardState(wrapper, model);
            }
        });
    });
    wrapper.querySelectorAll('.dragon-history-item').forEach((item) => {
        item.addEventListener('click', () => {
            if (item.dataset.taskId) window.location.hash = `#history/${item.dataset.taskId}`;
        });
    });
}

function renderDashboardState(wrapper, model) {
    if (!document.contains(wrapper)) return;
    const view = createDashboardModel({
        status: model.status,
        metrics: model.metrics,
        history: [],
        logs: [],
        peakVram: model.peakVram,
    });
    setText(wrapper, '[data-dashboard-title]', view.running ? '训练进行中' : '训练中心');
    setText(wrapper, '[data-dashboard-subtitle]', view.subtitle);
    setText(wrapper, '[data-dashboard-state-text]', stateText(view.state));
    setText(wrapper, '[data-dashboard-config]', view.configLabel);
    setText(wrapper, '[data-dashboard-run]', view.runLabel);
    setText(wrapper, '[data-dashboard-health]', view.health);
    setText(wrapper, '[data-dashboard-progress-text]', view.progressText);
    setText(wrapper, '[data-dashboard-progress-percent]', `${Math.round(view.progressPercent)}%`);
    setText(wrapper, '[data-dashboard-ring-loss]', view.loss);
    setText(wrapper, '[data-dashboard-ring-rate]', view.rate);
    setText(wrapper, '[data-dashboard-chart-count]', `最近 ${model.metrics.length} 个点`);
    setText(wrapper, '[data-dashboard-summary="loss-trend"]', view.lossTrend.value);
    setText(wrapper, '[data-dashboard-summary-detail="loss-trend"]', view.lossTrend.detail);
    setText(wrapper, '[data-dashboard-summary="eta"]', view.eta);
    setText(wrapper, '[data-dashboard-summary-detail="eta"]', `当前速度 ${view.rate}`);
    setText(wrapper, '[data-dashboard-summary="vram"]', view.vram);
    setText(wrapper, '[data-dashboard-summary-detail="vram"]', `峰值 ${view.vramPeak}`);
    setText(wrapper, '[data-dashboard-summary="gpu"]', view.gpu);
    setText(wrapper, '[data-dashboard-summary-detail="gpu"]', `当前温度 ${view.temp}`);
    const values = {
        loss: view.loss,
        lr: view.lr,
        step: view.stepText,
        rate: view.rate,
        eta: view.eta,
        epoch: view.epoch,
        vram: view.vram,
        gpu: view.gpu,
        temp: view.temp,
        'log-age': view.logAge,
    };
    Object.entries(values).forEach(([key, value]) => {
        setText(wrapper, `[data-dashboard-value="${key}"]`, value);
    });
    setText(wrapper, '[data-dashboard-value="vram-peak"]', view.vramPeak);
    const trendItem = wrapper.querySelector('[data-dashboard-summary="loss-trend"]')?.closest('.dragon-dashboard-overview-item');
    if (trendItem) trendItem.dataset.tone = view.lossTrend.tone;
    const trendIcon = wrapper.querySelector('[data-dashboard-summary-icon="loss-trend"]');
    if (trendIcon) trendIcon.innerHTML = renderIcon(view.lossTrend.icon);
    const stateBadge = wrapper.querySelector('[data-dashboard-state]');
    if (stateBadge) stateBadge.dataset.state = view.state;
    const dot = wrapper.querySelector('[data-dashboard-state] .dragon-nav-status-dot');
    if (dot) dot.dataset.state = view.state;
    const fill = wrapper.querySelector('[data-dashboard-progress-fill]');
    if (fill) fill.style.width = `${view.progressPercent}%`;
    const ring = wrapper.querySelector('[data-dashboard-ring]');
    if (ring) ring.style.setProperty('--dashboard-progress', `${view.progressPercent}%`);
    const stop = wrapper.querySelector('[data-dashboard-action="stop"]');
    if (stop) {
        stop.hidden = !view.running;
        stop.disabled = false;
        stop.innerHTML = `${renderIcon('stop', 'dragon-btn-icon')}<span>停止训练</span>`;
    }
    const logStatus = wrapper.querySelector('[data-dashboard-log-status]');
    if (logStatus) logStatus.textContent = view.running ? '实时连接中' : '等待训练';
    const chart = wrapper.querySelector('[data-dashboard-chart]');
    if (chart) chart.innerHTML = renderLossChart(model.metrics);
}

function appendStatusLog(wrapper, message) {
    const text = message.message || message.state;
    if (text) appendDashboardLog(wrapper, `[状态] ${text}`);
}

function appendDashboardLog(wrapper, text, level) {
    const log = wrapper.querySelector('[data-dashboard-log]');
    if (!log) return;
    log.querySelector('.dragon-empty-state')?.remove();
    appendLogLine(log, text, level);
}

function setText(root, selector, value) {
    const element = root.querySelector(selector);
    if (element) element.textContent = value == null || value === '' ? '-' : String(value);
}

function latestMetric(metrics, fallback) {
    return metrics.slice().reverse().find((item) => item && (item.loss !== undefined || item.lr !== undefined))
        || fallback
        || {};
}

function formatProgressText(progress, current, total) {
    const labels = {
        Training: '训练', training: '训练', Caching: '缓存', caching: '缓存',
        Compiling: '编译', compiling: '编译', Saving: '保存', saving: '保存',
    };
    const label = labels[progress.label] || progress.label || '训练';
    const rate = progress.rate ? ` · ${progress.rate}` : '';
    return `${label} ${formatStep(current)} / ${formatStep(total)}${rate}`;
}

function formatConfigLabel(status) {
    const variants = {
        lora: 'LoRA', lokr: 'LoKr', loha: 'LoHa', reft: 'ReFT',
        hydralora: 'HydraLoRA', glora: 'GLoRA', tlora: 'T-LoRA', vera: 'VeRA',
    };
    const presets = {
        default: '默认预设', low_vram: '低显存预设', low_vram_blockswap: '低显存块交换',
        balanced_16g: '16GB 均衡预设', debug: '调试预设', half: '半量预设',
        quarter: '四分之一预设', tenth: '十分之一预设', graft: 'Graft 预设',
    };
    const variant = status.variant ? (variants[status.variant] || status.variant) : '';
    const preset = status.preset ? (presets[status.preset] || status.preset) : '';
    return [variant, preset].filter(Boolean).join(' · ') || '尚未选择训练配置';
}

function formatHealth(status, state, current, total, latest) {
    if (status.anomaly_message || status.error_hint) {
        return status.anomaly_message || status.error_hint;
    }
    if (state === 'running' || state === 'training') {
        const parts = [];
        if (Number.isFinite(current)) parts.push(`步骤 ${formatStep(current)} / ${formatStep(total)}`);
        if (numberOrUndefined(latest.loss) !== undefined) parts.push(`损失 ${formatLoss(latest.loss)}`);
        if (numberOrUndefined(latest.lr) !== undefined) parts.push(`学习率 ${formatLr(latest.lr)}`);
        if (parts.length) return parts.join(' · ');
    }
    return status.last_log_line || stateText(state);
}

function progressPercent(current, total) {
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) return 0;
    return Math.max(0, Math.min(100, (current / total) * 100));
}

function formatStep(value) {
    return Number.isFinite(value) ? Math.round(value).toLocaleString('zh-CN') : '-';
}

function formatLoss(value) {
    const number = numberOrUndefined(value);
    return number === undefined ? '-' : number.toFixed(5);
}

function formatLossTrend(metrics, fallback) {
    const losses = metrics
        .map((item) => numberOrUndefined(item?.loss))
        .filter((value) => value !== undefined);
    const latest = losses.at(-1) ?? numberOrUndefined(fallback);
    if (latest === undefined) {
        return { value: '等待数据', detail: '尚无损失采样点', icon: 'chart', tone: 'neutral' };
    }
    if (losses.length < 2) {
        return { value: formatLoss(latest), detail: '等待更多采样点', icon: 'chart', tone: 'neutral' };
    }
    const first = losses[0];
    const change = first === 0 ? 0 : ((latest - first) / Math.abs(first)) * 100;
    if (Math.abs(change) < 0.05) {
        return { value: '基本稳定', detail: `${formatLoss(first)} 至 ${formatLoss(latest)}`, icon: 'activity', tone: 'neutral' };
    }
    const improving = change < 0;
    return {
        value: `${improving ? '下降' : '上升'} ${Math.abs(change).toFixed(1)}%`,
        detail: `${formatLoss(first)} 至 ${formatLoss(latest)}`,
        icon: improving ? 'trendDown' : 'trendUp',
        tone: improving ? 'positive' : 'caution',
    };
}

function formatVram(system) {
    const used = numberOrUndefined(system.vram_used_gb);
    const total = numberOrUndefined(system.vram_total_gb);
    if (used === undefined) return '-';
    return total === undefined ? `${used.toFixed(1)} GB` : `${used.toFixed(1)} / ${total.toFixed(1)} GB`;
}

function formatVramPeak(model) {
    const used = numberOrUndefined(model.peakVram);
    const total = numberOrUndefined(model.status?.latest_system?.vram_total_gb);
    if (used === undefined) return '-';
    return total === undefined ? `${used.toFixed(1)} GB` : `${used.toFixed(1)} / ${total.toFixed(1)} GB`;
}

function formatPercent(value) {
    const number = numberOrUndefined(value);
    return number === undefined ? '-' : `${Math.round(number)}%`;
}

function formatTemperature(value) {
    const number = numberOrUndefined(value);
    return number === undefined ? '-' : `${Math.round(number)}°C`;
}

function formatLogAge(value) {
    const timestamp = numberOrUndefined(value);
    if (timestamp === undefined) return '-';
    const seconds = Math.max(0, Math.round(Date.now() / 1000 - timestamp));
    return seconds < 5 ? '刚刚' : `${seconds} 秒前`;
}

function formatValue(value) {
    return value === undefined || value === null || value === '' ? '-' : String(value);
}

function numberOrUndefined(value) {
    const number = Number(value);
    return value === null || value === undefined || value === '' || !Number.isFinite(number)
        ? undefined
        : number;
}

function normalizeState(value) {
    const state = String(value || 'idle');
    return ['idle', 'running', 'training', 'queued', 'compiling', 'completed', 'error', 'stopped'].includes(state)
        ? state
        : 'unknown';
}

function stateText(state) {
    const map = {
        idle: '空闲', running: '训练中', training: '训练中', compiling: '编译中',
        queued: '排队中', completed: '已完成', error: '错误', stopped: '已停止', unknown: '未知',
    };
    return map[state] || '未知';
}

function renderHistoryItem(task) {
    const state = normalizeState(task.status || task.state);
    const name = task.output_name || task.task_name || task.id || '未命名任务';
    const time = task.started_at || task.created_at || '';
    return `
        <li class="dragon-history-item" data-task-id="${escapeHtml(task.id || '')}">
            <span class="dragon-history-item-name">${escapeHtml(name)}</span>
            <span class="dragon-history-item-state" data-state="${state}">${stateText(state)}</span>
            <span class="dragon-history-item-meta">${escapeHtml(time)}</span>
        </li>
    `;
}

function renderLogRecord(record) {
    const level = record.level || record.kind || '';
    const text = record.message || record.text || record.line || '';
    return `<div class="dragon-log-line" data-level="${escapeHtml(level)}">${escapeHtml(text)}</div>`;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[character]));
}
