/* Training history page: task list and detail views.
 * Fetches from /api/training/history and /api/training/history/{taskId}.
 */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderLossChart } from './live-training.js?v=dragon-ui-20260812v35';

const api = createApiClient();

export async function loadHistory(context = {}) {
    const taskId = context.taskId || null;
    if (taskId) return loadHistoryDetail(taskId);

    let data = {};
    try { data = await api('/api/training/history?limit=50'); } catch { /* server unavailable */ }
    const tasks = Array.isArray(data?.tasks) ? data.tasks : [];
    const html = renderHistoryList(tasks);
    return { html, onMount: (root) => bindHistoryList(root) };
}

async function loadHistoryDetail(taskId) {
    let payload = null;
    let error = '';
    try {
        payload = await api(`/api/training/history/${encodeURIComponent(taskId)}`);
        if (payload?.ok === false) error = payload.error || '加载训练记录失败';
    } catch (cause) {
        error = cause.message || '加载训练记录失败';
    }

    const html = error
        ? renderHistoryDetailError(taskId, error)
        : renderHistoryDetail(taskId, payload || {});
    return { html, onMount: (root) => bindHistoryDetail(root, taskId) };
}

function renderHistoryList(tasks) {
    if (!tasks.length) {
        return `
            <div class="dragon-page">
                <div class="dragon-page-hero dragon-reveal">
                    <h1>训练历史</h1>
                    <p>查看历史训练任务的配置、损失曲线和样张。</p>
                </div>
                <div class="dragon-empty-state dragon-reveal" data-stagger="1">
                    <p>暂无训练记录</p>
                </div>
            </div>
        `;
    }

    const groups = {};
    for (const task of tasks) {
        const group = task.config_group || task.variant || '未分组';
        if (!groups[group]) groups[group] = [];
        groups[group].push(task);
    }

    const groupSections = Object.entries(groups).map(([name, items], index) => `
        <div class="dragon-section dragon-reveal" data-stagger="${Math.min(index + 1, 6)}">
            <h2 class="dragon-section-title">${escapeHtml(name)}</h2>
            <p class="dragon-section-desc">${items.length} 个任务</p>
            <ul class="dragon-history-list">
                ${items.map(renderHistoryItem).join('')}
            </ul>
        </div>
    `).join('');

    return `
        <div class="dragon-page dragon-page-wide">
            <div class="dragon-page-hero dragon-reveal">
                <h1>训练历史</h1>
                <p>查看历史训练任务的配置、损失曲线和样张。共 ${tasks.length} 条记录。</p>
            </div>
            ${groupSections}
        </div>
    `;
}

function renderHistoryDetail(taskId, payload) {
    const task = payload.task || {};
    const metrics = Array.isArray(payload.metrics) ? payload.metrics : [];
    const logs = Array.isArray(payload.logs) ? payload.logs : [];
    const name = task.output_name || task.task_name || task.id || taskId;
    const state = task.state || task.status || 'unknown';
    const lossPoints = metrics.filter((item) => item.loss != null);
    const lastLoss = lossPoints.length ? Number(lossPoints[lossPoints.length - 1].loss).toFixed(5) : '-';
    const lastStep = lastMetricValue(metrics, 'step');
    const lastLr = lastMetricValue(metrics, 'lr');
    const configFile = task.config_file || payload.config_file || '';
    const runDir = task.run_dir || payload.run_dir || '';
    const startedAt = task.started_at || task.created_at || '';
    const finishedAt = task.finished_at || task.completed_at || '';
    const logPreview = logs.slice(-8).map(renderLogLine).join('') || '<div class="dragon-empty-state"><p>暂无日志</p></div>';

    return `
        <div class="dragon-page dragon-page-wide dragon-history-detail-page" data-history-detail="${escapeAttribute(taskId)}">
            <div class="dragon-page-hero dragon-reveal">
                <div class="dragon-history-detail-toolbar">
                    <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-back>返回列表</button>
                    <span class="dragon-history-item-state" data-state="${escapeAttribute(state)}">${stateText(state)}</span>
                </div>
                <span class="dragon-eyebrow">训练记录</span>
                <h1>${escapeHtml(name)}</h1>
                <p>${escapeHtml(configFile || taskId)}</p>
            </div>

            <div class="dragon-metrics-grid dragon-reveal" data-stagger="1">
                ${metricTile('最终损失', lastLoss)}
                ${metricTile('最后步数', lastStep ?? '-')}
                ${metricTile('学习率', formatLr(lastLr))}
                ${metricTile('Loss 点', lossPoints.length || task.metric_count || 0)}
                ${metricTile('日志条数', logs.length || task.log_count || 0)}
                ${metricTile('任务 ID', taskId)}
            </div>

            <section class="dragon-section dragon-reveal" data-stagger="2">
                <h2 class="dragon-section-title">运行信息</h2>
                <div class="dragon-history-detail-kv">
                    ${detailRow('配置', configFile)}
                    ${detailRow('运行目录', runDir)}
                    ${detailRow('开始时间', startedAt)}
                    ${detailRow('结束时间', finishedAt)}
                    ${detailRow('任务类型', task.job === 'preprocess' ? '预处理' : '训练')}
                </div>
            </section>

            <section class="dragon-section dragon-reveal" data-stagger="3">
                <h2 class="dragon-section-title">损失曲线</h2>
                <p class="dragon-section-desc">最近 ${metrics.length} 条记录。</p>
                <div class="dragon-chart-container">${renderLossChart(metrics)}</div>
            </section>

            <section class="dragon-section dragon-reveal" data-stagger="4">
                <div class="dragon-section-header-row">
                    <div><span class="dragon-eyebrow">日志</span><h2 class="dragon-section-title">最近输出</h2></div>
                    <a class="dragon-btn dragon-btn-ghost dragon-btn-sm" href="/api/training/history/${encodeURIComponent(taskId)}/logs/download">下载日志</a>
                </div>
                <div class="dragon-log-panel dragon-history-log-panel">${logPreview}</div>
            </section>
        </div>
    `;
}

function renderHistoryDetailError(taskId, error) {
    return `
        <div class="dragon-page">
            <div class="dragon-page-hero dragon-reveal">
                <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-back>返回列表</button>
                <h1>训练历史</h1>
                <p>${escapeHtml(error)}</p>
            </div>
            <div class="dragon-empty-state dragon-reveal" data-stagger="1">
                <p>无法加载任务 ${escapeHtml(taskId)}</p>
            </div>
        </div>
    `;
}

function renderHistoryItem(task) {
    const name = task.output_name || task.task_name || task.id || '-';
    const state = task.status || task.state || 'unknown';
    const time = task.started_at || task.created_at || '';
    const steps = task.total_steps || task.steps || '';
    const loss = task.final_loss != null ? Number(task.final_loss).toFixed(4) : '';

    return `
        <li class="dragon-history-item" data-task-id="${escapeAttribute(task.id || '')}" role="button" tabindex="0">
            <span class="dragon-history-item-name">${escapeHtml(name)}</span>
            ${loss ? `<span class="dragon-history-item-loss">损失: ${loss}</span>` : ''}
            ${steps ? `<span class="dragon-history-item-steps">${steps} 步</span>` : ''}
            <span class="dragon-history-item-state" data-state="${escapeAttribute(state)}">${stateText(state)}</span>
            <span class="dragon-history-item-meta">${escapeHtml(time)}</span>
        </li>
    `;
}

function renderLogLine(record) {
    const line = record.line || record.message || record.text || JSON.stringify(record);
    const level = record.level || '';
    return `<div class="dragon-log-line"${level ? ` data-level="${escapeAttribute(level)}"` : ''}>${escapeHtml(line)}</div>`;
}

function detailRow(label, value) {
    if (!value) return '';
    return `<div class="dragon-history-detail-row"><span>${escapeHtml(label)}</span><strong class="dragon-text-mono">${escapeHtml(value)}</strong></div>`;
}

function metricTile(label, value) {
    return `<div class="dragon-metric-tile"><div class="dragon-metric-value">${escapeHtml(value ?? '-')}</div><div class="dragon-metric-label">${label}</div></div>`;
}

function bindHistoryList(root) {
    root.querySelectorAll('.dragon-history-item[data-task-id]').forEach((item) => {
        const open = () => {
            const taskId = item.dataset.taskId;
            if (taskId) window.location.hash = `#history/${encodeURIComponent(taskId)}`;
        };
        item.addEventListener('click', open);
        item.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                open();
            }
        });
    });
}

function bindHistoryDetail(root) {
    root.querySelector('[data-history-back]')?.addEventListener('click', () => {
        window.location.hash = '#history';
    });
}

function lastMetricValue(metrics, key) {
    for (let index = metrics.length - 1; index >= 0; index -= 1) {
        const value = metrics[index]?.[key];
        if (value != null && value !== '') return value;
    }
    return null;
}

function formatLr(value) {
    if (value == null || value === '') return '-';
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toExponential(3) : String(value);
}

function stateText(state) {
    const map = {
        idle: '空闲', running: '训练中', training: '训练中',
        queued: '排队中', completed: '已完成', error: '错误',
        stopped: '已停止', unknown: '未知',
    };
    return map[state] || state;
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
