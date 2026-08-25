/* View-model and compact renderers for the live training master-detail workspace. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { isRunningState, stateText } from './live-training-state.js?v=dragon-ui-20260825v46';

const ERROR_STATES = new Set(['error', 'failed', 'interrupted', 'unavailable']);

export function liveWorkspaceMode(state) {
    const normalized = String(state || '').toLowerCase();
    if (ERROR_STATES.has(normalized)) return 'error';
    return isRunningState(normalized) ? 'running' : 'idle';
}

export function applyWorkspaceSnapshot(model, queuePayload = null, historyPayload = null) {
    if (queuePayload && typeof queuePayload === 'object') {
        const items = Array.isArray(queuePayload.items) ? queuePayload.items : [];
        const summary = queuePayload.summary && typeof queuePayload.summary === 'object' ? queuePayload.summary : countQueueStates(items);
        model.queueSummary = {
            queued: finiteNumber(summary.queued),
            running: finiteNumber(summary.running),
            error: finiteNumber(summary.error),
        };
        model.queueItems = items.filter((item) => ['running', 'queued', 'error'].includes(String(item.state))).slice(0, 5);
        model.queueEtaSeconds = estimatedQueueSeconds(model.queueItems);
    }
    if (historyPayload && typeof historyPayload === 'object') {
        model.recentTasks = Array.isArray(historyPayload.tasks) ? historyPayload.tasks.slice(0, 5) : [];
    }
    return model;
}

export function renderLiveSidebar(model) {
    return `
        <aside class="dragon-live-sidebar" aria-label="训练任务导航">
            <div class="dragon-live-sidebar-head">
                <span class="dragon-eyebrow">任务调度</span>
                <strong>训练队列</strong>
                <a href="#queue" aria-label="打开完整训练队列">${renderIcon('list')}</a>
            </div>
            <div data-live-sidebar-body>${renderLiveSidebarBody(model)}</div>
        </aside>`;
}

export function renderLiveSidebarBody(model) {
    const summary = model.queueSummary || {};
    const activeItems = Array.isArray(model.queueItems) ? model.queueItems : [];
    const recentTasks = Array.isArray(model.recentTasks) ? model.recentTasks : [];
    return `
        <section class="dragon-live-queue-summary" aria-label="队列摘要">
            <div><span>排队中</span><strong data-live-queue-count>${finiteNumber(summary.queued)}</strong></div>
            <div><span>正在运行</span><strong>${finiteNumber(summary.running)}</strong></div>
            <p><span>预计总耗时</span><strong>${formatQueueEta(model.queueEtaSeconds, summary.queued)}</strong></p>
        </section>
        ${activeItems.length ? `<section class="dragon-live-sidebar-section"><h2>活动队列</h2><div class="dragon-live-task-list">${activeItems.map(renderQueueTask).join('')}</div></section>` : ''}
        <section class="dragon-live-sidebar-section dragon-live-recent-section">
            <div class="dragon-live-sidebar-section-head"><h2>最近任务</h2><a href="#history">全部</a></div>
            <div class="dragon-live-task-list">${recentTasks.length ? recentTasks.map(renderHistoryTask).join('') : '<p class="dragon-live-sidebar-empty">暂无历史任务</p>'}</div>
        </section>`;
}

export function renderLiveStatePanel(model, mode) {
    const error = mode === 'error';
    return `
        <section class="dragon-live-state-panel" data-live-section="${mode}" ${liveWorkspaceMode(model.state) === mode ? '' : 'hidden'}>
            <span class="dragon-live-state-symbol" data-tone="${error ? 'error' : 'idle'}">${renderIcon(error ? 'activity' : 'terminal')}</span>
            <div><span class="dragon-eyebrow">${error ? 'MONITOR ALERT' : 'TRAINING IDLE'}</span><h2>${error ? '训练监控出现异常' : '当前无正在运行的任务'}</h2><p data-live-${mode}-message>${escapeHtml(error ? (model.lastActivity || '训练已异常中断，请检查日志或历史任务。') : '创建训练或打开队列，监控数据会在任务启动后自动接入。')}</p></div>
            <div class="dragon-live-state-actions">
                ${error ? '<a class="dragon-btn dragon-btn-secondary" href="#history">查看历史</a><button class="dragon-btn dragon-btn-primary" type="button" data-tool-action="retry">重新连接</button>' : `<a class="dragon-btn dragon-btn-primary" href="#config/training-config">${renderIcon('filePlus', 'dragon-btn-icon')}<span>快速创建新训练</span></a><a class="dragon-btn dragon-btn-secondary" href="#queue">查看队列</a>`}
            </div>
        </section>`;
}

export function lossDelta(metrics = []) {
    const values = metrics.map((item) => Number(item?.loss)).filter(Number.isFinite);
    if (values.length < 2) return { text: '等待下一步数据', tone: 'neutral' };
    const delta = values.at(-1) - values.at(-2);
    const prefix = delta > 0 ? '+' : '';
    return { text: `较上步 ${prefix}${delta.toFixed(4)}`, tone: delta <= 0 ? 'good' : 'warning' };
}

export function hardwarePercent(value, total) {
    const current = Number(value);
    const maximum = Number(total);
    if (!Number.isFinite(current) || !Number.isFinite(maximum) || maximum <= 0) return 0;
    return Math.max(0, Math.min(100, (current / maximum) * 100));
}

function renderQueueTask(item) {
    const state = String(item.state || 'queued');
    const title = item.variant || basename(item.source_config_file || item.runtime_config_file) || item.id || '未命名任务';
    return `<a class="dragon-live-task" href="#queue" data-state="${escapeHtml(state)}"><span class="dragon-live-task-dot"></span><span><strong>${escapeHtml(title)}</strong><small>${escapeHtml(stateText(state))}</small></span></a>`;
}

function renderHistoryTask(task) {
    const id = String(task.id || '');
    const state = String(task.state || task.status || 'unknown');
    const title = task.name || task.variant || task.history_run_label || id || '未命名任务';
    const time = task.started_at_text || task.finished_at_text || '';
    return `<a class="dragon-live-task" href="#history/${encodeURIComponent(id)}" data-state="${escapeHtml(state)}"><span class="dragon-live-task-dot"></span><span><strong>${escapeHtml(title)}</strong><small>${escapeHtml(stateText(state))}${time ? ` · ${escapeHtml(shortTime(time))}` : ''}</small></span></a>`;
}

function estimatedQueueSeconds(items) {
    let found = false;
    const total = items.reduce((sum, item) => {
        const value = Number(item.estimated_duration_seconds ?? item.estimated_seconds ?? item.eta_seconds);
        if (!Number.isFinite(value) || value <= 0) return sum;
        found = true;
        return sum + value;
    }, 0);
    return found ? total : null;
}

function formatQueueEta(seconds, queued) {
    if (!Number(queued)) return '队列空闲';
    if (seconds == null || !Number.isFinite(Number(seconds))) return '等待任务估算';
    const minutes = Math.max(1, Math.ceil(Number(seconds) / 60));
    if (minutes < 60) return `约 ${minutes} 分钟`;
    const hours = Math.floor(minutes / 60);
    const rest = minutes % 60;
    return rest ? `约 ${hours} 小时 ${rest} 分` : `约 ${hours} 小时`;
}

function countQueueStates(items) {
    return items.reduce((counts, item) => {
        const state = String(item.state || '');
        if (Object.hasOwn(counts, state)) counts[state] += 1;
        return counts;
    }, { queued: 0, running: 0, error: 0 });
}

function basename(value) { return String(value || '').replaceAll('\\', '/').split('/').pop() || ''; }
function shortTime(value) { return String(value).replace(/^\d{4}-/, '').slice(0, 11); }
function finiteNumber(value) { return Number.isFinite(Number(value)) ? Number(value) : 0; }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
