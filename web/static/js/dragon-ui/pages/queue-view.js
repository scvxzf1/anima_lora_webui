/* Presentational helpers for the Dragon training queue workspace. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { renderStatusRegion, renderToolHero } from './tool-page.js?v=dragon-ui-20260814v43';

export const QUEUE_FILTERS = [
    ['active', '待处理'],
    ['all', '全部'],
    ['queued', '等待'],
    ['running', '运行'],
    ['error', '异常'],
    ['done', '完成'],
    ['canceled', '已取消'],
];

const STATE_LABELS = {
    queued: '等待',
    running: '运行中',
    done: '已完成',
    error: '异常',
    canceled: '已取消',
};

export function normalizeQueueSnapshot(payload = {}) {
    const items = Array.isArray(payload.items) ? payload.items.map((item) => ({ ...item })) : [];
    const fallback = countQueueStates(items);
    const summary = payload.summary && typeof payload.summary === 'object'
        ? { ...fallback, ...payload.summary }
        : fallback;
    return {
        ok: payload.ok !== false,
        error: payload.error || '',
        message: payload.message || '',
        paused: Boolean(payload.paused),
        failure_policy: payload.failure_policy === 'continue' ? 'continue' : 'pause',
        auto_retry: Boolean(payload.auto_retry),
        max_attempts: clampNumber(payload.max_attempts, 1, 10, 1),
        retry_backoff_sec: clampNumber(payload.retry_backoff_sec, 0, 3600, 0),
        status: String(payload.status || 'idle'),
        current_item_id: String(payload.current_item_id || ''),
        summary: {
            total: numberOr(summary.total, fallback.total),
            queued: numberOr(summary.queued, fallback.queued),
            running: numberOr(summary.running, fallback.running),
            done: numberOr(summary.done, fallback.done),
            error: numberOr(summary.error, fallback.error),
            canceled: numberOr(summary.canceled, fallback.canceled),
        },
        items,
    };
}

export function renderQueuePage(model, ui = {}) {
    const filter = ui.filter || 'active';
    const visibleItems = filterQueueItems(model.items, filter);
    const queuedItems = model.items.filter((item) => item.state === 'queued');
    const queuedPositions = new Map(queuedItems.map((item, index) => [String(item.id || ''), index]));
    const isRunning = model.summary.running > 0 || model.status === 'running';
    const queueState = model.paused
        ? '队列已暂停'
        : (isRunning ? '正在执行' : (model.summary.queued > 0 ? '等待调度' : '队列空闲'));
    const badgeState = escapeAttribute(model.paused ? 'paused' : model.status);
    const badge = `
        <span class="dragon-queue-state-badge" data-state="${badgeState}">
            <span class="dragon-nav-status-dot" data-state="${badgeState}"></span>
            ${queueState}
        </span>
    `;
    const heroActions = `
        <button class="dragon-btn dragon-btn-secondary" type="button" data-queue-action="toggle-pause">
            ${renderIcon(model.paused ? 'activity' : 'stop', 'dragon-btn-icon')}
            <span>${model.paused ? '继续队列' : '暂停队列'}</span>
        </button>
        <button class="dragon-btn dragon-btn-ghost" type="button" data-queue-action="refresh">
            ${renderIcon('refresh', 'dragon-btn-icon')}<span>刷新</span>
        </button>
    `;

    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-queue-manager" data-queue-root>
            ${renderToolHero({
                eyebrow: '训练编排',
                title: '训练队列',
                description: '管理等待顺序、失败策略和重试规则；运行日志与性能指标请前往“实时训练”。',
                badge,
                actions: heroActions,
            })}

            ${renderStatusRegion('data-queue-feedback', ui.feedback?.message || '', ui.feedback?.tone || '')}

            <section class="dragon-stat-grid dragon-queue-stats dragon-reveal" data-stagger="1" aria-label="队列统计">
                ${renderStat('全部', model.summary.total, 'all', filter)}
                ${renderStat('等待', model.summary.queued, 'queued', filter)}
                ${renderStat('运行', model.summary.running, 'running', filter)}
                ${renderStat('异常', model.summary.error, 'error', filter)}
                ${renderStat('完成', model.summary.done, 'done', filter)}
                ${renderStat('取消', model.summary.canceled, 'canceled', filter)}
            </section>

            <div class="dragon-queue-layout">
                <section class="dragon-tool-panel dragon-queue-worklist dragon-reveal" data-stagger="2">
                    <div class="dragon-queue-list-head">
                        <div>
                            <span class="dragon-eyebrow">任务列表</span>
                            <h2>${filterLabel(filter)}</h2>
                            <p>显示 ${visibleItems.length} / ${model.items.length} 个任务</p>
                        </div>
                        <div class="dragon-queue-filter" role="group" aria-label="筛选队列任务">
                            ${QUEUE_FILTERS.map(([value, label]) => `
                                <button type="button" data-queue-filter="${value}" aria-pressed="${filter === value}">${label}</button>
                            `).join('')}
                        </div>
                    </div>
                    <div class="dragon-queue-list" data-queue-list>
                        ${visibleItems.length
                            ? visibleItems.map((item) => renderQueueItem(item, model, queuedPositions)).join('')
                            : renderQueueEmpty(model.items.length, filter)}
                    </div>
                </section>

                <aside class="dragon-queue-sidebar">
                    ${renderPolicyPanel(model, ui.draft)}
                    ${renderBulkPanel(model)}
                </aside>
            </div>
        </div>
    `;
}

export function stateLabel(state) {
    return STATE_LABELS[String(state || '')] || '未知';
}

export function queueItemTitle(item = {}) {
    const checkpoint = String(item.resume_info?.checkpoint_name || item.resume_info?.checkpoint || '').trim();
    if (checkpoint) return `续训 · ${basename(checkpoint)}`;
    const variant = String(item.variant || '').trim();
    const preset = String(item.preset || '').trim();
    const config = basename(item.source_config_file || item.runtime_config_file || '');
    return [variant || config, preset].filter(Boolean).join(' · ') || String(item.id || '未命名任务');
}

function renderStat(label, value, filter, activeFilter) {
    return `
        <button class="dragon-stat-tile dragon-queue-stat" type="button" data-queue-filter="${filter}" aria-pressed="${activeFilter === filter}">
            <span>${label}</span><strong>${Number(value) || 0}</strong>
        </button>
    `;
}

function renderQueueItem(item, model, queuedPositions) {
    const state = String(item.state || 'unknown');
    const id = String(item.id || '');
    const queuedIndex = queuedPositions.get(id);
    const queuedTotal = queuedPositions.size;
    const runtimePath = String(item.runtime_config_file || '').trim();
    const sourcePath = String(item.source_config_file || '').trim();
    const time = queueItemTime(item);
    const attempt = Math.max(1, Number(item.attempt) || 1);
    const maxAttempts = clampNumber(item.max_attempts ?? model.max_attempts, 1, 10, model.max_attempts);
    const gpu = Array.isArray(item.gpu_whitelist) && item.gpu_whitelist.length
        ? `GPU ${item.gpu_whitelist.join(', ')}`
        : '自动选择 GPU';
    const message = String(item.message || '').trim();
    const current = id && id === model.current_item_id;
    const classes = ['dragon-queue-card'];
    if (current) classes.push('dragon-queue-card-current');
    if (state === 'error') classes.push('dragon-queue-card-error');

    return `
        <article class="${classes.join(' ')}" data-item-id="${escapeAttribute(id)}" data-state="${escapeAttribute(state)}">
            <div class="dragon-queue-card-main">
                <div class="dragon-queue-card-heading">
                    <div class="dragon-queue-card-title">
                        <span class="dragon-queue-state" data-state="${escapeAttribute(state)}">${stateLabel(state)}</span>
                        <h3>${escapeHtml(queueItemTitle(item))}</h3>
                        ${item.requires_preprocess ? '<span class="dragon-queue-chip">含预处理</span>' : ''}
                        ${item.retry_of ? '<span class="dragon-queue-chip">重试任务</span>' : ''}
                    </div>
                    <span class="dragon-queue-item-time">${escapeHtml(time)}</span>
                </div>

                <div class="dragon-queue-card-meta">
                    <span>${escapeHtml(item.kind === 'resume' ? '续训' : '训练')}</span>
                    <span>${escapeHtml(item.methods_subdir || 'gui-methods')}</span>
                    <span>${escapeHtml(gpu)}</span>
                    <span>尝试 ${attempt} / ${maxAttempts}</span>
                </div>

                ${message ? `<p class="dragon-queue-message">${escapeHtml(message)}</p>` : ''}

                <dl class="dragon-queue-paths">
                    ${runtimePath ? renderPathRow('运行配置', runtimePath) : ''}
                    ${sourcePath && sourcePath !== runtimePath ? renderPathRow('源配置', sourcePath) : ''}
                </dl>
            </div>

            <div class="dragon-queue-card-actions" aria-label="${escapeAttribute(queueItemTitle(item))} 操作">
                ${state === 'queued' ? renderMoveActions(queuedIndex, queuedTotal) : ''}
                ${['done', 'error', 'canceled'].includes(state) ? `<button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-item-action="retry">重试</button>` : ''}
                ${(state === 'queued' || state === 'running')
                    ? `<button class="dragon-btn dragon-btn-ghost dragon-btn-sm dragon-btn-danger" type="button" data-item-action="cancel">${state === 'running' ? '停止' : '取消'}</button>`
                    : `<button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-item-action="remove">移出列表</button>`}
            </div>
        </article>
    `;
}

function renderMoveActions(position, total) {
    const first = position === 0;
    const last = position === total - 1;
    return `
        <div class="dragon-queue-move-group" role="group" aria-label="调整等待顺序">
            ${moveButton('top', '置顶', first)}
            ${moveButton('up', '上移', first)}
            ${moveButton('down', '下移', last)}
            ${moveButton('bottom', '置底', last)}
        </div>
    `;
}

function moveButton(direction, label, disabled) {
    return `<button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-item-action="move" data-direction="${direction}" ${disabled ? 'disabled' : ''}>${label}</button>`;
}

function renderPathRow(label, value) {
    return `<div><dt>${label}</dt><dd title="${escapeAttribute(value)}" translate="no">${escapeHtml(value)}</dd></div>`;
}

function renderPolicyPanel(model, draft = {}) {
    const failurePolicy = draft.failure_policy ?? model.failure_policy;
    const autoRetry = draft.auto_retry ?? model.auto_retry;
    const maxAttempts = draft.max_attempts ?? model.max_attempts;
    const retryBackoff = draft.retry_backoff_sec ?? model.retry_backoff_sec;
    return `
        <form class="dragon-tool-panel dragon-queue-policy dragon-reveal" data-stagger="3" data-queue-settings-form>
            <div class="dragon-tool-panel-head">
                <div><span class="dragon-eyebrow">调度策略</span><h2>失败与重试</h2></div>
                <span class="dragon-tool-note">仅覆盖当前队列</span>
            </div>
            <label class="dragon-queue-field">
                <span>任务失败后</span>
                <select class="dragon-select" name="failure_policy" autocomplete="off">
                    <option value="pause" ${failurePolicy === 'pause' ? 'selected' : ''}>暂停队列，等待处理</option>
                    <option value="continue" ${failurePolicy === 'continue' ? 'selected' : ''}>继续执行后续任务</option>
                </select>
            </label>
            <label class="dragon-queue-check">
                <input type="checkbox" name="auto_retry" ${autoRetry ? 'checked' : ''}>
                <span><strong>自动重试可恢复异常</strong><small>最大次数包含首次运行；失败策略为“暂停”时，重试任务会等待手动继续。</small></span>
            </label>
            <div class="dragon-queue-policy-grid">
                <label class="dragon-queue-field">
                    <span>最大尝试次数</span>
                    <input class="dragon-input" type="number" name="max_attempts" min="1" max="10" step="1" inputmode="numeric" autocomplete="off" value="${escapeAttribute(maxAttempts)}">
                </label>
                <label class="dragon-queue-field">
                    <span>重试等待（秒）</span>
                    <input class="dragon-input" type="number" name="retry_backoff_sec" min="0" max="3600" step="1" inputmode="numeric" autocomplete="off" value="${escapeAttribute(retryBackoff)}">
                </label>
            </div>
            <button class="dragon-btn dragon-btn-primary" type="submit" data-queue-settings-save>保存队列策略</button>
        </form>
    `;
}

function renderBulkPanel(model) {
    const active = model.summary.queued + model.summary.running;
    const hasActiveRuntime = active > 0 || model.status === 'running';
    return `
        <section class="dragon-tool-panel dragon-queue-bulk dragon-reveal" data-stagger="4">
            <div class="dragon-tool-panel-head">
                <div><span class="dragon-eyebrow">批量操作</span><h2>队列维护</h2></div>
            </div>
            <div class="dragon-queue-bulk-group">
                ${bulkButton('cancel-waiting', '取消全部等待', model.summary.queued < 1)}
                ${bulkButton('abort-after-current', '中止后续', model.summary.queued < 1)}
                ${bulkButton('cancel-all', '取消全部', active < 1, true)}
                ${bulkButton('force-abort', '强制中止', !hasActiveRuntime, true)}
            </div>
            <div class="dragon-queue-bulk-divider"></div>
            <div class="dragon-queue-bulk-group">
                ${bulkButton('clear-completed', `清理完成（${model.summary.done}）`, model.summary.done < 1)}
                ${bulkButton('clear-canceled', `清理取消（${model.summary.canceled}）`, model.summary.canceled < 1)}
            </div>
            <p>清理和移出列表只删除队列记录，不删除训练目录、日志、权重或历史任务。</p>
        </section>
    `;
}

function bulkButton(action, label, disabled, danger = false) {
    return `<button class="dragon-btn dragon-btn-secondary${danger ? ' dragon-btn-danger' : ''}" type="button" data-queue-action="${action}" ${disabled ? 'disabled' : ''}>${label}</button>`;
}

function renderQueueEmpty(total, filter) {
    const text = total
        ? `当前筛选“${filterLabel(filter)}”没有任务。`
        : '队列为空。请从训练配置页预检后加入训练队列。';
    return `<div class="dragon-empty-state dragon-queue-empty"><p>${text}</p></div>`;
}

function filterQueueItems(items, filter) {
    if (filter === 'all') return items;
    if (filter === 'active') return items.filter((item) => ['queued', 'running', 'error'].includes(item.state));
    return items.filter((item) => item.state === filter);
}

function filterLabel(filter) {
    return QUEUE_FILTERS.find(([value]) => value === filter)?.[1] || '待处理';
}

function queueItemTime(item) {
    if (item.state === 'running') return item.started_at_text || item.created_at_text || formatTimestamp(item.started_at || item.created_at);
    if (['done', 'error', 'canceled'].includes(item.state)) return item.finished_at_text || item.started_at_text || formatTimestamp(item.finished_at || item.started_at);
    return item.created_at_text || formatTimestamp(item.created_at);
}

function formatTimestamp(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) return '时间未知';
    return new Intl.DateTimeFormat('zh-CN', {
        month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
    }).format(new Date(numeric * 1000));
}

function countQueueStates(items) {
    const counts = { total: items.length, queued: 0, running: 0, done: 0, error: 0, canceled: 0 };
    items.forEach((item) => {
        if (Object.hasOwn(counts, item.state)) counts[item.state] += 1;
    });
    return counts;
}

function numberOr(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

function clampNumber(value, min, max, fallback) {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.min(max, Math.max(min, number));
}

function basename(value) {
    const normalized = String(value || '').replaceAll('\\', '/').replace(/\/+$/, '');
    return normalized.split('/').pop() || '';
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
