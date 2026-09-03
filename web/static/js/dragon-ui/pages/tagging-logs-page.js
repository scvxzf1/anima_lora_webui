/* Dedicated bounded in-memory log viewer for tagging jobs. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { createVisibilityPoller } from '../visibility-poller.js?v=dragon-ui-20260826v2';
import {
    clearTaggingLogs,
    loadTaggingJobs,
    loadTaggingLogs,
    loadTaggingSettings,
    saveTaggingSettings,
} from './tagging-api.js?v=dragon-ui-20260901v6';
import { returnToTaggingWorkspace } from './tagging-workspace-state.js?v=dragon-ui-20260831v4';

const api = createApiClient();
const POLL_INTERVAL_MS = 1500;
const LOG_DOM_WINDOW = 400;

export async function loadTaggingLogsPage() {
    const [settingsPayload, jobsPayload] = await Promise.all([
        loadTaggingSettings(api),
        loadTaggingJobs(api),
    ]);
    const settings = settingsPayload.settings || settingsPayload;
    const retention = Number(settings.log_retention_lines || 200);
    const logsPayload = await loadTaggingLogs(api, { limit: retention });
    const state = {
        active: true,
        settings,
        jobs: jobsPayload.jobs || [],
        lines: logsPayload.lines || [],
        lastSequence: Number(logsPayload.last_sequence || 0),
        retentionLines: Number(logsPayload.retention_lines || retention),
        buffered: Number(logsPayload.buffered || 0),
        jobFilter: '',
        saving: false,
        error: '',
        notice: '',
        root: null,
        cleanup: null,
        pollTimer: null,
        logPoller: null,
        requestId: 0,
    };
    return {
        html: renderPage(state),
        onMount: (root) => mountPage(root, state),
        onUnmount: () => disposePage(state),
    };
}

function mountPage(root, state) {
    state.root = root;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    root.addEventListener('click', (event) => handleClick(state, event), options);
    root.addEventListener('change', (event) => handleChange(state, event), options);
    root.addEventListener('submit', (event) => handleSubmit(state, event), options);
    state.cleanup = () => controller.abort();
    state.logPoller = createVisibilityPoller({
        delay: POLL_INTERVAL_MS,
        poll: () => pollLogs(state),
    });
    scrollLogToBottom(state);
    schedulePoll(state);
}

function disposePage(state) {
    state.active = false;
    state.requestId += 1;
    clearPoll(state);
    state.logPoller = null;
    state.cleanup?.();
}

function renderPage(state) {
    return `<div class="dragon-page dragon-page-wide dragon-caption-page dragon-tagging-tool-page dragon-tagging-logs-page" data-tagging-logs-page>
        <header class="dragon-tagging-tool-header">
            <div><button class="dragon-icon-button" type="button" data-logs-back aria-label="返回打标工作台" title="返回">${renderIcon('chevronDown')}</button><span><span class="dragon-eyebrow">MEMORY LOG</span><h1>打标日志</h1></span></div>
            <span class="dragon-tagging-memory-badge">${renderIcon('memory')}<b data-logs-buffered>${state.buffered}</b> / <b data-logs-retention>${state.retentionLines}</b></span>
        </header>
        <div data-logs-feedback>${feedback(state)}</div>
        <section class="dragon-tagging-log-shell">
            <form class="dragon-tagging-log-toolbar" data-logs-settings-form>
                <label class="dragon-field"><span>任务</span><select class="dragon-select" name="job_id" data-logs-job-filter><option value="">全部任务</option>${jobOptions(state)}</select></label>
                <label class="dragon-field"><span>内存保留行数</span><input class="dragon-input" type="number" name="log_retention_lines" min="50" max="5000" step="50" value="${state.retentionLines}"></label>
                <button class="dragon-btn dragon-btn-secondary" type="submit" data-logs-save ${state.saving ? 'disabled' : ''}>${renderIcon('save', 'dragon-btn-icon')}<span data-logs-save-label>${state.saving ? '保存中…' : '保存'}</span></button>
                <button class="dragon-btn dragon-btn-danger" type="button" data-logs-clear ${state.lines.length ? '' : 'disabled'}>${renderIcon('trash', 'dragon-btn-icon')}<span>清空日志</span></button>
            </form>
            <div class="dragon-tagging-log-window" data-logs-window tabindex="0" role="log" aria-live="off">${renderLogLines(state)}</div>
        </section>
    </div>`;
}

function renderLogLines(state) {
    if (!state.lines.length) return `<div class="dragon-tagging-log-empty"><span>${renderIcon('terminal')}</span><strong>暂无日志</strong></div>`;
    const visible = state.lines.slice(-LOG_DOM_WINDOW);
    const hidden = state.lines.length - visible.length;
    return `${hidden > 0 ? `<div class="dragon-tagging-log-truncated">较早 ${hidden} 行仍保留在内存中</div>` : ''}<ol>${visible.map(renderLogLine).join('')}</ol>`;
}

function renderLogLine(line) {
    return `<li data-level="${escapeAttribute(line.level || 'info')}" data-sequence="${Number(line.sequence || 0)}"><time>${escapeHtml(line.timestamp_text || '')}</time><span>${escapeHtml(levelLabel(line.level))}</span><code>${escapeHtml(line.message || '')}</code>${line.job_id ? `<small title="${escapeAttribute(line.job_id)}">${escapeHtml(line.job_id.slice(0, 8))}</small>` : ''}</li>`;
}

function handleClick(state, event) {
    const target = event.target.closest?.('[data-logs-back], [data-logs-clear]');
    if (!target) return;
    if (target.matches('[data-logs-back]')) return returnToTaggingWorkspace();
    if (target.matches('[data-logs-clear]')) return run(() => clearLogs(state));
}

function handleChange(state, event) {
    if (!event.target.matches('[data-logs-job-filter]')) return;
    state.jobFilter = event.target.value || '';
    run(() => reloadLogs(state));
}

function handleSubmit(state, event) {
    if (!event.target.matches('[data-logs-settings-form]')) return;
    event.preventDefault();
    const value = Number(event.target.elements.log_retention_lines?.value || 200);
    run(() => saveRetention(state, value));
}

async function saveRetention(state, value) {
    if (state.saving) return;
    state.saving = true;
    syncLogControls(state);
    try {
        const settings = await saveTaggingSettings(api, { log_retention_lines: value });
        state.settings = settings.settings || settings;
        state.retentionLines = Number(state.settings.log_retention_lines || value);
        state.notice = '日志保留行数已保存。';
        state.error = '';
        await reloadLogs(state);
    } catch (error) {
        state.error = error.message || '保存日志设置失败';
    } finally {
        state.saving = false;
        syncLogControls(state);
        syncFeedback(state);
    }
}

async function clearLogs(state) {
    if (!window.confirm('清空当前进程内的打标日志吗？')) return;
    try {
        const payload = await clearTaggingLogs(api);
        state.lines = payload.lines || [];
        state.buffered = Number(payload.buffered || 0);
        state.lastSequence = Number(payload.last_sequence || state.lastSequence);
        state.notice = '内存日志已清空。';
        state.error = '';
    } catch (error) {
        state.error = error.message || '清空日志失败';
    }
    renderLogWindow(state);
    syncLogControls(state);
    syncFeedback(state);
}

async function reloadLogs(state) {
    const requestId = ++state.requestId;
    try {
        const payload = await loadTaggingLogs(api, {
            after: 0,
            limit: state.retentionLines,
            jobId: state.jobFilter,
        });
        if (!state.active || requestId !== state.requestId) return;
        state.lines = payload.lines || [];
        state.lastSequence = Number(payload.last_sequence || 0);
        state.buffered = Number(payload.buffered || 0);
        state.retentionLines = Number(payload.retention_lines || state.retentionLines);
        renderLogWindow(state, { keepBottom: true });
        syncLogControls(state);
        syncFeedback(state);
    } catch (error) {
        if (state.active && requestId === state.requestId) {
            state.error = error.message || '读取日志失败';
            syncFeedback(state);
        }
    }
}

async function pollLogs(state) {
    const requestId = ++state.requestId;
    try {
        const payload = await loadTaggingLogs(api, {
            after: state.lastSequence,
            limit: state.retentionLines,
            jobId: state.jobFilter,
        });
        if (!state.active || requestId !== state.requestId) return;
        state.lastSequence = Number(payload.last_sequence || state.lastSequence);
        state.buffered = Number(payload.buffered || state.buffered);
        state.retentionLines = Number(payload.retention_lines || state.retentionLines);
        if (payload.lines?.length) {
            state.lines.push(...payload.lines);
            state.lines = state.lines.slice(-state.retentionLines);
            appendLogLines(state, payload.lines);
        } else {
            syncCounters(state);
        }
    } catch (error) {
        if (state.active) {
            const message = error.message || '刷新日志失败';
            if (state.error !== message) {
                state.error = message;
                syncFeedback(state);
            }
        }
    }
}

function schedulePoll(state) {
    if (!state.active || !state.logPoller) return;
    state.logPoller.start();
}

function clearPoll(state) {
    state.logPoller?.stop();
    state.pollTimer = null;
}

function renderLogWindow(state, { keepBottom = false } = {}) {
    const windowNode = state.root?.querySelector('[data-logs-window]');
    if (!windowNode) return;
    const wasNearBottom = keepBottom || windowNode.scrollHeight - windowNode.scrollTop - windowNode.clientHeight < 80;
    windowNode.innerHTML = renderLogLines(state);
    syncCounters(state);
    if (wasNearBottom) scrollLogToBottom(state);
}

function appendLogLines(state, lines) {
    const windowNode = state.root?.querySelector('[data-logs-window]');
    const list = windowNode?.querySelector('ol');
    if (!windowNode || !list) return renderLogWindow(state);
    const wasNearBottom = windowNode.scrollHeight - windowNode.scrollTop - windowNode.clientHeight < 80;
    list.insertAdjacentHTML('beforeend', lines.map(renderLogLine).join(''));
    let removedHeight = 0;
    while (list.children.length > LOG_DOM_WINDOW) {
        const first = list.firstElementChild;
        if (!first) break;
        removedHeight += Number(first.offsetHeight || 0);
        first.remove();
    }
    if (!wasNearBottom && removedHeight) windowNode.scrollTop = Math.max(0, windowNode.scrollTop - removedHeight);
    const hidden = Math.max(0, state.lines.length - list.children.length);
    let marker = windowNode.querySelector('.dragon-tagging-log-truncated');
    if (hidden > 0) {
        if (!marker) {
            windowNode.insertAdjacentHTML('afterbegin', '<div class="dragon-tagging-log-truncated"></div>');
            marker = windowNode.querySelector('.dragon-tagging-log-truncated');
        }
        if (marker) marker.textContent = `较早 ${hidden} 行仍保留在内存中`;
    } else marker?.remove();
    syncCounters(state);
    if (wasNearBottom) scrollLogToBottom(state);
}

function rerender(state, { keepBottom = false } = {}) {
    if (!state.root) return;
    const oldWindow = state.root.querySelector('[data-logs-window]');
    const scrollTop = oldWindow?.scrollTop || 0;
    const wasNearBottom = keepBottom || (oldWindow && oldWindow.scrollHeight - oldWindow.scrollTop - oldWindow.clientHeight < 80);
    state.root.innerHTML = renderPage(state);
    const nextWindow = state.root.querySelector('[data-logs-window]');
    if (nextWindow) nextWindow.scrollTop = wasNearBottom ? nextWindow.scrollHeight : scrollTop;
}

function syncCounters(state) {
    const buffered = state.root?.querySelector('[data-logs-buffered]');
    const retention = state.root?.querySelector('[data-logs-retention]');
    if (buffered) buffered.textContent = String(state.buffered);
    if (retention) retention.textContent = String(state.retentionLines);
}

function syncLogControls(state) {
    const form = state.root?.querySelector('[data-logs-settings-form]');
    if (!form) return;
    const input = form.elements.log_retention_lines;
    if (input && globalThis.document?.activeElement !== input) input.value = String(state.retentionLines);
    const save = form.querySelector('[data-logs-save]');
    if (save) {
        save.disabled = state.saving;
        const label = save.querySelector('[data-logs-save-label]');
        if (label) label.textContent = state.saving ? '保存中…' : '保存';
    }
    const clear = form.querySelector('[data-logs-clear]');
    if (clear) clear.disabled = !state.lines.length;
}

function syncFeedback(state) {
    const host = state.root?.querySelector('[data-logs-feedback]');
    if (host) host.innerHTML = feedback(state);
}

function scrollLogToBottom(state) {
    const windowNode = state.root?.querySelector('[data-logs-window]');
    if (windowNode) windowNode.scrollTop = windowNode.scrollHeight;
}

function jobOptions(state) {
    return state.jobs.map((job) => `<option value="${escapeAttribute(job.id)}" ${job.id === state.jobFilter ? 'selected' : ''}>${escapeHtml(job.created_at_text || job.id)} · ${Number(job.total || 0)} 张</option>`).join('');
}

function feedback(state) {
    return `${state.error ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="error" role="alert">${escapeHtml(state.error)}</div>` : ''}${state.notice ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="success" role="status">${escapeHtml(state.notice)}</div>` : ''}`;
}

function levelLabel(value) {
    return { debug: '调试', info: '信息', success: '成功', warning: '警告', error: '错误' }[value] || '信息';
}

function run(fn) {
    Promise.resolve().then(fn).catch((error) => console.error('[dragon-tagging-logs]', error));
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
