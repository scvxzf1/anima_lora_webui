/* Log search, export and local-view controls for Dragon live training. */

import { downloadText } from '../../shared/download.js?v=dragon-ui-20260812v35';
import { logRecordText, visibleLogs } from './live-training-state.js?v=dragon-ui-20260825v46';
import { renderLogs } from './live-training-view.js?v=dragon-ui-20260825v46';

export function createLiveLogBindings(root) {
    const query = (selector) => root.querySelector(selector);
    const actions = Object.fromEntries(['copy', 'download', 'pause', 'clear'].map((action) => [
        action,
        query(`[data-live-log-action="${action}"]`),
    ]));
    return {
        root,
        search: query('[data-live-log-search]'),
        container: query('[data-live-log]'),
        feedback: query('[data-live-log-feedback]'),
        count: query('[data-live-log-count]'),
        visibleCount: query('[data-live-log-visible-count]'),
        pause: actions.pause,
        pauseLabel: actions.pause?.querySelector('span') || null,
        actions,
    };
}

export function bindLiveLogTools(bindings, model, renderState) {
    const cleanups = [];
    const listen = (node, type, handler) => {
        if (!node) return;
        node.addEventListener(type, handler);
        cleanups.push(() => node.removeEventListener(type, handler));
    };
    let searchTimer = null;
    listen(bindings.search, 'input', () => {
        const search = bindings.search;
        model.logQuery = search.value;
        if (searchTimer) window.clearTimeout(searchTimer);
        searchTimer = window.setTimeout(() => {
            searchTimer = null;
            renderLogView(bindings, model, { forceBottom: false });
        }, 100);
    });
    listen(bindings.actions.copy, 'click', async () => {
        const text = logText(visibleLogs(model));
        if (!text) return showLogFeedback(bindings, '当前视图没有可复制的日志。', 'warning');
        try {
            await copyText(text);
            showLogFeedback(bindings, `已复制 ${visibleLogs(model).length} 条日志。`, 'success');
        } catch {
            showLogFeedback(bindings, '复制失败，请选择日志文本后手动复制。', 'error');
        }
    });
    listen(bindings.actions.download, 'click', () => {
        const records = visibleLogs(model);
        if (!records.length) return showLogFeedback(bindings, '当前视图没有可下载的日志。', 'warning');
        downloadText(`${logText(records)}\n`, logFilename());
        showLogFeedback(bindings, `已下载 ${records.length} 条日志。`, 'success');
    });
    listen(bindings.actions.pause, 'click', () => {
        model.autoScroll = !model.autoScroll;
        renderState({ chart: false, logs: false });
        showLogFeedback(bindings, model.autoScroll ? '已恢复自动滚动。' : '已暂停自动滚动；实时日志仍会继续接收。', 'info');
        if (model.autoScroll) scrollLogToBottom(bindings);
    });
    listen(bindings.actions.clear, 'click', () => {
        model.logClearBeforeId = maximumLogId(model.logs);
        renderLogView(bindings, model, { forceBottom: false });
        showLogFeedback(bindings, '已清空当前前端视图；后端日志未删除，新日志仍会继续显示。', 'info');
    });
    return () => {
        if (searchTimer) window.clearTimeout(searchTimer);
        searchTimer = null;
        cleanups.forEach((cleanup) => cleanup());
    };
}

export function renderLogView(bindings, model, options = {}) {
    const { container } = bindings;
    if (!container) return;
    const records = visibleLogs(model);
    container.innerHTML = renderLogs(records, {
        cleared: Number(model.logClearBeforeId || 0) > 0,
        filtering: Boolean(String(model.logQuery || '').trim()),
    });
    updateLogSummary(bindings, model, records.length);
    if (options.forceBottom || model.autoScroll) scrollLogToBottom(bindings);
}

export function updateLogSummary(bindings, model, visibleCount = visibleLogs(model).length) {
    const total = Number(model.logClearBeforeId || 0) > 0
        ? model.logs.filter((record) => Number(record.id || 0) > Number(model.logClearBeforeId || 0)).length
        : model.logs.length;
    const filtered = Boolean(String(model.logQuery || '').trim());
    model.visibleLogCount = visibleCount;
    model.visibleLogTotal = total;
    const text = filtered ? `${visibleCount} / ${total} 条` : `${total} 条`;
    setText(bindings.count, text);
    setText(bindings.visibleCount, `${visibleCount} 条可见`);
    const { pause } = bindings;
    if (pause) {
        const active = String(model.autoScroll);
        if (pause.getAttribute('aria-pressed') !== active) pause.setAttribute('aria-pressed', active);
        if (pause.dataset.active !== active) pause.dataset.active = active;
        setText(bindings.pauseLabel, '自动滚屏');
        const title = model.autoScroll ? '关闭自动滚屏' : '开启自动滚屏';
        if (pause.title !== title) pause.title = title;
    }
}

function logText(records) {
    return records.map((record) => {
        const timestamp = formatLogTimestamp(record.ts);
        const kind = String(record.level || record.kind || '').trim();
        return [timestamp ? `[${timestamp}]` : '', kind ? `[${kind}]` : '', logRecordText(record)].filter(Boolean).join(' ');
    }).join('\n');
}

function formatLogTimestamp(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds <= 0) return '';
    return new Intl.DateTimeFormat(undefined, {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }).format(new Date(seconds * 1000));
}

function logFilename() {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    return `training-live-${stamp}.log`;
}

function maximumLogId(records) {
    return records.reduce((maximum, record) => Math.max(maximum, Number(record.id || 0)), 0);
}

async function copyText(text) {
    if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(text);
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    textarea.remove();
    if (!copied) throw new Error('copy failed');
}

function scrollLogToBottom(bindings) {
    const { container } = bindings;
    if (container) container.scrollTop = container.scrollHeight;
}

function showLogFeedback(bindings, message, tone) {
    const { feedback } = bindings;
    if (!feedback) return;
    feedback.textContent = message;
    feedback.dataset.tone = tone;
}

function setText(node, value) {
    if (node && node.textContent !== value) node.textContent = value;
}
