/* Log search, export and local-view controls for Dragon live training. */

import { downloadText } from '../../shared/download.js?v=dragon-ui-20260812v35';
import { logRecordText, visibleLogs } from './live-training-state.js?v=dragon-ui-20260825v46';
import { renderLogs } from './live-training-view.js?v=dragon-ui-20260825v46';

export function bindLiveLogTools(root, model, renderState) {
    const search = root.querySelector('[data-live-log-search]');
    search?.addEventListener('input', () => {
        model.logQuery = search.value;
        renderLogView(root, model, { forceBottom: false });
    });
    root.querySelector('[data-live-log-action="copy"]')?.addEventListener('click', async () => {
        const text = logText(visibleLogs(model));
        if (!text) return showLogFeedback(root, '当前视图没有可复制的日志。', 'warning');
        try {
            await copyText(text);
            showLogFeedback(root, `已复制 ${visibleLogs(model).length} 条日志。`, 'success');
        } catch {
            showLogFeedback(root, '复制失败，请选择日志文本后手动复制。', 'error');
        }
    });
    root.querySelector('[data-live-log-action="download"]')?.addEventListener('click', () => {
        const records = visibleLogs(model);
        if (!records.length) return showLogFeedback(root, '当前视图没有可下载的日志。', 'warning');
        downloadText(`${logText(records)}\n`, logFilename());
        showLogFeedback(root, `已下载 ${records.length} 条日志。`, 'success');
    });
    root.querySelector('[data-live-log-action="pause"]')?.addEventListener('click', () => {
        model.autoScroll = !model.autoScroll;
        renderState({ chart: false, logs: false });
        showLogFeedback(root, model.autoScroll ? '已恢复自动滚动。' : '已暂停自动滚动；实时日志仍会继续接收。', 'info');
        if (model.autoScroll) scrollLogToBottom(root);
    });
    root.querySelector('[data-live-log-action="clear"]')?.addEventListener('click', () => {
        model.logClearBeforeId = maximumLogId(model.logs);
        renderLogView(root, model, { forceBottom: false });
        showLogFeedback(root, '已清空当前前端视图；后端日志未删除，新日志仍会继续显示。', 'info');
    });
}

export function renderLogView(root, model, options = {}) {
    const container = root.querySelector('[data-live-log]');
    if (!container) return;
    const records = visibleLogs(model);
    container.innerHTML = renderLogs(records, {
        cleared: Number(model.logClearBeforeId || 0) > 0,
        filtering: Boolean(String(model.logQuery || '').trim()),
    });
    updateLogSummary(root, model, records.length);
    if (options.forceBottom || model.autoScroll) scrollLogToBottom(root);
}

export function updateLogSummary(root, model, visibleCount = visibleLogs(model).length) {
    const total = Number(model.logClearBeforeId || 0) > 0
        ? model.logs.filter((record) => Number(record.id || 0) > Number(model.logClearBeforeId || 0)).length
        : model.logs.length;
    const filtered = Boolean(String(model.logQuery || '').trim());
    const text = filtered ? `${visibleCount} / ${total} 条` : `${total} 条`;
    setText(root, '[data-live-log-count]', text);
    setText(root, '[data-live-log-visible-count]', `${visibleCount} 条可见`);
    const pause = root.querySelector('[data-live-log-action="pause"]');
    if (pause) {
        pause.setAttribute('aria-pressed', String(model.autoScroll));
        pause.dataset.active = String(model.autoScroll);
        const label = pause.querySelector('span');
        if (label) label.textContent = '自动滚屏';
        pause.title = model.autoScroll ? '关闭自动滚屏' : '开启自动滚屏';
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

function scrollLogToBottom(root) {
    const container = root.querySelector('[data-live-log]');
    if (container) container.scrollTop = container.scrollHeight;
}

function showLogFeedback(root, message, tone) {
    const feedback = root.querySelector('[data-live-log-feedback]');
    if (!feedback) return;
    feedback.textContent = message;
    feedback.dataset.tone = tone;
}

function setText(root, selector, value) {
    const node = root.querySelector(selector);
    if (node) node.textContent = value;
}
