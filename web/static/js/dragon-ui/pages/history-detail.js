/* Dragon training history detail controller and resource lifecycle. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { copyText } from '../../shared/dom.js?v=dragon-ui-20260812v35';
import { createHistoryMetricsController } from './history-metrics-controller.js?v=dragon-ui-20260826v1';
import { createHistoryLogController } from './history-log-controller.js?v=dragon-ui-20260826v1';
import { createHistorySampleController } from './history-sample-controller.js?v=dragon-ui-20260826v1';
import { activateHistoryDetailTab, normalizeHistoryDetailTab } from './history-detail-tabs.js?v=dragon-ui-20260816v2';
import { resolveHistoryReturnNavigation } from '../history-return-navigation.js?v=dragon-ui-20260825v1';
import { renderHistoryDetailError, renderHistoryDetailPage } from './history-view.js?v=dragon-ui-20260826v99';
import { taskDisplayName } from './history-model.js?v=dragon-ui-20260828v3';

const api = createApiClient();
const HISTORY_LOG_PAGE_SIZE = 360;

export async function loadHistoryDetail(taskId, requestedTab) {
    const returnNavigation = resolveHistoryReturnNavigation(taskId);
    const activeTab = normalizeHistoryDetailTab(requestedTab);
    const imagesPromise = safeApi(
        `/api/preview/images?source=training&task_id=${encodeURIComponent(taskId)}&limit=8`,
        '读取训练样张失败',
    );
    const weightsPromise = safeApi(
        `/api/preview/weights?task_id=${encodeURIComponent(taskId)}`,
        '读取训练权重失败',
    );
    const payload = await safeApi(
        `/api/training/history/${encodeURIComponent(taskId)}?include_logs=0`,
        '加载训练记录失败',
    );
    if (payload.ok === false) {
        return {
            html: renderHistoryDetailError(taskId, payload.error || '加载训练记录失败', returnNavigation),
            onMount: (root) => bindHistoryNavigation(root, returnNavigation),
        };
    }

    const task = payload.task || {};
    const [images, weights, resume, logPage] = await Promise.all([
        imagesPromise,
        weightsPromise,
        task.job === 'training'
            ? safeApi(`/api/training/history/${encodeURIComponent(taskId)}/resume-options`, '读取续训检查点失败')
            : Promise.resolve({ ok: true, checkpoints: [], message: '预处理任务不支持续训。' }),
        activeTab === 'logs' ? loadHistoryLogPage(taskId) : Promise.resolve(null),
    ]);
    applyHistoryLogPage(payload, logPage);
    const model = {
        taskId,
        activeTab,
        payload,
        images,
        weights,
        resume,
        loadLogRange: (offset, limit) => loadHistoryLogPage(taskId, offset, limit),
        searchLog: (query, cursor, direction) => loadHistoryLogMatch(taskId, query, cursor, direction),
        ensureLogs: async () => {
            if (model.logsLoaded) return;
            applyHistoryLogPage(payload, await loadHistoryLogPage(taskId));
            model.logsLoaded = true;
        },
        logsLoaded: Boolean(logPage),
        lossChart: '',
        systemCharts: '',
        returnNavigation,
    };
    let detailController = null;
    let mountedRoot = null;
    return {
        html: renderHistoryDetailPage(model),
        onMount: (root) => { mountedRoot = root; detailController = bindHistoryDetail(root, model); },
        onUnmount: () => { detailController?.dispose(); mountedRoot = null; },
        onRouteUpdate: (context) => updateHistoryDetailRoute(context, taskId),
    };

    function updateHistoryDetailRoute(context, mountedTaskId) {
        if (context.taskId !== mountedTaskId || !mountedRoot) return false;
        model.activeTab = activateHistoryDetailTab(mountedRoot, context.sub);
        detailController?.activateTab(model.activeTab);
        scrollHistoryDetailContent(mountedRoot);
        return true;
    }
}

function applyHistoryLogPage(payload, logPage) {
    const logs = Array.isArray(logPage?.logs) ? logPage.logs : [];
    payload.logs = logs;
    payload.limits = {
        ...(payload.limits || {}),
        logs_total: Number(logPage?.total) || Number(payload.limits?.logs_total) || 0,
        logs_returned: logs.length,
        logs_truncated: Boolean(logPage?.has_more_before || logPage?.has_more_after),
        logs_offset: Number(logPage?.offset) || 0,
        logs_paged: true,
    };
}

async function loadHistoryLogPage(taskId, offset = null, limit = HISTORY_LOG_PAGE_SIZE) {
    const params = new URLSearchParams({ limit: String(limit) });
    if (offset != null) params.set('offset', String(offset));
    const payload = await safeApi(
        `/api/training/history/${encodeURIComponent(taskId)}/logs?${params.toString()}`,
        '读取历史日志失败',
    );
    if (payload.ok === false) throw new Error(payload.error || '读取历史日志失败');
    return payload;
}

async function loadHistoryLogMatch(taskId, query, cursor, direction) {
    const params = new URLSearchParams({
        query: String(query || ''),
        cursor: String(cursor || 0),
        direction: String(direction || 'forward'),
    });
    const payload = await safeApi(
        `/api/training/history/${encodeURIComponent(taskId)}/logs/search?${params.toString()}`,
        '搜索历史日志失败',
    );
    if (payload.ok === false) throw new Error(payload.error || '搜索历史日志失败');
    return payload;
}

function bindHistoryNavigation(root, returnNavigation = null) {
    root.querySelector('[data-history-back]')?.addEventListener('click', () => {
        window.location.hash = returnNavigation?.hash || '#history';
    });
    root.querySelector('[data-history-detail-refresh]')?.addEventListener('click', () => {
        window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    });
}

function bindHistoryDetail(root, model) {
    bindHistoryNavigation(root, model.returnNavigation);
    activateHistoryDetailTab(root, model.activeTab);
    const logController = createHistoryLogController(root, model);
    const sampleController = createHistorySampleController(root, Array.isArray(model.images.images) ? model.images.images : []);
    const metricsController = createHistoryMetricsController(root, model);
    logController.activateTab(model.activeTab);
    sampleController.activateTab(model.activeTab);
    metricsController.activateTab(model.activeTab);
    root.querySelectorAll('[data-history-resume-mode]').forEach((button) => {
        button.addEventListener('click', () => resumeHistoryTask(root, model, button.dataset.historyResumeMode));
    });
    root.querySelectorAll('[data-history-weight-copy]').forEach((button) => {
        button.addEventListener('click', () => copyHistoryWeightPath(button));
    });
    root.querySelectorAll('[data-history-path-copy]').forEach((button) => {
        button.addEventListener('click', () => copyHistoryTaskPath(button));
    });
    root.querySelector('[data-history-resume-shortcut]')?.addEventListener('click', (event) => {
        if (model.activeTab !== 'overview') return;
        event.preventDefault();
        scrollToHistoryElement(root.querySelector('.dragon-history-resume-panel'));
    });
    return {
        activateTab(tab) {
            logController.activateTab(tab);
            sampleController.activateTab(tab);
            metricsController.activateTab(tab);
        },
        dispose() {
            metricsController.dispose();
            logController.dispose();
            sampleController.dispose();
        },
    };
}

async function copyHistoryWeightPath(button) {
    return copyHistoryPath(button, String(button.dataset.historyWeightCopy || '').trim());
}

async function copyHistoryTaskPath(button) {
    return copyHistoryPath(button, String(button.dataset.historyPathCopy || '').trim());
}

async function copyHistoryPath(button, path) {
    const label = button.querySelector('span');
    const originalLabel = label?.textContent || '复制路径';
    const originalTitle = button.title;
    button.disabled = true;
    try {
        if (!path) throw new Error('missing path');
        await copyText(path);
        if (label) label.textContent = '已复制';
        button.title = path;
        showHistoryToast('路径已复制', 'success');
        globalThis.navigator?.vibrate?.(18);
    } catch {
        if (label) label.textContent = '复制失败';
        button.title = path ? '复制失败，请检查浏览器剪贴板权限' : '这个权重没有可复制的本地路径';
        showHistoryToast('复制失败', 'error');
    } finally {
        window.setTimeout(() => {
            if (!button.isConnected) return;
            if (label) label.textContent = originalLabel;
            button.title = originalTitle;
            button.disabled = false;
        }, 1400);
    }
}

function showHistoryToast(message, tone = 'success') {
    document.querySelector('[data-dragon-history-toast]')?.remove();
    const toast = document.createElement('div');
    toast.className = 'dragon-history-toast';
    toast.dataset.dragonHistoryToast = '';
    toast.dataset.tone = tone;
    toast.setAttribute('role', 'status');
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('visible'));
    window.setTimeout(() => {
        toast.classList.remove('visible');
        window.setTimeout(() => toast.remove(), 180);
    }, 1500);
}

function scrollHistoryDetailContent(root) {
    scrollToHistoryElement(root?.querySelector('[data-history-detail-content]'));
}

function scrollToHistoryElement(element) {
    if (!element || typeof window === 'undefined') return;
    const navHeight = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--dragon-nav-height')) || 44;
    const top = Math.max(0, window.scrollY + element.getBoundingClientRect().top - navHeight - 58);
    window.scrollTo({ top, behavior: 'auto' });
}

async function resumeHistoryTask(root, model, mode) {
    const select = root.querySelector('[data-history-resume-checkpoint]');
    const checkpoint = String(select?.value || '').trim();
    if (!checkpoint) {
        setResumeStatus(root, '请先选择一个可用的训练状态目录。', 'error');
        select?.focus();
        return;
    }
    const queueMode = mode === 'queue';
    const action = queueMode ? '加入续训队列' : '立即继续训练';
    const confirmed = window.confirm(`确认${action}“${taskDisplayName(model.payload.task)}”吗？\n\n将使用历史配置快照和所选训练状态目录创建一条新任务记录。`);
    if (!confirmed) return;

    setResumeBusy(root, true);
    setResumeStatus(root, queueMode ? '正在加入续训队列…' : '正在启动续训…', 'info');
    const endpoint = queueMode ? '/api/training/queue/resume' : '/api/training/resume';
    const payload = await safeApi(endpoint, queueMode ? '加入续训队列失败' : '续训启动失败', {
        method: 'POST',
        body: JSON.stringify({ task_id: model.taskId, checkpoint }),
    });
    if (payload.ok === false) {
        setResumeBusy(root, false);
        setResumeStatus(root, `${payload.error || '续训操作失败'}。请检查任务状态与检查点后重试。`, 'error');
        return;
    }
    setResumeStatus(root, payload.message || (queueMode ? '续训任务已加入队列。' : '续训已启动。'), 'success');
    window.location.hash = queueMode ? '#queue' : '#live-training';
}

function setResumeBusy(root, busy) {
    root.querySelectorAll('[data-history-resume-mode]').forEach((button) => { button.disabled = busy; });
    const select = root.querySelector('[data-history-resume-checkpoint]');
    if (select) select.disabled = busy;
}

function setResumeStatus(root, message, tone) {
    const status = root.querySelector('[data-history-resume-status]');
    if (!status) return;
    status.textContent = message;
    status.dataset.tone = tone;
    status.classList.toggle('dragon-config-feedback-visible', Boolean(message));
}

async function safeApi(url, fallback, options) {
    try {
        const payload = await api(url, options);
        if (payload?.ok === false) return { ...payload, error: payload.error || fallback };
        return payload || { ok: true };
    } catch (error) {
        return { ok: false, error: error.message || fallback };
    }
}
