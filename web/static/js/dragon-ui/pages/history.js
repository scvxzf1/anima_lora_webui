/* Dragon training history controller: list filters, detail resources, and resume actions. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { copyText } from '../../shared/dom.js?v=dragon-ui-20260812v35';
import { renderHistoryMetricsChart, bindHistoryChart } from './history-chart.js?v=dragon-ui-20260816v2';
import {
    bindHistorySystemCharts,
    renderHistorySystemCharts,
} from './history-system-charts.js?v=dragon-ui-20260816v1';
import { bindHistoryLogViewer } from './history-log-viewer.js?v=dragon-ui-20260816v3';
import { bindHistorySampleDialog } from './history-sample-dialog.js?v=dragon-ui-20260819v2';
import {
    activateHistoryDetailTab,
    normalizeHistoryDetailTab,
} from './history-detail-tabs.js?v=dragon-ui-20260816v2';
import { scanForReveal } from '../animations.js?v=dragon-ui-20260816v67';
import { switchToClassicUI } from '../../shared/ui-mode.js?v=dragon-ui-20260814v43';
import {
    createHistoryCollectionWorkspace,
    renderHistoryCollectionWorkbench,
} from './history-collections.js?v=dragon-ui-20260816v6';
import { bindHistoryCollectionWorkbench } from './history-collections-controller.js?v=dragon-ui-20260816v4';
import {
    renderHistoryDetailError,
    renderHistoryDetailPage,
    renderHistoryPage,
    renderHistoryResults,
    renderHistoryStats,
    renderHistorySummary,
    taskDisplayName,
} from './history-view.js?v=dragon-ui-20260819v92';

const api = createApiClient();

export async function loadHistory(context = {}) {
    const taskId = context.taskId || null;
    if (taskId) return loadHistoryDetail(taskId, context.sub);

    const [payload, collectionSettings] = await Promise.all([
        safeApi('/api/training/history?limit=200&include_archived=1', '读取训练历史失败'),
        safeApi('/api/training/history/collections/settings', '读取历史分组失败'),
    ]);
    const state = {
        tasks: Array.isArray(payload.tasks) ? payload.tasks : [],
        error: payload.ok === false ? (payload.error || '读取训练历史失败') : '',
        filters: defaultHistoryFilters(),
        workspace: createHistoryCollectionWorkspace(collectionSettings),
        requestSequence: 0,
    };
    return {
        html: renderHistoryPage({
            ...state,
            resultsHtml: renderHistoryCollectionWorkbench(state.tasks, state.filters, state.workspace),
        }),
        onMount: (root) => bindHistoryList(root, state),
        onUnmount: () => { state.requestSequence += 1; },
    };
}

async function loadHistoryDetail(taskId, requestedTab) {
    const payload = await safeApi(
        `/api/training/history/${encodeURIComponent(taskId)}`,
        '加载训练记录失败',
    );
    if (payload.ok === false) {
        return {
            html: renderHistoryDetailError(taskId, payload.error || '加载训练记录失败'),
            onMount: bindHistoryNavigation,
        };
    }

    const task = payload.task || {};
    const [images, weights, resume] = await Promise.all([
        safeApi(`/api/preview/images?source=training&task_id=${encodeURIComponent(taskId)}&limit=8`, '读取训练样张失败'),
        safeApi(`/api/preview/weights?task_id=${encodeURIComponent(taskId)}`, '读取训练权重失败'),
        task.job === 'training'
            ? safeApi(`/api/training/history/${encodeURIComponent(taskId)}/resume-options`, '读取续训检查点失败')
            : Promise.resolve({ ok: true, checkpoints: [], message: '预处理任务不支持续训。' }),
    ]);
    const model = {
        taskId,
        activeTab: normalizeHistoryDetailTab(requestedTab),
        payload,
        images,
        weights,
        resume,
        lossChart: renderHistoryMetricsChart(Array.isArray(payload.metrics) ? payload.metrics : []),
        systemCharts: renderHistorySystemCharts(payload.system, payload.limits),
    };
    let cleanup = null;
    let mountedRoot = null;
    return {
        html: renderHistoryDetailPage(model),
        onMount: (root) => { mountedRoot = root; cleanup = bindHistoryDetail(root, model); },
        onUnmount: () => { cleanup?.(); mountedRoot = null; },
        onRouteUpdate: (context) => updateHistoryDetailRoute(context, taskId),
    };

    function updateHistoryDetailRoute(context, mountedTaskId) {
        if (context.taskId !== mountedTaskId || !mountedRoot) return false;
        model.activeTab = activateHistoryDetailTab(mountedRoot, context.sub);
        scrollHistoryDetailContent(mountedRoot);
        return true;
    }
}

function bindHistoryList(root, state) {
    root.querySelectorAll('[data-history-filter]').forEach((control) => {
        const key = control.dataset.historyFilter;
        if (!key || !(key in state.filters)) return;
        const eventName = control.matches('input[type="search"]') ? 'input' : 'change';
        control.addEventListener(eventName, () => {
            state.filters[key] = control.value || historyFilterDefault(key);
            if (key !== 'archived' && key !== 'sort' && expandArchiveScopeForMatches(state)) {
                syncHistoryFilterControls(root, state);
            }
            updateHistoryStats(root, state);
            updateHistoryResults(root, state);
        });
    });
    root.querySelectorAll('[data-history-stat]').forEach((button) => {
        button.addEventListener('click', () => {
            applyHistoryStatFilter(state, button.dataset.historyStat || 'all');
            syncHistoryFilterControls(root, state);
            updateHistoryStats(root, state);
            updateHistoryResults(root, state);
        });
    });
    root.querySelector('[data-history-refresh]')?.addEventListener('click', () => refreshHistory(root, state));
    root.querySelectorAll('[data-history-classic]').forEach((button) => {
        button.addEventListener('click', () => switchToClassicUI('training'));
    });
    bindHistoryCollectionWorkbench(root, state, { renderResults: updateHistoryResults, setStatus: setHistoryStatus });
}

function updateHistoryResults(root, state) {
    const model = renderHistoryResults(state.tasks, state.filters);
    const results = root.querySelector('[data-history-results]');
    const count = root.querySelector('[data-history-count]');
    const summary = root.querySelector('[data-history-summary]');
    if (results) {
        results.innerHTML = renderHistoryCollectionWorkbench(
            state.tasks,
            state.filters,
            state.workspace,
        );
        // Filter updates replace the result DOM after the page observer has run.
        // Make replacement groups visible immediately instead of leaving them at
        // the default .dragon-reveal opacity: 0 while waiting for an observer tick.
        results.querySelectorAll('.dragon-reveal').forEach((element) => {
            element.classList.add('dragon-in-view');
        });
    }
    if (count) count.textContent = `${model.visibleCount} / ${state.tasks.length} 条记录`;
    if (summary) summary.textContent = renderHistorySummary(state.tasks, state.filters);
    scanForReveal();
}

async function refreshHistory(root, state) {
    const sequence = ++state.requestSequence;
    const button = root.querySelector('[data-history-refresh]');
    if (button) button.disabled = true;
    setHistoryStatus(root, '正在刷新训练历史…', 'info');
    const payload = await safeApi('/api/training/history?limit=200&include_archived=1', '刷新训练历史失败');
    if (sequence !== state.requestSequence) return;
    if (button) button.disabled = false;
    if (payload.ok === false) {
        setHistoryStatus(root, `${payload.error || '刷新训练历史失败'}。请检查 WebUI 服务后重试。`, 'error');
        return;
    }
    state.tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
    state.error = '';
    updateHistoryStats(root, state);
    updateHistoryResults(root, state);
    setHistoryStatus(root, `已刷新，共 ${state.tasks.length} 条训练记录。`, 'success');
}

const HISTORY_FILTER_DEFAULTS = Object.freeze({
    search: '', kind: 'all', status: 'all', archived: 'active', source: 'all',
    trainingVariant: 'all', preprocessPrecision: 'all', blockSwapPrecision: 'all',
    baseCompute: 'all', precisionPreference: 'all', sort: 'newest',
});

function defaultHistoryFilters() { return { ...HISTORY_FILTER_DEFAULTS }; }
function historyFilterDefault(key) { return HISTORY_FILTER_DEFAULTS[key] ?? 'all'; }

function syncHistoryFilterControls(root, state) {
    root.querySelectorAll('[data-history-filter]').forEach((control) => {
        const key = control.dataset.historyFilter;
        if (key in state.filters) control.value = state.filters[key];
    });
}

function updateHistoryStats(root, state) {
    const stats = root.querySelector('.dragon-history-stats');
    if (stats) stats.innerHTML = renderHistoryStats(state.tasks, state.filters);
    root.querySelectorAll('[data-history-stat]').forEach((button) => {
        button.onclick = () => {
            applyHistoryStatFilter(state, button.dataset.historyStat || 'all');
            syncHistoryFilterControls(root, state);
            updateHistoryStats(root, state);
            updateHistoryResults(root, state);
        };
    });
}

function expandArchiveScopeForMatches(state) {
    if (state.filters.archived !== 'active') return false;
    if (renderHistoryResults(state.tasks, state.filters).visibleCount > 0) return false;
    const expanded = { ...state.filters, archived: 'all' };
    if (renderHistoryResults(state.tasks, expanded).visibleCount === 0) return false;
    state.filters.archived = 'all';
    return true;
}

function applyHistoryStatFilter(state, stat) {
    state.filters = { ...HISTORY_FILTER_DEFAULTS, sort: state.filters.sort || 'newest' };
    if (stat === 'training' || stat === 'preprocess') state.filters.kind = stat;
    else if (stat === 'error') state.filters.status = 'error';
    else if (stat === 'archived') state.filters.archived = 'archived';
    else if (stat === 'queue') state.filters.source = 'queue';
}

function bindHistoryNavigation(root) {
    root.querySelector('[data-history-back]')?.addEventListener('click', () => {
        window.location.hash = '#history';
    });
    root.querySelector('[data-history-detail-refresh]')?.addEventListener('click', () => {
        window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    });
}

function bindHistoryDetail(root, model) {
    bindHistoryNavigation(root);
    activateHistoryDetailTab(root, model.activeTab);
    const unbindChart = bindHistoryChart(root, Array.isArray(model.payload.metrics) ? model.payload.metrics : []);
    const unbindSystemCharts = bindHistorySystemCharts(root, model.payload.system);
    const unbindLogViewer = bindHistoryLogViewer(root, model.payload.logs, {
        total: model.payload.limits?.logs_total,
    });
    root.querySelectorAll('[data-history-resume-mode]').forEach((button) => {
        button.addEventListener('click', () => resumeHistoryTask(root, model, button.dataset.historyResumeMode));
    });
    root.querySelectorAll('[data-history-weight-copy]').forEach((button) => {
        button.addEventListener('click', () => copyHistoryWeightPath(button));
    });
    const unbindSampleDialog = bindHistorySampleDialog(root, Array.isArray(model.images.images) ? model.images.images : []);
    root.querySelector('[data-history-resume-shortcut]')?.addEventListener('click', (event) => {
        if (model.activeTab !== 'overview') return;
        event.preventDefault();
        scrollToHistoryElement(root.querySelector('.dragon-history-resume-panel'));
    });
    return () => {
        unbindChart?.();
        unbindSystemCharts?.();
        unbindLogViewer?.();
        unbindSampleDialog?.();
    };
}

async function copyHistoryWeightPath(button) {
    const path = String(button.dataset.historyWeightCopy || '').trim();
    const label = button.querySelector('span');
    const originalLabel = label?.textContent || '复制路径';
    const originalTitle = button.title;
    button.disabled = true;
    try {
        if (!path) throw new Error('missing weight path');
        await copyText(path);
        if (label) label.textContent = '已复制';
        button.title = path;
    } catch {
        if (label) label.textContent = '复制失败';
        button.title = path ? '复制失败，请检查浏览器剪贴板权限' : '这个权重没有可复制的本地路径';
    } finally {
        window.setTimeout(() => {
            if (!button.isConnected) return;
            if (label) label.textContent = originalLabel;
            button.title = originalTitle;
            button.disabled = false;
        }, 1400);
    }
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

function setHistoryStatus(root, message, tone) {
    const status = root.querySelector('[data-history-status-region]');
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
