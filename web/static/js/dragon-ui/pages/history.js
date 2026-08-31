/* Dragon training history list controller. Detail-only code is loaded on demand. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { scanForReveal } from '../animations.js?v=dragon-ui-20260824v69';
import { switchToClassicUI } from '../../shared/ui-mode.js?v=dragon-ui-20260814v45';
import { createHistoryCollectionWorkspace, renderHistoryCollectionWorkbench } from './history-collections.js?v=dragon-ui-20260826v10';
import { bindHistoryCollectionWorkbench } from './history-collections-controller.js?v=dragon-ui-20260826v6';
import { createHistorySearchIndex, filterHistoryTasks } from './history-model.js?v=dragon-ui-20260828v3';
import { activeHistoryStat, renderHistoryPage, renderHistoryStats, renderHistorySummary } from './history-list-view.js?v=dragon-ui-20260828v3';

const api = createApiClient();

export async function loadHistory() {
    const [payload, collectionSettings] = await Promise.all([
        safeApi('/api/training/history?limit=200&include_archived=1', '读取训练历史失败'),
        safeApi('/api/training/history/collections/settings', '读取历史分组失败'),
    ]);
    const state = {
        tasks: Array.isArray(payload.tasks) ? payload.tasks : [],
        error: payload.ok === false ? (payload.error || '读取训练历史失败') : '',
        filters: defaultHistoryFilters(),
        searchIndex: createHistorySearchIndex(Array.isArray(payload.tasks) ? payload.tasks : []),
        workspace: createHistoryCollectionWorkspace(collectionSettings),
        requestSequence: 0,
    };
    const filtered = filterHistoryTasks(state.tasks, state.filters, state.searchIndex);
    let cleanup = null;
    return {
        html: renderHistoryPage({
            ...state,
            visibleCount: filtered.length,
            resultsHtml: renderHistoryCollectionWorkbench(state.tasks, state.filters, state.workspace, filtered),
        }),
        onMount: (root) => { cleanup = bindHistoryList(root, state); },
        onUnmount: () => { state.requestSequence += 1; cleanup?.(); },
    };
}

function bindHistoryList(root, state) {
    let filterTimer = null;
    const applyFilters = (key) => {
        filterTimer = null;
        if (!document.contains(root)) return;
        const filtered = filterHistoryTasks(state.tasks, state.filters, state.searchIndex);
        const resolved = resolveArchiveScopeForMatches(state, key, filtered);
        if (resolved !== filtered) {
            syncHistoryFilterControls(root, state);
        }
        updateHistoryStats(root, state);
        updateHistoryResults(root, state, resolved);
    };
    root.querySelectorAll('[data-history-filter]').forEach((control) => {
        const key = control.dataset.historyFilter;
        if (!key || !(key in state.filters)) return;
        const isSearch = control.matches('input[type="search"]');
        control.addEventListener(isSearch ? 'input' : 'change', () => {
            state.filters[key] = control.value || historyFilterDefault(key);
            if (filterTimer) window.clearTimeout(filterTimer);
            if (isSearch) filterTimer = window.setTimeout(() => applyFilters(key), 100);
            else applyFilters(key);
        });
    });
    const stats = root.querySelector('.dragon-history-stats');
    const handleStatClick = (event) => {
        const button = event.target.closest?.('[data-history-stat]');
        if (!button || !stats?.contains(button)) return;
        applyHistoryStatFilter(state, button.dataset.historyStat || 'all');
        syncHistoryFilterControls(root, state);
        updateHistoryStats(root, state);
        updateHistoryResults(root, state);
    };
    stats?.addEventListener('click', handleStatClick);
    root.querySelector('[data-history-refresh]')?.addEventListener('click', () => refreshHistory(root, state));
    root.querySelectorAll('[data-history-classic]').forEach((button) => {
        button.addEventListener('click', () => switchToClassicUI('training'));
    });
    bindHistoryCollectionWorkbench(root, state, { renderResults: updateHistoryResults, setStatus: setHistoryStatus });
    return () => {
        if (filterTimer) window.clearTimeout(filterTimer);
        filterTimer = null;
        stats?.removeEventListener('click', handleStatClick);
    };
}

function updateHistoryResults(root, state, filteredTasks = null) {
    const filtered = Array.isArray(filteredTasks)
        ? filteredTasks
        : filterHistoryTasks(state.tasks, state.filters, state.searchIndex);
    const results = root.querySelector('[data-history-results]');
    const count = root.querySelector('[data-history-count]');
    const summary = root.querySelector('[data-history-summary]');
    if (results) {
        results.innerHTML = renderHistoryCollectionWorkbench(state.tasks, state.filters, state.workspace, filtered);
        results.querySelectorAll('.dragon-reveal').forEach((element) => {
            element.classList.add('dragon-in-view');
        });
    }
    if (count) count.textContent = `${filtered.length} / ${state.tasks.length} 条记录`;
    if (summary) summary.textContent = renderHistorySummary(state.tasks, state.filters, filtered.length);
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
    state.searchIndex = createHistorySearchIndex(state.tasks);
    state.error = '';
    updateHistoryStats(root, state, { refreshCounts: true });
    updateHistoryResults(root, state);
    setHistoryStatus(root, `已刷新，共 ${state.tasks.length} 条训练记录。`, 'success');
}

const HISTORY_FILTER_DEFAULTS = Object.freeze({
    search: '', kind: 'all', status: 'all', archived: 'active', source: 'all',
    modelFamily: 'all', trainingVariant: 'all', preprocessPrecision: 'all', blockSwapPrecision: 'all',
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

function updateHistoryStats(root, state, { refreshCounts = false } = {}) {
    const stats = root.querySelector('.dragon-history-stats');
    if (stats && refreshCounts) stats.innerHTML = renderHistoryStats(state.tasks, state.filters);
    const active = activeHistoryStat(state.filters);
    root.querySelectorAll('[data-history-stat]').forEach((button) => {
        button.classList.toggle('active', button.dataset.historyStat === active);
    });
}

function resolveArchiveScopeForMatches(state, key, filtered) {
    if (key === 'archived' || key === 'sort') return filtered;
    if (state.filters.archived !== 'active' || filtered.length > 0) return filtered;
    const expanded = { ...state.filters, archived: 'all' };
    const expandedTasks = filterHistoryTasks(state.tasks, expanded, state.searchIndex);
    if (expandedTasks.length === 0) return filtered;
    state.filters.archived = 'all';
    return expandedTasks;
}

function applyHistoryStatFilter(state, stat) {
    state.filters = { ...HISTORY_FILTER_DEFAULTS, sort: state.filters.sort || 'newest' };
    if (stat === 'training' || stat === 'preprocess') state.filters.kind = stat;
    else if (stat === 'error') state.filters.status = 'error';
    else if (stat === 'archived') state.filters.archived = 'archived';
    else if (stat === 'queue') state.filters.source = 'queue';
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
