/**
 * Training history list/manager helpers.
 * Moved out of anima-app mechanical chunks.
 */
import { renderContinueTrainingSource } from '../anima-app/helpers/training-source-bridge.js?v=module-bootstrap-20260711-ir2';
import { historyManagerBaseFilteredTasks, historyManagerVisibleTasks, historyTaskIds, historyTaskIsArchived, renderHistoryBulkBar, renderHistoryCollectionsWorkbench, renderHistoryManagerStats, selectedHistoryTasks, syncHistorySelectionWithTasks } from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260711-ir2';
import { createHistoryTaskItem, renderHistoryDetailDialog } from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260711-ir2';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { appendLog } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    renderPreviewTaskSelect,
    setPreviewStatus,
} from '../anima-app/helpers/preview-view-bridge.js?v=module-bootstrap-20260711-ir2';
import { api } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir2';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir2';

const historyState = getHistoryState();
const trainingState = getTrainingState();
const HISTORY_REFRESH_BUTTON_LABELS = Object.freeze({
    'btn-refresh-history': '刷新任务列表',
    'btn-history-manager-refresh': '刷新',
});
let historyListLoadPromise = null;
let historyRefreshFeedbackTimer = null;

function setHistoryRefreshButtonState(state = 'idle') {
    if (historyRefreshFeedbackTimer) {
        clearTimeout(historyRefreshFeedbackTimer);
        historyRefreshFeedbackTimer = null;
    }
    const labels = {
        idle: HISTORY_REFRESH_BUTTON_LABELS,
        pending: {
            'btn-refresh-history': '刷新中...',
            'btn-history-manager-refresh': '刷新中...',
        },
        ok: {
            'btn-refresh-history': '已刷新',
            'btn-history-manager-refresh': '已刷新',
        },
        error: {
            'btn-refresh-history': '刷新失败',
            'btn-history-manager-refresh': '失败',
        },
    }[state] || HISTORY_REFRESH_BUTTON_LABELS;
    const pending = state === 'pending';
    for (const [id, defaultLabel] of Object.entries(HISTORY_REFRESH_BUTTON_LABELS)) {
        const button = document.getElementById(id);
        if (!button) continue;
        button.disabled = pending;
        button.setAttribute('aria-busy', pending ? 'true' : 'false');
        button.dataset.refreshState = state;
        button.textContent = labels[id] || defaultLabel;
    }
    if (!pending && state !== 'idle') {
        historyRefreshFeedbackTimer = setTimeout(() => {
            for (const [id, defaultLabel] of Object.entries(HISTORY_REFRESH_BUTTON_LABELS)) {
                const button = document.getElementById(id);
                if (!button) continue;
                button.disabled = false;
                button.setAttribute('aria-busy', 'false');
                button.dataset.refreshState = 'idle';
                button.textContent = defaultLabel;
            }
            historyRefreshFeedbackTimer = null;
        }, 1400);
    }
}

export async function loadTrainingHistoryList(options = {}) {
        if (location.protocol === 'file:') return;
        const announce = Boolean(options?.announce);
        if (historyListLoadPromise) {
            if (announce) setHistoryRefreshButtonState('pending');
            return historyListLoadPromise;
        }
        if (announce) setHistoryRefreshButtonState('pending');
        historyListLoadPromise = (async () => {
            let failed = false;
            try {
                const params = new URLSearchParams();
                params.set('include_archived', '1');
                const suffix = params.toString() ? `?${params.toString()}` : '';
                const payload = await api(`/api/training/history${suffix}`);
                historyState.historyTasks = payload.tasks || [];
                await loadHistoryCollectionSettings();
                renderTrainingHistoryList();
                renderHistoryManager();
                renderPreviewTaskSelect();
                renderContinueTrainingSource();
                setPreviewStatus('', '');
            } catch (e) {
                failed = true;
                const list = document.getElementById('task-history-list');
                if (list) list.textContent = '读取任务列表失败';
                const managerList = document.getElementById('history-manager-list');
                if (managerList) managerList.textContent = '读取历史任务失败';
                renderPreviewTaskSelect();
                setPreviewStatus('读取训练任务列表失败: ' + e.message, 'error');
            } finally {
                if (announce) setHistoryRefreshButtonState(failed ? 'error' : 'ok');
                historyListLoadPromise = null;
            }
        })();
        return await historyListLoadPromise;
    }

export async function loadHistoryCollectionSettings() {
        if (location.protocol === 'file:') return;
        try {
            const payload = await api('/api/training/history/collections/settings');
            if (payload.ok !== false) {
                historyState.historyCollectionSettings = normalizeHistoryCollectionSettings(payload);
            }
        } catch (e) {
            appendLog(`[状态] 读取历史集合设置失败: ${e.message}`);
        }
    }

export async function saveHistoryCollectionSettings(nextSettings) {
        historyState.historyCollectionSettings = normalizeHistoryCollectionSettings(nextSettings);
        if (location.protocol === 'file:') {
            renderHistoryManager();
            return historyState.historyCollectionSettings;
        }
        try {
            const payload = await api('/api/training/history/collections/settings', {
                method: 'PUT',
                body: JSON.stringify(historyState.historyCollectionSettings),
            });
            if (payload.ok === false) throw new Error(payload.error || '保存集合设置失败');
            historyState.historyCollectionSettings = normalizeHistoryCollectionSettings(payload);
            renderHistoryManager();
            return historyState.historyCollectionSettings;
        } catch (e) {
            appendLog(`[状态] 保存历史集合设置失败: ${e.message}`);
            renderHistoryManager();
            return historyState.historyCollectionSettings;
        }
    }

export function normalizeHistoryCollectionSettings(payload = {}) {
        return {
            collection_order: uniqueStringList(payload.collection_order),
            config_group_order: normalizeHistoryConfigGroupOrder(payload.config_group_order),
        };
    }

export function uniqueStringList(value) {
        const list = Array.isArray(value) ? value : [];
        const out = [];
        const seen = new Set();
        for (const item of list) {
            const text = String(item || '').trim();
            if (!text || seen.has(text)) continue;
            out.push(text);
            seen.add(text);
        }
        return out;
    }

export function normalizeHistoryConfigGroupOrder(value) {
        if (!value || typeof value !== 'object') return {};
        const out = {};
        for (const [key, order] of Object.entries(value)) {
            const cleanKey = String(key || '').trim();
            const cleanOrder = uniqueStringList(order);
            if (cleanKey && cleanOrder.length) out[cleanKey] = cleanOrder;
        }
        return out;
    }

export function renderTrainingHistoryList() {
        const list = document.getElementById('task-history-list');
        if (!list) return;
        list.innerHTML = '';
        const recentTasks = recentTrainingSidebarTasks();
        if (!recentTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'task-history-empty';
            empty.textContent = historyState.historyTasks.length
                ? '最近没有未归档训练任务；归档和预处理请到历史任务大界面查看。'
                : '暂无历史任务。下一次训练启动后会自动记录。';
            list.appendChild(empty);
            return;
        }
        for (const task of recentTasks) {
            list.appendChild(createHistoryTaskItem(task));
        }
    }

export function recentTrainingSidebarTasks() {
        return historyState.historyTasks
            .filter((task) => task.job === 'training' && !historyTaskIsArchived(task))
            .sort((a, b) => {
                const aTime = Number(a.started_at || a.updated_at || 0);
                const bTime = Number(b.started_at || b.updated_at || 0);
                return (bTime - aTime) || String(b.id || '').localeCompare(String(a.id || ''), 'zh-CN');
            })
            .slice(0, 6);
    }

export function renderHistoryManager() {
        const panel = document.getElementById('training-history-manager');
        if (!panel) return;
        syncHistorySelectionWithTasks();
        renderHistoryManagerStats();
        const status = document.getElementById('history-manager-status');
        const list = document.getElementById('history-manager-list');
        const tablePanel = list?.closest('.history-table-panel');
        const selectAll = document.getElementById('history-select-all');
        const mergeBtn = document.getElementById('btn-history-manager-merge');
        if (!list) return;
        const baseVisible = historyManagerBaseFilteredTasks();
        const visible = historyManagerVisibleTasks(baseVisible);
        historyState.historyCurrentVisibleTaskIds = historyTaskIds(visible);
        if (tablePanel) tablePanel.classList.add('collections-mode');
        if (status) {
            const archivedCount = historyState.historyTasks.filter(historyTaskIsArchived).length;
            status.textContent = [
                `共 ${historyState.historyTasks.length} 条记录`,
                `当前分组 ${visible.length} 条`,
                `筛选后 ${baseVisible.length} 条`,
                `归档 ${archivedCount} 条`,
                historyState.historyDropFeedback.message,
            ].filter(Boolean).join(' · ');
            status.dataset.feedbackTone = historyState.historyDropFeedback.tone || '';
        }
        list.innerHTML = '';
        if (!historyState.historyTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'history-manager-empty';
            empty.textContent = '暂无历史任务。';
            list.appendChild(empty);
        } else {
            renderHistoryManagerItems(list, baseVisible);
        }
        if (selectAll) {
            const visibleIds = historyState.historyCurrentVisibleTaskIds;
            const selectedVisible = visibleIds.filter((id) => historyState.selectedHistoryTaskIds.has(id)).length;
            selectAll.checked = visibleIds.length > 0 && selectedVisible === visibleIds.length;
            selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleIds.length;
        }
        renderHistoryBulkBar();
        if (mergeBtn) {
            mergeBtn.disabled = selectedHistoryTasks().filter((task) => task.job === 'training').length === 0;
        }
        renderHistoryDetailDialog();
    }

export function renderHistoryManagerItems(list, visible) {
        list.dataset.groupMode = 'collections';
        renderHistoryCollectionsWorkbench(list, visible);
    }

export function resetTrainingExpandedStateOnLeave() {
        if (trainingState.trainingViewMode === 'history') {
            renderHistoryManager();
        }
    }

