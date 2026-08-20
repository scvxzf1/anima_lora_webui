/* Dragon history collection workbench controller. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import {
    clearOrderedDropTargetIf,
    clearOrderedDropTargets,
    scheduleOrderedRowDropTarget,
    setOrderedDropTarget,
} from '../ordered-drag-target.js?v=dragon-ui-20260816v1';
import {
    HISTORY_COLLECTION_UNGROUPED,
    collectionKey,
    collectionValue,
    historyTaskCollection,
    normalizeCollectionSettings,
} from './history-collections.js?v=dragon-ui-20260816v6';

const api = createApiClient();
const HISTORY_DRAG_TARGET_OPTIONS = Object.freeze({
    frameKey: 'historyDragFrame',
    pendingKey: 'historyPendingDrop',
    targetKey: 'historyDropTarget',
    rowClassPrefix: 'dragon-history-collection-drop',
});

async function safeCollectionApi(url, fallback, options) {
    try {
        const payload = await api(url, options);
        if (payload?.ok === false) return { ...payload, error: payload.error || fallback };
        return payload || { ok: true };
    } catch (error) {
        return { ok: false, error: error.message || fallback };
    }
}

export function bindHistoryCollectionWorkbench(root, state, { renderResults, setStatus }) {
    const results = root.querySelector('[data-history-results]');
    if (!results) return;
    results.addEventListener('click', (event) => handleCollectionClick(event, root, state, { renderResults, setStatus }));
    results.addEventListener('change', (event) => handleCollectionChange(event, root, state, { renderResults, setStatus }));
    results.addEventListener('dragstart', (event) => handleCollectionDragStart(event, root, state));
    results.addEventListener('dragend', (event) => finishCollectionDrag(root, state, event.target));
    results.addEventListener('dragover', (event) => handleCollectionDragOver(event, state));
    results.addEventListener('dragleave', (event) => handleCollectionDragLeave(event, state));
    results.addEventListener('drop', (event) => handleCollectionDrop(event, root, state, { renderResults, setStatus }));
}

async function handleCollectionClick(event, root, state, deps) {
    const { renderResults } = deps;
    const button = event.target.closest('[data-history-collection-action]');
    if (!button) return;
    const action = button.dataset.historyCollectionAction;
    if (action === 'select') {
        state.workspace.activeKey = button.dataset.historyCollectionKey || HISTORY_COLLECTION_UNGROUPED;
        state.workspace.selectedTaskIds.clear();
    } else if (action === 'toggle-config') toggleConfigGroup(state, button.dataset.historyConfigKey);
    else if (action === 'refresh') window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    else if (action === 'new') await createCollection(root, state, deps);
    else if (action === 'assign') await assignSelectedCollection(root, state, deps);
    else if (action === 'clear-selected') await setTasksCollection(root, state, [...state.workspace.selectedTaskIds], '', true, deps);
    else if (action === 'rename') await renameCollection(root, state, button.dataset.historyCollectionValue || '', deps);
    else if (action === 'delete') await deleteCollection(root, state, button.dataset.historyCollectionValue || '', deps);
    renderResults(root, state);
}

function handleCollectionChange(event, root, state, { renderResults }) {
    const task = event.target.closest('[data-history-select-task]');
    if (task) updateTaskSelection(state, [task.dataset.historySelectTask], task.checked);
    const group = event.target.closest('[data-history-select-group]');
    if (group) updateTaskSelection(state, parseTaskIds(group.dataset.historySelectGroup), group.checked);
    renderResults(root, state);
}

function handleCollectionDragStart(event, root, state) {
    const collection = event.target.closest('[data-history-drag-collection]');
    if (collection) {
        const value = String(collection.dataset.historyDragCollection || '').trim();
        if (!value) return;
        state.workspace.draggedCollection = value;
        state.workspace.dragTaskIds = [];
        collection.classList.add('dragging');
        root.classList.add('dragon-dataset-dragging');
        event.dataTransfer?.setData('text/plain', `collection:${value}`);
        if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
        return;
    }
    const source = event.target.closest('[data-history-drag-task-ids]');
    if (!source) return;
    const sourceIds = parseTaskIds(source.dataset.historyDragTaskIds);
    const selected = sourceIds.some((id) => state.workspace.selectedTaskIds.has(id));
    state.workspace.draggedCollection = '';
    state.workspace.dragTaskIds = selected ? [...state.workspace.selectedTaskIds] : sourceIds;
    source.classList.add('dragging');
    root.classList.add('dragon-dataset-dragging');
    event.dataTransfer?.setData('text/plain', state.workspace.dragTaskIds.join(','));
    if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
}

function handleCollectionDragOver(event, state) {
    const row = event.target.closest('[data-history-drop-collection]');
    const dropzone = event.target.closest('[data-history-collection-dropzone]');
    if (state.workspace.draggedCollection) {
        if (dropzone) {
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            setOrderedDropTarget(state.workspace, dropzone, 'dropzone', '', HISTORY_DRAG_TARGET_OPTIONS);
            return;
        }
        const targetValue = String(row?.dataset.historyDropCollection || '').trim();
        if (!row || !targetValue || targetValue === state.workspace.draggedCollection) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        scheduleOrderedRowDropTarget(state.workspace, row, event.clientY, HISTORY_DRAG_TARGET_OPTIONS);
        return;
    }
    if (!row || !state.workspace.dragTaskIds.length) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
    scheduleOrderedRowDropTarget(state.workspace, row, event.clientY, HISTORY_DRAG_TARGET_OPTIONS);
}

function handleCollectionDragLeave(event, state) {
    const row = event.target.closest('[data-history-drop-collection]');
    const dropzone = event.target.closest('[data-history-collection-dropzone]');
    const node = dropzone || row;
    if (!node) return;
    if (event.relatedTarget && node.contains(event.relatedTarget)) return;
    clearOrderedDropTargetIf(state.workspace, node, HISTORY_DRAG_TARGET_OPTIONS);
}

async function handleCollectionDrop(event, root, state, deps) {
    const { renderResults } = deps;
    const row = event.target.closest('[data-history-drop-collection]');
    const dropzone = event.target.closest('[data-history-collection-dropzone]');
    if (!row && !dropzone) return;
    event.preventDefault();
    event.stopPropagation();
    if (state.workspace.draggedCollection) {
        const targetValue = String(row?.dataset.historyDropCollection || '').trim();
        const position = row ? historyRowDropPosition(row, event.clientY) : 'after';
        await reorderCollection(root, state, state.workspace.draggedCollection, targetValue, position, deps);
        finishCollectionDrag(root, state);
        renderResults(root, state);
        return;
    }
    if (!row) return;
    const transferIds = parseTaskIds(event.dataTransfer?.getData('text/plain'));
    const ids = state.workspace.dragTaskIds.length ? state.workspace.dragTaskIds : transferIds;
    const targetGroup = row.dataset.historyDropCollection || '';
    clearOrderedDropTargets(state.workspace, HISTORY_DRAG_TARGET_OPTIONS);
    await setTasksCollection(root, state, ids, targetGroup, true, deps);
    finishCollectionDrag(root, state);
    renderResults(root, state);
}

async function reorderCollection(root, state, source, target, position, deps) {
    const values = uniqueCollectionValues(state);
    const withoutSource = values.filter((value) => value !== source);
    let index = target ? withoutSource.indexOf(target) : withoutSource.length;
    if (index < 0) index = withoutSource.length;
    if (target && position === 'after') index += 1;
    withoutSource.splice(index, 0, source);
    return saveCollectionSettings(root, state, {
        ...state.workspace.settings,
        collection_order: withoutSource,
    }, deps);
}

function uniqueCollectionValues(state) {
    const values = [
        ...(state.workspace.settings.collection_order || []),
        ...state.tasks.map((task) => historyTaskCollection(task)),
    ];
    return [...new Set(values.map((value) => String(value || '').trim()).filter(Boolean))];
}

function historyRowDropPosition(row, clientY) {
    const rect = row.getBoundingClientRect();
    return clientY < rect.top + rect.height / 2 ? 'before' : 'after';
}

function finishCollectionDrag(root, state, source = null) {
    source?.closest?.('[data-history-drag-task-ids], [data-history-drag-collection]')?.classList.remove('dragging');
    root.querySelectorAll('.dragging').forEach((node) => node.classList.remove('dragging'));
    root.classList.remove('dragon-dataset-dragging');
    state.workspace.dragTaskIds = [];
    state.workspace.draggedCollection = '';
    clearOrderedDropTargets(state.workspace, HISTORY_DRAG_TARGET_OPTIONS);
}

function toggleConfigGroup(state, key) {
    if (!key) return;
    if (state.workspace.expandedConfigKeys.has(key)) state.workspace.expandedConfigKeys.delete(key);
    else state.workspace.expandedConfigKeys.add(key);
}

function updateTaskSelection(state, ids, selected) {
    for (const id of ids.filter(Boolean)) {
        if (selected) state.workspace.selectedTaskIds.add(id);
        else state.workspace.selectedTaskIds.delete(id);
    }
}

async function createCollection(root, state, deps) {
    const name = window.prompt('新建分组名称：', '');
    const clean = String(name || '').trim();
    if (!clean) return;
    const order = [...state.workspace.settings.collection_order.filter((item) => item !== clean), clean];
    if (await saveCollectionSettings(root, state, { ...state.workspace.settings, collection_order: order }, deps)) {
        state.workspace.activeKey = collectionKey(clean);
    }
}

async function assignSelectedCollection(root, state, deps) {
    const current = collectionValue(state.workspace.activeKey);
    const name = window.prompt('将已选任务设置到分组：', current);
    if (name === null) return;
    const clean = String(name).trim();
    if (!clean) return setTasksCollection(root, state, [...state.workspace.selectedTaskIds], '', true, deps);
    if (!state.workspace.settings.collection_order.includes(clean)) {
        const saved = await saveCollectionSettings(root, state, { ...state.workspace.settings, collection_order: [...state.workspace.settings.collection_order, clean] }, deps);
        if (!saved) return;
    }
    if (await setTasksCollection(root, state, [...state.workspace.selectedTaskIds], clean, true, deps)) {
        state.workspace.activeKey = collectionKey(clean);
    }
}

async function renameCollection(root, state, oldValue, deps) {
    const next = window.prompt(`重命名分组“${oldValue}”：`, oldValue);
    const clean = String(next || '').trim();
    if (!clean || clean === oldValue) return;
    const ids = state.tasks.filter((task) => historyTaskCollection(task) === oldValue).map((task) => task.id);
    if (ids.length && !(await setTasksCollection(root, state, ids, clean, false, deps))) return;
    const order = state.workspace.settings.collection_order.map((item) => item === oldValue ? clean : item);
    if (await saveCollectionSettings(root, state, { ...state.workspace.settings, collection_order: order }, deps)) {
        state.workspace.activeKey = collectionKey(clean);
    }
}

async function deleteCollection(root, state, value, deps) {
    const ids = state.tasks.filter((task) => historyTaskCollection(task) === value).map((task) => task.id);
    if (!window.confirm(`确认删除分组“${value}”吗？\n\n${ids.length} 条任务将回到未分类，历史记录和训练产物不会删除。`)) return;
    if (ids.length && !(await setTasksCollection(root, state, ids, '', false, deps))) return;
    const order = state.workspace.settings.collection_order.filter((item) => item !== value);
    if (await saveCollectionSettings(root, state, { ...state.workspace.settings, collection_order: order }, deps)) {
        state.workspace.activeKey = HISTORY_COLLECTION_UNGROUPED;
    }
}

async function setTasksCollection(root, state, ids, group, clearSelection = true, { setStatus }) {
    const taskIds = [...new Set(ids)].filter(Boolean);
    if (!taskIds.length) return false;
    setStatus(root, `正在整理 ${taskIds.length} 条历史任务…`, 'info');
    const payload = await safeCollectionApi('/api/training/history/batch', '设置历史分组失败', {
        method: 'POST', body: JSON.stringify({ action: 'set_group', task_ids: taskIds, group }),
    });
    if (payload.ok === false) {
        setStatus(root, payload.error || '设置历史分组失败', 'error');
        return false;
    }
    const idSet = new Set(taskIds);
    state.tasks.forEach((task) => { if (idSet.has(task.id)) task.group = group; });
    if (clearSelection) taskIds.forEach((id) => state.workspace.selectedTaskIds.delete(id));
    setStatus(root, `已将 ${taskIds.length} 条任务${group ? `归入“${group}”` : '移回未分类'}。`, 'success');
    return true;
}

async function saveCollectionSettings(root, state, settings, { setStatus }) {
    const payload = await safeCollectionApi('/api/training/history/collections/settings', '保存历史分组失败', {
        method: 'PUT', body: JSON.stringify(settings),
    });
    if (payload.ok === false) {
        setStatus(root, payload.error || '保存历史分组失败', 'error');
        return false;
    }
    state.workspace.settings = normalizeCollectionSettings(payload);
    setStatus(root, '历史分组已保存。', 'success');
    return true;
}

function parseTaskIds(value) {
    return String(value || '').split(',').map((item) => item.trim()).filter(Boolean);
}
