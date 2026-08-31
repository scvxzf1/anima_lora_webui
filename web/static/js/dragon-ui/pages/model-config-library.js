/* Grouped model-configuration library with ordered native drag/drop. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import {
    clearActiveOrderedDropTarget,
    clearOrderedDropTargetIf,
    clearOrderedDropTargets,
    scheduleOrderedRowDropTarget,
    setOrderedDropTarget,
} from '../ordered-drag-target.js?v=dragon-ui-20260816v1';
import { familyLabel, filterModelItems } from './model-config-state.js?v=dragon-ui-20260824-zimage-v1';

const ITEM_DROP_OPTIONS = Object.freeze({
    frameKey: 'modelItemDragFrame',
    pendingKey: 'modelItemPendingDrop',
    targetKey: 'modelItemDropTarget',
    rowClassPrefix: 'dragon-model-config-item-drop',
    groupTargetClass: 'dragon-model-config-group-drop-target',
});

const GROUP_DROP_OPTIONS = Object.freeze({
    frameKey: 'modelGroupDragFrame',
    pendingKey: 'modelGroupPendingDrop',
    targetKey: 'modelGroupDropTarget',
    rowClassPrefix: 'dragon-model-config-group-drop',
});

export function renderModelConfigList(state) {
    const visibleItems = filterModelItems(state.items, state.search);
    const visibleIds = new Set(visibleItems.map((item) => item.id));
    const itemsById = new Map(state.items.map((item) => [item.id, item]));
    const searching = Boolean(String(state.search || '').trim());
    const groups = state.groups
        .map((group) => ({
            ...group,
            items: group.item_ids.map((itemId) => itemsById.get(itemId)).filter((item) => item && visibleIds.has(item.id)),
        }))
        .filter((group) => group.items.length || !searching);
    if (!groups.length) return '<div class="dragon-model-config-list-empty">没有匹配的模型配置</div>';
    return groups.map((group, groupIndex) => renderGroup(group, groupIndex, state, searching)).join('');
}

function renderGroup(group, groupIndex, state, searching) {
    const groupDraggable = !searching && state.groups.length > 1;
    return `
        <section class="dragon-model-config-group" data-model-group="${escapeAttribute(group.id)}" data-model-drop-group="${escapeAttribute(group.id)}">
            <header class="dragon-model-config-group-header" data-model-group-row="${escapeAttribute(group.id)}">
                <span class="dragon-model-config-group-drag-handle" draggable="${groupDraggable}" title="拖动调整分组顺序" aria-hidden="true">${renderIcon('grip')}</span>
                <span class="dragon-model-config-group-title"><strong title="${escapeAttribute(group.label)}">${escapeHtml(group.label)}</strong><small>${group.items.length}</small></span>
                <span class="dragon-model-config-group-actions">
                    ${iconButton('chevronUp', '上移分组', 'data-model-group-move="-1"', group, groupIndex === 0 || searching)}
                    ${iconButton('chevronDown', '下移分组', 'data-model-group-move="1"', group, groupIndex === state.groups.length - 1 || searching)}
                    ${iconButton('edit', '重命名分组', 'data-model-group-action="rename"', group, false)}
                    ${iconButton('trash', '删除分组', 'data-model-group-action="delete"', group, state.groups.length <= 1, true)}
                </span>
            </header>
            <div class="dragon-model-config-group-items">
                ${group.items.map((item, itemIndex) => renderItem(item, itemIndex, group, state, searching)).join('')}
                <div class="dragon-model-config-dropzone" data-model-dropzone="${escapeAttribute(group.id)}" data-empty="${group.items.length ? 'false' : 'true'}"></div>
            </div>
        </section>
    `;
}

function renderItem(item, itemIndex, group, state, searching) {
    const path = item.pretrained_model_name_or_path || '基础模型路径未填写';
    return `
        <div class="dragon-model-config-row" data-model-row="${escapeAttribute(item.id)}" data-model-item-group="${escapeAttribute(group.id)}">
            <span class="dragon-model-config-drag-handle" draggable="${!searching}" title="拖动调整配置位置" aria-hidden="true">${renderIcon('grip')}</span>
            <button class="dragon-model-config-list-item" type="button" data-model-id="${escapeAttribute(item.id)}" data-active="${item.id === state.selectedId}" aria-pressed="${item.id === state.selectedId}" ${item.id === state.selectedId ? 'aria-current="true"' : ''}>
                <span><strong>${escapeHtml(item.name || '未命名配置')}</strong><small title="${escapeAttribute(path)}">${familyLabel(item.model_family)} · ${escapeHtml(path)}</small></span>
                ${item.id === state.defaultId ? '<em>默认</em>' : ''}
            </button>
            <span class="dragon-model-config-row-actions" aria-label="调整配置顺序">
                ${itemOrderButton(item, -1, itemIndex === 0 || searching, '上移')}
                ${itemOrderButton(item, 1, itemIndex === group.items.length - 1 || searching, '下移')}
            </span>
        </div>
    `;
}

function iconButton(icon, label, action, group, disabled, danger = false) {
    const className = `dragon-model-config-group-action${danger ? ' dragon-model-config-group-action-danger' : ''}`;
    return `<button class="${className}" type="button" ${action} data-group-id="${escapeAttribute(group.id)}" aria-label="${label} ${escapeAttribute(group.label)}" title="${label}" ${disabled ? 'disabled' : ''}>${renderIcon(icon)}</button>`;
}

function itemOrderButton(item, offset, disabled, label) {
    const icon = offset < 0 ? 'chevronUp' : 'chevronDown';
    return `<button class="dragon-icon-btn" type="button" data-model-item-move="${offset}" data-model-id="${escapeAttribute(item.id)}" aria-label="${label} ${escapeAttribute(item.name)}" title="${label}" ${disabled ? 'disabled' : ''}>${renderIcon(icon)}</button>`;
}

export function bindModelConfigDrag(root, state, { onPlaceItem, onPlaceGroup }) {
    bindDragRecovery(root, state);
    root.querySelectorAll('[data-model-row]').forEach((row) => bindItemRow(root, state, row, onPlaceItem));
    root.querySelectorAll('[data-model-dropzone]').forEach((dropzone) => bindItemDropzone(root, state, dropzone, onPlaceItem));
    root.querySelectorAll('[data-model-drop-group]').forEach((group) => bindGroupDrop(root, state, group, onPlaceItem, onPlaceGroup));
    root.querySelectorAll('[data-model-group-row]').forEach((row) => bindGroupRow(root, state, row));
}

function bindItemRow(root, state, row, onPlaceItem) {
    const handle = row.querySelector('.dragon-model-config-drag-handle[draggable="true"]');
    handle?.addEventListener('click', (event) => event.stopPropagation());
    handle?.addEventListener('dragstart', (event) => {
        finishDrag(root, state);
        state.draggedModelId = row.dataset.modelRow || '';
        state.modelSuppressClickUntil = performance.now() + 300;
        root.classList.add('dragon-model-config-item-dragging');
        row.classList.add('dragon-model-config-row-dragging');
        event.dataTransfer?.setData('text/plain', state.draggedModelId);
        if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
    });
    handle?.addEventListener('dragend', () => finishDrag(root, state));
    row.addEventListener('dragover', (event) => {
        if (!state.draggedModelId || row.dataset.modelRow === state.draggedModelId) return;
        event.preventDefault();
        event.stopPropagation();
        scheduleOrderedRowDropTarget(state, row, event.clientY, ITEM_DROP_OPTIONS);
    });
    row.addEventListener('drop', (event) => {
        if (!state.draggedModelId) return;
        event.preventDefault();
        event.stopPropagation();
        const rect = row.getBoundingClientRect();
        const position = event.clientY < rect.top + rect.height / 2 ? 'before' : 'after';
        const itemId = state.draggedModelId;
        const groupId = row.dataset.modelItemGroup || '';
        const anchorId = row.dataset.modelRow || '';
        finishDrag(root, state);
        onPlaceItem(itemId, groupId, anchorId, position);
    });
}

function bindItemDropzone(root, state, dropzone, onPlaceItem) {
    dropzone.addEventListener('dragover', (event) => {
        if (!state.draggedModelId) return;
        event.preventDefault();
        event.stopPropagation();
        setOrderedDropTarget(state, dropzone, 'dropzone', '', ITEM_DROP_OPTIONS);
    });
    dropzone.addEventListener('dragleave', (event) => {
        if (event.relatedTarget && dropzone.contains(event.relatedTarget)) return;
        clearOrderedDropTargetIf(state, dropzone, ITEM_DROP_OPTIONS);
    });
    dropzone.addEventListener('drop', (event) => {
        if (!state.draggedModelId) return;
        event.preventDefault();
        event.stopPropagation();
        const itemId = state.draggedModelId;
        const groupId = dropzone.dataset.modelDropzone || '';
        finishDrag(root, state);
        onPlaceItem(itemId, groupId, '', 'after');
    });
}

function bindGroupDrop(root, state, group, onPlaceItem, onPlaceGroup) {
    group.addEventListener('dragover', (event) => {
        if (state.draggedModelGroupId) {
            if (group.dataset.modelGroup === state.draggedModelGroupId) return;
            event.preventDefault();
            event.stopPropagation();
            scheduleOrderedRowDropTarget(state, group, event.clientY, GROUP_DROP_OPTIONS);
            return;
        }
        if (!state.draggedModelId || event.target.closest?.('[data-model-row], [data-model-dropzone]')) return;
        event.preventDefault();
        setOrderedDropTarget(state, group, 'group', '', ITEM_DROP_OPTIONS);
    });
    group.addEventListener('dragleave', (event) => {
        if (event.relatedTarget && group.contains(event.relatedTarget)) return;
        clearOrderedDropTargetIf(state, group, ITEM_DROP_OPTIONS);
    });
    group.addEventListener('drop', (event) => {
        if (state.draggedModelGroupId) {
            event.preventDefault();
            event.stopPropagation();
            placeDraggedGroup(root, state, group, event.clientY, onPlaceGroup);
            return;
        }
        if (!state.draggedModelId || event.target.closest?.('[data-model-row], [data-model-dropzone]')) return;
        event.preventDefault();
        const itemId = state.draggedModelId;
        const groupId = group.dataset.modelDropGroup || '';
        finishDrag(root, state);
        onPlaceItem(itemId, groupId, '', 'after');
    });
}

function bindGroupRow(root, state, row) {
    const handle = row.querySelector('.dragon-model-config-group-drag-handle[draggable="true"]');
    handle?.addEventListener('dragstart', (event) => {
        finishDrag(root, state);
        state.draggedModelGroupId = row.dataset.modelGroupRow || '';
        root.classList.add('dragon-model-config-group-dragging');
        row.closest('[data-model-group]')?.classList.add('dragon-model-config-group-dragging-source');
        event.dataTransfer?.setData('text/plain', state.draggedModelGroupId);
        if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
    });
    handle?.addEventListener('dragend', () => finishDrag(root, state));
}

function placeDraggedGroup(root, state, targetGroup, clientY, onPlaceGroup) {
    const rect = targetGroup.getBoundingClientRect();
    const after = clientY >= rect.top + rect.height / 2;
    const sourceId = state.draggedModelGroupId;
    const targetId = targetGroup.dataset.modelGroup || '';
    const sourceIndex = state.groups.findIndex((group) => group.id === sourceId);
    const targetIndex = state.groups.findIndex((group) => group.id === targetId);
    let insertionIndex = targetIndex + (after ? 1 : 0);
    if (sourceIndex < insertionIndex) insertionIndex -= 1;
    finishDrag(root, state);
    onPlaceGroup(sourceId, insertionIndex);
}

function bindDragRecovery(root, state) {
    if (state.modelDragRecovery) return;
    const finish = () => finishDrag(root, state);
    const cancelOnEscape = (event) => { if (event.key === 'Escape') finish(); };
    window.addEventListener('dragend', finish, true);
    window.addEventListener('blur', finish);
    document.addEventListener('keydown', cancelOnEscape);
    state.modelDragRecovery = { finish, cancelOnEscape };
}

export function disposeModelConfigDrag(root, state) {
    const recovery = state.modelDragRecovery;
    if (recovery) {
        window.removeEventListener('dragend', recovery.finish, true);
        window.removeEventListener('blur', recovery.finish);
        document.removeEventListener('keydown', recovery.cancelOnEscape);
        state.modelDragRecovery = null;
    }
    finishDrag(root, state);
}

function finishDrag(root, state) {
    state.draggedModelId = '';
    state.draggedModelGroupId = '';
    root.classList.remove('dragon-model-config-item-dragging', 'dragon-model-config-group-dragging');
    root.querySelectorAll('.dragon-model-config-row-dragging, .dragon-model-config-group-dragging-source').forEach((node) => {
        node.classList.remove('dragon-model-config-row-dragging', 'dragon-model-config-group-dragging-source');
    });
    clearOrderedDropTargets(state, ITEM_DROP_OPTIONS);
    clearOrderedDropTargets(state, GROUP_DROP_OPTIONS);
}

export function clearModelConfigDropFeedback(state) {
    clearActiveOrderedDropTarget(state, ITEM_DROP_OPTIONS);
    clearActiveOrderedDropTarget(state, GROUP_DROP_OPTIONS);
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
