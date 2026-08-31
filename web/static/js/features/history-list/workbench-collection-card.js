/**
 * History collection workbench card UI.
 */
import {
    beginHistoryCollectionDrag,
    dropHistoryCollectionToSort,
    dropHistoryTasksToCollection,
    finishHistoryCollectionDrag,
    finishHistoryCollectionPointerDrag,
    historyCollectionOrderDragEnter,
    historyCollectionOrderDragLeave,
    historyDropTargetDragEnter,
    historyDropTargetDragLeave,
    startHistoryCollectionMouseDrag,
    startHistoryCollectionPointerDrag,
    startHistoryCollectionTouchDrag,
} from '../anima-app/helpers/history-collection-drag-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    applySelectedHistoryTasksToCollection,
    clearHistoryCollection,
    createHistoryManagerGroupButton,
    createHistoryMoreActions,
    historyCompactGroupMetaParts,
    renameHistoryCollection,
} from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260831-release-v1';
import { renderHistoryManager } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260831-release-v1';
import { selectHistoryCollectionInWorkbench } from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    moveHistoryCollection,
} from './workbench-order.js?v=module-bootstrap-20260831-release-v1';

const historyState = getHistoryState();

export function createHistoryCollectionWorkbenchCard(collection, selectedTaskCount = 0, allCollections = []) {
    const card = document.createElement('article');
    card.className = ['history-collection-card', 'nav-card', collection.is_ungrouped ? 'ungrouped' : ''].filter(Boolean).join(' ');
    const dropTargetId = `collection:${collection.value || '__ungrouped__'}`;
    const canSortCollection = !collection.is_ungrouped && Boolean(collection.value);
    card.dataset.collectionKey = collection.key || '';
    card.dataset.collectionValue = collection.value || '';
    if (canSortCollection) {
        card.classList.add('sortable');
    }
    if (historyState.selectedHistoryCollectionKey === collection.key) {
        card.classList.add('active');
    }
    if (
        historyState.historyCollectionWorkbenchTarget
        && collection.value === historyState.historyCollectionWorkbenchTarget
    ) {
        card.classList.add('target');
    }
    if (historyState.historyDragState.activeDropTarget === dropTargetId) {
        card.classList.add('drop-active');
    }
    if (
        historyState.historyCollectionDragState.sourceValue
        && historyState.historyCollectionDragState.sourceValue === collection.value
    ) {
        card.classList.add('sort-source');
    }
    if (
        historyState.historyCollectionDragState.active &&
        historyState.historyCollectionDragState.activeDropTarget === `collection-sort:${collection.value || '__ungrouped__'}`
    ) {
        card.classList.add(
            'sort-active',
            historyState.historyCollectionDragState.dropPosition === 'before' ? 'sort-before' : 'sort-after'
        );
    }
    card.tabIndex = 0;
    card.setAttribute('role', 'button');
    card.setAttribute('aria-pressed', historyState.selectedHistoryCollectionKey === collection.key ? 'true' : 'false');
    card.addEventListener('dragenter', (event) => {
        if (historyCollectionOrderDragEnter(event, collection, card)) return;
        historyDropTargetDragEnter(event, dropTargetId, card);
    });
    card.addEventListener('dragover', (event) => {
        if (historyCollectionOrderDragEnter(event, collection, card)) return;
        historyDropTargetDragEnter(event, dropTargetId, card);
    });
    card.addEventListener('dragleave', (event) => {
        if (historyCollectionOrderDragLeave(event, collection, card)) return;
        historyDropTargetDragLeave(event, dropTargetId, card);
    });
    card.addEventListener('drop', async (event) => {
        if (await dropHistoryCollectionToSort(event, collection, allCollections)) return;
        dropHistoryTasksToCollection(event, collection.value || '', collection.label);
    });
    card.addEventListener('click', () => {
        selectHistoryCollectionInWorkbench(collection.key);
    });
    card.addEventListener('keydown', (event) => {
        if (event.target !== card) return;
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        selectHistoryCollectionInWorkbench(collection.key);
    });

    const head = document.createElement('div');
    head.className = 'history-collection-card-head';
    const title = document.createElement('div');
    title.className = 'history-collection-card-title';
    const titleRow = document.createElement('div');
    titleRow.className = 'history-collection-card-title-row';
    const passiveHandle = document.createElement('span');
    passiveHandle.className = 'history-drag-handle history-drag-handle-passive';
    passiveHandle.textContent = '⋮⋮';
    passiveHandle.title = collection.is_ungrouped ? '未分类分组不可排序' : '拖拽分组调整顺序';
    passiveHandle.setAttribute('aria-hidden', 'true');
    if (canSortCollection) {
        const dragHandle = document.createElement('button');
        dragHandle.type = 'button';
        dragHandle.className = 'history-drag-handle history-collection-drag-handle';
        dragHandle.textContent = '⋮⋮';
        dragHandle.title = '拖拽分组调整顺序';
        dragHandle.setAttribute('aria-label', '拖拽分组调整顺序');
        dragHandle.draggable = true;
        dragHandle.addEventListener('click', (event) => event.stopPropagation());
        dragHandle.addEventListener('pointerdown', (event) => startHistoryCollectionPointerDrag(event, collection, allCollections, dragHandle));
        dragHandle.addEventListener('mousedown', (event) => {
            event.stopPropagation();
            if (!('PointerEvent' in window)) startHistoryCollectionMouseDrag(event, collection, allCollections, dragHandle);
        });
        dragHandle.addEventListener('touchstart', (event) => {
            event.stopPropagation();
            if (!('PointerEvent' in window)) startHistoryCollectionTouchDrag(event, collection, allCollections, dragHandle);
        }, { passive: false });
        dragHandle.addEventListener('dragstart', (event) => {
            if (historyState.historyCollectionPointerDrag) finishHistoryCollectionPointerDrag(false);
            beginHistoryCollectionDrag(event, collection);
        });
        dragHandle.addEventListener('dragend', () => finishHistoryCollectionDrag());
        titleRow.appendChild(dragHandle);
    } else {
        titleRow.appendChild(passiveHandle);
    }
    const name = document.createElement('strong');
    name.textContent = collection.is_ungrouped ? '未分类' : collection.label;
    name.title = name.textContent;
    titleRow.appendChild(name);
    const meta = document.createElement('span');
    meta.className = 'history-compact-meta';
    meta.textContent = historyCompactGroupMetaParts(collection.tasks, [
        `${collection.groups.length} 组`,
    ]).join(' · ');
    title.append(titleRow, meta);

    const actions = document.createElement('div');
    actions.className = 'history-collection-card-actions';
    const joinSelectedBtn = createHistoryManagerGroupButton(
        collection.is_ungrouped ? '未分类' : '移入',
        () => applySelectedHistoryTasksToCollection(collection.value),
    );
    if (selectedTaskCount > 0) actions.append(joinSelectedBtn);
    if (!collection.is_ungrouped) {
        actions.append(
            createHistoryManagerGroupButton(
                historyState.historyCollectionWorkbenchTarget === collection.value ? '取消目标' : '目标',
                () => {
                    historyState.historyCollectionWorkbenchTarget =
                        historyState.historyCollectionWorkbenchTarget === collection.value ? '' : collection.value;
                    renderHistoryManager();
                },
            ),
            createHistoryMoreActions([
                createHistoryManagerGroupButton('置顶', () => moveHistoryCollection(collection, 'top', allCollections)),
                createHistoryManagerGroupButton('上移', () => moveHistoryCollection(collection, 'up', allCollections)),
                createHistoryManagerGroupButton('下移', () => moveHistoryCollection(collection, 'down', allCollections)),
                createHistoryManagerGroupButton('置底', () => moveHistoryCollection(collection, 'bottom', allCollections)),
                createHistoryManagerGroupButton('重命名', () => renameHistoryCollection(collection)),
                createHistoryManagerGroupButton('清空', () => clearHistoryCollection(collection)),
            ]),
        );
    }
    head.append(title, actions);
    card.appendChild(head);
    return card;
}
