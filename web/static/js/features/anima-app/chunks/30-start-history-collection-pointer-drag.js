/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    HISTORY_COLLECTION_DRAG_MIME,
    HISTORY_UNGROUPED_COLLECTION_KEY,
} from '../helpers/app-constants.js?v=module-bootstrap-20260831-release-v1';
import {
    autoScrollHistoryCollectionPointerDrag,
    canBeginHistoryCollectionSort,
    clearHistoryCollectionSortIndicators,
    clearHistoryDropTarget,
    configureHistoryCollectionDragBridge,
    createHistoryCollectionPointerDragImage,
    finishHistoryCollectionDrag,
    finishHistoryCollectionPointerDrag,
    finishHistoryDrag,
    historyCollectionEventPoint,
    historyCollectionPointerTargetFromPoint,
    historyDraggedTasksAlreadyInCollection,
    moveHistoryCollectionPointerDragImage,
    readHistoryDraggedTaskIds,
} from '../helpers/history-collection-drag-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    applyHistoryTaskIdsToCollection,
    collectionOrderValues,
    ensureHistoryCollectionOrderValue,
    historyCollectionsForWorkbench,
    historyCollectionSelectOptions,
    historyManagerFilteredTasks,
    historyTaskIsArchived,
} from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260831-release-v1';
import { renderHistoryManager, saveHistoryCollectionSettings, uniqueStringList } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260831-release-v1';

const historyState = getHistoryState();
const trainingState = getTrainingState();

    export function startHistoryCollectionPointerDrag(event, collection, allCollections = [], handle = null, options = { pointer: true }) {
        const usePointer = options.pointer !== false && 'pointerId' in event;
        if (historyState.historyCollectionPointerDrag) return;
        if ((usePointer || options.mouse) && 'button' in event && event.button !== 0) return;
        if (usePointer && event.isPrimary === false) return;
        if (!canBeginHistoryCollectionSort(collection)) {
            event.preventDefault();
            event.stopPropagation();
            return;
        }
        const startPoint = historyCollectionEventPoint(event);
        if (!startPoint) return;
        event.preventDefault();
        event.stopPropagation();
        const sourceValue = String(collection?.value || '').trim();
        const pointerId = usePointer ? event.pointerId : null;
        const dragHandle = handle || event.currentTarget;
        const drag = {
            sourceValue,
            allCollections,
            handle: dragHandle,
            pointerId,
            startX: startPoint.x,
            startY: startPoint.y,
            active: false,
            image: null,
            currentDrop: null,
        };
        const moveDrag = (moveEvent) => {
            const point = historyCollectionEventPoint(moveEvent);
            if (!point) return;
            const distance = Math.hypot(point.x - drag.startX, point.y - drag.startY);
            if (!drag.active) {
                if (distance < 5) return;
                closeHistoryDropPopover(false);
                finishHistoryDrag();
                historyState.historyCollectionDragState = {
                    active: true,
                    sourceValue,
                    activeDropTarget: '',
                    dropPosition: 'after',
                    pending: false,
                };
                drag.active = true;
                drag.image = createHistoryCollectionPointerDragImage(collection.label || sourceValue);
                dragHandle?.closest('.history-collection-card')?.classList.add('sort-source');
                dragHandle?.classList.add('dragging');
                document.body.classList.add('history-collection-pointer-drag-active');
                document.querySelector('.history-collections-workbench')?.classList.add('collection-reordering');
            }
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            moveHistoryCollectionPointerDragImage(drag.image, point.x, point.y);
            autoScrollHistoryCollectionPointerDrag(point.x, point.y);
            drag.currentDrop = historyCollectionPointerTargetFromPoint(point.x, point.y, allCollections);
            if (drag.currentDrop) {
                setHistoryCollectionSortTarget(drag.currentDrop.value, drag.currentDrop.position, drag.currentDrop.element);
            } else {
                clearHistoryCollectionSortIndicators();
            }
        };
        drag.onMove = (moveEvent) => {
            if (moveEvent.pointerId !== pointerId) return;
            moveDrag(moveEvent);
        };
        drag.onUp = (upEvent) => {
            if (upEvent.pointerId !== pointerId) return;
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishHistoryCollectionPointerDrag(true);
        };
        drag.onCancel = (cancelEvent) => {
            if (cancelEvent.pointerId !== pointerId) return;
            finishHistoryCollectionPointerDrag(false);
        };
        drag.onMouseMove = (moveEvent) => moveDrag(moveEvent);
        drag.onMouseUp = (upEvent) => {
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishHistoryCollectionPointerDrag(true);
        };
        drag.onTouchMove = (moveEvent) => moveDrag(moveEvent);
        drag.onTouchEnd = (touchEvent) => {
            touchEvent.preventDefault();
            touchEvent.stopPropagation();
            finishHistoryCollectionPointerDrag(true);
        };
        drag.onTouchCancel = () => finishHistoryCollectionPointerDrag(false);
        drag.onKeydown = (keyEvent) => {
            if (keyEvent.key === 'Escape') finishHistoryCollectionPointerDrag(false);
        };
        historyState.historyCollectionPointerDrag = drag;
        if (usePointer) {
            try {
                dragHandle?.setPointerCapture?.(pointerId);
            } catch (e) {
                /* 某些浏览器会让原生拖拽抢占捕获，文档级监听仍作为兜底。 */
            }
            document.addEventListener('pointermove', drag.onMove, { passive: false });
            document.addEventListener('pointerup', drag.onUp, { passive: false });
            document.addEventListener('pointercancel', drag.onCancel, { passive: false });
        } else if (options.touch) {
            document.addEventListener('touchmove', drag.onTouchMove, { passive: false });
            document.addEventListener('touchend', drag.onTouchEnd, { passive: false });
            document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false });
        } else {
            document.addEventListener('mousemove', drag.onMouseMove, { passive: false });
            document.addEventListener('mouseup', drag.onMouseUp, { passive: false });
        }
        document.addEventListener('keydown', drag.onKeydown);
    }

    export function startHistoryCollectionMouseDrag(event, collection, allCollections = [], handle = null) {
        startHistoryCollectionPointerDrag(event, collection, allCollections, handle, { pointer: false, mouse: true });
    }

    export function startHistoryCollectionTouchDrag(event, collection, allCollections = [], handle = null) {
        startHistoryCollectionPointerDrag(event, collection, allCollections, handle, { pointer: false, touch: true });
    }

    export function readHistoryDraggedCollectionValue(event) {
        const fallback = historyState.historyCollectionDragState.sourceValue || '';
        try {
            const direct = event?.dataTransfer?.getData(HISTORY_COLLECTION_DRAG_MIME);
            if (direct) return String(direct || '').trim();
        } catch (e) {
            /* 某些浏览器只允许在 drop 事件中读取 DataTransfer。 */
        }
        return String(fallback || '').trim();
    }

    export function historyCollectionDropPosition(event, element, collection) {
        if (collection?.is_ungrouped) return 'after';
        const rect = element?.getBoundingClientRect?.();
        if (!rect) return 'after';
        return Number(event?.clientY || 0) < rect.top + (rect.height / 2) ? 'before' : 'after';
    }

    export function setHistoryCollectionSortTarget(targetValue, position, element) {
        historyState.historyCollectionDragState.activeDropTarget = `collection-sort:${targetValue || '__ungrouped__'}`;
        historyState.historyCollectionDragState.dropPosition = position === 'before' ? 'before' : 'after';
        document.querySelectorAll('.history-collection-card.sort-active').forEach((item) => {
            if (item !== element) item.classList.remove('sort-active', 'sort-before', 'sort-after');
        });
        element?.classList.add(
            'sort-active',
            historyState.historyCollectionDragState.dropPosition === 'before' ? 'sort-before' : 'sort-after'
        );
    }

    export function clearHistoryCollectionSortTarget(targetValue, element) {
        if (historyState.historyCollectionDragState.activeDropTarget === `collection-sort:${targetValue || '__ungrouped__'}`) {
            historyState.historyCollectionDragState.activeDropTarget = '';
        }
        element?.classList.remove('sort-active', 'sort-before', 'sort-after');
    }

    export function historyCollectionOrderDragEnter(event, collection, element) {
        if (!historyState.historyCollectionDragState.active || historyState.historyCollectionDragState.pending) return false;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        const targetValue = String(collection?.value || '').trim();
        const position = historyCollectionDropPosition(event, element, collection);
        setHistoryCollectionSortTarget(targetValue, position, element);
        return true;
    }

    export function historyCollectionOrderDragLeave(event, collection, element) {
        if (!historyState.historyCollectionDragState.active) return false;
        if (element?.contains(event.relatedTarget)) return true;
        clearHistoryCollectionSortTarget(collection?.value || '', element);
        return true;
    }

    export function moveItemNearList(list, sourceValue, targetValue, position = 'after') {
        const out = uniqueStringList(list);
        const source = String(sourceValue || '').trim();
        const target = String(targetValue || '').trim();
        if (!source || !out.includes(source)) return out;
        const original = [...out];
        out.splice(out.indexOf(source), 1);
        let index = 0;
        if (target) {
            const targetIndex = out.indexOf(target);
            index = targetIndex < 0 ? out.length : targetIndex + (position === 'after' ? 1 : 0);
        }
        out.splice(Math.max(0, Math.min(out.length, index)), 0, source);
        return out.length === original.length && out.every((value, idx) => value === original[idx]) ? original : out;
    }

    export async function reorderHistoryCollectionValue(sourceValue, targetValue, position, allCollections = null) {
        const source = String(sourceValue || '').trim();
        if (!source) return false;
        const collections = allCollections || historyCollectionsForWorkbench(historyState.historyTasks);
        const currentOrder = collectionOrderValues(collections);
        const nextOrder = moveItemNearList(currentOrder, source, targetValue, position);
        if (nextOrder.length === currentOrder.length && nextOrder.every((value, idx) => value === currentOrder[idx])) {
            return false;
        }
        await saveHistoryCollectionSettings({
            ...historyState.historyCollectionSettings,
            collection_order: nextOrder,
        });
        return true;
    }

    export async function dropHistoryCollectionToSort(event, targetCollection, allCollections = []) {
        if (!historyState.historyCollectionDragState.active) return false;
        event.preventDefault();
        event.stopPropagation();
        const source = readHistoryDraggedCollectionValue(event);
        const target = String(targetCollection?.value || '').trim();
        const position = historyState.historyCollectionDragState.dropPosition
            || historyCollectionDropPosition(event, event.currentTarget, targetCollection);
        clearHistoryCollectionSortTarget(target, event.currentTarget);
        if (!source) {
            setHistoryDropFeedback('没有可排序的分组。', 'error');
            finishHistoryCollectionDrag();
            return true;
        }
        if (source === target) {
            setHistoryDropFeedback('分组顺序未变化。', 'ok');
            finishHistoryCollectionDrag();
            return true;
        }
        historyState.historyCollectionDragState.pending = true;
        try {
            const changed = await reorderHistoryCollectionValue(source, target, position, allCollections);
            setHistoryDropFeedback(changed ? `已调整「${source}」的位置。` : '分组顺序未变化。', 'ok');
        } catch (e) {
            setHistoryDropFeedback(`调整分组顺序失败: ${e.message}`, 'error');
        } finally {
            finishHistoryCollectionDrag();
            renderHistoryManager();
        }
        return true;
    }

    export async function dropHistoryTasksToCollection(event, groupValue, label) {
        event.preventDefault();
        event.stopPropagation();
        const taskIds = readHistoryDraggedTaskIds(event);
        const clean = String(groupValue || '').trim();
        clearHistoryDropTarget(historyState.historyDragState.activeDropTarget, event.currentTarget);
        if (!taskIds.length) {
            setHistoryDropFeedback('没有可移动的历史任务。', 'error');
            finishHistoryDrag();
            return;
        }
        if (historyDraggedTasksAlreadyInCollection(taskIds, clean)) {
            setHistoryDropFeedback(`已在${clean ? `分组「${clean}」` : '未分类'}中。`, 'ok');
            finishHistoryDrag();
            return;
        }
        historyState.historyDragState.pending = true;
        document.querySelector('.history-collections-workbench')?.classList.add('drop-pending');
        try {
            const res = await applyHistoryTaskIdsToCollection(taskIds, clean, { clearSelection: true });
            if (res === null) {
                setHistoryDropFeedback('移动失败，列表未更改。', 'error');
            } else {
                historyState.selectedHistoryCollectionKey = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;
                setHistoryDropFeedback(`${taskIds.length} 条任务已移动到${clean ? `「${label || clean}」` : '未分类'}。`, 'ok');
            }
        } catch (e) {
            setHistoryDropFeedback(`移动失败: ${e.message}`, 'error');
        } finally {
            historyState.historyDragState.pending = false;
            finishHistoryDrag();
            renderHistoryManager();
        }
    }

    export function defaultHistoryCollectionName() {
        const now = new Date();
        const yyyy = String(now.getFullYear());
        const mm = String(now.getMonth() + 1).padStart(2, '0');
        const dd = String(now.getDate()).padStart(2, '0');
        return uniqueHistoryCollectionName(`未分配_${yyyy}${mm}${dd}`);
    }

    export function uniqueHistoryCollectionName(base) {
        const cleanBase = String(base || '').trim().slice(0, 48) || '未分配';
        const existing = new Set([
            ...historyCollectionSelectOptions().map((item) => String(item.value || '').trim()).filter(Boolean),
            ...uniqueStringList(historyState.historyCollectionSettings.collection_order || []),
        ]);
        if (!existing.has(cleanBase)) return cleanBase;
        for (let index = 2; index < 1000; index += 1) {
            const suffix = `_${index}`;
            const candidate = `${cleanBase.slice(0, Math.max(1, 48 - suffix.length))}${suffix}`;
            if (!existing.has(candidate)) return candidate;
        }
        return cleanBase;
    }

    export function openHistoryNewCollectionPopover(event = null, taskIds = []) {
        const rect = event?.currentTarget?.getBoundingClientRect?.();
        const x = Number(event?.clientX || 0) || (rect ? Math.round(rect.left + rect.width / 2) : Math.round(window.innerWidth / 2));
        const y = Number(event?.clientY || 0) || (rect ? Math.round(rect.bottom + 8) : Math.round(window.innerHeight / 2));
        historyState.historyDragState.active = false;
        historyState.historyDragState.activeDropTarget = '';
        historyState.historyDragState.popover = {
            open: true,
            x,
            y,
            taskIds: uniqueStringList(taskIds),
            defaultName: defaultHistoryCollectionName(),
        };
        finishHistoryDrag();
        renderHistoryManager();
    }

    export function renderHistoryDropPopover(workbench) {
        if (historyState.historyDropPopoverOutsideHandler) {
            document.removeEventListener('mousedown', historyState.historyDropPopoverOutsideHandler);
            historyState.historyDropPopoverOutsideHandler = null;
        }
        const state = historyState.historyDragState.popover;
        if (!state.open) return;

        const popover = document.createElement('form');
        popover.className = 'history-drop-popover';
        popover.noValidate = true;
        const width = 320;
        const height = 152;
        const left = Math.min(Math.max(8, state.x), Math.max(8, window.innerWidth - width - 8));
        const top = Math.min(Math.max(8, state.y), Math.max(8, window.innerHeight - height - 8));
        popover.style.left = `${left}px`;
        popover.style.top = `${top}px`;

        const label = document.createElement('label');
        const title = document.createElement('span');
        title.textContent = state.taskIds.length ? `${state.taskIds.length} 条任务归入新分组` : '新建分组';
        const input = document.createElement('input');
        input.type = 'text';
        input.maxLength = 48;
        input.value = state.defaultName || defaultHistoryCollectionName();
        input.placeholder = '分组名称';
        input.disabled = historyState.historyDragState.pending;
        label.append(title, input);

        const actions = document.createElement('div');
        actions.className = 'history-drop-popover-actions';
        const cancel = document.createElement('button');
        cancel.type = 'button';
        cancel.className = 'task-history-action';
        cancel.textContent = '取消';
        cancel.disabled = historyState.historyDragState.pending;
        cancel.addEventListener('click', () => closeHistoryDropPopover());
        const submit = document.createElement('button');
        submit.type = 'submit';
        submit.className = 'task-history-action primary';
        submit.textContent = historyState.historyDragState.pending
            ? '保存中...'
            : (state.taskIds.length ? '新建并移动' : '新建分组');
        submit.disabled = historyState.historyDragState.pending || !input.value.trim();
        actions.append(cancel, submit);

        const updateSubmitState = () => {
            submit.disabled = historyState.historyDragState.pending || !input.value.trim();
        };
        input.addEventListener('input', updateSubmitState);
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                event.preventDefault();
                closeHistoryDropPopover();
            }
            if (event.key === 'Enter') {
                event.preventDefault();
                if (!submit.disabled) submitHistoryDropPopover(input.value);
            }
        });
        popover.addEventListener('submit', (event) => {
            event.preventDefault();
            if (!submit.disabled) submitHistoryDropPopover(input.value);
        });
        popover.append(label, actions);
        workbench.appendChild(popover);

        requestAnimationFrame(() => {
            input.focus();
            input.select();
        });
        const outsideHandler = (event) => {
            if (!popover.contains(event.target)) closeHistoryDropPopover();
        };
        historyState.historyDropPopoverOutsideHandler = outsideHandler;
        setTimeout(() => {
            if (historyState.historyDropPopoverOutsideHandler === outsideHandler) {
                document.addEventListener('mousedown', outsideHandler);
            }
        }, 0);
    }

    export function closeHistoryDropPopover(render = true) {
        if (historyState.historyDropPopoverOutsideHandler) {
            document.removeEventListener('mousedown', historyState.historyDropPopoverOutsideHandler);
            historyState.historyDropPopoverOutsideHandler = null;
        }
        document.querySelectorAll('.history-drop-popover').forEach((popover) => popover.remove());
        historyState.historyDragState.pending = false;
        historyState.historyDragState.popover = {
            open: false,
            x: 0,
            y: 0,
            taskIds: [],
            defaultName: '',
        };
        if (render) renderHistoryManager();
    }

    export async function submitHistoryDropPopover(name) {
        const clean = String(name || '').trim().slice(0, 48);
        const taskIds = uniqueStringList(historyState.historyDragState.popover.taskIds);
        if (!clean || historyState.historyDragState.pending) return;
        historyState.historyDragState.popover.defaultName = clean;
        historyState.historyDragState.pending = true;
        renderHistoryManager();
        try {
            if (taskIds.length) {
                const res = await applyHistoryTaskIdsToCollection(taskIds, clean, { clearSelection: true });
                if (res === null) {
                    historyState.historyDragState.pending = false;
                    setHistoryDropFeedback('新建分组失败，列表未更改。', 'error');
                    renderHistoryManager();
                    return;
                }
                setHistoryDropFeedback(`${taskIds.length} 条任务已移动到「${clean}」。`, 'ok');
            } else {
                await ensureHistoryCollectionOrderValue(clean);
                setHistoryDropFeedback(`已新建分组「${clean}」。`, 'ok');
            }
            historyState.selectedHistoryCollectionKey = `collection:${clean}`;
            closeHistoryDropPopover(false);
        } catch (e) {
            historyState.historyDragState.pending = false;
            setHistoryDropFeedback(`新建分组失败: ${e.message}`, 'error');
        } finally {
            renderHistoryManager();
        }
    }

    export function setHistoryDropFeedback(message, tone = '') {
        historyState.historyDropFeedback = { message: String(message || ''), tone: String(tone || '') };
        if (historyState.historyDropFeedbackTimer) clearTimeout(historyState.historyDropFeedbackTimer);
        const status = document.getElementById('history-manager-status');
        if (status && historyState.historyDropFeedback.message) {
            const visible = historyManagerFilteredTasks();
            const archivedCount = historyState.historyTasks.filter(historyTaskIsArchived).length;
            status.textContent = [
                `共 ${historyState.historyTasks.length} 条记录`,
                `当前分组 ${visible.length} 条`,
                `归档 ${archivedCount} 条`,
                historyState.historyDropFeedback.message,
            ].filter(Boolean).join(' · ');
            status.dataset.feedbackTone = historyState.historyDropFeedback.tone;
        }
        historyState.historyDropFeedbackTimer = setTimeout(() => {
            historyState.historyDropFeedback = { message: '', tone: '' };
            historyState.historyDropFeedbackTimer = null;
            if (trainingState.trainingViewMode === 'history') renderHistoryManager();
        }, 2600);
    }

configureHistoryCollectionDragBridge({
    startHistoryCollectionPointerDrag,
    startHistoryCollectionMouseDrag,
    startHistoryCollectionTouchDrag,
    readHistoryDraggedCollectionValue,
    historyCollectionDropPosition,
    setHistoryCollectionSortTarget,
    clearHistoryCollectionSortTarget,
    historyCollectionOrderDragEnter,
    historyCollectionOrderDragLeave,
    moveItemNearList,
    reorderHistoryCollectionValue,
    dropHistoryCollectionToSort,
    dropHistoryTasksToCollection,
    defaultHistoryCollectionName,
    uniqueHistoryCollectionName,
    openHistoryNewCollectionPopover,
    renderHistoryDropPopover,
    closeHistoryDropPopover,
    submitHistoryDropPopover,
    setHistoryDropFeedback,
});
