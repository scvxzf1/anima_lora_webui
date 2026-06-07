/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.startHistoryConfigGroupPointerDrag = function startHistoryConfigGroupPointerDrag(event, group, options = {}, handle = null, fallback = { pointer: true }) {
        const usePointer = fallback.pointer !== false && 'pointerId' in event;
        if (historyConfigGroupPointerDrag) return;
        if ((usePointer || fallback.mouse) && 'button' in event && event.button !== 0) return;
        if (usePointer && event.isPrimary === false) return;
        if (!canBeginHistoryConfigGroupDrag(group)) {
            event.preventDefault();
            event.stopPropagation();
            return;
        }
        const startPoint = historyCollectionEventPoint(event);
        if (!startPoint) return;
        event.stopPropagation();
        if (fallback.touch) event.preventDefault();
        const taskIds = uniqueStringList(historyDragTaskIdsForGroup(group));
        const sourceKey = configGroupKey(group);
        const collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        const dragHandle = handle || event.currentTarget;
        const pointerId = usePointer ? event.pointerId : null;
        const drag = {
            sourceKey,
            collectionKey,
            taskIds,
            groups: options.groups || [],
            collection: options.collection || null,
            handle: dragHandle,
            pointerId,
            startX: startPoint.x,
            startY: startPoint.y,
            active: false,
            image: null,
            currentDrop: null,
            currentCollectionDrop: null,
        };
        const moveDrag = (moveEvent) => {
            const point = historyCollectionEventPoint(moveEvent);
            if (!point) return;
            const distance = Math.hypot(point.x - drag.startX, point.y - drag.startY);
            if (!drag.active) {
                if (distance < 5) return;
                closeHistoryDropPopover(false);
                historyDragState = {
                    ...historyDragState,
                    active: true,
                    taskIds,
                    sourceGroupKey: sourceKey,
                    activeDropTarget: '',
                    popover: {
                        open: false,
                        x: 0,
                        y: 0,
                        taskIds: [],
                        defaultName: '',
                    },
                };
                historyConfigGroupSortState = {
                    active: Boolean(sourceKey),
                    sourceKey,
                    collectionKey,
                    activeDropTarget: '',
                    dropPosition: 'after',
                    pending: false,
                };
                drag.active = true;
                drag.image = createHistoryDragImage(taskIds.length);
                dragHandle?.classList.add('dragging');
                dragHandle?.closest('.history-config-group-card')?.classList.add('config-sort-source');
                document.body.classList.add('history-config-group-pointer-drag-active');
                document.querySelector('.history-collections-workbench')?.classList.add('dragging');
            }
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            moveHistoryCollectionPointerDragImage(drag.image, point.x, point.y);
            autoScrollHistoryCollectionPointerDrag(point.x, point.y);
            drag.currentCollectionDrop = historyCollectionDropTargetFromPoint(point.x, point.y);
            if (drag.currentCollectionDrop) {
                drag.currentDrop = null;
                setHistoryDropTarget(`collection:${drag.currentCollectionDrop.dataset.collectionValue || '__ungrouped__'}`, drag.currentCollectionDrop);
                clearHistoryConfigGroupSortIndicators();
                return;
            }
            clearHistoryDropIndicators();
            drag.currentDrop = historyConfigGroupPointerTargetFromPoint(point.x, point.y, drag.groups, drag.collection);
            if (drag.currentDrop && drag.currentDrop.key !== drag.sourceKey) {
                setHistoryConfigGroupSortTarget(drag.currentDrop.key, drag.currentDrop.position, drag.currentDrop.element);
            } else {
                drag.currentDrop = null;
                clearHistoryConfigGroupSortIndicators();
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
            finishHistoryConfigGroupPointerDrag(true);
        };
        drag.onCancel = (cancelEvent) => {
            if (cancelEvent.pointerId !== pointerId) return;
            finishHistoryConfigGroupPointerDrag(false);
        };
        drag.onMouseMove = (moveEvent) => moveDrag(moveEvent);
        drag.onMouseUp = (upEvent) => {
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishHistoryConfigGroupPointerDrag(true);
        };
        drag.onTouchMove = (moveEvent) => moveDrag(moveEvent);
        drag.onTouchEnd = (touchEvent) => {
            touchEvent.preventDefault();
            touchEvent.stopPropagation();
            finishHistoryConfigGroupPointerDrag(true);
        };
        drag.onTouchCancel = () => finishHistoryConfigGroupPointerDrag(false);
        drag.onKeydown = (keyEvent) => {
            if (keyEvent.key === 'Escape') finishHistoryConfigGroupPointerDrag(false);
        };
        historyConfigGroupPointerDrag = drag;
        if (usePointer) {
            try {
                dragHandle?.setPointerCapture?.(pointerId);
            } catch (e) {
                /* 某些浏览器会让原生拖拽抢占捕获，文档级监听仍作为兜底。 */
            }
            document.addEventListener('pointermove', drag.onMove, { passive: false });
            document.addEventListener('pointerup', drag.onUp, { passive: false });
            document.addEventListener('pointercancel', drag.onCancel, { passive: false });
        } else if (fallback.touch) {
            document.addEventListener('touchmove', drag.onTouchMove, { passive: false });
            document.addEventListener('touchend', drag.onTouchEnd, { passive: false });
            document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false });
        } else {
            document.addEventListener('mousemove', drag.onMouseMove, { passive: false });
            document.addEventListener('mouseup', drag.onMouseUp, { passive: false });
        }
        document.addEventListener('keydown', drag.onKeydown);
    }

    globalThis.startHistoryConfigGroupMouseDrag = function startHistoryConfigGroupMouseDrag(event, group, options = {}, handle = null) {
        startHistoryConfigGroupPointerDrag(event, group, options, handle, { pointer: false, mouse: true });
    }

    globalThis.startHistoryConfigGroupTouchDrag = function startHistoryConfigGroupTouchDrag(event, group, options = {}, handle = null) {
        startHistoryConfigGroupPointerDrag(event, group, options, handle, { pointer: false, touch: true });
    }

    globalThis.reorderHistoryConfigGroupValue = async function reorderHistoryConfigGroupValue(sourceKey, targetKey, position, groups = [], collection = null) {
        const source = String(sourceKey || '').trim();
        const target = String(targetKey || '').trim();
        if (!source || !target) return false;
        const collectionKey = historyCollectionStorageKey(collection || '__all__');
        const currentOrder = configGroupOrderValues(groups, collection);
        const nextOrder = moveItemNearList(currentOrder, source, target, position);
        if (nextOrder.length === currentOrder.length && nextOrder.every((value, idx) => value === currentOrder[idx])) {
            return false;
        }
        await saveHistoryCollectionSettings({
            ...historyCollectionSettings,
            config_group_order: {
                ...(historyCollectionSettings.config_group_order || {}),
                [collectionKey]: nextOrder,
            },
        });
        return true;
    }

    globalThis.dropHistoryConfigGroupToSort = async function dropHistoryConfigGroupToSort(event, targetGroup, options = {}) {
        if (!historyConfigGroupSortState.active) return false;
        const source = readHistoryDraggedConfigGroup(event);
        const targetKey = configGroupKey(targetGroup);
        const collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        if (!source.groupKey || !targetKey || source.collectionKey !== collectionKey) return false;
        event.preventDefault();
        event.stopPropagation();
        const position = historyConfigGroupSortState.dropPosition || historyConfigGroupDropPosition(event, event.currentTarget);
        clearHistoryConfigGroupSortTarget(targetKey, event.currentTarget);
        if (source.groupKey === targetKey) {
            setHistoryDropFeedback('配置分组顺序未变化。', 'ok');
            finishHistoryDrag();
            return true;
        }
        historyConfigGroupSortState.pending = true;
        try {
            const changed = await reorderHistoryConfigGroupValue(
                source.groupKey,
                targetKey,
                position,
                options.groups || [],
                options.collection || null,
            );
            setHistoryDropFeedback(changed ? '已调整配置分组顺序。' : '配置分组顺序未变化。', 'ok');
        } catch (e) {
            setHistoryDropFeedback(`调整配置分组顺序失败: ${e.message}`, 'error');
        } finally {
            finishHistoryDrag();
            renderHistoryManager();
        }
        return true;
    }

    globalThis.readHistoryDraggedTaskIds = function readHistoryDraggedTaskIds(event) {
        const fallback = historyDragState.taskIds || [];
        const transfer = event?.dataTransfer;
        const sources = [];
        try {
            sources.push(transfer?.getData(HISTORY_TASK_DRAG_MIME));
            sources.push(transfer?.getData('text/plain'));
        } catch (e) {
            /* 某些浏览器只允许在 drop 事件中读取 DataTransfer。 */
        }
        for (const raw of sources) {
            if (!raw) continue;
            try {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed)) return uniqueStringList(parsed);
            } catch (e) {
                const text = String(raw || '').trim();
                if (text) return uniqueStringList([text]);
            }
        }
        return uniqueStringList(fallback);
    }

    globalThis.setHistoryDropTarget = function setHistoryDropTarget(id, element) {
        if (!historyDragState.active || historyDragState.pending) return;
        historyDragState.activeDropTarget = id;
        document.querySelectorAll('.history-collection-card.drop-active').forEach((item) => {
            if (item !== element) item.classList.remove('drop-active');
        });
        element?.classList.add('drop-active');
    }

    globalThis.clearHistoryDropTarget = function clearHistoryDropTarget(id, element) {
        if (historyDragState.activeDropTarget === id) {
            historyDragState.activeDropTarget = '';
        }
        element?.classList.remove('drop-active');
    }

    globalThis.clearHistoryDropIndicators = function clearHistoryDropIndicators() {
        historyDragState.activeDropTarget = '';
        document.querySelectorAll('.history-collection-card.drop-active').forEach((item) => {
            item.classList.remove('drop-active');
        });
    }

    globalThis.historyTasksByIds = function historyTasksByIds(ids) {
        const taskMap = new Map(historyTasks.map((task) => [task.id, task]));
        return uniqueStringList(ids).map((id) => taskMap.get(id)).filter(Boolean);
    }

    globalThis.historyDraggedTasksAlreadyInCollection = function historyDraggedTasksAlreadyInCollection(ids, groupValue) {
        const clean = String(groupValue || '').trim();
        const taskIds = uniqueStringList(ids);
        const tasks = historyTasksByIds(taskIds);
        return tasks.length === taskIds.length && tasks.every((task) => historyTaskCollectionValue(task) === clean);
    }

    globalThis.historyDropTargetDragEnter = function historyDropTargetDragEnter(event, targetId, element) {
        if (!historyDragState.active || historyDragState.pending) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        setHistoryDropTarget(targetId, element);
    }

    globalThis.historyDropTargetDragLeave = function historyDropTargetDragLeave(event, targetId, element) {
        if (element?.contains(event.relatedTarget)) return;
        clearHistoryDropTarget(targetId, element);
    }

    globalThis.canBeginHistoryCollectionSort = function canBeginHistoryCollectionSort(collection) {
        const value = String(collection?.value || '').trim();
        return Boolean(value && !collection?.is_ungrouped && !historyCollectionDragState.pending && !historyDragState.pending);
    }

    globalThis.beginHistoryCollectionDrag = function beginHistoryCollectionDrag(event, collection) {
        const value = String(collection?.value || '').trim();
        if (!canBeginHistoryCollectionSort(collection)) {
            event.preventDefault();
            return;
        }
        closeHistoryDropPopover(false);
        finishHistoryDrag();
        historyCollectionDragState = {
            active: true,
            sourceValue: value,
            activeDropTarget: '',
            dropPosition: 'after',
            pending: false,
        };
        if (event.dataTransfer) {
            event.dataTransfer.setData(HISTORY_COLLECTION_DRAG_MIME, value);
            event.dataTransfer.setData('text/plain', JSON.stringify({ type: 'history-collection', value }));
            event.dataTransfer.effectAllowed = 'move';
        }
        event.currentTarget?.closest('.history-collection-card')?.classList.add('sort-source');
        document.querySelector('.history-collections-workbench')?.classList.add('collection-reordering');
    }

    globalThis.finishHistoryCollectionDrag = function finishHistoryCollectionDrag() {
        historyCollectionDragState = {
            active: false,
            sourceValue: '',
            activeDropTarget: '',
            dropPosition: 'after',
            pending: false,
        };
        document.querySelectorAll('.history-collection-card.sort-active, .history-collection-card.sort-source').forEach((item) => {
            item.classList.remove('sort-active', 'sort-before', 'sort-after', 'sort-source');
        });
        document.querySelector('.history-collections-workbench')?.classList.remove('collection-reordering');
    }

    globalThis.clearHistoryCollectionSortIndicators = function clearHistoryCollectionSortIndicators() {
        historyCollectionDragState.activeDropTarget = '';
        document.querySelectorAll('.history-collection-card.sort-active').forEach((item) => {
            item.classList.remove('sort-active', 'sort-before', 'sort-after');
        });
    }

    globalThis.createHistoryCollectionPointerDragImage = function createHistoryCollectionPointerDragImage(label) {
        removeHistoryDragImage();
        const image = document.createElement('div');
        image.className = 'history-drag-image history-collection-drag-image-pointer';
        image.textContent = label || '历史分组';
        document.body.appendChild(image);
        historyDragImageElement = image;
        return image;
    }

    globalThis.moveHistoryCollectionPointerDragImage = function moveHistoryCollectionPointerDragImage(image, x, y) {
        if (!image) return;
        image.style.left = `${x + 14}px`;
        image.style.top = `${y + 14}px`;
    }

    globalThis.historyCollectionForPointerCard = function historyCollectionForPointerCard(card, allCollections = []) {
        if (!card) return null;
        const key = String(card.dataset.collectionKey || '').trim();
        const value = String(card.dataset.collectionValue || '').trim();
        return (allCollections || []).find((collection) => collection.key === key)
            || (allCollections || []).find((collection) => String(collection.value || '').trim() === value)
            || null;
    }

    globalThis.historyCollectionPointerTargetForCard = function historyCollectionPointerTargetForCard(card, x, y, allCollections = []) {
        const collection = historyCollectionForPointerCard(card, allCollections);
        if (!collection) return null;
        return {
            element: card,
            collection,
            value: String(collection.value || '').trim(),
            position: historyCollectionDropPosition({ clientY: y }, card, collection),
        };
    }

    globalThis.nearestHistoryCollectionPointerTarget = function nearestHistoryCollectionPointerTarget(x, y, allCollections = []) {
        let best = null;
        document.querySelectorAll('.history-collection-card.nav-card').forEach((card) => {
            if (!card?.isConnected) return;
            const rect = card.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
            const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
            const distance = Math.hypot(dx, dy);
            const maxDistance = Math.max(22, Math.min(72, rect.height * 0.9));
            if (distance > maxDistance || (best && distance >= best.distance)) return;
            const target = historyCollectionPointerTargetForCard(card, x, y, allCollections);
            if (target) best = { ...target, distance };
        });
        if (!best) return null;
        const { distance, ...target } = best;
        return target;
    }

    globalThis.historyCollectionPointerTargetFromPoint = function historyCollectionPointerTargetFromPoint(x, y, allCollections = []) {
        const origin = document.elementFromPoint(x, y);
        const card = origin instanceof Element ? origin.closest('.history-collection-card.nav-card') : null;
        return historyCollectionPointerTargetForCard(card, x, y, allCollections)
            || nearestHistoryCollectionPointerTarget(x, y, allCollections);
    }

    globalThis.findHistoryCollectionPointerScroller = function findHistoryCollectionPointerScroller(origin) {
        let node = origin instanceof Element ? origin : null;
        while (node && node !== document.body) {
            const style = window.getComputedStyle(node);
            if (/(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight) {
                return node;
            }
            node = node.parentElement;
        }
        return document.scrollingElement;
    }

    globalThis.autoScrollHistoryCollectionPointerDrag = function autoScrollHistoryCollectionPointerDrag(x, y) {
        const origin = document.elementFromPoint(x, y);
        const scroller = findHistoryCollectionPointerScroller(origin);
        if (!scroller) return;
        const rect = scroller === document.scrollingElement
            ? { top: 0, bottom: window.innerHeight }
            : scroller.getBoundingClientRect();
        const margin = 42;
        let delta = 0;
        if (y < rect.top + margin) delta = -14;
        else if (y > rect.bottom - margin) delta = 14;
        if (delta) scroller.scrollBy({ top: delta, behavior: 'auto' });
    }

    globalThis.cleanupHistoryCollectionPointerDrag = function cleanupHistoryCollectionPointerDrag() {
        const drag = historyCollectionPointerDrag;
        if (!drag) return null;
        document.removeEventListener('pointermove', drag.onMove);
        document.removeEventListener('pointerup', drag.onUp);
        document.removeEventListener('pointercancel', drag.onCancel);
        document.removeEventListener('mousemove', drag.onMouseMove);
        document.removeEventListener('mouseup', drag.onMouseUp);
        document.removeEventListener('touchmove', drag.onTouchMove);
        document.removeEventListener('touchend', drag.onTouchEnd);
        document.removeEventListener('touchcancel', drag.onTouchCancel);
        document.removeEventListener('keydown', drag.onKeydown);
        try {
            if (drag.pointerId !== null && drag.pointerId !== undefined) {
                drag.handle?.releasePointerCapture?.(drag.pointerId);
            }
        } catch (e) {
            /* 指针可能已被浏览器释放，忽略即可。 */
        }
        removeHistoryDragImage();
        drag.handle.classList.remove('dragging');
        document.body.classList.remove('history-collection-pointer-drag-active');
        historyCollectionPointerDrag = null;
        return drag;
    }

    globalThis.historyCollectionEventPoint = function historyCollectionEventPoint(event) {
        const touch = event.changedTouches?.[0] || event.touches?.[0];
        const x = touch?.clientX ?? event.clientX;
        const y = touch?.clientY ?? event.clientY;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        return { x, y };
    }

    globalThis.finishHistoryCollectionPointerDrag = async function finishHistoryCollectionPointerDrag(commit = false) {
        const drag = cleanupHistoryCollectionPointerDrag();
        if (!drag) return;
        const source = drag.sourceValue;
        const target = commit && drag.active ? drag.currentDrop : null;
        if (!target || !source) {
            if (drag.active) finishHistoryCollectionDrag();
            return;
        }
        if (source === target.value) {
            setHistoryDropFeedback('分组顺序未变化。', 'ok');
            finishHistoryCollectionDrag();
            return;
        }
        historyCollectionDragState.pending = true;
        try {
            const changed = await reorderHistoryCollectionValue(source, target.value, target.position, drag.allCollections);
            setHistoryDropFeedback(changed ? `已调整「${source}」的位置。` : '分组顺序未变化。', 'ok');
        } catch (e) {
            setHistoryDropFeedback(`调整分组顺序失败: ${e.message}`, 'error');
        } finally {
            finishHistoryCollectionDrag();
            renderHistoryManager();
        }
    }
