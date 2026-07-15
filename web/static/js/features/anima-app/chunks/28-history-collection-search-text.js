/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    HISTORY_CONFIG_GROUP_DRAG_MIME,
    HISTORY_TASK_DRAG_MIME,
    HISTORY_UNGROUPED_COLLECTION_KEY,
} from '../helpers/app-constants.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    closeHistoryDropPopover,
    configureHistoryCollectionDragBridge,
    historyDraggedTasksAlreadyInCollection,
    reorderHistoryConfigGroupValue,
    setHistoryDropFeedback,
} from '../helpers/history-collection-drag-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    applyHistoryTaskIdsToCollection,
    configureHistoryCollectionsBridge,
    configGroupKey,
    enrichHistoryCollection,
    groupHistoryTasksByCollection,
    historyCollectionComparator,
    historyCollectionStorageKey,
    historyGroupDisplayLabel,
    historyTaskSearchText,
    historyTaskIds,
} from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { renderHistoryManager, uniqueStringList } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

const historyState = getHistoryState();

    export function historyCollectionSearchText(collection) {
        return [
            collection.label,
            collection.value,
            ...collection.groups.map((group) => historyGroupDisplayLabel(group)),
            ...collection.groups.map((group) => group.source_label || group.history_source_config_file || ''),
            ...collection.tasks.map((task) => historyTaskSearchText(task)),
        ].filter(Boolean).join('\n').toLowerCase();
    }

    export function historyConfigGroupSearchText(group) {
        return [
            historyGroupDisplayLabel(group),
            group.source_label,
            group.history_source_config_file,
            group.fallback_group_label,
            ...group.tasks.map((task) => historyTaskSearchText(task)),
        ].filter(Boolean).join('\n').toLowerCase();
    }

    export function createEmptyHistoryCollection(value = '') {
        const clean = String(value || '').trim();
        return enrichHistoryCollection({
            key: clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY,
            label: clean || '未分类',
            value: clean,
            is_ungrouped: !clean,
            tasks: [],
        });
    }

    export function createHistoryCollectionSearchEmptyCollection() {
        return enrichHistoryCollection({
            key: 'collection:__search_empty__',
            label: '无匹配分组',
            value: '__search_empty__',
            is_ungrouped: false,
            tasks: [],
        });
    }

    export function normalizeHistoryCollectionForWorkbench(collection) {
        const clean = String(collection?.value || '').trim();
        return enrichHistoryCollection({
            ...(collection || {}),
            key: clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY,
            label: clean || '未分类',
            value: clean,
            is_ungrouped: !clean,
            tasks: collection?.tasks || [],
        });
    }

    export function historyCollectionsForWorkbench(tasks) {
        const byKey = new Map();
        for (const collection of groupHistoryTasksByCollection(tasks || [])) {
            const normalized = normalizeHistoryCollectionForWorkbench(collection);
            byKey.set(normalized.key, normalized);
        }
        if (!byKey.has(HISTORY_UNGROUPED_COLLECTION_KEY)) {
            const ungrouped = createEmptyHistoryCollection();
            byKey.set(ungrouped.key, ungrouped);
        }
        for (const value of uniqueStringList(historyState.historyCollectionSettings.collection_order || [])) {
            const clean = String(value || '').trim();
            const key = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;
            if (clean && !byKey.has(key)) {
                byKey.set(key, createEmptyHistoryCollection(clean));
            }
        }
        return Array.from(byKey.values()).sort(historyCollectionComparator);
    }

    export function historyCollectionSelectOptions() {
        const collections = historyCollectionsForWorkbench(historyState.historyTasks);
        return collections.map((collection) => ({
            key: collection.key,
            label: collection.label,
            value: collection.value || '',
            task_count: collection.tasks.length,
            group_count: collection.groups.length,
            search_text: historyCollectionSearchText(collection),
        }));
    }

    export function historyCollectionOptionSearchText(item) {
        return [
            item.label,
            item.value,
            item.search_text,
        ].filter(Boolean).join('\n').toLowerCase();
    }

    export function historyCollectionsPanelTitle(titleText, metaText) {
        const title = document.createElement('div');
        title.className = 'history-collections-panel-title';
        const titleEl = document.createElement('strong');
        titleEl.textContent = titleText;
        const meta = document.createElement('span');
        meta.textContent = metaText;
        title.append(titleEl, meta);
        return title;
    }

    export function createHistoryCollectionsToolbarButton(label, handler, disabled = false, tone = '') {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = ['task-history-action', tone].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.disabled = Boolean(disabled);
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            if (!btn.disabled) handler(event);
        });
        return btn;
    }

    export function stopHistoryGroupButtonPropagation(button) {
        button.addEventListener('click', (event) => event.stopPropagation(), { capture: true });
        return button;
    }

    export function historyDragTaskIdsForGroup(group) {
        const groupIds = historyTaskIds(group?.tasks || []);
        const visible = new Set(historyState.historyCurrentVisibleTaskIds);
        const selectedIds = Array.from(historyState.selectedHistoryTaskIds)
            .filter((id) => id && (!visible.size || visible.has(id)));
        if (selectedIds.some((id) => groupIds.includes(id))) {
            return selectedIds;
        }
        return groupIds;
    }

    export function createHistoryDragImage(count) {
        removeHistoryDragImage();
        const image = document.createElement('div');
        image.className = 'history-drag-image';
        image.textContent = `${count} 条任务`;
        document.body.appendChild(image);
        historyState.historyDragImageElement = image;
        return image;
    }

    export function removeHistoryDragImage() {
        if (historyState.historyDragImageElement?.parentNode) {
            historyState.historyDragImageElement.parentNode.removeChild(historyState.historyDragImageElement);
        }
        historyState.historyDragImageElement = null;
    }

    export function canBeginHistoryConfigGroupDrag(group) {
        return Boolean(!historyState.historyDragState.pending && !historyState.historyConfigGroupSortState.pending && historyTaskIds(group?.tasks || []).length);
    }

    export function beginHistoryConfigGroupDrag(event, group, options = {}) {
        if (historyState.historyDragState.pending) {
            event.preventDefault();
            return;
        }
        const taskIds = uniqueStringList(historyDragTaskIdsForGroup(group));
        if (!taskIds.length) {
            event.preventDefault();
            return;
        }
        closeHistoryDropPopover(false);
        historyState.historyDragState = {
            ...historyState.historyDragState,
            active: true,
            taskIds,
            sourceGroupKey: configGroupKey(group),
            activeDropTarget: '',
            popover: {
                open: false,
                x: 0,
                y: 0,
                taskIds: [],
                defaultName: '',
            },
        };
        const payload = JSON.stringify(taskIds);
        const groupKey = configGroupKey(group);
        const collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        historyState.historyConfigGroupSortState = {
            active: Boolean(groupKey),
            sourceKey: groupKey,
            collectionKey,
            activeDropTarget: '',
            dropPosition: 'after',
            pending: false,
        };
        if (event.dataTransfer) {
            event.dataTransfer.setData(HISTORY_TASK_DRAG_MIME, payload);
            event.dataTransfer.setData(HISTORY_CONFIG_GROUP_DRAG_MIME, JSON.stringify({ groupKey, collectionKey }));
            event.dataTransfer.setData('text/plain', payload);
            event.dataTransfer.effectAllowed = 'move';
            const dragImage = createHistoryDragImage(taskIds.length);
            event.dataTransfer.setDragImage(dragImage, 18, 18);
        }
        document.querySelector('.history-collections-workbench')?.classList.add('dragging');
    }

    export function finishHistoryDrag() {
        removeHistoryDragImage();
        removeHistoryConfigGroupDropPreview();
        historyState.historyDragState.active = false;
        historyState.historyDragState.taskIds = [];
        historyState.historyDragState.sourceGroupKey = '';
        historyState.historyDragState.activeDropTarget = '';
        historyState.historyConfigGroupSortState = {
            active: false,
            sourceKey: '',
            collectionKey: '',
            activeDropTarget: '',
            dropPosition: 'after',
            pending: false,
        };
        document.querySelectorAll('.history-collection-card.drop-active').forEach((item) => {
            item.classList.remove('drop-active');
        });
        document.querySelectorAll('.history-config-group-card.config-sort-active, .history-config-group-card.config-sort-source').forEach((item) => {
            item.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after', 'config-sort-source');
        });
        document.querySelector('.history-collections-workbench')?.classList.remove('dragging');
    }

    export function readHistoryDraggedConfigGroup(event) {
        try {
            const raw = event?.dataTransfer?.getData(HISTORY_CONFIG_GROUP_DRAG_MIME);
            if (raw) {
                const parsed = JSON.parse(raw);
                return {
                    groupKey: String(parsed?.groupKey || '').trim(),
                    collectionKey: historyCollectionStorageKey(parsed?.collectionKey || '__all__'),
                };
            }
        } catch (e) {
            /* DataTransfer 在部分浏览器只能在 drop 阶段读取。 */
        }
        return {
            groupKey: historyState.historyConfigGroupSortState.sourceKey || '',
            collectionKey: historyState.historyConfigGroupSortState.collectionKey || '__all__',
        };
    }

    export function historyConfigGroupDropPosition(event, element) {
        const rect = element?.getBoundingClientRect?.();
        if (!rect) return 'after';
        return Number(event?.clientY || 0) < rect.top + (rect.height / 2) ? 'before' : 'after';
    }

    export function removeHistoryConfigGroupDropPreview() {
        if (historyState.historyConfigGroupDropPreviewElement?.parentNode) {
            historyState.historyConfigGroupDropPreviewElement.parentNode.removeChild(historyState.historyConfigGroupDropPreviewElement);
        }
        historyState.historyConfigGroupDropPreviewElement = null;
    }

    export function ensureHistoryConfigGroupDropPreview() {
        if (historyState.historyConfigGroupDropPreviewElement?.isConnected) return historyState.historyConfigGroupDropPreviewElement;
        const preview = document.createElement('div');
        preview.className = 'history-config-group-drop-preview';
        preview.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.textContent = '释放后插入到这里';
        preview.appendChild(label);
        historyState.historyConfigGroupDropPreviewElement = preview;
        return preview;
    }

    export function placeHistoryConfigGroupDropPreview(element, position) {
        const parent = element?.parentElement;
        if (!parent) return;
        const preview = ensureHistoryConfigGroupDropPreview();
        const placement = position === 'before' ? 'before' : 'after';
        const parentStyle = window.getComputedStyle(parent);
        const gap = Number.parseFloat(parentStyle.rowGap || parentStyle.gap || '0') || 0;
        const top = placement === 'before'
            ? Math.max(0, element.offsetTop - (gap / 2))
            : element.offsetTop + element.offsetHeight + (gap / 2);
        preview.dataset.position = placement;
        preview.style.top = `${top}px`;
        if (preview.parentElement !== parent) parent.appendChild(preview);
    }

    export function setHistoryConfigGroupSortTarget(targetKey, position, element) {
        historyState.historyConfigGroupSortState.activeDropTarget = `config-sort:${targetKey || ''}`;
        historyState.historyConfigGroupSortState.dropPosition = position === 'before' ? 'before' : 'after';
        document.querySelectorAll('.history-config-group-card.config-sort-active').forEach((item) => {
            if (item !== element) item.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
        });
        element?.classList.add(
            'config-sort-active',
            historyState.historyConfigGroupSortState.dropPosition === 'before' ? 'config-sort-before' : 'config-sort-after',
        );
        placeHistoryConfigGroupDropPreview(element, historyState.historyConfigGroupSortState.dropPosition);
    }

    export function clearHistoryConfigGroupSortTarget(targetKey, element) {
        if (historyState.historyConfigGroupSortState.activeDropTarget === `config-sort:${targetKey || ''}`) {
            historyState.historyConfigGroupSortState.activeDropTarget = '';
            removeHistoryConfigGroupDropPreview();
        }
        element?.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
    }

    export function clearHistoryConfigGroupSortIndicators() {
        historyState.historyConfigGroupSortState.activeDropTarget = '';
        removeHistoryConfigGroupDropPreview();
        document.querySelectorAll('.history-config-group-card.config-sort-active').forEach((item) => {
            item.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
        });
    }

    export function historyConfigGroupOrderDragEnter(event, group, element, options = {}) {
        if (!historyState.historyConfigGroupSortState.active || historyState.historyConfigGroupSortState.pending) return false;
        const source = readHistoryDraggedConfigGroup(event);
        const targetKey = configGroupKey(group);
        const collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        if (!source.groupKey || !targetKey || source.groupKey === targetKey || source.collectionKey !== collectionKey) return false;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        const position = historyConfigGroupDropPosition(event, element);
        setHistoryConfigGroupSortTarget(targetKey, position, element);
        return true;
    }

    export function historyConfigGroupOrderDragLeave(event, group, element) {
        if (!historyState.historyConfigGroupSortState.active) return false;
        if (element?.contains(event.relatedTarget)) return true;
        if (historyState.historyConfigGroupDropPreviewElement?.contains(event.relatedTarget)) return true;
        if (event.relatedTarget instanceof Element && event.relatedTarget.closest('.history-config-group-card-list')) return true;
        clearHistoryConfigGroupSortTarget(configGroupKey(group), element);
        return true;
    }

    export function historyConfigGroupForPointerCard(card, groups = []) {
        if (!card) return null;
        const key = String(card.dataset.configGroupKey || '').trim();
        return (groups || []).find((group) => configGroupKey(group) === key) || null;
    }

    export function historyConfigGroupPointerTargetForCard(card, x, y, groups = [], collection = null) {
        const group = historyConfigGroupForPointerCard(card, groups);
        if (!group) return null;
        return {
            element: card,
            group,
            key: configGroupKey(group),
            collectionKey: historyCollectionStorageKey(collection || '__all__'),
            position: historyConfigGroupDropPosition({ clientY: y }, card),
        };
    }

    export function nearestHistoryConfigGroupPointerTarget(x, y, groups = [], collection = null) {
        let best = null;
        document.querySelectorAll('.history-config-group-card').forEach((card) => {
            if (!card?.isConnected) return;
            const rect = card.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
            const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
            const distance = Math.hypot(dx, dy);
            const maxDistance = Math.max(24, Math.min(76, rect.height * 0.9));
            if (distance > maxDistance || (best && distance >= best.distance)) return;
            const target = historyConfigGroupPointerTargetForCard(card, x, y, groups, collection);
            if (target) best = { ...target, distance };
        });
        if (!best) return null;
        const { distance, ...target } = best;
        return target;
    }

    export function historyConfigGroupPointerTargetFromPoint(x, y, groups = [], collection = null) {
        const origin = document.elementFromPoint(x, y);
        const card = origin instanceof Element ? origin.closest('.history-config-group-card') : null;
        return historyConfigGroupPointerTargetForCard(card, x, y, groups, collection)
            || nearestHistoryConfigGroupPointerTarget(x, y, groups, collection);
    }

    export function historyCollectionDropTargetFromPoint(x, y) {
        const origin = document.elementFromPoint(x, y);
        return origin instanceof Element ? origin.closest('.history-collection-card.nav-card') : null;
    }

    export function cleanupHistoryConfigGroupPointerDrag() {
        const drag = historyState.historyConfigGroupPointerDrag;
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
        drag.handle?.classList.remove('dragging');
        document.body.classList.remove('history-config-group-pointer-drag-active');
        historyState.historyConfigGroupPointerDrag = null;
        return drag;
    }

    export async function finishHistoryConfigGroupPointerDrag(commit = false) {
        const drag = cleanupHistoryConfigGroupPointerDrag();
        if (!drag) return;
        const target = commit && drag.active ? drag.currentDrop : null;
        const collectionTarget = commit && drag.active ? drag.currentCollectionDrop : null;
        if (collectionTarget && drag.taskIds.length) {
            await dropHistoryTasksToCollectionLikePointer(collectionTarget, drag.taskIds);
            return;
        }
        if (!target || !drag.sourceKey) {
            if (drag.active) finishHistoryDrag();
            return;
        }
        if (target.key === drag.sourceKey) {
            setHistoryDropFeedback('配置分组顺序未变化。', 'ok');
            finishHistoryDrag();
            return;
        }
        historyState.historyConfigGroupSortState.pending = true;
        try {
            const changed = await reorderHistoryConfigGroupValue(
                drag.sourceKey,
                target.key,
                target.position,
                drag.groups,
                drag.collection,
            );
            setHistoryDropFeedback(changed ? '已调整配置分组顺序。' : '配置分组顺序未变化。', 'ok');
        } catch (e) {
            setHistoryDropFeedback(`调整配置分组顺序失败: ${e.message}`, 'error');
        } finally {
            finishHistoryDrag();
            renderHistoryManager();
        }
    }

    export async function dropHistoryTasksToCollectionLikePointer(targetCard, taskIds) {
        const groupValue = String(targetCard.dataset.collectionValue || '').trim();
        const label = targetCard.querySelector('.history-collection-card-title strong')?.textContent || groupValue || '未分类';
        const clean = groupValue;
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

    configureHistoryCollectionsBridge({
        historyCollectionSearchText,
        historyConfigGroupSearchText,
        createEmptyHistoryCollection,
        createHistoryCollectionSearchEmptyCollection,
        normalizeHistoryCollectionForWorkbench,
        historyCollectionsForWorkbench,
        historyCollectionSelectOptions,
        historyCollectionOptionSearchText,
        historyCollectionsPanelTitle,
        createHistoryCollectionsToolbarButton,
        stopHistoryGroupButtonPropagation,
    });

    configureHistoryCollectionDragBridge({
        historyDragTaskIdsForGroup,
        createHistoryDragImage,
        removeHistoryDragImage,
        canBeginHistoryConfigGroupDrag,
        beginHistoryConfigGroupDrag,
        finishHistoryDrag,
        readHistoryDraggedConfigGroup,
        historyConfigGroupDropPosition,
        removeHistoryConfigGroupDropPreview,
        ensureHistoryConfigGroupDropPreview,
        placeHistoryConfigGroupDropPreview,
        setHistoryConfigGroupSortTarget,
        clearHistoryConfigGroupSortTarget,
        clearHistoryConfigGroupSortIndicators,
        historyConfigGroupOrderDragEnter,
        historyConfigGroupOrderDragLeave,
        historyConfigGroupForPointerCard,
        historyConfigGroupPointerTargetForCard,
        nearestHistoryConfigGroupPointerTarget,
        historyConfigGroupPointerTargetFromPoint,
        historyCollectionDropTargetFromPoint,
        cleanupHistoryConfigGroupPointerDrag,
        finishHistoryConfigGroupPointerDrag,
        dropHistoryTasksToCollectionLikePointer,
    });
