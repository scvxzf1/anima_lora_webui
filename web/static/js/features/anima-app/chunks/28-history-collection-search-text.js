/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.historyCollectionSearchText = function historyCollectionSearchText(collection) {
        return [
            collection.label,
            collection.value,
            ...collection.groups.map((group) => historyGroupDisplayLabel(group)),
            ...collection.groups.map((group) => group.source_label || group.history_source_config_file || ''),
            ...collection.tasks.map((task) => historyTaskSearchText(task)),
        ].filter(Boolean).join('\n').toLowerCase();
    }

    globalThis.historyConfigGroupSearchText = function historyConfigGroupSearchText(group) {
        return [
            historyGroupDisplayLabel(group),
            group.source_label,
            group.history_source_config_file,
            group.fallback_group_label,
            ...group.tasks.map((task) => historyTaskSearchText(task)),
        ].filter(Boolean).join('\n').toLowerCase();
    }

    globalThis.createEmptyHistoryCollection = function createEmptyHistoryCollection(value = '') {
        const clean = String(value || '').trim();
        return enrichHistoryCollection({
            key: clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY,
            label: clean || '未分类',
            value: clean,
            is_ungrouped: !clean,
            tasks: [],
        });
    }

    globalThis.createHistoryCollectionSearchEmptyCollection = function createHistoryCollectionSearchEmptyCollection() {
        return enrichHistoryCollection({
            key: 'collection:__search_empty__',
            label: '无匹配分组',
            value: '__search_empty__',
            is_ungrouped: false,
            tasks: [],
        });
    }

    globalThis.normalizeHistoryCollectionForWorkbench = function normalizeHistoryCollectionForWorkbench(collection) {
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

    globalThis.historyCollectionsForWorkbench = function historyCollectionsForWorkbench(tasks) {
        const byKey = new Map();
        for (const collection of groupHistoryTasksByCollection(tasks || [])) {
            const normalized = normalizeHistoryCollectionForWorkbench(collection);
            byKey.set(normalized.key, normalized);
        }
        if (!byKey.has(HISTORY_UNGROUPED_COLLECTION_KEY)) {
            const ungrouped = createEmptyHistoryCollection();
            byKey.set(ungrouped.key, ungrouped);
        }
        for (const value of uniqueStringList(historyCollectionSettings.collection_order || [])) {
            const clean = String(value || '').trim();
            const key = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;
            if (clean && !byKey.has(key)) {
                byKey.set(key, createEmptyHistoryCollection(clean));
            }
        }
        return Array.from(byKey.values()).sort(historyCollectionComparator);
    }

    globalThis.historyCollectionSelectOptions = function historyCollectionSelectOptions() {
        const collections = historyCollectionsForWorkbench(historyTasks);
        return collections.map((collection) => ({
            key: collection.key,
            label: collection.label,
            value: collection.value || '',
            task_count: collection.tasks.length,
            group_count: collection.groups.length,
            search_text: historyCollectionSearchText(collection),
        }));
    }

    globalThis.historyCollectionOptionSearchText = function historyCollectionOptionSearchText(item) {
        return [
            item.label,
            item.value,
            item.search_text,
        ].filter(Boolean).join('\n').toLowerCase();
    }

    globalThis.historyCollectionsPanelTitle = function historyCollectionsPanelTitle(titleText, metaText) {
        const title = document.createElement('div');
        title.className = 'history-collections-panel-title';
        const titleEl = document.createElement('strong');
        titleEl.textContent = titleText;
        const meta = document.createElement('span');
        meta.textContent = metaText;
        title.append(titleEl, meta);
        return title;
    }

    globalThis.createHistoryCollectionsToolbarButton = function createHistoryCollectionsToolbarButton(label, handler, disabled = false, tone = '') {
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

    globalThis.stopHistoryGroupButtonPropagation = function stopHistoryGroupButtonPropagation(button) {
        button.addEventListener('click', (event) => event.stopPropagation(), { capture: true });
        return button;
    }

    globalThis.historyDragTaskIdsForGroup = function historyDragTaskIdsForGroup(group) {
        const groupIds = historyTaskIds(group?.tasks || []);
        const visible = new Set(historyCurrentVisibleTaskIds);
        const selectedIds = Array.from(selectedHistoryTaskIds)
            .filter((id) => id && (!visible.size || visible.has(id)));
        if (selectedIds.some((id) => groupIds.includes(id))) {
            return selectedIds;
        }
        return groupIds;
    }

    globalThis.createHistoryDragImage = function createHistoryDragImage(count) {
        removeHistoryDragImage();
        const image = document.createElement('div');
        image.className = 'history-drag-image';
        image.textContent = `${count} 条任务`;
        document.body.appendChild(image);
        historyDragImageElement = image;
        return image;
    }

    globalThis.removeHistoryDragImage = function removeHistoryDragImage() {
        if (historyDragImageElement?.parentNode) {
            historyDragImageElement.parentNode.removeChild(historyDragImageElement);
        }
        historyDragImageElement = null;
    }

    globalThis.canBeginHistoryConfigGroupDrag = function canBeginHistoryConfigGroupDrag(group) {
        return Boolean(!historyDragState.pending && !historyConfigGroupSortState.pending && historyTaskIds(group?.tasks || []).length);
    }

    globalThis.beginHistoryConfigGroupDrag = function beginHistoryConfigGroupDrag(event, group, options = {}) {
        if (historyDragState.pending) {
            event.preventDefault();
            return;
        }
        const taskIds = uniqueStringList(historyDragTaskIdsForGroup(group));
        if (!taskIds.length) {
            event.preventDefault();
            return;
        }
        closeHistoryDropPopover(false);
        historyDragState = {
            ...historyDragState,
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
        historyConfigGroupSortState = {
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

    globalThis.finishHistoryDrag = function finishHistoryDrag() {
        removeHistoryDragImage();
        removeHistoryConfigGroupDropPreview();
        historyDragState.active = false;
        historyDragState.taskIds = [];
        historyDragState.sourceGroupKey = '';
        historyDragState.activeDropTarget = '';
        historyConfigGroupSortState = {
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

    globalThis.readHistoryDraggedConfigGroup = function readHistoryDraggedConfigGroup(event) {
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
            groupKey: historyConfigGroupSortState.sourceKey || '',
            collectionKey: historyConfigGroupSortState.collectionKey || '__all__',
        };
    }

    globalThis.historyConfigGroupDropPosition = function historyConfigGroupDropPosition(event, element) {
        const rect = element?.getBoundingClientRect?.();
        if (!rect) return 'after';
        return Number(event?.clientY || 0) < rect.top + (rect.height / 2) ? 'before' : 'after';
    }

    globalThis.removeHistoryConfigGroupDropPreview = function removeHistoryConfigGroupDropPreview() {
        if (historyConfigGroupDropPreviewElement?.parentNode) {
            historyConfigGroupDropPreviewElement.parentNode.removeChild(historyConfigGroupDropPreviewElement);
        }
        historyConfigGroupDropPreviewElement = null;
    }

    globalThis.ensureHistoryConfigGroupDropPreview = function ensureHistoryConfigGroupDropPreview() {
        if (historyConfigGroupDropPreviewElement?.isConnected) return historyConfigGroupDropPreviewElement;
        const preview = document.createElement('div');
        preview.className = 'history-config-group-drop-preview';
        preview.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.textContent = '释放后插入到这里';
        preview.appendChild(label);
        historyConfigGroupDropPreviewElement = preview;
        return preview;
    }

    globalThis.placeHistoryConfigGroupDropPreview = function placeHistoryConfigGroupDropPreview(element, position) {
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

    globalThis.setHistoryConfigGroupSortTarget = function setHistoryConfigGroupSortTarget(targetKey, position, element) {
        historyConfigGroupSortState.activeDropTarget = `config-sort:${targetKey || ''}`;
        historyConfigGroupSortState.dropPosition = position === 'before' ? 'before' : 'after';
        document.querySelectorAll('.history-config-group-card.config-sort-active').forEach((item) => {
            if (item !== element) item.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
        });
        element?.classList.add(
            'config-sort-active',
            historyConfigGroupSortState.dropPosition === 'before' ? 'config-sort-before' : 'config-sort-after',
        );
        placeHistoryConfigGroupDropPreview(element, historyConfigGroupSortState.dropPosition);
    }

    globalThis.clearHistoryConfigGroupSortTarget = function clearHistoryConfigGroupSortTarget(targetKey, element) {
        if (historyConfigGroupSortState.activeDropTarget === `config-sort:${targetKey || ''}`) {
            historyConfigGroupSortState.activeDropTarget = '';
            removeHistoryConfigGroupDropPreview();
        }
        element?.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
    }

    globalThis.clearHistoryConfigGroupSortIndicators = function clearHistoryConfigGroupSortIndicators() {
        historyConfigGroupSortState.activeDropTarget = '';
        removeHistoryConfigGroupDropPreview();
        document.querySelectorAll('.history-config-group-card.config-sort-active').forEach((item) => {
            item.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
        });
    }

    globalThis.historyConfigGroupOrderDragEnter = function historyConfigGroupOrderDragEnter(event, group, element, options = {}) {
        if (!historyConfigGroupSortState.active || historyConfigGroupSortState.pending) return false;
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

    globalThis.historyConfigGroupOrderDragLeave = function historyConfigGroupOrderDragLeave(event, group, element) {
        if (!historyConfigGroupSortState.active) return false;
        if (element?.contains(event.relatedTarget)) return true;
        if (historyConfigGroupDropPreviewElement?.contains(event.relatedTarget)) return true;
        if (event.relatedTarget instanceof Element && event.relatedTarget.closest('.history-config-group-card-list')) return true;
        clearHistoryConfigGroupSortTarget(configGroupKey(group), element);
        return true;
    }

    globalThis.historyConfigGroupForPointerCard = function historyConfigGroupForPointerCard(card, groups = []) {
        if (!card) return null;
        const key = String(card.dataset.configGroupKey || '').trim();
        return (groups || []).find((group) => configGroupKey(group) === key) || null;
    }

    globalThis.historyConfigGroupPointerTargetForCard = function historyConfigGroupPointerTargetForCard(card, x, y, groups = [], collection = null) {
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

    globalThis.nearestHistoryConfigGroupPointerTarget = function nearestHistoryConfigGroupPointerTarget(x, y, groups = [], collection = null) {
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

    globalThis.historyConfigGroupPointerTargetFromPoint = function historyConfigGroupPointerTargetFromPoint(x, y, groups = [], collection = null) {
        const origin = document.elementFromPoint(x, y);
        const card = origin instanceof Element ? origin.closest('.history-config-group-card') : null;
        return historyConfigGroupPointerTargetForCard(card, x, y, groups, collection)
            || nearestHistoryConfigGroupPointerTarget(x, y, groups, collection);
    }

    globalThis.historyCollectionDropTargetFromPoint = function historyCollectionDropTargetFromPoint(x, y) {
        const origin = document.elementFromPoint(x, y);
        return origin instanceof Element ? origin.closest('.history-collection-card.nav-card') : null;
    }

    globalThis.cleanupHistoryConfigGroupPointerDrag = function cleanupHistoryConfigGroupPointerDrag() {
        const drag = historyConfigGroupPointerDrag;
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
        historyConfigGroupPointerDrag = null;
        return drag;
    }

    globalThis.finishHistoryConfigGroupPointerDrag = async function finishHistoryConfigGroupPointerDrag(commit = false) {
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
        historyConfigGroupSortState.pending = true;
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

    globalThis.dropHistoryTasksToCollectionLikePointer = async function dropHistoryTasksToCollectionLikePointer(targetCard, taskIds) {
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
        historyDragState.pending = true;
        document.querySelector('.history-collections-workbench')?.classList.add('drop-pending');
        try {
            const res = await applyHistoryTaskIdsToCollection(taskIds, clean, { clearSelection: true });
            if (res === null) {
                setHistoryDropFeedback('移动失败，列表未更改。', 'error');
            } else {
                selectedHistoryCollectionKey = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;
                setHistoryDropFeedback(`${taskIds.length} 条任务已移动到${clean ? `「${label || clean}」` : '未分类'}。`, 'ok');
            }
        } catch (e) {
            setHistoryDropFeedback(`移动失败: ${e.message}`, 'error');
        } finally {
            historyDragState.pending = false;
            finishHistoryDrag();
            renderHistoryManager();
        }
    }
