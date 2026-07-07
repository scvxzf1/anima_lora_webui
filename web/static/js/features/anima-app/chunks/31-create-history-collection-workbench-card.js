/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { HISTORY_UNGROUPED_COLLECTION_KEY } from '../helpers/app-constants.js?v=module-bootstrap-20260707-93';
import {
    beginHistoryCollectionDrag,
    beginHistoryConfigGroupDrag,
    dropHistoryConfigGroupToSort,
    dropHistoryCollectionToSort,
    dropHistoryTasksToCollection,
    finishHistoryCollectionDrag,
    finishHistoryCollectionPointerDrag,
    finishHistoryConfigGroupPointerDrag,
    finishHistoryDrag,
    historyConfigGroupOrderDragEnter,
    historyConfigGroupOrderDragLeave,
    historyCollectionOrderDragEnter,
    historyCollectionOrderDragLeave,
    historyDropTargetDragEnter,
    historyDropTargetDragLeave,
    placeHistoryConfigGroupDropPreview,
    startHistoryConfigGroupMouseDrag,
    startHistoryConfigGroupPointerDrag,
    startHistoryConfigGroupTouchDrag,
    startHistoryCollectionMouseDrag,
    startHistoryCollectionPointerDrag,
    startHistoryCollectionTouchDrag,
} from '../helpers/history-collection-drag-bridge.js?v=module-bootstrap-20260707-93';
import {
    applySelectedHistoryTasksToCollection,
    clearHistoryCollection,
    clearHistoryCollectionForTasks,
    commonHistoryCollectionValue,
    configureHistoryCollectionsBridge,
    configGroupKey,
    createHistoryConfigGroupMergeButton,
    createHistoryConfigGroupPreviewButton,
    createHistoryManagerGroupButton,
    groupHistoryTasks,
    historyCompactGroupMetaParts,
    historyContinueLabel,
    historyGroupDisplayLabel,
    historyCollectionsForWorkbench,
    historyManagerGroupMetaParts,
    historyQueueLabel,
    historyResumeLabel,
    historyTaskCollectionKey,
    historyTaskCollectionLabel,
    historyTaskDisplayName, historyTaskIds, historyTaskIsArchived, historyTasksAllSelected,
    compactHistoryContinueLabel, compactHistoryPathLabel, compactHistoryQueueLabel, compactHistoryResumeLabel,
    createHistoryManagerRow, createHistoryMoreActions, historyTaskSortComparator,
    renameHistoryCollection,
    setHistoryCollectionForTasks,
    setHistoryCollectionForTasksDirect,
    toggleHistoryTaskSelection,
} from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260707-93';
import {
    archiveHistoryTask,
    createHistoryActionButton,
    createHistoryTaskConfigButton,
    createHistoryTaskPreviewButton,
    deleteHistoryTask,
    loadHistoryTask,
} from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260707-93';
import { historyStateLabel } from '../helpers/history-timeline-bridge.js?v=module-bootstrap-20260707-93';
import { renderHistoryManager, saveHistoryCollectionSettings, uniqueStringList } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260707-93';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260707-93';

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
            historyState.selectedHistoryCollectionKey = collection.key;
            renderHistoryManager();
        });
        card.addEventListener('keydown', (event) => {
            if (event.target !== card) return;
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            historyState.selectedHistoryCollectionKey = collection.key;
            renderHistoryManager();
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

    export function createHistoryConfigGroupWorkbenchCard(group, splitCollections, options = {}) {
        const card = document.createElement('article');
        card.className = ['history-config-group-card', historyTasksAllSelected(group.tasks) ? 'selected' : ''].filter(Boolean).join(' ');
        card.classList.add('draggable');
        const groupKey = configGroupKey(group);
        card.dataset.configGroupKey = groupKey;
        card.dataset.collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        if (historyState.historyConfigGroupSortState.sourceKey && historyState.historyConfigGroupSortState.sourceKey === groupKey) {
            card.classList.add('config-sort-source');
        }
        if (
            historyState.historyConfigGroupSortState.active
            && historyState.historyConfigGroupSortState.activeDropTarget === `config-sort:${groupKey}`
        ) {
            card.classList.add(
                'config-sort-active',
                historyState.historyConfigGroupSortState.dropPosition === 'before' ? 'config-sort-before' : 'config-sort-after',
            );
            requestAnimationFrame(() => {
                if (card.isConnected && historyState.historyConfigGroupSortState.activeDropTarget === `config-sort:${groupKey}`) {
                    placeHistoryConfigGroupDropPreview(card, historyState.historyConfigGroupSortState.dropPosition);
                }
            });
        }
        card.addEventListener('dragenter', (event) => {
            historyConfigGroupOrderDragEnter(event, group, card, options);
        });
        card.addEventListener('dragover', (event) => {
            historyConfigGroupOrderDragEnter(event, group, card, options);
        });
        card.addEventListener('dragleave', (event) => {
            historyConfigGroupOrderDragLeave(event, group, card);
        });
        card.addEventListener('drop', async (event) => {
            if (await dropHistoryConfigGroupToSort(event, group, options)) return;
        });
        const ids = historyTaskIds(group.tasks);
        const selectedCount = ids.filter((id) => historyState.selectedHistoryTaskIds.has(id)).length;

        const select = document.createElement('label');
        select.className = 'history-config-group-select';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = ids.length > 0 && selectedCount === ids.length;
        checkbox.indeterminate = selectedCount > 0 && selectedCount < ids.length;
        checkbox.addEventListener('change', () => toggleHistoryTaskSelection(group.tasks));
        select.append(checkbox, document.createTextNode('选择分组'));

        const handle = document.createElement('button');
        handle.type = 'button';
        handle.className = 'history-drag-handle history-config-group-drag-handle';
        handle.textContent = '⋮⋮';
        handle.title = '拖拽配置分组调整顺序或移到右侧分组';
        handle.setAttribute('aria-label', '拖拽配置分组调整顺序或移到右侧分组');
        handle.draggable = true;
        handle.addEventListener('click', (event) => event.stopPropagation());
        handle.addEventListener('pointerdown', (event) => startHistoryConfigGroupPointerDrag(event, group, options, handle));
        handle.addEventListener('mousedown', (event) => {
            event.stopPropagation();
            if (!('PointerEvent' in window)) startHistoryConfigGroupMouseDrag(event, group, options, handle);
        });
        handle.addEventListener('touchstart', (event) => {
            event.stopPropagation();
            if (!('PointerEvent' in window)) startHistoryConfigGroupTouchDrag(event, group, options, handle);
        }, { passive: false });
        handle.addEventListener('dragstart', (event) => {
            if (historyState.historyConfigGroupPointerDrag) finishHistoryConfigGroupPointerDrag(false);
            beginHistoryConfigGroupDrag(event, group, options);
        });
        handle.addEventListener('dragend', () => finishHistoryDrag());

        if ((group.tasks || []).length === 1) {
            const task = group.tasks[0];
            card.classList.add('single-task');

            const main = document.createElement('button');
            main.type = 'button';
            main.className = 'history-config-group-card-main history-single-task-main';
            main.addEventListener('click', () => loadHistoryTask(task.id));

            const titleRow = document.createElement('span');
            titleRow.className = 'history-single-task-title-row';
            const title = document.createElement('strong');
            title.textContent = historyGroupDisplayLabel(group);
            title.title = title.textContent;
            const state = document.createElement('span');
            state.className = ['history-row-state', task.state || 'unknown'].join(' ');
            state.textContent = [
                task.job === 'preprocess' ? '预处理' : '训练',
                historyStateLabel(task.state),
                historyTaskIsArchived(task) ? '已归档' : '',
            ].filter(Boolean).join(' · ');
            titleRow.append(title, state);

            const taskName = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
            const sourceLabel = compactHistoryPathLabel(group.source_label || group.fallback_group_label || group.label);
            const timeText = `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`;
            const dataText = `${task.metric_count || 0} loss / ${task.log_count || 0} log`;
            const meta = document.createElement('span');
            meta.className = 'history-compact-meta';
            meta.textContent = [
                taskName && taskName !== title.textContent ? taskName : '',
                sourceLabel && sourceLabel !== title.textContent ? `源:${sourceLabel}` : '',
                compactHistoryQueueLabel(task),
                compactHistoryContinueLabel(task),
                compactHistoryResumeLabel(task),
                timeText,
                dataText,
            ].filter(Boolean).join(' · ');
            meta.title = [
                `任务: ${taskName}`,
                group.source_label ? `源配置: ${group.source_label}` : `配置组: ${group.fallback_group_label || group.label}`,
                historyQueueLabel(task),
                historyContinueLabel(task),
                historyResumeLabel(task),
                timeText,
                dataText,
            ].filter(Boolean).join(' · ');
            titleRow.appendChild(meta);
            main.appendChild(titleRow);

            const actions = document.createElement('div');
            actions.className = 'history-config-group-card-actions history-single-task-actions';
            if (historyState.historyCollectionWorkbenchTarget) {
                actions.append(
                    createHistoryManagerGroupButton(
                        '目标',
                        () => setHistoryCollectionForTasksDirect(group.tasks, historyState.historyCollectionWorkbenchTarget)
                    )
                );
            }
            if (task.job === 'training') {
                actions.append(createHistoryTaskPreviewButton(task));
            }
            actions.append(
                createHistoryActionButton('查看', () => loadHistoryTask(task.id)),
                createHistoryMoreActions([
                    createHistoryManagerGroupButton('置顶', () => moveHistoryConfigGroup(group, 'top', options.groups, options.collection)),
                    createHistoryManagerGroupButton('上移', () => moveHistoryConfigGroup(group, 'up', options.groups, options.collection)),
                    createHistoryManagerGroupButton('下移', () => moveHistoryConfigGroup(group, 'down', options.groups, options.collection)),
                    createHistoryManagerGroupButton('置底', () => moveHistoryConfigGroup(group, 'bottom', options.groups, options.collection)),
                    createHistoryTaskConfigButton(task),
                    createHistoryConfigGroupMergeButton(group),
                    createHistoryManagerGroupButton('设置分组', () => setHistoryCollectionForTasks(group.tasks, commonHistoryCollectionValue(group.tasks), historyGroupDisplayLabel(group))),
                    createHistoryManagerGroupButton('清除分组', () => clearHistoryCollectionForTasks(group.tasks, historyGroupDisplayLabel(group))),
                    createHistoryActionButton(historyTaskIsArchived(task) ? '取消归档' : '归档', () => archiveHistoryTask(task)),
                    createHistoryActionButton('删除', () => deleteHistoryTask(task), 'danger'),
                ]),
            );

            const head = document.createElement('div');
            head.className = 'history-config-group-card-head';
            head.append(select, handle, main, actions);
            card.appendChild(head);
            return card;
        }

        const main = document.createElement('div');
        main.className = 'history-config-group-card-main';
        const title = document.createElement('strong');
        title.textContent = historyGroupDisplayLabel(group);
        title.title = title.textContent;
        const collections = historyCollectionNamesForTasks(group.tasks);
        const split = splitCollections?.get(configGroupKey(group));
        const meta = document.createElement('span');
        meta.className = 'history-compact-meta';
        const sourceLabel = compactHistoryPathLabel(group.source_label || group.fallback_group_label || group.label);
        meta.textContent = historyCompactGroupMetaParts(group.tasks, [
            sourceLabel && sourceLabel !== title.textContent ? `源:${sourceLabel}` : '',
            split && split.size > 1 ? `跨 ${split.size} 组` : '',
        ]).join(' · ');
        meta.title = historyManagerGroupMetaParts(group.tasks, [
            group.source_label ? `源配置: ${group.source_label}` : `配置组: ${group.fallback_group_label || group.label}`,
            collections.length ? `当前分组: ${collections.join(' / ')}` : '当前分组: 未分类',
            split && split.size > 1 ? `分布在 ${split.size} 个分组` : '',
        ]).join(' · ');
        main.append(title, meta);

        const actions = document.createElement('div');
        actions.className = 'history-config-group-card-actions';
        const trainingCount = group.tasks.filter((task) => task.job === 'training').length;
        if (historyState.historyCollectionWorkbenchTarget) {
            actions.append(
                createHistoryManagerGroupButton(
                    '目标',
                    () => setHistoryCollectionForTasksDirect(group.tasks, historyState.historyCollectionWorkbenchTarget)
                )
            );
        }
        if (trainingCount) {
            actions.append(createHistoryConfigGroupPreviewButton(group));
        }
        actions.append(createHistoryMoreActions([
            createHistoryManagerGroupButton('置顶', () => moveHistoryConfigGroup(group, 'top', options.groups, options.collection)),
            createHistoryManagerGroupButton('上移', () => moveHistoryConfigGroup(group, 'up', options.groups, options.collection)),
            createHistoryManagerGroupButton('下移', () => moveHistoryConfigGroup(group, 'down', options.groups, options.collection)),
            createHistoryManagerGroupButton('置底', () => moveHistoryConfigGroup(group, 'bottom', options.groups, options.collection)),
            createHistoryConfigGroupMergeButton(group),
            createHistoryManagerGroupButton('设置分组', () => setHistoryCollectionForTasks(group.tasks, commonHistoryCollectionValue(group.tasks), historyGroupDisplayLabel(group))),
            createHistoryManagerGroupButton('清除分组', () => clearHistoryCollectionForTasks(group.tasks, historyGroupDisplayLabel(group))),
        ]));

        const head = document.createElement('div');
        head.className = 'history-config-group-card-head';
        head.append(select, handle, main, actions);
        card.appendChild(head);
        const taskList = document.createElement('div');
        taskList.className = 'history-config-group-task-list';
        for (const task of group.tasks) {
            taskList.appendChild(createHistoryManagerRow(task));
        }
        card.appendChild(taskList);
        return card;
    }

    export function historyCollectionNamesForTasks(tasks) {
        const names = Array.from(new Set((tasks || []).map(historyTaskCollectionLabel).filter(Boolean)));
        return names.length ? names : ['未分类'];
    }

    export function moveItemInList(list, value, direction) {
        const out = uniqueStringList(list);
        const item = String(value || '').trim();
        const index = out.indexOf(item);
        if (!item || index < 0) return out;
        out.splice(index, 1);
        if (direction === 'top') out.unshift(item);
        else if (direction === 'bottom') out.push(item);
        else if (direction === 'up') out.splice(Math.max(0, index - 1), 0, item);
        else if (direction === 'down') out.splice(Math.min(out.length, index + 1), 0, item);
        else out.splice(index, 0, item);
        return out;
    }

    export function collectionOrderValues(collections) {
        const available = (collections || []).filter((collection) => !collection.is_ungrouped && collection.value).map((collection) => collection.value);
        const out = historyState.historyCollectionSettings.collection_order.filter((value) => available.includes(value));
        for (const value of available) {
            if (!out.includes(value)) out.push(value);
        }
        return out;
    }

    export async function moveHistoryCollection(collection, direction, allCollections = []) {
        if (!collection || collection.is_ungrouped || !collection.value) return;
        await moveHistoryCollectionValue(collection.value, direction, allCollections);
    }

    export async function moveHistoryCollectionValue(value, direction, allCollections = null) {
        const group = String(value || '').trim();
        if (!group) return;
        const collections = allCollections || historyCollectionsForWorkbench(historyState.historyTasks);
        const order = moveItemInList(collectionOrderValues(collections), group, direction);
        await saveHistoryCollectionSettings({
            ...historyState.historyCollectionSettings,
            collection_order: order,
        });
    }

    export async function ensureHistoryCollectionOrderValue(value) {
        const group = String(value || '').trim();
        if (!group || historyState.historyCollectionSettings.collection_order.includes(group)) return;
        await saveHistoryCollectionSettings({
            ...historyState.historyCollectionSettings,
            collection_order: [...historyState.historyCollectionSettings.collection_order, group],
        });
    }

    export function configGroupOrderValues(groups, collection) {
        const key = historyCollectionStorageKey(collection || '__all__');
        const available = (groups || []).map(configGroupKey).filter(Boolean);
        const saved = historyState.historyCollectionSettings.config_group_order?.[key] || [];
        const out = saved.filter((value) => available.includes(value));
        for (const value of available) {
            if (!out.includes(value)) out.push(value);
        }
        return out;
    }

    export async function moveHistoryConfigGroup(group, direction, groups = [], collection = null) {
        const groupKey = configGroupKey(group);
        if (!groupKey) return;
        const collectionKey = historyCollectionStorageKey(collection || '__all__');
        const order = moveItemInList(configGroupOrderValues(groups, collection), groupKey, direction);
        await saveHistoryCollectionSettings({
            ...historyState.historyCollectionSettings,
            config_group_order: {
                ...(historyState.historyCollectionSettings.config_group_order || {}),
                [collectionKey]: order,
            },
        });
    }

    export function groupHistoryTasksByCollection(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const key = historyTaskCollectionKey(task);
            if (!map.has(key)) {
                map.set(key, {
                    key,
                    label: historyTaskCollectionLabel(task),
                    value: historyTaskCollectionValue(task),
                    is_ungrouped: !historyTaskCollectionValue(task),
                    tasks: [],
                });
            }
            map.get(key).tasks.push(task);
        }
        return Array.from(map.values())
            .map(enrichHistoryCollection)
            .sort(historyCollectionComparator);
    }

    export function historyCollectionComparator(a, b) {
        if (a.is_ungrouped !== b.is_ungrouped) return a.is_ungrouped ? -1 : 1;
        const order = historyState.historyCollectionSettings.collection_order || [];
        const aIndex = a.value ? order.indexOf(a.value) : -1;
        const bIndex = b.value ? order.indexOf(b.value) : -1;
        if (aIndex !== bIndex) {
            if (aIndex < 0) return -1;
            if (bIndex < 0) return 1;
            return aIndex - bIndex;
        }
        return (b.latest_started_at - a.latest_started_at) || a.label.localeCompare(b.label, 'zh-CN');
    }

    export function historyCollectionStorageKey(collection) {
        if (!collection) return '__all__';
        if (typeof collection === 'string') {
            if (!collection || collection === 'collection:__all__') return '__all__';
            if (collection === HISTORY_UNGROUPED_COLLECTION_KEY) return '__ungrouped__';
            return collection.startsWith('collection:') ? collection.slice('collection:'.length) : collection;
        }
        if (collection.is_ungrouped) return '__ungrouped__';
        return String(collection.value || '').trim() || '__ungrouped__';
    }

    export function historyCollectionByKey(collections, key) {
        return (collections || []).find((collection) => collection.key === key) || null;
    }

    export function sortedHistoryConfigGroups(groups, collectionKey = '__all__') {
        const storageKey = historyCollectionStorageKey(collectionKey);
        const order = historyState.historyCollectionSettings.config_group_order?.[storageKey] || [];
        return [...(groups || [])].sort((a, b) => {
            const aKey = configGroupKey(a);
            const bKey = configGroupKey(b);
            const aIndex = order.indexOf(aKey);
            const bIndex = order.indexOf(bKey);
            if (aIndex !== bIndex) {
                if (aIndex < 0) return -1;
                if (bIndex < 0) return 1;
                return aIndex - bIndex;
            }
            const aTime = Math.max(0, ...a.tasks.map((task) => Number(task.started_at || 0)));
            const bTime = Math.max(0, ...b.tasks.map((task) => Number(task.started_at || 0)));
            return (bTime - aTime) || historyGroupDisplayLabel(a).localeCompare(historyGroupDisplayLabel(b), 'zh-CN');
        });
    }

    export function enrichHistoryCollection(collection) {
        const tasks = [...(collection.tasks || [])].sort(historyTaskSortComparator(historyState.historyManagerFilters.sort));
        const groups = sortedHistoryConfigGroups(
            groupHistoryTasks(tasks).map(sortHistoryManagerGroupTasks),
            historyCollectionStorageKey(collection),
        );
        return {
            ...collection,
            tasks,
            groups,
            latest_started_at: Math.max(0, ...tasks.map((task) => Number(task.started_at || 0))),
        };
    }

    export function sortHistoryManagerGroupTasks(group) {
        return {
            ...group,
            tasks: [...(group.tasks || [])].sort(historyTaskSortComparator(historyState.historyManagerFilters.sort)),
        };
    }

    export function historyTaskCollectionValue(task) {
        return String(task?.group || '').trim();
    }

    configureHistoryCollectionsBridge({
        createHistoryCollectionWorkbenchCard,
        createHistoryConfigGroupWorkbenchCard,
        historyCollectionNamesForTasks,
        moveItemInList,
        collectionOrderValues,
        moveHistoryCollection,
        moveHistoryCollectionValue,
        ensureHistoryCollectionOrderValue,
        configGroupOrderValues,
        moveHistoryConfigGroup,
        groupHistoryTasksByCollection,
        historyCollectionComparator,
        historyCollectionStorageKey,
        historyCollectionByKey,
        sortedHistoryConfigGroups,
        enrichHistoryCollection,
        sortHistoryManagerGroupTasks,
        historyTaskCollectionValue,
    });
