/**
 * History config-group workbench card UI.
 */
import {
    beginHistoryConfigGroupDrag,
    dropHistoryConfigGroupToSort,
    dropHistoryTasksToCollection,
    finishHistoryConfigGroupPointerDrag,
    finishHistoryDrag,
    historyConfigGroupOrderDragEnter,
    historyConfigGroupOrderDragLeave,
    historyDropTargetDragEnter,
    historyDropTargetDragLeave,
    placeHistoryConfigGroupDropPreview,
    startHistoryConfigGroupMouseDrag,
    startHistoryConfigGroupPointerDrag,
    startHistoryConfigGroupTouchDrag,
} from '../anima-app/helpers/history-collection-drag-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    clearHistoryCollectionForTasks,
    commonHistoryCollectionValue,
    configGroupKey,
    isHistoryConfigGroupExpanded,
    toggleHistoryConfigGroupExpanded,
    createHistoryConfigGroupMergeButton,
    createHistoryConfigGroupPreviewButton,
    createHistoryManagerGroupButton,
    createHistoryManagerRow,
    createHistoryMoreActions,
    historyCompactGroupMetaParts,
    historyContinueLabel,
    historyGroupDisplayLabel,
    historyManagerGroupMetaParts,
    historyQueueLabel,
    historyResumeLabel,
    historyTaskDisplayName,
    historyTaskIds,
    historyTaskIsArchived,
    historyTasksAllSelected,
    compactHistoryContinueLabel,
    compactHistoryPathLabel,
    compactHistoryQueueLabel,
    compactHistoryResumeLabel,
    setHistoryCollectionForTasks,
    setHistoryCollectionForTasksDirect,
    toggleHistoryTaskSelection,
} from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    archiveHistoryTask,
    createHistoryActionButton,
    createHistoryTaskConfigButton,
    createHistoryTaskPreviewButton,
    deleteHistoryTask,
    loadHistoryTask,
} from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { historyStateLabel } from '../anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    historyCollectionNamesForTasks,
    historyCollectionStorageKey,
    moveHistoryConfigGroup,
} from './workbench-order.js?v=module-bootstrap-20260809-nf4-v2';

const historyState = getHistoryState();

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
        card.dataset.historyTaskId = String(task.id || '');

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
        state.dataset.liveHistoryState = '1';
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

    const expanded = isHistoryConfigGroupExpanded(group, options.collection);
    const hasLiveTasks = (group.tasks || []).some((task) => task?.state === 'running' || task?.state === 'compiling');
    card.classList.toggle('is-expanded', expanded);
    card.classList.toggle('is-live', hasLiveTasks);
    card.dataset.expanded = expanded ? '1' : '0';
    card.dataset.live = hasLiveTasks ? '1' : '0';

    const toggle = createHistoryManagerGroupButton(
        expanded ? '收起' : '展开',
        () => toggleHistoryConfigGroupExpanded(group, options.collection),
    );
    toggle.classList.add('history-config-group-toggle');
    toggle.title = expanded ? '收起任务列表' : '展开任务列表';
    toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    actions.prepend(toggle);

    const head = document.createElement('div');
    head.className = 'history-config-group-card-head';
    head.append(select, handle, main, actions);
    card.appendChild(head);

    if (expanded) {
        const taskList = document.createElement('div');
        taskList.className = 'history-config-group-task-list';
        for (const task of group.tasks) {
            taskList.appendChild(createHistoryManagerRow(task));
        }
        card.appendChild(taskList);
    } else {
        const summary = document.createElement('div');
        summary.className = 'history-config-group-collapse-summary';
        const liveTasks = (group.tasks || []).filter((task) => task?.state === 'running' || task?.state === 'compiling');
        if (liveTasks.length) {
            const live = liveTasks[0];
            const liveLabel = live.state === 'compiling' ? '编译中' : '运行中';
            const liveName = historyTaskDisplayName(live) || historyGroupDisplayLabel(group);
            const dataText = `${live.metric_count || 0} loss / ${live.log_count || 0} log`;
            summary.classList.add('is-live');
            summary.dataset.historyTaskId = String(live.id || '');
            summary.dataset.liveHistorySummary = '1';
            summary.dataset.liveHistoryGroupSize = String(group.tasks.length);
            summary.textContent = [
                `监控中 · ${liveLabel}`,
                liveName,
                dataText,
                `共 ${group.tasks.length} 条 · 点击展开`,
            ].filter(Boolean).join(' · ');
        } else {
            summary.textContent = `已折叠 ${group.tasks.length} 条任务 · 点击“展开”查看`;
        }
        summary.title = summary.textContent;
        summary.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            toggleHistoryConfigGroupExpanded(group, options.collection);
        });
        card.appendChild(summary);
    }
    return card;
}
