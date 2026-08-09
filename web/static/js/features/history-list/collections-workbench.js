/**
 * History collections workbench rendering and filter/search helpers.
 */
import { HISTORY_UNGROUPED_COLLECTION_KEY } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260809-nf4-v2';
import { openHistoryNewCollectionPopover, renderHistoryDropPopover } from '../anima-app/helpers/history-collection-drag-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    applySelectedHistoryTasksToCollection,
    clearSelectedHistoryCollection,
    createEmptyHistoryCollection,
    createHistoryCollectionSearchEmptyCollection,
    createHistoryCollectionWorkbenchCard,
    createHistoryConfigGroupWorkbenchCard,
    createHistoryCollectionsToolbarButton,
    groupHistoryTasks,
    historyConfigGroupCollectionMap,
    configGroupKey,
    historyCollectionByKey,
    historyCollectionSearchText,
    historyCollectionStorageKey,
    historyCollectionsForWorkbench,
    historyCollectionsPanelTitle,
    historyConfigGroupSearchText,
    historyContinueLabel,
    historyGroupDisplayLabel,
    historyQueueLabel,
    historyResumeLabel,
    historyTaskCollectionLabel,
    historyTaskDisplayName,
    historyTaskIds,
    historyTaskIsArchived,
    renderHistoryBulkBar,
    selectedHistoryTasks,
    historyTaskCollectionValue,
    runLabelFromPath,
    sortHistoryManagerGroupTasks,
    sortedHistoryConfigGroups,
    syncHistoryFilterControls,
} from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    archiveHistoryTask,
    createHistoryActionButton,
    createHistoryTaskConfigButton,
    createHistoryTaskPreviewButton,
    deleteHistoryTask,
    groupSelectedHistoryTasks,
    isHistoryDetailDialogOpen,
    loadHistoryTask,
} from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { historyStateLabel } from '../anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { renderHistoryManager } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { fillHistoryWorkbenchCardLists } from './workbench-chunk-fill.js?v=module-bootstrap-20260809-nf4-v2';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';

const historyState = getHistoryState();

function historyConfigGroupExpandKey(group, collection = null) {
    const collectionPart = historyCollectionStorageKey(collection || historyState.selectedHistoryCollectionKey || '__all__');
    return `${collectionPart}::${configGroupKey(group)}`;
}

export function historyConfigGroupHasLiveTasks(group) {
    return (group?.tasks || []).some((task) => task?.state === 'running' || task?.state === 'compiling');
}

export function isHistoryConfigGroupExpanded(group, collection = null) {
    return historyState.expandedHistoryConfigGroupKeys.has(historyConfigGroupExpandKey(group, collection));
}

export function toggleHistoryConfigGroupExpanded(group, collection = null) {
    const key = historyConfigGroupExpandKey(group, collection);
    if (historyState.expandedHistoryConfigGroupKeys.has(key)) {
        historyState.expandedHistoryConfigGroupKeys.delete(key);
        // Remember manual collapse so auto-expand does not immediately reopen this live group.
        historyState.collapsedHistoryConfigGroupKeys.add(key);
    } else {
        historyState.expandedHistoryConfigGroupKeys.add(key);
        historyState.collapsedHistoryConfigGroupKeys.delete(key);
    }
    refreshHistoryWorkbenchConfigPanel();
}

function ensureLiveHistoryConfigGroupsExpanded(groups, collection = null) {
    for (const group of groups || []) {
        if (!historyConfigGroupHasLiveTasks(group)) continue;
        const key = historyConfigGroupExpandKey(group, collection);
        if (historyState.collapsedHistoryConfigGroupKeys.has(key)) continue;
        historyState.expandedHistoryConfigGroupKeys.add(key);
    }
}

function workbenchCollectionsCacheKey(tasks) {
    // Include live-changing task fields so running jobs re-render with fresh
    // state / log / metric counts instead of reusing stale collection objects.
    const ids = (tasks || []).map((task) => [
        task?.id || '',
        task?.group || '',
        task?.state || '',
        task?.archived ? 1 : 0,
        task?.job || '',
        Number(task?.log_count || 0),
        Number(task?.metric_count || 0),
        Number(task?.started_at || 0),
        Number(task?.finished_at || 0),
        Number(task?.updated_at || 0),
        String(task?.name || ''),
        String(task?.message || ''),
    ].join('\u0002')).join('\n');
    const order = (historyState.historyCollectionSettings.collection_order || []).join('\n');
    const groupOrder = JSON.stringify(historyState.historyCollectionSettings.config_group_order || {});
    const sort = historyState.historyManagerFilters?.sort || 'newest';
    return `${sort}\u0001${order}\u0001${groupOrder}\u0001${ids}`;
}

function getHistoryWorkbenchCollections(tasks) {
    const key = workbenchCollectionsCacheKey(tasks);
    const cache = historyState.historyWorkbenchCollectionsCache;
    if (cache && cache.key === key && Array.isArray(cache.collections)) {
        return cache.collections;
    }
    const collections = historyCollectionsForWorkbench(tasks);
    historyState.historyWorkbenchCollectionsCache = { key, collections };
    return collections;
}

function buildHistoryWorkbenchModel(visible) {
    const allCollections = getHistoryWorkbenchCollections(visible);
    if (
        historyState.historyCollectionWorkbenchTarget
        && !allCollections.some((item) => item.value === historyState.historyCollectionWorkbenchTarget)
    ) {
        historyState.historyCollectionWorkbenchTarget = '';
    }
    const smartSearch = historySmartSearchTerms();
    const collectionSearchTerms = historySearchTerms(historyState.historyCollectionSearch, smartSearch.collection);
    const configSearchTerms = historySearchTerms(historyState.historyConfigGroupSearch, smartSearch.config);
    const configSearch = configSearchTerms.join(' ');
    const visibleCollections = visibleHistoryCollectionsForSearch(allCollections, collectionSearchTerms);
    const selectedCollection = selectedHistoryCollectionForWorkbench(allCollections, collectionSearchTerms);
    const scopedTasks = selectedCollection.tasks || [];
    // Prefer precomputed groups on the collection object when present.
    const baseGroups = Array.isArray(selectedCollection.groups) && selectedCollection.groups.length
        ? selectedCollection.groups.map(sortHistoryManagerGroupTasks)
        : groupHistoryTasks(scopedTasks).map(sortHistoryManagerGroupTasks);
    const configGroups = sortedHistoryConfigGroups(
        baseGroups,
        historyCollectionStorageKey(selectedCollection),
    );
    const visibleConfigGroups = configGroups.filter((group) =>
        historySearchTextMatches(historyConfigGroupSearchText(group), configSearchTerms)
        || (configSearch && historyConfigGroupSearchText(group).includes(configSearch))
    );
    ensureLiveHistoryConfigGroupsExpanded(visibleConfigGroups, selectedCollection);
    const currentVisibleTasks = uniqueHistoryTasks(visibleConfigGroups.flatMap((group) => group.tasks || []));
    historyState.historyCurrentVisibleTaskIds = historyTaskIds(currentVisibleTasks);
    const selectedTasks = currentVisibleTasks.filter(
        (task) => task.id && historyState.selectedHistoryTaskIds.has(task.id)
    );
    const selectedGroups = selectedHistoryConfigGroups(visibleConfigGroups);
    const splitCollections = historyConfigGroupCollectionMap(visible);
    return {
        allCollections,
        visibleCollections,
        selectedCollection,
        configGroups,
        visibleConfigGroups,
        currentVisibleTasks,
        selectedTasks,
        selectedGroups,
        splitCollections,
    };
}

function updateHistoryWorkbenchChrome(workbench, model) {
    if (!workbench || !model) return;
    const {
        allCollections,
        visibleCollections,
        selectedCollection,
        configGroups,
        visibleConfigGroups,
        currentVisibleTasks,
        selectedTasks,
        selectedGroups,
    } = model;

    const desc = workbench.querySelector('.history-collections-title span');
    if (desc) {
        desc.textContent = `左侧: ${selectedCollection.is_ungrouped ? '未分类' : selectedCollection.label} · 右侧切换/拖拽归类`;
    }

    const stats = workbench.querySelector('.history-collections-stats');
    if (stats) {
        const values = [
            allCollections.filter((item) => !item.is_ungrouped).length,
            currentVisibleTasks.length,
            visibleConfigGroups.length,
            selectedGroups.length,
        ];
        Array.from(stats.querySelectorAll('strong')).forEach((el, index) => {
            if (values[index] !== undefined) el.textContent = String(values[index]);
        });
    }

    const target = workbench.querySelector('.history-collections-toolbar > span');
    if (target) {
        target.textContent = [
            `当前: ${selectedCollection.label}`,
            historyState.historyCollectionWorkbenchTarget ? `目标: ${historyState.historyCollectionWorkbenchTarget}` : '',
            selectedTasks.length ? `已选: ${selectedTasks.length}` : '未选',
        ].filter(Boolean).join(' · ');
    }

    const configTitleMeta = workbench.querySelector('.history-current-group-content .history-collections-panel-title span');
    if (configTitleMeta) {
        configTitleMeta.textContent = `${visibleConfigGroups.length}/${configGroups.length} 组 · ${currentVisibleTasks.length} 条`;
    }
    const configTitleStrong = workbench.querySelector('.history-current-group-content .history-collections-panel-title strong');
    if (configTitleStrong) {
        configTitleStrong.textContent = selectedCollection.is_ungrouped
            ? '未分类任务'
            : `${selectedCollection.label} 内的任务`;
    }

    const collectionTitleMeta = workbench.querySelector('.history-collection-nav .history-collections-panel-title span');
    if (collectionTitleMeta) {
        collectionTitleMeta.textContent = `${visibleCollections.length}/${allCollections.length} 组`;
    }

    workbench.querySelectorAll('.history-collection-card.nav-card').forEach((card) => {
        const key = String(card.dataset.collectionKey || '');
        const active = key && key === historyState.selectedHistoryCollectionKey;
        card.classList.toggle('active', active);
        card.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
}

function fillHistoryWorkbenchConfigList(configList, model, signal) {
    if (!configList || !model) return;
    configList.innerHTML = '';
    const {
        visibleConfigGroups,
        splitCollections,
        configGroups,
        selectedCollection,
    } = model;
    if (!visibleConfigGroups.length) {
        const empty = document.createElement('div');
        empty.className = 'history-current-group-empty';
        empty.textContent = selectedCollection.is_ungrouped ? '未分类暂无任务。' : '该分组暂无任务。';
        configList.appendChild(empty);
        return;
    }
    fillHistoryWorkbenchCardLists({
        configList,
        collectionList: document.createElement('div'),
        visibleConfigGroups,
        visibleCollections: [],
        splitCollections,
        configGroups,
        selectedCollection,
        selectedTasksLength: model.selectedTasks.length,
        allCollections: model.allCollections,
        createHistoryConfigGroupWorkbenchCard,
        createHistoryCollectionWorkbenchCard,
        emptyConfigMessage: selectedCollection.is_ungrouped ? '未分类暂无任务。' : '该分组暂无任务。',
        signal,
        configOnly: true,
    });
}

export function refreshHistoryWorkbenchConfigPanel(options = {}) {
    const list = document.getElementById('history-manager-list');
    const workbench = list?.querySelector('.history-collections-workbench');
    if (!list || !workbench) {
        renderHistoryManager();
        return;
    }
    if (historyState.historyWorkbenchRenderSignal) {
        historyState.historyWorkbenchRenderSignal.cancelled = true;
    }
    const renderSignal = { cancelled: false };
    historyState.historyWorkbenchRenderSignal = renderSignal;
    const visible = options.visible || historyManagerBaseFilteredTasks();
    const model = buildHistoryWorkbenchModel(visible);
    updateHistoryWorkbenchChrome(workbench, model);
    const configList = workbench.querySelector('.history-config-group-card-list');
    fillHistoryWorkbenchConfigList(configList, model, renderSignal);

    const selectAll = document.getElementById('history-select-all');
    if (selectAll) {
        const visibleIds = historyState.historyCurrentVisibleTaskIds;
        const selectedVisible = visibleIds.filter((id) => historyState.selectedHistoryTaskIds.has(id)).length;
        selectAll.checked = visibleIds.length > 0 && selectedVisible === visibleIds.length;
        selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleIds.length;
    }
    const status = document.getElementById('history-manager-status');
    if (status) {
        const archivedCount = historyState.historyTasks.filter(historyTaskIsArchived).length;
        status.textContent = [
            `共 ${historyState.historyTasks.length} 条记录`,
            `当前分组 ${model.currentVisibleTasks.length} 条`,
            `筛选后 ${(options.baseCount != null ? options.baseCount : visible.length)} 条`,
            `归档 ${archivedCount} 条`,
            historyState.historyDropFeedback.message,
        ].filter(Boolean).join(' · ');
        status.dataset.feedbackTone = historyState.historyDropFeedback.tone || '';
    }
    renderHistoryBulkBar();
    const mergeBtn = document.getElementById('btn-history-manager-merge');
    if (mergeBtn) {
        mergeBtn.disabled = selectedHistoryTasks().filter((task) => task.job === 'training').length === 0;
    }
}

export function selectHistoryCollectionInWorkbench(collectionKey) {
    const key = String(collectionKey || '').trim();
    if (!key) return;
    if (historyState.selectedHistoryCollectionKey === key) return;
    historyState.selectedHistoryCollectionKey = key;
    refreshHistoryWorkbenchConfigPanel();
}

export function renderHistoryCollectionsWorkbench(list, visible) {
    if (historyState.historyWorkbenchRenderSignal) {
        historyState.historyWorkbenchRenderSignal.cancelled = true;
    }
    const renderSignal = { cancelled: false };
    historyState.historyWorkbenchRenderSignal = renderSignal;
    const model = buildHistoryWorkbenchModel(visible);
    const {
        allCollections,
        visibleCollections,
        selectedCollection,
        configGroups,
        visibleConfigGroups,
        currentVisibleTasks,
        selectedTasks,
        selectedGroups,
        splitCollections,
    } = model;

    const workbench = document.createElement('div');
    workbench.className = 'history-collections-workbench compact';
    if (historyState.historyDragState.active) workbench.classList.add('dragging');
    if (historyState.historyDragState.pending) workbench.classList.add('drop-pending');
    if (historyState.historyCollectionDragState.active) workbench.classList.add('collection-reordering');

    const head = document.createElement('div');
    head.className = 'history-collections-head';
    const title = document.createElement('div');
    title.className = 'history-collections-title';
    const heading = document.createElement('strong');
    heading.textContent = '历史分组';
    const desc = document.createElement('span');
    desc.textContent = `左侧: ${selectedCollection.is_ungrouped ? '未分类' : selectedCollection.label} · 右侧切换/拖拽归类`;
    title.append(heading, desc);

    const stats = document.createElement('div');
    stats.className = 'history-collections-stats';
    [
        ['分组', allCollections.filter((item) => !item.is_ungrouped).length],
        ['当前任务', currentVisibleTasks.length],
        ['配置组', visibleConfigGroups.length],
        ['已选分组', selectedGroups.length],
    ].forEach(([label, value]) => {
        const item = document.createElement('div');
        item.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
        stats.appendChild(item);
    });
    head.append(title, stats);
    workbench.appendChild(head);

    const toolbar = document.createElement('div');
    toolbar.className = 'history-collections-toolbar';
    const target = document.createElement('span');
    target.textContent = [
        `当前: ${selectedCollection.label}`,
        historyState.historyCollectionWorkbenchTarget ? `目标: ${historyState.historyCollectionWorkbenchTarget}` : '',
        selectedTasks.length ? `已选: ${selectedTasks.length}` : '未选',
    ].filter(Boolean).join(' · ');
    toolbar.appendChild(target);
    toolbar.append(
        createHistoryCollectionsToolbarButton('设置分组', () => groupSelectedHistoryTasks(), !selectedTasks.length),
        createHistoryCollectionsToolbarButton('清除分组', () => clearSelectedHistoryCollection(), !selectedTasks.length),
    );
    if (historyState.historyCollectionWorkbenchTarget) {
        toolbar.appendChild(createHistoryCollectionsToolbarButton(
            '加入目标',
            () => applySelectedHistoryTasksToCollection(historyState.historyCollectionWorkbenchTarget),
            !selectedTasks.length,
        ));
    }
    workbench.appendChild(toolbar);

    const body = document.createElement('div');
    body.className = 'history-collections-body';

    const configPanel = document.createElement('section');
    configPanel.className = 'history-collections-panel current-content history-current-group-content';
    configPanel.appendChild(historyCollectionsPanelTitle(
        selectedCollection.is_ungrouped ? '未分类任务' : `${selectedCollection.label} 内的任务`,
        `${visibleConfigGroups.length}/${configGroups.length} 组 · ${currentVisibleTasks.length} 条`,
    ));
    const configList = document.createElement('div');
    configList.className = 'history-config-group-card-list';
    configPanel.appendChild(configList);

    const collectionPanel = document.createElement('section');
    collectionPanel.className = 'history-collections-panel collection-nav history-collection-nav';
    const collectionPanelHead = document.createElement('div');
    collectionPanelHead.className = 'history-collection-nav-head';
    collectionPanelHead.appendChild(historyCollectionsPanelTitle('分组导航', `${visibleCollections.length}/${allCollections.length} 组`));
    const createBtn = createHistoryCollectionsToolbarButton('新建分组', (event) => openHistoryNewCollectionPopover(event, []));
    createBtn.classList.add('history-collection-create-btn');
    collectionPanelHead.appendChild(createBtn);
    collectionPanel.appendChild(collectionPanelHead);
    const collectionList = document.createElement('div');
    collectionList.className = 'history-collection-card-list';
    fillHistoryWorkbenchCardLists({
        configList,
        collectionList,
        visibleConfigGroups,
        visibleCollections,
        splitCollections,
        configGroups,
        selectedCollection,
        selectedTasksLength: selectedTasks.length,
        allCollections,
        createHistoryConfigGroupWorkbenchCard,
        createHistoryCollectionWorkbenchCard,
        emptyConfigMessage: selectedCollection.is_ungrouped ? '未分类暂无任务。' : '该分组暂无任务。',
        signal: renderSignal,
    });
    collectionPanel.appendChild(collectionList);

    body.append(configPanel, collectionPanel);
    workbench.appendChild(body);
    renderHistoryDropPopover(workbench);
    list.appendChild(workbench);
}

export function renderHistoryManagerStats() {
    const el = document.getElementById('history-manager-stats');
    if (!el) return;
    const counts = {
        total: historyState.historyTasks.length,
        training: historyState.historyTasks.filter((task) => task.job === 'training').length,
        preprocess: historyState.historyTasks.filter((task) => task.job === 'preprocess').length,
        error: historyState.historyTasks.filter((task) => ['error', 'interrupted'].includes(task.state)).length,
        archived: historyState.historyTasks.filter(historyTaskIsArchived).length,
        queue: historyState.historyTasks.filter((task) => task.from_queue || task.queue_item_id).length,
    };
    el.innerHTML = '';
    [
        ['全部', counts.total, 'all'],
        ['训练', counts.training, 'training'],
        ['预处理', counts.preprocess, 'preprocess'],
        ['异常/中断', counts.error, 'error'],
        ['归档', counts.archived, 'archived'],
        ['来自队列', counts.queue, 'queue'],
    ].forEach(([label, value, state]) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = ['history-manager-stat', state, historyStatFilterIsActive(state) ? 'active' : ''].filter(Boolean).join(' ');
        item.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
        item.addEventListener('click', () => applyHistoryStatFilter(state));
        el.appendChild(item);
    });
}

export function applyHistoryStatFilter(state) {
    historyState.historyCollectionSearch = '';
    historyState.historyConfigGroupSearch = '';
    const next = {
        search: '',
        kind: 'all',
        state: 'all',
        archived: 'all',
        source: 'all',
        trainingVariant: 'all',
        preprocessPrecision: 'all',
        blockSwapPrecision: 'all',
        baseCompute: 'all',
        precisionPreference: 'all',
        sort: historyState.historyManagerFilters.sort || 'newest',
    };
    if (state === 'training' || state === 'preprocess') {
        next.kind = state;
    } else if (state === 'error') {
        next.state = 'error';
    } else if (state === 'archived') {
        next.archived = 'archived';
    } else if (state === 'queue') {
        next.source = 'queue';
    }
    historyState.historyManagerFilters = next;
    syncHistoryFilterControls();
    renderHistoryManager();
}

export function historyStatFilterIsActive(state) {
    const searchEmpty =
        !String(historyState.historyManagerFilters.search || '').trim() &&
        !String(historyState.historyCollectionSearch || '').trim() &&
        !String(historyState.historyConfigGroupSearch || '').trim();
    const base =
        searchEmpty &&
        Boolean(historyState.historyManagerFilters.sort || 'newest') &&
        (historyState.historyManagerFilters.trainingVariant || 'all') === 'all' &&
        (historyState.historyManagerFilters.preprocessPrecision || 'all') === 'all' &&
        (historyState.historyManagerFilters.blockSwapPrecision || 'all') === 'all' &&
        (historyState.historyManagerFilters.baseCompute || 'all') === 'all' &&
        (historyState.historyManagerFilters.precisionPreference || 'all') === 'all' &&
        (state === 'archived'
            ? historyState.historyManagerFilters.archived === 'archived'
            : (historyState.historyManagerFilters.archived || 'active') === 'all');
    if (!base) return false;
    if (state === 'all') {
        return historyState.historyManagerFilters.kind === 'all' &&
            historyState.historyManagerFilters.state === 'all' &&
            historyState.historyManagerFilters.source === 'all';
    }
    if (state === 'training' || state === 'preprocess') {
        return historyState.historyManagerFilters.kind === state &&
            historyState.historyManagerFilters.state === 'all' &&
            historyState.historyManagerFilters.source === 'all';
    }
    if (state === 'error') {
        return historyState.historyManagerFilters.kind === 'all' &&
            historyState.historyManagerFilters.state === 'error' &&
            historyState.historyManagerFilters.source === 'all';
    }
    if (state === 'archived') {
        return historyState.historyManagerFilters.kind === 'all' &&
            historyState.historyManagerFilters.state === 'all' &&
            historyState.historyManagerFilters.archived === 'archived' &&
            historyState.historyManagerFilters.source === 'all';
    }
    if (state === 'queue') {
        return historyState.historyManagerFilters.kind === 'all' &&
            historyState.historyManagerFilters.state === 'all' &&
            historyState.historyManagerFilters.source === 'queue';
    }
    return false;
}

export function historyManagerFilteredTasks() {
    return historyManagerVisibleTasks(historyManagerBaseFilteredTasks());
}

export function historyManagerBaseFilteredTasks() {
    const search = historySmartSearchTerms().global;
    const visible = historyState.historyTasks.filter((task) => {
        if (historyState.historyManagerFilters.kind !== 'all' && task.job !== historyState.historyManagerFilters.kind) return false;
        if (historyState.historyManagerFilters.state !== 'all') {
            if (historyState.historyManagerFilters.state === 'error') {
                if (!['error', 'interrupted'].includes(task.state)) return false;
            } else if (task.state !== historyState.historyManagerFilters.state) {
                return false;
            }
        }
        const archived = historyTaskIsArchived(task);
        if (historyState.historyManagerFilters.archived === 'active' && archived) return false;
        if (historyState.historyManagerFilters.archived === 'archived' && !archived) return false;
        if (!historyTaskMatchesSourceFilter(task, historyState.historyManagerFilters.source)) return false;
        if (!historyTaskMatchesChipFilters(task, historyState.historyManagerFilters)) return false;
        if (search && !historyTaskSearchText(task).includes(search)) return false;
        return true;
    });
    visible.sort(historyTaskSortComparator(historyState.historyManagerFilters.sort));
    return visible;
}

export function historyManagerVisibleTasks(baseTasks) {
    const base = baseTasks || [];
    const smartSearch = historySmartSearchTerms();
    const collectionSearchTerms = historySearchTerms(historyState.historyCollectionSearch, smartSearch.collection);
    const configSearchTerms = historySearchTerms(historyState.historyConfigGroupSearch, smartSearch.config);
    const collections = getHistoryWorkbenchCollections(base);
    const selectedCollection = selectedHistoryCollectionForWorkbench(collections, collectionSearchTerms);
    const visibleGroups = (selectedCollection.groups || [])
        .filter((group) => historySearchTextMatches(historyConfigGroupSearchText(group), configSearchTerms));
    return uniqueHistoryTasks(visibleGroups.flatMap((group) => group.tasks || []));
}

export function uniqueHistoryTasks(tasks) {
    const seen = new Set();
    const out = [];
    for (const task of tasks || []) {
        const key = task?.id || `${task?.history_dir || ''}:${out.length}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(task);
    }
    out.sort(historyTaskSortComparator(historyState.historyManagerFilters.sort));
    return out;
}

export function historyTaskMatchesSourceFilter(task, filter) {
    if (!filter || filter === 'all') return true;
    if (filter === 'queue') return Boolean(task.from_queue || task.queue_item_id);
    if (filter === 'resume') return Boolean(task.resume_from?.source_task_id);
    if (filter === 'continue') return task.training_mode === 'continue_lora';
    return true;
}

export function historyTaskMatchesChipFilters(task, filters = {}) {
    const trainingVariant = filters.trainingVariant || 'all';
    if (trainingVariant !== 'all') {
        const value = String(task?.training_variant || '').trim().toLowerCase();
        if (value !== trainingVariant) return false;
    }
    const preprocessPrecision = filters.preprocessPrecision || 'all';
    if (preprocessPrecision !== 'all') {
        const value = String(task?.preprocess_precision || '').trim().toLowerCase();
        if (value !== preprocessPrecision) return false;
    }
    const blockSwapPrecision = filters.blockSwapPrecision || 'all';
    if (blockSwapPrecision !== 'all') {
        const value = String(task?.block_swap_precision || '').trim().toLowerCase();
        if (value !== blockSwapPrecision) return false;
    }
    const baseCompute = filters.baseCompute || 'all';
    if (baseCompute !== 'all') {
        const value = String(task?.base_compute || '').trim().toLowerCase();
        if (value !== baseCompute) return false;
    }
    const precisionPreference = filters.precisionPreference || 'all';
    if (precisionPreference !== 'all') {
        const value = String(task?.precision_preference || '').trim().toLowerCase();
        if (value !== precisionPreference) return false;
    }
    return true;
}

export function historyTaskSearchText(task) {
    return [
        task.id,
        historyTaskDisplayName(task),
        task.name,
        task.group,
        task.history_group_label,
        task.history_source_config_file,
        task.history_run_label,
        task.methods_subdir,
        task.variant,
        task.training_variant,
        task.preprocess_precision,
        task.block_swap_precision,
        task.base_compute,
        task.precision_preference,
        task.preset,
        task.run_dir,
        task.training_output_dir,
        task.output_dir,
        task.message,
    ].filter(Boolean).join('\n').toLowerCase();
}

export function historyTaskMatchesCollectionSearch(task, search) {
    return [
        historyTaskCollectionLabel(task),
        historyTaskCollectionValue(task),
        historyTaskSearchText(task),
    ].filter(Boolean).join('\n').toLowerCase().includes(search);
}

export function historySmartSearchTerms() {
    const raw = String(historyState.historyManagerFilters.search || '').trim();
    const terms = { global: '', collection: '', config: '' };
    const match = raw.match(/^([^:：]+)\s*[:：]\s*(.*)$/);
    if (!match) {
        terms.global = raw.toLowerCase();
        return terms;
    }
    const prefix = match[1].trim().toLowerCase();
    const value = match[2].trim().toLowerCase();
    if (!value) return terms;
    if (['组', '集合', 'group', 'collection'].includes(prefix)) {
        terms.collection = value;
    } else if (['配置', '配置组', 'config'].includes(prefix)) {
        terms.config = value;
    } else {
        terms.global = raw.toLowerCase();
    }
    return terms;
}

export function historySearchTerms(...values) {
    return values.map((value) => String(value || '').trim().toLowerCase()).filter(Boolean);
}

export function historySearchTextMatches(text, terms) {
    const haystack = String(text || '').toLowerCase();
    return terms.every((term) => haystack.includes(term));
}

export function historyCollectionMatchesSearch(collection, terms) {
    if (!terms.length) return true;
    const text = historyCollectionSearchText(collection);
    const phrase = terms.join(' ');
    return historySearchTextMatches(text, terms) || Boolean(phrase && text.includes(phrase));
}

export function visibleHistoryCollectionsForSearch(collections, terms) {
    return (collections || []).filter((collection) => historyCollectionMatchesSearch(collection, terms));
}

export function selectedHistoryCollectionForWorkbench(collections, collectionSearchTerms = []) {
    const allCollections = collections || [];
    const visibleCollections = visibleHistoryCollectionsForSearch(allCollections, collectionSearchTerms);
    if (collectionSearchTerms.length && !visibleCollections.length) {
        historyState.selectedHistoryCollectionKey = 'collection:__search_empty__';
        return createHistoryCollectionSearchEmptyCollection();
    }
    const candidates = collectionSearchTerms.length ? visibleCollections : allCollections;
    const selected = historyCollectionByKey(candidates, historyState.selectedHistoryCollectionKey)
        || (collectionSearchTerms.length ? candidates[0] : null)
        || historyCollectionByKey(allCollections, historyState.selectedHistoryCollectionKey)
        || historyCollectionByKey(allCollections, HISTORY_UNGROUPED_COLLECTION_KEY)
        || createEmptyHistoryCollection();
    historyState.selectedHistoryCollectionKey = selected.key;
    return selected;
}

export function historyTaskSortComparator(mode) {
    return (a, b) => {
        if (mode === 'oldest') return (Number(a.started_at || 0) - Number(b.started_at || 0));
        if (mode === 'loss') return (Number(b.metric_count || 0) - Number(a.metric_count || 0)) || (Number(b.started_at || 0) - Number(a.started_at || 0));
        if (mode === 'logs') return (Number(b.log_count || 0) - Number(a.log_count || 0)) || (Number(b.started_at || 0) - Number(a.started_at || 0));
        if (mode === 'name') return historyTaskDisplayName(a).localeCompare(historyTaskDisplayName(b), 'zh-CN');
        return (Number(b.started_at || 0) - Number(a.started_at || 0));
    };
}

export function createHistoryManagerRow(task) {
    const row = document.createElement('article');
    row.className = 'history-manager-row';
    row.dataset.historyTaskId = String(task.id || '');
    if (historyState.viewingHistoryTaskId === task.id && isHistoryDetailDialogOpen()) row.classList.add('active');
    if (historyTaskIsArchived(task)) row.classList.add('archived');

    const select = document.createElement('label');
    select.className = 'history-row-select';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = historyState.selectedHistoryTaskIds.has(task.id);
    checkbox.addEventListener('change', () => {
        if (checkbox.checked) historyState.selectedHistoryTaskIds.add(task.id);
        else historyState.selectedHistoryTaskIds.delete(task.id);
        renderHistoryManager();
    });
    select.appendChild(checkbox);

    const main = document.createElement('button');
    main.type = 'button';
    main.className = 'history-row-main';
    main.addEventListener('click', () => loadHistoryTask(task.id));
    const title = document.createElement('strong');
    title.textContent = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
    title.title = title.textContent;
    const meta = document.createElement('span');
    meta.className = 'history-compact-meta';
    meta.textContent = [
        compactHistoryPathLabel(task.history_source_config_file || `${task.methods_subdir || '-'} / ${task.variant || '-'}`),
        compactHistoryQueueLabel(task),
        compactHistoryContinueLabel(task),
        compactHistoryResumeLabel(task),
    ].filter(Boolean).join(' · ');
    meta.title = [
        task.history_source_config_file || `${task.methods_subdir || '-'} / ${task.variant || '-'}`,
        historyQueueLabel(task),
        historyContinueLabel(task),
        historyResumeLabel(task),
    ].filter(Boolean).join(' · ');
    main.append(title, meta);

    const state = document.createElement('div');
    state.className = ['history-row-state', task.state || 'unknown'].join(' ');
    state.dataset.liveHistoryState = '1';
    state.textContent = [
        task.job === 'preprocess' ? '预处理' : '训练',
        historyStateLabel(task.state),
        historyTaskIsArchived(task) ? '已归档' : '',
    ].filter(Boolean).join(' · ');

    const time = document.createElement('div');
    time.className = 'history-row-time';
    time.textContent = `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`;

    const data = document.createElement('div');
    data.className = 'history-row-data';
    data.dataset.liveHistoryCounts = 'manager';
    data.textContent = `${task.metric_count || 0} loss / ${task.log_count || 0} log`;

    const actions = document.createElement('div');
    actions.className = 'history-row-actions';
    if (task.job === 'training') {
        actions.append(
            createHistoryTaskPreviewButton(task),
        );
    }
    actions.append(
        createHistoryActionButton('查看', () => loadHistoryTask(task.id)),
        createHistoryMoreActions([
            createHistoryTaskConfigButton(task),
            createHistoryActionButton(historyTaskIsArchived(task) ? '取消归档' : '归档', () => archiveHistoryTask(task)),
            createHistoryActionButton('删除', () => deleteHistoryTask(task), 'danger'),
        ]),
    );

    row.append(select, main, state, time, data, actions);
    return row;
}

export function compactHistoryPathLabel(value) {
    const text = String(value || '').trim();
    if (!text) return '';
    return runLabelFromPath(text) || text;
}

export function compactHistoryQueueLabel(task) {
    if (!Boolean(task?.from_queue) && !String(task?.queue_item_id || '').trim()) return '';
    const attempt = Number(task?.queue_attempt || 1);
    return attempt > 1 ? `队列#${attempt}` : '队列';
}

export function compactHistoryContinueLabel(task) {
    if (task?.training_mode !== 'continue_lora') return '';
    const kind = String(task.continue_from_weight_kind || 'LoRA').trim() || 'LoRA';
    const name = compactHistoryPathLabel(task.continue_from_weight_name || '');
    return name ? `续训 ${kind}:${name}` : `续训 ${kind}`;
}

export function compactHistoryResumeLabel(task) {
    const resume = task?.resume_from || {};
    if (!resume || typeof resume !== 'object') return '';
    const checkpoint = compactHistoryPathLabel(resume.checkpoint_name || '');
    const step = resume.checkpoint_step !== undefined && resume.checkpoint_step !== null
        ? String(resume.checkpoint_step).trim()
        : '';
    if (checkpoint && step) return `恢复 ${checkpoint}@${step}`;
    if (checkpoint) return `恢复 ${checkpoint}`;
    if (step) return `恢复 step ${step}`;
    return resume.source_task_id ? '恢复' : '';
}

export function createHistoryMoreActions(buttons) {
    const menu = document.createElement('details');
    menu.className = 'history-more-actions';
    menu.addEventListener('click', (event) => event.stopPropagation());
    const summary = document.createElement('summary');
    summary.textContent = '...';
    summary.title = '更多历史任务操作';
    summary.setAttribute('aria-label', '更多历史任务操作');
    const body = document.createElement('div');
    body.className = 'history-more-actions-menu';
    for (const button of buttons.filter(Boolean)) {
        body.appendChild(button);
    }
    menu.append(summary, body);
    return menu;
}

export function selectedHistoryConfigGroups(groups) {
    return (groups || []).filter(
        (group) => historyTaskIds(group.tasks).some((id) => historyState.selectedHistoryTaskIds.has(id))
    );
}
