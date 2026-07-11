const historyCollectionsHandlers = Object.create(null);

function requireHistoryCollectionsHandler(name) {
    const handler = historyCollectionsHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[history-collections] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureHistoryCollectionsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            historyCollectionsHandlers[key] = handler;
        }
    }
}

export function historyTaskCollectionLabel(...args) { return requireHistoryCollectionsHandler('historyTaskCollectionLabel')(...args); }
export function historyTaskCollectionKey(...args) { return requireHistoryCollectionsHandler('historyTaskCollectionKey')(...args); }
export function historyConfigGroupCollectionMap(...args) { return requireHistoryCollectionsHandler('historyConfigGroupCollectionMap')(...args); }
export function historyTaskIds(...args) { return requireHistoryCollectionsHandler('historyTaskIds')(...args); }
export function historyTasksAllSelected(...args) { return requireHistoryCollectionsHandler('historyTasksAllSelected')(...args); }
export function toggleHistoryTaskSelection(...args) { return requireHistoryCollectionsHandler('toggleHistoryTaskSelection')(...args); }
export function historyManagerGroupMetaParts(...args) { return requireHistoryCollectionsHandler('historyManagerGroupMetaParts')(...args); }
export function historyCompactGroupMetaParts(...args) { return requireHistoryCollectionsHandler('historyCompactGroupMetaParts')(...args); }
export function commonHistoryCollectionValue(...args) { return requireHistoryCollectionsHandler('commonHistoryCollectionValue')(...args); }
export function createHistoryManagerGroupButton(...args) { return requireHistoryCollectionsHandler('createHistoryManagerGroupButton')(...args); }
export function createHistoryConfigGroupMergeButton(...args) { return requireHistoryCollectionsHandler('createHistoryConfigGroupMergeButton')(...args); }
export function createHistoryConfigGroupPreviewButton(...args) { return requireHistoryCollectionsHandler('createHistoryConfigGroupPreviewButton')(...args); }
export function canPreviewHistoryConfigGroup(...args) { return requireHistoryCollectionsHandler('canPreviewHistoryConfigGroup')(...args); }
export function setHistoryCollectionForTasks(...args) { return requireHistoryCollectionsHandler('setHistoryCollectionForTasks')(...args); }
export function renameHistoryCollection(...args) { return requireHistoryCollectionsHandler('renameHistoryCollection')(...args); }
export function clearHistoryCollection(...args) { return requireHistoryCollectionsHandler('clearHistoryCollection')(...args); }
export function renameHistoryCollectionOrderValue(...args) { return requireHistoryCollectionsHandler('renameHistoryCollectionOrderValue')(...args); }
export function renameHistoryConfigGroupOrderKey(...args) { return requireHistoryCollectionsHandler('renameHistoryConfigGroupOrderKey')(...args); }
export function removeHistoryCollectionSettingValue(...args) { return requireHistoryCollectionsHandler('removeHistoryCollectionSettingValue')(...args); }
export function setHistoryCollectionForTasksDirect(...args) { return requireHistoryCollectionsHandler('setHistoryCollectionForTasksDirect')(...args); }
export function applySelectedHistoryTasksToCollection(...args) { return requireHistoryCollectionsHandler('applySelectedHistoryTasksToCollection')(...args); }
export function applyHistoryTaskIdsToCollection(...args) { return requireHistoryCollectionsHandler('applyHistoryTaskIdsToCollection')(...args); }
export function clearSelectedHistoryCollection(...args) { return requireHistoryCollectionsHandler('clearSelectedHistoryCollection')(...args); }
export function clearHistoryCollectionForTasks(...args) { return requireHistoryCollectionsHandler('clearHistoryCollectionForTasks')(...args); }
export function archiveHistoryTasksByIds(...args) { return requireHistoryCollectionsHandler('archiveHistoryTasksByIds')(...args); }
export function deleteHistoryTasksByIds(...args) { return requireHistoryCollectionsHandler('deleteHistoryTasksByIds')(...args); }
export function syncHistorySelectionWithTasks(...args) { return requireHistoryCollectionsHandler('syncHistorySelectionWithTasks')(...args); }
export function selectedHistoryTasks(...args) { return requireHistoryCollectionsHandler('selectedHistoryTasks')(...args); }
export function renderHistoryBulkBar(...args) { return requireHistoryCollectionsHandler('renderHistoryBulkBar')(...args); }
export function syncHistoryFilterControls(...args) { return requireHistoryCollectionsHandler('syncHistoryFilterControls')(...args); }
export function historyManagerFilterDefault(...args) { return requireHistoryCollectionsHandler('historyManagerFilterDefault')(...args); }
export function openHistoryCollectionsWorkbench(...args) { return requireHistoryCollectionsHandler('openHistoryCollectionsWorkbench')(...args); }
export function createHistoryCollectionWorkbenchCard(...args) { return requireHistoryCollectionsHandler('createHistoryCollectionWorkbenchCard')(...args); }
export function createHistoryConfigGroupWorkbenchCard(...args) { return requireHistoryCollectionsHandler('createHistoryConfigGroupWorkbenchCard')(...args); }
export function historyCollectionNamesForTasks(...args) { return requireHistoryCollectionsHandler('historyCollectionNamesForTasks')(...args); }
export function moveItemInList(...args) { return requireHistoryCollectionsHandler('moveItemInList')(...args); }
export function collectionOrderValues(...args) { return requireHistoryCollectionsHandler('collectionOrderValues')(...args); }
export function moveHistoryCollection(...args) { return requireHistoryCollectionsHandler('moveHistoryCollection')(...args); }
export function moveHistoryCollectionValue(...args) { return requireHistoryCollectionsHandler('moveHistoryCollectionValue')(...args); }
export function ensureHistoryCollectionOrderValue(...args) { return requireHistoryCollectionsHandler('ensureHistoryCollectionOrderValue')(...args); }
export function configGroupOrderValues(...args) { return requireHistoryCollectionsHandler('configGroupOrderValues')(...args); }
export function moveHistoryConfigGroup(...args) { return requireHistoryCollectionsHandler('moveHistoryConfigGroup')(...args); }
export function groupHistoryTasksByCollection(...args) { return requireHistoryCollectionsHandler('groupHistoryTasksByCollection')(...args); }
export function historyCollectionComparator(...args) { return requireHistoryCollectionsHandler('historyCollectionComparator')(...args); }
export function historyCollectionStorageKey(...args) { return requireHistoryCollectionsHandler('historyCollectionStorageKey')(...args); }
export function historyCollectionByKey(...args) { return requireHistoryCollectionsHandler('historyCollectionByKey')(...args); }
export function sortedHistoryConfigGroups(...args) { return requireHistoryCollectionsHandler('sortedHistoryConfigGroups')(...args); }
export function enrichHistoryCollection(...args) { return requireHistoryCollectionsHandler('enrichHistoryCollection')(...args); }
export function sortHistoryManagerGroupTasks(...args) { return requireHistoryCollectionsHandler('sortHistoryManagerGroupTasks')(...args); }
export function historyTaskCollectionValue(...args) { return requireHistoryCollectionsHandler('historyTaskCollectionValue')(...args); }
export function historyCollectionSearchText(...args) { return requireHistoryCollectionsHandler('historyCollectionSearchText')(...args); }
export function historyConfigGroupSearchText(...args) { return requireHistoryCollectionsHandler('historyConfigGroupSearchText')(...args); }
export function createEmptyHistoryCollection(...args) { return requireHistoryCollectionsHandler('createEmptyHistoryCollection')(...args); }
export function createHistoryCollectionSearchEmptyCollection(...args) { return requireHistoryCollectionsHandler('createHistoryCollectionSearchEmptyCollection')(...args); }
export function normalizeHistoryCollectionForWorkbench(...args) { return requireHistoryCollectionsHandler('normalizeHistoryCollectionForWorkbench')(...args); }
export function historyCollectionsForWorkbench(...args) { return requireHistoryCollectionsHandler('historyCollectionsForWorkbench')(...args); }
export function historyCollectionSelectOptions(...args) { return requireHistoryCollectionsHandler('historyCollectionSelectOptions')(...args); }
export function historyCollectionOptionSearchText(...args) { return requireHistoryCollectionsHandler('historyCollectionOptionSearchText')(...args); }
export function historyCollectionsPanelTitle(...args) { return requireHistoryCollectionsHandler('historyCollectionsPanelTitle')(...args); }
export function createHistoryCollectionsToolbarButton(...args) { return requireHistoryCollectionsHandler('createHistoryCollectionsToolbarButton')(...args); }
export function stopHistoryGroupButtonPropagation(...args) { return requireHistoryCollectionsHandler('stopHistoryGroupButtonPropagation')(...args); }
export function groupHistoryTasks(...args) { return requireHistoryCollectionsHandler('groupHistoryTasks')(...args); }
export function historyConfigGroupFromTask(...args) { return requireHistoryCollectionsHandler('historyConfigGroupFromTask')(...args); }
export function configGroupKey(...args) { return requireHistoryCollectionsHandler('configGroupKey')(...args); }
export function enrichHistoryGroup(...args) { return requireHistoryCollectionsHandler('enrichHistoryGroup')(...args); }
export function historyTaskDisplayName(...args) { return requireHistoryCollectionsHandler('historyTaskDisplayName')(...args); }
export function historyTaskIsArchived(...args) { return requireHistoryCollectionsHandler('historyTaskIsArchived')(...args); }
export function historyTaskRunPath(...args) { return requireHistoryCollectionsHandler('historyTaskRunPath')(...args); }
export function historyResumeLabel(...args) { return requireHistoryCollectionsHandler('historyResumeLabel')(...args); }
export function historyQueueLabel(...args) { return requireHistoryCollectionsHandler('historyQueueLabel')(...args); }
export function historyContinueLabel(...args) { return requireHistoryCollectionsHandler('historyContinueLabel')(...args); }
export function historyContinuePathLabel(...args) { return requireHistoryCollectionsHandler('historyContinuePathLabel')(...args); }
export function runLabelFromPath(...args) { return requireHistoryCollectionsHandler('runLabelFromPath')(...args); }
export function historyGroupDisplayLabel(...args) { return requireHistoryCollectionsHandler('historyGroupDisplayLabel')(...args); }
export function createHistoryGroupHeading(...args) { return requireHistoryCollectionsHandler('createHistoryGroupHeading')(...args); }
export function renderHistoryCollectionsWorkbench(...args) { return requireHistoryCollectionsHandler('renderHistoryCollectionsWorkbench')(...args); }
export function refreshHistoryWorkbenchConfigPanel(...args) { return requireHistoryCollectionsHandler('refreshHistoryWorkbenchConfigPanel')(...args); }
export function selectHistoryCollectionInWorkbench(...args) { return requireHistoryCollectionsHandler('selectHistoryCollectionInWorkbench')(...args); }
export function toggleHistoryConfigGroupExpanded(...args) { return requireHistoryCollectionsHandler('toggleHistoryConfigGroupExpanded')(...args); }
export function isHistoryConfigGroupExpanded(...args) { return requireHistoryCollectionsHandler('isHistoryConfigGroupExpanded')(...args); }
export function renderHistoryManagerStats(...args) { return requireHistoryCollectionsHandler('renderHistoryManagerStats')(...args); }
export function applyHistoryStatFilter(...args) { return requireHistoryCollectionsHandler('applyHistoryStatFilter')(...args); }
export function historyStatFilterIsActive(...args) { return requireHistoryCollectionsHandler('historyStatFilterIsActive')(...args); }
export function historyManagerFilteredTasks(...args) { return requireHistoryCollectionsHandler('historyManagerFilteredTasks')(...args); }
export function historyManagerBaseFilteredTasks(...args) { return requireHistoryCollectionsHandler('historyManagerBaseFilteredTasks')(...args); }
export function historyManagerVisibleTasks(...args) { return requireHistoryCollectionsHandler('historyManagerVisibleTasks')(...args); }
export function uniqueHistoryTasks(...args) { return requireHistoryCollectionsHandler('uniqueHistoryTasks')(...args); }
export function historyTaskMatchesSourceFilter(...args) { return requireHistoryCollectionsHandler('historyTaskMatchesSourceFilter')(...args); }
export function historyTaskSearchText(...args) { return requireHistoryCollectionsHandler('historyTaskSearchText')(...args); }
export function historyTaskMatchesCollectionSearch(...args) { return requireHistoryCollectionsHandler('historyTaskMatchesCollectionSearch')(...args); }
export function historySmartSearchTerms(...args) { return requireHistoryCollectionsHandler('historySmartSearchTerms')(...args); }
export function historySearchTerms(...args) { return requireHistoryCollectionsHandler('historySearchTerms')(...args); }
export function historySearchTextMatches(...args) { return requireHistoryCollectionsHandler('historySearchTextMatches')(...args); }
export function historyCollectionMatchesSearch(...args) { return requireHistoryCollectionsHandler('historyCollectionMatchesSearch')(...args); }
export function visibleHistoryCollectionsForSearch(...args) { return requireHistoryCollectionsHandler('visibleHistoryCollectionsForSearch')(...args); }
export function selectedHistoryCollectionForWorkbench(...args) { return requireHistoryCollectionsHandler('selectedHistoryCollectionForWorkbench')(...args); }
export function historyTaskSortComparator(...args) { return requireHistoryCollectionsHandler('historyTaskSortComparator')(...args); }
export function createHistoryManagerRow(...args) { return requireHistoryCollectionsHandler('createHistoryManagerRow')(...args); }
export function compactHistoryPathLabel(...args) { return requireHistoryCollectionsHandler('compactHistoryPathLabel')(...args); }
export function compactHistoryQueueLabel(...args) { return requireHistoryCollectionsHandler('compactHistoryQueueLabel')(...args); }
export function compactHistoryContinueLabel(...args) { return requireHistoryCollectionsHandler('compactHistoryContinueLabel')(...args); }
export function compactHistoryResumeLabel(...args) { return requireHistoryCollectionsHandler('compactHistoryResumeLabel')(...args); }
export function createHistoryMoreActions(...args) { return requireHistoryCollectionsHandler('createHistoryMoreActions')(...args); }
export function selectedHistoryConfigGroups(...args) { return requireHistoryCollectionsHandler('selectedHistoryConfigGroups')(...args); }
