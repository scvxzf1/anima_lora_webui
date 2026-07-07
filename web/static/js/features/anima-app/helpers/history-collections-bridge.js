const legacyRoot = globalThis;

const historyCollectionsBridge = {
    historyTaskCollectionLabel: (...args) => legacyRoot.historyTaskCollectionLabel?.(...args),
    historyTaskCollectionKey: (...args) => legacyRoot.historyTaskCollectionKey?.(...args),
    historyConfigGroupCollectionMap: (...args) => legacyRoot.historyConfigGroupCollectionMap?.(...args),
    historyTaskIds: (...args) => legacyRoot.historyTaskIds?.(...args),
    historyTasksAllSelected: (...args) => legacyRoot.historyTasksAllSelected?.(...args),
    toggleHistoryTaskSelection: (...args) => legacyRoot.toggleHistoryTaskSelection?.(...args),
    historyManagerGroupMetaParts: (...args) => legacyRoot.historyManagerGroupMetaParts?.(...args),
    historyCompactGroupMetaParts: (...args) => legacyRoot.historyCompactGroupMetaParts?.(...args),
    commonHistoryCollectionValue: (...args) => legacyRoot.commonHistoryCollectionValue?.(...args),
    createHistoryManagerGroupButton: (...args) => legacyRoot.createHistoryManagerGroupButton?.(...args),
    createHistoryConfigGroupMergeButton: (...args) => legacyRoot.createHistoryConfigGroupMergeButton?.(...args),
    createHistoryConfigGroupPreviewButton: (...args) => legacyRoot.createHistoryConfigGroupPreviewButton?.(...args),
    canPreviewHistoryConfigGroup: (...args) => legacyRoot.canPreviewHistoryConfigGroup?.(...args),
    setHistoryCollectionForTasks: (...args) => legacyRoot.setHistoryCollectionForTasks?.(...args),
    renameHistoryCollection: (...args) => legacyRoot.renameHistoryCollection?.(...args),
    clearHistoryCollection: (...args) => legacyRoot.clearHistoryCollection?.(...args),
    renameHistoryCollectionOrderValue: (...args) => legacyRoot.renameHistoryCollectionOrderValue?.(...args),
    renameHistoryConfigGroupOrderKey: (...args) => legacyRoot.renameHistoryConfigGroupOrderKey?.(...args),
    removeHistoryCollectionSettingValue: (...args) => legacyRoot.removeHistoryCollectionSettingValue?.(...args),
    setHistoryCollectionForTasksDirect: (...args) => legacyRoot.setHistoryCollectionForTasksDirect?.(...args),
    applySelectedHistoryTasksToCollection: (...args) => legacyRoot.applySelectedHistoryTasksToCollection?.(...args),
    applyHistoryTaskIdsToCollection: (...args) => legacyRoot.applyHistoryTaskIdsToCollection?.(...args),
    clearSelectedHistoryCollection: (...args) => legacyRoot.clearSelectedHistoryCollection?.(...args),
    clearHistoryCollectionForTasks: (...args) => legacyRoot.clearHistoryCollectionForTasks?.(...args),
    archiveHistoryTasksByIds: (...args) => legacyRoot.archiveHistoryTasksByIds?.(...args),
    deleteHistoryTasksByIds: (...args) => legacyRoot.deleteHistoryTasksByIds?.(...args),
    syncHistorySelectionWithTasks: (...args) => legacyRoot.syncHistorySelectionWithTasks?.(...args),
    selectedHistoryTasks: (...args) => legacyRoot.selectedHistoryTasks?.(...args),
    renderHistoryBulkBar: (...args) => legacyRoot.renderHistoryBulkBar?.(...args),
    syncHistoryFilterControls: (...args) => legacyRoot.syncHistoryFilterControls?.(...args),
    historyManagerFilterDefault: (...args) => legacyRoot.historyManagerFilterDefault?.(...args),
    openHistoryCollectionsWorkbench: (...args) => legacyRoot.openHistoryCollectionsWorkbench?.(...args),
    createHistoryCollectionWorkbenchCard: (...args) => legacyRoot.createHistoryCollectionWorkbenchCard?.(...args),
    createHistoryConfigGroupWorkbenchCard: (...args) => legacyRoot.createHistoryConfigGroupWorkbenchCard?.(...args),
    historyCollectionNamesForTasks: (...args) => legacyRoot.historyCollectionNamesForTasks?.(...args),
    moveItemInList: (...args) => legacyRoot.moveItemInList?.(...args),
    collectionOrderValues: (...args) => legacyRoot.collectionOrderValues?.(...args),
    moveHistoryCollection: (...args) => legacyRoot.moveHistoryCollection?.(...args),
    moveHistoryCollectionValue: (...args) => legacyRoot.moveHistoryCollectionValue?.(...args),
    ensureHistoryCollectionOrderValue: (...args) => legacyRoot.ensureHistoryCollectionOrderValue?.(...args),
    configGroupOrderValues: (...args) => legacyRoot.configGroupOrderValues?.(...args),
    moveHistoryConfigGroup: (...args) => legacyRoot.moveHistoryConfigGroup?.(...args),
    groupHistoryTasksByCollection: (...args) => legacyRoot.groupHistoryTasksByCollection?.(...args),
    historyCollectionComparator: (...args) => legacyRoot.historyCollectionComparator?.(...args),
    historyCollectionStorageKey: (...args) => legacyRoot.historyCollectionStorageKey?.(...args),
    historyCollectionByKey: (...args) => legacyRoot.historyCollectionByKey?.(...args),
    sortedHistoryConfigGroups: (...args) => legacyRoot.sortedHistoryConfigGroups?.(...args),
    enrichHistoryCollection: (...args) => legacyRoot.enrichHistoryCollection?.(...args),
    sortHistoryManagerGroupTasks: (...args) => legacyRoot.sortHistoryManagerGroupTasks?.(...args),
    historyTaskCollectionValue: (...args) => legacyRoot.historyTaskCollectionValue?.(...args),
    historyCollectionSearchText: (...args) => legacyRoot.historyCollectionSearchText?.(...args),
    historyConfigGroupSearchText: (...args) => legacyRoot.historyConfigGroupSearchText?.(...args),
    createEmptyHistoryCollection: (...args) => legacyRoot.createEmptyHistoryCollection?.(...args),
    createHistoryCollectionSearchEmptyCollection: (...args) => legacyRoot.createHistoryCollectionSearchEmptyCollection?.(...args),
    normalizeHistoryCollectionForWorkbench: (...args) => legacyRoot.normalizeHistoryCollectionForWorkbench?.(...args),
    historyCollectionsForWorkbench: (...args) => legacyRoot.historyCollectionsForWorkbench?.(...args),
    historyCollectionSelectOptions: (...args) => legacyRoot.historyCollectionSelectOptions?.(...args),
    historyCollectionOptionSearchText: (...args) => legacyRoot.historyCollectionOptionSearchText?.(...args),
    historyCollectionsPanelTitle: (...args) => legacyRoot.historyCollectionsPanelTitle?.(...args),
    createHistoryCollectionsToolbarButton: (...args) => legacyRoot.createHistoryCollectionsToolbarButton?.(...args),
    stopHistoryGroupButtonPropagation: (...args) => legacyRoot.stopHistoryGroupButtonPropagation?.(...args),
    groupHistoryTasks: (...args) => legacyRoot.groupHistoryTasks?.(...args),
    historyConfigGroupFromTask: (...args) => legacyRoot.historyConfigGroupFromTask?.(...args),
    configGroupKey: (...args) => legacyRoot.configGroupKey?.(...args),
    enrichHistoryGroup: (...args) => legacyRoot.enrichHistoryGroup?.(...args),
    historyTaskDisplayName: (...args) => legacyRoot.historyTaskDisplayName?.(...args),
    historyTaskIsArchived: (...args) => legacyRoot.historyTaskIsArchived?.(...args),
    historyTaskRunPath: (...args) => legacyRoot.historyTaskRunPath?.(...args),
    historyResumeLabel: (...args) => legacyRoot.historyResumeLabel?.(...args),
    historyQueueLabel: (...args) => legacyRoot.historyQueueLabel?.(...args),
    historyContinueLabel: (...args) => legacyRoot.historyContinueLabel?.(...args),
    historyContinuePathLabel: (...args) => legacyRoot.historyContinuePathLabel?.(...args),
    runLabelFromPath: (...args) => legacyRoot.runLabelFromPath?.(...args),
    historyGroupDisplayLabel: (...args) => legacyRoot.historyGroupDisplayLabel?.(...args),
    createHistoryGroupHeading: (...args) => legacyRoot.createHistoryGroupHeading?.(...args),
    renderHistoryCollectionsWorkbench: (...args) => legacyRoot.renderHistoryCollectionsWorkbench?.(...args),
    renderHistoryManagerStats: (...args) => legacyRoot.renderHistoryManagerStats?.(...args),
    applyHistoryStatFilter: (...args) => legacyRoot.applyHistoryStatFilter?.(...args),
    historyStatFilterIsActive: (...args) => legacyRoot.historyStatFilterIsActive?.(...args),
    historyManagerFilteredTasks: (...args) => legacyRoot.historyManagerFilteredTasks?.(...args),
    historyManagerBaseFilteredTasks: (...args) => legacyRoot.historyManagerBaseFilteredTasks?.(...args),
    historyManagerVisibleTasks: (...args) => legacyRoot.historyManagerVisibleTasks?.(...args),
    uniqueHistoryTasks: (...args) => legacyRoot.uniqueHistoryTasks?.(...args),
    historyTaskMatchesSourceFilter: (...args) => legacyRoot.historyTaskMatchesSourceFilter?.(...args),
    historyTaskSearchText: (...args) => legacyRoot.historyTaskSearchText?.(...args),
    historyTaskMatchesCollectionSearch: (...args) => legacyRoot.historyTaskMatchesCollectionSearch?.(...args),
    historySmartSearchTerms: (...args) => legacyRoot.historySmartSearchTerms?.(...args),
    historySearchTerms: (...args) => legacyRoot.historySearchTerms?.(...args),
    historySearchTextMatches: (...args) => legacyRoot.historySearchTextMatches?.(...args),
    historyCollectionMatchesSearch: (...args) => legacyRoot.historyCollectionMatchesSearch?.(...args),
    visibleHistoryCollectionsForSearch: (...args) => legacyRoot.visibleHistoryCollectionsForSearch?.(...args),
    selectedHistoryCollectionForWorkbench: (...args) => legacyRoot.selectedHistoryCollectionForWorkbench?.(...args),
    historyTaskSortComparator: (...args) => legacyRoot.historyTaskSortComparator?.(...args),
    createHistoryManagerRow: (...args) => legacyRoot.createHistoryManagerRow?.(...args),
    compactHistoryPathLabel: (...args) => legacyRoot.compactHistoryPathLabel?.(...args),
    compactHistoryQueueLabel: (...args) => legacyRoot.compactHistoryQueueLabel?.(...args),
    compactHistoryContinueLabel: (...args) => legacyRoot.compactHistoryContinueLabel?.(...args),
    compactHistoryResumeLabel: (...args) => legacyRoot.compactHistoryResumeLabel?.(...args),
    createHistoryMoreActions: (...args) => legacyRoot.createHistoryMoreActions?.(...args),
    selectedHistoryConfigGroups: (...args) => legacyRoot.selectedHistoryConfigGroups?.(...args),
};

export function configureHistoryCollectionsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in historyCollectionsBridge) {
            historyCollectionsBridge[key] = handler;
        }
    }
}

export const historyTaskCollectionLabel = (...args) => historyCollectionsBridge.historyTaskCollectionLabel(...args);
export const historyTaskCollectionKey = (...args) => historyCollectionsBridge.historyTaskCollectionKey(...args);
export const historyConfigGroupCollectionMap = (...args) => historyCollectionsBridge.historyConfigGroupCollectionMap(...args);
export const historyTaskIds = (...args) => historyCollectionsBridge.historyTaskIds(...args);
export const historyTasksAllSelected = (...args) => historyCollectionsBridge.historyTasksAllSelected(...args);
export const toggleHistoryTaskSelection = (...args) => historyCollectionsBridge.toggleHistoryTaskSelection(...args);
export const historyManagerGroupMetaParts = (...args) => historyCollectionsBridge.historyManagerGroupMetaParts(...args);
export const historyCompactGroupMetaParts = (...args) => historyCollectionsBridge.historyCompactGroupMetaParts(...args);
export const commonHistoryCollectionValue = (...args) => historyCollectionsBridge.commonHistoryCollectionValue(...args);
export const createHistoryManagerGroupButton = (...args) => historyCollectionsBridge.createHistoryManagerGroupButton(...args);
export const createHistoryConfigGroupMergeButton = (...args) => historyCollectionsBridge.createHistoryConfigGroupMergeButton(...args);
export const createHistoryConfigGroupPreviewButton = (...args) => historyCollectionsBridge.createHistoryConfigGroupPreviewButton(...args);
export const canPreviewHistoryConfigGroup = (...args) => historyCollectionsBridge.canPreviewHistoryConfigGroup(...args);
export const setHistoryCollectionForTasks = (...args) => historyCollectionsBridge.setHistoryCollectionForTasks(...args);
export const renameHistoryCollection = (...args) => historyCollectionsBridge.renameHistoryCollection(...args);
export const clearHistoryCollection = (...args) => historyCollectionsBridge.clearHistoryCollection(...args);
export const renameHistoryCollectionOrderValue = (...args) => historyCollectionsBridge.renameHistoryCollectionOrderValue(...args);
export const renameHistoryConfigGroupOrderKey = (...args) => historyCollectionsBridge.renameHistoryConfigGroupOrderKey(...args);
export const removeHistoryCollectionSettingValue = (...args) => historyCollectionsBridge.removeHistoryCollectionSettingValue(...args);
export const setHistoryCollectionForTasksDirect = (...args) => historyCollectionsBridge.setHistoryCollectionForTasksDirect(...args);
export const applySelectedHistoryTasksToCollection = (...args) => historyCollectionsBridge.applySelectedHistoryTasksToCollection(...args);
export const applyHistoryTaskIdsToCollection = (...args) => historyCollectionsBridge.applyHistoryTaskIdsToCollection(...args);
export const clearSelectedHistoryCollection = (...args) => historyCollectionsBridge.clearSelectedHistoryCollection(...args);
export const clearHistoryCollectionForTasks = (...args) => historyCollectionsBridge.clearHistoryCollectionForTasks(...args);
export const archiveHistoryTasksByIds = (...args) => historyCollectionsBridge.archiveHistoryTasksByIds(...args);
export const deleteHistoryTasksByIds = (...args) => historyCollectionsBridge.deleteHistoryTasksByIds(...args);
export const syncHistorySelectionWithTasks = (...args) => historyCollectionsBridge.syncHistorySelectionWithTasks(...args);
export const selectedHistoryTasks = (...args) => historyCollectionsBridge.selectedHistoryTasks(...args);
export const renderHistoryBulkBar = (...args) => historyCollectionsBridge.renderHistoryBulkBar(...args);
export const syncHistoryFilterControls = (...args) => historyCollectionsBridge.syncHistoryFilterControls(...args);
export const historyManagerFilterDefault = (...args) => historyCollectionsBridge.historyManagerFilterDefault(...args);
export const openHistoryCollectionsWorkbench = (...args) => historyCollectionsBridge.openHistoryCollectionsWorkbench(...args);
export const createHistoryCollectionWorkbenchCard = (...args) => historyCollectionsBridge.createHistoryCollectionWorkbenchCard(...args);
export const createHistoryConfigGroupWorkbenchCard = (...args) => historyCollectionsBridge.createHistoryConfigGroupWorkbenchCard(...args);
export const historyCollectionNamesForTasks = (...args) => historyCollectionsBridge.historyCollectionNamesForTasks(...args);
export const moveItemInList = (...args) => historyCollectionsBridge.moveItemInList(...args);
export const collectionOrderValues = (...args) => historyCollectionsBridge.collectionOrderValues(...args);
export const moveHistoryCollection = (...args) => historyCollectionsBridge.moveHistoryCollection(...args);
export const moveHistoryCollectionValue = (...args) => historyCollectionsBridge.moveHistoryCollectionValue(...args);
export const ensureHistoryCollectionOrderValue = (...args) => historyCollectionsBridge.ensureHistoryCollectionOrderValue(...args);
export const configGroupOrderValues = (...args) => historyCollectionsBridge.configGroupOrderValues(...args);
export const moveHistoryConfigGroup = (...args) => historyCollectionsBridge.moveHistoryConfigGroup(...args);
export const groupHistoryTasksByCollection = (...args) => historyCollectionsBridge.groupHistoryTasksByCollection(...args);
export const historyCollectionComparator = (...args) => historyCollectionsBridge.historyCollectionComparator(...args);
export const historyCollectionStorageKey = (...args) => historyCollectionsBridge.historyCollectionStorageKey(...args);
export const historyCollectionByKey = (...args) => historyCollectionsBridge.historyCollectionByKey(...args);
export const sortedHistoryConfigGroups = (...args) => historyCollectionsBridge.sortedHistoryConfigGroups(...args);
export const enrichHistoryCollection = (...args) => historyCollectionsBridge.enrichHistoryCollection(...args);
export const sortHistoryManagerGroupTasks = (...args) => historyCollectionsBridge.sortHistoryManagerGroupTasks(...args);
export const historyTaskCollectionValue = (...args) => historyCollectionsBridge.historyTaskCollectionValue(...args);
export const historyCollectionSearchText = (...args) => historyCollectionsBridge.historyCollectionSearchText(...args);
export const historyConfigGroupSearchText = (...args) => historyCollectionsBridge.historyConfigGroupSearchText(...args);
export const createEmptyHistoryCollection = (...args) => historyCollectionsBridge.createEmptyHistoryCollection(...args);
export const createHistoryCollectionSearchEmptyCollection = (...args) => historyCollectionsBridge.createHistoryCollectionSearchEmptyCollection(...args);
export const normalizeHistoryCollectionForWorkbench = (...args) => historyCollectionsBridge.normalizeHistoryCollectionForWorkbench(...args);
export const historyCollectionsForWorkbench = (...args) => historyCollectionsBridge.historyCollectionsForWorkbench(...args);
export const historyCollectionSelectOptions = (...args) => historyCollectionsBridge.historyCollectionSelectOptions(...args);
export const historyCollectionOptionSearchText = (...args) => historyCollectionsBridge.historyCollectionOptionSearchText(...args);
export const historyCollectionsPanelTitle = (...args) => historyCollectionsBridge.historyCollectionsPanelTitle(...args);
export const createHistoryCollectionsToolbarButton = (...args) => historyCollectionsBridge.createHistoryCollectionsToolbarButton(...args);
export const stopHistoryGroupButtonPropagation = (...args) => historyCollectionsBridge.stopHistoryGroupButtonPropagation(...args);
export const groupHistoryTasks = (...args) => historyCollectionsBridge.groupHistoryTasks(...args);
export const historyConfigGroupFromTask = (...args) => historyCollectionsBridge.historyConfigGroupFromTask(...args);
export const configGroupKey = (...args) => historyCollectionsBridge.configGroupKey(...args);
export const enrichHistoryGroup = (...args) => historyCollectionsBridge.enrichHistoryGroup(...args);
export const historyTaskDisplayName = (...args) => historyCollectionsBridge.historyTaskDisplayName(...args);
export const historyTaskIsArchived = (...args) => historyCollectionsBridge.historyTaskIsArchived(...args);
export const historyTaskRunPath = (...args) => historyCollectionsBridge.historyTaskRunPath(...args);
export const historyResumeLabel = (...args) => historyCollectionsBridge.historyResumeLabel(...args);
export const historyQueueLabel = (...args) => historyCollectionsBridge.historyQueueLabel(...args);
export const historyContinueLabel = (...args) => historyCollectionsBridge.historyContinueLabel(...args);
export const historyContinuePathLabel = (...args) => historyCollectionsBridge.historyContinuePathLabel(...args);
export const runLabelFromPath = (...args) => historyCollectionsBridge.runLabelFromPath(...args);
export const historyGroupDisplayLabel = (...args) => historyCollectionsBridge.historyGroupDisplayLabel(...args);
export const createHistoryGroupHeading = (...args) => historyCollectionsBridge.createHistoryGroupHeading(...args);
export const renderHistoryCollectionsWorkbench = (...args) => historyCollectionsBridge.renderHistoryCollectionsWorkbench(...args);
export const renderHistoryManagerStats = (...args) => historyCollectionsBridge.renderHistoryManagerStats(...args);
export const applyHistoryStatFilter = (...args) => historyCollectionsBridge.applyHistoryStatFilter(...args);
export const historyStatFilterIsActive = (...args) => historyCollectionsBridge.historyStatFilterIsActive(...args);
export const historyManagerFilteredTasks = (...args) => historyCollectionsBridge.historyManagerFilteredTasks(...args);
export const historyManagerBaseFilteredTasks = (...args) => historyCollectionsBridge.historyManagerBaseFilteredTasks(...args);
export const historyManagerVisibleTasks = (...args) => historyCollectionsBridge.historyManagerVisibleTasks(...args);
export const uniqueHistoryTasks = (...args) => historyCollectionsBridge.uniqueHistoryTasks(...args);
export const historyTaskMatchesSourceFilter = (...args) => historyCollectionsBridge.historyTaskMatchesSourceFilter(...args);
export const historyTaskSearchText = (...args) => historyCollectionsBridge.historyTaskSearchText(...args);
export const historyTaskMatchesCollectionSearch = (...args) => historyCollectionsBridge.historyTaskMatchesCollectionSearch(...args);
export const historySmartSearchTerms = (...args) => historyCollectionsBridge.historySmartSearchTerms(...args);
export const historySearchTerms = (...args) => historyCollectionsBridge.historySearchTerms(...args);
export const historySearchTextMatches = (...args) => historyCollectionsBridge.historySearchTextMatches(...args);
export const historyCollectionMatchesSearch = (...args) => historyCollectionsBridge.historyCollectionMatchesSearch(...args);
export const visibleHistoryCollectionsForSearch = (...args) => historyCollectionsBridge.visibleHistoryCollectionsForSearch(...args);
export const selectedHistoryCollectionForWorkbench = (...args) => historyCollectionsBridge.selectedHistoryCollectionForWorkbench(...args);
export const historyTaskSortComparator = (...args) => historyCollectionsBridge.historyTaskSortComparator(...args);
export const createHistoryManagerRow = (...args) => historyCollectionsBridge.createHistoryManagerRow(...args);
export const compactHistoryPathLabel = (...args) => historyCollectionsBridge.compactHistoryPathLabel(...args);
export const compactHistoryQueueLabel = (...args) => historyCollectionsBridge.compactHistoryQueueLabel(...args);
export const compactHistoryContinueLabel = (...args) => historyCollectionsBridge.compactHistoryContinueLabel(...args);
export const compactHistoryResumeLabel = (...args) => historyCollectionsBridge.compactHistoryResumeLabel(...args);
export const createHistoryMoreActions = (...args) => historyCollectionsBridge.createHistoryMoreActions(...args);
export const selectedHistoryConfigGroups = (...args) => historyCollectionsBridge.selectedHistoryConfigGroups(...args);
