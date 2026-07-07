const legacyRoot = globalThis;

const historyTaskActionsBridge = {
    createHistoryTaskItem: (...args) => legacyRoot.createHistoryTaskItem?.(...args),
    compactPathLabel: (...args) => legacyRoot.compactPathLabel?.(...args),
    createHistoryActionButton: (...args) => legacyRoot.createHistoryActionButton?.(...args),
    createHistoryTaskPreviewButton: (...args) => legacyRoot.createHistoryTaskPreviewButton?.(...args),
    createHistoryTaskConfigButton: (...args) => legacyRoot.createHistoryTaskConfigButton?.(...args),
    applyHistoryTaskIdsBatchAction: (...args) => legacyRoot.applyHistoryTaskIdsBatchAction?.(...args),
    applyHistoryBatchAction: (...args) => legacyRoot.applyHistoryBatchAction?.(...args),
    archiveSelectedHistoryTasks: (...args) => legacyRoot.archiveSelectedHistoryTasks?.(...args),
    groupSelectedHistoryTasks: (...args) => legacyRoot.groupSelectedHistoryTasks?.(...args),
    deleteSelectedHistoryTasks: (...args) => legacyRoot.deleteSelectedHistoryTasks?.(...args),
    mergeSelectedHistoryTasks: (...args) => legacyRoot.mergeSelectedHistoryTasks?.(...args),
    historyBatchDeleteUnavailable: (...args) => legacyRoot.historyBatchDeleteUnavailable?.(...args),
    deleteHistoryTasksWithLegacyEndpoint: (...args) => legacyRoot.deleteHistoryTasksWithLegacyEndpoint?.(...args),
    deleteHistoryTasksThorough: (...args) => legacyRoot.deleteHistoryTasksThorough?.(...args),
    showHistoryDeletePreviewDialog: (...args) => legacyRoot.showHistoryDeletePreviewDialog?.(...args),
    renameHistoryTask: (...args) => legacyRoot.renameHistoryTask?.(...args),
    regroupHistoryTask: (...args) => legacyRoot.regroupHistoryTask?.(...args),
    archiveHistoryTask: (...args) => legacyRoot.archiveHistoryTask?.(...args),
    deleteHistoryTask: (...args) => legacyRoot.deleteHistoryTask?.(...args),
    updateHistoryTaskMeta: (...args) => legacyRoot.updateHistoryTaskMeta?.(...args),
    historyTaskLabel: (...args) => legacyRoot.historyTaskLabel?.(...args),
    showHistoryTaskInputDialog: (...args) => legacyRoot.showHistoryTaskInputDialog?.(...args),
    showHistoryCollectionSelectDialog: (...args) => legacyRoot.showHistoryCollectionSelectDialog?.(...args),
    showHistoryTaskConfirmDialog: (...args) => legacyRoot.showHistoryTaskConfirmDialog?.(...args),
    showHistoryTaskMessageDialog: (...args) => legacyRoot.showHistoryTaskMessageDialog?.(...args),
    showHistoryTaskDialog: (...args) => legacyRoot.showHistoryTaskDialog?.(...args),
    normalizeHistoryDetailTab: (...args) => legacyRoot.normalizeHistoryDetailTab?.(...args),
    renderHistoryManagerDetail: (...args) => legacyRoot.renderHistoryManagerDetail?.(...args),
    renderHistoryDetailDialog: (...args) => legacyRoot.renderHistoryDetailDialog?.(...args),
    closeHistoryDetailDialog: (...args) => legacyRoot.closeHistoryDetailDialog?.(...args),
    isHistoryDetailDialogOpen: (...args) => legacyRoot.isHistoryDetailDialogOpen?.(...args),
    shouldRenderInlineResumePanel: (...args) => legacyRoot.shouldRenderInlineResumePanel?.(...args),
    clearViewingHistoryTaskContext: (...args) => legacyRoot.clearViewingHistoryTaskContext?.(...args),
    handleHistoryDetailWindowKeydown: (...args) => legacyRoot.handleHistoryDetailWindowKeydown?.(...args),
    restorePreviewWorkspaceFromHistoryDetail: (...args) => legacyRoot.restorePreviewWorkspaceFromHistoryDetail?.(...args),
    activateHistoryDetailPreview: (...args) => legacyRoot.activateHistoryDetailPreview?.(...args),
    clearHistoryManagerDetail: (...args) => legacyRoot.clearHistoryManagerDetail?.(...args),
    selectedHistoryManagerResumeCheckpoint: (...args) => legacyRoot.selectedHistoryManagerResumeCheckpoint?.(...args),
    resumeTrainingFromHistoryDetail: (...args) => legacyRoot.resumeTrainingFromHistoryDetail?.(...args),
    loadHistoryTask: (...args) => legacyRoot.loadHistoryTask?.(...args),
    openSidebarHistoryTask: (...args) => legacyRoot.openSidebarHistoryTask?.(...args),
    refreshHistoryView: (...args) => legacyRoot.refreshHistoryView?.(...args),
    loadConfigGroupTimeline: (...args) => legacyRoot.loadConfigGroupTimeline?.(...args),
    historyTaskStepOffset: (...args) => legacyRoot.historyTaskStepOffset?.(...args),
    historyLossChartPoints: (...args) => legacyRoot.historyLossChartPoints?.(...args),
    renderHistoryTask: (...args) => legacyRoot.renderHistoryTask?.(...args),
};

export function configureHistoryTaskActionsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in historyTaskActionsBridge) {
            historyTaskActionsBridge[key] = handler;
        }
    }
}

export function createHistoryTaskItem(...args) { return historyTaskActionsBridge.createHistoryTaskItem(...args); }
export function compactPathLabel(...args) { return historyTaskActionsBridge.compactPathLabel(...args); }
export function createHistoryActionButton(...args) { return historyTaskActionsBridge.createHistoryActionButton(...args); }
export function createHistoryTaskPreviewButton(...args) { return historyTaskActionsBridge.createHistoryTaskPreviewButton(...args); }
export function createHistoryTaskConfigButton(...args) { return historyTaskActionsBridge.createHistoryTaskConfigButton(...args); }
export function applyHistoryTaskIdsBatchAction(...args) { return historyTaskActionsBridge.applyHistoryTaskIdsBatchAction(...args); }
export function applyHistoryBatchAction(...args) { return historyTaskActionsBridge.applyHistoryBatchAction(...args); }
export function archiveSelectedHistoryTasks(...args) { return historyTaskActionsBridge.archiveSelectedHistoryTasks(...args); }
export function groupSelectedHistoryTasks(...args) { return historyTaskActionsBridge.groupSelectedHistoryTasks(...args); }
export function deleteSelectedHistoryTasks(...args) { return historyTaskActionsBridge.deleteSelectedHistoryTasks(...args); }
export function mergeSelectedHistoryTasks(...args) { return historyTaskActionsBridge.mergeSelectedHistoryTasks(...args); }
export function historyBatchDeleteUnavailable(...args) { return historyTaskActionsBridge.historyBatchDeleteUnavailable(...args); }
export function deleteHistoryTasksWithLegacyEndpoint(...args) { return historyTaskActionsBridge.deleteHistoryTasksWithLegacyEndpoint(...args); }
export function deleteHistoryTasksThorough(...args) { return historyTaskActionsBridge.deleteHistoryTasksThorough(...args); }
export function showHistoryDeletePreviewDialog(...args) { return historyTaskActionsBridge.showHistoryDeletePreviewDialog(...args); }
export function renameHistoryTask(...args) { return historyTaskActionsBridge.renameHistoryTask(...args); }
export function regroupHistoryTask(...args) { return historyTaskActionsBridge.regroupHistoryTask(...args); }
export function archiveHistoryTask(...args) { return historyTaskActionsBridge.archiveHistoryTask(...args); }
export function deleteHistoryTask(...args) { return historyTaskActionsBridge.deleteHistoryTask(...args); }
export function updateHistoryTaskMeta(...args) { return historyTaskActionsBridge.updateHistoryTaskMeta(...args); }
export function historyTaskLabel(...args) { return historyTaskActionsBridge.historyTaskLabel(...args); }
export function showHistoryTaskInputDialog(...args) { return historyTaskActionsBridge.showHistoryTaskInputDialog(...args); }
export function showHistoryCollectionSelectDialog(...args) { return historyTaskActionsBridge.showHistoryCollectionSelectDialog(...args); }
export function showHistoryTaskConfirmDialog(...args) { return historyTaskActionsBridge.showHistoryTaskConfirmDialog(...args); }
export function showHistoryTaskMessageDialog(...args) { return historyTaskActionsBridge.showHistoryTaskMessageDialog(...args); }
export function showHistoryTaskDialog(...args) { return historyTaskActionsBridge.showHistoryTaskDialog(...args); }
export function normalizeHistoryDetailTab(...args) { return historyTaskActionsBridge.normalizeHistoryDetailTab(...args); }
export function renderHistoryManagerDetail(...args) { return historyTaskActionsBridge.renderHistoryManagerDetail(...args); }
export function renderHistoryDetailDialog(...args) { return historyTaskActionsBridge.renderHistoryDetailDialog(...args); }
export function closeHistoryDetailDialog(...args) { return historyTaskActionsBridge.closeHistoryDetailDialog(...args); }
export function isHistoryDetailDialogOpen(...args) { return historyTaskActionsBridge.isHistoryDetailDialogOpen(...args); }
export function shouldRenderInlineResumePanel(...args) { return historyTaskActionsBridge.shouldRenderInlineResumePanel(...args); }
export const clearViewingHistoryTaskContext = (...args) => historyTaskActionsBridge.clearViewingHistoryTaskContext(...args);
export function handleHistoryDetailWindowKeydown(...args) { return historyTaskActionsBridge.handleHistoryDetailWindowKeydown(...args); }
export function restorePreviewWorkspaceFromHistoryDetail(...args) { return historyTaskActionsBridge.restorePreviewWorkspaceFromHistoryDetail(...args); }
export function activateHistoryDetailPreview(...args) { return historyTaskActionsBridge.activateHistoryDetailPreview(...args); }
export function clearHistoryManagerDetail(...args) { return historyTaskActionsBridge.clearHistoryManagerDetail(...args); }
export function selectedHistoryManagerResumeCheckpoint(...args) { return historyTaskActionsBridge.selectedHistoryManagerResumeCheckpoint(...args); }
export function resumeTrainingFromHistoryDetail(...args) { return historyTaskActionsBridge.resumeTrainingFromHistoryDetail(...args); }
export function loadHistoryTask(...args) { return historyTaskActionsBridge.loadHistoryTask(...args); }
export function openSidebarHistoryTask(...args) { return historyTaskActionsBridge.openSidebarHistoryTask(...args); }
export function refreshHistoryView(...args) { return historyTaskActionsBridge.refreshHistoryView(...args); }
export function loadConfigGroupTimeline(...args) { return historyTaskActionsBridge.loadConfigGroupTimeline(...args); }
export function historyTaskStepOffset(...args) { return historyTaskActionsBridge.historyTaskStepOffset(...args); }
export function historyLossChartPoints(...args) { return historyTaskActionsBridge.historyLossChartPoints(...args); }
export function renderHistoryTask(...args) { return historyTaskActionsBridge.renderHistoryTask(...args); }
