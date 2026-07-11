const historyTaskActionsHandlers = Object.create(null);

function requireHistoryTaskActionsHandler(name) {
    const handler = historyTaskActionsHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[history-task-actions] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureHistoryTaskActionsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            historyTaskActionsHandlers[key] = handler;
        }
    }
}

export function createHistoryTaskItem(...args) { return requireHistoryTaskActionsHandler('createHistoryTaskItem')(...args); }
export function compactPathLabel(...args) { return requireHistoryTaskActionsHandler('compactPathLabel')(...args); }
export function createHistoryActionButton(...args) { return requireHistoryTaskActionsHandler('createHistoryActionButton')(...args); }
export function createHistoryTaskPreviewButton(...args) { return requireHistoryTaskActionsHandler('createHistoryTaskPreviewButton')(...args); }
export function createHistoryTaskConfigButton(...args) { return requireHistoryTaskActionsHandler('createHistoryTaskConfigButton')(...args); }
export function applyHistoryTaskIdsBatchAction(...args) { return requireHistoryTaskActionsHandler('applyHistoryTaskIdsBatchAction')(...args); }
export function applyHistoryBatchAction(...args) { return requireHistoryTaskActionsHandler('applyHistoryBatchAction')(...args); }
export function archiveSelectedHistoryTasks(...args) { return requireHistoryTaskActionsHandler('archiveSelectedHistoryTasks')(...args); }
export function groupSelectedHistoryTasks(...args) { return requireHistoryTaskActionsHandler('groupSelectedHistoryTasks')(...args); }
export function deleteSelectedHistoryTasks(...args) { return requireHistoryTaskActionsHandler('deleteSelectedHistoryTasks')(...args); }
export function mergeSelectedHistoryTasks(...args) { return requireHistoryTaskActionsHandler('mergeSelectedHistoryTasks')(...args); }
export function historyBatchDeleteUnavailable(...args) { return requireHistoryTaskActionsHandler('historyBatchDeleteUnavailable')(...args); }
export function deleteHistoryTasksWithLegacyEndpoint(...args) { return requireHistoryTaskActionsHandler('deleteHistoryTasksWithLegacyEndpoint')(...args); }
export function deleteHistoryTasksThorough(...args) { return requireHistoryTaskActionsHandler('deleteHistoryTasksThorough')(...args); }
export function showHistoryDeletePreviewDialog(...args) { return requireHistoryTaskActionsHandler('showHistoryDeletePreviewDialog')(...args); }
export function renameHistoryTask(...args) { return requireHistoryTaskActionsHandler('renameHistoryTask')(...args); }
export function regroupHistoryTask(...args) { return requireHistoryTaskActionsHandler('regroupHistoryTask')(...args); }
export function archiveHistoryTask(...args) { return requireHistoryTaskActionsHandler('archiveHistoryTask')(...args); }
export function deleteHistoryTask(...args) { return requireHistoryTaskActionsHandler('deleteHistoryTask')(...args); }
export function updateHistoryTaskMeta(...args) { return requireHistoryTaskActionsHandler('updateHistoryTaskMeta')(...args); }
export function historyTaskLabel(...args) { return requireHistoryTaskActionsHandler('historyTaskLabel')(...args); }
export function showHistoryTaskInputDialog(...args) { return requireHistoryTaskActionsHandler('showHistoryTaskInputDialog')(...args); }
export function showHistoryCollectionSelectDialog(...args) { return requireHistoryTaskActionsHandler('showHistoryCollectionSelectDialog')(...args); }
export function showHistoryTaskConfirmDialog(...args) { return requireHistoryTaskActionsHandler('showHistoryTaskConfirmDialog')(...args); }
export function showHistoryTaskMessageDialog(...args) { return requireHistoryTaskActionsHandler('showHistoryTaskMessageDialog')(...args); }
export function showHistoryTaskDialog(...args) { return requireHistoryTaskActionsHandler('showHistoryTaskDialog')(...args); }
export function normalizeHistoryDetailTab(...args) { return requireHistoryTaskActionsHandler('normalizeHistoryDetailTab')(...args); }
export function renderHistoryManagerDetail(...args) { return requireHistoryTaskActionsHandler('renderHistoryManagerDetail')(...args); }
export function renderHistoryDetailDialog(...args) { return requireHistoryTaskActionsHandler('renderHistoryDetailDialog')(...args); }
export function closeHistoryDetailDialog(...args) { return requireHistoryTaskActionsHandler('closeHistoryDetailDialog')(...args); }
export function isHistoryDetailDialogOpen(...args) { return requireHistoryTaskActionsHandler('isHistoryDetailDialogOpen')(...args); }
export function shouldRenderInlineResumePanel(...args) { return requireHistoryTaskActionsHandler('shouldRenderInlineResumePanel')(...args); }
export const clearViewingHistoryTaskContext = (...args) => requireHistoryTaskActionsHandler('clearViewingHistoryTaskContext')(...args);
export function handleHistoryDetailWindowKeydown(...args) { return requireHistoryTaskActionsHandler('handleHistoryDetailWindowKeydown')(...args); }
export function restorePreviewWorkspaceFromHistoryDetail(...args) { return requireHistoryTaskActionsHandler('restorePreviewWorkspaceFromHistoryDetail')(...args); }
export function activateHistoryDetailPreview(...args) { return requireHistoryTaskActionsHandler('activateHistoryDetailPreview')(...args); }
export function clearHistoryManagerDetail(...args) { return requireHistoryTaskActionsHandler('clearHistoryManagerDetail')(...args); }
export function selectedHistoryManagerResumeCheckpoint(...args) { return requireHistoryTaskActionsHandler('selectedHistoryManagerResumeCheckpoint')(...args); }
export function resumeTrainingFromHistoryDetail(...args) { return requireHistoryTaskActionsHandler('resumeTrainingFromHistoryDetail')(...args); }
export function loadHistoryTask(...args) { return requireHistoryTaskActionsHandler('loadHistoryTask')(...args); }
export function openSidebarHistoryTask(...args) { return requireHistoryTaskActionsHandler('openSidebarHistoryTask')(...args); }
export function refreshHistoryView(...args) { return requireHistoryTaskActionsHandler('refreshHistoryView')(...args); }
export function loadConfigGroupTimeline(...args) { return requireHistoryTaskActionsHandler('loadConfigGroupTimeline')(...args); }
export function historyTaskStepOffset(...args) { return requireHistoryTaskActionsHandler('historyTaskStepOffset')(...args); }
export function historyLossChartPoints(...args) { return requireHistoryTaskActionsHandler('historyLossChartPoints')(...args); }
export function renderHistoryTask(...args) { return requireHistoryTaskActionsHandler('renderHistoryTask')(...args); }
