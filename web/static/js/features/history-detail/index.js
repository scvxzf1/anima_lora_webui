import { fetchHistoryTask } from './api.js?v=module-bootstrap-20260608-11';
import { createHistoryDetailDialog } from './dialog.js?v=module-bootstrap-20260608-11';
import { createHistoryDetailState, normalizeHistoryDetailTab } from './state.js?v=module-bootstrap-20260608-11';

export function createHistoryDetailFeature(ctx, deps) {
    const state = createHistoryDetailState();
    const dialog = createHistoryDetailDialog({ ctx, state, deps });

    async function loadHistoryTask(taskId, options = {}) {
        try {
            const payload = await fetchHistoryTask(ctx, taskId);
            if (!payload.ok) {
                alert(payload.error || '读取历史任务失败');
                return;
            }
            if (options.detailTab) {
                state.detailTab = normalizeHistoryDetailTab(options.detailTab);
            }
            state.mainTaskReturn = null;
            deps.setViewingHistoryTaskContext({
                taskId,
                viewMode: 'live',
                task: payload.task || null,
                configGroup: null,
                timelineSelection: [],
            });
            dialog.setResumeLoadingForTask(taskId);
            state.curve.hoverStep = null;
            dialog.renderHistoryManagerDetail(payload, { open: true });
            deps.renderTrainingHistoryList();
            deps.renderHistoryManager();
            await dialog.loadResumeOptionsForTask(taskId);
        } catch (e) {
            alert('读取历史任务失败: ' + e.message);
        }
    }

    function clearHistoryDetailState() {
        state.currentPayload = null;
        state.detailTab = 'overview';
        state.returnState = null;
        state.mainTaskReturn = null;
        state.curve.hoverStep = null;
    }

    function resetCurveHover() {
        state.curve.hoverStep = null;
    }

    return {
        loadHistoryTask,
        renderHistoryManagerDetail: dialog.renderHistoryManagerDetail,
        renderHistoryDetailDialog: dialog.renderHistoryDetailDialog,
        closeHistoryDetailDialog: dialog.closeHistoryDetailDialog,
        isHistoryDetailDialogOpen: dialog.isHistoryDetailDialogOpen,
        handleHistoryDetailWindowKeydown: dialog.handleHistoryDetailWindowKeydown,
        loadResumeOptionsForTask: dialog.loadResumeOptionsForTask,
        clearResumeOptions: dialog.clearResumeOptions,
        renderResumePanelState: dialog.renderResumePanelState,
        selectedResumeCheckpoint: dialog.selectedResumeCheckpoint,
        resumeTrainingFromCheckpoint: dialog.resumeTrainingFromCheckpoint,
        selectedHistoryManagerResumeCheckpoint: dialog.selectedHistoryManagerResumeCheckpoint,
        resumeTrainingFromHistoryDetail: dialog.resumeTrainingFromHistoryDetail,
        setResumeStatus: dialog.setResumeStatus,
        getCurrentPayload: dialog.getCurrentPayload,
        getActiveTab: dialog.getActiveTab,
        setActiveTab: dialog.setActiveTab,
        bindHistoryDetailEvents: dialog.bindHistoryDetailEvents,
        clearHistoryDetailState,
        resetCurveHover,
        normalizeHistoryDetailTab,
    };
}
