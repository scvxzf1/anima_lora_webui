import { createHistoryResumeActions } from './actions.js?v=module-bootstrap-20260707-93';
import { createHistoryResumeDetailRenderer } from './detail.js?v=module-bootstrap-20260707-93';
import { createHistoryResumePanelRenderer } from './panel.js?v=module-bootstrap-20260707-93';

export function createHistoryResumeFeature({ ctx, state, deps, slots }) {
    const panel = createHistoryResumePanelRenderer({ state, deps, slots });
    const actions = createHistoryResumeActions({
        ctx,
        state,
        deps,
        slots,
        renderResumePanelState: panel.renderResumePanelState,
    });
    const detail = createHistoryResumeDetailRenderer({ ctx, state, deps, slots, actions });

    return {
        clearResumeOptions: actions.clearResumeOptions,
        loadResumeOptionsForTask: actions.loadResumeOptionsForTask,
        renderResumePanelState: panel.renderResumePanelState,
        renderHistoryDetailResume: detail.renderHistoryDetailResume,
        selectedResumeCheckpoint: actions.selectedResumeCheckpoint,
        resumeTrainingFromCheckpoint: actions.resumeTrainingFromCheckpoint,
        selectedHistoryManagerResumeCheckpoint: actions.selectedHistoryManagerResumeCheckpoint,
        resumeTrainingFromHistoryDetail: actions.resumeTrainingFromHistoryDetail,
        setResumeStatus: actions.setResumeStatus,
        setResumeLoadingForTask: actions.setResumeLoadingForTask,
    };
}
