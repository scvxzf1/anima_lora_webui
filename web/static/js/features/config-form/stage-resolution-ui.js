/**
 * Stage schedule dialog UI public surface.
 * Implementation split across stage-resolution-ui-widgets.js and stage-resolution-ui-dialog.js.
 */
export {
    createStageResolutionSummary,
    createStageResolutionChartPanel,
    createStageResolutionEditor,
    createStageResolutionTable,
    syncStageResolutionEditorInputs,
} from './stage-resolution-ui-widgets.js?v=module-bootstrap-20260831-release-v1';

export {
    drawStageResolutionChart,
    renderStageResolutionDialog,
    createOpenStageResolutionDialogButton,
    openStageResolutionDialog,
    createStageScheduleInlineSummary,
    resolveStageScheduleTargetFile,
    listStageScheduleTargetCandidates,
} from './stage-resolution-ui-dialog.js?v=module-bootstrap-20260901-dialog-v1';
