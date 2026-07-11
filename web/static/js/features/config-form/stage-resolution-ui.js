/**
 * Stage schedule dialog UI public surface.
 * Implementation split across stage-resolution-ui-widgets.js and stage-resolution-ui-dialog.js.
 */
export {
    createStageResolutionSummary,
    createStageResolutionChartPanel,
    createStageResolutionEditor,
    createStageResolutionTable,
} from './stage-resolution-ui-widgets.js?v=module-bootstrap-20260711-ir2';

export {
    drawStageResolutionChart,
    renderStageResolutionDialog,
    createOpenStageResolutionDialogButton,
    openStageResolutionDialog,
    createStageScheduleInlineSummary,
} from './stage-resolution-ui-dialog.js?v=module-bootstrap-20260711-ir2';
