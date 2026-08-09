/**
 * Compatibility barrel for former chunk 03.
 * Implementations live in config-form/step-estimate.js, dataset-editor/load.js,
 * and live-training/dashboard-ui.js.
 */
export {
    loadStepEstimate,
    createStepEstimatePanel,
    scheduleStepEstimatePanelRefresh,
    updateStepEstimatePanel,
} from '../../config-form/step-estimate.js?v=module-bootstrap-20260809-nf4-v2';
export {
    loadDatasetEditor,
    loadDatasetPresets,
    loadDatasetPreset,
} from '../../dataset-editor/load.js?v=module-bootstrap-20260809-nf4-v2';
export {
    setText,
    metricValueIsEmpty,
    setMetricText,
    setEtaMetricText,
    resetLiveMetricPlaceholders,
    updateDashboardProgressIdleState,
    setTrainingDashboardHeadState,
    syncLossChartEmptyState,
    syncLiveChartControls,
    renderLiveChartPanel,
    updateTrainingToolbarState,
} from '../../live-training/dashboard-ui.js?v=module-bootstrap-20260809-nf4-v2';
