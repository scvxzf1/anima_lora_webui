/**
 * Stage schedule public surface.
 * Implementation split across stage-resolution-model.js and stage-resolution-ui.js.
 */
export {
    STAGE_COLORS,
    clamp01,
    defaultStageScheduleStages,
    hydrateStageScheduleFromConfig,
    hydrateStageScheduleFromDatasetPreset,
    listSubsetOptions,
    normalizeRawStages,
    normalizedStageResolutionStages,
    pctLabel,
    pickDatasetRows,
    readTotalSteps,
    resolveStageScheduleSource,
    stageResolutionMetrics,
    stageResolutionStatus,
    stageSchedulePayload,
    toFraction,
} from './stage-resolution-model.js?v=module-bootstrap-20260831-release-v1';

export {
    createOpenStageResolutionDialogButton,
    createStageResolutionChartPanel,
    createStageResolutionEditor,
    createStageResolutionSummary,
    createStageResolutionTable,
    createStageScheduleInlineSummary,
    drawStageResolutionChart,
    openStageResolutionDialog,
    renderStageResolutionDialog,
    resolveStageScheduleTargetFile,
    listStageScheduleTargetCandidates,
} from './stage-resolution-ui.js?v=module-bootstrap-20260831-release-v1';

// Compatibility exports used by old chunk imports / quick-preset panel co-location.
export {
    createFillGlobalModelPathsButton,
    createResourceQuickPresetsButton,
    createResourceQuickPresetPanel,
    createNoDatasetRegularizationQuickPresetsButton,
    createNoDatasetRegularizationQuickPresetPanel,
} from './stage-resolution-presets.js?v=module-bootstrap-20260831-release-v1';
