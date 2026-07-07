const legacyRoot = globalThis;

const historyTimelineBridge = {
    renderConfigGroupTimeline: (...args) => legacyRoot.renderConfigGroupTimeline?.(...args),
    formatGroupTimelineLogRecord: (...args) => legacyRoot.formatGroupTimelineLogRecord?.(...args),
    configGroupTimelineSummary: (...args) => legacyRoot.configGroupTimelineSummary?.(...args),
    formatStepRange: (...args) => legacyRoot.formatStepRange?.(...args),
    renderConfigGroupPaths: (...args) => legacyRoot.renderConfigGroupPaths?.(...args),
    configGroupLabel: (...args) => legacyRoot.configGroupLabel?.(...args),
    metricsWithProgressFallback: (...args) => legacyRoot.metricsWithProgressFallback?.(...args),
    metricIdentity: (...args) => legacyRoot.metricIdentity?.(...args),
    returnToLiveTraining: (...args) => legacyRoot.returnToLiveTraining?.(...args),
    loadResumeOptionsForTask: (...args) => legacyRoot.loadResumeOptionsForTask?.(...args),
    clearResumeOptions: (...args) => legacyRoot.clearResumeOptions?.(...args),
    renderResumePanelState: (...args) => legacyRoot.renderResumePanelState?.(...args),
    selectedResumeCheckpoint: (...args) => legacyRoot.selectedResumeCheckpoint?.(...args),
    resumeTrainingFromCheckpoint: (...args) => legacyRoot.resumeTrainingFromCheckpoint?.(...args),
    queueResumeTrainingFromCheckpoint: (...args) => legacyRoot.queueResumeTrainingFromCheckpoint?.(...args),
    setResumeStatus: (...args) => legacyRoot.setResumeStatus?.(...args),
    renderHistoryPaths: (...args) => legacyRoot.renderHistoryPaths?.(...args),
    runtimePathItems: (...args) => legacyRoot.runtimePathItems?.(...args),
    historyAbsolutePath: (...args) => legacyRoot.historyAbsolutePath?.(...args),
    historyProjectRoot: (...args) => legacyRoot.historyProjectRoot?.(...args),
    historyLooksProjectRelativePath: (...args) => legacyRoot.historyLooksProjectRelativePath?.(...args),
    historyCleanPath: (...args) => legacyRoot.historyCleanPath?.(...args),
    historyIsAbsolutePath: (...args) => legacyRoot.historyIsAbsolutePath?.(...args),
    historyIsSpecialPath: (...args) => legacyRoot.historyIsSpecialPath?.(...args),
    historyTrimPath: (...args) => legacyRoot.historyTrimPath?.(...args),
    historyJoinPath: (...args) => legacyRoot.historyJoinPath?.(...args),
    installSelectableHistoryPathText: (...args) => legacyRoot.installSelectableHistoryPathText?.(...args),
    historyArtifactUrl: (...args) => legacyRoot.historyArtifactUrl?.(...args),
    historyStateLabel: (...args) => legacyRoot.historyStateLabel?.(...args),
};

export function configureHistoryTimelineBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in historyTimelineBridge) {
            historyTimelineBridge[key] = handler;
        }
    }
}

export const renderConfigGroupTimeline = (...args) => historyTimelineBridge.renderConfigGroupTimeline(...args);
export const formatGroupTimelineLogRecord = (...args) => historyTimelineBridge.formatGroupTimelineLogRecord(...args);
export const configGroupTimelineSummary = (...args) => historyTimelineBridge.configGroupTimelineSummary(...args);
export const formatStepRange = (...args) => historyTimelineBridge.formatStepRange(...args);
export const renderConfigGroupPaths = (...args) => historyTimelineBridge.renderConfigGroupPaths(...args);
export const configGroupLabel = (...args) => historyTimelineBridge.configGroupLabel(...args);
export const metricsWithProgressFallback = (...args) => historyTimelineBridge.metricsWithProgressFallback(...args);
export const metricIdentity = (...args) => historyTimelineBridge.metricIdentity(...args);
export const returnToLiveTraining = (...args) => historyTimelineBridge.returnToLiveTraining(...args);
export const loadResumeOptionsForTask = (...args) => historyTimelineBridge.loadResumeOptionsForTask(...args);
export const clearResumeOptions = (...args) => historyTimelineBridge.clearResumeOptions(...args);
export const renderResumePanelState = (...args) => historyTimelineBridge.renderResumePanelState(...args);
export const selectedResumeCheckpoint = (...args) => historyTimelineBridge.selectedResumeCheckpoint(...args);
export const resumeTrainingFromCheckpoint = (...args) => historyTimelineBridge.resumeTrainingFromCheckpoint(...args);
export const queueResumeTrainingFromCheckpoint = (...args) => historyTimelineBridge.queueResumeTrainingFromCheckpoint(...args);
export const setResumeStatus = (...args) => historyTimelineBridge.setResumeStatus(...args);
export const renderHistoryPaths = (...args) => historyTimelineBridge.renderHistoryPaths(...args);
export const runtimePathItems = (...args) => historyTimelineBridge.runtimePathItems(...args);
export const historyAbsolutePath = (...args) => historyTimelineBridge.historyAbsolutePath(...args);
export const historyProjectRoot = (...args) => historyTimelineBridge.historyProjectRoot(...args);
export const historyLooksProjectRelativePath = (...args) => historyTimelineBridge.historyLooksProjectRelativePath(...args);
export const historyCleanPath = (...args) => historyTimelineBridge.historyCleanPath(...args);
export const historyIsAbsolutePath = (...args) => historyTimelineBridge.historyIsAbsolutePath(...args);
export const historyIsSpecialPath = (...args) => historyTimelineBridge.historyIsSpecialPath(...args);
export const historyTrimPath = (...args) => historyTimelineBridge.historyTrimPath(...args);
export const historyJoinPath = (...args) => historyTimelineBridge.historyJoinPath(...args);
export const installSelectableHistoryPathText = (...args) => historyTimelineBridge.installSelectableHistoryPathText(...args);
export const historyArtifactUrl = (...args) => historyTimelineBridge.historyArtifactUrl(...args);
export const historyStateLabel = (...args) => historyTimelineBridge.historyStateLabel(...args);
