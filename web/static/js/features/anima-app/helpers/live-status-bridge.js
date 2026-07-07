const legacyRoot = globalThis;

const liveStatusBridge = {
    updateProgress: (...args) => legacyRoot.updateProgress?.(...args),
    updateMetrics: (...args) => legacyRoot.updateMetrics?.(...args),
    updateStatus: (...args) => legacyRoot.updateStatus?.(...args),
    liveStatusState: (...args) => legacyRoot.liveStatusState?.(...args),
    terminalStatusMessage: (...args) => legacyRoot.terminalStatusMessage?.(...args),
    resetLiveSystemPeaks: (...args) => legacyRoot.resetLiveSystemPeaks?.(...args),
    clearRuntimeInfo: (...args) => legacyRoot.clearRuntimeInfo?.(...args),
    applyRuntimeInfoToState: (...args) => legacyRoot.applyRuntimeInfoToState?.(...args),
    renderCurrentRuntimePaths: (...args) => legacyRoot.renderCurrentRuntimePaths?.(...args),
    currentRuntimeTaskInfo: (...args) => legacyRoot.currentRuntimeTaskInfo?.(...args),
    updateSystem: (...args) => legacyRoot.updateSystem?.(...args),
    formatRuntimeVram: (...args) => legacyRoot.formatRuntimeVram?.(...args),
    renderTrainingRunSummary: (...args) => legacyRoot.renderTrainingRunSummary?.(...args),
    renderLiveTrainingDashboard: (...args) => legacyRoot.renderLiveTrainingDashboard?.(...args),
    trainingEtaMetricInfo: (...args) => legacyRoot.trainingEtaMetricInfo?.(...args),
    markTrainingActivity: (...args) => legacyRoot.markTrainingActivity?.(...args),
    refreshTrainingHealth: (...args) => legacyRoot.refreshTrainingHealth?.(...args),
    formatDuration: (...args) => legacyRoot.formatDuration?.(...args),
};

export function configureLiveStatusBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in liveStatusBridge) {
            liveStatusBridge[key] = handler;
        }
    }
}

export function updateProgress(...args) { return liveStatusBridge.updateProgress(...args); }
export function updateMetrics(...args) { return liveStatusBridge.updateMetrics(...args); }
export function updateStatus(...args) { return liveStatusBridge.updateStatus(...args); }
export function liveStatusState(...args) { return liveStatusBridge.liveStatusState(...args); }
export function terminalStatusMessage(...args) { return liveStatusBridge.terminalStatusMessage(...args); }
export function resetLiveSystemPeaks(...args) { return liveStatusBridge.resetLiveSystemPeaks(...args); }
export function clearRuntimeInfo(...args) { return liveStatusBridge.clearRuntimeInfo(...args); }
export function applyRuntimeInfoToState(...args) { return liveStatusBridge.applyRuntimeInfoToState(...args); }
export function renderCurrentRuntimePaths(...args) { return liveStatusBridge.renderCurrentRuntimePaths(...args); }
export function currentRuntimeTaskInfo(...args) { return liveStatusBridge.currentRuntimeTaskInfo(...args); }
export function updateSystem(...args) { return liveStatusBridge.updateSystem(...args); }
export function formatRuntimeVram(...args) { return liveStatusBridge.formatRuntimeVram(...args); }
export function renderTrainingRunSummary(...args) { return liveStatusBridge.renderTrainingRunSummary(...args); }
export function renderLiveTrainingDashboard(...args) { return liveStatusBridge.renderLiveTrainingDashboard(...args); }
export function trainingEtaMetricInfo(...args) { return liveStatusBridge.trainingEtaMetricInfo(...args); }
export function markTrainingActivity(...args) { return liveStatusBridge.markTrainingActivity(...args); }
export function refreshTrainingHealth(...args) { return liveStatusBridge.refreshTrainingHealth(...args); }
export function formatDuration(...args) { return liveStatusBridge.formatDuration(...args); }
