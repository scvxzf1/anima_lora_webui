const liveStatusHandlers = Object.create(null);

function requireLiveStatusHandler(name) {
    const handler = liveStatusHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[live-status] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureLiveStatusBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            liveStatusHandlers[key] = handler;
        }
    }
}

export function updateProgress(...args) { return requireLiveStatusHandler('updateProgress')(...args); }
export function updateMetrics(...args) { return requireLiveStatusHandler('updateMetrics')(...args); }
export function updateStatus(...args) { return requireLiveStatusHandler('updateStatus')(...args); }
export function liveStatusState(...args) { return requireLiveStatusHandler('liveStatusState')(...args); }
export function terminalStatusMessage(...args) { return requireLiveStatusHandler('terminalStatusMessage')(...args); }
export function resetLiveSystemPeaks(...args) { return requireLiveStatusHandler('resetLiveSystemPeaks')(...args); }
export function clearRuntimeInfo(...args) { return requireLiveStatusHandler('clearRuntimeInfo')(...args); }
export function applyRuntimeInfoToState(...args) { return requireLiveStatusHandler('applyRuntimeInfoToState')(...args); }
export function renderCurrentRuntimePaths(...args) { return requireLiveStatusHandler('renderCurrentRuntimePaths')(...args); }
export function currentRuntimeTaskInfo(...args) { return requireLiveStatusHandler('currentRuntimeTaskInfo')(...args); }
export function updateSystem(...args) { return requireLiveStatusHandler('updateSystem')(...args); }
export function formatRuntimeVram(...args) { return requireLiveStatusHandler('formatRuntimeVram')(...args); }
export function renderTrainingRunSummary(...args) { return requireLiveStatusHandler('renderTrainingRunSummary')(...args); }
export function renderLiveTrainingDashboard(...args) { return requireLiveStatusHandler('renderLiveTrainingDashboard')(...args); }
export function trainingEtaMetricInfo(...args) { return requireLiveStatusHandler('trainingEtaMetricInfo')(...args); }
export function markTrainingActivity(...args) { return requireLiveStatusHandler('markTrainingActivity')(...args); }
export function refreshTrainingHealth(...args) { return requireLiveStatusHandler('refreshTrainingHealth')(...args); }
export function formatDuration(...args) { return requireLiveStatusHandler('formatDuration')(...args); }
