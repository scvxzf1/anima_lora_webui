const historyTimelineHandlers = Object.create(null);

function requireHistoryTimelineHandler(name) {
    const handler = historyTimelineHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[history-timeline] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureHistoryTimelineBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            historyTimelineHandlers[key] = handler;
        }
    }
}

export function renderConfigGroupTimeline(...args) { return requireHistoryTimelineHandler('renderConfigGroupTimeline')(...args); }
export function formatGroupTimelineLogRecord(...args) { return requireHistoryTimelineHandler('formatGroupTimelineLogRecord')(...args); }
export function configGroupTimelineSummary(...args) { return requireHistoryTimelineHandler('configGroupTimelineSummary')(...args); }
export function formatStepRange(...args) { return requireHistoryTimelineHandler('formatStepRange')(...args); }
export function renderConfigGroupPaths(...args) { return requireHistoryTimelineHandler('renderConfigGroupPaths')(...args); }
export function configGroupLabel(...args) { return requireHistoryTimelineHandler('configGroupLabel')(...args); }
export function metricsWithProgressFallback(...args) { return requireHistoryTimelineHandler('metricsWithProgressFallback')(...args); }
export function metricIdentity(...args) { return requireHistoryTimelineHandler('metricIdentity')(...args); }
export function returnToLiveTraining(...args) { return requireHistoryTimelineHandler('returnToLiveTraining')(...args); }
export function loadResumeOptionsForTask(...args) { return requireHistoryTimelineHandler('loadResumeOptionsForTask')(...args); }
export function clearResumeOptions(...args) { return requireHistoryTimelineHandler('clearResumeOptions')(...args); }
export function renderResumePanelState(...args) { return requireHistoryTimelineHandler('renderResumePanelState')(...args); }
export function selectedResumeCheckpoint(...args) { return requireHistoryTimelineHandler('selectedResumeCheckpoint')(...args); }
export function resumeTrainingFromCheckpoint(...args) { return requireHistoryTimelineHandler('resumeTrainingFromCheckpoint')(...args); }
export function queueResumeTrainingFromCheckpoint(...args) { return requireHistoryTimelineHandler('queueResumeTrainingFromCheckpoint')(...args); }
export function setResumeStatus(...args) { return requireHistoryTimelineHandler('setResumeStatus')(...args); }
export function renderHistoryPaths(...args) { return requireHistoryTimelineHandler('renderHistoryPaths')(...args); }
export function runtimePathItems(...args) { return requireHistoryTimelineHandler('runtimePathItems')(...args); }
export function historyAbsolutePath(...args) { return requireHistoryTimelineHandler('historyAbsolutePath')(...args); }
export function historyProjectRoot(...args) { return requireHistoryTimelineHandler('historyProjectRoot')(...args); }
export function historyLooksProjectRelativePath(...args) { return requireHistoryTimelineHandler('historyLooksProjectRelativePath')(...args); }
export function historyCleanPath(...args) { return requireHistoryTimelineHandler('historyCleanPath')(...args); }
export function historyIsAbsolutePath(...args) { return requireHistoryTimelineHandler('historyIsAbsolutePath')(...args); }
export function historyIsSpecialPath(...args) { return requireHistoryTimelineHandler('historyIsSpecialPath')(...args); }
export function historyTrimPath(...args) { return requireHistoryTimelineHandler('historyTrimPath')(...args); }
export function historyJoinPath(...args) { return requireHistoryTimelineHandler('historyJoinPath')(...args); }
export function installSelectableHistoryPathText(...args) { return requireHistoryTimelineHandler('installSelectableHistoryPathText')(...args); }
export function historyArtifactUrl(...args) { return requireHistoryTimelineHandler('historyArtifactUrl')(...args); }
export function historyStateLabel(...args) { return requireHistoryTimelineHandler('historyStateLabel')(...args); }
