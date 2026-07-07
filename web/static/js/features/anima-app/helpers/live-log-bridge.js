const legacyRoot = globalThis;

const liveLogBridge = {
    stopTraining: (...args) => legacyRoot.stopTraining?.(...args),
    connectWebSocket: (...args) => legacyRoot.connectWebSocket?.(...args),
    handleWsMessage: (...args) => legacyRoot.handleWsMessage?.(...args),
    appendLog: (...args) => legacyRoot.appendLog?.(...args),
    appendLogRecord: (...args) => legacyRoot.appendLogRecord?.(...args),
    renderLogOutputLines: (...args) => legacyRoot.renderLogOutputLines?.(...args),
    currentLogOutputLines: (...args) => legacyRoot.currentLogOutputLines?.(...args),
    resetLogOutputLines: (...args) => legacyRoot.resetLogOutputLines?.(...args),
    scheduleLogRenderBatch: (...args) => legacyRoot.scheduleLogRenderBatch?.(...args),
    logLineTone: (...args) => legacyRoot.logLineTone?.(...args),
    scheduleLogFlush: (...args) => legacyRoot.scheduleLogFlush?.(...args),
    flushLogBuffer: (...args) => legacyRoot.flushLogBuffer?.(...args),
    replayTrainingLogs: (...args) => legacyRoot.replayTrainingLogs?.(...args),
    replayMetricsHistory: (...args) => legacyRoot.replayMetricsHistory?.(...args),
    replayMetricsFromLogRecord: (...args) => legacyRoot.replayMetricsFromLogRecord?.(...args),
    setLogStatus: (...args) => legacyRoot.setLogStatus?.(...args),
    updateLogStatusText: (...args) => legacyRoot.updateLogStatusText?.(...args),
    setTrainingHealthNotice: (...args) => legacyRoot.setTrainingHealthNotice?.(...args),
    recoverLiveTrainingState: (...args) => legacyRoot.recoverLiveTrainingState?.(...args),
};

export function configureLiveLogBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in liveLogBridge) {
            liveLogBridge[key] = handler;
        }
    }
}

export function stopTraining(...args) { return liveLogBridge.stopTraining(...args); }
export function connectWebSocket(...args) { return liveLogBridge.connectWebSocket(...args); }
export function handleWsMessage(...args) { return liveLogBridge.handleWsMessage(...args); }
export function appendLog(...args) { return liveLogBridge.appendLog(...args); }
export function appendLogRecord(...args) { return liveLogBridge.appendLogRecord(...args); }
export function renderLogOutputLines(...args) { return liveLogBridge.renderLogOutputLines(...args); }
export function currentLogOutputLines(...args) { return liveLogBridge.currentLogOutputLines(...args); }
export function resetLogOutputLines(...args) { return liveLogBridge.resetLogOutputLines(...args); }
export function scheduleLogRenderBatch(...args) { return liveLogBridge.scheduleLogRenderBatch(...args); }
export function logLineTone(...args) { return liveLogBridge.logLineTone(...args); }
export function scheduleLogFlush(...args) { return liveLogBridge.scheduleLogFlush(...args); }
export function flushLogBuffer(...args) { return liveLogBridge.flushLogBuffer(...args); }
export function replayTrainingLogs(...args) { return liveLogBridge.replayTrainingLogs(...args); }
export function replayMetricsHistory(...args) { return liveLogBridge.replayMetricsHistory(...args); }
export function replayMetricsFromLogRecord(...args) { return liveLogBridge.replayMetricsFromLogRecord(...args); }
export function setLogStatus(...args) { return liveLogBridge.setLogStatus(...args); }
export function updateLogStatusText(...args) { return liveLogBridge.updateLogStatusText(...args); }
export function setTrainingHealthNotice(...args) { return liveLogBridge.setTrainingHealthNotice(...args); }
export function recoverLiveTrainingState(...args) { return liveLogBridge.recoverLiveTrainingState(...args); }
