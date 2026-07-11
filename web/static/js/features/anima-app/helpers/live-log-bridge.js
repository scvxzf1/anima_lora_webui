const liveLogHandlers = Object.create(null);

function requireLiveLogHandler(name) {
    const handler = liveLogHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[live-log] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureLiveLogBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            liveLogHandlers[key] = handler;
        }
    }
}

export function stopTraining(...args) { return requireLiveLogHandler('stopTraining')(...args); }
export function connectWebSocket(...args) { return requireLiveLogHandler('connectWebSocket')(...args); }
export function handleWsMessage(...args) { return requireLiveLogHandler('handleWsMessage')(...args); }
export function appendLog(...args) { return requireLiveLogHandler('appendLog')(...args); }
export function appendLogRecord(...args) { return requireLiveLogHandler('appendLogRecord')(...args); }
export function renderLogOutputLines(...args) { return requireLiveLogHandler('renderLogOutputLines')(...args); }
export function currentLogOutputLines(...args) { return requireLiveLogHandler('currentLogOutputLines')(...args); }
export function resetLogOutputLines(...args) { return requireLiveLogHandler('resetLogOutputLines')(...args); }
export function scheduleLogRenderBatch(...args) { return requireLiveLogHandler('scheduleLogRenderBatch')(...args); }
export function logLineTone(...args) { return requireLiveLogHandler('logLineTone')(...args); }
export function scheduleLogFlush(...args) { return requireLiveLogHandler('scheduleLogFlush')(...args); }
export function flushLogBuffer(...args) { return requireLiveLogHandler('flushLogBuffer')(...args); }
export function replayTrainingLogs(...args) { return requireLiveLogHandler('replayTrainingLogs')(...args); }
export function replayMetricsHistory(...args) { return requireLiveLogHandler('replayMetricsHistory')(...args); }
export function replayMetricsFromLogRecord(...args) { return requireLiveLogHandler('replayMetricsFromLogRecord')(...args); }
export function setLogStatus(...args) { return requireLiveLogHandler('setLogStatus')(...args); }
export function updateLogStatusText(...args) { return requireLiveLogHandler('updateLogStatusText')(...args); }
export function setTrainingHealthNotice(...args) { return requireLiveLogHandler('setTrainingHealthNotice')(...args); }
export function recoverLiveTrainingState(...args) { return requireLiveLogHandler('recoverLiveTrainingState')(...args); }
