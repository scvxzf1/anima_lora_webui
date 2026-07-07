let scheduleStatusPollHandler = null;
let pollStatusHandler = null;

export function configureStatusPollingBridge(handlers = {}) {
    if (typeof handlers.scheduleStatusPoll === 'function') {
        scheduleStatusPollHandler = handlers.scheduleStatusPoll;
    }
    if (typeof handlers.pollStatus === 'function') {
        pollStatusHandler = handlers.pollStatus;
    }
}

export function scheduleStatusPoll(...args) {
    if (!scheduleStatusPollHandler) {
        throw new Error('status polling bridge is not configured');
    }
    return scheduleStatusPollHandler(...args);
}

export function pollStatus(...args) {
    if (!pollStatusHandler) {
        throw new Error('status polling bridge is not configured');
    }
    return pollStatusHandler(...args);
}
