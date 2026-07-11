const historyDetailHandlers = Object.create(null);

function requireHistoryDetailHandler(name) {
    const handler = historyDetailHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[history-detail] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureHistoryDetailBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            historyDetailHandlers[key] = handler;
        }
    }
}

export function ensureHistoryDetailFeature(...args) { return requireHistoryDetailHandler('ensureHistoryDetailFeature')(...args); }
export function getHistoryDetailFeature(...args) { return requireHistoryDetailHandler('getHistoryDetailFeature')(...args); }
export function isHistoryReviewMode(...args) { return requireHistoryDetailHandler('isHistoryReviewMode')(...args); }
