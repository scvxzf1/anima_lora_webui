const legacyRoot = globalThis;

const historyDetailBridge = {
    ensureHistoryDetailFeature: (...args) => legacyRoot.ensureHistoryDetailFeature?.(...args),
    getHistoryDetailFeature: () => legacyRoot.historyDetailFeature ?? null,
    isHistoryReviewMode: (...args) => legacyRoot.isHistoryReviewMode?.(...args),
};

export function configureHistoryDetailBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in historyDetailBridge) {
            historyDetailBridge[key] = handler;
        }
    }
}

export function ensureHistoryDetailFeature(...args) {
    return historyDetailBridge.ensureHistoryDetailFeature(...args);
}

export function getHistoryDetailFeature() {
    return historyDetailBridge.getHistoryDetailFeature();
}

export function isHistoryReviewMode(...args) {
    return historyDetailBridge.isHistoryReviewMode(...args);
}
