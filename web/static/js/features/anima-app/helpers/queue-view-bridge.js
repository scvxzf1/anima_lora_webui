const legacyRoot = globalThis;

const queueViewBridge = {
    loadTrainingQueue: (...args) => legacyRoot.loadTrainingQueue?.(...args),
    updateTrainingQueueFromPayload: (...args) => legacyRoot.updateTrainingQueueFromPayload?.(...args),
    renderTrainingQueue: (...args) => legacyRoot.renderTrainingQueue?.(...args),
    refreshQueueRunningProgressViews: (...args) => legacyRoot.refreshQueueRunningProgressViews?.(...args),
    showTrainingView: (...args) => legacyRoot.showTrainingView?.(...args),
    trainingViewTabs: (...args) => legacyRoot.trainingViewTabs?.(...args),
    focusTrainingViewTab: (...args) => legacyRoot.focusTrainingViewTab?.(...args),
    activateTrainingViewTabButton: (...args) => legacyRoot.activateTrainingViewTabButton?.(...args),
    moveTrainingViewTabFocus: (...args) => legacyRoot.moveTrainingViewTabFocus?.(...args),
    bindTrainingViewTabKeyboard: (...args) => legacyRoot.bindTrainingViewTabKeyboard?.(...args),
    renderTrainingViewMode: (...args) => legacyRoot.renderTrainingViewMode?.(...args),
    resetTrainingExpandedStateOnLeave: (...args) => legacyRoot.resetTrainingExpandedStateOnLeave?.(...args),
};

export function configureQueueViewBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in queueViewBridge) {
            queueViewBridge[key] = handler;
        }
    }
}

export function loadTrainingQueue(...args) { return queueViewBridge.loadTrainingQueue(...args); }
export function updateTrainingQueueFromPayload(...args) { return queueViewBridge.updateTrainingQueueFromPayload(...args); }
export function renderTrainingQueue(...args) { return queueViewBridge.renderTrainingQueue(...args); }
export function refreshQueueRunningProgressViews(...args) { return queueViewBridge.refreshQueueRunningProgressViews(...args); }
export function showTrainingView(...args) { return queueViewBridge.showTrainingView(...args); }
export function trainingViewTabs(...args) { return queueViewBridge.trainingViewTabs(...args); }
export function focusTrainingViewTab(...args) { return queueViewBridge.focusTrainingViewTab(...args); }
export function activateTrainingViewTabButton(...args) { return queueViewBridge.activateTrainingViewTabButton(...args); }
export function moveTrainingViewTabFocus(...args) { return queueViewBridge.moveTrainingViewTabFocus(...args); }
export function bindTrainingViewTabKeyboard(...args) { return queueViewBridge.bindTrainingViewTabKeyboard(...args); }
export function renderTrainingViewMode(...args) { return queueViewBridge.renderTrainingViewMode(...args); }
export function resetTrainingExpandedStateOnLeave(...args) { return queueViewBridge.resetTrainingExpandedStateOnLeave(...args); }
