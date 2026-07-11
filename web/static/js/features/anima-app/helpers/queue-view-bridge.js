const queueViewHandlers = Object.create(null);

function requireQueueViewHandler(name) {
    const handler = queueViewHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[queue-view] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureQueueViewBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            queueViewHandlers[key] = handler;
        }
    }
}

export function loadTrainingQueue(...args) { return requireQueueViewHandler('loadTrainingQueue')(...args); }
export function updateTrainingQueueFromPayload(...args) { return requireQueueViewHandler('updateTrainingQueueFromPayload')(...args); }
export function renderTrainingQueue(...args) { return requireQueueViewHandler('renderTrainingQueue')(...args); }
export function refreshQueueRunningProgressViews(...args) { return requireQueueViewHandler('refreshQueueRunningProgressViews')(...args); }
export function showTrainingView(...args) { return requireQueueViewHandler('showTrainingView')(...args); }
export function trainingViewTabs(...args) { return requireQueueViewHandler('trainingViewTabs')(...args); }
export function focusTrainingViewTab(...args) { return requireQueueViewHandler('focusTrainingViewTab')(...args); }
export function activateTrainingViewTabButton(...args) { return requireQueueViewHandler('activateTrainingViewTabButton')(...args); }
export function moveTrainingViewTabFocus(...args) { return requireQueueViewHandler('moveTrainingViewTabFocus')(...args); }
export function bindTrainingViewTabKeyboard(...args) { return requireQueueViewHandler('bindTrainingViewTabKeyboard')(...args); }
export function renderTrainingViewMode(...args) { return requireQueueViewHandler('renderTrainingViewMode')(...args); }
export function resetTrainingExpandedStateOnLeave(...args) { return requireQueueViewHandler('resetTrainingExpandedStateOnLeave')(...args); }
