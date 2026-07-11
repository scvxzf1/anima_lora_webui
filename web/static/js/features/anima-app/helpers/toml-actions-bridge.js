const tomlActionsHandlers = Object.create(null);

function requireTomlActionsHandler(name) {
    const handler = tomlActionsHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[toml-actions] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureTomlActionsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            tomlActionsHandlers[key] = handler;
        }
    }
}

export function moveCurrentTomlToGroup(...args) { return requireTomlActionsHandler('moveCurrentTomlToGroup')(...args); }
export function getMovableTomlGroups(...args) { return requireTomlActionsHandler('getMovableTomlGroups')(...args); }
export function deleteTomlGroupButtonTitle(...args) { return requireTomlActionsHandler('deleteTomlGroupButtonTitle')(...args); }
export function canDeleteTomlGroup(...args) { return requireTomlActionsHandler('canDeleteTomlGroup')(...args); }
export function showMoveTomlDialog(...args) { return requireTomlActionsHandler('showMoveTomlDialog')(...args); }
export function deleteTomlGroup(...args) { return requireTomlActionsHandler('deleteTomlGroup')(...args); }
export function deleteTomlFile(...args) { return requireTomlActionsHandler('deleteTomlFile')(...args); }
export function isMissingTomlFileResponse(...args) { return requireTomlActionsHandler('isMissingTomlFileResponse')(...args); }
export function handleDeletedTomlSelection(...args) { return requireTomlActionsHandler('handleDeletedTomlSelection')(...args); }
export function clearCurrentTomlSelection(...args) { return requireTomlActionsHandler('clearCurrentTomlSelection')(...args); }
export function restoreSystemTomlPresets(...args) { return requireTomlActionsHandler('restoreSystemTomlPresets')(...args); }
