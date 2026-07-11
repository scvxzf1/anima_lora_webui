const tomlDragHandlers = Object.create(null);

function requireTomlDragHandler(name) {
    const handler = tomlDragHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[toml-drag] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureTomlDragBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            tomlDragHandlers[key] = handler;
        }
    }
}

export function canDropTomlFileToGroup(...args) { return requireTomlDragHandler('canDropTomlFileToGroup')(...args); }
export function isTomlFileDraggable(...args) { return requireTomlDragHandler('isTomlFileDraggable')(...args); }
export function createTomlGroupDragHandle(...args) { return requireTomlDragHandler('createTomlGroupDragHandle')(...args); }
export function placeTomlGroup(...args) { return requireTomlDragHandler('placeTomlGroup')(...args); }
export function placeTomlFile(...args) { return requireTomlDragHandler('placeTomlFile')(...args); }
export function tomlFileDragOptions(...args) { return requireTomlDragHandler('tomlFileDragOptions')(...args); }
export function tomlGroupDragOptions(...args) { return requireTomlDragHandler('tomlGroupDragOptions')(...args); }
export function populateTomlFileSelect(...args) { return requireTomlDragHandler('populateTomlFileSelect')(...args); }
export function renderTomlFileGroups(...args) { return requireTomlDragHandler('renderTomlFileGroups')(...args); }
export function renderTomlFileGroupList(...args) { return requireTomlDragHandler('renderTomlFileGroupList')(...args); }
export function createTomlGroupActions(...args) { return requireTomlDragHandler('createTomlGroupActions')(...args); }
export function exportableTomlGroupFiles(...args) { return requireTomlDragHandler('exportableTomlGroupFiles')(...args); }
export function exportTomlGroup(...args) { return requireTomlDragHandler('exportTomlGroup')(...args); }
export function exportTomlGroupFilename(...args) { return requireTomlDragHandler('exportTomlGroupFilename')(...args); }
export function queueableTomlGroupFiles(...args) { return requireTomlDragHandler('queueableTomlGroupFiles')(...args); }
export function tomlItemQueueVariant(...args) { return requireTomlDragHandler('tomlItemQueueVariant')(...args); }
export function tomlItemQueueEntry(...args) { return requireTomlDragHandler('tomlItemQueueEntry')(...args); }
export function tomlGroupQueueFailureLabel(...args) { return requireTomlDragHandler('tomlGroupQueueFailureLabel')(...args); }
export function showTomlGroupQueueConfirmDialog(...args) { return requireTomlDragHandler('showTomlGroupQueueConfirmDialog')(...args); }
export function enqueueTomlGroupToQueue(...args) { return requireTomlDragHandler('enqueueTomlGroupToQueue')(...args); }
export function createTomlGroupActionButton(...args) { return requireTomlDragHandler('createTomlGroupActionButton')(...args); }
export function runTomlGroupAction(...args) { return requireTomlDragHandler('runTomlGroupAction')(...args); }
export function createTomlFileButton(...args) { return requireTomlDragHandler('createTomlFileButton')(...args); }
