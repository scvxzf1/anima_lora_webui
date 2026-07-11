const tomlActionStateHandlers = Object.create(null);

function requireTomlActionStateHandler(name) {
    const handler = tomlActionStateHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[toml-action-state] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureTomlActionStateBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            tomlActionStateHandlers[key] = handler;
        }
    }
}

export function updateTomlActionState(...args) { return requireTomlActionStateHandler('updateTomlActionState')(...args); }
export function isTomlLocked(...args) { return requireTomlActionStateHandler('isTomlLocked')(...args); }
export function applyTomlLockState(...args) { return requireTomlActionStateHandler('applyTomlLockState')(...args); }
export function setTomlEditorLocked(...args) { return requireTomlActionStateHandler('setTomlEditorLocked')(...args); }
export function updateTomlEditorPanelState(...args) { return requireTomlActionStateHandler('updateTomlEditorPanelState')(...args); }
export function toggleTomlEditorPanel(...args) { return requireTomlActionStateHandler('toggleTomlEditorPanel')(...args); }
export function copyTomlEditorContent(...args) { return requireTomlActionStateHandler('copyTomlEditorContent')(...args); }
export function tomlLockLabel(...args) { return requireTomlActionStateHandler('tomlLockLabel')(...args); }
export function tomlFileDisplayParts(...args) { return requireTomlActionStateHandler('tomlFileDisplayParts')(...args); }
export function tomlFileDisplayName(...args) { return requireTomlActionStateHandler('tomlFileDisplayName')(...args); }
export function lockTomlButtonTitle(...args) { return requireTomlActionStateHandler('lockTomlButtonTitle')(...args); }
export function deleteTomlButtonTitle(...args) { return requireTomlActionStateHandler('deleteTomlButtonTitle')(...args); }
export function resetTomlDeleteConfirm(...args) { return requireTomlActionStateHandler('resetTomlDeleteConfirm')(...args); }
export function armTomlDeleteConfirm(...args) { return requireTomlActionStateHandler('armTomlDeleteConfirm')(...args); }
export function resetTomlSaveConfirm(...args) { return requireTomlActionStateHandler('resetTomlSaveConfirm')(...args); }
export function armTomlSaveConfirm(...args) { return requireTomlActionStateHandler('armTomlSaveConfirm')(...args); }
export function setTomlStatus(...args) { return requireTomlActionStateHandler('setTomlStatus')(...args); }
export function applyTomlToConfig(...args) { return requireTomlActionStateHandler('applyTomlToConfig')(...args); }
export function toggleTomlUserLock(...args) { return requireTomlActionStateHandler('toggleTomlUserLock')(...args); }
export function toggleTomlGroupLock(...args) { return requireTomlActionStateHandler('toggleTomlGroupLock')(...args); }
export function createTomlGroup(...args) { return requireTomlActionStateHandler('createTomlGroup')(...args); }
export function renameTomlGroup(...args) { return requireTomlActionStateHandler('renameTomlGroup')(...args); }
