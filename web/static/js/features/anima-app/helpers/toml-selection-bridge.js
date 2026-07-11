const tomlSelectionHandlers = Object.create(null);

function requireTomlSelectionHandler(name) {
    const handler = tomlSelectionHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[toml-selection] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureTomlSelectionBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            tomlSelectionHandlers[key] = handler;
        }
    }
}

export function updateTomlSelectionUI(...args) { return requireTomlSelectionHandler('updateTomlSelectionUI')(...args); }
export function isTomlDirty(...args) { return requireTomlSelectionHandler('isTomlDirty')(...args); }
export function currentFormConfigFile(...args) { return requireTomlSelectionHandler('currentFormConfigFile')(...args); }
export function hasUnsavedFormChanges(...args) { return requireTomlSelectionHandler('hasUnsavedFormChanges')(...args); }
export function hasPendingConfigChanges(...args) { return requireTomlSelectionHandler('hasPendingConfigChanges')(...args); }
export function currentTomlEditorContentForFile(...args) { return requireTomlSelectionHandler('currentTomlEditorContentForFile')(...args); }
export function confirmDiscardTomlChanges(...args) { return requireTomlSelectionHandler('confirmDiscardTomlChanges')(...args); }
export function confirmUnsavedDiscard(...args) { return requireTomlSelectionHandler('confirmUnsavedDiscard')(...args); }
export function collectPendingConfigChangeDetails(...args) { return requireTomlSelectionHandler('collectPendingConfigChangeDetails')(...args); }
export function originalValueForChange(...args) { return requireTomlSelectionHandler('originalValueForChange')(...args); }
export function summarizeDatasetEditorState(...args) { return requireTomlSelectionHandler('summarizeDatasetEditorState')(...args); }
export function summarizeTextChange(...args) { return requireTomlSelectionHandler('summarizeTextChange')(...args); }
export function formatConfigChangeValue(...args) { return requireTomlSelectionHandler('formatConfigChangeValue')(...args); }
export function showConfigSwitchToast(...args) { return requireTomlSelectionHandler('showConfigSwitchToast')(...args); }
export function handlePendingConfigSwitch(...args) { return requireTomlSelectionHandler('handlePendingConfigSwitch')(...args); }
export function pendingConfigSwitchState(...args) { return requireTomlSelectionHandler('pendingConfigSwitchState')(...args); }
export function pendingToastLabel(...args) { return requireTomlSelectionHandler('pendingToastLabel')(...args); }
export function sharedHistoryTaskDialogParts(...args) { return requireTomlSelectionHandler('sharedHistoryTaskDialogParts')(...args); }
export function sharedHistoryTaskDialogIsOpen(...args) { return requireTomlSelectionHandler('sharedHistoryTaskDialogIsOpen')(...args); }
export function openSharedHistoryTaskDialog(...args) { return requireTomlSelectionHandler('openSharedHistoryTaskDialog')(...args); }
export function closeSharedHistoryTaskDialog(...args) { return requireTomlSelectionHandler('closeSharedHistoryTaskDialog')(...args); }
export function savePendingConfigSwitchChanges(...args) { return requireTomlSelectionHandler('savePendingConfigSwitchChanges')(...args); }
export function showUnsavedConfigSwitchDialog(...args) { return requireTomlSelectionHandler('showUnsavedConfigSwitchDialog')(...args); }
export function createConfigSwitchDialogBody(...args) { return requireTomlSelectionHandler('createConfigSwitchDialogBody')(...args); }
export function createConfigSwitchChangeValue(...args) { return requireTomlSelectionHandler('createConfigSwitchChangeValue')(...args); }
export function showAppConfirmDialog(...args) { return requireTomlSelectionHandler('showAppConfirmDialog')(...args); }
export function updateTomlDirtyState(...args) { return requireTomlSelectionHandler('updateTomlDirtyState')(...args); }
export function updateChangedFieldMarks(...args) { return requireTomlSelectionHandler('updateChangedFieldMarks')(...args); }
export function configFieldInputChanged(...args) { return requireTomlSelectionHandler('configFieldInputChanged')(...args); }
export function updateTomlBadges(...args) { return requireTomlSelectionHandler('updateTomlBadges')(...args); }
export function setBadge(...args) { return requireTomlSelectionHandler('setBadge')(...args); }
