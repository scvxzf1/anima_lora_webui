const legacyRoot = globalThis;

const tomlSelectionBridge = {
    updateTomlSelectionUI: (...args) => legacyRoot.updateTomlSelectionUI?.(...args),
    isTomlDirty: (...args) => legacyRoot.isTomlDirty?.(...args),
    currentFormConfigFile: (...args) => legacyRoot.currentFormConfigFile?.(...args),
    hasUnsavedFormChanges: (...args) => legacyRoot.hasUnsavedFormChanges?.(...args),
    hasPendingConfigChanges: (...args) => legacyRoot.hasPendingConfigChanges?.(...args),
    currentTomlEditorContentForFile: (...args) => legacyRoot.currentTomlEditorContentForFile?.(...args),
    confirmDiscardTomlChanges: (...args) => legacyRoot.confirmDiscardTomlChanges?.(...args),
    confirmUnsavedDiscard: (...args) => legacyRoot.confirmUnsavedDiscard?.(...args),
    collectPendingConfigChangeDetails: (...args) => legacyRoot.collectPendingConfigChangeDetails?.(...args),
    originalValueForChange: (...args) => legacyRoot.originalValueForChange?.(...args),
    summarizeDatasetEditorState: (...args) => legacyRoot.summarizeDatasetEditorState?.(...args),
    summarizeTextChange: (...args) => legacyRoot.summarizeTextChange?.(...args),
    formatConfigChangeValue: (...args) => legacyRoot.formatConfigChangeValue?.(...args),
    showConfigSwitchToast: (...args) => legacyRoot.showConfigSwitchToast?.(...args),
    handlePendingConfigSwitch: (...args) => legacyRoot.handlePendingConfigSwitch?.(...args),
    pendingConfigSwitchState: (...args) => legacyRoot.pendingConfigSwitchState?.(...args),
    pendingToastLabel: (...args) => legacyRoot.pendingToastLabel?.(...args),
    sharedHistoryTaskDialogParts: (...args) => legacyRoot.sharedHistoryTaskDialogParts?.(...args),
    sharedHistoryTaskDialogIsOpen: (...args) => legacyRoot.sharedHistoryTaskDialogIsOpen?.(...args),
    openSharedHistoryTaskDialog: (...args) => legacyRoot.openSharedHistoryTaskDialog?.(...args),
    closeSharedHistoryTaskDialog: (...args) => legacyRoot.closeSharedHistoryTaskDialog?.(...args),
    savePendingConfigSwitchChanges: (...args) => legacyRoot.savePendingConfigSwitchChanges?.(...args),
    showUnsavedConfigSwitchDialog: (...args) => legacyRoot.showUnsavedConfigSwitchDialog?.(...args),
    createConfigSwitchDialogBody: (...args) => legacyRoot.createConfigSwitchDialogBody?.(...args),
    createConfigSwitchChangeValue: (...args) => legacyRoot.createConfigSwitchChangeValue?.(...args),
    showAppConfirmDialog: (...args) => legacyRoot.showAppConfirmDialog?.(...args),
    updateTomlDirtyState: (...args) => legacyRoot.updateTomlDirtyState?.(...args),
    updateChangedFieldMarks: (...args) => legacyRoot.updateChangedFieldMarks?.(...args),
    configFieldInputChanged: (...args) => legacyRoot.configFieldInputChanged?.(...args),
    updateTomlBadges: (...args) => legacyRoot.updateTomlBadges?.(...args),
    setBadge: (...args) => legacyRoot.setBadge?.(...args),
};

export function configureTomlSelectionBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in tomlSelectionBridge) {
            tomlSelectionBridge[key] = handler;
        }
    }
}

export function updateTomlSelectionUI(...args) { return tomlSelectionBridge.updateTomlSelectionUI(...args); }
export function isTomlDirty(...args) { return tomlSelectionBridge.isTomlDirty(...args); }
export function currentFormConfigFile(...args) { return tomlSelectionBridge.currentFormConfigFile(...args); }
export function hasUnsavedFormChanges(...args) { return tomlSelectionBridge.hasUnsavedFormChanges(...args); }
export function hasPendingConfigChanges(...args) { return tomlSelectionBridge.hasPendingConfigChanges(...args); }
export function currentTomlEditorContentForFile(...args) { return tomlSelectionBridge.currentTomlEditorContentForFile(...args); }
export function confirmDiscardTomlChanges(...args) { return tomlSelectionBridge.confirmDiscardTomlChanges(...args); }
export function confirmUnsavedDiscard(...args) { return tomlSelectionBridge.confirmUnsavedDiscard(...args); }
export function collectPendingConfigChangeDetails(...args) { return tomlSelectionBridge.collectPendingConfigChangeDetails(...args); }
export function originalValueForChange(...args) { return tomlSelectionBridge.originalValueForChange(...args); }
export function summarizeDatasetEditorState(...args) { return tomlSelectionBridge.summarizeDatasetEditorState(...args); }
export function summarizeTextChange(...args) { return tomlSelectionBridge.summarizeTextChange(...args); }
export function formatConfigChangeValue(...args) { return tomlSelectionBridge.formatConfigChangeValue(...args); }
export function showConfigSwitchToast(...args) { return tomlSelectionBridge.showConfigSwitchToast(...args); }
export function handlePendingConfigSwitch(...args) { return tomlSelectionBridge.handlePendingConfigSwitch(...args); }
export function pendingConfigSwitchState(...args) { return tomlSelectionBridge.pendingConfigSwitchState(...args); }
export function pendingToastLabel(...args) { return tomlSelectionBridge.pendingToastLabel(...args); }
export function sharedHistoryTaskDialogParts(...args) { return tomlSelectionBridge.sharedHistoryTaskDialogParts(...args); }
export function sharedHistoryTaskDialogIsOpen(...args) { return tomlSelectionBridge.sharedHistoryTaskDialogIsOpen(...args); }
export function openSharedHistoryTaskDialog(...args) { return tomlSelectionBridge.openSharedHistoryTaskDialog(...args); }
export function closeSharedHistoryTaskDialog(...args) { return tomlSelectionBridge.closeSharedHistoryTaskDialog(...args); }
export function savePendingConfigSwitchChanges(...args) { return tomlSelectionBridge.savePendingConfigSwitchChanges(...args); }
export function showUnsavedConfigSwitchDialog(...args) { return tomlSelectionBridge.showUnsavedConfigSwitchDialog(...args); }
export function createConfigSwitchDialogBody(...args) { return tomlSelectionBridge.createConfigSwitchDialogBody(...args); }
export function createConfigSwitchChangeValue(...args) { return tomlSelectionBridge.createConfigSwitchChangeValue(...args); }
export function showAppConfirmDialog(...args) { return tomlSelectionBridge.showAppConfirmDialog(...args); }
export function updateTomlDirtyState(...args) { return tomlSelectionBridge.updateTomlDirtyState(...args); }
export function updateChangedFieldMarks(...args) { return tomlSelectionBridge.updateChangedFieldMarks(...args); }
export function configFieldInputChanged(...args) { return tomlSelectionBridge.configFieldInputChanged(...args); }
export function updateTomlBadges(...args) { return tomlSelectionBridge.updateTomlBadges(...args); }
export function setBadge(...args) { return tomlSelectionBridge.setBadge(...args); }
