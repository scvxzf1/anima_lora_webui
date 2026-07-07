const legacyRoot = globalThis;

const tomlActionStateBridge = {
    updateTomlActionState: (...args) => legacyRoot.updateTomlActionState?.(...args),
    isTomlLocked: (...args) => legacyRoot.isTomlLocked?.(...args),
    applyTomlLockState: (...args) => legacyRoot.applyTomlLockState?.(...args),
    setTomlEditorLocked: (...args) => legacyRoot.setTomlEditorLocked?.(...args),
    updateTomlEditorPanelState: (...args) => legacyRoot.updateTomlEditorPanelState?.(...args),
    toggleTomlEditorPanel: (...args) => legacyRoot.toggleTomlEditorPanel?.(...args),
    copyTomlEditorContent: (...args) => legacyRoot.copyTomlEditorContent?.(...args),
    tomlLockLabel: (...args) => legacyRoot.tomlLockLabel?.(...args),
    tomlFileDisplayParts: (...args) => legacyRoot.tomlFileDisplayParts?.(...args),
    tomlFileDisplayName: (...args) => legacyRoot.tomlFileDisplayName?.(...args),
    lockTomlButtonTitle: (...args) => legacyRoot.lockTomlButtonTitle?.(...args),
    deleteTomlButtonTitle: (...args) => legacyRoot.deleteTomlButtonTitle?.(...args),
    resetTomlDeleteConfirm: (...args) => legacyRoot.resetTomlDeleteConfirm?.(...args),
    armTomlDeleteConfirm: (...args) => legacyRoot.armTomlDeleteConfirm?.(...args),
    resetTomlSaveConfirm: (...args) => legacyRoot.resetTomlSaveConfirm?.(...args),
    armTomlSaveConfirm: (...args) => legacyRoot.armTomlSaveConfirm?.(...args),
    setTomlStatus: (...args) => legacyRoot.setTomlStatus?.(...args),
    applyTomlToConfig: (...args) => legacyRoot.applyTomlToConfig?.(...args),
    toggleTomlUserLock: (...args) => legacyRoot.toggleTomlUserLock?.(...args),
    toggleTomlGroupLock: (...args) => legacyRoot.toggleTomlGroupLock?.(...args),
    createTomlGroup: (...args) => legacyRoot.createTomlGroup?.(...args),
    renameTomlGroup: (...args) => legacyRoot.renameTomlGroup?.(...args),
};

export function configureTomlActionStateBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in tomlActionStateBridge) {
            tomlActionStateBridge[key] = handler;
        }
    }
}

export const updateTomlActionState = (...args) => tomlActionStateBridge.updateTomlActionState(...args);
export const isTomlLocked = (...args) => tomlActionStateBridge.isTomlLocked(...args);
export const applyTomlLockState = (...args) => tomlActionStateBridge.applyTomlLockState(...args);
export const setTomlEditorLocked = (...args) => tomlActionStateBridge.setTomlEditorLocked(...args);
export const updateTomlEditorPanelState = (...args) => tomlActionStateBridge.updateTomlEditorPanelState(...args);
export const toggleTomlEditorPanel = (...args) => tomlActionStateBridge.toggleTomlEditorPanel(...args);
export const copyTomlEditorContent = (...args) => tomlActionStateBridge.copyTomlEditorContent(...args);
export const tomlLockLabel = (...args) => tomlActionStateBridge.tomlLockLabel(...args);
export const tomlFileDisplayParts = (...args) => tomlActionStateBridge.tomlFileDisplayParts(...args);
export const tomlFileDisplayName = (...args) => tomlActionStateBridge.tomlFileDisplayName(...args);
export const lockTomlButtonTitle = (...args) => tomlActionStateBridge.lockTomlButtonTitle(...args);
export const deleteTomlButtonTitle = (...args) => tomlActionStateBridge.deleteTomlButtonTitle(...args);
export const resetTomlDeleteConfirm = (...args) => tomlActionStateBridge.resetTomlDeleteConfirm(...args);
export const armTomlDeleteConfirm = (...args) => tomlActionStateBridge.armTomlDeleteConfirm(...args);
export const resetTomlSaveConfirm = (...args) => tomlActionStateBridge.resetTomlSaveConfirm(...args);
export const armTomlSaveConfirm = (...args) => tomlActionStateBridge.armTomlSaveConfirm(...args);
export const setTomlStatus = (...args) => tomlActionStateBridge.setTomlStatus(...args);
export const applyTomlToConfig = (...args) => tomlActionStateBridge.applyTomlToConfig(...args);
export const toggleTomlUserLock = (...args) => tomlActionStateBridge.toggleTomlUserLock(...args);
export const toggleTomlGroupLock = (...args) => tomlActionStateBridge.toggleTomlGroupLock(...args);
export const createTomlGroup = (...args) => tomlActionStateBridge.createTomlGroup(...args);
export const renameTomlGroup = (...args) => tomlActionStateBridge.renameTomlGroup(...args);
