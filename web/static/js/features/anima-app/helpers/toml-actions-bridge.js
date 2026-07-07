const legacyRoot = globalThis;

const tomlActionsBridge = {
    moveCurrentTomlToGroup: (...args) => legacyRoot.moveCurrentTomlToGroup?.(...args),
    getMovableTomlGroups: (...args) => legacyRoot.getMovableTomlGroups?.(...args),
    deleteTomlGroupButtonTitle: (...args) => legacyRoot.deleteTomlGroupButtonTitle?.(...args),
    canDeleteTomlGroup: (...args) => legacyRoot.canDeleteTomlGroup?.(...args),
    showMoveTomlDialog: (...args) => legacyRoot.showMoveTomlDialog?.(...args),
    deleteTomlGroup: (...args) => legacyRoot.deleteTomlGroup?.(...args),
    deleteTomlFile: (...args) => legacyRoot.deleteTomlFile?.(...args),
    isMissingTomlFileResponse: (...args) => legacyRoot.isMissingTomlFileResponse?.(...args),
    handleDeletedTomlSelection: (...args) => legacyRoot.handleDeletedTomlSelection?.(...args),
    clearCurrentTomlSelection: (...args) => legacyRoot.clearCurrentTomlSelection?.(...args),
    restoreSystemTomlPresets: (...args) => legacyRoot.restoreSystemTomlPresets?.(...args),
};

export function configureTomlActionsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in tomlActionsBridge) {
            tomlActionsBridge[key] = handler;
        }
    }
}

export const moveCurrentTomlToGroup = (...args) => tomlActionsBridge.moveCurrentTomlToGroup(...args);
export const getMovableTomlGroups = (...args) => tomlActionsBridge.getMovableTomlGroups(...args);
export const deleteTomlGroupButtonTitle = (...args) => tomlActionsBridge.deleteTomlGroupButtonTitle(...args);
export const canDeleteTomlGroup = (...args) => tomlActionsBridge.canDeleteTomlGroup(...args);
export const showMoveTomlDialog = (...args) => tomlActionsBridge.showMoveTomlDialog(...args);
export const deleteTomlGroup = (...args) => tomlActionsBridge.deleteTomlGroup(...args);
export const deleteTomlFile = (...args) => tomlActionsBridge.deleteTomlFile(...args);
export const isMissingTomlFileResponse = (...args) => tomlActionsBridge.isMissingTomlFileResponse(...args);
export const handleDeletedTomlSelection = (...args) => tomlActionsBridge.handleDeletedTomlSelection(...args);
export const clearCurrentTomlSelection = (...args) => tomlActionsBridge.clearCurrentTomlSelection(...args);
export const restoreSystemTomlPresets = (...args) => tomlActionsBridge.restoreSystemTomlPresets(...args);
