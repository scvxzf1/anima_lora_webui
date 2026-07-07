const legacyRoot = globalThis;

const tomlIoBridge = {
    importTomlFile: (...args) => legacyRoot.importTomlFile?.(...args),
    handleTomlImport: (...args) => legacyRoot.handleTomlImport?.(...args),
    exportTomlFile: (...args) => legacyRoot.exportTomlFile?.(...args),
    downloadTomlContent: (...args) => legacyRoot.downloadTomlContent?.(...args),
    triggerDownload: (...args) => legacyRoot.triggerDownload?.(...args),
    downloadBlob: (...args) => legacyRoot.downloadBlob?.(...args),
    createTomlZipBlob: (...args) => legacyRoot.createTomlZipBlob?.(...args),
    uniqueZipEntryName: (...args) => legacyRoot.uniqueZipEntryName?.(...args),
    saveTomlAs: (...args) => legacyRoot.saveTomlAs?.(...args),
    createBlankPresetFromLoraTemplate: (...args) => legacyRoot.createBlankPresetFromLoraTemplate?.(...args),
    previewPatchedTomlContent: (...args) => legacyRoot.previewPatchedTomlContent?.(...args),
    showTomlSaveAsDialog: (...args) => legacyRoot.showTomlSaveAsDialog?.(...args),
    saveAsTargetGroups: (...args) => legacyRoot.saveAsTargetGroups?.(...args),
    moveTomlFileToGroup: (...args) => legacyRoot.moveTomlFileToGroup?.(...args),
    normalizeTomlSaveAsPath: (...args) => legacyRoot.normalizeTomlSaveAsPath?.(...args),
    exportTomlFilename: (...args) => legacyRoot.exportTomlFilename?.(...args),
    isFixedSystemTomlGroup: (...args) => legacyRoot.isFixedSystemTomlGroup?.(...args),
    isDatasetConfigGroup: (...args) => legacyRoot.isDatasetConfigGroup?.(...args),
    isTrainingTomlGroup: (...args) => legacyRoot.isTrainingTomlGroup?.(...args),
    filterTrainingTomlGroups: (...args) => legacyRoot.filterTrainingTomlGroups?.(...args),
    shouldShowTomlGroup: (...args) => legacyRoot.shouldShowTomlGroup?.(...args),
    reorderTomlFileGroups: (...args) => legacyRoot.reorderTomlFileGroups?.(...args),
    getSortableTomlGroups: (...args) => legacyRoot.getSortableTomlGroups?.(...args),
    isTomlGroupDraggable: (...args) => legacyRoot.isTomlGroupDraggable?.(...args),
};

export function configureTomlIoBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in tomlIoBridge) {
            tomlIoBridge[key] = handler;
        }
    }
}

export function importTomlFile(...args) { return tomlIoBridge.importTomlFile(...args); }
export function handleTomlImport(...args) { return tomlIoBridge.handleTomlImport(...args); }
export function exportTomlFile(...args) { return tomlIoBridge.exportTomlFile(...args); }
export function downloadTomlContent(...args) { return tomlIoBridge.downloadTomlContent(...args); }
export function triggerDownload(...args) { return tomlIoBridge.triggerDownload(...args); }
export function downloadBlob(...args) { return tomlIoBridge.downloadBlob(...args); }
export function createTomlZipBlob(...args) { return tomlIoBridge.createTomlZipBlob(...args); }
export function uniqueZipEntryName(...args) { return tomlIoBridge.uniqueZipEntryName(...args); }
export function saveTomlAs(...args) { return tomlIoBridge.saveTomlAs(...args); }
export function createBlankPresetFromLoraTemplate(...args) { return tomlIoBridge.createBlankPresetFromLoraTemplate(...args); }
export function previewPatchedTomlContent(...args) { return tomlIoBridge.previewPatchedTomlContent(...args); }
export function showTomlSaveAsDialog(...args) { return tomlIoBridge.showTomlSaveAsDialog(...args); }
export function saveAsTargetGroups(...args) { return tomlIoBridge.saveAsTargetGroups(...args); }
export function moveTomlFileToGroup(...args) { return tomlIoBridge.moveTomlFileToGroup(...args); }
export function normalizeTomlSaveAsPath(...args) { return tomlIoBridge.normalizeTomlSaveAsPath(...args); }
export function exportTomlFilename(...args) { return tomlIoBridge.exportTomlFilename(...args); }
export function isFixedSystemTomlGroup(...args) { return tomlIoBridge.isFixedSystemTomlGroup(...args); }
export function isDatasetConfigGroup(...args) { return tomlIoBridge.isDatasetConfigGroup(...args); }
export function isTrainingTomlGroup(...args) { return tomlIoBridge.isTrainingTomlGroup(...args); }
export function filterTrainingTomlGroups(...args) { return tomlIoBridge.filterTrainingTomlGroups(...args); }
export function shouldShowTomlGroup(...args) { return tomlIoBridge.shouldShowTomlGroup(...args); }
export function reorderTomlFileGroups(...args) { return tomlIoBridge.reorderTomlFileGroups(...args); }
export function getSortableTomlGroups(...args) { return tomlIoBridge.getSortableTomlGroups(...args); }
export function isTomlGroupDraggable(...args) { return tomlIoBridge.isTomlGroupDraggable(...args); }
