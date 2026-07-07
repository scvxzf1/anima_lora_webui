const legacyRoot = globalThis;

const datasetPresetActionsBridge = {
    applySelectedDatasetPresetToCurrentConfig: (...args) => legacyRoot.applySelectedDatasetPresetToCurrentConfig?.(...args),
    saveDatasetPresetEditor: (...args) => legacyRoot.saveDatasetPresetEditor?.(...args),
    createNewDatasetPreset: (...args) => legacyRoot.createNewDatasetPreset?.(...args),
    copyDatasetPreset: (...args) => legacyRoot.copyDatasetPreset?.(...args),
    renameDatasetPreset: (...args) => legacyRoot.renameDatasetPreset?.(...args),
    copyDatasetPresetToName: (...args) => legacyRoot.copyDatasetPresetToName?.(...args),
    deleteDatasetPreset: (...args) => legacyRoot.deleteDatasetPreset?.(...args),
    importDatasetPreset: (...args) => legacyRoot.importDatasetPreset?.(...args),
    handleDatasetPresetImport: (...args) => legacyRoot.handleDatasetPresetImport?.(...args),
    exportDatasetPreset: (...args) => legacyRoot.exportDatasetPreset?.(...args),
    datasetPresetPathFromName: (...args) => legacyRoot.datasetPresetPathFromName?.(...args),
    showDatasetPresetNameDialog: (...args) => legacyRoot.showDatasetPresetNameDialog?.(...args),
    setDatasetPresetStatus: (...args) => legacyRoot.setDatasetPresetStatus?.(...args),
    createDatasetPresetGroup: (...args) => legacyRoot.createDatasetPresetGroup?.(...args),
    renameDatasetPresetGroup: (...args) => legacyRoot.renameDatasetPresetGroup?.(...args),
    deleteDatasetPresetGroup: (...args) => legacyRoot.deleteDatasetPresetGroup?.(...args),
    placeDatasetPresetGroup: (...args) => legacyRoot.placeDatasetPresetGroup?.(...args),
    placeDatasetPresetFile: (...args) => legacyRoot.placeDatasetPresetFile?.(...args),
};

export function configureDatasetPresetActionsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in datasetPresetActionsBridge) {
            datasetPresetActionsBridge[key] = handler;
        }
    }
}

export const applySelectedDatasetPresetToCurrentConfig = (...args) => datasetPresetActionsBridge.applySelectedDatasetPresetToCurrentConfig(...args);
export const saveDatasetPresetEditor = (...args) => datasetPresetActionsBridge.saveDatasetPresetEditor(...args);
export const createNewDatasetPreset = (...args) => datasetPresetActionsBridge.createNewDatasetPreset(...args);
export const copyDatasetPreset = (...args) => datasetPresetActionsBridge.copyDatasetPreset(...args);
export const renameDatasetPreset = (...args) => datasetPresetActionsBridge.renameDatasetPreset(...args);
export const copyDatasetPresetToName = (...args) => datasetPresetActionsBridge.copyDatasetPresetToName(...args);
export const deleteDatasetPreset = (...args) => datasetPresetActionsBridge.deleteDatasetPreset(...args);
export const importDatasetPreset = (...args) => datasetPresetActionsBridge.importDatasetPreset(...args);
export const handleDatasetPresetImport = (...args) => datasetPresetActionsBridge.handleDatasetPresetImport(...args);
export const exportDatasetPreset = (...args) => datasetPresetActionsBridge.exportDatasetPreset(...args);
export const datasetPresetPathFromName = (...args) => datasetPresetActionsBridge.datasetPresetPathFromName(...args);
export const showDatasetPresetNameDialog = (...args) => datasetPresetActionsBridge.showDatasetPresetNameDialog(...args);
export const setDatasetPresetStatus = (...args) => datasetPresetActionsBridge.setDatasetPresetStatus(...args);
export const createDatasetPresetGroup = (...args) => datasetPresetActionsBridge.createDatasetPresetGroup(...args);
export const renameDatasetPresetGroup = (...args) => datasetPresetActionsBridge.renameDatasetPresetGroup(...args);
export const deleteDatasetPresetGroup = (...args) => datasetPresetActionsBridge.deleteDatasetPresetGroup(...args);
export const placeDatasetPresetGroup = (...args) => datasetPresetActionsBridge.placeDatasetPresetGroup(...args);
export const placeDatasetPresetFile = (...args) => datasetPresetActionsBridge.placeDatasetPresetFile(...args);
