const datasetPresetActionsHandlers = Object.create(null);

function requireDatasetPresetActionsHandler(name) {
    const handler = datasetPresetActionsHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[dataset-preset-actions] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureDatasetPresetActionsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            datasetPresetActionsHandlers[key] = handler;
        }
    }
}

export function applySelectedDatasetPresetToCurrentConfig(...args) { return requireDatasetPresetActionsHandler('applySelectedDatasetPresetToCurrentConfig')(...args); }
export function saveDatasetPresetEditor(...args) { return requireDatasetPresetActionsHandler('saveDatasetPresetEditor')(...args); }
export function createNewDatasetPreset(...args) { return requireDatasetPresetActionsHandler('createNewDatasetPreset')(...args); }
export function copyDatasetPreset(...args) { return requireDatasetPresetActionsHandler('copyDatasetPreset')(...args); }
export function renameDatasetPreset(...args) { return requireDatasetPresetActionsHandler('renameDatasetPreset')(...args); }
export function copyDatasetPresetToName(...args) { return requireDatasetPresetActionsHandler('copyDatasetPresetToName')(...args); }
export function deleteDatasetPreset(...args) { return requireDatasetPresetActionsHandler('deleteDatasetPreset')(...args); }
export function importDatasetPreset(...args) { return requireDatasetPresetActionsHandler('importDatasetPreset')(...args); }
export function handleDatasetPresetImport(...args) { return requireDatasetPresetActionsHandler('handleDatasetPresetImport')(...args); }
export function exportDatasetPreset(...args) { return requireDatasetPresetActionsHandler('exportDatasetPreset')(...args); }
export function datasetPresetPathFromName(...args) { return requireDatasetPresetActionsHandler('datasetPresetPathFromName')(...args); }
export function showDatasetPresetNameDialog(...args) { return requireDatasetPresetActionsHandler('showDatasetPresetNameDialog')(...args); }
export function setDatasetPresetStatus(...args) { return requireDatasetPresetActionsHandler('setDatasetPresetStatus')(...args); }
export function createDatasetPresetGroup(...args) { return requireDatasetPresetActionsHandler('createDatasetPresetGroup')(...args); }
export function renameDatasetPresetGroup(...args) { return requireDatasetPresetActionsHandler('renameDatasetPresetGroup')(...args); }
export function deleteDatasetPresetGroup(...args) { return requireDatasetPresetActionsHandler('deleteDatasetPresetGroup')(...args); }
export function placeDatasetPresetGroup(...args) { return requireDatasetPresetActionsHandler('placeDatasetPresetGroup')(...args); }
export function placeDatasetPresetFile(...args) { return requireDatasetPresetActionsHandler('placeDatasetPresetFile')(...args); }
