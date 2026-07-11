const tomlIoHandlers = Object.create(null);

function requireTomlIoHandler(name) {
    const handler = tomlIoHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[toml-io] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureTomlIoBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            tomlIoHandlers[key] = handler;
        }
    }
}

export function importTomlFile(...args) { return requireTomlIoHandler('importTomlFile')(...args); }
export function handleTomlImport(...args) { return requireTomlIoHandler('handleTomlImport')(...args); }
export function exportTomlFile(...args) { return requireTomlIoHandler('exportTomlFile')(...args); }
export function downloadTomlContent(...args) { return requireTomlIoHandler('downloadTomlContent')(...args); }
export function triggerDownload(...args) { return requireTomlIoHandler('triggerDownload')(...args); }
export function downloadBlob(...args) { return requireTomlIoHandler('downloadBlob')(...args); }
export function createTomlZipBlob(...args) { return requireTomlIoHandler('createTomlZipBlob')(...args); }
export function uniqueZipEntryName(...args) { return requireTomlIoHandler('uniqueZipEntryName')(...args); }
export function saveTomlAs(...args) { return requireTomlIoHandler('saveTomlAs')(...args); }
export function createBlankPresetFromLoraTemplate(...args) { return requireTomlIoHandler('createBlankPresetFromLoraTemplate')(...args); }
export function previewPatchedTomlContent(...args) { return requireTomlIoHandler('previewPatchedTomlContent')(...args); }
export function showTomlSaveAsDialog(...args) { return requireTomlIoHandler('showTomlSaveAsDialog')(...args); }
export function saveAsTargetGroups(...args) { return requireTomlIoHandler('saveAsTargetGroups')(...args); }
export function moveTomlFileToGroup(...args) { return requireTomlIoHandler('moveTomlFileToGroup')(...args); }
export function normalizeTomlSaveAsPath(...args) { return requireTomlIoHandler('normalizeTomlSaveAsPath')(...args); }
export function exportTomlFilename(...args) { return requireTomlIoHandler('exportTomlFilename')(...args); }
export function isFixedSystemTomlGroup(...args) { return requireTomlIoHandler('isFixedSystemTomlGroup')(...args); }
export function isDatasetConfigGroup(...args) { return requireTomlIoHandler('isDatasetConfigGroup')(...args); }
export function isTrainingTomlGroup(...args) { return requireTomlIoHandler('isTrainingTomlGroup')(...args); }
export function filterTrainingTomlGroups(...args) { return requireTomlIoHandler('filterTrainingTomlGroups')(...args); }
export function shouldShowTomlGroup(...args) { return requireTomlIoHandler('shouldShowTomlGroup')(...args); }
export function reorderTomlFileGroups(...args) { return requireTomlIoHandler('reorderTomlFileGroups')(...args); }
export function getSortableTomlGroups(...args) { return requireTomlIoHandler('getSortableTomlGroups')(...args); }
export function isTomlGroupDraggable(...args) { return requireTomlIoHandler('isTomlGroupDraggable')(...args); }
