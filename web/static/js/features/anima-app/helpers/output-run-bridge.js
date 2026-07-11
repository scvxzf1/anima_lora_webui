const outputRunHandlers = Object.create(null);

function requireOutputRunHandler(name) {
    const handler = outputRunHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[output-run] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureOutputRunBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            outputRunHandlers[key] = handler;
        }
    }
}

export function loadOutputRunConfig(...args) { return requireOutputRunHandler('loadOutputRunConfig')(...args); }
export function preferredOutputRunKind(...args) { return requireOutputRunHandler('preferredOutputRunKind')(...args); }
export function renderOutputRunManager(...args) { return requireOutputRunHandler('renderOutputRunManager')(...args); }
export function renderOutputRunList(...args) { return requireOutputRunHandler('renderOutputRunList')(...args); }
export function renderOutputRunDetail(...args) { return requireOutputRunHandler('renderOutputRunDetail')(...args); }
export function renderOutputRunSaveAsControls(...args) { return requireOutputRunHandler('renderOutputRunSaveAsControls')(...args); }
export function filteredOutputRuns(...args) { return requireOutputRunHandler('filteredOutputRuns')(...args); }
export function selectedOutputRun(...args) { return requireOutputRunHandler('selectedOutputRun')(...args); }
export function updateOutputRunSelectionUI(...args) { return requireOutputRunHandler('updateOutputRunSelectionUI')(...args); }
export function updateOutputRunActionState(...args) { return requireOutputRunHandler('updateOutputRunActionState')(...args); }
export function setButtonDisabled(...args) { return requireOutputRunHandler('setButtonDisabled')(...args); }
export function copyOutputRunConfigContent(...args) { return requireOutputRunHandler('copyOutputRunConfigContent')(...args); }
export function exportOutputRunConfig(...args) { return requireOutputRunHandler('exportOutputRunConfig')(...args); }
export function openOutputRunSaveAs(...args) { return requireOutputRunHandler('openOutputRunSaveAs')(...args); }
export function closeOutputRunSaveAs(...args) { return requireOutputRunHandler('closeOutputRunSaveAs')(...args); }
export function outputRunSaveAsDefaultName(...args) { return requireOutputRunHandler('outputRunSaveAsDefaultName')(...args); }
export function confirmOutputRunSaveAs(...args) { return requireOutputRunHandler('confirmOutputRunSaveAs')(...args); }
export function selectAndApplyTomlFile(...args) { return requireOutputRunHandler('selectAndApplyTomlFile')(...args); }
export function loadTomlFile(...args) { return requireOutputRunHandler('loadTomlFile')(...args); }
export function saveTomlFile(...args) { return requireOutputRunHandler('saveTomlFile')(...args); }
export function saveRawTomlContent(...args) { return requireOutputRunHandler('saveRawTomlContent')(...args); }
export function saveFormPatchToToml(...args) { return requireOutputRunHandler('saveFormPatchToToml')(...args); }
