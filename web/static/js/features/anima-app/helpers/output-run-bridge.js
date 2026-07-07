const legacyRoot = globalThis;

const outputRunBridge = {
    loadOutputRunConfig: (...args) => legacyRoot.loadOutputRunConfig?.(...args),
    preferredOutputRunKind: (...args) => legacyRoot.preferredOutputRunKind?.(...args),
    renderOutputRunManager: (...args) => legacyRoot.renderOutputRunManager?.(...args),
    renderOutputRunList: (...args) => legacyRoot.renderOutputRunList?.(...args),
    renderOutputRunDetail: (...args) => legacyRoot.renderOutputRunDetail?.(...args),
    renderOutputRunSaveAsControls: (...args) => legacyRoot.renderOutputRunSaveAsControls?.(...args),
    filteredOutputRuns: (...args) => legacyRoot.filteredOutputRuns?.(...args),
    selectedOutputRun: (...args) => legacyRoot.selectedOutputRun?.(...args),
    updateOutputRunSelectionUI: (...args) => legacyRoot.updateOutputRunSelectionUI?.(...args),
    updateOutputRunActionState: (...args) => legacyRoot.updateOutputRunActionState?.(...args),
    setButtonDisabled: (...args) => legacyRoot.setButtonDisabled?.(...args),
    copyOutputRunConfigContent: (...args) => legacyRoot.copyOutputRunConfigContent?.(...args),
    exportOutputRunConfig: (...args) => legacyRoot.exportOutputRunConfig?.(...args),
    openOutputRunSaveAs: (...args) => legacyRoot.openOutputRunSaveAs?.(...args),
    closeOutputRunSaveAs: (...args) => legacyRoot.closeOutputRunSaveAs?.(...args),
    outputRunSaveAsDefaultName: (...args) => legacyRoot.outputRunSaveAsDefaultName?.(...args),
    confirmOutputRunSaveAs: (...args) => legacyRoot.confirmOutputRunSaveAs?.(...args),
    selectAndApplyTomlFile: (...args) => legacyRoot.selectAndApplyTomlFile?.(...args),
    loadTomlFile: (...args) => legacyRoot.loadTomlFile?.(...args),
    saveTomlFile: (...args) => legacyRoot.saveTomlFile?.(...args),
    saveRawTomlContent: (...args) => legacyRoot.saveRawTomlContent?.(...args),
    saveFormPatchToToml: (...args) => legacyRoot.saveFormPatchToToml?.(...args),
};

export function configureOutputRunBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in outputRunBridge) {
            outputRunBridge[key] = handler;
        }
    }
}

export const loadOutputRunConfig = (...args) => outputRunBridge.loadOutputRunConfig(...args);
export const preferredOutputRunKind = (...args) => outputRunBridge.preferredOutputRunKind(...args);
export const renderOutputRunManager = (...args) => outputRunBridge.renderOutputRunManager(...args);
export const renderOutputRunList = (...args) => outputRunBridge.renderOutputRunList(...args);
export const renderOutputRunDetail = (...args) => outputRunBridge.renderOutputRunDetail(...args);
export const renderOutputRunSaveAsControls = (...args) => outputRunBridge.renderOutputRunSaveAsControls(...args);
export const filteredOutputRuns = (...args) => outputRunBridge.filteredOutputRuns(...args);
export const selectedOutputRun = (...args) => outputRunBridge.selectedOutputRun(...args);
export const updateOutputRunSelectionUI = (...args) => outputRunBridge.updateOutputRunSelectionUI(...args);
export const updateOutputRunActionState = (...args) => outputRunBridge.updateOutputRunActionState(...args);
export const setButtonDisabled = (...args) => outputRunBridge.setButtonDisabled(...args);
export const copyOutputRunConfigContent = (...args) => outputRunBridge.copyOutputRunConfigContent(...args);
export const exportOutputRunConfig = (...args) => outputRunBridge.exportOutputRunConfig(...args);
export const openOutputRunSaveAs = (...args) => outputRunBridge.openOutputRunSaveAs(...args);
export const closeOutputRunSaveAs = (...args) => outputRunBridge.closeOutputRunSaveAs(...args);
export const outputRunSaveAsDefaultName = (...args) => outputRunBridge.outputRunSaveAsDefaultName(...args);
export const confirmOutputRunSaveAs = (...args) => outputRunBridge.confirmOutputRunSaveAs(...args);
export const selectAndApplyTomlFile = (...args) => outputRunBridge.selectAndApplyTomlFile(...args);
export const loadTomlFile = (...args) => outputRunBridge.loadTomlFile(...args);
export const saveTomlFile = (...args) => outputRunBridge.saveTomlFile(...args);
export const saveRawTomlContent = (...args) => outputRunBridge.saveRawTomlContent(...args);
export const saveFormPatchToToml = (...args) => outputRunBridge.saveFormPatchToToml(...args);
