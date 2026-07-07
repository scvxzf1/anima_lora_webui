const legacyRoot = globalThis;

const previewViewBridge = {
    loadPreviewSettings: (...args) => legacyRoot.loadPreviewSettings?.(...args),
    savePreviewSettings: (...args) => legacyRoot.savePreviewSettings?.(...args),
    resetPreviewSettings: (...args) => legacyRoot.resetPreviewSettings?.(...args),
    loadPreviewImages: (...args) => legacyRoot.loadPreviewImages?.(...args),
    loadPreviewWeights: (...args) => legacyRoot.loadPreviewWeights?.(...args),
    setPreviewSource: (...args) => legacyRoot.setPreviewSource?.(...args),
    openTrainingPreview: (...args) => legacyRoot.openTrainingPreview?.(...args),
    openCurrentTrainingPreview: (...args) => legacyRoot.openCurrentTrainingPreview?.(...args),
    openLiveSamplingPreview: (...args) => legacyRoot.openLiveSamplingPreview?.(...args),
    openHistoryConfigGroupPreview: (...args) => legacyRoot.openHistoryConfigGroupPreview?.(...args),
    normalizePreviewGroup: (...args) => legacyRoot.normalizePreviewGroup?.(...args),
    renderPreviewTaskSelect: (...args) => legacyRoot.renderPreviewTaskSelect?.(...args),
    changePreviewTask: (...args) => legacyRoot.changePreviewTask?.(...args),
    togglePreviewWeightSort: (...args) => legacyRoot.togglePreviewWeightSort?.(...args),
    openPreviewDialog: (...args) => legacyRoot.openPreviewDialog?.(...args),
    closePreviewImageDialog: (...args) => legacyRoot.closePreviewImageDialog?.(...args),
    openPreviewPanel: (...args) => legacyRoot.openPreviewPanel?.(...args),
    closePreviewPanel: (...args) => legacyRoot.closePreviewPanel?.(...args),
    restorePreviewWorkspaceAfterPanelClose: (...args) => legacyRoot.restorePreviewWorkspaceAfterPanelClose?.(...args),
    setPreviewStatus: (...args) => legacyRoot.setPreviewStatus?.(...args),
    createPreviewDetailRow: (...args) => legacyRoot.createPreviewDetailRow?.(...args),
    createPreviewDetailBlock: (...args) => legacyRoot.createPreviewDetailBlock?.(...args),
    renderDatasetImageDialogDetails: (...args) => legacyRoot.renderDatasetImageDialogDetails?.(...args),
    formatTotalPixels: (...args) => legacyRoot.formatTotalPixels?.(...args),
    copyText: (...args) => legacyRoot.copyText?.(...args),
    formatBytes: (...args) => legacyRoot.formatBytes?.(...args),
};

export function configurePreviewViewBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in previewViewBridge) {
            previewViewBridge[key] = handler;
        }
    }
}

export function loadPreviewSettings(...args) { return previewViewBridge.loadPreviewSettings(...args); }
export function savePreviewSettings(...args) { return previewViewBridge.savePreviewSettings(...args); }
export function resetPreviewSettings(...args) { return previewViewBridge.resetPreviewSettings(...args); }
export function loadPreviewImages(...args) { return previewViewBridge.loadPreviewImages(...args); }
export function loadPreviewWeights(...args) { return previewViewBridge.loadPreviewWeights(...args); }
export function setPreviewSource(...args) { return previewViewBridge.setPreviewSource(...args); }
export function openTrainingPreview(...args) { return previewViewBridge.openTrainingPreview(...args); }
export function openCurrentTrainingPreview(...args) { return previewViewBridge.openCurrentTrainingPreview(...args); }
export function openLiveSamplingPreview(...args) { return previewViewBridge.openLiveSamplingPreview(...args); }
export function openHistoryConfigGroupPreview(...args) { return previewViewBridge.openHistoryConfigGroupPreview(...args); }
export function normalizePreviewGroup(...args) { return previewViewBridge.normalizePreviewGroup(...args); }
export function renderPreviewTaskSelect(...args) { return previewViewBridge.renderPreviewTaskSelect(...args); }
export function changePreviewTask(...args) { return previewViewBridge.changePreviewTask(...args); }
export function togglePreviewWeightSort(...args) { return previewViewBridge.togglePreviewWeightSort(...args); }
export function openPreviewDialog(...args) { return previewViewBridge.openPreviewDialog(...args); }
export function closePreviewImageDialog(...args) { return previewViewBridge.closePreviewImageDialog(...args); }
export function openPreviewPanel(...args) { return previewViewBridge.openPreviewPanel(...args); }
export function closePreviewPanel(...args) { return previewViewBridge.closePreviewPanel(...args); }
export function restorePreviewWorkspaceAfterPanelClose(...args) { return previewViewBridge.restorePreviewWorkspaceAfterPanelClose(...args); }
export function setPreviewStatus(...args) { return previewViewBridge.setPreviewStatus(...args); }
export function createPreviewDetailRow(...args) { return previewViewBridge.createPreviewDetailRow(...args); }
export function createPreviewDetailBlock(...args) { return previewViewBridge.createPreviewDetailBlock(...args); }
export function renderDatasetImageDialogDetails(...args) { return previewViewBridge.renderDatasetImageDialogDetails(...args); }
export function formatTotalPixels(...args) { return previewViewBridge.formatTotalPixels(...args); }
export function copyText(...args) { return previewViewBridge.copyText(...args); }
export function formatBytes(...args) { return previewViewBridge.formatBytes(...args); }
