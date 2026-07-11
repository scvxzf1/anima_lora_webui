const previewViewHandlers = Object.create(null);

function requirePreviewViewHandler(name) {
    const handler = previewViewHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[preview-view] bridge not configured: ${name}`);
    }
    return handler;
}

export function configurePreviewViewBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            previewViewHandlers[key] = handler;
        }
    }
}

export function loadPreviewSettings(...args) { return requirePreviewViewHandler('loadPreviewSettings')(...args); }
export function savePreviewSettings(...args) { return requirePreviewViewHandler('savePreviewSettings')(...args); }
export function resetPreviewSettings(...args) { return requirePreviewViewHandler('resetPreviewSettings')(...args); }
export function loadPreviewImages(...args) { return requirePreviewViewHandler('loadPreviewImages')(...args); }
export function loadPreviewWeights(...args) { return requirePreviewViewHandler('loadPreviewWeights')(...args); }
export function setPreviewSource(...args) { return requirePreviewViewHandler('setPreviewSource')(...args); }
export function openTrainingPreview(...args) { return requirePreviewViewHandler('openTrainingPreview')(...args); }
export function openCurrentTrainingPreview(...args) { return requirePreviewViewHandler('openCurrentTrainingPreview')(...args); }
export function openLiveSamplingPreview(...args) { return requirePreviewViewHandler('openLiveSamplingPreview')(...args); }
export function openHistoryConfigGroupPreview(...args) { return requirePreviewViewHandler('openHistoryConfigGroupPreview')(...args); }
export function normalizePreviewGroup(...args) { return requirePreviewViewHandler('normalizePreviewGroup')(...args); }
export function renderPreviewTaskSelect(...args) { return requirePreviewViewHandler('renderPreviewTaskSelect')(...args); }
export function changePreviewTask(...args) { return requirePreviewViewHandler('changePreviewTask')(...args); }
export function togglePreviewWeightSort(...args) { return requirePreviewViewHandler('togglePreviewWeightSort')(...args); }
export function openPreviewDialog(...args) { return requirePreviewViewHandler('openPreviewDialog')(...args); }
export function closePreviewImageDialog(...args) { return requirePreviewViewHandler('closePreviewImageDialog')(...args); }
export function openPreviewPanel(...args) { return requirePreviewViewHandler('openPreviewPanel')(...args); }
export function closePreviewPanel(...args) { return requirePreviewViewHandler('closePreviewPanel')(...args); }
export function restorePreviewWorkspaceAfterPanelClose(...args) { return requirePreviewViewHandler('restorePreviewWorkspaceAfterPanelClose')(...args); }
export function setPreviewStatus(...args) { return requirePreviewViewHandler('setPreviewStatus')(...args); }
export function createPreviewDetailRow(...args) { return requirePreviewViewHandler('createPreviewDetailRow')(...args); }
export function createPreviewDetailBlock(...args) { return requirePreviewViewHandler('createPreviewDetailBlock')(...args); }
export function renderDatasetImageDialogDetails(...args) { return requirePreviewViewHandler('renderDatasetImageDialogDetails')(...args); }
export function formatTotalPixels(...args) { return requirePreviewViewHandler('formatTotalPixels')(...args); }
export function copyText(...args) { return requirePreviewViewHandler('copyText')(...args); }
export function formatBytes(...args) { return requirePreviewViewHandler('formatBytes')(...args); }
