/**
 * Mechanical split from the former monolithic app closure.
 * Keep image-test feature wiring out of 01-scope-state so the scope chunk stays small.
 */
const ctx = globalThis.ctx;

globalThis.imageTestFeature = null;
globalThis.ensureImageTestFeature = function ensureImageTestFeature() {
    if (imageTestFeature) return imageTestFeature;
    imageTestFeature = createImageTestFeature(ctx, {
        getCurrentConfig: () => currentConfig,
        getCurrentTomlFile: () => currentTomlFile,
        getSelectionMeta: () => ({
            method: globalThis.val('method-select'),
            variant: globalThis.val('variant-select'),
            preset: globalThis.val('preset-select') || 'default',
        }),
        precisionPreferenceFromConfig: (cfg) => globalThis.precisionPreferenceFromConfig(cfg),
        openPreviewDialog: (image) => globalThis.ensurePreviewFeature().openPreviewDialog(image),
    });
    return imageTestFeature;
}
