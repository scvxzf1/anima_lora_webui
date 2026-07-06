/**
 * Mechanical split from the former monolithic app closure.
 * Keep image-test feature wiring out of 01-scope-state so the scope chunk stays small.
 */
export function createImageTestFeatureBridge(runtime) {
    const ctx = runtime.ctx;
    let imageTestFeature = null;

    function ensureImageTestFeature() {
        if (imageTestFeature) return imageTestFeature;
        imageTestFeature = globalThis.createImageTestFeature(ctx, {
            getCurrentConfig: () => globalThis.currentConfig,
            getCurrentTomlFile: () => globalThis.currentTomlFile,
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

    return {
        get imageTestFeature() {
            return imageTestFeature;
        },
        set imageTestFeature(value) {
            imageTestFeature = value;
        },
        ensureImageTestFeature,
    };
}
