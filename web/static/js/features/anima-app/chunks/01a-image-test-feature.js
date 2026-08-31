/**
 * Mechanical split from the former monolithic app closure.
 * Keep image-test feature wiring out of 01-scope-state so the scope chunk stays small.
 */
import { ensurePreviewFeature } from '../helpers/feature-ensurers.js?v=module-bootstrap-20260831-release-v1';
import { precisionPreferenceFromConfig } from '../helpers/config-values.js?v=module-bootstrap-20260831-release-v1';
import { createImageTestFeature } from '../../image-test/index.js?v=module-bootstrap-20260831-release-v1';

export function createImageTestFeatureBridge(runtime) {
    const ctx = runtime.ctx;
    const configState = runtime.state.config;
    const tomlState = runtime.state.toml;
    const dom = runtime.dom;
    let imageTestFeature = null;

    function ensureImageTestFeature() {
        if (imageTestFeature) return imageTestFeature;
        imageTestFeature = createImageTestFeature(ctx, {
            getCurrentConfig: () => configState.currentConfig,
            getModelFamily: () => {
                const globalSettings = runtime.state.appShell.globalSettings;
                return configState.currentConfig?.model_family
                    || globalSettings?.model_family
                    || 'anima';
            },
            getCurrentTomlFile: () => tomlState.currentTomlFile,
            getSelectionMeta: () => ({
                method: dom.val('method-select'),
                variant: dom.val('variant-select'),
                preset: dom.val('preset-select') || 'default',
            }),
            precisionPreferenceFromConfig,
            openPreviewDialog: (...args) => ensurePreviewFeature().openPreviewDialog(...args),
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
