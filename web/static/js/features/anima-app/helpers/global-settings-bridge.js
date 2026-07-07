const legacyRoot = globalThis;

const globalSettingsBridge = {
    resolveGlobalUIScaleDefaultValue: (...args) => legacyRoot.resolveGlobalUIScaleDefaultValue?.(...args),
    syncGlobalUIScaleOverrideField: (...args) => legacyRoot.syncGlobalUIScaleOverrideField?.(...args),
    syncAllGlobalUIScaleOverrideFields: (...args) => legacyRoot.syncAllGlobalUIScaleOverrideFields?.(...args),
    applyGlobalUIScaleOverrideInputs: (...args) => legacyRoot.applyGlobalUIScaleOverrideInputs?.(...args),
    collectGlobalUIScaleOverridePayload: (...args) => legacyRoot.collectGlobalUIScaleOverridePayload?.(...args),
    loadGlobalSettings: (...args) => legacyRoot.loadGlobalSettings?.(...args),
    saveGlobalSettings: (...args) => legacyRoot.saveGlobalSettings?.(...args),
    resetGlobalSettings: (...args) => legacyRoot.resetGlobalSettings?.(...args),
    setGlobalSettingsStatus: (...args) => legacyRoot.setGlobalSettingsStatus?.(...args),
    applyGlobalSettingsToInputs: (...args) => legacyRoot.applyGlobalSettingsToInputs?.(...args),
    collectGlobalSettingsPayload: (...args) => legacyRoot.collectGlobalSettingsPayload?.(...args),
    getGlobalModelPathOverrides: (...args) => legacyRoot.getGlobalModelPathOverrides?.(...args),
    toggleGlobalSettingHelp: (...args) => legacyRoot.toggleGlobalSettingHelp?.(...args),
};

export function configureGlobalSettingsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in globalSettingsBridge) {
            globalSettingsBridge[key] = handler;
        }
    }
}

export function resolveGlobalUIScaleDefaultValue(...args) { return globalSettingsBridge.resolveGlobalUIScaleDefaultValue(...args); }
export function syncGlobalUIScaleOverrideField(...args) { return globalSettingsBridge.syncGlobalUIScaleOverrideField(...args); }
export function syncAllGlobalUIScaleOverrideFields(...args) { return globalSettingsBridge.syncAllGlobalUIScaleOverrideFields(...args); }
export function applyGlobalUIScaleOverrideInputs(...args) { return globalSettingsBridge.applyGlobalUIScaleOverrideInputs(...args); }
export function collectGlobalUIScaleOverridePayload(...args) { return globalSettingsBridge.collectGlobalUIScaleOverridePayload(...args); }
export function loadGlobalSettings(...args) { return globalSettingsBridge.loadGlobalSettings(...args); }
export function saveGlobalSettings(...args) { return globalSettingsBridge.saveGlobalSettings(...args); }
export function resetGlobalSettings(...args) { return globalSettingsBridge.resetGlobalSettings(...args); }
export function setGlobalSettingsStatus(...args) { return globalSettingsBridge.setGlobalSettingsStatus(...args); }
export function applyGlobalSettingsToInputs(...args) { return globalSettingsBridge.applyGlobalSettingsToInputs(...args); }
export function collectGlobalSettingsPayload(...args) { return globalSettingsBridge.collectGlobalSettingsPayload(...args); }
export function getGlobalModelPathOverrides(...args) { return globalSettingsBridge.getGlobalModelPathOverrides(...args); }
export function toggleGlobalSettingHelp(...args) { return globalSettingsBridge.toggleGlobalSettingHelp(...args); }
