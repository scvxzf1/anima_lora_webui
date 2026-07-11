const globalSettingsHandlers = Object.create(null);

function requireGlobalSettingsHandler(name) {
    const handler = globalSettingsHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[global-settings] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureGlobalSettingsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            globalSettingsHandlers[key] = handler;
        }
    }
}

export function resolveGlobalUIScaleDefaultValue(...args) { return requireGlobalSettingsHandler('resolveGlobalUIScaleDefaultValue')(...args); }
export function syncGlobalUIScaleOverrideField(...args) { return requireGlobalSettingsHandler('syncGlobalUIScaleOverrideField')(...args); }
export function syncAllGlobalUIScaleOverrideFields(...args) { return requireGlobalSettingsHandler('syncAllGlobalUIScaleOverrideFields')(...args); }
export function applyGlobalUIScaleOverrideInputs(...args) { return requireGlobalSettingsHandler('applyGlobalUIScaleOverrideInputs')(...args); }
export function collectGlobalUIScaleOverridePayload(...args) { return requireGlobalSettingsHandler('collectGlobalUIScaleOverridePayload')(...args); }
export function loadGlobalSettings(...args) { return requireGlobalSettingsHandler('loadGlobalSettings')(...args); }
export function saveGlobalSettings(...args) { return requireGlobalSettingsHandler('saveGlobalSettings')(...args); }
export function resetGlobalSettings(...args) { return requireGlobalSettingsHandler('resetGlobalSettings')(...args); }
export function setGlobalSettingsStatus(...args) { return requireGlobalSettingsHandler('setGlobalSettingsStatus')(...args); }
export function applyGlobalSettingsToInputs(...args) { return requireGlobalSettingsHandler('applyGlobalSettingsToInputs')(...args); }
export function collectGlobalSettingsPayload(...args) { return requireGlobalSettingsHandler('collectGlobalSettingsPayload')(...args); }
export function getGlobalModelPathOverrides(...args) { return requireGlobalSettingsHandler('getGlobalModelPathOverrides')(...args); }
export function toggleGlobalSettingHelp(...args) { return requireGlobalSettingsHandler('toggleGlobalSettingHelp')(...args); }
