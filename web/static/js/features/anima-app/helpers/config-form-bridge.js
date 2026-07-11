const configFormHandlers = Object.create(null);

function requireConfigFormHandler(name) {
    const handler = configFormHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[config-form] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureConfigFormBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            configFormHandlers[key] = handler;
        }
    }
}

export function saveDatasetEditor(...args) { return requireConfigFormHandler('saveDatasetEditor')(...args); }
export function collectChangedFormValues(...args) { return requireConfigFormHandler('collectChangedFormValues')(...args); }
export function networkArgInputChanged(...args) { return requireConfigFormHandler('networkArgInputChanged')(...args); }
export function networkArgFieldValueFromConfig(...args) { return requireConfigFormHandler('networkArgFieldValueFromConfig')(...args); }
export function collectNetworkArgsFromForm(...args) { return requireConfigFormHandler('collectNetworkArgsFromForm')(...args); }
export function prepareFormPatchValues(...args) { return requireConfigFormHandler('prepareFormPatchValues')(...args); }
export function shouldSkipUiDefaultField(...args) { return requireConfigFormHandler('shouldSkipUiDefaultField')(...args); }
export function syncConfigDraftFromForm(...args) { return requireConfigFormHandler('syncConfigDraftFromForm')(...args); }
export function updateConfigDraftFromInput(...args) { return requireConfigFormHandler('updateConfigDraftFromInput')(...args); }
export function originalConfigFieldValue(...args) { return requireConfigFormHandler('originalConfigFieldValue')(...args); }
export function displayConfigFieldValue(...args) { return requireConfigFormHandler('displayConfigFieldValue')(...args); }
export function configDraftValueChanged(...args) { return requireConfigFormHandler('configDraftValueChanged')(...args); }
export function isActiveNetworkArgFieldKey(...args) { return requireConfigFormHandler('isActiveNetworkArgFieldKey')(...args); }
export function readFieldInputValue(...args) { return requireConfigFormHandler('readFieldInputValue')(...args); }
export function readLoKrEnabled(...args) { return requireConfigFormHandler('readLoKrEnabled')(...args); }
export function updateLoKrFieldState(...args) { return requireConfigFormHandler('updateLoKrFieldState')(...args); }
export function readVeRAEnabled(...args) { return requireConfigFormHandler('readVeRAEnabled')(...args); }
export function readDoRAAvailable(...args) { return requireConfigFormHandler('readDoRAAvailable')(...args); }
export function setDoRADraftValue(...args) { return requireConfigFormHandler('setDoRADraftValue')(...args); }
export function updateDoRAFieldState(...args) { return requireConfigFormHandler('updateDoRAFieldState')(...args); }
export function updateVeRAFieldState(...args) { return requireConfigFormHandler('updateVeRAFieldState')(...args); }
export function currentLossWeightingScheme(...args) { return requireConfigFormHandler('currentLossWeightingScheme')(...args); }
export function lossWeightingFieldState(...args) { return requireConfigFormHandler('lossWeightingFieldState')(...args); }
export function lossWeightingDisabledHint(...args) { return requireConfigFormHandler('lossWeightingDisabledHint')(...args); }
export function applyLossWeightingFieldInputState(...args) { return requireConfigFormHandler('applyLossWeightingFieldInputState')(...args); }
export function updateLossWeightingFieldState(...args) { return requireConfigFormHandler('updateLossWeightingFieldState')(...args); }
