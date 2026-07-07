const legacyRoot = globalThis;

const configFormBridge = {
    saveDatasetEditor: (...args) => legacyRoot.saveDatasetEditor?.(...args),
    collectChangedFormValues: (...args) => legacyRoot.collectChangedFormValues?.(...args),
    networkArgInputChanged: (...args) => legacyRoot.networkArgInputChanged?.(...args),
    networkArgFieldValueFromConfig: (...args) => legacyRoot.networkArgFieldValueFromConfig?.(...args),
    collectNetworkArgsFromForm: (...args) => legacyRoot.collectNetworkArgsFromForm?.(...args),
    prepareFormPatchValues: (...args) => legacyRoot.prepareFormPatchValues?.(...args),
    shouldSkipUiDefaultField: (...args) => legacyRoot.shouldSkipUiDefaultField?.(...args),
    syncConfigDraftFromForm: (...args) => legacyRoot.syncConfigDraftFromForm?.(...args),
    updateConfigDraftFromInput: (...args) => legacyRoot.updateConfigDraftFromInput?.(...args),
    originalConfigFieldValue: (...args) => legacyRoot.originalConfigFieldValue?.(...args),
    displayConfigFieldValue: (...args) => legacyRoot.displayConfigFieldValue?.(...args),
    configDraftValueChanged: (...args) => legacyRoot.configDraftValueChanged?.(...args),
    isActiveNetworkArgFieldKey: (...args) => legacyRoot.isActiveNetworkArgFieldKey?.(...args),
    readFieldInputValue: (...args) => legacyRoot.readFieldInputValue?.(...args),
    readLoKrEnabled: (...args) => legacyRoot.readLoKrEnabled?.(...args),
    updateLoKrFieldState: (...args) => legacyRoot.updateLoKrFieldState?.(...args),
    readVeRAEnabled: (...args) => legacyRoot.readVeRAEnabled?.(...args),
    readDoRAAvailable: (...args) => legacyRoot.readDoRAAvailable?.(...args),
    setDoRADraftValue: (...args) => legacyRoot.setDoRADraftValue?.(...args),
    updateDoRAFieldState: (...args) => legacyRoot.updateDoRAFieldState?.(...args),
    updateVeRAFieldState: (...args) => legacyRoot.updateVeRAFieldState?.(...args),
    currentLossWeightingScheme: (...args) => legacyRoot.currentLossWeightingScheme?.(...args),
    lossWeightingFieldState: (...args) => legacyRoot.lossWeightingFieldState?.(...args),
    lossWeightingDisabledHint: (...args) => legacyRoot.lossWeightingDisabledHint?.(...args),
    applyLossWeightingFieldInputState: (...args) => legacyRoot.applyLossWeightingFieldInputState?.(...args),
    updateLossWeightingFieldState: (...args) => legacyRoot.updateLossWeightingFieldState?.(...args),
};

export function configureConfigFormBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in configFormBridge) {
            configFormBridge[key] = handler;
        }
    }
}

export const saveDatasetEditor = (...args) => configFormBridge.saveDatasetEditor(...args);
export const collectChangedFormValues = (...args) => configFormBridge.collectChangedFormValues(...args);
export const networkArgInputChanged = (...args) => configFormBridge.networkArgInputChanged(...args);
export const networkArgFieldValueFromConfig = (...args) => configFormBridge.networkArgFieldValueFromConfig(...args);
export const collectNetworkArgsFromForm = (...args) => configFormBridge.collectNetworkArgsFromForm(...args);
export const prepareFormPatchValues = (...args) => configFormBridge.prepareFormPatchValues(...args);
export const shouldSkipUiDefaultField = (...args) => configFormBridge.shouldSkipUiDefaultField(...args);
export const syncConfigDraftFromForm = (...args) => configFormBridge.syncConfigDraftFromForm(...args);
export const updateConfigDraftFromInput = (...args) => configFormBridge.updateConfigDraftFromInput(...args);
export const originalConfigFieldValue = (...args) => configFormBridge.originalConfigFieldValue(...args);
export const displayConfigFieldValue = (...args) => configFormBridge.displayConfigFieldValue(...args);
export const configDraftValueChanged = (...args) => configFormBridge.configDraftValueChanged(...args);
export const isActiveNetworkArgFieldKey = (...args) => configFormBridge.isActiveNetworkArgFieldKey(...args);
export const readFieldInputValue = (...args) => configFormBridge.readFieldInputValue(...args);
export const readLoKrEnabled = (...args) => configFormBridge.readLoKrEnabled(...args);
export const updateLoKrFieldState = (...args) => configFormBridge.updateLoKrFieldState(...args);
export const readVeRAEnabled = (...args) => configFormBridge.readVeRAEnabled(...args);
export const readDoRAAvailable = (...args) => configFormBridge.readDoRAAvailable(...args);
export const setDoRADraftValue = (...args) => configFormBridge.setDoRADraftValue(...args);
export const updateDoRAFieldState = (...args) => configFormBridge.updateDoRAFieldState(...args);
export const updateVeRAFieldState = (...args) => configFormBridge.updateVeRAFieldState(...args);
export const currentLossWeightingScheme = (...args) => configFormBridge.currentLossWeightingScheme(...args);
export const lossWeightingFieldState = (...args) => configFormBridge.lossWeightingFieldState(...args);
export const lossWeightingDisabledHint = (...args) => configFormBridge.lossWeightingDisabledHint(...args);
export const applyLossWeightingFieldInputState = (...args) => configFormBridge.applyLossWeightingFieldInputState(...args);
export const updateLossWeightingFieldState = (...args) => configFormBridge.updateLossWeightingFieldState(...args);
