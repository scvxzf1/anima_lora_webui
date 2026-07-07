const legacyRoot = globalThis;

const configFieldUiBridge = {
    appendSamplePromptRow: (...args) => legacyRoot.appendSamplePromptRow?.(...args),
    createSamplePromptTextField: (...args) => legacyRoot.createSamplePromptTextField?.(...args),
    createSamplePromptInputField: (...args) => legacyRoot.createSamplePromptInputField?.(...args),
    clearSamplePromptRow: (...args) => legacyRoot.clearSamplePromptRow?.(...args),
    updateSamplePromptRemoveButtons: (...args) => legacyRoot.updateSamplePromptRemoveButtons?.(...args),
    isNumericField: (...args) => legacyRoot.isNumericField?.(...args),
    isIntegerNumericField: (...args) => legacyRoot.isIntegerNumericField?.(...args),
    allowsNegativeNumberField: (...args) => legacyRoot.allowsNegativeNumberField?.(...args),
    createSelectInput: (...args) => legacyRoot.createSelectInput?.(...args),
    selectUsesStrictOptions: (...args) => legacyRoot.selectUsesStrictOptions?.(...args),
    strictSelectCurrentValueLabel: (...args) => legacyRoot.strictSelectCurrentValueLabel?.(...args),
    fieldValueType: (...args) => legacyRoot.fieldValueType?.(...args),
    fieldValueTypeForKey: (...args) => legacyRoot.fieldValueTypeForKey?.(...args),
    optionValue: (...args) => legacyRoot.optionValue?.(...args),
    optionLabel: (...args) => legacyRoot.optionLabel?.(...args),
    generateDefaultHelp: (...args) => legacyRoot.generateDefaultHelp?.(...args),
    sectionTitleForField: (...args) => legacyRoot.sectionTitleForField?.(...args),
    createHelpContent: (...args) => legacyRoot.createHelpContent?.(...args),
    addHelpSection: (...args) => legacyRoot.addHelpSection?.(...args),
    getHelpSpec: (...args) => legacyRoot.getHelpSpec?.(...args),
};

export function configureConfigFieldUiBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in configFieldUiBridge) {
            configFieldUiBridge[key] = handler;
        }
    }
}

export const appendSamplePromptRow = (...args) => configFieldUiBridge.appendSamplePromptRow(...args);
export const createSamplePromptTextField = (...args) => configFieldUiBridge.createSamplePromptTextField(...args);
export const createSamplePromptInputField = (...args) => configFieldUiBridge.createSamplePromptInputField(...args);
export const clearSamplePromptRow = (...args) => configFieldUiBridge.clearSamplePromptRow(...args);
export const updateSamplePromptRemoveButtons = (...args) => configFieldUiBridge.updateSamplePromptRemoveButtons(...args);
export const isNumericField = (...args) => configFieldUiBridge.isNumericField(...args);
export const isIntegerNumericField = (...args) => configFieldUiBridge.isIntegerNumericField(...args);
export const allowsNegativeNumberField = (...args) => configFieldUiBridge.allowsNegativeNumberField(...args);
export const createSelectInput = (...args) => configFieldUiBridge.createSelectInput(...args);
export const selectUsesStrictOptions = (...args) => configFieldUiBridge.selectUsesStrictOptions(...args);
export const strictSelectCurrentValueLabel = (...args) => configFieldUiBridge.strictSelectCurrentValueLabel(...args);
export const fieldValueType = (...args) => configFieldUiBridge.fieldValueType(...args);
export const fieldValueTypeForKey = (...args) => configFieldUiBridge.fieldValueTypeForKey(...args);
export const optionValue = (...args) => configFieldUiBridge.optionValue(...args);
export const optionLabel = (...args) => configFieldUiBridge.optionLabel(...args);
export const generateDefaultHelp = (...args) => configFieldUiBridge.generateDefaultHelp(...args);
export const sectionTitleForField = (...args) => configFieldUiBridge.sectionTitleForField(...args);
export const createHelpContent = (...args) => configFieldUiBridge.createHelpContent(...args);
export const addHelpSection = (...args) => configFieldUiBridge.addHelpSection(...args);
export const getHelpSpec = (...args) => configFieldUiBridge.getHelpSpec(...args);
