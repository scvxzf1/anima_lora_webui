const configFieldUiHandlers = Object.create(null);

function requireConfigFieldUiHandler(name) {
    const handler = configFieldUiHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[config-field-ui] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureConfigFieldUiBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            configFieldUiHandlers[key] = handler;
        }
    }
}

export function appendSamplePromptRow(...args) { return requireConfigFieldUiHandler('appendSamplePromptRow')(...args); }
export function createSamplePromptTextField(...args) { return requireConfigFieldUiHandler('createSamplePromptTextField')(...args); }
export function createSamplePromptInputField(...args) { return requireConfigFieldUiHandler('createSamplePromptInputField')(...args); }
export function clearSamplePromptRow(...args) { return requireConfigFieldUiHandler('clearSamplePromptRow')(...args); }
export function updateSamplePromptRemoveButtons(...args) { return requireConfigFieldUiHandler('updateSamplePromptRemoveButtons')(...args); }
export function isNumericField(...args) { return requireConfigFieldUiHandler('isNumericField')(...args); }
export function isIntegerNumericField(...args) { return requireConfigFieldUiHandler('isIntegerNumericField')(...args); }
export function allowsNegativeNumberField(...args) { return requireConfigFieldUiHandler('allowsNegativeNumberField')(...args); }
export function createSelectInput(...args) { return requireConfigFieldUiHandler('createSelectInput')(...args); }
export function selectUsesStrictOptions(...args) { return requireConfigFieldUiHandler('selectUsesStrictOptions')(...args); }
export function strictSelectCurrentValueLabel(...args) { return requireConfigFieldUiHandler('strictSelectCurrentValueLabel')(...args); }
export function fieldValueType(...args) { return requireConfigFieldUiHandler('fieldValueType')(...args); }
export function fieldValueTypeForKey(...args) { return requireConfigFieldUiHandler('fieldValueTypeForKey')(...args); }
export function optionValue(...args) { return requireConfigFieldUiHandler('optionValue')(...args); }
export function optionLabel(...args) { return requireConfigFieldUiHandler('optionLabel')(...args); }
export function generateDefaultHelp(...args) { return requireConfigFieldUiHandler('generateDefaultHelp')(...args); }
export function sectionTitleForField(...args) { return requireConfigFieldUiHandler('sectionTitleForField')(...args); }
export function createHelpContent(...args) { return requireConfigFieldUiHandler('createHelpContent')(...args); }
export function addHelpSection(...args) { return requireConfigFieldUiHandler('addHelpSection')(...args); }
export function getHelpSpec(...args) { return requireConfigFieldUiHandler('getHelpSpec')(...args); }
