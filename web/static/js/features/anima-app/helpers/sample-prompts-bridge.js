const samplePromptsHandlers = Object.create(null);

function requireSamplePromptsHandler(name) {
    const handler = samplePromptsHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[sample-prompts] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureSamplePromptsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            samplePromptsHandlers[key] = handler;
        }
    }
}

export function currentSamplePromptText(...args) { return requireSamplePromptsHandler('currentSamplePromptText')(...args); }
export function normalizeSamplePromptsPath(...args) { return requireSamplePromptsHandler('normalizeSamplePromptsPath')(...args); }
export function isEditableSamplePromptsTextFilePath(...args) { return requireSamplePromptsHandler('isEditableSamplePromptsTextFilePath')(...args); }
export function isSamplePromptsFilePath(...args) { return requireSamplePromptsHandler('isSamplePromptsFilePath')(...args); }
export function loadSamplePrompts(...args) { return requireSamplePromptsHandler('loadSamplePrompts')(...args); }
export function saveSamplePrompts(...args) { return requireSamplePromptsHandler('saveSamplePrompts')(...args); }
