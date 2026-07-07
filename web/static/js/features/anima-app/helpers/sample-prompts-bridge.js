const legacyRoot = globalThis;

const samplePromptsBridge = {
    currentSamplePromptText: (...args) => legacyRoot.currentSamplePromptText?.(...args),
    normalizeSamplePromptsPath: (...args) => legacyRoot.normalizeSamplePromptsPath?.(...args),
    isEditableSamplePromptsTextFilePath: (...args) => legacyRoot.isEditableSamplePromptsTextFilePath?.(...args),
    isSamplePromptsFilePath: (...args) => legacyRoot.isSamplePromptsFilePath?.(...args),
    loadSamplePrompts: (...args) => legacyRoot.loadSamplePrompts?.(...args),
    saveSamplePrompts: (...args) => legacyRoot.saveSamplePrompts?.(...args),
};

export function configureSamplePromptsBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in samplePromptsBridge) {
            samplePromptsBridge[key] = handler;
        }
    }
}

export function currentSamplePromptText(...args) { return samplePromptsBridge.currentSamplePromptText(...args); }
export function normalizeSamplePromptsPath(...args) { return samplePromptsBridge.normalizeSamplePromptsPath(...args); }
export function isEditableSamplePromptsTextFilePath(...args) { return samplePromptsBridge.isEditableSamplePromptsTextFilePath(...args); }
export function isSamplePromptsFilePath(...args) { return samplePromptsBridge.isSamplePromptsFilePath(...args); }
export function loadSamplePrompts(...args) { return samplePromptsBridge.loadSamplePrompts(...args); }
export function saveSamplePrompts(...args) { return samplePromptsBridge.saveSamplePrompts(...args); }
