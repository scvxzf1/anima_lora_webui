const legacyRoot = globalThis;

const tomlManagerBridge = {
    updateConfigPageSummary: (...args) => legacyRoot.updateConfigPageSummary?.(...args),
    setTomlManagerMode: (...args) => legacyRoot.setTomlManagerMode?.(...args),
    switchTomlManagerMode: (...args) => legacyRoot.switchTomlManagerMode?.(...args),
    loadTomlFileList: (...args) => legacyRoot.loadTomlFileList?.(...args),
    loadDefaultTomlFile: (...args) => legacyRoot.loadDefaultTomlFile?.(...args),
    loadOutputRuns: (...args) => legacyRoot.loadOutputRuns?.(...args),
};

export function configureTomlManagerBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in tomlManagerBridge) {
            tomlManagerBridge[key] = handler;
        }
    }
}

export const updateConfigPageSummary = (...args) => tomlManagerBridge.updateConfigPageSummary(...args);
export const setTomlManagerMode = (...args) => tomlManagerBridge.setTomlManagerMode(...args);
export const switchTomlManagerMode = (...args) => tomlManagerBridge.switchTomlManagerMode(...args);
export const loadTomlFileList = (...args) => tomlManagerBridge.loadTomlFileList(...args);
export const loadDefaultTomlFile = (...args) => tomlManagerBridge.loadDefaultTomlFile(...args);
export const loadOutputRuns = (...args) => tomlManagerBridge.loadOutputRuns(...args);
