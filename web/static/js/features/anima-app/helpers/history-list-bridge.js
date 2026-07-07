const legacyRoot = globalThis;

const historyListBridge = {
    loadTrainingHistoryList: (...args) => legacyRoot.loadTrainingHistoryList?.(...args),
    loadHistoryCollectionSettings: (...args) => legacyRoot.loadHistoryCollectionSettings?.(...args),
    saveHistoryCollectionSettings: (...args) => legacyRoot.saveHistoryCollectionSettings?.(...args),
    normalizeHistoryCollectionSettings: (...args) => legacyRoot.normalizeHistoryCollectionSettings?.(...args),
    uniqueStringList: (...args) => legacyRoot.uniqueStringList?.(...args),
    normalizeHistoryConfigGroupOrder: (...args) => legacyRoot.normalizeHistoryConfigGroupOrder?.(...args),
    renderTrainingHistoryList: (...args) => legacyRoot.renderTrainingHistoryList?.(...args),
    recentTrainingSidebarTasks: (...args) => legacyRoot.recentTrainingSidebarTasks?.(...args),
    renderHistoryManager: (...args) => legacyRoot.renderHistoryManager?.(...args),
    renderHistoryManagerItems: (...args) => legacyRoot.renderHistoryManagerItems?.(...args),
};

export function configureHistoryListBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in historyListBridge) {
            historyListBridge[key] = handler;
        }
    }
}

export function loadTrainingHistoryList(...args) { return historyListBridge.loadTrainingHistoryList(...args); }
export function loadHistoryCollectionSettings(...args) { return historyListBridge.loadHistoryCollectionSettings(...args); }
export function saveHistoryCollectionSettings(...args) { return historyListBridge.saveHistoryCollectionSettings(...args); }
export function normalizeHistoryCollectionSettings(...args) { return historyListBridge.normalizeHistoryCollectionSettings(...args); }
export function uniqueStringList(...args) { return historyListBridge.uniqueStringList(...args); }
export function normalizeHistoryConfigGroupOrder(...args) { return historyListBridge.normalizeHistoryConfigGroupOrder(...args); }
export function renderTrainingHistoryList(...args) { return historyListBridge.renderTrainingHistoryList(...args); }
export function recentTrainingSidebarTasks(...args) { return historyListBridge.recentTrainingSidebarTasks(...args); }
export function renderHistoryManager(...args) { return historyListBridge.renderHistoryManager(...args); }
export function renderHistoryManagerItems(...args) { return historyListBridge.renderHistoryManagerItems(...args); }
