const historyListHandlers = Object.create(null);

function requireHistoryListHandler(name) {
    const handler = historyListHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[history-list] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureHistoryListBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            historyListHandlers[key] = handler;
        }
    }
}

export function loadTrainingHistoryList(...args) { return requireHistoryListHandler('loadTrainingHistoryList')(...args); }
export function loadHistoryCollectionSettings(...args) { return requireHistoryListHandler('loadHistoryCollectionSettings')(...args); }
export function saveHistoryCollectionSettings(...args) { return requireHistoryListHandler('saveHistoryCollectionSettings')(...args); }
export function normalizeHistoryCollectionSettings(...args) { return requireHistoryListHandler('normalizeHistoryCollectionSettings')(...args); }
export function uniqueStringList(...args) { return requireHistoryListHandler('uniqueStringList')(...args); }
export function normalizeHistoryConfigGroupOrder(...args) { return requireHistoryListHandler('normalizeHistoryConfigGroupOrder')(...args); }
export function renderTrainingHistoryList(...args) { return requireHistoryListHandler('renderTrainingHistoryList')(...args); }
export function syncRecentHistorySidebarSelection(...args) { return requireHistoryListHandler('syncRecentHistorySidebarSelection')(...args); }
export function recentTrainingSidebarTasks(...args) { return requireHistoryListHandler('recentTrainingSidebarTasks')(...args); }
export function renderHistoryManager(...args) { return requireHistoryListHandler('renderHistoryManager')(...args); }
export function renderHistoryManagerItems(...args) { return requireHistoryListHandler('renderHistoryManagerItems')(...args); }
