let historyState = null;

export function configureHistoryStateBridge(state) {
    historyState = state || null;
}

export function getHistoryState() {
    if (!historyState) {
        throw new Error('history state bridge is not configured');
    }
    return historyState;
}
