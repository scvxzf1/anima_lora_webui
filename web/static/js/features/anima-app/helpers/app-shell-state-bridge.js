let appShellState = null;

export function configureAppShellStateBridge(state) {
    appShellState = state || null;
}

export function getAppShellState() {
    if (!appShellState) {
        throw new Error('app shell state bridge is not configured');
    }
    return appShellState;
}
