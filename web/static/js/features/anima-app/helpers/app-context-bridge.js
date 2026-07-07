let appContext = null;

export function configureAppContextBridge(ctx) {
    appContext = ctx || null;
}

export function getAppContext() {
    if (!appContext) {
        throw new Error('app context bridge is not configured');
    }
    return appContext;
}
