const preflightDialogHandlers = Object.create(null);

function requirePreflightDialogHandler(name) {
    const handler = preflightDialogHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[preflight-dialog] bridge not configured: ${name}`);
    }
    return handler;
}

export function configurePreflightDialogBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            preflightDialogHandlers[key] = handler;
        }
    }
}

export function showPreflightPendingDialog(...args) { return requirePreflightDialogHandler('showPreflightPendingDialog')(...args); }
export function renderPreflightPending(...args) { return requirePreflightDialogHandler('renderPreflightPending')(...args); }
export function showPreflightRequestError(...args) { return requirePreflightDialogHandler('showPreflightRequestError')(...args); }
export function isPreflightDialogOpen(...args) { return requirePreflightDialogHandler('isPreflightDialogOpen')(...args); }
export function waitForPreflightDialogClose(...args) { return requirePreflightDialogHandler('waitForPreflightDialogClose')(...args); }
export function renderPreflightResult(...args) { return requirePreflightDialogHandler('renderPreflightResult')(...args); }
export function preflightCanStartPreprocess(...args) { return requirePreflightDialogHandler('preflightCanStartPreprocess')(...args); }
export function startPreprocessFromPreflight(...args) { return requirePreflightDialogHandler('startPreprocessFromPreflight')(...args); }
export function currentTrainingConfigFile(...args) { return requirePreflightDialogHandler('currentTrainingConfigFile')(...args); }
export function preflightPlainText(...args) { return requirePreflightDialogHandler('preflightPlainText')(...args); }
