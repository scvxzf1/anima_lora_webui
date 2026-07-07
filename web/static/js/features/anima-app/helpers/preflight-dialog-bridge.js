const legacyRoot = globalThis;

const preflightDialogBridge = {
    showPreflightPendingDialog: (...args) => legacyRoot.showPreflightPendingDialog?.(...args),
    renderPreflightPending: (...args) => legacyRoot.renderPreflightPending?.(...args),
    showPreflightRequestError: (...args) => legacyRoot.showPreflightRequestError?.(...args),
    isPreflightDialogOpen: (...args) => legacyRoot.isPreflightDialogOpen?.(...args),
    waitForPreflightDialogClose: (...args) => legacyRoot.waitForPreflightDialogClose?.(...args),
    renderPreflightResult: (...args) => legacyRoot.renderPreflightResult?.(...args),
    preflightCanStartPreprocess: (...args) => legacyRoot.preflightCanStartPreprocess?.(...args),
    startPreprocessFromPreflight: (...args) => legacyRoot.startPreprocessFromPreflight?.(...args),
    currentTrainingConfigFile: (...args) => legacyRoot.currentTrainingConfigFile?.(...args),
    preflightPlainText: (...args) => legacyRoot.preflightPlainText?.(...args),
};

export function configurePreflightDialogBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in preflightDialogBridge) {
            preflightDialogBridge[key] = handler;
        }
    }
}

export function showPreflightPendingDialog(...args) { return preflightDialogBridge.showPreflightPendingDialog(...args); }
export function renderPreflightPending(...args) { return preflightDialogBridge.renderPreflightPending(...args); }
export function showPreflightRequestError(...args) { return preflightDialogBridge.showPreflightRequestError(...args); }
export function isPreflightDialogOpen(...args) { return preflightDialogBridge.isPreflightDialogOpen(...args); }
export function waitForPreflightDialogClose(...args) { return preflightDialogBridge.waitForPreflightDialogClose(...args); }
export function renderPreflightResult(...args) { return preflightDialogBridge.renderPreflightResult(...args); }
export function preflightCanStartPreprocess(...args) { return preflightDialogBridge.preflightCanStartPreprocess(...args); }
export function startPreprocessFromPreflight(...args) { return preflightDialogBridge.startPreprocessFromPreflight(...args); }
export function currentTrainingConfigFile(...args) { return preflightDialogBridge.currentTrainingConfigFile(...args); }
export function preflightPlainText(...args) { return preflightDialogBridge.preflightPlainText(...args); }
