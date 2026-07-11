const appShellStartupHandlers = Object.create(null);

function requireAppShellStartupHandler(name) {
    const handler = appShellStartupHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[app-shell-startup] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureAppShellStartupBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            appShellStartupHandlers[key] = handler;
        }
    }
}

export function startAnimaApp(...args) { return requireAppShellStartupHandler('startAnimaApp')(...args); }
export function openTutorialDialog(...args) { return requireAppShellStartupHandler('openTutorialDialog')(...args); }
export function loadVariants(...args) { return requireAppShellStartupHandler('loadVariants')(...args); }
export function loadConfig(...args) { return requireAppShellStartupHandler('loadConfig')(...args); }
export function reloadCurrentConfig(...args) { return requireAppShellStartupHandler('reloadCurrentConfig')(...args); }
export function renderConfigForm(...args) { return requireAppShellStartupHandler('renderConfigForm')(...args); }
export function syncConfigDraftFromForm(...args) { return requireAppShellStartupHandler('syncConfigDraftFromForm')(...args); }
export function parseNetworkArgMap(...args) { return requireAppShellStartupHandler('parseNetworkArgMap')(...args); }
export function normalizeNetworkArgArray(...args) { return requireAppShellStartupHandler('normalizeNetworkArgArray')(...args); }
export function getThemeController(...args) { return requireAppShellStartupHandler('getThemeController')(...args); }
export function getUiScaleController(...args) { return requireAppShellStartupHandler('getUiScaleController')(...args); }
export function getGpuPicker(...args) { return requireAppShellStartupHandler('getGpuPicker')(...args); }
export function getTabController(...args) { return requireAppShellStartupHandler('getTabController')(...args); }
