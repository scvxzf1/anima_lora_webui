const legacyRoot = globalThis;

const appShellStartupBridge = {
    startAnimaApp: (...args) => legacyRoot.startAnimaApp?.(...args),
    openTutorialDialog: (...args) => legacyRoot.openTutorialDialog?.(...args),
    loadVariants: (...args) => legacyRoot.loadVariants?.(...args),
    loadConfig: (...args) => legacyRoot.loadConfig?.(...args),
    reloadCurrentConfig: (...args) => legacyRoot.reloadCurrentConfig?.(...args),
    renderConfigForm: (...args) => legacyRoot.renderConfigForm?.(...args),
    syncConfigDraftFromForm: (...args) => legacyRoot.syncConfigDraftFromForm?.(...args),
    parseNetworkArgMap: (...args) => legacyRoot.parseNetworkArgMap?.(...args),
    normalizeNetworkArgArray: (...args) => legacyRoot.normalizeNetworkArgArray?.(...args),
    getThemeController: () => legacyRoot.themeController ?? null,
    getUiScaleController: () => legacyRoot.uiScaleController ?? null,
    getGpuPicker: () => legacyRoot.gpuPicker ?? null,
    getTabController: () => legacyRoot.tabController ?? null,
};

export function configureAppShellStartupBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in appShellStartupBridge) {
            appShellStartupBridge[key] = handler;
        }
    }
}

export function startAnimaApp(...args) { return appShellStartupBridge.startAnimaApp(...args); }
export function openTutorialDialog(...args) { return appShellStartupBridge.openTutorialDialog(...args); }
export function loadVariants(...args) { return appShellStartupBridge.loadVariants(...args); }
export function loadConfig(...args) { return appShellStartupBridge.loadConfig(...args); }
export function reloadCurrentConfig(...args) { return appShellStartupBridge.reloadCurrentConfig(...args); }
export function renderConfigForm(...args) { return appShellStartupBridge.renderConfigForm(...args); }
export function syncConfigDraftFromForm(...args) { return appShellStartupBridge.syncConfigDraftFromForm(...args); }
export function parseNetworkArgMap(...args) { return appShellStartupBridge.parseNetworkArgMap(...args); }
export function normalizeNetworkArgArray(...args) { return appShellStartupBridge.normalizeNetworkArgArray(...args); }
export function getThemeController() { return appShellStartupBridge.getThemeController(); }
export function getUiScaleController() { return appShellStartupBridge.getUiScaleController(); }
export function getGpuPicker() { return appShellStartupBridge.getGpuPicker(); }
export function getTabController() { return appShellStartupBridge.getTabController(); }
