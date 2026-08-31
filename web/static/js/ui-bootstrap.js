const CLASSIC_ENTRY = '/static/app.js?v=module-bootstrap-20260828-model-family-filter-v1';
const CLASSIC_STYLESHEET = '/static/style.css?v=frontend-chain-20260809-model-config1';
const UI_MODE_ENTRY = '/static/js/shared/ui-mode.js?v=dragon-ui-20260816v48';
const DRAGON_ENTRY = '/static/js/dragon-ui/index.js?v=dragon-ui-20260831v21';
const DRAGON_STYLESHEET = '/static/css/dragon-style.css?v=dragon-ui-20260828v155';
const STYLESHEET_TIMEOUT_MS = 10_000;

let activeDragonCleanup = null;

function setBootState(state) {
    document.documentElement.dataset.appBoot = state;
}

export function resolveRequestedUIMode(search = '', storedMode = null) {
    const requestedMode = new URLSearchParams(search).get('ui');
    if (requestedMode === 'classic' || requestedMode === 'dragon') return requestedMode;
    return storedMode === 'classic' ? 'classic' : 'dragon';
}

function stylesheetHref(mode) {
    return mode === 'dragon' ? DRAGON_STYLESHEET : CLASSIC_STYLESHEET;
}

function stylesheetElement() {
    let stylesheet = document.getElementById('app-ui-stylesheet');
    if (stylesheet) return stylesheet;
    stylesheet = document.createElement('link');
    stylesheet.id = 'app-ui-stylesheet';
    stylesheet.rel = 'stylesheet';
    document.head.append(stylesheet);
    return stylesheet;
}

function waitForStylesheet(stylesheet) {
    if (stylesheet.dataset.loadError === 'true') {
        return Promise.reject(new Error(`failed to load UI stylesheet: ${stylesheet.href}`));
    }
    if (stylesheet.dataset.loaded === 'true' || stylesheet.sheet) return Promise.resolve();
    return new Promise((resolve, reject) => {
        const timer = window.setTimeout(() => {
            cleanup();
            reject(new Error(`timed out loading UI stylesheet: ${stylesheet.href}`));
        }, STYLESHEET_TIMEOUT_MS);
        const cleanup = () => {
            window.clearTimeout(timer);
            stylesheet.removeEventListener('load', onLoad);
            stylesheet.removeEventListener('error', onError);
        };
        const onLoad = () => {
            stylesheet.dataset.loaded = 'true';
            delete stylesheet.dataset.loadError;
            cleanup();
            resolve();
        };
        const onError = () => {
            stylesheet.dataset.loadError = 'true';
            delete stylesheet.dataset.loaded;
            cleanup();
            reject(new Error(`failed to load UI stylesheet: ${stylesheet.href}`));
        };
        stylesheet.addEventListener('load', onLoad, { once: true });
        stylesheet.addEventListener('error', onError, { once: true });
    });
}

export async function activateUIStylesheet(mode) {
    const normalizedMode = mode === 'dragon' ? 'dragon' : 'classic';
    const stylesheet = stylesheetElement();
    const href = stylesheetHref(normalizedMode);
    if (stylesheet.dataset.uiStylesheet !== normalizedMode || !stylesheet.href.endsWith(href)) {
        delete stylesheet.dataset.loaded;
        delete stylesheet.dataset.loadError;
        stylesheet.dataset.uiStylesheet = normalizedMode;
        stylesheet.href = href;
    }
    await waitForStylesheet(stylesheet);
}

function resetDragonShell() {
    const dragonRoot = document.getElementById('dragon-root');
    const dragonNav = document.getElementById('dragon-nav');
    const dragonMain = document.getElementById('dragon-main');
    if (dragonRoot) dragonRoot.hidden = true;
    dragonNav?.replaceChildren();
    dragonMain?.replaceChildren();
    dragonMain?.removeAttribute('aria-busy');
    dragonMain?.removeAttribute('data-ui-scale');
    dragonMain?.style.removeProperty('zoom');
    document.documentElement.style.removeProperty('--dragon-user-scale');
    document.documentElement.style.removeProperty('color-scheme');
    delete document.documentElement.dataset.dragonTheme;
    delete document.body.dataset.dragonMobileMenuOpen;
    delete document.body.dataset.dragonUi;
}

export async function cleanupDragonUI() {
    const cleanup = activeDragonCleanup;
    activeDragonCleanup = null;
    if (typeof cleanup === 'function') await cleanup();
    resetDragonShell();
}

async function bootClassicUI({ classicLoader, modeLoader, stylesheetLoader }) {
    const wasHidden = document.body.hidden;
    document.body.hidden = true;
    try {
        await cleanupDragonUI();
        document.documentElement.dataset.uiMode = 'classic';
        await stylesheetLoader('classic');
        const [classicModule, modeModule] = await Promise.all([
            classicLoader(),
            modeLoader(),
        ]);
        modeModule.initClassicUiSwitch();
        await classicModule.startClassicUI();
        modeModule.activateRequestedClassicTab();
        setBootState('classic');
    } finally {
        document.body.hidden = wasHidden;
    }
}

async function bootDragonUI({ dragonLoader, stylesheetLoader }) {
    await stylesheetLoader('dragon');
    const dragonRoot = document.getElementById('dragon-root');
    if (!dragonRoot) throw new Error('#dragon-root mount point not found');
    dragonRoot.hidden = false;
    const dragonModule = await dragonLoader();
    try {
        const cleanup = await dragonModule.initDragonUI();
        activeDragonCleanup = typeof cleanup === 'function'
            ? cleanup
            : dragonModule.destroyDragonUI;
    } catch (error) {
        await dragonModule.destroyDragonUI?.();
        throw error;
    }
    setBootState('dragon');
}

export async function bootstrapUI({
    classicLoader = () => import(CLASSIC_ENTRY),
    modeLoader = () => import(UI_MODE_ENTRY),
    dragonLoader = () => import(DRAGON_ENTRY),
    stylesheetLoader = activateUIStylesheet,
} = {}) {
    const requestedMode = document.documentElement.dataset.uiMode
        || (document.body.hasAttribute('data-dragon-ui') ? 'dragon' : 'classic');
    if (requestedMode !== 'dragon') {
        delete document.documentElement.dataset.dragonFallback;
        await bootClassicUI({ classicLoader, modeLoader, stylesheetLoader });
        return;
    }

    try {
        await bootDragonUI({ dragonLoader, stylesheetLoader });
        delete document.documentElement.dataset.dragonFallback;
    } catch (error) {
        console.error('[dragon-ui] failed to start; falling back to classic UI', error);
        document.documentElement.dataset.dragonFallback = 'true';
        await bootClassicUI({ classicLoader, modeLoader, stylesheetLoader });
    }
}

if (typeof document !== 'undefined') {
    bootstrapUI().catch((error) => {
        setBootState('error');
        if (document.body) document.body.hidden = false;
        console.error('[ui-bootstrap] failed to start any UI', error);
    });
}
