/* Dragon trainer UI entry point.
 * Initializes theme, navigation, router, and loads the default page.
 * Activated when body[data-dragon-ui] is set (default UI mode).
 */

import { initTheme } from './theme.js?v=dragon-ui-20260814v45';
import { destroyNav, initNav } from './nav.js?v=dragon-ui-20260902-training-nav-v3';
import { canLeaveCurrentPage, destroyRouter, initRouter, isCurrentPage, navigate, refreshCurrentRoute } from './router.js?v=dragon-ui-20260902-training-nav-v3';
import { isConfigCategory } from './category-map.js?v=dragon-ui-20260826v45';
import { destroyAnimations, initScrollAnimations, initParallax } from './animations.js?v=dragon-ui-20260824v69';
import { destroyDragonMotion, initDragonMotion } from './motion.js?v=dragon-ui-20260824v1';
import { applyDragonConfigChromeSettings } from './config-chrome.js?v=dragon-ui-20260825v1';
import { trackHistoryDetailEntry } from './history-return-navigation.js?v=dragon-ui-20260825v1';
import { createDragonPageLoaders } from './page-loaders.js?v=dragon-ui-20260902-krea2-pp-v1';
import { clearDragonRouteStyles } from './route-styles.js?v=dragon-ui-20260902-training-nav-v3';
import { createApiClient } from '../shared/api.js?v=dragon-ui-20260812v35';
import { loadAndApplyDragonUIScale } from './ui-scale.js?v=dragon-ui-20260814v43';

const dragonApi = createApiClient();
let routeChangeSequence = 0;
let activeCleanupCallbacks = [];

export async function initDragonUI() {
    destroyDragonUI();
    const cleanupCallbacks = [];
    try {
        const cleanupTheme = initTheme();
        if (typeof cleanupTheme === 'function') cleanupCallbacks.push(cleanupTheme);
        const globalSettings = await loadAndApplyDragonUIScale(dragonApi);
        const cleanupMotion = initDragonMotion(globalSettings || {});
        applyDragonConfigChromeSettings(globalSettings || {});
        cleanupCallbacks.push(cleanupMotion);
        const mount = document.getElementById('dragon-main');
        if (!mount) throw new Error('#dragon-main mount point not found');

        const loaders = createDragonPageLoaders();

        initRouter(mount, loaders);
        initNav(async (route) => {
            if (!(await requestLeaveApproval())) return false;
            if (route.type === 'external') return true;
            return navigate(route);
        });
        initScrollAnimations();
        initParallax();
        const handleMotionChange = () => {
            destroyAnimations();
            initScrollAnimations();
            initParallax();
        };
        window.addEventListener('dragon-motion-change', handleMotionChange);
        cleanupCallbacks.push(() => window.removeEventListener('dragon-motion-change', handleMotionChange));

        let acceptedHash = window.location.hash;
        let hashChangeSequence = 0;
        let leaveCheckPromise = null;
        const requestLeaveApproval = () => {
            if (!leaveCheckPromise) {
                leaveCheckPromise = Promise.resolve(canLeaveCurrentPage()).finally(() => { leaveCheckPromise = null; });
            }
            return leaveCheckPromise;
        };
        const handleWindowHashChange = async () => {
            const nextHash = window.location.hash;
            const sequence = ++hashChangeSequence;
            const staysOnDatasetPage = isCurrentPage('dataset-editor') && nextHash.startsWith('#dataset-editor');
            const staysOnConfigCategory = configCategoryFromHash(acceptedHash) === configCategoryFromHash(nextHash)
                && configCategoryFromHash(nextHash) !== null;
            if (!staysOnDatasetPage && !staysOnConfigCategory && !(await requestLeaveApproval())) {
                if (sequence !== hashChangeSequence) return;
                history.replaceState(null, '', `${window.location.pathname}${window.location.search}${acceptedHash || ''}`);
                window.dispatchEvent(new CustomEvent('dragon-route-restored'));
                return;
            }
            if (sequence !== hashChangeSequence) return;
            trackHistoryDetailEntry(acceptedHash, nextHash);
            acceptedHash = nextHash;
            await handleHashChange();
        };
        const handleRouteRefresh = async () => {
            if (await requestLeaveApproval()) refreshCurrentRoute();
        };
        window.addEventListener('hashchange', handleWindowHashChange);
        window.addEventListener('dragon-refresh-route', handleRouteRefresh);
        cleanupCallbacks.push(
            () => window.removeEventListener('hashchange', handleWindowHashChange),
            () => window.removeEventListener('dragon-refresh-route', handleRouteRefresh),
        );

        await handleHashChange();
        acceptedHash = window.location.hash;
        activeCleanupCallbacks = cleanupCallbacks;
        return destroyDragonUI;
    } catch (error) {
        for (const cleanup of cleanupCallbacks.reverse()) cleanup();
        destroyNav();
        destroyRouter();
        destroyAnimations();
        clearDragonRouteStyles();
        throw error;
    }
}

function configCategoryFromHash(hash) {
    const parts = String(hash || '').replace(/^#/, '').split('/');
    return parts[0] === 'config' && isConfigCategory(parts[1]) ? parts[1] : null;
}

export function destroyDragonUI() {
    routeChangeSequence += 1;
    const cleanupCallbacks = activeCleanupCallbacks;
    activeCleanupCallbacks = [];
    for (const cleanup of cleanupCallbacks.reverse()) cleanup();
    destroyNav();
    destroyRouter();
    destroyAnimations();
    destroyDragonMotion();
    clearDragonRouteStyles();
}

async function handleHashChange() {
    const mount = document.getElementById('dragon-main');
    if (!mount) return;

    try {
        const hash = window.location.hash.slice(1);
        const sequence = ++routeChangeSequence;
        await loadAndApplyDragonUIScale(dragonApi, pageTypeForHash(hash));
        if (sequence !== routeChangeSequence) return;
        if (hash) {
            const parts = hash.split('/');
            if (parts[0] === 'config' && parts[1]) {
                if (isConfigCategory(parts[1])) {
                    await navigate({ type: 'category', categoryId: parts[1], subId: parts[2] || null });
                } else {
                    await navigate({ type: 'sub', subId: parts[1] });
                }
            } else if (parts[0] === 'dashboard') {
                await navigate({ type: 'page', page: 'dashboard' });
            } else if (parts[0] === 'page' && parts[1]) {
                await navigate({ type: 'page', page: normalizePageType(parts[1]) });
            } else if (parts[0] === 'history') {
                await navigate({
                    type: 'page',
                    page: 'history',
                    taskId: decodeHashPart(parts[1]),
                    sub: parts[2] || null,
                });
            } else if (parts[0]) {
                const page = normalizePageType(parts[0]);
                if (isPageAlias(parts[0])) await navigate({ type: 'page', page });
                else await navigate({ type: 'sub', subId: parts[0] });
            } else {
                await navigate({ type: 'page', page: 'dashboard' });
            }
        } else {
            await navigate({ type: 'page', page: 'dashboard' });
        }
    } catch (err) {
        console.error('[dragon-ui] 页面加载失败', err);
        mount.innerHTML = '<div class="dragon-empty-state"><p>页面加载失败，请检查服务器连接</p></div>';
    }
}

function decodeHashPart(value) {
    if (!value) return null;
    try {
        return decodeURIComponent(value);
    } catch {
        return value;
    }
}

function pageTypeForHash(hash) {
    const parts = String(hash || '').split('/');
    if (parts[0] === 'config') return 'config';
    if (parts[0] === 'history') return 'history';
    if (parts[0] === 'page') return normalizePageType(parts[1] || 'dashboard');
    return normalizePageType(parts[0] || 'dashboard');
}

function normalizePageType(value) {
    const page = String(value || '').trim();
    return page === 'tagging' ? 'captioning' : page;
}

function isPageAlias(value) {
    return ['captioning', 'tagging'].includes(String(value || '').trim());
}
