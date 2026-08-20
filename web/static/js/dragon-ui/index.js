/* Dragon trainer UI entry point.
 * Initializes theme, navigation, router, and loads the default page.
 * Activated when body[data-dragon-ui] is set (default UI mode).
 */

import { initTheme } from './theme.js?v=dragon-ui-20260814v45';
import { destroyNav, initNav } from './nav.js?v=dragon-ui-20260817v70';
import { canLeaveCurrentPage, destroyRouter, initRouter, isCurrentPage, navigate, refreshCurrentRoute } from './router.js?v=dragon-ui-20260816v68';
import { isConfigCategory } from './category-map.js?v=dragon-ui-20260814v43';
import { destroyAnimations, initScrollAnimations, initParallax } from './animations.js?v=dragon-ui-20260816v67';
import { loadDashboard } from './pages/dashboard.js?v=dragon-ui-20260814v43';
import { loadConfigPage } from './pages/config-page.js?v=dragon-ui-20260819v78';
import { loadLiveTraining } from './pages/live-training.js?v=dragon-ui-20260814v43';
import { loadHistory } from './pages/history.js?v=dragon-ui-20260819v92';
import { loadQueue } from './pages/queue.js?v=dragon-ui-20260814v43';
import { loadWeightAnalysis } from './pages/weight-analysis.js?v=dragon-ui-20260814v43';
import { loadImageTest } from './pages/image-test.js?v=dragon-ui-20260814v43';
import { loadEnvironment } from './pages/environment.js?v=dragon-ui-20260814v43';
import { loadDatasetEditor } from './pages/dataset-editor.js?v=dragon-ui-20260816v70';
import { loadModelConfig } from './pages/model-config.js?v=dragon-ui-20260817v5';
import { loadGlobalSettings } from './pages/global-settings.js?v=dragon-ui-20260814v43';
import { loadPreviewWorkspace } from './pages/preview-workspace.js?v=dragon-ui-20260814v43';
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
        await loadAndApplyDragonUIScale(dragonApi);
        const mount = document.getElementById('dragon-main');
        if (!mount) throw new Error('#dragon-main mount point not found');

        const loaders = {
            dashboard: loadDashboard,
            config: loadConfigPage,
            'live-training': loadLiveTraining,
            history: loadHistory,
            queue: loadQueue,
            'weight-analysis': loadWeightAnalysis,
            'image-test': loadImageTest,
            environment: loadEnvironment,
            'dataset-editor': loadDatasetEditor,
            'model-config': loadModelConfig,
            'global-settings': loadGlobalSettings,
            'preview-workspace': loadPreviewWorkspace,
        };

        initRouter(mount, loaders);
        initNav(async (route) => {
            if (!canLeaveCurrentPage()) return false;
            if (route.type === 'external') return true;
            return navigate(route);
        });
        initScrollAnimations();
        initParallax();

        let acceptedHash = window.location.hash;
        const handleWindowHashChange = async () => {
            const nextHash = window.location.hash;
            const staysOnDatasetPage = isCurrentPage('dataset-editor') && nextHash.startsWith('#dataset-editor');
            if (!staysOnDatasetPage && !canLeaveCurrentPage()) {
                history.replaceState(null, '', `${window.location.pathname}${window.location.search}${acceptedHash || ''}`);
                window.dispatchEvent(new CustomEvent('dragon-route-restored'));
                return;
            }
            acceptedHash = nextHash;
            await handleHashChange();
        };
        const handleRouteRefresh = () => {
            if (canLeaveCurrentPage()) refreshCurrentRoute();
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
        throw error;
    }
}

export function destroyDragonUI() {
    routeChangeSequence += 1;
    const cleanupCallbacks = activeCleanupCallbacks;
    activeCleanupCallbacks = [];
    for (const cleanup of cleanupCallbacks.reverse()) cleanup();
    destroyNav();
    destroyRouter();
    destroyAnimations();
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
                await navigate({ type: 'page', page: parts[1] });
            } else if (parts[0] === 'history') {
                await navigate({
                    type: 'page',
                    page: 'history',
                    taskId: decodeHashPart(parts[1]),
                    sub: parts[2] || null,
                });
            } else if (parts[0]) {
                await navigate({ type: 'sub', subId: parts[0] });
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
    if (parts[0] === 'page') return parts[1] || 'dashboard';
    return parts[0] || 'dashboard';
}
