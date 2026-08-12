/* Apple-style UI entry point for Anima LoRA Web UI.
 * Initializes theme, navigation, router, and loads the default page.
 * Activated when body[data-apple-ui] is set (default UI mode).
 */

import { initTheme } from './theme.js?v=apple-ui-20260812v33';
import { initNav } from './nav.js?v=apple-ui-20260812v33';
import { initRouter, navigate, refreshCurrentRoute } from './router.js?v=apple-ui-20260812v33';
import { isConfigCategory } from './category-map.js?v=apple-ui-20260812v33';
import { initScrollAnimations, initParallax } from './animations.js?v=apple-ui-20260812v33';
import { loadDashboard } from './pages/dashboard.js?v=apple-ui-20260812v33';
import { loadConfigPage } from './pages/config-page.js?v=apple-ui-20260812v33';
import { loadLiveTraining } from './pages/live-training.js?v=apple-ui-20260812v33';
import { loadHistory } from './pages/history.js?v=apple-ui-20260812v33';
import { loadQueue } from './pages/queue.js?v=apple-ui-20260812v33';
import { loadWeightAnalysis } from './pages/weight-analysis.js?v=apple-ui-20260812v33';
import { loadImageTest } from './pages/image-test.js?v=apple-ui-20260812v33';
import { loadEnvironment } from './pages/environment.js?v=apple-ui-20260812v33';
import { loadDatasetEditor } from './pages/dataset-editor.js?v=apple-ui-20260812v33';
import { loadModelConfig } from './pages/model-config.js?v=apple-ui-20260812v33';
import { loadGlobalSettings } from './pages/global-settings.js?v=apple-ui-20260812v33';
import { loadPreviewWorkspace } from './pages/preview-workspace.js?v=apple-ui-20260812v33';

export async function initAppleUI() {
    initTheme();
    const mount = document.getElementById('apple-main');
    if (!mount) throw new Error('#apple-main mount point not found');

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
    initNav(async (route) => { await navigate(route); });
    initScrollAnimations();
    initParallax();

    window.addEventListener('hashchange', handleHashChange);
    await handleHashChange();

    window.addEventListener('apple-refresh-route', () => refreshCurrentRoute());
}

async function handleHashChange() {
    const mount = document.getElementById('apple-main');
    if (!mount) return;

    try {
        const hash = window.location.hash.slice(1);
        if (hash) {
            const parts = hash.split('/');
            if (parts[0] === 'config' && parts[1]) {
                if (isConfigCategory(parts[1])) {
                    await navigate({ type: 'category', categoryId: parts[1], subId: parts[2] || null });
                } else {
                    await navigate({ type: 'sub', subId: parts[1] });
                }
            } else if (parts[0] === 'page' && parts[1]) {
                await navigate({ type: 'page', page: parts[1] });
            } else if (parts[0]) {
                await navigate({ type: 'sub', subId: parts[0] });
            } else {
                await navigate({ type: 'page', page: 'dashboard' });
            }
        } else {
            await navigate({ type: 'page', page: 'dashboard' });
        }
    } catch (err) {
        console.error('[apple-ui] 页面加载失败', err);
        mount.innerHTML = '<div class="apple-empty-state"><p>页面加载失败，请检查服务器连接</p></div>';
    }
}
