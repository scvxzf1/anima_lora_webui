import { ensureDragonRouteStyles } from './route-styles.js?v=dragon-ui-20260902-training-nav-v3';

function lazyPage(importer, exportName, contextArgs = (context) => [context]) {
    let loaderPromise = null;
    return async (context) => {
        if (!loaderPromise) {
            loaderPromise = importer().then((module) => {
                const loader = module[exportName];
                if (typeof loader !== 'function') throw new Error(`Dragon page loader ${exportName} is unavailable`);
                return loader;
            });
        }
        try {
            return await (await loaderPromise)(...contextArgs(context));
        } catch (error) {
            loaderPromise = null;
            throw error;
        }
    };
}

function styledPage(styleKey, loader) {
    return async (context) => {
        await ensureDragonRouteStyles(typeof styleKey === 'function' ? styleKey(context) : styleKey);
        return loader(context);
    };
}

export function createDragonPageLoaders() {
    const loadHistoryList = lazyPage(
        () => import('./pages/history.js?v=dragon-ui-20260902v1'),
        'loadHistory',
    );
    const loadHistoryDetail = lazyPage(
        () => import('./pages/history-detail.js?v=dragon-ui-20260826v6'),
        'loadHistoryDetail',
        (context = {}) => [context.taskId, context.sub],
    );
    const loadCaptioning = lazyPage(
        () => import('./pages/captioning.js?v=dragon-ui-20260902v16'),
        'loadCaptioning',
    );
    return {
        dashboard: styledPage('dashboard', lazyPage(() => import('./pages/dashboard.js?v=dragon-ui-20260826v45'), 'loadDashboard')),
        config: styledPage('config', lazyPage(() => import('./pages/config-page.js?v=dragon-ui-20260903-pp-multimodel-v1'), 'loadConfigPage')),
        'live-training': styledPage('live', lazyPage(() => import('./pages/live-training.js?v=dragon-ui-20260826v54'), 'loadLiveTraining')),
        history: styledPage(
            (context = {}) => context.taskId ? 'history-detail' : 'history-list',
            (context = {}) => context.taskId ? loadHistoryDetail(context) : loadHistoryList(context),
        ),
        queue: styledPage('pages', lazyPage(() => import('./pages/queue.js?v=dragon-ui-20260825v2'), 'loadQueue')),
        'weight-analysis': styledPage('pages', lazyPage(() => import('./pages/weight-analysis.js?v=dragon-ui-20260814v43'), 'loadWeightAnalysis')),
        'image-test': styledPage('pages', lazyPage(() => import('./pages/image-test.js?v=dragon-ui-20260824v114'), 'loadImageTest')),
        environment: styledPage('pages', lazyPage(() => import('./pages/environment.js?v=dragon-ui-20260814v43'), 'loadEnvironment')),
        'dataset-editor': styledPage('dataset', lazyPage(() => import('./pages/dataset-editor.js?v=dragon-ui-20260902v138'), 'loadDatasetEditor')),
        'model-config': styledPage('pages', lazyPage(() => import('./pages/model-config.js?v=dragon-ui-20260824-zimage-v1'), 'loadModelConfig')),
        'global-settings': styledPage('pages', lazyPage(() => import('./pages/global-settings.js?v=dragon-ui-20260902v47'), 'loadGlobalSettings')),
        'preview-workspace': styledPage('pages', lazyPage(() => import('./pages/preview-workspace.js?v=dragon-ui-20260814v43'), 'loadPreviewWorkspace')),
        captioning: styledPage('captioning', loadCaptioning),
        tagging: styledPage('captioning', loadCaptioning),
        'captioning-prompts': styledPage('captioning', lazyPage(
            () => import('./pages/tagging-prompt-presets-page.js?v=dragon-ui-20260901v4'),
            'loadTaggingPromptPresetsPage',
        )),
        'captioning-providers': styledPage('captioning', lazyPage(
            () => import('./pages/tagging-provider-profiles-page.js?v=dragon-ui-20260902v5'),
            'loadTaggingProviderProfilesPage',
        )),
        'captioning-results': styledPage('captioning', lazyPage(
            () => import('./pages/tagging-results-controller.js?v=dragon-ui-20260902v9'),
            'loadTaggingResultsPage',
        )),
        'captioning-logs': styledPage('captioning', lazyPage(
            () => import('./pages/tagging-logs-page.js?v=dragon-ui-20260901v5'),
            'loadTaggingLogsPage',
        )),
    };
}
