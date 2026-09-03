/* Lazy loader and lifecycle boundary for the dataset image preview. */

export function createDatasetPreviewController(api, state) {
    let modulePromise = null;
    let refreshCleanup = null;
    let disposed = false;

    const load = async () => {
        if (!modulePromise) {
            modulePromise = import('./dataset-editor-preview.js?v=dragon-ui-20260902v53').catch((error) => {
                modulePromise = null;
                throw error;
            });
        }
        const module = await modulePromise;
        if (disposed) throw new Error('数据集预览页面已关闭');
        if (!refreshCleanup) refreshCleanup = module.bindDatasetPreviewRefresh(api, state);
        return module;
    };

    return {
        async open(datasetIndex) {
            const module = await load();
            return module.openDatasetPreview(api, state, datasetIndex);
        },
        dispose() {
            disposed = true;
            refreshCleanup?.();
            refreshCleanup = null;
        },
    };
}
