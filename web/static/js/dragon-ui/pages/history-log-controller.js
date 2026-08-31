/* Lazy resource controller for the history detail log tab. */

export function createHistoryLogController(root, model) {
    let viewerCleanup = null;
    let loading = null;
    let disposed = false;

    const ensureViewer = () => {
        if (disposed || viewerCleanup) return loading;
        if (!loading) loading = loadViewer(root, model).then((cleanup) => {
            if (disposed) {
                cleanup?.();
                return;
            }
            viewerCleanup = cleanup;
        }).catch((error) => {
            loading = null;
            setLogStatus(root, error?.message || '加载历史日志失败');
        });
        return loading;
    };

    return {
        activateTab(tab) { if (tab === 'logs') void ensureViewer(); },
        dispose() {
            disposed = true;
            viewerCleanup?.();
            viewerCleanup = null;
        },
    };
}

async function loadViewer(root, model) {
    setLogStatus(root, '正在加载日志…');
    const [, module] = await Promise.all([
        model.ensureLogs(),
        import('./history-log-viewer.js?v=dragon-ui-20260826v9'),
    ]);
    return module.bindHistoryLogViewer(root, model.payload.logs, {
        total: model.payload.limits?.logs_total,
        offset: model.payload.limits?.logs_offset,
        loadRange: model.loadLogRange,
        searchMatch: model.searchLog,
    });
}

function setLogStatus(root, message) {
    const status = root.querySelector('[data-history-log-window-status]');
    if (status) status.textContent = message;
}
