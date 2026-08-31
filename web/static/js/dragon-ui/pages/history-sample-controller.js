/* Lazy interaction binding for the history artifacts sample dialog. */

export function createHistorySampleController(root, images) {
    let cleanup = null;
    let loading = null;
    let disposed = false;
    const activate = () => {
        if (disposed || cleanup || loading) return;
        loading = import('./history-sample-dialog.js?v=dragon-ui-20260826v4')
            .then((module) => {
                loading = null;
                const nextCleanup = module.bindHistorySampleDialog(root, images);
                if (disposed) nextCleanup?.();
                else cleanup = nextCleanup;
            })
            .catch(() => { loading = null; });
    };
    return {
        activateTab(tab) { if (tab === 'artifacts') activate(); },
        dispose() {
            disposed = true;
            cleanup?.();
            cleanup = null;
        },
    };
}
