/* Lazy rendering and lifecycle for history detail charts. */

export function createHistoryMetricsController(root, model) {
    let loading = null;
    let cleanups = [];
    let rendered = Boolean(model.lossChart || model.systemCharts);
    let disposed = false;
    const activate = () => {
        if (disposed || rendered || loading) return;
        loading = Promise.all([
            import('./history-chart.js?v=dragon-ui-20260826v6'),
            import('./history-system-charts.js?v=dragon-ui-20260826v7'),
        ]).then(([chart, system]) => {
            loading = null;
            if (disposed) return;
            const metrics = Array.isArray(model.payload.metrics) ? model.payload.metrics : [];
            const chartContainer = root.querySelector('[data-history-chart-container]');
            const systemHost = root.querySelector('[data-history-system-host]');
            if (chartContainer) chartContainer.innerHTML = chart.renderHistoryMetricsChart(metrics);
            if (systemHost) systemHost.innerHTML = system.renderHistorySystemCharts(model.payload.system, model.payload.limits);
            cleanups = [chart.bindHistoryChart(root, metrics), system.bindHistorySystemCharts(root, model.payload.system)];
            rendered = true;
        }).catch(() => { loading = null; });
    };
    return {
        activateTab(tab) { if (tab === 'metrics') activate(); },
        dispose() {
            disposed = true;
            cleanups.forEach((cleanup) => cleanup?.());
            cleanups = [];
        },
    };
}
