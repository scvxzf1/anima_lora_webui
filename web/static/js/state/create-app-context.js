export function createAppContext(deps) {
    const ctx = {
        api: deps.api,
        catalog: deps.catalog,
        dom: deps.dom,
        download: deps.download,
        format: deps.format,
        MetricsChart: deps.MetricsChart,
    };
    return Object.freeze(ctx);
}
