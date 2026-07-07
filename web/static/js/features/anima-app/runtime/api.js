export function createRuntimeApi(ctx) {
    const api = function runtimeApi(url, opts = {}) {
        return ctx.api(url, opts);
    };
    api.request = api;
    api.datasetPresetApi = async function datasetPresetApi(url, opts = {}) {
        const timeoutMs = Number(opts.timeoutMs || 15000);
        const requestOpts = { ...opts };
        delete requestOpts.timeoutMs;
        let timeoutId = null;
        try {
            return await Promise.race([
                api(url, requestOpts),
                new Promise((_, reject) => {
                    timeoutId = window.setTimeout(() => {
                        reject(new Error('数据集预设请求超时，请查看终端日志或刷新预设列表'));
                    }, timeoutMs);
                }),
            ]);
        } finally {
            if (timeoutId !== null) {
                window.clearTimeout(timeoutId);
            }
        }
    };
    return Object.freeze(api);
}
