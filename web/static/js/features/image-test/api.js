export async function fetchImageTestStatus(ctx) {
    return ctx.api('/api/image-test/status');
}

export async function startImageTestRequest(ctx, payload) {
    return ctx.api('/api/image-test/start', {
        method: 'POST',
        body: JSON.stringify(payload),
    });
}

export async function stopImageTestRequest(ctx) {
    return ctx.api('/api/image-test/stop', {
        method: 'POST',
    });
}

export async function fetchImageTestWeights(ctx) {
    return ctx.api('/api/analysis/weights');
}

export async function fetchImageTestImages(ctx, limit = 24) {
    const params = new URLSearchParams({
        source: 'inference',
        limit: String(limit),
    });
    return ctx.api(`/api/preview/images?${params.toString()}`);
}
