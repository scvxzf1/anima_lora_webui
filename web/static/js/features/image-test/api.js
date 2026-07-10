export async function fetchImageTestStatus(ctx) {
    return ctx.api('/api/image-test/status');
}

export async function deleteImageTestImagesRequest(ctx, payload) {
    return ctx.api('/api/image-test/images', {
        method: 'DELETE',
        body: JSON.stringify(payload),
    });
}

export async function resolveImageTestWeightPathRequest(ctx, payload) {
    return ctx.api('/api/image-test/resolve-weight', {
        method: 'POST',
        body: JSON.stringify(payload),
    });
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

export async function fetchImageTestGpus(ctx) {
    return ctx.api('/api/training/gpus');
}

export async function fetchImageTestImages(ctx, limit = 24, historyRange = '7') {
    const params = new URLSearchParams({
        source: 'inference',
        limit: String(limit),
    });
    const normalizedRange = String(historyRange || '').trim().toLowerCase();
    if (normalizedRange && normalizedRange !== 'all') {
        params.set('days', normalizedRange);
    }
    return ctx.api(`/api/preview/images?${params.toString()}`);
}
