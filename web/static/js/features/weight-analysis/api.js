export async function fetchAnalysisWeights(ctx) {
    return ctx.api('/api/analysis/weights');
}

export async function inspectAnalysisWeight(ctx, path) {
    return ctx.api('/api/analysis/inspect', {
        method: 'POST',
        body: JSON.stringify({ path }),
    });
}

export async function inspectAnalysisWeightFile(ctx, file) {
    const form = new FormData();
    form.append('file', file, file.name || 'uploaded.safetensors');
    return ctx.api('/api/analysis/inspect-upload', {
        method: 'POST',
        headers: {},
        body: form,
    });
}
