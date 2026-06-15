export async function fetchHistoryTask(ctx, taskId) {
    return ctx.api(`/api/training/history/${encodeURIComponent(taskId)}`);
}

export async function fetchResumeOptions(ctx, taskId) {
    return ctx.api(`/api/training/history/${encodeURIComponent(taskId)}/resume-options`);
}

export async function fetchHistoryResumeWeights(ctx, taskId) {
    return ctx.api(`/api/preview/weights?task_id=${encodeURIComponent(taskId)}`);
}

export async function inspectContinueLoraWeight(ctx, { path, variant, preset, methodsSubdir, configFile }) {
    return ctx.api('/api/training/continue-lora/inspect', {
        method: 'POST',
        body: JSON.stringify({
            path,
            variant,
            preset,
            methods_subdir: methodsSubdir,
            config_file: configFile || '',
        }),
    });
}

export async function postResumeTraining(ctx, { taskId, checkpoint, queueMode, gpuWhitelist }) {
    return ctx.api(queueMode ? '/api/training/queue/resume' : '/api/training/resume', {
        method: 'POST',
        body: JSON.stringify({
            task_id: taskId,
            checkpoint,
            gpu_whitelist: gpuWhitelist,
        }),
    });
}
