export async function fetchHistoryTask(ctx, taskId) {
    return ctx.api(`/api/training/history/${encodeURIComponent(taskId)}`);
}

export async function fetchResumeOptions(ctx, taskId) {
    return ctx.api(`/api/training/history/${encodeURIComponent(taskId)}/resume-options`);
}

export async function fetchHistoryResumeWeights(ctx, taskId) {
    return ctx.api(`/api/preview/weights?task_id=${encodeURIComponent(taskId)}`);
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
