export function fetchTrainingQueue(ctx) {
    return ctx.api('/api/training/queue');
}

export function enqueueTrainingQueue(ctx, options = {}) {
    return ctx.api('/api/training/queue/start', {
        method: 'POST',
        body: JSON.stringify({
            variant: options.variant,
            preset: options.preset,
            methods_subdir: options.methodsSubdir,
            config_file: options.configFile,
            extra_args: [],
            gpu_whitelist: options.gpuWhitelist || [],
            confirmed: true,
            confirm_preprocess: Boolean(options.willAutoPreprocess),
            start_paused: Boolean(options.startPaused),
            ...(options.continuePayload || {}),
        }),
    });
}

export function enqueueTrainingQueueBatch(ctx, options = {}) {
    return enqueueTrainingQueueBatchAt(ctx, '/api/training/queue/batch/start', options);
}

export function enqueueTrainingQueueBatchAlias(ctx, options = {}) {
    return enqueueTrainingQueueBatchAt(ctx, '/api/training/queue/batch-start', options);
}

export function enqueueTrainingQueueBatchRoot(ctx, options = {}) {
    return enqueueTrainingQueueBatchAt(ctx, '/api/training/queue', options);
}

function enqueueTrainingQueueBatchAt(ctx, url, options = {}) {
    return ctx.api(url, {
        method: 'POST',
        body: JSON.stringify({
            items: options.items || [],
            preset: options.preset,
            gpu_whitelist: options.gpuWhitelist || [],
            start_paused: options.startPaused !== false,
        }),
    });
}

export function resumeTrainingQueue(ctx, options = {}) {
    return ctx.api('/api/training/queue/resume', {
        method: 'POST',
        body: JSON.stringify({
            task_id: options.taskId,
            checkpoint: options.checkpoint,
            gpu_whitelist: options.gpuWhitelist || [],
        }),
    });
}

export function moveTrainingQueueItem(ctx, itemId, direction) {
    return ctx.api(`/api/training/queue/${encodeURIComponent(itemId)}/move`, {
        method: 'POST',
        body: JSON.stringify({ direction }),
    });
}

export function deleteTrainingQueueItem(ctx, itemId) {
    return ctx.api(`/api/training/queue/${encodeURIComponent(itemId)}`, { method: 'DELETE' });
}

export function retryTrainingQueueItem(ctx, itemId) {
    return ctx.api(`/api/training/queue/${encodeURIComponent(itemId)}/retry`, { method: 'POST' });
}

export function cancelWaitingTrainingQueue(ctx) {
    return ctx.api('/api/training/queue/cancel-waiting', { method: 'POST' });
}

export function cancelAllTrainingQueue(ctx) {
    return ctx.api('/api/training/queue/cancel-all', { method: 'POST' });
}

export function abortTrainingQueueAfterCurrent(ctx) {
    return ctx.api('/api/training/queue/abort-after-current', { method: 'POST' });
}

export function forceAbortTrainingQueue(ctx) {
    return ctx.api('/api/training/queue/force-abort', { method: 'POST' });
}

export function clearCompletedTrainingQueue(ctx) {
    return ctx.api('/api/training/queue/clear-completed', { method: 'POST' });
}

export function clearCanceledTrainingQueue(ctx) {
    return ctx.api('/api/training/queue/clear-canceled', { method: 'POST' });
}

export function updateTrainingQueueSettingsRequest(ctx, patch) {
    return ctx.api('/api/training/queue/settings', {
        method: 'POST',
        body: JSON.stringify(patch),
    });
}
