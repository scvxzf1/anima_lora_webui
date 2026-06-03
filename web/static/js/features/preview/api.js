export async function fetchPreviewSettings(ctx, state) {
    const taskQuery = state.selectedTaskId && !state.selectedGroup
        ? `?task_id=${encodeURIComponent(state.selectedTaskId)}`
        : '';
    return ctx.api('/api/preview/settings' + taskQuery);
}

export async function savePreviewSettingsRequest(ctx, values) {
    return ctx.api('/api/preview/settings', {
        method: 'PUT',
        body: JSON.stringify(values),
    });
}

export async function fetchPreviewImages(ctx, state, deps) {
    const params = new URLSearchParams({ source: state.source });
    addTrainingPreviewParams(params, state, deps);
    return ctx.api(`/api/preview/images?${params.toString()}`);
}

export async function fetchPreviewWeights(ctx, state, deps) {
    const params = new URLSearchParams({ source: 'training' });
    addTrainingPreviewParams(params, state, deps);
    return ctx.api(`/api/preview/weights?${params.toString()}`);
}

function addTrainingPreviewParams(params, state, deps) {
    if (state.source !== 'training') return;
    if (state.selectedGroup) {
        params.set('mode', 'config_group');
        params.set('methods_subdir', state.selectedGroup.methods_subdir);
        params.set('variant', state.selectedGroup.variant);
        params.set('preset', state.selectedGroup.preset || 'default');
        if (state.selectedGroup.history_group_key) {
            params.set('group_key', state.selectedGroup.history_group_key);
        }
        params.set('include_archived', deps.getShowArchivedHistory() ? '1' : '0');
    } else if (state.selectedTaskId) {
        params.set('task_id', state.selectedTaskId);
    }
}
