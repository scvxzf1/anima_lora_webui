export function createPreviewState() {
    return {
        settings: null,
        source: 'training',
        selectedTaskId: '',
        selectedGroup: null,
        requestSeq: 0,
        weightRequestSeq: 0,
        weightSortDirection: 'asc',
        trainingSampleState: null,
        panel: {
            open: false,
            previousTab: '',
            restoreTrainingView: '',
        },
    };
}

export function normalizePreviewGroup(group) {
    if (!group) return null;
    return {
        methods_subdir: group.methods_subdir || '',
        variant: group.variant || '',
        preset: group.preset || 'default',
        history_group_key: group.history_group_key || '',
        history_group_label: group.history_group_label || group.label || '',
        history_source_config_file: group.history_source_config_file || group.source_label || '',
        label: group.label || group.history_group_label || group.history_source_config_file || `${group.methods_subdir || '-'} / ${group.variant || '-'}`,
    };
}

export function selectedPreviewSelectValue(state) {
    if (state.selectedGroup) return encodePreviewGroupValue(state.selectedGroup);
    if (state.selectedTaskId) return encodePreviewTaskValue(state.selectedTaskId);
    return '';
}

export function encodePreviewTaskValue(taskId) {
    return `task:${taskId || ''}`;
}

export function encodePreviewGroupValue(group) {
    const payload = [
        group.methods_subdir || '',
        group.variant || '',
        group.preset || 'default',
        group.history_group_key || '',
        group.history_group_label || '',
        group.history_source_config_file || '',
    ].map((value) => encodeURIComponent(value)).join('|');
    return `group:${payload}`;
}

export function decodePreviewGroupValue(value) {
    if (!String(value || '').startsWith('group:')) return null;
    const parts = String(value).slice(6).split('|').map((item) => decodeURIComponent(item));
    if (!parts[0] || !parts[1]) return null;
    return {
        methods_subdir: parts[0],
        variant: parts[1],
        preset: parts[2] || 'default',
        history_group_key: parts[3] || '',
        history_group_label: parts[4] || '',
        history_source_config_file: parts[5] || '',
        label: parts[4] || parts[5] || `${parts[0]} / ${parts[1]} / ${parts[2] || 'default'}`,
    };
}

export function applyPreviewSelectionValue(state, value) {
    const group = decodePreviewGroupValue(value);
    if (group) {
        state.selectedGroup = group;
        state.selectedTaskId = '';
        return;
    }
    state.selectedGroup = null;
    state.selectedTaskId = String(value || '').startsWith('task:') ? String(value).slice(5) : '';
}

export function previewSourceLabel(source) {
    return {
        training: '训练过程中采样结果',
        inference: '推理预览',
        custom: '自定义路径',
    }[source] || '预览图';
}
