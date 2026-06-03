export function setResumeLoadingForTask(state, taskId) {
    state.resumeOptions = {
        loading: true,
        taskId,
        checkpoints: [],
        defaultCheckpoint: '',
        error: '',
        message: '正在读取可续训检查点...',
        diagnostic: {},
    };
    state.resumeWeights = {
        loading: true,
        taskId,
        weights: [],
        error: '',
        message: '正在读取可热启动权重...',
    };
}

export function clearResumeState(state) {
    state.resumeOptions = {
        loading: false,
        taskId: '',
        checkpoints: [],
        defaultCheckpoint: '',
        error: '',
        message: '',
        diagnostic: {},
    };
    state.resumeWeights = {
        loading: false,
        taskId: '',
        weights: [],
        error: '',
        message: '',
    };
}

export function selectedResumeCheckpointFromState(state) {
    const select = document.getElementById('resume-checkpoint-select');
    const value = select?.value || '';
    if (!value) return null;
    return state.resumeOptions.checkpoints.find((item) => item.path === value) || null;
}

export function selectedHistoryManagerResumeCheckpointFromState(state) {
    const select = document.getElementById('history-manager-resume-select');
    const value = select?.value || '';
    if (!value) return null;
    return state.resumeOptions.checkpoints.find((item) => item.path === value) || null;
}

export function resumeCheckpointOptionLabel(item) {
    return [
        item.kind_label || '训练状态',
        resumeCheckpointProgressText(item),
        item.scope_label || '',
        item.name || '',
    ].filter(Boolean).join(' · ');
}

export function resumeCheckpointProgressText(item) {
    const parts = [];
    if (item.epoch != null) parts.push(`Epoch ${item.epoch}`);
    if (item.step != null) parts.push(`Step ${item.step}`);
    return parts.join(' / ') || '步数未知';
}

export function resumeSummaryLine(label, value) {
    const row = document.createElement('div');
    const key = document.createElement('span');
    key.textContent = label;
    const valEl = document.createElement('code');
    valEl.textContent = value || '-';
    row.append(key, valEl);
    return row;
}
