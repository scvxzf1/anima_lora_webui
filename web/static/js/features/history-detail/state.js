export const HISTORY_DETAIL_TABS = [
    { key: 'overview', label: '概览' },
    { key: 'analysis', label: '训练分析' },
    { key: 'preview', label: '样张与权重' },
    { key: 'logs', label: '日志' },
    { key: 'config_files', label: '配置与文件' },
];

const HISTORY_DETAIL_TAB_ALIASES = {
    resume: 'overview',
    chart: 'analysis',
    system: 'analysis',
    samples: 'preview',
    weights: 'preview',
    config: 'config_files',
    paths: 'config_files',
};

export function normalizeHistoryDetailTab(tab) {
    const normalized = HISTORY_DETAIL_TAB_ALIASES[tab] || tab || 'overview';
    return HISTORY_DETAIL_TABS.some((item) => item.key === normalized) ? normalized : 'overview';
}

export function createHistoryDetailState() {
    return {
        detailTab: 'overview',
        currentPayload: null,
        returnState: null,
        mainTaskReturn: null,
        resumeOptions: createEmptyResumeOptions(),
        resumeWeights: createEmptyResumeWeights(),
        curve: {
            showRaw: true,
            showSmooth: true,
            showLoss: true,
            showLr: true,
            smoothWindow: 15,
            rangeMode: 'all',
            customStart: '',
            customEnd: '',
            hoverStep: null,
        },
        logs: {
            payloadKey: '',
            query: '',
            level: 'all',
            matchIndex: 0,
        },
    };
}

export function createEmptyResumeOptions(overrides = {}) {
    return {
        loading: false,
        taskId: '',
        checkpoints: [],
        defaultCheckpoint: '',
        error: '',
        message: '',
        diagnostic: {},
        ...overrides,
    };
}

export function createEmptyResumeWeights(overrides = {}) {
    return {
        loading: false,
        taskId: '',
        weights: [],
        error: '',
        message: '',
        ...overrides,
    };
}

export function setHistoryDetailTab(state, tab) {
    state.detailTab = normalizeHistoryDetailTab(tab);
    return state.detailTab;
}

export function resetHistoryDetailViewState(state) {
    state.currentPayload = null;
    state.detailTab = 'overview';
    state.returnState = null;
    state.mainTaskReturn = null;
    state.curve.hoverStep = null;
}

export function setResumeLoadingState(state, taskId) {
    state.resumeOptions = createEmptyResumeOptions({
        loading: true,
        taskId,
        message: '正在读取可续训检查点...',
    });
    state.resumeWeights = createEmptyResumeWeights({
        loading: true,
        taskId,
        message: '正在读取可热启动权重...',
    });
}

export function clearResumeState(state) {
    state.resumeOptions = createEmptyResumeOptions();
    state.resumeWeights = createEmptyResumeWeights();
}
