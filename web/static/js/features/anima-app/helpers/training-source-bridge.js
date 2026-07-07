const trainingSourceBridge = {
    configTrainingSourceMode: () => 'fresh',
    setConfigTrainingSourceMode: async () => {},
    auditConfigTrainingSourceOnEnter: async () => true,
    renderContinueTrainingSource: () => {},
    trainingSourceLaunchReadiness: () => ({ ready: true, checking: false, reason: '' }),
    trainingSourceLaunchBlockReason: () => '训练来源审查未通过。',
    ensureTrainingSourceReadyForLaunch: async () => true,
    continueTrainingRequestPayload: () => ({}),
    auditConfigFullResumeSource: async () => false,
    auditConfigWeightHotstartSource: async () => false,
    startConfigFullResumeSource: async () => {},
    handleConfigFullResumeTaskChange: async () => {},
    handleConfigFullResumeCheckpointChange: () => {},
    trainingSourceLaunchSummary: () => '\n\n训练来源: 从零训练',
    refreshContinueTrainingSourceCompatibility: async () => true,
    selectContinueLoraWeight: async () => false,
    clearContinueTrainingSource: () => {},
};

export function configureTrainingSourceBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in trainingSourceBridge) {
            trainingSourceBridge[key] = handler;
        }
    }
}

export function configTrainingSourceMode(...args) {
    return trainingSourceBridge.configTrainingSourceMode(...args);
}

export function setConfigTrainingSourceMode(...args) {
    return trainingSourceBridge.setConfigTrainingSourceMode(...args);
}

export function auditConfigTrainingSourceOnEnter(...args) {
    return trainingSourceBridge.auditConfigTrainingSourceOnEnter(...args);
}

export function renderContinueTrainingSource(...args) {
    return trainingSourceBridge.renderContinueTrainingSource(...args);
}

export function trainingSourceLaunchReadiness(...args) {
    return trainingSourceBridge.trainingSourceLaunchReadiness(...args);
}

export function trainingSourceLaunchBlockReason(...args) {
    return trainingSourceBridge.trainingSourceLaunchBlockReason(...args);
}

export function ensureTrainingSourceReadyForLaunch(...args) {
    return trainingSourceBridge.ensureTrainingSourceReadyForLaunch(...args);
}

export function continueTrainingRequestPayload(...args) {
    return trainingSourceBridge.continueTrainingRequestPayload(...args);
}

export function auditConfigFullResumeSource(...args) {
    return trainingSourceBridge.auditConfigFullResumeSource(...args);
}

export function auditConfigWeightHotstartSource(...args) {
    return trainingSourceBridge.auditConfigWeightHotstartSource(...args);
}

export function startConfigFullResumeSource(...args) {
    return trainingSourceBridge.startConfigFullResumeSource(...args);
}

export function handleConfigFullResumeTaskChange(...args) {
    return trainingSourceBridge.handleConfigFullResumeTaskChange(...args);
}

export function handleConfigFullResumeCheckpointChange(...args) {
    return trainingSourceBridge.handleConfigFullResumeCheckpointChange(...args);
}

export function trainingSourceLaunchSummary(...args) {
    return trainingSourceBridge.trainingSourceLaunchSummary(...args);
}

export function refreshContinueTrainingSourceCompatibility(...args) {
    return trainingSourceBridge.refreshContinueTrainingSourceCompatibility(...args);
}

export function selectContinueLoraWeight(...args) {
    return trainingSourceBridge.selectContinueLoraWeight(...args);
}

export function clearContinueTrainingSource(...args) {
    return trainingSourceBridge.clearContinueTrainingSource(...args);
}
