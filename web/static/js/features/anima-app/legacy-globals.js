export function installLegacyGlobals(runtime) {
    globalThis.__animaRuntime = runtime;
    globalThis.ctx = runtime.ctx;
    globalThis.__animaAppContext = runtime.ctx;
}

export function installLegacyStateGlobals(runtime) {
    for (const bucket of Object.values(runtime.state)) {
        for (const key of Object.keys(bucket)) {
            Object.defineProperty(globalThis, key, {
                configurable: true,
                enumerable: true,
                get: () => bucket[key],
                set: (value) => {
                    bucket[key] = value;
                },
            });
        }
    }
}

export function installLegacyImageTestFeature(runtime, bridge) {
    runtime.features.imageTest = bridge;
    Object.defineProperty(globalThis, 'imageTestFeature', {
        configurable: true,
        enumerable: true,
        get: () => bridge.imageTestFeature,
        set: (value) => {
            bridge.imageTestFeature = value;
        },
    });
    globalThis.ensureImageTestFeature = bridge.ensureImageTestFeature;
}

export function installLegacyStatusPolling(runtime, bridge) {
    runtime.features.statusPolling = bridge;
    globalThis.trainingStatusPollDelayMs = bridge.trainingStatusPollDelayMs;
    globalThis.scheduleStatusPoll = bridge.scheduleStatusPoll;
    globalThis.pollStatus = bridge.pollStatus;
    globalThis.refreshTrainingSidebarSummariesFromPoll = bridge.refreshTrainingSidebarSummariesFromPoll;
    globalThis.applyStatusSnapshotFallbacks = bridge.applyStatusSnapshotFallbacks;
    globalThis.hasStatusPayload = bridge.hasStatusPayload;
}
