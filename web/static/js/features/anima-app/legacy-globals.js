export function installLegacyGlobals(runtime) {
    globalThis.ctx = runtime.ctx;
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
    globalThis.ensureImageTestFeature = bridge.ensureImageTestFeature;
}

export function installLegacyStatusPolling(runtime, bridge) {
    runtime.features.statusPolling = bridge;
    globalThis.scheduleStatusPoll = bridge.scheduleStatusPoll;
    globalThis.pollStatus = bridge.pollStatus;
}
