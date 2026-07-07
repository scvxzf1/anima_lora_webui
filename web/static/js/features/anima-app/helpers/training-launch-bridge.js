const legacyRoot = globalThis;

const trainingLaunchBridge = {
    startTraining: (...args) => legacyRoot.startTraining?.(...args),
    queueCurrentTrainingFromConfig: (...args) => legacyRoot.queueCurrentTrainingFromConfig?.(...args),
    runPreflight: (...args) => legacyRoot.runPreflight?.(...args),
    isCliOnlySpdSource: (...args) => legacyRoot.isCliOnlySpdSource?.(...args),
    currentTrainingConfigIsRuntime: (...args) => legacyRoot.currentTrainingConfigIsRuntime?.(...args),
    chooseTrainingLaunchMode: (...args) => legacyRoot.chooseTrainingLaunchMode?.(...args),
    confirmTrainingLaunch: (...args) => legacyRoot.confirmTrainingLaunch?.(...args),
    startTrainingUnchecked: (...args) => legacyRoot.startTrainingUnchecked?.(...args),
    enqueueTrainingFromConfig: (...args) => legacyRoot.enqueueTrainingFromConfig?.(...args),
    enqueueTrainingQueueRequest: (...args) => legacyRoot.enqueueTrainingQueueRequest?.(...args),
    enqueueTrainingQueueBatchRequest: (...args) => legacyRoot.enqueueTrainingQueueBatchRequest?.(...args),
    enterLiveTrainingForNewRun: (...args) => legacyRoot.enterLiveTrainingForNewRun?.(...args),
    showPreflightDialog: (...args) => legacyRoot.showPreflightDialog?.(...args),
};

export function configureTrainingLaunchBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in trainingLaunchBridge) {
            trainingLaunchBridge[key] = handler;
        }
    }
}

export const startTraining = (...args) => trainingLaunchBridge.startTraining(...args);
export const queueCurrentTrainingFromConfig = (...args) => trainingLaunchBridge.queueCurrentTrainingFromConfig(...args);
export const runPreflight = (...args) => trainingLaunchBridge.runPreflight(...args);
export const isCliOnlySpdSource = (...args) => trainingLaunchBridge.isCliOnlySpdSource(...args);
export const currentTrainingConfigIsRuntime = (...args) => trainingLaunchBridge.currentTrainingConfigIsRuntime(...args);
export const chooseTrainingLaunchMode = (...args) => trainingLaunchBridge.chooseTrainingLaunchMode(...args);
export const confirmTrainingLaunch = (...args) => trainingLaunchBridge.confirmTrainingLaunch(...args);
export const startTrainingUnchecked = (...args) => trainingLaunchBridge.startTrainingUnchecked(...args);
export const enqueueTrainingFromConfig = (...args) => trainingLaunchBridge.enqueueTrainingFromConfig(...args);
export const enqueueTrainingQueueRequest = (...args) => trainingLaunchBridge.enqueueTrainingQueueRequest(...args);
export const enqueueTrainingQueueBatchRequest = (...args) => trainingLaunchBridge.enqueueTrainingQueueBatchRequest(...args);
export const enterLiveTrainingForNewRun = (...args) => trainingLaunchBridge.enterLiveTrainingForNewRun(...args);
export const showPreflightDialog = (...args) => trainingLaunchBridge.showPreflightDialog(...args);
