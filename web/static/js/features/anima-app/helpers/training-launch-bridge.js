const trainingLaunchHandlers = Object.create(null);

function requireTrainingLaunchHandler(name) {
    const handler = trainingLaunchHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[training-launch] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureTrainingLaunchBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            trainingLaunchHandlers[key] = handler;
        }
    }
}

export function startTraining(...args) { return requireTrainingLaunchHandler('startTraining')(...args); }
export function queueCurrentTrainingFromConfig(...args) { return requireTrainingLaunchHandler('queueCurrentTrainingFromConfig')(...args); }
export function runPreflight(...args) { return requireTrainingLaunchHandler('runPreflight')(...args); }
export function isCliOnlySpdSource(...args) { return requireTrainingLaunchHandler('isCliOnlySpdSource')(...args); }
export function currentTrainingConfigIsRuntime(...args) { return requireTrainingLaunchHandler('currentTrainingConfigIsRuntime')(...args); }
export function chooseTrainingLaunchMode(...args) { return requireTrainingLaunchHandler('chooseTrainingLaunchMode')(...args); }
export function confirmTrainingLaunch(...args) { return requireTrainingLaunchHandler('confirmTrainingLaunch')(...args); }
export function startTrainingUnchecked(...args) { return requireTrainingLaunchHandler('startTrainingUnchecked')(...args); }
export function enqueueTrainingFromConfig(...args) { return requireTrainingLaunchHandler('enqueueTrainingFromConfig')(...args); }
export function enqueueTrainingQueueRequest(...args) { return requireTrainingLaunchHandler('enqueueTrainingQueueRequest')(...args); }
export function enqueueTrainingQueueBatchRequest(...args) { return requireTrainingLaunchHandler('enqueueTrainingQueueBatchRequest')(...args); }
export function enterLiveTrainingForNewRun(...args) { return requireTrainingLaunchHandler('enterLiveTrainingForNewRun')(...args); }
export function showPreflightDialog(...args) { return requireTrainingLaunchHandler('showPreflightDialog')(...args); }
