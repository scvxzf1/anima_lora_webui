let trainingState = null;

export function configureTrainingStateBridge(state) {
    trainingState = state || null;
}

export function getTrainingState() {
    if (!trainingState) {
        throw new Error('training state bridge is not configured');
    }
    return trainingState;
}

export function getTrainingSourceState() {
    return getTrainingState().trainingSourceState;
}

export function getContinueTrainingSource() {
    return getTrainingState().continueTrainingSource;
}

export function setContinueTrainingSource(value) {
    getTrainingState().continueTrainingSource = value || null;
    return getTrainingState().continueTrainingSource;
}
