let configState = null;

export function configureConfigStateBridge(state) {
    configState = state || null;
}

export function getConfigState() {
    if (!configState) {
        throw new Error('config state bridge is not configured');
    }
    return configState;
}
