let tomlState = null;

export function configureTomlStateBridge(state) {
    tomlState = state || null;
}

export function getTomlState() {
    if (!tomlState) {
        throw new Error('toml state bridge is not configured');
    }
    return tomlState;
}
