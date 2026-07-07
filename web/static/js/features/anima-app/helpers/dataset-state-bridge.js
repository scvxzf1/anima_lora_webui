let datasetState = null;

export function configureDatasetStateBridge(state) {
    datasetState = state || null;
}

export function getDatasetState() {
    if (!datasetState) {
        throw new Error('dataset state bridge is not configured');
    }
    return datasetState;
}
