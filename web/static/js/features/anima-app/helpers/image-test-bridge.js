let ensureImageTestFeatureHandler = null;

export function configureImageTestBridge(handler) {
    ensureImageTestFeatureHandler = typeof handler === 'function' ? handler : null;
}

export function ensureImageTestFeature(...args) {
    if (!ensureImageTestFeatureHandler) {
        throw new Error('image test bridge is not configured');
    }
    return ensureImageTestFeatureHandler(...args);
}
