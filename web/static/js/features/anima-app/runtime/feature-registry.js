export function createFeatureRegistry() {
    return {
        register(name, feature) {
            this[name] = feature;
            return feature;
        },
        get(name) {
            return this[name] || null;
        },
        dispose(name) {
            const feature = this[name];
            feature?.destroy?.();
            delete this[name];
        },
        disposeAll() {
            for (const [name, feature] of Object.entries(this)) {
                if (typeof feature === 'function') continue;
                feature?.destroy?.();
                delete this[name];
            }
        },
    };
}
