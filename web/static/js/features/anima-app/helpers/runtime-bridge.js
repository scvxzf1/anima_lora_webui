let runtimeApi = null;
let runtimeDom = null;

export function configureRuntimeBridge(runtime) {
    runtimeApi = runtime?.api || null;
    runtimeDom = runtime?.dom || null;
}

function requireRuntimeApi() {
    if (!runtimeApi) {
        throw new Error('runtime bridge api is not configured');
    }
    return runtimeApi;
}

function requireRuntimeDom() {
    if (!runtimeDom) {
        throw new Error('runtime bridge dom is not configured');
    }
    return runtimeDom;
}

export function api(...args) {
    return requireRuntimeApi()(...args);
}

export function datasetPresetApi(...args) {
    return requireRuntimeApi().datasetPresetApi(...args);
}

export function val(...args) {
    return requireRuntimeDom().val(...args);
}

export function populateSelect(...args) {
    return requireRuntimeDom().populateSelect(...args);
}
