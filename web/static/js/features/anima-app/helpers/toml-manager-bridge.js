const tomlManagerHandlers = Object.create(null);

function requireTomlManagerHandler(name) {
    const handler = tomlManagerHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[toml-manager] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureTomlManagerBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            tomlManagerHandlers[key] = handler;
        }
    }
}

export function updateConfigPageSummary(...args) { return requireTomlManagerHandler('updateConfigPageSummary')(...args); }
export function setTomlManagerMode(...args) { return requireTomlManagerHandler('setTomlManagerMode')(...args); }
export function switchTomlManagerMode(...args) { return requireTomlManagerHandler('switchTomlManagerMode')(...args); }
export function loadTomlFileList(...args) { return requireTomlManagerHandler('loadTomlFileList')(...args); }
export function loadDefaultTomlFile(...args) { return requireTomlManagerHandler('loadDefaultTomlFile')(...args); }
export function loadOutputRuns(...args) { return requireTomlManagerHandler('loadOutputRuns')(...args); }
