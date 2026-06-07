const TOML_GROUP_OPEN_KEY = 'anima.tomlGroupOpen';

export function readTomlGroupState(storage = window.localStorage) {
    try {
        return JSON.parse(storage.getItem(TOML_GROUP_OPEN_KEY) || '{}') || {};
    } catch {
        return {};
    }
}

export function writeTomlGroupState(state, storage = window.localStorage) {
    storage.setItem(TOML_GROUP_OPEN_KEY, JSON.stringify(state));
}
