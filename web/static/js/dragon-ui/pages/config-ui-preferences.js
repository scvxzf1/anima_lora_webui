const STORAGE_KEY = 'anima_dragon_config_ui';

function readPreferences() {
    try {
        const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
        return value && typeof value === 'object' ? value : {};
    } catch {
        return {};
    }
}

function writePreferences(patch) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...readPreferences(), ...patch }));
    } catch {
        // Storage may be disabled; route state still works for the current page.
    }
}

export function preferredConfigSubId(requestedSubId, category) {
    if (requestedSubId || category?.id !== 'training-config') return requestedSubId;
    return readPreferences().viewMode === 'all' ? 'all' : requestedSubId;
}

export function presetLibraryCollapsed(fallback = false) {
    const stored = readPreferences().presetCollapsed;
    return typeof stored === 'boolean' ? stored : fallback;
}

export function persistPresetLibraryCollapsed(collapsed) {
    writePreferences({ presetCollapsed: Boolean(collapsed) });
}

export function persistConfigViewMode(isAll) {
    writePreferences({ viewMode: isAll ? 'all' : 'grouped' });
}

export function bindConfigViewPreference(root) {
    const links = [...root.querySelectorAll('[data-config-view-mode]')];
    const handlers = links.map((link) => {
        const handler = () => persistConfigViewMode(link.dataset.configViewMode === 'all');
        link.addEventListener('click', handler);
        return [link, handler];
    });
    return () => handlers.forEach(([link, handler]) => link.removeEventListener('click', handler));
}
