import { normalizeConfigVisibilityLevel } from './config-field-tiers.js?v=auto-block-swap-20260908-v3';

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
    return 'all';
}

export function presetLibraryCollapsed(fallback = false) {
    const stored = readPreferences().workspaceVersion === 1 ? readPreferences().presetCollapsed : undefined;
    return typeof stored === 'boolean' ? stored : fallback;
}

export function persistPresetLibraryCollapsed(collapsed) {
    writePreferences({ presetCollapsed: Boolean(collapsed), workspaceVersion: 1 });
}

export function persistConfigViewMode(isAll) {
    writePreferences({ viewMode: isAll ? 'all' : 'grouped' });
}

export function preferredConfigBilingual(fallback = false) {
    const safeFallback = typeof fallback === 'boolean' ? fallback : false;
    const stored = readPreferences().bilingual;
    return typeof stored === 'boolean' ? stored : safeFallback;
}

export function persistConfigBilingual(enabled) {
    writePreferences({ bilingual: Boolean(enabled) });
}

export function preferredConfigVisibilityLevel(fallback = 'all') {
    const stored = readPreferences();
    return normalizeConfigVisibilityLevel(stored.fieldScopeVersion === 1 ? stored.visibilityLevel : undefined, fallback);
}

export function persistConfigVisibilityLevel(level) {
    writePreferences({ visibilityLevel: normalizeConfigVisibilityLevel(level), fieldScopeVersion: 1 });
}

export function preferredConfigHideUnavailable(fallback = false) {
    const stored = readPreferences().hideUnavailable;
    return typeof stored === 'boolean' ? stored : Boolean(fallback);
}

export function persistConfigHideUnavailable(enabled) {
    writePreferences({ hideUnavailable: Boolean(enabled) });
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
