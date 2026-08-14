/* Dragon UI scale application. Global scale changes the root; page overrides use zoom. */

const MIN_SCALE = 25;
const MAX_SCALE = 400;
const DEFAULT_SCALE = 100;

const PAGE_SCALE_KEYS = Object.freeze({
    config: 'ui_scale_config',
    'dataset-editor': 'ui_scale_datasets',
    'live-training': 'ui_scale_training',
    'weight-analysis': 'ui_scale_weight_analysis',
    'image-test': 'ui_scale_image_test',
    'global-settings': 'ui_scale_settings',
    'model-config': 'ui_scale_model_config',
    environment: 'ui_scale_environment',
});

export function clampUIScale(value, fallback = DEFAULT_SCALE) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.round(numeric)));
}

export function resolveBaseScale(settings = {}) {
    return clampUIScale(settings.ui_scale ?? settings.defaults?.ui_scale ?? DEFAULT_SCALE);
}

export function resolvePageScale(settings = {}, pageType = '', baseScale = resolveBaseScale(settings)) {
    const key = PAGE_SCALE_KEYS[pageType];
    if (!key) return baseScale;
    const raw = settings[key] ?? settings.defaults?.[key] ?? '';
    return String(raw ?? '').trim() ? clampUIScale(raw, baseScale) : baseScale;
}

export function applyDragonUIScale(settings = {}, pageType = currentPageType()) {
    const baseScale = resolveBaseScale(settings);
    const pageScale = resolvePageScale(settings, pageType, baseScale);
    document.documentElement.style.setProperty('--dragon-user-scale', String(baseScale / 100));
    const mount = document.getElementById('dragon-main');
    if (mount) {
        const zoom = pageScale / DEFAULT_SCALE;
        mount.dataset.uiScale = String(pageScale);
        if (Math.abs(zoom - 1) < 0.001) mount.style.removeProperty('zoom');
        else mount.style.setProperty('zoom', String(zoom));
    }
    return { baseScale, pageScale };
}

export async function loadAndApplyDragonUIScale(api, pageType = currentPageType()) {
    if (typeof api !== 'function') return null;
    try {
        const settings = await api('/api/settings/global');
        if (settings?.ok === false) return null;
        applyDragonUIScale(settings || {}, pageType);
        return settings;
    } catch {
        return null;
    }
}

function currentPageType() {
    const hash = window.location.hash.slice(1).split('/')[0];
    if (hash === 'config') return 'config';
    if (hash === 'history') return 'history';
    return hash || 'dashboard';
}
