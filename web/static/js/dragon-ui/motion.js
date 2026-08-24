/* Dragon motion preference. The persisted setting and OS reduced-motion
 * preference are combined into one runtime state shared by all UI effects.
 */

const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

let preferenceEnabled = true;
let mediaQuery = null;
let mediaChangeHandler = null;

export function resolveDragonMotionEnabled(settings = {}, systemReduced = systemPrefersReducedMotion()) {
    const raw = settings.dragon_motion_enabled ?? settings.defaults?.dragon_motion_enabled ?? true;
    return normalizeBoolean(raw, true) && !systemReduced;
}

export function initDragonMotion(settings = {}) {
    destroyDragonMotion();
    preferenceEnabled = normalizeBoolean(
        settings.dragon_motion_enabled ?? settings.defaults?.dragon_motion_enabled,
        true,
    );
    mediaQuery = typeof window.matchMedia === 'function' ? window.matchMedia(REDUCED_MOTION_QUERY) : null;
    mediaChangeHandler = () => syncMotionState();
    if (typeof mediaQuery?.addEventListener === 'function') mediaQuery.addEventListener('change', mediaChangeHandler);
    else if (typeof mediaQuery?.addListener === 'function') mediaQuery.addListener(mediaChangeHandler);
    syncMotionState();
    return destroyDragonMotion;
}

export function applyDragonMotionSetting(settings = {}) {
    preferenceEnabled = normalizeBoolean(
        settings.dragon_motion_enabled ?? settings.defaults?.dragon_motion_enabled,
        true,
    );
    return syncMotionState();
}

export function isDragonMotionEnabled() {
    return document.documentElement.dataset.dragonMotion !== 'disabled';
}

export function dragonScrollBehavior() {
    return isDragonMotionEnabled() ? 'smooth' : 'auto';
}

export function destroyDragonMotion() {
    if (typeof mediaQuery?.removeEventListener === 'function' && mediaChangeHandler) {
        mediaQuery.removeEventListener('change', mediaChangeHandler);
    } else if (typeof mediaQuery?.removeListener === 'function' && mediaChangeHandler) {
        mediaQuery.removeListener(mediaChangeHandler);
    }
    mediaQuery = null;
    mediaChangeHandler = null;
    delete document.documentElement.dataset.dragonMotion;
}

function syncMotionState() {
    const enabled = preferenceEnabled && !(mediaQuery?.matches ?? systemPrefersReducedMotion());
    document.documentElement.dataset.dragonMotion = enabled ? 'enabled' : 'disabled';
    if (!enabled) {
        document.querySelectorAll('.dragon-reveal').forEach((element) => element.classList.add('dragon-in-view'));
    }
    window.dispatchEvent(new CustomEvent('dragon-motion-change', { detail: { enabled } }));
    return enabled;
}

function systemPrefersReducedMotion() {
    return typeof window.matchMedia === 'function' && window.matchMedia(REDUCED_MOTION_QUERY).matches;
}

function normalizeBoolean(value, fallback) {
    if (value == null || value === '') return fallback;
    if (typeof value === 'boolean') return value;
    return ['1', 'true', 'yes', 'on'].includes(String(value).trim().toLowerCase());
}
