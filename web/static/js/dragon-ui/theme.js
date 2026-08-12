/* Theme management for the Dragon trainer UI.
 * Follows system preference via prefers-color-scheme, with manual override.
 */

const STORAGE_KEY = 'anima_dragon_theme';

function getSystemTheme() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function getStoredTheme() {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        return stored === 'light' || stored === 'dark' ? stored : null;
    } catch {
        return null;
    }
}

export function getActiveTheme() {
    return getStoredTheme() || getSystemTheme();
}

export function getThemePreference() {
    return getStoredTheme() || 'system';
}

export function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.dragonTheme = theme;
}

export function initTheme() {
    const theme = getActiveTheme();
    applyTheme(theme);

    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', (e) => {
        if (!getStoredTheme()) {
            applyTheme(e.matches ? 'dark' : 'light');
        }
    });
}

export function setThemePreference(preference) {
    const normalized = ['light', 'dark'].includes(preference) ? preference : 'system';
    try {
        if (normalized === 'system') localStorage.removeItem(STORAGE_KEY);
        else localStorage.setItem(STORAGE_KEY, normalized);
    } catch { /* ignore */ }
    applyTheme(normalized === 'system' ? getSystemTheme() : normalized);
    return normalized;
}
