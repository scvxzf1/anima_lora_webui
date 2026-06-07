export function createThemeController({
    storageKey,
    getLossChart,
    chartTheme,
    storage = window.localStorage,
    root = document.documentElement,
} = {}) {
    function currentTheme() {
        return root.dataset.theme === 'light' ? 'light' : 'dark';
    }

    function storedTheme() {
        try {
            return storage.getItem(storageKey);
        } catch (_) {
            return null;
        }
    }

    function saveTheme(theme) {
        try {
            storage.setItem(storageKey, theme);
        } catch (_) {
            // 忽略浏览器禁用本地存储的情况，当前页面仍然可以完成切换。
        }
    }

    function applyTheme(theme) {
        const safeTheme = theme === 'light' ? 'light' : 'dark';
        root.dataset.theme = safeTheme;
        const toggle = document.getElementById('theme-toggle');
        const label = document.getElementById('theme-toggle-text');
        if (toggle) {
            const isLight = safeTheme === 'light';
            toggle.setAttribute('aria-pressed', String(isLight));
            toggle.title = isLight ? '切换到深色主题' : '切换到浅色主题';
        }
        if (label) label.textContent = safeTheme === 'light' ? '深色主题' : '浅色主题';
        getLossChart?.()?.setTheme?.(chartTheme());
    }

    function initThemeToggle() {
        applyTheme(storedTheme() || currentTheme());
        const toggle = document.getElementById('theme-toggle');
        if (!toggle) return;
        toggle.addEventListener('click', () => {
            const next = currentTheme() === 'light' ? 'dark' : 'light';
            applyTheme(next);
            saveTheme(next);
        });
    }

    return {
        currentTheme,
        applyTheme,
        initThemeToggle,
    };
}
