/**
 * Header language toggle placeholder.
 * Persists zh-CN / en choice and updates the button label only.
 * Full UI i18n can plug into applyLanguage later.
 */

export function createLanguageController({
    storageKey = 'anima_lora_language',
    storage = window.localStorage,
    root = document.documentElement,
} = {}) {
    const SUPPORTED = new Set(['zh-CN', 'en']);

    function normalizeLanguage(value) {
        if (!value) return 'zh-CN';
        const raw = String(value).trim();
        if (SUPPORTED.has(raw)) return raw;
        const lower = raw.toLowerCase();
        if (lower === 'en' || lower.startsWith('en-')) return 'en';
        if (lower === 'zh' || lower.startsWith('zh')) return 'zh-CN';
        return 'zh-CN';
    }

    function currentLanguage() {
        return normalizeLanguage(root.dataset.lang || root.lang || 'zh-CN');
    }

    function storedLanguage() {
        try {
            return normalizeLanguage(storage.getItem(storageKey));
        } catch (_) {
            return null;
        }
    }

    function saveLanguage(language) {
        try {
            storage.setItem(storageKey, language);
        } catch (_) {
            // Ignore storage failures; in-page toggle still works.
        }
    }

    function languageButtonLabel(language) {
        // Show the language you can switch TO (same pattern as theme toggle).
        return language === 'en' ? '中文' : 'English';
    }

    function languageButtonTitle(language) {
        return language === 'en' ? 'Switch to Simplified Chinese' : '切换到 English';
    }

    function applyLanguage(language) {
        const safe = normalizeLanguage(language);
        root.dataset.lang = safe;
        root.lang = safe === 'en' ? 'en' : 'zh-CN';
        const toggle = document.getElementById('language-toggle');
        const label = document.getElementById('language-toggle-text');
        if (toggle) {
            toggle.setAttribute('aria-pressed', String(safe === 'en'));
            toggle.dataset.language = safe;
            toggle.title = languageButtonTitle(safe);
        }
        if (label) label.textContent = languageButtonLabel(safe);
        // Placeholder hook for future full UI translation.
        root.dispatchEvent(new CustomEvent('anima-language-change', {
            detail: { language: safe },
        }));
    }

    function initLanguageToggle() {
        applyLanguage(storedLanguage() || currentLanguage());
        const toggle = document.getElementById('language-toggle');
        if (!toggle) return;
        toggle.addEventListener('click', () => {
            const next = currentLanguage() === 'en' ? 'zh-CN' : 'en';
            applyLanguage(next);
            saveLanguage(next);
        });
    }

    return {
        currentLanguage,
        applyLanguage,
        initLanguageToggle,
    };
}
