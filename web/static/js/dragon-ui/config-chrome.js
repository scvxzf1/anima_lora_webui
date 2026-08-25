const CONFIG_CHROME_DEFAULTS = Object.freeze({
    helpAlwaysVisible: false,
    tagsAlwaysVisible: false,
});

function settingEnabled(value, fallback) {
    if (value == null) return fallback;
    if (typeof value === 'string') return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
    return Boolean(value);
}

export function normalizeDragonConfigChromeSettings(settings = {}) {
    return {
        helpAlwaysVisible: settingEnabled(
            settings.dragon_config_help_always_visible,
            CONFIG_CHROME_DEFAULTS.helpAlwaysVisible,
        ),
        tagsAlwaysVisible: settingEnabled(
            settings.dragon_config_tags_always_visible,
            CONFIG_CHROME_DEFAULTS.tagsAlwaysVisible,
        ),
    };
}

export function applyDragonConfigChromeSettings(settings = {}) {
    const normalized = normalizeDragonConfigChromeSettings(settings);
    document.documentElement.dataset.dragonConfigHelp = normalized.helpAlwaysVisible ? 'always' : 'contextual';
    document.documentElement.dataset.dragonConfigTags = normalized.tagsAlwaysVisible ? 'always' : 'contextual';
    return normalized;
}
