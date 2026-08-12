const UI_MODE_STORAGE_KEY = 'anima_ui_mode';
const CLASSIC_TARGET_STORAGE_KEY = 'anima_classic_target_tab';

function switchMode(mode, targetTab = '') {
    try {
        localStorage.setItem(UI_MODE_STORAGE_KEY, mode);
        if (targetTab) localStorage.setItem(CLASSIC_TARGET_STORAGE_KEY, targetTab);
    } catch (_) {
        // The query parameter still performs the switch when storage is disabled.
    }
    const url = new URL(window.location.href);
    url.searchParams.set('ui', mode);
    url.hash = '';
    window.location.assign(url.href);
}

export function switchToClassicUI(targetTab = '') {
    switchMode('classic', targetTab);
}

export function switchToAppleUI() {
    switchMode('apple');
}

export function initClassicUiSwitch() {
    document.getElementById('classic-apple-ui-toggle')?.addEventListener('click', switchToAppleUI);
}

export function activateRequestedClassicTab() {
    let target = '';
    try {
        target = localStorage.getItem(CLASSIC_TARGET_STORAGE_KEY) || '';
        localStorage.removeItem(CLASSIC_TARGET_STORAGE_KEY);
    } catch (_) {
        return;
    }
    if (target) document.querySelector(`[data-tab="${CSS.escape(target)}"]`)?.click();
}
