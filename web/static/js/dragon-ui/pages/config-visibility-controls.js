import { configVisibilityLabel, normalizeConfigVisibilityLevel } from './config-field-tiers.js?v=auto-block-swap-20260908-v3';

export function bindConfigVisibilityControls(root, {
    state,
    onChange,
    onVisibilityChange,
    onHideUnavailableChange,
} = {}) {
    const menu = root.querySelector('[data-config-visibility-menu]');
    const displayMenu = root.querySelector('[data-config-display-menu]');
    const trigger = root.querySelector('[data-config-visibility-trigger]');
    const label = root.querySelector('[data-config-visibility-label]');
    const options = [...(menu?.querySelectorAll('[data-config-visibility-level]') || [])];
    const unavailableToggle = root.querySelector('[data-config-hide-unavailable]');
    const candidateToggle = root.querySelector('[data-config-show-all-candidates]');
    const candidateToggleLabel = candidateToggle?.querySelector('[data-config-candidate-toggle-label]');
    if (!menu && !unavailableToggle && !candidateToggle) return null;

    const sync = () => {
        const level = normalizeConfigVisibilityLevel(state?.configVisibilityLevel);
        if (label) label.textContent = configVisibilityLabel(level);
        if (trigger) trigger.dataset.active = String(level !== 'all');
        options.forEach((option) => {
            option.setAttribute('aria-checked', String(option.dataset.configVisibilityLevel === level));
        });
        if (unavailableToggle) {
            const active = Boolean(state?.hideUnavailable);
            unavailableToggle.dataset.active = String(active);
            unavailableToggle.setAttribute('aria-pressed', String(active));
        }
        if (candidateToggle) {
            const active = Boolean(state?.showAllCandidates);
            candidateToggle.dataset.active = String(active);
            candidateToggle.setAttribute('aria-pressed', String(active));
            candidateToggle.setAttribute('aria-label', active ? '返回当前适用配置项' : '查看全部候选配置项');
            if (candidateToggleLabel) candidateToggleLabel.textContent = active ? '返回当前适用' : '查看全部候选';
        }
    };

    const optionHandlers = options.map((option) => {
        const handler = () => {
            const level = normalizeConfigVisibilityLevel(option.dataset.configVisibilityLevel);
            if (state) state.configVisibilityLevel = level;
            onVisibilityChange?.(level);
            if (menu) menu.open = false;
            sync();
            onChange?.();
        };
        option.addEventListener('click', handler);
        return [option, handler];
    });
    const unavailableHandler = () => {
        if (state) state.hideUnavailable = !Boolean(state.hideUnavailable);
        onHideUnavailableChange?.(Boolean(state?.hideUnavailable));
        sync();
        onChange?.();
    };
    const candidateHandler = () => {
        if (state) state.showAllCandidates = !Boolean(state.showAllCandidates);
        sync();
        onChange?.();
    };
    const outsideHandler = (event) => {
        if (menu?.open && !menu.contains(event.target)) menu.open = false;
        if (displayMenu?.open && !displayMenu.contains(event.target)) displayMenu.open = false;
    };
    const escapeHandler = (event) => {
        if (event.key === 'Escape' && menu?.open) {
            menu.open = false;
            trigger?.focus();
        } else if (event.key === 'Escape' && displayMenu?.open) {
            displayMenu.open = false;
            displayMenu.querySelector('summary')?.focus();
        }
    };
    unavailableToggle?.addEventListener('click', unavailableHandler);
    candidateToggle?.addEventListener('click', candidateHandler);
    document.addEventListener('click', outsideHandler);
    document.addEventListener('keydown', escapeHandler);
    sync();

    return () => {
        optionHandlers.forEach(([option, handler]) => option.removeEventListener('click', handler));
        unavailableToggle?.removeEventListener('click', unavailableHandler);
        candidateToggle?.removeEventListener('click', candidateHandler);
        document.removeEventListener('click', outsideHandler);
        document.removeEventListener('keydown', escapeHandler);
    };
}
