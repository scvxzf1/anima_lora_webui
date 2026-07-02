/**
 * UI 缩放控制器
 * 默认比例仍作用于 root；局部覆盖通过容器 zoom 做相对缩放。
 */

export function createUIScaleController({
    root = document.documentElement,
    topLevelFields = [],
    historyDetailFields = [],
} = {}) {
    const DEFAULT_SCALE = 100;
    const MIN_SCALE = 25;
    const MAX_SCALE = 400;

    function clampScale(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return DEFAULT_SCALE;
        return Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.round(num)));
    }

    function resolveBaseScale(settings) {
        return clampScale(settings?.ui_scale ?? settings?.defaults?.ui_scale ?? DEFAULT_SCALE);
    }

    function resolveOverrideScale(settings, key, baseScale) {
        const raw = settings?.[key] ?? settings?.defaults?.[key] ?? '';
        return String(raw ?? '').trim() ? clampScale(raw) : clampScale(baseScale);
    }

    function applyUIScale(scale) {
        const safeScale = clampScale(scale);
        root.style.setProperty('--ui-scale', safeScale / 100);
        root.style.fontSize = `${safeScale}%`;
        return safeScale;
    }

    function applyScopedZoom(element, effectiveScale, baseScale) {
        if (!element) return;
        const base = clampScale(baseScale);
        const effective = clampScale(effectiveScale);
        const zoom = effective / base;
        element.dataset.uiScale = String(effective);
        if (Math.abs(zoom - 1) < 0.001) {
            element.style.removeProperty('zoom');
            element.style.removeProperty('--ui-scope-zoom');
            return;
        }
        element.style.setProperty('zoom', String(zoom));
        element.style.setProperty('--ui-scope-zoom', String(zoom));
    }

    function applyTopLevelScales(settings, baseScale = resolveBaseScale(settings)) {
        for (const field of topLevelFields) {
            const target = document.getElementById(`tab-${field.tab}`);
            const effectiveScale = resolveOverrideScale(settings, field.key, baseScale);
            applyScopedZoom(target, effectiveScale, baseScale);
        }
        return baseScale;
    }

    function applyHistoryDetailScale(settings, activeHistoryDetailTab = 'overview', baseScale = resolveBaseScale(settings)) {
        const field = historyDetailFields.find((item) => item.detailTab === activeHistoryDetailTab);
        const target = document.getElementById('history-detail-content');
        const effectiveScale = field
            ? resolveOverrideScale(settings, field.key, baseScale)
            : baseScale;
        applyScopedZoom(target, effectiveScale, baseScale);
        if (target) target.dataset.historyDetailTab = activeHistoryDetailTab || 'overview';
        return effectiveScale;
    }

    function applyScaleFromSettings(settings, { activeHistoryDetailTab = 'overview' } = {}) {
        if (!settings) return;
        const baseScale = resolveBaseScale(settings);
        applyUIScale(baseScale);
        applyTopLevelScales(settings, baseScale);
        applyHistoryDetailScale(settings, activeHistoryDetailTab, baseScale);
    }

    function initUIScale() {
        applyScaleFromSettings({ ui_scale: DEFAULT_SCALE });
    }

    return {
        applyUIScale,
        applyScopedZoom,
        applyTopLevelScales,
        applyHistoryDetailScale,
        applyScaleFromSettings,
        initUIScale,
        resolveBaseScale,
        resolveOverrideScale,
        clampScale,
        DEFAULT_SCALE,
        MIN_SCALE,
        MAX_SCALE,
    };
}
