/**
 * UI缩放控制器
 * 管理整个WebUI的缩放比例
 */

export function createUIScaleController({
    root = document.documentElement,
} = {}) {
    const DEFAULT_SCALE = 100;
    const MIN_SCALE = 25;
    const MAX_SCALE = 400;

    /**
     * 应用UI缩放
     * @param {number} scale - 缩放百分比 (25-400)
     */
    function applyUIScale(scale) {
        const safeScale = clampScale(scale);
        root.style.setProperty('--ui-scale', safeScale / 100);
        root.style.fontSize = `${safeScale}%`;
    }

    /**
     * 限制缩放值在有效范围内
     * @param {any} value - 输入值
     * @returns {number} 限制后的缩放值
     */
    function clampScale(value) {
        const num = Number(value);
        if (!Number.isFinite(num) || num < MIN_SCALE) return DEFAULT_SCALE;
        return Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.round(num)));
    }

    /**
     * 从全局设置中读取并应用UI缩放
     * @param {object} settings - 全局设置对象
     */
    function applyScaleFromSettings(settings) {
        if (!settings) return;
        const scale = settings.ui_scale ?? settings.defaults?.ui_scale ?? DEFAULT_SCALE;
        applyUIScale(scale);
    }

    /**
     * 初始化UI缩放（在页面加载时调用）
     */
    function initUIScale() {
        // 初始应用默认缩放
        applyUIScale(DEFAULT_SCALE);
    }

    return {
        applyUIScale,
        applyScaleFromSettings,
        initUIScale,
        clampScale,
        DEFAULT_SCALE,
        MIN_SCALE,
        MAX_SCALE,
    };
}
