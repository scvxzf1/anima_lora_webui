/**
 * Lightweight field presentation helpers for provenance badges and dirty summaries.
 * Frontend best-effort sources: config file value vs FORM_UI_DEFAULTS vs draft dirty.
 */

/**
 * @typedef {Object} FieldPresentation
 * @property {string} key
 * @property {unknown} value
 * @property {'config'|'ui_default'|'draft'|'unknown'} source
 * @property {boolean} isUiDefault
 * @property {boolean} isDirty
 * @property {string[]} conflicts
 */

/**
 * @param {string} key
 * @param {object} options
 * @param {Record<string, unknown>} [options.currentConfig]
 * @param {Record<string, unknown>} [options.uiDefaults]
 * @param {boolean} [options.isDirty]
 * @param {unknown} [options.value]
 * @returns {FieldPresentation}
 */
export function buildFieldPresentation(key, options = {}) {
    const currentConfig = options.currentConfig || {};
    const uiDefaults = options.uiDefaults || {};
    const isDirty = Boolean(options.isDirty);
    const hasConfig = Object.prototype.hasOwnProperty.call(currentConfig, key);
    const hasUiDefault = Object.prototype.hasOwnProperty.call(uiDefaults, key);
    let source = 'unknown';
    if (isDirty) source = 'draft';
    else if (hasConfig) source = 'config';
    else if (hasUiDefault) source = 'ui_default';
    const value = options.value !== undefined
        ? options.value
        : (hasConfig ? currentConfig[key] : (hasUiDefault ? uiDefaults[key] : undefined));
    const conflicts = [];
    if (hasConfig && hasUiDefault && currentConfig[key] !== uiDefaults[key] && !isDirty) {
        // not a real conflict — config wins; leave empty for now
    }
    return {
        key: String(key || ''),
        value,
        source,
        isUiDefault: !hasConfig && hasUiDefault,
        isDirty,
        conflicts,
    };
}

export function fieldSourceBadgeLabel(presentation) {
    if (!presentation) return '';
    if (presentation.isDirty) return '已改';
    // Keep labels distinct: color alone is not enough for provenance.
    if (presentation.source === 'ui_default') return '默认';
    if (presentation.source === 'config') return '配置';
    return '';
}

/**
 * @param {Record<string, unknown>} dirtyValues
 * @param {object} [options]
 * @param {number} [options.maxKeys]
 * @returns {string}
 */
export function summarizeDirtyDiff(dirtyValues, options = {}) {
    const maxKeys = Number.isFinite(Number(options.maxKeys)) ? Number(options.maxKeys) : 8;
    const entries = Object.entries(dirtyValues || {});
    if (!entries.length) return '没有待保存的字段修改';
    const shown = entries.slice(0, maxKeys).map(([key, value]) => {
        let text;
        if (value === null || value === undefined) text = '空';
        else if (typeof value === 'object') text = Array.isArray(value) ? `[${value.length}]` : '{…}';
        else {
            text = String(value);
            if (text.length > 24) text = `${text.slice(0, 21)}…`;
        }
        return `${key}=${text}`;
    });
    const more = entries.length > maxKeys ? ` 等共 ${entries.length} 项` : '';
    return `保存前将写入 ${entries.length} 个字段：${shown.join('；')}${more}`;
}
