export function froBenchmark(info) {
    const ratio = Number(info.mid_late_vs_early_ratio);
    if (Number.isFinite(ratio)) {
        const pct = Math.abs(ratio - 1) * 100;
        if (ratio > 1.08) return `中后段均值约高 ${pct.toFixed(0)}%`;
        if (ratio < 0.92) return `中后段均值约低 ${pct.toFixed(0)}%`;
        return '中后段与早段接近';
    }
    return '需结合同类权重横向比较';
}

export function paramBenchmark(value) {
    const params = Number(value || 0);
    if (!Number.isFinite(params) || params <= 0) return '暂无规模基准';
    if (params >= 20_000_000) return '相比常见 LoRA 偏大';
    if (params <= 1_000_000) return '相比常见 LoRA 偏小';
    return '常见 LoRA 参数规模';
}

export function blockBenchmark(value) {
    const count = Number(value || 0);
    if (count >= 20) return '覆盖较完整';
    if (count >= 8) return '覆盖中等';
    return '覆盖较窄';
}

export function ratioBenchmark(ratio) {
    if (!Number.isFinite(ratio)) return '早段缺少可比数据';
    if (ratio >= 1.25) return '中后段明显更强';
    if (ratio >= 1.08) return '中后段略强';
    if (ratio <= 0.85) return '早段更活跃';
    return '深度分布均衡';
}

export function formatNumber(value, options = {}) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '-';
    if (options.compact && Math.abs(number) >= 1000) return number.toExponential(1);
    if (Math.abs(number) >= 100) return number.toFixed(1);
    if (Math.abs(number) >= 1) return number.toFixed(3);
    if (number === 0) return '0';
    return number.toExponential(2);
}

export function formatInteger(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.round(number).toLocaleString('zh-CN') : '-';
}

export function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : '-';
}

export function shortComponent(component) {
    return String(component || '')
        .replace('self_attn_', 'SA ')
        .replace('cross_attn_', 'CA ')
        .replace('_proj', '')
        .replace('output', 'out')
        .replace('mlp_layer', 'MLP');
}

export function fileNameFromPath(value) {
    return String(value || '').split(/[\\/]/).filter(Boolean).pop() || '';
}

export function escapeText(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
