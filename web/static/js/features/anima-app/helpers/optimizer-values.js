import { parseArrayValue } from './form-values.js?v=module-bootstrap-20260711-ir6';

export function normalizeOptimizerType(value) {
    return String(value ?? '').trim().toLowerCase();
}

export function optimizerArgEntryKey(raw) {
    const text = String(raw || '').trim();
    const splitAt = text.indexOf('=');
    return splitAt > 0 ? text.slice(0, splitAt).trim().toLowerCase() : '';
}

export function optimizerArgEntryValue(raw) {
    const text = String(raw || '').trim();
    const splitAt = text.indexOf('=');
    return splitAt > 0 ? text.slice(splitAt + 1).trim() : '';
}

export function normalizeOptimizerArgArray(value) {
    if (Array.isArray(value)) return value.map((item) => String(item));
    if (typeof value === 'string' && value.trim()) return parseArrayValue(value).map((item) => String(item));
    return [];
}

export function cameBetasNeedPatch(rawBetas) {
    const parts = String(rawBetas || '')
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
    return parts.length === 2;
}

export function normalizeCameOptimizerArgs(args) {
    const result = normalizeOptimizerArgArray(args);
    let betasIndex = -1;
    for (let index = 0; index < result.length; index += 1) {
        if (optimizerArgEntryKey(result[index]) === 'betas') {
            betasIndex = index;
            break;
        }
    }
    if (betasIndex < 0) {
        return result;
    }
    const rawBetas = optimizerArgEntryValue(result[betasIndex]);
    if (cameBetasNeedPatch(rawBetas)) {
        result[betasIndex] = 'betas=0.9,0.999,0.9999';
    }
    return result;
}
