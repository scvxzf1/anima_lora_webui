export function parseNumberValue(raw, fallback) {
    const trimmed = String(raw).trim();
    if (trimmed === '' && fallback === '') return '';
    if (trimmed === '') return fallback;
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : fallback;
}

export function parseArrayValue(raw) {
    const trimmed = String(raw).trim();
    if (!trimmed) return [];
    try {
        const parsed = JSON.parse(trimmed);
        return Array.isArray(parsed) ? parsed : [parsed];
    } catch {
        return trimmed.split(',').map((item) => item.trim()).filter(Boolean);
    }
}

export function isBooleanLikeValue(value) {
    return value === true || value === false || value === 'true' || value === 'false';
}

export function normalizeBooleanLikeValue(value) {
    return value === true || value === 'true';
}

export function isNumberLikeValue(value) {
    if (typeof value === 'number') return Number.isFinite(value);
    if (typeof value !== 'string') return false;
    const trimmed = value.trim();
    return trimmed !== '' && Number.isFinite(Number(trimmed));
}

export function valuesEqual(a, b) {
    if (isBooleanLikeValue(a) && isBooleanLikeValue(b)) {
        return normalizeBooleanLikeValue(a) === normalizeBooleanLikeValue(b);
    }
    if (isNumberLikeValue(a) && isNumberLikeValue(b)) {
        return Number(a) === Number(b);
    }
    return JSON.stringify(a) === JSON.stringify(b);
}

export function valuesEqualForFieldType(a, b, valueType = '') {
    if (!valuesEqual(a, b)) return false;
    if (valueType === 'number') {
        return typeof a === 'number' && Number.isFinite(a)
            && typeof b === 'number' && Number.isFinite(b);
    }
    if (valueType === 'boolean') {
        return typeof a === 'boolean' && typeof b === 'boolean';
    }
    if (valueType === 'array') {
        return Array.isArray(a) && Array.isArray(b);
    }
    if (valueType === 'string') {
        return typeof a === 'string' && typeof b === 'string';
    }
    return true;
}

export function normalizeMultilineText(value) {
    return String(value || '')
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .join('\n');
}
