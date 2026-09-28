export function stripNetworkArgQuotes(value) {
    const text = String(value || '').trim();
    if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
        return text.slice(1, -1);
    }
    return text;
}

export function parseNetworkArgEntry(raw) {
    const text = String(raw || '').trim();
    const splitAt = text.indexOf('=');
    if (splitAt <= 0) return null;
    const arg = text.slice(0, splitAt).trim();
    if (!arg) return null;
    return {
        arg,
        value: stripNetworkArgQuotes(text.slice(splitAt + 1).trim()),
        raw: text,
    };
}

export function parseBooleanNetworkArg(value, fallback = false) {
    if (typeof value === 'boolean') return value;
    if (value === 1 || value === 0) return Boolean(value);
    const text = String(value ?? '').trim().toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(text)) return true;
    if (['0', 'false', 'no', 'off'].includes(text)) return false;
    return Boolean(fallback);
}

export function coerceNetworkArgValue(value, spec) {
    if (spec.valueType === 'boolean' || spec.valueType === 'booleanInt') {
        return parseBooleanNetworkArg(value, spec.default);
    }
    if (spec.valueType === 'integer') {
        const n = Number(value);
        return Number.isFinite(n) ? Math.trunc(n) : spec.default;
    }
    if (spec.valueType === 'number') {
        const n = Number(value);
        return Number.isFinite(n) ? n : spec.default;
    }
    return String(value ?? spec.default ?? '');
}

export function formatNetworkArgValue(spec, value) {
    if (spec.valueType === 'booleanInt') return parseBooleanNetworkArg(value, spec.default) ? '1' : '0';
    if (spec.valueType === 'boolean') return parseBooleanNetworkArg(value, spec.default) ? 'true' : 'false';
    if (spec.valueType === 'integer') {
        const n = Number(value);
        return Number.isFinite(n) ? String(Math.trunc(n)) : String(spec.default);
    }
    if (spec.valueType === 'number') {
        const n = Number(value);
        return Number.isFinite(n) ? String(n) : String(spec.default);
    }
    return String(value ?? '').trim();
}

export function formatNetworkArg(spec, value) {
    return `${spec.arg}=${formatNetworkArgValue(spec, value)}`;
}
