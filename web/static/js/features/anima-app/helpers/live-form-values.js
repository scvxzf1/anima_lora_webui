function liveFieldInput(key) {
    return document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
}

function liveFieldRawValue(input) {
    return input.type === 'checkbox' ? input.checked : input.value;
}

export function readLiveNumber(key, fallback) {
    const input = liveFieldInput(key);
    if (!input) return Number(fallback) || 0;
    const n = Number(liveFieldRawValue(input));
    return Number.isFinite(n) && n > 0 ? n : (Number(fallback) || 0);
}

export function readNonnegativeLiveNumber(key, fallback = 0) {
    const fallbackNumber = Math.max(0, Number(fallback) || 0);
    const input = liveFieldInput(key);
    if (!input) return fallbackNumber;
    const trimmed = String(liveFieldRawValue(input)).trim();
    if (!trimmed) return fallbackNumber;
    const n = Number(trimmed);
    return Number.isFinite(n) && n >= 0 ? n : fallbackNumber;
}

export function readOptionalLiveNumber(key) {
    const input = liveFieldInput(key);
    if (!input) return null;
    const trimmed = String(liveFieldRawValue(input)).trim();
    if (!trimmed) return null;
    const n = Number(trimmed);
    return Number.isFinite(n) && n > 0 ? n : null;
}
