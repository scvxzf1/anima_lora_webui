export function isTruthy(value) {
    return value === true || value === 1 || value === '1' || String(value).toLowerCase() === 'true';
}

export function normalizeLoraAdapterKind(value) {
    const text = String(value ?? '').trim().toLowerCase();
    if (text === 'loha' || text === 'lokr' || text === 'glora' || text === 'vera') return text;
    return 'lora';
}

export function loraAdapterKindFromConfig(config = {}) {
    if (isTruthy(config?.use_glora)) return 'glora';
    if (isTruthy(config?.use_vera)) return 'vera';
    if (isTruthy(config?.use_lokr)) return 'lokr';
    if (isTruthy(config?.use_loha)) return 'loha';
    return 'lora';
}

export function loraAdapterFlagsForKind(kind) {
    const normalized = normalizeLoraAdapterKind(kind);
    return {
        use_glora: normalized === 'glora',
        use_loha: normalized === 'loha',
        use_lokr: normalized === 'lokr',
        use_vera: normalized === 'vera',
    };
}

export function loraAdapterFlagsMatchConfig(kind, config = {}) {
    const flags = loraAdapterFlagsForKind(kind);
    return isTruthy(config?.use_glora) === flags.use_glora
        && isTruthy(config?.use_loha) === flags.use_loha
        && isTruthy(config?.use_lokr) === flags.use_lokr
        && isTruthy(config?.use_vera) === flags.use_vera;
}

export function normalizePrecisionPreference(value) {
    const normalized = String(value || '').trim().toLowerCase();
    if (normalized === 'fp16' || normalized === 'fp32') return normalized;
    return 'bf16';
}

export function precisionPreferenceFromConfig(config = {}) {
    const mixedPrecision = String(config?.mixed_precision || '').trim().toLowerCase();
    if (mixedPrecision === 'no') return 'fp32';
    if (mixedPrecision === 'fp16' || isTruthy(config?.full_fp16)) return 'fp16';
    return 'bf16';
}

export function precisionPreferencePatch(preference, baseConfig = {}) {
    const normalized = normalizePrecisionPreference(preference);
    const patch = {
        mixed_precision: normalized === 'fp32' ? 'no' : normalized,
    };
    if (Object.prototype.hasOwnProperty.call(baseConfig || {}, 'full_fp16') || isTruthy(baseConfig?.full_fp16)) {
        patch.full_fp16 = false;
    }
    if (Object.prototype.hasOwnProperty.call(baseConfig || {}, 'full_bf16') || isTruthy(baseConfig?.full_bf16)) {
        patch.full_bf16 = false;
    }
    return patch;
}
