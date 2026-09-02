const KREA2_MODEL_FAMILIES = new Set(['krea2', 'krea2_raw']);

export function normalizeModelFamily(value) {
    return String(value ?? '').trim().toLowerCase();
}

export function isKrea2ModelFamily(value) {
    return KREA2_MODEL_FAMILIES.has(normalizeModelFamily(value));
}
