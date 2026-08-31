/* One-shot context passed from the dataset editor to the tagging workbench. */

export const TAGGING_PREFILL_STORAGE_KEY = 'dragon-tagging-prefill-v1';

const MAX_SELECTED_FILES = 500;

export function writeTaggingPrefill(value = {}) {
    const payload = {
        dataset_file: cleanString(value.dataset_file || value.file, 2048),
        dataset_index: normalizeIndex(value.dataset_index),
        source: normalizeSource(value.source),
        image_file: cleanString(value.image_file, 4096),
        prompt: cleanString(value.prompt, 10000),
        selected_files: Array.isArray(value.selected_files)
            ? value.selected_files.map((item) => cleanString(item, 4096)).filter(Boolean).slice(0, MAX_SELECTED_FILES)
            : [],
    };
    if (!payload.dataset_file) return false;
    try {
        globalThis.sessionStorage?.setItem(TAGGING_PREFILL_STORAGE_KEY, JSON.stringify(payload));
        return true;
    } catch {
        return false;
    }
}

export function consumeTaggingPrefill() {
    let raw = '';
    try {
        raw = globalThis.sessionStorage?.getItem(TAGGING_PREFILL_STORAGE_KEY) || '';
        globalThis.sessionStorage?.removeItem(TAGGING_PREFILL_STORAGE_KEY);
    } catch {
        return {};
    }
    if (!raw) return {};
    try {
        const value = JSON.parse(raw);
        if (!value || typeof value !== 'object') return {};
        return {
            dataset_file: cleanString(value.dataset_file || value.file, 2048),
            dataset_index: normalizeIndex(value.dataset_index),
            source: normalizeSource(value.source),
            image_file: cleanString(value.image_file, 4096),
            prompt: cleanString(value.prompt, 10000),
            selected_files: Array.isArray(value.selected_files)
                ? value.selected_files.map((item) => cleanString(item, 4096)).filter(Boolean).slice(0, MAX_SELECTED_FILES)
                : [],
        };
    } catch {
        return {};
    }
}

function cleanString(value, maxLength) {
    return String(value ?? '').trim().slice(0, maxLength);
}

function normalizeIndex(value) {
    const index = Number(value);
    return Number.isInteger(index) && index >= 0 ? index : 0;
}

function normalizeSource(value) {
    return String(value || '').toLowerCase() === 'training' ? 'training' : 'source';
}
