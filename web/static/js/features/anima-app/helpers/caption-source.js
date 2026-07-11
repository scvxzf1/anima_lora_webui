import { CAPTION_SOURCE_MODE_OPTIONS } from '../../../config/catalog.js?v=module-bootstrap-20260711-ir6';

export function normalizeCaptionSourceMode(value, preferJson = false) {
    const raw = String(value || '').trim().toLowerCase().replace(/-/g, '_');
    const allowed = new Set(CAPTION_SOURCE_MODE_OPTIONS.map((option) => option.value));
    if (allowed.has(raw)) return raw;
    if (raw === 'captions.json' || raw === 'diffpipeforge') return 'captions_json';
    if (raw === '.json' || raw === 'same_stem_json') return 'json';
    if (raw === '.txt' || raw === 'text') return 'txt';
    return preferJson ? 'json' : 'auto';
}

export function captionSourceModeLabel(value) {
    const mode = normalizeCaptionSourceMode(value);
    const option = CAPTION_SOURCE_MODE_OPTIONS.find((item) => item.value === mode);
    return option ? `${option.label} (${option.detail})` : mode;
}
