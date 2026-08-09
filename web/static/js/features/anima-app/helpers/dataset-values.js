import {
    DATASET_SETTING_KEYS,
    DEFAULT_NL_TAG_MIX,
} from '../../../config/catalog.js?v=module-bootstrap-20260809-nf4-v2';
import { normalizeCaptionSourceMode } from './caption-source.js?v=module-bootstrap-20260809-nf4-v2';

export function normalizeNlTagMix(raw) {
    const source = raw && typeof raw === 'object' ? raw : {};
    const enabled = source.enabled === true || source.enabled === 'true';
    const parsedRatio = Number(source.tag_ratio ?? source.tagRatio ?? DEFAULT_NL_TAG_MIX.tag_ratio);
    const tagRatio = Number.isFinite(parsedRatio)
        ? Math.min(1, Math.max(0, parsedRatio > 1 ? parsedRatio / 100 : parsedRatio))
        : DEFAULT_NL_TAG_MIX.tag_ratio;
    return {
        enabled,
        tag_ratio: tagRatio,
    };
}

export function nlTagMixSummary(mix) {
    const normalized = normalizeNlTagMix(mix);
    const tagPercent = Math.round(normalized.tag_ratio * 100);
    return `${tagPercent}% tag + ${100 - tagPercent}% nl`;
}

export function normalizeTriggerClone(raw) {
    const source = raw && typeof raw === 'object' ? raw : {};
    return {
        enabled: source.enabled === true || source.enabled === 'true',
        prompt: String(source.prompt || '').trim(),
        num_repeats: Math.max(1, Number.parseInt(source.num_repeats || 1, 10) || 1),
    };
}

export function normalizeDatasetEditorRows(rows) {
    return (rows || [])
        .filter((row) => row && typeof row === 'object')
        .map((row) => ({
            source_dir: String(row.source_dir || row.source_image_dir || ''),
            image_dir: String(row.image_dir || row.resized_image_dir || ''),
            cache_dir: String(row.cache_dir || row.lora_cache_dir || ''),
            num_repeats: Math.max(1, Number.parseInt(row.num_repeats || 1, 10) || 1),
            recursive: row.recursive !== false && row.recursive !== 'false',
            path_pattern: String(row.path_pattern || '*').trim() || '*',
            is_reg: row.is_reg === true,
            nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix),
            trigger_clone: normalizeTriggerClone(row.trigger_clone),
            settings: normalizeDatasetRowSettings(row),
        }));
}

export function datasetRowsForPayload(rows) {
    return normalizeDatasetEditorRows(rows).map((row) => ({
        source_dir: row.source_dir,
        image_dir: row.image_dir,
        cache_dir: row.cache_dir,
        num_repeats: row.num_repeats,
        recursive: row.recursive,
        path_pattern: row.path_pattern,
        is_reg: row.is_reg,
        nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix),
        trigger_clone: normalizeTriggerClone(row.trigger_clone),
        settings: normalizeDatasetDefaults(row.settings || {}),
    }));
}

export function normalizeDatasetRowSettings(row) {
    if (row.settings && typeof row.settings === 'object') {
        return normalizeDatasetDefaults(row.settings);
    }
    if ([...DATASET_SETTING_KEYS].some((key) => key in row)) {
        return normalizeDatasetDefaults(row);
    }
    return {};
}

export function normalizeDatasetDefaults(defaults) {
    const raw = defaults && typeof defaults === 'object' ? defaults : {};
    const preferJson = raw.prefer_json_caption === true || raw.prefer_json_caption === 'true';
    const captionSourceMode = normalizeCaptionSourceMode(raw.caption_source_mode, preferJson);
    const validationSeed = Number.parseInt(raw.validation_seed ?? 42, 10);
    const priorLossWeight = Number(raw.prior_loss_weight ?? 1.0);
    return {
        resolution: Math.max(1, Number.parseInt(raw.resolution || 1024, 10) || 1024),
        prior_loss_weight: Number.isFinite(priorLossWeight) ? Math.max(0, priorLossWeight) : 1.0,
        enable_bucket: raw.enable_bucket !== false && raw.enable_bucket !== 'false',
        min_bucket_reso: Math.max(1, Number.parseInt(raw.min_bucket_reso || 256, 10) || 256),
        max_bucket_reso: Math.max(1, Number.parseInt(raw.max_bucket_reso || 1024, 10) || 1024),
        bucket_reso_steps: Math.max(1, Number.parseInt(raw.bucket_reso_steps || 64, 10) || 64),
        bucket_no_upscale: raw.bucket_no_upscale === true || raw.bucket_no_upscale === 'true',
        validation_split: Math.max(0, Number(raw.validation_split ?? 0) || 0),
        validation_split_num: Math.max(0, Number.parseInt(raw.validation_split_num || 0, 10) || 0),
        validation_seed: Number.isFinite(validationSeed) ? Math.max(0, validationSeed) : 42,
        caption_extension: String(raw.caption_extension || '.txt'),
        keep_tokens: Math.max(0, Number.parseInt(raw.keep_tokens ?? 3, 10) || 0),
        prefer_json_caption: preferJson,
        caption_source_mode: captionSourceMode,
    };
}
