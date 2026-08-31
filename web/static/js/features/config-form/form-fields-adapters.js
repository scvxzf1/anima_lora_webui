/**
 * LoRA adapter / optimizer compatibility helpers for config form fields.
 */
import {
    FORM_UI_DEFAULTS,
    LOKR_SCOPED_FIELD_KEYS,
    VERA_SCOPED_FIELD_KEYS,
} from '../../config/catalog.js?v=module-bootstrap-20260831-release-v1';
import { valuesEqual } from '../anima-app/helpers/form-values.js?v=module-bootstrap-20260831-release-v1';
import {
    loraAdapterFlagsForKind,
    loraAdapterFlagsMatchConfig,
    loraAdapterKindFromConfig,
    normalizeLoraAdapterKind,
    normalizePrecisionPreference,
    precisionPreferenceFromConfig,
    precisionPreferencePatch,
} from '../anima-app/helpers/config-values.js?v=module-bootstrap-20260831-release-v1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { normalizeCameOptimizerArgs, normalizeOptimizerType } from '../anima-app/helpers/optimizer-values.js?v=module-bootstrap-20260831-release-v1';
import { setDoRADraftValue } from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260831-release-v1';
import { parseNetworkArgEntry } from '../anima-app/helpers/network-args.js?v=module-bootstrap-20260831-release-v1';

const configState = getConfigState();

const LOKR_NETWORK_ARG_KEYS = new Set([
    'lokr_use_einsum',
    'lokr_decompose_w2',
    'lokr_full_factor',
    'lokr_allow_legacy_dim',
    'lokr_factor_group_size',
    'lokr_project_chunk_bytes',
]);

function currentConfigState() {
    return configState.currentConfig || {};
}

function markScopedFieldDeleted(values, key, currentConfig) {
    // 后端 _normalize_patch_value 对未知键透传；null 在 _patch_toml_top_level 里不会删除。
    // 通过空字符串 + 专用后端删除键处理，或直接不写。这里用 delete 语义：
    // 仅当原配置存在该键时才请求删除，避免无意义 patch。
    if (Object.prototype.hasOwnProperty.call(currentConfig || {}, key)) {
        values[key] = null;
    } else {
        delete values[key];
    }
}

function stripInactiveLoraAdapterFields(values, kind, currentConfig) {
    const next = { ...values };
    if (kind !== 'lokr') {
        for (const key of LOKR_SCOPED_FIELD_KEYS) {
            markScopedFieldDeleted(next, key, currentConfig);
        }
    }
    if (kind !== 'vera') {
        for (const key of VERA_SCOPED_FIELD_KEYS) {
            markScopedFieldDeleted(next, key, currentConfig);
        }
    }
    if (kind !== 'lora') {
        if (Object.prototype.hasOwnProperty.call(currentConfig || {}, 'dora_wd') || 'dora_wd' in next) {
            next.dora_wd = false;
        }
    }
    // 清理 network_args 里的 LoKr 专属参数，避免普通 LoRA 配置残留。
    if (kind !== 'lokr') {
        const baseArgs = Array.isArray(next.network_args)
            ? next.network_args
            : (Array.isArray(currentConfig?.network_args) ? currentConfig.network_args : null);
        if (baseArgs) {
            const filtered = baseArgs.filter((raw) => {
                const parsed = parseNetworkArgEntry(raw);
                if (!parsed) return true;
                return !LOKR_NETWORK_ARG_KEYS.has(parsed.arg);
            });
            if (!valuesEqual(filtered, baseArgs)) {
                next.network_args = filtered;
            }
        }
    }
    return next;
}

export function applyLoraAdapterDraft(kind) {
    const configFormState = configState.configFormState;
    const currentConfig = currentConfigState();
    const normalized = normalizeLoraAdapterKind(kind);
    const originalKind = loraAdapterKindFromConfig(currentConfig);
    if (normalized === originalKind && loraAdapterFlagsMatchConfig(normalized, currentConfig)) {
        configFormState.draftValues.delete('lora_adapter_kind');
    } else {
        configFormState.draftValues.set('lora_adapter_kind', normalized);
    }
    if (normalized !== 'lora') {
        setDoRADraftValue(false);
    }
    // 切离 LoKr/VeRA 时清掉 draft 中的专属字段，避免后续 collect 写回。
    if (normalized !== 'lokr') {
        for (const key of LOKR_SCOPED_FIELD_KEYS) configFormState.draftValues.delete(key);
    }
    if (normalized !== 'vera') {
        for (const key of VERA_SCOPED_FIELD_KEYS) configFormState.draftValues.delete(key);
    }
    configFormState.draftValues.delete('use_glora');
    configFormState.draftValues.delete('use_loha');
    configFormState.draftValues.delete('use_lokr');
    configFormState.draftValues.delete('use_vera');
}

export function readLiveLoraAdapterKind() {
    const configFormState = configState.configFormState;
    const currentConfig = currentConfigState();
    if (configFormState.draftValues.has('lora_adapter_kind')) {
        return normalizeLoraAdapterKind(configFormState.draftValues.get('lora_adapter_kind'));
    }
    const input = document.querySelector('#config-form .field-input[data-key="lora_adapter_kind"]');
    if (input) {
        return normalizeLoraAdapterKind(input.value);
    }
    return loraAdapterKindFromConfig(currentConfig);
}

export function applyLoraAdapterPatch(values, options = {}) {
    const configFormState = configState.configFormState;
    const currentConfig = currentConfigState();
    const kindChanged = configFormState.draftValues.has('lora_adapter_kind');
    const nextKind = kindChanged
        ? normalizeLoraAdapterKind(configFormState.draftValues.get('lora_adapter_kind'))
        : loraAdapterKindFromConfig({ ...currentConfig, ...values });
    const flags = loraAdapterFlagsForKind(nextKind);

    // 用户改了结构：写互斥 flags，并补默认专属字段。
    if (kindChanged) {
        values.use_glora = flags.use_glora;
        values.use_loha = flags.use_loha;
        values.use_lokr = flags.use_lokr;
        values.use_vera = flags.use_vera;
        if (nextKind !== 'lora') {
            values.dora_wd = false;
        }
        if (flags.use_lokr && !('lokr_factor' in values) && !('lokr_factor' in currentConfig)) {
            values.lokr_factor = FORM_UI_DEFAULTS.lokr_factor;
        }
        if (flags.use_lokr && !('lokr_use_einsum' in values) && !('lokr_use_einsum' in currentConfig)) {
            values.lokr_use_einsum = FORM_UI_DEFAULTS.lokr_use_einsum;
        }
        if (flags.use_lokr && !('lokr_decompose_w2' in values) && !('lokr_decompose_w2' in currentConfig)) {
            values.lokr_decompose_w2 = FORM_UI_DEFAULTS.lokr_decompose_w2;
        }
        if (flags.use_lokr && !('lokr_factor_group_size' in values) && !('lokr_factor_group_size' in currentConfig)) {
            values.lokr_factor_group_size = FORM_UI_DEFAULTS.lokr_factor_group_size;
        }
        if (flags.use_lokr && !('lokr_project_chunk_bytes' in values) && !('lokr_project_chunk_bytes' in currentConfig)) {
            values.lokr_project_chunk_bytes = FORM_UI_DEFAULTS.lokr_project_chunk_bytes;
        }
        if (flags.use_vera && !('vera_projection_prng_key' in values) && !('vera_projection_prng_key' in currentConfig)) {
            values.vera_projection_prng_key = FORM_UI_DEFAULTS.vera_projection_prng_key;
        }
        if (flags.use_vera && !('vera_d_initial' in values) && !('vera_d_initial' in currentConfig)) {
            values.vera_d_initial = FORM_UI_DEFAULTS.vera_d_initial;
        }
        if (flags.use_vera && !('vera_save_projection' in values) && !('vera_save_projection' in currentConfig)) {
            values.vera_save_projection = FORM_UI_DEFAULTS.vera_save_projection;
        }
    }

    // dirty 检测路径不要注入 null 删除键 / dora_wd:false，否则「已修改 0 项」仍弹未保存。
    // 真正保存（stripInactive=true）时再清理无关适配器字段残留。
    if (options.stripInactive || kindChanged) {
        return stripInactiveLoraAdapterFields(values, nextKind, currentConfig);
    }
    return values;
}

export function applyOptimizerCompatibilityPatch(values) {
    const currentConfig = currentConfigState();
    const nextValues = { ...values };
    const optimizerType = 'optimizer_type' in nextValues ? nextValues.optimizer_type : currentConfig.optimizer_type;
    if (normalizeOptimizerType(optimizerType) !== 'came') return nextValues;
    const baseArgs = 'optimizer_args' in nextValues ? nextValues.optimizer_args : currentConfig.optimizer_args;
    const normalizedArgs = normalizeCameOptimizerArgs(baseArgs);
    if (!valuesEqual(normalizedArgs, baseArgs || [])) {
        nextValues.optimizer_args = normalizedArgs;
    }
    return nextValues;
}
