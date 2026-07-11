/**
 * LoRA adapter / optimizer compatibility helpers for config form fields.
 */
import { valuesEqual } from '../anima-app/helpers/form-values.js?v=module-bootstrap-20260711-ir1';
import {
    loraAdapterFlagsForKind,
    loraAdapterFlagsMatchConfig,
    loraAdapterKindFromConfig,
    normalizeLoraAdapterKind,
    normalizePrecisionPreference,
    precisionPreferenceFromConfig,
    precisionPreferencePatch,
} from '../anima-app/helpers/config-values.js?v=module-bootstrap-20260711-ir1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { normalizeCameOptimizerArgs, normalizeOptimizerType } from '../anima-app/helpers/optimizer-values.js?v=module-bootstrap-20260711-ir1';
import { setDoRADraftValue } from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260711-ir1';

const configState = getConfigState();

function currentConfigState() {
    return configState.currentConfig || {};
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
        return normalizeLoraAdapterKind(readFieldInputValue(input, loraAdapterKindFromConfig(currentConfig)));
    }
    return loraAdapterKindFromConfig(currentConfig);
}

export function applyLoraAdapterPatch(values) {
    const configFormState = configState.configFormState;
    const currentConfig = currentConfigState();
    if (!configFormState.draftValues.has('lora_adapter_kind')) return values;
    const nextKind = normalizeLoraAdapterKind(configFormState.draftValues.get('lora_adapter_kind'));
    const flags = loraAdapterFlagsForKind(nextKind);
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
