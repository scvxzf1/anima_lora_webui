/**
 * Adapter-specific config form enablement and cleanup state.
 */
import { FORM_UI_DEFAULTS } from '../../config/catalog.js?v=module-bootstrap-20260809-nf4-v2';
import { LOSS_WEIGHTING_DEPENDENT_FIELDS } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260809-nf4-v2';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    configDraftValueChanged,
    originalConfigFieldValue,
} from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { readLiveLoraAdapterKind } from './form-fields-adapters.js?v=module-bootstrap-20260809-nf4-v2';
import { readFieldInputValue } from './config-value-collector.js?v=module-bootstrap-20260809-nf4-v2';

const configState = getConfigState();

function currentConfigState() {
    return configState.currentConfig || {};
}

export function readLoKrEnabled() {
    return readLiveLoraAdapterKind() === 'lokr';
}

export function updateLoKrFieldState() {
    const enabled = readLoKrEnabled();
    const inputs = [
        document.querySelector('#config-form .field-input[data-key="lokr_factor"]'),
        document.querySelector('#config-form .field-input[data-key="lokr_use_einsum"]'),
        document.querySelector('#config-form .field-input[data-key="lokr_decompose_w2"]'),
        document.querySelector('#config-form .field-input[data-key="lokr_factor_group_size"]'),
        document.querySelector('#config-form .field-input[data-key="lokr_project_chunk_bytes"]'),
    ].filter(Boolean);
    for (const input of inputs) {
        input.disabled = !enabled;
        input.title = enabled ? '' : '启用 LoKr 后生效';
        const row = input.closest('.field-row');
        if (row) row.classList.toggle('field-row-disabled', !enabled);
    }
}

export function readVeRAEnabled() {
    return readLiveLoraAdapterKind() === 'vera';
}

export function readDoRAAvailable() {
    return readLiveLoraAdapterKind() === 'lora';
}

export function setDoRADraftValue(value) {
    const configFormState = configState.configFormState;
    const original = originalConfigFieldValue('dora_wd');
    if (configDraftValueChanged('dora_wd', value, original, { persistDefaultFields: true })) {
        configFormState.draftValues.set('dora_wd', value);
    } else {
        configFormState.draftValues.delete('dora_wd');
    }
}

export function updateDoRAFieldState() {
    const input = document.querySelector('#config-form .field-input[data-key="dora_wd"]');
    if (!input) return;
    const enabled = readDoRAAvailable();
    if (!enabled) {
        input.checked = false;
        setDoRADraftValue(false);
    }
    input.disabled = !enabled;
    input.title = enabled ? '' : 'DoRA 仅支持普通 LoRA；切到 LoHa/LoKr/GLoRA/VeRA 时会自动关闭';
    const row = input.closest('.field-row');
    if (row) row.classList.toggle('field-row-disabled', !enabled);
}

export function updateVeRAFieldState() {
    const enabled = readVeRAEnabled();
    const inputs = [
        document.querySelector('#config-form .field-input[data-key="vera_projection_prng_key"]'),
        document.querySelector('#config-form .field-input[data-key="vera_d_initial"]'),
        document.querySelector('#config-form .field-input[data-key="vera_save_projection"]'),
    ].filter(Boolean);
    for (const input of inputs) {
        input.disabled = !enabled;
        input.title = enabled ? '' : '启用 VeRA 后生效';
        const row = input.closest('.field-row');
        if (row) row.classList.toggle('field-row-disabled', !enabled);
    }
}

export function currentLossWeightingScheme() {
    const configFormState = configState.configFormState;
    const currentConfig = currentConfigState();
    const input = document.querySelector('#config-form .field-input[data-key="weighting_scheme"]');
    if (input) {
        return String(readFieldInputValue(input, originalConfigFieldValue('weighting_scheme')) || 'uniform');
    }
    if (configFormState.draftValues.has('weighting_scheme')) {
        return String(configFormState.draftValues.get('weighting_scheme') || 'uniform');
    }
    return String(currentConfig?.weighting_scheme ?? FORM_UI_DEFAULTS.weighting_scheme ?? 'uniform');
}

export function lossWeightingFieldState(key) {
    const requiredScheme = LOSS_WEIGHTING_DEPENDENT_FIELDS.get(key);
    if (!requiredScheme) return { enabled: true, requiredScheme: '', currentScheme: currentLossWeightingScheme() };
    const currentScheme = currentLossWeightingScheme();
    return {
        enabled: currentScheme === requiredScheme,
        requiredScheme,
        currentScheme,
    };
}

export function lossWeightingDisabledHint(key, state = lossWeightingFieldState(key)) {
    if (!state.requiredScheme) return '';
    return `仅 weighting_scheme = ${state.requiredScheme} 时生效；当前 ${state.currentScheme || 'uniform'}，不生效。`;
}

export function applyLossWeightingFieldInputState(input, key) {
    if (!input || !LOSS_WEIGHTING_DEPENDENT_FIELDS.has(key)) return;
    const state = lossWeightingFieldState(key);
    input.disabled = !state.enabled;
    input.title = state.enabled ? '' : lossWeightingDisabledHint(key, state);
}

export function updateLossWeightingFieldState() {
    for (const key of LOSS_WEIGHTING_DEPENDENT_FIELDS.keys()) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) continue;
        const state = lossWeightingFieldState(key);
        applyLossWeightingFieldInputState(input, key);
        const row = input.closest('.field-row');
        if (!row) continue;
        row.classList.toggle('field-row-disabled', !state.enabled);
        const hint = row.querySelector('.field-state-hint');
        if (hint) {
            hint.textContent = state.enabled ? '' : lossWeightingDisabledHint(key, state);
            hint.hidden = state.enabled;
        }
    }
}
