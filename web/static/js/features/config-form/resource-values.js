/**
 * Resource quick-preset helpers + global model path fill.
 * Extracted from anima-app chunk 06.
 */
import { SELECTIVE_CHECKPOINT_STRENGTH } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260714-stage-dataset5';
import { originalConfigFieldValue, readFieldInputValue } from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { handleFormFieldChange } from './form-fields.js?v=module-bootstrap-20260714-stage-dataset5';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { openModelConfigPickerDialog } from '../model-configs/index.js?v=model-configs-20260809-1';

const configState = getConfigState();

export function strongerSelectiveCheckpointValue(current, fallback) {
    const currentKey = String(current ?? '').trim() || 'off';
    const fallbackKey = String(fallback ?? '').trim() || 'off';
    const currentStrength = SELECTIVE_CHECKPOINT_STRENGTH.get(currentKey);
    const fallbackStrength = SELECTIVE_CHECKPOINT_STRENGTH.get(fallbackKey);
    if (currentStrength === undefined) return fallbackKey;
    if (fallbackStrength === undefined) return currentKey;
    return currentStrength >= fallbackStrength ? currentKey : fallbackKey;
}

export function resourceQuickCurrentValue(key) {
    const configFormState = configState.configFormState;
    const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
    if (input) {
        return readFieldInputValue(input, originalConfigFieldValue(key));
    }
    if (configFormState.draftValues.has(key)) {
        return configFormState.draftValues.get(key);
    }
    return originalConfigFieldValue(key);
}

export async function fillGlobalModelPathsIntoConfigForm() {
    const selected = await openModelConfigPickerDialog();
    if (!selected) return;
    const entries = [
        ['model_family', selected.model_family],
        ['pretrained_model_name_or_path', selected.pretrained_model_name_or_path],
        ['qwen3', selected.qwen3],
        ['vae', selected.vae],
    ];

    let applied = 0;
    for (const [key, value] of entries) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) continue;
        input.value = value;
        applied += 1;
    }
    const familyInput = document.querySelector('#config-form .field-input[data-key="model_family"]');
    handleFormFieldChange(familyInput ? { target: familyInput } : undefined);
    setTomlStatus(
        applied ? 'ok' : 'error',
        applied
            ? `已应用“${selected.name}”，请保存当前配置后再训练`
            : '当前表单没有可覆盖的模型配置字段'
    );
}
