/**
 * Resource quick-preset helpers + global model path fill.
 * Extracted from anima-app chunk 06.
 */
import { SELECTIVE_CHECKPOINT_STRENGTH } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260809-nf4-v2';
import { originalConfigFieldValue, readFieldInputValue } from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { handleFormFieldChange } from './form-fields.js?v=module-bootstrap-20260809-nf4-v2';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { openModelConfigPickerDialog } from '../model-configs/index.js?v=module-bootstrap-20260809-nf4-v2';
import { setFieldInputValue } from './field-input.js?v=module-bootstrap-20260809-nf4-v2';
import { modelFamilyFormDefaults } from './model-family-defaults.js?v=module-bootstrap-20260824-zimage-defaults-v1';

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
    const configFormState = configState.configFormState;
    const entries = [
        ['model_family', selected.model_family],
        ['pretrained_model_name_or_path', selected.pretrained_model_name_or_path],
        ['qwen3', selected.qwen3],
        ['vae', selected.vae],
    ];
    const familyDefaults = modelFamilyFormDefaults(selected.model_family);

    let applied = 0;
    for (const [key, value] of entries) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input && key === 'model_family') {
            const original = String(originalConfigFieldValue(key) ?? '');
            const next = String(value ?? '');
            if (next === original) {
                configFormState.draftValues.delete(key);
            } else {
                configFormState.draftValues.set(key, next);
            }
            applied += 1;
            continue;
        }
        if (!input) continue;
        input.value = value;
        applied += 1;
    }
    for (const [key, value] of familyDefaults) {
        setFieldInputValue(key, value);
        applied += 1;
    }
    const familyInput = document.querySelector('#config-form .field-input[data-key="model_family"]');
    handleFormFieldChange({
        target: familyInput || { dataset: { key: 'model_family' }, value: selected.model_family },
    });
    setTomlStatus(
        applied ? 'ok' : 'error',
        applied
            ? `已应用“${selected.name}”${familyDefaults.length ? '并同步模型兼容参数' : ''}，请保存当前配置后再训练`
            : '当前表单没有可覆盖的模型配置字段'
    );
}
