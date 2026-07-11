/**
 * Resource quick-preset helpers + global model path fill.
 * Extracted from anima-app chunk 06.
 */
import {
    GLOBAL_MODEL_PATH_FIELDS,
} from '../../config/catalog.js?v=module-bootstrap-20260711-ir2';
import { SELECTIVE_CHECKPOINT_STRENGTH } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260711-ir2';
import { getAppShellState } from '../anima-app/helpers/app-shell-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { originalConfigFieldValue, readFieldInputValue } from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260711-ir2';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { handleFormFieldChange } from './form-fields.js?v=module-bootstrap-20260711-ir2';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { loadGlobalSettings, getGlobalModelPathOverrides } from '../anima-app/helpers/global-settings-bridge.js?v=module-bootstrap-20260711-ir2';
import { showAppConfirmDialog } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir2';

const appShellState = getAppShellState();
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
    if (!appShellState.globalSettings && location.protocol !== 'file:') {
        await loadGlobalSettings();
    }
    const overrides = getGlobalModelPathOverrides();
    const entries = GLOBAL_MODEL_PATH_FIELDS
        .map(([key]) => [key, overrides[key]])
        .filter(([, value]) => String(value || '').trim());
    if (!entries.length) {
        setTomlStatus('error', '全局设置里还没有可填写的基础模型路径');
        return;
    }

    const confirmed = await showAppConfirmDialog({
        title: '是否确认覆盖',
        description: '填写全局路径配置',
        message: '将用全局设置里的基础模型路径覆盖当前配置表单中的同名字段。',
        confirmText: '是',
        cancelText: '否',
    });
    if (!confirmed) return;

    let applied = 0;
    for (const [key, value] of entries) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) continue;
        input.value = value;
        applied += 1;
    }
    handleFormFieldChange();
    setTomlStatus(
        applied ? 'ok' : 'error',
        applied ? '已填写全局路径配置，请保存当前配置后再训练' : '当前表单没有可覆盖的基础模型路径字段'
    );
}
