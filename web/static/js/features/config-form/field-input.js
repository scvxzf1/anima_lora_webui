/**
 * Config form field input helpers extracted from anima-app chunk 13.
 */
import {
    configDraftValueChanged,
    originalConfigFieldValue,
    updateConfigDraftFromInput,
} from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260711-ir6';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { escapeHtml as sharedEscapeHtml } from '../../shared/format.js?v=module-bootstrap-20260711-ir6';

const configState = getConfigState();

export function setFieldInputValue(key, value) {
    const configFormState = configState.configFormState;
    const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
    if (!input) {
        const original = originalConfigFieldValue(key);
        if (configDraftValueChanged(key, value, original)) {
            configFormState.draftValues.set(key, value);
        } else {
            configFormState.draftValues.delete(key);
        }
        return;
    }
    if (input.type === 'checkbox') {
        input.checked = Boolean(value);
    } else {
        input.value = value ?? '';
    }
    updateConfigDraftFromInput(input);
}

export function escapeHtml(value) {
    return sharedEscapeHtml(value);
}
