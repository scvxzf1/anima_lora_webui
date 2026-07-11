/**
 * Config form field rows, live change handling, and field input factory.
 */
import { updateChoiceGuide } from './choice-guide-ui.js?v=module-bootstrap-20260711-ir6';
import { updateStepEstimatePanel } from './step-estimate.js?v=module-bootstrap-20260711-ir6';
import { valuesEqual } from '../anima-app/helpers/form-values.js?v=module-bootstrap-20260711-ir6';
import { collectLiveCompatIssues, formatLiveCompatStatus } from './live-compat.js?v=module-bootstrap-20260711-ir6';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { buildFieldPresentation, fieldSourceBadgeLabel } from './field-presentation.js?v=module-bootstrap-20260711-ir6';
import { isTruthy } from '../anima-app/helpers/config-values.js?v=module-bootstrap-20260711-ir6';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    formatFieldName,
    shouldRenderSelectInput,
} from '../anima-app/helpers/config-field-display.js?v=module-bootstrap-20260711-ir6';
import {
    allowsNegativeNumberField,
    createHelpContent,
    createSelectInput,
    fieldValueTypeForKey,
    isIntegerNumericField,
    isNumericField,
} from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    CONFIG_FORM_INTERNAL_KEYS,
    FIELD_OPTIONS,
    FORM_UI_DEFAULTS,
    help,
} from '../../config/catalog.js?v=module-bootstrap-20260711-ir6';
import { LOSS_WEIGHTING_DEPENDENT_FIELDS } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260711-ir6';
import {
    applyLossWeightingFieldInputState,
    collectNetworkArgsFromForm,
    displayConfigFieldValue,
    isActiveNetworkArgFieldKey,
    originalConfigFieldValue,
    readDoRAAvailable,
    readFieldInputValue,
    readLoKrEnabled,
    readVeRAEnabled,
    syncConfigDraftFromForm,
    updateDoRAFieldState,
    updateLoKrFieldState,
    updateLossWeightingFieldState,
    updateVeRAFieldState,
} from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260711-ir6';
import { updateTomlDirtyState } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    createSamplePromptAddButton,
    createSamplePromptTextModeButton,
    createSamplePromptsEditor,
    createSamplePromptsPathInput,
} from './form-fields-sample.js?v=module-bootstrap-20260711-ir6';

let updateNoDatasetRegularizationModePanelCallback = () => {};
const configState = getConfigState();

function currentConfigState() {
    return configState.currentConfig || {};
}

export function configureNoDatasetRegularizationModePanelUpdater(updater) {
    updateNoDatasetRegularizationModePanelCallback = typeof updater === 'function' ? updater : () => {};
}

const PREPROCESS_MEMORY_PROFILE_VALUES = {
    auto: { preprocess_vae_cache_batch_size: 'auto', preprocess_text_cache_batch_size: 'auto' },
    low_vram: { preprocess_vae_cache_batch_size: 1, preprocess_text_cache_batch_size: 4 },
    balanced: { preprocess_vae_cache_batch_size: 2, preprocess_text_cache_batch_size: 8 },
    speed: { preprocess_vae_cache_batch_size: 4, preprocess_text_cache_batch_size: 16 },
};

function configFieldInputByKey(key) {
    return [...document.querySelectorAll('#config-form .field-input[data-key]')]
        .find((input) => input.dataset.key === key);
}

function setConfigFieldInputValue(key, value) {
    const input = configFieldInputByKey(key);
    if (!input) return;
    if (input.type === 'checkbox') {
        input.checked = value === true || value === 'true';
        return;
    }
    input.value = value ?? '';
}

function applyPreprocessMemoryProfileSelection(event) {
    const target = event?.target;
    if (target?.dataset?.key !== 'preprocess_memory_profile') return;
    const profile = String(target.value || 'auto').trim().replace(/-/g, '_');
    const values = PREPROCESS_MEMORY_PROFILE_VALUES[profile] || PREPROCESS_MEMORY_PROFILE_VALUES.auto;
    for (const [key, value] of Object.entries(values)) {
        setConfigFieldInputValue(key, value);
    }
}

export function createFieldRow(key, value) {
    const configFormState = configState.configFormState;
    const row = document.createElement('div');
    row.className = 'field-row';
    row.dataset.key = key;
    if (key === 'sample_prompts') row.classList.add('field-row-sample-prompts');
    const hasDraftValue = configFormState.draftValues.has(key);
    const originalValue = originalConfigFieldValue(key);
    const displayValue = displayConfigFieldValue(key, value);

    const main = document.createElement('div');
    main.className = 'field-main';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'field-name';
    nameSpan.textContent = formatFieldName(key);
    nameSpan.title = `${key}，点击聚焦输入项`;
    const presentation = buildFieldPresentation(key, {
        currentConfig: currentConfigState(),
        uiDefaults: FORM_UI_DEFAULTS,
        isDirty: hasDraftValue,
        value: displayValue,
    });
    const badgeLabel = fieldSourceBadgeLabel(presentation);
    if (badgeLabel) {
        const badge = document.createElement('span');
        badge.className = 'field-source-badge';
        badge.dataset.source = presentation.source;
        badge.dataset.key = key;
        if (presentation.isDirty) badge.classList.add('is-dirty');
        badge.textContent = badgeLabel;
        badge.title = `字段来源：${presentation.source}`;
        nameSpan.appendChild(badge);
    }

    const input = createFieldInput(key, displayValue, { originalValue, hasDraftValue });
    input.dataset.key = key;
    input.dataset.valueType = fieldValueTypeForKey(key, originalValue);
    input.addEventListener('input', handleFormFieldChange);
    input.addEventListener('change', handleFormFieldChange);
    nameSpan.addEventListener('click', () => focusConfigFieldInput(input));

    if (key === 'sample_prompts' && configState.samplePromptsMode !== 'path') {
        const labelStack = document.createElement('div');
        labelStack.className = 'field-label-stack';
        labelStack.appendChild(nameSpan);

        const rowsWrap = input.querySelector('.sample-prompts-rows');
        if (rowsWrap) {
            const labelActions = document.createElement('div');
            labelActions.className = 'field-label-actions';
            labelActions.appendChild(createSamplePromptAddButton(rowsWrap));
            labelActions.appendChild(createSamplePromptTextModeButton(input));
            labelStack.appendChild(labelActions);
        }
        main.appendChild(labelStack);
    } else {
        main.appendChild(nameSpan);
    }
    main.appendChild(input);

    const btn = document.createElement('button');
    btn.className = 'info-toggle';
    btn.textContent = '?';
    btn.type = 'button';
    btn.title = '查看填写建议、好处、代价、风险和推荐';
    btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        const helpDiv = row.querySelector('.field-help');
        if (helpDiv) helpDiv.classList.toggle('visible');
    });
    main.appendChild(btn);
    row.appendChild(main);

    if (LOSS_WEIGHTING_DEPENDENT_FIELDS.has(key)) {
        const stateHint = document.createElement('p');
        stateHint.className = 'field-state-hint';
        stateHint.hidden = true;
        row.appendChild(stateHint);
    }

    const helpDiv = document.createElement('div');
    helpDiv.className = 'field-help';
    helpDiv.appendChild(createHelpContent(key, value));
    row.appendChild(helpDiv);

    return row;
}

function focusConfigFieldInput(input) {
    if (!input) return;
    const target = input.matches?.('input, textarea, select, button')
        ? input
        : input.querySelector?.('input, textarea, select, button');
    if (!target || target.disabled) return;
    target.focus();
    const selectableTypes = new Set(['email', 'password', 'search', 'tel', 'text', 'url']);
    if (
        typeof target.select === 'function'
        && (target.tagName === 'TEXTAREA' || selectableTypes.has(target.type || ''))
    ) {
        try {
            target.select();
        } catch {
            // 少数输入类型可聚焦但不能选中文本，保留聚焦结果即可。
        }
    }
}

export function handleFormFieldChange(event) {
    applyPreprocessMemoryProfileSelection(event);
    syncConfigDraftFromForm();
    updateTomlDirtyState();
    updateStepEstimatePanel();
    updateLoKrFieldState();
    updateVeRAFieldState();
    updateDoRAFieldState();
    updateLossWeightingFieldState();
    updateNoDatasetRegularizationModePanelCallback();
    updateChoiceGuideFromLiveForm();
    updateLiveCompatWarningsFromForm();
}

function updateLiveCompatWarningsFromForm() {
    const liveConfig = liveConfigFromForm();
    const issues = collectLiveCompatIssues(liveConfig);
    const statusEl = document.getElementById('toml-status');
    const previous = String(statusEl?.textContent || '');
    const wasLiveCompat = previous.includes('live 兼容');
    if (!issues.length) {
        // Clear only our live-compat sticky message; leave other status text alone.
        if (wasLiveCompat) setTomlStatus('', '');
        return;
    }
    const message = formatLiveCompatStatus(issues);
    const severity = issues.some((item) => item.severity === 'error') ? 'error' : 'pending';
    setTomlStatus(severity, message);
}

function updateChoiceGuideFromLiveForm() {
    const currentConfig = currentConfigState();
    if (!currentConfig || Object.keys(currentConfig).length === 0) return;
    updateChoiceGuide(liveConfigFromForm());
}

function liveConfigFromForm() {
    const configFormState = configState.configFormState;
    const currentConfig = currentConfigState();
    syncConfigDraftFromForm();
    const rawNetworkArgsChanged = configFormState.draftValues.has('network_args');
    const liveConfig = { ...(currentConfig || {}) };
    for (const [key, next] of configFormState.draftValues.entries()) {
        if (!key) continue;
        if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
        if (isActiveNetworkArgFieldKey(key)) continue;
        if (key === 'lora_adapter_kind') {
            const nextKind = normalizeLoraAdapterKind(next);
            Object.assign(liveConfig, loraAdapterFlagsForKind(nextKind));
            if (nextKind !== 'lora') liveConfig.dora_wd = false;
            continue;
        }
        if (key === 'precision_preference') {
            Object.assign(liveConfig, precisionPreferencePatch(next, currentConfig));
            continue;
        }
        liveConfig[key] = next;
    }
    liveConfig.network_args = collectNetworkArgsFromForm(liveConfig, { skipUnchangedInputs: rawNetworkArgsChanged }).networkArgs;
    return liveConfig;
}

function createFieldInput(key, value, options = {}) {
    if (key === 'sample_prompts') {
        if (configState.samplePromptsMode === 'path') {
            return createSamplePromptsPathInput(value);
        }
        return createSamplePromptsEditor(value, options.originalValue, options.hasDraftValue);
    }
    const fieldOptions = FIELD_OPTIONS[key];
    if (shouldRenderSelectInput(key, value)) {
        return createSelectInput(key, value, fieldOptions);
    }

    let input;
    const typeSource = options.originalValue ?? value;
    if (typeof typeSource === 'boolean') {
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = value === true || value === 'true';
    } else {
        input = document.createElement('input');
        input.type = isNumericField(key, typeSource) ? 'number' : 'text';
        if (input.type === 'number') {
            input.step = isIntegerNumericField(key, typeSource) ? '1' : '0.01';
            if (!allowsNegativeNumberField(key)) input.min = '0';
        }
        input.value = Array.isArray(value) ? JSON.stringify(value) : (value ?? '');
    }
    input.className = 'field-input';
    if (key === 'dora_wd') {
        const enabled = readDoRAAvailable();
        input.disabled = !enabled;
        input.title = enabled ? '' : 'DoRA 仅支持普通 LoRA；切到 LoHa/LoKr/GLoRA/VeRA 时会自动关闭';
        if (!enabled) input.checked = false;
    }
    if (key === 'lokr_factor' || key === 'lokr_use_einsum' || key === 'lokr_decompose_w2' || key === 'lokr_factor_group_size' || key === 'lokr_project_chunk_bytes') {
        input.disabled = !readLoKrEnabled();
        input.title = input.disabled ? '启用 LoKr 后生效' : '';
    }
    if (key === 'vera_projection_prng_key' || key === 'vera_d_initial' || key === 'vera_save_projection') {
        input.disabled = !readVeRAEnabled();
        input.title = input.disabled ? '启用 VeRA 后生效' : '';
    }
    applyLossWeightingFieldInputState(input, key);
    return input;
}
