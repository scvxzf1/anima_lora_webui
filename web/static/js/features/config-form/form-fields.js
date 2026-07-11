/**
 * Config form field row / adapter / sample-prompt editor helpers.
 * Extracted from anima-app chunk 14.
 */
import { updateChoiceGuide } from './choice-guide-ui.js?v=module-bootstrap-20260711-ir1';
import { updateStepEstimatePanel } from '../anima-app/chunks/03-parse-network-arg-entry.js?v=module-bootstrap-20260711-ir1';
import { valuesEqual } from '../anima-app/helpers/form-values.js?v=module-bootstrap-20260711-ir1';
import { collectLiveCompatIssues, formatLiveCompatStatus } from './live-compat.js?v=module-bootstrap-20260711-ir1';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { buildFieldPresentation, fieldSourceBadgeLabel } from './field-presentation.js?v=module-bootstrap-20260711-ir1';
import {
    isTruthy,
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
import {
    formatFieldName,
    shouldRenderSelectInput,
} from '../anima-app/helpers/config-field-display.js?v=module-bootstrap-20260711-ir1';
import {
    allowsNegativeNumberField,
    appendSamplePromptRow,
    createHelpContent,
    createSelectInput,
    fieldValueTypeForKey,
    isIntegerNumericField,
    isNumericField,
    updateSamplePromptRemoveButtons,
} from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    blankSamplePromptRow,
    parseSamplePromptRows,
    samplePromptsContentNeedsTextMode,
    serializeSamplePromptsEditor,
} from '../sample-prompts/model.js?v=module-bootstrap-20260711-ir1';
import {
    CONFIG_FORM_INTERNAL_KEYS,
    FIELD_OPTIONS,
    FORM_UI_DEFAULTS,
    help,
} from '../../config/catalog.js?v=module-bootstrap-20260711-ir1';
import { LOSS_WEIGHTING_DEPENDENT_FIELDS } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260711-ir1';
import { applyLossWeightingFieldInputState, collectNetworkArgsFromForm, displayConfigFieldValue, isActiveNetworkArgFieldKey, originalConfigFieldValue, readDoRAAvailable, readFieldInputValue, readLoKrEnabled, readVeRAEnabled, setDoRADraftValue, syncConfigDraftFromForm, updateDoRAFieldState, updateLoKrFieldState, updateLossWeightingFieldState, updateVeRAFieldState } from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260711-ir1';
import { updateTomlDirtyState } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir1';

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

    function createSamplePromptsPathInput(value) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input';
        input.value = value ?? '';
        input.title = '当前 sample_prompts 指向非 .txt 文件，保留为文件路径。';
        return input;
    }

    function createSamplePromptsEditor(value, originalValue = value, touched = false) {
        const editor = document.createElement('div');
        editor.className = 'field-input sample-prompts-editor';
        editor.dataset.originalContent = originalValue ?? '';
        editor.dataset.touched = touched ? '1' : '0';

        const rows = document.createElement('div');
        rows.className = 'sample-prompts-rows';

        editor.appendChild(rows);

        editor.addEventListener('input', (event) => {
            if (event.target?.closest?.('.sample-prompt-row')) {
                markSamplePromptsEditorTouched(editor);
            }
        });
        editor.addEventListener('change', (event) => {
            if (event.target?.closest?.('.sample-prompt-row')) {
                markSamplePromptsEditorTouched(editor);
            }
        });

        renderSamplePromptRows(editor, value ?? '');
        return editor;
    }

    function createSamplePromptAddButton(rowsWrap) {
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small sample-prompts-add-btn';
        addBtn.textContent = '添加行';
        addBtn.addEventListener('click', () => {
            const editor = rowsWrap.closest('.sample-prompts-editor');
            if (editor?.dataset.mode === 'text') {
                const textarea = editor.querySelector('.sample-prompts-textarea');
                if (textarea) {
                    if (textarea.value && !textarea.value.endsWith('\n')) textarea.value += '\n';
                    textarea.focus();
                    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
                    markSamplePromptsEditorTouched(editor);
                    handleFormFieldChange();
                    return;
                }
            }
            appendSamplePromptRow(rowsWrap, blankSamplePromptRow());
            markSamplePromptsEditorTouched(editor);
            handleFormFieldChange();
        });
        return addBtn;
    }

    function createSamplePromptTextModeButton(editor) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-small sample-prompts-add-btn sample-prompts-mode-btn';
        btn.dataset.samplePromptsModeToggle = '1';
        updateSamplePromptModeButtonState(btn, editor);
        btn.addEventListener('click', () => {
            if (editor.dataset.mode === 'text') {
                switchSamplePromptsEditorToTableMode(editor);
            } else {
                switchSamplePromptsEditorToTextMode(editor);
            }
            updateSamplePromptModeButtonState(btn, editor);
            markSamplePromptsEditorTouched(editor);
            handleFormFieldChange();
        });
        return btn;
    }

    function updateSamplePromptModeButtonState(btn, editor) {
        if (!btn || !editor) return;
        const textMode = editor.dataset.mode === 'text';
        btn.textContent = textMode ? '表格模式' : '文本模式';
        btn.title = textMode ? '切回按列编辑提示词' : '保留注释、空行和原始参数格式';
        btn.setAttribute('aria-pressed', String(textMode));
    }

    export function setSamplePromptsEditorContent(editor, content) {
        if (!editor) return;
        editor.dataset.originalContent = content || '';
        editor.dataset.touched = '0';
        renderSamplePromptRows(editor, content || '');
        updateSamplePromptModeButtonState(editor.closest('.field-row')?.querySelector('[data-sample-prompts-mode-toggle]'), editor);
    }

    export function markSamplePromptsEditorTouched(editor) {
        if (editor) editor.dataset.touched = '1';
    }

    function renderSamplePromptRows(editor, content) {
        const rowsWrap = editor.querySelector('.sample-prompts-rows');
        if (!rowsWrap) return;
        rowsWrap.innerHTML = '';
        editor.dataset.mode = samplePromptsContentNeedsTextMode(content) ? 'text' : 'table';
        if (editor.dataset.mode === 'text') {
            const textarea = document.createElement('textarea');
            textarea.className = 'sample-prompts-textarea';
            textarea.value = content || '';
            textarea.spellcheck = false;
            textarea.addEventListener('input', () => markSamplePromptsEditorTouched(editor));
            rowsWrap.appendChild(textarea);
            return;
        }
        const rows = parseSamplePromptRows(content);
        for (const row of rows) {
            appendSamplePromptRow(rowsWrap, row);
        }
        updateSamplePromptRemoveButtons(rowsWrap);
    }

    function switchSamplePromptsEditorToTextMode(editor) {
        if (!editor || editor.dataset.mode === 'text') return;
        const rowsWrap = editor.querySelector('.sample-prompts-rows');
        if (!rowsWrap) return;
        const text = serializeSamplePromptsEditor(editor);
        rowsWrap.innerHTML = '';
        editor.dataset.mode = 'text';
        const textarea = document.createElement('textarea');
        textarea.className = 'sample-prompts-textarea';
        textarea.value = text;
        textarea.spellcheck = false;
        textarea.addEventListener('input', () => markSamplePromptsEditorTouched(editor));
        rowsWrap.appendChild(textarea);
        textarea.focus();
    }

    function switchSamplePromptsEditorToTableMode(editor) {
        if (!editor || editor.dataset.mode !== 'text') return;
        const rowsWrap = editor.querySelector('.sample-prompts-rows');
        if (!rowsWrap) return;
        const text = serializeSamplePromptsEditor(editor);
        rowsWrap.innerHTML = '';
        editor.dataset.mode = 'table';
        for (const row of parseSamplePromptRows(text)) {
            appendSamplePromptRow(rowsWrap, row);
        }
        updateSamplePromptRemoveButtons(rowsWrap);
        rowsWrap.querySelector('[data-sample-prompt-field="prompt"]')?.focus();
    }
