/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.loraAdapterKindFromConfig = function loraAdapterKindFromConfig(config = currentConfig) {
        if (isTruthy(config?.use_glora)) return 'glora';
        if (isTruthy(config?.use_vera)) return 'vera';
        if (isTruthy(config?.use_lokr)) return 'lokr';
        if (isTruthy(config?.use_loha)) return 'loha';
        return 'lora';
    }

    globalThis.loraAdapterFlagsForKind = function loraAdapterFlagsForKind(kind) {
        const normalized = normalizeLoraAdapterKind(kind);
        return {
            use_glora: normalized === 'glora',
            use_loha: normalized === 'loha',
            use_lokr: normalized === 'lokr',
            use_vera: normalized === 'vera',
        };
    }

    globalThis.applyLoraAdapterDraft = function applyLoraAdapterDraft(kind) {
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

    globalThis.readLiveLoraAdapterKind = function readLiveLoraAdapterKind() {
        if (configFormState.draftValues.has('lora_adapter_kind')) {
            return normalizeLoraAdapterKind(configFormState.draftValues.get('lora_adapter_kind'));
        }
        const input = document.querySelector('#config-form .field-input[data-key="lora_adapter_kind"]');
        if (input) {
            return normalizeLoraAdapterKind(readFieldInputValue(input, loraAdapterKindFromConfig(currentConfig)));
        }
        return loraAdapterKindFromConfig(currentConfig);
    }

    globalThis.applyLoraAdapterPatch = function applyLoraAdapterPatch(values) {
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

    globalThis.normalizeOptimizerType = function normalizeOptimizerType(value) {
        return String(value ?? '').trim().toLowerCase();
    }

    globalThis.optimizerArgEntryKey = function optimizerArgEntryKey(raw) {
        const text = String(raw || '').trim();
        const splitAt = text.indexOf('=');
        return splitAt > 0 ? text.slice(0, splitAt).trim().toLowerCase() : '';
    }

    globalThis.optimizerArgEntryValue = function optimizerArgEntryValue(raw) {
        const text = String(raw || '').trim();
        const splitAt = text.indexOf('=');
        return splitAt > 0 ? text.slice(splitAt + 1).trim() : '';
    }

    globalThis.normalizeOptimizerArgArray = function normalizeOptimizerArgArray(value) {
        if (Array.isArray(value)) return value.map((item) => String(item));
        if (typeof value === 'string' && value.trim()) return parseArrayValue(value).map((item) => String(item));
        return [];
    }

    globalThis.cameBetasNeedPatch = function cameBetasNeedPatch(rawBetas) {
        const parts = String(rawBetas || '')
            .split(',')
            .map((item) => item.trim())
            .filter(Boolean);
        return parts.length === 2;
    }

    globalThis.normalizeCameOptimizerArgs = function normalizeCameOptimizerArgs(args) {
        const result = normalizeOptimizerArgArray(args);
        let betasIndex = -1;
        for (let index = 0; index < result.length; index += 1) {
            if (optimizerArgEntryKey(result[index]) === 'betas') {
                betasIndex = index;
                break;
            }
        }
        if (betasIndex < 0) {
            return result;
        }
        const rawBetas = optimizerArgEntryValue(result[betasIndex]);
        if (cameBetasNeedPatch(rawBetas)) {
            result[betasIndex] = 'betas=0.9,0.999,0.9999';
        }
        return result;
    }

    globalThis.applyOptimizerCompatibilityPatch = function applyOptimizerCompatibilityPatch(values) {
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

    globalThis.loraAdapterFlagsMatchConfig = function loraAdapterFlagsMatchConfig(kind, config = currentConfig) {
        const flags = loraAdapterFlagsForKind(kind);
        return isTruthy(config?.use_glora) === flags.use_glora
            && isTruthy(config?.use_loha) === flags.use_loha
            && isTruthy(config?.use_lokr) === flags.use_lokr
            && isTruthy(config?.use_vera) === flags.use_vera;
    }

    globalThis.compactList = function compactList(items) {
        return items.filter((item) => item !== undefined && item !== null && String(item).trim() !== '');
    }

    globalThis.valueDetail = function valueDetail(key, value) {
        if (value === undefined || value === null || value === '') return '';
        return `${FIELD_LABEL_ZH[key] || key}: ${formatChoiceValue(value)}`;
    }

    globalThis.flagDetail = function flagDetail(key, label, value) {
        if (value === undefined || value === null || value === '') return '';
        return `${label}: ${isTruthy(value) ? '开启' : '关闭'}`;
    }

    globalThis.formatChoiceValue = function formatChoiceValue(value) {
        if (Array.isArray(value)) return value.join(', ');
        if (typeof value === 'boolean') return value ? 'true' : 'false';
        return String(value);
    }

    globalThis.createFieldRow = function createFieldRow(key, value) {
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
        nameSpan.title = key;

        const input = createFieldInput(key, displayValue, { originalValue, hasDraftValue });
        input.dataset.key = key;
        input.dataset.valueType = fieldValueTypeForKey(key, originalValue);
        input.addEventListener('input', handleFormFieldChange);
        input.addEventListener('change', handleFormFieldChange);

        if (key === 'sample_prompts' && samplePromptsMode !== 'path') {
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

    globalThis.handleFormFieldChange = function handleFormFieldChange() {
        syncConfigDraftFromForm();
        updateTomlDirtyState();
        updateStepEstimatePanel();
        updateLoKrFieldState();
        updateVeRAFieldState();
        updateDoRAFieldState();
        updateLossWeightingFieldState();
        updateChoiceGuideFromLiveForm();
    }

    globalThis.updateChoiceGuideFromLiveForm = function updateChoiceGuideFromLiveForm() {
        if (!currentConfig || Object.keys(currentConfig).length === 0) return;
        updateChoiceGuide(liveConfigFromForm());
    }

    globalThis.liveConfigFromForm = function liveConfigFromForm() {
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
            liveConfig[key] = next;
        }
        liveConfig.network_args = collectNetworkArgsFromForm(liveConfig, { skipUnchangedInputs: rawNetworkArgsChanged }).networkArgs;
        return liveConfig;
    }

    globalThis.formatFieldName = function formatFieldName(key) {
        const label = FIELD_LABEL_ZH[key];
        return label ? `${label} / ${key}` : key;
    }

    globalThis.createFieldInput = function createFieldInput(key, value, options = {}) {
        if (key === 'sample_prompts') {
            if (samplePromptsMode === 'path') {
                return createSamplePromptsPathInput(value);
            }
            return createSamplePromptsEditor(value, options.originalValue, options.hasDraftValue);
        }
        const fieldOptions = FIELD_OPTIONS[key];
        if (fieldOptions && !Array.isArray(value)) {
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
        if (key === 'lokr_factor' || key === 'lokr_factor_group_size' || key === 'lokr_project_chunk_bytes') {
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

    globalThis.createSamplePromptsPathInput = function createSamplePromptsPathInput(value) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input';
        input.value = value ?? '';
        input.title = '当前 sample_prompts 指向非 .txt 文件，保留为文件路径。';
        return input;
    }

    globalThis.createSamplePromptsEditor = function createSamplePromptsEditor(value, originalValue = value, touched = false) {
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

    globalThis.createSamplePromptAddButton = function createSamplePromptAddButton(rowsWrap) {
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

    globalThis.createSamplePromptTextModeButton = function createSamplePromptTextModeButton(editor) {
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

    globalThis.updateSamplePromptModeButtonState = function updateSamplePromptModeButtonState(btn, editor) {
        if (!btn || !editor) return;
        const textMode = editor.dataset.mode === 'text';
        btn.textContent = textMode ? '表格模式' : '文本模式';
        btn.title = textMode ? '切回按列编辑提示词' : '保留注释、空行和原始参数格式';
        btn.setAttribute('aria-pressed', String(textMode));
    }

    globalThis.setSamplePromptsEditorContent = function setSamplePromptsEditorContent(editor, content) {
        if (!editor) return;
        editor.dataset.originalContent = content || '';
        editor.dataset.touched = '0';
        renderSamplePromptRows(editor, content || '');
        updateSamplePromptModeButtonState(editor.closest('.field-row')?.querySelector('[data-sample-prompts-mode-toggle]'), editor);
    }

    globalThis.markSamplePromptsEditorTouched = function markSamplePromptsEditorTouched(editor) {
        if (editor) editor.dataset.touched = '1';
    }

    globalThis.renderSamplePromptRows = function renderSamplePromptRows(editor, content) {
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

    globalThis.switchSamplePromptsEditorToTextMode = function switchSamplePromptsEditorToTextMode(editor) {
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

    globalThis.switchSamplePromptsEditorToTableMode = function switchSamplePromptsEditorToTableMode(editor) {
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
