/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.deleteDatasetPresetGroup = async function deleteDatasetPresetGroup(group) {
        if (!group?.id || !group.deletable) return;
        const count = (group.files || []).length;
        const ok = await showHistoryTaskConfirmDialog({
            title: '删除数据集分组',
            description: group.label || group.id,
            message: count > 0
                ? `只删除这个分组，不删除其中 ${count} 个数据集 TOML；这些文件会回到默认数据集分组。`
                : '只删除这个空分组，不会删除任何 TOML 文件。',
            confirmText: '删除分组',
            danger: true,
        });
        if (!ok) return;
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '删除数据集分组失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组已删除', 'ok');
        } catch (e) {
            setDatasetPresetStatus('删除数据集分组失败: ' + e.message, 'error');
        }
    }

    globalThis.placeDatasetPresetGroup = async function placeDatasetPresetGroup(payload, index) {
        const groupId = payload?.groupId;
        if (!groupId) return;
        if (datasetPresetState.search.trim()) {
            setDatasetPresetStatus('筛选数据集预设时不能拖动排序，请先清空搜索', 'error');
            return;
        }
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'group', group: groupId, scope: 'dataset', index }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '调整数据集分组位置失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组位置已更新', 'ok');
        } catch (e) {
            setDatasetPresetStatus('调整数据集分组位置失败: ' + e.message, 'error');
        }
    }

    globalThis.placeDatasetPresetFile = async function placeDatasetPresetFile(payload, groupId, index) {
        const file = payload?.file;
        if (!file || !groupId) return;
        if (datasetPresetState.search.trim()) {
            setDatasetPresetStatus('筛选数据集预设时不能拖动排序，请先清空搜索', 'error');
            return;
        }
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'file', file, group: groupId, index }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '数据集预设位置调整失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集预设位置已更新', 'ok');
        } catch (e) {
            setDatasetPresetStatus('数据集预设位置调整失败: ' + e.message, 'error');
        }
    }

    globalThis.saveDatasetEditor = async function saveDatasetEditor(options = {}) {
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const targetFile = options.trainFile || currentTrainingSource.file || currentTomlFile || '';
        const targetContent = options.trainContent ?? (document.getElementById('toml-editor')?.value || '');
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        if (!rows.length || rows.some((row) => !row.source_dir.trim())) {
            setTomlStatus('error', '请至少填写一个原始数据集路径');
            return null;
        }
        try {
            const res = await api('/api/config/datasets', {
                method: 'PUT',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    train_file: targetFile,
                    train_content: targetContent,
                    prefer_existing_dataset_config: options.preferExistingDatasetConfig !== false,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetEditorState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存数据集配置失败');
                return null;
            }
            if (typeof res.train_content === 'string' && res.train_content) {
                const editor = document.getElementById('toml-editor');
                if (editor && targetFile === (currentTomlFile || val('toml-file-select'))) {
                    editor.value = res.train_content;
                    tomlSavedContent = res.train_content;
                }
            }
        datasetEditorState = {
            loading: false,
            loaded: true,
            dirty: false,
            dataset_config: res.dataset_config || datasetEditorState.dataset_config,
            datasets: normalizeDatasetEditorRows(res.datasets || rows),
            defaults: normalizeDatasetDefaults(res.defaults || datasetEditorState.defaults || {}),
            error: '',
        };
        currentConfig.dataset_config = datasetEditorState.dataset_config;
        if (datasetEditorState.datasets[0]) {
            currentConfig.source_image_dir = datasetEditorState.datasets[0].source_dir;
            currentConfig.resized_image_dir = datasetEditorState.datasets[0].image_dir;
            currentConfig.lora_cache_dir = datasetEditorState.datasets[0].cache_dir;
            }
            syncDatasetEditorToCompatFields();
            renderDatasetEditor();
            updateTomlDirtyState();
            await loadStepEstimate();
            if (options.reloadList !== false) {
                await loadTomlFileList(targetFile);
            }
            return res;
        } catch (e) {
            setTomlStatus('error', '保存数据集配置失败: ' + e.message);
            return null;
        }
    }

    globalThis.collectChangedFormValues = function collectChangedFormValues(options = {}) {
        syncConfigDraftFromForm(options);
        const values = {};
        for (const [key, next] of configFormState.draftValues.entries()) {
            if (!key) continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (isActiveNetworkArgFieldKey(key)) {
                continue;
            }
            if (key === 'sample_prompts') {
                if (samplePromptsMode === 'path') {
                    const original = typeof currentConfig.sample_prompts === 'string' ? currentConfig.sample_prompts : '';
                    if (!valuesEqual(next, original)) {
                        values[key] = next;
                    }
                    continue;
                }
                if (String(next || '') !== String(samplePromptsContent || '')) {
                    values[key] = next;
                }
                continue;
            }
            if (key === 'lora_adapter_kind') {
                continue;
            }
            const hasOriginal = key in currentConfig;
            const original = hasOriginal ? currentConfig[key] : FORM_UI_DEFAULTS[key];
            if (!hasOriginal) {
                if (shouldSkipUiDefaultField(key, next, options)) continue;
                values[key] = next;
                continue;
            }
            if (!valuesEqual(next, original)) {
                values[key] = next;
            }
        }
        const rawNetworkArgsChanged = 'network_args' in values;
        const merged = collectNetworkArgsFromForm(
            { network_args: values.network_args ?? currentConfig.network_args },
            { skipUnchangedInputs: rawNetworkArgsChanged },
        );
        if (merged.changed) {
            values.network_args = merged.networkArgs;
        } else if ('network_args' in values) {
            delete values.network_args;
        }
        if (values.use_lokr === true && !('lokr_factor' in values) && !('lokr_factor' in currentConfig)) {
            values.lokr_factor = FORM_UI_DEFAULTS.lokr_factor;
        }
        if (values.use_lokr === true && !('lokr_factor_group_size' in values) && !('lokr_factor_group_size' in currentConfig)) {
            values.lokr_factor_group_size = FORM_UI_DEFAULTS.lokr_factor_group_size;
        }
        if (values.use_lokr === true && !('lokr_project_chunk_bytes' in values) && !('lokr_project_chunk_bytes' in currentConfig)) {
            values.lokr_project_chunk_bytes = FORM_UI_DEFAULTS.lokr_project_chunk_bytes;
        }
        return applyLoraAdapterPatch(values);
    }

    globalThis.networkArgInputChanged = function networkArgInputChanged(input) {
        const spec = NETWORK_ARG_FIELD_MAP.get(input.dataset.key);
        if (!spec) return false;
        const original = networkArgFieldValueFromConfig(spec, currentConfig);
        const next = readFieldInputValue(input, original);
        return !valuesEqual(next, original);
    }

    globalThis.networkArgFieldValueFromConfig = function networkArgFieldValueFromConfig(spec, config = currentConfig) {
        const argMap = parseNetworkArgMap(config?.network_args);
        return coerceNetworkArgValue(argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default, spec);
    }

    globalThis.collectNetworkArgsFromForm = function collectNetworkArgsFromForm(baseConfig = currentConfig, options = {}) {
        const baseArgs = normalizeNetworkArgArray(baseConfig?.network_args);
        const formValues = new Map();
        const changedKeys = new Set();
        const applyNetworkArgFormValue = (key, value) => {
            if (!isActiveNetworkArgFieldKey(key)) return;
            const spec = NETWORK_ARG_FIELD_MAP.get(key);
            const original = networkArgFieldValueFromConfig(spec, currentConfig);
            formValues.set(spec.arg, { spec, value });
            if (!valuesEqual(value, original)) changedKeys.add(spec.key);
        };

        for (const [key, value] of configFormState.draftValues.entries()) {
            applyNetworkArgFormValue(key, value);
        }
        const inputs = [...document.querySelectorAll('#config-form .field-input[data-key]')]
            .filter((input) => isActiveNetworkArgFieldKey(input.dataset.key));
        for (const input of inputs) {
            if (options.skipUnchangedInputs && !networkArgInputChanged(input)) continue;
            const spec = NETWORK_ARG_FIELD_MAP.get(input.dataset.key);
            const original = networkArgFieldValueFromConfig(spec, currentConfig);
            applyNetworkArgFormValue(input.dataset.key, readFieldInputValue(input, original));
        }

        if (!formValues.size) {
            return { networkArgs: baseArgs, changed: !valuesEqual(baseArgs, currentConfig.network_args || []) };
        }

        const result = [];
        const seenArgs = new Set();
        for (const raw of baseArgs) {
            const parsed = parseNetworkArgEntry(raw);
            if (!parsed || !formValues.has(parsed.arg)) {
                result.push(raw);
                continue;
            }
            seenArgs.add(parsed.arg);
            const { spec, value } = formValues.get(parsed.arg);
            result.push(formatNetworkArg(spec, value));
        }

        for (const { spec, value } of formValues.values()) {
            if (seenArgs.has(spec.arg)) continue;
            if (!changedKeys.has(spec.key)) continue;
            result.push(formatNetworkArg(spec, value));
        }

        return {
            networkArgs: result,
            changed: !valuesEqual(result, currentConfig.network_args || []),
        };
    }

    globalThis.formatNetworkArg = function formatNetworkArg(spec, value) {
        return `${spec.arg}=${formatNetworkArgValue(spec, value)}`;
    }

    globalThis.formatNetworkArgValue = function formatNetworkArgValue(spec, value) {
        if (spec.valueType === 'booleanInt') return parseBooleanNetworkArg(value, spec.default) ? '1' : '0';
        if (spec.valueType === 'boolean') return parseBooleanNetworkArg(value, spec.default) ? 'true' : 'false';
        if (spec.valueType === 'integer') {
            const n = Number(value);
            return Number.isFinite(n) ? String(Math.trunc(n)) : String(spec.default);
        }
        if (spec.valueType === 'number') {
            const n = Number(value);
            return Number.isFinite(n) ? String(n) : String(spec.default);
        }
        return String(value ?? '').trim();
    }

    globalThis.prepareFormPatchValues = async function prepareFormPatchValues(values) {
        const nextValues = applyOptimizerCompatibilityPatch(values);
        if ('sample_prompts' in nextValues && samplePromptsMode !== 'path') {
            const promptText = String(nextValues.sample_prompts || '');
            if (promptText.trim()) {
                const saved = await saveSamplePrompts(promptText);
                nextValues.sample_prompts = saved.file || samplePromptsPath;
            } else {
                nextValues.sample_prompts = '';
            }
        }
        return nextValues;
    }

    globalThis.shouldSkipUiDefaultField = function shouldSkipUiDefaultField(key, value, options = {}) {
        if (!(key in FORM_UI_DEFAULTS)) return false;
        if (options.persistDefaultFields && FORM_UI_PERSIST_DEFAULT_FIELDS.has(key)) return false;
        if (OPTIONAL_EMPTY_FIELDS.has(key) && value === '') return true;
        return valuesEqual(value, FORM_UI_DEFAULTS[key]);
    }

    globalThis.readFieldInputValue = function readFieldInputValue(input, originalValue) {
        if (input.classList?.contains('sample-prompts-editor')) {
            if (input.dataset.touched !== '1') return input.dataset.originalContent || '';
            return serializeSamplePromptsEditor(input);
        }
        if (input.tagName === 'TEXTAREA') return normalizeMultilineText(input.value);
        if (input.type === 'checkbox') return input.checked;
        const raw = input.value;
        switch (input.dataset.valueType || fieldValueType(originalValue)) {
            case 'number':
                if (String(raw).trim() === '' && OPTIONAL_EMPTY_NUMBER_FIELDS.has(input.dataset.key)) return '';
                return parseNumberValue(raw, originalValue);
            case 'boolean':
                return raw === 'true';
            case 'array':
                return parseArrayValue(raw);
            default:
                return raw;
        }
    }

    globalThis.readLoKrEnabled = function readLoKrEnabled() {
        return readLiveLoraAdapterKind() === 'lokr';
    }

    globalThis.updateLoKrFieldState = function updateLoKrFieldState() {
        const enabled = readLoKrEnabled();
        const inputs = [
            document.querySelector('#config-form .field-input[data-key="lokr_factor"]'),
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

    globalThis.readVeRAEnabled = function readVeRAEnabled() {
        return readLiveLoraAdapterKind() === 'vera';
    }

    globalThis.readDoRAAvailable = function readDoRAAvailable() {
        return readLiveLoraAdapterKind() === 'lora';
    }

    globalThis.setDoRADraftValue = function setDoRADraftValue(value) {
        const original = originalConfigFieldValue('dora_wd');
        if (configDraftValueChanged('dora_wd', value, original, { persistDefaultFields: true })) {
            configFormState.draftValues.set('dora_wd', value);
        } else {
            configFormState.draftValues.delete('dora_wd');
        }
    }

    globalThis.updateDoRAFieldState = function updateDoRAFieldState() {
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

    globalThis.updateVeRAFieldState = function updateVeRAFieldState() {
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

    globalThis.currentLossWeightingScheme = function currentLossWeightingScheme() {
        const input = document.querySelector('#config-form .field-input[data-key="weighting_scheme"]');
        if (input) {
            return String(readFieldInputValue(input, originalConfigFieldValue('weighting_scheme')) || 'uniform');
        }
        if (configFormState.draftValues.has('weighting_scheme')) {
            return String(configFormState.draftValues.get('weighting_scheme') || 'uniform');
        }
        return String(currentConfig?.weighting_scheme ?? FORM_UI_DEFAULTS.weighting_scheme ?? 'uniform');
    }

    globalThis.lossWeightingFieldState = function lossWeightingFieldState(key) {
        const requiredScheme = LOSS_WEIGHTING_DEPENDENT_FIELDS.get(key);
        if (!requiredScheme) return { enabled: true, requiredScheme: '', currentScheme: currentLossWeightingScheme() };
        const currentScheme = currentLossWeightingScheme();
        return {
            enabled: currentScheme === requiredScheme,
            requiredScheme,
            currentScheme,
        };
    }

    globalThis.lossWeightingDisabledHint = function lossWeightingDisabledHint(key, state = lossWeightingFieldState(key)) {
        if (!state.requiredScheme) return '';
        return `仅 weighting_scheme = ${state.requiredScheme} 时生效；当前 ${state.currentScheme || 'uniform'}，不生效。`;
    }

    globalThis.applyLossWeightingFieldInputState = function applyLossWeightingFieldInputState(input, key) {
        if (!input || !LOSS_WEIGHTING_DEPENDENT_FIELDS.has(key)) return;
        const state = lossWeightingFieldState(key);
        input.disabled = !state.enabled;
        input.title = state.enabled ? '' : lossWeightingDisabledHint(key, state);
    }

    globalThis.updateLossWeightingFieldState = function updateLossWeightingFieldState() {
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

    globalThis.parseNumberValue = function parseNumberValue(raw, fallback) {
        const trimmed = String(raw).trim();
        if (trimmed === '' && fallback === '') return '';
        if (trimmed === '') return fallback;
        const n = Number(trimmed);
        return Number.isFinite(n) ? n : fallback;
    }

    globalThis.parseArrayValue = function parseArrayValue(raw) {
        const trimmed = String(raw).trim();
        if (!trimmed) return [];
        try {
            const parsed = JSON.parse(trimmed);
            return Array.isArray(parsed) ? parsed : [parsed];
        } catch {
            return trimmed.split(',').map((item) => item.trim()).filter(Boolean);
        }
    }

    globalThis.valuesEqual = function valuesEqual(a, b) {
        if (isBooleanLikeValue(a) && isBooleanLikeValue(b)) {
            return normalizeBooleanLikeValue(a) === normalizeBooleanLikeValue(b);
        }
        if (isNumberLikeValue(a) && isNumberLikeValue(b)) {
            return Number(a) === Number(b);
        }
        return JSON.stringify(a) === JSON.stringify(b);
    }

    globalThis.isBooleanLikeValue = function isBooleanLikeValue(value) {
        return value === true || value === false || value === 'true' || value === 'false';
    }

    globalThis.normalizeBooleanLikeValue = function normalizeBooleanLikeValue(value) {
        return value === true || value === 'true';
    }

    globalThis.isNumberLikeValue = function isNumberLikeValue(value) {
        if (typeof value === 'number') return Number.isFinite(value);
        if (typeof value !== 'string') return false;
        const trimmed = value.trim();
        return trimmed !== '' && Number.isFinite(Number(trimmed));
    }

    globalThis.normalizeMultilineText = function normalizeMultilineText(value) {
        return String(value || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean)
            .join('\n');
    }
