/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { coerceNetworkArgValue, formatNetworkArg, parseNetworkArgEntry } from '../helpers/network-args.js?v=module-bootstrap-20260711-ir2';
import {
    datasetRowsForPayload,
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../helpers/dataset-values.js?v=module-bootstrap-20260711-ir2';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { configDraftValueChanged, configureConfigFormBridge, displayConfigFieldValue, isActiveNetworkArgFieldKey, originalConfigFieldValue, syncConfigDraftFromForm } from '../helpers/config-form-bridge.js?v=module-bootstrap-20260711-ir2';
import { fieldValueType } from '../helpers/config-field-ui-bridge.js?v=module-bootstrap-20260711-ir2';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    configureDatasetPresetActionsBridge,
    setDatasetPresetStatus,
} from '../helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260711-ir2';
import { loadTomlFileList } from '../helpers/toml-manager-bridge.js?v=module-bootstrap-20260711-ir2';
import { normalizeMultilineText, parseArrayValue, parseNumberValue, valuesEqual } from '../helpers/form-values.js?v=module-bootstrap-20260711-ir2';
import {
    normalizePrecisionPreference,
    precisionPreferenceFromConfig,
    precisionPreferencePatch,
} from '../helpers/config-values.js?v=module-bootstrap-20260711-ir2';
import { api, val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir2';
import { loadDatasetPresets, loadStepEstimate } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260711-ir2';
import { serializeSamplePromptsEditor } from '../../sample-prompts/model.js?v=module-bootstrap-20260711-ir2';
import { renderConfigDatasetPicker } from './06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260711-ir2';
import { syncDatasetEditorToCompatFields } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260711-ir2';
import { applyLoraAdapterPatch, applyOptimizerCompatibilityPatch, readLiveLoraAdapterKind } from './14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260711-ir2';
import {
    CONFIG_FORM_INTERNAL_KEYS,
    FORM_UI_DEFAULTS,
    FORM_UI_PERSIST_DEFAULT_FIELDS,
    NETWORK_ARG_FIELD_MAP,
    OPTIONAL_EMPTY_FIELDS,
    OPTIONAL_EMPTY_NUMBER_FIELDS,
} from '../../../config/catalog.js?v=module-bootstrap-20260711-ir2';
import { LOSS_WEIGHTING_DEPENDENT_FIELDS } from '../helpers/app-constants.js?v=module-bootstrap-20260711-ir2';
import { normalizeNetworkArgArray, parseNetworkArgMap } from '../helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260711-ir2';
import { renderDatasetEditor } from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir2';
import { showHistoryTaskConfirmDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260711-ir2';
import { currentTomlEditorContentForFile, updateTomlDirtyState } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir2';
import { saveSamplePrompts } from '../helpers/sample-prompts-bridge.js?v=module-bootstrap-20260711-ir2';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    setTomlStatus,
} from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir2';

const configState = getConfigState();
const datasetState = getDatasetState();
const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

    export async function deleteDatasetPresetGroup(group) {
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

    export async function placeDatasetPresetGroup(payload, index) {
        const groupId = payload?.groupId;
        const datasetPresetState = currentDatasetPresetState();
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

    export async function placeDatasetPresetFile(payload, groupId, index) {
        const file = payload?.file;
        const datasetPresetState = currentDatasetPresetState();
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

    export async function saveDatasetEditor(options = {}) {
        const currentTrainingSource = currentTrainingSourceState();
        const currentConfig = currentConfigState();
        const datasetEditorState = currentDatasetEditorState();
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const targetFile = options.trainFile || currentTrainingSource.file || tomlState.currentTomlFile || '';
        const targetContent = options.trainContent ?? currentTomlEditorContentForFile(targetFile);
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
                    config_values: collectChangedFormValues({ persistDefaultFields: true }),
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存数据集配置失败');
                return null;
            }
            if (typeof res.train_content === 'string' && res.train_content) {
                const editor = document.getElementById('toml-editor');
                if (editor && targetFile === (tomlState.currentTomlFile || val('toml-file-select'))) {
                    editor.value = res.train_content;
                    tomlState.tomlSavedContent = res.train_content;
                }
            }
            datasetState.datasetEditorState = {
                loading: false,
                loaded: true,
                dirty: false,
                dataset_config: res.dataset_config || datasetEditorState.dataset_config,
                datasets: normalizeDatasetEditorRows(res.datasets || rows),
                defaults: normalizeDatasetDefaults(res.defaults || datasetEditorState.defaults || {}),
                error: '',
            };
            const nextDatasetEditorState = currentDatasetEditorState();
            const nextDatasetConfig = nextDatasetEditorState.dataset_config || '';
            currentConfig.dataset_config = nextDatasetConfig;
            datasetState.selectedConfigDatasetFile = nextDatasetConfig;
            datasetState.selectedConfigDatasetSummary = nextDatasetConfig ? (res.summary || null) : null;
            datasetState.configDatasetPreviewState = {
                file: '',
                loading: false,
                payload: null,
                error: '',
            };
            if (nextDatasetEditorState.datasets[0]) {
                currentConfig.source_image_dir = nextDatasetEditorState.datasets[0].source_dir;
                currentConfig.resized_image_dir = nextDatasetEditorState.datasets[0].image_dir;
                currentConfig.lora_cache_dir = nextDatasetEditorState.datasets[0].cache_dir;
            }
            syncDatasetEditorToCompatFields();
            renderDatasetEditor();
            renderConfigDatasetPicker();
            updateTomlDirtyState();
            await loadDatasetPresets({ selectCurrent: false, manage: false });
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

    export function collectChangedFormValues(options = {}) {
        const configFormState = configState.configFormState;
        const currentConfig = currentConfigState();
        syncConfigDraftFromForm(options);
        const values = {};
        for (const [key, next] of configFormState.draftValues.entries()) {
            if (!key) continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (isActiveNetworkArgFieldKey(key)) {
                continue;
            }
            if (key === 'precision_preference') {
                const original = precisionPreferenceFromConfig(currentConfig);
                const normalized = normalizePrecisionPreference(next);
                if (!valuesEqual(normalized, original)) {
                    values[key] = normalized;
                }
                continue;
            }
            if (key === 'sample_prompts') {
                if (configState.samplePromptsMode === 'path') {
                    const original = typeof currentConfig.sample_prompts === 'string' ? currentConfig.sample_prompts : '';
                    if (!valuesEqual(next, original)) {
                        values[key] = next;
                    }
                    continue;
                }
                if (String(next || '') !== String(configState.samplePromptsContent || '')) {
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
        if (values.use_lokr === true && !('lokr_use_einsum' in values) && !('lokr_use_einsum' in currentConfig)) {
            values.lokr_use_einsum = FORM_UI_DEFAULTS.lokr_use_einsum;
        }
        if (values.use_lokr === true && !('lokr_decompose_w2' in values) && !('lokr_decompose_w2' in currentConfig)) {
            values.lokr_decompose_w2 = FORM_UI_DEFAULTS.lokr_decompose_w2;
        }
        if (values.use_lokr === true && !('lokr_factor_group_size' in values) && !('lokr_factor_group_size' in currentConfig)) {
            values.lokr_factor_group_size = FORM_UI_DEFAULTS.lokr_factor_group_size;
        }
        if (values.use_lokr === true && !('lokr_project_chunk_bytes' in values) && !('lokr_project_chunk_bytes' in currentConfig)) {
            values.lokr_project_chunk_bytes = FORM_UI_DEFAULTS.lokr_project_chunk_bytes;
        }
        if (
            options.persistDefaultFields
            && !('preprocess_precision_preference' in values)
            && !Object.prototype.hasOwnProperty.call(currentConfig || {}, 'preprocess_precision_preference')
        ) {
            values.preprocess_precision_preference = normalizePrecisionPreference(
                displayConfigFieldValue(
                    'preprocess_precision_preference',
                    originalConfigFieldValue('preprocess_precision_preference'),
                ),
            );
        }
        return applyLoraAdapterPatch(values);
    }

    export function networkArgInputChanged(input) {
        const currentConfig = currentConfigState();
        const spec = NETWORK_ARG_FIELD_MAP.get(input.dataset.key);
        if (!spec) return false;
        const original = networkArgFieldValueFromConfig(spec, currentConfig);
        const next = readFieldInputValue(input, original);
        return !valuesEqual(next, original);
    }

    export function networkArgFieldValueFromConfig(spec, config = currentConfigState()) {
        const argMap = parseNetworkArgMap(config?.network_args);
        return coerceNetworkArgValue(argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default, spec);
    }

    export function collectNetworkArgsFromForm(baseConfig = currentConfigState(), options = {}) {
        const configFormState = configState.configFormState;
        const currentConfig = currentConfigState();
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

    export async function prepareFormPatchValues(values) {
        const currentConfig = currentConfigState();
        const nextValues = applyOptimizerCompatibilityPatch(values);
        if ('precision_preference' in nextValues) {
            Object.assign(nextValues, precisionPreferencePatch(nextValues.precision_preference, currentConfig));
            delete nextValues.precision_preference;
        }
        if ('sample_prompts' in nextValues && configState.samplePromptsMode !== 'path') {
            const promptText = String(nextValues.sample_prompts || '');
            if (promptText.trim()) {
                const saved = await saveSamplePrompts(promptText);
                nextValues.sample_prompts = saved.file || configState.samplePromptsPath;
            } else {
                nextValues.sample_prompts = '';
            }
        }
        return nextValues;
    }

    export function shouldSkipUiDefaultField(key, value, options = {}) {
        if (!(key in FORM_UI_DEFAULTS)) return false;
        if (options.persistDefaultFields && FORM_UI_PERSIST_DEFAULT_FIELDS.has(key)) return false;
        if (OPTIONAL_EMPTY_FIELDS.has(key) && value === '') return true;
        return valuesEqual(value, FORM_UI_DEFAULTS[key]);
    }

    export function readFieldInputValue(input, originalValue) {
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

configureDatasetPresetActionsBridge({
    deleteDatasetPresetGroup,
    placeDatasetPresetGroup,
    placeDatasetPresetFile,
});

configureConfigFormBridge({
    saveDatasetEditor,
    collectChangedFormValues,
    networkArgInputChanged,
    networkArgFieldValueFromConfig,
    collectNetworkArgsFromForm,
    prepareFormPatchValues,
    shouldSkipUiDefaultField,
    readFieldInputValue,
    readLoKrEnabled,
    updateLoKrFieldState,
    readVeRAEnabled,
    readDoRAAvailable,
    setDoRADraftValue,
    updateDoRAFieldState,
    updateVeRAFieldState,
    currentLossWeightingScheme,
    lossWeightingFieldState,
    lossWeightingDisabledHint,
    applyLossWeightingFieldInputState,
    updateLossWeightingFieldState,
});
