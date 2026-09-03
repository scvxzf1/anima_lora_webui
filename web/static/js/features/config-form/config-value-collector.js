/**
 * Config form value collection and patch preparation.
 */
import {
    CONFIG_FORM_INTERNAL_KEYS,
    FORM_UI_DEFAULTS,
    FORM_UI_PERSIST_DEFAULT_FIELDS,
    NETWORK_ARG_FIELD_MAP,
    OPTIONAL_EMPTY_FIELDS,
    OPTIONAL_EMPTY_NUMBER_FIELDS,
} from '../../config/catalog.js?v=module-bootstrap-20260902-lokr-backend-v4';
import {
    coerceNetworkArgValue,
    formatNetworkArg,
    parseNetworkArgEntry,
} from '../anima-app/helpers/network-args.js?v=module-bootstrap-20260831-release-v1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    displayConfigFieldValue,
    isActiveNetworkArgFieldKey,
    originalConfigFieldValue,
    syncConfigDraftFromForm,
} from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    fieldValueType,
    fieldValueTypeForKey,
} from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    normalizeMultilineText,
    parseArrayValue,
    parseNumberValue,
    valuesEqual,
    valuesEqualForFieldType,
} from '../anima-app/helpers/form-values.js?v=module-bootstrap-20260831-release-v1';
import {
    normalizePrecisionPreference,
    precisionPreferenceFromConfig,
    precisionPreferencePatch,
} from '../anima-app/helpers/config-values.js?v=module-bootstrap-20260831-release-v1';
import {
    normalizeNetworkArgArray,
    parseNetworkArgMap,
} from '../anima-app/helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260831-release-v1';
import { saveSamplePrompts } from '../anima-app/helpers/sample-prompts-bridge.js?v=module-bootstrap-20260831-release-v1';
import { serializeSamplePromptsEditor } from '../sample-prompts/model.js?v=module-bootstrap-20260831-release-v1';
import {
    applyLoraAdapterPatch,
    applyOptimizerCompatibilityPatch,
} from './form-fields-adapters.js?v=module-bootstrap-20260902-lokr-backend-v4';

const configState = getConfigState();

function currentConfigState() {
    return configState.currentConfig || {};
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
        // Stage schedule is dialog-managed (no form input). Always keep draft values.
        if (key === 'stage_schedule' || key === 'stage_schedule_enabled') {
            const original = currentConfig[key];
            if (!(key in currentConfig) || !valuesEqual(next, original)) {
                values[key] = next;
            }
            continue;
        }
        const hasOriginal = key in currentConfig;
        const original = hasOriginal ? currentConfig[key] : FORM_UI_DEFAULTS[key];
        if (!hasOriginal) {
            if (shouldSkipUiDefaultField(key, next, options)) continue;
            values[key] = next;
            continue;
        }
        if (!valuesEqualForFieldType(
            next,
            original,
            fieldValueTypeForKey(key, original),
        )) {
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
    // 保存路径（persistDefaultFields）会 strip 无关适配器字段；
    // dirty 检测路径不 strip，避免假未保存。
    return applyLoraAdapterPatch(values, { stripInactive: Boolean(options.persistDefaultFields) });
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
    const fallback = Object.prototype.hasOwnProperty.call(config || {}, spec.key)
        ? config[spec.key]
        : spec.default;
    return coerceNetworkArgValue(argMap.has(spec.arg) ? argMap.get(spec.arg) : fallback, spec);
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
