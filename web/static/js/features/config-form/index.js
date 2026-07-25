/**
 * Config form draft/render helpers.
 * Moved out of anima-app mechanical chunks.
 */
import { applyLoraAdapterDraft, applyOptimizerCompatibilityPatch } from './form-fields.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    CHIMERA_UI_DEFAULT_FIELDS,
    CONFIG_FORM_INTERNAL_KEYS,
    CONFIG_FORM_MERGED_FIELDS,
    DATASET_BLUEPRINT_FIELDS,
    DEPRECATED_CONFIG_FORM_FIELDS,
    FORM_SECTION_DEFS,
    FORM_UI_DEFAULTS,
    IP_ADAPTER_UI_DEFAULT_FIELDS,
    LOKR_SCOPED_FIELD_KEYS,
    METHOD_SCOPED_CONFIG_FORM_FIELDS,
    NETWORK_ARG_FIELD_MAP,
    NETWORK_ARG_FIELD_SPECS,
    RETIRED_CONFIG_FORM_FIELDS,
    SOFT_TOKENS_UI_DEFAULT_FIELDS,
    SPD_UI_DEFAULT_FIELDS,
    VERA_SCOPED_FIELD_KEYS,
} from '../../config/catalog.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    isTruthy,
    loraAdapterFlagsMatchConfig,
    loraAdapterKindFromConfig,
    normalizeLoraAdapterKind,
    normalizePrecisionPreference,
    precisionPreferenceFromConfig,
} from '../anima-app/helpers/config-values.js?v=module-bootstrap-20260714-stage-dataset5';
import { readLiveLoraAdapterKind } from './form-fields-adapters.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    configureConfigFormBridge,
    networkArgFieldValueFromConfig,
    readFieldInputValue,
    shouldSkipUiDefaultField,
    updateDoRAFieldState,
    updateLoKrFieldState,
    updateLossWeightingFieldState,
    updateVeRAFieldState,
} from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { coerceNetworkArgValue, parseNetworkArgEntry } from '../anima-app/helpers/network-args.js?v=module-bootstrap-20260714-stage-dataset5';
import { parseArrayValue, valuesEqual } from '../anima-app/helpers/form-values.js?v=module-bootstrap-20260714-stage-dataset5';
import { activeMethodKey } from './method-key.js?v=module-bootstrap-20260714-stage-dataset5';
import { appendConfigGroupsByCategory, createConfigGroupEntry } from './group-entry.js?v=module-bootstrap-20260714-stage-dataset5';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { currentSamplePromptText } from '../anima-app/helpers/sample-prompts-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

const configState = getConfigState();
const configFormState = configState.configFormState;
// network_args 专属字段按 activeNetworkArgFamilies 暴露，不再无条件常显。
const ALWAYS_VISIBLE_NETWORK_ARG_FIELDS = new Set();

function currentConfigState() { return configState.currentConfig || {}; }

    export function resetConfigFormDraft() {
        configState.configFormState.draftValues.clear();
    }

    export function applyConfigCompatibilityDrafts() {
        const patch = applyOptimizerCompatibilityPatch({});
        for (const [key, value] of Object.entries(patch)) {
            configFormState.draftValues.set(key, value);
            const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
            if (!input) continue;
            if (input.type === 'checkbox') {
                input.checked = Boolean(value);
            } else {
                input.value = Array.isArray(value) ? JSON.stringify(value) : (value ?? '');
            }
            input.title = `${input.title ? `${input.title}\n` : ''}已自动修正兼容性参数，请保存当前配置后再训练。`;
        }
        return patch;
    }

    export function syncConfigDraftFromForm(options = {}) {
        document.querySelectorAll('#config-form .field-input[data-key]').forEach((input) => {
            updateConfigDraftFromInput(input, options);
        });
    }

    export function updateConfigDraftFromInput(input, options = {}) {
        const key = input?.dataset?.key;
        if (!key || CONFIG_FORM_INTERNAL_KEYS.has(key)) return;
        if (key === 'stage_schedule' || key === 'stage_schedule_enabled') return;
        const original = originalConfigFieldValue(key);
        const next = readFieldInputValue(input, original);
        if (key === 'lora_adapter_kind') {
            applyLoraAdapterDraft(next);
            return;
        }
        if (configDraftValueChanged(key, next, original, options)) {
            configFormState.draftValues.set(key, next);
        } else {
            configFormState.draftValues.delete(key);
        }
    }

    export function originalConfigFieldValue(key) {
        const currentConfig = currentConfigState();
        if (key === 'sample_prompts' && configState.samplePromptsMode !== 'path') {
            return configState.samplePromptsContent || '';
        }
        if (key === 'precision_preference') {
            return precisionPreferenceFromConfig(currentConfig);
        }
        if (isActiveNetworkArgFieldKey(key)) {
            return networkArgFieldValueFromConfig(NETWORK_ARG_FIELD_MAP.get(key), currentConfig);
        }
        if (key === 'lora_adapter_kind') {
            return loraAdapterKindFromConfig(currentConfig);
        }
        if (key in currentConfig) return currentConfig[key];
        return FORM_UI_DEFAULTS[key];
    }

    export function displayConfigFieldValue(key, value) {
        const currentConfig = currentConfigState();
        if (key === 'lora_adapter_kind') {
            return configFormState.draftValues.has(key)
                ? configFormState.draftValues.get(key)
                : loraAdapterKindFromConfig(currentConfig);
        }
        if (key === 'precision_preference') {
            return configFormState.draftValues.has(key)
                ? normalizePrecisionPreference(configFormState.draftValues.get(key))
                : precisionPreferenceFromConfig(currentConfig);
        }
        return configFormState.draftValues.has(key)
            ? configFormState.draftValues.get(key)
            : value;
    }

    export function configDraftValueChanged(key, next, original = originalConfigFieldValue(key), options = {}) {
        const currentConfig = currentConfigState();
        if (key === 'sample_prompts' && configState.samplePromptsMode !== 'path') {
            return String(next || '') !== String(configState.samplePromptsContent || '');
        }
        if (key === 'precision_preference') {
            return normalizePrecisionPreference(next) !== precisionPreferenceFromConfig(currentConfig);
        }
        if (isActiveNetworkArgFieldKey(key)) {
            return !valuesEqual(next, original);
        }
        if (key === 'lora_adapter_kind') {
            return normalizeLoraAdapterKind(next) !== normalizeLoraAdapterKind(original)
                || !loraAdapterFlagsMatchConfig(next, currentConfig);
        }
        const hasOriginal = key in currentConfig;
        if (!hasOriginal && shouldSkipUiDefaultField(key, next, options)) return false;
        return !valuesEqual(next, original);
    }

    export function renderConfigForm(config) {
        const container = document.getElementById('config-form');
        container.innerHTML = '';

        const fieldsByKey = {};
        for (const [key, value] of Object.entries(config)) {
            if (key === 'output_dir') continue;
            if (key === 'general' || key === 'datasets') continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (CONFIG_FORM_MERGED_FIELDS?.has?.(key)) continue;
            if (shouldSkipConfigFormField(key, config)) continue;
            if (DATASET_BLUEPRINT_FIELDS.has(key)) continue;
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) continue;
            fieldsByKey[key] = value;
        }
        for (const [key, value] of Object.entries(FORM_UI_DEFAULTS)) {
            if (key === 'output_dir') continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (CONFIG_FORM_MERGED_FIELDS?.has?.(key)) continue;
            if (shouldSkipConfigFormField(key, config)) continue;
            if (DATASET_BLUEPRINT_FIELDS.has(key)) continue;
            if (!shouldExposeUiDefaultField(key, config, fieldsByKey)) continue;
            if (!(key in fieldsByKey)) fieldsByKey[key] = value;
        }
        applyNetworkArgFields(fieldsByKey, config);
        fieldsByKey.sample_prompts = currentSamplePromptText(config);

        const consumed = new Set();
        const sectionEntries = [];
        for (const section of FORM_SECTION_DEFS) {
            if (!shouldRenderConfigSection(section, config)) continue;
            const fields = collectSectionFields(fieldsByKey, section.keys, consumed);
            if (fields.length > 0) {
                sectionEntries.push(createConfigGroupEntry(
                    section.title,
                    fields,
                    section.className || '',
                    section.description || '',
                    section.open,
                    section.notice || ''
                ));
            }
        }

        const otherFields = Object.entries(fieldsByKey).filter(([key]) => !consumed.has(key));
        if (otherFields.length > 0) {
            sectionEntries.push(createConfigGroupEntry(
                '其他高级选项',
                otherFields,
                '',
                '未归类的新字段或低频字段；保留给高级调试使用。'
            ));
        }
        appendConfigGroupsByCategory(container, sectionEntries);
        updateLoKrFieldState();
        updateVeRAFieldState();
        updateDoRAFieldState();
        updateLossWeightingFieldState();
    }

    export function shouldRenderConfigSection(section, config = currentConfigState()) {
        if (!section?.method) return true;
        return activeMethodKey(config) === section.method;
    }

    export function resolveLiveLoraAdapterKind(config = currentConfigState()) {
        // 优先读表单 draft / DOM，保证切换 lora_adapter_kind 后立刻反映。
        try {
            return readLiveLoraAdapterKind();
        } catch {
            return loraAdapterKindFromConfig(config);
        }
    }

    export function isLoraAdapterScopedFieldActive(key, config = currentConfigState()) {
        if (LOKR_SCOPED_FIELD_KEYS.has(key)) {
            return resolveLiveLoraAdapterKind(config) === 'lokr';
        }
        if (VERA_SCOPED_FIELD_KEYS.has(key)) {
            return resolveLiveLoraAdapterKind(config) === 'vera';
        }
        if (key === 'dora_wd') {
            return resolveLiveLoraAdapterKind(config) === 'lora';
        }
        return true;
    }

    export function isConvrotScopedFieldActive(key, config = currentConfigState()) {
        const convrotKeys = new Set([
            'convrot_group_size',
            'convrot_scope',
            'convrot_hadamard',
            'convrot_min_in_features',
            'convrot_largest_in_features_only',
            'convrot_large_layer_mode',
            'convrot_large_min_in_features',
        ]);
        if (!convrotKeys.has(key)) return true;
        const baseCompute = String(config?.base_compute ?? 'bf16').trim().toLowerCase() || 'bf16';
        return baseCompute === 'w8a16_convrot' || baseCompute === 'w8a8_convrot';
    }

    export function shouldSkipConfigFormField(key, config = currentConfigState()) {
        if (key === 'stage_schedule' || key === 'stage_schedule_enabled') return true;
        if (CONFIG_FORM_MERGED_FIELDS?.has?.(key)) return true;
        if (DEPRECATED_CONFIG_FORM_FIELDS.has(key)) return true;
        if (RETIRED_CONFIG_FORM_FIELDS.has(key)) return true;
        if (key === 'mixed_precision' || key === 'full_fp16' || key === 'full_bf16') return true;
        // 即使 config 里残留了 lokr_factor / vera_*，非对应适配器也不展示。
        if (LOKR_SCOPED_FIELD_KEYS.has(key) || VERA_SCOPED_FIELD_KEYS.has(key) || key === 'dora_wd') {
            if (!isLoraAdapterScopedFieldActive(key, config)) return true;
        }
        if (
            key === 'convrot_group_size'
            || key === 'convrot_scope'
            || key === 'convrot_hadamard'
            || key === 'convrot_min_in_features'
            || key === 'convrot_largest_in_features_only'
            || key === 'convrot_large_layer_mode'
            || key === 'convrot_large_min_in_features'
        ) {
            if (!isConvrotScopedFieldActive(key, config)) return true;
        }
        const scopedFamilies = METHOD_SCOPED_CONFIG_FORM_FIELDS.get(key);
        if (!scopedFamilies) return false;
        return !scopedFamilies.has(activeMethodKey(config));
    }

    export function shouldExposeUiDefaultField(key, config, fieldsByKey = {}) {
        if (key in fieldsByKey) return true;
        if (NETWORK_ARG_FIELD_MAP.has(key)) {
            if (ALWAYS_VISIBLE_NETWORK_ARG_FIELDS.has(key)) return true;
            return isActiveNetworkArgFieldKey(key, config);
        }
        if (LOKR_SCOPED_FIELD_KEYS.has(key) || VERA_SCOPED_FIELD_KEYS.has(key) || key === 'dora_wd') {
            return isLoraAdapterScopedFieldActive(key, config);
        }
        if (
            key === 'convrot_group_size'
            || key === 'convrot_scope'
            || key === 'convrot_hadamard'
            || key === 'convrot_min_in_features'
            || key === 'convrot_largest_in_features_only'
            || key === 'convrot_large_layer_mode'
            || key === 'convrot_large_min_in_features'
        ) {
            return isConvrotScopedFieldActive(key, config);
        }
        const family = activeMethodKey(config);
        if (SPD_UI_DEFAULT_FIELDS.has(key)) return family === 'spd';
        if (CHIMERA_UI_DEFAULT_FIELDS.has(key)) return family === 'chimera';
        if (IP_ADAPTER_UI_DEFAULT_FIELDS.has(key)) return family === 'ip_adapter';
        if (SOFT_TOKENS_UI_DEFAULT_FIELDS.has(key)) return family === 'soft_tokens';
        return true;
    }

    export function applyNetworkArgFields(fieldsByKey, config) {
        const specs = activeNetworkArgSpecs(config);
        if (!specs.length) return;
        const argMap = parseNetworkArgMap(config?.network_args);
        for (const spec of specs) {
            const rawValue = argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default;
            fieldsByKey[spec.key] = coerceNetworkArgValue(rawValue, spec);
        }
    }

    export function isActiveNetworkArgFieldKey(key, config = currentConfigState()) {
        return activeNetworkArgSpecs(config).some((spec) => spec.key === key);
    }

    export function collectSectionFields(fieldsByKey, orderedKeys, consumed) {
        const fields = [];
        for (const key of orderedKeys) {
            if (consumed.has(key) || !(key in fieldsByKey)) continue;
            fields.push([key, fieldsByKey[key]]);
            consumed.add(key);
        }
        return fields;
    }

    export function activeNetworkArgSpecs(config = currentConfigState()) {
        const families = activeNetworkArgFamilies(config);
        // 仅按当前适配器/方法家族暴露；残留 network_args 不再把无关字段拉回表单。
        return NETWORK_ARG_FIELD_SPECS.filter((spec) => families.has(spec.family));
    }

    export function activeNetworkArgFamilies(config = currentConfigState()) {
        const families = new Set();
        const moduleName = String(config?.network_module || '');
        const method = activeMethodKey(config);
        const adapterKind = resolveLiveLoraAdapterKind(config);
        if (method === 'soft_tokens' || moduleName.includes('soft_tokens')) families.add('soft_tokens');
        // LoKr network_args 仅在当前适配器结构为 lokr 时生效。
        if (adapterKind === 'lokr' || method === 'lokr') families.add('lokr');
        if (method === 'ip_adapter' || isTruthy(config?.use_ip_adapter) || moduleName.includes('ip_adapter')) {
            families.add('ip_adapter');
        }
        if (method === 'easycontrol' || isTruthy(config?.use_easycontrol) || moduleName.includes('easycontrol')) {
            families.add('easycontrol');
        }
        return families;
    }
    export function parseNetworkArgMap(networkArgs) {
        const map = new Map();
        for (const raw of normalizeNetworkArgArray(networkArgs)) {
            const parsed = parseNetworkArgEntry(raw);
            if (parsed) map.set(parsed.arg, parsed.value);
        }
        return map;
    }

    export function normalizeNetworkArgArray(networkArgs) {
        if (Array.isArray(networkArgs)) return networkArgs.map((item) => String(item));
        if (typeof networkArgs === 'string' && networkArgs.trim()) return parseArrayValue(networkArgs).map((item) => String(item));
        return [];
    }

configureConfigFormBridge({
    syncConfigDraftFromForm,
    updateConfigDraftFromInput,
    originalConfigFieldValue,
    displayConfigFieldValue,
    configDraftValueChanged,
    isActiveNetworkArgFieldKey,
});
