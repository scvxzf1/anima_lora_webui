import { FORM_UI_DEFAULTS, NETWORK_ARG_FIELD_MAP } from '../../config/catalog/defaults.js?v=dragon-ui-20260830v2';
import { coerceNetworkArgValue, formatNetworkArg, parseNetworkArgEntry } from '../../features/anima-app/helpers/network-args.js?v=dragon-ui-20260812v35';
import { loraAdapterFlagsForKind, loraAdapterKindFromConfig, precisionPreferenceFromConfig, precisionPreferencePatch } from '../../features/anima-app/helpers/config-values.js?v=dragon-ui-20260812v35';
import { isBooleanConfigField, normalizeBooleanConfigValue } from './config-field-types.js?v=dragon-ui-20260830v1';

function networkArgMap(config) {
    const entries = Array.isArray(config?.network_args) ? config.network_args : [];
    return new Map(entries.map(parseNetworkArgEntry).filter(Boolean).map((item) => [item.arg, item.value]));
}

export function displayConfigValue(key, config) {
    if (key === 'lora_adapter_kind') return loraAdapterKindFromConfig(config);
    if (key === 'precision_preference') return precisionPreferenceFromConfig(config);
    const spec = NETWORK_ARG_FIELD_MAP.get(key);
    if (spec) {
        const args = networkArgMap(config);
        return coerceNetworkArgValue(args.has(spec.arg) ? args.get(spec.arg) : spec.default, spec);
    }
    if (isBooleanConfigField(key, config?.[key])) {
        return normalizeBooleanConfigValue(key, config?.[key]);
    }
    return config?.[key] ?? FORM_UI_DEFAULTS[key] ?? '';
}

export function serializeConfigValue(input, originalValue) {
    if (input.classList?.contains?.('dragon-toggle')) return input.dataset.checked === 'true';
    if (Array.isArray(originalValue)) {
        return input.value.split(/[\n,]/).map((item) => item.trim()).filter(Boolean);
    }
    if (input.type === 'checkbox') return input.checked;
    if (typeof originalValue === 'boolean' || isBooleanConfigField(input.dataset?.key, originalValue)) {
        return normalizeBooleanConfigValue(input.dataset?.key, input.value, originalValue);
    }
    if (typeof originalValue === 'number' || input.type === 'number') {
        return input.value === '' ? '' : Number(input.value);
    }
    return input.value;
}

export function prepareConfigPatch(changedValues, originalConfig) {
    const patch = { ...changedValues };
    if ('lora_adapter_kind' in patch) {
        Object.assign(patch, loraAdapterFlagsForKind(patch.lora_adapter_kind));
        delete patch.lora_adapter_kind;
    }
    if ('precision_preference' in patch) {
        Object.assign(patch, precisionPreferencePatch(patch.precision_preference, originalConfig));
        delete patch.precision_preference;
    }
    mergeNetworkArgs(patch, originalConfig);
    return patch;
}

function mergeNetworkArgs(patch, originalConfig) {
    const changedSpecs = [...NETWORK_ARG_FIELD_MAP.values()].filter((spec) => spec.key in patch);
    if (!changedSpecs.length) return;
    const replacements = new Map(changedSpecs.map((spec) => [spec.arg, formatNetworkArg(spec, patch[spec.key])]));
    const next = [];
    const seen = new Set();
    for (const raw of Array.isArray(originalConfig.network_args) ? originalConfig.network_args : []) {
        const parsed = parseNetworkArgEntry(raw);
        if (!parsed || !replacements.has(parsed.arg)) {
            next.push(raw);
            continue;
        }
        next.push(replacements.get(parsed.arg));
        seen.add(parsed.arg);
    }
    for (const [arg, value] of replacements) {
        if (!seen.has(arg)) next.push(value);
    }
    changedSpecs.forEach((spec) => { delete patch[spec.key]; });
    patch.network_args = next;
}

export function configValueForControl(value) {
    return Array.isArray(value) ? value.join('\n') : value;
}
