import {
    LOKR_SCOPED_FIELD_KEYS,
    VERA_SCOPED_FIELD_KEYS,
} from '../../config/catalog/defaults.js?v=auto-block-swap-20260908-v3';
import { modelFamilyCapability, normalizeModelFamily } from '../../features/config-form/model-family.js?v=qwen-image-21-v2';
import { configFieldCatalogEntry } from './config-field-catalog.js?v=auto-block-swap-20260908-v3';
import { normalizeBooleanConfigValue } from './config-field-types.js?v=auto-block-swap-20260908-v3';

const ORTHOGONAL_METHODS = new Set(['ortholora', 'tlora']);
const FAMILY_SCOPED_METHOD_CLUSTERS = new Set([
    'adapter',
    'orthogonal',
    'reft',
    'routing',
    'fei',
    'chimera',
    'ip-adapter',
    'easycontrol',
    'soft-tokens',
    'spd-audit',
]);
const FAMILY_SCOPED_METHOD_FIELD_KEYS = new Set([
    'network_module',
    'network_args',
    'lora_adapter_kind',
    'ip_features_cache_to_disk',
    'weight_decay',
    'ip_diagnostics_epochs',
]);
const CONVROT_FIELD_KEYS = new Set([
    'convrot_group_size',
    'convrot_scope',
    'convrot_hadamard',
    'convrot_min_in_features',
    'convrot_largest_in_features_only',
    'convrot_large_layer_mode',
    'convrot_large_min_in_features',
]);
const IP_PE_CHILD_KEYS = new Set(['pe_lora_rank', 'pe_lora_alpha', 'pe_lora_layer_from']);

const METHOD_LABELS = Object.freeze({
    chimera: 'ChimeraHydra',
    easycontrol: 'EasyControl',
    hydralora: 'HydraLoRA',
    ip_adapter: 'IP-Adapter',
    reft: 'ReFT',
    soft_tokens: 'Soft Tokens',
    spd: 'SPD',
});

function contextValue(context, key, fallback = undefined) {
    if (Object.prototype.hasOwnProperty.call(context?.values || {}, key)) return context.values[key];
    return fallback;
}

function toggleEnabled(context, key, fallback = false) {
    return normalizeBooleanConfigValue(key, contextValue(context, key, fallback), fallback);
}

function choiceEnabled(value) {
    if (value === true || value === 1) return true;
    return !['', '0', 'false', 'none', 'off', 'disabled'].includes(String(value ?? '').trim().toLowerCase());
}

function shown(catalog) {
    return {
        visible: true,
        reason: '',
        code: null,
        controlledBy: catalog.controlledBy,
    };
}

function hidden(catalog, reason, code) {
    return {
        visible: false,
        reason,
        code,
        controlledBy: catalog.controlledBy,
    };
}

function methodHidden(catalog, method, expected) {
    return hidden(
        catalog,
        `该参数属于${expected}，当前方法为 ${METHOD_LABELS[method] || method}。`,
        'method-context',
    );
}

function isFamilyScopedMethodField(key, catalog) {
    return FAMILY_SCOPED_METHOD_FIELD_KEYS.has(key)
        || (catalog.stage === 'method' && FAMILY_SCOPED_METHOD_CLUSTERS.has(catalog.cluster));
}

export function configFieldDisclosure(key, context = {}) {
    const catalog = configFieldCatalogEntry(key);
    const method = String(context.method || 'lora').trim().toLowerCase();
    const adapter = String(context.adapter || 'lora').trim().toLowerCase();
    const modelFamily = normalizeModelFamily(context.modelFamily || 'anima');
    const familyCapability = modelFamilyCapability(modelFamily);

    if (catalog.location === 'audit_only' && !(catalog.cluster === 'spd-audit' && method === 'spd')) {
        return hidden(catalog, '该参数不进入常规训练流，仅在候选审计中显示。', 'audit-only');
    }

    // Adapter kind is a persistent training-context choice. Capability checks
    // may disable it, but the selector itself must not disappear.
    if (key === 'lora_adapter_kind') return shown(catalog);

    const familyScopedMethodField = isFamilyScopedMethodField(key, catalog);
    if (!familyCapability && familyScopedMethodField) {
        return hidden(
            catalog,
            `未识别模型族 ${modelFamily}，无法确认该方法分支是否受支持。`,
            'model-family-unknown',
        );
    }

    if (LOKR_SCOPED_FIELD_KEYS.has(key) && adapter !== 'lokr') {
        return hidden(catalog, '该参数仅在 LoKr Adapter 下显示。', 'adapter-context');
    }
    if (VERA_SCOPED_FIELD_KEYS.has(key) && adapter !== 'vera') {
        return hidden(catalog, '该参数仅在 VeRA Adapter 下显示。', 'adapter-context');
    }
    if (key === 'dora_wd' && (
        adapter !== 'lora'
        || method !== 'lora'
    )) {
        return hidden(catalog, 'DoRA 只作为普通 LoRA 的可选扩展显示。', 'adapter-context');
    }

    const useOrtho = toggleEnabled(context, 'use_ortho');
    const useTimestepMask = toggleEnabled(context, 'use_timestep_mask');
    const sharedOrthogonalField = key === 'channel_scaling_alpha' && ['chimera', 'spd'].includes(method);
    if (catalog.cluster === 'orthogonal' && !(
        sharedOrthogonalField
        ||
        ORTHOGONAL_METHODS.has(method)
        || useOrtho
        || useTimestepMask
    )) {
        return methodHidden(catalog, method, ' OrthoLoRA / T-LoRA');
    }

    const addReft = toggleEnabled(context, 'add_reft');
    if (catalog.cluster === 'reft' && method !== 'reft' && !addReft) {
        return methodHidden(catalog, method, ' ReFT');
    }

    const moeStyle = contextValue(context, 'use_moe_style', false);
    const routingActive = method === 'hydralora' || choiceEnabled(moeStyle);
    if (catalog.cluster === 'routing' && !routingActive) {
        return methodHidden(catalog, method, ' HydraLoRA / MoE 路由');
    }
    if (catalog.cluster === 'fei' && (!routingActive || String(contextValue(context, 'router_source', 'none')).toLowerCase() !== 'fei')) {
        return hidden(catalog, '先在 MoE 路由中选择 FEI 作为 router source。', 'feature-disabled');
    }

    const chimeraActive = method === 'chimera' || toggleEnabled(context, 'use_chimera_hydra');
    if (catalog.cluster === 'chimera' && !chimeraActive) {
        return methodHidden(catalog, method, ' ChimeraHydra');
    }

    if (catalog.cluster === 'ip-adapter') {
        if (method !== 'ip_adapter') return methodHidden(catalog, method, ' IP-Adapter');
        if (key !== 'use_ip_adapter' && !toggleEnabled(context, 'use_ip_adapter')) {
            return hidden(catalog, '请先启用 IP-Adapter。', 'feature-disabled');
        }
        if (IP_PE_CHILD_KEYS.has(key) && !toggleEnabled(context, 'pe_lora_enabled')) {
            return hidden(catalog, '请先启用 PE LoRA。', 'feature-disabled');
        }
    }
    if (key === 'ip_diagnostics_epochs' && method !== 'ip_adapter') {
        return methodHidden(catalog, method, ' IP-Adapter');
    }
    if (key === 'ip_features_cache_to_disk' && method !== 'ip_adapter') {
        return methodHidden(catalog, method, ' IP-Adapter');
    }

    if (key === 'weight_decay' && method !== 'spd') {
        return methodHidden(catalog, method, ' SPD');
    }

    if (catalog.cluster === 'easycontrol') {
        if (method !== 'easycontrol') return methodHidden(catalog, method, ' EasyControl');
        if (key !== 'use_easycontrol' && !toggleEnabled(context, 'use_easycontrol')) {
            return hidden(catalog, '请先启用 EasyControl。', 'feature-disabled');
        }
    }

    if (catalog.cluster === 'soft-tokens' && method !== 'soft_tokens') {
        return methodHidden(catalog, method, ' Soft Tokens');
    }

    const baseCompute = String(context.baseCompute || contextValue(context, 'base_compute', 'bf16')).toLowerCase();
    if (CONVROT_FIELD_KEYS.has(key) && !['w8a16_convrot', 'w8a8_convrot'].includes(baseCompute)) {
        return hidden(catalog, 'ConvRot 参数仅在 W8A16/W8A8 ConvRot 计算模式下显示。', 'feature-disabled');
    }

    return shown(catalog);
}
