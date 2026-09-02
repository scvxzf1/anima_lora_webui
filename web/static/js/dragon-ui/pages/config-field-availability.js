import {
    CHIMERA_UI_DEFAULT_FIELDS,
    IP_ADAPTER_UI_DEFAULT_FIELDS,
    LOKR_SCOPED_FIELD_KEYS,
    METHOD_SCOPED_CONFIG_FORM_FIELDS,
    NETWORK_ARG_FIELD_MAP,
    SPD_UI_DEFAULT_FIELDS,
    VERA_SCOPED_FIELD_KEYS,
} from '../../config/catalog/defaults.js?v=dragon-ui-20260902-lokr-backend-v4';
import { normalizeBooleanConfigValue } from './config-field-types.js?v=dragon-ui-20260902-lokr-backend-v4';
import {
    modelFamilySupportsPipelineParallel,
    normalizeModelFamily,
} from '../../features/config-form/model-family.js?v=module-bootstrap-20260903-pp-multimodel-v1';

const CONVROT_FIELD_KEYS = new Set([
    'convrot_group_size',
    'convrot_scope',
    'convrot_hadamard',
    'convrot_min_in_features',
    'convrot_largest_in_features_only',
    'convrot_large_layer_mode',
    'convrot_large_min_in_features',
]);

const PIPELINE_PARALLEL_FIELD_KEYS = new Set([
    'pipeline_parallel',
    'pipeline_parallel_stages',
    'pipeline_parallel_microbatches',
    'pipeline_parallel_schedule',
    'pipeline_parallel_split',
]);

const FAMILY_LABELS = Object.freeze({
    lokr: 'LoKr',
    vera: 'VeRA',
    soft_tokens: 'Soft Tokens',
    ip_adapter: 'IP-Adapter',
    easycontrol: 'EasyControl',
});

const METHOD_LABELS = Object.freeze({
    spd: 'SPD',
    chimera: 'ChimeraHydra',
    ip_adapter: 'IP-Adapter',
    easycontrol: 'EasyControl',
    soft_tokens: 'Soft Tokens',
});

const LORA_ADAPTER_KINDS = new Set(['lora', 'loha', 'lokr', 'glora', 'vera']);

function unavailable(reason, code) {
    return { enabled: false, reason, code };
}

function familyEnabled(family, adapter, method) {
    return {
        lokr: adapter === 'lokr' || method === 'lokr',
        vera: adapter === 'vera' || method === 'vera',
        soft_tokens: method === 'soft_tokens',
        ip_adapter: method === 'ip_adapter',
        easycontrol: method === 'easycontrol',
    }[family] === true;
}

export function resolveConfigAdapterKind(values = {}) {
    const selected = String(values.lora_adapter_kind ?? '').trim().toLowerCase();
    if (LORA_ADAPTER_KINDS.has(selected)) return selected;
    if (normalizeBooleanConfigValue('use_glora', values.use_glora)) return 'glora';
    if (normalizeBooleanConfigValue('use_vera', values.use_vera)) return 'vera';
    if (normalizeBooleanConfigValue('use_lokr', values.use_lokr)) return 'lokr';
    if (normalizeBooleanConfigValue('use_loha', values.use_loha)) return 'loha';
    return 'lora';
}

export function configFieldAvailability(key, context = {}) {
    const method = String(context.method || 'lora').trim().toLowerCase();
    const adapter = String(context.adapter || 'lora').trim().toLowerCase();
    const baseCompute = String(context.baseCompute || 'bf16').trim().toLowerCase();
    const modelFamily = normalizeModelFamily(context.modelFamily || 'anima');
    const pipelineParallel = normalizeBooleanConfigValue(
        'pipeline_parallel',
        context.pipelineParallel,
    );

    if (PIPELINE_PARALLEL_FIELD_KEYS.has(key) && !modelFamilySupportsPipelineParallel(modelFamily)) {
        return unavailable(
            `流水线并行尚未为当前模型族 ${modelFamily} 声明分层能力。`,
            'pipeline-parallel-model-family',
        );
    }
    if (key !== 'pipeline_parallel' && PIPELINE_PARALLEL_FIELD_KEYS.has(key) && !pipelineParallel) {
        return unavailable(
            '请先开启流水线并行。',
            'pipeline-parallel-disabled',
        );
    }

    if (CONVROT_FIELD_KEYS.has(key) && !['w8a16_convrot', 'w8a8_convrot'].includes(baseCompute)) {
        return unavailable(
            `当前基础计算类型为 ${baseCompute}；ConvRot 仅适用于 W8A16 或 W8A8。请先切换基础计算类型。`,
            'convrot-base-compute',
        );
    }

    const methodScope = METHOD_SCOPED_CONFIG_FORM_FIELDS.get(key);
    if (methodScope && !methodScope.has(method)) {
        const expected = [...methodScope].map((name) => METHOD_LABELS[name] || name).join('、');
        return unavailable(`该参数仅适用于 ${expected}，当前方法为 ${METHOD_LABELS[method] || method}。`, 'method-scope');
    }

    if (SPD_UI_DEFAULT_FIELDS.has(key) && method !== 'spd') {
        return unavailable(`该参数属于 SPD 实验，当前方法为 ${METHOD_LABELS[method] || method}。切换到 SPD 后可编辑。`, 'spd-method');
    }

    if (CHIMERA_UI_DEFAULT_FIELDS.has(key) && key !== 'use_chimera_hydra' && method !== 'chimera') {
        return unavailable(`该参数属于 ChimeraHydra，当前方法为 ${METHOD_LABELS[method] || method}。切换方法后可编辑。`, 'chimera-method');
    }

    if (IP_ADAPTER_UI_DEFAULT_FIELDS.has(key) && method !== 'ip_adapter') {
        return unavailable(`该参数属于 IP-Adapter，当前方法为 ${METHOD_LABELS[method] || method}。切换方法后可编辑。`, 'ip-adapter-method');
    }

    const spec = NETWORK_ARG_FIELD_MAP.get(key);
    if (spec && !familyEnabled(spec.family, adapter, method)) {
        return unavailable(`该参数属于 ${FAMILY_LABELS[spec.family] || spec.family}，当前适配器为 ${FAMILY_LABELS[adapter] || adapter}。`, 'adapter-family');
    }

    if (LOKR_SCOPED_FIELD_KEYS.has(key) && adapter !== 'lokr') {
        return unavailable(`该参数仅适用于 LoKr，当前适配器为 ${FAMILY_LABELS[adapter] || adapter}。请先启用 LoKr。`, 'lokr-adapter');
    }

    if (VERA_SCOPED_FIELD_KEYS.has(key) && adapter !== 'vera') {
        return unavailable(`该参数仅适用于 VeRA，当前适配器为 ${FAMILY_LABELS[adapter] || adapter}。请先启用 VeRA。`, 'vera-adapter');
    }

    if (key === 'dora_wd' && adapter !== 'lora') {
        return unavailable(`DoRA 权重衰减仅适用于普通 LoRA，当前适配器为 ${FAMILY_LABELS[adapter] || adapter}。`, 'dora-adapter');
    }

    return { enabled: true, reason: '', code: null };
}
