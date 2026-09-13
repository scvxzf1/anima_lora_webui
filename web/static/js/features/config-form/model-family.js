const FALLBACK_MODEL_FAMILIES = Object.freeze([
    {
        name: 'anima',
        aliases: ['anima'],
        supported_network_specs: null,
        supports_method_adapters: true,
        plain_lora_only: false,
        supported_attention_modes: ['flash', 'torch', 'mem_efficient', 'sageattn', 'flex', 'xformers', 'sdpa'],
        pipeline_parallel: { configurable: true, runtime_available: false },
    },
    {
        name: 'krea2_raw',
        aliases: ['krea2', 'krea2_raw'],
        supported_network_specs: null,
        supports_method_adapters: false,
        plain_lora_only: false,
        supported_attention_modes: ['torch', 'flash', 'sdpa'],
        pipeline_parallel: { configurable: true, runtime_available: false },
    },
    {
        name: 'z_image',
        aliases: ['zimage', 'z_image'],
        supported_network_specs: null,
        supports_method_adapters: false,
        plain_lora_only: false,
        supported_attention_modes: ['flash', 'torch', 'sdpa'],
        pipeline_parallel: { configurable: true, runtime_available: false },
    },
]);

let capabilityByFamily = new Map();
let canonicalByAlias = new Map();
let capabilityLoadPromise = null;

function itemsFromPayload(payload) {
    if (Array.isArray(payload)) return payload;
    if (payload && Array.isArray(payload.items)) return payload.items;
    return [];
}

export function configureModelFamilyCapabilities(payload) {
    const items = itemsFromPayload(payload);
    if (!items.length) throw new Error('model-family capability catalog is empty');

    const nextCapabilities = new Map();
    const nextAliases = new Map();
    items.forEach((item) => {
        const name = String(item?.name || '').trim().toLowerCase().replaceAll('-', '_');
        if (!name) return;
        const normalized = { ...item, name };
        nextCapabilities.set(name, normalized);
        [name, ...(Array.isArray(item.aliases) ? item.aliases : [])].forEach((alias) => {
            const key = String(alias || '').trim().toLowerCase().replaceAll('-', '_');
            if (key) nextAliases.set(key, name);
        });
    });
    if (!nextCapabilities.size) {
        throw new Error('model-family capability catalog has no valid entries');
    }
    capabilityByFamily = nextCapabilities;
    canonicalByAlias = nextAliases;
    return [...capabilityByFamily.values()];
}

configureModelFamilyCapabilities(FALLBACK_MODEL_FAMILIES);

export async function loadModelFamilyCapabilities(api) {
    if (typeof api !== 'function') throw new TypeError('api must be a function');
    if (!capabilityLoadPromise) {
        capabilityLoadPromise = Promise.resolve()
            .then(() => api('/api/config/model-families'))
            .then((payload) => configureModelFamilyCapabilities(payload))
            .catch((error) => {
                capabilityLoadPromise = null;
                console.warn('[model-family] capability catalog unavailable; using fallback', error);
                return [...capabilityByFamily.values()];
            });
    }
    return capabilityLoadPromise;
}

export function normalizeModelFamily(value) {
    const normalized = String(value ?? '').trim().toLowerCase().replaceAll('-', '_');
    return canonicalByAlias.get(normalized) || normalized;
}

export function modelFamilyCapability(value) {
    return capabilityByFamily.get(normalizeModelFamily(value)) || null;
}

export function modelFamilyPipelineCapability(value) {
    return modelFamilyCapability(value)?.pipeline_parallel || null;
}

export function modelFamilySupportsPipelineParallel(value) {
    const capability = modelFamilyPipelineCapability(value);
    return capability?.configurable === true && capability?.runtime_available === true;
}

export function modelFamilyOptionSupported(fieldKey, family, option) {
    if (fieldKey === 'auto_block_swap_preference' && option === 'ram') {
        return isKrea2ModelFamily(family);
    }
    const capability = modelFamilyCapability(family);
    if (!capability) return true;
    const value = String(option ?? '').trim();
    if (fieldKey === 'attn_mode') {
        return (capability.supported_attention_modes || []).includes(value.toLowerCase());
    }
    // Variant selection is open; concrete network/forward contracts validate execution.
    return true;
}

export function modelFamilySelectOptions(fieldKey, family, options, currentValue) {
    const supported = (Array.isArray(options) ? options : []).filter((option) => (
        modelFamilyOptionSupported(fieldKey, family, option)
    ));
    if (
        currentValue !== null
        && currentValue !== undefined
        && !supported.some((option) => String(option) === String(currentValue))
    ) {
        supported.push(currentValue);
    }
    return supported.map((value) => ({
        value,
        supported: modelFamilyOptionSupported(fieldKey, family, value),
    }));
}

export function isKrea2ModelFamily(value) {
    return normalizeModelFamily(value) === 'krea2_raw';
}
