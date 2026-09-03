const FALLBACK_MODEL_FAMILIES = Object.freeze([
    {
        name: 'anima',
        aliases: ['anima'],
        pipeline_parallel: { configurable: true, runtime_available: false },
    },
    {
        name: 'krea2_raw',
        aliases: ['krea2', 'krea2_raw'],
        pipeline_parallel: { configurable: true, runtime_available: false },
    },
    {
        name: 'z_image',
        aliases: ['zimage', 'z_image'],
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

export function modelFamilyPipelineCapability(value) {
    return capabilityByFamily.get(normalizeModelFamily(value))?.pipeline_parallel || null;
}

export function modelFamilySupportsPipelineParallel(value) {
    return modelFamilyPipelineCapability(value)?.configurable === true;
}

export function isKrea2ModelFamily(value) {
    return normalizeModelFamily(value) === 'krea2_raw';
}
