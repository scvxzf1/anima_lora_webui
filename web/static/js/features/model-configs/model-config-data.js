export const MODEL_CONFIG_PATH_FIELDS = Object.freeze([
    {
        key: 'pretrained_model_name_or_path',
        inputId: 'model-config-dit-path',
        label: '基础 DiT 模型',
    },
    {
        key: 'qwen3',
        inputId: 'model-config-qwen3-path',
        label: 'Qwen3 文本编码器',
    },
    {
        key: 'vae',
        inputId: 'model-config-vae-path',
        label: 'VAE 模型',
    },
]);

const MODEL_CONFIG_FAMILIES = new Set(['anima', 'krea2_raw', 'z_image']);

export function modelFamilyLabel(value) {
    if (value === 'krea2_raw') return 'Krea-2';
    if (value === 'z_image') return 'Z-Image';
    return 'Anima';
}

export function cleanModelConfigItem(item = {}) {
    return {
        id: String(item.id || '').trim(),
        name: String(item.name || '').trim(),
        model_family: MODEL_CONFIG_FAMILIES.has(item.model_family) ? item.model_family : 'anima',
        ...Object.fromEntries(MODEL_CONFIG_PATH_FIELDS.map(({ key }) => [key, String(item[key] || '').trim()])),
    };
}

export function modelConfigValidationError(item, items = []) {
    const clean = cleanModelConfigItem(item);
    if (!clean.name) return '请输入配置名称';
    const duplicate = items.some((candidate) => (
        candidate.id !== clean.id
        && String(candidate.name || '').trim().toLocaleLowerCase() === clean.name.toLocaleLowerCase()
    ));
    if (duplicate) return '配置名称不能重复';
    for (const field of MODEL_CONFIG_PATH_FIELDS) {
        if (!clean[field.key]) return `请填写${field.label}`;
    }
    return '';
}

export function moveModelConfig(items, sourceId, targetId, position = 'before') {
    const sourceIndex = items.findIndex((item) => item.id === sourceId);
    const targetIndex = items.findIndex((item) => item.id === targetId);
    if (sourceIndex < 0 || targetIndex < 0 || sourceIndex === targetIndex) return [...items];
    const next = [...items];
    const [source] = next.splice(sourceIndex, 1);
    const adjustedTarget = next.findIndex((item) => item.id === targetId);
    const insertIndex = position === 'after' ? adjustedTarget + 1 : adjustedTarget;
    next.splice(insertIndex, 0, source);
    return next;
}

export function moveModelConfigByOffset(items, itemId, offset) {
    const index = items.findIndex((item) => item.id === itemId);
    const targetIndex = index + offset;
    if (index < 0 || targetIndex < 0 || targetIndex >= items.length) return [...items];
    const next = [...items];
    [next[index], next[targetIndex]] = [next[targetIndex], next[index]];
    return next;
}

export function modelConfigRequest(items, defaultId, revision) {
    return {
        revision,
        default_id: defaultId,
        items: items.map(cleanModelConfigItem),
    };
}
