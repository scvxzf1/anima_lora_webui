/* Pure helpers for Dragon's ordered global model configuration library. */

export const MODEL_PATH_FIELDS = Object.freeze([
    ['pretrained_model_name_or_path', '基础 DiT 模型'],
    ['qwen3', 'Qwen3 文本编码器'],
    ['vae', 'VAE 模型'],
]);

export function cleanModelItem(item = {}) {
    return {
        id: String(item.id || '').trim(),
        name: String(item.name || '').trim(),
        model_family: item.model_family === 'krea2_raw' ? 'krea2_raw' : 'anima',
        ...Object.fromEntries(MODEL_PATH_FIELDS.map(([key]) => [key, String(item[key] || '').trim()])),
    };
}

export function cloneModelItems(items = []) {
    return items.map((item) => ({ ...cleanModelItem(item) }));
}

export function familyLabel(value) {
    return value === 'krea2_raw' ? 'Krea-2' : 'Anima';
}

export function filterModelItems(items, query) {
    const needle = String(query || '').trim().toLocaleLowerCase();
    if (!needle) return items;
    return items.filter((item) => [
        item.name,
        familyLabel(item.model_family),
        ...MODEL_PATH_FIELDS.map(([key]) => item[key]),
    ].some((value) => String(value || '').toLocaleLowerCase().includes(needle)));
}

export function moveModelItem(items, itemId, offset) {
    const index = items.findIndex((item) => item.id === itemId);
    const target = index + offset;
    if (index < 0 || target < 0 || target >= items.length) return [...items];
    const next = [...items];
    [next[index], next[target]] = [next[target], next[index]];
    return next;
}

export function uniqueDraftName(items, base = '新建模型配置') {
    const names = new Set(items.map((item) => item.name.toLocaleLowerCase()));
    let candidate = base;
    let suffix = 1;
    while (names.has(candidate.toLocaleLowerCase())) candidate = `${base} ${++suffix}`;
    return candidate;
}

export function validateModelItems(items, defaultId) {
    if (!items.length) return { message: '至少需要保留 1 个模型配置' };
    if (!items.some((item) => item.id === defaultId)) return { message: '请指定 1 个默认模型配置' };
    const names = new Set();
    for (const rawItem of items) {
        const item = cleanModelItem(rawItem);
        if (!item.name) return { itemId: item.id, field: 'name', message: '请输入配置名称' };
        const normalizedName = item.name.toLocaleLowerCase();
        if (names.has(normalizedName)) return { itemId: item.id, field: 'name', message: `配置名称“${item.name}”重复` };
        names.add(normalizedName);
        const missing = MODEL_PATH_FIELDS.find(([key]) => !item[key]);
        if (missing) return { itemId: item.id, field: missing[0], message: `请填写${missing[1]}` };
    }
    return null;
}

export function serializeModelState(items, defaultId) {
    return JSON.stringify({ defaultId, items: items.map(cleanModelItem) });
}
