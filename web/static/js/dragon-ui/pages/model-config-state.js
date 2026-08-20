/* Pure helpers for Dragon's ordered global model configuration library. */

export const MODEL_PATH_FIELDS = Object.freeze([
    ['pretrained_model_name_or_path', '基础 DiT 模型'],
    ['qwen3', 'Qwen3 文本编码器'],
    ['vae', 'VAE 模型'],
]);

export const DEFAULT_MODEL_GROUP = Object.freeze({
    id: 'ungrouped',
    label: '未分组',
    item_ids: [],
});

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

export function cleanModelGroup(group = {}) {
    return {
        id: String(group.id || '').trim(),
        label: String(group.label || '').trim(),
        item_ids: [...new Set((Array.isArray(group.item_ids) ? group.item_ids : [])
            .map((itemId) => String(itemId || '').trim())
            .filter(Boolean))],
    };
}

export function normalizeModelGroups(groups = [], items = []) {
    const knownIds = new Set(items.map((item) => item.id));
    const source = groups.length ? groups : [{ ...DEFAULT_MODEL_GROUP, item_ids: items.map((item) => item.id) }];
    const assigned = new Set();
    const normalized = source.map((rawGroup, index) => {
        const group = cleanModelGroup(rawGroup);
        const itemIds = group.item_ids.filter((itemId) => {
            if (!knownIds.has(itemId) || assigned.has(itemId)) return false;
            assigned.add(itemId);
            return true;
        });
        return {
            id: group.id || `model-group-${index + 1}`,
            label: group.label || `分组 ${index + 1}`,
            item_ids: itemIds,
        };
    });
    const missing = items.map((item) => item.id).filter((itemId) => !assigned.has(itemId));
    normalized[0].item_ids.push(...missing);
    return normalized;
}

export function cloneModelGroups(groups = [], items = null) {
    const source = Array.isArray(items) ? normalizeModelGroups(groups, items) : groups.map(cleanModelGroup);
    return source.map((group) => ({ ...group, item_ids: [...group.item_ids] }));
}

export function modelGroupForItem(groups, itemId) {
    return groups.find((group) => group.item_ids.includes(itemId)) || null;
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

export function moveModelGroup(groups, groupId, offset) {
    const index = groups.findIndex((group) => group.id === groupId);
    const target = index + offset;
    if (index < 0 || target < 0 || target >= groups.length) return cloneModelGroups(groups);
    const next = cloneModelGroups(groups);
    [next[index], next[target]] = [next[target], next[index]];
    return next;
}

export function placeModelGroup(groups, groupId, targetIndex) {
    const next = cloneModelGroups(groups);
    const sourceIndex = next.findIndex((group) => group.id === groupId);
    if (sourceIndex < 0) return next;
    const [group] = next.splice(sourceIndex, 1);
    next.splice(Math.max(0, Math.min(targetIndex, next.length)), 0, group);
    return next;
}

export function moveModelItemInGroups(groups, itemId, offset) {
    const next = cloneModelGroups(groups);
    const group = modelGroupForItem(next, itemId);
    if (!group) return next;
    const index = group.item_ids.indexOf(itemId);
    const target = index + offset;
    if (target < 0 || target >= group.item_ids.length) return next;
    [group.item_ids[index], group.item_ids[target]] = [group.item_ids[target], group.item_ids[index]];
    return next;
}

export function placeModelItem(groups, itemId, targetGroupId, anchorItemId = '', position = 'after') {
    const next = cloneModelGroups(groups);
    next.forEach((group) => {
        group.item_ids = group.item_ids.filter((candidate) => candidate !== itemId);
    });
    const target = next.find((group) => group.id === targetGroupId);
    if (!target) return cloneModelGroups(groups);
    const anchorIndex = anchorItemId ? target.item_ids.indexOf(anchorItemId) : -1;
    const insertIndex = anchorIndex < 0
        ? target.item_ids.length
        : anchorIndex + (position === 'before' ? 0 : 1);
    target.item_ids.splice(insertIndex, 0, itemId);
    return next;
}

export function removeModelGroup(groups, groupId) {
    if (groups.length <= 1) return cloneModelGroups(groups);
    const next = cloneModelGroups(groups);
    const index = next.findIndex((group) => group.id === groupId);
    if (index < 0) return next;
    const [removed] = next.splice(index, 1);
    const target = next[Math.max(0, index - 1)] || next[0];
    target.item_ids.push(...removed.item_ids);
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

export function validateModelGroups(groups, items) {
    if (!groups.length) return { message: '至少需要保留 1 个模型配置分组' };
    const knownIds = new Set(items.map((item) => item.id));
    const groupIds = new Set();
    const labels = new Set();
    const assigned = new Set();
    for (const rawGroup of groups) {
        const group = cleanModelGroup(rawGroup);
        if (!group.id || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(group.id)) {
            return { groupId: group.id, message: '模型配置分组 ID 无效' };
        }
        if (groupIds.has(group.id)) return { groupId: group.id, message: '模型配置分组 ID 不能重复' };
        groupIds.add(group.id);
        if (!group.label) return { groupId: group.id, message: '请填写分组名称' };
        if (group.label.length > 80) return { groupId: group.id, message: '分组名称不能超过 80 个字符' };
        const normalizedLabel = group.label.toLocaleLowerCase();
        if (labels.has(normalizedLabel)) return { groupId: group.id, message: `分组名称“${group.label}”重复` };
        labels.add(normalizedLabel);
        for (const itemId of group.item_ids) {
            if (!knownIds.has(itemId)) return { groupId: group.id, message: '分组引用了不存在的模型配置' };
            if (assigned.has(itemId)) return { groupId: group.id, message: '每个模型配置只能属于一个分组' };
            assigned.add(itemId);
        }
    }
    if (assigned.size !== knownIds.size) return { message: '每个模型配置都必须属于一个分组' };
    return null;
}

export function serializeModelState(items, defaultId, groups = []) {
    return JSON.stringify({
        defaultId,
        items: items.map(cleanModelItem),
        groups: normalizeModelGroups(groups, items),
    });
}
