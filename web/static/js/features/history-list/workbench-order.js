/**
 * History workbench ordering and collection grouping helpers.
 */
import { HISTORY_UNGROUPED_COLLECTION_KEY } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260831-release-v1';
import {
    configGroupKey,
    groupHistoryTasks,
    historyCollectionsForWorkbench,
    historyGroupDisplayLabel,
    historyTaskCollectionKey,
    historyTaskCollectionLabel,
    historyTaskSortComparator,
} from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260831-release-v1';
import { saveHistoryCollectionSettings, uniqueStringList } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260831-release-v1';

const historyState = getHistoryState();

export function historyCollectionNamesForTasks(tasks) {
    const names = Array.from(new Set((tasks || []).map(historyTaskCollectionLabel).filter(Boolean)));
    return names.length ? names : ['未分类'];
}

export function moveItemInList(list, value, direction) {
    const out = uniqueStringList(list);
    const item = String(value || '').trim();
    const index = out.indexOf(item);
    if (!item || index < 0) return out;
    out.splice(index, 1);
    if (direction === 'top') out.unshift(item);
    else if (direction === 'bottom') out.push(item);
    else if (direction === 'up') out.splice(Math.max(0, index - 1), 0, item);
    else if (direction === 'down') out.splice(Math.min(out.length, index + 1), 0, item);
    else out.splice(index, 0, item);
    return out;
}

export function collectionOrderValues(collections) {
    const available = (collections || []).filter((collection) => !collection.is_ungrouped && collection.value).map((collection) => collection.value);
    const out = historyState.historyCollectionSettings.collection_order.filter((value) => available.includes(value));
    for (const value of available) {
        if (!out.includes(value)) out.push(value);
    }
    return out;
}

export async function moveHistoryCollection(collection, direction, allCollections = []) {
    if (!collection || collection.is_ungrouped || !collection.value) return;
    await moveHistoryCollectionValue(collection.value, direction, allCollections);
}

export async function moveHistoryCollectionValue(value, direction, allCollections = null) {
    const group = String(value || '').trim();
    if (!group) return;
    const collections = allCollections || historyCollectionsForWorkbench(historyState.historyTasks);
    const order = moveItemInList(collectionOrderValues(collections), group, direction);
    await saveHistoryCollectionSettings({
        ...historyState.historyCollectionSettings,
        collection_order: order,
    });
}

export async function ensureHistoryCollectionOrderValue(value) {
    const group = String(value || '').trim();
    if (!group || historyState.historyCollectionSettings.collection_order.includes(group)) return;
    await saveHistoryCollectionSettings({
        ...historyState.historyCollectionSettings,
        collection_order: [...historyState.historyCollectionSettings.collection_order, group],
    });
}

export function configGroupOrderValues(groups, collection) {
    const key = historyCollectionStorageKey(collection || '__all__');
    const available = (groups || []).map(configGroupKey).filter(Boolean);
    const saved = historyState.historyCollectionSettings.config_group_order?.[key] || [];
    const out = saved.filter((value) => available.includes(value));
    for (const value of available) {
        if (!out.includes(value)) out.push(value);
    }
    return out;
}

export async function moveHistoryConfigGroup(group, direction, groups = [], collection = null) {
    const groupKey = configGroupKey(group);
    if (!groupKey) return;
    const collectionKey = historyCollectionStorageKey(collection || '__all__');
    const order = moveItemInList(configGroupOrderValues(groups, collection), groupKey, direction);
    await saveHistoryCollectionSettings({
        ...historyState.historyCollectionSettings,
        config_group_order: {
            ...(historyState.historyCollectionSettings.config_group_order || {}),
            [collectionKey]: order,
        },
    });
}

export function groupHistoryTasksByCollection(tasks) {
    const map = new Map();
    for (const task of tasks) {
        const key = historyTaskCollectionKey(task);
        if (!map.has(key)) {
            map.set(key, {
                key,
                label: historyTaskCollectionLabel(task),
                value: historyTaskCollectionValue(task),
                is_ungrouped: !historyTaskCollectionValue(task),
                tasks: [],
            });
        }
        map.get(key).tasks.push(task);
    }
    return Array.from(map.values())
        .map(enrichHistoryCollection)
        .sort(historyCollectionComparator);
}

export function historyCollectionComparator(a, b) {
    if (a.is_ungrouped !== b.is_ungrouped) return a.is_ungrouped ? -1 : 1;
    const order = historyState.historyCollectionSettings.collection_order || [];
    const aIndex = a.value ? order.indexOf(a.value) : -1;
    const bIndex = b.value ? order.indexOf(b.value) : -1;
    if (aIndex !== bIndex) {
        if (aIndex < 0) return -1;
        if (bIndex < 0) return 1;
        return aIndex - bIndex;
    }
    return (b.latest_started_at - a.latest_started_at) || a.label.localeCompare(b.label, 'zh-CN');
}

export function historyCollectionStorageKey(collection) {
    if (!collection) return '__all__';
    if (typeof collection === 'string') {
        if (!collection || collection === 'collection:__all__') return '__all__';
        if (collection === HISTORY_UNGROUPED_COLLECTION_KEY) return '__ungrouped__';
        return collection.startsWith('collection:') ? collection.slice('collection:'.length) : collection;
    }
    if (collection.is_ungrouped) return '__ungrouped__';
    return String(collection.value || '').trim() || '__ungrouped__';
}

export function historyCollectionByKey(collections, key) {
    return (collections || []).find((collection) => collection.key === key) || null;
}

export function sortedHistoryConfigGroups(groups, collectionKey = '__all__') {
    const storageKey = historyCollectionStorageKey(collectionKey);
    const order = historyState.historyCollectionSettings.config_group_order?.[storageKey] || [];
    return [...(groups || [])].sort((a, b) => {
        const aKey = configGroupKey(a);
        const bKey = configGroupKey(b);
        const aIndex = order.indexOf(aKey);
        const bIndex = order.indexOf(bKey);
        if (aIndex !== bIndex) {
            if (aIndex < 0) return -1;
            if (bIndex < 0) return 1;
            return aIndex - bIndex;
        }
        const aTime = Math.max(0, ...a.tasks.map((task) => Number(task.started_at || 0)));
        const bTime = Math.max(0, ...b.tasks.map((task) => Number(task.started_at || 0)));
        return (bTime - aTime) || historyGroupDisplayLabel(a).localeCompare(historyGroupDisplayLabel(b), 'zh-CN');
    });
}

export function enrichHistoryCollection(collection) {
    const tasks = [...(collection.tasks || [])].sort(historyTaskSortComparator(historyState.historyManagerFilters.sort));
    const groups = sortedHistoryConfigGroups(
        groupHistoryTasks(tasks).map(sortHistoryManagerGroupTasks),
        historyCollectionStorageKey(collection),
    );
    return {
        ...collection,
        tasks,
        groups,
        latest_started_at: Math.max(0, ...tasks.map((task) => Number(task.started_at || 0))),
    };
}

export function sortHistoryManagerGroupTasks(group) {
    return {
        ...group,
        tasks: [...(group.tasks || [])].sort(historyTaskSortComparator(historyState.historyManagerFilters.sort)),
    };
}

export function historyTaskCollectionValue(task) {
    return String(task?.group || '').trim();
}
