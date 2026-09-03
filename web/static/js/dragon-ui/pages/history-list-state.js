/* In-memory state preserved while the history list is temporarily unmounted. */

let memorySnapshot = null;

export function restoreHistoryListState(defaultFilters = {}, workspace = {}) {
    if (!memorySnapshot) return { filters: { ...defaultFilters }, workspace };

    const filters = { ...defaultFilters };
    for (const key of Object.keys(filters)) {
        const value = memorySnapshot.filters?.[key];
        if (typeof value === 'string') filters[key] = value;
    }

    return {
        filters,
        workspace: {
            ...workspace,
            activeKey: memorySnapshot.activeKey || workspace.activeKey,
            selectedTaskIds: new Set(memorySnapshot.selectedTaskIds),
            expandedConfigKeys: new Set(memorySnapshot.expandedConfigKeys),
            initializedExpansion: memorySnapshot.initializedExpansion,
        },
    };
}

export function saveHistoryListState(state = {}) {
    const workspace = state.workspace || {};
    memorySnapshot = {
        filters: { ...(state.filters || {}) },
        activeKey: typeof workspace.activeKey === 'string' ? workspace.activeKey : '',
        selectedTaskIds: stringValues(workspace.selectedTaskIds),
        expandedConfigKeys: stringValues(workspace.expandedConfigKeys),
        initializedExpansion: workspace.initializedExpansion === true,
    };
}

function stringValues(values) {
    return [...(values || [])].filter((value) => typeof value === 'string' && value);
}
