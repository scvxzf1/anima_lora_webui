/* Incremental dirty-state tracking and stable DOM bindings for config forms. */

export function createConfigDirtyBindings(root) {
    const fields = new Map([...root.querySelectorAll('[data-config-field-key]')].map((field) => [
        field.dataset.configFieldKey,
        {
            field,
            reset: field.querySelector(':scope > [data-config-reset-field], .dragon-field-label-actions [data-config-reset-field]'),
        },
    ]));
    const changedOnly = root.querySelector('[data-config-changed-only]');
    return {
        fields,
        count: root.querySelector('[data-config-dirty-count]'),
        changedOnly,
        changedOnlyLabel: changedOnly?.querySelector('span') || null,
    };
}

export function replaceConfigDirtyKeys(state, changedKeys) {
    state.dirtyKeys = new Set(changedKeys);
}

export function updateConfigDirtyKey(state, key, baselineValue) {
    if (!key) return;
    if (configValuesEqual(state.draftValues[key], baselineValue)) state.dirtyKeys.delete(key);
    else state.dirtyKeys.add(key);
}

export function renderConfigDirtyState(bindings, state, changedKey = null) {
    const entries = changedKey
        ? [bindings.fields.get(changedKey)].filter(Boolean)
        : [...bindings.fields.values()];
    entries.forEach(({ field, reset }) => {
        const dirty = state.dirtyKeys.has(field.dataset.configFieldKey);
        setDataset(field, 'dirty', String(dirty));
        if (reset && reset.hidden === dirty) reset.hidden = !dirty;
    });

    setText(bindings.count, state.dirty ? `已修改 ${state.dirtyKeys.size} 项` : '未修改');
    const changedOnly = bindings.changedOnly;
    if (changedOnly) {
        if (changedOnly.disabled === state.dirty) changedOnly.disabled = !state.dirty;
        setDataset(changedOnly, 'active', String(state.showChangedOnly));
        setText(bindings.changedOnlyLabel, state.showChangedOnly ? '显示全部' : '查看修改');
    }
}

function configValuesEqual(left, right) {
    return JSON.stringify(left ?? '') === JSON.stringify(right ?? '');
}

function setDataset(node, key, value) {
    if (node.dataset[key] !== value) node.dataset[key] = value;
}

function setText(node, value) {
    if (node && node.textContent !== value) node.textContent = value;
}
