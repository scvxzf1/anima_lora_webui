/* Shared ordered drag/drop target state used by dataset presets and history collections. */

export function scheduleOrderedRowDropTarget(state, row, clientY, options) {
    const { frameKey, pendingKey } = options;
    state[pendingKey] = { row, clientY };
    if (state[frameKey]) return;
    state[frameKey] = requestAnimationFrame(() => {
        state[frameKey] = 0;
        const pending = state[pendingKey];
        state[pendingKey] = null;
        if (!pending?.row?.isConnected) return;
        const rect = pending.row.getBoundingClientRect();
        const position = pending.clientY < rect.top + rect.height / 2 ? 'before' : 'after';
        setOrderedDropTarget(state, pending.row, 'row', position, options);
    });
}

export function setOrderedDropTarget(state, node, kind, position = '', options) {
    const { targetKey, rowClassPrefix, dropzoneOverAttribute = 'over', groupTargetClass } = options;
    const active = state[targetKey];
    if (active?.node === node && active.kind === kind && active.position === position) return;
    clearActiveOrderedDropTarget(state, options);
    if (kind === 'row') node.classList.add(`${rowClassPrefix}-${position}`);
    else if (kind === 'dropzone') node.dataset[dropzoneOverAttribute] = 'true';
    else if (kind === 'group' && groupTargetClass) node.classList.add(groupTargetClass);
    state[targetKey] = { node, kind, position };
}

export function clearActiveOrderedDropTarget(state, options) {
    const { targetKey, rowClassPrefix, dropzoneOverAttribute = 'over', groupTargetClass } = options;
    const active = state[targetKey];
    if (!active?.node) return;
    if (active.kind === 'row') active.node.classList.remove(`${rowClassPrefix}-before`, `${rowClassPrefix}-after`);
    else if (active.kind === 'dropzone') active.node.dataset[dropzoneOverAttribute] = 'false';
    else if (active.kind === 'group' && groupTargetClass) active.node.classList.remove(groupTargetClass);
    state[targetKey] = null;
}

export function clearOrderedDropTargetIf(state, node, options) {
    if (state[options.targetKey]?.node === node) clearActiveOrderedDropTarget(state, options);
}

export function clearOrderedDropTargets(state, options) {
    if (state[options.frameKey]) cancelAnimationFrame(state[options.frameKey]);
    state[options.frameKey] = 0;
    state[options.pendingKey] = null;
    clearActiveOrderedDropTarget(state, options);
}
