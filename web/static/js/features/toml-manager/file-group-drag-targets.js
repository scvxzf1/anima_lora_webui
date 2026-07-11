/**
 * File-group drop target wiring (row/list/header).
 */
import {
    clearFileGroupDropTarget,
    configFileDropIndex,
    configGroupDropIndex,
    fileGroupContainsRelatedTarget,
    finishFileGroupDrag,
    markFileGroupDropTarget,
    registerFileGroupDropTarget,
} from './file-group-drag-core.js?v=module-bootstrap-20260711-ir1';

export function setupFileGroupRowDropTarget(row, group, targetFile, options) {
    registerFileGroupDropTarget(row, ({ payload, y }) => {
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
        if (payload.file === targetFile && payload.groupId === group?.id) return null;
        if (!options.canDropToGroup(group, payload)) return null;
        const rect = row.getBoundingClientRect();
        const placeAfter = y > rect.top + rect.height / 2;
        const position = placeAfter ? 'after' : 'before';
        return {
            position,
            drop: async () => {
                const index = configFileDropIndex(group, targetFile, placeAfter, payload.file);
                await options.onDrop(payload, group.id, index);
            },
        };
    });
    const updateDropTarget = (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (payload.file === targetFile && payload.groupId === group?.id) return;
        if (!options.canDropToGroup(group, payload)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        const rect = row.getBoundingClientRect();
        const placeAfter = event.clientY > rect.top + rect.height / 2;
        row.dataset.dropPosition = placeAfter ? 'after' : 'before';
        markFileGroupDropTarget(row, placeAfter ? 'after' : 'before');
    };
    row.addEventListener('dragenter', updateDropTarget);
    row.addEventListener('dragover', updateDropTarget);
    row.addEventListener('dragleave', (event) => {
        if (fileGroupContainsRelatedTarget(row, event)) return;
        clearFileGroupDropTarget(row);
    });
    row.addEventListener('drop', async (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (payload.file === targetFile && payload.groupId === group?.id) return;
        if (!options.canDropToGroup(group, payload)) return;
        event.preventDefault();
        event.stopPropagation();
        const placeAfter = row.dataset.dropPosition === 'after';
        const index = configFileDropIndex(group, targetFile, placeAfter, payload.file);
        await options.onDrop(payload, group.id, index);
        finishFileGroupDrag();
    });
}

export function setupFileGroupListDropTarget(list, group, options) {
    registerFileGroupDropTarget(list, ({ payload, origin }) => {
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
        if (originClosest(origin, options.rowSelector)) return null;
        if (!options.canDropToGroup(group, payload)) return null;
        return {
            position: 'inside',
            drop: async () => {
                const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
                await options.onDrop(payload, group.id, index);
            },
        };
    });
    const updateDropTarget = (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (eventTargetClosest(event, options.rowSelector)) return;
        if (!options.canDropToGroup(group, payload)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        markFileGroupDropTarget(list, 'inside');
    };
    list.addEventListener('dragenter', updateDropTarget);
    list.addEventListener('dragover', updateDropTarget);
    list.addEventListener('dragleave', (event) => {
        if (fileGroupContainsRelatedTarget(list, event)) return;
        clearFileGroupDropTarget(list);
    });
    list.addEventListener('drop', async (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (eventTargetClosest(event, options.rowSelector)) return;
        if (!options.canDropToGroup(group, payload)) return;
        event.preventDefault();
        event.stopPropagation();
        const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
        await options.onDrop(payload, group.id, index);
        finishFileGroupDrag();
    });
}

export function setupFileGroupHeaderDropTarget(node, group, options) {
    registerFileGroupDropTarget(node, ({ payload }) => {
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
        if (!options.canDropToGroup(group, payload)) return null;
        return {
            position: 'inside',
            drop: async () => {
                const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
                await options.onDrop(payload, group.id, index);
            },
        };
    });
    const updateDropTarget = (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (!options.canDropToGroup(group, payload)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        markFileGroupDropTarget(node, 'inside');
    };
    node.addEventListener('dragenter', updateDropTarget);
    node.addEventListener('dragover', updateDropTarget);
    node.addEventListener('dragleave', (event) => {
        if (fileGroupContainsRelatedTarget(node, event)) return;
        clearFileGroupDropTarget(node);
    });
    node.addEventListener('drop', async (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (!options.canDropToGroup(group, payload)) return;
        event.preventDefault();
        event.stopPropagation();
        const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
        await options.onDrop(payload, group.id, index);
        finishFileGroupDrag();
    });
}

// file-group-drag-targets module end
