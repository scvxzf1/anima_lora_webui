/**
 * Config/TOML group drop-target wiring.
 */
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    clearFileGroupDropTarget,
    configGroupDropIndex,
    fileGroupContainsRelatedTarget,
    finishFileGroupDrag,
    markFileGroupDropTarget,
    registerFileGroupDropTarget,
} from './file-group-drag.js?v=module-bootstrap-20260831-release-v1';

const datasetState = getDatasetState();

function currentFileGroupDragState() {
    return datasetState.fileGroupDragState || null;
}

export function setupConfigGroupDropTarget(node, group, options) {
    registerFileGroupDropTarget(node, ({ payload, y }) => {
        if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return null;
        if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return null;
        const rect = node.getBoundingClientRect();
        const placeAfter = y > rect.top + rect.height / 2;
        const position = placeAfter ? 'after' : 'before';
        return {
            position,
            drop: async () => {
                const index = configGroupDropIndex(options.getSortableGroups(), group.id, placeAfter, payload.groupId);
                await options.onDrop(payload, index);
            },
        };
    });
    const updateDropTarget = (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return;
        if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        const rect = node.getBoundingClientRect();
        const placeAfter = event.clientY > rect.top + rect.height / 2;
        node.dataset.dropPosition = placeAfter ? 'after' : 'before';
        markFileGroupDropTarget(node, placeAfter ? 'after' : 'before');
    };
    node.addEventListener('dragenter', updateDropTarget);
    node.addEventListener('dragover', updateDropTarget);
    node.addEventListener('dragleave', (event) => {
        if (fileGroupContainsRelatedTarget(node, event)) return;
        clearFileGroupDropTarget(node);
    });
    node.addEventListener('drop', async (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return;
        if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return;
        event.preventDefault();
        event.stopPropagation();
        const placeAfter = node.dataset.dropPosition === 'after';
        const index = configGroupDropIndex(options.getSortableGroups(), group.id, placeAfter, payload.groupId);
        await options.onDrop(payload, index);
        finishFileGroupDrag();
    });
}
