/**
 * File-group drop target wiring (row/list/header).
 * Same-list reorder: always resolve by pointer Y, then submit full DOM order.
 */
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    clearFileGroupDropTarget,
    eventTargetClosest,
    fileGroupContainsRelatedTarget,
    finishFileGroupDrag,
    markFileGroupDropTarget,
    moveFileNearList,
    originClosest,
    registerFileGroupDropTarget,
} from './file-group-drag-core.js?v=module-bootstrap-20260809-nf4-v2';

const datasetState = getDatasetState();

function currentFileGroupDragState() {
    return datasetState.fileGroupDragState || null;
}

function rowPathFromNode(row) {
    return String(row?.dataset?.file || '').trim();
}

export function fileGroupOrderFromDom(list, rowSelector) {
    if (!(list instanceof Element)) return [];
    return Array.from(list.querySelectorAll(rowSelector || '.dataset-preset-row, .toml-file-row-wrap'))
        .map((node) => rowPathFromNode(node))
        .filter(Boolean);
}

function rowDropPosition(row, y) {
    const rect = row.getBoundingClientRect();
    if (!rect || rect.height <= 0) return 'after';
    return y < rect.top + rect.height / 2 ? 'before' : 'after';
}

function nearestRowByY(rows, payloadFile, y) {
    let best = null;
    for (const row of rows) {
        const file = rowPathFromNode(row);
        if (!file || file === payloadFile) continue;
        const rect = row.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) continue;
        // 点在行内：优先用该行；点在行外：按到行边界距离。
        const inside = y >= rect.top && y <= rect.bottom;
        const dy = inside ? 0 : (y < rect.top ? rect.top - y : y - rect.bottom);
        const midBias = Math.abs(y - (rect.top + rect.height / 2)) * 0.0001;
        const score = dy + midBias;
        if (!best || score < best.score || (score === best.score && inside && !best.inside)) {
            best = { row, file, score, inside, rect };
        }
    }
    return best;
}

/**
 * 始终按 Y 选目标行，不信任 drop 事件落在哪一行。
 * 提交完整 nextOrder，后端直接写 order。
 */
export function resolveFileGroupSameListDrop(list, group, options, payload, y) {
    if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
    if (!options.canDropToGroup(group, payload)) return null;

    const rowSelector = options.rowSelector || '.dataset-preset-row, .toml-file-row-wrap';
    const rows = Array.from(list?.querySelectorAll?.(rowSelector) || [])
        .filter((node) => node instanceof Element);
    if (!rows.length) {
        return {
            node: list,
            position: 'inside',
            drop: async () => {
                await options.onDrop(payload, group.id, 0, { order: [payload.file] });
            },
        };
    }

    const nearest = nearestRowByY(rows, payload.file, y);
    if (!nearest) {
        const order = fileGroupOrderFromDom(list, rowSelector).filter((path) => path !== payload.file);
        order.push(payload.file);
        return {
            node: list,
            position: 'inside',
            drop: async () => {
                await options.onDrop(payload, group.id, order.length - 1, { order });
            },
        };
    }

    const targetFile = nearest.file;
    const position = rowDropPosition(nearest.row, y);
    const currentOrder = fileGroupOrderFromDom(list, rowSelector);
    // 跨组时 source 不在 currentOrder；moveFileNearList 会把它插入到 anchor 旁。
    const nextOrder = moveFileNearList(currentOrder, payload.file, targetFile, position);
    const remaining = currentOrder.filter((path) => path !== payload.file);
    const anchorIndex = remaining.indexOf(targetFile);
    const index = anchorIndex < 0
        ? remaining.length
        : anchorIndex + (position === 'after' ? 1 : 0);

    // 仅同组且顺序未变时跳过；跨组 nextOrder 必含 source，不会误判 no-op。
    const unchanged = nextOrder.length === currentOrder.length
        && nextOrder.every((path, idx) => path === currentOrder[idx]);

    return {
        node: nearest.row,
        position,
        drop: async () => {
            if (unchanged) return;
            await options.onDrop(payload, group.id, index, {
                anchor: targetFile,
                position,
                order: nextOrder,
            });
        },
    };
}

export function setupFileGroupRowDropTarget(row, group, targetFile, options) {
    registerFileGroupDropTarget(row, ({ payload, y }) => {
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
        if (payload.file === targetFile && payload.groupId === group?.id) return null;
        if (!options.canDropToGroup(group, payload)) return null;
        const list = row.parentElement;
        return resolveFileGroupSameListDrop(list, group, options, payload, y);
    });

    const updateDropTarget = (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (payload.file === targetFile && payload.groupId === group?.id) return;
        if (!options.canDropToGroup(group, payload)) return;
        const list = row.parentElement;
        const resolved = resolveFileGroupSameListDrop(list, group, options, payload, event.clientY);
        if (!resolved) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        row._animaRowDrop = resolved;
        if (resolved.position === 'before' || resolved.position === 'after') {
            resolved.node.dataset.dropPosition = resolved.position;
        }
        markFileGroupDropTarget(resolved.node, resolved.position);
    };

    row.addEventListener('dragenter', updateDropTarget);
    row.addEventListener('dragover', updateDropTarget);
    row.addEventListener('dragleave', (event) => {
        if (fileGroupContainsRelatedTarget(row, event)) return;
        row._animaRowDrop = null;
        clearFileGroupDropTarget(row);
    });
    row.addEventListener('drop', async (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        const list = row.parentElement;
        const resolved = resolveFileGroupSameListDrop(list, group, options, payload, event.clientY)
            || row._animaRowDrop;
        row._animaRowDrop = null;
        if (!resolved) return;
        event.preventDefault();
        event.stopPropagation();
        await resolved.drop();
        finishFileGroupDrag();
    });
}

export function setupFileGroupListDropTarget(list, group, options) {
    registerFileGroupDropTarget(list, ({ payload, origin, y }) => {
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
        if (originClosest(origin, options.rowSelector)) return null;
        return resolveFileGroupSameListDrop(list, group, options, payload, y);
    });

    const updateDropTarget = (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (eventTargetClosest(event, options.rowSelector)) return;
        const resolved = resolveFileGroupSameListDrop(list, group, options, payload, event.clientY);
        if (!resolved) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        list._animaListDrop = resolved;
        if (resolved.position === 'before' || resolved.position === 'after') {
            resolved.node.dataset.dropPosition = resolved.position;
        }
        markFileGroupDropTarget(resolved.node, resolved.position);
    };

    list.addEventListener('dragenter', updateDropTarget);
    list.addEventListener('dragover', updateDropTarget);
    list.addEventListener('dragleave', (event) => {
        if (fileGroupContainsRelatedTarget(list, event)) return;
        list._animaListDrop = null;
        clearFileGroupDropTarget(list);
    });
    list.addEventListener('drop', async (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (eventTargetClosest(event, options.rowSelector)) return;
        const resolved = resolveFileGroupSameListDrop(list, group, options, payload, event.clientY)
            || list._animaListDrop;
        list._animaListDrop = null;
        if (!resolved) return;
        event.preventDefault();
        event.stopPropagation();
        await resolved.drop();
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
                const order = (group?.files || [])
                    .map((item) => item?.path)
                    .filter((path) => path && path !== payload.file);
                order.push(payload.file);
                await options.onDrop(payload, group.id, order.length - 1, { order });
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
        const order = (group?.files || [])
            .map((item) => item?.path)
            .filter((path) => path && path !== payload.file);
        order.push(payload.file);
        await options.onDrop(payload, group.id, order.length - 1, { order });
        finishFileGroupDrag();
    });
}

// file-group-drag-targets module end
