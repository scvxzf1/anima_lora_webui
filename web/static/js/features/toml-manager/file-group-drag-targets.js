/**
 * File-group drop target wiring (row/list/header).
 */
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    clearFileGroupDropTarget,
    configFileDropIndex,
    eventTargetClosest,
    fileGroupContainsRelatedTarget,
    finishFileGroupDrag,
    markFileGroupDropTarget,
    originClosest,
    registerFileGroupDropTarget,
} from './file-group-drag-core.js?v=module-bootstrap-20260711-ir6';

const datasetState = getDatasetState();

function currentFileGroupDragState() {
    return datasetState.fileGroupDragState || null;
}

function rowPathFromNode(row) {
    return row?.dataset?.file || '';
}

/**
 * 同组排序的唯一落点算法：
 * 1. 从 DOM 读取当前行顺序
 * 2. 先排除正在拖动的项
 * 3. 用各行中点决定插入下标
 * 这样 “1 拖到 4 下方” 不会再卡成 after(2)。
 */
export function resolveFileGroupRowPlacement(listOrRows, options, payload, y) {
    if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
    const rows = (listOrRows instanceof Element
        ? Array.from(listOrRows.querySelectorAll?.(options.rowSelector) || [])
        : Array.from(listOrRows || [])
    ).filter((node) => node instanceof Element);

    const items = [];
    for (const row of rows) {
        const file = rowPathFromNode(row);
        if (!file || file === payload.file) continue;
        const rect = row.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) continue;
        items.push({ row, file, rect });
    }
    if (!items.length) {
        return {
            node: listOrRows instanceof Element ? listOrRows : null,
            position: 'inside',
            index: 0,
            placeAfter: false,
            targetFile: '',
        };
    }

    for (let index = 0; index < items.length; index += 1) {
        const item = items[index];
        const mid = item.rect.top + item.rect.height / 2;
        if (y < mid) {
            return {
                node: item.row,
                position: 'before',
                index,
                placeAfter: false,
                targetFile: item.file,
            };
        }
    }

    const last = items[items.length - 1];
    return {
        node: last.row,
        position: 'after',
        index: items.length,
        placeAfter: true,
        targetFile: last.file,
    };
}

function resolveFileGroupListDropTarget(list, group, options, payload, y) {
    if (!options.canDropToGroup(group, payload)) return null;
    const placement = resolveFileGroupRowPlacement(list, options, payload, y);
    if (!placement) return null;

    if (!placement.targetFile) {
        return {
            node: list,
            position: 'inside',
            drop: async () => {
                const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
                await options.onDrop(payload, group.id, index);
            },
        };
    }

    return {
        node: placement.node,
        position: placement.position,
        drop: async () => {
            // placement.index 已是“剔除拖动项后”的插入下标，直接使用，避免二次换算漂移。
            await options.onDrop(payload, group.id, placement.index);
        },
    };
}

function resolveFileGroupRowDropTarget(row, group, targetFile, options, payload, y) {
    if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
    if (payload.file === targetFile && payload.groupId === group?.id) return null;
    if (!options.canDropToGroup(group, payload)) return null;

    const list = row.parentElement;
    if (list) {
        const placement = resolveFileGroupRowPlacement(list, options, payload, y);
        if (placement?.targetFile) {
            return {
                node: placement.node,
                position: placement.position,
                drop: async () => {
                    await options.onDrop(payload, group.id, placement.index);
                },
            };
        }
    }

    // 兜底：没有父列表时退回单行 before/after。
    const rect = row.getBoundingClientRect();
    const placeAfter = y > rect.top + rect.height / 2;
    const position = placeAfter ? 'after' : 'before';
    return {
        node: row,
        position,
        drop: async () => {
            const index = configFileDropIndex(group, targetFile, placeAfter, payload.file);
            await options.onDrop(payload, group.id, index);
        },
    };
}

export function setupFileGroupRowDropTarget(row, group, targetFile, options) {
    registerFileGroupDropTarget(row, ({ payload, y }) => (
        resolveFileGroupRowDropTarget(row, group, targetFile, options, payload, y)
    ));
    const updateDropTarget = (event) => {
        const payload = currentFileGroupDragState();
        const resolved = resolveFileGroupRowDropTarget(row, group, targetFile, options, payload, event.clientY);
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
        // 松手时按最终坐标重算，避免沿用旧 hover 落点。
        const resolved = resolveFileGroupRowDropTarget(row, group, targetFile, options, payload, event.clientY)
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
        return resolveFileGroupListDropTarget(list, group, options, payload, y);
    });
    const updateDropTarget = (event) => {
        const payload = currentFileGroupDragState();
        if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
        if (eventTargetClosest(event, options.rowSelector)) return;
        const resolved = resolveFileGroupListDropTarget(list, group, options, payload, event.clientY);
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
        const resolved = resolveFileGroupListDropTarget(list, group, options, payload, event.clientY)
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
