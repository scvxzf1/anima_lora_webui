/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { FILE_GROUP_DROP_TARGET_ATTR } from '../helpers/app-constants.js?v=module-bootstrap-20260707-93';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import {
    beginFileGroupDrag,
    canBeginFileGroupDrag,
    createFileGroupPointerDragImage,
    eventTargetClosest,
    moveFileGroupPointerDragImage,
    registerFileGroupDropTarget,
    removeFileGroupDragImage,
    setFileGroupDragData,
} from './07-render-config-dataset-picker-dialog.js?v=module-bootstrap-20260707-93';

const datasetState = getDatasetState();

function currentFileGroupDragState() {
    return datasetState.fileGroupDragState || null;
}

function currentFileGroupPointerDrag() {
    return datasetState.fileGroupPointerDrag || null;
}

function currentFileGroupDropPreviewElement() {
    return datasetState.fileGroupDropPreviewElement || null;
}

function currentFileGroupDropTargets() {
    return datasetState.fileGroupDropTargets;
}

function currentFileGroupDropTargetNodes() {
    return datasetState.fileGroupDropTargetNodes;
}

function currentFileGroupActiveDropTargetNode() {
    return datasetState.fileGroupActiveDropTargetNode || null;
}

function currentFileGroupActiveDropPosition() {
    return datasetState.fileGroupActiveDropPosition || '';
}

    function originClosest(origin, selector) {
        return origin instanceof Element ? origin.closest(selector) : null;
    }

    function resolveFileGroupPointerDropTarget(x, y) {
        const payload = currentFileGroupDragState();
        const origin = document.elementFromPoint(x, y);
        let node = origin;
        while (node && node !== document.documentElement) {
            if (node instanceof Element && node.hasAttribute(FILE_GROUP_DROP_TARGET_ATTR)) {
                const resolve = currentFileGroupDropTargets().get(node);
                const target = resolve?.({ payload, x, y, origin });
                if (target) return { node, ...target };
            }
            node = node.parentElement;
        }
        return resolveNearestFileGroupDropTarget(x, y, origin, payload);
    }

    function resolveNearestFileGroupDropTarget(x, y, origin, payload) {
        let best = null;
        const dropTargetNodes = currentFileGroupDropTargetNodes();
        for (const node of dropTargetNodes) {
            if (!node?.isConnected || !(node instanceof Element)) {
                dropTargetNodes.delete(node);
                continue;
            }
            const rect = node.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) continue;
            const maxDistance = Math.max(26, Math.min(90, rect.height * 0.85));
            const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
            const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
            const distance = Math.hypot(dx, dy);
            if (distance > maxDistance) continue;
            const resolve = currentFileGroupDropTargets().get(node);
            const target = resolve?.({ payload, x, y, origin });
            if (!target) continue;
            if (!best || distance < best.distance) best = { node, distance, ...target };
        }
        if (!best) return null;
        const { distance, ...target } = best;
        return target;
    }

    function markResolvedFileGroupDropTarget(target) {
        if (!target) {
            clearFileGroupDropIndicators();
            return;
        }
        if (target.position === 'before' || target.position === 'after') {
            target.node.dataset.dropPosition = target.position;
        }
        markFileGroupDropTarget(target.node, target.position);
    }

    function removeFileGroupDropPreview() {
        const preview = currentFileGroupDropPreviewElement();
        if (preview?.parentNode) {
            preview.parentNode.removeChild(preview);
        }
        datasetState.fileGroupDropPreviewElement = null;
    }

    function ensureFileGroupDropPreview() {
        const existingPreview = currentFileGroupDropPreviewElement();
        if (existingPreview?.isConnected) return existingPreview;
        const preview = document.createElement('div');
        preview.className = 'file-group-drop-preview';
        preview.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.textContent = '释放后插入到这里';
        preview.appendChild(label);
        document.body.appendChild(preview);
        datasetState.fileGroupDropPreviewElement = preview;
        return preview;
    }

    function placeFileGroupDropPreview(node, position) {
        if (!node || position !== 'before' && position !== 'after') {
            removeFileGroupDropPreview();
            return;
        }
        const rect = node.getBoundingClientRect?.();
        if (!rect || rect.width <= 0 || rect.height <= 0) {
            removeFileGroupDropPreview();
            return;
        }
        const preview = ensureFileGroupDropPreview();
        preview.dataset.position = position;
        preview.style.left = `${rect.left + 4}px`;
        preview.style.top = `${position === 'before' ? rect.top : rect.bottom}px`;
        preview.style.width = `${Math.max(40, rect.width - 8)}px`;
    }

    function findScrollableFileGroupAncestor(origin) {
        let node = origin instanceof Element ? origin : null;
        while (node && node !== document.body) {
            const style = window.getComputedStyle(node);
            if (/(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight) {
                return node;
            }
            node = node.parentElement;
        }
        return document.scrollingElement;
    }

    export function autoScrollFileGroupPointerDrag(x, y) {
        const origin = document.elementFromPoint(x, y);
        const scroller = findScrollableFileGroupAncestor(origin);
        if (!scroller) return;
        const rect = scroller === document.scrollingElement
            ? { top: 0, bottom: window.innerHeight }
            : scroller.getBoundingClientRect();
        const margin = 46;
        const speed = 16;
        let delta = 0;
        if (y < rect.top + margin) {
            delta = -speed;
        } else if (y > rect.bottom - margin) {
            delta = speed;
        }
        if (delta) scroller.scrollBy({ top: delta, behavior: 'auto' });
    }

    function cleanupFileGroupPointerDrag() {
        const drag = currentFileGroupPointerDrag();
        if (!drag) return null;
        document.removeEventListener('pointermove', drag.onMove);
        document.removeEventListener('pointerup', drag.onUp);
        document.removeEventListener('pointercancel', drag.onCancel);
        document.removeEventListener('mousemove', drag.onMouseMove);
        document.removeEventListener('mouseup', drag.onMouseUp);
        document.removeEventListener('keydown', drag.onKeydown);
        try {
            if (drag.pointerId !== null && drag.pointerId !== undefined) {
                drag.handle.releasePointerCapture?.(drag.pointerId);
            }
        } catch (e) {
            /* 指针可能已被浏览器释放，忽略即可。 */
        }
        removeFileGroupDragImage(drag.image);
        document.body.classList.remove('file-group-pointer-drag-active');
        drag.handle?.classList.remove('dragging');
        datasetState.fileGroupPointerDrag = null;
        return drag;
    }

    function finishFileGroupPointerDrag(commit = false) {
        const drag = cleanupFileGroupPointerDrag();
        if (!drag) return;
        const target = commit && drag.active ? drag.currentDrop : null;
        finishFileGroupDrag();
        if (!target) return;
        Promise.resolve(target.drop()).catch((e) => {
            console.error('拖拽位置更新失败', e);
        });
    }

    function startFileGroupFallbackDrag(event, payload, handle, disabled, options = {}) {
        if (disabled || currentFileGroupPointerDrag()) return;
        if ((options.pointer || options.mouse) && 'button' in event && event.button !== 0) return;
        if (options.pointer && event.isPrimary === false) return;
        event.preventDefault();
        event.stopPropagation();
        const pointerId = options.pointer ? event.pointerId : null;
        const drag = {
            payload,
            handle,
            pointerId,
            startX: event.clientX,
            startY: event.clientY,
            active: false,
            image: null,
            currentDrop: null,
        };
        const moveDrag = (moveEvent) => {
            const distance = Math.hypot(moveEvent.clientX - drag.startX, moveEvent.clientY - drag.startY);
            if (!drag.active) {
                if (distance < 4) return;
                if (!canBeginFileGroupDrag(payload, disabled)) {
                    finishFileGroupPointerDrag(false);
                    return;
                }
                beginFileGroupDrag(payload, handle);
                drag.active = true;
                drag.image = createFileGroupPointerDragImage(payload);
                document.body.classList.add('file-group-pointer-drag-active');
            }
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            moveFileGroupPointerDragImage(drag.image, moveEvent.clientX, moveEvent.clientY);
            autoScrollFileGroupPointerDrag(moveEvent.clientX, moveEvent.clientY);
            drag.currentDrop = resolveFileGroupPointerDropTarget(moveEvent.clientX, moveEvent.clientY);
            markResolvedFileGroupDropTarget(drag.currentDrop);
        };
        drag.onMove = (moveEvent) => {
            if (moveEvent.pointerId !== pointerId) return;
            moveDrag(moveEvent);
        };
        drag.onUp = (upEvent) => {
            if (upEvent.pointerId !== pointerId) return;
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishFileGroupPointerDrag(true);
        };
        drag.onMouseMove = (moveEvent) => {
            moveDrag(moveEvent);
        };
        drag.onMouseUp = (upEvent) => {
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishFileGroupPointerDrag(true);
        };
        drag.onCancel = (cancelEvent) => {
            if (cancelEvent.pointerId !== pointerId) return;
            finishFileGroupPointerDrag(false);
        };
        drag.onKeydown = (keyEvent) => {
            if (keyEvent.key === 'Escape') finishFileGroupPointerDrag(false);
        };
        datasetState.fileGroupPointerDrag = drag;
        const addMouseFallbackListeners = () => {
            document.addEventListener('mousemove', drag.onMouseMove, { passive: false });
            document.addEventListener('mouseup', drag.onMouseUp, { passive: false });
        };
        if (options.pointer) {
            try {
                handle.setPointerCapture?.(pointerId);
            } catch (e) {
                /* 浏览器可能已切换到原生拖拽流程，继续使用文档级监听兜底。 */
            }
            document.addEventListener('pointermove', drag.onMove, { passive: false });
            document.addEventListener('pointerup', drag.onUp, { passive: false });
            document.addEventListener('pointercancel', drag.onCancel, { passive: false });
            addMouseFallbackListeners();
        } else {
            addMouseFallbackListeners();
        }
        document.addEventListener('keydown', drag.onKeydown);
    }

    function startFileGroupPointerDrag(event, payload, handle, disabled) {
        startFileGroupFallbackDrag(event, payload, handle, disabled, { pointer: true });
    }

    function startFileGroupMouseDrag(event, payload, handle, disabled) {
        startFileGroupFallbackDrag(event, payload, handle, disabled, { mouse: true });
    }

    export function markFileGroupDropTarget(node, position) {
        if (!node || !position) {
            clearFileGroupDropIndicators();
            return;
        }
        const normalizedPosition = position === 'before' || position === 'after' ? position : 'inside';
        if (currentFileGroupActiveDropTargetNode() === node
            && currentFileGroupActiveDropPosition() === normalizedPosition) {
            node.classList.add(`file-group-drop-${normalizedPosition}`);
            placeFileGroupDropPreview(node, normalizedPosition);
            return;
        }
        clearFileGroupDropIndicators({ keepPreview: true });
        datasetState.fileGroupActiveDropTargetNode = node;
        datasetState.fileGroupActiveDropPosition = normalizedPosition;
        node.classList.add(`file-group-drop-${normalizedPosition}`);
        placeFileGroupDropPreview(node, normalizedPosition);
    }

    export function createFileGroupDragHandle(payload, options = {}) {
        const handle = document.createElement('button');
        const disabled = Boolean(options.disabled);
        handle.type = 'button';
        handle.className = ['file-group-drag-handle', disabled ? 'disabled' : ''].filter(Boolean).join(' ');
        handle.setAttribute('aria-label', options.label || '拖动调整位置');
        handle.title = options.title || '拖动调整位置';
        handle.textContent = '⋮⋮';
        handle.tabIndex = disabled ? -1 : 0;
        handle.draggable = !disabled;
        handle.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
        });
        handle.addEventListener('pointerdown', (event) => startFileGroupPointerDrag(event, payload, handle, disabled));
        handle.addEventListener('mousedown', (event) => {
            event.stopPropagation();
            startFileGroupMouseDrag(event, payload, handle, disabled);
        });
        handle.addEventListener('dragstart', (event) => {
            if (currentFileGroupPointerDrag()) {
                event.preventDefault();
                event.stopPropagation();
                return;
            }
            if (!canBeginFileGroupDrag(payload, disabled)) {
                event.preventDefault();
                return;
            }
            event.stopPropagation();
            beginFileGroupDrag(payload, handle);
            setFileGroupDragData(event, payload);
        });
        handle.addEventListener('dragend', () => {
            handle.classList.remove('dragging');
            finishFileGroupDrag();
        });
        return handle;
    }

    export function finishFileGroupDrag() {
        currentFileGroupDragState()?.sourceElement?.classList.remove('file-group-dragging');
        datasetState.fileGroupDragState = null;
        clearFileGroupDropIndicators();
    }

    function clearFileGroupDropIndicators(options = {}) {
        if (!options.keepPreview) {
            removeFileGroupDropPreview();
        }
        document.querySelectorAll('.file-group-drop-before, .file-group-drop-after, .file-group-drop-inside').forEach((node) => {
            node.classList.remove('file-group-drop-before', 'file-group-drop-after', 'file-group-drop-inside');
        });
        datasetState.fileGroupActiveDropTargetNode = null;
        datasetState.fileGroupActiveDropPosition = '';
    }

    function configFileDropIndex(group, targetFile, placeAfter, draggedFile) {
        const files = (group?.files || [])
            .map((item) => item?.path)
            .filter((path) => path && path !== draggedFile);
        const targetIndex = files.indexOf(targetFile);
        if (targetIndex < 0) return files.length;
        return targetIndex + (placeAfter ? 1 : 0);
    }

    export function configGroupDropIndex(groups, targetGroupId, placeAfter, draggedGroupId) {
        const ids = (groups || [])
            .map((group) => group?.id)
            .filter((id) => id && id !== draggedGroupId);
        const targetIndex = ids.indexOf(targetGroupId);
        if (targetIndex < 0) return ids.length;
        return targetIndex + (placeAfter ? 1 : 0);
    }

    export function fileGroupContainsRelatedTarget(node, event) {
        const related = event?.relatedTarget;
        return related instanceof Node && node.contains(related);
    }

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
            row.classList.remove('file-group-drop-before', 'file-group-drop-after');
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
            list.classList.remove('file-group-drop-inside');
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
            node.classList.remove('file-group-drop-inside');
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
