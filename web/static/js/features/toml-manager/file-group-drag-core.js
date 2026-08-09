/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { FILE_GROUP_DROP_TARGET_ATTR } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260809-nf4-v2';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
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

// File-group drag primitives (from former chunk 07).
export function eventTargetClosest(event, selector) {
        const target = event?.target;
        return target instanceof Element ? target.closest(selector) : null;
    }

function createFileGroupDragImage(payload) {
        const image = document.createElement('div');
        image.className = 'file-group-drag-image';
        image.textContent = payload.file || payload.groupId || '移动项目';
        document.body.appendChild(image);
        return image;
    }

export function removeFileGroupDragImage(image) {
        if (image?.parentNode) image.parentNode.removeChild(image);
    }

export function setFileGroupDragData(event, payload) {
        const data = payload.file || payload.groupId || payload.target || 'move';
        const transfer = event?.dataTransfer;
        if (!transfer) return;
        let image = null;
        try {
            transfer.setData('text/plain', data);
            transfer.setData('application/x-anima-file-group', JSON.stringify({
                target: payload.target || '',
                scope: payload.scope || '',
                file: payload.file || '',
                groupId: payload.groupId || '',
            }));
            transfer.effectAllowed = 'move';
            image = createFileGroupDragImage(payload);
            transfer.setDragImage(image, 12, 12);
        } catch (e) {
            /* 部分浏览器会限制 DataTransfer 写入；内存态拖拽仍可继续。 */
        } finally {
            if (image) window.setTimeout(() => removeFileGroupDragImage(image), 0);
        }
    }

export function canBeginFileGroupDrag(payload, disabled) {
        if (disabled || (payload.canDrag && !payload.canDrag())) {
            if (payload.blockedMessage) payload.blockedMessage();
            return false;
        }
        return true;
    }

export function beginFileGroupDrag(payload, handle) {
        datasetState.fileGroupDragState = payload;
        payload.sourceElement?.classList.add('file-group-dragging');
        handle?.classList.add('dragging');
    }

export function createFileGroupPointerDragImage(payload) {
        const image = createFileGroupDragImage(payload);
        image.classList.add('file-group-drag-image-pointer');
        return image;
    }

export function moveFileGroupPointerDragImage(image, x, y) {
        if (!image) return;
        image.style.left = `${x + 14}px`;
        image.style.top = `${y + 14}px`;
    }


export function fileGroupDropTargetPriority(node, position) {
    if (!(node instanceof Element)) return 50;
    if (node.matches('.dataset-preset-row, .toml-file-row-wrap')) return 0;
    if (position === 'before' || position === 'after') return 1;
    if (node.matches('summary') || node.tagName === 'SUMMARY') return 3;
    if (position === 'inside') return 4;
    return 2;
}

export function registerFileGroupDropTarget(node, resolve) {
        node.setAttribute(FILE_GROUP_DROP_TARGET_ATTR, '1');
        datasetState.fileGroupDropTargets.set(node, resolve);
        datasetState.fileGroupDropTargetNodes.add(node);
    }

    export function originClosest(origin, selector) {
        return origin instanceof Element ? origin.closest(selector) : null;
    }

    function resolveFileGroupPointerDropTarget(x, y) {
        const payload = currentFileGroupDragState();
        const origin = document.elementFromPoint(x, y);
        let node = origin;
        let best = null;
        while (node && node !== document.documentElement) {
            if (node instanceof Element && node.hasAttribute(FILE_GROUP_DROP_TARGET_ATTR)) {
                const resolve = currentFileGroupDropTargets().get(node);
                const target = resolve?.({ payload, x, y, origin });
                if (target) {
                    const resolvedNode = target.node instanceof Element ? target.node : node;
                    const candidate = { ...target, node: resolvedNode, distance: 0 };
                    const priority = fileGroupDropTargetPriority(candidate.node, candidate.position);
                    if (!best || priority < best.priority) {
                        best = { ...candidate, priority };
                    }
                    // 精确行命中时直接返回，避免再被祖先 list/header 覆盖。
                    if (priority === 0) {
                        const { priority: _priority, distance: _distance, ...result } = best;
                        return result;
                    }
                }
            }
            node = node.parentElement;
        }
        if (best) {
            const { priority: _priority, distance: _distance, ...result } = best;
            return result;
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
            const resolvedNode = target.node instanceof Element ? target.node : node;
            const priority = fileGroupDropTargetPriority(resolvedNode, target.position);
            const candidate = { ...target, node: resolvedNode, distance, priority };
            if (
                !best
                || candidate.priority < best.priority
                || (candidate.priority === best.priority && candidate.distance < best.distance)
            ) {
                best = candidate;
            }
        }
        if (!best) return null;
        const { distance, priority, ...target } = best;
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
        // 先挂 body，真正定位时会挪到父列表（与历史任务预览一致）。
        document.body.appendChild(preview);
        datasetState.fileGroupDropPreviewElement = preview;
        return preview;
    }

    function placeFileGroupDropPreview(node, position) {
        if (!node || position !== 'before' && position !== 'after') {
            removeFileGroupDropPreview();
            return;
        }
        // 对齐历史任务：预览挂到父列表，用 offsetTop，滚动时也不会和指示线错位。
        const parent = node.parentElement;
        if (!parent) {
            removeFileGroupDropPreview();
            return;
        }
        parent.classList.add('file-group-drop-host');
        const preview = ensureFileGroupDropPreview();
        const placement = position === 'before' ? 'before' : 'after';
        const parentStyle = window.getComputedStyle(parent);
        const gap = Number.parseFloat(parentStyle.rowGap || parentStyle.gap || '0') || 0;
        const top = placement === 'before'
            ? Math.max(0, node.offsetTop - (gap / 2))
            : node.offsetTop + node.offsetHeight + (gap / 2);
        preview.dataset.position = placement;
        preview.style.left = '';
        preview.style.width = '';
        preview.style.top = `${top}px`;
        if (preview.parentElement !== parent) parent.appendChild(preview);
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

    function finishFileGroupPointerDrag(commit = false, event = null) {
        const drag = cleanupFileGroupPointerDrag();
        if (!drag) return;
        let target = null;
        if (commit && drag.active) {
            // 松手时按最终坐标重算，避免 currentDrop 停在同组较早的旧行上。
            const finalX = Number.isFinite(event?.clientX) ? event.clientX : null;
            const finalY = Number.isFinite(event?.clientY) ? event.clientY : null;
            if (finalX !== null && finalY !== null) {
                target = resolveFileGroupPointerDropTarget(finalX, finalY);
            } else {
                target = drag.currentDrop;
            }
        }
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
            finishFileGroupPointerDrag(true, upEvent);
        };
        drag.onMouseMove = (moveEvent) => {
            moveDrag(moveEvent);
        };
        drag.onMouseUp = (upEvent) => {
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishFileGroupPointerDrag(true, upEvent);
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

export function clearFileGroupDropTarget(node) {
    node?.classList.remove('file-group-drop-before', 'file-group-drop-after', 'file-group-drop-inside');
    if (!node || currentFileGroupActiveDropTargetNode() === node) {
        clearFileGroupDropIndicators();
    }
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


export function moveFileNearList(list, sourceValue, targetValue, position = 'after') {
    // 同组：source 已在 list，先摘再插。
    // 跨组：source 不在目标 list，直接插到 anchor 旁（不能 early-return 成 no-op）。
    const out = [];
    const seen = new Set();
    for (const raw of list || []) {
        const value = String(raw || '').trim();
        if (!value || seen.has(value)) continue;
        seen.add(value);
        out.push(value);
    }
    const source = String(sourceValue || '').trim();
    const target = String(targetValue || '').trim();
    if (!source) return out;
    const original = [...out];
    const sourceIndex = out.indexOf(source);
    if (sourceIndex >= 0) out.splice(sourceIndex, 1);
    let index = out.length;
    if (target) {
        const targetIndex = out.indexOf(target);
        index = targetIndex < 0 ? out.length : targetIndex + (position === 'after' ? 1 : 0);
    }
    out.splice(Math.max(0, Math.min(out.length, index)), 0, source);
    return out.length === original.length && out.every((value, idx) => value === original[idx]) ? original : out;
}

export function configFileDropIndex(group, targetFile, placeAfter, draggedFile) {
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
