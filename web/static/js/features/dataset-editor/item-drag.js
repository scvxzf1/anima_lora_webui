/**
 * Dataset editor item drag/drop and list item shell.
 * Extracted from former chunk 10 / config-input.
 */
import { datasetEditorStateForActivePanel } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { normalizeDatasetEditorRows } from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260714-stage-dataset5';
import { compactPathLabel } from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    autoScrollFileGroupPointerDrag,
    fileGroupContainsRelatedTarget,
} from '../toml-manager/file-group-drag.js?v=module-bootstrap-20260714-stage-dataset5';
import { createDatasetEditorRow } from './row.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { renderDatasetEditor } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { moveDatasetEditorRow, moveDatasetEditorRowToIndex } from './mutations.js?v=module-bootstrap-20260714-stage-dataset5';

const datasetState = getDatasetState();
let datasetEditorDragState = null;
// Pointer/touch fallback drag session. Module-local (same pattern as datasetEditorDragState);
// dataset-state.js still has a leftover null field from the split, unused elsewhere.
let datasetEditorPointerDrag = null;

function datasetEditorDragRows() {
            return normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
        }

function datasetEditorCanDrag() {
            return datasetEditorDragRows().length > 1;
        }

function datasetEditorDragLabel(index) {
            const row = datasetEditorDragRows()[index] || {};
            const path = String(row.source_dir || row.image_dir || '').trim();
            return path ? compactPathLabel(path) : `SUBSET ${index + 1}`;
        }

function datasetEditorDragFullPath(index) {
            const row = datasetEditorDragRows()[index] || {};
            return String(row.source_dir || row.image_dir || '').trim();
        }

function createDatasetEditorDragImage(index) {
            const image = document.createElement('div');
            image.className = 'dataset-editor-drag-image';
            const fullPath = datasetEditorDragFullPath(index);
            image.textContent = datasetEditorDragLabel(index);
            if (fullPath) image.title = fullPath;
            document.body.appendChild(image);
            return image;
        }

function removeDatasetEditorDragImage(image) {
            if (image?.parentNode) image.parentNode.removeChild(image);
        }

function moveDatasetEditorDragImage(image, x, y) {
            if (!image) return;
            image.style.left = `${x + 14}px`;
            image.style.top = `${y + 14}px`;
        }

function beginDatasetEditorDrag(index, item, handle) {
            datasetEditorDragState = { index, sourceElement: item, handle };
            item?.classList.add('dataset-editor-item-dragging');
            handle?.classList.add('dragging');
            document.body.classList.add('dataset-editor-pointer-drag-active');
        }

function clearDatasetEditorDropIndicators() {
            document.querySelectorAll('.dataset-editor-drop-before, .dataset-editor-drop-after').forEach((node) => {
                node.classList.remove('dataset-editor-drop-before', 'dataset-editor-drop-after');
            });
        }

function finishDatasetEditorDrag() {
            datasetEditorDragState?.sourceElement?.classList.remove('dataset-editor-item-dragging');
            datasetEditorDragState?.handle?.classList.remove('dragging');
            document.body.classList.remove('dataset-editor-pointer-drag-active');
            datasetEditorDragState = null;
            clearDatasetEditorDropIndicators();
        }

function datasetEditorDropTargetFromPoint(x, y) {
            const items = [...document.querySelectorAll('#dataset-editor .dataset-editor-item')];
            if (!items.length) return null;
            let best = null;
            for (const item of items) {
                const targetIndex = Number.parseInt(item.dataset.index || '-1', 10);
                if (!Number.isInteger(targetIndex) || targetIndex < 0) continue;
                if (targetIndex === datasetEditorDragState?.index) continue;
                const rect = item.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) continue;
                const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
                const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
                const distance = Math.hypot(dx, dy);
                if (distance > Math.max(30, rect.height * 0.8)) continue;
                const placeAfter = y > rect.top + rect.height / 2;
                if (!best || distance < best.distance) {
                    best = { node: item, targetIndex, placeAfter, distance };
                }
            }
            return best;
        }

function markDatasetEditorDropTarget(target) {
            clearDatasetEditorDropIndicators();
            if (!target?.node) return;
            target.node.classList.add(target.placeAfter ? 'dataset-editor-drop-after' : 'dataset-editor-drop-before');
        }

function datasetEditorEventPoint(event) {
            const touch = event.changedTouches?.[0] || event.touches?.[0];
            const x = touch?.clientX ?? event.clientX;
            const y = touch?.clientY ?? event.clientY;
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            return { x, y };
        }

function finishDatasetEditorPointerDrag(commit = false) {
            const drag = datasetEditorPointerDrag;
            if (!drag) return;
            document.removeEventListener('pointermove', drag.onPointerMove);
            document.removeEventListener('pointerup', drag.onPointerUp);
            document.removeEventListener('pointercancel', drag.onPointerCancel);
            document.removeEventListener('mousemove', drag.onMouseMove);
            document.removeEventListener('mouseup', drag.onMouseUp);
            document.removeEventListener('touchmove', drag.onTouchMove);
            document.removeEventListener('touchend', drag.onTouchEnd);
            document.removeEventListener('touchcancel', drag.onTouchCancel);
            document.removeEventListener('keydown', drag.onKeydown);
            try {
                if (drag.pointerId !== null && drag.pointerId !== undefined) {
                    drag.handle.releasePointerCapture?.(drag.pointerId);
                }
            } catch (e) {
                /* 指针捕获可能已被浏览器释放，继续清理拖拽态。 */
            }
            removeDatasetEditorDragImage(drag.image);
            const target = commit && drag.active ? drag.currentDrop : null;
            datasetEditorPointerDrag = null;
            finishDatasetEditorDrag();
            if (target) {
                moveDatasetEditorRow(drag.index, target.targetIndex, target.placeAfter);
            }
        }

function startDatasetEditorFallbackDrag(event, index, item, handle, options = {}) {
            if (!datasetEditorCanDrag() || datasetEditorPointerDrag) return;
            if ((options.pointer || options.mouse) && 'button' in event && event.button !== 0) return;
            if (options.pointer && event.isPrimary === false) return;
            const startPoint = datasetEditorEventPoint(event);
            if (!startPoint) return;
            event.preventDefault();
            event.stopPropagation();
            const pointerId = options.pointer ? event.pointerId : null;
            const drag = {
                index,
                item,
                handle,
                pointerId,
                startX: startPoint.x,
                startY: startPoint.y,
                active: false,
                image: null,
                currentDrop: null,
            };
            const moveDrag = (moveEvent) => {
                const point = datasetEditorEventPoint(moveEvent);
                if (!point) return;
                const distance = Math.hypot(point.x - drag.startX, point.y - drag.startY);
                if (!drag.active) {
                    if (distance < 4) return;
                    beginDatasetEditorDrag(index, item, handle);
                    drag.active = true;
                    drag.image = createDatasetEditorDragImage(index);
                }
                moveEvent.preventDefault();
                moveEvent.stopPropagation();
                moveDatasetEditorDragImage(drag.image, point.x, point.y);
                autoScrollFileGroupPointerDrag(point.x, point.y);
                drag.currentDrop = datasetEditorDropTargetFromPoint(point.x, point.y);
                markDatasetEditorDropTarget(drag.currentDrop);
            };
            drag.onPointerMove = (moveEvent) => {
                if (moveEvent.pointerId !== pointerId) return;
                moveDrag(moveEvent);
            };
            drag.onPointerUp = (upEvent) => {
                if (upEvent.pointerId !== pointerId) return;
                upEvent.preventDefault();
                upEvent.stopPropagation();
                finishDatasetEditorPointerDrag(true);
            };
            drag.onPointerCancel = (cancelEvent) => {
                if (cancelEvent.pointerId !== pointerId) return;
                finishDatasetEditorPointerDrag(false);
            };
            drag.onMouseMove = (moveEvent) => moveDrag(moveEvent);
            drag.onMouseUp = (upEvent) => {
                upEvent.preventDefault();
                upEvent.stopPropagation();
                finishDatasetEditorPointerDrag(true);
            };
            drag.onTouchMove = (moveEvent) => moveDrag(moveEvent);
            drag.onTouchEnd = (touchEvent) => {
                touchEvent.preventDefault();
                touchEvent.stopPropagation();
                finishDatasetEditorPointerDrag(true);
            };
            drag.onTouchCancel = () => finishDatasetEditorPointerDrag(false);
            drag.onKeydown = (keyEvent) => {
                if (keyEvent.key === 'Escape') finishDatasetEditorPointerDrag(false);
            };
            datasetEditorPointerDrag = drag;
            if (options.pointer) {
                try {
                    handle.setPointerCapture?.(pointerId);
                } catch (e) {
                    /* 某些浏览器禁用按钮指针捕获，文档级监听仍可兜底。 */
                }
                document.addEventListener('pointermove', drag.onPointerMove, { passive: false });
                document.addEventListener('pointerup', drag.onPointerUp, { passive: false });
                document.addEventListener('pointercancel', drag.onPointerCancel, { passive: false });
            } else if (options.touch) {
                document.addEventListener('touchmove', drag.onTouchMove, { passive: false });
                document.addEventListener('touchend', drag.onTouchEnd, { passive: false });
                document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false });
            } else {
                document.addEventListener('mousemove', drag.onMouseMove, { passive: false });
                document.addEventListener('mouseup', drag.onMouseUp, { passive: false });
            }
            document.addEventListener('keydown', drag.onKeydown);
        }

function startDatasetEditorPointerDrag(event, index, item, handle) {
            startDatasetEditorFallbackDrag(event, index, item, handle, { pointer: true });
        }

function startDatasetEditorMouseDrag(event, index, item, handle) {
            startDatasetEditorFallbackDrag(event, index, item, handle, { mouse: true });
        }

function startDatasetEditorTouchDrag(event, index, item, handle) {
            startDatasetEditorFallbackDrag(event, index, item, handle, { touch: true });
        }

export function createDatasetEditorDragHandle(index, item) {
            const handle = document.createElement('button');
            const disabled = !datasetEditorCanDrag();
            handle.type = 'button';
            handle.className = ['dataset-editor-drag-handle', disabled ? 'disabled' : ''].filter(Boolean).join(' ');
            handle.textContent = '⋮⋮';
            handle.title = disabled ? '至少两组数据集时可以拖动排序' : '拖动排序；也可用 Alt+方向键移动';
            handle.setAttribute('aria-label', `拖动排序第 ${index + 1} 组数据集`);
            handle.draggable = !disabled;
            handle.tabIndex = disabled ? -1 : 0;
            handle.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
            });
            handle.addEventListener('keydown', (event) => {
                if (disabled || !event.altKey || !['ArrowUp', 'ArrowDown'].includes(event.key)) return;
                event.preventDefault();
                const targetIndex = event.key === 'ArrowUp' ? index - 1 : index + 1;
                moveDatasetEditorRowToIndex(index, targetIndex);
            });
            handle.addEventListener('pointerdown', (event) => startDatasetEditorPointerDrag(event, index, item, handle));
            handle.addEventListener('mousedown', (event) => {
                event.stopPropagation();
                startDatasetEditorMouseDrag(event, index, item, handle);
            });
            handle.addEventListener('touchstart', (event) => {
                event.stopPropagation();
                startDatasetEditorTouchDrag(event, index, item, handle);
            }, { passive: false });
            handle.addEventListener('dragstart', (event) => {
                if (datasetEditorPointerDrag) finishDatasetEditorPointerDrag(false);
                if (disabled) {
                    event.preventDefault();
                    return;
                }
                event.stopPropagation();
                beginDatasetEditorDrag(index, item, handle);
                if (event.dataTransfer) {
                    try {
                        event.dataTransfer.setData('text/plain', String(index));
                        event.dataTransfer.setData('application/x-anima-dataset-row', String(index));
                        event.dataTransfer.effectAllowed = 'move';
                        const image = createDatasetEditorDragImage(index);
                        event.dataTransfer.setDragImage(image, 12, 12);
                        window.setTimeout(() => removeDatasetEditorDragImage(image), 0);
                    } catch (e) {
                        /* 原生 DataTransfer 失败时，pointer 兜底仍可完成排序。 */
                    }
                }
            });
            handle.addEventListener('dragend', () => finishDatasetEditorDrag());
            return handle;
        }

function setupDatasetEditorItemDropTarget(item, targetIndex) {
            const updateDropTarget = (event) => {
                const sourceIndex = datasetEditorDragState?.index;
                if (!Number.isInteger(sourceIndex) || sourceIndex === targetIndex) return;
                event.preventDefault();
                event.stopPropagation();
                if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
                const rect = item.getBoundingClientRect();
                const placeAfter = event.clientY > rect.top + rect.height / 2;
                item.dataset.dropPosition = placeAfter ? 'after' : 'before';
                markDatasetEditorDropTarget({ node: item, placeAfter });
            };
            item.addEventListener('dragenter', updateDropTarget);
            item.addEventListener('dragover', updateDropTarget);
            item.addEventListener('dragleave', (event) => {
                if (fileGroupContainsRelatedTarget(item, event)) return;
                item.classList.remove('dataset-editor-drop-before', 'dataset-editor-drop-after');
            });
            item.addEventListener('drop', (event) => {
                const sourceIndex = datasetEditorDragState?.index;
                if (!Number.isInteger(sourceIndex) || sourceIndex === targetIndex) return;
                event.preventDefault();
                event.stopPropagation();
                const placeAfter = item.dataset.dropPosition === 'after';
                finishDatasetEditorDrag();
                moveDatasetEditorRow(sourceIndex, targetIndex, placeAfter);
            });
        }

export function createDatasetEditorItem(row, index) {
            const item = document.createElement('div');
            item.className = 'dataset-editor-item';
            item.dataset.index = String(index);
            const datasetState = getDatasetState();
            item.classList.toggle('is-selected', index === datasetState.selectedDatasetIndex);
            setupDatasetEditorItemDropTarget(item, index);
            item.addEventListener('click', (event) => {
                if (event.target.closest('button, input, select, textarea, a, label, summary, .dataset-editor-drag-handle')) return;
                if (datasetState.selectedDatasetIndex === index) return;
                datasetState.selectedDatasetIndex = index;
                renderDatasetEditor();
            });
            item.append(createDatasetEditorRow(row, index, item));
            return item;
        }
