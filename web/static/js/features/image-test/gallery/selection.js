import { DEFAULT_SELECTION_NOTE } from './constants.js?v=module-bootstrap-20260711-ir1';
import { imageKey } from './image-meta.js?v=module-bootstrap-20260711-ir1';

/**
 * 选择状态 / 工具栏 / 批量删除。
 *
 * @param {object} deps
 * @param {object} deps.state
 * @param {Function|undefined} deps.requestImageDelete
 * @param {() => any[]} deps.selectedImagesInDisplayOrder
 */
export function createSelectionApi({
    state,
    requestImageDelete,
    selectedImagesInDisplayOrder,
}) {
    function toggleImageSelection(imageKeyValue, options = {}) {
        const next = options.additive ? new Set(state.selectedKeys) : new Set();
        if (options.shiftKey && state.lastSelectedKey) {
            const range = visibleSelectionRange(state.lastSelectedKey, imageKeyValue);
            if (range.length) {
                range.forEach((key) => next.add(key));
            } else {
                toggleSingleSelection(next, imageKeyValue, { additive: options.additive });
            }
        } else {
            toggleSingleSelection(next, imageKeyValue, { additive: options.additive });
        }
        state.selectedKeys = next;
        state.lastSelectedKey = imageKeyValue;
        setSelectionStatus('', '');
        syncCardSelectionStates();
        syncSelectionToolbar();
    }

    function toggleSingleSelection(selectedKeys, imageKeyValue, options = {}) {
        if (options.additive) {
            if (selectedKeys.has(imageKeyValue)) {
                selectedKeys.delete(imageKeyValue);
            } else {
                selectedKeys.add(imageKeyValue);
            }
            return;
        }
        if (state.selectedKeys.size === 1 && state.selectedKeys.has(imageKeyValue)) {
            selectedKeys.delete(imageKeyValue);
        } else {
            selectedKeys.clear();
            selectedKeys.add(imageKeyValue);
        }
    }

    function visibleSelectionRange(fromKey, toKey) {
        const start = state.visibleOrderedKeys.indexOf(fromKey);
        const end = state.visibleOrderedKeys.indexOf(toKey);
        if (start === -1 || end === -1) return [];
        const [from, to] = start < end ? [start, end] : [end, start];
        return state.visibleOrderedKeys.slice(from, to + 1);
    }

    function pruneSelectionToFilteredImages() {
        const allowed = new Set(state.filteredOrderedKeys);
        state.selectedKeys = new Set(
            [...state.selectedKeys].filter((key) => allowed.has(key)),
        );
        if (!allowed.has(state.lastSelectedKey)) {
            state.lastSelectedKey = '';
        }
    }

    function syncCardSelectionStates() {
        document.querySelectorAll('#image-test-grid [data-image-key]').forEach((card) => {
            const selected = state.selectedKeys.has(card.getAttribute('data-image-key') || '');
            card.classList.toggle('is-selected', selected);
            card.setAttribute('aria-checked', selected ? 'true' : 'false');
        });
    }

    function clearSelection() {
        state.selectedKeys = new Set();
        state.lastSelectedKey = '';
        setSelectionStatus('', '');
        syncCardSelectionStates();
        syncSelectionToolbar();
    }

    function syncSelectionToolbar() {
        const toolbar = document.getElementById('image-test-selection-toolbar');
        const summary = document.getElementById('image-test-selection-summary');
        const note = document.getElementById('image-test-selection-note');
        const exportButton = document.getElementById('btn-image-test-export-merged');
        const exportOriginalsButton = document.getElementById('btn-image-test-export-originals');
        if (!toolbar || !summary || !note || !exportButton || !exportOriginalsButton) return;

        const selectedCount = state.selectedKeys.size;
        toolbar.hidden = selectedCount <= 0;
        summary.textContent = `已选 ${selectedCount} 张`;
        note.textContent = state.selectionMessage || DEFAULT_SELECTION_NOTE;
        note.classList.remove('is-success', 'is-warning', 'is-error');
        if (state.selectionTone) {
            note.classList.add(`is-${state.selectionTone}`);
        }
        exportButton.disabled = selectedCount <= 0 || state.exportPending || state.rawExportPending;
        exportOriginalsButton.disabled = selectedCount <= 0 || state.exportPending || state.rawExportPending;
        exportButton.textContent = state.exportPending ? '正在导出...' : '导出合并图';
        exportOriginalsButton.textContent = state.rawExportPending ? '打包中...' : '打包原图';
    }

    function setSelectionStatus(message, tone = '') {
        state.selectionMessage = message;
        state.selectionTone = tone;
    }

    function syncCardDeleteStates() {
        document.querySelectorAll('.image-test-history-card[data-image-key]').forEach((card) => {
            const busy = state.deletePending || state.exportPending || state.rawExportPending;
            card.classList.toggle('is-busy', busy);
            const deleteBtn = card.querySelector('[data-image-action="delete"]');
            if (deleteBtn instanceof HTMLButtonElement) {
                deleteBtn.disabled = busy;
            }
        });
    }

    async function deleteSelectedImages() {
        await deleteImagesWithConfirmation(selectedImagesInDisplayOrder());
    }

    async function deleteImagesWithConfirmation(images) {
        const targets = dedupeImages(images);
        if (!targets.length || state.deletePending || state.exportPending || state.rawExportPending) {
            return null;
        }
        if (typeof requestImageDelete !== 'function') {
            const error = '当前页面未接入删图能力。';
            setSelectionStatus(error, 'error');
            syncSelectionToolbar();
            return { ok: false, error };
        }
        const confirmed = window.confirm(buildDeleteConfirmationMessage(targets));
        if (!confirmed) {
            return null;
        }
        let failed = false;
        state.deletePending = true;
        setSelectionStatus(`正在从硬盘永久删除 ${targets.length} 张图片...`, 'warning');
        syncSelectionToolbar();
        try {
            const result = await requestImageDelete(targets);
            if (Number(result?.deleted_count || 0) > 0) {
                removeDeletedImagesFromSelection(targets);
            }
            if (result?.ok === false && !Number(result?.deleted_count || 0)) {
                failed = true;
                setSelectionStatus(result.error || result.message || '删除失败。', 'error');
            }
            return result;
        } catch (error) {
            failed = true;
            setSelectionStatus(`删除失败：${error?.message || '无法删除图片。'}`, 'error');
            return { ok: false, error: error?.message || '无法删除图片。' };
        } finally {
            state.deletePending = false;
            if (!failed) {
                setSelectionStatus('', '');
            }
            syncSelectionToolbar();
        }
    }

    function dedupeImages(images) {
        const seen = new Set();
        const result = [];
        (Array.isArray(images) ? images : []).forEach((image) => {
            const key = imageKey(image);
            if (!key || seen.has(key)) return;
            seen.add(key);
            result.push(image);
        });
        return result;
    }

    function removeDeletedImagesFromSelection(images) {
        const deletedKeys = new Set(dedupeImages(images).map((image) => imageKey(image)));
        if (!deletedKeys.size) return;
        deletedKeys.forEach((key) => state.selectedKeys.delete(key));
        if (deletedKeys.has(state.lastSelectedKey)) {
            state.lastSelectedKey = '';
        }
        syncCardSelectionStates();
        syncSelectionToolbar();
    }

    function buildDeleteConfirmationMessage(images) {
        const count = Array.isArray(images) ? images.length : 0;
        return `确认从硬盘永久删除选中的 ${count} 张图片吗？此操作不可恢复。`;
    }

    return {
        toggleImageSelection,
        toggleSingleSelection,
        visibleSelectionRange,
        pruneSelectionToFilteredImages,
        syncCardSelectionStates,
        clearSelection,
        syncSelectionToolbar,
        setSelectionStatus,
        syncCardDeleteStates,
        deleteSelectedImages,
        deleteImagesWithConfirmation,
        dedupeImages,
        removeDeletedImagesFromSelection,
        buildDeleteConfirmationMessage,
    };
}
