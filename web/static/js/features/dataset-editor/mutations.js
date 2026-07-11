/**
 * Dataset editor row mutations and experimental scope helpers.
 * Extracted from anima-app chunk 13.
 */
import { DEFAULT_TRIGGER_CLONE } from '../../config/catalog.js?v=module-bootstrap-20260711-ir6';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
    normalizeNlTagMix,
    normalizeTriggerClone,
} from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260711-ir6';
import {
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    renderDatasetPresetHeader,
} from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir6';
import { setFieldInputValue } from '../config-form/field-input.js?v=module-bootstrap-20260711-ir6';
import { updateTomlDirtyState } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir6';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { updateStepEstimatePanel } from '../config-form/step-estimate.js?v=module-bootstrap-20260711-ir6';

const datasetState = getDatasetState();

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

function datasetExperimentalScopeSelectionsState() {
    return datasetState.datasetExperimentalScopeSelections;
}

export function updateDatasetEditorRowsSettingValue(indices, key, value, options = {}) {
    const state = datasetEditorStateForActivePanel();
    const rows = normalizeDatasetEditorRows(state.datasets);
    const targets = datasetValidTargetIndices(indices, rows.length);
    if (!targets.length) return;
    for (const targetIndex of targets) {
        const settings = normalizeDatasetDefaults(rows[targetIndex].settings || state.defaults || {});
        settings[key] = value;
        rows[targetIndex].settings = settings;
    }
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.datasets = rows;
    } else {
        datasetState.datasetEditorState.datasets = rows;
    }
    markDatasetEditorDirty();
    if (options.render === 'item') {
        refreshDatasetEditorItems(targets) || renderDatasetEditor();
    } else if (options.render) {
        renderDatasetEditor();
    }
}

export function updateDatasetEditorRowNlTagMix(index, nextMix, options = {}) {
    updateDatasetEditorRowsNlTagMix([index], nextMix, options);
}

export function updateDatasetEditorRowsNlTagMix(indices, nextMix, options = {}) {
    const state = datasetEditorStateForActivePanel();
    const rows = normalizeDatasetEditorRows(state.datasets);
    const targets = datasetValidTargetIndices(indices, rows.length);
    if (!targets.length) return;
    const mix = normalizeNlTagMix(nextMix);
    for (const targetIndex of targets) {
        rows[targetIndex].nl_tag_mix = mix;
    }
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.datasets = rows;
    } else {
        datasetState.datasetEditorState.datasets = rows;
    }
    markDatasetEditorDirty();
    if (options.render === false) {
        return;
    }
    refreshDatasetEditorItems(targets) || renderDatasetEditor();
}

export function updateDatasetEditorRowTriggerClone(index, nextClone, options = {}) {
    const state = datasetEditorStateForActivePanel();
    const rows = normalizeDatasetEditorRows(state.datasets);
    if (!rows[index]) return;
    rows[index].trigger_clone = normalizeTriggerClone({
        ...rows[index].trigger_clone,
        ...nextClone,
    });
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.datasets = rows;
    } else {
        datasetState.datasetEditorState.datasets = rows;
    }
    markDatasetEditorDirty();
    if (options.render) {
        renderDatasetEditor();
    }
}

function datasetExperimentalScopeKey(index) {
    return `${isDatasetTabActive() ? 'dataset-preset' : 'config-dataset'}:${index}`;
}

export function datasetExperimentalScopeIndices(index, total = null) {
    const state = datasetEditorStateForActivePanel();
    const count = total ?? normalizeDatasetEditorRows(state.datasets).length;
    const key = datasetExperimentalScopeKey(index);
    const raw = datasetExperimentalScopeSelectionsState().get(key) || [index];
    const selected = datasetValidTargetIndices(raw, count);
    if (!selected.length && index >= 0 && index < count) {
        selected.push(index);
    }
    datasetExperimentalScopeSelectionsState().set(key, selected);
    return selected;
}

export function setDatasetExperimentalScopeIndices(index, indices) {
    const state = datasetEditorStateForActivePanel();
    const count = normalizeDatasetEditorRows(state.datasets).length;
    const selected = datasetValidTargetIndices(indices, count);
    if (!selected.length && index >= 0 && index < count) {
        selected.push(index);
    }
    datasetExperimentalScopeSelectionsState().set(datasetExperimentalScopeKey(index), selected);
}

export function datasetValidTargetIndices(indices, count) {
    return [...new Set((indices || [])
        .map((value) => Number.parseInt(value, 10))
        .filter((value) => Number.isInteger(value) && value >= 0 && value < count))]
        .sort((left, right) => left - right);
}

function setDatasetEditorRowsAfterSort(rows) {
    datasetExperimentalScopeSelectionsState().clear();
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.datasets = rows;
    } else {
        datasetState.datasetEditorState.datasets = rows;
    }
    markDatasetEditorDirty();
    renderDatasetEditor();
}

export function moveDatasetEditorRow(sourceIndex, targetIndex, placeAfter = false) {
    const rows = normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
    if (rows.length <= 1) return false;
    if (sourceIndex < 0 || sourceIndex >= rows.length || targetIndex < 0 || targetIndex >= rows.length) return false;
    if (sourceIndex === targetIndex) return false;
    let insertIndex = targetIndex + (placeAfter ? 1 : 0);
    if (sourceIndex < insertIndex) insertIndex -= 1;
    insertIndex = Math.max(0, Math.min(rows.length - 1, insertIndex));
    if (insertIndex === sourceIndex) return false;
    const [moved] = rows.splice(sourceIndex, 1);
    rows.splice(insertIndex, 0, moved);
    setDatasetEditorRowsAfterSort(rows);
    return true;
}

export function moveDatasetEditorRowToIndex(sourceIndex, targetIndex) {
    const rows = normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
    const clamped = Math.max(0, Math.min(rows.length - 1, targetIndex));
    if (clamped === sourceIndex) return false;
    return moveDatasetEditorRow(sourceIndex, clamped, clamped > sourceIndex);
}

export function markDatasetEditorDirty() {
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.dirty = true;
        datasetState.datasetPresetState.status = '有未保存的数据集修改';
        renderDatasetPresetHeader();
    } else {
        datasetState.datasetEditorState.dirty = true;
        updateTomlDirtyState();
        updateStepEstimatePanel();
    }
    const dirty = document.querySelector('#dataset-editor .dataset-editor-dirty');
    if (dirty) {
        dirty.classList.add('active');
        dirty.textContent = '有未保存的数据集修改';
    }
}

export function addDatasetEditorRow() {
    const state = datasetEditorStateForActivePanel();
    const rows = normalizeDatasetEditorRows(state.datasets);
    rows.push({
        source_dir: '',
        image_dir: '',
        cache_dir: '',
        num_repeats: 1,
        trigger_clone: normalizeTriggerClone(DEFAULT_TRIGGER_CLONE),
        settings: normalizeDatasetDefaults(state.defaults || {}),
    });
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.datasets = rows;
        datasetState.datasetPresetState.dirty = true;
    } else {
        datasetState.datasetEditorState.datasets = rows;
        datasetState.datasetEditorState.dirty = true;
    }
    renderDatasetEditor();
    if (!isDatasetTabActive()) {
        updateTomlDirtyState();
        updateStepEstimatePanel();
    }
}

export function removeDatasetEditorRow(index) {
    const state = datasetEditorStateForActivePanel();
    const rows = normalizeDatasetEditorRows(state.datasets);
    if (rows.length <= 1) return;
    rows.splice(index, 1);
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.datasets = rows;
        datasetState.datasetPresetState.dirty = true;
    } else {
        datasetState.datasetEditorState.datasets = rows;
        datasetState.datasetEditorState.dirty = true;
    }
    renderDatasetEditor();
    if (!isDatasetTabActive()) {
        updateTomlDirtyState();
        updateStepEstimatePanel();
    }
}

export function syncDatasetEditorToCompatFields() {
    const datasetEditorState = currentDatasetEditorState();
    const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
    const first = rows[0];
    if (!first) return;
    setFieldInputValue('source_image_dir', first.source_dir);
    setFieldInputValue('resized_image_dir', first.image_dir);
    setFieldInputValue('lora_cache_dir', first.cache_dir);
    if (datasetEditorState.dataset_config) {
        setFieldInputValue('dataset_config', datasetEditorState.dataset_config);
    }
}
