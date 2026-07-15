/**
 * Dataset preset list shell and page summary.
 * Extracted from former chunk 07 / dataset dialog module.
 */
import {
    configureDatasetRenderBridge,
    createDatasetPresetGroupNode,
    readDatasetPresetGroupState,
    renderDatasetEditor,
} from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { datasetPresetGroupsForDisplay } from '../anima-app/helpers/dataset-presets.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { escapeHtml } from '../config-form/field-input.js?v=module-bootstrap-20260714-stage-dataset5';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

export function createDatasetEditor() {
        const panel = document.createElement('div');
        panel.id = 'dataset-editor';
        panel.className = 'dataset-editor';
        renderDatasetEditor(panel);
        return panel;
    }

    function renderDatasetPresetList() {
        const list = document.getElementById('dataset-preset-list');
        if (!list) return;
        list.innerHTML = '';
        const datasetPresetState = currentDatasetPresetState();
        const presets = datasetPresetState.presets || [];
        const groups = datasetPresetGroupsForDisplay();
        updateDatasetPresetPageSummary();
        const showErrorAsEmptyState = datasetPresetState.error && !presets.length;
        if (datasetPresetState.loading && !presets.length) {
            const loading = document.createElement('p');
            loading.className = 'dataset-preset-empty';
            loading.textContent = '正在读取数据集预设...';
            list.appendChild(loading);
            return;
        }
        if (showErrorAsEmptyState) {
            const error = document.createElement('p');
            error.className = 'dataset-preset-empty error';
            error.textContent = datasetPresetState.error;
            list.appendChild(error);
        }
        if (!presets.length && !groups.length) {
            const empty = document.createElement('p');
            empty.className = 'dataset-preset-empty';
            empty.textContent = datasetPresetState.error ? '读取数据集预设失败。' : '还没有数据集预设。';
            list.appendChild(empty);
            return;
        }
        if (!groups.length) {
            const empty = document.createElement('p');
            empty.className = 'dataset-preset-empty';
            empty.textContent = '没有匹配的数据集预设。';
            list.appendChild(empty);
            return;
        }
        const stored = readDatasetPresetGroupState();
        for (const group of groups) {
            list.appendChild(createDatasetPresetGroupNode(group, stored));
        }
    }

export function updateDatasetPresetPageSummary() {
        const summary = document.getElementById('dataset-page-summary');
        if (!summary) return;
        const datasetPresetState = currentDatasetPresetState();
        const presets = datasetPresetState.presets || [];
        const groups = datasetPresetState.groups || [];
        const totalDatasets = presets.reduce((sum, preset) => sum + Number(preset.summary?.dataset_count || 0), 0);
        const totalRepeats = presets.reduce((sum, preset) => sum + Number(preset.summary?.repeat_total || 0), 0);
        summary.innerHTML = '';
        [
            ['预设', presets.length],
            ['分组', groups.length || 1],
            ['子集', totalDatasets],
            ['重复', totalRepeats],
        ].forEach(([label, value]) => {
            const item = document.createElement('span');
            item.className = 'dataset-page-summary-stat';
            item.innerHTML = `<strong>${escapeHtml(String(value))}</strong><small>${escapeHtml(label)}</small>`;
            summary.appendChild(item);
        });
        if (datasetPresetState.dirty) {
            const dirty = document.createElement('span');
            dirty.className = 'dataset-page-summary-dirty';
            dirty.textContent = '当前预设未保存';
            summary.appendChild(dirty);
        }
    }

configureDatasetRenderBridge({
    renderDatasetPresetList,
});
