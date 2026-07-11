/**
 * Dataset editor panel rendering and active-row helpers.
 */

import { captureDatasetExperimentalOpenStates } from './inline-help.js?v=module-bootstrap-20260711-ir2';
import { help } from '../../config/catalog.js?v=module-bootstrap-20260711-ir2';
import { datasetConfigLabel, datasetConfigValue } from '../anima-app/helpers/dataset-config-fields.js?v=module-bootstrap-20260711-ir2';
import { createHelpContent } from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260711-ir2';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260711-ir2';
import { createDatasetConfigInput } from './config-input.js?v=module-bootstrap-20260711-ir2';
import { createDatasetEditorItem } from './item-drag.js?v=module-bootstrap-20260711-ir2';
import { addDatasetEditorRow, datasetValidTargetIndices } from './mutations.js?v=module-bootstrap-20260711-ir2';
import { createDatasetEditorToolbarActions } from './toolbar.js?v=module-bootstrap-20260711-ir2';

const configState = getConfigState();
const datasetState = getDatasetState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

export function renderDatasetEditor(existingPanel = null) {
    const panel = existingPanel || document.getElementById('dataset-editor');
    if (!panel) return;
    captureDatasetExperimentalOpenStates(panel);
    panel.innerHTML = '';
    const state = datasetEditorStateForActivePanel();

    const header = document.createElement('div');
    header.className = 'dataset-editor-header';
    const title = document.createElement('div');
    title.innerHTML = '<strong>多数据集路径</strong><span>每一行是一组数据：填写原始图路径、重复次数和分桶参数；缩放图与 LoRA 缓存会在训练运行目录中自动生成。</span>';
    const actions = document.createElement('div');
    actions.className = 'dataset-editor-actions';
    const toolbar = createDatasetEditorToolbarActions();
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn btn-small';
    addBtn.textContent = '添加数据集';
    addBtn.title = '新增一组数据集路径。适合把多个角色、画风或批次一起训练，并给每组设置独立重复次数。';
    addBtn.addEventListener('click', addDatasetEditorRow);
    actions.append(toolbar, addBtn);
    header.append(title, actions);
    panel.appendChild(header);

    if (state.loading) {
        const loading = document.createElement('p');
        loading.className = 'dataset-editor-message';
        loading.textContent = '正在读取数据集配置...';
        panel.appendChild(loading);
        return;
    }
    if (state.error) {
        const error = document.createElement('p');
        error.className = 'dataset-editor-message error';
        error.textContent = state.error;
        panel.appendChild(error);
    }

    const rows = state.datasets.length
        ? state.datasets
        : normalizeDatasetEditorRows([{
            source_dir: currentConfigState().source_image_dir || '',
            image_dir: currentConfigState().resized_image_dir || '',
            cache_dir: currentConfigState().lora_cache_dir || '',
            num_repeats: 1,
            settings: normalizeDatasetDefaults(state.defaults || {}),
        }]);
    if (!state.datasets.length) {
        setActiveDatasetRows(rows);
    }

    panel.appendChild(createDatasetDefaultsEditor());

    const list = document.createElement('div');
    list.className = 'dataset-editor-list';
    rows.forEach((row, index) => {
        list.appendChild(createDatasetEditorItem(row, index));
    });
    panel.appendChild(list);

    const footer = document.createElement('div');
    footer.className = 'dataset-editor-footer';
    const configPath = document.createElement('code');
    configPath.textContent = activeDatasetFileLabel();
    const dirty = document.createElement('span');
    dirty.className = activeDatasetDirty() ? 'dataset-editor-dirty active' : 'dataset-editor-dirty';
    dirty.textContent = activeDatasetDirty() ? '有未保存的数据集修改' : '数据集路径已同步';
    footer.append(configPath, dirty);
    panel.appendChild(footer);
    if (isDatasetTabActive()) {
        renderDatasetPresetHeader();
    }
}

export function refreshDatasetEditorItem(index) {
    const panel = document.getElementById('dataset-editor');
    if (!panel) return false;
    const state = datasetEditorStateForActivePanel();
    if (state.loading || state.error) return false;
    const rows = normalizeDatasetEditorRows(state.datasets);
    const row = rows[index];
    if (!row) return false;
    const list = panel.querySelector('.dataset-editor-list');
    const currentItem = list?.querySelector(`.dataset-editor-item[data-index="${index}"]`);
    if (!list || !currentItem) return false;
    const nextItem = createDatasetEditorItem(row, index);
    list.replaceChild(nextItem, currentItem);
    return true;
}

export function refreshDatasetEditorItems(indices) {
    const targets = datasetValidTargetIndices(indices, normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets).length);
    if (!targets.length) return false;
    let updated = false;
    for (const index of targets) {
        updated = refreshDatasetEditorItem(index) || updated;
    }
    return updated;
}

export function datasetEditorStateForActivePanel() {
    return isDatasetTabActive() ? currentDatasetPresetState() : currentDatasetEditorState();
}

export function isDatasetTabActive() {
    return Boolean(document.getElementById('tab-datasets')?.classList.contains('active'));
}

export function setActiveDatasetRows(rows) {
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.datasets = rows;
    } else {
        datasetState.datasetEditorState.datasets = rows;
    }
}

export function activeDatasetFileLabel() {
    if (isDatasetTabActive()) {
        return currentDatasetPresetState().selectedFile || '保存后生成 configs/datasets/<名称>.toml';
    }
    return currentDatasetEditorState().dataset_config || currentConfigState().dataset_config || '保存后自动生成 configs/datasets/<当前配置>.toml';
}

export function activeDatasetDirty() {
    return isDatasetTabActive() ? currentDatasetPresetState().dirty : currentDatasetEditorState().dirty;
}

function createDatasetDefaultsEditor() {
    const state = datasetEditorStateForActivePanel();
    const defaults = normalizeDatasetDefaults(state.defaults || {});
    if (isDatasetTabActive()) {
        datasetState.datasetPresetState.defaults = defaults;
    } else {
        datasetState.datasetEditorState.defaults = defaults;
    }
    const wrap = document.createElement('div');
    wrap.className = 'dataset-defaults-list';

    const heading = document.createElement('div');
    heading.className = 'dataset-defaults-heading';
    heading.innerHTML = '<strong>通用标注设置</strong><span>这里只保留 keep_tokens；文本标注扩展名等兼容项在每组数据集的高级区配置。</span>';
    wrap.appendChild(heading);

    const fields = [
        ['keep_tokens', 'number'],
    ];

    for (const [key, type, layout] of fields) {
        const row = document.createElement('div');
        row.className = [
            'dataset-config-field',
            layout === 'wide' ? 'wide' : '',
            layout === 'switch' ? 'switch' : '',
        ].filter(Boolean).join(' ');
        row.dataset.key = key;

        const label = document.createElement('label');
        label.className = 'dataset-config-label';
        const nameSpan = document.createElement('span');
        nameSpan.className = 'field-name';
        nameSpan.textContent = datasetConfigLabel(key);
        nameSpan.title = key;
        label.appendChild(nameSpan);

        const btn = document.createElement('button');
        btn.className = 'info-toggle';
        btn.textContent = '?';
        btn.type = 'button';
        btn.title = '查看填写建议、好处、代价、风险和推荐';
        btn.addEventListener('click', () => {
            btn.classList.toggle('active');
            row.querySelector('.field-help')?.classList.toggle('visible');
        });
        label.appendChild(btn);
        row.appendChild(label);

        const input = createDatasetConfigInput(key, type, defaults);
        row.appendChild(input);

        const helpDiv = document.createElement('div');
        helpDiv.className = 'field-help';
        helpDiv.appendChild(createHelpContent(key, datasetConfigValue(key, defaults)));
        row.appendChild(helpDiv);
        wrap.appendChild(row);
    }
    return wrap;
}
