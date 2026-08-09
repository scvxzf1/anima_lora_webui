/**
 * Config form dataset-preset picker panel.
 * Extracted from anima-app chunk 06.
 */
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { datasetPresetByFile } from '../anima-app/helpers/dataset-presets.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    ensureConfigDatasetPreview,
    renderConfigDatasetPickerDialog,
} from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadDatasetPresets } from '../dataset-editor/load.js?v=module-bootstrap-20260714-stage-dataset5';

const configState = getConfigState();
const datasetState = getDatasetState();

function currentConfigState() {
    return configState.currentConfig || {};
}

export function createConfigDatasetPicker() {
    const panel = document.createElement('div');
    panel.id = 'config-dataset-picker';
    panel.className = 'config-dataset-picker';
    renderConfigDatasetPicker(panel);
    return panel;
}

export function renderConfigDatasetPicker(existingPanel = null) {
    const panel = existingPanel || document.getElementById('config-dataset-picker');
    if (!panel) return;
    panel.innerHTML = '';

    const header = document.createElement('div');
    header.className = 'config-dataset-picker-header';
    const title = document.createElement('div');
    title.innerHTML = '<strong>数据集预设</strong><span>当前配置只保留选择摘要；搜索、选择和预览在弹窗中完成。</span>';
    const actions = document.createElement('div');
    actions.className = 'config-dataset-picker-actions';
    const openBtn = document.createElement('button');
    openBtn.type = 'button';
    openBtn.className = 'btn btn-small';
    openBtn.textContent = datasetState.selectedConfigDatasetFile ? '更换预设' : '选择预设';
    openBtn.title = '打开数据集预设弹窗，可以搜索并查看第一张原始图预览。';
    openBtn.addEventListener('click', openConfigDatasetPickerDialog);
    const manageBtn = document.createElement('button');
    manageBtn.type = 'button';
    manageBtn.className = 'btn btn-small';
    manageBtn.textContent = '管理数据集';
    manageBtn.addEventListener('click', () => document.querySelector('[data-tab="datasets"]')?.click());
    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.className = 'btn btn-small';
    refreshBtn.textContent = '刷新预设';
    refreshBtn.addEventListener('click', () => loadDatasetPresets({ selectCurrent: false, manage: false }));
    actions.append(openBtn, manageBtn, refreshBtn);
    header.append(title, actions);
    panel.appendChild(header);

    const body = document.createElement('div');
    body.className = 'config-dataset-picker-body';
    body.appendChild(createConfigDatasetCurrentSummary());
    panel.appendChild(body);
    ensureConfigDatasetPreview();
}

function createConfigDatasetCurrentSummary() {
    const currentConfig = currentConfigState();
    const preset = datasetPresetByFile(datasetState.selectedConfigDatasetFile);
    const summary = datasetState.selectedConfigDatasetSummary || preset?.summary || {};
    const wrap = document.createElement('div');
    wrap.className = 'config-dataset-current';

    const info = document.createElement('div');
    info.className = 'config-dataset-current-info';
    const label = document.createElement('span');
    label.className = 'config-dataset-current-label';
    label.textContent = datasetState.selectedConfigDatasetFile ? '当前选中' : '当前状态';
    const title = document.createElement('strong');
    title.textContent = datasetState.selectedConfigDatasetFile
        ? (preset?.label || preset?.filename || datasetState.selectedConfigDatasetFile)
        : '不使用独立数据集预设';
    const path = document.createElement('code');
    path.textContent = datasetState.selectedConfigDatasetFile || '沿用当前训练配置文件中的数据集字段';
    info.append(label, title, path);

    const meta = document.createElement('div');
    meta.className = 'config-dataset-current-meta';
    const state = document.createElement('span');
    const isDirtySelection = datasetState.selectedConfigDatasetFile !== (currentConfig.dataset_config || '');
    state.className = [
        'config-dataset-current-state',
        isDirtySelection ? 'dirty' : 'synced',
    ].join(' ');
    state.textContent = isDirtySelection
        ? '未保存'
        : '已同步';
    const count = document.createElement('span');
    count.textContent = datasetState.selectedConfigDatasetFile
        ? `${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}`
        : '当前配置';
    const source = document.createElement('span');
    source.textContent = datasetState.selectedConfigDatasetFile && summary.source_dir
        ? `原始路径: ${summary.source_dir}`
        : '保存当前配置后才会写入训练 TOML';
    meta.append(state, count, source);

    wrap.append(info, meta);
    return wrap;
}

export function isConfigDatasetPickerDialogOpen() {
    return Boolean(document.getElementById('config-dataset-picker-dialog')?.open);
}

function openConfigDatasetPickerDialog() {
    const dialog = document.getElementById('config-dataset-picker-dialog');
    if (!dialog) return;
    renderConfigDatasetPickerDialog();
    ensureConfigDatasetPreview();
    if (dialog.showModal && !dialog.open) {
        dialog.showModal();
    } else if (!dialog.open) {
        dialog.setAttribute('open', 'open');
    }
    const search = dialog.querySelector('.config-dataset-search');
    if (search) {
        search.focus({ preventScroll: true });
        search.setSelectionRange(search.value.length, search.value.length);
    }
}

export function closeConfigDatasetPickerDialog() {
    const dialog = document.getElementById('config-dataset-picker-dialog');
    if (dialog?.open) dialog.close();
}

function openUnnamedDatasetDialog() {
    const dialog = document.getElementById('unnamed-dataset-dialog');
    if (!dialog) return;
    if (dialog.showModal && !dialog.open) {
        dialog.showModal();
    } else if (!dialog.open) {
        dialog.setAttribute('open', 'open');
    }
}

// Keep for parity with legacy chunk surface if external listeners call it via global bridge later.
export { openUnnamedDatasetDialog };
