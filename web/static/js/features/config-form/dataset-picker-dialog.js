/**
 * Config dataset picker dialog, dataset editor shell, and file-group drag primitives.
 * Extracted from anima-app chunk 07.
 */
import { FILE_GROUP_DROP_TARGET_ATTR } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260711-ir1';
import {
    datasetPresetByFile,
    datasetPresetGroupsForDisplay,
    datasetPresetSummaryByFile,
} from '../anima-app/helpers/dataset-presets.js?v=module-bootstrap-20260711-ir1';
import {
    configureDatasetRenderBridge,
    createDatasetPresetGroupNode,
    readDatasetPresetGroupState,
    renderDatasetEditor,
} from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { datasetPresetApi } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';
import { updateTomlDirtyState } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadStepEstimate } from './step-estimate.js?v=module-bootstrap-20260711-ir1';
import { renderConfigDatasetPicker } from './dataset-picker.js?v=module-bootstrap-20260711-ir1';
import { escapeHtml } from './field-input.js?v=module-bootstrap-20260711-ir1';

const configState = getConfigState();
const datasetState = getDatasetState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentConfigDatasetPreviewState() {
    return datasetState.configDatasetPreviewState || {};
}

    function renderConfigDatasetPickerDialog() {
        const dialog = document.getElementById('config-dataset-picker-dialog');
        const body = document.getElementById('config-dataset-picker-dialog-body');
        if (!dialog || !body) return;
        body.innerHTML = '';

        const toolbar = document.createElement('div');
        toolbar.className = 'config-dataset-dialog-toolbar';
        const search = document.createElement('input');
        search.type = 'search';
        search.className = 'field-input config-dataset-search';
        search.placeholder = '搜索数据集预设、路径或原始目录';
        search.value = datasetState.configDatasetPickerSearch;
        search.addEventListener('input', () => {
            const cursor = search.selectionStart ?? search.value.length;
            datasetState.configDatasetPickerSearch = search.value;
            renderConfigDatasetPickerDialog();
            const nextSearch = document.querySelector('#config-dataset-picker-dialog .config-dataset-search');
            if (nextSearch) {
                nextSearch.focus();
                nextSearch.setSelectionRange(cursor, cursor);
            }
        });
        toolbar.appendChild(search);
        body.appendChild(toolbar);

        const workspace = document.createElement('div');
        workspace.className = 'config-dataset-workspace config-dataset-dialog-workspace';
        workspace.appendChild(createConfigDatasetPresetList());
        workspace.appendChild(createConfigDatasetPresetPreview());
        body.appendChild(workspace);
    }

    function createConfigDatasetPresetList() {
        const list = document.createElement('div');
        list.className = 'config-dataset-preset-list';
        const noneBtn = createConfigDatasetPresetButton(null);
        list.appendChild(noneBtn);

        const presets = filteredConfigDatasetPresets();
        if (!presets.length && datasetState.configDatasetPickerSearch.trim()) {
            const empty = document.createElement('p');
            empty.className = 'config-dataset-picker-empty';
            empty.textContent = '没有匹配的数据集预设。';
            list.appendChild(empty);
            return list;
        }

        for (const preset of presets) {
            list.appendChild(createConfigDatasetPresetButton(preset));
        }
        return list;
    }

    function createConfigDatasetPresetButton(preset) {
        const isNone = !preset;
        const file = isNone ? '' : preset.path;
        const summary = preset?.summary || {};
        const active = file === datasetState.selectedConfigDatasetFile;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'config-dataset-preset-option',
            active ? 'active' : '',
            preset?.readonly ? 'readonly' : '',
        ].filter(Boolean).join(' ');
        btn.dataset.file = file;
        const title = document.createElement('strong');
        title.textContent = isNone
            ? '不使用独立数据集预设'
            : (preset.label || preset.filename || preset.path || '未命名预设');
        const path = document.createElement('span');
        path.textContent = isNone ? '沿用当前训练配置文件中的数据集字段' : preset.path;
        const meta = document.createElement('small');
        meta.textContent = isNone
            ? '当前配置'
            : `${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}${preset.readonly ? ' · 只读' : ''}`;
        btn.append(title, path, meta);
        btn.addEventListener('click', () => selectConfigDatasetPreset(file));
        return btn;
    }

    function filteredConfigDatasetPresets() {
        const keyword = datasetState.configDatasetPickerSearch.trim().toLowerCase();
        const presets = currentDatasetPresetState().presets || [];
        if (!keyword) return presets;
        return presets.filter((preset) => {
            const summary = preset.summary || {};
            return [
                preset.label,
                preset.filename,
                preset.path,
                summary.source_dir,
                            ].some((value) => String(value || '').toLowerCase().includes(keyword));
        });
    }

    function createConfigDatasetPresetPreview() {
        const preview = document.createElement('div');
        preview.className = 'config-dataset-preview';
        const summary = document.createElement('div');
        summary.className = 'config-dataset-summary';
        summary.appendChild(createConfigDatasetSummary());
        preview.appendChild(summary);
        preview.appendChild(createConfigDatasetPreviewImage());
        return preview;
    }

    function createConfigDatasetPreviewImage() {
        const box = document.createElement('div');
        box.className = 'config-dataset-preview-image';
        const state = currentConfigDatasetPreviewState();
        if (!datasetState.selectedConfigDatasetFile) {
            box.classList.add('empty');
            box.textContent = '选择一个数据集预设后，这里会显示第一张原始图。';
            return box;
        }
        if (state.file !== datasetState.selectedConfigDatasetFile || state.loading) {
            box.classList.add('empty');
            box.textContent = '正在读取第一张原始图...';
            return box;
        }
        if (state.error) {
            box.classList.add('empty');
            box.textContent = state.error;
            return box;
        }
        const image = Array.isArray(state.payload?.images) ? state.payload.images[0] : null;
        if (!image) {
            box.classList.add('empty');
            box.textContent = state.payload?.message || '没有找到可预览的原始图。';
            return box;
        }
        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name || '数据集预览图';
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            box.classList.add('empty');
            box.textContent = '预览图加载失败。';
        });
        const caption = document.createElement('div');
        caption.className = 'config-dataset-preview-caption';
        const name = document.createElement('strong');
        name.textContent = image.name || '-';
        const path = document.createElement('span');
        path.textContent = state.payload?.directory || image.file || '';
        caption.append(name, path);
        box.append(img, caption);
        return box;
    }

    function createConfigDatasetSummary() {
        const wrap = document.createElement('div');
        const preset = datasetPresetByFile(datasetState.selectedConfigDatasetFile);
        const summary = datasetState.selectedConfigDatasetSummary || preset?.summary || {};
        if (!datasetState.selectedConfigDatasetFile) {
            wrap.className = 'config-dataset-summary-empty';
            wrap.textContent = '未选择独立数据集预设；训练会沿用当前配置文件里的数据集字段。';
            return wrap;
        }
        const items = [
            ['预设文件', datasetState.selectedConfigDatasetFile],
            ['数据组数', String(summary.dataset_count || 0)],
            ['重复合计', String(summary.repeat_total || 0)],
            ['第 1 组原始路径', summary.source_dir || '-'],
        ];
        if (datasetState.selectedConfigDatasetFile !== (currentConfigState().dataset_config || '')) {
            items.unshift(['状态', '未保存，保存当前配置后生效']);
        }
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            row.append(key, valEl);
            wrap.appendChild(row);
        }
        return wrap;
    }

    async function selectConfigDatasetPreset(file) {
        datasetState.selectedConfigDatasetFile = file || '';
        datasetState.selectedConfigDatasetSummary = datasetPresetSummaryByFile(datasetState.selectedConfigDatasetFile);
        datasetState.configDatasetPreviewState = {
            file: '',
            loading: false,
            payload: null,
            error: '',
        };
        renderConfigDatasetPicker();
        renderConfigDatasetPickerDialog();
        updateTomlDirtyState();
        await loadStepEstimate();
    }

    function ensureConfigDatasetPreview() {
        const previewState = currentConfigDatasetPreviewState();
        if (!datasetState.selectedConfigDatasetFile) return;
        if (previewState.file === datasetState.selectedConfigDatasetFile
            && (previewState.loading || previewState.payload || previewState.error)) {
            return;
        }
        loadConfigDatasetPresetPreview(datasetState.selectedConfigDatasetFile);
    }

    async function loadConfigDatasetPresetPreview(file) {
        if (!file || location.protocol === 'file:') return;
        const requestSeq = ++datasetState.configDatasetPreviewRequestSeq;
        datasetState.configDatasetPreviewState = {
            file,
            loading: true,
            payload: null,
            error: '',
        };
        renderConfigDatasetPreviewArea();
        try {
            const params = new URLSearchParams({
                file,
                dataset_index: '0',
                source: 'source',
                limit: '1',
            });
            const payload = await datasetPresetApi(`/api/config/dataset-presets/images?${params.toString()}`);
            if (requestSeq !== datasetState.configDatasetPreviewRequestSeq || file !== datasetState.selectedConfigDatasetFile) return;
            if (!payload.ok) throw new Error(payload.error || '读取数据集预览失败');
            datasetState.configDatasetPreviewState = {
                file,
                loading: false,
                payload,
                error: '',
            };
        } catch (e) {
            if (requestSeq !== datasetState.configDatasetPreviewRequestSeq || file !== datasetState.selectedConfigDatasetFile) return;
            datasetState.configDatasetPreviewState = {
                file,
                loading: false,
                payload: null,
                error: e.message || '读取数据集预览失败',
            };
        }
        renderConfigDatasetPreviewArea();
    }

    function renderConfigDatasetPreviewArea() {
        const previews = document.querySelectorAll('.config-dataset-preview');
        if (!previews.length) return;
        previews.forEach((preview) => {
            preview.innerHTML = '';
            const summary = document.createElement('div');
            summary.className = 'config-dataset-summary';
            summary.appendChild(createConfigDatasetSummary());
            preview.appendChild(summary);
            preview.appendChild(createConfigDatasetPreviewImage());
        });
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

export function registerFileGroupDropTarget(node, resolve) {
        node.setAttribute(FILE_GROUP_DROP_TARGET_ATTR, '1');
        datasetState.fileGroupDropTargets.set(node, resolve);
        datasetState.fileGroupDropTargetNodes.add(node);
    }

configureDatasetRenderBridge({
    renderConfigDatasetPickerDialog,
    ensureConfigDatasetPreview,
    renderDatasetPresetList,
});
