/**
 * Config form dataset-preset picker dialog and preview.
 * Extracted from former chunk 07 / dataset dialog module.
 */
import {
    datasetPresetByFile,
    datasetPresetGroupsForDisplay,
    datasetPresetSummaryByFile,
} from '../anima-app/helpers/dataset-presets.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    configureDatasetRenderBridge,
} from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { datasetPresetApi } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { updateTomlDirtyState } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadStepEstimate } from './step-estimate.js?v=module-bootstrap-20260714-stage-dataset5';
import { renderConfigDatasetPicker } from './dataset-picker.js?v=module-bootstrap-20260714-stage-dataset5';

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
            datasetState.configDatasetPickerSearch = search.value;
            renderConfigDatasetPickerResults(body);
        });
        toolbar.appendChild(search);
        body.appendChild(toolbar);

        renderConfigDatasetPickerResults(body);
    }

    function renderConfigDatasetPickerResults(body) {
        const workspace = document.createElement('div');
        workspace.className = 'config-dataset-workspace config-dataset-dialog-workspace';
        workspace.appendChild(createConfigDatasetPresetList());
        workspace.appendChild(createConfigDatasetPresetPreview());
        const currentWorkspace = body.querySelector('.config-dataset-dialog-workspace');
        if (currentWorkspace) currentWorkspace.replaceWith(workspace);
        else body.appendChild(workspace);
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

    function updateConfigDatasetPresetListActive() {
        const buttons = document.querySelectorAll('#config-dataset-picker-dialog .config-dataset-preset-option');
        buttons.forEach((btn) => {
            const file = btn.dataset.file || '';
            btn.classList.toggle('active', file === datasetState.selectedConfigDatasetFile);
        });
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
        updateConfigDatasetPresetListActive();
        renderConfigDatasetPreviewArea();
        ensureConfigDatasetPreview();
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

configureDatasetRenderBridge({
    renderConfigDatasetPickerDialog,
    ensureConfigDatasetPreview,
});
