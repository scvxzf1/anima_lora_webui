/**
 * Dataset editor / preset loaders.
 * Extracted from anima-app chunk 03.
 */
import { HIDDEN_DATASET_PRESET_FILES } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260711-ir2';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260711-ir2';
import {
    datasetPresetSummaryByFile,
    orderDatasetPresetsForGroups,
    selectedDatasetConfigOverride,
    sortDatasetPresetGroups,
} from '../anima-app/helpers/dataset-presets.js?v=module-bootstrap-20260711-ir2';
import {
    renderConfigDatasetPickerDialog,
    renderDatasetEditor,
    renderDatasetPresetHeader,
    renderDatasetPresetList,
    isDatasetTabActive,
} from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir2';
import { isCliOnlySpdSource } from '../anima-app/helpers/training-launch-bridge.js?v=module-bootstrap-20260711-ir2';
import { confirmUnsavedDiscard } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir2';
import { api, datasetPresetApi, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir2';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { currentTrainingConfigFile } from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    isConfigDatasetPickerDialogOpen,
    renderConfigDatasetPicker,
} from '../config-form/dataset-picker.js?v=module-bootstrap-20260711-ir2';

const configState = getConfigState();
const datasetState = getDatasetState();
const trainingState = getTrainingState();

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

export async function loadDatasetEditor(parentSeq = configState.configLoadSeq) {
    const requestSeq = ++datasetState.datasetLoadSeq;
    const currentTrainingSource = currentTrainingSourceState();
    const datasetEditorState = currentDatasetEditorState();
    const variant = currentTrainingSource.method || val('variant-select');
    const preset = val('preset-select');
    const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
    if (!variant || location.protocol === 'file:') return;
    if (isCliOnlySpdSource(variant, methodsSubdir)) {
        datasetState.datasetEditorState = {
            ...datasetEditorState,
            loading: false,
            loaded: false,
            error: 'SPD 是 CLI 实验配置，不使用 Web 数据集编辑器。',
        };
        renderDatasetEditor();
        return;
    }
    datasetState.datasetEditorState.loading = true;
    datasetState.datasetEditorState.error = '';
    renderDatasetEditor();
    try {
        const params = new URLSearchParams({ variant, preset, methods_subdir: methodsSubdir });
        const configFile = currentTrainingConfigFile();
        if (configFile) params.set('config_file', configFile);
        const datasetConfigOverride = selectedDatasetConfigOverride();
        if (datasetConfigOverride !== null) params.set('dataset_config', datasetConfigOverride);
        const data = await api(`/api/config/datasets?${params.toString()}`);
        if (parentSeq !== configState.configLoadSeq || requestSeq !== datasetState.datasetLoadSeq) return;
        if (!data.ok) {
            throw new Error(data.error || '读取数据集配置失败');
        }
        datasetState.datasetEditorState = {
            loading: false,
            loaded: true,
            dirty: false,
            dataset_config: data.dataset_config || '',
            datasets: normalizeDatasetEditorRows(data.datasets || []),
            defaults: normalizeDatasetDefaults(data.defaults || {}),
            error: '',
        };
    } catch (e) {
        if (parentSeq !== configState.configLoadSeq || requestSeq !== datasetState.datasetLoadSeq) return;
        datasetState.datasetEditorState = {
            ...datasetEditorState,
            loading: false,
            loaded: false,
            defaults: normalizeDatasetDefaults(datasetEditorState.defaults || {}),
            error: e.message || '读取数据集配置失败',
        };
    }
    renderDatasetEditor();
}

export async function loadDatasetPresets(options = {}) {
    if (location.protocol === 'file:') return false;
    const requestSeq = ++datasetState.datasetPresetLoadSeq;
    const datasetPresetState = currentDatasetPresetState();
    const managePresets = options.manage === true || (options.manage !== false && isDatasetTabActive());
    if (managePresets) {
        datasetState.datasetPresetState.loading = true;
        renderDatasetPresetList();
    }
    try {
        const data = await datasetPresetApi('/api/config/dataset-presets');
        if (requestSeq !== datasetState.datasetPresetLoadSeq) return false;
        if (!data.ok) throw new Error(data.error || '读取数据集预设失败');
        const presets = (Array.isArray(data.presets) ? data.presets : [])
            .filter((preset) => !HIDDEN_DATASET_PRESET_FILES.has(preset.path));
        const presetPaths = new Set(presets.map((preset) => preset.path));
        const groups = (Array.isArray(data.groups) ? data.groups : [])
            .map((group) => ({
                ...group,
                files: (Array.isArray(group.files) ? group.files : [])
                    .filter((preset) => presetPaths.has(preset.path) && !HIDDEN_DATASET_PRESET_FILES.has(preset.path)),
            }))
            .filter((group) => group.kind === 'dataset' || group.files.length);
        const sortedGroups = sortDatasetPresetGroups(groups);
        datasetState.datasetPresetState.presets = orderDatasetPresetsForGroups(presets, sortedGroups);
        datasetState.datasetPresetState.groups = sortedGroups;
        if (managePresets) {
            datasetState.datasetPresetState.loading = false;
        }
        datasetState.datasetPresetState.error = '';
        datasetState.selectedConfigDatasetSummary = datasetPresetSummaryByFile(datasetState.selectedConfigDatasetFile);
        renderConfigDatasetPicker();
        if (!managePresets) {
            if (isConfigDatasetPickerDialogOpen()) {
                renderConfigDatasetPickerDialog();
            }
            return true;
        }
        const preserveDirtySelection = datasetPresetState.dirty;
        const selectedDatasetVisible = presets.some((preset) => preset.path === datasetPresetState.selectedFile);
        if (!selectedDatasetVisible && !preserveDirtySelection) {
            datasetState.datasetPresetState.selectedFile = '';
        }
        if (!preserveDirtySelection && options.selectCurrent !== false && datasetState.selectedConfigDatasetFile && !datasetPresetState.selectedFile && presets.some((preset) => preset.path === datasetState.selectedConfigDatasetFile)) {
            datasetState.datasetPresetState.selectedFile = datasetState.selectedConfigDatasetFile;
        }
        if (!preserveDirtySelection && !datasetPresetState.selectedFile && presets.length) {
            datasetState.datasetPresetState.selectedFile = presets[0].path;
        }
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        if (datasetPresetState.selectedFile && !datasetPresetState.dirty) {
            await loadDatasetPreset(datasetPresetState.selectedFile);
        } else {
            renderDatasetEditor();
        }
        return true;
    } catch (e) {
        if (requestSeq !== datasetState.datasetPresetLoadSeq) return false;
        if (managePresets) {
            datasetState.datasetPresetState.loading = false;
        }
        datasetState.datasetPresetState.error = e.message || '读取数据集预设失败';
        if (managePresets) {
            renderDatasetPresetList();
            renderDatasetPresetHeader();
        } else {
            renderConfigDatasetPicker();
            if (isConfigDatasetPickerDialogOpen()) {
                renderConfigDatasetPickerDialog();
            }
        }
        if (options.throwOnError) {
            throw e;
        }
        return false;
    }
}

export async function loadDatasetPreset(file) {
    const datasetPresetState = currentDatasetPresetState();
    if (!file) return;
    if (datasetPresetState.dirty && !(await confirmUnsavedDiscard('当前数据集预设有未保存修改，切换会丢弃这些修改。是否继续？'))) {
        renderDatasetPresetList();
        return;
    }
    datasetPresetState.selectedFile = file;
    datasetPresetState.loading = true;
    datasetPresetState.error = '';
    renderDatasetPresetList();
    renderDatasetPresetHeader();
    renderDatasetEditor();
    try {
        const data = await datasetPresetApi(`/api/config/dataset-presets/read?file=${encodeURIComponent(file)}`);
        if (!data.ok) throw new Error(data.error || '读取数据集预设失败');
        datasetState.datasetPresetState = {
            ...datasetPresetState,
            loading: false,
            dirty: false,
            isNew: false,
            selectedFile: data.file || file,
            datasets: normalizeDatasetEditorRows(data.datasets || []),
            defaults: normalizeDatasetDefaults(data.defaults || {}),
            readonly: Boolean(data.readonly || data.meta?.locked),
            error: '',
            status: '',
        };
    } catch (e) {
        datasetState.datasetPresetState = {
            ...datasetPresetState,
            loading: false,
            error: e.message || '读取数据集预设失败',
        };
    }
    renderDatasetPresetList();
    renderDatasetPresetHeader();
    renderDatasetEditor();
}
