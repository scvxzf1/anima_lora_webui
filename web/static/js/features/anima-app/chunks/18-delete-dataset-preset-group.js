/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    datasetRowsForPayload,
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../helpers/dataset-values.js?v=module-bootstrap-20260809-nf4-v2';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { configureConfigFormBridge } from '../helpers/config-form-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    configureDatasetPresetActionsBridge,
    setDatasetPresetStatus,
} from '../helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { loadTomlFileList } from '../helpers/toml-manager-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { api, val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { loadDatasetPresets, loadStepEstimate } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260809-nf4-v2';
import { renderConfigDatasetPicker } from './06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260809-nf4-v2';
import { syncDatasetEditorToCompatFields } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260809-nf4-v2';
import { renderDatasetEditor } from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { showHistoryTaskConfirmDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { currentTomlEditorContentForFile, updateTomlDirtyState } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { setTomlStatus } from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    collectChangedFormValues,
    collectNetworkArgsFromForm,
    networkArgFieldValueFromConfig,
    networkArgInputChanged,
    prepareFormPatchValues,
    readFieldInputValue,
    shouldSkipUiDefaultField,
} from '../../config-form/config-value-collector.js?v=module-bootstrap-20260809-nf4-v2';
import {
    applyLossWeightingFieldInputState,
    currentLossWeightingScheme,
    lossWeightingDisabledHint,
    lossWeightingFieldState,
    readDoRAAvailable,
    readLoKrEnabled,
    readVeRAEnabled,
    setDoRADraftValue,
    updateDoRAFieldState,
    updateLoKrFieldState,
    updateLossWeightingFieldState,
    updateVeRAFieldState,
} from '../../config-form/adapter-field-state.js?v=module-bootstrap-20260809-nf4-v2';

const configState = getConfigState();
const datasetState = getDatasetState();
const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

    export async function deleteDatasetPresetGroup(group) {
        if (!group?.id || !group.deletable) return;
        const count = (group.files || []).length;
        const ok = await showHistoryTaskConfirmDialog({
            title: '删除数据集分组',
            description: group.label || group.id,
            message: count > 0
                ? `只删除这个分组，不删除其中 ${count} 个数据集 TOML；这些文件会回到默认数据集分组。`
                : '只删除这个空分组，不会删除任何 TOML 文件。',
            confirmText: '删除分组',
            danger: true,
        });
        if (!ok) return;
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '删除数据集分组失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组已删除', 'ok');
        } catch (e) {
            setDatasetPresetStatus('删除数据集分组失败: ' + e.message, 'error');
        }
    }

    export async function placeDatasetPresetGroup(payload, index) {
        const groupId = payload?.groupId;
        const datasetPresetState = currentDatasetPresetState();
        if (!groupId) return;
        if (datasetPresetState.search.trim()) {
            setDatasetPresetStatus('筛选数据集预设时不能拖动排序，请先清空搜索', 'error');
            return;
        }
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'group', group: groupId, scope: 'dataset', index }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '调整数据集分组位置失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组位置已更新', 'ok');
        } catch (e) {
            setDatasetPresetStatus('调整数据集分组位置失败: ' + e.message, 'error');
        }
    }

    export async function placeDatasetPresetFile(payload, groupId, index, placeOptions = {}) {
        const file = payload?.file;
        const datasetPresetState = currentDatasetPresetState();
        if (!file || !groupId) return;
        if (datasetPresetState.search.trim()) {
            setDatasetPresetStatus('筛选数据集预设时不能拖动排序，请先清空搜索', 'error');
            return;
        }
        try {
            const body = {
                target: 'file',
                file,
                group: groupId,
                index,
            };
            if (placeOptions?.anchor) {
                body.anchor = placeOptions.anchor;
                body.position = placeOptions.position === 'before' ? 'before' : 'after';
            }
            if (Array.isArray(placeOptions?.order) && placeOptions.order.length) {
                body.order = placeOptions.order;
            }
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '数据集预设位置调整失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集预设位置已更新', 'ok');
        } catch (e) {
            setDatasetPresetStatus('数据集预设位置调整失败: ' + e.message, 'error');
        }
    }

    export async function saveDatasetEditor(options = {}) {
        const currentTrainingSource = currentTrainingSourceState();
        const currentConfig = currentConfigState();
        const datasetEditorState = currentDatasetEditorState();
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const targetFile = options.trainFile || currentTrainingSource.file || tomlState.currentTomlFile || '';
        const targetContent = options.trainContent ?? currentTomlEditorContentForFile(targetFile);
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        if (!rows.length || rows.some((row) => !row.source_dir.trim())) {
            setTomlStatus('error', '请至少填写一个原始数据集路径');
            return null;
        }
        try {
            const res = await api('/api/config/datasets', {
                method: 'PUT',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    train_file: targetFile,
                    train_content: targetContent,
                    prefer_existing_dataset_config: options.preferExistingDatasetConfig !== false,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetEditorState.defaults || {}),
                    config_values: collectChangedFormValues({ persistDefaultFields: true }),
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存数据集配置失败');
                return null;
            }
            if (typeof res.train_content === 'string' && res.train_content) {
                const editor = document.getElementById('toml-editor');
                if (editor && targetFile === (tomlState.currentTomlFile || val('toml-file-select'))) {
                    editor.value = res.train_content;
                    tomlState.tomlSavedContent = res.train_content;
                }
            }
            datasetState.datasetEditorState = {
                loading: false,
                loaded: true,
                dirty: false,
                dataset_config: res.dataset_config || datasetEditorState.dataset_config,
                datasets: normalizeDatasetEditorRows(res.datasets || rows),
                defaults: normalizeDatasetDefaults(res.defaults || datasetEditorState.defaults || {}),
                stage_schedule_enabled: res.stage_schedule_enabled ?? datasetEditorState.stage_schedule_enabled,
                stage_schedule: Array.isArray(res.stage_schedule)
                    ? res.stage_schedule
                    : (datasetEditorState.stage_schedule || []),
                error: '',
            };
            const nextDatasetEditorState = currentDatasetEditorState();
            const nextDatasetConfig = nextDatasetEditorState.dataset_config || '';
            currentConfig.dataset_config = nextDatasetConfig;
            datasetState.selectedConfigDatasetFile = nextDatasetConfig;
            datasetState.selectedConfigDatasetSummary = nextDatasetConfig ? (res.summary || null) : null;
            datasetState.configDatasetPreviewState = {
                file: '',
                loading: false,
                payload: null,
                error: '',
            };
            if (nextDatasetEditorState.datasets[0]) {
                currentConfig.source_image_dir = nextDatasetEditorState.datasets[0].source_dir;
                currentConfig.resized_image_dir = nextDatasetEditorState.datasets[0].image_dir;
                currentConfig.lora_cache_dir = nextDatasetEditorState.datasets[0].cache_dir;
            }
            syncDatasetEditorToCompatFields();
            renderDatasetEditor();
            renderConfigDatasetPicker();
            updateTomlDirtyState();
            await loadDatasetPresets({ selectCurrent: false, manage: false });
            await loadStepEstimate();
            if (options.reloadList !== false) {
                await loadTomlFileList(targetFile);
            }
            return res;
        } catch (e) {
            setTomlStatus('error', '保存数据集配置失败: ' + e.message);
            return null;
        }
    }

configureDatasetPresetActionsBridge({
    deleteDatasetPresetGroup,
    placeDatasetPresetGroup,
    placeDatasetPresetFile,
});

configureConfigFormBridge({
    saveDatasetEditor,
    collectChangedFormValues,
    networkArgInputChanged,
    networkArgFieldValueFromConfig,
    collectNetworkArgsFromForm,
    prepareFormPatchValues,
    shouldSkipUiDefaultField,
    readFieldInputValue,
    readLoKrEnabled,
    updateLoKrFieldState,
    readVeRAEnabled,
    readDoRAAvailable,
    setDoRADraftValue,
    updateDoRAFieldState,
    updateVeRAFieldState,
    currentLossWeightingScheme,
    lossWeightingFieldState,
    lossWeightingDisabledHint,
    applyLossWeightingFieldInputState,
    updateLossWeightingFieldState,
});
