/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    datasetRowsForPayload,
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../helpers/dataset-values.js?v=module-bootstrap-20260711-ir6';
import {
    datasetPresetByFile,
    datasetPresetSummaryByFile,
} from '../helpers/dataset-presets.js?v=module-bootstrap-20260711-ir6';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { configureDatasetPresetActionsBridge } from '../helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    renderDatasetEditor,
    renderDatasetPresetHeader,
    renderDatasetPresetList,
} from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir6';
import { showHistoryTaskInputDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260711-ir6';
import { confirmUnsavedDiscard, currentTomlEditorContentForFile, showAppConfirmDialog } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir6';
import { api, datasetPresetApi, val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { loadDatasetPreset, loadDatasetPresets, loadStepEstimate } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260711-ir6';
import { renderConfigDatasetPicker } from './06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260711-ir6';
import {
    setTomlStatus,
} from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir6';

const configState = getConfigState();
const datasetState = getDatasetState();
const tomlState = getTomlState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

    export async function applySelectedDatasetPresetToCurrentConfig(file) {
        const currentConfig = currentConfigState();
        const nextDataset = datasetState.selectedConfigDatasetFile || '';
        const currentDataset = currentConfig.dataset_config || '';
        if (!nextDataset || nextDataset === currentDataset) {
            if (!nextDataset && currentDataset) {
                const res = await api('/api/config/raw', {
                    method: 'PATCH',
                    body: JSON.stringify({
                        file,
                        values: { dataset_config: '' },
                        content: currentTomlEditorContentForFile(file),
                    }),
                });
                if (!res.ok) {
                    setTomlStatus('error', res.error || '清除数据集预设失败');
                    return null;
                }
                if (typeof res.content === 'string' && file === (tomlState.currentTomlFile || val('toml-file-select'))) {
                    const editor = document.getElementById('toml-editor');
                    if (editor) {
                        editor.value = res.content;
                        tomlState.tomlSavedContent = res.content;
                    }
                }
                currentConfig.dataset_config = '';
                return { applied: true, response: res };
            }
            return { applied: false };
        }
        try {
            const res = await datasetPresetApi('/api/config/dataset-presets/apply', {
                method: 'POST',
                body: JSON.stringify({
                    dataset_file: nextDataset,
                    train_file: file,
                    train_content: currentTomlEditorContentForFile(file),
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '应用数据集预设失败');
                return null;
            }
            if (typeof res.train_content === 'string' && file === (tomlState.currentTomlFile || val('toml-file-select'))) {
                const editor = document.getElementById('toml-editor');
                if (editor) {
                    editor.value = res.train_content;
                    tomlState.tomlSavedContent = res.train_content;
                }
            }
            currentConfig.dataset_config = res.dataset_config || nextDataset;
            const values = res.values || {};
            for (const [key, value] of Object.entries(values)) {
                currentConfig[key] = value;
            }
            return { applied: true, response: res };
        } catch (e) {
            setTomlStatus('error', '应用数据集预设失败: ' + e.message);
            return null;
        }
    }

    export async function saveDatasetPresetEditor() {
        const datasetPresetState = currentDatasetPresetState();
        if (datasetPresetState.readonly) {
            setDatasetPresetStatus('系统数据集预设只读，请复制后编辑', 'error');
            return null;
        }
        let file = datasetPresetState.selectedFile || '';
        const wasUnnamedPreset = !file;
        if (!file) {
            const name = await showDatasetPresetNameDialog({
                title: '保存数据集预设',
                description: '当前预设还没有文件名。请输入一个名称，保存到 configs/datasets/。',
                confirmText: '保存预设',
            });
            if (name === null) return null;
            file = datasetPresetPathFromName(name);
            datasetPresetState.selectedFile = file;
        }
        const rows = normalizeDatasetEditorRows(datasetPresetState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        if (!rows.length || rows.some((row) => !row.source_dir.trim())) {
            setDatasetPresetStatus('请至少填写一个原始数据集路径', 'error');
            return null;
        }
        datasetPresetState.loading = true;
        datasetPresetState.error = '';
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
        try {
            const res = await datasetPresetApi('/api/config/dataset-presets', {
                method: 'PUT',
                body: JSON.stringify({
                    file,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                    overwrite: !(datasetPresetState.isNew || wasUnnamedPreset),
                }),
            });
            if (!res.ok) {
                datasetState.datasetPresetState.loading = false;
                renderDatasetPresetList();
                renderDatasetPresetHeader();
                renderDatasetEditor();
                setDatasetPresetStatus(res.error || '保存数据集预设失败', 'error');
                return null;
            }
            datasetState.datasetPresetState = {
                ...datasetPresetState,
                loading: false,
                selectedFile: res.file || file,
                datasets: normalizeDatasetEditorRows(res.datasets || rows),
                defaults: normalizeDatasetDefaults(res.defaults || datasetPresetState.defaults || {}),
                dirty: false,
                isNew: false,
                readonly: false,
                status: res.message || '已保存数据集预设',
            };
            const listRefreshed = await loadDatasetPresets({ selectCurrent: false, manage: true });
            if (!listRefreshed) {
                const message = datasetPresetState.error
                    ? `已保存 ${res.file || file}，但刷新左侧列表失败: ${datasetPresetState.error}`
                    : `已保存 ${res.file || file}，但左侧列表刷新失败，请点“刷新”或查看终端日志。`;
                setDatasetPresetStatus(message, 'warn');
                return res;
            }
            await loadDatasetPreset(datasetState.datasetPresetState.selectedFile);
            setDatasetPresetStatus(res.message || '已保存数据集预设', 'ok');
            if (datasetState.selectedConfigDatasetFile === datasetState.datasetPresetState.selectedFile) {
                datasetState.selectedConfigDatasetSummary = datasetPresetSummaryByFile(datasetState.selectedConfigDatasetFile);
                await loadStepEstimate();
            }
            return res;
        } catch (e) {
            datasetState.datasetPresetState.loading = false;
            renderDatasetPresetList();
            renderDatasetPresetHeader();
            renderDatasetEditor();
            setDatasetPresetStatus('保存数据集预设失败: ' + e.message, 'error');
            return null;
        }
    }

    export async function createNewDatasetPreset() {
        const datasetPresetState = currentDatasetPresetState();
        if (datasetPresetState.dirty && !(await confirmUnsavedDiscard('当前数据集预设有未保存修改，新建会丢弃这些修改。是否继续？'))) return;
        const name = await showDatasetPresetNameDialog({
            title: '新建数据集预设',
            description: '输入新预设名称，稍后保存时会写入 configs/datasets/。',
            confirmText: '创建预设',
        });
        if (name === null) return;
        const nextFile = datasetPresetPathFromName(name);
        if (datasetPresetByFile(nextFile)) {
            setDatasetPresetStatus('数据集预设已存在，请换一个名称或使用复制/重命名', 'error');
            return;
        }
        datasetState.datasetPresetState = {
            ...datasetPresetState,
            selectedFile: nextFile,
            datasets: normalizeDatasetEditorRows([{
                source_dir: '',
                image_dir: '',
                cache_dir: '',
                num_repeats: 1,
                settings: normalizeDatasetDefaults({}),
            }]),
            defaults: normalizeDatasetDefaults({}),
            dirty: true,
            isNew: true,
            readonly: false,
            error: '',
            status: '新预设尚未保存',
        };
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
    }

    export async function copyDatasetPreset() {
        const datasetPresetState = currentDatasetPresetState();
        if (!datasetPresetState.selectedFile) return;
        const name = await showDatasetPresetNameDialog({
            title: '复制数据集预设',
            description: '使用当前编辑器中的内容复制为新的数据集预设。',
            value: `${datasetPresetState.selectedFile.split('/').pop().replace(/\.toml$/i, '')}_copy`,
            confirmText: '复制预设',
        });
        if (name === null) return;
        const rows = normalizeDatasetEditorRows(datasetPresetState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        try {
            const res = await datasetPresetApi('/api/config/dataset-presets/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    name,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '复制数据集预设失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            await loadDatasetPreset(res.file);
            setDatasetPresetStatus('已复制数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('复制数据集预设失败: ' + e.message, 'error');
        }
    }

    export async function renameDatasetPreset() {
        const datasetPresetState = currentDatasetPresetState();
        const oldFile = datasetPresetState.selectedFile;
        if (!oldFile || datasetPresetState.readonly) return;
        const name = await showDatasetPresetNameDialog({
            title: '重命名数据集预设',
            description: '会先保存为新预设，再删除旧 TOML；图片、缩放图和缓存目录不受影响。',
            value: oldFile.split('/').pop().replace(/\.toml$/i, ''),
            confirmText: '重命名',
        });
        if (name === null) return;
        const nextFile = datasetPresetPathFromName(name);
        if (nextFile === oldFile) return;
        const saved = await copyDatasetPresetToName(name);
        if (!saved) return;
        try {
            const del = await datasetPresetApi(`/api/config/dataset-presets?file=${encodeURIComponent(oldFile)}`, { method: 'DELETE' });
            if (!del.ok) {
                setDatasetPresetStatus(del.error || '新预设已保存，但旧预设删除失败', 'error');
                return;
            }
            if (datasetState.selectedConfigDatasetFile === oldFile) datasetState.selectedConfigDatasetFile = nextFile;
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            await loadDatasetPreset(nextFile);
            renderConfigDatasetPicker();
            setDatasetPresetStatus('已重命名数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('重命名数据集预设失败: ' + e.message, 'error');
        }
    }

    export async function copyDatasetPresetToName(name) {
        const datasetPresetState = currentDatasetPresetState();
        try {
            const res = await datasetPresetApi('/api/config/dataset-presets/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    name,
                    datasets: datasetRowsForPayload(datasetPresetState.datasets),
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '保存新数据集预设失败', 'error');
                return null;
            }
            return res;
        } catch (e) {
            setDatasetPresetStatus('保存新数据集预设失败: ' + e.message, 'error');
            return null;
        }
    }

    export async function deleteDatasetPreset() {
        const datasetPresetState = currentDatasetPresetState();
        const file = datasetPresetState.selectedFile;
        if (!file || datasetPresetState.readonly) return;
        const ok = await showAppConfirmDialog({
            title: '删除数据集预设',
            description: file,
            message: '只删除数据集预设 TOML，不删除图片、缩放图或缓存目录。',
            confirmText: '删除预设',
            danger: true,
        });
        if (!ok) return;
        try {
            const res = await datasetPresetApi(`/api/config/dataset-presets?file=${encodeURIComponent(file)}`, { method: 'DELETE' });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '删除数据集预设失败', 'error');
                return;
            }
            if (datasetState.selectedConfigDatasetFile === file) {
                datasetState.selectedConfigDatasetFile = '';
                datasetState.selectedConfigDatasetSummary = null;
            }
            datasetState.datasetPresetState.selectedFile = '';
            datasetState.datasetPresetState.dirty = false;
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            renderConfigDatasetPicker();
            setDatasetPresetStatus('已删除数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('删除数据集预设失败: ' + e.message, 'error');
        }
    }

    export function importDatasetPreset() {
        document.getElementById('dataset-import-input')?.click();
    }

    export async function handleDatasetPresetImport(event) {
        const fileInput = event.target;
        const file = fileInput.files?.[0];
        if (!file) return;
        try {
            const content = await file.text();
            const name = await showDatasetPresetNameDialog({
                title: '导入数据集预设',
                description: '输入导入后的预设名称，文件会保存到 configs/datasets/。',
                value: file.name.replace(/\.toml$/i, ''),
                confirmText: '导入预设',
            });
            if (name === null) return;
            datasetState.datasetPresetState.loading = true;
            datasetState.datasetPresetState.error = '';
            renderDatasetPresetList();
            renderDatasetPresetHeader();
            renderDatasetEditor();
            const target = datasetPresetPathFromName(name);
            const res = await datasetPresetApi('/api/config/dataset-presets/import', {
                method: 'POST',
                body: JSON.stringify({ name, content }),
            });
            if (!res.ok) {
                datasetState.datasetPresetState.loading = false;
                renderDatasetPresetList();
                renderDatasetPresetHeader();
                renderDatasetEditor();
                setDatasetPresetStatus(res.error || '导入数据集预设失败', 'error');
                return;
            }
            datasetState.datasetPresetState = {
                ...currentDatasetPresetState(),
                loading: false,
                dirty: false,
                isNew: false,
                selectedFile: res.file || target,
                datasets: normalizeDatasetEditorRows(res.datasets || []),
                defaults: normalizeDatasetDefaults(res.defaults || {}),
                readonly: false,
                error: '',
            };
            const listRefreshed = await loadDatasetPresets({ selectCurrent: false, manage: true });
            if (!listRefreshed) {
                setDatasetPresetStatus(`已导入 ${res.file || target}，但刷新左侧列表失败，请点“刷新”或查看终端日志。`, 'warn');
                return;
            }
            await loadDatasetPreset(res.file || target);
            setDatasetPresetStatus('已导入数据集预设', 'ok');
        } catch (e) {
            datasetState.datasetPresetState.loading = false;
            renderDatasetPresetList();
            renderDatasetPresetHeader();
            renderDatasetEditor();
            setDatasetPresetStatus('导入数据集预设失败: ' + e.message, 'error');
        } finally {
            fileInput.value = '';
        }
    }

    export async function exportDatasetPreset() {
        const file = currentDatasetPresetState().selectedFile;
        if (!file) return;
        try {
            const data = await datasetPresetApi(`/api/config/dataset-presets/read?file=${encodeURIComponent(file)}`);
            if (!data.ok) {
                setDatasetPresetStatus(data.error || '导出数据集预设失败', 'error');
                return;
            }
            const blob = new Blob([data.content || ''], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = file.split('/').pop() || 'dataset.toml';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            setDatasetPresetStatus('已导出数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('导出数据集预设失败: ' + e.message, 'error');
        }
    }

    export function datasetPresetPathFromName(name) {
        const stem = String(name || '')
            .replace(/\.toml$/i, '')
            .replace(/\\/g, '/')
            .split('/')
            .pop()
            .replace(/[^A-Za-z0-9_-]+/g, '_')
            .replace(/^_+|_+$/g, '') || 'dataset';
        return `configs/datasets/${stem}.toml`;
    }

    export async function showDatasetPresetNameDialog(options = {}) {
        const name = await showHistoryTaskInputDialog({
            title: options.title || '数据集预设名称',
            description: options.description || '请输入数据集预设名称。',
            label: options.label || '预设名称',
            value: options.value || '',
            placeholder: options.placeholder || '例如 rokkotsu_goddess_v2',
            confirmText: options.confirmText || '确认',
        });
        if (name === null) return null;
        const clean = name.trim();
        if (!clean) {
            setDatasetPresetStatus('请输入数据集预设名称', 'error');
            return null;
        }
        return clean;
    }

    export function setDatasetPresetStatus(message, level = '') {
        datasetState.datasetPresetState.status = message || '';
        const header = document.getElementById('dataset-preset-header');
        if (!header) return;
        let status = header.querySelector('.dataset-preset-status');
        if (!status) {
            status = document.createElement('div');
            status.className = 'dataset-preset-status';
            header.appendChild(status);
        }
        status.textContent = message || '';
        status.className = ['dataset-preset-status', level].filter(Boolean).join(' ');
    }

    export async function createDatasetPresetGroup() {
        const label = await showHistoryTaskInputDialog({
            title: '新建数据集分组',
            description: '只用于整理 configs/datasets 下的数据集预设，不会修改训练配置内容。',
            label: '分组名称',
            placeholder: '例如：角色数据集 / 试验数据集 / 正式数据集',
            confirmText: '创建分组',
        });
        if (label === null) return;
        if (!label.trim()) {
            setDatasetPresetStatus('分组名称不能为空', 'error');
            return;
        }
        try {
            const res = await api('/api/config/file-groups', {
                method: 'POST',
                body: JSON.stringify({ label: label.trim(), kind: 'dataset' }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '创建数据集分组失败', 'error');
                return;
            }
            if (res.group?.kind !== 'dataset') {
                setDatasetPresetStatus('后端仍是旧版本，请重启 WebUI 后再创建数据集分组', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组已创建', 'ok');
        } catch (e) {
            setDatasetPresetStatus('创建数据集分组失败: ' + e.message, 'error');
        }
    }

    export async function renameDatasetPresetGroup(group) {
        if (!group?.id || !group.renamable) return;
        const label = await showHistoryTaskInputDialog({
            title: '重命名数据集分组',
            description: '只修改左侧分组名称，不会改动数据集 TOML 文件路径。',
            label: '分组名称',
            value: group.label || group.id,
            placeholder: '例如：正式数据集',
            confirmText: '保存名称',
        });
        if (label === null) return;
        if (!label.trim()) {
            setDatasetPresetStatus('分组名称不能为空', 'error');
            return;
        }
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'PATCH',
                body: JSON.stringify({ label: label.trim() }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '重命名数据集分组失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组已重命名', 'ok');
        } catch (e) {
            setDatasetPresetStatus('重命名数据集分组失败: ' + e.message, 'error');
        }
    }

configureDatasetPresetActionsBridge({
    applySelectedDatasetPresetToCurrentConfig,
    saveDatasetPresetEditor,
    createNewDatasetPreset,
    copyDatasetPreset,
    renameDatasetPreset,
    copyDatasetPresetToName,
    deleteDatasetPreset,
    importDatasetPreset,
    handleDatasetPresetImport,
    exportDatasetPreset,
    datasetPresetPathFromName,
    showDatasetPresetNameDialog,
    setDatasetPresetStatus,
    createDatasetPresetGroup,
    renameDatasetPresetGroup,
});
