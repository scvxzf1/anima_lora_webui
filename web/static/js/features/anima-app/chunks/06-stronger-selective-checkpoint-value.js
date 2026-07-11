/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    CONFIG_COMPACT_FIELD_GROUPS,
    GLOBAL_MODEL_PATH_FIELDS,
} from '../../../config/catalog.js?v=module-bootstrap-20260707-93';
import { SELECTIVE_CHECKPOINT_STRENGTH } from '../helpers/app-constants.js?v=module-bootstrap-20260711-1';
import { getAppShellState } from '../helpers/app-shell-state-bridge.js?v=module-bootstrap-20260707-93';
import { originalConfigFieldValue, readFieldInputValue } from '../helpers/config-form-bridge.js?v=module-bootstrap-20260707-93';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260707-93';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import { datasetPresetByFile } from '../helpers/dataset-presets.js?v=module-bootstrap-20260707-93';
import { historyTaskDisplayName } from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260707-93';
import {
    ensureConfigDatasetPreview,
    renderConfigDatasetPickerDialog,
} from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260707-93';
import { loadDatasetPresets } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260707-93';
import { createFieldRow, handleFormFieldChange } from './14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260707-93';
import {
    setTomlStatus,
    updateTomlActionState,
} from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260707-93';
import { loadGlobalSettings, getGlobalModelPathOverrides } from '../helpers/global-settings-bridge.js?v=module-bootstrap-20260707-93';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260707-93';
import { loadTrainingHistoryList } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260707-93';
import { api, val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260707-93';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260707-93';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260707-93';
import { currentTrainingConfigFile } from '../helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260707-93';

const appShellState = getAppShellState();
const configState = getConfigState();
const datasetState = getDatasetState();
const historyState = getHistoryState();
const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function currentContinueTrainingSource() {
    return trainingState.continueTrainingSource;
}


    export function strongerSelectiveCheckpointValue(current, fallback) {
        const currentKey = String(current ?? '').trim() || 'off';
        const fallbackKey = String(fallback ?? '').trim() || 'off';
        const currentStrength = SELECTIVE_CHECKPOINT_STRENGTH.get(currentKey);
        const fallbackStrength = SELECTIVE_CHECKPOINT_STRENGTH.get(fallbackKey);
        if (currentStrength === undefined) return fallbackKey;
        if (fallbackStrength === undefined) return currentKey;
        return currentStrength >= fallbackStrength ? currentKey : fallbackKey;
    }

    export function resourceQuickCurrentValue(key) {
        const configFormState = configState.configFormState;
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (input) {
            return readFieldInputValue(input, originalConfigFieldValue(key));
        }
        if (configFormState.draftValues.has(key)) {
            return configFormState.draftValues.get(key);
        }
        return originalConfigFieldValue(key);
    }

    export async function fillGlobalModelPathsIntoConfigForm() {
        if (!appShellState.globalSettings && location.protocol !== 'file:') {
            await loadGlobalSettings();
        }
        const overrides = getGlobalModelPathOverrides();
        const entries = GLOBAL_MODEL_PATH_FIELDS
            .map(([key]) => [key, overrides[key]])
            .filter(([, value]) => String(value || '').trim());
        if (!entries.length) {
            setTomlStatus('error', '全局设置里还没有可填写的基础模型路径');
            return;
        }

        const confirmed = await showAppConfirmDialog({
            title: '是否确认覆盖',
            description: '填写全局路径配置',
            message: '将用全局设置里的基础模型路径覆盖当前配置表单中的同名字段。',
            confirmText: '是',
            cancelText: '否',
        });
        if (!confirmed) return;

        let applied = 0;
        for (const [key, value] of entries) {
            const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
            if (!input) continue;
            input.value = value;
            applied += 1;
        }
        handleFormFieldChange();
        setTomlStatus(
            applied ? 'ok' : 'error',
            applied ? '已填写全局路径配置，请保存当前配置后再训练' : '当前表单没有可覆盖的基础模型路径字段'
        );
    }

    export function appendFieldRows(content, fields, groupClass) {
        const compactGroups = CONFIG_COMPACT_FIELD_GROUPS[groupClass] || [];
        const usedLayouts = new Set();
        let index = 0;

        while (index < fields.length) {
            const [key] = fields[index];
            const compactLayout = compactGroups.find((layout) => {
                if (usedLayouts.has(layout)) return false;
                return layout.keys.includes(key);
            });

            if (!compactLayout) {
                content.appendChild(createFieldRow(fields[index][0], fields[index][1]));
                index += 1;
                continue;
            }

            usedLayouts.add(compactLayout);
            const compactKeys = new Set(compactLayout.keys);
            const grid = document.createElement('div');
            grid.className = ['config-field-grid', compactLayout.className].filter(Boolean).join(' ');

            while (index < fields.length && compactKeys.has(fields[index][0])) {
                const [compactKey, compactValue] = fields[index];
                const row = createFieldRow(compactKey, compactValue);
                row.classList.add('field-row-compact');
                grid.appendChild(row);
                index += 1;
            }

            if (grid.childElementCount <= 1) {
                const onlyRow = grid.firstElementChild;
                if (onlyRow) content.appendChild(onlyRow);
            } else {
                normalizeCompactGridColumns(grid);
                appendCompactGridFillers(grid);
                content.appendChild(grid);
            }
        }
    }

    function appendCompactGridFillers(grid) {
        grid.querySelectorAll('.field-row-filler').forEach((node) => node.remove());
        const columnCount = compactGridColumnCount(grid);
        if (columnCount <= 1) return;
        const remainder = grid.childElementCount % columnCount;
        if (remainder === 0) return;
        const fillerCount = columnCount - remainder;
        for (let index = 0; index < fillerCount; index += 1) {
            grid.appendChild(createCompactGridFiller());
        }
    }

    function createCompactGridFiller() {
        const filler = document.createElement('div');
        filler.className = 'field-row field-row-compact field-row-filler';
        filler.setAttribute('aria-hidden', 'true');
        filler.setAttribute('role', 'presentation');
        return filler;
    }

    function compactGridColumnCount(grid) {
        if (!grid) return 0;
        if (grid.classList.contains('config-field-grid-5col')) return 5;
        if (grid.classList.contains('config-field-grid-4col')) return 4;
        if (grid.classList.contains('config-field-grid-3col')) return 3;
        if (grid.classList.contains('config-field-grid-2col')) return 2;
        return 2;
    }

    function preferredCompactGridColumns(grid) {
        if (!grid) return 0;
        if (grid.classList.contains('config-field-grid-5col')) return 5;
        if (grid.classList.contains('config-field-grid-4col')) return 4;
        if (grid.classList.contains('config-field-grid-3col')) return 3;
        if (grid.classList.contains('config-field-grid-2col')) return 2;
        return 0;
    }

    function normalizeCompactGridColumns(grid) {
        const count = grid.childElementCount;
        // Keep an explicit layout preference from CONFIG_COMPACT_FIELD_GROUPS
        // (e.g. four boolean flags as 2x2, not auto-upgraded to 4 columns).
        const preferred = preferredCompactGridColumns(grid);
        grid.classList.remove('config-field-grid-2col', 'config-field-grid-3col', 'config-field-grid-4col', 'config-field-grid-5col');
        if (preferred === 2) {
            grid.classList.add('config-field-grid-2col');
            return;
        }
        if (preferred === 3) {
            grid.classList.add('config-field-grid-3col');
            return;
        }
        if (preferred === 4) {
            grid.classList.add('config-field-grid-4col');
            return;
        }
        if (preferred === 5) {
            grid.classList.add('config-field-grid-5col');
            return;
        }
        if (count >= 4) {
            grid.classList.add('config-field-grid-4col');
        } else if (count === 3) {
            grid.classList.add('config-field-grid-3col');
        } else if (count === 2) {
            grid.classList.add('config-field-grid-2col');
        }
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
        if (isConfigDatasetPickerDialogOpen()) {
            renderConfigDatasetPickerDialog();
        }
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

    function renderContinueTrainingSource() {
        const summary = document.getElementById('continue-training-source-summary');
        const chooseBtn = document.getElementById('btn-open-continue-lora-dialog');
        const clearBtn = document.getElementById('btn-clear-continue-lora-source');
        if (!summary || !chooseBtn || !clearBtn) return;
        summary.innerHTML = '';
        const continueTrainingSource = currentContinueTrainingSource();
        if (!continueTrainingSource) {
            const title = document.createElement('strong');
            title.textContent = '从零开始';
            const detail = document.createElement('span');
            detail.textContent = '不加载已有权重';
            summary.append(title, detail);
            summary.className = 'continue-training-source-summary';
            chooseBtn.textContent = '选择 LoRA/LoHa/LoKr/GLoRA';
            clearBtn.hidden = true;
            updateTomlActionState(tomlState.currentTomlFile);
            return;
        }
        const title = document.createElement('strong');
        title.textContent = `权重热启动 ${continueTrainingSource.kind || 'LoRA'} · ${continueTrainingSource.name || '未命名权重'}`;
        const path = document.createElement('code');
        path.textContent = continueTrainingSource.abs_path || '';
        const state = document.createElement('span');
        state.className = continueTrainingSource.compatible === false ? 'warning' : 'ok';
        state.textContent = continueTrainingSource.compatible === false
            ? (continueTrainingSource.message || '当前配置不兼容')
            : '兼容 · 启动时会使用 --network_weights 与 --dim_from_weights';
        summary.append(title, path, state);
        summary.className = [
            'continue-training-source-summary',
            continueTrainingSource.compatible === false ? 'incompatible' : 'selected',
        ].join(' ');
        chooseBtn.textContent = '更换权重';
        clearBtn.hidden = false;
        updateTomlActionState(tomlState.currentTomlFile);
    }

    export function clearContinueTrainingSource() {
        trainingState.continueTrainingSource = null;
        renderContinueTrainingSource();
        setTomlStatus('ok', '已恢复为从零训练');
    }

    export async function openContinueLoraDialog() {
        const dialog = document.getElementById('continue-lora-dialog');
        if (!dialog) return;
        if (!historyState.historyTasks.length) {
            await loadTrainingHistoryList();
        }
        renderContinueLoraHistoryTasks();
        const input = document.getElementById('continue-lora-path-input');
        if (input && currentContinueTrainingSource()?.abs_path) {
            input.value = currentContinueTrainingSource().abs_path;
        }
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
        await loadContinueLoraWeights();
        document.getElementById('continue-lora-path-input')?.focus({ preventScroll: true });
    }

    function renderContinueLoraHistoryTasks() {
        const select = document.getElementById('continue-lora-history-task');
        if (!select) return;
        const previous = trainingState.continueLoraDialogState.taskId;
        const tasks = historyState.historyTasks.filter((task) => task.job === 'training');
        select.innerHTML = '';
        const latest = document.createElement('option');
        latest.value = '';
        latest.textContent = '最近一次训练输出';
        select.appendChild(latest);
        for (const task of tasks) {
            const option = document.createElement('option');
            option.value = task.id || '';
            option.textContent = historyTaskDisplayName(task) || task.id || '训练任务';
            select.appendChild(option);
        }
        if (previous && tasks.some((task) => task.id === previous)) {
            select.value = previous;
        } else {
            trainingState.continueLoraDialogState.taskId = '';
            select.value = '';
        }
    }

    export async function loadContinueLoraWeights() {
        const list = document.getElementById('continue-lora-weight-list');
        if (!list) return;
        trainingState.continueLoraDialogState.loading = true;
        trainingState.continueLoraDialogState.error = '';
        renderContinueLoraWeights();
        try {
            const params = new URLSearchParams();
            if (trainingState.continueLoraDialogState.taskId) {
                params.set('task_id', trainingState.continueLoraDialogState.taskId);
            }
            const suffix = params.toString() ? `?${params.toString()}` : '';
            const payload = await api(`/api/preview/weights${suffix}`);
            trainingState.continueLoraDialogState = {
                ...trainingState.continueLoraDialogState,
                loading: false,
                weights: payload.weights || [],
                error: payload.ok === false ? (payload.error || '读取权重失败') : '',
                message: payload.message || '',
            };
        } catch (e) {
            trainingState.continueLoraDialogState = {
                ...trainingState.continueLoraDialogState,
                loading: false,
                weights: [],
                error: e.message || '读取权重失败',
            };
        }
        renderContinueLoraWeights();
    }

    function renderContinueLoraWeights() {
        const list = document.getElementById('continue-lora-weight-list');
        if (!list) return;
        list.innerHTML = '';
        if (trainingState.continueLoraDialogState.loading) {
            list.textContent = '正在读取历史权重...';
            return;
        }
        if (trainingState.continueLoraDialogState.error) {
            list.textContent = trainingState.continueLoraDialogState.error;
            return;
        }
        if (!trainingState.continueLoraDialogState.weights.length) {
            list.textContent = trainingState.continueLoraDialogState.message || '没有可选择的 .safetensors 权重。';
            return;
        }
        for (const item of trainingState.continueLoraDialogState.weights) {
            const row = document.createElement('div');
            row.className = 'continue-lora-weight-item';
            const info = document.createElement('div');
            const name = document.createElement('strong');
            name.textContent = item.name || '未命名权重';
            const path = document.createElement('code');
            path.textContent = item.abs_path || item.file || '';
            info.append(name, path);
            const useBtn = document.createElement('button');
            useBtn.type = 'button';
            useBtn.className = 'btn btn-small btn-primary';
            useBtn.textContent = '热启动';
            useBtn.addEventListener('click', () => selectContinueLoraWeight(item.abs_path || item.file || ''));
            row.append(info, useBtn);
            list.appendChild(row);
        }
    }

    function setContinueLoraStatus(message, state = '') {
        const status = document.getElementById('continue-lora-inspect-status');
        if (!status) return;
        status.className = ['continue-lora-status', state].filter(Boolean).join(' ');
        status.textContent = message || '';
    }

    export async function requestContinueLoraInspection(path) {
        const currentTrainingSource = currentTrainingSourceState();
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        return api('/api/training/continue-lora/inspect', {
            method: 'POST',
            body: JSON.stringify({
                path,
                variant,
                preset,
                methods_subdir: methodsSubdir,
                config_file: currentTrainingConfigFile(),
            }),
        });
    }

    export async function selectContinueLoraWeight(path, options = {}) {
        const rawPath = String(path || '').trim();
        if (!rawPath) {
            setContinueLoraStatus('请填写 .safetensors 权重绝对路径。', 'error');
            return false;
        }
        setContinueLoraStatus('正在检查权重结构与当前变体兼容性...', 'pending');
        try {
            const payload = await requestContinueLoraInspection(rawPath);
            if (!payload.ok) {
                setContinueLoraStatus(payload.error || '权重检测失败。', 'error');
                if (!document.getElementById('continue-lora-dialog')?.open) {
                    alert(payload.error || '权重检测失败。');
                }
                return false;
            }
            if (!payload.compatible) {
                setContinueLoraStatus(payload.message || '当前配置与这个权重不兼容。', 'warning');
                if (!document.getElementById('continue-lora-dialog')?.open) {
                    alert(payload.message || '当前配置与这个权重不兼容。');
                }
                return false;
            }
            trainingState.continueTrainingSource = payload;
            renderContinueTrainingSource();
            setContinueLoraStatus(payload.message || '已选择权重热启动来源。', 'ok');
            setTomlStatus('ok', `训练来源已设置为权重热启动 ${payload.kind} · ${payload.name}`);
            if (options.switchToConfig !== false) {
                document.querySelector('[data-tab="config"]')?.click();
            }
            const dialog = document.getElementById('continue-lora-dialog');
            if (dialog?.open && options.keepDialogOpen !== true) dialog.close();
            return true;
        } catch (e) {
            setContinueLoraStatus('权重检测请求失败: ' + e.message, 'error');
            if (!document.getElementById('continue-lora-dialog')?.open) {
                alert('权重检测请求失败: ' + e.message);
            }
            return false;
        }
    }

    export async function refreshContinueTrainingSourceCompatibility() {
        const continueTrainingSource = currentContinueTrainingSource();
        if (!continueTrainingSource?.abs_path) {
            renderContinueTrainingSource();
            return true;
        }
        let payload;
        try {
            payload = await requestContinueLoraInspection(continueTrainingSource.abs_path);
        } catch (e) {
            trainingState.continueTrainingSource = {
                ...continueTrainingSource,
                compatible: false,
                message: '无法重新检查权重热启动来源: ' + e.message,
            };
            renderContinueTrainingSource();
            return false;
        }
        if (!payload.ok) {
            trainingState.continueTrainingSource = {
                ...continueTrainingSource,
                compatible: false,
                message: payload.error || '无法重新检查权重热启动来源。',
            };
            renderContinueTrainingSource();
            return false;
        }
        trainingState.continueTrainingSource = payload;
        renderContinueTrainingSource();
        return Boolean(payload.compatible);
    }
