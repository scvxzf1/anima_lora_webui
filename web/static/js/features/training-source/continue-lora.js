/**
 * Continue-training LoRA weight hotstart helpers.
 * Extracted from anima-app chunk 06.
 */
import { historyTaskDisplayName } from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    setTomlStatus,
    updateTomlActionState,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { loadTrainingHistoryList } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { api, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { currentTrainingConfigFile } from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260809-nf4-v2';

const historyState = getHistoryState();
const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function currentContinueTrainingSource() {
    return trainingState.continueTrainingSource;
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
