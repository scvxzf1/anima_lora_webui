/**
 * Config-page training source modes (fresh / full resume / weight hotstart).
 * Moved out of anima-app mechanical chunks.
 */
import { isLiveRunningState } from '../live-training/index.js?v=module-bootstrap-20260707-93';
import { historyTaskDisplayName } from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260707-93';
import { readNonnegativeLiveNumber, readOptionalLiveNumber } from '../anima-app/helpers/live-form-values.js?v=module-bootstrap-20260707-93';
import { configureTrainingSourceBridge } from '../anima-app/helpers/training-source-bridge.js?v=module-bootstrap-20260707-93';
import { enterLiveTrainingForNewRun } from '../anima-app/helpers/training-launch-bridge.js?v=module-bootstrap-20260707-93';
import { clearContinueTrainingSource as BASE_CLEAR_CONTINUE_SOURCE, refreshContinueTrainingSourceCompatibility as BASE_REFRESH_CONTINUE_SOURCE, selectContinueLoraWeight as BASE_SELECT_CONTINUE_WEIGHT } from '../anima-app/chunks/06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260707-93';
import {
    setTomlStatus,
    updateTomlActionState,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260707-93';
import { getGpuPicker } from '../anima-app/helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260707-93';
import { renderPreflightPending, showPreflightRequestError } from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260707-93';
import { appendLog } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260707-93';
import { getContinueTrainingSource, getTrainingSourceState, getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260707-93';
import { showTrainingView, updateTrainingQueueFromPayload } from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260707-93';
import { loadTrainingHistoryList } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260707-93';
import { api } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260707-93';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260707-93';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260707-93';
import { optionNodeLocal, summaryLine, textNode } from '../anima-app/helpers/training-source-ui.js?v=module-bootstrap-20260707-93';
export const CONFIG_TRAINING_SOURCE_DOM_CONTRACT = Object.freeze({
    required: Object.freeze([
        'continue-training-source',
        'continue-training-source-summary',
        'config-full-resume-panel',
        'config-full-resume-task-select',
        'config-full-resume-checkpoint-select',
        'config-full-resume-summary',
        'config-weight-hotstart-panel',
        'config-weight-hotstart-detail',
        'config-training-source-status',
    ]),
    optional: Object.freeze([
        'btn-refresh-config-full-resume',
        'btn-open-continue-lora-dialog',
        'btn-clear-continue-lora-source',
        'preflight-dialog',
    ]),
});
const historyState = getHistoryState();
const tomlState = getTomlState();
const trainingState = getTrainingState();
const trainingRuntime = trainingState.trainingRuntime;

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function ensureTrainingSourceState() {
    const state = getTrainingSourceState();
    state.mode = normalizeConfigTrainingSourceMode(state.mode);
    state.full_resume = {
        task_id: '',
        checkpoint: '',
        checkpoints: [],
        default_checkpoint: '',
        current_step: null,
        target_total_steps: null,
        remaining_steps: null,
        resume_available: false,
        audit_status: 'idle',
        unavailable_reason: '',
        estimate_error: '',
        message: '',
        diagnostic: {},
        ...(state.full_resume || {}),
    };
    state.weight_hotstart = {
        abs_path: '',
        name: '',
        kind: '',
        compatible: false,
        audit_status: 'idle',
        unavailable_reason: '',
        ...(state.weight_hotstart || {}),
    };
    return state;
}
function normalizeConfigTrainingSourceMode(mode) {
    return ['fresh', 'full_resume', 'weight_hotstart'].includes(mode) ? mode : 'fresh';
}

export function configTrainingSourceMode() {
    return ensureTrainingSourceState().mode;
}
export async function setConfigTrainingSourceMode(mode, options = {}) {
    const state = ensureTrainingSourceState();
    state.mode = normalizeConfigTrainingSourceMode(mode);
    renderContinueTrainingSource();
    if (options.audit === false) return;
    if (state.mode === 'full_resume') {
        await auditConfigFullResumeSource({ force: true });
    } else if (state.mode === 'weight_hotstart') {
        await auditConfigWeightHotstartSource();
    } else {
        state.audit_status = 'ok';
        renderContinueTrainingSource();
    }
}
export async function auditConfigTrainingSourceOnEnter() {
    renderContinueTrainingSource();
    const mode = configTrainingSourceMode();
    if (mode === 'full_resume') return auditConfigFullResumeSource({ force: true });
    if (mode === 'weight_hotstart') return auditConfigWeightHotstartSource();
    return true;
}
export function renderContinueTrainingSource() {
    const state = ensureTrainingSourceState();
    const section = document.getElementById('continue-training-source');
    const summary = document.getElementById('continue-training-source-summary');
    if (!section || !summary) return;
    section.dataset.mode = state.mode;
    renderTrainingSourceModeButtons(state.mode);
    renderTrainingSourceSummary(summary, state);
    renderConfigFullResumePanel(state);
    renderConfigWeightHotstartPanel(state);
    renderConfigTrainingSourceStatus(state);
    updateTomlActionState(tomlState.currentTomlFile);
}
function renderTrainingSourceModeButtons(mode) {
    document.querySelectorAll('[data-training-source-mode]').forEach((btn) => {
        const active = btn.dataset.trainingSourceMode === mode;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
}
function renderTrainingSourceSummary(summary, state) {
    summary.innerHTML = '';
    summary.className = ['continue-training-source-summary', state.mode].join(' ');
    if (state.mode === 'full_resume') {
        const selected = selectedConfigFullResumeCheckpointFromState();
        const task = configFullResumeTaskById(state.full_resume.task_id);
        const usable = configFullResumeCheckpointUsable(selected);
        summary.classList.toggle('selected', usable);
        summary.classList.toggle('incompatible', Boolean(!usable));
        summary.append(
            textNode('strong', `完整续训${task ? ` · ${configFullResumeTaskLabel(task)}` : ''}`),
            textNode('span', selected ? configResumeRemainingText(selected) : (state.full_resume.message || '请选择历史训练任务和 checkpoint-state')),
        );
        if (selected?.path) summary.appendChild(textNode('code', selected.path));
        return;
    }
    if (state.mode === 'weight_hotstart') {
        syncWeightHotstartFieldsFromContinue();
        const source = getContinueTrainingSource();
        summary.classList.toggle('selected', Boolean(source?.abs_path && source.compatible !== false));
        summary.classList.toggle('incompatible', Boolean(!source?.abs_path || source.compatible === false));
        summary.append(
            textNode('strong', source?.abs_path ? `权重热启动 ${source.kind || 'LoRA'} · ${source.name || '未命名权重'}` : '权重热启动未选择'),
            textNode('span', source?.abs_path ? '只加载网络权重，从 Step 0 按当前配置训练。' : '请选择并审查 .safetensors 权重。'),
        );
        if (source?.abs_path) summary.appendChild(textNode('code', source.abs_path));
        const audit = state.weight_hotstart.audit_status;
        const stateLine = textNode('span', trainingSourceWeightStateText(state));
        stateLine.className = audit === 'ok' ? 'ok' : audit === 'checking' ? 'warning' : 'warning';
        summary.appendChild(stateLine);
        return;
    }
    summary.append(
        textNode('strong', '从零训练'),
        textNode('span', '不加载历史权重，不恢复 optimizer、scheduler 或已完成步数。'),
    );
}
function renderConfigFullResumePanel(state) {
    const panel = document.getElementById('config-full-resume-panel');
    const taskSelect = document.getElementById('config-full-resume-task-select');
    const checkpointSelect = document.getElementById('config-full-resume-checkpoint-select');
    const summary = document.getElementById('config-full-resume-summary');
    const refresh = document.getElementById('btn-refresh-config-full-resume');
    if (!panel || !taskSelect || !checkpointSelect || !summary) return;
    const active = state.mode === 'full_resume';
    panel.hidden = !active;
    const full = state.full_resume;
    const loading = full.audit_status === 'checking';

    const tasks = configResumeTaskCandidates();
    taskSelect.innerHTML = '';
    if (!tasks.length) {
        taskSelect.appendChild(optionNodeLocal('', '没有训练历史任务'));
    } else {
        taskSelect.appendChild(optionNodeLocal('', '选择历史训练任务'));
        for (const task of tasks) {
            taskSelect.appendChild(optionNodeLocal(task.id || '', configFullResumeTaskLabel(task)));
        }
    }
    taskSelect.value = full.task_id || '';
    taskSelect.disabled = loading;

    checkpointSelect.innerHTML = '';
    if (loading) {
        checkpointSelect.appendChild(optionNodeLocal('', '正在审查 checkpoint-state...'));
    } else if (full.checkpoints.length) {
        for (const item of full.checkpoints) {
            checkpointSelect.appendChild(optionNodeLocal(item.path, configResumeCheckpointLabel(item)));
        }
        checkpointSelect.value = full.checkpoint || full.default_checkpoint || full.checkpoints[0]?.path || '';
    } else {
        checkpointSelect.appendChild(optionNodeLocal('', '未找到完整续训状态目录'));
    }
    checkpointSelect.disabled = loading || !full.checkpoints.length;
    if (refresh) refresh.disabled = loading;

    summary.innerHTML = '';
    const selected = selectedConfigFullResumeCheckpointFromState();
    if (selected) {
        summary.append(
            summaryLine('状态目录', selected.path),
            summaryLine('已训练到', selected.step != null ? `Step ${selected.step}` : '步数未知'),
            summaryLine('目标 Step', selected.target_total_steps != null ? `Step ${selected.target_total_steps}` : '无法确认'),
            summaryLine('剩余步数', selected.remaining_steps != null ? String(selected.remaining_steps) : '无法确认'),
            summaryLine('检查点路径状态', configFullResumeCheckpointUsable(selected) ? 'train_state.json 可读取' : '不可用'),
        );
        if (selected.unavailable_reason) summary.appendChild(summaryLine('不可用原因', selected.unavailable_reason));
        if (selected.estimate_error) summary.appendChild(summaryLine('步数估算', `无法确认剩余步数: ${selected.estimate_error}`));
    } else {
        const p = document.createElement('p');
        p.textContent = full.message || full.unavailable_reason || '请选择一个历史任务，审查完成后才允许完整续训。';
        summary.appendChild(p);
    }
}
function renderConfigWeightHotstartPanel(state) {
    const panel = document.getElementById('config-weight-hotstart-panel');
    const chooseBtn = document.getElementById('btn-open-continue-lora-dialog');
    const clearBtn = document.getElementById('btn-clear-continue-lora-source');
    const detail = document.getElementById('config-weight-hotstart-detail');
    if (!panel || !chooseBtn || !clearBtn || !detail) return;
    panel.hidden = state.mode !== 'weight_hotstart';
    const source = getContinueTrainingSource();
    chooseBtn.textContent = source?.abs_path ? '更换权重' : '选择权重';
    clearBtn.hidden = !source?.abs_path;
    detail.innerHTML = '';
    if (!source?.abs_path) {
        detail.textContent = '权重热启动需要先选择 .safetensors，并通过类型、格式和当前变体兼容性检查。';
        return;
    }
    detail.append(
        summaryLine('权重文件', source.name || source.abs_path),
        summaryLine('权重类型', source.kind || '-'),
        summaryLine('权重路径', source.abs_path),
        summaryLine('恢复范围', '不恢复 optimizer / scheduler / 已完成步数'),
    );
}
function renderConfigTrainingSourceStatus(state) {
    const status = document.getElementById('config-training-source-status');
    if (!status) return;
    const readiness = trainingSourceLaunchReadiness();
    status.className = [
        'config-training-source-status',
        readiness.checking ? 'pending' : readiness.ready ? 'ok' : 'error',
    ].join(' ');
    if (state.mode === 'fresh') {
        status.textContent = '当前方案：从零训练。';
    } else if (readiness.checking) {
        status.textContent = '正在审查续接来源，审查完成前不能启动训练或加入队列。';
    } else {
        status.textContent = readiness.ready ? trainingSourceReadyText(state) : readiness.reason;
    }
}
function trainingSourceReadyText(state) {
    if (state.mode === 'full_resume') return '完整续训审查通过，将使用历史任务冻结配置快照。';
    if (state.mode === 'weight_hotstart') return '权重热启动审查通过，将按当前配置从 Step 0 训练。';
    return '从零训练可启动。';
}
export function trainingSourceLaunchReadiness() {
    const state = ensureTrainingSourceState();
    if (state.mode === 'fresh') return { ready: true, checking: false, reason: '' };
    if (state.mode === 'full_resume') {
        const full = state.full_resume;
        const selected = selectedConfigFullResumeCheckpointFromState();
        const appendCompleted = configFullResumeCanAppendCompletedCheckpoint(selected);
        if (full.audit_status === 'checking') return { ready: false, checking: true, reason: '正在审查续接来源' };
        if (!full.task_id) return { ready: false, checking: false, reason: '请选择要完整续训的历史训练任务。' };
        if (!selected) return { ready: false, checking: false, reason: full.unavailable_reason || '未找到可用 checkpoint-state/train_state.json。' };
        if (selected.resume_available === false && !appendCompleted) return { ready: false, checking: false, reason: selected.unavailable_reason || '这个检查点不可用于完整续训。' };
        if (full.audit_status !== 'ok' && !appendCompleted) return { ready: false, checking: false, reason: full.unavailable_reason || '完整续训来源审查未通过。' };
        return { ready: true, checking: false, reason: '' };
    }
    const source = getContinueTrainingSource();
    const weight = state.weight_hotstart;
    if (weight.audit_status === 'checking') return { ready: false, checking: true, reason: '正在审查续接来源' };
    if (!source?.abs_path) return { ready: false, checking: false, reason: '请选择可加载的 .safetensors 权重后再启动权重热启动。' };
    if (source.compatible === false || weight.audit_status !== 'ok') {
        return { ready: false, checking: false, reason: weight.unavailable_reason || source.message || '权重热启动审查未通过。' };
    }
    return { ready: true, checking: false, reason: '' };
}
export function trainingSourceLaunchBlockReason() {
    return trainingSourceLaunchReadiness().reason || '训练来源审查未通过。';
}
export async function ensureTrainingSourceReadyForLaunch() {
    const mode = configTrainingSourceMode();
    if (mode === 'full_resume') return auditConfigFullResumeSource({ force: true });
    if (mode === 'weight_hotstart') return auditConfigWeightHotstartSource();
    ensureTrainingSourceState().audit_status = 'ok';
    renderContinueTrainingSource();
    return true;
}
export function continueTrainingRequestPayload() {
    const source = getContinueTrainingSource();
    if (configTrainingSourceMode() !== 'weight_hotstart' || !source) return {};
    return {
        continue_from_weight_abs_path: source.abs_path || '',
        continue_from_weight_name: source.name || '',
        continue_from_weight_kind: source.kind || '',
    };
}
export async function auditConfigFullResumeSource(options = {}) {
    const state = ensureTrainingSourceState();
    const full = state.full_resume;
    if (!historyState.historyTasks.length && typeof loadTrainingHistoryList === 'function') {
        await loadTrainingHistoryList();
    }
    ensureFullResumeTaskSelection();
    if (!full.task_id) {
        setFullResumeAudit('error', '没有可用于完整续训的历史训练任务。');
        renderContinueTrainingSource();
        return false;
    }
    full.audit_status = 'checking';
    full.unavailable_reason = '';
    state.audit_status = 'checking';
    renderContinueTrainingSource();
    const seq = (state.fullResumeAuditSeq || 0) + 1;
    state.fullResumeAuditSeq = seq;
    try {
        const payload = await api(`/api/training/history/${encodeURIComponent(full.task_id)}/resume-options`);
        if (seq !== state.fullResumeAuditSeq) return false;
        const checkpoints = Array.isArray(payload.checkpoints) ? payload.checkpoints : [];
        full.checkpoints = checkpoints;
        full.default_checkpoint = payload.default_checkpoint || '';
        full.message = payload.message || '';
        full.diagnostic = payload.diagnostic || {};
        const selectedPath = chooseFullResumeCheckpointPath(full);
        full.checkpoint = selectedPath;
        const selected = selectedConfigFullResumeCheckpointFromState();
        syncFullResumeCheckpointFields(selected);
        if (payload.ok === false) {
            setFullResumeAudit('error', payload.error || '读取完整续训检查点失败。');
        } else if (!selected) {
            setFullResumeAudit('error', payload.message || '未找到 checkpoint-state/train_state.json，完整续训不可用。');
        } else if (selected.resume_available === false && !configFullResumeCanAppendCompletedCheckpoint(selected)) {
            setFullResumeAudit('error', selected.unavailable_reason || '这个检查点不可用于完整续训。');
        } else {
            setFullResumeAudit('ok', '');
        }
    } catch (e) {
        if (seq !== state.fullResumeAuditSeq) return false;
        full.checkpoints = [];
        full.checkpoint = '';
        setFullResumeAudit('error', '完整续训审查失败: ' + e.message);
    }
    renderContinueTrainingSource();
    return trainingSourceLaunchReadiness().ready;
}
function setFullResumeAudit(status, reason) {
    const state = ensureTrainingSourceState();
    state.full_resume.audit_status = status;
    state.full_resume.unavailable_reason = reason || '';
    state.audit_status = status;
}
export async function auditConfigWeightHotstartSource() {
    const state = ensureTrainingSourceState();
    if (!getContinueTrainingSource()?.abs_path) {
        syncWeightHotstartAuditFromContinue(false, '请选择可加载的 .safetensors 权重后再启动权重热启动。');
        renderContinueTrainingSource();
        return false;
    }
    state.weight_hotstart.audit_status = 'checking';
    state.audit_status = 'checking';
    renderContinueTrainingSource();
    if (!BASE_REFRESH_CONTINUE_SOURCE) {
        syncWeightHotstartAuditFromContinue(false, '权重审查入口未初始化。');
        renderContinueTrainingSource();
        return false;
    }
    const ok = await BASE_REFRESH_CONTINUE_SOURCE();
    syncWeightHotstartAuditFromContinue(ok);
    renderContinueTrainingSource();
    return trainingSourceLaunchReadiness().ready;
}
function syncWeightHotstartAuditFromContinue(ok = null, reason = '') {
    const state = ensureTrainingSourceState();
    const source = syncWeightHotstartFieldsFromContinue();
    const weight = state.weight_hotstart;
    if (!source.abs_path) {
        weight.audit_status = state.mode === 'weight_hotstart' ? 'error' : 'idle';
        weight.unavailable_reason = reason || '请选择可加载的 .safetensors 权重后再启动权重热启动。';
    } else if (weight.audit_status === 'checking' && ok == null && !reason) {
        state.audit_status = 'checking';
        return;
    } else if (ok === false || source.compatible === false) {
        weight.audit_status = 'error';
        weight.unavailable_reason = reason || source.message || '权重热启动审查未通过。';
    } else {
        weight.audit_status = 'ok';
        weight.unavailable_reason = '';
    }
    if (state.mode === 'weight_hotstart') state.audit_status = weight.audit_status;
}
function syncWeightHotstartFieldsFromContinue() {
    const source = getContinueTrainingSource() || {};
    const weight = ensureTrainingSourceState().weight_hotstart;
    weight.abs_path = source.abs_path || '';
    weight.name = source.name || '';
    weight.kind = source.kind || '';
    weight.compatible = Boolean(source.abs_path && source.compatible !== false);
    return source;
}
function configFullResumeDurationOverrides() {
    const epochs = readOptionalLiveNumber('max_train_epochs');
    if (epochs) return { max_train_epochs: epochs };
    const steps = readNonnegativeLiveNumber('max_train_steps', 0) || 0;
    return steps > 0 ? { max_train_steps: steps } : {};
}
function configFullResumeDurationText(duration) {
    if (duration?.max_train_epochs) return `从所选检查点再追加 ${duration.max_train_epochs} 轮。`;
    if (duration?.max_train_steps) return `从所选检查点再追加 ${duration.max_train_steps} 步。`;
    return '未填写当前训练时长时，将沿用历史冻结配置的剩余步数。';
}
function configFullResumeCanAppendCompletedCheckpoint(item) {
    const duration = configFullResumeDurationOverrides();
    const step = Number(item?.step), target = Number(item?.target_total_steps);
    const integrity = item?.state_integrity || {};
    return Boolean((duration.max_train_epochs || duration.max_train_steps) && Number.isFinite(step) && Number.isFinite(target) && step >= target && integrity.complete !== false);
}
function configFullResumeCheckpointUsable(item) { return Boolean(item && (item.resume_available !== false || configFullResumeCanAppendCompletedCheckpoint(item))); }
export async function startConfigFullResumeSource(queueMode = false) {
    if (!queueMode && isLiveRunningState(trainingRuntime.state)) {
        setTomlStatus('error', '当前已有训练或预处理在运行，请改用“加入队列”。', { persist: true });
        return;
    }
    if (!(await ensureTrainingSourceReadyForLaunch())) {
        setTomlStatus('error', trainingSourceLaunchBlockReason(), { persist: true });
        return;
    }
    const state = ensureTrainingSourceState();
    const selected = selectedConfigFullResumeCheckpointFromState();
    const task = configFullResumeTaskById(state.full_resume.task_id);
    if (!selected || !task) return;
    const durationOverrides = configFullResumeDurationOverrides();
    const ok = await showAppConfirmDialog({
        title: queueMode ? '完整续训加入队列' : '开始完整续训',
        description: configFullResumeTaskLabel(task),
        message: `将使用历史任务冻结配置快照，并从 ${selected.name || selected.path} 恢复 optimizer、scheduler 和已完成步数。${configFullResumeDurationText(durationOverrides)}其它配置页表单不会覆盖这次完整续训。`,
        confirmText: queueMode ? '加入队列' : '开始完整续训',
    });
    if (!ok) return;
    renderPreflightPending({
        title: queueMode ? '完整续训加入队列' : '启动完整续训',
        message: '正在提交完整续训请求...',
        detail: '后端会再次检查 checkpoint-state/train_state.json 和剩余步数。',
    });
    try {
        const res = await api(queueMode ? '/api/training/queue/resume' : '/api/training/resume', {
            method: 'POST',
            body: JSON.stringify({
                task_id: task.id,
                checkpoint: selected.path,
                duration_overrides: durationOverrides,
                gpu_whitelist: getGpuPicker()?.selectedGpuPayload?.() ?? [],
            }),
        });
        if (!res.ok) {
            showPreflightRequestError(res.error || '完整续训启动失败');
            return;
        }
        document.getElementById('preflight-dialog')?.close(queueMode ? 'queued' : 'training-started');
        if (queueMode) {
            updateTrainingQueueFromPayload(res);
            document.querySelector('[data-tab="training"]')?.click();
            showTrainingView('queue');
        } else {
            enterLiveTrainingForNewRun();
            await loadTrainingHistoryList();
        }
        appendLog(`[状态] ${res.message || (queueMode ? '完整续训已加入队列' : '完整续训已启动')}: ${selected.name || selected.path}`);
    } catch (e) {
        showPreflightRequestError('完整续训请求失败: ' + e.message);
    }
}
function ensureFullResumeTaskSelection() {
    const full = ensureTrainingSourceState().full_resume;
    const tasks = configResumeTaskCandidates();
    if (!tasks.some((task) => task.id === full.task_id)) {
        full.task_id = tasks[0]?.id || '';
        full.checkpoint = '';
        full.checkpoints = [];
    }
}
function chooseFullResumeCheckpointPath(full) {
    const exists = full.checkpoints.some((item) => item.path === full.checkpoint);
    if (exists) return full.checkpoint;
    const available = full.checkpoints.find((item) => item.resume_available !== false);
    return full.default_checkpoint || available?.path || full.checkpoints[0]?.path || '';
}
function selectedConfigFullResumeCheckpointFromState() {
    const full = ensureTrainingSourceState().full_resume;
    return full.checkpoints.find((item) => item.path === full.checkpoint) || null;
}
function syncFullResumeCheckpointFields(selected) {
    const full = ensureTrainingSourceState().full_resume;
    full.current_step = selected?.step ?? null;
    full.target_total_steps = selected?.target_total_steps ?? null;
    full.remaining_steps = selected?.remaining_steps ?? null;
    full.resume_available = configFullResumeCheckpointUsable(selected);
    full.estimate_error = selected?.estimate_error || '';
}
export async function handleConfigFullResumeTaskChange(value) {
    const full = ensureTrainingSourceState().full_resume;
    full.task_id = value || '';
    full.checkpoint = '';
    full.checkpoints = [];
    await auditConfigFullResumeSource({ force: true });
}
export function handleConfigFullResumeCheckpointChange(value) {
    const full = ensureTrainingSourceState().full_resume;
    full.checkpoint = value || '';
    const selected = selectedConfigFullResumeCheckpointFromState();
    syncFullResumeCheckpointFields(selected);
    const usable = configFullResumeCheckpointUsable(selected);
    setFullResumeAudit(
        usable ? 'ok' : 'error',
        usable ? '' : (selected?.unavailable_reason || (!selected ? '请选择可用的 checkpoint-state。' : '')),
    );
    renderContinueTrainingSource();
}
function configResumeTaskCandidates() {
    return (historyState.historyTasks || [])
        .filter((task) => task?.job === 'training' && task.id)
        .sort((a, b) => (Number(b.started_at || b.updated_at || 0) - Number(a.started_at || a.updated_at || 0)));
}
function configFullResumeTaskById(taskId) {
    return configResumeTaskCandidates().find((task) => task.id === taskId) || null;
}
function configFullResumeTaskLabel(task) {
    return (typeof historyTaskDisplayName === 'function' ? historyTaskDisplayName(task) : '') || task.label || task.name || task.id || '训练任务';
}
function configResumeCheckpointLabel(item) {
    const unavailable = item.resume_available === false ? '不可用' : '';
    return [
        item.kind_label || '训练状态',
        configResumeProgressText(item),
        item.scope_label || '',
        unavailable,
        item.name || '',
    ].filter(Boolean).join(' · ');
}
function configResumeProgressText(item) {
    const parts = [];
    if (item?.epoch != null) parts.push(`Epoch ${item.epoch}`);
    if (item?.step != null) parts.push(`Step ${item.step}`);
    return parts.join(' / ') || '步数未知';
}
function configResumeRemainingText(item) {
    if (item?.step != null && item?.target_total_steps != null) {
        const remaining = item.remaining_steps != null ? item.remaining_steps : Math.max(0, Number(item.target_total_steps) - Number(item.step));
        return `已训练到 Step ${item.step} / 目标 Step ${item.target_total_steps} / 剩余 ${remaining}`;
    }
    if (item?.estimate_error) return `${configResumeProgressText(item)} / 无法确认剩余步数`;
    return configResumeProgressText(item);
}
function trainingSourceWeightStateText(state) {
    const weight = state.weight_hotstart;
    if (weight.audit_status === 'checking') return '正在审查续接来源';
    if (weight.audit_status === 'ok') return '审查通过 · 不恢复 optimizer / scheduler / 已完成步数';
    return weight.unavailable_reason || '等待权重审查';
}
export function trainingSourceLaunchSummary() {
    const state = ensureTrainingSourceState();
    if (state.mode === 'full_resume') {
        const selected = selectedConfigFullResumeCheckpointFromState();
        const task = configFullResumeTaskById(state.full_resume.task_id);
        return `\n\n训练来源: 完整续训\n历史任务: ${task ? configFullResumeTaskLabel(task) : '-'}\n状态目录: ${selected?.path || '-'}\n${selected ? configResumeRemainingText(selected) : ''}`;
    }
    if (state.mode === 'weight_hotstart' && getContinueTrainingSource()?.abs_path) {
        const source = getContinueTrainingSource();
        return `\n\n训练来源: 权重热启动 ${source.kind || 'LoRA'} · ${source.name || ''}\n基于权重: ${source.abs_path}\n说明: 不恢复 optimizer、scheduler 和已完成步数`;
    }
    return '\n\n训练来源: 从零训练';
}
export async function refreshContinueTrainingSourceCompatibility() {
    if (!getContinueTrainingSource()?.abs_path) {
        syncWeightHotstartAuditFromContinue(configTrainingSourceMode() !== 'weight_hotstart');
        renderContinueTrainingSource();
        return configTrainingSourceMode() !== 'weight_hotstart';
    }
    if (configTrainingSourceMode() === 'weight_hotstart') {
        ensureTrainingSourceState().weight_hotstart.audit_status = 'checking';
        renderContinueTrainingSource();
    }
    const ok = await BASE_REFRESH_CONTINUE_SOURCE();
    syncWeightHotstartAuditFromContinue(ok);
    renderContinueTrainingSource();
    return ok;
}
export async function selectContinueLoraWeight(path, options = {}) {
    const ok = await BASE_SELECT_CONTINUE_WEIGHT(path, options);
    if (ok) {
        ensureTrainingSourceState().mode = 'weight_hotstart';
        syncWeightHotstartAuditFromContinue(true);
        renderContinueTrainingSource();
    }
    return ok;
}
export function clearContinueTrainingSource() {
    BASE_CLEAR_CONTINUE_SOURCE();
    const state = ensureTrainingSourceState();
    state.mode = 'fresh';
    state.weight_hotstart = {
        abs_path: '',
        name: '',
        kind: '',
        compatible: false,
        audit_status: 'idle',
        unavailable_reason: '',
    };
    renderContinueTrainingSource();
    setTomlStatus('ok', '已恢复为从零训练');
}
configureTrainingSourceBridge({ configTrainingSourceMode, setConfigTrainingSourceMode, auditConfigTrainingSourceOnEnter, renderContinueTrainingSource, trainingSourceLaunchReadiness, trainingSourceLaunchBlockReason, ensureTrainingSourceReadyForLaunch, continueTrainingRequestPayload, auditConfigFullResumeSource, auditConfigWeightHotstartSource, startConfigFullResumeSource, handleConfigFullResumeTaskChange, handleConfigFullResumeCheckpointChange, trainingSourceLaunchSummary, refreshContinueTrainingSourceCompatibility, selectContinueLoraWeight, clearContinueTrainingSource });
