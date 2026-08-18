/* Dragon training history controller: list filters, detail resources, and resume actions. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { renderLossChart } from './live-training.js?v=dragon-ui-20260814v43';
import { scanForReveal } from '../animations.js?v=dragon-ui-20260812v35';
import { bindHistorySampleDialog } from './history-sample-dialog.js?v=dragon-ui-20260819v2';
import {
    renderHistoryDetailError,
    renderHistoryDetailPage,
    renderHistoryPage,
    renderHistoryResults,
    taskDisplayName,
} from './history-view.js?v=dragon-ui-20260819v92';

const api = createApiClient();

export async function loadHistory(context = {}) {
    const taskId = context.taskId || null;
    if (taskId) return loadHistoryDetail(taskId);

    const payload = await safeApi('/api/training/history?limit=200', '读取训练历史失败');
    const state = {
        tasks: Array.isArray(payload.tasks) ? payload.tasks : [],
        error: payload.ok === false ? (payload.error || '读取训练历史失败') : '',
        query: '',
        status: 'all',
        requestSequence: 0,
    };
    return {
        html: renderHistoryPage(state),
        onMount: (root) => bindHistoryList(root, state),
        onUnmount: () => { state.requestSequence += 1; },
    };
}

async function loadHistoryDetail(taskId) {
    const payload = await safeApi(
        `/api/training/history/${encodeURIComponent(taskId)}`,
        '加载训练记录失败',
    );
    if (payload.ok === false) {
        return {
            html: renderHistoryDetailError(taskId, payload.error || '加载训练记录失败'),
            onMount: bindHistoryNavigation,
        };
    }

    const task = payload.task || {};
    const [images, weights, resume] = await Promise.all([
        safeApi(`/api/preview/images?source=training&task_id=${encodeURIComponent(taskId)}&limit=8`, '读取训练样张失败'),
        safeApi(`/api/preview/weights?task_id=${encodeURIComponent(taskId)}`, '读取训练权重失败'),
        task.job === 'training'
            ? safeApi(`/api/training/history/${encodeURIComponent(taskId)}/resume-options`, '读取续训检查点失败')
            : Promise.resolve({ ok: true, checkpoints: [], message: '预处理任务不支持续训。' }),
    ]);
    const model = {
        taskId,
        payload,
        images,
        weights,
        resume,
        lossChart: renderLossChart(Array.isArray(payload.metrics) ? payload.metrics : []),
    };
    return { html: renderHistoryDetailPage(model), onMount: (root) => bindHistoryDetail(root, model) };
}

function bindHistoryList(root, state) {
    const queryInput = root.querySelector('[data-history-search]');
    const statusSelect = root.querySelector('[data-history-status]');
    queryInput?.addEventListener('input', () => {
        state.query = queryInput.value || '';
        updateHistoryResults(root, state);
    });
    statusSelect?.addEventListener('change', () => {
        state.status = statusSelect.value || 'all';
        updateHistoryResults(root, state);
    });
    root.querySelector('[data-history-refresh]')?.addEventListener('click', () => refreshHistory(root, state));
}

function updateHistoryResults(root, state) {
    const model = renderHistoryResults(state.tasks, state.query, state.status);
    const results = root.querySelector('[data-history-results]');
    const count = root.querySelector('[data-history-count]');
    if (results) results.innerHTML = model.html;
    if (count) count.textContent = `${model.visibleCount} / ${state.tasks.length} 条记录`;
    scanForReveal();
}

async function refreshHistory(root, state) {
    const sequence = ++state.requestSequence;
    const button = root.querySelector('[data-history-refresh]');
    if (button) button.disabled = true;
    setHistoryStatus(root, '正在刷新训练历史…', 'info');
    const payload = await safeApi('/api/training/history?limit=200', '刷新训练历史失败');
    if (sequence !== state.requestSequence) return;
    if (button) button.disabled = false;
    if (payload.ok === false) {
        setHistoryStatus(root, `${payload.error || '刷新训练历史失败'}。请检查 WebUI 服务后重试。`, 'error');
        return;
    }
    state.tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
    state.error = '';
    updateHistoryResults(root, state);
    setHistoryStatus(root, `已刷新，共 ${state.tasks.length} 条训练记录。`, 'success');
}

function bindHistoryNavigation(root) {
    root.querySelector('[data-history-back]')?.addEventListener('click', () => {
        window.location.hash = '#history';
    });
    root.querySelector('[data-history-detail-refresh]')?.addEventListener('click', () => {
        window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    });
}

function bindHistoryDetail(root, model) {
    bindHistoryNavigation(root);
    root.querySelectorAll('[data-history-resume-mode]').forEach((button) => {
        button.addEventListener('click', () => resumeHistoryTask(root, model, button.dataset.historyResumeMode));
    });
    bindHistorySampleDialog(root, Array.isArray(model.images.images) ? model.images.images : []);
}

async function resumeHistoryTask(root, model, mode) {
    const select = root.querySelector('[data-history-resume-checkpoint]');
    const checkpoint = String(select?.value || '').trim();
    if (!checkpoint) {
        setResumeStatus(root, '请先选择一个可用的训练状态目录。', 'error');
        select?.focus();
        return;
    }
    const queueMode = mode === 'queue';
    const action = queueMode ? '加入续训队列' : '立即继续训练';
    const confirmed = window.confirm(`确认${action}“${taskDisplayName(model.payload.task)}”吗？\n\n将使用历史配置快照和所选训练状态目录创建一条新任务记录。`);
    if (!confirmed) return;

    setResumeBusy(root, true);
    setResumeStatus(root, queueMode ? '正在加入续训队列…' : '正在启动续训…', 'info');
    const endpoint = queueMode ? '/api/training/queue/resume' : '/api/training/resume';
    const payload = await safeApi(endpoint, queueMode ? '加入续训队列失败' : '续训启动失败', {
        method: 'POST',
        body: JSON.stringify({ task_id: model.taskId, checkpoint }),
    });
    if (payload.ok === false) {
        setResumeBusy(root, false);
        setResumeStatus(root, `${payload.error || '续训操作失败'}。请检查任务状态与检查点后重试。`, 'error');
        return;
    }
    setResumeStatus(root, payload.message || (queueMode ? '续训任务已加入队列。' : '续训已启动。'), 'success');
    window.location.hash = queueMode ? '#queue' : '#live-training';
}

function setResumeBusy(root, busy) {
    root.querySelectorAll('[data-history-resume-mode]').forEach((button) => { button.disabled = busy; });
    const select = root.querySelector('[data-history-resume-checkpoint]');
    if (select) select.disabled = busy;
}

function setResumeStatus(root, message, tone) {
    const status = root.querySelector('[data-history-resume-status]');
    if (!status) return;
    status.textContent = message;
    status.dataset.tone = tone;
    status.classList.toggle('dragon-config-feedback-visible', Boolean(message));
}

function setHistoryStatus(root, message, tone) {
    const status = root.querySelector('[data-history-status-region]');
    if (!status) return;
    status.textContent = message;
    status.dataset.tone = tone;
    status.classList.toggle('dragon-config-feedback-visible', Boolean(message));
}

async function safeApi(url, fallback, options) {
    try {
        const payload = await api(url, options);
        if (payload?.ok === false) return { ...payload, error: payload.error || fallback };
        return payload || { ok: true };
    } catch (error) {
        return { ok: false, error: error.message || fallback };
    }
}
