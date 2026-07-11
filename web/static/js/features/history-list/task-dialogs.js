/**
 * History collection select dialogs and task detail/resume actions.
 */
import {
    formatSystemPercent,
    formatSystemTemperature,
    formatSystemVram,
    historySystemSummary,
} from '../history-detail/system.js?v=module-bootstrap-20260711-ir6';
import { ensurePreviewFeature } from '../anima-app/helpers/feature-ensurers.js?v=module-bootstrap-20260711-ir6';
import {
    historyCollectionOptionSearchText,
    historyCollectionSelectOptions,
    historyTaskDisplayName,
    moveHistoryCollectionValue,
} from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260711-ir6';
import { clearResumeOptions, historyStateLabel, metricsWithProgressFallback, renderConfigGroupTimeline, renderHistoryPaths, renderResumePanelState } from '../anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260711-ir6';
import { renderTrainingRunSummary } from '../anima-app/helpers/live-status-bridge.js?v=module-bootstrap-20260711-ir6';
import { formatLr, lastValue, readConfigNumber } from '../live-training/index.js?v=module-bootstrap-20260711-ir6';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { api } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir6';
import { escapeHtml } from '../config-form/field-input.js?v=module-bootstrap-20260711-ir6';
import { setEtaMetricText, setMetricText, setText, setTrainingDashboardHeadState, syncLossChartEmptyState, updateDashboardProgressIdleState, updateTrainingToolbarState } from '../live-training/dashboard-ui.js?v=module-bootstrap-20260711-ir6';
import { ensureHistoryDetailFeature } from '../anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260711-ir6';
import { closeSharedHistoryTaskDialog, openSharedHistoryTaskDialog, sharedHistoryTaskDialogIsOpen, sharedHistoryTaskDialogParts } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir6';
import { renderLogOutputLines, setLogStatus } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir6';
import { showTrainingView } from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260711-ir6';
import { loadTrainingHistoryList, renderHistoryManager, renderTrainingHistoryList, syncRecentHistorySidebarSelection } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir6';

const historyState = getHistoryState();
const trainingState = getTrainingState();
const tomlState = getTomlState();
const SIDEBAR_HISTORY_CACHE_LIMIT = 8;
const SIDEBAR_HISTORY_LOG_RENDER_LIMIT = 800;

function rememberSidebarHistoryPayload(taskId, payload) {
    const id = String(taskId || '').trim();
    if (!id || !payload?.ok) return;
    const cache = historyState.sidebarHistoryPayloadCache;
    if (cache.has(id)) cache.delete(id);
    cache.set(id, payload);
    while (cache.size > SIDEBAR_HISTORY_CACHE_LIMIT) {
        const oldest = cache.keys().next().value;
        cache.delete(oldest);
    }
}

function sidebarHistoryLogLines(logs) {
    const lines = (logs || []).map((record) => `${record.kind === 'progress' ? '[进度] ' : ''}${record.line || ''}`);
    if (lines.length <= SIDEBAR_HISTORY_LOG_RENDER_LIMIT) return lines;
    const omitted = lines.length - SIDEBAR_HISTORY_LOG_RENDER_LIMIT;
    return [
        `[提示] 日志较多，侧栏仅渲染最近 ${SIDEBAR_HISTORY_LOG_RENDER_LIMIT} 行（已省略 ${omitted} 行）。完整日志可在历史详情中查看。`,
        ...lines.slice(-SIDEBAR_HISTORY_LOG_RENDER_LIMIT),
    ];
}



export function showHistoryCollectionSelectDialog(options) {
    const wrap = document.createElement('div');
    wrap.className = 'history-collection-select-dialog';
    let selectedValue = String(options.value || '').trim();
    let query = selectedValue;

    const field = document.createElement('label');
    field.className = 'history-task-dialog-field';
    const label = document.createElement('span');
    label.textContent = '搜索或新建集合';
    const input = document.createElement('input');
    input.type = 'search';
    input.value = query;
    input.placeholder = '输入集合名，或从下方选择已有集合';
    input.className = 'history-task-dialog-input';
    field.append(label, input);

    const hint = document.createElement('p');
    hint.className = 'history-collection-select-hint';
    hint.textContent = '下拉列表按手动顺序排列；输入不存在的集合名会在保存时新建集合。';

    const orderActions = document.createElement('div');
    orderActions.className = 'history-collection-select-order';
    const list = document.createElement('div');
    list.className = 'history-collection-select-list';

    const renderOptions = () => {
        const optionsList = historyCollectionSelectOptions();
        const search = query.trim().toLowerCase();
        const visible = optionsList.filter((item) => !search || historyCollectionOptionSearchText(item).includes(search));
        list.innerHTML = '';
        if (!visible.length) {
            const empty = document.createElement('div');
            empty.className = 'history-collection-select-empty';
            empty.textContent = query.trim() ? `将新建集合: ${query.trim()}` : '暂无集合。';
            list.appendChild(empty);
        }
        for (const item of visible) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = ['history-collection-select-option', selectedValue === item.value ? 'selected' : ''].filter(Boolean).join(' ');
            btn.innerHTML = `<strong>${escapeHtml(item.label)}</strong><span>${item.task_count} 条任务 · ${item.group_count} 个配置分组</span>`;
            btn.addEventListener('click', () => {
                selectedValue = item.value;
                query = item.value;
                input.value = item.value;
                renderOptions();
            });
            list.appendChild(btn);
        }
        orderActions.querySelectorAll('button').forEach((btn) => {
            btn.disabled = !selectedValue;
        });
    };

    ['置顶', '上移', '下移', '置底'].forEach((labelText) => {
        const direction = { '置顶': 'top', '上移': 'up', '下移': 'down', '置底': 'bottom' }[labelText];
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'task-history-action';
        btn.textContent = labelText;
        btn.addEventListener('click', async (event) => {
            event.preventDefault();
            event.stopPropagation();
            await moveHistoryCollectionValue(selectedValue, direction);
            renderOptions();
        });
        orderActions.appendChild(btn);
    });

    input.addEventListener('input', () => {
        query = input.value || '';
        selectedValue = query.trim();
        renderOptions();
    });

    wrap.append(field, hint, orderActions, list);
    renderOptions();

    return showHistoryTaskDialog({
        title: options.title,
        description: options.description,
        body: wrap,
        confirmText: options.confirmText || '保存集合',
        onOpen: () => {
            input.focus();
            input.select();
        },
        getValue: () => query.trim(),
    });
}

export function showHistoryTaskConfirmDialog(options) {
    const wrap = document.createElement('div');
    wrap.className = 'history-task-dialog-message';
    const strong = document.createElement('strong');
    strong.textContent = options.description || '';
    const p = document.createElement('p');
    p.textContent = options.message || '';
    wrap.append(strong, p);
    return showHistoryTaskDialog({
        title: options.title,
        description: '',
        body: wrap,
        confirmText: options.confirmText || '确认',
        cancelText: options.cancelText || '取消',
        cancelPrimary: options.cancelPrimary,
        danger: options.danger,
        getValue: () => true,
    });
}

export function showHistoryTaskMessageDialog(options = {}) {
    const wrap = document.createElement('div');
    wrap.className = ['history-task-dialog-message', `tone-${options.tone || 'info'}`].filter(Boolean).join(' ');
    const message = document.createElement('p');
    message.textContent = options.message || '';
    if (options.message) wrap.appendChild(message);

    const detailLines = (options.detailLines || []).map((line) => String(line || '').trim()).filter(Boolean);
    if (detailLines.length) {
        const list = document.createElement('pre');
        list.className = 'history-task-dialog-detail-list';
        list.textContent = detailLines.join('\n');
        wrap.appendChild(list);
    }

    return showHistoryTaskDialog({
        title: options.title || '提示',
        description: options.description || '',
        body: wrap,
        confirmText: options.confirmText || '知道了',
        hideCancel: true,
        getValue: () => true,
    });
}

export function showHistoryTaskDialog(options) {
    const parts = sharedHistoryTaskDialogParts();
    if (!parts) {
        return Promise.resolve(null);
    }
    const { dialog, title, desc, body, cancelBtn, confirmBtn, closeBtn, form } = parts;
    if (tomlState.sharedDialogBusy || sharedHistoryTaskDialogIsOpen(dialog)) {
        return Promise.resolve(null);
    }
    tomlState.sharedDialogBusy = true;

    title.textContent = options.title || '任务操作';
    desc.textContent = options.description || '';
    body.innerHTML = '';
    if (options.body) body.appendChild(options.body);
    cancelBtn.textContent = options.cancelText || '取消';
    cancelBtn.classList.toggle('btn-primary', Boolean(options.cancelPrimary));
    cancelBtn.hidden = Boolean(options.hideCancel);
    confirmBtn.textContent = options.confirmText || '确认';
    confirmBtn.disabled = false;
    confirmBtn.classList.toggle('btn-danger', Boolean(options.danger));
    confirmBtn.classList.toggle('btn-primary', !options.danger);
    dialog.returnValue = '';

    return new Promise((resolve) => {
        let settled = false;
        const closeClick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            closeSharedHistoryTaskDialog(dialog, event.currentTarget?.value || 'cancel', handleClose);
        };
        const submitDialog = (event) => {
            event.preventDefault();
            const value = event.submitter?.value || 'confirm';
            if (value === 'confirm' && confirmBtn.disabled) return;
            closeSharedHistoryTaskDialog(dialog, value, handleClose);
        };
        const keydownDialog = (event) => {
            if (event.key !== 'Escape') return;
            event.preventDefault();
            closeSharedHistoryTaskDialog(dialog, 'cancel', handleClose);
        };
        const cleanup = () => {
            dialog.removeEventListener('close', handleClose);
            form?.removeEventListener('submit', submitDialog);
            closeBtn?.removeEventListener('click', closeClick);
            cancelBtn.removeEventListener('click', closeClick);
            confirmBtn.removeEventListener('click', closeClick);
            dialog.removeEventListener('keydown', keydownDialog);
            document.body.classList.remove('history-task-dialog-fallback-open');
            tomlState.sharedDialogBusy = false;
            cancelBtn.hidden = false;
            cancelBtn.classList.remove('btn-primary');
            confirmBtn.classList.remove('btn-danger');
            confirmBtn.classList.add('btn-primary');
        };
        const handleClose = () => {
            if (settled) return;
            settled = true;
            cleanup();
            if (dialog.returnValue === 'confirm') {
                resolve(options.getValue ? options.getValue() : true);
            } else {
                resolve(null);
            }
        };
        dialog.addEventListener('close', handleClose);
        form?.addEventListener('submit', submitDialog);
        closeBtn?.addEventListener('click', closeClick);
        cancelBtn.addEventListener('click', closeClick);
        confirmBtn.addEventListener('click', closeClick);
        dialog.addEventListener('keydown', keydownDialog);
        try {
            openSharedHistoryTaskDialog(dialog);
        } catch (e) {
            cleanup();
            resolve(null);
            return;
        }
        requestAnimationFrame(() => {
            if (options.onOpen) {
                options.onOpen();
            } else {
                confirmBtn.focus();
            }
        });
    });
}

export function normalizeHistoryDetailTab(tab) {
    return ensureHistoryDetailFeature().normalizeHistoryDetailTab(tab);
}

export function renderHistoryManagerDetail(payload = ensureHistoryDetailFeature().getCurrentPayload(), options = {}) {
    return ensureHistoryDetailFeature().renderHistoryManagerDetail(payload, options);
}

export function renderHistoryDetailDialog(payload = ensureHistoryDetailFeature().getCurrentPayload(), options = {}) {
    return ensureHistoryDetailFeature().renderHistoryDetailDialog(payload, options);
}

export function closeHistoryDetailDialog() {
    return ensureHistoryDetailFeature().closeHistoryDetailDialog();
}

export function isHistoryDetailDialogOpen() {
    return ensureHistoryDetailFeature().isHistoryDetailDialogOpen();
}

export function shouldRenderInlineResumePanel() {
    return historyState.historyViewMode !== 'live' && trainingState.trainingViewMode === 'live';
}

export function clearViewingHistoryTaskContext(payload = null) {
    if (payload?.mode === 'config_group') return;
    historyState.viewingHistoryTaskId = '';
    historyState.currentHistoryTaskForResume = null;
    if (historyState.historyViewMode !== 'config_group') {
        historyState.historyViewMode = 'live';
        historyState.currentHistoryConfigGroup = null;
        historyState.currentHistoryTimelineSelection = [];
    }
    renderResumePanelState();
}

export function handleHistoryDetailWindowKeydown(event) {
    return ensureHistoryDetailFeature().handleHistoryDetailWindowKeydown(event);
}

export function restorePreviewWorkspaceFromHistoryDetail() {
    return ensurePreviewFeature().restorePreviewWorkspaceFromHistoryDetail();
}

export function activateHistoryDetailPreview(payload) {
    return ensurePreviewFeature().activateHistoryDetailPreview(payload);
}

export function clearHistoryManagerDetail() {
    historyState.viewingHistoryTaskId = '';
    historyState.historyViewMode = 'live';
    historyState.currentHistoryTaskForResume = null;
    historyState.currentHistoryConfigGroup = null;
    historyState.currentHistoryTimelineSelection = [];
    ensureHistoryDetailFeature().clearHistoryDetailState();
    closeHistoryDetailDialog();
    clearResumeOptions();
    renderTrainingHistoryList();
    renderHistoryManager();
}

export function selectedHistoryManagerResumeCheckpoint() {
    return ensureHistoryDetailFeature().selectedHistoryManagerResumeCheckpoint();
}

export async function resumeTrainingFromHistoryDetail(queueMode) {
    return ensureHistoryDetailFeature().resumeTrainingFromHistoryDetail(queueMode);
}

export async function loadHistoryTask(taskId, options = {}) {
    return ensureHistoryDetailFeature().loadHistoryTask(taskId, options);
}

export async function openSidebarHistoryTask(taskId) {
    const id = String(taskId || '').trim();
    if (!id) return;
    const requestId = (historyState.sidebarHistoryRequestId || 0) + 1;
    historyState.sidebarHistoryRequestId = requestId;

    // Paint selection immediately so rapid sidebar switching feels responsive.
    showTrainingView('live');
    closeHistoryDetailDialog();
    historyState.historyViewMode = 'task';
    historyState.viewingHistoryTaskId = id;
    historyState.currentHistoryConfigGroup = null;
    historyState.currentHistoryTimelineSelection = [];
    const cached = historyState.sidebarHistoryPayloadCache.get(id);
    if (cached?.task) {
        historyState.currentHistoryTaskForResume = cached.task;
    }
    ensureHistoryDetailFeature().clearHistoryDetailState();
    ensureHistoryDetailFeature().resetCurveHover();
    syncRecentHistorySidebarSelection();
    if (cached?.ok) {
        renderHistoryTask(cached, { stickLogsToBottom: false });
    }

    try {
        const payload = await api(`/api/training/history/${encodeURIComponent(id)}`);
        if (requestId !== historyState.sidebarHistoryRequestId) return;
        if (!payload.ok) {
            await showHistoryTaskMessageDialog({
                title: '读取历史任务失败',
                message: payload.error || '读取历史任务失败',
                tone: 'error',
            });
            return;
        }
        rememberSidebarHistoryPayload(id, payload);
        historyState.currentHistoryTaskForResume = payload.task || null;
        renderHistoryTask(payload, { stickLogsToBottom: true });
        syncRecentHistorySidebarSelection();
    } catch (e) {
        if (requestId !== historyState.sidebarHistoryRequestId) return;
        await showHistoryTaskMessageDialog({
            title: '读取历史任务失败',
            message: e.message,
            tone: 'error',
        });
    }
}

export async function refreshHistoryView() {
    if (historyState.historyViewMode === 'config_group' && historyState.currentHistoryConfigGroup) {
        await loadConfigGroupTimeline(historyState.currentHistoryConfigGroup, {
            taskIds: historyState.currentHistoryTimelineSelection,
            skipSelectionDialog: true,
        });
        return;
    }
    if (!historyState.viewingHistoryTaskId) return;
    await openSidebarHistoryTask(historyState.viewingHistoryTaskId);
}

export async function loadConfigGroupTimeline(group, options = {}) {
    if (!group?.history_group_key && (!group?.methods_subdir || !group?.variant)) return;
    const taskIds = Array.isArray(options.taskIds) ? options.taskIds.filter(Boolean) : [];
    const query = new URLSearchParams({
        methods_subdir: group.methods_subdir || '',
        variant: group.variant || '',
        preset: group.preset || 'default',
        include_archived: historyState.showArchivedHistory ? '1' : '0',
    });
    if (!taskIds.length && group.history_group_key) {
        query.set('group_key', group.history_group_key);
    }
    for (const taskId of taskIds) {
        query.append('task_id', taskId);
    }
    try {
        const payload = await api(`/api/training/history/config-group/timeline?${query.toString()}`);
        if (!payload.ok) {
            await showHistoryTaskMessageDialog({
                title: '读取配置分组失败',
                message: payload.error || '读取配置分组合并日志失败',
                tone: 'error',
            });
            return;
        }
        if (options.detailTab) {
            ensureHistoryDetailFeature().setActiveTab(options.detailTab);
        }
        historyState.historyViewMode = 'config_group';
        historyState.viewingHistoryTaskId = '';
        showTrainingView('history');
        historyState.currentHistoryConfigGroup = payload.group || group;
        historyState.currentHistoryTimelineSelection = (payload.summary?.selected_task_ids || taskIds || []).filter(Boolean);
        historyState.currentHistoryTaskForResume = null;
        clearResumeOptions();
        ensureHistoryDetailFeature().resetCurveHover();
        renderTrainingHistoryList();
        renderConfigGroupTimeline(payload);
        renderHistoryManagerDetail(payload, { open: true });
    } catch (e) {
        await showHistoryTaskMessageDialog({
            title: '读取配置分组失败',
            message: e.message,
            tone: 'error',
        });
    }
}

export function historyTaskStepOffset(task) {
    const resume = task?.resume_from || {};
    const step = Number(resume.checkpoint_step || 0);
    return Number.isFinite(step) && step > 0 ? step : 0;
}

export function historyLossChartPoints(lossPoints, task) {
    const offset = historyTaskStepOffset(task);
    const out = [];
    let maxStep = null;
    for (const item of lossPoints || []) {
        const rawStep = Number(item.step);
        if (!Number.isFinite(rawStep)) continue;
        const step = rawStep + offset;
        if (maxStep !== null && step < maxStep) continue;
        if (maxStep === null || step > maxStep) maxStep = step;
        out.push({
            step,
            loss: item.loss,
            rawStep,
            displayStepOffset: offset,
            ts: item.ts,
            rate: item.rate,
            lr: item.lr,
            sourceTaskLabel: task ? historyTaskDisplayName(task) : '',
            sourceTaskId: task?.id || '',
        });
    }
    return out;
}

export function renderHistoryTask(payload, options = {}) {
    const task = payload.task || {};
    historyState.currentHistoryTaskForResume = task;
    const banner = document.getElementById('history-view-banner');
    const bannerTitle = document.getElementById('history-view-title');
    if (banner) banner.hidden = false;
    if (bannerTitle) {
        bannerTitle.textContent = `历史任务: ${historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`} · ${historyStateLabel(task.state)}`;
    }
    setText('training-run-state', '历史');
    const stateEl = document.getElementById('training-run-state');
    if (stateEl) stateEl.className = 'training-run-state history';
    setTrainingDashboardHeadState('history');
    updateTrainingToolbarState('history', '历史');
    setText('training-run-title', historyTaskDisplayName(task) || '历史任务');
    setText('training-run-meta', [
        task.methods_subdir ? `方法目录 ${task.methods_subdir}` : '',
        task.variant ? `配置 ${task.variant}` : '',
        task.preset ? `预设 ${task.preset}` : '',
    ].filter(Boolean).join(' · ') || '历史任务记录');
    renderTrainingRunSummary([
        ['运行目录', task.run_dir],
        ['输出', task.output_dir],
        ['样张', task.sample_dir],
    ], '该任务没有记录运行目录。');
    document.getElementById('train-variant').textContent = task.variant || '-';
    document.getElementById('train-preset').textContent = task.preset || '-';
    document.getElementById('progress-bar').style.width = task.state === 'idle' ? '100%' : '0%';
    document.getElementById('progress-text').textContent = `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`;
    updateDashboardProgressIdleState(true);
    setMetricText('metric-vram', 'N/A');
    setMetricText('metric-vram-peak', 'N/A');
    setMetricText('metric-gpu', 'N/A');
    setMetricText('metric-gpu-peak', 'N/A');
    setMetricText('metric-temp', 'N/A');
    setMetricText('metric-temp-peak', 'N/A');
    setMetricText('metric-log-age', task.finished_at_text ? '已结束' : '历史');
    setMetricText('metric-rate', 'N/A');

    const logs = payload.logs || [];
    const metrics = metricsWithProgressFallback(payload.metrics || [], logs);
    const lossPoints = metrics.filter((item) => item.loss !== undefined);
    const chartPoints = historyLossChartPoints(lossPoints, task);
    trainingState.lossChart?.setXLabel?.('step');
    trainingState.lossChart?.setScaleMode?.('step', {
        xRange: {
            min: chartPoints[0]?.step,
            max: chartPoints[chartPoints.length - 1]?.step,
        },
    });
    trainingState.lossChart?.setData(chartPoints, { keepAll: true });
    syncLossChartEmptyState();
    const lastMetric = metrics[metrics.length - 1] || {};
    const lastLossMetric = lossPoints[lossPoints.length - 1] || {};
    const lastChartPoint = chartPoints[chartPoints.length - 1] || {};
    const configLr = readConfigNumber(payload.config_toml, 'learning_rate');
    const system = payload.system || [];
    const lastSystem = system[system.length - 1] || {};
    const systemSummary = historySystemSummary(payload);
    setMetricText('metric-loss', lastMetric.loss !== undefined ? Number(lastMetric.loss).toFixed(5) : 'N/A');
    setMetricText('metric-lr', formatLr(lastValue(metrics, 'lr') ?? configLr));
    setMetricText('metric-step', lastChartPoint.step ?? lastValue(metrics, 'step') ?? lastLossMetric.step ?? 'N/A');
    setMetricText('metric-rate', lastValue(metrics, 'rate') || 'N/A');
    setMetricText('metric-vram',
        lastSystem.vram_used_gb !== undefined ? `${lastSystem.vram_used_gb}/${lastSystem.vram_total_gb} GB` : 'N/A');
    setMetricText('metric-vram-peak',
        systemSummary.hasSystem ? formatSystemVram(systemSummary.peakVramRecord) : 'N/A');
    if (lastSystem.gpu_util !== undefined) {
        setMetricText('metric-gpu', `${lastSystem.gpu_util}%${lastSystem.gpu_temp ? ` ${lastSystem.gpu_temp}°C` : ''}`);
    } else {
        setMetricText('metric-gpu', 'N/A');
    }
    setMetricText('metric-gpu-peak',
        systemSummary.hasSystem ? formatSystemPercent(systemSummary.peakGpu) : 'N/A');
    setMetricText('metric-temp',
        lastSystem.gpu_temp !== undefined ? formatSystemTemperature(lastSystem.gpu_temp) : 'N/A');
    setMetricText('metric-temp-peak',
        systemSummary.hasSystem ? formatSystemTemperature(systemSummary.peakTemp) : 'N/A');
    setEtaMetricText({
        text: task.finished_at_text || '历史',
        empty: !task.finished_at_text,
        title: task.finished_at_text ? '历史任务完成时间。' : '历史任务未记录完成时间。',
    });

    const logEl = document.getElementById('log-output');
    const logLines = sidebarHistoryLogLines(logs);
    renderLogOutputLines(logLines, { stickToBottom: options.stickLogsToBottom !== false });
    if (options.stickLogsToBottom !== false && logEl) logEl.scrollTop = logEl.scrollHeight;
    const totalLogs = (payload.logs || []).length;
    const renderedLogs = Math.min(totalLogs, SIDEBAR_HISTORY_LOG_RENDER_LIMIT);
    setLogStatus(
        totalLogs > renderedLogs
            ? `历史 · 渲染 ${renderedLogs}/${totalLogs} 行`
            : `历史 · ${totalLogs} 行`,
        'warning',
    );

    const health = document.getElementById('training-health');
    health.className = 'training-health';
    health.textContent = [
        task.message || '历史任务记录',
        task.history_dir ? `历史目录: ${task.history_dir}` : '',
        task.output_dir ? `输出目录: ${task.output_dir}` : '',
        task.sample_dir ? `样张目录: ${task.sample_dir}` : '',
    ].filter(Boolean).join(' · ');

    const configPanel = document.getElementById('history-config-panel');
    const configTitle = document.getElementById('history-config-title');
    const configOutput = document.getElementById('history-config-output');
    if (configPanel) configPanel.hidden = false;
    if (configTitle) configTitle.textContent = '任务配置快照';
    if (configOutput) configOutput.textContent = payload.config_toml || '# 无配置快照';
    renderHistoryPaths(task);
    renderResumePanelState();
    if (trainingState.trainingViewMode === 'history') renderHistoryManagerDetail(payload);
}
