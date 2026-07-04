import {
    fetchPreviewImages,
    fetchPreviewSettings,
    fetchPreviewWeights,
    savePreviewSettingsRequest,
} from './api.js?v=module-bootstrap-20260704-1';
import {
    createPreviewDetailBlock,
    createPreviewDetailRow,
    createPreviewDialog,
} from './dialog.js?v=module-bootstrap-20260704-1';
import { createPreviewImages } from './images.js?v=module-bootstrap-20260704-1';
import {
    applyPreviewSelectionValue,
    createPreviewState,
    encodePreviewGroupValue,
    encodePreviewTaskValue,
    normalizePreviewGroup,
    previewSourceLabel,
    selectedPreviewSelectValue,
} from './state.js?v=module-bootstrap-20260704-1';
import { createPreviewWeights } from './weights.js?v=module-bootstrap-20260704-1';
import { createPreviewWorkspace } from './workspace.js?v=module-bootstrap-20260704-1';

export function createPreviewFeature(ctx, deps) {
    const state = createPreviewState();
    const dialog = createPreviewDialog({ ctx, deps });
    const workspace = createPreviewWorkspace({
        state,
        deps,
        closePreviewImageDialog: dialog.closePreviewImageDialog,
    });
    const images = createPreviewImages({
        ctx,
        state,
        deps,
        openPreviewDialog: dialog.openPreviewDialog,
        syncPreviewPanelSubtitle: workspace.syncPreviewPanelSubtitle,
    });
    const weights = createPreviewWeights({
        ctx,
        state,
        deps,
        syncPreviewPanelSubtitle: workspace.syncPreviewPanelSubtitle,
    });

    async function loadPreviewSettings() {
        if (location.protocol === 'file:') return;
        try {
            state.settings = await fetchPreviewSettings(ctx, state);
            document.getElementById('preview-training-dir').value = state.settings.training_dir || '';
            document.getElementById('preview-inference-dir').value = state.settings.inference_dir || '';
            document.getElementById('preview-custom-dir').value = state.settings.custom_dir || '';
            updatePreviewDirectorySummary();
            renderPreviewTaskSelect();
        } catch (e) {
            setPreviewStatus('读取路径设置失败: ' + e.message, 'error');
        }
    }

    async function savePreviewSettings() {
        try {
            const res = await savePreviewSettingsRequest(ctx, {
                training_dir: ctx.dom.val('preview-training-dir'),
                inference_dir: ctx.dom.val('preview-inference-dir'),
                custom_dir: ctx.dom.val('preview-custom-dir'),
            });
            if (!res.ok) {
                setPreviewStatus(res.error || '保存失败', 'error');
                return;
            }
            setPreviewStatus(res.message || '路径设置已保存', 'ok');
            await loadPreviewSettings();
            await loadPreviewImages();
        } catch (e) {
            setPreviewStatus('保存失败: ' + e.message, 'error');
        }
    }

    async function resetPreviewSettings() {
        if (!state.settings?.defaults) return;
        document.getElementById('preview-training-dir').value = state.settings.defaults.training_dir || 'output/ckpt/sample';
        document.getElementById('preview-inference-dir').value = state.settings.defaults.inference_dir || 'output/tests';
        document.getElementById('preview-custom-dir').value = state.settings.defaults.custom_dir || '';
        await savePreviewSettings();
    }

    async function loadPreviewImages() {
        if (location.protocol === 'file:') {
            images.setPreviewEmpty('静态打开没有后端 API，无法读取项目预览图。');
            return;
        }
        const requestSeq = ++state.requestSeq;
        images.setPreviewLoading();
        try {
            if (!deps.getHistoryTasks().length) {
                await deps.loadTrainingHistoryList();
            }
            if (!state.settings) {
                await loadPreviewSettings();
            }
            const payload = await fetchPreviewImages(ctx, state, deps);
            if (requestSeq !== state.requestSeq) return;
            if (!payload.ok) {
                images.setPreviewEmpty(payload.error || '读取预览图失败');
                return;
            }
            images.renderPreviewImages(payload);
            state.trainingSampleState = payload.sample_config || state.trainingSampleState;
            deps.setTrainingSampleState(state.trainingSampleState);
            loadPreviewWeights();
        } catch (e) {
            if (requestSeq !== state.requestSeq) return;
            images.setPreviewEmpty('读取预览图失败: ' + e.message);
        }
    }

    async function loadPreviewWeights() {
        const requestSeq = ++state.weightRequestSeq;
        if (location.protocol === 'file:') {
            if (requestSeq === state.weightRequestSeq) {
                weights.renderPreviewWeights({ ok: true, weights: [], message: '静态打开没有后端 API。' });
            }
            return;
        }
        if (state.source !== 'training') {
            if (requestSeq === state.weightRequestSeq) {
                weights.renderPreviewWeights({
                    ok: true,
                    weights: [],
                    message: '权重文件只随训练来源显示。',
                });
            }
            return;
        }
        try {
            const payload = await fetchPreviewWeights(ctx, state, deps);
            if (requestSeq !== state.weightRequestSeq) return;
            weights.renderPreviewWeights(payload);
        } catch (e) {
            if (requestSeq !== state.weightRequestSeq) return;
            weights.renderPreviewWeights({ ok: false, weights: [], error: '读取权重文件失败: ' + e.message });
        }
    }

    function setPreviewSource(source) {
        state.source = source || 'training';
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.previewSource === state.source);
        });
        updatePreviewTaskVisibility();
        updatePreviewDirectorySummary();
        state.weightRequestSeq += 1;
        loadPreviewImages();
    }

    async function openTrainingPreview(options = {}) {
        state.source = 'training';
        state.selectedTaskId = '';
        state.selectedGroup = null;
        if (options.group) {
            state.selectedGroup = normalizePreviewGroup(options.group);
        } else if (options.taskId) {
            state.selectedTaskId = String(options.taskId || '');
        }
        state.settings = null;
        state.weightRequestSeq += 1;
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.previewSource === 'training');
        });
        if (options.group) {
            await openHistoryConfigGroupPreview(options.group);
        } else if (options.taskId) {
            await deps.loadHistoryTask(String(options.taskId || ''), { detailTab: 'preview' });
        } else {
            workspace.openPreviewPanel();
            updatePreviewTaskVisibility();
            renderPreviewTaskSelect();
            workspace.syncPreviewPanelSubtitle();
            await loadPreviewSettings();
            await loadPreviewImages();
        }
    }

    function openCurrentTrainingPreview(event) {
        event?.preventDefault?.();
        event?.stopPropagation?.();
        openTrainingPreview();
    }

    async function openLiveSamplingPreview(event) {
        event?.preventDefault?.();
        event?.stopPropagation?.();
        const historyTaskId = deps.getTrainingViewMode() === 'live'
            ? String(deps.getViewingHistoryTaskId?.() || '')
            : '';
        state.source = 'training';
        state.selectedTaskId = historyTaskId;
        state.selectedGroup = null;
        state.settings = null;
        state.weightRequestSeq += 1;
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.previewSource === 'training');
        });
        workspace.openPreviewPanel({ mode: 'sampling' });
        updatePreviewTaskVisibility();
        renderPreviewTaskSelect();
        workspace.syncPreviewPanelSubtitle();
        await loadPreviewSettings();
        await loadPreviewImages();
    }

    async function openHistoryConfigGroupPreview(group) {
        await deps.loadConfigGroupTimeline(group, { skipSelectionDialog: true, detailTab: 'preview' });
    }

    function renderPreviewTaskSelect() {
        const select = document.getElementById('preview-training-task');
        if (!select) return;
        const previousValue = selectedPreviewSelectValue(state);
        select.innerHTML = '';
        const liveOption = document.createElement('option');
        liveOption.value = '';
        select.appendChild(liveOption);

        const trainingTasks = deps.getHistoryTasks()
            .filter((task) => task.job === 'training' && (deps.getShowArchivedHistory() || !deps.historyTaskIsArchived(task)))
            .sort((a, b) => Number(b.started_at || 0) - Number(a.started_at || 0));
        liveOption.textContent = trainingTasks.length
            ? `当前任务或最新运行目录 · ${trainingTasks.length} 个历史训练`
            : '当前任务或最新运行目录 · 暂无历史训练';

        const groups = previewTrainingGroups(trainingTasks);
        if (groups.length) {
            const groupOptions = document.createElement('optgroup');
            groupOptions.label = '训练分组合并';
            for (const group of groups) {
                const option = document.createElement('option');
                option.value = encodePreviewGroupValue(group);
                option.textContent = `${group.label} · ${group.tasks.length} 次训练`;
                groupOptions.appendChild(option);
            }
            select.appendChild(groupOptions);
        }

        if (trainingTasks.length) {
            const taskOptions = document.createElement('optgroup');
            taskOptions.label = '单个训练任务';
            for (const task of trainingTasks) {
                const option = document.createElement('option');
                option.value = encodePreviewTaskValue(task.id);
                option.textContent = [
                    task.name || `${task.methods_subdir || '-'} / ${task.variant || '-'}`,
                    task.started_at_text || task.id,
                    deps.historyStateLabel(task.state),
                ].filter(Boolean).join(' · ');
                taskOptions.appendChild(option);
            }
            select.appendChild(taskOptions);
        }

        const values = Array.from(select.options).map((option) => option.value);
        const nextValue = values.includes(previousValue) ? previousValue : '';
        applyPreviewSelectionValue(state, nextValue);
        select.value = nextValue;
        select.disabled = false;
        updatePreviewTaskVisibility();
    }

    function previewTrainingGroups(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const group = deps.historyConfigGroupFromTask(task);
            if (!map.has(group.key)) {
                map.set(group.key, { ...group, tasks: [] });
            }
            map.get(group.key).tasks.push(task);
        }
        return Array.from(map.values())
            .filter((group) => group.tasks.length > 0)
            .sort((a, b) => {
                const aTime = Math.max(...a.tasks.map((task) => Number(task.started_at || 0)));
                const bTime = Math.max(...b.tasks.map((task) => Number(task.started_at || 0)));
                return (bTime - aTime) || a.label.localeCompare(b.label, 'zh-CN');
            });
    }

    function updatePreviewTaskVisibility() {
        const field = document.getElementById('preview-training-task-field');
        if (field) field.hidden = state.source !== 'training';
        workspace.syncPreviewPanelSubtitle();
    }

    async function changePreviewTask(taskId) {
        applyPreviewSelectionValue(state, taskId || '');
        state.settings = null;
        state.weightRequestSeq += 1;
        await loadPreviewSettings();
        await loadPreviewImages();
    }

    function activateHistoryDetailPreview(payload) {
        return activateHistoryDetailPreviewTarget(historyDetailPreviewTarget(payload));
    }

    async function activateHistoryDetailPreviewTarget(target) {
        if (!target) return;
        state.source = 'training';
        state.selectedTaskId = target.taskId || '';
        state.selectedGroup = target.group || null;
        state.settings = null;
        state.weightRequestSeq += 1;
        workspace.mountPreviewWorkspaceInHistoryDetail();
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.previewSource === 'training');
        });
        updatePreviewTaskVisibility();
        renderPreviewTaskSelect();
        workspace.syncPreviewPanelSubtitle();
        await loadPreviewSettings();
        await loadPreviewImages();
    }

    function historyDetailPreviewTarget(payload) {
        const task = payload?.task || null;
        if (task) {
            return task.job === 'training' && task.id
                ? { taskId: String(task.id) }
                : null;
        }
        const group = payload?.group || null;
        if (payload?.mode === 'config_group' && deps.canPreviewHistoryConfigGroup(group)) {
            return { group: normalizePreviewGroup(group) };
        }
        return null;
    }

    function updateRuntimeSampleState({ sampleDir, sampleConfig }) {
        if (sampleDir !== undefined && state.settings) {
            state.settings.current_task_sample_dir = sampleDir || '';
            state.settings.effective_training_dir = state.settings.current_task_sample_dir || state.settings.training_dir;
            updatePreviewDirectorySummary();
        }
        if (sampleConfig !== undefined) {
            state.trainingSampleState = sampleConfig || null;
            deps.setTrainingSampleState(state.trainingSampleState);
        }
    }

    function updatePreviewDirectorySummary() {
        const el = document.getElementById('preview-current-dir');
        if (!el || !state.settings) return;
        if (state.source === 'training') {
            el.textContent = state.settings.effective_training_dir || state.settings.training_dir || '-';
        } else if (state.source === 'inference') {
            el.textContent = state.settings.inference_dir || '-';
        } else {
            el.textContent = state.settings.custom_dir || '-';
        }
        workspace.syncPreviewPanelSubtitle();
    }

    function setPreviewStatus(text, status = '') {
        const el = document.getElementById('preview-settings-status');
        if (!el) return;
        el.textContent = text;
        el.className = `preview-status ${status}`.trim();
    }

    return {
        loadPreviewSettings,
        savePreviewSettings,
        resetPreviewSettings,
        loadPreviewImages,
        loadPreviewWeights,
        setPreviewSource,
        changePreviewTask,
        openTrainingPreview,
        openCurrentTrainingPreview,
        openLiveSamplingPreview,
        openHistoryConfigGroupPreview,
        openPreviewPanel: workspace.openPreviewPanel,
        closePreviewPanel: workspace.closePreviewPanel,
        mountPreviewWorkspaceInHistoryDetail: workspace.mountPreviewWorkspaceInHistoryDetail,
        restorePreviewWorkspaceFromHistoryDetail: workspace.restorePreviewWorkspaceFromHistoryDetail,
        activateHistoryDetailPreview,
        updateRuntimeSampleState,
        renderPreviewTaskSelect,
        setPreviewStatus,
        normalizePreviewGroup,
        openPreviewDialog: dialog.openPreviewDialog,
        closePreviewImageDialog: dialog.closePreviewImageDialog,
        createPreviewDetailRow,
        createPreviewDetailBlock,
        restorePreviewWorkspaceAfterPanelClose: workspace.restorePreviewWorkspaceAfterPanelClose,
        togglePreviewWeightSort: () => weights.togglePreviewWeightSort(loadPreviewWeights),
        previewSourceLabel,
    };
}
