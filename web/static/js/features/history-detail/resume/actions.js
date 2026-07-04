import {
    fetchHistoryResumeWeights,
    fetchResumeOptions,
    inspectContinueLoraWeight,
    postResumeTraining,
} from '../api.js?v=module-bootstrap-20260704-1';
import {
    clearResumeState,
    selectedHistoryManagerResumeCheckpointFromState,
    selectedResumeCheckpointFromState,
    setResumeLoadingForTask as setResumeLoadingForTaskState,
} from './state.js?v=module-bootstrap-20260704-1';

export function createHistoryResumeActions({ ctx, state, deps, slots, renderResumePanelState }) {
    function setResumeLoadingForTask(taskId) {
        setResumeLoadingForTaskState(state, taskId);
        renderResumePanelState();
    }

    function setResumeStatus(text, status = '') {
        const el = document.getElementById('resume-training-status');
        if (!el) return;
        el.textContent = text || '';
        el.className = ['resume-status', status].filter(Boolean).join(' ');
    }

    async function loadResumeOptionsForTask(taskId = deps.getViewingHistoryTaskId()) {
        if (!taskId) {
            clearResumeOptions();
            return;
        }
        setResumeLoadingForTaskState(state, taskId);
        renderResumePanelState();
        try {
            const payload = await fetchResumeOptions(ctx, taskId);
            if (taskId !== deps.getViewingHistoryTaskId()) return;
            if (!payload.ok) {
                state.resumeOptions = {
                    loading: false,
                    taskId,
                    checkpoints: [],
                    defaultCheckpoint: '',
                    error: payload.error || '读取续训检查点失败',
                    message: '',
                    diagnostic: payload.diagnostic || {},
                };
                await loadHistoryResumeWeights(taskId);
                renderResumePanelState();
                return;
            }
            state.resumeOptions = {
                loading: false,
                taskId,
                checkpoints: payload.checkpoints || [],
                defaultCheckpoint: payload.default_checkpoint || '',
                error: '',
                message: payload.message || '',
                diagnostic: payload.diagnostic || {},
            };
            await loadHistoryResumeWeights(taskId);
            renderResumePanelState();
        } catch (e) {
            if (taskId !== deps.getViewingHistoryTaskId()) return;
            state.resumeOptions = {
                loading: false,
                taskId,
                checkpoints: [],
                defaultCheckpoint: '',
                error: '读取续训检查点失败: ' + e.message,
                message: '',
                diagnostic: {},
            };
            state.resumeWeights = {
                loading: false,
                taskId,
                weights: [],
                error: '',
                message: '',
            };
            renderResumePanelState();
        }
    }

    async function loadHistoryResumeWeights(taskId = deps.getViewingHistoryTaskId()) {
        if (!taskId) {
            state.resumeWeights = {
                loading: false,
                taskId: '',
                weights: [],
                error: '',
                message: '',
            };
            return;
        }
        state.resumeWeights = {
            loading: true,
            taskId,
            weights: [],
            error: '',
            message: '正在读取可热启动权重...',
        };
        renderResumePanelState();
        try {
            const payload = await fetchHistoryResumeWeights(ctx, taskId);
            if (taskId !== deps.getViewingHistoryTaskId()) return;
            const rawWeights = Array.isArray(payload.weights) ? payload.weights : [];
            state.resumeWeights = {
                loading: true,
                taskId,
                weights: rawWeights,
                error: payload.ok ? '' : (payload.error || '读取历史权重失败'),
                message: rawWeights.length ? '正在审查可热启动权重...' : (payload.message || ''),
            };
            renderResumePanelState();
            const weights = await reviewHistoryResumeWeights(rawWeights);
            if (taskId !== deps.getViewingHistoryTaskId()) return;
            state.resumeWeights = {
                loading: false,
                taskId,
                weights,
                error: payload.ok ? '' : (payload.error || '读取历史权重失败'),
                message: payload.message || '',
            };
        } catch (e) {
            if (taskId !== deps.getViewingHistoryTaskId()) return;
            state.resumeWeights = {
                loading: false,
                taskId,
                weights: [],
                error: '读取历史权重失败: ' + e.message,
                message: '',
            };
        }
    }

    async function reviewHistoryResumeWeights(weights) {
        return Promise.all((weights || []).map(async (item) => {
            const weightPath = item.abs_path || item.file || '';
            if (!weightPath) {
                return {
                    ...item,
                    inspect_status: 'error',
                    inspect_compatible: false,
                    inspect_message: '权重路径为空',
                };
            }
            try {
                const payload = await inspectHistoryResumeWeight(weightPath);
                const compatible = Boolean(payload.ok && payload.compatible);
                return {
                    ...item,
                    inspect_status: compatible ? 'ok' : 'error',
                    inspect_compatible: compatible,
                    inspect_kind: payload.kind || '',
                    inspect_message: payload.message || payload.error || (compatible ? '审查通过' : '权重审查未通过'),
                };
            } catch (e) {
                return {
                    ...item,
                    inspect_status: 'error',
                    inspect_compatible: false,
                    inspect_message: '权重审查失败: ' + e.message,
                };
            }
        }));
    }

    async function inspectHistoryResumeWeight(path) {
        if (typeof deps.inspectContinueLoraWeight === 'function') {
            return deps.inspectContinueLoraWeight(path);
        }
        const task = deps.getCurrentHistoryTaskForResume?.() || {};
        return inspectContinueLoraWeight(ctx, {
            path,
            variant: task.variant || '',
            preset: task.preset || 'default',
            methodsSubdir: task.methods_subdir || 'gui-methods',
            configFile: task.history_source_config_file || task.runtime_config_file || '',
        });
    }

    function clearResumeOptions() {
        clearResumeState(state);
        deps.setCurrentHistoryTaskForResume(null);
        renderResumePanelState();
    }

    async function resumeTrainingFromCheckpoint() {
        if (!deps.getViewingHistoryTaskId()) return;
        const selected = selectedResumeCheckpoint();
        if (!selected) {
            setResumeStatus('请先选择一个可续训状态目录。', 'error');
            return;
        }
        const taskName = deps.historyTaskLabel(deps.getCurrentHistoryTaskForResume() || {});
        const ok = await deps.showHistoryTaskConfirmDialog({
            title: '从检查点继续训练',
            description: taskName,
            message: `将使用这个历史任务的配置快照，并从 ${selected.name} 继续训练。训练会恢复优化器、学习率调度器和已完成步数；启动后会生成一个新的训练任务记录。`,
            confirmText: '确认开始续训',
        });
        if (!ok) return;

        setResumeStatus('正在启动续训...', '');
        try {
            const res = await postResumeTraining(ctx, {
                taskId: deps.getViewingHistoryTaskId(),
                checkpoint: selected.path,
                queueMode: false,
                gpuWhitelist: deps.selectedGpuPayload(),
            });
            if (!res.ok) {
                setResumeStatus(res.error || '续训启动失败', 'error');
                return;
            }
            const message = res.message || '续训已启动';
            setResumeStatus(message, 'ok');
            await deps.loadTrainingHistoryList();
            deps.returnToLiveTraining();
            deps.appendLog(`[状态] ${message}: ${selected.name}`);
        } catch (e) {
            setResumeStatus('续训启动失败: ' + e.message, 'error');
        }
    }

    async function resumeTrainingFromHistoryDetail(queueMode) {
        if (!deps.getViewingHistoryTaskId()) return;
        const selected = selectedHistoryManagerResumeCheckpoint();
        if (!selected) return;
        const taskName = deps.historyTaskLabel(deps.getCurrentHistoryTaskForResume() || {});
        const ok = await deps.showHistoryTaskConfirmDialog({
            title: queueMode ? '续训加入队列' : '从检查点继续训练',
            description: taskName,
            message: queueMode
                ? `将使用这个历史任务的配置快照，并从 ${selected.name} 续训。任务会排队等待当前训练结束后自动启动。`
                : `将使用这个历史任务的配置快照，并从 ${selected.name} 继续训练。训练会恢复优化器、学习率调度器和已完成步数。`,
            confirmText: queueMode ? '加入队列' : '确认开始续训',
        });
        if (!ok) return;
        try {
            const res = await postResumeTraining(ctx, {
                taskId: deps.getViewingHistoryTaskId(),
                checkpoint: selected.path,
                queueMode,
                gpuWhitelist: deps.selectedGpuPayload(),
            });
            if (!res.ok) {
                alert(res.error || '续训启动失败');
                return;
            }
            if (queueMode) {
                deps.updateTrainingQueueFromPayload(res);
                slots.closeHistoryDetailDialog();
                deps.showTrainingView('queue');
            } else {
                await deps.loadTrainingHistoryList();
                deps.returnToLiveTraining();
            }
        } catch (e) {
            alert('续训启动失败: ' + e.message);
        }
    }

    function selectedResumeCheckpoint() {
        return selectedResumeCheckpointFromState(state);
    }

    function selectedHistoryManagerResumeCheckpoint() {
        return selectedHistoryManagerResumeCheckpointFromState(state);
    }

    return {
        clearResumeOptions,
        loadHistoryResumeWeights,
        loadResumeOptionsForTask,
        resumeTrainingFromCheckpoint,
        resumeTrainingFromHistoryDetail,
        selectedHistoryManagerResumeCheckpoint,
        selectedResumeCheckpoint,
        setResumeStatus,
        setResumeLoadingForTask,
    };
}
