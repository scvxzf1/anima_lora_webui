/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
export function createStatusPollingBridge(target = globalThis) {
    // Keep polling bookkeeping inside the bridge while old callers still use global function names.
    let trainingSidebarSummaryLastRefreshAt = 0;
    let trainingSidebarSummaryLastTaskId = '';
    let trainingSidebarSummaryLastStatus = '';
    let trainingSidebarSummaryRefreshPromise = null;
    let trainingStatusPollFailures = Number(target.trainingStatusPollFailures || 0);
    let trainingStatusPollTimer = target.trainingStatusPollTimer || null;
    let trainingStatusPollPromise = target.trainingStatusPollPromise || null;
    let trainingStatusPollForceReplayMetrics = Boolean(target.trainingStatusPollForceReplayMetrics);

    // ── 状态轮询 ──
    function trainingStatusPollDelayMs() {
        const visible = !document.hidden;
        const wsOpen = ws?.readyState === WebSocket.OPEN;
        const running = isLiveRunningState();
        if (!visible) return wsOpen ? (running ? 30000 : 120000) : 60000;
        if (!wsOpen) return running ? 5000 : 15000;
        return running ? 10000 : 60000;
    }

    function scheduleStatusPoll(options = {}) {
        if (location.protocol === 'file:') return;
        if (trainingStatusPollTimer) {
            window.clearTimeout(trainingStatusPollTimer);
            trainingStatusPollTimer = null;
        }
        const delay = options.immediate ? 0 : trainingStatusPollDelayMs();
        trainingStatusPollTimer = window.setTimeout(() => {
            trainingStatusPollTimer = null;
            void pollStatus({ forceReplayMetrics: options.forceReplayMetrics === true });
        }, delay);
    }

    async function pollStatus(options = {}) {
        if (options.forceReplayMetrics) {
            trainingStatusPollForceReplayMetrics = true;
        }
        if (trainingStatusPollPromise) return trainingStatusPollPromise;
        trainingStatusPollPromise = (async () => {
            if (isHistoryReviewMode()) return;
            try {
                const status = await api('/api/training/status');
                if (status.ok === false) throw new Error(status.error || '读取训练状态失败');
                const forceReplayMetrics = trainingStatusPollForceReplayMetrics;
                trainingStatusPollForceReplayMetrics = false;
                if (trainingStatusPollFailures) {
                    trainingStatusPollFailures = 0;
                    updateLogStatusText();
                }
                updateStatus({
                    state: status.status,
                    variant: status.variant,
                    preset: status.preset,
                    methods_subdir: status.methods_subdir,
                    job: status.job,
                    last_output_at: status.last_output_at,
                    last_log_id: status.last_log_id,
                    last_log_line: status.last_log_line,
                    error_hint: status.error_hint,
                    anomaly_message: status.anomaly_message || '',
                    output_dir: status.output_dir,
                    sample_dir: status.sample_dir,
                    sample_config: status.sample_config,
                    run_dir: status.run_dir,
                    runtime_config_file: status.runtime_config_file,
                    original_config_file: status.original_config_file,
                    dataset_config_file: status.dataset_config_file,
                    model_cache_dir: status.model_cache_dir,
                    dataset_cache_dir: status.dataset_cache_dir,
                    training_output_dir: status.training_output_dir,
                    logs_dir: status.logs_dir,
                });
                applyStatusSnapshotFallbacks(status);
                refreshTrainingSidebarSummariesFromPoll(status);
                if ((status.last_log_id || 0) > trainingRuntime.lastLogId) {
                    await replayTrainingLogs();
                } else if (forceReplayMetrics || isLiveRunningState()) {
                    await replayMetricsHistory();
                }
            } catch (e) {
                trainingStatusPollFailures += 1;
                if (trainingStatusPollFailures < 3) return;
                const message = `训练状态轮询连续失败 ${trainingStatusPollFailures} 次: ${e.message}`;
                setLogStatus('状态轮询失败', 'error');
                setTrainingHealthNotice(message, 'error');
                if (trainingStatusPollFailures === 3) appendLog(`[状态] ${message}`);
            }
        })();
        try {
            return await trainingStatusPollPromise;
        } finally {
            trainingStatusPollPromise = null;
            scheduleStatusPoll();
        }
    }

    function refreshTrainingSidebarSummariesFromPoll(status = {}) {
        if (location.protocol === 'file:') return null;
        const taskId = String(status.task_id || '').trim();
        const state = String(status.status || '').trim();
        const live = isLiveRunningState(state);
        const knownTask = taskId && Array.isArray(historyTasks)
            && historyTasks.some((task) => String(task.id || '') === taskId);
        const taskChanged = taskId && taskId !== trainingSidebarSummaryLastTaskId;
        const statusChanged = taskId && state && state !== trainingSidebarSummaryLastStatus;
        const now = Date.now();
        const stale = now - trainingSidebarSummaryLastRefreshAt >= 15000;
        const shouldRefresh = taskChanged || statusChanged || (taskId && !knownTask) || (live && stale);
        if (!shouldRefresh) return trainingSidebarSummaryRefreshPromise;
        if (trainingSidebarSummaryRefreshPromise) return trainingSidebarSummaryRefreshPromise;
        trainingSidebarSummaryLastTaskId = taskId || trainingSidebarSummaryLastTaskId;
        trainingSidebarSummaryLastStatus = state || trainingSidebarSummaryLastStatus;
        trainingSidebarSummaryLastRefreshAt = now;
        trainingSidebarSummaryRefreshPromise = Promise.all([
            loadTrainingQueue(),
            loadTrainingHistoryList(),
        ]).catch((e) => {
            appendLog(`[状态] 刷新训练侧栏失败: ${e.message}`);
        }).finally(() => {
            trainingSidebarSummaryRefreshPromise = null;
        });
        return trainingSidebarSummaryRefreshPromise;
    }

    function applyStatusSnapshotFallbacks(status = {}) {
        if (!isLiveRunningState(status.status)) return;
        if (hasStatusPayload(status.latest_progress)) {
            updateProgress(status.latest_progress, { replay: true });
        }
        if (hasStatusPayload(status.latest_metric)) {
            updateMetrics(status.latest_metric, { replay: true });
        }
        if (hasStatusPayload(status.latest_system)) {
            updateSystem(status.latest_system, { replay: true });
        }
    }

    function hasStatusPayload(value) {
        return value && typeof value === 'object' && Object.keys(value).length > 0;
    }

    return {
        trainingStatusPollDelayMs,
        scheduleStatusPoll,
        pollStatus,
        refreshTrainingSidebarSummariesFromPoll,
        applyStatusSnapshotFallbacks,
        hasStatusPayload,
    };
}
