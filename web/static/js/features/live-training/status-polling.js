/**
 * Live training status polling bridge.
 * Moved out of anima-app mechanical chunks.
 */
import { isLiveRunningState } from './index.js?v=module-bootstrap-20260711-ir1';
import {
    updateMetrics,
    updateProgress,
    updateStatus,
    updateSystem,
} from '../anima-app/helpers/live-status-bridge.js?v=module-bootstrap-20260711-ir1';
import { isHistoryReviewMode } from '../anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    appendLog,
    replayMetricsHistory,
    replayTrainingLogs,
    setLogStatus,
    setTrainingHealthNotice,
    updateLogStatusText,
} from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir1';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadTrainingQueue } from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadTrainingHistoryList } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260711-ir1';
import { api } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';

export function createStatusPollingBridge(target = globalThis) {
    // Keep polling bookkeeping inside the bridge while old callers still use global function names.
    let trainingSidebarSummaryLastRefreshAt = 0;
    let trainingSidebarSummaryLastTaskId = '';
    let trainingSidebarSummaryLastStatus = '';
    let trainingSidebarSummaryRefreshPromise = null;

    function readHistoryTasks() {
        try {
            return getHistoryState().historyTasks;
        } catch {
            return target.historyTasks;
        }
    }

    // ── 状态轮询 ──
    function trainingStatusPollDelayMs() {
        const visible = !document.hidden;
        const wsOpen = target.ws?.readyState === WebSocket.OPEN;
        const running = isLiveRunningState(target.trainingRuntime?.state);
        // Keep a short fallback cadence while running so resource cards still
        // refresh from latest_system if a WS system frame is delayed/missed.
        if (!visible) return wsOpen ? (running ? 15000 : 120000) : 60000;
        if (!wsOpen) return running ? 2000 : 15000;
        return running ? 4000 : 60000;
    }

    function scheduleStatusPoll(options = {}) {
        if (location.protocol === 'file:') return;
        if (target.trainingStatusPollTimer) {
            window.clearTimeout(target.trainingStatusPollTimer);
            target.trainingStatusPollTimer = null;
        }
        const delay = options.immediate ? 0 : trainingStatusPollDelayMs();
        target.trainingStatusPollTimer = window.setTimeout(() => {
            target.trainingStatusPollTimer = null;
            void pollStatus({ forceReplayMetrics: options.forceReplayMetrics === true });
        }, delay);
    }

    async function pollStatus(options = {}) {
        if (options.forceReplayMetrics) {
            target.trainingStatusPollForceReplayMetrics = true;
        }
        if (target.trainingStatusPollPromise) return target.trainingStatusPollPromise;
        target.trainingStatusPollPromise = (async () => {
            if (isHistoryReviewMode()) return;
            try {
                const status = await api('/api/training/status');
                if (status.ok === false) throw new Error(status.error || '读取训练状态失败');
                const forceReplayMetrics = target.trainingStatusPollForceReplayMetrics;
                target.trainingStatusPollForceReplayMetrics = false;
                if (target.trainingStatusPollFailures) {
                    target.trainingStatusPollFailures = 0;
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
                const shouldReplayRecoveredArtifacts = shouldReplayRecoveredLiveArtifacts(status);
                if (shouldReplayRecoveredArtifacts && (status.last_log_id || 0) > (target.trainingRuntime?.lastLogId || 0)) {
                    await replayTrainingLogs();
                } else if (shouldReplayRecoveredArtifacts && (forceReplayMetrics || isLiveRunningState(target.trainingRuntime?.state))) {
                    await replayMetricsHistory();
                }
                return status;
            } catch (e) {
                target.trainingStatusPollFailures += 1;
                if (target.trainingStatusPollFailures < 3) return;
                const message = `训练状态轮询连续失败 ${target.trainingStatusPollFailures} 次: ${e.message}`;
                setLogStatus('状态轮询失败', 'error');
                setTrainingHealthNotice(message, 'error');
                if (target.trainingStatusPollFailures === 3) appendLog(`[状态] ${message}`);
            }
        })();
        try {
            return await target.trainingStatusPollPromise;
        } finally {
            target.trainingStatusPollPromise = null;
            scheduleStatusPoll();
        }
    }

    function refreshTrainingSidebarSummariesFromPoll(status = {}) {
        if (location.protocol === 'file:') return null;
        const taskId = String(status.task_id || '').trim();
        const state = String(status.status || '').trim();
        const live = isLiveRunningState(state);
        const historyTasks = readHistoryTasks();
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

    function shouldReplayRecoveredLiveArtifacts(status = {}) {
        const state = String(status.status || '').trim();
        return isLiveRunningState(state) || state === 'error';
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
        shouldReplayRecoveredLiveArtifacts,
        applyStatusSnapshotFallbacks,
        hasStatusPayload,
    };
}
