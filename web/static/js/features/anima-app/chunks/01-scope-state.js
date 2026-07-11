/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    configurePreviewFeatureEnsurer,
    configureQueueFeatureEnsurer,
} from '../helpers/feature-ensurers.js?v=module-bootstrap-20260711-ir6';
import { getAppShellState } from '../helpers/app-shell-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getAppContext } from '../helpers/app-context-bridge.js?v=module-bootstrap-20260711-ir6';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    configTrainingSourceMode,
    continueTrainingRequestPayload,
    ensureTrainingSourceReadyForLaunch,
    refreshContinueTrainingSourceCompatibility,
    selectContinueLoraWeight,
    startConfigFullResumeSource,
    trainingSourceLaunchBlockReason,
} from '../helpers/training-source-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    currentTrainingConfigIsRuntime,
    isCliOnlySpdSource,
    showPreflightDialog,
} from '../helpers/training-launch-bridge.js?v=module-bootstrap-20260711-ir6';
import { canPreviewHistoryConfigGroup, historyConfigGroupFromTask, historyTaskIsArchived, runLabelFromPath } from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260711-ir6';
import { historyTaskLabel, loadConfigGroupTimeline, loadHistoryTask, showHistoryTaskConfirmDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260711-ir6';
import { configGroupLabel, historyStateLabel } from '../helpers/history-timeline-bridge.js?v=module-bootstrap-20260711-ir6';
import { getGpuPicker } from '../helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260711-ir6';
import { ensureHistoryDetailFeature } from '../helpers/history-detail-bridge.js?v=module-bootstrap-20260711-ir6';
import { hasPendingConfigChanges, showAppConfirmDialog } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir6';
import { currentTrainingConfigFile, renderPreflightPending, showPreflightRequestError } from '../helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260711-ir6';
import { appendLog } from '../helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir6';
import { renderTrainingViewMode, showTrainingView } from '../helpers/queue-view-bridge.js?v=module-bootstrap-20260711-ir6';
import { loadTrainingHistoryList } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260711-ir6';
import { renderDatasetImageDialogDetails } from '../helpers/preview-view-bridge.js?v=module-bootstrap-20260711-ir6';
import { setTomlStatus, updateTomlActionState } from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir6';

const ctx = getAppContext();
const appShellState = getAppShellState();
const datasetState = getDatasetState();
const historyState = getHistoryState();
const tomlState = getTomlState();
const trainingState = getTrainingState();
    // ── 状态 ──
    configureQueueFeatureEnsurer(ctx, appShellState, {
            appendLog,
            showAppConfirmDialog,
            setTomlStatus,
            currentTrainingConfigFile,
            getTomlManagerMode: () => tomlState.tomlManagerMode,
            getOutputRunFile: () => datasetState.outputRunState.file,
            getOutputRunSelectedRun: () => datasetState.outputRunState.selectedRun,
            getCurrentTomlFile: () => tomlState.currentTomlFile,
            hasPendingConfigChanges,
            updateTomlActionState,
            getCurrentTrainingSource: () => trainingState.currentTrainingSource,
            isCliOnlySpdSource,
            hasContinueTrainingSource: () => Boolean(trainingState.continueTrainingSource),
            continueTrainingSourceMessage: () => trainingState.continueTrainingSource?.message || '',
            refreshContinueTrainingSourceCompatibility,
            getTrainingSourceMode: () => configTrainingSourceMode(),
            ensureTrainingSourceReadyForLaunch,
            trainingSourceLaunchBlockReason,
            queueConfigFullResumeSource: () => startConfigFullResumeSource(true),
            currentTrainingConfigIsRuntime,
            renderPreflightPending,
            continueTrainingRequestPayload: () => continueTrainingRequestPayload(),
            showPreflightDialog,
            showPreflightRequestError,
            selectedGpuPayload: () => getGpuPicker()?.selectedGpuPayload?.() ?? [],
            showTrainingView,
            getTrainingRuntime: () => trainingState.trainingRuntime,
            renderTrainingViewMode,
            runLabelFromPath,
            getViewingHistoryTaskId: () => historyState.viewingHistoryTaskId,
            selectedResumeCheckpoint: () => ensureHistoryDetailFeature().selectedResumeCheckpoint(),
            setResumeStatus: (text, state = '') => ensureHistoryDetailFeature().setResumeStatus(text, state),
            historyTaskLabel,
            getCurrentHistoryTaskForResume: () => historyState.currentHistoryTaskForResume,
            showHistoryTaskConfirmDialog,
    });
    configurePreviewFeatureEnsurer(ctx, appShellState, {
            getHistoryTasks: () => historyState.historyTasks,
            getShowArchivedHistory: () => historyState.showArchivedHistory,
            loadTrainingHistoryList,
            loadHistoryTask,
            loadConfigGroupTimeline,
            showTrainingView,
            getTrainingViewMode: () => trainingState.trainingViewMode,
            getViewingHistoryTaskId: () => historyState.viewingHistoryTaskId,
            getTrainingRuntime: () => trainingState.trainingRuntime,
            setTrainingSampleState: (value) => {
                if (value) trainingState.trainingRuntime.sampleConfig = value;
            },
            historyTaskIsArchived,
            historyStateLabel,
            historyConfigGroupFromTask,
            canPreviewHistoryConfigGroup,
            configGroupLabel,
            selectContinueLoraWeight,
            renderDatasetImageDialogDetails,
    });
