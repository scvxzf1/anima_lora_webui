/**
 * DOM event bindings for the main app shell.
 */
import {
    GLOBAL_UI_OVERRIDE_FIELDS,
    help,
} from '../../config/catalog.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    ensureEnvironmentCheckFeature,
    ensureQueueFeature,
    ensureWeightAnalysisFeature,
} from '../anima-app/helpers/feature-ensurers.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    closeOutputRunSaveAs,
    confirmOutputRunSaveAs,
    copyOutputRunConfigContent,
    exportOutputRunConfig,
    loadTomlFile,
    openOutputRunSaveAs,
    renderOutputRunList,
    saveTomlFile,
    selectAndApplyTomlFile,
} from '../anima-app/helpers/output-run-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadOutputRuns, setTomlManagerMode, switchTomlManagerMode } from '../anima-app/helpers/toml-manager-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    deleteTomlFile,
    moveCurrentTomlToGroup,
    restoreSystemTomlPresets,
} from '../anima-app/helpers/toml-actions-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    queueCurrentTrainingFromConfig,
    startTraining,
} from '../anima-app/helpers/training-launch-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    copyDatasetPreset,
    createDatasetPresetGroup,
    createNewDatasetPreset,
    deleteDatasetPreset,
    exportDatasetPreset,
    handleDatasetPresetImport,
    importDatasetPreset,
    renameDatasetPreset,
    saveDatasetPresetEditor,
} from '../anima-app/helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { renderDatasetPresetList } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { historyManagerFilterDefault, openHistoryCollectionsWorkbench } from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    archiveSelectedHistoryTasks,
    deleteSelectedHistoryTasks,
    groupSelectedHistoryTasks,
    mergeSelectedHistoryTasks,
    refreshHistoryView,
} from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadResumeOptionsForTask, queueResumeTrainingFromCheckpoint, renderResumePanelState, resumeTrainingFromCheckpoint, returnToLiveTraining } from '../anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getAppShellState } from '../anima-app/helpers/app-shell-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getAppContext } from '../anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadConfig, loadVariants, openTutorialDialog, reloadCurrentConfig } from '../anima-app/helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { ensureHistoryDetailFeature, isHistoryReviewMode } from '../anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { ensureImageTestFeature } from '../anima-app/helpers/image-test-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { confirmDiscardTomlChanges, updateTomlDirtyState } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { createBlankPresetFromLoraTemplate, exportTomlFile, handleTomlImport, importTomlFile, saveTomlAs } from '../anima-app/helpers/toml-io-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    auditConfigFullResumeSource,
    clearContinueTrainingSource,
    handleConfigFullResumeCheckpointChange,
    handleConfigFullResumeTaskChange,
    selectContinueLoraWeight,
    setConfigTrainingSourceMode,
} from '../anima-app/helpers/training-source-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { resetLogOutputLines, stopTraining, updateLogStatusText } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { setCurrentTrainingSourceFromVariant } from '../training-source/source-state.js?v=module-bootstrap-20260714-stage-dataset5';
import { confirmBeforeConfigSelectionChange, rememberSelectionSnapshot, updateChoiceGuide } from '../config-form/choice-guide-ui.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadDatasetPreviewImages } from '../dataset-editor/preview.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadDatasetPresets } from '../dataset-editor/load.js?v=module-bootstrap-20260714-stage-dataset5';
import { renderLiveChartPanel } from '../live-training/dashboard-ui.js?v=module-bootstrap-20260714-stage-dataset5';
import { selectConfigCategory, updateConfigStickyPlacement } from '../config-form/group-entry.js?v=module-bootstrap-20260714-stage-dataset5';
import { closeConfigDatasetPickerDialog } from '../config-form/dataset-picker.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadContinueLoraWeights, openContinueLoraDialog } from '../training-source/continue-lora.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    applyTomlToConfig,
    copyTomlEditorContent,
    toggleTomlEditorPanel,
    toggleTomlUserLock,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { saveGlobalSettings, resetGlobalSettings, toggleGlobalSettingHelp, syncGlobalUIScaleOverrideField, syncAllGlobalUIScaleOverrideFields } from '../anima-app/helpers/global-settings-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { savePreviewSettings, resetPreviewSettings, loadPreviewImages, loadPreviewWeights, setPreviewSource, openCurrentTrainingPreview, openLiveSamplingPreview, closePreviewPanel, togglePreviewWeightSort, changePreviewTask, restorePreviewWorkspaceAfterPanelClose } from '../anima-app/helpers/preview-view-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { bindTrainingViewTabKeyboard, showTrainingView } from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadTrainingHistoryList, renderHistoryManager } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { installBeginnerTooltips } from './beginner-tooltips.js?v=module-bootstrap-20260714-stage-dataset5';
import { SETUP_EVENT_DOM_CONTRACT } from './event-listeners-contract.js?v=module-bootstrap-20260714-stage-dataset5';
import { debounce } from '../../shared/debounce.js?v=module-bootstrap-20260714-stage-dataset5';
import { bindBrowseDialogBackdropClose } from '../../shared/dialog.js?v=module-bootstrap-20260714-stage-dataset5';

const ctx = getAppContext();
const appShellState = getAppShellState();
const datasetState = getDatasetState();
const historyState = getHistoryState();

const scheduleHistoryManagerRender = debounce(() => {
    renderHistoryManager();
}, 150);

// Browse/preview dialogs: click backdrop to close.
// Confirm dialogs (history-task-dialog / preflight) stay button+Esc only.
bindBrowseDialogBackdropClose([
    'tutorial-dialog',
    'preview-dialog',
    'dataset-preview-dialog',
    'config-dataset-picker-dialog',
    'continue-lora-dialog',
    'dataset-experimental-dialog',
    'stage-resolution-dialog',
    // image-test-layer-dialog / preview-panel-dialog / history-detail-dialog
    // already have dedicated close handlers elsewhere.
]);
const tomlState = getTomlState();
const trainingState = getTrainingState();

const REQUIRED_SETUP_EVENT_DOM_IDS = new Set(SETUP_EVENT_DOM_CONTRACT.required);

export function setupEventListeners() {
    const on = (id, eventName, handler, listenerOptions) => {
        return ctx.dom.bindEvent(id, eventName, handler, {
            contract: 'setupEventListeners',
            listenerOptions,
            required: REQUIRED_SETUP_EVENT_DOM_IDS.has(id),
        });
    };
    installBeginnerTooltips();
    on('method-select', 'change', async () => {
        if (!(await confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换方法会重新加载表单并丢弃这些修改。是否继续？'))) {
            return;
        }
        updateChoiceGuide();
        const variants = await loadVariants({ reset: true });
        if (variants.length) {
            await loadConfig();
        }
        rememberSelectionSnapshot();
    });
    on('variant-select', 'change', async () => {
        if (!(await confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换变体会重新加载表单并丢弃这些修改。是否继续？'))) {
            return;
        }
        setCurrentTrainingSourceFromVariant(val('variant-select'));
        updateChoiceGuide();
        await loadConfig();
        rememberSelectionSnapshot();
    });
    on('preset-select', 'change', async () => {
        if (!(await confirmBeforeConfigSelectionChange('当前配置有未保存修改，切换预设会重新加载表单并丢弃这些修改。是否继续？'))) {
            return;
        }
        updateChoiceGuide();
        await loadConfig();
        rememberSelectionSnapshot();
    });
    on('btn-load-config', 'click', reloadCurrentConfig);
    on('btn-start-from-config', 'click', startTraining);
    on('btn-queue-from-config', 'click', queueCurrentTrainingFromConfig);
    on('live-chart-toggle-lr', 'change', (event) => {
        trainingState.liveChartState.showLr = Boolean(event.target.checked);
        renderLiveChartPanel();
    });
    on('live-chart-range', 'change', (event) => {
        trainingState.liveChartState.rangeMode = event.target.value || 'all';
        renderLiveChartPanel();
    });
    document.querySelectorAll('[data-sticky-config-category]').forEach((btn) => {
        btn.addEventListener('click', () => selectConfigCategory(btn.dataset.stickyConfigCategory, { scrollToForm: true }));
    });
    window.addEventListener('resize', () => requestAnimationFrame(updateConfigStickyPlacement));
    on('btn-open-continue-lora-dialog', 'click', openContinueLoraDialog);
    on('btn-clear-continue-lora-source', 'click', clearContinueTrainingSource);
    document.querySelectorAll('[data-training-source-mode]').forEach((btn) => {
        btn.addEventListener('click', () => setConfigTrainingSourceMode(btn.dataset.trainingSourceMode || 'fresh'));
    });
    on('config-full-resume-task-select', 'change', (event) => {
        handleConfigFullResumeTaskChange(event.target.value || '');
    });
    on('config-full-resume-checkpoint-select', 'change', (event) => {
        handleConfigFullResumeCheckpointChange(event.target.value || '');
    });
    on('btn-refresh-config-full-resume', 'click', () => auditConfigFullResumeSource({ force: true }));
    on('btn-inspect-continue-lora-path', 'click', () => {
        selectContinueLoraWeight(document.getElementById('continue-lora-path-input')?.value || '');
    });
    on('continue-lora-history-task', 'change', (event) => {
        trainingState.continueLoraDialogState.taskId = event.target.value || '';
        loadContinueLoraWeights();
    });
    on('btn-refresh-continue-lora-weights', 'click', loadContinueLoraWeights);
    on('btn-open-tutorial', 'click', openTutorialDialog);
    on('btn-stop-training', 'click', stopTraining);
    on('btn-open-queue-manager', 'click', () => showTrainingView('queue'));
    on('btn-training-queue-view', 'click', () => showTrainingView('queue'));
    on('btn-training-history-view', 'click', () => showTrainingView('history'));
    on('btn-open-history-manager', 'click', () => showTrainingView('history'));
    ensureQueueFeature().bindQueueEvents();
    ensureWeightAnalysisFeature(ctx, appShellState).bindWeightAnalysisEvents();
    ensureEnvironmentCheckFeature(ctx, appShellState).bindEnvironmentCheckEvents();
    ensureImageTestFeature().bindImageTestEvents();
    on('btn-apply-toml', 'click', applyTomlToConfig);
    on('btn-move-toml-group', 'click', moveCurrentTomlToGroup);
    on('btn-create-blank-preset', 'click', createBlankPresetFromLoraTemplate);
    on('btn-save-toml', 'click', saveTomlFile);
    on('btn-toggle-toml-editor', 'click', toggleTomlEditorPanel);
    on('btn-copy-toml', 'click', copyTomlEditorContent);
    on('btn-save-toml-direct', 'click', () => saveTomlFile({ mode: 'editor' }));
    on('btn-import-toml', 'click', importTomlFile);
    on('btn-export-toml', 'click', exportTomlFile);
    on('btn-save-as-toml', 'click', saveTomlAs);
    on('btn-lock-toml', 'click', toggleTomlUserLock);
    on('btn-delete-toml', 'click', deleteTomlFile);
    on('btn-restore-system-toml', 'click', restoreSystemTomlPresets);
    on('toml-import-input', 'change', handleTomlImport);
    on('btn-toml-mode-project', 'click', () => switchTomlManagerMode('project'));
    on('btn-toml-mode-output', 'click', () => switchTomlManagerMode('output'));
    on('btn-refresh-output-runs', 'click', () => loadOutputRuns({ keepSelection: true }));
    on('btn-copy-output-config', 'click', copyOutputRunConfigContent);
    on('btn-export-output-config', 'click', exportOutputRunConfig);
    on('btn-save-output-config-as', 'click', openOutputRunSaveAs);
    on('btn-confirm-output-config-save-as', 'click', confirmOutputRunSaveAs);
    on('btn-cancel-output-config-save-as', 'click', closeOutputRunSaveAs);
    on('output-run-search', 'input', (event) => {
        datasetState.outputRunState = { ...datasetState.outputRunState, search: event.target.value || '' };
        renderOutputRunList();
    });
    on('btn-new-dataset-preset', 'click', createNewDatasetPreset);
    on('btn-copy-dataset-preset', 'click', copyDatasetPreset);
    on('btn-rename-dataset-preset', 'click', renameDatasetPreset);
    on('btn-import-dataset-preset', 'click', importDatasetPreset);
    on('dataset-import-input', 'change', handleDatasetPresetImport);
    on('btn-export-dataset-preset', 'click', exportDatasetPreset);
    on('btn-delete-dataset-preset', 'click', deleteDatasetPreset);
    on('btn-save-dataset-preset', 'click', saveDatasetPresetEditor);
    on('btn-create-dataset-preset-group', 'click', createDatasetPresetGroup);
    on('btn-refresh-dataset-presets', 'click', () => loadDatasetPresets({ selectCurrent: false, manage: true }));
    on('dataset-preset-search', 'input', (event) => {
        datasetState.datasetPresetState.search = event.target.value || '';
        renderDatasetPresetList();
    });
    on('btn-refresh-dataset-preview', 'click', loadDatasetPreviewImages);
    on('btn-config-dataset-dialog-refresh', 'click', () => loadDatasetPresets({ selectCurrent: false, manage: false }));
    on('btn-config-dataset-dialog-manage', 'click', () => {
        closeConfigDatasetPickerDialog();
        document.querySelector('[data-tab="datasets"]')?.click();
    });
    on('btn-reload-toml', 'click', async () => {
        const file = tomlState.currentTomlFile || val('toml-file-select');
        if (file && (await confirmDiscardTomlChanges('当前 TOML 有未保存修改，重新读取文件会丢失这些修改。是否继续？'))) {
            loadTomlFile(file, { force: true });
        }
    });
    on('toml-file-select', 'change', (e) => {
        selectAndApplyTomlFile(e.target.value);
    });
    on('toml-editor', 'input', updateTomlDirtyState);
    on('btn-clear-log', 'click', () => {
        if (isHistoryReviewMode()) return;
        resetLogOutputLines();
        trainingState.trainingRuntime.logBuffer = [];
        trainingState.trainingRuntime.logFlushPending = false;
        trainingState.trainingRuntime.logLineCount = 0;
        updateLogStatusText();
    });
    on('btn-refresh-history', 'click', () => loadTrainingHistoryList({ announce: true }));
    on('btn-preview-training-results', 'click', openCurrentTrainingPreview);
    on('btn-live-sampling-preview', 'click', openLiveSamplingPreview);
    on('btn-history-manager-refresh', 'click', () => loadTrainingHistoryList({ announce: true }));
    on('btn-history-collections-workbench', 'click', openHistoryCollectionsWorkbench);
    on('btn-history-manager-merge', 'click', mergeSelectedHistoryTasks);
    on('btn-history-bulk-archive', 'click', () => archiveSelectedHistoryTasks(true));
    on('btn-history-bulk-unarchive', 'click', () => archiveSelectedHistoryTasks(false));
    on('btn-history-bulk-group', 'click', groupSelectedHistoryTasks);
    on('btn-history-bulk-delete', 'click', deleteSelectedHistoryTasks);
    on('history-select-all', 'change', (event) => {
        const visible = historyState.historyCurrentVisibleTaskIds;
        if (event.target.checked) {
            visible.forEach((id) => historyState.selectedHistoryTaskIds.add(id));
        } else {
            visible.forEach((id) => historyState.selectedHistoryTaskIds.delete(id));
        }
        renderHistoryManager();
    });
    const historyFilterMap = {
        'history-manager-search': 'search',
        'history-filter-kind': 'kind',
        'history-filter-state': 'state',
        'history-filter-archived': 'archived',
        'history-filter-source': 'source',
        'history-filter-training-variant': 'trainingVariant',
        'history-filter-preprocess-precision': 'preprocessPrecision',
        'history-filter-block-swap-precision': 'blockSwapPrecision',
        'history-filter-base-compute': 'baseCompute',
        'history-filter-precision-preference': 'precisionPreference',
        'history-sort-mode': 'sort',
    };
    for (const [id, key] of Object.entries(historyFilterMap)) {
        on(id, id === 'history-manager-search' ? 'input' : 'change', (event) => {
            const value = event.target.value || historyManagerFilterDefault(key);
            historyState.historyManagerFilters[key] = value;
            if (id === 'history-manager-search') {
                scheduleHistoryManagerRender();
            } else {
                scheduleHistoryManagerRender.cancel();
                renderHistoryManager();
            }
        });
    }
    on('history-collection-search', 'input', (event) => {
        historyState.historyCollectionSearch = event.target.value || '';
        scheduleHistoryManagerRender();
    });
    on('history-config-group-search', 'input', (event) => {
        historyState.historyConfigGroupSearch = event.target.value || '';
        scheduleHistoryManagerRender();
    });
    ensureHistoryDetailFeature().bindHistoryDetailEvents();
    on('btn-live-training', 'click', returnToLiveTraining);
    bindTrainingViewTabKeyboard();
    on('btn-refresh-history-view', 'click', refreshHistoryView);
    on('btn-close-history', 'click', returnToLiveTraining);
    on('btn-refresh-resume-options', 'click', () => loadResumeOptionsForTask());
    on('btn-resume-training', 'click', resumeTrainingFromCheckpoint);
    on('btn-queue-resume-training', 'click', queueResumeTrainingFromCheckpoint);
    on('resume-checkpoint-select', 'change', renderResumePanelState);
    on('history-show-archived', 'change', (e) => {
        historyState.showArchivedHistory = e.target.checked;
        loadTrainingHistoryList();
    });
    document.querySelectorAll('.preview-source-btn').forEach((btn) => {
        btn.addEventListener('click', () => setPreviewSource(btn.dataset.previewSource));
    });
    on('btn-refresh-preview', 'click', loadPreviewImages);
    on('btn-refresh-weights', 'click', loadPreviewWeights);
    on('btn-sort-weights', 'click', togglePreviewWeightSort);
    on('btn-save-preview-settings', 'click', savePreviewSettings);
    on('btn-reset-preview-settings', 'click', resetPreviewSettings);
    on('btn-close-preview-panel', 'click', closePreviewPanel);
    on('preview-panel-dialog', 'click', (event) => {
        if (event.target === event.currentTarget) closePreviewPanel();
    });
    on('preview-panel-dialog', 'close', restorePreviewWorkspaceAfterPanelClose);
    on('btn-save-global-settings', 'click', saveGlobalSettings);
    on('btn-reset-global-settings', 'click', resetGlobalSettings);
    document.querySelectorAll('.global-setting-help-toggle').forEach((btn) => {
        btn.addEventListener('click', () => toggleGlobalSettingHelp(btn));
    });
    on('global-ui-scale', 'input', () => {
        syncAllGlobalUIScaleOverrideFields({ preserveCustom: true });
    });
    on('global-ui-scale', 'change', () => {
        syncAllGlobalUIScaleOverrideFields({ preserveCustom: true });
    });
    GLOBAL_UI_OVERRIDE_FIELDS.forEach((field) => {
        on(field.followDefaultId, 'change', () => {
            syncGlobalUIScaleOverrideField(field);
        });
    });
    on('preview-training-task', 'change', (e) => changePreviewTask(e.target.value));

    setTomlManagerMode('project');
}
