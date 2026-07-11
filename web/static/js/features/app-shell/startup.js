/**
 * App shell startup + history-detail feature wiring.
 * Moved out of anima-app mechanical chunks.
 */
import { MetricsChart } from '../../../chart.js?v=module-bootstrap-20260711-ir1';
import { createGpuPicker } from './gpu-picker.js?v=module-bootstrap-20260711-ir1';
import { createTabController } from './tabs.js?v=module-bootstrap-20260711-ir1';
import { createThemeController } from './theme.js?v=module-bootstrap-20260711-ir1';
import { createLanguageController } from './language.js?v=module-bootstrap-20260711-ir1';
import { createUIScaleController } from './ui-scale.js?v=module-bootstrap-20260711-ir1';
import { createHistoryDetailFeature } from '../history-detail/index.js?v=module-bootstrap-20260711-ir1';
import { formatLr, lastValue } from '../live-training/index.js?v=module-bootstrap-20260711-ir1';
import { setupEventListeners } from './event-listeners.js?v=module-bootstrap-20260711-ir1';
import {
    loadDatasetPresets,
    loadStepEstimate,
    renderLiveChartPanel,
    resetLiveMetricPlaceholders,
    scheduleStepEstimatePanelRefresh,
    syncLiveChartControls,
    syncLossChartEmptyState,
} from '../anima-app/chunks/03-parse-network-arg-entry.js?v=module-bootstrap-20260711-ir1';
import {
    clearCurrentTrainingSource,
    setCurrentTrainingSourceFromVariant,
} from '../training-source/source-state.js?v=module-bootstrap-20260711-ir1';
import {
    rememberSelectionSnapshot,
    updateChoiceGuide,
} from '../config-form/choice-guide-ui.js?v=module-bootstrap-20260711-ir1';
import { requestContinueLoraInspection } from '../anima-app/chunks/06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260711-ir1';
import {
    GLOBAL_UI_HISTORY_DETAIL_OVERRIDE_FIELDS,
    GLOBAL_UI_TOP_LEVEL_OVERRIDE_FIELDS,
} from '../../config/catalog.js?v=module-bootstrap-20260711-ir1';
import { GPU_WHITELIST_STORAGE_KEY, LANGUAGE_STORAGE_KEY, THEME_STORAGE_KEY } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260711-ir1';
import { datasetPresetSummaryByFile } from '../anima-app/helpers/dataset-presets.js?v=module-bootstrap-20260711-ir1';
import { isDatasetTabActive } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir1';
import { ensureEnvironmentCheckFeature, ensureWeightAnalysisFeature } from '../anima-app/helpers/feature-ensurers.js?v=module-bootstrap-20260711-ir1';
import { makeHistoryArtifactUrl } from '../anima-app/helpers/history-artifacts.js?v=module-bootstrap-20260711-ir1';
import {
    auditConfigTrainingSourceOnEnter,
    refreshContinueTrainingSourceCompatibility,
    renderContinueTrainingSource,
    selectContinueLoraWeight,
} from '../anima-app/helpers/training-source-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadDefaultTomlFile, loadTomlFileList } from '../anima-app/helpers/toml-manager-bridge.js?v=module-bootstrap-20260711-ir1';
import { refreshTrainingHealth } from '../anima-app/helpers/live-status-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    canPreviewHistoryConfigGroup,
    historyContinueLabel,
    historyQueueLabel,
    historyResumeLabel,
    historyTaskDisplayName,
    historyTaskIsArchived,
} from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    activateHistoryDetailPreview,
    archiveHistoryTask,
    clearViewingHistoryTaskContext,
    createHistoryActionButton,
    createHistoryTaskPreviewButton,
    deleteHistoryTask,
    historyLossChartPoints,
    historyTaskLabel,
    renameHistoryTask,
    restorePreviewWorkspaceFromHistoryDetail,
    shouldRenderInlineResumePanel,
    showHistoryTaskConfirmDialog,
} from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    configGroupLabel,
    configGroupTimelineSummary,
    formatGroupTimelineLogRecord,
    formatStepRange,
    historyStateLabel,
    metricsWithProgressFallback,
    returnToLiveTraining,
    runtimePathItems,
} from '../anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260711-ir1';
import { setTomlStatus, updateTomlActionState } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { configureAppShellStartupBridge } from '../anima-app/helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260711-ir1';
import { getAppShellState } from '../anima-app/helpers/app-shell-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getAppContext } from '../anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260711-ir1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getHistoryState } from '../anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadGlobalSettings } from '../anima-app/helpers/global-settings-bridge.js?v=module-bootstrap-20260711-ir1';
import { configureHistoryDetailBridge } from '../anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260711-ir1';
import { ensureImageTestFeature } from '../anima-app/helpers/image-test-bridge.js?v=module-bootstrap-20260711-ir1';
import { confirmDiscardTomlChanges, updateTomlDirtyState } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadSamplePrompts } from '../anima-app/helpers/sample-prompts-bridge.js?v=module-bootstrap-20260711-ir1';
import { api, populateSelect, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';
import { downloadBlob } from '../anima-app/helpers/toml-io-bridge.js?v=module-bootstrap-20260711-ir1';
import { currentTrainingConfigFile } from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260711-ir1';
import { appendLog, connectWebSocket, logLineTone, recoverLiveTrainingState } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir1';
import { scheduleStatusPoll } from '../anima-app/helpers/status-polling-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    loadTrainingQueue,
    showTrainingView,
    resetTrainingExpandedStateOnLeave,
    updateTrainingQueueFromPayload,
} from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadTrainingHistoryList, renderHistoryManager, renderTrainingHistoryList } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadTomlFile } from '../anima-app/helpers/output-run-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadPreviewSettings, normalizePreviewGroup, copyText } from '../anima-app/helpers/preview-view-bridge.js?v=module-bootstrap-20260711-ir1';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    applyConfigCompatibilityDrafts,
    normalizeNetworkArgArray,
    parseNetworkArgMap,
    renderConfigForm,
    resetConfigFormDraft,
    syncConfigDraftFromForm,
} from '../config-form/index.js?v=module-bootstrap-20260711-ir1';

const ctx = getAppContext();
const appShellState = getAppShellState();
const configState = getConfigState();
const datasetState = getDatasetState();
const historyState = getHistoryState();
const tomlState = getTomlState();
const trainingState = getTrainingState();
let historyDetailFeature = null;
let themeController = null;
let uiScaleController = null;
let gpuPicker = null;
let tabController = null;

function currentConfigState() { return configState.currentConfig || {}; }

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function currentContinueTrainingSource() {
    return trainingState.continueTrainingSource || null;
}

function setPreviewEmpty(message) {
    const empty = document.getElementById('preview-empty');
    if (!empty) return;
    empty.textContent = message;
    empty.hidden = false;
    const grid = document.getElementById('preview-grid');
    if (grid) grid.innerHTML = '';
}

    export function ensureHistoryDetailFeature() {
        if (historyDetailFeature) return historyDetailFeature;
        historyDetailFeature = createHistoryDetailFeature(ctx, {
            setViewingHistoryTaskContext: ({
                taskId = '',
                viewMode = 'live',
                task = null,
                configGroup = null,
                timelineSelection = [],
            } = {}) => {
                historyState.viewingHistoryTaskId = taskId || '';
                historyState.historyViewMode = viewMode || 'live';
                historyState.currentHistoryTaskForResume = task || null;
                historyState.currentHistoryConfigGroup = configGroup || null;
                historyState.currentHistoryTimelineSelection = Array.isArray(timelineSelection) ? timelineSelection : [];
            },
            getViewingHistoryTaskId: () => historyState.viewingHistoryTaskId,
            getCurrentHistoryTaskForResume: () => historyState.currentHistoryTaskForResume,
            setCurrentHistoryTaskForResume: (task) => { historyState.currentHistoryTaskForResume = task || null; },
            renderTrainingHistoryList,
            renderHistoryManager,
            loadTrainingHistoryList,
            showTrainingView,
            returnToLiveTraining,
            clearViewingHistoryTaskContext,
            shouldRenderInlineResumePanel,
            getTrainingViewMode: () => trainingState.trainingViewMode,
            getTrainingRuntime: () => trainingState.trainingRuntime,
            activateHistoryDetailPreview,
            restorePreviewWorkspaceFromHistoryDetail,
            updateTrainingQueueFromPayload,
            appendLog,
            historyTaskDisplayName,
            historyTaskLabel,
            historyStateLabel,
            historyQueueLabel,
            historyResumeLabel,
            historyContinueLabel,
            historyTaskIsArchived,
            createHistoryActionButton,
            createHistoryTaskPreviewButton,
            renameHistoryTask,
            archiveHistoryTask,
            deleteHistoryTask,
            canPreviewHistoryConfigGroup,
            normalizePreviewGroup,
            configGroupLabel,
            runtimePathItems,
            historyArtifactUrl: makeHistoryArtifactUrl,
            copyText,
            downloadBlob,
            selectedGpuPayload: () => gpuPicker.selectedGpuPayload(),
            inspectContinueLoraWeight: requestContinueLoraInspection,
            selectContinueLoraWeight,
            showHistoryTaskConfirmDialog,
            formatLr,
            lastValue,
            metricsWithProgressFallback,
            historyLossChartPoints,
            formatStepRange,
            configGroupTimelineSummary,
            formatGroupTimelineLogRecord,
            logLineTone,
            applyHistoryDetailUIScale: (detailTab) => {
                uiScaleController?.applyHistoryDetailScale?.(appShellState.globalSettings || {}, detailTab || 'overview');
            },
        });
        return historyDetailFeature;
    }

    // ── 初始化 ──

    export async function startAnimaApp() {
        themeController = createThemeController({
            storageKey: THEME_STORAGE_KEY,
            getLossChart: () => trainingState.lossChart,
            chartTheme,
        });
        const languageController = createLanguageController({
            storageKey: LANGUAGE_STORAGE_KEY,
        });
        uiScaleController = createUIScaleController({
            topLevelFields: GLOBAL_UI_TOP_LEVEL_OVERRIDE_FIELDS,
            historyDetailFields: GLOBAL_UI_HISTORY_DETAIL_OVERRIDE_FIELDS,
        });
        gpuPicker = createGpuPicker({
            storageKey: GPU_WHITELIST_STORAGE_KEY,
            api,
        });
        tabController = createTabController({
            loadDatasetPresets,
            loadGlobalSettings,
            ensureWeightAnalysisFeature: () => ensureWeightAnalysisFeature(ctx, appShellState),
            ensureEnvironmentCheckFeature: () => ensureEnvironmentCheckFeature(ctx, appShellState),
            ensureImageTestFeature,
            resetTrainingExpandedStateOnLeave,
            resizeLiveChart: () => trainingState.lossChart?.resize?.(),
            auditConfigTrainingSourceOnEnter,
        });

        const boot = async () => {
            themeController.initThemeToggle();
            languageController.initLanguageToggle();
            uiScaleController.initUIScale();
            tabController.setupTabs();
            trainingState.lossChart = new MetricsChart(document.getElementById('loss-chart'), {
                emptyText: '',
                showLr: trainingState.liveChartState.showLr,
                rangeMode: trainingState.liveChartState.rangeMode,
            });
            trainingState.lossChart.setTheme(chartTheme());
            resetLiveMetricPlaceholders();
            syncLossChartEmptyState();
            syncLiveChartControls();
            renderLiveChartPanel();
            setupEventListeners();
            gpuPicker.initGpuPickerEvents();
            await loadInitialData();
            if (location.protocol !== 'file:') {
                connectWebSocket();
                recoverLiveTrainingState();
                scheduleStatusPoll();
                setInterval(refreshTrainingHealth, 1000);
                document.addEventListener('visibilitychange', () => {
                    scheduleStatusPoll({ immediate: !document.hidden });
                    if (!document.hidden) recoverLiveTrainingState();
                });
                window.addEventListener('online', () => {
                    scheduleStatusPoll({ immediate: true });
                    recoverLiveTrainingState();
                });
            }
        };

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', boot, { once: true });
        } else {
            await boot();
        }
    };
    export function chartTheme() {
        const trainingRoot = document.getElementById('tab-training');
        const styles = getComputedStyle(trainingRoot || document.documentElement);
        const rootStyles = getComputedStyle(document.documentElement);
        const read = (...names) => {
            for (const name of names) {
                const value = styles.getPropertyValue(name).trim() || rootStyles.getPropertyValue(name).trim();
                if (value) return value;
            }
            return '';
        };
        return {
            color: read('--training-accent', '--accent') || '#4fc3f7',
            grid: read('--training-border', '--chart-grid') || '#2a3a5e',
            text: read('--training-muted', '--text-dim') || '#8892a4',
            tooltipBg: read('--training-panel-bg', '--bg-card') || '#16213e',
            tooltipBorder: read('--training-border', '--border') || '#2a3a5e',
            tooltipText: read('--training-text', '--text') || '#e0e0e0',
            highlight: read('--warning') || '#f0c36a',
            crosshair: read('--training-accent', '--accent') || '#4fc3f7',
            lr: read('--warning') || '#f0c36a',
        };
    }

    export function isHistoryReviewMode() {
        return historyState.historyViewMode !== 'live';
    }

    export function openTutorialDialog() {
        const dialog = document.getElementById('tutorial-dialog');
        if (!dialog) return;
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
    }


    function asItemList(payload) {
        // C-R6 progressive envelope: prefer {ok, items}, keep bare array compatibility.
        if (Array.isArray(payload)) return payload;
        if (payload && Array.isArray(payload.items)) return payload.items;
        return [];
    }

    // ── 加载初始数据 ──
    export async function loadInitialData() {
        if (location.protocol === 'file:') {
            await gpuPicker.loadGpuOptions();
            showStandaloneWarning();
            return;
        }
        try {
            const [methodsPayload, presetsPayload, help] = await Promise.all([
                api('/api/methods'),
                api('/api/presets'),
                api('/api/config/field-help'),
            ]);
            configState.fieldHelp = help;
            populateSelect('method-select', asItemList(methodsPayload), 'lora');
            populateSelect('preset-select', asItemList(presetsPayload), 'default');
            await gpuPicker.loadGpuOptions();
            const variants = await loadVariants();
            const tomlListPromise = loadTomlFileList('', { deferDefaultLoad: true })
                .then(() => ({ ok: true }))
                .catch((error) => ({ ok: false, error }));
            await loadDatasetPresets({ selectCurrent: false, manage: isDatasetTabActive() });
            if (variants.length) {
                await loadConfig();
            }
            const tomlListResult = await tomlListPromise;
            if (!tomlListResult.ok) throw tomlListResult.error;
            if (!tomlState.currentTomlFile) {
                await loadDefaultTomlFile();
            }
            rememberSelectionSnapshot();
            await loadTrainingQueue();
            await loadTrainingHistoryList();
            await loadPreviewSettings();
            await loadGlobalSettings();
            returnToLiveTraining({ refresh: false });
        } catch (e) {
            console.error('初始化失败:', e);
        }
    }

    export function showStandaloneWarning() {
        const form = document.getElementById('config-form');
        form.innerHTML = '';
        const panel = document.createElement('div');
        panel.className = 'standalone-warning';
        panel.innerHTML = [
            '<strong>当前是 file:// 静态打开模式，无法读取或保存项目配置。</strong>',
            '<p>请在项目根目录启动 Web 服务后访问 <code>http://127.0.0.1:20102/</code>：</p>',
            '<pre>.venv/bin/python -m web --host 127.0.0.1 --port 20102</pre>',
        ].join('');
        form.appendChild(panel);
        setTomlStatus('error', '静态打开没有后端 API，保存/另存为/读取配置不可用', { persist: true });
        setPreviewEmpty('静态打开没有后端 API，无法读取项目预览图。');
    }

    export async function loadVariants({ reset = false } = {}) {
        const method = val('method-select');
        const variantsPayload = await api(`/api/methods/${method}/variants`);
        const variants = asItemList(variantsPayload);
        populateSelect('variant-select', variants, reset ? (variants[0] || method) : method);
        const selectedVariant = val('variant-select');
        if (!selectedVariant) {
            clearCurrentTrainingSource();
            setTomlStatus('error', `方法 ${method} 暂无可训练变体，已阻止加载配置`, { persist: true });
            updateChoiceGuide();
            return [];
        }
        setCurrentTrainingSourceFromVariant(selectedVariant);
        updateChoiceGuide();
        return variants;
    }

    export async function loadConfig() {
        const requestSeq = ++configState.configLoadSeq;
        const currentTrainingSource = currentTrainingSourceState();
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        if (!variant) return;
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const params = new URLSearchParams({ variant, preset, methods_subdir: methodsSubdir });
        const configFile = currentTrainingConfigFile();
        if (configFile) params.set('config_file', configFile);
        const data = await api(`/api/config/merged?${params.toString()}`);
        if (requestSeq !== configState.configLoadSeq) return;
        if (data?.ok === false) {
            setTomlStatus('error', data.error || '读取配置失败');
            return;
        }
        resetConfigFormDraft();
        configState.currentConfig = data;
        datasetState.selectedConfigDatasetFile = data.dataset_config || '';
        datasetState.selectedConfigDatasetSummary = datasetPresetSummaryByFile(datasetState.selectedConfigDatasetFile);
        renderConfigForm(data);
        scheduleStepEstimatePanelRefresh();
        const compatibilityPatch = applyConfigCompatibilityDrafts();
        renderContinueTrainingSource();
        if (currentContinueTrainingSource()?.abs_path) {
            await refreshContinueTrainingSourceCompatibility();
        }
        if (configState.samplePromptsMode === 'editor-file') {
            loadSamplePrompts(configState.samplePromptsPath, requestSeq);
        } else {
            configState.samplePromptsLoadSeq += 1;
        }
        loadStepEstimate(requestSeq);
        updateChoiceGuide();
        updateTomlActionState(tomlState.currentTomlFile);
        // 同步加载对应的 TOML 文件到右侧编辑器
        const tomlFile = currentTrainingSource.file || `configs/${methodsSubdir}/${variant}.toml`;
        if (tomlState.tomlFiles.includes(tomlFile) && tomlState.currentTomlFile !== tomlFile) {
            await loadTomlFile(tomlFile, { force: true });
        }
        if (Object.keys(compatibilityPatch).length > 0) {
            updateTomlDirtyState();
            setTomlStatus('error', '已自动修正 CAME optimizer_args 的 betas 格式，请保存当前配置后再训练。', { persist: true });
        }
    }

    export async function reloadCurrentConfig() {
        if (!(await confirmDiscardTomlChanges('当前配置有未保存修改，刷新会重新读取表单和数据集设置并丢弃这些修改。是否继续？'))) {
            return;
        }
        await loadConfig();
        rememberSelectionSnapshot();
    }

    // ── 配置表单渲染 ──

configureHistoryDetailBridge({
    ensureHistoryDetailFeature,
    getHistoryDetailFeature: () => historyDetailFeature,
    isHistoryReviewMode,
});

configureAppShellStartupBridge({
    startAnimaApp,
    openTutorialDialog,
    loadVariants,
    loadConfig,
    reloadCurrentConfig,
    renderConfigForm,
    syncConfigDraftFromForm,
    parseNetworkArgMap,
    normalizeNetworkArgArray,
    getThemeController: () => themeController,
    getUiScaleController: () => uiScaleController,
    getGpuPicker: () => gpuPicker,
    getTabController: () => tabController,
});
