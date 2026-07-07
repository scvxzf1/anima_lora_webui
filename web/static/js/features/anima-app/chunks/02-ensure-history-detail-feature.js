/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { MetricsChart } from '../../../../chart.js?v=module-bootstrap-20260707-93';
import { applyLoraAdapterDraft, applyOptimizerCompatibilityPatch } from './14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260707-93';
import { createGpuPicker } from '../../app-shell/gpu-picker.js?v=module-bootstrap-20260707-93';
import { createTabController } from '../../app-shell/tabs.js?v=module-bootstrap-20260707-93';
import { createThemeController } from '../../app-shell/theme.js?v=module-bootstrap-20260707-93';
import { createUIScaleController } from '../../app-shell/ui-scale.js?v=module-bootstrap-20260707-93';
import { createHistoryDetailFeature } from '../../history-detail/index.js?v=module-bootstrap-20260707-93';
import { formatLr, lastValue } from '../../live-training/index.js?v=module-bootstrap-20260707-93';
import { setupEventListeners } from './36-setup-event-listeners.js?v=module-bootstrap-20260707-93';
import { loadDatasetPresets, loadStepEstimate, renderLiveChartPanel, resetLiveMetricPlaceholders, scheduleStepEstimatePanelRefresh, syncLiveChartControls, syncLossChartEmptyState } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260707-93';
import { activeMethodKey, clearCurrentTrainingSource, rememberSelectionSnapshot, setCurrentTrainingSourceFromVariant, updateChoiceGuide } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260707-93';
import { requestContinueLoraInspection } from './06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260707-93';
import {
    CHIMERA_UI_DEFAULT_FIELDS,
    CONFIG_FORM_INTERNAL_KEYS,
    CONFIG_FORM_MERGED_FIELDS,
    DATASET_BLUEPRINT_FIELDS,
    DEPRECATED_CONFIG_FORM_FIELDS,
    FORM_SECTION_DEFS,
    FORM_UI_DEFAULTS,
    GLOBAL_UI_HISTORY_DETAIL_OVERRIDE_FIELDS,
    GLOBAL_UI_TOP_LEVEL_OVERRIDE_FIELDS,
    IP_ADAPTER_UI_DEFAULT_FIELDS,
    METHOD_SCOPED_CONFIG_FORM_FIELDS,
    NETWORK_ARG_FIELD_MAP,
    NETWORK_ARG_FIELD_SPECS,
    RETIRED_CONFIG_FORM_FIELDS,
    SOFT_TOKENS_UI_DEFAULT_FIELDS,
    SPD_UI_DEFAULT_FIELDS,
    help,
} from '../../../config/catalog.js?v=module-bootstrap-20260707-93';
import { GPU_WHITELIST_STORAGE_KEY, THEME_STORAGE_KEY } from '../helpers/app-constants.js?v=module-bootstrap-20260707-93';
import { isTruthy, loraAdapterFlagsMatchConfig, loraAdapterKindFromConfig, normalizeLoraAdapterKind, normalizePrecisionPreference, precisionPreferenceFromConfig } from '../helpers/config-values.js?v=module-bootstrap-20260707-93';
import { datasetPresetSummaryByFile } from '../helpers/dataset-presets.js?v=module-bootstrap-20260707-93';
import { isDatasetTabActive } from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260707-93';
import { ensureEnvironmentCheckFeature, ensureWeightAnalysisFeature } from '../helpers/feature-ensurers.js?v=module-bootstrap-20260707-93';
import { configureConfigFormBridge, networkArgFieldValueFromConfig, readFieldInputValue, shouldSkipUiDefaultField, updateDoRAFieldState, updateLoKrFieldState, updateLossWeightingFieldState, updateVeRAFieldState } from '../helpers/config-form-bridge.js?v=module-bootstrap-20260707-93';
import { makeHistoryArtifactUrl } from '../helpers/history-artifacts.js?v=module-bootstrap-20260707-93';
import { coerceNetworkArgValue, parseNetworkArgEntry } from '../helpers/network-args.js?v=module-bootstrap-20260707-93';
import { parseArrayValue, valuesEqual } from '../helpers/form-values.js?v=module-bootstrap-20260707-93';
import { auditConfigTrainingSourceOnEnter, refreshContinueTrainingSourceCompatibility, renderContinueTrainingSource, selectContinueLoraWeight } from '../helpers/training-source-bridge.js?v=module-bootstrap-20260707-93';
import { loadDefaultTomlFile, loadTomlFileList } from '../helpers/toml-manager-bridge.js?v=module-bootstrap-20260707-93';
import { refreshTrainingHealth } from '../helpers/live-status-bridge.js?v=module-bootstrap-20260707-93';
import { canPreviewHistoryConfigGroup, historyContinueLabel, historyQueueLabel, historyResumeLabel, historyTaskDisplayName, historyTaskIsArchived } from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260707-93';
import { activateHistoryDetailPreview, archiveHistoryTask, clearViewingHistoryTaskContext, createHistoryActionButton, createHistoryTaskPreviewButton, deleteHistoryTask, historyLossChartPoints, historyTaskLabel, renameHistoryTask, restorePreviewWorkspaceFromHistoryDetail, shouldRenderInlineResumePanel, showHistoryTaskConfirmDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260707-93';
import { configGroupLabel, configGroupTimelineSummary, formatGroupTimelineLogRecord, formatStepRange, historyStateLabel, metricsWithProgressFallback, returnToLiveTraining, runtimePathItems } from '../helpers/history-timeline-bridge.js?v=module-bootstrap-20260707-93';
import { appendConfigGroupsByCategory, createConfigGroupEntry } from './04-create-config-group-entry.js?v=module-bootstrap-20260707-93';
import { setTomlStatus, updateTomlActionState } from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260707-93';
import { configureAppShellStartupBridge } from '../helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260707-93';
import { getAppShellState } from '../helpers/app-shell-state-bridge.js?v=module-bootstrap-20260707-93';
import { getAppContext } from '../helpers/app-context-bridge.js?v=module-bootstrap-20260707-93';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260707-93';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260707-93';
import { loadGlobalSettings } from '../helpers/global-settings-bridge.js?v=module-bootstrap-20260707-93';
import { configureHistoryDetailBridge } from '../helpers/history-detail-bridge.js?v=module-bootstrap-20260707-93';
import { ensureImageTestFeature } from '../helpers/image-test-bridge.js?v=module-bootstrap-20260707-93';
import { confirmDiscardTomlChanges, updateTomlDirtyState } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260707-93';
import { currentSamplePromptText, loadSamplePrompts } from '../helpers/sample-prompts-bridge.js?v=module-bootstrap-20260707-93';
import { api, populateSelect, val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260707-93';
import { downloadBlob } from '../helpers/toml-io-bridge.js?v=module-bootstrap-20260707-93';
import { currentTrainingConfigFile } from '../helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260707-93';
import { appendLog, connectWebSocket, logLineTone, recoverLiveTrainingState } from '../helpers/live-log-bridge.js?v=module-bootstrap-20260707-93';
import { scheduleStatusPoll } from '../helpers/status-polling-bridge.js?v=module-bootstrap-20260707-93';
import { loadTrainingQueue, showTrainingView, resetTrainingExpandedStateOnLeave, updateTrainingQueueFromPayload } from '../helpers/queue-view-bridge.js?v=module-bootstrap-20260707-93';
import { loadTrainingHistoryList, renderHistoryManager, renderTrainingHistoryList } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260707-93';
import { loadTomlFile } from '../helpers/output-run-bridge.js?v=module-bootstrap-20260707-93';
import { loadPreviewSettings, normalizePreviewGroup, copyText } from '../helpers/preview-view-bridge.js?v=module-bootstrap-20260707-93';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260707-93';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260707-93';

const ctx = getAppContext();
const appShellState = getAppShellState();
const configState = getConfigState();
const datasetState = getDatasetState();
const historyState = getHistoryState();
const tomlState = getTomlState();
const trainingState = getTrainingState();
const configFormState = configState.configFormState;
const ALWAYS_VISIBLE_NETWORK_ARG_FIELDS = new Set(['lokr_use_einsum', 'lokr_decompose_w2', 'lokr_factor_group_size', 'lokr_project_chunk_bytes']);
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

    // ── 加载初始数据 ──
    export async function loadInitialData() {
        if (location.protocol === 'file:') {
            await gpuPicker.loadGpuOptions();
            showStandaloneWarning();
            return;
        }
        try {
            const [methods, presets, help] = await Promise.all([
                api('/api/methods'),
                api('/api/presets'),
                api('/api/config/field-help'),
            ]);
            configState.fieldHelp = help;
            populateSelect('method-select', methods, 'lora');
            populateSelect('preset-select', presets, 'default');
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
        const variants = await api(`/api/methods/${method}/variants`);
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
    export function resetConfigFormDraft() {
        configState.configFormState.draftValues.clear();
    }

    export function applyConfigCompatibilityDrafts() {
        const patch = applyOptimizerCompatibilityPatch({});
        for (const [key, value] of Object.entries(patch)) {
            configFormState.draftValues.set(key, value);
            const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
            if (!input) continue;
            if (input.type === 'checkbox') {
                input.checked = Boolean(value);
            } else {
                input.value = Array.isArray(value) ? JSON.stringify(value) : (value ?? '');
            }
            input.title = `${input.title ? `${input.title}\n` : ''}已自动修正兼容性参数，请保存当前配置后再训练。`;
        }
        return patch;
    }

    export function syncConfigDraftFromForm(options = {}) {
        document.querySelectorAll('#config-form .field-input[data-key]').forEach((input) => {
            updateConfigDraftFromInput(input, options);
        });
    }

    export function updateConfigDraftFromInput(input, options = {}) {
        const key = input?.dataset?.key;
        if (!key || CONFIG_FORM_INTERNAL_KEYS.has(key)) return;
        const original = originalConfigFieldValue(key);
        const next = readFieldInputValue(input, original);
        if (key === 'lora_adapter_kind') {
            applyLoraAdapterDraft(next);
            return;
        }
        if (configDraftValueChanged(key, next, original, options)) {
            configFormState.draftValues.set(key, next);
        } else {
            configFormState.draftValues.delete(key);
        }
    }

    export function originalConfigFieldValue(key) {
        const currentConfig = currentConfigState();
        if (key === 'sample_prompts' && configState.samplePromptsMode !== 'path') {
            return configState.samplePromptsContent || '';
        }
        if (key === 'precision_preference') {
            return precisionPreferenceFromConfig(currentConfig);
        }
        if (isActiveNetworkArgFieldKey(key)) {
            return networkArgFieldValueFromConfig(NETWORK_ARG_FIELD_MAP.get(key), currentConfig);
        }
        if (key === 'lora_adapter_kind') {
            return loraAdapterKindFromConfig(currentConfig);
        }
        if (key in currentConfig) return currentConfig[key];
        return FORM_UI_DEFAULTS[key];
    }

    export function displayConfigFieldValue(key, value) {
        const currentConfig = currentConfigState();
        if (key === 'lora_adapter_kind') {
            return configFormState.draftValues.has(key)
                ? configFormState.draftValues.get(key)
                : loraAdapterKindFromConfig(currentConfig);
        }
        if (key === 'precision_preference') {
            return configFormState.draftValues.has(key)
                ? normalizePrecisionPreference(configFormState.draftValues.get(key))
                : precisionPreferenceFromConfig(currentConfig);
        }
        return configFormState.draftValues.has(key)
            ? configFormState.draftValues.get(key)
            : value;
    }

    export function configDraftValueChanged(key, next, original = originalConfigFieldValue(key), options = {}) {
        const currentConfig = currentConfigState();
        if (key === 'sample_prompts' && configState.samplePromptsMode !== 'path') {
            return String(next || '') !== String(configState.samplePromptsContent || '');
        }
        if (key === 'precision_preference') {
            return normalizePrecisionPreference(next) !== precisionPreferenceFromConfig(currentConfig);
        }
        if (isActiveNetworkArgFieldKey(key)) {
            return !valuesEqual(next, original);
        }
        if (key === 'lora_adapter_kind') {
            return normalizeLoraAdapterKind(next) !== normalizeLoraAdapterKind(original)
                || !loraAdapterFlagsMatchConfig(next, currentConfig);
        }
        const hasOriginal = key in currentConfig;
        if (!hasOriginal && shouldSkipUiDefaultField(key, next, options)) return false;
        return !valuesEqual(next, original);
    }

    export function renderConfigForm(config) {
        const container = document.getElementById('config-form');
        container.innerHTML = '';

        const fieldsByKey = {};
        for (const [key, value] of Object.entries(config)) {
            if (key === 'output_dir') continue;
            if (key === 'general' || key === 'datasets') continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (CONFIG_FORM_MERGED_FIELDS?.has?.(key)) continue;
            if (shouldSkipConfigFormField(key, config)) continue;
            if (DATASET_BLUEPRINT_FIELDS.has(key)) continue;
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) continue;
            fieldsByKey[key] = value;
        }
        for (const [key, value] of Object.entries(FORM_UI_DEFAULTS)) {
            if (key === 'output_dir') continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (CONFIG_FORM_MERGED_FIELDS?.has?.(key)) continue;
            if (shouldSkipConfigFormField(key, config)) continue;
            if (DATASET_BLUEPRINT_FIELDS.has(key)) continue;
            if (!shouldExposeUiDefaultField(key, config, fieldsByKey)) continue;
            if (!(key in fieldsByKey)) fieldsByKey[key] = value;
        }
        applyNetworkArgFields(fieldsByKey, config);
        fieldsByKey.sample_prompts = currentSamplePromptText(config);

        const consumed = new Set();
        const sectionEntries = [];
        for (const section of FORM_SECTION_DEFS) {
            if (!shouldRenderConfigSection(section, config)) continue;
            const fields = collectSectionFields(fieldsByKey, section.keys, consumed);
            if (fields.length > 0) {
                sectionEntries.push(createConfigGroupEntry(
                    section.title,
                    fields,
                    section.className || '',
                    section.description || '',
                    section.open,
                    section.notice || ''
                ));
            }
        }

        const otherFields = Object.entries(fieldsByKey).filter(([key]) => !consumed.has(key));
        if (otherFields.length > 0) {
            sectionEntries.push(createConfigGroupEntry(
                '其他高级选项',
                otherFields,
                '',
                '未归类的新字段或低频字段；保留给高级调试使用。'
            ));
        }
        appendConfigGroupsByCategory(container, sectionEntries);
        updateLoKrFieldState();
        updateVeRAFieldState();
        updateDoRAFieldState();
        updateLossWeightingFieldState();
    }

    export function shouldRenderConfigSection(section, config = currentConfigState()) {
        if (!section?.method) return true;
        return activeMethodKey(config) === section.method;
    }

    export function shouldSkipConfigFormField(key, config = currentConfigState()) {
        if (CONFIG_FORM_MERGED_FIELDS?.has?.(key)) return true;
        if (DEPRECATED_CONFIG_FORM_FIELDS.has(key)) return true;
        if (RETIRED_CONFIG_FORM_FIELDS.has(key)) return true;
        if (key === 'mixed_precision' || key === 'full_fp16' || key === 'full_bf16') return true;
        const scopedFamilies = METHOD_SCOPED_CONFIG_FORM_FIELDS.get(key);
        if (!scopedFamilies) return false;
        return !scopedFamilies.has(activeMethodKey(config));
    }

    export function shouldExposeUiDefaultField(key, config, fieldsByKey = {}) {
        if (key in fieldsByKey) return true;
        if (NETWORK_ARG_FIELD_MAP.has(key)) return ALWAYS_VISIBLE_NETWORK_ARG_FIELDS.has(key);
        const family = activeMethodKey(config);
        if (SPD_UI_DEFAULT_FIELDS.has(key)) return family === 'spd';
        if (CHIMERA_UI_DEFAULT_FIELDS.has(key)) return family === 'chimera';
        if (IP_ADAPTER_UI_DEFAULT_FIELDS.has(key)) return family === 'ip_adapter';
        if (SOFT_TOKENS_UI_DEFAULT_FIELDS.has(key)) return family === 'soft_tokens';
        return true;
    }

    export function applyNetworkArgFields(fieldsByKey, config) {
        const specs = activeNetworkArgSpecs(config);
        if (!specs.length) return;
        const argMap = parseNetworkArgMap(config?.network_args);
        for (const spec of specs) {
            const rawValue = argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default;
            fieldsByKey[spec.key] = coerceNetworkArgValue(rawValue, spec);
        }
    }

    export function isActiveNetworkArgFieldKey(key, config = currentConfigState()) {
        return activeNetworkArgSpecs(config).some((spec) => spec.key === key);
    }

    export function collectSectionFields(fieldsByKey, orderedKeys, consumed) {
        const fields = [];
        for (const key of orderedKeys) {
            if (consumed.has(key) || !(key in fieldsByKey)) continue;
            fields.push([key, fieldsByKey[key]]);
            consumed.add(key);
        }
        return fields;
    }

    export function activeNetworkArgSpecs(config = currentConfigState()) {
        const families = activeNetworkArgFamilies(config);
        const argMap = parseNetworkArgMap(config?.network_args);
        return NETWORK_ARG_FIELD_SPECS.filter((spec) =>
            families.has(spec.family) || argMap.has(spec.arg)
        );
    }

    export function activeNetworkArgFamilies(config = currentConfigState()) {
        const families = new Set();
        const moduleName = String(config?.network_module || '');
        const method = activeMethodKey(config);
        if (method === 'soft_tokens' || moduleName.includes('soft_tokens')) families.add('soft_tokens');
        if (method === 'lokr' || isTruthy(config?.use_lokr)) families.add('lokr');
        if (method === 'ip_adapter' || isTruthy(config?.use_ip_adapter) || moduleName.includes('ip_adapter')) {
            families.add('ip_adapter');
        }
        if (method === 'easycontrol' || isTruthy(config?.use_easycontrol) || moduleName.includes('easycontrol')) {
            families.add('easycontrol');
        }
        return families;
    }
    export function parseNetworkArgMap(networkArgs) {
        const map = new Map();
        for (const raw of normalizeNetworkArgArray(networkArgs)) {
            const parsed = parseNetworkArgEntry(raw);
            if (parsed) map.set(parsed.arg, parsed.value);
        }
        return map;
    }

    export function normalizeNetworkArgArray(networkArgs) {
        if (Array.isArray(networkArgs)) return networkArgs.map((item) => String(item));
        if (typeof networkArgs === 'string' && networkArgs.trim()) return parseArrayValue(networkArgs).map((item) => String(item));
        return [];
    }

configureHistoryDetailBridge({
    ensureHistoryDetailFeature,
    getHistoryDetailFeature: () => historyDetailFeature,
    isHistoryReviewMode,
});

configureConfigFormBridge({
    syncConfigDraftFromForm,
    updateConfigDraftFromInput,
    originalConfigFieldValue,
    displayConfigFieldValue,
    configDraftValueChanged,
    isActiveNetworkArgFieldKey,
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
