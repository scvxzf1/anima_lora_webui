import { createPreviewFeature } from './preview/index.js?v=module-bootstrap-20260603-6';
import { createQueueFeature } from './queue/index.js?v=module-bootstrap-20260603-6';
import { createHistoryDetailFeature } from './history-detail/index.js?v=module-bootstrap-20260603-6';
import {
    formatSystemPercent,
    formatSystemTemperature,
    formatSystemVram,
    historySystemSummary,
} from './history-detail/system.js?v=module-bootstrap-20260603-6';
import { formatCompactNumber, numberOrNull } from './history-detail/ui.js?v=module-bootstrap-20260603-6';

function formatLossValue(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(5) : '-';
}

/**
 * Anima LoRA Web UI — legacy feature container.
 *
 * 这个文件是第一阶段拆分的过渡层：保留旧业务行为，同时通过 ctx 接入新模块化基础设施。
 * 后续 feature 继续从这里拆出 createXFeature(ctx)，不要把这里当成新的长期上帝文件。
 */
export function createLegacyApp(ctx) {
    const { MetricsChart } = ctx;

    (function () {
    'use strict';

    // ── 状态 ──
    let fieldHelp = {};
    let currentConfig = {};
    let ws = null;
    let lossChart = null;
    let stepCounter = 0;
    let tomlStatusTimer = null;
    let tomlFiles = [];
    let tomlFileGroups = [];
    let tomlFileMeta = {};
    let currentTomlFile = '';
    let tomlSavedContent = '';
    let tomlDeleteConfirmFile = '';
    let tomlDeleteConfirmTimer = null;
    let tomlSaveConfirmFile = '';
    let tomlSaveConfirmTimer = null;
    let tomlManagerMode = 'project';
    let configSwitchToastTimer = null;
    let sharedDialogBusy = false;
	    let tomlGroupActionBusy = false;
	    let fileGroupDragState = null;
	    let fileGroupPointerDrag = null;
	    let fileGroupDropPreviewElement = null;
	    let fileGroupActiveDropTargetNode = null;
	    let fileGroupActiveDropPosition = '';
	    let datasetEditorDragState = null;
	    let datasetEditorPointerDrag = null;
	    const fileGroupDropTargets = new WeakMap();
    const fileGroupDropTargetNodes = new Set();
    const FILE_GROUP_DROP_TARGET_ATTR = 'data-file-group-drop-target';
    let configLoadSeq = 0;
    let datasetLoadSeq = 0;
    let stepEstimateSeq = 0;
    let samplePromptsLoadSeq = 0;
    let datasetPresetLoadSeq = 0;
    let datasetPreviewLoadSeq = 0;
    let configGroupHintSeq = 0;
    const configFormState = {
        activeCategory: 'required',
        showAdvanced: false,
        search: '',
        expandedGroups: new Set(),
        collapsedGroups: new Set(),
        draftValues: new Map(),
    };
    const RESOURCE_QUICK_PRESETS = [
        {
            id: 'gpu_full',
            label: '全 GPU',
            note: '显存充足优先；最快，不做 block swap。',
            values: {
                blocks_to_swap: 0,
                block_swap_transfer_dtype: 'bf16',
                selective_checkpoint: 'off',
                block_swap_profile_jsonl: 'off',
                unsloth_offload_checkpointing: false,
                torch_compile: true,
            },
        },
        {
            id: 'balanced_16g',
            label: 'Balanced 16G',
            note: '推荐 16GB；约省 4GB，速度损失较低。',
            values: {
                blocks_to_swap: 12,
                block_swap_transfer_dtype: 'bf16',
                selective_checkpoint: 'off',
                block_swap_profile_jsonl: 'auto',
                unsloth_offload_checkpointing: false,
                torch_compile: true,
            },
        },
        {
            id: 'fp8_swap_test',
            label: 'FP8 测试',
            note: '实验项；压缩 frozen base block 传输，只用于对照测试。',
            values: {
                blocks_to_swap: 12,
                block_swap_transfer_dtype: 'fp8_e4m3',
                selective_checkpoint: 'off',
                block_swap_profile_jsonl: 'auto',
                unsloth_offload_checkpointing: false,
                torch_compile: true,
            },
        },
        {
            id: 'vram_saver',
            label: '更省显存',
            note: '交换 16 块；更省显存，训练会更慢。',
            values: {
                blocks_to_swap: 16,
                block_swap_transfer_dtype: 'bf16',
                selective_checkpoint: 'off',
                block_swap_profile_jsonl: 'auto',
                unsloth_offload_checkpointing: false,
                torch_compile: true,
            },
        },
        {
            id: 'oom_fallback',
            label: 'OOM 兜底',
            note: '仍然 OOM 时用；开启 mlp_only 重算。',
            values: {
                blocks_to_swap: 12,
                block_swap_transfer_dtype: 'bf16',
                selective_checkpoint: 'mlp_only',
                block_swap_profile_jsonl: 'auto',
                unsloth_offload_checkpointing: false,
                torch_compile: true,
            },
        },
    ];
    let datasetCaptionSourceHelpSeq = 0;
    let choiceGuideHintSeq = 0;
    const selectionSnapshot = {
        method: '',
        variant: '',
        preset: '',
    };
    let currentStepEstimate = null;
    let datasetEditorState = {
        loading: false,
        loaded: false,
        dirty: false,
        dataset_config: '',
        datasets: [],
        defaults: {},
        error: '',
    };
    let datasetPresetState = {
        loading: false,
        dirty: false,
        isNew: false,
        selectedFile: '',
        presets: [],
        groups: [],
        search: '',
        datasets: [],
        defaults: {},
        readonly: false,
        error: '',
        status: '',
    };
    const datasetPreviewState = {
        datasetIndex: 0,
        source: 'source',
        payload: null,
    };
    const HIDDEN_DATASET_PRESET_FILES = new Set([
        'configs/datasets/easycontrol.toml',
        'configs/datasets/ip_adapter.toml',
    ]);
    const DATASET_PRESET_GROUP_STATE_KEY = 'anima_lora_dataset_preset_groups_v2';
    let selectedConfigDatasetFile = '';
    let selectedConfigDatasetSummary = null;
    let outputRunState = {
        loading: false,
        runs: [],
        selectedRun: '',
        selectedKind: 'original',
        search: '',
        content: '',
        file: '',
        outputRoot: '',
        error: '',
        saveAsOpen: false,
    };
    let configDatasetPickerSearch = '';
    let configDatasetPreviewRequestSeq = 0;
    let configDatasetPreviewState = {
        file: '',
        loading: false,
        payload: null,
        error: '',
    };
    const DEFAULT_SAMPLE_PROMPTS_PATH = 'configs/sample_prompts.txt';
    let samplePromptsPath = DEFAULT_SAMPLE_PROMPTS_PATH;
    let samplePromptsContent = '';
    let samplePromptsMode = 'editor-inline';
    let viewingHistoryTaskId = '';
    let historyViewMode = 'live';
    let currentHistoryTaskForResume = null;
    let currentHistoryConfigGroup = null;
    let currentHistoryTimelineSelection = [];
    const STAGE_RESOLUTION_STEPS_PER_EPOCH = 1000;
    let stageResolutionState = {
        enabled: false,
        selectedIndex: 0,
        stages: [
            { name: 'EP1', epochs: 1, maxSide: 1024, downRange: 256, manualRepeats: false, repeats: 1 },
            { name: 'EP2', epochs: 1, maxSide: 1536, downRange: 512, manualRepeats: false, repeats: 1 },
        ],
    };
    let liveChartState = {
        showLr: true,
        rangeMode: 'all',
    };
    let continueTrainingSource = null;
    let continueLoraDialogState = {
        loading: false,
        taskId: '',
        weights: [],
        error: '',
        message: '',
    };
    let trainingViewMode = 'live';
    let historyTasks = [];
    let showArchivedHistory = false;
    let selectedHistoryTaskIds = new Set();
    let historyManagerFilters = {
        search: '',
        kind: 'all',
        state: 'all',
        archived: 'active',
        source: 'all',
        sort: 'newest',
    };
    let historyCollectionWorkbenchTarget = '';
    let historyCollectionSettings = {
        collection_order: [],
        config_group_order: {},
    };
    let historyCollectionSearch = '';
    let historyConfigGroupSearch = '';
    const HISTORY_UNGROUPED_COLLECTION_KEY = 'collection:__ungrouped__';
    let selectedHistoryCollectionKey = HISTORY_UNGROUPED_COLLECTION_KEY;
    let historyCurrentVisibleTaskIds = [];
    const HISTORY_TASK_DRAG_MIME = 'application/x-anima-history-task-ids';
    const HISTORY_COLLECTION_DRAG_MIME = 'application/x-anima-history-collection';
    const HISTORY_CONFIG_GROUP_DRAG_MIME = 'application/x-anima-history-config-group';
    let historyDragState = {
        active: false,
        taskIds: [],
        sourceGroupKey: '',
        activeDropTarget: '',
        pending: false,
        popover: {
            open: false,
            x: 0,
            y: 0,
            taskIds: [],
            defaultName: '',
        },
    };
    let historyCollectionDragState = {
        active: false,
        sourceValue: '',
        activeDropTarget: '',
        dropPosition: 'after',
        pending: false,
    };
    let historyConfigGroupSortState = {
        active: false,
        sourceKey: '',
        collectionKey: '',
        activeDropTarget: '',
        dropPosition: 'after',
        pending: false,
    };
    let historyConfigGroupPointerDrag = null;
    let historyCollectionPointerDrag = null;
    let historyDragImageElement = null;
    let historyConfigGroupDropPreviewElement = null;
    let historyDropPopoverOutsideHandler = null;
    let historyDropFeedback = { message: '', tone: '' };
    let historyDropFeedbackTimer = null;
    const THEME_STORAGE_KEY = 'anima_lora_theme';
    const GPU_WHITELIST_STORAGE_KEY = 'anima_lora_gpu_whitelist';
    let availableGpus = [];
    let selectedGpuWhitelist = [];
    let currentTrainingSource = {
        method: 'lora',
        methods_subdir: 'gui-methods',
        file: 'configs/gui-methods/lora.toml',
    };
    const {
        BLANK_PRESET_TEMPLATE_FILE,
        BLANK_PRESET_TEMPLATE_LABEL,
        FORM_UI_DEFAULTS,
        OPTIONAL_EMPTY_FIELDS,
        OPTIONAL_EMPTY_NUMBER_FIELDS,
        FORM_UI_PERSIST_DEFAULT_FIELDS,
        CONFIG_FORM_INTERNAL_KEYS,
        CONFIG_FORM_MERGED_FIELDS,
        DEPRECATED_CONFIG_FORM_FIELDS,
        RETIRED_CONFIG_FORM_FIELDS,
        METHOD_SCOPED_CONFIG_FORM_FIELDS,
        DATASET_EDITOR_COMPAT_FIELDS,
        DATASET_BLUEPRINT_FIELDS,
        DATASET_SETTING_KEYS,
        DEFAULT_NL_TAG_MIX,
        DEFAULT_TRIGGER_CLONE,
        CAPTION_SOURCE_MODE_OPTIONS,
        NETWORK_ARG_FIELD_SPECS,
        NETWORK_ARG_FIELD_MAP,
        NETWORK_ARG_SPEC_BY_ARG,
        SPD_UI_DEFAULT_FIELDS,
        CHIMERA_UI_DEFAULT_FIELDS,
        IP_ADAPTER_UI_DEFAULT_FIELDS,
        SOFT_TOKENS_UI_DEFAULT_FIELDS,
        MAX_LOG_LINES,
        GLOBAL_MODEL_PATH_FIELDS,
        GLOBAL_SETTING_INPUTS,
        FORM_SECTION_DEFS,
        FORM_CATEGORY_DEFS,
        STICKY_CONFIG_CATEGORY_IDS,
        ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS,
        FORM_CATEGORY_SECTION_MAP,
        CONFIG_COMPACT_FIELD_GROUPS,
        VARIANT_METHOD_FAMILY,
        EXTRA_FIELD_HELP_ZH,
        FIELD_LABEL_ZH,
        FIELD_OPTIONS,
        METHOD_GUIDE_ZH,
        VARIANT_GUIDE_ZH,
        PRESET_GUIDE_ZH,
        FIELD_HELP_ZH,
        choiceHelp,
        help,
    } = ctx.catalog;
    const datasetExperimentalScopeSelections = new Map();
    const trainingRuntime = {
        state: 'idle',
        variant: '',
        preset: '',
        methodsSubdir: '',
        job: '',
        lastOutputAt: 0,
        lastUiActivityAt: 0,
        lastGpuUtil: null,
        lastGpuTemp: null,
        lastVramUsedGb: null,
        lastVramTotalGb: null,
        peakGpuUtil: null,
        peakGpuTemp: null,
        peakVramUsedGb: null,
        quietHintShown: false,
        lastLogId: 0,
        logLineCount: 0,
        logBuffer: [],
        logFlushPending: false,
        outputDir: '',
        sampleDir: '',
        sampleConfig: null,
        runDir: '',
        runtimeConfigFile: '',
        originalConfigFile: '',
        datasetConfigFile: '',
        modelCacheDir: '',
        datasetCacheDir: '',
        trainingOutputDir: '',
        logsDir: '',
        progressCurrent: 0,
        progressTotal: 0,
        progressLabel: '',
        progressRate: '',
        progressSecondsPerStep: null,
        progressUpdatedAt: 0,
    };
    let globalSettings = null;
    let previewFeature = null;
    let queueFeature = null;
    let historyDetailFeature = null;
    function ensureQueueFeature() {
        if (queueFeature) return queueFeature;
        queueFeature = createQueueFeature(ctx, {
            appendLog,
            showAppConfirmDialog,
            setTomlStatus,
            currentTrainingConfigFile,
            getTomlManagerMode: () => tomlManagerMode,
            getOutputRunFile: () => outputRunState.file,
            getOutputRunSelectedRun: () => outputRunState.selectedRun,
            getCurrentTomlFile: () => currentTomlFile,
            hasPendingConfigChanges,
            updateTomlActionState,
            getCurrentTrainingSource: () => currentTrainingSource,
            isCliOnlySpdSource,
            hasContinueTrainingSource: () => Boolean(continueTrainingSource),
            continueTrainingSourceMessage: () => continueTrainingSource?.message || '',
            refreshContinueTrainingSourceCompatibility,
            currentTrainingConfigIsRuntime,
            renderPreflightPending,
            continueTrainingRequestPayload,
            showPreflightDialog,
            showPreflightRequestError,
            selectedGpuPayload,
            showTrainingView,
            getTrainingRuntime: () => trainingRuntime,
            renderTrainingViewMode,
            runLabelFromPath,
            getViewingHistoryTaskId: () => viewingHistoryTaskId,
            selectedResumeCheckpoint: () => ensureHistoryDetailFeature().selectedResumeCheckpoint(),
            setResumeStatus: (text, state = '') => ensureHistoryDetailFeature().setResumeStatus(text, state),
            historyTaskLabel,
            getCurrentHistoryTaskForResume: () => currentHistoryTaskForResume,
            showHistoryTaskConfirmDialog,
        });
        return queueFeature;
    }
    function ensurePreviewFeature() {
        if (previewFeature) return previewFeature;
        previewFeature = createPreviewFeature(ctx, {
            getHistoryTasks: () => historyTasks,
            getShowArchivedHistory: () => showArchivedHistory,
            loadTrainingHistoryList,
            loadHistoryTask,
            loadConfigGroupTimeline,
            showTrainingView,
            getTrainingViewMode: () => trainingViewMode,
            getTrainingRuntime: () => trainingRuntime,
            setTrainingSampleState: (value) => {
                if (value) trainingRuntime.sampleConfig = value;
            },
            historyTaskIsArchived,
            historyStateLabel,
            historyConfigGroupFromTask,
            canPreviewHistoryConfigGroup,
            configGroupLabel,
            selectContinueLoraWeight,
            renderDatasetImageDialogDetails,
        });
        return previewFeature;
    }

    function makeHistoryArtifactUrl(task, artifactKey, options = {}) {
        const taskId = String(task?.id || '').trim();
        const key = String(artifactKey || '').trim();
        if (!taskId || !key) return '#';
        const params = new URLSearchParams();
        if (options.download) params.set('download', '1');
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return `/api/training/history/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(key)}${suffix}`;
    }

    function ensureHistoryDetailFeature() {
        if (historyDetailFeature) return historyDetailFeature;
        historyDetailFeature = createHistoryDetailFeature(ctx, {
            setViewingHistoryTaskContext: ({
                taskId = '',
                viewMode = 'live',
                task = null,
                configGroup = null,
                timelineSelection = [],
            } = {}) => {
                viewingHistoryTaskId = taskId || '';
                historyViewMode = viewMode || 'live';
                currentHistoryTaskForResume = task || null;
                currentHistoryConfigGroup = configGroup || null;
                currentHistoryTimelineSelection = Array.isArray(timelineSelection) ? timelineSelection : [];
            },
            getViewingHistoryTaskId: () => viewingHistoryTaskId,
            getCurrentHistoryTaskForResume: () => currentHistoryTaskForResume,
            setCurrentHistoryTaskForResume: (task) => { currentHistoryTaskForResume = task || null; },
            renderTrainingHistoryList,
            renderHistoryManager,
            loadTrainingHistoryList,
            showTrainingView,
            returnToLiveTraining,
            clearViewingHistoryTaskContext,
            shouldRenderInlineResumePanel,
            getTrainingViewMode: () => trainingViewMode,
            getTrainingRuntime: () => trainingRuntime,
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
            selectedGpuPayload,
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
        });
        return historyDetailFeature;
    }
    // ── 初始化 ──
    document.addEventListener('DOMContentLoaded', async () => {
        initThemeToggle();
        setupTabs();
        lossChart = new MetricsChart(document.getElementById('loss-chart'), {
            emptyText: '',
            showLr: liveChartState.showLr,
            rangeMode: liveChartState.rangeMode,
        });
        lossChart.setTheme(chartTheme());
        resetLiveMetricPlaceholders();
        syncLossChartEmptyState();
        syncLiveChartControls();
        renderLiveChartPanel();
        setupEventListeners();
        initGpuPickerEvents();
        await loadInitialData();
        if (location.protocol !== 'file:') {
            connectWebSocket();
            pollStatus();
            setInterval(pollStatus, 10000);
            setInterval(refreshTrainingHealth, 1000);
        }
    });

    function currentTheme() {
        return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
    }

    function storedTheme() {
        try {
            return localStorage.getItem(THEME_STORAGE_KEY);
        } catch (_) {
            return null;
        }
    }

    function saveTheme(theme) {
        try {
            localStorage.setItem(THEME_STORAGE_KEY, theme);
        } catch (_) {
            // 忽略浏览器禁用本地存储的情况，当前页面仍然可以完成切换。
        }
    }

    function applyTheme(theme) {
        const safeTheme = theme === 'light' ? 'light' : 'dark';
        document.documentElement.dataset.theme = safeTheme;
        const toggle = document.getElementById('theme-toggle');
        const label = document.getElementById('theme-toggle-text');
        if (toggle) {
            const isLight = safeTheme === 'light';
            toggle.setAttribute('aria-pressed', String(isLight));
            toggle.title = isLight ? '切换到深色主题' : '切换到浅色主题';
        }
        if (label) label.textContent = safeTheme === 'light' ? '深色主题' : '浅色主题';
        lossChart?.setTheme?.(chartTheme());
    }

    function initThemeToggle() {
        applyTheme(storedTheme() || currentTheme());
        const toggle = document.getElementById('theme-toggle');
        if (!toggle) return;
        toggle.addEventListener('click', () => {
            const next = currentTheme() === 'light' ? 'dark' : 'light';
            applyTheme(next);
            saveTheme(next);
        });
    }

    function loadStoredGpuWhitelist() {
        try {
            const parsed = JSON.parse(localStorage.getItem(GPU_WHITELIST_STORAGE_KEY) || '[]');
            if (!Array.isArray(parsed)) return [];
            return parsed
                .map((item) => Number(item))
                .filter((item, index, list) => Number.isInteger(item) && item >= 0 && list.indexOf(item) === index);
        } catch (_) {
            return [];
        }
    }

    function saveGpuWhitelist() {
        try {
            localStorage.setItem(GPU_WHITELIST_STORAGE_KEY, JSON.stringify(selectedGpuWhitelist));
        } catch (_) {
            // 浏览器禁用 localStorage 时，本次页面内选择仍然有效。
        }
    }

    async function loadGpuOptions() {
        selectedGpuWhitelist = loadStoredGpuWhitelist();
        renderGpuPicker();
        if (location.protocol === 'file:') {
            updateGpuPickerNote('静态打开无法读取本机 GPU；选择会在服务模式下生效。');
            return;
        }
        try {
            const payload = await api('/api/training/gpus');
            availableGpus = Array.isArray(payload.gpus) ? payload.gpus : [];
            selectedGpuWhitelist = sanitizeGpuWhitelist(selectedGpuWhitelist);
            saveGpuWhitelist();
            renderGpuPicker();
        } catch (e) {
            availableGpus = [];
            renderGpuPicker();
            updateGpuPickerNote('读取 GPU 列表失败，训练会使用默认可见 GPU。');
        }
    }

    function sanitizeGpuWhitelist(list) {
        const selected = Array.isArray(list) ? list : [];
        if (!availableGpus.length) return selected.filter((item) => Number.isInteger(item) && item >= 0);
        const known = new Set(availableGpus.map((gpu) => Number(gpu.index)));
        return selected.filter((item) => known.has(item));
    }

    function renderGpuPicker() {
        const toggle = document.getElementById('gpu-picker-toggle');
        const list = document.getElementById('gpu-option-list');
        const allCheckbox = document.getElementById('gpu-all-checkbox');
        if (!toggle || !list || !allCheckbox) return;

        const selected = new Set(selectedGpuWhitelist);
        const allSelected = selectedGpuWhitelist.length === 0;
        allCheckbox.checked = allSelected;
        allCheckbox.indeterminate = false;
        allCheckbox.disabled = allSelected;
        toggle.textContent = gpuPickerSummary();
        toggle.title = gpuPickerTitle();
        list.innerHTML = '';

        if (!availableGpus.length) {
            const empty = document.createElement('div');
            empty.className = 'gpu-picker-note';
            empty.textContent = '未读取到 NVIDIA GPU；保持“全部 GPU”时会沿用系统默认可见设备。';
            list.appendChild(empty);
            updateGpuPickerNote('选择为空表示不限制 GPU，训练使用系统默认可见设备。');
            return;
        }

        for (const gpu of availableGpus) {
            const index = Number(gpu.index);
            const option = document.createElement('label');
            option.className = 'gpu-option';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = String(index);
            checkbox.checked = selected.has(index);
            checkbox.addEventListener('change', () => toggleGpuSelection(index, checkbox.checked));

            const body = document.createElement('span');
            const name = document.createElement('span');
            name.className = 'gpu-option-name';
            name.textContent = gpu.label || `GPU ${index} · ${gpu.name || '未命名显卡'}`;
            body.appendChild(name);
            const meta = document.createElement('span');
            meta.className = 'gpu-option-meta';
            meta.textContent = gpu.memory_total_gb
                ? `显存 ${gpu.memory_total_gb} GB · 训练时写入 CUDA_VISIBLE_DEVICES=${index}`
                : `训练时写入 CUDA_VISIBLE_DEVICES=${index}`;
            body.appendChild(meta);
            option.append(checkbox, body);
            list.appendChild(option);
        }

        updateGpuPickerNote(allSelected
            ? '当前不限制 GPU，训练会使用系统默认可见设备。'
            : `当前训练白名单: ${selectedGpuWhitelist.join(', ')}`);
    }

    function gpuPickerSummary() {
        if (!selectedGpuWhitelist.length) return '全部 GPU';
        const names = selectedGpuWhitelist.map((index) => {
            const gpu = availableGpus.find((item) => Number(item.index) === Number(index));
            return gpu?.name ? `GPU ${index} · ${gpu.name}` : `GPU ${index}`;
        });
        if (names.length <= 2) return names.join(' / ');
        return `${names.slice(0, 2).join(' / ')} 等 ${names.length} 张`;
    }

    function gpuPickerTitle() {
        return [
            '选择训练时允许使用的 GPU 白名单。',
            '留空/全部 GPU 表示不覆盖系统默认可见设备。',
            '选择会保存在本机浏览器，并在开始训练或自动预处理后训练时生效。',
        ].join('\n');
    }

    function updateGpuPickerNote(text) {
        const note = document.getElementById('gpu-picker-note');
        if (note) note.textContent = text;
    }

    function setGpuWhitelist(next) {
        selectedGpuWhitelist = sanitizeGpuWhitelist(next);
        saveGpuWhitelist();
        renderGpuPicker();
    }

    function toggleGpuSelection(index, checked) {
        const selected = new Set(selectedGpuWhitelist);
        if (checked) selected.add(index);
        else selected.delete(index);
        setGpuWhitelist([...selected].sort((a, b) => a - b));
    }

    function selectedGpuPayload() {
        return selectedGpuWhitelist.slice().sort((a, b) => a - b);
    }

    function closeGpuPickerPanel() {
        const panel = document.getElementById('gpu-picker-panel');
        const toggle = document.getElementById('gpu-picker-toggle');
        if (!panel || !toggle) return;
        panel.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
    }

    function initGpuPickerEvents() {
        const picker = document.getElementById('gpu-picker');
        const toggle = document.getElementById('gpu-picker-toggle');
        const panel = document.getElementById('gpu-picker-panel');
        const allCheckbox = document.getElementById('gpu-all-checkbox');
        if (!picker || !toggle || !panel || !allCheckbox) return;
        toggle.addEventListener('click', () => {
            const nextOpen = panel.hidden;
            panel.hidden = !nextOpen;
            toggle.setAttribute('aria-expanded', String(nextOpen));
        });
        allCheckbox.addEventListener('change', () => setGpuWhitelist([]));
        document.addEventListener('click', (event) => {
            if (!picker.contains(event.target)) closeGpuPickerPanel();
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') closeGpuPickerPanel();
        });
    }

    function chartTheme() {
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

    function isHistoryReviewMode() {
        return historyViewMode !== 'live';
    }

    function openTutorialDialog() {
        const dialog = document.getElementById('tutorial-dialog');
        if (!dialog) return;
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
    }

    // ── Tab 切换 ──
    function normalizeTopLevelTabState() {
        const activeButton = document.querySelector('.tab-btn.active');
        const activeName = activeButton?.dataset.tab || '';
        const hasUsableActiveTab =
            activeName &&
            activeName !== 'preview' &&
            document.getElementById(`tab-${activeName}`);
        const fallbackButton = document.querySelector('[data-tab="training"]') || document.querySelector('[data-tab="config"]');
        const nextButton = hasUsableActiveTab ? activeButton : fallbackButton;
        const nextName = nextButton?.dataset.tab || '';
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn === nextButton);
        });
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.toggle('active', tab.id === `tab-${nextName}`);
        });
        document.getElementById('tab-preview')?.classList.remove('active');
    }

    function setupTabs() {
        normalizeTopLevelTabState();
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const previousTab = document.querySelector('.tab-btn.active')?.dataset.tab || '';
                const nextTab = btn.dataset.tab || '';
                if (previousTab === 'training' && nextTab !== 'training') {
                    resetTrainingExpandedStateOnLeave();
                }
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('tab-' + nextTab).classList.add('active');
                if (nextTab === 'datasets') {
                    loadDatasetPresets({ manage: true });
                }
                if (nextTab === 'training' && lossChart?.resize) {
                    lossChart.resize();
                }
                if (nextTab === 'settings') {
                    loadGlobalSettings();
                }
            });
        });
    }

    // ── 加载初始数据 ──
    async function loadInitialData() {
        if (location.protocol === 'file:') {
            await loadGpuOptions();
            showStandaloneWarning();
            return;
        }
        try {
            const [methods, presets, help] = await Promise.all([
                api('/api/methods'),
                api('/api/presets'),
                api('/api/config/field-help'),
            ]);
            fieldHelp = help;
            populateSelect('method-select', methods, 'lora');
            populateSelect('preset-select', presets, 'default');
            await loadGpuOptions();
            const variants = await loadVariants();
            await loadDatasetPresets({ selectCurrent: false, manage: isDatasetTabActive() });
            if (variants.length) {
                await loadConfig();
            }
            await loadTomlFileList();
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

    function showStandaloneWarning() {
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

    async function loadVariants({ reset = false } = {}) {
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

    async function loadConfig() {
        const requestSeq = ++configLoadSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        if (!variant) return;
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const data = await api(`/api/config/merged?variant=${encodeURIComponent(variant)}&preset=${encodeURIComponent(preset)}&methods_subdir=${encodeURIComponent(methodsSubdir)}`);
        if (requestSeq !== configLoadSeq) return;
        if (data?.ok === false) {
            setTomlStatus('error', data.error || '读取配置失败');
            return;
        }
        resetConfigFormDraft();
        currentConfig = data;
        selectedConfigDatasetFile = currentConfig.dataset_config || '';
        selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
        renderConfigForm(currentConfig);
        renderContinueTrainingSource();
        if (continueTrainingSource?.abs_path) {
            await refreshContinueTrainingSourceCompatibility();
        }
        if (samplePromptsMode === 'editor-file') {
            loadSamplePrompts(samplePromptsPath, requestSeq);
        } else {
            samplePromptsLoadSeq += 1;
        }
        loadStepEstimate(requestSeq);
        updateChoiceGuide();
        updateTomlActionState(currentTomlFile);
        // 同步加载对应的 TOML 文件到右侧编辑器
        const tomlFile = currentTrainingSource.file || `configs/${methodsSubdir}/${variant}.toml`;
        if (tomlFiles.includes(tomlFile) && currentTomlFile !== tomlFile) {
            loadTomlFile(tomlFile, { force: true });
        }
    }

    async function reloadCurrentConfig() {
        if (!(await confirmDiscardTomlChanges('当前配置有未保存修改，刷新会重新读取表单和数据集设置并丢弃这些修改。是否继续？'))) {
            return;
        }
        await loadConfig();
        rememberSelectionSnapshot();
    }

    // ── 配置表单渲染 ──
    function resetConfigFormDraft() {
        configFormState.draftValues.clear();
    }

    function syncConfigDraftFromForm(options = {}) {
        document.querySelectorAll('#config-form .field-input[data-key]').forEach((input) => {
            updateConfigDraftFromInput(input, options);
        });
    }

    function updateConfigDraftFromInput(input, options = {}) {
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

    function originalConfigFieldValue(key) {
        if (key === 'sample_prompts' && samplePromptsMode !== 'path') {
            return samplePromptsContent || '';
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

    function displayConfigFieldValue(key, value) {
        if (key === 'lora_adapter_kind') {
            return configFormState.draftValues.has(key)
                ? configFormState.draftValues.get(key)
                : loraAdapterKindFromConfig(currentConfig);
        }
        return configFormState.draftValues.has(key)
            ? configFormState.draftValues.get(key)
            : value;
    }

    function configDraftValueChanged(key, next, original = originalConfigFieldValue(key), options = {}) {
        if (key === 'sample_prompts' && samplePromptsMode !== 'path') {
            return String(next || '') !== String(samplePromptsContent || '');
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

    function renderConfigForm(config) {
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
                    section.description || ''
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
    }

    function shouldRenderConfigSection(section, config = currentConfig) {
        if (!section?.method) return true;
        return activeMethodKey(config) === section.method;
    }

    function shouldSkipConfigFormField(key, config = currentConfig) {
        if (CONFIG_FORM_MERGED_FIELDS?.has?.(key)) return true;
        if (DEPRECATED_CONFIG_FORM_FIELDS.has(key)) return true;
        if (RETIRED_CONFIG_FORM_FIELDS.has(key)) return true;
        const scopedFamilies = METHOD_SCOPED_CONFIG_FORM_FIELDS.get(key);
        if (!scopedFamilies) return false;
        return !scopedFamilies.has(activeMethodKey(config));
    }

    function shouldExposeUiDefaultField(key, config, fieldsByKey = {}) {
        if (key in fieldsByKey) return true;
        if (NETWORK_ARG_FIELD_MAP.has(key)) return false;
        const family = activeMethodKey(config);
        if (SPD_UI_DEFAULT_FIELDS.has(key)) return family === 'spd';
        if (CHIMERA_UI_DEFAULT_FIELDS.has(key)) return family === 'chimera';
        if (IP_ADAPTER_UI_DEFAULT_FIELDS.has(key)) return family === 'ip_adapter';
        if (SOFT_TOKENS_UI_DEFAULT_FIELDS.has(key)) return family === 'soft_tokens';
        return true;
    }

    function applyNetworkArgFields(fieldsByKey, config) {
        const specs = activeNetworkArgSpecs(config);
        if (!specs.length) return;
        const argMap = parseNetworkArgMap(config?.network_args);
        for (const spec of specs) {
            const rawValue = argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default;
            fieldsByKey[spec.key] = coerceNetworkArgValue(rawValue, spec);
        }
    }

    function isActiveNetworkArgFieldKey(key, config = currentConfig) {
        return activeNetworkArgSpecs(config).some((spec) => spec.key === key);
    }

    function collectSectionFields(fieldsByKey, orderedKeys, consumed) {
        const fields = [];
        for (const key of orderedKeys) {
            if (consumed.has(key) || !(key in fieldsByKey)) continue;
            fields.push([key, fieldsByKey[key]]);
            consumed.add(key);
        }
        return fields;
    }

    function activeNetworkArgSpecs(config = currentConfig) {
        const families = activeNetworkArgFamilies(config);
        const argMap = parseNetworkArgMap(config?.network_args);
        return NETWORK_ARG_FIELD_SPECS.filter((spec) =>
            families.has(spec.family) || argMap.has(spec.arg)
        );
    }

    function activeNetworkArgFamilies(config = currentConfig) {
        const families = new Set();
        const moduleName = String(config?.network_module || '');
        const method = activeMethodKey(config);
        if (method === 'soft_tokens' || moduleName.includes('soft_tokens')) families.add('soft_tokens');
        if (method === 'ip_adapter' || isTruthy(config?.use_ip_adapter) || moduleName.includes('ip_adapter')) {
            families.add('ip_adapter');
        }
        if (method === 'easycontrol' || isTruthy(config?.use_easycontrol) || moduleName.includes('easycontrol')) {
            families.add('easycontrol');
        }
        return families;
    }

    function parseNetworkArgMap(networkArgs) {
        const map = new Map();
        for (const raw of normalizeNetworkArgArray(networkArgs)) {
            const parsed = parseNetworkArgEntry(raw);
            if (parsed) map.set(parsed.arg, parsed.value);
        }
        return map;
    }

    function normalizeNetworkArgArray(networkArgs) {
        if (Array.isArray(networkArgs)) return networkArgs.map((item) => String(item));
        if (typeof networkArgs === 'string' && networkArgs.trim()) return parseArrayValue(networkArgs).map((item) => String(item));
        return [];
    }

    function parseNetworkArgEntry(raw) {
        const text = String(raw || '').trim();
        const splitAt = text.indexOf('=');
        if (splitAt <= 0) return null;
        const arg = text.slice(0, splitAt).trim();
        if (!arg) return null;
        return {
            arg,
            value: stripNetworkArgQuotes(text.slice(splitAt + 1).trim()),
            raw: text,
        };
    }

    function stripNetworkArgQuotes(value) {
        const text = String(value || '').trim();
        if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
            return text.slice(1, -1);
        }
        return text;
    }

    function coerceNetworkArgValue(value, spec) {
        if (spec.valueType === 'boolean' || spec.valueType === 'booleanInt') {
            return parseBooleanNetworkArg(value, spec.default);
        }
        if (spec.valueType === 'integer') {
            const n = Number(value);
            return Number.isFinite(n) ? Math.trunc(n) : spec.default;
        }
        if (spec.valueType === 'number') {
            const n = Number(value);
            return Number.isFinite(n) ? n : spec.default;
        }
        return String(value ?? spec.default ?? '');
    }

    function parseBooleanNetworkArg(value, fallback = false) {
        if (typeof value === 'boolean') return value;
        if (value === 1 || value === 0) return Boolean(value);
        const text = String(value ?? '').trim().toLowerCase();
        if (['1', 'true', 'yes', 'on'].includes(text)) return true;
        if (['0', 'false', 'no', 'off'].includes(text)) return false;
        return Boolean(fallback);
    }

    async function loadStepEstimate(parentSeq = configLoadSeq) {
        const requestSeq = ++stepEstimateSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant || location.protocol === 'file:') return;
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            currentStepEstimate = null;
            updateStepEstimatePanel();
            return;
        }
        try {
            const datasetParam = selectedConfigDatasetFile ? `&dataset_config=${encodeURIComponent(selectedConfigDatasetFile)}` : '';
            const data = await api(`/api/config/steps?variant=${encodeURIComponent(variant)}&preset=${encodeURIComponent(preset)}&methods_subdir=${encodeURIComponent(methodsSubdir)}${datasetParam}`);
            if (parentSeq !== configLoadSeq || requestSeq !== stepEstimateSeq) return;
            currentStepEstimate = data?.ok === false ? null : data;
        } catch {
            if (parentSeq !== configLoadSeq || requestSeq !== stepEstimateSeq) return;
            currentStepEstimate = null;
        }
        updateStepEstimatePanel();
    }

    async function loadDatasetEditor(parentSeq = configLoadSeq) {
        const requestSeq = ++datasetLoadSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant || location.protocol === 'file:') return;
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            datasetEditorState = {
                ...datasetEditorState,
                loading: false,
                loaded: false,
                error: 'SPD 是 CLI 实验配置，不使用 Web 数据集编辑器。',
            };
            renderDatasetEditor();
            return;
        }
        datasetEditorState.loading = true;
        datasetEditorState.error = '';
        renderDatasetEditor();
        try {
            const data = await api(`/api/config/datasets?variant=${encodeURIComponent(variant)}&preset=${encodeURIComponent(preset)}&methods_subdir=${encodeURIComponent(methodsSubdir)}`);
            if (parentSeq !== configLoadSeq || requestSeq !== datasetLoadSeq) return;
            if (!data.ok) {
                throw new Error(data.error || '读取数据集配置失败');
            }
            datasetEditorState = {
                loading: false,
                loaded: true,
                dirty: false,
                dataset_config: data.dataset_config || '',
                datasets: normalizeDatasetEditorRows(data.datasets || []),
                defaults: normalizeDatasetDefaults(data.defaults || {}),
                error: '',
            };
        } catch (e) {
            if (parentSeq !== configLoadSeq || requestSeq !== datasetLoadSeq) return;
            datasetEditorState = {
                ...datasetEditorState,
                loading: false,
                loaded: false,
                defaults: normalizeDatasetDefaults(datasetEditorState.defaults || {}),
                error: e.message || '读取数据集配置失败',
            };
        }
        renderDatasetEditor();
    }

    async function loadDatasetPresets(options = {}) {
        if (location.protocol === 'file:') return;
        const requestSeq = ++datasetPresetLoadSeq;
        const managePresets = options.manage === true || (options.manage !== false && isDatasetTabActive());
        if (managePresets) {
            datasetPresetState.loading = true;
            renderDatasetPresetList();
        }
        try {
            const data = await api('/api/config/dataset-presets');
            if (requestSeq !== datasetPresetLoadSeq) return;
            if (!data.ok) throw new Error(data.error || '读取数据集预设失败');
            const presets = (Array.isArray(data.presets) ? data.presets : [])
                .filter((preset) => !HIDDEN_DATASET_PRESET_FILES.has(preset.path));
            const presetPaths = new Set(presets.map((preset) => preset.path));
            const groups = (Array.isArray(data.groups) ? data.groups : [])
                .map((group) => ({
                    ...group,
                    files: (Array.isArray(group.files) ? group.files : [])
                        .filter((preset) => presetPaths.has(preset.path) && !HIDDEN_DATASET_PRESET_FILES.has(preset.path)),
                }))
                .filter((group) => group.kind === 'dataset' || group.files.length);
            const sortedGroups = sortDatasetPresetGroups(groups);
            datasetPresetState.presets = orderDatasetPresetsForGroups(presets, sortedGroups);
            datasetPresetState.groups = sortedGroups;
            if (managePresets) {
                datasetPresetState.loading = false;
            }
            datasetPresetState.error = '';
            selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
            renderConfigDatasetPicker();
            if (!managePresets) {
                if (isConfigDatasetPickerDialogOpen()) {
                    renderConfigDatasetPickerDialog();
                }
                return;
            }
            const preserveDirtySelection = datasetPresetState.dirty;
            const selectedDatasetVisible = presets.some((preset) => preset.path === datasetPresetState.selectedFile);
            if (!selectedDatasetVisible && !preserveDirtySelection) {
                datasetPresetState.selectedFile = '';
            }
            if (!preserveDirtySelection && options.selectCurrent !== false && selectedConfigDatasetFile && !datasetPresetState.selectedFile && presets.some((preset) => preset.path === selectedConfigDatasetFile)) {
                datasetPresetState.selectedFile = selectedConfigDatasetFile;
            }
            if (!preserveDirtySelection && !datasetPresetState.selectedFile && presets.length) {
                datasetPresetState.selectedFile = presets[0].path;
            }
            renderDatasetPresetList();
            renderDatasetPresetHeader();
            if (datasetPresetState.selectedFile && !datasetPresetState.dirty) {
                await loadDatasetPreset(datasetPresetState.selectedFile);
            } else {
                renderDatasetEditor();
            }
        } catch (e) {
            if (requestSeq !== datasetPresetLoadSeq) return;
            if (managePresets) {
                datasetPresetState.loading = false;
            }
            datasetPresetState.error = e.message || '读取数据集预设失败';
            if (managePresets) {
                renderDatasetPresetList();
                renderDatasetPresetHeader();
            } else {
                renderConfigDatasetPicker();
                if (isConfigDatasetPickerDialogOpen()) {
                    renderConfigDatasetPickerDialog();
                }
            }
        }
    }

    async function loadDatasetPreset(file) {
        if (!file) return;
        if (datasetPresetState.dirty && !(await confirmUnsavedDiscard('当前数据集预设有未保存修改，切换会丢弃这些修改。是否继续？'))) {
            renderDatasetPresetList();
            return;
        }
        datasetPresetState.selectedFile = file;
        datasetPresetState.loading = true;
        datasetPresetState.error = '';
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
        try {
            const data = await api(`/api/config/dataset-presets/read?file=${encodeURIComponent(file)}`);
            if (!data.ok) throw new Error(data.error || '读取数据集预设失败');
            datasetPresetState = {
                ...datasetPresetState,
                loading: false,
                dirty: false,
                isNew: false,
                selectedFile: data.file || file,
                datasets: normalizeDatasetEditorRows(data.datasets || []),
                defaults: normalizeDatasetDefaults(data.defaults || {}),
                readonly: Boolean(data.readonly || data.meta?.locked),
                error: '',
                status: '',
            };
        } catch (e) {
            datasetPresetState = {
                ...datasetPresetState,
                loading: false,
                error: e.message || '读取数据集预设失败',
            };
        }
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
    }

    function createStepEstimatePanel() {
        const panel = document.createElement('div');
        panel.id = 'step-estimate-panel';
        panel.className = 'step-estimate-panel';
        panel.innerHTML = [
            '<div class="step-estimate-title">预计训练步数</div>',
            '<div class="step-estimate-grid">',
            '<div><span>数据集</span><strong id="step-dataset-count">-</strong></div>',
            '<div><span>训练图片</span><strong id="step-train-images">-</strong></div>',
            '<div><span>重复后样本</span><strong id="step-repeated-images">-</strong></div>',
            '<div><span>有效批大小</span><strong id="step-effective-batch">-</strong></div>',
            '<div><span>每轮步数</span><strong id="step-per-epoch">-</strong></div>',
            '<div><span>最大训练轮数</span><strong id="step-max-train-epochs">-</strong></div>',
            '<div><span>总步数</span><strong id="step-total">-</strong></div>',
            '</div>',
            '<div id="step-dataset-breakdown" class="step-dataset-breakdown"></div>',
            '<p id="step-estimate-note" class="step-estimate-note"></p>',
        ].join('');
        return panel;
    }

    function updateStepEstimatePanel() {
        const panel = document.getElementById('step-estimate-panel');
        if (!panel || !currentStepEstimate) return;

        const epochs = readOptionalLiveNumber('max_train_epochs');
        const batchSize = readLiveNumber('train_batch_size', currentStepEstimate.train_batch_size || 1);
        const gradAccum = readLiveNumber('gradient_accumulation_steps', currentStepEstimate.gradient_accumulation_steps || 1);
        const sampleRatio = readLiveNumber('sample_ratio', currentStepEstimate.sample_ratio || 1);
        const maxTrainSteps = readNonnegativeLiveNumber('max_train_steps', currentStepEstimate.max_train_steps ?? 0);
        const datasets = liveDatasetRowsForEstimate();
        const trainImages = datasets.reduce((sum, row) => sum + Number(row.train_image_count || 0), 0);
        const weightedImages = datasets.reduce((sum, row) => sum + (Number(row.train_image_count || 0) * Number(row.num_repeats || 1)), 0);
        const effectiveBatch = Math.max(1, batchSize * gradAccum);
        const repeatedImages = Math.max(0, Math.floor(weightedImages * sampleRatio));
        const stepsPerEpoch = repeatedImages ? Math.ceil(repeatedImages / effectiveBatch) : 0;
        const durationMode = epochs ? 'epochs' : (maxTrainSteps > 0 ? 'steps' : 'unset');
        const totalSteps = durationMode === 'epochs' ? stepsPerEpoch * epochs : maxTrainSteps;

        setText('step-dataset-count', String(datasets.length || 0));
        setText('step-train-images', String(trainImages));
        setText('step-repeated-images', `${repeatedImages} = ${weightedImages} x ${sampleRatio}`);
        setText('step-effective-batch', `${effectiveBatch} = ${batchSize} x ${gradAccum}`);
        setText('step-per-epoch', String(stepsPerEpoch));
        setText('step-max-train-epochs', durationMode === 'epochs' ? String(epochs) : '未设置');
        const totalLabel = durationMode === 'epochs'
            ? `${totalSteps} = ${stepsPerEpoch} x ${epochs}`
            : (durationMode === 'steps' ? `${totalSteps} = max_train_steps` : '未配置');
        setText('step-total', totalLabel);
        renderStepDatasetBreakdown(datasets);
        const note = durationMode === 'epochs'
            ? `公式: 向上取整(重复后样本 / 有效批大小) = 每轮步数；每轮步数 x max_train_epochs(${epochs}) = 总步数。max_train_epochs 已设置，max_train_steps 此时不生效。`
            : (durationMode === 'steps'
                ? `当前未设置 max_train_epochs，训练将直接按 max_train_steps=${maxTrainSteps} 作为固定总步数运行。若填写 epoch，则会按每轮步数重新推导总步数。`
                : `当前未设置 max_train_epochs，且 max_train_steps=0 表示不启用固定步数。启动训练前需要设置最大训练轮数，或把最大训练步数填成正数。`);
        setText('step-estimate-note', note);
    }

    function liveDatasetRowsForEstimate() {
        const baseRows = Array.isArray(currentStepEstimate?.datasets) ? currentStepEstimate.datasets : [];
        return baseRows.length ? baseRows : [{
            index: 1,
            source_dir: currentStepEstimate?.source_dir || '',
            image_dir: currentStepEstimate?.resized_dir || '',
            cache_dir: currentStepEstimate?.lora_cache_dir || '',
            source_image_count: currentStepEstimate?.source_image_count || 0,
            resized_image_count: currentStepEstimate?.resized_image_count || 0,
            train_image_count: currentStepEstimate?.train_image_count || 0,
            num_repeats: currentStepEstimate?.dataset_num_repeats || 1,
            weighted_image_count: currentStepEstimate?.weighted_image_count || 0,
            uses_preprocessed_images: currentStepEstimate?.uses_preprocessed_images || false,
        }];
    }

    function renderStepDatasetBreakdown(datasets) {
        const container = document.getElementById('step-dataset-breakdown');
        if (!container) return;
        container.innerHTML = '';
        if (!datasets.length) {
            const empty = document.createElement('div');
            empty.className = 'step-dataset-row muted';
            empty.textContent = '还没有可估算的数据集。';
            container.appendChild(empty);
            return;
        }
        for (const row of datasets) {
            const item = document.createElement('div');
            item.className = 'step-dataset-row';
            const trainCount = Number(row.train_image_count || 0);
            const repeats = Number(row.num_repeats || 1);
            const weighted = trainCount * repeats;
            const source = row.uses_preprocessed_images ? '缩放图' : '原始图';
            item.innerHTML = [
                `<strong>第 ${row.index || 1} 组</strong>`,
                `<span>${source} ${trainCount} 张 x 重复 ${repeats} = ${weighted} 样本</span>`,
                `<code>${escapeHtml(row.source_dir || row.image_dir || '-')}</code>`,
            ].join('');
            container.appendChild(item);
        }
    }

    function readLiveNumber(key, fallback) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return Number(fallback) || 0;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const n = Number(raw);
        return Number.isFinite(n) && n > 0 ? n : (Number(fallback) || 0);
    }

    function readNonnegativeLiveNumber(key, fallback = 0) {
        const fallbackNumber = Math.max(0, Number(fallback) || 0);
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return fallbackNumber;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const trimmed = String(raw).trim();
        if (!trimmed) return fallbackNumber;
        const n = Number(trimmed);
        return Number.isFinite(n) && n >= 0 ? n : fallbackNumber;
    }

    function readOptionalLiveNumber(key) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return null;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const trimmed = String(raw).trim();
        if (!trimmed) return null;
        const n = Number(trimmed);
        return Number.isFinite(n) && n > 0 ? n : null;
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = text;
        if (el.classList.contains('metric-value')) {
            const empty = metricValueIsEmpty(text);
            el.classList.toggle('metric-empty', empty);
            el.closest('.metric-item')?.classList.toggle('is-empty', empty);
        }
    }

    function metricValueIsEmpty(value) {
        const text = String(value ?? '').trim();
        return !text || text === '-' || text.toUpperCase() === 'N/A';
    }

    function setMetricText(id, value) {
        const text = metricValueIsEmpty(value) ? 'N/A' : String(value);
        setText(id, text);
    }

    function setEtaMetricText(info = {}) {
        const el = document.getElementById('metric-eta');
        if (!el) return;
        const text = String(info.text || '').trim() || '待计算';
        el.textContent = text;
        el.title = info.title || '';
        const empty = info.empty !== undefined ? Boolean(info.empty) : (text === '待计算' || metricValueIsEmpty(text));
        el.classList.toggle('metric-empty', empty);
        el.closest('.metric-item')?.classList.toggle('is-empty', empty);
    }

    function resetLiveMetricPlaceholders(options = {}) {
        const includePrimary = options.primary !== false;
        const ids = [
            ...(includePrimary ? ['metric-loss', 'metric-lr', 'metric-step', 'metric-rate'] : ['metric-rate']),
            'metric-vram',
            'metric-vram-peak',
            'metric-gpu',
            'metric-gpu-peak',
            'metric-temp',
            'metric-temp-peak',
            'metric-log-age',
        ];
        ids.forEach((id) => setMetricText(id, 'N/A'));
        setEtaMetricText({ text: '待计算', empty: true, title: '需要进度总数和速度后计算预计完成时间。' });
    }

    function updateDashboardProgressIdleState(active = null) {
        const wrap = document.querySelector('#tab-training .training-dashboard-progress');
        const text = document.getElementById('progress-text');
        if (!wrap) return;
        const hasProgress = active !== null
            ? Boolean(active)
            : Number(trainingRuntime.progressTotal || 0) > 0;
        wrap.classList.toggle('is-idle', !hasProgress);
        if (!hasProgress && text) {
            text.textContent = '暂无正在运行的任务目录...';
        }
    }

    function syncLossChartEmptyState() {
        const shell = document.getElementById('loss-chart-shell');
        if (!shell) return;
        const pointCount = Array.isArray(lossChart?.data) ? lossChart.data.length : 0;
        shell.classList.toggle('is-empty', pointCount < 2);
        renderLiveChartPanel();
    }

    function syncLiveChartControls() {
        const lrToggle = document.getElementById('live-chart-toggle-lr');
        if (lrToggle) lrToggle.checked = liveChartState.showLr;
        const rangeSelect = document.getElementById('live-chart-range');
        if (rangeSelect) rangeSelect.value = liveChartState.rangeMode;
    }

    function liveChartVisiblePoints(points = []) {
        const all = Array.isArray(points) ? points : [];
        const match = String(liveChartState.rangeMode || 'all').match(/^last(\d+)$/);
        if (!match) return all;
        const count = Number(match[1]);
        return Number.isFinite(count) && count > 0 ? all.slice(-count) : all;
    }

    function renderLiveChartPanel() {
        const points = Array.isArray(lossChart?.data) ? lossChart.data : [];
        lossChart?.setDisplayOptions?.({
            showLr: liveChartState.showLr,
            rangeMode: liveChartState.rangeMode,
        });
        const visible = liveChartVisiblePoints(points);
        const latest = visible[visible.length - 1] || null;
        const latestLr = [...visible].reverse().find((point) => numberOrNull(point.lr) !== null) || null;
        setLiveChartStat('live-chart-stat-loss', latest ? formatLossValue(latest.value) : 'N/A');
        setLiveChartStat('live-chart-stat-lr', latestLr ? formatLr(latestLr.lr) : 'N/A');
        setLiveChartStat('live-chart-stat-points', visible.length ? `${visible.length}/${points.length}` : '0', !visible.length);
        setLiveChartStat('live-chart-stat-range', liveChartStepRangeText(visible), !visible.length);
        const lrLegend = document.getElementById('live-chart-lr-legend');
        if (lrLegend) {
            lrLegend.classList.toggle('muted', !liveChartState.showLr || !latestLr);
        }
    }

    function setLiveChartStat(id, value, empty = null) {
        const el = document.getElementById(id);
        if (!el) return;
        const text = metricValueIsEmpty(value) ? 'N/A' : String(value);
        el.textContent = text;
        const isEmpty = empty === null ? metricValueIsEmpty(text) : Boolean(empty);
        el.closest('.live-chart-stat')?.classList.toggle('is-empty', isEmpty);
    }

    function liveChartStepRangeText(points = []) {
        if (!points.length) return 'N/A';
        const first = points[0]?.step;
        const last = points[points.length - 1]?.step;
        return `${formatStepLabel(first)} - ${formatStepLabel(last)}`;
    }

    function formatStepLabel(value) {
        const number = Number(value);
        return Number.isFinite(number) ? String(Math.round(number)) : '-';
    }

    function updateTrainingToolbarState(state, label) {
        const safeState = state || 'idle';
        const stateEl = document.getElementById('training-toolbar-state');
        const textEl = document.getElementById('training-toolbar-state-text');
        if (stateEl) stateEl.className = `training-toolbar-state ${safeState}`;
        if (textEl) textEl.textContent = label || '空闲';
    }

    function createConfigGroupEntry(name, fields, extraClass = '', description = '') {
        const categoryId = FORM_CATEGORY_SECTION_MAP.get(name) || 'advanced';
        return {
            name,
            fields,
            extraClass,
            description,
            categoryId,
        };
    }

    function appendConfigGroupsByCategory(container, groups) {
        if (!groups.length) {
            container.appendChild(createConfigFormEmpty('当前配置没有可编辑字段。'));
            return;
        }
        const buckets = new Map(FORM_CATEGORY_DEFS.map((category) => [category.id, []]));
        for (const group of groups) {
            const categoryId = group.categoryId || FORM_CATEGORY_SECTION_MAP.get(group.name) || 'advanced';
            const bucket = buckets.get(categoryId) || buckets.get('advanced');
            bucket.push(group);
        }

        const searchText = normalizeConfigSearch(configFormState.search);
        const categories = FORM_CATEGORY_DEFS.filter((category) => {
            const categoryGroups = buckets.get(category.id) || [];
            return categoryGroups.length && configCategoryVisible(category, searchText);
        });
        const activeCategory = normalizeConfigActiveCategory(categories);
        updateConfigStickyDirectory(categories, buckets, activeCategory, searchText);
        updateConfigStickyPlacement();
        const renderedGroups = [];
        const sourceCategories = searchText
            ? categories
            : categories.filter((category) => category.id === activeCategory);
        for (const category of FORM_CATEGORY_DEFS) {
            if (!sourceCategories.some((item) => item.id === category.id)) continue;
            for (const group of buckets.get(category.id) || []) {
                const filtered = filterConfigGroupEntry(group, searchText);
                if (filtered) renderedGroups.push(filtered);
            }
        }

        const shell = document.createElement('div');
        shell.className = ['config-form-shell', searchText ? 'searching' : '', configFormState.showAdvanced ? 'advanced-visible' : 'basic-only'].filter(Boolean).join(' ');

        const main = document.createElement('div');
        main.className = 'config-form-main';
        main.appendChild(createConfigFormControls(groups, renderedGroups, searchText));
        const groupList = document.createElement('div');
        groupList.className = 'config-form-group-list';
        if (!renderedGroups.length) {
            groupList.appendChild(createConfigFormEmpty(searchText ? '没有匹配的配置项。' : '这个分类暂无可编辑项。'));
        } else {
            for (const group of renderedGroups) {
                groupList.appendChild(createGroup(group.name, group.fields, group.extraClass, group.description, searchText));
            }
        }
        main.appendChild(groupList);
        shell.appendChild(main);
        container.appendChild(shell);
        updateChangedFieldMarks();
        requestAnimationFrame(updateConfigStickyPlacement);
    }

    function normalizeConfigSearch(value) {
        return String(value || '').trim().toLowerCase();
    }

    function configCategoryVisible(category, searchText = '') {
        return Boolean(searchText) || configFormState.showAdvanced || !category.advanced;
    }

    function normalizeConfigActiveCategory(categories) {
        if (!categories.length) return '';
        const ids = new Set(categories.map((category) => category.id));
        if (!ids.has(configFormState.activeCategory)) {
            configFormState.activeCategory = categories[0].id;
        }
        return configFormState.activeCategory;
    }

    function selectConfigCategory(categoryId, options = {}) {
        if (!categoryId) return;
        const category = FORM_CATEGORY_DEFS.find((item) => item.id === categoryId);
        syncConfigDraftFromForm();
        if (category?.advanced) {
            configFormState.showAdvanced = true;
        }
        configFormState.activeCategory = categoryId;
        configFormState.search = '';
        renderConfigForm(currentConfig);
        if (options.scrollToForm) {
            requestAnimationFrame(() => scrollConfigFormContentToTop('smooth'));
        }
    }

    function scrollConfigFormContentToTop(behavior = 'auto') {
        const scroller = document.querySelector('#tab-config .config-left');
        if (!scroller) return;
        scroller.scrollTo({ top: 0, behavior });
    }

    function updateConfigStickyDirectory(categories, buckets, activeCategory, searchText) {
        const visibleCategories = new Set(categories.map((category) => category.id));
        document.querySelectorAll('[data-sticky-config-category]').forEach((btn) => {
            const categoryId = btn.dataset.stickyConfigCategory || '';
            const category = FORM_CATEGORY_DEFS.find((item) => item.id === categoryId);
            const categoryGroups = buckets.get(categoryId) || [];
            const hasFields = categoryGroups.length > 0;
            const enabled = Boolean(category && STICKY_CONFIG_CATEGORY_IDS.has(categoryId) && (visibleCategories.has(categoryId) || (category.advanced && hasFields)));
            const fieldCount = categoryGroups.reduce((sum, group) => sum + group.fields.length, 0);
            btn.hidden = !category || !STICKY_CONFIG_CATEGORY_IDS.has(categoryId);
            btn.disabled = !enabled;
            btn.classList.toggle('active', enabled && categoryId === activeCategory && !searchText);
            btn.setAttribute('aria-current', enabled && categoryId === activeCategory && !searchText ? 'true' : 'false');
            btn.title = enabled ? `切换到${category.title}配置` : '当前配置没有这个分类';
            const count = btn.querySelector('em');
            if (count) count.textContent = `${fieldCount} 项`;
        });
    }

    function updateConfigStickyPlacement() {
        const bar = document.getElementById('config-sticky-actions');
        const workspace = document.getElementById('config-form-workspace');
        if (!bar || !workspace || workspace.hidden) return;
        const rect = workspace.getBoundingClientRect();
        if (rect.width <= 0) return;
        const sidePadding = Math.min(16, Math.max(0, rect.width / 18));
        const maxWidth = Math.max(0, rect.width - sidePadding * 2);
        const width = Math.min(1040, maxWidth);
        const left = rect.left + Math.max(sidePadding, (rect.width - width) / 2);
        bar.style.setProperty('--config-sticky-left', `${Math.round(left)}px`);
        bar.style.setProperty('--config-sticky-width', `${Math.round(width)}px`);

        const scroller = workspace.querySelector('.config-left');
        if (!scroller) return;
        const barRect = bar.getBoundingClientRect();
        const barStyle = window.getComputedStyle(bar);
        const bottomOffset = Number.parseFloat(barStyle.bottom) || 20;
        const safeSpace = Math.ceil(barRect.height + bottomOffset + 18);
        const scrollerRect = scroller.getBoundingClientRect();
        const availableHeight = Math.max(180, window.innerHeight - scrollerRect.top - 16);
        workspace.style.setProperty('--config-sticky-safe-space', `${safeSpace}px`);
        workspace.style.setProperty('--config-left-max-height', `${Math.round(availableHeight)}px`);
    }

    function createConfigFormControls(allGroups, renderedGroups, searchText) {
        const controls = document.createElement('div');
        controls.className = 'config-form-controls';

        const searchLabel = document.createElement('label');
        searchLabel.className = 'config-search-box';
        const searchCaption = document.createElement('span');
        searchCaption.textContent = '搜索配置项';
        const search = document.createElement('input');
        search.id = 'config-search-input';
        search.type = 'search';
        search.placeholder = '输入学习率、caption、network_dim 或中文名称';
        search.value = configFormState.search;
        search.addEventListener('input', (event) => {
            syncConfigDraftFromForm();
            configFormState.search = event.target.value || '';
            renderConfigForm(currentConfig);
            requestAnimationFrame(() => {
                const next = document.getElementById('config-search-input');
                if (next) {
                    next.focus();
                    const length = next.value.length;
                    next.setSelectionRange(length, length);
                }
            });
        });
        searchLabel.append(searchCaption, search);
        controls.appendChild(searchLabel);

        const advanced = document.createElement('label');
        advanced.className = 'config-advanced-toggle';
        const advancedInput = document.createElement('input');
        advancedInput.id = 'config-advanced-toggle';
        advancedInput.type = 'checkbox';
        advancedInput.checked = configFormState.showAdvanced;
        advancedInput.addEventListener('change', (event) => {
            syncConfigDraftFromForm();
            configFormState.showAdvanced = event.target.checked;
            renderConfigForm(currentConfig);
        });
        const advancedText = document.createElement('span');
        advancedText.textContent = '显示高级配置';
        advanced.append(advancedInput, advancedText);
        controls.appendChild(advanced);

        const resetBtn = document.createElement('button');
        resetBtn.id = 'btn-reset-config-changes';
        resetBtn.type = 'button';
        resetBtn.className = 'btn btn-small';
        resetBtn.textContent = '重置当前改动';
        resetBtn.title = '重新读取当前配置文件，丢弃尚未保存的表单修改。';
        resetBtn.addEventListener('click', reloadCurrentConfig);
        controls.appendChild(resetBtn);

        const summary = document.createElement('div');
        summary.className = 'config-form-summary';
        const total = allGroups.reduce((sum, group) => sum + group.fields.length, 0);
        const rendered = renderedGroups.reduce((sum, group) => sum + group.fields.length, 0);
        summary.innerHTML = [
            `<span>${searchText ? '匹配' : '显示'} <strong>${rendered}</strong> / ${total} 项</span>`,
            '<span>已修改 <strong id="config-modified-count">0</strong> 项</span>',
        ].join('');
        controls.appendChild(summary);
        return controls;
    }

    function filterConfigGroupEntry(group, searchText) {
        if (!searchText) return group;
        const groupMatched = configTextMatches([group.name, group.description, group.categoryId], searchText);
        const fields = groupMatched
            ? group.fields
            : group.fields.filter(([key, value]) => configFieldMatchesSearch(key, value, searchText));
        if (!fields.length) return null;
        return { ...group, fields };
    }

    function configFieldMatchesSearch(key, value, searchText) {
        return configTextMatches([
            key,
            formatFieldName(key),
            value,
            fieldHelp[key] ? JSON.stringify(fieldHelp[key]) : '',
        ], searchText);
    }

    function configTextMatches(parts, searchText) {
        return parts.some((part) => String(part ?? '').toLowerCase().includes(searchText));
    }

    function createConfigFormEmpty(text) {
        const empty = document.createElement('div');
        empty.className = 'config-form-empty';
        empty.textContent = text;
        return empty;
    }

    function configCategoryIsAdvanced(categoryId) {
        return Boolean(FORM_CATEGORY_DEFS.find((category) => category.id === categoryId)?.advanced);
    }

    function configGroupIsCollapsed(name, searchText = '') {
        if (searchText) return false;
        if (configFormState.collapsedGroups.has(name)) return true;
        if (configFormState.expandedGroups.has(name)) return false;
        if (configFormState.activeCategory === 'advanced' && ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS.has(name)) return false;
        return configCategoryIsAdvanced(FORM_CATEGORY_SECTION_MAP.get(name) || 'advanced');
    }

    function createGroup(name, fields, extraClass = '', description = '', searchText = '') {
        const section = document.createElement('section');
        section.className = ['config-group', extraClass].filter(Boolean).join(' ');
        section.dataset.groupName = name;

        const header = document.createElement('div');
        header.className = 'config-group-title';
        const title = document.createElement('span');
        title.textContent = `${name} (${fields.length} 项)`;
        header.appendChild(title);

        const content = document.createElement('div');
        content.className = 'config-group-body';
        const collapsed = searchText ? configGroupIsCollapsed(name, searchText) : configGroupIsCollapsed(name);
        content.hidden = collapsed;
        const collapseBtn = document.createElement('button');
        collapseBtn.type = 'button';
        collapseBtn.className = 'config-group-collapse';
        collapseBtn.textContent = collapsed ? '展开' : '收起';
        collapseBtn.setAttribute('aria-expanded', String(!collapsed));
        collapseBtn.title = collapsed ? '展开这个配置区' : '收起这个配置区';
        collapseBtn.addEventListener('click', () => {
            const nextCollapsed = !content.hidden;
            content.hidden = nextCollapsed;
            collapseBtn.textContent = nextCollapsed ? '展开' : '收起';
            collapseBtn.setAttribute('aria-expanded', String(!nextCollapsed));
            collapseBtn.title = nextCollapsed ? '展开这个配置区' : '收起这个配置区';
            if (nextCollapsed) {
                configFormState.collapsedGroups.add(name);
                configFormState.expandedGroups.delete(name);
            } else {
                configFormState.expandedGroups.add(name);
                configFormState.collapsedGroups.delete(name);
            }
        });
        let hint = null;
        if (description) {
            const hintId = `config-group-hint-${++configGroupHintSeq}`;
            const btn = document.createElement('button');
            btn.className = 'info-toggle config-group-info-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '展开分组说明';
            btn.setAttribute('aria-label', `${name} 说明`);
            btn.setAttribute('aria-controls', hintId);
            btn.setAttribute('aria-expanded', 'false');
            header.appendChild(btn);

            hint = document.createElement('p');
            hint.className = 'config-group-hint';
            hint.id = hintId;
            hint.hidden = true;
            hint.textContent = description;
            btn.addEventListener('click', () => {
                const nextVisible = hint.hidden;
                hint.hidden = !nextVisible;
                btn.classList.toggle('active', nextVisible);
                btn.setAttribute('aria-expanded', String(nextVisible));
                btn.title = nextVisible ? '收起分组说明' : '展开分组说明';
            });
            content.appendChild(hint);
        }
        const titleActions = document.createElement('div');
        titleActions.className = 'config-group-title-actions';
        if (extraClass === 'config-group-model') {
            titleActions.appendChild(createFillGlobalModelPathsButton());
        }
        if (extraClass === 'config-group-resource') {
            titleActions.appendChild(createResourceQuickPresetsButton(content, collapseBtn));
        }
        titleActions.appendChild(collapseBtn);
        header.appendChild(titleActions);
        section.appendChild(header);
        if (extraClass === 'config-group-data') {
            content.appendChild(createConfigDatasetPicker());
        }
        if (extraClass === 'config-group-resource') {
            content.appendChild(createResourceQuickPresetPanel());
        }
        appendFieldRows(content, fields, extraClass);
        if (extraClass === 'config-group-steps') {
            content.appendChild(createStepEstimatePanel());
            updateStepEstimatePanel();
        }
        section.appendChild(content);
        return section;
    }

    function createOpenStageResolutionDialogButton() {
        const btn = document.createElement('button');
        btn.id = 'btn-open-stage-resolution-dialog';
        btn.type = 'button';
        btn.className = 'btn btn-small config-group-title-action';
        btn.textContent = '阶段调度';
        btn.title = '打开阶段分辨率调度面板';
        btn.addEventListener('click', openStageResolutionDialog);
        return btn;
    }

    function openStageResolutionDialog() {
        const dialog = document.getElementById('stage-resolution-dialog');
        if (!dialog) return;
        renderStageResolutionDialog();
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
        requestAnimationFrame(drawStageResolutionChart);
    }

    function normalizedStageResolutionStages() {
        if (!Array.isArray(stageResolutionState.stages) || !stageResolutionState.stages.length) {
            stageResolutionState.stages = [
                { name: 'EP1', epochs: 1, maxSide: 1024, downRange: 256, manualRepeats: false, repeats: 1 },
            ];
        }
        stageResolutionState.stages = stageResolutionState.stages.map((stage, index) => ({
            name: String(stage.name || `EP${index + 1}`).trim() || `EP${index + 1}`,
            epochs: Number(stage.epochs) || 0,
            maxSide: Number(stage.maxSide) || 0,
            downRange: Number(stage.downRange) || 0,
            manualRepeats: Boolean(stage.manualRepeats),
            repeats: Math.max(1, Math.round(Number(stage.repeats) || 1)),
        }));
        stageResolutionState.selectedIndex = Math.max(
            0,
            Math.min(stageResolutionState.selectedIndex || 0, stageResolutionState.stages.length - 1)
        );
        return stageResolutionState.stages;
    }

    function stageResolutionMetrics() {
        stageResolutionState.enabled = Boolean(stageResolutionState.enabled);
        const stages = normalizedStageResolutionStages();
        let cursorStep = 0;
        const ranges = stages.map((stage, index) => {
            const epochs = Number(stage.epochs);
            const maxSide = Number(stage.maxSide);
            const downRange = Number(stage.downRange);
            const minSide = maxSide - downRange;
            const startStep = cursorStep;
            const steps = Math.max(0, epochs) * STAGE_RESOLUTION_STEPS_PER_EPOCH;
            cursorStep += steps;
            const problems = [];
            const warnings = [];
            if (!Number.isFinite(epochs) || epochs <= 0) problems.push('epochs 必须大于 0');
            if (!Number.isFinite(maxSide) || maxSide <= 0) problems.push('单边最大值无效');
            if (!Number.isFinite(downRange) || downRange <= 0) problems.push('向下波动必须大于 0');
            if (Number.isFinite(minSide) && minSide <= 0) problems.push('单边最小值无效');
            if (Number.isFinite(minSide) && Number.isFinite(maxSide) && minSide >= maxSide) problems.push('范围为空');
            return {
                ...stage,
                index,
                startStep,
                endStep: cursorStep,
                steps,
                minSide,
                imageCount: null,
                autoRepeats: stage.manualRepeats ? stage.repeats : 1,
                problems,
                warnings,
            };
        });

        for (let i = 0; i < ranges.length; i += 1) {
            for (let j = i + 1; j < ranges.length; j += 1) {
                const a = ranges[i];
                const b = ranges[j];
                if (a.problems.length || b.problems.length) continue;
                const overlaps = Math.max(a.minSide, b.minSide) < Math.min(a.maxSide, b.maxSide);
                if (overlaps) {
                    a.warnings.push('范围重叠');
                    b.warnings.push('范围重叠');
                }
            }
        }
        const sorted = ranges
            .filter((item) => !item.problems.length)
            .slice()
            .sort((a, b) => a.minSide - b.minSide);
        for (let i = 1; i < sorted.length; i += 1) {
            if (sorted[i].minSide > sorted[i - 1].maxSide) {
                sorted[i - 1].warnings.push('存在断档');
                sorted[i].warnings.push('存在断档');
            }
        }

        const problemCount = ranges.filter((item) => item.problems.length).length;
        const warningCount = ranges.filter((item) => item.warnings.length).length;
        return {
            enabled: stageResolutionState.enabled,
            stages: ranges,
            totalSteps: cursorStep,
            problemCount,
            warningCount,
            selected: ranges[stageResolutionState.selectedIndex] || ranges[0],
        };
    }

    function stageResolutionStatus(stage) {
        if (stage.problems.length) return { tone: 'error', text: stage.problems[0] };
        if (stage.warnings.length) return { tone: 'warning', text: stage.warnings[0] };
        return { tone: 'ok', text: '就绪' };
    }

    function renderStageResolutionDialog() {
        const body = document.getElementById('stage-resolution-dialog-body');
        if (!body) return;
        const metrics = stageResolutionMetrics();
        body.innerHTML = '';
        body.appendChild(createStageResolutionSummary(metrics));

        const workspace = document.createElement('div');
        workspace.className = 'stage-resolution-workspace';
        workspace.appendChild(createStageResolutionChartPanel());
        workspace.appendChild(createStageResolutionEditor(metrics.selected));
        body.appendChild(workspace);

        body.appendChild(createStageResolutionTable(metrics.stages));
        requestAnimationFrame(drawStageResolutionChart);
    }

    function createStageResolutionSummary(metrics) {
        const wrap = document.createElement('div');
        wrap.className = 'stage-resolution-summary';
        const rows = [
            ['调度状态', metrics.enabled ? '已启用' : '未启用'],
            ['阶段数', `${metrics.stages.length}`],
            ['预计 steps', `${metrics.totalSteps}`],
            ['配置检查', metrics.problemCount ? `${metrics.problemCount} 项错误` : (metrics.warningCount ? `${metrics.warningCount} 项提示` : '就绪')],
            ['图片统计', '待接入'],
        ];
        wrap.appendChild(createStageResolutionEnableControl(metrics.enabled));
        rows.forEach(([label, value]) => {
            const item = document.createElement('div');
            item.className = 'stage-resolution-summary-item';
            const strong = document.createElement('strong');
            strong.textContent = value;
            const span = document.createElement('span');
            span.textContent = label;
            item.append(strong, span);
            wrap.appendChild(item);
        });
        return wrap;
    }

    function createStageResolutionEnableControl(enabled) {
        const item = document.createElement('label');
        item.className = 'stage-resolution-summary-item stage-resolution-enable-control';
        const input = document.createElement('input');
        input.id = 'stage-resolution-enable-toggle';
        input.type = 'checkbox';
        input.checked = enabled;
        input.addEventListener('change', (event) => {
            setStageResolutionEnabled(event.target.checked);
        });
        const copy = document.createElement('span');
        const strong = document.createElement('strong');
        strong.textContent = '启用阶段调度';
        const hint = document.createElement('span');
        hint.textContent = enabled ? '将用于阶段方案' : '草稿，不影响训练';
        copy.append(strong, hint);
        item.append(input, copy);
        return item;
    }

    function setStageResolutionEnabled(enabled) {
        stageResolutionState.enabled = Boolean(enabled);
        renderStageResolutionDialog();
    }

    function createStageResolutionChartPanel() {
        const panel = document.createElement('section');
        panel.className = 'stage-resolution-chart-panel';
        const header = document.createElement('div');
        header.className = 'stage-resolution-panel-head';
        const title = document.createElement('div');
        title.innerHTML = '<strong>阶段折线</strong><span>点表示阶段，阴影表示该阶段的单边范围。</span>';
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small';
        addBtn.textContent = '新增阶段';
        addBtn.addEventListener('click', addStageResolutionPoint);
        header.append(title, addBtn);
        const canvas = document.createElement('canvas');
        canvas.id = 'stage-resolution-chart';
        canvas.width = 720;
        canvas.height = 280;
        canvas.addEventListener('click', selectStageResolutionPointFromCanvas);
        panel.append(header, canvas);
        return panel;
    }

    function createStageResolutionEditor(stage) {
        const aside = document.createElement('aside');
        aside.className = 'stage-resolution-editor';
        const head = document.createElement('div');
        head.className = 'stage-resolution-panel-head';
        const title = document.createElement('div');
        title.innerHTML = '<strong>当前点编辑器</strong><span>修改后立即同步折线图和阶段表。</span>';
        head.appendChild(title);
        aside.appendChild(head);
        if (!stage) return aside;

        const fields = document.createElement('div');
        fields.className = 'stage-resolution-fields';
        fields.append(
            createStageResolutionInput('阶段名', 'name', stage.name, 'text'),
            createStageResolutionInput('epochs', 'epochs', stage.epochs, 'number'),
            createStageResolutionInput('单边最大值', 'maxSide', stage.maxSide, 'number'),
            createStageResolutionInput('向下波动', 'downRange', stage.downRange, 'number'),
            createStageResolutionReadonly('单边最小值', `${Math.max(0, stage.minSide || 0)}`),
            createStageResolutionReadonly('预计图片数', '待统计'),
            createStageResolutionRepeats(stage)
        );
        aside.appendChild(fields);
        return aside;
    }

    function createStageResolutionInput(labelText, key, value, type) {
        const label = document.createElement('label');
        label.className = 'stage-resolution-field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = type;
        input.value = value;
        input.dataset.stageField = key;
        if (type === 'number') {
            input.min = key === 'epochs' ? '1' : '0';
            input.step = '1';
        }
        input.addEventListener('input', updateSelectedStageResolutionField);
        label.append(span, input);
        return label;
    }

    function createStageResolutionReadonly(labelText, value) {
        const label = document.createElement('label');
        label.className = 'stage-resolution-field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const output = document.createElement('output');
        output.textContent = value;
        label.append(span, output);
        return label;
    }

    function createStageResolutionRepeats(stage) {
        const wrap = document.createElement('div');
        wrap.className = 'stage-resolution-field stage-resolution-repeat-field';
        const label = document.createElement('label');
        const check = document.createElement('input');
        check.type = 'checkbox';
        check.checked = stage.manualRepeats;
        check.addEventListener('change', (event) => {
            updateStageResolutionStage(stage.index, { manualRepeats: event.target.checked });
        });
        label.append(check, document.createTextNode('手动 repeats'));
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '1';
        input.step = '1';
        input.value = stage.autoRepeats;
        input.disabled = !stage.manualRepeats;
        input.addEventListener('input', (event) => {
            updateStageResolutionStage(stage.index, { repeats: Math.max(1, Math.round(Number(event.target.value) || 1)) });
        });
        wrap.append(label, input);
        return wrap;
    }

    function createStageResolutionTable(stages) {
        const section = document.createElement('section');
        section.className = 'stage-resolution-table-panel';
        const head = document.createElement('div');
        head.className = 'stage-resolution-panel-head';
        const title = document.createElement('div');
        title.innerHTML = '<strong>阶段表</strong><span>每行对应一个阶段点。</span>';
        head.appendChild(title);
        const tableWrap = document.createElement('div');
        tableWrap.className = 'stage-resolution-table-wrap';
        const table = document.createElement('table');
        table.className = 'stage-resolution-table';
        table.innerHTML = '<thead><tr><th>阶段</th><th>epochs</th><th>单边最大</th><th>向下波动</th><th>step 范围</th><th>分辨率范围</th><th>图片</th><th>repeats</th><th>状态</th><th>操作</th></tr></thead>';
        const tbody = document.createElement('tbody');
        stages.forEach((stage) => tbody.appendChild(createStageResolutionTableRow(stage)));
        table.appendChild(tbody);
        tableWrap.appendChild(table);
        section.append(head, tableWrap);
        return section;
    }

    function createStageResolutionTableRow(stage) {
        const tr = document.createElement('tr');
        const selected = stage.index === stageResolutionState.selectedIndex;
        const status = stageResolutionStatus(stage);
        tr.className = selected ? 'selected' : '';
        tr.dataset.stageIndex = String(stage.index);
        tr.append(
            stageResolutionTableInputCell(stage, 'name', stage.name, 'text'),
            stageResolutionTableInputCell(stage, 'epochs', stage.epochs, 'number'),
            stageResolutionTableInputCell(stage, 'maxSide', stage.maxSide, 'number'),
            stageResolutionTableInputCell(stage, 'downRange', stage.downRange, 'number'),
            stageResolutionTableCell(`${stage.startStep}-${stage.endStep}`),
            stageResolutionTableCell(`${Math.max(0, stage.minSide)}-${stage.maxSide}`),
            stageResolutionTableCell('待统计'),
            stageResolutionTableCell(`${stage.autoRepeats}`),
            stageResolutionStatusCell(status),
            stageResolutionActionCell(stage)
        );
        tr.addEventListener('click', (event) => {
            if (event.target.closest('button')) return;
            selectStageResolutionPoint(stage.index);
        });
        return tr;
    }

    function stageResolutionTableInputCell(stage, key, value, type) {
        const td = document.createElement('td');
        const input = document.createElement('input');
        input.className = 'stage-resolution-table-input';
        input.type = type;
        input.value = value;
        if (type === 'number') {
            input.min = key === 'epochs' ? '1' : '0';
            input.step = '1';
        }
        input.addEventListener('click', (event) => event.stopPropagation());
        input.addEventListener('input', (event) => {
            const next = key === 'name' ? event.target.value : Number(event.target.value);
            updateStageResolutionStage(stage.index, { [key]: next });
        });
        td.appendChild(input);
        return td;
    }

    function stageResolutionTableCell(text) {
        const td = document.createElement('td');
        td.textContent = text;
        return td;
    }

    function stageResolutionStatusCell(status) {
        const td = document.createElement('td');
        const badge = document.createElement('span');
        badge.className = `stage-resolution-status ${status.tone}`;
        badge.textContent = status.text;
        td.appendChild(badge);
        return td;
    }

    function stageResolutionActionCell(stage) {
        const td = document.createElement('td');
        td.className = 'stage-resolution-actions';
        const up = stageResolutionActionButton('↑', '上移', () => moveStageResolutionPoint(stage.index, -1));
        const down = stageResolutionActionButton('↓', '下移', () => moveStageResolutionPoint(stage.index, 1));
        const del = stageResolutionActionButton('删', '删除', () => deleteStageResolutionPoint(stage.index));
        up.disabled = stage.index <= 0;
        down.disabled = stage.index >= stageResolutionState.stages.length - 1;
        del.disabled = stageResolutionState.stages.length <= 1;
        td.append(up, down, del);
        return td;
    }

    function stageResolutionActionButton(text, title, handler) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-small';
        btn.textContent = text;
        btn.title = title;
        btn.addEventListener('click', handler);
        return btn;
    }

    function updateSelectedStageResolutionField(event) {
        const key = event.target.dataset.stageField;
        const value = key === 'name' ? event.target.value : Number(event.target.value);
        updateStageResolutionStage(stageResolutionState.selectedIndex, { [key]: value });
    }

    function updateStageResolutionStage(index, patch) {
        const stages = normalizedStageResolutionStages();
        if (!stages[index]) return;
        stageResolutionState.stages[index] = { ...stages[index], ...patch };
        renderStageResolutionDialog();
    }

    function addStageResolutionPoint() {
        const stages = normalizedStageResolutionStages();
        const last = stages[stages.length - 1] || { maxSide: 1024, downRange: 256 };
        stages.push({
            name: `EP${stages.length + 1}`,
            epochs: 1,
            maxSide: Math.max(256, Number(last.maxSide || 1024) + 512),
            downRange: Math.max(64, Number(last.downRange || 256)),
            manualRepeats: false,
            repeats: 1,
        });
        stageResolutionState.selectedIndex = stages.length - 1;
        renderStageResolutionDialog();
    }

    function deleteStageResolutionPoint(index) {
        const stages = normalizedStageResolutionStages();
        if (stages.length <= 1) return;
        stages.splice(index, 1);
        stageResolutionState.selectedIndex = Math.max(0, Math.min(index, stages.length - 1));
        renderStageResolutionDialog();
    }

    function moveStageResolutionPoint(index, direction) {
        const stages = normalizedStageResolutionStages();
        const nextIndex = index + direction;
        if (nextIndex < 0 || nextIndex >= stages.length) return;
        [stages[index], stages[nextIndex]] = [stages[nextIndex], stages[index]];
        stageResolutionState.selectedIndex = nextIndex;
        renderStageResolutionDialog();
    }

    function selectStageResolutionPoint(index) {
        stageResolutionState.selectedIndex = index;
        renderStageResolutionDialog();
    }

    function selectStageResolutionPointFromCanvas(event) {
        const canvas = event.currentTarget;
        const points = canvas._stageResolutionPoints || [];
        if (!points.length) return;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        let nearest = points[0];
        for (const point of points) {
            if (Math.abs(point.x - x) < Math.abs(nearest.x - x)) nearest = point;
        }
        selectStageResolutionPoint(nearest.index);
    }

    function drawStageResolutionChart() {
        const canvas = document.getElementById('stage-resolution-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const metrics = stageResolutionMetrics();
        const stages = metrics.stages;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(320, Math.floor(rect.width || 720));
        const height = Math.max(220, Math.floor(rect.height || 280));
        const ratio = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);

        const styles = getComputedStyle(document.documentElement);
        const accent = styles.getPropertyValue('--accent').trim() || '#4fc3f7';
        const warning = styles.getPropertyValue('--warning').trim() || '#f0c36a';
        const danger = styles.getPropertyValue('--danger').trim() || '#ef5350';
        const grid = styles.getPropertyValue('--chart-grid').trim() || '#2a3a5e';
        const text = styles.getPropertyValue('--text-dim').trim() || '#8892a4';
        const success = styles.getPropertyValue('--success').trim() || '#22c55e';
        const pad = { top: 24, right: 38, bottom: 38, left: 46 };
        const plotW = width - pad.left - pad.right;
        const plotH = height - pad.top - pad.bottom;
        if (!stages.length || plotW <= 0 || plotH <= 0) return;

        const values = stages.flatMap((stage) => [stage.maxSide, Math.max(0, stage.minSide)]);
        let minY = Math.min(...values);
        let maxY = Math.max(...values);
        if (minY === maxY) {
            minY -= 128;
            maxY += 128;
        }
        minY = Math.max(0, Math.floor(minY / 128) * 128);
        maxY = Math.ceil(maxY / 128) * 128;
        const yFor = (value) => pad.top + (1 - ((value - minY) / Math.max(1, maxY - minY))) * plotH;
        const xFor = (index) => stages.length === 1
            ? pad.left + plotW / 2
            : pad.left + (plotW * index / (stages.length - 1));

        ctx.strokeStyle = grid;
        ctx.lineWidth = 0.5;
        ctx.fillStyle = text;
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i += 1) {
            const y = pad.top + (plotH * i / 4);
            const value = maxY - ((maxY - minY) * i / 4);
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(width - pad.right, y);
            ctx.stroke();
            ctx.fillText(String(Math.round(value)), pad.left - 8, y + 3);
        }

        const points = [];
        stages.forEach((stage, index) => {
            const x = xFor(index);
            const yMax = yFor(stage.maxSide);
            const yMin = yFor(Math.max(0, stage.minSide));
            const status = stageResolutionStatus(stage);
            const color = status.tone === 'error' ? danger : (status.tone === 'warning' ? warning : accent);
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.16;
            ctx.fillRect(x - 24, yMax, 48, Math.max(2, yMin - yMax));
            ctx.globalAlpha = 1;
            points.push({ x, y: yMax, index });
        });

        ctx.strokeStyle = accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((point, index) => {
            if (index === 0) ctx.moveTo(point.x, point.y);
            else ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();

        points.forEach((point) => {
            const stage = stages[point.index];
            const status = stageResolutionStatus(stage);
            const selected = point.index === stageResolutionState.selectedIndex;
            const color = status.tone === 'error' ? danger : (status.tone === 'warning' ? warning : success);
            ctx.fillStyle = color;
            ctx.strokeStyle = selected ? warning : accent;
            ctx.lineWidth = selected ? 3 : 1.5;
            ctx.beginPath();
            ctx.arc(point.x, point.y, selected ? 6 : 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = text;
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`${stage.startStep}-${stage.endStep}`, point.x, height - 18);
        });
        canvas._stageResolutionPoints = points;
    }

    function createFillGlobalModelPathsButton() {
        const btn = document.createElement('button');
        btn.id = 'btn-fill-global-model-paths';
        btn.type = 'button';
        btn.className = 'btn btn-small config-group-title-action';
        btn.textContent = '填写全局路径配置';
        btn.title = '用全局设置里的三项基础模型路径覆盖当前配置表单';
        btn.addEventListener('click', () => {
            fillGlobalModelPathsIntoConfigForm().catch((e) => {
                setTomlStatus('error', '填写全局路径配置失败: ' + e.message);
            });
        });
        return btn;
    }

    function createResourceQuickPresetsButton(content, collapseBtn) {
        const btn = document.createElement('button');
        btn.id = 'btn-resource-quick-presets';
        btn.type = 'button';
        btn.className = 'btn btn-small config-group-title-action config-resource-quick-toggle';
        btn.textContent = '快速填写';
        btn.title = '显示显存与速度预设，一键填写当前表单';
        btn.setAttribute('aria-expanded', 'false');
        btn.addEventListener('click', () => {
            const panel = content.querySelector('.config-resource-quick-presets');
            if (!panel) return;
            const nextVisible = panel.hidden;
            panel.hidden = !nextVisible;
            btn.classList.toggle('active', nextVisible);
            btn.setAttribute('aria-expanded', String(nextVisible));
            btn.title = nextVisible ? '收起显存与速度快速预设' : '显示显存与速度预设，一键填写当前表单';
            if (nextVisible && content.hidden) {
                content.hidden = false;
                collapseBtn.textContent = '收起';
                collapseBtn.setAttribute('aria-expanded', 'true');
                collapseBtn.title = '收起这个配置区';
                configFormState.expandedGroups.add('显存与速度');
                configFormState.collapsedGroups.delete('显存与速度');
            }
        });
        return btn;
    }

    function createResourceQuickPresetPanel() {
        const panel = document.createElement('div');
        panel.className = 'config-resource-quick-presets';
        panel.hidden = true;
        panel.setAttribute('aria-label', '显存与速度快速预设');

        const label = document.createElement('span');
        label.className = 'config-resource-quick-label';
        label.textContent = '快速预设';
        panel.appendChild(label);

        for (const preset of RESOURCE_QUICK_PRESETS) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'btn btn-small config-resource-preset-btn';
            btn.dataset.resourcePreset = preset.id;
            btn.textContent = preset.label;
            btn.title = preset.note;
            btn.addEventListener('click', () => applyResourceQuickPreset(preset));
            panel.appendChild(btn);
        }
        return panel;
    }

    function applyResourceQuickPreset(preset) {
        for (const [key, value] of Object.entries(preset.values)) {
            setFieldInputValue(key, value);
        }
        handleFormFieldChange();
        setTomlStatus('ok', `已填写显存与速度预设: ${preset.label}`);
    }

    async function fillGlobalModelPathsIntoConfigForm() {
        if (!globalSettings && location.protocol !== 'file:') {
            await loadGlobalSettings();
        }
        const overrides = getGlobalModelPathOverrides();
        const entries = GLOBAL_MODEL_PATH_FIELDS
            .map(([key]) => [key, overrides[key]])
            .filter(([, value]) => String(value || '').trim());
        if (!entries.length) {
            setTomlStatus('error', '全局设置里还没有可填写的基础模型路径');
            return;
        }

        const confirmed = await showAppConfirmDialog({
            title: '是否确认覆盖',
            description: '填写全局路径配置',
            message: '将用全局设置里的基础模型路径覆盖当前配置表单中的同名字段。',
            confirmText: '是',
            cancelText: '否',
        });
        if (!confirmed) return;

        let applied = 0;
        for (const [key, value] of entries) {
            const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
            if (!input) continue;
            input.value = value;
            applied += 1;
        }
        handleFormFieldChange();
        setTomlStatus(
            applied ? '已填写全局路径配置，请保存当前配置后再训练' : '当前表单没有可覆盖的基础模型路径字段',
            applied ? 'ok' : 'error'
        );
    }

    function appendFieldRows(content, fields, groupClass) {
        const compactGroups = CONFIG_COMPACT_FIELD_GROUPS[groupClass] || [];
        const usedLayouts = new Set();
        let index = 0;

        while (index < fields.length) {
            const [key] = fields[index];
            const compactLayout = compactGroups.find((layout) => {
                if (usedLayouts.has(layout)) return false;
                return layout.keys.includes(key);
            });

            if (!compactLayout) {
                content.appendChild(createFieldRow(fields[index][0], fields[index][1]));
                index += 1;
                continue;
            }

            usedLayouts.add(compactLayout);
            const compactKeys = new Set(compactLayout.keys);
            const grid = document.createElement('div');
            grid.className = ['config-field-grid', compactLayout.className].filter(Boolean).join(' ');

            while (index < fields.length && compactKeys.has(fields[index][0])) {
                const [compactKey, compactValue] = fields[index];
                const row = createFieldRow(compactKey, compactValue);
                row.classList.add('field-row-compact');
                grid.appendChild(row);
                index += 1;
            }

            if (grid.childElementCount <= 1) {
                const onlyRow = grid.firstElementChild;
                if (onlyRow) content.appendChild(onlyRow);
            } else {
                normalizeCompactGridColumns(grid);
                content.appendChild(grid);
            }
        }
    }

    function normalizeCompactGridColumns(grid) {
        const count = grid.childElementCount;
        grid.classList.remove('config-field-grid-3col', 'config-field-grid-4col');
        if (count >= 4) {
            grid.classList.add('config-field-grid-4col');
        } else if (count === 3) {
            grid.classList.add('config-field-grid-3col');
        }
    }

    function createConfigDatasetPicker() {
        const panel = document.createElement('div');
        panel.id = 'config-dataset-picker';
        panel.className = 'config-dataset-picker';
        renderConfigDatasetPicker(panel);
        return panel;
    }

    function renderConfigDatasetPicker(existingPanel = null) {
        const panel = existingPanel || document.getElementById('config-dataset-picker');
        if (!panel) return;
        panel.innerHTML = '';

        const header = document.createElement('div');
        header.className = 'config-dataset-picker-header';
        const title = document.createElement('div');
        title.innerHTML = '<strong>数据集预设</strong><span>当前配置只保留选择摘要；搜索、选择和预览在弹窗中完成。</span>';
        const actions = document.createElement('div');
        actions.className = 'config-dataset-picker-actions';
        const openBtn = document.createElement('button');
        openBtn.type = 'button';
        openBtn.className = 'btn btn-small';
        openBtn.textContent = selectedConfigDatasetFile ? '更换预设' : '选择预设';
        openBtn.title = '打开数据集预设弹窗，可以搜索并查看第一张原始图预览。';
        openBtn.addEventListener('click', openConfigDatasetPickerDialog);
        const manageBtn = document.createElement('button');
        manageBtn.type = 'button';
        manageBtn.className = 'btn btn-small';
        manageBtn.textContent = '管理数据集';
        manageBtn.addEventListener('click', () => document.querySelector('[data-tab="datasets"]')?.click());
        const refreshBtn = document.createElement('button');
        refreshBtn.type = 'button';
        refreshBtn.className = 'btn btn-small';
        refreshBtn.textContent = '刷新预设';
        refreshBtn.addEventListener('click', () => loadDatasetPresets({ selectCurrent: false, manage: false }));
        actions.append(openBtn, manageBtn, refreshBtn);
        header.append(title, actions);
        panel.appendChild(header);

        const body = document.createElement('div');
        body.className = 'config-dataset-picker-body';
        body.appendChild(createConfigDatasetCurrentSummary());
        panel.appendChild(body);
        if (isConfigDatasetPickerDialogOpen()) {
            renderConfigDatasetPickerDialog();
        }
        ensureConfigDatasetPreview();
    }

    function createConfigDatasetCurrentSummary() {
        const preset = datasetPresetByFile(selectedConfigDatasetFile);
        const summary = selectedConfigDatasetSummary || preset?.summary || {};
        const wrap = document.createElement('div');
        wrap.className = 'config-dataset-current';

        const info = document.createElement('div');
        info.className = 'config-dataset-current-info';
        const label = document.createElement('span');
        label.className = 'config-dataset-current-label';
        label.textContent = selectedConfigDatasetFile ? '当前选中' : '当前状态';
        const title = document.createElement('strong');
        title.textContent = selectedConfigDatasetFile
            ? (preset?.label || preset?.filename || selectedConfigDatasetFile)
            : '不使用独立数据集预设';
        const path = document.createElement('code');
        path.textContent = selectedConfigDatasetFile || '沿用当前训练配置文件中的数据集字段';
        info.append(label, title, path);

        const meta = document.createElement('div');
        meta.className = 'config-dataset-current-meta';
        const state = document.createElement('span');
        const isDirtySelection = selectedConfigDatasetFile !== (currentConfig.dataset_config || '');
        state.className = [
            'config-dataset-current-state',
            isDirtySelection ? 'dirty' : 'synced',
        ].join(' ');
        state.textContent = isDirtySelection
            ? '未保存'
            : '已同步';
        const count = document.createElement('span');
        count.textContent = selectedConfigDatasetFile
            ? `${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}`
            : '当前配置';
        const source = document.createElement('span');
        source.textContent = selectedConfigDatasetFile && summary.source_dir
            ? `原始路径: ${summary.source_dir}`
            : '保存当前配置后才会写入训练 TOML';
        meta.append(state, count, source);

        wrap.append(info, meta);
        return wrap;
    }

    function isConfigDatasetPickerDialogOpen() {
        return Boolean(document.getElementById('config-dataset-picker-dialog')?.open);
    }

    function openConfigDatasetPickerDialog() {
        const dialog = document.getElementById('config-dataset-picker-dialog');
        if (!dialog) return;
        renderConfigDatasetPickerDialog();
        ensureConfigDatasetPreview();
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
        const search = dialog.querySelector('.config-dataset-search');
        if (search) {
            search.focus({ preventScroll: true });
            search.setSelectionRange(search.value.length, search.value.length);
        }
    }

    function closeConfigDatasetPickerDialog() {
        const dialog = document.getElementById('config-dataset-picker-dialog');
        if (dialog?.open) dialog.close();
    }

    function openUnnamedDatasetDialog() {
        const dialog = document.getElementById('unnamed-dataset-dialog');
        if (!dialog) return;
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
    }

    function renderContinueTrainingSource() {
        const summary = document.getElementById('continue-training-source-summary');
        const chooseBtn = document.getElementById('btn-open-continue-lora-dialog');
        const clearBtn = document.getElementById('btn-clear-continue-lora-source');
        if (!summary || !chooseBtn || !clearBtn) return;
        summary.innerHTML = '';
        if (!continueTrainingSource) {
            const title = document.createElement('strong');
            title.textContent = '从零开始';
            const detail = document.createElement('span');
            detail.textContent = '不加载已有权重';
            summary.append(title, detail);
            summary.className = 'continue-training-source-summary';
            chooseBtn.textContent = '选择 LoRA/LoHa/LoKr';
            clearBtn.hidden = true;
            updateTomlActionState(currentTomlFile);
            return;
        }
        const title = document.createElement('strong');
        title.textContent = `继续训练 ${continueTrainingSource.kind || 'LoRA'} · ${continueTrainingSource.name || '未命名权重'}`;
        const path = document.createElement('code');
        path.textContent = continueTrainingSource.abs_path || '';
        const state = document.createElement('span');
        state.className = continueTrainingSource.compatible === false ? 'warning' : 'ok';
        state.textContent = continueTrainingSource.compatible === false
            ? (continueTrainingSource.message || '当前配置不兼容')
            : '兼容 · 启动时会使用 --network_weights 与 --dim_from_weights';
        summary.append(title, path, state);
        summary.className = [
            'continue-training-source-summary',
            continueTrainingSource.compatible === false ? 'incompatible' : 'selected',
        ].join(' ');
        chooseBtn.textContent = '更换';
        clearBtn.hidden = false;
        updateTomlActionState(currentTomlFile);
    }

    function continueTrainingRequestPayload() {
        if (!continueTrainingSource) return {};
        return {
            continue_from_weight_abs_path: continueTrainingSource.abs_path || '',
            continue_from_weight_name: continueTrainingSource.name || '',
            continue_from_weight_kind: continueTrainingSource.kind || '',
        };
    }

    function clearContinueTrainingSource() {
        continueTrainingSource = null;
        renderContinueTrainingSource();
        setTomlStatus('ok', '已恢复为从零开始训练');
    }

    async function openContinueLoraDialog() {
        const dialog = document.getElementById('continue-lora-dialog');
        if (!dialog) return;
        if (!historyTasks.length) {
            await loadTrainingHistoryList();
        }
        renderContinueLoraHistoryTasks();
        const input = document.getElementById('continue-lora-path-input');
        if (input && continueTrainingSource?.abs_path) {
            input.value = continueTrainingSource.abs_path;
        }
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
        await loadContinueLoraWeights();
        document.getElementById('continue-lora-path-input')?.focus({ preventScroll: true });
    }

    function renderContinueLoraHistoryTasks() {
        const select = document.getElementById('continue-lora-history-task');
        if (!select) return;
        const previous = continueLoraDialogState.taskId;
        const tasks = historyTasks.filter((task) => task.job === 'training');
        select.innerHTML = '';
        const latest = document.createElement('option');
        latest.value = '';
        latest.textContent = '最近一次训练输出';
        select.appendChild(latest);
        for (const task of tasks) {
            const option = document.createElement('option');
            option.value = task.id || '';
            option.textContent = historyTaskDisplayName(task) || task.id || '训练任务';
            select.appendChild(option);
        }
        if (previous && tasks.some((task) => task.id === previous)) {
            select.value = previous;
        } else {
            continueLoraDialogState.taskId = '';
            select.value = '';
        }
    }

    async function loadContinueLoraWeights() {
        const list = document.getElementById('continue-lora-weight-list');
        if (!list) return;
        continueLoraDialogState.loading = true;
        continueLoraDialogState.error = '';
        renderContinueLoraWeights();
        try {
            const params = new URLSearchParams();
            if (continueLoraDialogState.taskId) {
                params.set('task_id', continueLoraDialogState.taskId);
            }
            const suffix = params.toString() ? `?${params.toString()}` : '';
            const payload = await api(`/api/preview/weights${suffix}`);
            continueLoraDialogState = {
                ...continueLoraDialogState,
                loading: false,
                weights: payload.weights || [],
                error: payload.ok === false ? (payload.error || '读取权重失败') : '',
                message: payload.message || '',
            };
        } catch (e) {
            continueLoraDialogState = {
                ...continueLoraDialogState,
                loading: false,
                weights: [],
                error: e.message || '读取权重失败',
            };
        }
        renderContinueLoraWeights();
    }

    function renderContinueLoraWeights() {
        const list = document.getElementById('continue-lora-weight-list');
        if (!list) return;
        list.innerHTML = '';
        if (continueLoraDialogState.loading) {
            list.textContent = '正在读取历史权重...';
            return;
        }
        if (continueLoraDialogState.error) {
            list.textContent = continueLoraDialogState.error;
            return;
        }
        if (!continueLoraDialogState.weights.length) {
            list.textContent = continueLoraDialogState.message || '没有可选择的 .safetensors 权重。';
            return;
        }
        for (const item of continueLoraDialogState.weights) {
            const row = document.createElement('div');
            row.className = 'continue-lora-weight-item';
            const info = document.createElement('div');
            const name = document.createElement('strong');
            name.textContent = item.name || '未命名权重';
            const path = document.createElement('code');
            path.textContent = item.abs_path || item.file || '';
            info.append(name, path);
            const useBtn = document.createElement('button');
            useBtn.type = 'button';
            useBtn.className = 'btn btn-small btn-primary';
            useBtn.textContent = '继续训练';
            useBtn.addEventListener('click', () => selectContinueLoraWeight(item.abs_path || item.file || ''));
            row.append(info, useBtn);
            list.appendChild(row);
        }
    }

    function setContinueLoraStatus(message, state = '') {
        const status = document.getElementById('continue-lora-inspect-status');
        if (!status) return;
        status.className = ['continue-lora-status', state].filter(Boolean).join(' ');
        status.textContent = message || '';
    }

    async function requestContinueLoraInspection(path) {
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        return api('/api/training/continue-lora/inspect', {
            method: 'POST',
            body: JSON.stringify({
                path,
                variant,
                preset,
                methods_subdir: methodsSubdir,
                config_file: currentTrainingConfigFile(),
            }),
        });
    }

    async function selectContinueLoraWeight(path, options = {}) {
        const rawPath = String(path || '').trim();
        if (!rawPath) {
            setContinueLoraStatus('请填写 .safetensors 权重绝对路径。', 'error');
            return false;
        }
        setContinueLoraStatus('正在检查权重结构与当前变体兼容性...', 'pending');
        try {
            const payload = await requestContinueLoraInspection(rawPath);
            if (!payload.ok) {
                setContinueLoraStatus(payload.error || '权重检测失败。', 'error');
                if (!document.getElementById('continue-lora-dialog')?.open) {
                    alert(payload.error || '权重检测失败。');
                }
                return false;
            }
            if (!payload.compatible) {
                setContinueLoraStatus(payload.message || '当前配置与这个权重不兼容。', 'warning');
                if (!document.getElementById('continue-lora-dialog')?.open) {
                    alert(payload.message || '当前配置与这个权重不兼容。');
                }
                return false;
            }
            continueTrainingSource = payload;
            renderContinueTrainingSource();
            setContinueLoraStatus(payload.message || '已选择继续训练权重。', 'ok');
            setTomlStatus('ok', `训练来源已设置为继续训练 ${payload.kind} · ${payload.name}`);
            if (options.switchToConfig !== false) {
                document.querySelector('[data-tab="config"]')?.click();
            }
            const dialog = document.getElementById('continue-lora-dialog');
            if (dialog?.open && options.keepDialogOpen !== true) dialog.close();
            return true;
        } catch (e) {
            setContinueLoraStatus('权重检测请求失败: ' + e.message, 'error');
            if (!document.getElementById('continue-lora-dialog')?.open) {
                alert('权重检测请求失败: ' + e.message);
            }
            return false;
        }
    }

    async function refreshContinueTrainingSourceCompatibility() {
        if (!continueTrainingSource?.abs_path) {
            renderContinueTrainingSource();
            return true;
        }
        let payload;
        try {
            payload = await requestContinueLoraInspection(continueTrainingSource.abs_path);
        } catch (e) {
            continueTrainingSource = {
                ...continueTrainingSource,
                compatible: false,
                message: '无法重新检查继续训练权重: ' + e.message,
            };
            renderContinueTrainingSource();
            return false;
        }
        if (!payload.ok) {
            continueTrainingSource = {
                ...continueTrainingSource,
                compatible: false,
                message: payload.error || '无法重新检查继续训练权重。',
            };
            renderContinueTrainingSource();
            return false;
        }
        continueTrainingSource = payload;
        renderContinueTrainingSource();
        return Boolean(payload.compatible);
    }

    function renderConfigDatasetPickerDialog() {
        const dialog = document.getElementById('config-dataset-picker-dialog');
        const body = document.getElementById('config-dataset-picker-dialog-body');
        if (!dialog || !body) return;
        body.innerHTML = '';

        const toolbar = document.createElement('div');
        toolbar.className = 'config-dataset-dialog-toolbar';
        const search = document.createElement('input');
        search.type = 'search';
        search.className = 'field-input config-dataset-search';
        search.placeholder = '搜索数据集预设、路径或原始目录';
        search.value = configDatasetPickerSearch;
        search.addEventListener('input', () => {
            const cursor = search.selectionStart ?? search.value.length;
            configDatasetPickerSearch = search.value;
            renderConfigDatasetPickerDialog();
            const nextSearch = document.querySelector('#config-dataset-picker-dialog .config-dataset-search');
            if (nextSearch) {
                nextSearch.focus();
                nextSearch.setSelectionRange(cursor, cursor);
            }
        });
        toolbar.appendChild(search);
        body.appendChild(toolbar);

        const workspace = document.createElement('div');
        workspace.className = 'config-dataset-workspace config-dataset-dialog-workspace';
        workspace.appendChild(createConfigDatasetPresetList());
        workspace.appendChild(createConfigDatasetPresetPreview());
        body.appendChild(workspace);
    }

    function datasetPresetOptionLabel(preset) {
        const summary = preset?.summary || {};
        const name = preset?.label || preset?.filename || preset?.path || '未命名预设';
        const count = Number(summary.dataset_count || 0);
        const repeats = Number(summary.repeat_total || 0);
        const lock = preset?.readonly ? '只读 · ' : '';
        return `${lock}${name} · ${count || 0} 组 · 重复 ${repeats || 0}`;
    }

    function createConfigDatasetPresetList() {
        const list = document.createElement('div');
        list.className = 'config-dataset-preset-list';
        const noneBtn = createConfigDatasetPresetButton(null);
        list.appendChild(noneBtn);

        const presets = filteredConfigDatasetPresets();
        if (!presets.length && configDatasetPickerSearch.trim()) {
            const empty = document.createElement('p');
            empty.className = 'config-dataset-picker-empty';
            empty.textContent = '没有匹配的数据集预设。';
            list.appendChild(empty);
            return list;
        }

        for (const preset of presets) {
            list.appendChild(createConfigDatasetPresetButton(preset));
        }
        return list;
    }

    function createConfigDatasetPresetButton(preset) {
        const isNone = !preset;
        const file = isNone ? '' : preset.path;
        const summary = preset?.summary || {};
        const active = file === selectedConfigDatasetFile;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'config-dataset-preset-option',
            active ? 'active' : '',
            preset?.readonly ? 'readonly' : '',
        ].filter(Boolean).join(' ');
        btn.dataset.file = file;
        const title = document.createElement('strong');
        title.textContent = isNone
            ? '不使用独立数据集预设'
            : (preset.label || preset.filename || preset.path || '未命名预设');
        const path = document.createElement('span');
        path.textContent = isNone ? '沿用当前训练配置文件中的数据集字段' : preset.path;
        const meta = document.createElement('small');
        meta.textContent = isNone
            ? '当前配置'
            : `${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}${preset.readonly ? ' · 只读' : ''}`;
        btn.append(title, path, meta);
        btn.addEventListener('click', () => selectConfigDatasetPreset(file));
        return btn;
    }

    function filteredConfigDatasetPresets() {
        const keyword = configDatasetPickerSearch.trim().toLowerCase();
        const presets = datasetPresetState.presets || [];
        if (!keyword) return presets;
        return presets.filter((preset) => {
            const summary = preset.summary || {};
            return [
                preset.label,
                preset.filename,
                preset.path,
                summary.source_dir,
                            ].some((value) => String(value || '').toLowerCase().includes(keyword));
        });
    }

    function createConfigDatasetPresetPreview() {
        const preview = document.createElement('div');
        preview.className = 'config-dataset-preview';
        const summary = document.createElement('div');
        summary.className = 'config-dataset-summary';
        summary.appendChild(createConfigDatasetSummary());
        preview.appendChild(summary);
        preview.appendChild(createConfigDatasetPreviewImage());
        return preview;
    }

    function createConfigDatasetPreviewImage() {
        const box = document.createElement('div');
        box.className = 'config-dataset-preview-image';
        const state = configDatasetPreviewState;
        if (!selectedConfigDatasetFile) {
            box.classList.add('empty');
            box.textContent = '选择一个数据集预设后，这里会显示第一张原始图。';
            return box;
        }
        if (state.file !== selectedConfigDatasetFile || state.loading) {
            box.classList.add('empty');
            box.textContent = '正在读取第一张原始图...';
            return box;
        }
        if (state.error) {
            box.classList.add('empty');
            box.textContent = state.error;
            return box;
        }
        const image = Array.isArray(state.payload?.images) ? state.payload.images[0] : null;
        if (!image) {
            box.classList.add('empty');
            box.textContent = state.payload?.message || '没有找到可预览的原始图。';
            return box;
        }
        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name || '数据集预览图';
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            box.classList.add('empty');
            box.textContent = '预览图加载失败。';
        });
        const caption = document.createElement('div');
        caption.className = 'config-dataset-preview-caption';
        const name = document.createElement('strong');
        name.textContent = image.name || '-';
        const path = document.createElement('span');
        path.textContent = state.payload?.directory || image.file || '';
        caption.append(name, path);
        box.append(img, caption);
        return box;
    }

    function createConfigDatasetSummary() {
        const wrap = document.createElement('div');
        const preset = datasetPresetByFile(selectedConfigDatasetFile);
        const summary = selectedConfigDatasetSummary || preset?.summary || {};
        if (!selectedConfigDatasetFile) {
            wrap.className = 'config-dataset-summary-empty';
            wrap.textContent = '未选择独立数据集预设；训练会沿用当前配置文件里的数据集字段。';
            return wrap;
        }
        const items = [
            ['预设文件', selectedConfigDatasetFile],
            ['数据组数', String(summary.dataset_count || 0)],
            ['重复合计', String(summary.repeat_total || 0)],
            ['第 1 组原始路径', summary.source_dir || '-'],
        ];
        if (selectedConfigDatasetFile !== (currentConfig.dataset_config || '')) {
            items.unshift(['状态', '未保存，保存当前配置后生效']);
        }
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            row.append(key, valEl);
            wrap.appendChild(row);
        }
        return wrap;
    }

    async function selectConfigDatasetPreset(file) {
        selectedConfigDatasetFile = file || '';
        selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
        configDatasetPreviewState = {
            file: '',
            loading: false,
            payload: null,
            error: '',
        };
        renderConfigDatasetPicker();
        renderConfigDatasetPickerDialog();
        updateTomlDirtyState();
        await loadStepEstimate();
    }

    function datasetPresetByFile(file) {
        return (datasetPresetState.presets || []).find((item) => item.path === file) || null;
    }

    function datasetPresetSummaryByFile(file) {
        return datasetPresetByFile(file)?.summary || null;
    }

    function ensureConfigDatasetPreview() {
        if (!selectedConfigDatasetFile) return;
        if (configDatasetPreviewState.file === selectedConfigDatasetFile && (configDatasetPreviewState.loading || configDatasetPreviewState.payload || configDatasetPreviewState.error)) {
            return;
        }
        loadConfigDatasetPresetPreview(selectedConfigDatasetFile);
    }

    async function loadConfigDatasetPresetPreview(file) {
        if (!file || location.protocol === 'file:') return;
        const requestSeq = ++configDatasetPreviewRequestSeq;
        configDatasetPreviewState = {
            file,
            loading: true,
            payload: null,
            error: '',
        };
        renderConfigDatasetPreviewArea();
        try {
            const params = new URLSearchParams({
                file,
                dataset_index: '0',
                source: 'source',
                limit: '1',
            });
            const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
            if (requestSeq !== configDatasetPreviewRequestSeq || file !== selectedConfigDatasetFile) return;
            if (!payload.ok) throw new Error(payload.error || '读取数据集预览失败');
            configDatasetPreviewState = {
                file,
                loading: false,
                payload,
                error: '',
            };
        } catch (e) {
            if (requestSeq !== configDatasetPreviewRequestSeq || file !== selectedConfigDatasetFile) return;
            configDatasetPreviewState = {
                file,
                loading: false,
                payload: null,
                error: e.message || '读取数据集预览失败',
            };
        }
        renderConfigDatasetPreviewArea();
    }

    function renderConfigDatasetPreviewArea() {
        const previews = document.querySelectorAll('.config-dataset-preview');
        if (!previews.length) return;
        previews.forEach((preview) => {
            preview.innerHTML = '';
            const summary = document.createElement('div');
            summary.className = 'config-dataset-summary';
            summary.appendChild(createConfigDatasetSummary());
            preview.appendChild(summary);
            preview.appendChild(createConfigDatasetPreviewImage());
        });
    }

    function createDatasetEditor() {
        const panel = document.createElement('div');
        panel.id = 'dataset-editor';
        panel.className = 'dataset-editor';
        renderDatasetEditor(panel);
        return panel;
    }

    function renderDatasetPresetList() {
        const list = document.getElementById('dataset-preset-list');
        if (!list) return;
        list.innerHTML = '';
        const presets = datasetPresetState.presets || [];
        const groups = datasetPresetGroupsForDisplay();
        updateDatasetPresetPageSummary();
        const showErrorAsEmptyState = datasetPresetState.error && !presets.length;
        if (datasetPresetState.loading && !presets.length) {
            const loading = document.createElement('p');
            loading.className = 'dataset-preset-empty';
            loading.textContent = '正在读取数据集预设...';
            list.appendChild(loading);
            return;
        }
        if (showErrorAsEmptyState) {
            const error = document.createElement('p');
            error.className = 'dataset-preset-empty error';
            error.textContent = datasetPresetState.error;
            list.appendChild(error);
        }
        if (!presets.length && !groups.length) {
            const empty = document.createElement('p');
            empty.className = 'dataset-preset-empty';
            empty.textContent = datasetPresetState.error ? '读取数据集预设失败。' : '还没有数据集预设。';
            list.appendChild(empty);
            return;
        }
        if (!groups.length) {
            const empty = document.createElement('p');
            empty.className = 'dataset-preset-empty';
            empty.textContent = '没有匹配的数据集预设。';
            list.appendChild(empty);
            return;
        }
        const stored = readDatasetPresetGroupState();
        for (const group of groups) {
            list.appendChild(createDatasetPresetGroupNode(group, stored));
        }
    }

    function updateDatasetPresetPageSummary() {
        const summary = document.getElementById('dataset-page-summary');
        if (!summary) return;
        const presets = datasetPresetState.presets || [];
        const groups = datasetPresetState.groups || [];
        const totalDatasets = presets.reduce((sum, preset) => sum + Number(preset.summary?.dataset_count || 0), 0);
        const totalRepeats = presets.reduce((sum, preset) => sum + Number(preset.summary?.repeat_total || 0), 0);
        summary.innerHTML = '';
        [
            ['预设', presets.length],
            ['分组', groups.length || 1],
            ['子集', totalDatasets],
            ['重复', totalRepeats],
        ].forEach(([label, value]) => {
            const item = document.createElement('span');
            item.className = 'dataset-page-summary-stat';
            item.innerHTML = `<strong>${escapeHtml(String(value))}</strong><small>${escapeHtml(label)}</small>`;
            summary.appendChild(item);
        });
        if (datasetPresetState.dirty) {
            const dirty = document.createElement('span');
            dirty.className = 'dataset-page-summary-dirty';
            dirty.textContent = '当前预设未保存';
            summary.appendChild(dirty);
        }
    }

    function datasetPresetGroupsForDisplay() {
        const keyword = datasetPresetState.search.trim().toLowerCase();
        const presetMap = new Map((datasetPresetState.presets || []).map((preset) => [preset.path, preset]));
        const sourceGroups = (datasetPresetState.groups || []).length
            ? datasetPresetState.groups
            : [{
                id: 'datasets',
                label: '数据集配置',
                open: false,
                kind: 'dataset',
                files: datasetPresetState.presets || [],
                movable: true,
            }];
        const covered = new Set();
        const groups = [];

        for (const rawGroup of sourceGroups) {
            const files = (rawGroup.files || [])
                .map((item) => presetMap.get(item.path) || item)
                .filter((item) => item?.path && presetMap.has(item.path))
                .filter((item) => datasetPresetMatchesSearch(item, keyword));
            (rawGroup.files || []).forEach((item) => {
                if (item?.path && presetMap.has(item.path)) covered.add(item.path);
            });
            if (keyword && !files.length) continue;
            if (!files.length && rawGroup.kind !== 'dataset' && rawGroup.id !== 'datasets' && rawGroup.id !== 'unfiled_datasets') continue;
            groups.push({ ...rawGroup, files });
        }

        const ungrouped = (datasetPresetState.presets || [])
            .filter((preset) => !covered.has(preset.path))
            .filter((preset) => datasetPresetMatchesSearch(preset, keyword));
        if (ungrouped.length) {
            groups.push({
                id: 'unfiled_datasets',
                label: '未分组数据集配置',
                open: true,
                kind: 'dataset',
                movable: true,
                files: ungrouped,
            });
        }
        return sortDatasetPresetGroups(groups);
    }

    function isUnfiledDatasetGroup(group) {
        return group?.id === 'unfiled_datasets';
    }

    function sortDatasetPresetGroups(groups) {
        return [...groups].sort((a, b) => {
            if (isUnfiledDatasetGroup(a)) return -1;
            if (isUnfiledDatasetGroup(b)) return 1;
            return 0;
        });
    }

    function orderDatasetPresetsForGroups(presets, groups) {
        const presetMap = new Map((presets || []).map((preset) => [preset.path, preset]));
        const ordered = [];
        const seen = new Set();
        for (const group of sortDatasetPresetGroups(groups || [])) {
            for (const item of group.files || []) {
                if (!item?.path || seen.has(item.path) || !presetMap.has(item.path)) continue;
                ordered.push(presetMap.get(item.path));
                seen.add(item.path);
            }
        }
        for (const preset of presets || []) {
            if (!preset?.path || seen.has(preset.path)) continue;
            ordered.push(preset);
        }
        return ordered;
    }

    function datasetPresetMatchesSearch(preset, keyword) {
        if (!keyword) return true;
        const summary = preset?.summary || {};
        return [
            preset?.label,
            preset?.filename,
            preset?.path,
            summary.source_dir,
            summary.image_dir,
            summary.cache_dir,
        ].some((value) => String(value || '').toLowerCase().includes(keyword));
    }

    function eventTargetClosest(event, selector) {
        const target = event?.target;
        return target instanceof Element ? target.closest(selector) : null;
    }

    function createFileGroupDragImage(payload) {
        const image = document.createElement('div');
        image.className = 'file-group-drag-image';
        image.textContent = payload.file || payload.groupId || '移动项目';
        document.body.appendChild(image);
        return image;
    }

    function removeFileGroupDragImage(image) {
        if (image?.parentNode) image.parentNode.removeChild(image);
    }

    function setFileGroupDragData(event, payload) {
        const data = payload.file || payload.groupId || payload.target || 'move';
        const transfer = event?.dataTransfer;
        if (!transfer) return;
        let image = null;
        try {
            transfer.setData('text/plain', data);
            transfer.setData('application/x-anima-file-group', JSON.stringify({
                target: payload.target || '',
                scope: payload.scope || '',
                file: payload.file || '',
                groupId: payload.groupId || '',
            }));
            transfer.effectAllowed = 'move';
            image = createFileGroupDragImage(payload);
            transfer.setDragImage(image, 12, 12);
        } catch (e) {
            /* 部分浏览器会限制 DataTransfer 写入；内存态拖拽仍可继续。 */
        } finally {
            if (image) window.setTimeout(() => removeFileGroupDragImage(image), 0);
        }
    }

    function canBeginFileGroupDrag(payload, disabled) {
        if (disabled || (payload.canDrag && !payload.canDrag())) {
            if (payload.blockedMessage) payload.blockedMessage();
            return false;
        }
        return true;
    }

    function beginFileGroupDrag(payload, handle) {
        fileGroupDragState = payload;
        payload.sourceElement?.classList.add('file-group-dragging');
        handle?.classList.add('dragging');
    }

    function createFileGroupPointerDragImage(payload) {
        const image = createFileGroupDragImage(payload);
        image.classList.add('file-group-drag-image-pointer');
        return image;
    }

    function moveFileGroupPointerDragImage(image, x, y) {
        if (!image) return;
        image.style.left = `${x + 14}px`;
        image.style.top = `${y + 14}px`;
    }

    function registerFileGroupDropTarget(node, resolve) {
        node.setAttribute(FILE_GROUP_DROP_TARGET_ATTR, '1');
        fileGroupDropTargets.set(node, resolve);
        fileGroupDropTargetNodes.add(node);
    }

    function originClosest(origin, selector) {
        return origin instanceof Element ? origin.closest(selector) : null;
    }

    function resolveFileGroupPointerDropTarget(x, y) {
        const payload = fileGroupDragState;
        const origin = document.elementFromPoint(x, y);
        let node = origin;
        while (node && node !== document.documentElement) {
            if (node instanceof Element && node.hasAttribute(FILE_GROUP_DROP_TARGET_ATTR)) {
                const resolve = fileGroupDropTargets.get(node);
                const target = resolve?.({ payload, x, y, origin });
                if (target) return { node, ...target };
            }
            node = node.parentElement;
        }
        return resolveNearestFileGroupDropTarget(x, y, origin, payload);
    }

    function resolveNearestFileGroupDropTarget(x, y, origin, payload) {
        let best = null;
        for (const node of fileGroupDropTargetNodes) {
            if (!node?.isConnected || !(node instanceof Element)) {
                fileGroupDropTargetNodes.delete(node);
                continue;
            }
            const rect = node.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) continue;
            const maxDistance = Math.max(26, Math.min(90, rect.height * 0.85));
            const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
            const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
            const distance = Math.hypot(dx, dy);
            if (distance > maxDistance) continue;
            const resolve = fileGroupDropTargets.get(node);
            const target = resolve?.({ payload, x, y, origin });
            if (!target) continue;
            if (!best || distance < best.distance) best = { node, distance, ...target };
        }
        if (!best) return null;
        const { distance, ...target } = best;
        return target;
    }

    function markResolvedFileGroupDropTarget(target) {
        if (!target) {
            clearFileGroupDropIndicators();
            return;
        }
        if (target.position === 'before' || target.position === 'after') {
            target.node.dataset.dropPosition = target.position;
        }
        markFileGroupDropTarget(target.node, target.position);
    }

    function removeFileGroupDropPreview() {
        if (fileGroupDropPreviewElement?.parentNode) {
            fileGroupDropPreviewElement.parentNode.removeChild(fileGroupDropPreviewElement);
        }
        fileGroupDropPreviewElement = null;
    }

    function ensureFileGroupDropPreview() {
        if (fileGroupDropPreviewElement?.isConnected) return fileGroupDropPreviewElement;
        const preview = document.createElement('div');
        preview.className = 'file-group-drop-preview';
        preview.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.textContent = '释放后插入到这里';
        preview.appendChild(label);
        document.body.appendChild(preview);
        fileGroupDropPreviewElement = preview;
        return preview;
    }

    function placeFileGroupDropPreview(node, position) {
        if (!node || position !== 'before' && position !== 'after') {
            removeFileGroupDropPreview();
            return;
        }
        const rect = node.getBoundingClientRect?.();
        if (!rect || rect.width <= 0 || rect.height <= 0) {
            removeFileGroupDropPreview();
            return;
        }
        const preview = ensureFileGroupDropPreview();
        preview.dataset.position = position;
        preview.style.left = `${rect.left + 4}px`;
        preview.style.top = `${position === 'before' ? rect.top : rect.bottom}px`;
        preview.style.width = `${Math.max(40, rect.width - 8)}px`;
    }

    function findScrollableFileGroupAncestor(origin) {
        let node = origin instanceof Element ? origin : null;
        while (node && node !== document.body) {
            const style = window.getComputedStyle(node);
            if (/(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight) {
                return node;
            }
            node = node.parentElement;
        }
        return document.scrollingElement;
    }

    function autoScrollFileGroupPointerDrag(x, y) {
        const origin = document.elementFromPoint(x, y);
        const scroller = findScrollableFileGroupAncestor(origin);
        if (!scroller) return;
        const rect = scroller === document.scrollingElement
            ? { top: 0, bottom: window.innerHeight }
            : scroller.getBoundingClientRect();
        const margin = 46;
        const speed = 16;
        let delta = 0;
        if (y < rect.top + margin) {
            delta = -speed;
        } else if (y > rect.bottom - margin) {
            delta = speed;
        }
        if (delta) scroller.scrollBy({ top: delta, behavior: 'auto' });
    }

    function cleanupFileGroupPointerDrag() {
        const drag = fileGroupPointerDrag;
        if (!drag) return null;
        document.removeEventListener('pointermove', drag.onMove);
        document.removeEventListener('pointerup', drag.onUp);
        document.removeEventListener('pointercancel', drag.onCancel);
        document.removeEventListener('mousemove', drag.onMouseMove);
        document.removeEventListener('mouseup', drag.onMouseUp);
        document.removeEventListener('keydown', drag.onKeydown);
        try {
            if (drag.pointerId !== null && drag.pointerId !== undefined) {
                drag.handle.releasePointerCapture?.(drag.pointerId);
            }
        } catch (e) {
            /* 指针可能已被浏览器释放，忽略即可。 */
        }
        removeFileGroupDragImage(drag.image);
        document.body.classList.remove('file-group-pointer-drag-active');
        drag.handle?.classList.remove('dragging');
        fileGroupPointerDrag = null;
        return drag;
    }

    function finishFileGroupPointerDrag(commit = false) {
        const drag = cleanupFileGroupPointerDrag();
        if (!drag) return;
        const target = commit && drag.active ? drag.currentDrop : null;
        finishFileGroupDrag();
        if (!target) return;
        Promise.resolve(target.drop()).catch((e) => {
            console.error('拖拽位置更新失败', e);
        });
    }

    function startFileGroupFallbackDrag(event, payload, handle, disabled, options = {}) {
        if (disabled || fileGroupPointerDrag) return;
        if ((options.pointer || options.mouse) && 'button' in event && event.button !== 0) return;
        if (options.pointer && event.isPrimary === false) return;
        event.preventDefault();
        event.stopPropagation();
        const pointerId = options.pointer ? event.pointerId : null;
        const drag = {
            payload,
            handle,
            pointerId,
            startX: event.clientX,
            startY: event.clientY,
            active: false,
            image: null,
            currentDrop: null,
        };
        const moveDrag = (moveEvent) => {
            const distance = Math.hypot(moveEvent.clientX - drag.startX, moveEvent.clientY - drag.startY);
            if (!drag.active) {
                if (distance < 4) return;
                if (!canBeginFileGroupDrag(payload, disabled)) {
                    finishFileGroupPointerDrag(false);
                    return;
                }
                beginFileGroupDrag(payload, handle);
                drag.active = true;
                drag.image = createFileGroupPointerDragImage(payload);
                document.body.classList.add('file-group-pointer-drag-active');
            }
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            moveFileGroupPointerDragImage(drag.image, moveEvent.clientX, moveEvent.clientY);
            autoScrollFileGroupPointerDrag(moveEvent.clientX, moveEvent.clientY);
            drag.currentDrop = resolveFileGroupPointerDropTarget(moveEvent.clientX, moveEvent.clientY);
            markResolvedFileGroupDropTarget(drag.currentDrop);
        };
        drag.onMove = (moveEvent) => {
            if (moveEvent.pointerId !== pointerId) return;
            moveDrag(moveEvent);
        };
        drag.onUp = (upEvent) => {
            if (upEvent.pointerId !== pointerId) return;
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishFileGroupPointerDrag(true);
        };
        drag.onMouseMove = (moveEvent) => {
            moveDrag(moveEvent);
        };
        drag.onMouseUp = (upEvent) => {
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishFileGroupPointerDrag(true);
        };
        drag.onCancel = (cancelEvent) => {
            if (cancelEvent.pointerId !== pointerId) return;
            finishFileGroupPointerDrag(false);
        };
        drag.onKeydown = (keyEvent) => {
            if (keyEvent.key === 'Escape') finishFileGroupPointerDrag(false);
        };
        fileGroupPointerDrag = drag;
        const addMouseFallbackListeners = () => {
            document.addEventListener('mousemove', drag.onMouseMove, { passive: false });
            document.addEventListener('mouseup', drag.onMouseUp, { passive: false });
        };
        if (options.pointer) {
            try {
                handle.setPointerCapture?.(pointerId);
            } catch (e) {
                /* 浏览器可能已切换到原生拖拽流程，继续使用文档级监听兜底。 */
            }
            document.addEventListener('pointermove', drag.onMove, { passive: false });
            document.addEventListener('pointerup', drag.onUp, { passive: false });
            document.addEventListener('pointercancel', drag.onCancel, { passive: false });
            addMouseFallbackListeners();
        } else {
            addMouseFallbackListeners();
        }
        document.addEventListener('keydown', drag.onKeydown);
    }

    function startFileGroupPointerDrag(event, payload, handle, disabled) {
        startFileGroupFallbackDrag(event, payload, handle, disabled, { pointer: true });
    }

    function startFileGroupMouseDrag(event, payload, handle, disabled) {
        startFileGroupFallbackDrag(event, payload, handle, disabled, { mouse: true });
    }

    function createFileGroupDragHandle(payload, options = {}) {
        const handle = document.createElement('button');
        const disabled = Boolean(options.disabled);
        handle.type = 'button';
        handle.className = ['file-group-drag-handle', disabled ? 'disabled' : ''].filter(Boolean).join(' ');
        handle.setAttribute('aria-label', options.label || '拖动调整位置');
        handle.title = options.title || '拖动调整位置';
        handle.textContent = '⋮⋮';
        handle.tabIndex = disabled ? -1 : 0;
        handle.draggable = !disabled;
        handle.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
        });
        handle.addEventListener('pointerdown', (event) => startFileGroupPointerDrag(event, payload, handle, disabled));
        handle.addEventListener('mousedown', (event) => {
            event.stopPropagation();
            startFileGroupMouseDrag(event, payload, handle, disabled);
        });
        handle.addEventListener('dragstart', (event) => {
            if (fileGroupPointerDrag) {
                event.preventDefault();
                event.stopPropagation();
                return;
            }
            if (!canBeginFileGroupDrag(payload, disabled)) {
                event.preventDefault();
                return;
            }
            event.stopPropagation();
            beginFileGroupDrag(payload, handle);
            setFileGroupDragData(event, payload);
        });
        handle.addEventListener('dragend', () => {
            handle.classList.remove('dragging');
            finishFileGroupDrag();
        });
        return handle;
    }

    function finishFileGroupDrag() {
        fileGroupDragState?.sourceElement?.classList.remove('file-group-dragging');
        fileGroupDragState = null;
        clearFileGroupDropIndicators();
    }

    function clearFileGroupDropIndicators(options = {}) {
        if (!options.keepPreview) {
            removeFileGroupDropPreview();
        }
        document.querySelectorAll('.file-group-drop-before, .file-group-drop-after, .file-group-drop-inside').forEach((node) => {
            node.classList.remove('file-group-drop-before', 'file-group-drop-after', 'file-group-drop-inside');
        });
        fileGroupActiveDropTargetNode = null;
        fileGroupActiveDropPosition = '';
    }

    function markFileGroupDropTarget(node, position) {
        if (!node || !position) {
            clearFileGroupDropIndicators();
            return;
        }
        const normalizedPosition = position === 'before' || position === 'after' ? position : 'inside';
        if (fileGroupActiveDropTargetNode === node && fileGroupActiveDropPosition === normalizedPosition) {
            node.classList.add(`file-group-drop-${normalizedPosition}`);
            placeFileGroupDropPreview(node, normalizedPosition);
            return;
        }
        clearFileGroupDropIndicators({ keepPreview: true });
        fileGroupActiveDropTargetNode = node;
        fileGroupActiveDropPosition = normalizedPosition;
        node.classList.add(`file-group-drop-${normalizedPosition}`);
        placeFileGroupDropPreview(node, normalizedPosition);
    }

    function configFileDropIndex(group, targetFile, placeAfter, draggedFile) {
        const files = (group?.files || [])
            .map((item) => item?.path)
            .filter((path) => path && path !== draggedFile);
        const targetIndex = files.indexOf(targetFile);
        if (targetIndex < 0) return files.length;
        return targetIndex + (placeAfter ? 1 : 0);
    }

    function configGroupDropIndex(groups, targetGroupId, placeAfter, draggedGroupId) {
        const ids = (groups || [])
            .map((group) => group?.id)
            .filter((id) => id && id !== draggedGroupId);
        const targetIndex = ids.indexOf(targetGroupId);
        if (targetIndex < 0) return ids.length;
        return targetIndex + (placeAfter ? 1 : 0);
    }

    function fileGroupContainsRelatedTarget(node, event) {
        const related = event?.relatedTarget;
        return related instanceof Node && node.contains(related);
    }

    function setupFileGroupRowDropTarget(row, group, targetFile, options) {
        registerFileGroupDropTarget(row, ({ payload, y }) => {
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
            if (payload.file === targetFile && payload.groupId === group?.id) return null;
            if (!options.canDropToGroup(group, payload)) return null;
            const rect = row.getBoundingClientRect();
            const placeAfter = y > rect.top + rect.height / 2;
            const position = placeAfter ? 'after' : 'before';
            return {
                position,
                drop: async () => {
                    const index = configFileDropIndex(group, targetFile, placeAfter, payload.file);
                    await options.onDrop(payload, group.id, index);
                },
            };
        });
        const updateDropTarget = (event) => {
            const payload = fileGroupDragState;
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
            if (payload.file === targetFile && payload.groupId === group?.id) return;
            if (!options.canDropToGroup(group, payload)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            const rect = row.getBoundingClientRect();
            const placeAfter = event.clientY > rect.top + rect.height / 2;
            row.dataset.dropPosition = placeAfter ? 'after' : 'before';
            markFileGroupDropTarget(row, placeAfter ? 'after' : 'before');
        };
        row.addEventListener('dragenter', updateDropTarget);
        row.addEventListener('dragover', updateDropTarget);
        row.addEventListener('dragleave', (event) => {
            if (fileGroupContainsRelatedTarget(row, event)) return;
            row.classList.remove('file-group-drop-before', 'file-group-drop-after');
        });
        row.addEventListener('drop', async (event) => {
            const payload = fileGroupDragState;
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
            if (payload.file === targetFile && payload.groupId === group?.id) return;
            if (!options.canDropToGroup(group, payload)) return;
            event.preventDefault();
            event.stopPropagation();
            const placeAfter = row.dataset.dropPosition === 'after';
            const index = configFileDropIndex(group, targetFile, placeAfter, payload.file);
            await options.onDrop(payload, group.id, index);
            finishFileGroupDrag();
        });
    }

    function setupFileGroupListDropTarget(list, group, options) {
        registerFileGroupDropTarget(list, ({ payload, origin }) => {
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
            if (originClosest(origin, options.rowSelector)) return null;
            if (!options.canDropToGroup(group, payload)) return null;
            return {
                position: 'inside',
                drop: async () => {
                    const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
                    await options.onDrop(payload, group.id, index);
                },
            };
        });
        const updateDropTarget = (event) => {
            const payload = fileGroupDragState;
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
            if (eventTargetClosest(event, options.rowSelector)) return;
            if (!options.canDropToGroup(group, payload)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            markFileGroupDropTarget(list, 'inside');
        };
        list.addEventListener('dragenter', updateDropTarget);
        list.addEventListener('dragover', updateDropTarget);
        list.addEventListener('dragleave', (event) => {
            if (fileGroupContainsRelatedTarget(list, event)) return;
            list.classList.remove('file-group-drop-inside');
        });
        list.addEventListener('drop', async (event) => {
            const payload = fileGroupDragState;
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
            if (eventTargetClosest(event, options.rowSelector)) return;
            if (!options.canDropToGroup(group, payload)) return;
            event.preventDefault();
            event.stopPropagation();
            const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
            await options.onDrop(payload, group.id, index);
            finishFileGroupDrag();
        });
    }

    function setupFileGroupHeaderDropTarget(node, group, options) {
        registerFileGroupDropTarget(node, ({ payload }) => {
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return null;
            if (!options.canDropToGroup(group, payload)) return null;
            return {
                position: 'inside',
                drop: async () => {
                    const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
                    await options.onDrop(payload, group.id, index);
                },
            };
        });
        const updateDropTarget = (event) => {
            const payload = fileGroupDragState;
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
            if (!options.canDropToGroup(group, payload)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            markFileGroupDropTarget(node, 'inside');
        };
        node.addEventListener('dragenter', updateDropTarget);
        node.addEventListener('dragover', updateDropTarget);
        node.addEventListener('dragleave', (event) => {
            if (fileGroupContainsRelatedTarget(node, event)) return;
            node.classList.remove('file-group-drop-inside');
        });
        node.addEventListener('drop', async (event) => {
            const payload = fileGroupDragState;
            if (!payload || payload.target !== 'file' || payload.scope !== options.scope) return;
            if (!options.canDropToGroup(group, payload)) return;
            event.preventDefault();
            event.stopPropagation();
            const index = (group?.files || []).filter((item) => item?.path && item.path !== payload.file).length;
            await options.onDrop(payload, group.id, index);
            finishFileGroupDrag();
        });
    }

    function setupConfigGroupDropTarget(node, group, options) {
        registerFileGroupDropTarget(node, ({ payload, y }) => {
            if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return null;
            if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return null;
            const rect = node.getBoundingClientRect();
            const placeAfter = y > rect.top + rect.height / 2;
            const position = placeAfter ? 'after' : 'before';
            return {
                position,
                drop: async () => {
                    const index = configGroupDropIndex(options.getSortableGroups(), group.id, placeAfter, payload.groupId);
                    await options.onDrop(payload, index);
                },
            };
        });
        const updateDropTarget = (event) => {
            const payload = fileGroupDragState;
            if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return;
            if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            const rect = node.getBoundingClientRect();
            const placeAfter = event.clientY > rect.top + rect.height / 2;
            node.dataset.dropPosition = placeAfter ? 'after' : 'before';
            markFileGroupDropTarget(node, placeAfter ? 'after' : 'before');
        };
        node.addEventListener('dragenter', updateDropTarget);
        node.addEventListener('dragover', updateDropTarget);
        node.addEventListener('dragleave', (event) => {
            if (fileGroupContainsRelatedTarget(node, event)) return;
            node.classList.remove('file-group-drop-before', 'file-group-drop-after');
        });
        node.addEventListener('drop', async (event) => {
            const payload = fileGroupDragState;
            if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return;
            if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return;
            event.preventDefault();
            event.stopPropagation();
            const placeAfter = node.dataset.dropPosition === 'after';
            const index = configGroupDropIndex(options.getSortableGroups(), group.id, placeAfter, payload.groupId);
            await options.onDrop(payload, index);
            finishFileGroupDrag();
        });
    }

    function createDatasetPresetGroupNode(group, stored) {
        const files = group.files || [];
        const details = document.createElement('details');
        details.className = ['dataset-preset-group', !files.length ? 'empty' : '', group.locked ? 'readonly' : ''].filter(Boolean).join(' ');
        details.dataset.groupId = group.id || '';
        const containsSelected = files.some((preset) => preset.path === datasetPresetState.selectedFile);
        const shouldForceOpen = containsSelected || Boolean(datasetPresetState.search.trim());
        const defaultOpen = isUnfiledDatasetGroup(group);
        details.open = shouldForceOpen || (stored[group.id] ?? defaultOpen);
        details.addEventListener('toggle', () => {
            const next = readDatasetPresetGroupState();
            next[group.id] = details.open;
            writeDatasetPresetGroupState(next);
        });

        const summary = document.createElement('summary');
        const groupHandle = createDatasetPresetGroupDragHandle(group, details);
        if (groupHandle) summary.appendChild(groupHandle);
        const title = document.createElement('span');
        title.className = 'dataset-preset-group-title';
        title.textContent = `${group.label || group.id || '数据集分组'} (${(group.files || []).length})`;
        summary.appendChild(title);
        const actions = createDatasetPresetGroupActions(group);
        if (actions) summary.appendChild(actions);
        if (group.locked || group.user_group_locked) {
            const badge = document.createElement('em');
            badge.textContent = group.user_group_locked ? '分组锁定' : '只读';
            summary.appendChild(badge);
        }
        setupFileGroupHeaderDropTarget(summary, group, datasetPresetDragOptions());
        details.appendChild(summary);

        const list = document.createElement('div');
        list.className = 'dataset-preset-group-list';
        setupFileGroupListDropTarget(list, group, datasetPresetDragOptions());
        if (!files.length) {
            const empty = document.createElement('div');
            empty.className = 'dataset-preset-empty dataset-preset-empty-state';
            empty.textContent = datasetPresetState.search.trim() ? '此分组没有匹配项。' : '空分组，可将数据集预设移动到这里。';
            list.appendChild(empty);
        }
        files.forEach((preset) => {
            list.appendChild(createDatasetPresetGroupFileRow(preset, group));
        });
        details.appendChild(list);
        setupConfigGroupDropTarget(details, group, datasetPresetGroupDragOptions());
        return details;
    }

    function getSortableDatasetPresetGroups() {
        return (datasetPresetState.groups || [])
            .filter((group) => group.id && !group.system_locked && !group.locked && !group.user_group_locked && !isUnfiledDatasetGroup(group));
    }

    function createDatasetPresetGroupDragHandle(group, details) {
        const disabled = !isDatasetPresetGroupDraggable(group);
        return createFileGroupDragHandle({
            target: 'group',
            scope: 'dataset',
            groupId: group.id,
            sourceElement: details,
            canDrag: () => isDatasetPresetGroupDraggable(group),
            blockedMessage: () => setDatasetPresetStatus('该数据集分组不能拖动排序', 'error'),
        }, {
            disabled,
            label: `拖动数据集分组 ${group.label || group.id}`,
            title: disabled ? '该数据集分组不能拖动排序' : '拖动调整数据集分组顺序',
        });
    }

    function isDatasetPresetGroupDraggable(group) {
        return Boolean(group?.id && !datasetPresetState.search.trim() && !group.system_locked && !group.locked && !group.user_group_locked && !isUnfiledDatasetGroup(group));
    }

    function isDatasetPresetFileDraggable(preset, group) {
        return Boolean(preset?.path && group?.id && !datasetPresetState.search.trim() && !preset.readonly);
    }

    function datasetPresetCanDropToGroup(group, payload) {
        return Boolean(
            payload?.file &&
            group?.id &&
            !datasetPresetState.search.trim() &&
            (group.kind === 'dataset' || group.id === 'datasets' || group.id === 'unfiled_datasets') &&
            group.movable &&
            !group.locked &&
            !group.user_group_locked
        );
    }

    function datasetPresetDragOptions() {
        return {
            scope: 'dataset',
            rowSelector: '.dataset-preset-row',
            canDropToGroup: datasetPresetCanDropToGroup,
            onDrop: placeDatasetPresetFile,
        };
    }

    function datasetPresetGroupDragOptions() {
        return {
            scope: 'dataset',
            getSortableGroups: () => getSortableDatasetPresetGroups(),
            canDropOnGroup: (group) => isDatasetPresetGroupDraggable(group),
            onDrop: placeDatasetPresetGroup,
        };
    }

    function createDatasetPresetGroupActions(group) {
        const wrap = document.createElement('span');
        wrap.className = 'dataset-preset-group-actions';
        if (group.renamable) {
            wrap.appendChild(createDatasetPresetGroupActionButton('重命名', () => renameDatasetPresetGroup(group), {
                title: '重命名这个数据集分组',
            }));
        }
        if (group.deletable) {
            wrap.appendChild(createDatasetPresetGroupActionButton('删除分组', () => deleteDatasetPresetGroup(group), {
                title: `删除分组“${group.label || group.id}”；不会删除其中的 TOML 文件`,
                danger: true,
            }));
        }
        return wrap.childElementCount ? wrap : null;
    }

    function createDatasetPresetGroupActionButton(label, handler, options = {}) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'dataset-preset-group-action-btn',
            options.variant ? `dataset-preset-group-action-btn-${options.variant}` : '',
            options.danger ? 'danger' : '',
        ].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.disabled = Boolean(options.disabled);
        btn.title = options.title || label;
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            if (!btn.disabled) handler();
        });
        return btn;
    }

    function createDatasetPresetGroupFileRow(preset, group) {
        const row = document.createElement('div');
        row.className = 'dataset-preset-row';
        row.dataset.file = preset.path;
        row.dataset.groupId = group?.id || '';
        setupFileGroupRowDropTarget(row, group, preset.path, datasetPresetDragOptions());

        const dragHandle = createFileGroupDragHandle({
            target: 'file',
            scope: 'dataset',
            file: preset.path,
            groupId: group?.id || '',
            sourceElement: row,
            canDrag: () => isDatasetPresetFileDraggable(preset, group),
            blockedMessage: () => {
                const message = datasetPresetState.search.trim()
                    ? '筛选数据集预设时不能拖动排序，请先清空搜索'
                    : '该数据集预设不能拖动排序';
                setDatasetPresetStatus(message, 'error');
            },
        }, {
            disabled: !isDatasetPresetFileDraggable(preset, group),
            label: `拖动数据集预设 ${preset.label || preset.filename || preset.path}`,
            title: isDatasetPresetFileDraggable(preset, group)
                ? '拖动调整数据集预设位置或移动到其他分组'
                : '当前数据集预设不能拖动',
        });
        row.appendChild(dragHandle);

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'dataset-preset-item',
            preset.path === datasetPresetState.selectedFile ? 'active' : '',
            preset.readonly ? 'readonly' : '',
        ].filter(Boolean).join(' ');
        btn.dataset.file = preset.path;
        const summary = preset.summary || {};
        btn.innerHTML = [
            `<strong>${escapeHtml(preset.label || preset.filename || preset.path)}</strong>`,
            `<span>${escapeHtml(preset.path)}</span>`,
            `<small>${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}${preset.readonly ? ' · 只读' : ''}</small>`,
        ].join('');
        btn.addEventListener('click', () => loadDatasetPreset(preset.path));
        row.appendChild(btn);

        return row;
    }

    function readDatasetPresetGroupState() {
        try {
            return JSON.parse(localStorage.getItem(DATASET_PRESET_GROUP_STATE_KEY) || '{}') || {};
        } catch (_) {
            return {};
        }
    }

    function writeDatasetPresetGroupState(state) {
        try {
            localStorage.setItem(DATASET_PRESET_GROUP_STATE_KEY, JSON.stringify(state || {}));
        } catch (_) {
            // 忽略本地存储不可用；分组折叠状态不是关键数据。
        }
    }

    function renderDatasetPresetHeader() {
        const header = document.getElementById('dataset-preset-header');
        updateDatasetPresetPageSummary();
        if (!header) return;
        const file = datasetPresetState.selectedFile;
        const preset = datasetPresetByFile(file);
        const summary = preset?.summary || {};
        header.innerHTML = '';
        const title = document.createElement('div');
        title.className = 'dataset-preset-title-block';
        const datasetCount = Number(summary.dataset_count || datasetPresetState.datasets.length || 0);
        const repeatTotal = Number(summary.repeat_total || datasetPresetState.datasets.reduce((sum, row) => sum + Number(row.num_repeats || 1), 0) || 0);
        const status = datasetPresetState.dirty ? '未保存' : (datasetPresetState.readonly ? '只读' : '已同步');
        title.innerHTML = [
            '<span class="dataset-preset-breadcrumb">CONFIGS / DATASETS</span>',
            `<strong>${escapeHtml(preset?.label || preset?.filename || file || '新数据集预设')}</strong>`,
            `<span>${escapeHtml(file || '尚未保存')}</span>`,
        ].join('');
        const meta = document.createElement('div');
        meta.className = 'dataset-preset-meta';
        [
            ['状态', status, datasetPresetState.dirty ? 'warn' : (datasetPresetState.readonly ? 'lock' : 'ok')],
            ['子集', datasetCount],
            ['重复', repeatTotal],
            ['分辨率', summary.resolution ? `${summary.resolution}px` : '-'],
        ].forEach(([label, value, tone]) => {
            const stat = document.createElement('span');
            stat.className = ['dataset-preset-stat', tone ? `dataset-preset-stat-${tone}` : ''].filter(Boolean).join(' ');
            stat.innerHTML = `<small>${escapeHtml(String(label))}</small><strong>${escapeHtml(String(value))}</strong>`;
            meta.appendChild(stat);
        });
        header.append(title, meta);
        if (datasetPresetState.status) {
            const statusEl = document.createElement('span');
            statusEl.className = 'dataset-preset-status dataset-preset-inline-status';
            statusEl.textContent = datasetPresetState.status;
            header.appendChild(statusEl);
        }
        updateDatasetPresetActionState();
    }

    function updateDatasetPresetActionState() {
        const saveBtn = document.getElementById('btn-save-dataset-preset');
        if (saveBtn) {
            saveBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile || !datasetPresetState.dirty;
            saveBtn.title = datasetPresetState.readonly
                ? '系统数据集预设只读，请复制后编辑'
                : (datasetPresetState.dirty ? '保存当前数据集预设' : '当前数据集预设没有未保存修改');
        }
        const deleteBtn = document.getElementById('btn-delete-dataset-preset');
        if (deleteBtn) {
            deleteBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile;
            deleteBtn.title = datasetPresetState.readonly ? '系统数据集预设不能删除' : '只删除 TOML 预设，不删除图片或缓存目录';
        }
        const renameBtn = document.getElementById('btn-rename-dataset-preset');
        if (renameBtn) {
            renameBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile;
        }
        const copyBtn = document.getElementById('btn-copy-dataset-preset');
        if (copyBtn) copyBtn.disabled = !datasetPresetState.selectedFile;
        const exportBtn = document.getElementById('btn-export-dataset-preset');
        if (exportBtn) exportBtn.disabled = !datasetPresetState.selectedFile;
    }

    function renderDatasetEditor(existingPanel = null) {
        const panel = existingPanel || document.getElementById('dataset-editor');
        if (!panel) return;
        panel.innerHTML = '';
        const state = datasetEditorStateForActivePanel();

        const header = document.createElement('div');
        header.className = 'dataset-editor-header';
        const title = document.createElement('div');
        title.innerHTML = '<strong>多数据集路径</strong><span>每一行是一组数据：填写原始图路径、重复次数和分桶参数；缩放图与 LoRA 缓存会在训练运行目录中自动生成。</span>';
        const actions = document.createElement('div');
        actions.className = 'dataset-editor-actions';
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small';
        addBtn.textContent = '添加数据集';
        addBtn.title = '新增一组数据集路径。适合把多个角色、画风或批次一起训练，并给每组设置独立重复次数。';
        addBtn.addEventListener('click', addDatasetEditorRow);
        actions.append(addBtn);
        header.append(title, actions);
        panel.appendChild(header);

        if (state.loading) {
            const loading = document.createElement('p');
            loading.className = 'dataset-editor-message';
            loading.textContent = '正在读取数据集配置...';
            panel.appendChild(loading);
            return;
        }
        if (state.error) {
            const error = document.createElement('p');
            error.className = 'dataset-editor-message error';
            error.textContent = state.error;
            panel.appendChild(error);
        }

        const rows = state.datasets.length
            ? state.datasets
            : normalizeDatasetEditorRows([{
                source_dir: currentConfig.source_image_dir || '',
                image_dir: currentConfig.resized_image_dir || '',
                cache_dir: currentConfig.lora_cache_dir || '',
                num_repeats: 1,
                settings: normalizeDatasetDefaults(state.defaults || {}),
            }]);
        if (!state.datasets.length) {
            setActiveDatasetRows(rows);
        }

        panel.appendChild(createDatasetDefaultsEditor());

        const list = document.createElement('div');
        list.className = 'dataset-editor-list';
        rows.forEach((row, index) => {
            list.appendChild(createDatasetEditorItem(row, index));
        });
        panel.appendChild(list);

        const footer = document.createElement('div');
        footer.className = 'dataset-editor-footer';
        const configPath = document.createElement('code');
        configPath.textContent = activeDatasetFileLabel();
        const dirty = document.createElement('span');
        dirty.className = activeDatasetDirty() ? 'dataset-editor-dirty active' : 'dataset-editor-dirty';
        dirty.textContent = activeDatasetDirty() ? '有未保存的数据集修改' : '数据集路径已同步';
        footer.append(configPath, dirty);
        panel.appendChild(footer);
        if (isDatasetTabActive()) {
            renderDatasetPresetHeader();
        }
    }

    function datasetEditorStateForActivePanel() {
        return isDatasetTabActive() ? datasetPresetState : datasetEditorState;
    }

    function isDatasetTabActive() {
        return Boolean(document.getElementById('tab-datasets')?.classList.contains('active'));
    }

    function setActiveDatasetRows(rows) {
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
    }

    function activeDatasetFileLabel() {
        if (isDatasetTabActive()) {
            return datasetPresetState.selectedFile || '保存后生成 configs/datasets/<名称>.toml';
        }
        return datasetEditorState.dataset_config || currentConfig.dataset_config || '保存后自动生成 configs/datasets/<当前配置>.toml';
    }

    function activeDatasetDirty() {
        return isDatasetTabActive() ? datasetPresetState.dirty : datasetEditorState.dirty;
    }

    function createDatasetDefaultsEditor() {
        const state = datasetEditorStateForActivePanel();
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        if (isDatasetTabActive()) {
            datasetPresetState.defaults = defaults;
        } else {
            datasetEditorState.defaults = defaults;
        }
        const wrap = document.createElement('div');
        wrap.className = 'dataset-defaults-list';

        const heading = document.createElement('div');
        heading.className = 'dataset-defaults-heading';
        heading.innerHTML = '<strong>通用标注设置</strong><span>这里只保留 keep_tokens；文本标注扩展名等兼容项在每组数据集的高级区配置。</span>';
        wrap.appendChild(heading);

        const fields = [
            ['keep_tokens', 'number'],
        ];

        for (const [key, type, layout] of fields) {
            const row = document.createElement('div');
            row.className = [
                'dataset-config-field',
                layout === 'wide' ? 'wide' : '',
                layout === 'switch' ? 'switch' : '',
            ].filter(Boolean).join(' ');
            row.dataset.key = key;

            const label = document.createElement('label');
            label.className = 'dataset-config-label';
            const nameSpan = document.createElement('span');
            nameSpan.className = 'field-name';
            nameSpan.textContent = datasetConfigLabel(key);
            nameSpan.title = key;
            label.appendChild(nameSpan);

            const btn = document.createElement('button');
            btn.className = 'info-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '查看填写建议、好处、代价、风险和推荐';
            btn.addEventListener('click', () => {
                btn.classList.toggle('active');
                row.querySelector('.field-help')?.classList.toggle('visible');
            });
            label.appendChild(btn);
            row.appendChild(label);

            const input = createDatasetConfigInput(key, type, defaults);
            row.appendChild(input);

            const helpDiv = document.createElement('div');
            helpDiv.className = 'field-help';
            helpDiv.appendChild(createHelpContent(key, datasetConfigValue(key, defaults)));
            row.appendChild(helpDiv);
            wrap.appendChild(row);
        }
        return wrap;
    }

    function createDatasetConfigInput(key, type, defaults) {
        if (type === 'switch') {
            return createDatasetConfigSwitch(key, defaults);
        }

        let input;
        if (type === 'select') {
            input = document.createElement('select');
            const options = key === 'enable_bucket'
                ? [[true, '启用'], [false, '关闭']]
                : [[false, '允许放大'], [true, '不放大小图']];
            const current = Boolean(defaults[key]);
            for (const [value, label] of options) {
                const opt = document.createElement('option');
                opt.value = value ? 'true' : 'false';
                opt.textContent = label;
                opt.selected = value === current;
                input.appendChild(opt);
            }
            input.dataset.valueType = 'boolean';
        } else {
            input = document.createElement('input');
            input.type = type;
            input.dataset.valueType = type === 'number' ? 'number' : 'string';
            input.value = datasetConfigValue(key, defaults);
            if (type === 'number') {
                input.min = '0';
                input.step = key === 'validation_split' ? '0.001' : (key === 'resolution' || key.endsWith('_reso') || key === 'bucket_reso_steps' ? '16' : '1');
            }
        }
        input.className = 'field-input dataset-config-input';
        input.dataset.key = key;
        input.addEventListener('input', () => updateDatasetConfigValue(key, input));
        input.addEventListener('change', () => updateDatasetConfigValue(key, input));
        return input;
    }

    function createDatasetConfigSwitch(key, defaults) {
        const checked = Boolean(defaults[key]);
        const wrap = document.createElement('label');
        wrap.className = ['dataset-json-switch', checked ? 'enabled' : ''].filter(Boolean).join(' ');

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.className = 'dataset-json-switch-input';
        input.dataset.key = key;
        input.checked = checked;
        input.setAttribute('aria-label', datasetConfigLabel(key));

        const copy = document.createElement('span');
        copy.className = 'dataset-json-switch-copy';
        const title = document.createElement('span');
        title.className = 'dataset-json-switch-title';
        title.textContent = 'JSON 标注';
        const desc = document.createElement('span');
        desc.className = 'dataset-json-switch-desc';
        desc.textContent = '.json 优先，失败回退 .txt';
        copy.append(title, desc);

        const state = document.createElement('span');
        state.className = 'dataset-json-switch-state';
        state.textContent = checked ? '已启用' : '已关闭';

        const track = document.createElement('span');
        track.className = 'dataset-json-switch-track';
        track.setAttribute('aria-hidden', 'true');
        const thumb = document.createElement('span');
        thumb.className = 'dataset-json-switch-thumb';
        track.appendChild(thumb);

        input.addEventListener('change', () => {
            wrap.classList.toggle('enabled', input.checked);
            state.textContent = input.checked ? '已启用' : '已关闭';
            updateDatasetConfigValue(key, input);
        });

        wrap.append(input, copy, state, track);
        return wrap;
    }

    function datasetConfigLabel(key) {
        const labels = {
            resolution: '分辨率',
            enable_bucket: '启用长宽比分桶',
            min_bucket_reso: '最小桶边长',
            max_bucket_reso: '最大桶边长',
            bucket_reso_steps: '桶尺寸步长',
            bucket_no_upscale: '小图放大',
            validation_split: '验证集比例',
            validation_split_num: '固定验证数量',
            validation_seed: '验证随机种子',
            caption_extension: '文本标注扩展名',
            keep_tokens: '保留前置 token',
            prefer_json_caption: '优先 JSON 标注',
            caption_source_mode: '标注来源',
        };
        return `${labels[key] || FIELD_LABEL_ZH[key] || key} / ${key}`;
    }

    function datasetConfigValue(key, defaults) {
        return defaults[key] ?? '';
    }

	    function updateDatasetConfigValue(key, input) {
	        updateDatasetDefault(key, input);
	    }

	    function datasetEditorDragRows() {
	        return normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
	    }

	    function datasetEditorCanDrag() {
	        return datasetEditorDragRows().length > 1;
	    }

	    function datasetEditorDragLabel(index) {
	        const row = datasetEditorDragRows()[index] || {};
	        const path = String(row.source_dir || row.image_dir || '').trim();
	        return path ? compactPathLabel(path) : `SUBSET ${index + 1}`;
	    }

	    function createDatasetEditorDragImage(index) {
	        const image = document.createElement('div');
	        image.className = 'dataset-editor-drag-image';
	        image.textContent = datasetEditorDragLabel(index);
	        document.body.appendChild(image);
	        return image;
	    }

	    function removeDatasetEditorDragImage(image) {
	        if (image?.parentNode) image.parentNode.removeChild(image);
	    }

	    function moveDatasetEditorDragImage(image, x, y) {
	        if (!image) return;
	        image.style.left = `${x + 14}px`;
	        image.style.top = `${y + 14}px`;
	    }

	    function beginDatasetEditorDrag(index, item, handle) {
	        datasetEditorDragState = { index, sourceElement: item, handle };
	        item?.classList.add('dataset-editor-item-dragging');
	        handle?.classList.add('dragging');
	        document.body.classList.add('dataset-editor-pointer-drag-active');
	    }

	    function clearDatasetEditorDropIndicators() {
	        document.querySelectorAll('.dataset-editor-drop-before, .dataset-editor-drop-after').forEach((node) => {
	            node.classList.remove('dataset-editor-drop-before', 'dataset-editor-drop-after');
	        });
	    }

	    function finishDatasetEditorDrag() {
	        datasetEditorDragState?.sourceElement?.classList.remove('dataset-editor-item-dragging');
	        datasetEditorDragState?.handle?.classList.remove('dragging');
	        document.body.classList.remove('dataset-editor-pointer-drag-active');
	        datasetEditorDragState = null;
	        clearDatasetEditorDropIndicators();
	    }

	    function datasetEditorDropTargetFromPoint(x, y) {
	        const items = [...document.querySelectorAll('#dataset-editor .dataset-editor-item')];
	        if (!items.length) return null;
	        let best = null;
	        for (const item of items) {
	            const targetIndex = Number.parseInt(item.dataset.index || '-1', 10);
	            if (!Number.isInteger(targetIndex) || targetIndex < 0) continue;
	            if (targetIndex === datasetEditorDragState?.index) continue;
	            const rect = item.getBoundingClientRect();
	            if (rect.width <= 0 || rect.height <= 0) continue;
	            const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
	            const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
	            const distance = Math.hypot(dx, dy);
	            if (distance > Math.max(30, rect.height * 0.8)) continue;
	            const placeAfter = y > rect.top + rect.height / 2;
	            if (!best || distance < best.distance) {
	                best = { node: item, targetIndex, placeAfter, distance };
	            }
	        }
	        return best;
	    }

	    function markDatasetEditorDropTarget(target) {
	        clearDatasetEditorDropIndicators();
	        if (!target?.node) return;
	        target.node.classList.add(target.placeAfter ? 'dataset-editor-drop-after' : 'dataset-editor-drop-before');
	    }

	    function datasetEditorEventPoint(event) {
	        const touch = event.changedTouches?.[0] || event.touches?.[0];
	        const x = touch?.clientX ?? event.clientX;
	        const y = touch?.clientY ?? event.clientY;
	        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
	        return { x, y };
	    }

	    function finishDatasetEditorPointerDrag(commit = false) {
	        const drag = datasetEditorPointerDrag;
	        if (!drag) return;
	        document.removeEventListener('pointermove', drag.onPointerMove);
	        document.removeEventListener('pointerup', drag.onPointerUp);
	        document.removeEventListener('pointercancel', drag.onPointerCancel);
	        document.removeEventListener('mousemove', drag.onMouseMove);
	        document.removeEventListener('mouseup', drag.onMouseUp);
	        document.removeEventListener('touchmove', drag.onTouchMove);
	        document.removeEventListener('touchend', drag.onTouchEnd);
	        document.removeEventListener('touchcancel', drag.onTouchCancel);
	        document.removeEventListener('keydown', drag.onKeydown);
	        try {
	            if (drag.pointerId !== null && drag.pointerId !== undefined) {
	                drag.handle.releasePointerCapture?.(drag.pointerId);
	            }
	        } catch (e) {
	            /* 指针捕获可能已被浏览器释放，继续清理拖拽态。 */
	        }
	        removeDatasetEditorDragImage(drag.image);
	        const target = commit && drag.active ? drag.currentDrop : null;
	        datasetEditorPointerDrag = null;
	        finishDatasetEditorDrag();
	        if (target) {
	            moveDatasetEditorRow(drag.index, target.targetIndex, target.placeAfter);
	        }
	    }

	    function startDatasetEditorFallbackDrag(event, index, item, handle, options = {}) {
	        if (!datasetEditorCanDrag() || datasetEditorPointerDrag) return;
	        if ((options.pointer || options.mouse) && 'button' in event && event.button !== 0) return;
	        if (options.pointer && event.isPrimary === false) return;
	        const startPoint = datasetEditorEventPoint(event);
	        if (!startPoint) return;
	        event.preventDefault();
	        event.stopPropagation();
	        const pointerId = options.pointer ? event.pointerId : null;
	        const drag = {
	            index,
	            item,
	            handle,
	            pointerId,
	            startX: startPoint.x,
	            startY: startPoint.y,
	            active: false,
	            image: null,
	            currentDrop: null,
	        };
	        const moveDrag = (moveEvent) => {
	            const point = datasetEditorEventPoint(moveEvent);
	            if (!point) return;
	            const distance = Math.hypot(point.x - drag.startX, point.y - drag.startY);
	            if (!drag.active) {
	                if (distance < 4) return;
	                beginDatasetEditorDrag(index, item, handle);
	                drag.active = true;
	                drag.image = createDatasetEditorDragImage(index);
	            }
	            moveEvent.preventDefault();
	            moveEvent.stopPropagation();
	            moveDatasetEditorDragImage(drag.image, point.x, point.y);
	            autoScrollFileGroupPointerDrag(point.x, point.y);
	            drag.currentDrop = datasetEditorDropTargetFromPoint(point.x, point.y);
	            markDatasetEditorDropTarget(drag.currentDrop);
	        };
	        drag.onPointerMove = (moveEvent) => {
	            if (moveEvent.pointerId !== pointerId) return;
	            moveDrag(moveEvent);
	        };
	        drag.onPointerUp = (upEvent) => {
	            if (upEvent.pointerId !== pointerId) return;
	            upEvent.preventDefault();
	            upEvent.stopPropagation();
	            finishDatasetEditorPointerDrag(true);
	        };
	        drag.onPointerCancel = (cancelEvent) => {
	            if (cancelEvent.pointerId !== pointerId) return;
	            finishDatasetEditorPointerDrag(false);
	        };
	        drag.onMouseMove = (moveEvent) => moveDrag(moveEvent);
	        drag.onMouseUp = (upEvent) => {
	            upEvent.preventDefault();
	            upEvent.stopPropagation();
	            finishDatasetEditorPointerDrag(true);
	        };
	        drag.onTouchMove = (moveEvent) => moveDrag(moveEvent);
	        drag.onTouchEnd = (touchEvent) => {
	            touchEvent.preventDefault();
	            touchEvent.stopPropagation();
	            finishDatasetEditorPointerDrag(true);
	        };
	        drag.onTouchCancel = () => finishDatasetEditorPointerDrag(false);
	        drag.onKeydown = (keyEvent) => {
	            if (keyEvent.key === 'Escape') finishDatasetEditorPointerDrag(false);
	        };
	        datasetEditorPointerDrag = drag;
	        if (options.pointer) {
	            try {
	                handle.setPointerCapture?.(pointerId);
	            } catch (e) {
	                /* 某些浏览器禁用按钮指针捕获，文档级监听仍可兜底。 */
	            }
	            document.addEventListener('pointermove', drag.onPointerMove, { passive: false });
	            document.addEventListener('pointerup', drag.onPointerUp, { passive: false });
	            document.addEventListener('pointercancel', drag.onPointerCancel, { passive: false });
	        } else if (options.touch) {
	            document.addEventListener('touchmove', drag.onTouchMove, { passive: false });
	            document.addEventListener('touchend', drag.onTouchEnd, { passive: false });
	            document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false });
	        } else {
	            document.addEventListener('mousemove', drag.onMouseMove, { passive: false });
	            document.addEventListener('mouseup', drag.onMouseUp, { passive: false });
	        }
	        document.addEventListener('keydown', drag.onKeydown);
	    }

	    function startDatasetEditorPointerDrag(event, index, item, handle) {
	        startDatasetEditorFallbackDrag(event, index, item, handle, { pointer: true });
	    }

	    function startDatasetEditorMouseDrag(event, index, item, handle) {
	        startDatasetEditorFallbackDrag(event, index, item, handle, { mouse: true });
	    }

	    function startDatasetEditorTouchDrag(event, index, item, handle) {
	        startDatasetEditorFallbackDrag(event, index, item, handle, { touch: true });
	    }

	    function createDatasetEditorDragHandle(index, item) {
	        const handle = document.createElement('button');
	        const disabled = !datasetEditorCanDrag();
	        handle.type = 'button';
	        handle.className = ['dataset-editor-drag-handle', disabled ? 'disabled' : ''].filter(Boolean).join(' ');
	        handle.textContent = '⋮⋮';
	        handle.title = disabled ? '至少两组数据集时可以拖动排序' : '拖动排序；也可用 Alt+方向键移动';
	        handle.setAttribute('aria-label', `拖动排序第 ${index + 1} 组数据集`);
	        handle.draggable = !disabled;
	        handle.tabIndex = disabled ? -1 : 0;
	        handle.addEventListener('click', (event) => {
	            event.preventDefault();
	            event.stopPropagation();
	        });
	        handle.addEventListener('keydown', (event) => {
	            if (disabled || !event.altKey || !['ArrowUp', 'ArrowDown'].includes(event.key)) return;
	            event.preventDefault();
	            const targetIndex = event.key === 'ArrowUp' ? index - 1 : index + 1;
	            moveDatasetEditorRowToIndex(index, targetIndex);
	        });
	        handle.addEventListener('pointerdown', (event) => startDatasetEditorPointerDrag(event, index, item, handle));
	        handle.addEventListener('mousedown', (event) => {
	            event.stopPropagation();
	            startDatasetEditorMouseDrag(event, index, item, handle);
	        });
	        handle.addEventListener('touchstart', (event) => {
	            event.stopPropagation();
	            startDatasetEditorTouchDrag(event, index, item, handle);
	        }, { passive: false });
	        handle.addEventListener('dragstart', (event) => {
	            if (datasetEditorPointerDrag) finishDatasetEditorPointerDrag(false);
	            if (disabled) {
	                event.preventDefault();
	                return;
	            }
	            event.stopPropagation();
	            beginDatasetEditorDrag(index, item, handle);
	            if (event.dataTransfer) {
	                try {
	                    event.dataTransfer.setData('text/plain', String(index));
	                    event.dataTransfer.setData('application/x-anima-dataset-row', String(index));
	                    event.dataTransfer.effectAllowed = 'move';
	                    const image = createDatasetEditorDragImage(index);
	                    event.dataTransfer.setDragImage(image, 12, 12);
	                    window.setTimeout(() => removeDatasetEditorDragImage(image), 0);
	                } catch (e) {
	                    /* 原生 DataTransfer 失败时，pointer 兜底仍可完成排序。 */
	                }
	            }
	        });
	        handle.addEventListener('dragend', () => finishDatasetEditorDrag());
	        return handle;
	    }

	    function setupDatasetEditorItemDropTarget(item, targetIndex) {
	        const updateDropTarget = (event) => {
	            const sourceIndex = datasetEditorDragState?.index;
	            if (!Number.isInteger(sourceIndex) || sourceIndex === targetIndex) return;
	            event.preventDefault();
	            event.stopPropagation();
	            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
	            const rect = item.getBoundingClientRect();
	            const placeAfter = event.clientY > rect.top + rect.height / 2;
	            item.dataset.dropPosition = placeAfter ? 'after' : 'before';
	            markDatasetEditorDropTarget({ node: item, placeAfter });
	        };
	        item.addEventListener('dragenter', updateDropTarget);
	        item.addEventListener('dragover', updateDropTarget);
	        item.addEventListener('dragleave', (event) => {
	            if (fileGroupContainsRelatedTarget(item, event)) return;
	            item.classList.remove('dataset-editor-drop-before', 'dataset-editor-drop-after');
	        });
	        item.addEventListener('drop', (event) => {
	            const sourceIndex = datasetEditorDragState?.index;
	            if (!Number.isInteger(sourceIndex) || sourceIndex === targetIndex) return;
	            event.preventDefault();
	            event.stopPropagation();
	            const placeAfter = item.dataset.dropPosition === 'after';
	            finishDatasetEditorDrag();
	            moveDatasetEditorRow(sourceIndex, targetIndex, placeAfter);
	        });
	    }

	    function createDatasetEditorItem(row, index) {
	        const item = document.createElement('div');
	        item.className = 'dataset-editor-item';
	        item.dataset.index = String(index);
	        setupDatasetEditorItemDropTarget(item, index);
	        item.append(
	            createDatasetEditorRow(row, index, item),
	            createDatasetExperimentalFeaturesEditor(row, index),
	        );
	        return item;
	    }

	    function createDatasetEditorRow(row, index, item = null) {
	        const wrap = document.createElement('div');
	        wrap.className = 'dataset-editor-row';
	        wrap.dataset.index = String(index);
	        const head = document.createElement('div');
	        head.className = 'dataset-row-head';
        const titleBox = document.createElement('div');
        titleBox.className = 'dataset-row-title';
        const titleLine = document.createElement('div');
        titleLine.className = 'dataset-row-title-line';
        const mark = document.createElement('span');
        mark.className = 'dataset-row-mark';
        mark.textContent = '{}';
        const title = document.createElement('strong');
        title.textContent = `SUBSET ${index + 1} · 数据集组`;
        const subtitle = document.createElement('span');
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const mix = normalizeNlTagMix(row.nl_tag_mix);
        const triggerClone = normalizeTriggerClone(row.trigger_clone);
        const pathPattern = String(row.path_pattern || '*').trim() || '*';
        subtitle.textContent = [
            `${settings.resolution}px`,
            `桶 ${settings.min_bucket_reso}-${settings.max_bucket_reso}/${settings.bucket_reso_steps}`,
            `重复 ${row.num_repeats || 1}`,
            row.recursive === false ? '递归关闭' : '',
            pathPattern !== '*' ? `筛选 ${pathPattern}` : '',
            captionSourceModeLabel(settings.caption_source_mode),
            mix.enabled ? nlTagMixSummary(mix) : '',
            triggerClone.enabled ? `触发克隆 x${triggerClone.num_repeats}` : '',
	        ].filter(Boolean).join(' · ');
	        const dragHandle = createDatasetEditorDragHandle(index, item);
	        titleLine.append(mark, title);
	        titleBox.append(titleLine, subtitle);
        const headActions = document.createElement('div');
        headActions.className = 'dataset-row-head-actions';
        const badges = document.createElement('div');
        badges.className = 'dataset-row-badges';
        [
            ['分辨率', `${settings.resolution}px`],
            ['重复', row.num_repeats || 1],
            ['标注', CAPTION_SOURCE_MODE_OPTIONS.find((option) => option.value === normalizeCaptionSourceMode(settings.caption_source_mode))?.label || 'Auto'],
        ].forEach(([label, value]) => {
            const badge = document.createElement('span');
            badge.className = 'dataset-row-badge';
            badge.innerHTML = `<small>${escapeHtml(String(label))}</small><strong>${escapeHtml(String(value))}</strong>`;
            badges.appendChild(badge);
        });
        headActions.appendChild(badges);
        if (isDatasetTabActive()) {
            const previewBtn = document.createElement('button');
            previewBtn.type = 'button';
            previewBtn.className = 'btn btn-small';
            previewBtn.textContent = '预览图片和标注';
            previewBtn.disabled = !datasetPresetState.selectedFile || datasetPresetState.dirty;
            previewBtn.title = previewBtn.disabled
                ? '请先保存当前数据集预设，再预览磁盘中的图片和同名标注。'
                : '打开这一组数据集的原始图预览，并读取同名 caption 标注。';
            previewBtn.addEventListener('click', () => openDatasetPreview(index));
            headActions.appendChild(previewBtn);
        }
	        head.append(dragHandle, titleBox, headActions);
        wrap.appendChild(head);

        const paths = document.createElement('div');
        paths.className = 'dataset-row-paths';
        paths.appendChild(createDatasetPathField(index, 'source_dir', '原始数据集路径', row.source_dir, 'image_dataset'));
        wrap.appendChild(paths);

        wrap.appendChild(createDatasetRowCaptionSourceModeEditor(settings, index));
        wrap.appendChild(createDatasetNlTagMixEditor(row, index));
        wrap.appendChild(createDatasetRowSettingsEditor(row, index));

        const repeat = document.createElement('label');
        repeat.className = 'dataset-repeat-field';
        const repeatText = document.createElement('span');
        repeatText.textContent = '重复次数';
        repeatText.title = '这一组图片在每轮里重复使用几次。小数据集或重点角色可以适当提高，但过高会更容易过拟合。';
        const repeatInput = document.createElement('input');
        repeatInput.type = 'number';
        repeatInput.min = '1';
        repeatInput.step = '1';
        repeatInput.value = String(row.num_repeats || 1);
        repeatInput.title = '每轮训练中这组数据的重复倍率。1 表示正常使用一次，2 表示等效看两遍。';
        repeatInput.addEventListener('input', () => updateDatasetEditorRow(index, 'num_repeats', repeatInput.value));
        repeat.append(repeatText, repeatInput);

        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'btn btn-small danger dataset-remove-btn';
        remove.textContent = '删除';
        remove.disabled = datasetEditorStateForActivePanel().datasets.length <= 1;
        remove.title = remove.disabled ? '至少保留一组数据集路径' : '从当前 dataset_config 中移除这一组路径，不会删除磁盘文件。';
        remove.addEventListener('click', () => removeDatasetEditorRow(index));
        const bottomActions = document.createElement('div');
        bottomActions.className = 'dataset-row-bottom-actions';
        bottomActions.append(repeat, remove);
        wrap.appendChild(bottomActions);
        return wrap;
    }

    function createDatasetExperimentalFeaturesEditor(row, index) {
        const panel = document.createElement('details');
        panel.className = 'dataset-experimental-features';
        panel.dataset.index = String(index);
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const clone = normalizeTriggerClone(row.trigger_clone);
        const pathPattern = String(row.path_pattern || '*').trim() || '*';
        panel.open = clone.enabled
            || row.recursive === false
            || pathPattern !== '*'
            || (settings.caption_extension && settings.caption_extension !== '.txt');

        const head = document.createElement('summary');
        head.className = 'dataset-experimental-head';
        const title = document.createElement('strong');
        title.textContent = '实验性/高级/旧功能';
        const note = document.createElement('span');
        note.textContent = `对应第 ${index + 1} 组数据集；这些选项按当前这组数据集单独保存，收纳高级兼容项、旧格式入口和需要先小范围验证的功能。`;
        head.append(title, note);

        const body = document.createElement('div');
        body.className = 'dataset-experimental-body';
        body.append(
            createDatasetExperimentalScopePicker(index),
            createDatasetPathFilterEditor(row, index),
            createDatasetTriggerCloneEditor(row, index),
            createDatasetCaptionExtensionEditor(row, index),
        );

        panel.append(head, body);
        return panel;
    }

    function createDatasetPathFilterEditor(row, index) {
        const panel = document.createElement('div');
        panel.className = 'dataset-path-filter-advanced';
        panel.dataset.index = String(index);

        const recursive = document.createElement('label');
        recursive.className = 'dataset-path-filter-recursive';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = row.recursive !== false;
        checkbox.setAttribute('aria-label', '递归扫描子目录');
        checkbox.addEventListener('change', () => {
            updateDatasetEditorRow(index, 'recursive', checkbox.checked);
        });
        const recursiveCopy = document.createElement('span');
        recursiveCopy.innerHTML = '<strong>递归扫描子目录 / recursive</strong><small>默认开启；关闭后只读取原始路径第一层的图片。</small>';
        recursive.append(checkbox, recursiveCopy);

        const pattern = document.createElement('label');
        pattern.className = 'dataset-path-filter-pattern';
        const patternText = document.createElement('span');
        patternText.textContent = '路径筛选 / path_pattern';
        const patternInput = document.createElement('input');
        patternInput.type = 'text';
        patternInput.className = 'field-input';
        patternInput.value = String(row.path_pattern || '*').trim() || '*';
        patternInput.placeholder = '*';
        patternInput.title = '相对原始路径的 glob 筛选，例如 char_a/*；多个模式用 | 分隔。';
        patternInput.addEventListener('input', () => {
            updateDatasetEditorRow(index, 'path_pattern', patternInput.value);
        });
        pattern.append(patternText, patternInput);

        panel.append(recursive, pattern);
        return panel;
    }

    function createDatasetRowSettingsEditor(row, index) {
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const panel = document.createElement('div');
        panel.className = 'dataset-row-settings';
        const fields = [
            ['resolution', 'number'],
            ['enable_bucket', 'select'],
            ['min_bucket_reso', 'number'],
            ['max_bucket_reso', 'number'],
            ['bucket_reso_steps', 'number'],
            ['bucket_no_upscale', 'select'],
            ['validation_split', 'number'],
            ['validation_split_num', 'number'],
            ['validation_seed', 'number'],
        ];
        const helpDiv = document.createElement('div');
        helpDiv.className = 'field-help dataset-row-settings-help';

        for (const [key, type] of fields) {
            const field = document.createElement('div');
            field.className = 'dataset-row-setting-field';
            const labelRow = document.createElement('div');
            labelRow.className = 'dataset-row-setting-label';
            const label = document.createElement('span');
            label.className = 'field-name';
            label.textContent = datasetConfigLabel(key);
            label.title = key;
            labelRow.appendChild(label);

            const btn = document.createElement('button');
            btn.className = 'info-toggle dataset-row-help-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '查看填写建议、好处、代价、风险和推荐';
            btn.addEventListener('click', () => {
                const wasActive = btn.classList.contains('active');
                panel.querySelectorAll('.dataset-row-help-toggle.active').forEach((activeBtn) => {
                    activeBtn.classList.remove('active');
                });
                helpDiv.classList.remove('visible');
                helpDiv.innerHTML = '';
                if (wasActive) return;
                btn.classList.add('active');
                helpDiv.appendChild(createHelpContent(key, datasetConfigValue(key, settings)));
                helpDiv.classList.add('visible');
            });
            labelRow.appendChild(btn);

            field.appendChild(labelRow);
            field.appendChild(createDatasetRowSettingInput(index, key, type, settings));
            panel.appendChild(field);
        }
        panel.appendChild(helpDiv);
        return panel;
    }

    function createDatasetCaptionExtensionEditor(row, index) {
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const panel = document.createElement('div');
        panel.className = 'dataset-caption-extension-advanced';
        panel.dataset.index = String(index);

        const copy = document.createElement('div');
        copy.className = 'dataset-caption-extension-copy';
        const titleRow = document.createElement('div');
        titleRow.className = 'dataset-caption-extension-title-row';
        const title = document.createElement('strong');
        title.textContent = '文本标注扩展名 / caption_extension';
        const helpBtn = document.createElement('button');
        helpBtn.className = 'info-toggle dataset-caption-extension-help-toggle';
        helpBtn.textContent = '?';
        helpBtn.type = 'button';
        helpBtn.title = '查看填写建议、好处、代价、风险和推荐';
        titleRow.append(title, helpBtn);
        const desc = document.createElement('small');
        desc.textContent = '高级兼容项：仅在 txt 来源或 auto 回退到文本 sidecar 时使用。';
        copy.append(titleRow, desc);

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input dataset-caption-extension-input';
        input.value = settings.caption_extension || '.txt';
        input.placeholder = '.txt';
        input.setAttribute('aria-label', '文本标注扩展名');
        input.addEventListener('input', () => {
            updateDatasetEditorRowsSettingValue(
                datasetExperimentalScopeIndices(index),
                'caption_extension',
                input.value,
            );
        });
        input.addEventListener('change', () => {
            updateDatasetEditorRowsSettingValue(
                datasetExperimentalScopeIndices(index),
                'caption_extension',
                input.value,
                { render: true },
            );
        });

        const helpDiv = document.createElement('div');
        helpDiv.className = 'field-help dataset-caption-extension-help';
        helpDiv.appendChild(createHelpContent('caption_extension', settings.caption_extension || '.txt'));
        helpBtn.addEventListener('click', () => {
            helpBtn.classList.toggle('active');
            helpDiv.classList.toggle('visible');
        });

        panel.append(copy, input, helpDiv);
        return panel;
    }

    function createDatasetNlTagMixEditor(row, index) {
        const mix = normalizeNlTagMix(row.nl_tag_mix);
        const panel = document.createElement('div');
        panel.className = ['dataset-nl-tag-mix', mix.enabled ? 'enabled' : ''].filter(Boolean).join(' ');
        panel.dataset.index = String(index);

        const toggle = document.createElement('label');
        toggle.className = 'dataset-nl-tag-toggle';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = mix.enabled;
        checkbox.setAttribute('aria-label', 'captions格式nl/tag权重调整');
        checkbox.addEventListener('change', () => {
            updateDatasetEditorRowNlTagMix(index, {
                enabled: checkbox.checked,
                tag_ratio: mix.tag_ratio,
            });
        });
        const toggleText = document.createElement('span');
        toggleText.innerHTML = '<strong>captions格式nl/tag权重调整</strong><small>面向 DiffPipeForge captions.json 的多标注数据集优化；按短标签串和自然语言句子判断 tag/nl，按比例抽样重建运行时 captions.json，并写入 results.json。</small>';
        toggle.append(checkbox, toggleText);

        const ratio = document.createElement('label');
        ratio.className = 'dataset-nl-tag-ratio';
        const ratioHead = document.createElement('span');
        ratioHead.textContent = 'tag 占比';
        const ratioInput = document.createElement('input');
        ratioInput.type = 'range';
        ratioInput.min = '0';
        ratioInput.max = '100';
        ratioInput.step = '5';
        ratioInput.value = String(Math.round(mix.tag_ratio * 100));
        ratioInput.disabled = !mix.enabled;
        ratioInput.addEventListener('input', () => {
            updateDatasetEditorRowNlTagMix(index, {
                enabled: true,
                tag_ratio: Number(ratioInput.value) / 100,
            });
        });
        ratio.append(ratioHead, ratioInput);

        const ratioNumber = document.createElement('input');
        ratioNumber.type = 'number';
        ratioNumber.min = '0';
        ratioNumber.max = '100';
        ratioNumber.step = '5';
        ratioNumber.value = String(Math.round(mix.tag_ratio * 100));
        ratioNumber.disabled = !mix.enabled;
        ratioNumber.className = 'dataset-nl-tag-number';
        ratioNumber.setAttribute('aria-label', 'tag 占比百分比');
        ratioNumber.addEventListener('input', () => {
            updateDatasetEditorRowNlTagMix(index, {
                enabled: true,
                tag_ratio: Number(ratioNumber.value) / 100,
            });
        });

        const summary = document.createElement('output');
        summary.className = 'dataset-nl-tag-summary';
        summary.value = nlTagMixSummary(mix);
        summary.textContent = nlTagMixSummary(mix);

        panel.append(toggle, ratio, ratioNumber, summary);
        return panel;
    }

    function createDatasetExperimentalScopePicker(index) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        const selected = new Set(datasetExperimentalScopeIndices(index, rows.length));
        const scope = document.createElement('div');
        scope.className = 'dataset-experimental-scope';

        const copy = document.createElement('div');
        copy.className = 'dataset-experimental-scope-copy';
        const title = document.createElement('strong');
        title.textContent = '生效范围 / 对多数据集负责';
        const desc = document.createElement('span');
        desc.textContent = '选择这个实验框要同步写入的数据集组；保存时仍按每组独立配置落盘。';
        copy.append(title, desc);

        const actions = document.createElement('div');
        actions.className = 'dataset-experimental-scope-actions';
        const selectAll = document.createElement('button');
        selectAll.type = 'button';
        selectAll.className = 'btn btn-small';
        selectAll.textContent = '全选数据集';
        selectAll.disabled = rows.length <= 1 || selected.size === rows.length;
        selectAll.title = rows.length <= 1
            ? '当前只有一组数据集'
            : '让这个实验框同时负责所有数据集组。';
        selectAll.addEventListener('click', () => {
            setDatasetExperimentalScopeIndices(index, rows.map((_row, rowIndex) => rowIndex));
            renderDatasetEditor();
        });
        actions.appendChild(selectAll);

        const chips = document.createElement('div');
        chips.className = 'dataset-experimental-scope-chips';
        rows.forEach((_row, rowIndex) => {
            const chip = document.createElement('label');
            chip.className = ['dataset-scope-chip', selected.has(rowIndex) ? 'selected' : ''].filter(Boolean).join(' ');
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = selected.has(rowIndex);
            input.setAttribute('aria-label', `第 ${rowIndex + 1} 组数据集生效`);
            input.addEventListener('change', () => {
                const next = new Set(datasetExperimentalScopeIndices(index, rows.length));
                if (input.checked) {
                    next.add(rowIndex);
                } else {
                    next.delete(rowIndex);
                }
                if (!next.size) {
                    next.add(index);
                }
                setDatasetExperimentalScopeIndices(index, [...next]);
                renderDatasetEditor();
            });
            const text = document.createElement('span');
            text.textContent = `第 ${rowIndex + 1} 组`;
            chip.append(input, text);
            chips.appendChild(chip);
        });

        scope.append(copy, actions, chips);
        return scope;
    }

    function createDatasetTriggerCloneEditor(row, index) {
        const clone = normalizeTriggerClone(row.trigger_clone);
        const panel = document.createElement('div');
        panel.className = ['dataset-trigger-clone', clone.enabled ? 'enabled' : ''].filter(Boolean).join(' ');
        panel.dataset.index = String(index);

        const toggle = document.createElement('label');
        toggle.className = 'dataset-trigger-clone-toggle';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = clone.enabled;
        checkbox.setAttribute('aria-label', '触发提示词图像克隆');
        const toggleText = document.createElement('span');
        toggleText.innerHTML = '<strong>触发提示词图像克隆</strong><small>开启后，本次运行目录会生成额外训练子集；原始数据集不会被修改。</small>';
        checkbox.addEventListener('change', () => {
            updateDatasetEditorRowTriggerClone(index, {
                enabled: checkbox.checked,
            }, { render: true });
        });
        toggle.append(checkbox, toggleText);

        const prompt = document.createElement('label');
        prompt.className = 'dataset-trigger-clone-prompt';
        const promptText = document.createElement('span');
        promptText.textContent = '触发提示词';
        const promptInput = document.createElement('input');
        promptInput.type = 'text';
        promptInput.className = 'field-input';
        promptInput.value = clone.prompt;
        promptInput.placeholder = '例如 my_character_token';
        promptInput.disabled = !clone.enabled;
        promptInput.addEventListener('input', () => {
            updateDatasetEditorRowTriggerClone(index, {
                enabled: true,
                prompt: promptInput.value,
            });
        });
        prompt.append(promptText, promptInput);

        const repeats = document.createElement('label');
        repeats.className = 'dataset-trigger-clone-repeats';
        const repeatsText = document.createElement('span');
        repeatsText.textContent = '克隆循环次数';
        const repeatsInput = document.createElement('input');
        repeatsInput.type = 'number';
        repeatsInput.min = '1';
        repeatsInput.step = '1';
        repeatsInput.value = String(clone.num_repeats);
        repeatsInput.disabled = !clone.enabled;
        repeatsInput.addEventListener('input', () => {
            updateDatasetEditorRowTriggerClone(index, {
                enabled: true,
                num_repeats: repeatsInput.value,
            });
        });
        repeats.append(repeatsText, repeatsInput);

        const summary = document.createElement('span');
        summary.className = 'dataset-trigger-clone-summary';
        summary.textContent = clone.enabled
            ? `额外训练权重 x${clone.num_repeats}`
            : '默认关闭';

        panel.append(toggle, prompt, repeats, summary);
        return panel;
    }

    function normalizeCaptionSourceMode(value, preferJson = false) {
        const raw = String(value || '').trim().toLowerCase().replace(/-/g, '_');
        const allowed = new Set(CAPTION_SOURCE_MODE_OPTIONS.map((option) => option.value));
        if (allowed.has(raw)) return raw;
        if (raw === 'captions.json' || raw === 'diffpipeforge') return 'captions_json';
        if (raw === '.json' || raw === 'same_stem_json') return 'json';
        if (raw === '.txt' || raw === 'text') return 'txt';
        return preferJson ? 'json' : 'auto';
    }

    function captionSourceModeLabel(value) {
        const mode = normalizeCaptionSourceMode(value);
        const option = CAPTION_SOURCE_MODE_OPTIONS.find((item) => item.value === mode);
        return option ? `${option.label} (${option.detail})` : mode;
    }

    function createDatasetRowCaptionSourceModeEditor(settings, index) {
        const current = normalizeCaptionSourceMode(settings.caption_source_mode, settings.prefer_json_caption);
        const panel = document.createElement('div');
        panel.className = 'dataset-caption-source';
        panel.dataset.mode = current;

        const head = document.createElement('div');
        head.className = 'dataset-caption-source-head';
        const copy = document.createElement('div');
        copy.className = 'dataset-caption-source-copy';
        const titleRow = document.createElement('div');
        titleRow.className = 'dataset-caption-source-title-row';
        const title = document.createElement('strong');
        title.textContent = '标注来源 / caption_source_mode';
        const helpId = `dataset-caption-source-notes-${++datasetCaptionSourceHelpSeq}`;
        const helpBtn = document.createElement('button');
        helpBtn.className = 'info-toggle dataset-caption-source-help-toggle';
        helpBtn.type = 'button';
        helpBtn.textContent = '?';
        helpBtn.title = '展开标注来源注释';
        helpBtn.setAttribute('aria-label', '标注来源格式注释');
        helpBtn.setAttribute('aria-controls', helpId);
        helpBtn.setAttribute('aria-expanded', 'false');
        titleRow.append(title, helpBtn);
        const desc = document.createElement('span');
        desc.textContent = '默认 auto 自动识别；保存后预览和训练前预检测都会显示识别结果，也可以强制指定格式。';
        copy.append(titleRow, desc);
        const state = document.createElement('span');
        state.className = 'dataset-caption-source-state';
        state.textContent = captionSourceModeLabel(current);
        head.append(copy, state);

        const controls = document.createElement('div');
        controls.className = 'dataset-caption-source-options';
        CAPTION_SOURCE_MODE_OPTIONS.forEach((option) => {
            const label = document.createElement('label');
            label.className = ['dataset-caption-source-option', option.value === current ? 'selected' : ''].filter(Boolean).join(' ');
            const input = document.createElement('input');
            input.type = 'radio';
            input.name = `dataset-caption-source-${index}`;
            input.value = option.value;
            input.checked = option.value === current;
            input.setAttribute('aria-label', `${option.label} ${option.detail}`);
            input.addEventListener('change', () => {
                if (!input.checked) return;
                updateDatasetEditorRowsSettingValue(
                    [index],
                    'caption_source_mode',
                    option.value,
                    { render: true },
                );
            });
            const labelText = document.createElement('span');
            labelText.textContent = option.label;
            const detail = document.createElement('small');
            detail.textContent = option.detail;
            label.append(input, labelText, detail);
            controls.appendChild(label);
        });

        const notes = document.createElement('ul');
        notes.className = 'dataset-caption-source-notes';
        notes.id = helpId;
        notes.hidden = true;
        [
            '"1.png+1.txt"*n = sd-scripts格式标注',
            '"1.png+1.json"*n = AnimaLoraToolkit格式标注',
            '"png*n"+captions.json = DiffPipeForge格式标注',
            'caption_extension 仅影响 txt 来源或 auto 回退到文本标注；json / captions.json 模式会忽略它。',
        ].forEach((text) => {
            const item = document.createElement('li');
            item.textContent = text;
            notes.appendChild(item);
        });
        helpBtn.addEventListener('click', () => {
            const nextVisible = notes.hidden;
            notes.hidden = !nextVisible;
            helpBtn.classList.toggle('active', nextVisible);
            helpBtn.setAttribute('aria-expanded', String(nextVisible));
            helpBtn.title = nextVisible ? '收起标注来源注释' : '展开标注来源注释';
        });

        panel.append(head, controls, notes);
        return panel;
    }

    function createDatasetRowSettingInput(index, key, type, settings) {
        let input;
        if (type === 'select') {
            input = document.createElement('select');
            const options = key === 'enable_bucket'
                ? [[true, '启用'], [false, '关闭']]
                : [[false, '允许放大'], [true, '不放大小图']];
            const current = Boolean(settings[key]);
            for (const [value, label] of options) {
                const opt = document.createElement('option');
                opt.value = value ? 'true' : 'false';
                opt.textContent = label;
                opt.selected = value === current;
                input.appendChild(opt);
            }
        } else {
            input = document.createElement('input');
            input.type = type;
            input.value = datasetConfigValue(key, settings);
            if (type === 'number') {
                input.min = '0';
                input.step = key === 'validation_split' ? '0.001' : (key === 'resolution' || key.endsWith('_reso') || key === 'bucket_reso_steps' ? '16' : '1');
            }
        }
        input.className = 'field-input dataset-row-setting-input';
        input.addEventListener('input', () => updateDatasetEditorRowSetting(index, key, input));
        input.addEventListener('change', () => updateDatasetEditorRowSetting(index, key, input));
        return input;
    }

    function createDatasetPathField(index, key, label, value, placeholder) {
        const field = document.createElement('label');
        field.className = 'dataset-path-field';
        field.dataset.key = key;
        const text = document.createElement('span');
        text.textContent = label;
        const titles = {
            source_dir: '原始图片和 caption 所在目录。预处理从这里读图；缩放图和 LoRA 缓存会写入本次训练运行目录。',
            image_dir: '缩放图目录。预处理会把图片按分辨率/分桶规则写到这里；训练从这里枚举训练图片。',
            cache_dir: 'LoRA 缓存目录。VAE latent、文本编码器缓存、PE 特征缓存会写到这里；训练用它加速。',
        };
        text.title = titles[key] || label;
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input dataset-path-input';
        input.value = value || '';
        input.placeholder = placeholder;
        input.title = titles[key] || '';
        input.addEventListener('input', () => updateDatasetEditorRow(index, key, input.value));
        field.append(text, input);
        return field;
    }

    async function openDatasetPreview(index) {
        if (!datasetPresetState.selectedFile) {
            setDatasetPresetStatus('请先选择一个数据集预设', 'error');
            return;
        }
        if (datasetPresetState.dirty) {
            setDatasetPresetStatus('请先保存当前数据集预设，再打开预览', 'error');
            return;
        }
        datasetPreviewState.datasetIndex = index;
        datasetPreviewState.source = 'source';
        datasetPreviewState.payload = null;
        const dialog = document.getElementById('dataset-preview-dialog');
        renderDatasetPreviewDialog({ loading: true });
        if (dialog?.showModal && !dialog.open) {
            dialog.showModal();
        }
        await loadDatasetPreviewImages();
    }

    async function loadDatasetPreviewImages() {
        const file = datasetPresetState.selectedFile;
        if (!file) return;
        const requestSeq = ++datasetPreviewLoadSeq;
        renderDatasetPreviewDialog({ loading: true });
        try {
            const params = new URLSearchParams({
                file,
                dataset_index: String(datasetPreviewState.datasetIndex || 0),
                source: 'source',
                limit: '120',
            });
            const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
            if (requestSeq !== datasetPreviewLoadSeq) return;
            if (!payload.ok) throw new Error(payload.error || '读取数据集预览失败');
            datasetPreviewState.payload = payload;
            renderDatasetPreviewDialog();
        } catch (e) {
            if (requestSeq !== datasetPreviewLoadSeq) return;
            datasetPreviewState.payload = {
                ok: false,
                error: e.message || '读取数据集预览失败',
                images: [],
            };
            renderDatasetPreviewDialog();
        }
    }

    function renderDatasetPreviewDialog(options = {}) {
        const title = document.getElementById('dataset-preview-dialog-title');
        const meta = document.getElementById('dataset-preview-dialog-meta');
        const grid = document.getElementById('dataset-preview-grid');
        const details = document.getElementById('dataset-preview-details');
        const empty = document.getElementById('dataset-preview-empty');
        if (!title || !meta || !grid || !details || !empty) return;

        const datasetNo = Number(datasetPreviewState.datasetIndex || 0) + 1;
        title.textContent = `第 ${datasetNo} 组数据集预览`;
        if (options.loading) {
            meta.textContent = '正在读取图片和同名标注...';
            grid.innerHTML = '';
            details.innerHTML = '';
            empty.textContent = '正在读取数据集图片...';
            empty.hidden = false;
            return;
        }

        const payload = datasetPreviewState.payload || {};
        if (payload.error) {
            meta.textContent = payload.error;
            grid.innerHTML = '';
            details.innerHTML = '';
            empty.textContent = payload.error;
            empty.hidden = false;
            return;
        }

        const countText = `${payload.count || 0}/${payload.total || 0} 张`;
        const sourceLabel = payload.caption_source_label || captionSourceModeLabel(payload.caption_source_mode || 'auto');
        const detectedSummary = payload.caption_summary ? ` · 识别 ${payload.caption_summary}` : '';
        meta.textContent = `${payload.source_label || '原始图目录'} · ${payload.directory || '-'} · ${countText} · 标注来源 ${sourceLabel}${detectedSummary}`;
        renderDatasetPreviewDetails(payload);
        grid.innerHTML = '';
        const images = Array.isArray(payload.images) ? payload.images : [];
        if (!images.length) {
            empty.textContent = payload.message || '当前目录没有可预览图片。';
            empty.hidden = false;
            return;
        }
        empty.hidden = true;
        for (const image of images) {
            grid.appendChild(createDatasetPreviewCard(image));
        }
    }

    function renderDatasetPreviewDetails(payload) {
        const details = document.getElementById('dataset-preview-details');
        if (!details) return;
        details.innerHTML = '';
        const row = payload.row || {};
        const settings = normalizeDatasetDefaults(payload.settings || row.settings || {});
        const items = [
            ['数据集文件', payload.file || datasetPresetState.selectedFile || '-'],
            ['当前目录', payload.directory || '-'],
            ['原始路径', row.source_dir || '-'],
            ['重复次数', row.num_repeats ?? '-'],
            ['分辨率', settings.resolution || '-'],
            ['分桶', settings.enable_bucket ? `${settings.min_bucket_reso}-${settings.max_bucket_reso}/${settings.bucket_reso_steps}` : '关闭'],
            ['验证集', datasetPreviewValidationText(settings)],
            ['标注来源', payload.caption_source_label || captionSourceModeLabel(settings.caption_source_mode || 'auto')],
            ['识别结果', payload.caption_summary || '-'],
        ];
        for (const [label, value] of items) {
            details.appendChild(createPreviewDetailRow(label, String(value)));
        }
    }

    function datasetPreviewValidationText(settings) {
        if (Number(settings.validation_split_num || 0) > 0) return `固定 ${settings.validation_split_num} 张`;
        return `${settings.validation_split ?? 0}`;
    }

    function createDatasetPreviewCard(image) {
        const card = document.createElement('article');
        card.className = 'dataset-preview-card';
        const imageWrap = document.createElement('button');
        imageWrap.type = 'button';
        imageWrap.className = 'dataset-preview-image-btn';
        imageWrap.title = '点击在大图预览中查看。';
        imageWrap.addEventListener('click', () => openPreviewDialog(datasetPreviewImageToPreviewImage(image)));

        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name;
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            card.classList.add('dataset-preview-card-error');
            img.alt = '图片加载失败';
        });
        imageWrap.appendChild(img);

        const body = document.createElement('div');
        body.className = 'dataset-preview-card-body';
        const name = document.createElement('strong');
        name.textContent = image.name || '-';
        const file = document.createElement('span');
        file.textContent = image.file || '';
        body.append(name, file);

        const caption = image.caption || {};
        const captionBox = document.createElement('div');
        captionBox.className = ['dataset-preview-caption', caption.ok ? '' : 'missing'].filter(Boolean).join(' ');
        const captionHead = document.createElement('div');
        const captionTitle = document.createElement('span');
        const captionCount = Number(caption.caption_count || 0);
        const formatLabel = caption.format_label || caption.extension || '';
        captionTitle.textContent = caption.ok
            ? `标注 ${formatLabel}${captionCount > 1 ? ` · ${captionCount} 条` : ''}`
            : `缺少标注 · ${caption.source_label || captionSourceModeLabel(caption.source_mode || 'auto')}`;
        captionHead.appendChild(captionTitle);
        if (caption.file) {
            const copyBtn = document.createElement('button');
            copyBtn.type = 'button';
            copyBtn.className = 'btn btn-small';
            copyBtn.textContent = '复制标注';
            copyBtn.addEventListener('click', () => copyDatasetCaptionText(caption.text || '', copyBtn));
            captionHead.appendChild(copyBtn);
        }
        const pre = document.createElement('pre');
        pre.textContent = caption.ok ? (caption.text || '(空标注)') : '未按当前标注来源找到 caption 文件';
        captionBox.append(captionHead, pre);
        body.appendChild(captionBox);

        card.append(imageWrap, body);
        return card;
    }

    function datasetPreviewImageToPreviewImage(image) {
        return {
            ...image,
            detailContext: 'dataset',
            sample: {},
            source_task: null,
        };
    }

    async function copyDatasetCaptionText(text, button) {
        try {
            await copyText(text || '');
            const original = button.textContent;
            button.textContent = '已复制';
            button.classList.add('btn-primary');
            setTimeout(() => {
                button.textContent = original;
                button.classList.remove('btn-primary');
            }, 1000);
        } catch (e) {
            alert('复制标注失败: ' + e.message);
        }
    }

    function normalizeNlTagMix(raw) {
        const source = raw && typeof raw === 'object' ? raw : {};
        const enabled = source.enabled === true || source.enabled === 'true';
        const parsedRatio = Number(source.tag_ratio ?? source.tagRatio ?? DEFAULT_NL_TAG_MIX.tag_ratio);
        const tagRatio = Number.isFinite(parsedRatio)
            ? Math.min(1, Math.max(0, parsedRatio > 1 ? parsedRatio / 100 : parsedRatio))
            : DEFAULT_NL_TAG_MIX.tag_ratio;
        return {
            enabled,
            tag_ratio: tagRatio,
        };
    }

    function nlTagMixSummary(mix) {
        const normalized = normalizeNlTagMix(mix);
        const tagPercent = Math.round(normalized.tag_ratio * 100);
        return `${tagPercent}% tag + ${100 - tagPercent}% nl`;
    }

    function normalizeTriggerClone(raw) {
        const source = raw && typeof raw === 'object' ? raw : {};
        return {
            enabled: source.enabled === true || source.enabled === 'true',
            prompt: String(source.prompt || '').trim(),
            num_repeats: Math.max(1, Number.parseInt(source.num_repeats || 1, 10) || 1),
        };
    }

    function normalizeDatasetEditorRows(rows) {
        return (rows || [])
            .filter((row) => row && typeof row === 'object')
            .map((row) => ({
                source_dir: String(row.source_dir || row.source_image_dir || ''),
                image_dir: String(row.image_dir || row.resized_image_dir || ''),
                cache_dir: String(row.cache_dir || row.lora_cache_dir || ''),
                num_repeats: Math.max(1, Number.parseInt(row.num_repeats || 1, 10) || 1),
                recursive: row.recursive !== false && row.recursive !== 'false',
                path_pattern: String(row.path_pattern || '*').trim() || '*',
                nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix),
                trigger_clone: normalizeTriggerClone(row.trigger_clone),
                settings: normalizeDatasetRowSettings(row),
            }));
    }

    function datasetRowsForPayload(rows) {
        return normalizeDatasetEditorRows(rows).map((row) => ({
            source_dir: row.source_dir,
            image_dir: row.image_dir,
            cache_dir: row.cache_dir,
            num_repeats: row.num_repeats,
            recursive: row.recursive,
            path_pattern: row.path_pattern,
            nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix),
            trigger_clone: normalizeTriggerClone(row.trigger_clone),
            settings: normalizeDatasetDefaults(row.settings || {}),
        }));
    }

    function normalizeDatasetRowSettings(row) {
        if (row.settings && typeof row.settings === 'object') {
            return normalizeDatasetDefaults(row.settings);
        }
        if ([...DATASET_SETTING_KEYS].some((key) => key in row)) {
            return normalizeDatasetDefaults(row);
        }
        return {};
    }

    function normalizeDatasetDefaults(defaults) {
        const raw = defaults && typeof defaults === 'object' ? defaults : {};
        const preferJson = raw.prefer_json_caption === true || raw.prefer_json_caption === 'true';
        const captionSourceMode = normalizeCaptionSourceMode(raw.caption_source_mode, preferJson);
        const validationSeed = Number.parseInt(raw.validation_seed ?? 42, 10);
        return {
            resolution: Math.max(1, Number.parseInt(raw.resolution || 1024, 10) || 1024),
            enable_bucket: raw.enable_bucket !== false && raw.enable_bucket !== 'false',
            min_bucket_reso: Math.max(1, Number.parseInt(raw.min_bucket_reso || 256, 10) || 256),
            max_bucket_reso: Math.max(1, Number.parseInt(raw.max_bucket_reso || 1024, 10) || 1024),
            bucket_reso_steps: Math.max(1, Number.parseInt(raw.bucket_reso_steps || 64, 10) || 64),
            bucket_no_upscale: raw.bucket_no_upscale === true || raw.bucket_no_upscale === 'true',
            validation_split: Math.max(0, Number(raw.validation_split ?? 0.025) || 0),
            validation_split_num: Math.max(0, Number.parseInt(raw.validation_split_num || 0, 10) || 0),
            validation_seed: Number.isFinite(validationSeed) ? Math.max(0, validationSeed) : 42,
            caption_extension: String(raw.caption_extension || '.txt'),
            keep_tokens: Math.max(0, Number.parseInt(raw.keep_tokens || 3, 10) || 0),
            prefer_json_caption: preferJson,
            caption_source_mode: captionSourceMode,
        };
    }

    function updateDatasetDefault(key, input) {
        const state = datasetEditorStateForActivePanel();
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        if (input.type === 'checkbox') {
            defaults[key] = input.checked;
        } else if (input.tagName === 'SELECT') {
            defaults[key] = input.value === 'true';
        } else if (input.type === 'number') {
            defaults[key] = key === 'validation_split' ? Math.max(0, Number(input.value) || 0) : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            defaults[key] = input.value;
        }
        if (isDatasetTabActive()) {
            datasetPresetState.defaults = defaults;
        } else {
            datasetEditorState.defaults = defaults;
        }
        markDatasetEditorDirty();
    }

    function updateDatasetEditorRow(index, key, value) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (!rows[index]) return;
        if (key === 'source_dir' && rows[index].source_dir !== value) {
            rows[index].image_dir = '';
            rows[index].cache_dir = '';
        }
        rows[index][key] = key === 'num_repeats'
            ? Math.max(1, Number.parseInt(value || '1', 10) || 1)
            : value;
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        if (!isDatasetTabActive() && index === 0 && key === 'source_dir') {
            setFieldInputValue('source_image_dir', value);
        }
        markDatasetEditorDirty();
        if (key === 'num_repeats') {
            updateStepEstimatePanel();
        }
    }

    function updateDatasetEditorRowSetting(index, key, input) {
        let value;
        if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.tagName === 'SELECT') {
            value = input.value === 'true';
        } else if (input.type === 'number') {
            value = key === 'validation_split' ? Math.max(0, Number(input.value) || 0) : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            value = input.value;
        }
        updateDatasetEditorRowSettingValue(index, key, value);
    }

    function updateDatasetEditorRowSettingValue(index, key, value) {
        updateDatasetEditorRowsSettingValue([index], key, value);
    }

    function updateDatasetEditorRowsSettingValue(indices, key, value, options = {}) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        const targets = datasetValidTargetIndices(indices, rows.length);
        if (!targets.length) return;
        for (const targetIndex of targets) {
            const settings = normalizeDatasetDefaults(rows[targetIndex].settings || state.defaults || {});
            settings[key] = value;
            rows[targetIndex].settings = settings;
        }
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        if (options.render) {
            renderDatasetEditor();
        }
    }

    function updateDatasetEditorRowNlTagMix(index, nextMix) {
        updateDatasetEditorRowsNlTagMix([index], nextMix);
    }

    function updateDatasetEditorRowsNlTagMix(indices, nextMix) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        const targets = datasetValidTargetIndices(indices, rows.length);
        if (!targets.length) return;
        const mix = normalizeNlTagMix(nextMix);
        for (const targetIndex of targets) {
            rows[targetIndex].nl_tag_mix = mix;
        }
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        renderDatasetEditor();
    }

    function updateDatasetEditorRowTriggerClone(index, nextClone, options = {}) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (!rows[index]) return;
        rows[index].trigger_clone = normalizeTriggerClone({
            ...rows[index].trigger_clone,
            ...nextClone,
        });
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        if (options.render) {
            renderDatasetEditor();
        }
    }

    function datasetExperimentalScopeKey(index) {
        return `${isDatasetTabActive() ? 'dataset-preset' : 'config-dataset'}:${index}`;
    }

    function datasetExperimentalScopeIndices(index, total = null) {
        const state = datasetEditorStateForActivePanel();
        const count = total ?? normalizeDatasetEditorRows(state.datasets).length;
        const key = datasetExperimentalScopeKey(index);
        const raw = datasetExperimentalScopeSelections.get(key) || [index];
        const selected = datasetValidTargetIndices(raw, count);
        if (!selected.length && index >= 0 && index < count) {
            selected.push(index);
        }
        datasetExperimentalScopeSelections.set(key, selected);
        return selected;
    }

    function setDatasetExperimentalScopeIndices(index, indices) {
        const state = datasetEditorStateForActivePanel();
        const count = normalizeDatasetEditorRows(state.datasets).length;
        const selected = datasetValidTargetIndices(indices, count);
        if (!selected.length && index >= 0 && index < count) {
            selected.push(index);
        }
        datasetExperimentalScopeSelections.set(datasetExperimentalScopeKey(index), selected);
    }

	    function datasetValidTargetIndices(indices, count) {
	        return [...new Set((indices || [])
	            .map((value) => Number.parseInt(value, 10))
	            .filter((value) => Number.isInteger(value) && value >= 0 && value < count))]
	            .sort((left, right) => left - right);
	    }

	    function setDatasetEditorRowsAfterSort(rows) {
	        datasetExperimentalScopeSelections.clear();
	        if (isDatasetTabActive()) {
	            datasetPresetState.datasets = rows;
	        } else {
	            datasetEditorState.datasets = rows;
	        }
	        markDatasetEditorDirty();
	        renderDatasetEditor();
	    }

	    function moveDatasetEditorRow(sourceIndex, targetIndex, placeAfter = false) {
	        const rows = normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
	        if (rows.length <= 1) return false;
	        if (sourceIndex < 0 || sourceIndex >= rows.length || targetIndex < 0 || targetIndex >= rows.length) return false;
	        if (sourceIndex === targetIndex) return false;
	        let insertIndex = targetIndex + (placeAfter ? 1 : 0);
	        if (sourceIndex < insertIndex) insertIndex -= 1;
	        insertIndex = Math.max(0, Math.min(rows.length - 1, insertIndex));
	        if (insertIndex === sourceIndex) return false;
	        const [moved] = rows.splice(sourceIndex, 1);
	        rows.splice(insertIndex, 0, moved);
	        setDatasetEditorRowsAfterSort(rows);
	        return true;
	    }

	    function moveDatasetEditorRowToIndex(sourceIndex, targetIndex) {
	        const rows = normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
	        const clamped = Math.max(0, Math.min(rows.length - 1, targetIndex));
	        if (clamped === sourceIndex) return false;
	        return moveDatasetEditorRow(sourceIndex, clamped, clamped > sourceIndex);
	    }

	    function markDatasetEditorDirty() {
        if (isDatasetTabActive()) {
            datasetPresetState.dirty = true;
            datasetPresetState.status = '有未保存的数据集修改';
            renderDatasetPresetHeader();
        } else {
            datasetEditorState.dirty = true;
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
        const dirty = document.querySelector('#dataset-editor .dataset-editor-dirty');
        if (dirty) {
            dirty.classList.add('active');
            dirty.textContent = '有未保存的数据集修改';
        }
    }

    function addDatasetEditorRow() {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        rows.push({
            source_dir: '',
            image_dir: '',
            cache_dir: '',
            num_repeats: 1,
            trigger_clone: normalizeTriggerClone(DEFAULT_TRIGGER_CLONE),
            settings: normalizeDatasetDefaults(state.defaults || {}),
        });
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
            datasetPresetState.dirty = true;
        } else {
            datasetEditorState.datasets = rows;
            datasetEditorState.dirty = true;
        }
        renderDatasetEditor();
        if (!isDatasetTabActive()) {
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
    }

    function removeDatasetEditorRow(index) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (rows.length <= 1) return;
        rows.splice(index, 1);
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
            datasetPresetState.dirty = true;
        } else {
            datasetEditorState.datasets = rows;
            datasetEditorState.dirty = true;
        }
        renderDatasetEditor();
        if (!isDatasetTabActive()) {
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
    }

    function syncDatasetEditorToCompatFields() {
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        const first = rows[0];
        if (!first) return;
        setFieldInputValue('source_image_dir', first.source_dir);
        setFieldInputValue('resized_image_dir', first.image_dir);
        setFieldInputValue('lora_cache_dir', first.cache_dir);
        if (datasetEditorState.dataset_config) {
            setFieldInputValue('dataset_config', datasetEditorState.dataset_config);
        }
    }

    function setFieldInputValue(key, value) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return;
        if (input.type === 'checkbox') {
            input.checked = Boolean(value);
        } else {
            input.value = value ?? '';
        }
        updateConfigDraftFromInput(input);
    }

    function escapeHtml(value) {
        return ctx.format.escapeHtml(value);
    }

    function setCurrentTrainingSourceFromVariant(variant) {
        if (!variant) {
            clearCurrentTrainingSource();
            return;
        }
        if (val('method-select') === 'spd' || variant === 'spd') {
            currentTrainingSource = {
                method: 'spd',
                methods_subdir: 'methods',
                file: 'configs/methods/spd.toml',
            };
            return;
        }
        currentTrainingSource = {
            method: variant,
            methods_subdir: 'gui-methods',
            file: `configs/gui-methods/${variant}.toml`,
        };
    }

    function clearCurrentTrainingSource() {
        currentTrainingSource = {
            method: '',
            methods_subdir: '',
            file: '',
        };
    }

    function outputRunRuntimeFile(run = selectedOutputRun()) {
        const runtime = (run?.files || []).find((item) => item.kind === 'runtime');
        return runtime?.file || '';
    }

    function rememberSelectionSnapshot() {
        selectionSnapshot.method = val('method-select');
        selectionSnapshot.variant = val('variant-select');
        selectionSnapshot.preset = val('preset-select');
    }

    function restoreSelectionSnapshot() {
        const methodSelect = document.getElementById('method-select');
        const variantSelect = document.getElementById('variant-select');
        const presetSelect = document.getElementById('preset-select');
        if (methodSelect && selectionSnapshot.method && [...methodSelect.options].some((opt) => opt.value === selectionSnapshot.method)) {
            methodSelect.value = selectionSnapshot.method;
        }
        if (variantSelect && selectionSnapshot.variant && [...variantSelect.options].some((opt) => opt.value === selectionSnapshot.variant)) {
            variantSelect.value = selectionSnapshot.variant;
        }
        if (presetSelect && selectionSnapshot.preset && [...presetSelect.options].some((opt) => opt.value === selectionSnapshot.preset)) {
            presetSelect.value = selectionSnapshot.preset;
        }
        setCurrentTrainingSourceFromVariant(val('variant-select'));
        updateChoiceGuide();
    }

    async function confirmBeforeConfigSelectionChange(message) {
        const ok = await handlePendingConfigSwitch({
            targetLabel: '新的配置选择',
        });
        if (!ok) restoreSelectionSnapshot();
        return ok;
    }

    function updateChoiceGuide(config = currentConfig) {
        const container = document.getElementById('choice-guide');
        if (!container) return;
        container.innerHTML = '';
        const methodKey = activeMethodKey(config);
        container.appendChild(createChoiceCard('方法', methodKey, METHOD_GUIDE_ZH, defaultMethodGuide(), methodGuideFromConfig(methodKey, config)));
        const sourceKey = currentTrainingSource.method || val('variant-select');
        container.appendChild(createChoiceCard('配置', sourceKey, VARIANT_GUIDE_ZH, defaultVariantGuide(), configGuideFromCurrentSource(sourceKey, config)));
        const presetKey = val('preset-select');
        container.appendChild(createChoiceCard('预设', presetKey, PRESET_GUIDE_ZH, defaultPresetGuide(), presetGuideFromConfig(presetKey, config)));
    }

    function createChoiceCard(kind, key, guideMap, fallback, overrideGuide = null) {
        const guide = overrideGuide || guideMap[key] || fallback;
        const helpId = `choice-guide-hint-${++choiceGuideHintSeq}`;
        const card = document.createElement('article');
        card.className = 'choice-card';

        const heading = document.createElement('div');
        heading.className = 'choice-card-heading';
        const title = document.createElement('strong');
        title.textContent = `${kind}: ${key || '-'}`;
        const name = document.createElement('span');
        name.textContent = guide.title;
        heading.appendChild(title);
        heading.appendChild(name);
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'info-toggle choice-info-toggle';
        toggle.textContent = '?';
        toggle.title = `展开${kind}说明`;
        toggle.setAttribute('aria-label', `${kind}说明`);
        toggle.setAttribute('aria-expanded', 'false');
        toggle.setAttribute('aria-controls', helpId);
        heading.appendChild(toggle);
        card.appendChild(heading);

        const body = document.createElement('div');
        body.id = helpId;
        body.className = 'choice-card-body';
        body.hidden = true;
        body.appendChild(choiceLine('说明', guide.summary));
        body.appendChild(choiceLine('取舍', guide.tradeoff));
        body.appendChild(choiceLine('推荐', guide.recommend, 'choice-recommend'));
        if (Array.isArray(guide.details) && guide.details.length) {
            const details = document.createElement('ul');
            details.className = 'choice-details';
            for (const detail of guide.details) {
                const item = document.createElement('li');
                item.textContent = detail;
                details.appendChild(item);
            }
            body.appendChild(details);
        }
        toggle.addEventListener('click', () => {
            const nextOpen = body.hidden;
            body.hidden = !nextOpen;
            toggle.classList.toggle('active', nextOpen);
            toggle.setAttribute('aria-expanded', String(nextOpen));
            toggle.title = nextOpen ? `收起${kind}说明` : `展开${kind}说明`;
        });
        card.appendChild(body);
        return card;
    }

    function choiceLine(label, text, extraClass = '') {
        const line = document.createElement('p');
        line.className = extraClass;
        const strong = document.createElement('strong');
        strong.textContent = `${label}: `;
        line.appendChild(strong);
        line.appendChild(document.createTextNode(text));
        return line;
    }

    function defaultMethodGuide() {
        return choiceHelp(
            '自定义方法',
            '当前方法没有专门说明，通常表示它来自后端方法列表。',
            '请结合变体 TOML 判断实际训练行为。',
            '不确定时使用 lora。'
        );
    }

    function defaultVariantGuide() {
        return choiceHelp(
            '自定义变体',
            '当前变体对应一个 gui-methods TOML 文件，里面才是实际训练参数。',
            '自定义变体灵活，但需要自行确认字段组合是否合理。',
            '不确定时从内置 lora 变体复制再改。'
        );
    }

    function defaultPresetGuide() {
        return choiceHelp(
            '自定义预设',
            '当前预设来自 presets.toml 或自定义配置。',
            '它会覆盖部分硬件、采样或性能参数。',
            '不确定时使用 default。'
        );
    }

    function activeMethodKey(config = currentConfig) {
        const inferred = inferMethodFromConfig(config);
        if (inferred) return inferred;
        if (currentTrainingSource.methods_subdir === 'methods' && currentTrainingSource.method === 'spd') {
            return 'spd';
        }
        if (currentTrainingSource.methods_subdir === 'gui-methods') {
            return VARIANT_METHOD_FAMILY[currentTrainingSource.method] || val('method-select') || 'lora';
        }
        return val('method-select') || 'lora';
    }

    function inferMethodFromConfig(config) {
        if (!config || typeof config !== 'object') return '';
        const moduleName = String(config.network_module || '');
        if (currentTrainingSource.methods_subdir === 'methods' && currentTrainingSource.method === 'spd') return 'spd';
        if ('dit_path' in config && 'iterations' in config && currentTrainingSource.method === 'spd') return 'spd';
        if (isTruthy(config.use_lokr)) return 'lokr';
        if (isTruthy(config.use_loha)) return 'loha';
        if (isTruthy(config.use_easycontrol) || moduleName.includes('easycontrol')) return 'easycontrol';
        if (isTruthy(config.use_ip_adapter) || moduleName.includes('ip_adapter')) return 'ip_adapter';
        if (moduleName.includes('soft_tokens')) return 'soft_tokens';
        if (isTruthy(config.add_reft) || ('reft_dim' in config && Number(config.reft_dim) > 0)) return 'reft';
        if (
            isTruthy(config.use_hydra) ||
            isTruthy(config.use_sigma_router) ||
            String(config.use_moe_style || 'false') !== 'false' ||
            moduleName.includes('chimera') ||
            moduleName.includes('hydra')
        ) {
            if (moduleName.includes('chimera') || 'content_router_source' in config) return 'chimera';
            return 'hydralora';
        }
        if (isTruthy(config.use_timestep_mask)) return 'tlora';
        if (isTruthy(config.use_ortho)) return 'ortholora';
        return '';
    }

    function methodGuideFromConfig(methodKey, config = currentConfig) {
        const base = METHOD_GUIDE_ZH[methodKey] || defaultMethodGuide();
        const details = compactList([
            flagDetail('use_lokr', 'LoKr', config.use_lokr),
            flagDetail('use_loha', 'LoHa', config.use_loha),
            isTruthy(config.use_lokr) ? valueDetail('lokr_factor', config.lokr_factor) : '',
            valueDetail('network_dim', config.network_dim),
            valueDetail('network_alpha', config.network_alpha),
            valueDetail('learning_rate', config.learning_rate),
            valueDetail('max_train_epochs', config.max_train_epochs),
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 当前表单已读取关键训练字段。`,
            details,
        };
    }

    function configGuideFromCurrentSource(sourceKey, config = currentConfig) {
        const isImported = currentTrainingSource.methods_subdir === 'imported';
        const base = isImported
            ? choiceHelp(
                '导入训练配置',
                `当前表单来自 ${currentTrainingSource.file || '导入配置'}。`,
                '它会按 base.toml → 当前预设 → 该 TOML 的顺序合并；不会强行加入变体下拉。',
                '适合把历史训练配置作为独立入口继续查看、预检测或训练。'
            )
            : (VARIANT_GUIDE_ZH[sourceKey] || defaultVariantGuide());
        const details = compactList([
            currentTrainingSource.file ? `文件: ${currentTrainingSource.file}` : '',
            config.dataset_config ? `数据集配置: ${config.dataset_config}` : '',
            config.output_name ? `输出名称: ${config.output_name}` : '',
            globalSettings?.output_root ? `Web 输出根目录: ${globalSettings.output_root}` : '',
            config.source_image_dir ? `原始数据集: ${config.source_image_dir}` : '',
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 已读取当前 TOML 的路径和输出信息。`,
            details,
        };
    }

    function presetGuideFromConfig(presetKey, config = currentConfig) {
        const base = PRESET_GUIDE_ZH[presetKey] || defaultPresetGuide();
        const details = compactList([
            valueDetail('mixed_precision', config.mixed_precision),
            valueDetail('optimizer_type', config.optimizer_type),
            valueDetail('lr_scheduler', config.lr_scheduler),
            valueDetail('train_batch_size', config.train_batch_size),
            valueDetail('gradient_accumulation_steps', config.gradient_accumulation_steps),
            valueDetail('sample_ratio', config.sample_ratio),
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 当前已合并后的预设/配置值如下。`,
            details,
        };
    }

    function isTruthy(value) {
        return value === true || value === 1 || value === '1' || String(value).toLowerCase() === 'true';
    }

    function normalizeLoraAdapterKind(value) {
        const text = String(value ?? '').trim().toLowerCase();
        if (text === 'loha' || text === 'lokr') return text;
        return 'lora';
    }

    function loraAdapterKindFromConfig(config = currentConfig) {
        if (isTruthy(config?.use_lokr)) return 'lokr';
        if (isTruthy(config?.use_loha)) return 'loha';
        return 'lora';
    }

    function loraAdapterFlagsForKind(kind) {
        const normalized = normalizeLoraAdapterKind(kind);
        return {
            use_loha: normalized === 'loha',
            use_lokr: normalized === 'lokr',
        };
    }

    function applyLoraAdapterDraft(kind) {
        const normalized = normalizeLoraAdapterKind(kind);
        const originalKind = loraAdapterKindFromConfig(currentConfig);
        if (normalized === originalKind && loraAdapterFlagsMatchConfig(normalized, currentConfig)) {
            configFormState.draftValues.delete('lora_adapter_kind');
        } else {
            configFormState.draftValues.set('lora_adapter_kind', normalized);
        }
        configFormState.draftValues.delete('use_loha');
        configFormState.draftValues.delete('use_lokr');
    }

    function readLiveLoraAdapterKind() {
        if (configFormState.draftValues.has('lora_adapter_kind')) {
            return normalizeLoraAdapterKind(configFormState.draftValues.get('lora_adapter_kind'));
        }
        const input = document.querySelector('#config-form .field-input[data-key="lora_adapter_kind"]');
        if (input) {
            return normalizeLoraAdapterKind(readFieldInputValue(input, loraAdapterKindFromConfig(currentConfig)));
        }
        return loraAdapterKindFromConfig(currentConfig);
    }

    function applyLoraAdapterPatch(values) {
        if (!configFormState.draftValues.has('lora_adapter_kind')) return values;
        const nextKind = normalizeLoraAdapterKind(configFormState.draftValues.get('lora_adapter_kind'));
        const flags = loraAdapterFlagsForKind(nextKind);
        values.use_loha = flags.use_loha;
        values.use_lokr = flags.use_lokr;
        if (flags.use_lokr && !('lokr_factor' in values) && !('lokr_factor' in currentConfig)) {
            values.lokr_factor = FORM_UI_DEFAULTS.lokr_factor;
        }
        return values;
    }

    function loraAdapterFlagsMatchConfig(kind, config = currentConfig) {
        const flags = loraAdapterFlagsForKind(kind);
        return isTruthy(config?.use_loha) === flags.use_loha
            && isTruthy(config?.use_lokr) === flags.use_lokr;
    }

    function compactList(items) {
        return items.filter((item) => item !== undefined && item !== null && String(item).trim() !== '');
    }

    function valueDetail(key, value) {
        if (value === undefined || value === null || value === '') return '';
        return `${FIELD_LABEL_ZH[key] || key}: ${formatChoiceValue(value)}`;
    }

    function flagDetail(key, label, value) {
        if (value === undefined || value === null || value === '') return '';
        return `${label}: ${isTruthy(value) ? '开启' : '关闭'}`;
    }

    function formatChoiceValue(value) {
        if (Array.isArray(value)) return value.join(', ');
        if (typeof value === 'boolean') return value ? 'true' : 'false';
        return String(value);
    }

    function createFieldRow(key, value) {
        const row = document.createElement('div');
        row.className = 'field-row';
        row.dataset.key = key;
        if (key === 'sample_prompts') row.classList.add('field-row-sample-prompts');
        const hasDraftValue = configFormState.draftValues.has(key);
        const originalValue = originalConfigFieldValue(key);
        const displayValue = displayConfigFieldValue(key, value);

        const main = document.createElement('div');
        main.className = 'field-main';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'field-name';
        nameSpan.textContent = formatFieldName(key);
        nameSpan.title = key;

        const input = createFieldInput(key, displayValue, { originalValue, hasDraftValue });
        input.dataset.key = key;
        input.dataset.valueType = fieldValueTypeForKey(key, originalValue);
        input.addEventListener('input', handleFormFieldChange);
        input.addEventListener('change', handleFormFieldChange);

        if (key === 'sample_prompts' && samplePromptsMode !== 'path') {
            const labelStack = document.createElement('div');
            labelStack.className = 'field-label-stack';
            labelStack.appendChild(nameSpan);

            const rowsWrap = input.querySelector('.sample-prompts-rows');
            if (rowsWrap) {
                const labelActions = document.createElement('div');
                labelActions.className = 'field-label-actions';
                labelActions.appendChild(createSamplePromptAddButton(rowsWrap));
                labelActions.appendChild(createSamplePromptTextModeButton(input));
                labelStack.appendChild(labelActions);
            }
            main.appendChild(labelStack);
        } else {
            main.appendChild(nameSpan);
        }
        main.appendChild(input);

        const btn = document.createElement('button');
        btn.className = 'info-toggle';
        btn.textContent = '?';
        btn.type = 'button';
        btn.title = '查看填写建议、好处、代价、风险和推荐';
        btn.addEventListener('click', () => {
            btn.classList.toggle('active');
            const helpDiv = row.querySelector('.field-help');
            if (helpDiv) helpDiv.classList.toggle('visible');
        });
        main.appendChild(btn);
        row.appendChild(main);

        const helpDiv = document.createElement('div');
        helpDiv.className = 'field-help';
        helpDiv.appendChild(createHelpContent(key, value));
        row.appendChild(helpDiv);

        return row;
    }

    function handleFormFieldChange() {
        syncConfigDraftFromForm();
        updateTomlDirtyState();
        updateStepEstimatePanel();
        updateLoKrFieldState();
        updateChoiceGuideFromLiveForm();
    }

    function updateChoiceGuideFromLiveForm() {
        if (!currentConfig || Object.keys(currentConfig).length === 0) return;
        updateChoiceGuide(liveConfigFromForm());
    }

    function liveConfigFromForm() {
        syncConfigDraftFromForm();
        const liveConfig = { ...(currentConfig || {}) };
        for (const [key, next] of configFormState.draftValues.entries()) {
            if (!key) continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (isActiveNetworkArgFieldKey(key)) continue;
            if (key === 'lora_adapter_kind') {
                Object.assign(liveConfig, loraAdapterFlagsForKind(next));
                continue;
            }
            liveConfig[key] = next;
        }
        liveConfig.network_args = collectNetworkArgsFromForm(liveConfig).networkArgs;
        return liveConfig;
    }

    function formatFieldName(key) {
        const label = FIELD_LABEL_ZH[key];
        return label ? `${label} / ${key}` : key;
    }

    function createFieldInput(key, value, options = {}) {
        if (key === 'sample_prompts') {
            if (samplePromptsMode === 'path') {
                return createSamplePromptsPathInput(value);
            }
            return createSamplePromptsEditor(value, options.originalValue, options.hasDraftValue);
        }
        const fieldOptions = FIELD_OPTIONS[key];
        if (fieldOptions && !Array.isArray(value)) {
            return createSelectInput(key, value, fieldOptions);
        }

        let input;
        const typeSource = options.originalValue ?? value;
        if (typeof typeSource === 'boolean') {
            input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = value === true || value === 'true';
        } else {
            input = document.createElement('input');
            input.type = isNumericField(key, typeSource) ? 'number' : 'text';
            if (input.type === 'number') {
                input.step = isIntegerNumericField(key, typeSource) ? '1' : '0.01';
                if (!allowsNegativeNumberField(key)) input.min = '0';
            }
            input.value = Array.isArray(value) ? JSON.stringify(value) : (value ?? '');
        }
        input.className = 'field-input';
        if (key === 'lokr_factor') {
            input.disabled = !readLoKrEnabled();
            input.title = input.disabled ? '启用 LoKr 后生效' : '';
        }
        return input;
    }

    function createSamplePromptsPathInput(value) {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input';
        input.value = value ?? '';
        input.title = '当前 sample_prompts 指向非 .txt 文件，保留为文件路径。';
        return input;
    }

    function createSamplePromptsEditor(value, originalValue = value, touched = false) {
        const editor = document.createElement('div');
        editor.className = 'field-input sample-prompts-editor';
        editor.dataset.originalContent = originalValue ?? '';
        editor.dataset.touched = touched ? '1' : '0';

        const rows = document.createElement('div');
        rows.className = 'sample-prompts-rows';

        editor.appendChild(rows);

        editor.addEventListener('input', (event) => {
            if (event.target?.closest?.('.sample-prompt-row')) {
                markSamplePromptsEditorTouched(editor);
            }
        });
        editor.addEventListener('change', (event) => {
            if (event.target?.closest?.('.sample-prompt-row')) {
                markSamplePromptsEditorTouched(editor);
            }
        });

        renderSamplePromptRows(editor, value ?? '');
        return editor;
    }

    function createSamplePromptAddButton(rowsWrap) {
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small sample-prompts-add-btn';
        addBtn.textContent = '添加行';
        addBtn.addEventListener('click', () => {
            const editor = rowsWrap.closest('.sample-prompts-editor');
            if (editor?.dataset.mode === 'text') {
                const textarea = editor.querySelector('.sample-prompts-textarea');
                if (textarea) {
                    if (textarea.value && !textarea.value.endsWith('\n')) textarea.value += '\n';
                    textarea.focus();
                    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
                    markSamplePromptsEditorTouched(editor);
                    handleFormFieldChange();
                    return;
                }
            }
            appendSamplePromptRow(rowsWrap, blankSamplePromptRow());
            markSamplePromptsEditorTouched(editor);
            handleFormFieldChange();
        });
        return addBtn;
    }

    function createSamplePromptTextModeButton(editor) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-small sample-prompts-add-btn sample-prompts-mode-btn';
        btn.dataset.samplePromptsModeToggle = '1';
        updateSamplePromptModeButtonState(btn, editor);
        btn.addEventListener('click', () => {
            if (editor.dataset.mode === 'text') {
                switchSamplePromptsEditorToTableMode(editor);
            } else {
                switchSamplePromptsEditorToTextMode(editor);
            }
            updateSamplePromptModeButtonState(btn, editor);
            markSamplePromptsEditorTouched(editor);
            handleFormFieldChange();
        });
        return btn;
    }

    function updateSamplePromptModeButtonState(btn, editor) {
        if (!btn || !editor) return;
        const textMode = editor.dataset.mode === 'text';
        btn.textContent = textMode ? '表格模式' : '文本模式';
        btn.title = textMode ? '切回按列编辑提示词' : '保留注释、空行和原始参数格式';
        btn.setAttribute('aria-pressed', String(textMode));
    }

    function setSamplePromptsEditorContent(editor, content) {
        if (!editor) return;
        editor.dataset.originalContent = content || '';
        editor.dataset.touched = '0';
        renderSamplePromptRows(editor, content || '');
        updateSamplePromptModeButtonState(editor.closest('.field-row')?.querySelector('[data-sample-prompts-mode-toggle]'), editor);
    }

    function markSamplePromptsEditorTouched(editor) {
        if (editor) editor.dataset.touched = '1';
    }

    function renderSamplePromptRows(editor, content) {
        const rowsWrap = editor.querySelector('.sample-prompts-rows');
        if (!rowsWrap) return;
        rowsWrap.innerHTML = '';
        editor.dataset.mode = samplePromptsContentNeedsTextMode(content) ? 'text' : 'table';
        if (editor.dataset.mode === 'text') {
            const textarea = document.createElement('textarea');
            textarea.className = 'sample-prompts-textarea';
            textarea.value = content || '';
            textarea.spellcheck = false;
            textarea.addEventListener('input', () => markSamplePromptsEditorTouched(editor));
            rowsWrap.appendChild(textarea);
            return;
        }
        const rows = parseSamplePromptRows(content);
        for (const row of rows) {
            appendSamplePromptRow(rowsWrap, row);
        }
        updateSamplePromptRemoveButtons(rowsWrap);
    }

    function switchSamplePromptsEditorToTextMode(editor) {
        if (!editor || editor.dataset.mode === 'text') return;
        const rowsWrap = editor.querySelector('.sample-prompts-rows');
        if (!rowsWrap) return;
        const text = serializeSamplePromptsEditor(editor);
        rowsWrap.innerHTML = '';
        editor.dataset.mode = 'text';
        const textarea = document.createElement('textarea');
        textarea.className = 'sample-prompts-textarea';
        textarea.value = text;
        textarea.spellcheck = false;
        textarea.addEventListener('input', () => markSamplePromptsEditorTouched(editor));
        rowsWrap.appendChild(textarea);
        textarea.focus();
    }

    function switchSamplePromptsEditorToTableMode(editor) {
        if (!editor || editor.dataset.mode !== 'text') return;
        const rowsWrap = editor.querySelector('.sample-prompts-rows');
        if (!rowsWrap) return;
        const text = serializeSamplePromptsEditor(editor);
        rowsWrap.innerHTML = '';
        editor.dataset.mode = 'table';
        for (const row of parseSamplePromptRows(text)) {
            appendSamplePromptRow(rowsWrap, row);
        }
        updateSamplePromptRemoveButtons(rowsWrap);
        rowsWrap.querySelector('[data-sample-prompt-field="prompt"]')?.focus();
    }

    function appendSamplePromptRow(rowsWrap, row) {
        const item = document.createElement('div');
        item.className = 'sample-prompt-row';

        const promptField = createSamplePromptTextField('提示词', 'prompt', row.prompt || '');
        const heightField = createSamplePromptInputField('长 / h', 'height', row.height || '', 'number', '1');
        const widthField = createSamplePromptInputField('宽 / w', 'width', row.width || '', 'number', '1');
        const cfgField = createSamplePromptInputField('CFG / g', 'cfg', row.cfg || '', 'number', '0.1');
        const stepsField = createSamplePromptInputField('步数 / s', 'steps', row.steps || '', 'number', '1');
        const seedField = createSamplePromptInputField('种子 / d', 'seed', row.seed || '', 'number', '1');
        const extra = document.createElement('input');
        extra.type = 'hidden';
        extra.dataset.samplePromptField = 'extra';
        extra.value = row.extra || '';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn btn-small btn-subtle-danger sample-prompt-remove';
        removeBtn.textContent = '删除';
        removeBtn.addEventListener('click', () => {
            const editor = rowsWrap.closest('.sample-prompts-editor');
            const rowCount = rowsWrap.querySelectorAll('.sample-prompt-row').length;
            if (rowCount <= 1) {
                clearSamplePromptRow(item);
            } else {
                item.remove();
            }
            markSamplePromptsEditorTouched(editor);
            updateSamplePromptRemoveButtons(rowsWrap);
            handleFormFieldChange();
        });

        const rowActions = document.createElement('div');
        rowActions.className = 'sample-prompt-row-actions';
        rowActions.append(removeBtn);

        item.append(promptField, heightField, widthField, cfgField, stepsField, seedField, extra, rowActions);
        rowsWrap.appendChild(item);
        updateSamplePromptRemoveButtons(rowsWrap);
    }

    function createSamplePromptTextField(labelText, field, value) {
        const label = document.createElement('label');
        label.className = 'sample-prompt-field sample-prompt-field-text';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = 'text';
        input.dataset.samplePromptField = field;
        input.value = value || '';
        label.append(span, input);
        return label;
    }

    function createSamplePromptInputField(labelText, field, value, type = 'text', step = '') {
        const label = document.createElement('label');
        label.className = 'sample-prompt-field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = type;
        input.dataset.samplePromptField = field;
        input.value = value || '';
        if (type === 'number') {
            input.min = '0';
            input.step = step || '1';
        }
        label.append(span, input);
        return label;
    }

    function clearSamplePromptRow(row) {
        row.querySelectorAll('[data-sample-prompt-field]').forEach((input) => {
            input.value = '';
        });
    }

    function updateSamplePromptRemoveButtons(rowsWrap) {
        const rows = rowsWrap.querySelectorAll('.sample-prompt-row');
        rows.forEach((row) => {
            const button = row.querySelector('.sample-prompt-remove');
            if (!button) return;
            button.textContent = rows.length <= 1 ? '清空' : '删除';
        });
    }

    function blankSamplePromptRow() {
        return { prompt: '', height: '', width: '', cfg: '', steps: '', seed: '', extra: '' };
    }

    function samplePromptsContentNeedsTextMode(content) {
        const text = String(content || '');
        if (!text) return false;
        return text.split(/\r?\n/).some((line) => {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) return true;
            return serializeSamplePromptRow(parseSamplePromptLine(line)) !== trimmed;
        });
    }

    function parseSamplePromptRows(content) {
        const rows = String(content || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter((line) => line && !line.startsWith('#'))
            .map(parseSamplePromptLine);
        return rows.length ? rows : [blankSamplePromptRow()];
    }

    function parseSamplePromptLine(line) {
        const parts = String(line || '').trim().split(/\s+--/);
        const row = blankSamplePromptRow();
        row.prompt = (parts.shift() || '').trim();
        const extras = [];

        for (const rawPart of parts) {
            const part = rawPart.trim();
            let match = part.match(/^h\s+(\d+)$/i);
            if (match) {
                row.height = match[1];
                continue;
            }
            match = part.match(/^w\s+(\d+)$/i);
            if (match) {
                row.width = match[1];
                continue;
            }
            match = part.match(/^g\s+([\d.]+)$/i);
            if (match) {
                row.cfg = match[1];
                continue;
            }
            match = part.match(/^s\s+(\d+)$/i);
            if (match) {
                row.steps = match[1];
                continue;
            }
            match = part.match(/^d\s+(\d+)$/i);
            if (match) {
                row.seed = match[1];
                continue;
            }
            if (part) extras.push(`--${part}`);
        }
        row.extra = extras.join(' ');
        return row;
    }

    function serializeSamplePromptsEditor(editor) {
        if (editor.dataset.mode === 'text') {
            return editor.querySelector('.sample-prompts-textarea')?.value || '';
        }
        const rows = [];
        for (const rowEl of editor.querySelectorAll('.sample-prompt-row')) {
            const row = samplePromptRowFromElement(rowEl);
            const line = serializeSamplePromptRow(row);
            if (line) rows.push(line);
        }
        return rows.join('\n');
    }

    function samplePromptRowFromElement(rowEl) {
        const value = (field) => rowEl.querySelector(`[data-sample-prompt-field="${field}"]`)?.value?.trim() || '';
        return {
            prompt: value('prompt'),
            height: value('height'),
            width: value('width'),
            cfg: value('cfg'),
            steps: value('steps'),
            seed: value('seed'),
            extra: value('extra'),
        };
    }

    function serializeSamplePromptRow(row) {
        if (!row.prompt) return '';
        const args = [];
        if (row.width) args.push(`--w ${positiveIntegerText(row.width)}`);
        if (row.height) args.push(`--h ${positiveIntegerText(row.height)}`);
        if (row.steps) args.push(`--s ${positiveIntegerText(row.steps)}`);
        if (row.cfg) args.push(`--g ${positiveNumberText(row.cfg)}`);
        if (row.seed) args.push(`--d ${positiveIntegerText(row.seed)}`);
        if (row.extra) args.push(row.extra.trim());
        return [row.prompt.trim(), ...args.filter(Boolean)].join(' ');
    }

    function positiveIntegerText(value) {
        const n = Math.max(0, Math.floor(Number(value)));
        return Number.isFinite(n) ? String(n) : '';
    }

    function positiveNumberText(value) {
        const n = Number(value);
        if (!Number.isFinite(n) || n < 0) return '';
        return String(n);
    }

    function isNumericField(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) {
            return ['integer', 'number'].includes(networkArgSpec.valueType);
        }
        return typeof value === 'number' || [
            'max_train_epochs',
            'max_train_steps',
            'train_batch_size',
            'gradient_accumulation_steps',
            'sample_ratio',
            'sample_every_n_epochs',
            'sample_every_n_steps',
            'save_every_n_epochs',
            'checkpointing_epochs',
        ].includes(key);
    }

    function isIntegerNumericField(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) return networkArgSpec.valueType === 'integer';
        return [
            'max_train_epochs',
            'max_train_steps',
            'train_batch_size',
            'gradient_accumulation_steps',
            'sample_every_n_epochs',
            'sample_every_n_steps',
            'save_every_n_epochs',
            'checkpointing_epochs',
        ].includes(key) || Number.isInteger(value);
    }

    function allowsNegativeNumberField(key) {
        return ['b_cond_init', 'pe_lora_layer_from'].includes(key);
    }

    function createSelectInput(key, value, options) {
        const select = document.createElement('select');
        select.className = 'field-input field-select';
        select.dataset.valueType = fieldValueTypeForKey(key, value);
        const normalizedValue = optionValue(value);
        const normalizedOptions = options.map(optionValue);
        const displayOptions = [...options];
        if (!normalizedOptions.includes(normalizedValue)) {
            displayOptions.unshift(value);
        }

        for (const option of displayOptions) {
            const opt = document.createElement('option');
            opt.value = optionValue(option);
            opt.textContent = optionLabel(key, option);
            if (opt.value === normalizedValue) opt.selected = true;
            select.appendChild(opt);
        }
        return select;
    }

    Object.assign(FIELD_HELP_ZH, EXTRA_FIELD_HELP_ZH);

    function fieldValueType(value) {
        if (Array.isArray(value)) return 'array';
        if (typeof value === 'boolean') return 'boolean';
        if (typeof value === 'number') return 'number';
        return 'string';
    }

    function fieldValueTypeForKey(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) {
            if (networkArgSpec.valueType === 'boolean' || networkArgSpec.valueType === 'booleanInt') return 'boolean';
            if (networkArgSpec.valueType === 'integer' || networkArgSpec.valueType === 'number') return 'number';
            return 'string';
        }
        if (key === 'lora_adapter_kind') return 'string';
        if (key === 'use_lokr' || key === 'use_loha') return 'boolean';
        if (key === 'lokr_factor') return 'number';
        if (isNumericField(key, value)) return 'number';
        return fieldValueType(value);
    }

    function optionValue(value) {
        if (value === null || value === undefined) return '';
        if (typeof value === 'boolean') return value ? 'true' : 'false';
        return String(value);
    }

    function optionLabel(key, value) {
        if (key === 'lora_adapter_kind') {
            return {
                lora: '普通 LoRA',
                loha: 'LoHa',
                lokr: 'LoKr',
            }[normalizeLoraAdapterKind(value)] || String(value);
        }
        if (key === 'use_lokr') {
            return value === true || value === 'true' ? '启用 LoKr' : '普通 LoRA';
        }
        if (key === 'use_loha') {
            return value === true || value === 'true' ? '启用 LoHa' : '普通 LoRA';
        }
        if (key === 'use_moe_style' && (value === false || value === 'false')) {
            return '关闭专家路由 / false';
        }
        if (key === 'splice_position') {
            return value === 'front_of_padding' ? 'Padding 前沿 / front_of_padding' : '序列末尾 / end_of_sequence';
        }
        if (key === 'contrastive_negative_mode') {
            return {
                shuffled: '随机负样本 / shuffled',
                jaccard: 'Jaccard 降权 / jaccard',
                hard: '困难负样本 / hard',
            }[value] || String(value);
        }
        if (key === 'contrastive_objective') {
            return {
                infonce: 'InfoNCE / infonce',
                agsm: 'AGSM / agsm',
            }[value] || String(value);
        }
        if (value === true) return '开启 / true';
        if (value === false) return '关闭 / false';
        return String(value);
    }

    function generateDefaultHelp(key, value) {
        const typeStr = Array.isArray(value) ? '数组' :
            typeof value === 'boolean' ? '布尔值 (true/false)' :
            typeof value === 'number' ? '数值' : '字符串';
        const label = FIELD_LABEL_ZH[key] || key;
        const section = sectionTitleForField(key);
        const currentText = value === undefined ? '未设置' : JSON.stringify(value);
        return help(
            `${label} 是当前配置里的${section}字段，WebUI 暂时没有为它写专门教程。`,
            `按 ${typeStr} 填写。当前值: ${currentText}。如果你只是想正常训练，不需要为了“看懂它”而主动修改。`,
            ['保留这个字段可以完整复现当前 TOML 的训练行为。'],
            ['它通常属于低频或方法内部参数，改动后效果不一定能从字段名直观看出来。'],
            ['不了解来源时修改，可能导致训练启动失败、缓存失效，或让训练结果和预期不一致。'],
            '新手建议保持当前值；要改之前先看右侧 TOML 所属变体，或复制一份新配置做实验。'
        );
    }

    function sectionTitleForField(key) {
        for (const section of FORM_SECTION_DEFS) {
            if ((section.keys || []).includes(key)) return section.title;
        }
        if (String(key).includes('cache')) return '缓存/预处理';
        if (String(key).includes('sample')) return '训练中预览图';
        if (String(key).includes('router') || String(key).includes('repa') || String(key).includes('reft')) return '方法内部';
        return '高级配置';
    }

    function createHelpContent(key, value) {
        const spec = getHelpSpec(key, value);
        const content = document.createElement('div');
        content.className = 'help-content';
        addHelpSection(content, '作用', spec.summary, 'summary');
        addHelpSection(content, '怎么填', spec.fill, 'fill');
        addHelpSection(content, '好处', spec.benefit, 'benefit');
        addHelpSection(content, '代价', spec.cost, 'cost');
        addHelpSection(content, '风险', spec.risk, 'risk');
        addHelpSection(content, '推荐', spec.recommend, 'recommend');
        addHelpSection(content, 'PS', spec.ps, 'ps');
        return content;
    }

    function addHelpSection(parent, title, body, kind) {
        if (body === undefined || body === null || body === '') return;
        if (Array.isArray(body) && body.length === 0) return;

        const section = document.createElement('section');
        section.className = `help-section help-${kind}`;

        const heading = document.createElement('div');
        heading.className = 'help-heading';
        heading.textContent = title;
        section.appendChild(heading);

        if (Array.isArray(body)) {
            const list = document.createElement('ul');
            for (const item of body) {
                if (!item) continue;
                const li = document.createElement('li');
                li.textContent = item;
                list.appendChild(li);
            }
            section.appendChild(list);
        } else {
            const text = document.createElement('p');
            text.textContent = body;
            section.appendChild(text);
        }
        parent.appendChild(section);
    }

    function getHelpSpec(key, value) {
        // 优先使用内置中文说明
        if (FIELD_HELP_ZH[key]) return FIELD_HELP_ZH[key];
        // 其次从服务端获取的 field help 中取英文（作为兜底）
        const remote = fieldHelp[key];
        if (remote) {
            const remoteText = remote.en || remote.ko || '';
            if (remoteText) {
                const label = FIELD_LABEL_ZH[key] || key;
                return help(
                    `${label} 来自项目配置 schema 或方法配置，属于当前训练链路的一部分。`,
                    `${remoteText} 新手只需要确认当前值来自可信变体；不要为了试错随手改。`,
                    ['能保留上游配置说明，帮助你追踪字段来源。'],
                    ['英文说明通常偏开发者视角，仍需要结合当前方法和 TOML 判断。'],
                    ['如果字段和当前方法不匹配，可能训练启动后才暴露错误。'],
                    '不确定时保持当前变体默认值；需要实验时先另存为新配置。'
                );
            }
        }
        return generateDefaultHelp(key, value);
    }

    // ── TOML 编辑器 ──
    function updateConfigPageSummary(mode = tomlManagerMode) {
        const modeLabel = document.getElementById('config-sidebar-mode-label');
        const countLabel = document.getElementById('config-sidebar-file-count');
        const countName = document.getElementById('config-sidebar-count-label');
        const kicker = document.getElementById('config-workspace-kicker');
        const title = document.getElementById('config-workspace-title');
        const subtitle = document.getElementById('config-workspace-subtitle');
        if (modeLabel) modeLabel.textContent = mode === 'output' ? '快照' : '项目';
        if (countName) countName.textContent = mode === 'output' ? '运行目录' : '配置文件';
        if (kicker) kicker.textContent = mode === 'output' ? 'OUTPUT SNAPSHOT' : 'CONFIG PRESET';
        if (title) title.textContent = mode === 'output' ? '训练输出配置' : '训练配置';
        if (subtitle) {
            subtitle.textContent = mode === 'output'
                ? '查看全局输出文件夹里的训练快照，可复制为新的项目预设后继续编辑。'
                : '选择方法、变体、预设，编辑训练参数并引用数据集预设。';
        }
        if (!countLabel) return;
        if (mode === 'output') {
            countLabel.textContent = outputRunState.loading ? '...' : String((outputRunState.runs || []).length);
            return;
        }
        countLabel.textContent = String(tomlFiles.length || 0);
    }

    function setTomlManagerMode(mode) {
        const nextMode = mode === 'output' ? 'output' : 'project';
        tomlManagerMode = nextMode;
        document.querySelectorAll('.toml-mode-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.tomlMode === nextMode);
        });
        const projectManager = document.getElementById('toml-project-manager');
        const outputManager = document.getElementById('output-run-manager');
        const configWorkspace = document.getElementById('config-form-workspace');
        const stickyActions = document.getElementById('config-sticky-actions');
        const outputDetail = document.getElementById('output-run-detail-panel');
        const projectActions = document.querySelectorAll('.toml-primary-actions');
        const outputActions = document.getElementById('output-run-actions');
        if (projectManager) projectManager.hidden = nextMode !== 'project';
        if (outputManager) outputManager.hidden = nextMode !== 'output';
        if (configWorkspace) configWorkspace.hidden = nextMode !== 'project';
        if (stickyActions) stickyActions.hidden = nextMode !== 'project';
        if (outputDetail) outputDetail.hidden = nextMode !== 'output';
        projectActions.forEach((el) => {
            el.hidden = nextMode !== 'project';
        });
        if (outputActions) outputActions.hidden = nextMode !== 'output';
        updateConfigPageSummary(nextMode);
        if (nextMode === 'output') {
            const label = document.getElementById('toml-current-file');
            if (label) label.textContent = outputRunState.file || outputRunState.selectedRun || '训练输出配置';
            setBadge('toml-current-badge', false, '当前训练');
            setBadge('toml-trainable-badge', Boolean(outputRunState.file), '只读快照');
            setBadge('toml-lock-badge', Boolean(outputRunState.file), '只读');
            setBadge('toml-dirty-badge', false, '未保存');
            updateOutputRunActionState();
            if (!outputRunState.runs.length && !outputRunState.loading) {
                loadOutputRuns();
            } else {
                renderOutputRunManager();
            }
            return;
        }
        updateTomlSelectionUI(currentTomlFile);
        updateTomlDirtyState();
        requestAnimationFrame(updateConfigStickyPlacement);
    }

    async function switchTomlManagerMode(nextMode) {
        const normalizedMode = nextMode === 'output' ? 'output' : 'project';
        if (normalizedMode !== tomlManagerMode && normalizedMode === 'output' && hasPendingConfigChanges(currentTomlFile)) {
            if (!(await confirmDiscardTomlChanges('当前项目预设有未保存修改，切换到训练输出配置会暂时隐藏这些修改。是否继续？'))) {
                return false;
            }
        }
        setTomlManagerMode(normalizedMode);
        return true;
    }

    async function loadTomlFileList(preferredFile = '', options = {}) {
        const groups = await api('/api/config/file-groups?kind=training');
        tomlFileGroups = filterTrainingTomlGroups(groups);
        tomlFileMeta = {};
        tomlFiles = [];
        for (const group of tomlFileGroups) {
            for (const item of group.files || []) {
                tomlFiles.push(item.path);
                tomlFileMeta[item.path] = item;
            }
        }
        populateTomlFileSelect(reorderTomlFileGroups(tomlFileGroups));
        if (preferredFile && !tomlFiles.includes(preferredFile) && currentTomlFile === preferredFile) {
            await handleDeletedTomlSelection(preferredFile, '当前配置文件已不存在或已被删除');
            return;
        }
        if (preferredFile && tomlFiles.includes(preferredFile)) {
            await loadTomlFile(preferredFile, { force: options.force === true });
            return;
        }
        if (options.skipDefaultLoad) {
            updateTomlSelectionUI('');
            applyTomlLockState('');
            updateTomlDirtyState();
            return;
        }
        // 默认加载当前变体对应的文件
        const variant = currentTrainingSource.method || val('variant-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const target = currentTrainingSource.file || `configs/${methodsSubdir}/${variant}.toml`;
        if (tomlFiles.includes(target)) {
            await loadTomlFile(target);
        } else if (tomlFiles.length > 0) {
            await loadTomlFile(tomlFiles[0]);
        }
    }

    async function loadOutputRuns(options = {}) {
        if (location.protocol === 'file:') return;
        outputRunState = {
            ...outputRunState,
            loading: true,
            error: '',
        };
        renderOutputRunManager();
        try {
            const data = await api('/api/config/output-runs');
            if (!data.ok) throw new Error(data.error || '读取训练输出配置失败');
            const runs = Array.isArray(data.runs) ? data.runs : [];
            let selectedRun = outputRunState.selectedRun;
            if (selectedRun && !runs.some((item) => item.name === selectedRun)) selectedRun = '';
            if (!selectedRun && runs.length && options.keepSelection !== true) {
                selectedRun = runs[0].name || '';
            }
            outputRunState = {
                ...outputRunState,
                loading: false,
                runs,
                outputRoot: data.output_root || '',
                selectedRun,
                error: '',
            };
            renderOutputRunManager();
            if (selectedRun) {
                await loadOutputRunConfig(selectedRun, preferredOutputRunKind(selectedRun));
            } else {
                updateOutputRunSelectionUI();
            }
        } catch (e) {
            outputRunState = {
                ...outputRunState,
                loading: false,
                runs: [],
                content: '',
                file: '',
                error: e.message,
            };
            renderOutputRunManager();
            setTomlStatus('error', '读取训练输出配置失败: ' + e.message);
        }
    }

    async function loadOutputRunConfig(runName, kind = 'original') {
        const run = outputRunState.runs.find((item) => item.name === runName);
        if (!run) {
            outputRunState = { ...outputRunState, selectedRun: '', content: '', file: '' };
            renderOutputRunManager();
            return;
        }
        const available = new Set((run.files || []).map((item) => item.kind));
        const selectedKind = available.has(kind) ? kind : preferredOutputRunKind(runName);
        outputRunState = {
            ...outputRunState,
            selectedRun: runName,
            selectedKind,
            content: '读取中...',
            file: '',
            saveAsOpen: false,
            error: '',
        };
        renderOutputRunManager();
        try {
            const data = await api(`/api/config/output-runs/read?run=${encodeURIComponent(runName)}&kind=${encodeURIComponent(selectedKind)}`);
            if (!data.ok) throw new Error(data.error || '读取运行配置失败');
            outputRunState = {
                ...outputRunState,
                selectedRun: data.run || runName,
                selectedKind: data.kind || selectedKind,
                content: data.content || '',
                file: data.file || '',
                error: '',
            };
            renderOutputRunManager();
            setTomlStatus('', '');
        } catch (e) {
            outputRunState = {
                ...outputRunState,
                content: '',
                file: '',
                error: e.message,
            };
            renderOutputRunManager();
            setTomlStatus('error', '读取运行配置失败: ' + e.message);
        }
    }

    function preferredOutputRunKind(runName = outputRunState.selectedRun) {
        const run = outputRunState.runs.find((item) => item.name === runName);
        const kinds = (run?.files || []).map((item) => item.kind);
        if (kinds.includes(outputRunState.selectedKind)) return outputRunState.selectedKind;
        if (kinds.includes('original')) return 'original';
        if (kinds.includes('runtime')) return 'runtime';
        if (kinds.includes('dataset')) return 'dataset';
        return 'original';
    }

    function renderOutputRunManager() {
        renderOutputRunList();
        renderOutputRunDetail();
        updateOutputRunActionState();
        updateConfigPageSummary('output');
        if (tomlManagerMode === 'output') {
            updateOutputRunSelectionUI();
        }
    }

    function renderOutputRunList() {
        const container = document.getElementById('output-run-list');
        if (!container) return;
        container.innerHTML = '';
        if (outputRunState.loading) {
            const loading = document.createElement('div');
            loading.className = 'output-run-empty';
            loading.textContent = '正在读取全局输出文件夹...';
            container.appendChild(loading);
            return;
        }
        if (outputRunState.error) {
            const error = document.createElement('div');
            error.className = 'output-run-empty error';
            error.textContent = outputRunState.error;
            container.appendChild(error);
            return;
        }
        const runs = filteredOutputRuns();
        if (!runs.length) {
            const empty = document.createElement('div');
            empty.className = 'output-run-empty';
            empty.textContent = outputRunState.search
                ? '没有匹配的训练输出配置。'
                : `没有在 ${outputRunState.outputRoot || '输出文件夹'} 找到训练配置。`;
            container.appendChild(empty);
            return;
        }
        for (const run of runs) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'output-run-item';
            btn.classList.toggle('active', run.name === outputRunState.selectedRun);
            btn.dataset.run = run.name;
            btn.title = run.path || run.name;
            btn.addEventListener('click', () => loadOutputRunConfig(run.name, preferredOutputRunKind(run.name)));

            const name = document.createElement('strong');
            name.textContent = run.name;
            const meta = document.createElement('span');
            const fileLabels = (run.files || []).map((item) => item.label).join(' / ');
            meta.textContent = [run.mtime_text, fileLabels].filter(Boolean).join(' · ');
            const path = document.createElement('small');
            path.textContent = run.path || '';
            btn.append(name, meta, path);
            container.appendChild(btn);
        }
    }

    function renderOutputRunDetail() {
        const run = selectedOutputRun();
        const title = document.getElementById('output-run-title');
        const meta = document.getElementById('output-run-meta');
        const tabs = document.getElementById('output-run-kind-tabs');
        const viewer = document.getElementById('output-run-config-viewer');
        const saveAs = document.getElementById('output-run-save-as');
        if (title) title.textContent = run?.name || '未选择运行目录';
        if (meta) {
            meta.textContent = run
                ? [run.path, run.mtime_text].filter(Boolean).join(' · ')
                : `从 ${outputRunState.outputRoot || '全局输出文件夹'} 读取训练快照配置。`;
        }
        if (tabs) {
            tabs.innerHTML = '';
            const files = run?.files || [];
            if (!files.length) {
                const empty = document.createElement('span');
                empty.className = 'output-run-kind-empty';
                empty.textContent = '无可读 TOML';
                tabs.appendChild(empty);
            }
            for (const file of files) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'output-run-kind-btn';
                btn.classList.toggle('active', file.kind === outputRunState.selectedKind);
                btn.textContent = file.label;
                btn.title = file.file || file.filename;
                btn.addEventListener('click', () => loadOutputRunConfig(run.name, file.kind));
                tabs.appendChild(btn);
            }
        }
        if (viewer) {
            viewer.value = outputRunState.content || '';
            viewer.placeholder = run ? '这个运行目录没有可显示的配置内容。' : '选择左侧运行目录后查看配置。';
        }
        if (saveAs) {
            saveAs.hidden = !outputRunState.saveAsOpen;
        }
        renderOutputRunSaveAsControls();
    }

    function renderOutputRunSaveAsControls() {
        const input = document.getElementById('output-run-save-name');
        const select = document.getElementById('output-run-save-group');
        if (input && !input.value && outputRunState.saveAsOpen) {
            input.value = outputRunSaveAsDefaultName();
        }
        if (!select) return;
        const current = select.value || 'imported';
        select.innerHTML = '';
        const groups = saveAsTargetGroups();
        for (const group of groups) {
            const opt = document.createElement('option');
            opt.value = group.id;
            opt.textContent = group.label || group.id;
            select.appendChild(opt);
        }
        select.value = groups.some((group) => group.id === current) ? current : (groups[0]?.id || 'imported');
    }

    function filteredOutputRuns() {
        const query = outputRunState.search.trim().toLowerCase();
        if (!query) return outputRunState.runs;
        return outputRunState.runs.filter((run) => {
            const haystack = [
                run.name,
                run.path,
                run.mtime_text,
                ...(run.files || []).map((item) => `${item.kind} ${item.label} ${item.file}`),
            ].join(' ').toLowerCase();
            return haystack.includes(query);
        });
    }

    function selectedOutputRun() {
        return outputRunState.runs.find((item) => item.name === outputRunState.selectedRun) || null;
    }

    function updateOutputRunSelectionUI() {
        const label = document.getElementById('toml-current-file');
        if (label) {
            label.textContent = outputRunState.file || outputRunState.selectedRun || '训练输出配置';
        }
        setBadge('toml-current-badge', false, '当前训练');
        setBadge('toml-trainable-badge', Boolean(outputRunState.file), '只读快照');
        setBadge('toml-lock-badge', Boolean(outputRunState.file), '只读');
        setBadge('toml-dirty-badge', false, '未保存');
    }

    function updateOutputRunActionState() {
        const run = selectedOutputRun();
        const hasContent = Boolean(outputRunState.content && outputRunState.file);
        const hasOriginal = Boolean(run?.has_original);
        setButtonDisabled('btn-copy-output-config', !hasContent);
        setButtonDisabled('btn-export-output-config', !hasContent);
        const saveBtn = document.getElementById('btn-save-output-config-as');
        if (saveBtn) {
            saveBtn.disabled = !run || !hasOriginal;
            saveBtn.title = !run
                ? '请先选择一个训练运行目录'
                : (hasOriginal ? '把 config.original.toml 复制到 configs/imported，随后可在项目预设中编辑。' : '这个运行目录没有 config.original.toml，不能复制为项目预设。');
        }
    }

    function setButtonDisabled(id, disabled) {
        ctx.dom.setButtonDisabled(id, disabled);
    }

    async function copyOutputRunConfigContent() {
        const text = outputRunState.content || '';
        if (!text) return;
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                const viewer = document.getElementById('output-run-config-viewer');
                viewer?.focus();
                viewer?.select();
                document.execCommand('copy');
            }
            setTomlStatus('ok', '已复制训练输出配置内容');
        } catch (e) {
            setTomlStatus('error', '复制失败: ' + e.message);
        }
    }

    function exportOutputRunConfig() {
        if (!outputRunState.content) return;
        const filename = outputRunState.file
            ? outputRunState.file.split('/').pop()
            : `${outputRunState.selectedRun || 'output-run'}.${outputRunState.selectedKind}.toml`;
        const blob = new Blob([outputRunState.content], { type: 'application/toml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename || 'output-run-config.toml';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setTomlStatus('ok', `已导出 ${filename}`);
    }

    function openOutputRunSaveAs() {
        const run = selectedOutputRun();
        if (!run) {
            setTomlStatus('error', '请先选择一个训练运行目录');
            return;
        }
        if (!run.has_original) {
            setTomlStatus('error', '这个运行目录没有 config.original.toml，不能复制为项目预设');
            return;
        }
        outputRunState = { ...outputRunState, saveAsOpen: true };
        renderOutputRunManager();
        const input = document.getElementById('output-run-save-name');
        if (input) {
            input.value = outputRunSaveAsDefaultName();
            input.focus();
            input.select();
        }
    }

    function closeOutputRunSaveAs() {
        outputRunState = { ...outputRunState, saveAsOpen: false };
        renderOutputRunManager();
    }

    function outputRunSaveAsDefaultName() {
        const run = selectedOutputRun();
        const stem = String(run?.name || 'output_run')
            .replace(/-\d{8}-\d{6}(?:-\d+)?$/i, '')
            .replace(/[^A-Za-z0-9_.-]+/g, '_')
            .replace(/^_+|_+$/g, '') || 'output_run';
        return `${stem}_from_output`;
    }

    async function confirmOutputRunSaveAs() {
        const run = selectedOutputRun();
        if (!run) return;
        const name = val('output-run-save-name') || outputRunSaveAsDefaultName();
        const group = val('output-run-save-group') || 'imported';
        try {
            const res = await api('/api/config/output-runs/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    run: run.name,
                    name,
                    target_group: group,
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '复制为项目预设失败');
                return;
            }
            outputRunState = { ...outputRunState, saveAsOpen: false };
            if (hasPendingConfigChanges(currentTomlFile)) {
                if (!(await confirmDiscardTomlChanges('复制完成后会回到项目预设并加载新文件，当前项目预设里已有未保存修改。是否继续？'))) {
                    return;
                }
            }
            await switchTomlManagerMode('project');
            await loadTomlFileList(res.file, { force: true });
            await loadTomlFile(res.file, { force: true });
            setTomlStatus('ok', `已复制为新项目预设: ${res.file}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '复制为项目预设失败: ' + e.message);
        }
    }

    async function selectAndApplyTomlFile(filePath) {
        if (!filePath) return false;
        const previousFile = currentTomlFile;
        const previousSelect = document.getElementById('toml-file-select');
        const targetLabel = tomlFileDisplayName(filePath);
        const canSwitch = await handlePendingConfigSwitch({ targetLabel });
        if (!canSwitch) {
            if (previousSelect) previousSelect.value = previousFile || '';
            updateTomlSelectionUI(previousFile);
            return false;
        }
        await loadTomlFile(filePath, { force: true });
        const meta = tomlFileMeta[filePath];
        if (!meta?.trainable) {
            setTomlStatus('error', '已打开该配置文件，但它不是完整训练配置，不能加载为当前训练入口');
            return false;
        }
        await applyTomlToConfig({ silent: true });
        rememberSelectionSnapshot();
        setTomlStatus('ok', `已加载选中配置: ${meta.path || filePath}`);
        return true;
    }

    async function loadTomlFile(filePath, options = {}) {
        if (!options.force && !(await confirmDiscardTomlChanges('当前 TOML 有未保存修改，切换文件会丢失这些修改。是否继续？'))) {
            const select = document.getElementById('toml-file-select');
            if (select) select.value = currentTomlFile || '';
            return;
        }
        resetTomlDeleteConfirm();
        resetTomlSaveConfirm();
        const data = await api(`/api/config/raw?file=${encodeURIComponent(filePath)}`);
        if (data?.ok === false) {
            if (isMissingTomlFileResponse(data)) {
                await handleDeletedTomlSelection(filePath, data.error || '配置文件不存在或已被删除');
                return;
            }
            setTomlStatus('error', data.error || '读取配置文件失败');
            return;
        }
        currentTomlFile = filePath;
        document.getElementById('toml-file-select').value = filePath;
        tomlSavedContent = data.content || '';
        document.getElementById('toml-editor').value = tomlSavedContent;
        if (data.meta) tomlFileMeta[filePath] = data.meta;
        updateTomlSelectionUI(filePath);
        applyTomlLockState(filePath);
        updateTomlDirtyState();
        setTomlStatus('', '');
    }

    async function saveTomlFile(options = {}) {
        const file = currentTomlFile || val('toml-file-select');
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件，或使用“另存新配置”保存导入内容');
            return false;
        }
        if (isTomlLocked(file)) {
            setTomlStatus('error', '该配置文件已锁定，请使用“另存新配置”创建可编辑配置');
            return false;
        }
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(file);
        const directEditorSave = options.mode === 'editor';
        if (directEditorSave) {
            if (formDirty) {
                setTomlStatus('error', '左侧表单或数据集预设选择有未保存修改，请先使用“保存更新当前选中配置”处理后再直接保存 TOML');
                updateTomlActionState(file);
                return false;
            }
            if (!editorDirty) {
                setTomlStatus('error', '直接编辑器没有未保存的 TOML 文本修改');
                return false;
            }
            if (!options.skipConfirm && tomlSaveConfirmFile !== file) {
                armTomlSaveConfirm(file);
                return false;
            }
            resetTomlSaveConfirm({ update: false });
            return await saveRawTomlContent(file, document.getElementById('toml-editor').value, { reloadConfig: currentTrainingSource.file === file });
        }
        if (editorDirty && !formDirty && !options.skipConfirm && tomlSaveConfirmFile !== file) {
            armTomlSaveConfirm(file);
            return false;
        }
        resetTomlSaveConfirm({ update: false });
        if (currentTrainingSource.file === file) {
            const datasetApplied = await applySelectedDatasetPresetToCurrentConfig(file);
            if (!datasetApplied) return false;
            const datasetWasDirty = datasetEditorState.dirty;
            if (datasetWasDirty) {
                const datasetSaved = await saveDatasetEditor({ trainFile: file, reloadList: false });
                if (!datasetSaved) return false;
            }
            const changedValues = collectChangedFormValues({ persistDefaultFields: true });
            if (Object.keys(changedValues).length > 0) {
                return await saveFormPatchToToml(file, changedValues);
            }
            if (datasetApplied.applied || datasetWasDirty) {
                await loadConfig();
                await loadTomlFileList(file);
                updateTomlDirtyState();
                setTomlStatus('ok', datasetWasDirty ? '✓ 已保存数据集修改' : '✓ 已应用数据集预设');
                return true;
            }
        }
        const content = document.getElementById('toml-editor').value;
        return await saveRawTomlContent(file, content, { reloadConfig: currentTrainingSource.file === file });
    }

    async function saveRawTomlContent(file, content, options = {}) {
        try {
            const res = await api('/api/config/raw', {
                method: 'PUT',
                body: JSON.stringify({ file, content }),
            });
            if (res.ok) {
                tomlSavedContent = content;
                resetTomlSaveConfirm({ update: false });
                updateTomlDirtyState();
                setTomlStatus('ok', '✓ 已保存');
                await loadTomlFileList(file);
                if (options.reloadConfig) {
                    await loadConfig(); // 仅当前训练源被保存时刷新左侧表单
                }
                return true;
            } else {
                setTomlStatus('error', res.error || '保存失败');
                return false;
            }
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
            return false;
        }
    }

    async function saveFormPatchToToml(file, values) {
        try {
            const content = document.getElementById('toml-editor').value;
            const preparedValues = await prepareFormPatchValues(values);
            const res = await api('/api/config/raw', {
                method: 'PATCH',
                body: JSON.stringify({ file, values: preparedValues, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存失败');
                return false;
            }

            if (typeof res.content === 'string') {
                document.getElementById('toml-editor').value = res.content;
                tomlSavedContent = res.content;
            }
            resetTomlSaveConfirm({ update: false });
            await loadConfig();
            await loadTomlFileList(file);
            updateTomlDirtyState();
            setTomlStatus('ok', `✓ 已保存 ${res.changed?.length || Object.keys(preparedValues).length} 个表单修改`);
            return true;
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
            return false;
        }
    }

    async function applySelectedDatasetPresetToCurrentConfig(file) {
        const nextDataset = selectedConfigDatasetFile || '';
        const currentDataset = currentConfig.dataset_config || '';
        if (!nextDataset || nextDataset === currentDataset) {
            if (!nextDataset && currentDataset) {
                const res = await api('/api/config/raw', {
                    method: 'PATCH',
                    body: JSON.stringify({
                        file,
                        values: { dataset_config: '' },
                        content: document.getElementById('toml-editor')?.value || '',
                    }),
                });
                if (!res.ok) {
                    setTomlStatus('error', res.error || '清除数据集预设失败');
                    return null;
                }
                if (typeof res.content === 'string') {
                    const editor = document.getElementById('toml-editor');
                    if (editor) {
                        editor.value = res.content;
                        tomlSavedContent = res.content;
                    }
                }
                currentConfig.dataset_config = '';
                return { applied: true, response: res };
            }
            return { applied: false };
        }
        try {
            const res = await api('/api/config/dataset-presets/apply', {
                method: 'POST',
                body: JSON.stringify({
                    dataset_file: nextDataset,
                    train_file: file,
                    train_content: document.getElementById('toml-editor')?.value || '',
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '应用数据集预设失败');
                return null;
            }
            if (typeof res.train_content === 'string') {
                const editor = document.getElementById('toml-editor');
                if (editor) {
                    editor.value = res.train_content;
                    tomlSavedContent = res.train_content;
                }
            }
            currentConfig.dataset_config = res.dataset_config || nextDataset;
            const values = res.values || {};
            for (const [key, value] of Object.entries(values)) {
                currentConfig[key] = value;
            }
            return { applied: true, response: res };
        } catch (e) {
            setTomlStatus('error', '应用数据集预设失败: ' + e.message);
            return null;
        }
    }

    async function saveDatasetPresetEditor() {
        if (datasetPresetState.readonly) {
            setDatasetPresetStatus('系统数据集预设只读，请复制后编辑', 'error');
            return null;
        }
        let file = datasetPresetState.selectedFile || '';
        const wasUnnamedPreset = !file;
        if (!file) {
            const name = await showDatasetPresetNameDialog({
                title: '保存数据集预设',
                description: '当前预设还没有文件名。请输入一个名称，保存到 configs/datasets/。',
                confirmText: '保存预设',
            });
            if (name === null) return null;
            file = datasetPresetPathFromName(name);
            datasetPresetState.selectedFile = file;
        }
        const rows = normalizeDatasetEditorRows(datasetPresetState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        if (!rows.length || rows.some((row) => !row.source_dir.trim())) {
            setDatasetPresetStatus('请至少填写一个原始数据集路径', 'error');
            return null;
        }
        try {
            const res = await api('/api/config/dataset-presets', {
                method: 'PUT',
                body: JSON.stringify({
                    file,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                    overwrite: !(datasetPresetState.isNew || wasUnnamedPreset),
                }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '保存数据集预设失败', 'error');
                return null;
            }
            datasetPresetState = {
                ...datasetPresetState,
                selectedFile: res.file || file,
                datasets: normalizeDatasetEditorRows(res.datasets || rows),
                defaults: normalizeDatasetDefaults(res.defaults || datasetPresetState.defaults || {}),
                dirty: false,
                isNew: false,
                readonly: false,
                status: res.message || '已保存数据集预设',
            };
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            await loadDatasetPreset(datasetPresetState.selectedFile);
            setDatasetPresetStatus(res.message || '已保存数据集预设', 'ok');
            if (selectedConfigDatasetFile === datasetPresetState.selectedFile) {
                selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
                await loadStepEstimate();
            }
            return res;
        } catch (e) {
            setDatasetPresetStatus('保存数据集预设失败: ' + e.message, 'error');
            return null;
        }
    }

    async function createNewDatasetPreset() {
        if (datasetPresetState.dirty && !(await confirmUnsavedDiscard('当前数据集预设有未保存修改，新建会丢弃这些修改。是否继续？'))) return;
        const name = await showDatasetPresetNameDialog({
            title: '新建数据集预设',
            description: '输入新预设名称，稍后保存时会写入 configs/datasets/。',
            confirmText: '创建预设',
        });
        if (name === null) return;
        const nextFile = datasetPresetPathFromName(name);
        if (datasetPresetByFile(nextFile)) {
            setDatasetPresetStatus('数据集预设已存在，请换一个名称或使用复制/重命名', 'error');
            return;
        }
        datasetPresetState = {
            ...datasetPresetState,
            selectedFile: nextFile,
            datasets: normalizeDatasetEditorRows([{
                source_dir: '',
                image_dir: '',
                cache_dir: '',
                num_repeats: 1,
                settings: normalizeDatasetDefaults({}),
            }]),
            defaults: normalizeDatasetDefaults({}),
            dirty: true,
            isNew: true,
            readonly: false,
            error: '',
            status: '新预设尚未保存',
        };
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
    }

    async function copyDatasetPreset() {
        if (!datasetPresetState.selectedFile) return;
        const name = await showDatasetPresetNameDialog({
            title: '复制数据集预设',
            description: '使用当前编辑器中的内容复制为新的数据集预设。',
            value: `${datasetPresetState.selectedFile.split('/').pop().replace(/\.toml$/i, '')}_copy`,
            confirmText: '复制预设',
        });
        if (name === null) return;
        const rows = normalizeDatasetEditorRows(datasetPresetState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        try {
            const res = await api('/api/config/dataset-presets/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    name,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '复制数据集预设失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            await loadDatasetPreset(res.file);
            setDatasetPresetStatus('已复制数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('复制数据集预设失败: ' + e.message, 'error');
        }
    }

    async function renameDatasetPreset() {
        const oldFile = datasetPresetState.selectedFile;
        if (!oldFile || datasetPresetState.readonly) return;
        const name = await showDatasetPresetNameDialog({
            title: '重命名数据集预设',
            description: '会先保存为新预设，再删除旧 TOML；图片、缩放图和缓存目录不受影响。',
            value: oldFile.split('/').pop().replace(/\.toml$/i, ''),
            confirmText: '重命名',
        });
        if (name === null) return;
        const nextFile = datasetPresetPathFromName(name);
        if (nextFile === oldFile) return;
        const saved = await copyDatasetPresetToName(name);
        if (!saved) return;
        try {
            const del = await api(`/api/config/dataset-presets?file=${encodeURIComponent(oldFile)}`, { method: 'DELETE' });
            if (!del.ok) {
                setDatasetPresetStatus(del.error || '新预设已保存，但旧预设删除失败', 'error');
                return;
            }
            if (selectedConfigDatasetFile === oldFile) selectedConfigDatasetFile = nextFile;
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            await loadDatasetPreset(nextFile);
            renderConfigDatasetPicker();
            setDatasetPresetStatus('已重命名数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('重命名数据集预设失败: ' + e.message, 'error');
        }
    }

    async function copyDatasetPresetToName(name) {
        try {
            const res = await api('/api/config/dataset-presets/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    name,
                    datasets: datasetRowsForPayload(datasetPresetState.datasets),
                    defaults: normalizeDatasetDefaults(datasetPresetState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '保存新数据集预设失败', 'error');
                return null;
            }
            return res;
        } catch (e) {
            setDatasetPresetStatus('保存新数据集预设失败: ' + e.message, 'error');
            return null;
        }
    }

    async function deleteDatasetPreset() {
        const file = datasetPresetState.selectedFile;
        if (!file || datasetPresetState.readonly) return;
        const ok = await showAppConfirmDialog({
            title: '删除数据集预设',
            description: file,
            message: '只删除数据集预设 TOML，不删除图片、缩放图或缓存目录。',
            confirmText: '删除预设',
            danger: true,
        });
        if (!ok) return;
        try {
            const res = await api(`/api/config/dataset-presets?file=${encodeURIComponent(file)}`, { method: 'DELETE' });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '删除数据集预设失败', 'error');
                return;
            }
            if (selectedConfigDatasetFile === file) {
                selectedConfigDatasetFile = '';
                selectedConfigDatasetSummary = null;
            }
            datasetPresetState.selectedFile = '';
            datasetPresetState.dirty = false;
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            renderConfigDatasetPicker();
            setDatasetPresetStatus('已删除数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('删除数据集预设失败: ' + e.message, 'error');
        }
    }

    function importDatasetPreset() {
        document.getElementById('dataset-import-input')?.click();
    }

    async function handleDatasetPresetImport(event) {
        const fileInput = event.target;
        const file = fileInput.files?.[0];
        if (!file) return;
        try {
            const content = await file.text();
            const name = await showDatasetPresetNameDialog({
                title: '导入数据集预设',
                description: '输入导入后的预设名称，文件会保存到 configs/datasets/。',
                value: file.name.replace(/\.toml$/i, ''),
                confirmText: '导入预设',
            });
            if (name === null) return;
            const target = datasetPresetPathFromName(name);
            const res = await api('/api/config/raw/save-as', {
                method: 'POST',
                body: JSON.stringify({ file: target, content }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '导入数据集预设失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            await loadDatasetPreset(target);
            setDatasetPresetStatus('已导入数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('导入数据集预设失败: ' + e.message, 'error');
        } finally {
            fileInput.value = '';
        }
    }

    async function exportDatasetPreset() {
        const file = datasetPresetState.selectedFile;
        if (!file) return;
        try {
            const data = await api(`/api/config/dataset-presets/read?file=${encodeURIComponent(file)}`);
            if (!data.ok) {
                setDatasetPresetStatus(data.error || '导出数据集预设失败', 'error');
                return;
            }
            const blob = new Blob([data.content || ''], { type: 'text/plain;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = file.split('/').pop() || 'dataset.toml';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            setDatasetPresetStatus('已导出数据集预设', 'ok');
        } catch (e) {
            setDatasetPresetStatus('导出数据集预设失败: ' + e.message, 'error');
        }
    }

    function datasetPresetPathFromName(name) {
        const stem = String(name || '')
            .replace(/\.toml$/i, '')
            .replace(/\\/g, '/')
            .split('/')
            .pop()
            .replace(/[^A-Za-z0-9_-]+/g, '_')
            .replace(/^_+|_+$/g, '') || 'dataset';
        return `configs/datasets/${stem}.toml`;
    }

    async function showDatasetPresetNameDialog(options = {}) {
        const name = await showHistoryTaskInputDialog({
            title: options.title || '数据集预设名称',
            description: options.description || '请输入数据集预设名称。',
            label: options.label || '预设名称',
            value: options.value || '',
            placeholder: options.placeholder || '例如 rokkotsu_goddess_v2',
            confirmText: options.confirmText || '确认',
        });
        if (name === null) return null;
        const clean = name.trim();
        if (!clean) {
            setDatasetPresetStatus('请输入数据集预设名称', 'error');
            return null;
        }
        return clean;
    }

    function setDatasetPresetStatus(message, level = '') {
        datasetPresetState.status = message || '';
        const header = document.getElementById('dataset-preset-header');
        if (!header) return;
        let status = header.querySelector('.dataset-preset-status');
        if (!status) {
            status = document.createElement('div');
            status.className = 'dataset-preset-status';
            header.appendChild(status);
        }
        status.textContent = message || '';
        status.className = ['dataset-preset-status', level].filter(Boolean).join(' ');
    }

    async function createDatasetPresetGroup() {
        const label = await showHistoryTaskInputDialog({
            title: '新建数据集分组',
            description: '只用于整理 configs/datasets 下的数据集预设，不会修改训练配置内容。',
            label: '分组名称',
            placeholder: '例如：角色数据集 / 试验数据集 / 正式数据集',
            confirmText: '创建分组',
        });
        if (label === null) return;
        if (!label.trim()) {
            setDatasetPresetStatus('分组名称不能为空', 'error');
            return;
        }
        try {
            const res = await api('/api/config/file-groups', {
                method: 'POST',
                body: JSON.stringify({ label: label.trim(), kind: 'dataset' }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '创建数据集分组失败', 'error');
                return;
            }
            if (res.group?.kind !== 'dataset') {
                setDatasetPresetStatus('后端仍是旧版本，请重启 WebUI 后再创建数据集分组', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组已创建', 'ok');
        } catch (e) {
            setDatasetPresetStatus('创建数据集分组失败: ' + e.message, 'error');
        }
    }

    async function renameDatasetPresetGroup(group) {
        if (!group?.id || !group.renamable) return;
        const label = await showHistoryTaskInputDialog({
            title: '重命名数据集分组',
            description: '只修改左侧分组名称，不会改动数据集 TOML 文件路径。',
            label: '分组名称',
            value: group.label || group.id,
            placeholder: '例如：正式数据集',
            confirmText: '保存名称',
        });
        if (label === null) return;
        if (!label.trim()) {
            setDatasetPresetStatus('分组名称不能为空', 'error');
            return;
        }
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'PATCH',
                body: JSON.stringify({ label: label.trim() }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '重命名数据集分组失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组已重命名', 'ok');
        } catch (e) {
            setDatasetPresetStatus('重命名数据集分组失败: ' + e.message, 'error');
        }
    }

    async function deleteDatasetPresetGroup(group) {
        if (!group?.id || !group.deletable) return;
        const count = (group.files || []).length;
        const ok = await showHistoryTaskConfirmDialog({
            title: '删除数据集分组',
            description: group.label || group.id,
            message: count > 0
                ? `只删除这个分组，不删除其中 ${count} 个数据集 TOML；这些文件会回到默认数据集分组。`
                : '只删除这个空分组，不会删除任何 TOML 文件。',
            confirmText: '删除分组',
            danger: true,
        });
        if (!ok) return;
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '删除数据集分组失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组已删除', 'ok');
        } catch (e) {
            setDatasetPresetStatus('删除数据集分组失败: ' + e.message, 'error');
        }
    }

    async function placeDatasetPresetGroup(payload, index) {
        const groupId = payload?.groupId;
        if (!groupId) return;
        if (datasetPresetState.search.trim()) {
            setDatasetPresetStatus('筛选数据集预设时不能拖动排序，请先清空搜索', 'error');
            return;
        }
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'group', group: groupId, scope: 'dataset', index }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '调整数据集分组位置失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集分组位置已更新', 'ok');
        } catch (e) {
            setDatasetPresetStatus('调整数据集分组位置失败: ' + e.message, 'error');
        }
    }

    async function placeDatasetPresetFile(payload, groupId, index) {
        const file = payload?.file;
        if (!file || !groupId) return;
        if (datasetPresetState.search.trim()) {
            setDatasetPresetStatus('筛选数据集预设时不能拖动排序，请先清空搜索', 'error');
            return;
        }
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'file', file, group: groupId, index }),
            });
            if (!res.ok) {
                setDatasetPresetStatus(res.error || '数据集预设位置调整失败', 'error');
                return;
            }
            await loadDatasetPresets({ selectCurrent: false, manage: true });
            setDatasetPresetStatus(res.message || '数据集预设位置已更新', 'ok');
        } catch (e) {
            setDatasetPresetStatus('数据集预设位置调整失败: ' + e.message, 'error');
        }
    }

    async function saveDatasetEditor(options = {}) {
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const targetFile = options.trainFile || currentTrainingSource.file || currentTomlFile || '';
        const targetContent = options.trainContent ?? (document.getElementById('toml-editor')?.value || '');
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        const payloadRows = datasetRowsForPayload(rows);
        if (!rows.length || rows.some((row) => !row.source_dir.trim())) {
            setTomlStatus('error', '请至少填写一个原始数据集路径');
            return null;
        }
        try {
            const res = await api('/api/config/datasets', {
                method: 'PUT',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    train_file: targetFile,
                    train_content: targetContent,
                    prefer_existing_dataset_config: options.preferExistingDatasetConfig !== false,
                    datasets: payloadRows,
                    defaults: normalizeDatasetDefaults(datasetEditorState.defaults || {}),
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存数据集配置失败');
                return null;
            }
            if (typeof res.train_content === 'string' && res.train_content) {
                const editor = document.getElementById('toml-editor');
                if (editor && targetFile === (currentTomlFile || val('toml-file-select'))) {
                    editor.value = res.train_content;
                    tomlSavedContent = res.train_content;
                }
            }
        datasetEditorState = {
            loading: false,
            loaded: true,
            dirty: false,
            dataset_config: res.dataset_config || datasetEditorState.dataset_config,
            datasets: normalizeDatasetEditorRows(res.datasets || rows),
            defaults: normalizeDatasetDefaults(res.defaults || datasetEditorState.defaults || {}),
            error: '',
        };
        currentConfig.dataset_config = datasetEditorState.dataset_config;
        if (datasetEditorState.datasets[0]) {
            currentConfig.source_image_dir = datasetEditorState.datasets[0].source_dir;
            currentConfig.resized_image_dir = datasetEditorState.datasets[0].image_dir;
            currentConfig.lora_cache_dir = datasetEditorState.datasets[0].cache_dir;
            }
            syncDatasetEditorToCompatFields();
            renderDatasetEditor();
            updateTomlDirtyState();
            await loadStepEstimate();
            if (options.reloadList !== false) {
                await loadTomlFileList(targetFile);
            }
            return res;
        } catch (e) {
            setTomlStatus('error', '保存数据集配置失败: ' + e.message);
            return null;
        }
    }

    function collectChangedFormValues(options = {}) {
        syncConfigDraftFromForm(options);
        const values = {};
        for (const [key, next] of configFormState.draftValues.entries()) {
            if (!key) continue;
            if (CONFIG_FORM_INTERNAL_KEYS.has(key)) continue;
            if (isActiveNetworkArgFieldKey(key)) {
                continue;
            }
            if (key === 'sample_prompts') {
                if (samplePromptsMode === 'path') {
                    const original = typeof currentConfig.sample_prompts === 'string' ? currentConfig.sample_prompts : '';
                    if (!valuesEqual(next, original)) {
                        values[key] = next;
                    }
                    continue;
                }
                if (String(next || '') !== String(samplePromptsContent || '')) {
                    values[key] = next;
                }
                continue;
            }
            if (key === 'lora_adapter_kind') {
                continue;
            }
            const hasOriginal = key in currentConfig;
            const original = hasOriginal ? currentConfig[key] : FORM_UI_DEFAULTS[key];
            if (!hasOriginal) {
                if (shouldSkipUiDefaultField(key, next, options)) continue;
                values[key] = next;
                continue;
            }
            if (!valuesEqual(next, original)) {
                values[key] = next;
            }
        }
        const merged = collectNetworkArgsFromForm({ network_args: values.network_args ?? currentConfig.network_args });
        if (merged.changed) {
            values.network_args = merged.networkArgs;
        } else if ('network_args' in values) {
            delete values.network_args;
        }
        if (values.use_lokr === true && !('lokr_factor' in values) && !('lokr_factor' in currentConfig)) {
            values.lokr_factor = FORM_UI_DEFAULTS.lokr_factor;
        }
        return applyLoraAdapterPatch(values);
    }

    function networkArgInputChanged(input) {
        const spec = NETWORK_ARG_FIELD_MAP.get(input.dataset.key);
        if (!spec) return false;
        const original = networkArgFieldValueFromConfig(spec, currentConfig);
        const next = readFieldInputValue(input, original);
        return !valuesEqual(next, original);
    }

    function networkArgFieldValueFromConfig(spec, config = currentConfig) {
        const argMap = parseNetworkArgMap(config?.network_args);
        return coerceNetworkArgValue(argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default, spec);
    }

    function collectNetworkArgsFromForm(baseConfig = currentConfig) {
        const baseArgs = normalizeNetworkArgArray(baseConfig?.network_args);
        const formValues = new Map();
        const changedKeys = new Set();
        const applyNetworkArgFormValue = (key, value) => {
            if (!isActiveNetworkArgFieldKey(key)) return;
            const spec = NETWORK_ARG_FIELD_MAP.get(key);
            const original = networkArgFieldValueFromConfig(spec, currentConfig);
            formValues.set(spec.arg, { spec, value });
            if (!valuesEqual(value, original)) changedKeys.add(spec.key);
        };

        for (const [key, value] of configFormState.draftValues.entries()) {
            applyNetworkArgFormValue(key, value);
        }
        const inputs = [...document.querySelectorAll('#config-form .field-input[data-key]')]
            .filter((input) => isActiveNetworkArgFieldKey(input.dataset.key));
        for (const input of inputs) {
            const spec = NETWORK_ARG_FIELD_MAP.get(input.dataset.key);
            const original = networkArgFieldValueFromConfig(spec, currentConfig);
            applyNetworkArgFormValue(input.dataset.key, readFieldInputValue(input, original));
        }

        if (!formValues.size) {
            return { networkArgs: baseArgs, changed: !valuesEqual(baseArgs, currentConfig.network_args || []) };
        }

        const result = [];
        const seenArgs = new Set();
        for (const raw of baseArgs) {
            const parsed = parseNetworkArgEntry(raw);
            if (!parsed || !formValues.has(parsed.arg)) {
                result.push(raw);
                continue;
            }
            seenArgs.add(parsed.arg);
            const { spec, value } = formValues.get(parsed.arg);
            result.push(formatNetworkArg(spec, value));
        }

        for (const { spec, value } of formValues.values()) {
            if (seenArgs.has(spec.arg)) continue;
            if (!changedKeys.has(spec.key)) continue;
            result.push(formatNetworkArg(spec, value));
        }

        return {
            networkArgs: result,
            changed: !valuesEqual(result, currentConfig.network_args || []),
        };
    }

    function formatNetworkArg(spec, value) {
        return `${spec.arg}=${formatNetworkArgValue(spec, value)}`;
    }

    function formatNetworkArgValue(spec, value) {
        if (spec.valueType === 'booleanInt') return parseBooleanNetworkArg(value, spec.default) ? '1' : '0';
        if (spec.valueType === 'boolean') return parseBooleanNetworkArg(value, spec.default) ? 'true' : 'false';
        if (spec.valueType === 'integer') {
            const n = Number(value);
            return Number.isFinite(n) ? String(Math.trunc(n)) : String(spec.default);
        }
        if (spec.valueType === 'number') {
            const n = Number(value);
            return Number.isFinite(n) ? String(n) : String(spec.default);
        }
        return String(value ?? '').trim();
    }

    async function prepareFormPatchValues(values) {
        const nextValues = { ...values };
        if ('sample_prompts' in nextValues && samplePromptsMode !== 'path') {
            const promptText = String(nextValues.sample_prompts || '');
            if (promptText.trim()) {
                const saved = await saveSamplePrompts(promptText);
                nextValues.sample_prompts = saved.file || samplePromptsPath;
            } else {
                nextValues.sample_prompts = '';
            }
        }
        return nextValues;
    }

    function shouldSkipUiDefaultField(key, value, options = {}) {
        if (!(key in FORM_UI_DEFAULTS)) return false;
        if (options.persistDefaultFields && FORM_UI_PERSIST_DEFAULT_FIELDS.has(key)) return false;
        if (OPTIONAL_EMPTY_FIELDS.has(key) && value === '') return true;
        return valuesEqual(value, FORM_UI_DEFAULTS[key]);
    }

    function readFieldInputValue(input, originalValue) {
        if (input.classList?.contains('sample-prompts-editor')) {
            if (input.dataset.touched !== '1') return input.dataset.originalContent || '';
            return serializeSamplePromptsEditor(input);
        }
        if (input.tagName === 'TEXTAREA') return normalizeMultilineText(input.value);
        if (input.type === 'checkbox') return input.checked;
        const raw = input.value;
        switch (input.dataset.valueType || fieldValueType(originalValue)) {
            case 'number':
                if (String(raw).trim() === '' && OPTIONAL_EMPTY_NUMBER_FIELDS.has(input.dataset.key)) return '';
                return parseNumberValue(raw, originalValue);
            case 'boolean':
                return raw === 'true';
            case 'array':
                return parseArrayValue(raw);
            default:
                return raw;
        }
    }

    function readLoKrEnabled() {
        return readLiveLoraAdapterKind() === 'lokr';
    }

    function updateLoKrFieldState() {
        const factorInput = document.querySelector('#config-form .field-input[data-key="lokr_factor"]');
        if (!factorInput) return;
        const enabled = readLoKrEnabled();
        factorInput.disabled = !enabled;
        factorInput.title = enabled ? '' : '启用 LoKr 后生效';
        const row = factorInput.closest('.field-row');
        if (row) row.classList.toggle('field-row-disabled', !enabled);
    }

    function parseNumberValue(raw, fallback) {
        const trimmed = String(raw).trim();
        if (trimmed === '' && fallback === '') return '';
        if (trimmed === '') return fallback;
        const n = Number(trimmed);
        return Number.isFinite(n) ? n : fallback;
    }

    function parseArrayValue(raw) {
        const trimmed = String(raw).trim();
        if (!trimmed) return [];
        try {
            const parsed = JSON.parse(trimmed);
            return Array.isArray(parsed) ? parsed : [parsed];
        } catch {
            return trimmed.split(',').map((item) => item.trim()).filter(Boolean);
        }
    }

    function valuesEqual(a, b) {
        if (isBooleanLikeValue(a) && isBooleanLikeValue(b)) {
            return normalizeBooleanLikeValue(a) === normalizeBooleanLikeValue(b);
        }
        if (isNumberLikeValue(a) && isNumberLikeValue(b)) {
            return Number(a) === Number(b);
        }
        return JSON.stringify(a) === JSON.stringify(b);
    }

    function isBooleanLikeValue(value) {
        return value === true || value === false || value === 'true' || value === 'false';
    }

    function normalizeBooleanLikeValue(value) {
        return value === true || value === 'true';
    }

    function isNumberLikeValue(value) {
        if (typeof value === 'number') return Number.isFinite(value);
        if (typeof value !== 'string') return false;
        const trimmed = value.trim();
        return trimmed !== '' && Number.isFinite(Number(trimmed));
    }

    function normalizeMultilineText(value) {
        return String(value || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean)
            .join('\n');
    }

    function currentSamplePromptText(config) {
        const raw = typeof config.sample_prompts === 'string' ? config.sample_prompts.trim() : '';
        const previousMode = samplePromptsMode;
        const previousPath = samplePromptsPath;
        const previousContent = samplePromptsContent;
        samplePromptsPath = DEFAULT_SAMPLE_PROMPTS_PATH;
        samplePromptsContent = '';

        if (!raw) {
            samplePromptsMode = 'editor-inline';
            return FORM_UI_DEFAULTS.sample_prompts;
        }
        if (isEditableSamplePromptsTextFilePath(raw)) {
            const nextPath = normalizeSamplePromptsPath(raw);
            samplePromptsMode = 'editor-file';
            samplePromptsPath = nextPath;
            if (previousMode === 'editor-file' && previousPath === nextPath) {
                samplePromptsContent = previousContent || '';
                return samplePromptsContent || FORM_UI_DEFAULTS.sample_prompts;
            }
            return FORM_UI_DEFAULTS.sample_prompts;
        }
        if (isSamplePromptsFilePath(raw)) {
            samplePromptsMode = 'path';
            return raw;
        }

        samplePromptsMode = 'editor-inline';
        samplePromptsContent = raw;
        return raw;
    }

    function normalizeSamplePromptsPath(value) {
        return String(value || '').replace(/\\/g, '/').trim();
    }

    function isEditableSamplePromptsTextFilePath(value) {
        const text = normalizeSamplePromptsPath(value);
        if (!text.toLowerCase().endsWith('.txt')) return false;
        if (!text.startsWith('configs/')) return false;
        return !text.split('/').includes('..');
    }

    function isSamplePromptsFilePath(value) {
        const text = normalizeSamplePromptsPath(value).toLowerCase();
        return text.endsWith('.txt') || text.endsWith('.toml') || text.endsWith('.json');
    }

    async function loadSamplePrompts(filePath = samplePromptsPath, parentSeq = configLoadSeq) {
        if (location.protocol === 'file:') return;
        const requestSeq = ++samplePromptsLoadSeq;
        try {
            const data = await api(`/api/config/sample-prompts?file=${encodeURIComponent(filePath || samplePromptsPath)}`);
            if (parentSeq !== configLoadSeq || requestSeq !== samplePromptsLoadSeq) return;
            if (data?.ok === false) {
                throw new Error(data.error || '读取预览提示词失败');
            }
            samplePromptsPath = data.file || samplePromptsPath;
            samplePromptsContent = data.content || '';
            const input = document.querySelector('#config-form .field-input[data-key="sample_prompts"]');
            if (input) {
                if (configFormState.draftValues.has('sample_prompts')) return;
                if (input.classList?.contains('sample-prompts-editor')) {
                    setSamplePromptsEditorContent(input, samplePromptsContent);
                } else {
                    input.value = samplePromptsContent;
                }
            }
        } catch (e) {
            console.warn('读取预览提示词失败:', e);
        }
    }

    async function saveSamplePrompts(content) {
        const res = await api('/api/config/sample-prompts', {
            method: 'PUT',
            body: JSON.stringify({
                file: samplePromptsPath,
                train_config_file: currentTrainingSource.file || currentTomlFile || '',
                content,
            }),
        });
        if (!res.ok) {
            throw new Error(res.error || '保存预览提示词失败');
        }
        samplePromptsPath = res.file || samplePromptsPath;
        samplePromptsContent = res.content || '';
        return res;
    }

    async function importTomlFile() {
        if (!(await confirmDiscardTomlChanges('当前 TOML 有未保存修改，导入会覆盖编辑器内容。是否继续？'))) {
            return;
        }
        const input = document.getElementById('toml-import-input');
        if (!input) return;
        input.value = '';
        input.click();
    }

    function handleTomlImport(event) {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = () => {
            currentTomlFile = '';
            tomlSavedContent = '';
            document.getElementById('toml-current-file').textContent = `未保存导入: ${file.name}`;
            document.getElementById('toml-file-select').value = '';
            document.getElementById('toml-editor').value = reader.result || '';
            setTomlEditorLocked(false);
            updateTomlSelectionUI('');
            applyTomlLockState('');
            updateTomlDirtyState();
            setTomlStatus('ok', `已导入 ${file.name}，点击保存或另存为写入项目`, { persist: true });
        };
        reader.onerror = () => {
            setTomlStatus('error', '导入失败: 无法读取本地文件');
        };
        reader.readAsText(file, 'utf-8');
    }

    function exportTomlFile() {
        const content = document.getElementById('toml-editor').value;
        const file = currentTomlFile || val('toml-file-select');
        const filename = exportTomlFilename(file);
        downloadTomlContent(content, filename);
        setTomlStatus('ok', `已导出 ${filename}`);
    }

    function downloadTomlContent(content, filename) {
        ctx.download.downloadText(content, filename, 'application/toml;charset=utf-8');
    }

    function triggerDownload(url, filename) {
        ctx.download.triggerDownload(url, filename);
    }

    function downloadBlob(blob, filename) {
        ctx.download.downloadBlob(blob, filename);
    }

    function createTomlZipBlob(entries) {
        return ctx.download.createZipBlob(entries, uniqueZipEntryName);
    }

    function uniqueZipEntryName(name, usedNames) {
        const base = exportTomlFilename(name || 'config.toml');
        if (!usedNames.has(base)) {
            usedNames.add(base);
            return base;
        }
        const stem = base.replace(/\.toml$/i, '');
        let index = 2;
        let candidate = `${stem}-${index}.toml`;
        while (usedNames.has(candidate)) {
            index += 1;
            candidate = `${stem}-${index}.toml`;
        }
        usedNames.add(candidate);
        return candidate;
    }

    async function saveTomlAs() {
        const editor = document.getElementById('toml-editor');
        const currentFile = currentTomlFile;
        const target = await showTomlSaveAsDialog(currentFile);
        if (target === null) return;

        const file = normalizeTomlSaveAsPath(target?.name ?? target);
        const targetGroupId = target?.group || 'imported';
        if (!file) {
            setTomlStatus('error', '另存新配置失败: 请先输入新的配置名称');
            return;
        }
        if (file === currentFile) {
            setTomlStatus('error', '另存新配置失败: 新配置不能和当前选中文件同名');
            return;
        }
        if (tomlFiles.includes(file)) {
            setTomlStatus('error', `${file} 已存在，请换一个新的配置名称`);
            return;
        }

        try {
            const baseContent = editor.value;
            const preparedValues = await prepareFormPatchValues(collectChangedFormValues({ persistDefaultFields: true }));
            const content = Object.keys(preparedValues).length
                ? await previewPatchedTomlContent(file, baseContent, preparedValues)
                : baseContent;
            const res = await api('/api/config/raw/save-as', {
                method: 'POST',
                body: JSON.stringify({ file, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '另存为失败');
                return;
            }

            currentTrainingSource = {
                method: file.split('/').pop().replace(/\.toml$/i, ''),
                methods_subdir: 'imported',
                file,
            };
            currentTomlFile = file;
            tomlSavedContent = content;
            editor.value = content;
            const datasetApplied = await applySelectedDatasetPresetToCurrentConfig(file);
            if (!datasetApplied) {
                setTomlStatus('error', `新配置已创建: ${file}，但数据集预设应用失败，请修正后再次保存更新当前选中配置`, { persist: true });
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            if (datasetApplied.applied) {
                const editorAfterDataset = document.getElementById('toml-editor');
                tomlSavedContent = editorAfterDataset?.value || tomlSavedContent;
            }
            const moved = await moveTomlFileToGroup(file, targetGroupId);
            if (!moved) {
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            await loadTomlFileList(file);
            await applyTomlToConfig({ silent: true });
            updateTomlDirtyState();
            const groupLabel = saveAsTargetGroups().find((group) => group.id === targetGroupId)?.label || targetGroupId;
            setTomlStatus('ok', `已另存新配置: ${file} → ${groupLabel}`);
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function createBlankPresetFromLoraTemplate() {
        let templateContent = '';
        try {
            const data = await api(`/api/config/raw?file=${encodeURIComponent(BLANK_PRESET_TEMPLATE_FILE)}`);
            if (data?.ok === false) {
                setTomlStatus('error', data.error || '读取 LoRA 模板失败');
                return;
            }
            templateContent = typeof data.content === 'string' ? data.content : '';
        } catch (e) {
            setTomlStatus('error', '读取 LoRA 模板失败: ' + e.message);
            return;
        }
        if (!templateContent.trim()) {
            setTomlStatus('error', `读取 LoRA 模板失败: ${BLANK_PRESET_TEMPLATE_FILE} 内容为空或不存在`);
            return;
        }

        const target = await showTomlSaveAsDialog(BLANK_PRESET_TEMPLATE_FILE, {
            title: '创建空白预设配置',
            description: `以 ${BLANK_PRESET_TEMPLATE_LABEL} 为模板，并套用全局基础模型路径，创建一个新的可编辑项目预设。`,
            confirmText: '创建空白预设配置',
            hint: '新文件默认创建到 configs/imported/；分组只影响右侧列表归类。全局模型路径只作为初始默认值，创建后仍可在配置页覆盖。',
            currentText: `模板: ${BLANK_PRESET_TEMPLATE_LABEL} (${BLANK_PRESET_TEMPLATE_FILE})`,
        });
        if (target === null) return;

        const file = normalizeTomlSaveAsPath(target?.name ?? target);
        const targetGroupId = target?.group || 'imported';
        if (!file) {
            setTomlStatus('error', '创建空白预设配置失败: 请先输入新的配置名称');
            return;
        }
        if (file === BLANK_PRESET_TEMPLATE_FILE) {
            setTomlStatus('error', '创建空白预设配置失败: 不能覆盖 LoRA 标准模板');
            return;
        }
        if (tomlFiles.includes(file)) {
            setTomlStatus('error', `${file} 已存在，请换一个新的配置名称`);
            return;
        }

        const canSwitch = await handlePendingConfigSwitch({ targetLabel: `新的空白预设配置 ${file.split('/').pop() || file}` });
        if (!canSwitch) return;

        try {
            // 空白预设先复用模板，再把全局默认基础模型路径灌进去，减少新建后手工改三项的次数。
            const globalModelPathOverrides = getGlobalModelPathOverrides();
            const content = Object.keys(globalModelPathOverrides).length
                ? await previewPatchedTomlContent(file, templateContent, globalModelPathOverrides)
                : templateContent;
            const res = await api('/api/config/raw/save-as', {
                method: 'POST',
                body: JSON.stringify({ file, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '创建空白预设配置失败');
                return;
            }

            const moved = await moveTomlFileToGroup(file, targetGroupId);
            if (!moved) {
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            await loadTomlFileList(file, { force: true });
            await applyTomlToConfig({ silent: true });
            updateTomlDirtyState();
            const groupLabel = saveAsTargetGroups().find((group) => group.id === targetGroupId)?.label || targetGroupId;
            setTomlStatus('ok', `已创建空白预设配置: ${file} → ${groupLabel}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '创建空白预设配置失败: ' + e.message);
        }
    }

    async function previewPatchedTomlContent(file, content, values) {
        const res = await api('/api/config/raw/patch-preview', {
            method: 'POST',
            body: JSON.stringify({ file, content, values }),
        });
        if (!res.ok) {
            throw new Error(res.error || '应用表单修改失败');
        }
        return typeof res.content === 'string' ? res.content : content;
    }

    async function showTomlSaveAsDialog(currentFile, options = {}) {
        const wrap = document.createElement('div');
        wrap.className = 'toml-save-as-dialog-body';

        const label = document.createElement('label');
        label.className = 'history-task-dialog-field';
        const labelText = document.createElement('span');
        labelText.textContent = options.nameLabel || '新配置名称或 configs/ 路径';
        const input = document.createElement('input');
        input.type = 'text';
        input.value = '';
        input.placeholder = options.placeholder || '例如 rokkotsu_v2 或 configs/imported/rokkotsu_v2.toml';
        input.className = 'history-task-dialog-input';
        label.append(labelText, input);

        const groups = saveAsTargetGroups();
        const groupWrap = document.createElement('div');
        groupWrap.className = 'toml-save-as-group-list';
        const groupTitle = document.createElement('span');
        groupTitle.className = 'toml-save-as-group-title';
        groupTitle.textContent = '保存到分组';
        groupWrap.appendChild(groupTitle);

        const radios = [];
        for (const group of groups) {
            const option = document.createElement('label');
            option.className = 'toml-move-option';

            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'toml-save-as-target-group';
            radio.value = group.id;
            radio.checked = group.id === 'imported' || (!radios.length && !groups.some((item) => item.id === 'imported'));
            radios.push(radio);

            const text = document.createElement('span');
            const title = document.createElement('strong');
            title.textContent = group.label || group.id;
            const detail = document.createElement('small');
            detail.textContent = `${(group.files || []).length} 个配置`;
            text.append(title, detail);

            option.append(radio, text);
            groupWrap.appendChild(option);
        }

        const hint = document.createElement('p');
        hint.className = 'toml-save-as-hint';
        hint.textContent = options.hint || '只填写文件名时会创建到 configs/imported/，分组只影响右侧列表归类；必须使用新名称，不会覆盖当前选中配置。';

        const current = document.createElement('p');
        current.className = 'toml-save-as-current';
        current.textContent = options.currentText
            || (currentFile ? `当前选中配置: ${currentFile}` : '当前没有选中的配置文件，将使用编辑器内容创建新配置。');

        wrap.append(label, groupWrap, hint, current);

        return showHistoryTaskDialog({
            title: options.title || '另存新配置',
            description: options.description || '输入一个新名称，并选择它在右侧配置列表中的目标分组。',
            body: wrap,
            confirmText: options.confirmText || '创建配置文件',
            onOpen: () => input.focus(),
            getValue: () => {
                const checked = wrap.querySelector('input[name="toml-save-as-target-group"]:checked');
                return {
                    name: input.value,
                    group: checked?.value || 'imported',
                };
            },
        });
    }

    function saveAsTargetGroups() {
        const trainingGroups = filterTrainingTomlGroups(tomlFileGroups);
        const groups = reorderTomlFileGroups(trainingGroups)
            .filter((group) => group.trainable && group.movable && !group.locked && !group.user_group_locked);
        if (groups.some((group) => group.id === 'imported')) return groups;
        const imported = trainingGroups.find((group) => group.id === 'imported');
        if (imported && imported.trainable && !imported.locked && !imported.user_group_locked) {
            return [imported, ...groups];
        }
        return groups.length ? groups : [{
            id: 'imported',
            label: '导入配置',
            files: [],
        }];
    }

    async function moveTomlFileToGroup(file, groupId) {
        if (!groupId || groupId === 'imported') return true;
        try {
            const res = await api('/api/config/file-groups/move-file', {
                method: 'POST',
                body: JSON.stringify({ file, group: groupId }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '另存成功，但移动到指定分组失败');
                return false;
            }
            return true;
        } catch (e) {
            setTomlStatus('error', '另存成功，但移动分组请求失败: ' + e.message);
            return false;
        }
    }

    function normalizeTomlSaveAsPath(rawPath) {
        let file = String(rawPath || '').trim().replace(/\\/g, '/');
        file = file.replace(/^\/+/, '');
        if (!file) return '';
        if (!file.startsWith('configs/')) {
            file = `configs/imported/${file}`;
        }
        if (!file.toLowerCase().endsWith('.toml')) {
            file += '.toml';
        }
        return file;
    }

    function exportTomlFilename(filePath) {
        const base = String(filePath || '').split('/').filter(Boolean).pop();
        if (!base) return 'anima-config.toml';
        return base.toLowerCase().endsWith('.toml') ? base : `${base}.toml`;
    }

    function isFixedSystemTomlGroup(group) {
        return Boolean(
            group.id === 'web_config' ||
            group.id === 'presets' ||
            group.id === 'methods' ||
            group.id === 'gui_methods' ||
            group.system_locked
        );
    }

    function isDatasetConfigGroup(group) {
        if (!group) return false;
        const id = String(group.id || '');
        const kind = String(group.kind || '').toLowerCase();
        if (kind === 'dataset' || kind === 'datasets' || id === 'datasets' || id === 'unfiled_datasets') return true;
        return (group.files || []).some((item) => String(item.path || '').replace(/\\/g, '/').startsWith('configs/datasets/'));
    }

    function isTrainingTomlGroup(group) {
        return Boolean(group) && !isDatasetConfigGroup(group);
    }

    function filterTrainingTomlGroups(groups) {
        return (Array.isArray(groups) ? groups : []).filter(isTrainingTomlGroup);
    }

    function shouldShowTomlGroup(group) {
        return isTrainingTomlGroup(group) && !isFixedSystemTomlGroup(group);
    }

    function reorderTomlFileGroups(groups) {
        return [...(groups || [])]
            .map((group, index) => ({ group, index }))
            .filter(({ group }) => isTrainingTomlGroup(group) && (group.user_managed || group.lockable || (group.files || []).length > 0))
            .sort((a, b) => {
                const aFixed = isFixedSystemTomlGroup(a.group);
                const bFixed = isFixedSystemTomlGroup(b.group);
                if (aFixed !== bFixed) return aFixed ? 1 : -1;
                return a.index - b.index;
            })
            .map((item) => item.group);
    }

    function getSortableTomlGroups() {
        return [...(tomlFileGroups || [])]
            .filter((group) => isTomlGroupDraggable(group));
    }

    function isTomlGroupDraggable(group) {
        return Boolean(
            group?.id &&
            isTrainingTomlGroup(group) &&
            !isFixedSystemTomlGroup(group) &&
            !group.locked &&
            !group.user_group_locked &&
            (group.user_managed || group.lockable || (group.files || []).length > 0)
        );
    }

    function canDropTomlFileToGroup(group) {
        return Boolean(
            group?.id &&
            isTrainingTomlGroup(group) &&
            group.movable &&
            !group.locked &&
            !group.user_group_locked
        );
    }

    function isTomlFileDraggable(item) {
        return Boolean(item?.path && !item.locked && !hasPendingConfigChanges(currentTomlFile));
    }

    function createTomlGroupDragHandle(group, details) {
        const disabled = !isTomlGroupDraggable(group);
        return createFileGroupDragHandle({
            target: 'group',
            scope: 'training',
            groupId: group.id,
            sourceElement: details,
            canDrag: () => isTomlGroupDraggable(group),
            blockedMessage: () => setTomlStatus('error', '该配置分组不能拖动排序'),
        }, {
            disabled,
            label: `拖动配置分组 ${group.label || group.id}`,
            title: disabled ? '该配置分组不能拖动排序' : '拖动调整配置分组顺序',
        });
    }

    async function placeTomlGroup(payload, index) {
        const groupId = payload?.groupId;
        if (!groupId) return;
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'group', group: groupId, scope: 'training', index }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '调整分组位置失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组位置已更新');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function placeTomlFile(payload, groupId, index) {
        const file = payload?.file;
        if (!file || !groupId) return;
        if (hasPendingConfigChanges(currentTomlFile)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再拖动排序');
            updateTomlActionState(currentTomlFile);
            return;
        }
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'file', file, group: groupId, index }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '配置位置调整失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || file);
            setTomlStatus('ok', res.message || '配置位置已更新');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    function tomlFileDragOptions() {
        return {
            scope: 'training',
            rowSelector: '.toml-file-row-wrap',
            canDropToGroup: canDropTomlFileToGroup,
            onDrop: placeTomlFile,
        };
    }

    function tomlGroupDragOptions() {
        return {
            scope: 'training',
            getSortableGroups: () => getSortableTomlGroups(),
            canDropOnGroup: (group) => isTomlGroupDraggable(group),
            onDrop: placeTomlGroup,
        };
    }

    function populateTomlFileSelect(groups) {
        const sel = document.getElementById('toml-file-select');
        const prev = sel.value;
        sel.innerHTML = '';
        for (const group of groups) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = group.label || group.id || '配置文件';
            for (const item of group.files || []) {
                const opt = document.createElement('option');
                opt.value = item.path;
                opt.textContent = [tomlLockLabel(item), tomlFileDisplayName(item)].filter(Boolean).join(' / ');
                opt.dataset.locked = item.locked ? '1' : '0';
                optgroup.appendChild(opt);
            }
            sel.appendChild(optgroup);
        }
        if (tomlFiles.includes(prev)) {
            sel.value = prev;
        }
        renderTomlFileGroups(groups);
        updateConfigPageSummary('project');
    }

    function renderTomlFileGroups(groups) {
        const container = document.getElementById('toml-file-groups');
        if (!container) return;
        container.innerHTML = '';
        const stored = readTomlGroupState();

        const toolbar = document.createElement('div');
        toolbar.className = 'toml-group-toolbar';
        const createBtn = document.createElement('button');
        createBtn.type = 'button';
        createBtn.className = 'toml-group-action-btn';
        createBtn.textContent = '新建分组';
        createBtn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            runTomlGroupAction(createTomlGroup, createBtn);
        });
        toolbar.appendChild(createBtn);
        container.appendChild(toolbar);

        const visibleGroups = (groups || []).filter(shouldShowTomlGroup);
        if (visibleGroups.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'toml-file-group-empty';
            empty.textContent = '系统分组已隐藏。可点击“新建分组”创建自己的配置分组。';
            container.appendChild(empty);
        }

        for (const group of visibleGroups) {
            const details = document.createElement('details');
            details.className = 'toml-file-group';
            if (group.locked) details.classList.add('readonly');
            details.dataset.groupId = group.id;
            details.open = stored[group.id] ?? Boolean(group.open);
            details.addEventListener('toggle', () => {
                const next = readTomlGroupState();
                next[group.id] = details.open;
                writeTomlGroupState(next);
            });

            const summary = document.createElement('summary');
            const groupHandle = createTomlGroupDragHandle(group, details);
            if (groupHandle) summary.appendChild(groupHandle);
            const title = document.createElement('span');
            title.className = 'toml-group-title';
            title.textContent = `${group.label || group.id} (${(group.files || []).length})`;
            summary.appendChild(title);
            const actions = createTomlGroupActions(group);
            if (actions) summary.appendChild(actions);
            if (group.lockable) {
                const groupLockBtn = document.createElement('button');
                groupLockBtn.type = 'button';
                groupLockBtn.className = 'toml-group-lock-btn';
                groupLockBtn.textContent = group.user_group_locked ? '解除分组锁定' : '锁定分组';
                groupLockBtn.title = group.user_group_locked
                    ? '解除该分组的用户锁定'
                    : '锁定该分组内所有文件，防止误保存';
                groupLockBtn.addEventListener('click', (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();
                    runTomlGroupAction(() => toggleTomlGroupLock(group), groupLockBtn);
                });
                summary.appendChild(groupLockBtn);
            }
            if (group.locked) {
                const badge = document.createElement('em');
                badge.textContent = group.user_group_locked ? '分组锁定' : '锁定';
                summary.appendChild(badge);
            }
            setupFileGroupHeaderDropTarget(summary, group, tomlFileDragOptions());
            details.appendChild(summary);

            const list = document.createElement('div');
            list.className = 'toml-file-list';
            setupFileGroupListDropTarget(list, group, tomlFileDragOptions());
            const files = group.files || [];
            if (!files.length) {
                const empty = document.createElement('div');
                empty.className = 'toml-file-group-empty';
                empty.textContent = group.user_managed ? '空分组，可使用“移动”放入当前配置。' : '暂无配置文件。';
                list.appendChild(empty);
            }
            files.forEach((item, index) => {
                list.appendChild(createTomlFileButton(item, group, index, files.length));
            });
            details.appendChild(list);
            setupConfigGroupDropTarget(details, group, tomlGroupDragOptions());
            container.appendChild(details);
        }
        updateTomlSelectionUI(currentTomlFile);
    }

    function createTomlGroupActions(group) {
        const wrap = document.createElement('span');
        wrap.className = 'toml-group-actions';

        const queueableFiles = queueableTomlGroupFiles(group);
        wrap.appendChild(createTomlGroupActionButton('加入队列', () => enqueueTomlGroupToQueue(group), {
            title: queueableFiles.length
                ? `将该分组内 ${queueableFiles.length} 个可训练配置加入训练队列`
                : '该分组没有可加入队列的训练配置',
            disabled: !queueableFiles.length,
            variant: 'queue',
        }));
        if (group.renamable) {
            wrap.appendChild(createTomlGroupActionButton('重命名', () => renameTomlGroup(group), {
                title: '重命名这个配置分组',
            }));
        }
        const exportableFiles = exportableTomlGroupFiles(group);
        wrap.appendChild(createTomlGroupActionButton('导出分组', () => exportTomlGroup(group), {
            title: exportableFiles.length
                ? `将该分组内 ${exportableFiles.length} 个配置导出为一个 zip，内部保留独立 TOML 文件`
                : '该分组没有可导出的 TOML 文件',
            disabled: !exportableFiles.length,
            variant: 'export',
        }));
        wrap.appendChild(createTomlGroupActionButton('删除分组', () => deleteTomlGroup(group), {
            title: deleteTomlGroupButtonTitle(group),
            danger: true,
            disabled: !canDeleteTomlGroup(group),
        }));
        return wrap;
    }

    function exportableTomlGroupFiles(group) {
        return (group?.files || [])
            .filter((item) => item?.path && String(item.path).toLowerCase().endsWith('.toml'));
    }

    async function exportTomlGroup(group) {
        const files = exportableTomlGroupFiles(group);
        if (!files.length) {
            setTomlStatus('error', '该分组没有可导出的 TOML 文件');
            return;
        }
        if (hasPendingConfigChanges(currentTomlFile)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再导出分组');
            updateTomlActionState(currentTomlFile);
            return;
        }

        const filename = `${exportTomlGroupFilename(group)}.zip`;
        setTomlStatus('pending', `正在读取分组“${group.label || group.id}”中的 ${files.length} 个配置...`, { persist: true });

        try {
            const entries = await Promise.all(files.map(async (item) => {
                const path = String(item.path || '');
                const data = await api(`/api/config/raw?file=${encodeURIComponent(path)}`);
                if (data?.ok === false) {
                    throw new Error(`${path}: ${data.error || '读取失败'}`);
                }
                return {
                    name: item.filename || path,
                    content: data.content || '',
                };
            }));
            const blob = createTomlZipBlob(entries);
            downloadBlob(blob, filename);
            setTomlStatus('ok', `已导出分组“${group.label || group.id}”：1 个 zip，内含 ${files.length} 个独立 TOML 文件`, { persist: true });
        } catch (e) {
            setTomlStatus('error', `导出分组失败: ${e.message || e}`, { persist: true });
        }
    }

    function exportTomlGroupFilename(group) {
        const raw = String(group?.label || group?.id || 'toml-group').trim();
        const safe = raw.replace(/[\\/:*?"<>|\r\n\t]+/g, '_').replace(/\s+/g, '_').replace(/^[._]+|[._]+$/g, '');
        return safe || 'toml-group';
    }

    function queueableTomlGroupFiles(group) {
        return (group?.files || [])
            .filter((item) => item?.path && item.trainable)
            .filter((item) => !String(item.path || '').replace(/\\/g, '/').startsWith('configs/datasets/'));
    }

    function tomlItemQueueVariant(item) {
        if (item?.method) return item.method;
        const filename = String(item?.filename || item?.path || '').split('/').pop() || '';
        return filename.toLowerCase().endsWith('.toml') ? filename.slice(0, -5) : filename;
    }

    async function showTomlGroupQueueConfirmDialog(group, files) {
        const wrap = document.createElement('div');
        wrap.className = 'history-task-dialog-message toml-group-queue-dialog';

        const strong = document.createElement('strong');
        strong.textContent = `${group.label || group.id || '配置分组'} · ${files.length} 个配置`;
        const message = document.createElement('p');
        message.textContent = `确认后会按当前 GPU 选择和当前预设 ${val('preset-select') || 'default'}，把该分组内可训练配置逐个冻结并加入队列；队列会保持暂停，等待你手动继续。`;

        const list = document.createElement('div');
        list.className = 'toml-group-queue-list';
        files.slice(0, 12).forEach((item) => {
            const row = document.createElement('code');
            row.textContent = item.path || tomlFileDisplayName(item);
            list.appendChild(row);
        });
        if (files.length > 12) {
            const more = document.createElement('span');
            more.textContent = `还有 ${files.length - 12} 个配置...`;
            list.appendChild(more);
        }

        wrap.append(strong, message, list);
        return showHistoryTaskDialog({
            title: '批量加入训练队列',
            description: '这会创建独立运行配置，不会修改原 TOML 文件。',
            body: wrap,
            confirmText: '确认加入队列',
            cancelText: '取消',
            getValue: () => true,
        });
    }

    async function enqueueTomlGroupToQueue(group) {
        const files = queueableTomlGroupFiles(group);
        if (!files.length) {
            setTomlStatus('error', '该分组没有可加入队列的训练配置');
            return;
        }
        if (hasPendingConfigChanges(currentTomlFile)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再批量加入队列');
            updateTomlActionState(currentTomlFile);
            return;
        }
        const confirmed = await showTomlGroupQueueConfirmDialog(group, files);
        if (!confirmed) return;

        const preset = val('preset-select') || 'default';
        let queued = 0;
        let failure = null;
        for (const item of files) {
            const variant = tomlItemQueueVariant(item);
            const methodsSubdir = item.methods_subdir || 'imported';
            renderPreflightPending({
                title: '批量加入训练队列',
                message: `正在加入 ${queued + 1}/${files.length}: ${item.filename || item.label || item.path}`,
                detail: '正在为这个配置创建独立运行配置；如果某个配置预检测失败，批量操作会停在该配置。',
            });
            if (!variant || isCliOnlySpdSource(variant, methodsSubdir)) {
                failure = { item, error: variant ? 'SPD CLI 实验配置不能通过 Web 队列启动' : '配置缺少可训练变体名称' };
                break;
            }
            try {
                const res = await enqueueTrainingQueueRequest({
                    variant,
                    preset,
                    methodsSubdir,
                    configFile: item.path,
                    willAutoPreprocess: true,
                    startPaused: true,
                    continuePayload: {},
                });
                if (!res.ok) {
                    failure = { item, error: res.error || '加入队列失败', preflight: res.preflight };
                    break;
                }
                queued += 1;
                updateTrainingQueueFromPayload(res);
            } catch (e) {
                failure = { item, error: e.message || '请求失败' };
                break;
            }
        }

        if (queued > 0) {
            document.querySelector('[data-tab="training"]')?.click();
            showTrainingView('queue');
            appendLog(`[状态] 已将分组“${group.label || group.id}”中的 ${queued} 个配置加入训练队列`);
        }
        if (failure) {
            const fileLabel = failure.item?.path || tomlFileDisplayName(failure.item);
            const message = `批量加入队列已停止：${fileLabel}，${failure.error}`;
            setTomlStatus('error', queued ? `已加入 ${queued} 个配置；${message}` : message, { persist: true });
            if (failure.preflight) {
                showPreflightDialog(failure.preflight, false, { willAutoPreprocess: true });
            } else {
                showPreflightRequestError(message);
            }
            return;
        }

        const dialog = document.getElementById('preflight-dialog');
        if (dialog?.open) dialog.close('queued-group');
        setTomlStatus('ok', `已将 ${queued} 个配置加入训练队列`, { persist: true });
    }

    function createTomlGroupActionButton(label, handler, options = {}) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'toml-group-action-btn',
            options.variant ? `toml-group-action-btn-${options.variant}` : '',
            options.danger ? 'danger' : '',
        ].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.disabled = Boolean(options.disabled);
        btn.title = options.title || label;
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            if (!btn.disabled) runTomlGroupAction(handler, btn);
        });
        return btn;
    }

    function runTomlGroupAction(handler, button = null) {
        if (tomlGroupActionBusy) return;
        tomlGroupActionBusy = true;
        if (button) button.disabled = true;
        Promise.resolve()
            .then(handler)
            .catch((e) => {
                setTomlStatus('error', '分组操作失败: ' + e.message);
            })
            .finally(() => {
                tomlGroupActionBusy = false;
                if (button?.isConnected) button.disabled = false;
            });
    }

    function createTomlFileButton(item, group = null) {
        const row = document.createElement('div');
        row.className = 'toml-file-row-wrap';
        row.dataset.file = item.path;
        row.dataset.groupId = group?.id || item.group || '';
        setupFileGroupRowDropTarget(row, group, item.path, tomlFileDragOptions());

        const dragHandle = createFileGroupDragHandle({
            target: 'file',
            scope: 'training',
            file: item.path,
            groupId: group?.id || item.group || '',
            sourceElement: row,
            canDrag: () => isTomlFileDraggable(item),
            blockedMessage: () => {
                const message = hasPendingConfigChanges(currentTomlFile)
                    ? '当前配置尚未保存，请先保存或放弃修改后再拖动排序'
                    : '该配置文件不能拖动排序';
                setTomlStatus('error', message);
            },
        }, {
            disabled: !isTomlFileDraggable(item),
            label: `拖动配置文件 ${tomlFileDisplayName(item)}`,
            title: isTomlFileDraggable(item)
                ? '拖动调整配置文件位置或移动到其他分组'
                : '当前配置文件不能拖动',
        });
        row.appendChild(dragHandle);

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'toml-file-item';
        if (item.locked) btn.classList.add('readonly');
        btn.dataset.file = item.path;
        btn.title = tomlFileDisplayName(item);
        btn.addEventListener('click', () => selectAndApplyTomlFile(item.path));

        const name = document.createElement('span');
        name.className = 'toml-file-name';
        name.textContent = item.label || item.path;
        btn.appendChild(name);

        const meta = document.createElement('span');
        meta.className = 'toml-file-meta';
        const tags = [];
        if (item.filename && item.filename !== item.label) tags.push(item.filename);
        if (currentTrainingSource.file === item.path) tags.push('当前训练');
        const lockLabel = tomlLockLabel(item);
        if (lockLabel) tags.push(lockLabel);
        tags.push(item.trainable ? '可训练' : '非训练');
        tags.push(item.path);
        meta.textContent = tags.join(' / ');
        btn.appendChild(meta);
        row.appendChild(btn);
        return row;
    }

    function updateTomlSelectionUI(filePath) {
        document.querySelectorAll('.toml-file-item').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.file === filePath);
        });
        const label = document.getElementById('toml-current-file');
        if (label) label.textContent = filePath ? tomlFileDisplayName(filePath) : '未保存导入内容';
        const applyBtn = document.getElementById('btn-apply-toml');
        if (applyBtn) {
            const meta = tomlFileMeta[filePath];
            const dirty = hasPendingConfigChanges(filePath);
            applyBtn.disabled = !meta?.trainable || dirty;
            applyBtn.title = dirty
                ? '当前配置尚未保存，请先保存或另存为'
                : (meta?.trainable ? '将该配置作为当前表单和训练入口' : '该文件不是完整训练配置');
        }
        updateTomlBadges(filePath);
    }

    function isTomlDirty() {
        const editor = document.getElementById('toml-editor');
        if (!editor) return false;
        return editor.value !== tomlSavedContent;
    }

    function hasUnsavedFormChanges(filePath = currentTomlFile) {
        if (!filePath || currentTrainingSource.file !== filePath) return false;
        if (!currentConfig || Object.keys(currentConfig).length === 0) return false;
        return datasetEditorState.dirty
            || selectedConfigDatasetFile !== (currentConfig.dataset_config || '')
            || Object.keys(collectChangedFormValues()).length > 0;
    }

    function hasPendingConfigChanges(filePath = currentTomlFile) {
        return isTomlDirty() || hasUnsavedFormChanges(filePath);
    }

    async function confirmDiscardTomlChanges(message) {
        if (!hasPendingConfigChanges(currentTomlFile)) return true;
        return confirmUnsavedDiscard(message);
    }

    function confirmUnsavedDiscard(message) {
        return showAppConfirmDialog({
            title: '未保存更改',
            description: '当前页面有尚未保存的修改',
            message,
            confirmText: '继续并丢弃',
            cancelText: '留在当前页面',
            danger: true,
        });
    }

    function collectPendingConfigChangeDetails(pending = pendingConfigSwitchState()) {
        const changes = [];
        if (pending.formDirty) {
            if (selectedConfigDatasetFile !== (currentConfig.dataset_config || '')) {
                changes.push({
                    label: '数据集预设 / dataset_config',
                    original: currentConfig.dataset_config || '未设置',
                    next: selectedConfigDatasetFile || '未设置',
                });
            }
            for (const [key, nextValue] of Object.entries(collectChangedFormValues())) {
                changes.push({
                    label: formatFieldName(key),
                    original: originalValueForChange(key),
                    next: nextValue,
                });
            }
            if (datasetEditorState.dirty) {
                changes.push({
                    label: '多数据集路径与参数',
                    original: currentConfig.dataset_config || '当前配置内的数据集字段',
                    next: summarizeDatasetEditorState(datasetEditorState),
                });
            }
        }
        if (pending.editorDirty) {
            const editorValue = document.getElementById('toml-editor')?.value || '';
            changes.push({
                label: '直接编辑 TOML',
                original: summarizeTextChange(tomlSavedContent),
                next: summarizeTextChange(editorValue),
            });
        }
        return changes;
    }

    function originalValueForChange(key) {
        if (key === 'sample_prompts' && samplePromptsMode !== 'path') {
            return samplePromptsContent || '';
        }
        if (isActiveNetworkArgFieldKey(key)) {
            return networkArgFieldValueFromConfig(NETWORK_ARG_FIELD_MAP.get(key), currentConfig);
        }
        if (key in currentConfig) return currentConfig[key];
        return FORM_UI_DEFAULTS[key];
    }

    function summarizeDatasetEditorState(state) {
        const rows = normalizeDatasetEditorRows(state.datasets || []);
        const parts = rows.map((row, index) => {
            const settings = normalizeDatasetDefaults(row.settings || state.defaults || {});
            const triggerClone = normalizeTriggerClone(row.trigger_clone);
            return [
                `第 ${index + 1} 组`,
                row.source_dir || '未设置原始路径',
                `重复 ${row.num_repeats || 1}`,
                `${settings.resolution}px`,
                `标注 ${captionSourceModeLabel(settings.caption_source_mode)}`,
                triggerClone.enabled ? `触发克隆 x${triggerClone.num_repeats}` : '',
            ].filter(Boolean).join(' · ');
        });
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        parts.push(`通用 keep_tokens ${defaults.keep_tokens}`);
        return parts.join('\n');
    }

    function summarizeTextChange(text) {
        const value = String(text || '');
        const lines = value.split(/\r?\n/).length;
        const chars = value.length;
        const preview = value.split(/\r?\n/).find((line) => line.trim()) || '空内容';
        return `${lines} 行 / ${chars} 字符\n${preview}`;
    }

    function formatConfigChangeValue(value) {
        let text;
        if (typeof value === 'string') {
            text = value;
        } else {
            try {
                text = JSON.stringify(value, null, 2);
            } catch {
                text = String(value);
            }
        }
        if (text === '') text = '空';
        return text.length > 600 ? `${text.slice(0, 600)}\n...` : text;
    }

    function showConfigSwitchToast(filePath, stateText) {
        const toast = document.getElementById('config-switch-toast');
        if (!toast) return;
        if (configSwitchToastTimer) {
            clearTimeout(configSwitchToastTimer);
            configSwitchToastTimer = null;
        }
        const file = (filePath || currentTomlFile || '当前配置').split('/').pop() || '当前配置';
        toast.textContent = `${file}，${stateText}`;
        toast.hidden = false;
        configSwitchToastTimer = setTimeout(() => {
            toast.hidden = true;
            configSwitchToastTimer = null;
        }, 2000);
    }

    async function handlePendingConfigSwitch({ targetLabel = '' } = {}) {
        const pending = pendingConfigSwitchState();
        if (!pending.hasChanges) return true;
        const action = await showUnsavedConfigSwitchDialog({ pending, targetLabel });
        if (action === 'cancel') return false;
        if (action === 'discard') {
            showConfigSwitchToast(pendingToastLabel(pending), '更改未保存');
            return true;
        }
        const saved = await savePendingConfigSwitchChanges(pending);
        if (!saved) return false;
        showConfigSwitchToast(pendingToastLabel(pending), '更改保存完成');
        return true;
    }

    function pendingConfigSwitchState() {
        const editorFile = currentTomlFile || val('toml-file-select') || '';
        const formFile = currentTrainingSource.file || '';
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(formFile);
        const dirtyFiles = [];
        if (editorDirty && editorFile) dirtyFiles.push(editorFile);
        if (formDirty && formFile && !dirtyFiles.includes(formFile)) dirtyFiles.push(formFile);
        const canSave = dirtyFiles.length > 0 && dirtyFiles.every((file) => file && !isTomlLocked(file));
        return {
            editorDirty,
            editorFile,
            formDirty,
            formFile,
            dirtyFiles,
            sourceFile: dirtyFiles[0] || formFile || editorFile || '',
            hasChanges: editorDirty || formDirty,
            canSave,
        };
    }

    function pendingToastLabel(pending) {
        const files = pending?.dirtyFiles || [];
        if (files.length > 1) {
            const first = files[0].split('/').pop() || files[0];
            return `${first} 等 ${files.length} 个配置`;
        }
        return pending?.sourceFile || currentTomlFile || '当前配置';
    }

    function sharedHistoryTaskDialogParts() {
        const dialog = document.getElementById('history-task-dialog');
        const title = document.getElementById('history-task-dialog-title');
        const desc = document.getElementById('history-task-dialog-desc');
        const body = document.getElementById('history-task-dialog-body');
        const cancelBtn = document.getElementById('history-task-dialog-cancel');
        const confirmBtn = document.getElementById('history-task-dialog-confirm');
        const closeBtn = dialog?.querySelector('.history-task-dialog-header button[value="cancel"]');
        const form = dialog?.querySelector('form');
        if (!dialog || !title || !desc || !body || !cancelBtn || !confirmBtn) return null;
        return { dialog, title, desc, body, cancelBtn, confirmBtn, closeBtn, form };
    }

    function sharedHistoryTaskDialogIsOpen(dialog) {
        return Boolean(dialog?.open || dialog?.hasAttribute?.('open'));
    }

    function openSharedHistoryTaskDialog(dialog) {
        document.body.classList.remove('history-task-dialog-fallback-open');
        if (typeof dialog.showModal === 'function') {
            try {
                dialog.showModal();
                return;
            } catch (e) {
                if (sharedHistoryTaskDialogIsOpen(dialog)) return;
            }
        }
        dialog.setAttribute('open', 'open');
        dialog.setAttribute('role', 'dialog');
        dialog.setAttribute('aria-modal', 'true');
        document.body.classList.add('history-task-dialog-fallback-open');
    }

    function closeSharedHistoryTaskDialog(dialog, value, fallbackClose) {
        dialog.returnValue = value || '';
        if (typeof dialog.close === 'function' && sharedHistoryTaskDialogIsOpen(dialog)) {
            try {
                dialog.close(dialog.returnValue);
                return;
            } catch (e) {
                /* 部分浏览器的 dialog close 可能在 fallback 状态下抛错，继续手动关闭。 */
            }
        }
        dialog.removeAttribute('open');
        fallbackClose();
    }

    async function savePendingConfigSwitchChanges(pending) {
        if (pending.editorDirty) {
            const savedEditor = await saveTomlFile({ skipConfirm: true, source: 'switch' });
            if (!savedEditor) return false;
        }
        if (pending.formDirty && (!pending.editorDirty || pending.formFile !== pending.editorFile)) {
            if (currentTomlFile !== pending.formFile) {
                await loadTomlFile(pending.formFile, { force: true });
            }
            const savedForm = await saveTomlFile({ skipConfirm: true, source: 'switch' });
            if (!savedForm) return false;
        }
        return true;
    }

    function showUnsavedConfigSwitchDialog({ pending = pendingConfigSwitchState(), targetLabel = '' } = {}) {
        const parts = sharedHistoryTaskDialogParts();
        if (!parts) {
            return Promise.resolve('cancel');
        }
        const { dialog, title, desc, body, cancelBtn, confirmBtn, closeBtn, form } = parts;
        if (sharedDialogBusy || sharedHistoryTaskDialogIsOpen(dialog)) {
            return Promise.resolve('cancel');
        }
        sharedDialogBusy = true;

        title.textContent = '有更改待保存';
        desc.textContent = targetLabel ? `即将切换到 ${targetLabel}` : '即将切换配置';
        body.innerHTML = '';
        body.appendChild(createConfigSwitchDialogBody(pending));
        cancelBtn.textContent = '放弃未保存的更改';
        cancelBtn.value = 'discard';
        confirmBtn.textContent = '保存更改并切换';
        confirmBtn.value = 'save';
        confirmBtn.disabled = !pending.canSave;
        confirmBtn.title = pending.canSave ? '' : '存在只读配置，不能直接保存；请先另存为可编辑配置，或放弃未保存更改后切换。';
        confirmBtn.classList.remove('btn-danger');
        confirmBtn.classList.add('btn-primary');
        cancelBtn.hidden = false;
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
                const value = event.submitter?.value || (confirmBtn.disabled ? 'cancel' : 'save');
                if (value === 'save' && confirmBtn.disabled) return;
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
                sharedDialogBusy = false;
                cancelBtn.hidden = false;
                cancelBtn.value = 'cancel';
                confirmBtn.value = 'confirm';
                confirmBtn.title = '';
                if (closeBtn) closeBtn.value = 'cancel';
            };
            const handleClose = () => {
                if (settled) return;
                settled = true;
                const action = dialog.returnValue === 'save'
                    ? 'save'
                    : (dialog.returnValue === 'discard' ? 'discard' : 'cancel');
                cleanup();
                resolve(action);
            };
            dialog.addEventListener('close', handleClose);
            form?.addEventListener('submit', submitDialog);
            closeBtn?.addEventListener('click', closeClick);
            cancelBtn.addEventListener('click', closeClick);
            confirmBtn.addEventListener('click', closeClick);
            dialog.addEventListener('keydown', keydownDialog);
            try {
                openSharedHistoryTaskDialog(dialog);
            } catch {
                cleanup();
                resolve('cancel');
                return;
            }
            requestAnimationFrame(() => cancelBtn.focus());
        });
    }

    function createConfigSwitchDialogBody(pending = pendingConfigSwitchState()) {
        const wrap = document.createElement('div');
        wrap.className = 'config-switch-dialog-body';

        const intro = document.createElement('p');
        intro.textContent = '当前配置有未保存修改。请先选择保存后切换，或放弃这些修改后继续切换。';
        wrap.appendChild(intro);

        const list = document.createElement('div');
        list.className = 'config-switch-change-list';
        const changes = collectPendingConfigChangeDetails(pending);
        if (!changes.length) {
            const empty = document.createElement('p');
            empty.textContent = '检测到未保存状态，但没有可展示的字段差异。';
            list.appendChild(empty);
        }
        for (const change of changes) {
            const item = document.createElement('article');
            item.className = 'config-switch-change-item';

            const label = document.createElement('strong');
            label.textContent = change.label;
            item.appendChild(label);

            const values = document.createElement('div');
            values.className = 'config-switch-change-values';
            values.appendChild(createConfigSwitchChangeValue('原始', change.original));
            values.appendChild(createConfigSwitchChangeValue('未保存的更改', change.next));
            item.appendChild(values);
            list.appendChild(item);
        }
        wrap.appendChild(list);
        return wrap;
    }

    function createConfigSwitchChangeValue(labelText, value) {
        const box = document.createElement('div');
        box.className = 'config-switch-change-value';
        const label = document.createElement('span');
        label.textContent = labelText;
        const code = document.createElement('code');
        code.textContent = formatConfigChangeValue(value);
        box.append(label, code);
        return box;
    }

    function showAppConfirmDialog(options) {
        return showHistoryTaskConfirmDialog({
            title: options.title || '确认操作',
            description: options.description || '',
            message: options.message || '',
            confirmText: options.confirmText || '确认',
            cancelText: options.cancelText || '取消',
            danger: options.danger,
        }).then(Boolean);
    }

    function updateTomlDirtyState() {
        if (!hasPendingConfigChanges(currentTomlFile)) {
            resetTomlSaveConfirm({ update: false });
        }
        updateChangedFieldMarks();
        updateTomlBadges(currentTomlFile);
        updateTomlActionState(currentTomlFile);
    }

    function updateChangedFieldMarks() {
        let changedCount = 0;
        document.querySelectorAll('#config-form .field-input[data-key]').forEach((input) => {
            const changed = configFieldInputChanged(input);
            input.closest('.field-row')?.classList.toggle('field-row-changed', changed);
            if (changed) changedCount += 1;
        });
        for (const [key, value] of configFormState.draftValues.entries()) {
            if (!document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`)
                && configDraftValueChanged(key, value)) {
                changedCount += 1;
            }
        }
        const count = document.getElementById('config-modified-count');
        if (count) count.textContent = String(changedCount);
    }

    function configFieldInputChanged(input) {
        const key = input?.dataset?.key;
        if (!key || CONFIG_FORM_INTERNAL_KEYS.has(key)) return false;
        const original = originalConfigFieldValue(key);
        const next = readFieldInputValue(input, original);
        return configDraftValueChanged(key, next, original);
    }

    function updateTomlBadges(filePath) {
        const meta = tomlFileMeta[filePath];
        setBadge('toml-current-badge', Boolean(filePath && currentTrainingSource.file === filePath), '当前训练');
        setBadge('toml-trainable-badge', Boolean(filePath), meta?.trainable ? '可训练' : '非训练');
        setBadge('toml-lock-badge', Boolean(meta?.locked), tomlLockLabel(meta) || '只读');
        setBadge('toml-dirty-badge', hasPendingConfigChanges(filePath), '未保存');
    }

    function setBadge(id, visible, text) {
        const badge = document.getElementById(id);
        if (!badge) return;
        badge.hidden = !visible;
        badge.textContent = text;
    }

    function updateTomlActionState(filePath) {
        const meta = tomlFileMeta[filePath];
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(filePath);
        const dirty = editorDirty || formDirty;
        const saveBtn = document.getElementById('btn-save-toml');
        if (saveBtn) {
            saveBtn.disabled = Boolean(meta?.locked) || !filePath || !dirty;
            saveBtn.textContent = '保存更新当前选中配置';
            saveBtn.classList.remove('btn-confirm-danger');
            saveBtn.title = meta?.locked
                ? '该配置文件已锁定，请使用新名称另存新配置后编辑'
                : (dirty
                    ? (formDirty
                        ? '把左侧表单、数据集预设选择和采样提示词等修改写回当前选中的 TOML；保存后训练会使用这些新值。'
                        : '把直接编辑器里的 TOML 文本写回当前文件。')
                    : '当前配置没有未保存修改，不需要保存。');
        }
        updateTomlEditorPanelState(filePath);
        const applyBtn = document.getElementById('btn-apply-toml');
        if (applyBtn) {
            applyBtn.disabled = !meta?.trainable || dirty;
            applyBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : (meta?.trainable
                    ? '把右侧选中的 TOML 加载到左侧表单，并把它设为“开始训练”使用的配置。'
                    : '该文件不是完整训练配置，不能作为训练入口。');
        }
        const moveBtn = document.getElementById('btn-move-toml-group');
        if (moveBtn) {
            const canMove = Boolean(filePath && meta && !meta.locked && !dirty && getMovableTomlGroups(meta.group).length > 0);
            moveBtn.disabled = !canMove;
            moveBtn.title = dirty
                ? '当前配置尚未保存，请先保存或放弃修改后再移动分组位置'
                : (meta?.locked
                    ? `${tomlLockLabel(meta) || '只读'}配置不能移动分组位置`
                    : (canMove ? '只调整右侧配置文件列表里的分组归属，不会改 TOML 内容或磁盘路径。' : '当前没有其他可移入的分组'));
        }
        const reloadBtn = document.getElementById('btn-reload-toml');
        if (reloadBtn) {
            reloadBtn.disabled = !filePath;
            reloadBtn.title = '从磁盘重新读取当前配置文件；未保存的编辑会被丢弃，但不会切换训练入口。';
        }
        const lockBtn = document.getElementById('btn-lock-toml');
        if (lockBtn) {
            const hasFile = Boolean(filePath && meta);
            const isSystemOrGroupLocked = Boolean(meta?.system_locked || meta?.group_locked);
            lockBtn.disabled = !hasFile || isSystemOrGroupLocked || dirty;
            lockBtn.textContent = meta?.user_locked ? '解除锁定' : '锁定当前文件';
            lockBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : lockTomlButtonTitle(meta);
        }
        const deleteBtn = document.getElementById('btn-delete-toml');
        if (deleteBtn) {
            const canDelete = Boolean(filePath && meta && !meta.locked && !dirty);
            if (!canDelete) resetTomlDeleteConfirm({ update: false });
            const confirming = canDelete && tomlDeleteConfirmFile === filePath;
            deleteBtn.disabled = !canDelete;
            deleteBtn.textContent = confirming ? '确认删除配置' : '删除当前配置';
            deleteBtn.classList.toggle('btn-confirm-danger', confirming);
            deleteBtn.title = dirty
                ? '当前配置尚未保存，请先保存或放弃修改后再删除'
                : (confirming ? '再次点击才会真正删除当前配置文件' : deleteTomlButtonTitle(meta));
        }
        const restoreBtn = document.getElementById('btn-restore-system-toml');
        if (restoreBtn) {
            restoreBtn.disabled = dirty;
            restoreBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : '从项目内置版本还原系统预设；会覆盖系统预设文件，还原前会自动备份。用户导入配置不会被还原。';
        }
        const startBtn = document.getElementById('btn-start-from-config');
        const trainingConfigFile = currentTrainingConfigFile();
        const canStart = Boolean(trainingConfigFile) && !dirty;
        if (startBtn) {
            startBtn.disabled = !canStart;
            startBtn.textContent = '开始训练';
            startBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : (canStart ? '运行训练前预检测，通过后选择立即启动或加入队列。' : '请先选择可训练配置文件');
        }
        const queueBtn = document.getElementById('btn-queue-from-config');
        const canQueue = Boolean(trainingConfigFile) && !dirty;
        if (queueBtn) {
            queueBtn.disabled = !canQueue;
            queueBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : (canQueue ? '把当前训练配置直接冻结并加入训练队列。' : '请先选择可训练配置文件');
        }
    }

    function readTomlGroupState() {
        try {
            return JSON.parse(localStorage.getItem('anima.tomlGroupOpen') || '{}') || {};
        } catch {
            return {};
        }
    }

    function writeTomlGroupState(state) {
        localStorage.setItem('anima.tomlGroupOpen', JSON.stringify(state));
    }

    function isTomlLocked(filePath) {
        return Boolean(tomlFileMeta[filePath]?.locked);
    }

    function applyTomlLockState(filePath) {
        const locked = isTomlLocked(filePath);
        setTomlEditorLocked(locked);
        updateTomlActionState(filePath);
    }

    function setTomlEditorLocked(locked) {
        const editor = document.getElementById('toml-editor');
        editor.readOnly = locked;
        editor.title = locked ? '该配置文件已锁定，只能导出或使用新名称另存新配置' : '';
    }

    function updateTomlEditorPanelState(filePath = currentTomlFile) {
        const panel = document.getElementById('toml-edit-panel');
        const manager = document.getElementById('config-project-workspace') || document.querySelector('.toml-manager');
        const directEditor = document.getElementById('config-direct-editor');
        const toggleBtn = document.getElementById('btn-toggle-toml-editor');
        const saveDirectBtn = document.getElementById('btn-save-toml-direct');
        const copyBtn = document.getElementById('btn-copy-toml');
        const meta = tomlFileMeta[filePath];
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(filePath);
        const dirty = editorDirty || formDirty;
        const locked = Boolean(meta?.locked);
        const confirming = Boolean(filePath && tomlSaveConfirmFile === filePath);
        if (toggleBtn) {
            const open = Boolean(panel && !panel.hidden && tomlManagerMode === 'project');
            if (manager) manager.classList.toggle('toml-edit-open', open);
            if (directEditor) directEditor.hidden = !open;
            toggleBtn.disabled = !filePath;
            toggleBtn.textContent = open ? '收起配置文件编辑' : '直接编辑配置文件';
            toggleBtn.classList.toggle('active', open);
            toggleBtn.title = open
                ? '收起二级配置文件编辑界面；不会自动保存修改。'
                : '展开二级界面，查看、复制或直接编辑当前 TOML。适合批量改字段，保存时需要二次确认。';
        }
        if (saveDirectBtn) {
            saveDirectBtn.disabled = locked || !filePath || !editorDirty || formDirty;
            saveDirectBtn.textContent = confirming ? '确认保存配置文件' : '保存配置文件';
            saveDirectBtn.classList.toggle('btn-confirm-danger', confirming);
            saveDirectBtn.title = locked
                ? '该配置文件已锁定，请使用新名称另存新配置后编辑'
                : (formDirty
                    ? '左侧表单或数据集还有未保存修改，请先用“保存更新当前选中配置”保存。'
                    : (editorDirty
                    ? (confirming ? '再次点击才会真正写入磁盘；请确认 TOML 内容没有语法错误。' : '第一次点击进入确认，第二次点击保存，防止误覆盖配置文件。')
                    : '直接编辑器没有未保存的 TOML 文本修改'));
        }
        if (copyBtn) {
            copyBtn.disabled = !filePath && !document.getElementById('toml-editor')?.value;
            copyBtn.title = '复制当前编辑器里的 TOML 内容，方便备份、对比或发给别人排查。';
        }
    }

    function toggleTomlEditorPanel() {
        const panel = document.getElementById('toml-edit-panel');
        if (!panel) return;
        if (!currentTomlFile) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        panel.hidden = !panel.hidden;
        updateTomlEditorPanelState(currentTomlFile);
        if (!panel.hidden) {
            document.getElementById('toml-editor')?.focus();
        }
    }

    async function copyTomlEditorContent() {
        const editor = document.getElementById('toml-editor');
        if (!editor) return;
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(editor.value);
            } else {
                editor.focus();
                editor.select();
                document.execCommand('copy');
            }
            setTomlStatus('ok', '已复制当前配置内容');
        } catch (e) {
            setTomlStatus('error', '复制失败: ' + e.message);
        }
    }

    function tomlLockLabel(meta) {
        if (!meta?.locked) return '';
        if (meta.system_locked) return '系统只读';
        if (meta.user_locked) return '用户锁定';
        if (meta.user_group_locked) return '分组锁定';
        if (meta.group_locked) return '分组只读';
        return meta.lock_reason_label || '只读';
    }

    function tomlFileDisplayParts(fileOrMeta) {
        const meta = typeof fileOrMeta === 'string'
            ? (tomlFileMeta[fileOrMeta] || { path: fileOrMeta })
            : (fileOrMeta || {});
        const path = meta.path || '';
        const filename = meta.filename || (path ? path.split('/').pop() : '');
        const label = meta.label || '';
        const parts = [];
        if (label && label !== filename && label !== path) parts.push(label);
        if (filename) parts.push(filename);
        if (path && path !== filename) parts.push(path);
        return parts;
    }

    function tomlFileDisplayName(fileOrMeta) {
        const parts = tomlFileDisplayParts(fileOrMeta);
        return parts.length ? parts.join(' / ') : '未命名配置文件';
    }

    function lockTomlButtonTitle(meta) {
        if (!meta) return '请先选择一个配置文件';
        if (meta.system_locked) return '系统预设已内置锁定，不能手动解除';
        if (meta.group_locked) return '该文件属于只读分组，不能手动解除';
        if (meta.user_group_locked) return '该文件所在分组已锁定，请在分组标题解除锁定';
        if (meta.user_locked) return '解除你为该文件设置的锁定';
        return '锁定当前文件，防止误保存';
    }

    function deleteTomlButtonTitle(meta) {
        if (!meta) return '请先选择一个配置文件';
        if (meta.locked) return `${tomlLockLabel(meta) || '只读'}配置不能删除`;
        return '删除当前选中的配置文件';
    }

    function resetTomlDeleteConfirm(options = {}) {
        if (tomlDeleteConfirmTimer) {
            clearTimeout(tomlDeleteConfirmTimer);
            tomlDeleteConfirmTimer = null;
        }
        if (!tomlDeleteConfirmFile) return;
        tomlDeleteConfirmFile = '';
        if (options.update !== false) {
            updateTomlActionState(currentTomlFile);
        }
    }

    function armTomlDeleteConfirm(file) {
        resetTomlDeleteConfirm({ update: false });
        tomlDeleteConfirmFile = file;
        tomlDeleteConfirmTimer = setTimeout(() => {
            resetTomlDeleteConfirm();
            setTomlStatus('', '');
        }, 8000);
        updateTomlActionState(file);
        setTomlStatus('error', `再次点击“确认删除配置”才会删除: ${file}`);
    }

    function resetTomlSaveConfirm(options = {}) {
        if (tomlSaveConfirmTimer) {
            clearTimeout(tomlSaveConfirmTimer);
            tomlSaveConfirmTimer = null;
        }
        if (!tomlSaveConfirmFile) return;
        tomlSaveConfirmFile = '';
        if (options.update !== false) {
            updateTomlActionState(currentTomlFile);
        }
    }

    function armTomlSaveConfirm(file) {
        resetTomlSaveConfirm({ update: false });
        tomlSaveConfirmFile = file;
        tomlSaveConfirmTimer = setTimeout(() => {
            resetTomlSaveConfirm();
            setTomlStatus('', '');
        }, 8000);
        updateTomlActionState(file);
        setTomlStatus('error', `再次点击“确认保存”才会写入当前配置: ${file}`);
    }

    function setTomlStatus(cls, text, options = {}) {
        const el = document.getElementById('toml-status');
        if (tomlStatusTimer) {
            clearTimeout(tomlStatusTimer);
            tomlStatusTimer = null;
        }
        el.className = cls;
        el.textContent = text;
        if (cls === 'ok' && !options.persist) {
            tomlStatusTimer = setTimeout(() => {
                el.textContent = '';
                tomlStatusTimer = null;
            }, 3000);
        }
    }

    async function applyTomlToConfig(options = {}) {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再加载选中配置');
            updateTomlActionState(file);
            return;
        }
        if (!meta?.trainable) {
            setTomlStatus('error', '该文件不是完整训练配置，不能加载选中配置');
            return;
        }

        currentTrainingSource = {
            method: meta.method,
            methods_subdir: meta.methods_subdir || 'gui-methods',
            file: meta.path,
        };

        if (meta.methods_subdir === 'methods' && meta.method === 'spd') {
            const methodSelect = document.getElementById('method-select');
            if ([...methodSelect.options].some((opt) => opt.value === 'spd')) {
                methodSelect.value = 'spd';
            }
            const variantSelect = document.getElementById('variant-select');
            const variants = await api('/api/methods/spd/variants');
            populateSelect('variant-select', variants, 'spd');
        } else if (meta.methods_subdir === 'gui-methods') {
            const methodFamily = VARIANT_METHOD_FAMILY[meta.method] || meta.method || 'lora';
            const methodSelect = document.getElementById('method-select');
            if ([...methodSelect.options].some((opt) => opt.value === methodFamily)) {
                methodSelect.value = methodFamily;
            }
            const variantSelect = document.getElementById('variant-select');
            if (![...variantSelect.options].some((opt) => opt.value === meta.method)) {
                const variants = await api(`/api/methods/${encodeURIComponent(methodFamily)}/variants`);
                populateSelect('variant-select', variants, meta.method);
            }
            if ([...variantSelect.options].some((opt) => opt.value === meta.method)) {
                variantSelect.value = meta.method;
            }
        }

        await loadConfig();
        renderTomlFileGroups(reorderTomlFileGroups(tomlFileGroups));
        updateTomlDirtyState();
        rememberSelectionSnapshot();
        if (!options.silent) {
            setTomlStatus('ok', `已应用 ${meta.path} 到表单`);
        }
    }

    async function toggleTomlUserLock() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (!file || !meta) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再调整锁定');
            updateTomlActionState(file);
            return;
        }
        if (meta.system_locked) {
            setTomlStatus('error', '系统预设已内置锁定，不能手动解除');
            return;
        }
        if (meta.group_locked) {
            setTomlStatus('error', '该文件属于只读分组，不能手动解除');
            return;
        }
        if (meta.user_group_locked) {
            setTomlStatus('error', '该文件所在分组已锁定，请在分组标题解除锁定');
            return;
        }

        const nextLocked = !meta.user_locked;
        const message = nextLocked
            ? `锁定 ${file}？锁定后不能直接保存，仍可使用新名称另存新配置。`
            : `解除 ${file} 的用户锁定？解除后可以直接编辑保存。`;
        if (!(await showAppConfirmDialog({
            title: nextLocked ? '锁定配置文件' : '解除配置锁定',
            description: file,
            message,
            confirmText: nextLocked ? '锁定' : '解除锁定',
            danger: nextLocked,
        }))) return;

        try {
            const res = await api('/api/config/lock', {
                method: 'POST',
                body: JSON.stringify({ file, locked: nextLocked }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '锁定操作失败');
                if (res.meta) tomlFileMeta[file] = res.meta;
                updateTomlDirtyState();
                return;
            }
            if (res.meta) tomlFileMeta[file] = res.meta;
            await loadTomlFileList(file);
            applyTomlLockState(file);
            updateTomlDirtyState();
            setTomlStatus('ok', res.message || (nextLocked ? '已锁定当前文件' : '已解除用户锁定'));
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function toggleTomlGroupLock(groupOrId) {
        const group = typeof groupOrId === 'string'
            ? tomlFileGroups.find((item) => item.id === groupOrId)
            : groupOrId;
        if (!group) {
            setTomlStatus('error', '分组不存在，请先刷新文件列表');
            return;
        }
        if (!group.lockable) {
            setTomlStatus('error', '该分组不能手动锁定或解锁');
            return;
        }

        const nextLocked = !group.user_group_locked;
        const sourceGroupIds = group.sourceGroupIds?.length ? group.sourceGroupIds : [group.id];
        const message = nextLocked
            ? `锁定分组“${group.label || group.id}”？该分组内文件将不能直接保存，仍可使用新名称另存新配置。`
            : `解除分组“${group.label || group.id}”的锁定？解除后该分组内文件可恢复编辑保存。`;
        if (!(await showAppConfirmDialog({
            title: nextLocked ? '锁定配置分组' : '解除分组锁定',
            description: group.label || group.id,
            message,
            confirmText: nextLocked ? '锁定分组' : '解除锁定',
            danger: nextLocked,
        }))) return;

        try {
            let lastResponse = null;
            for (const groupId of sourceGroupIds) {
                const res = await api('/api/config/group-lock', {
                    method: 'POST',
                    body: JSON.stringify({ group: groupId, locked: nextLocked }),
                });
                if (!res.ok) {
                    setTomlStatus('error', res.error || '分组锁定操作失败');
                    return;
                }
                lastResponse = res;
            }
            await loadTomlFileList(currentTomlFile || '');
            applyTomlLockState(currentTomlFile);
            updateTomlDirtyState();
            setTomlStatus('ok', lastResponse?.message || (nextLocked ? '已锁定当前分组' : '已解除分组锁定'));
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function createTomlGroup() {
        const label = await showHistoryTaskInputDialog({
            title: '新建配置分组',
            description: '用于整理右侧 TOML 配置文件。新分组默认可训练，可移入 imported 配置。',
            label: '分组名称',
            placeholder: '例如：角色配置 / 试验配置 / 正式配置',
            confirmText: '创建分组',
        });
        if (label === null) return;
        if (!label.trim()) {
            setTomlStatus('error', '分组名称不能为空');
            return;
        }
        try {
            const res = await api('/api/config/file-groups', {
                method: 'POST',
                body: JSON.stringify({ label: label.trim(), kind: 'training' }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '创建分组失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已创建');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function renameTomlGroup(group) {
        const label = await showHistoryTaskInputDialog({
            title: '重命名配置分组',
            description: '只修改分组显示名称，不会改动配置文件路径。',
            label: '分组名称',
            value: group.label || group.id,
            placeholder: '例如：正式配置',
            confirmText: '保存名称',
        });
        if (label === null) return;
        if (!label.trim()) {
            setTomlStatus('error', '分组名称不能为空');
            return;
        }
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'PATCH',
                body: JSON.stringify({ label: label.trim() }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '重命名分组失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已重命名');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function moveCurrentTomlToGroup() {
        const file = currentTomlFile || val('toml-file-select');
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再移动分组');
            updateTomlActionState(file);
            return;
        }
        const meta = tomlFileMeta[file];
        if (meta?.locked) {
            setTomlStatus('error', `${tomlLockLabel(meta) || '只读'}配置不能移动分组`);
            return;
        }

        const groups = getMovableTomlGroups(meta?.group);
        if (!groups.length) {
            setTomlStatus('error', '当前没有其他可移入的分组，请先新建分组或解除目标分组锁定');
            return;
        }
        const targetGroupId = await showMoveTomlDialog(file, meta, groups);
        if (!targetGroupId) return;
        try {
            const res = await api('/api/config/file-groups/move-file', {
                method: 'POST',
                body: JSON.stringify({ file, group: targetGroupId }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '移动分组失败');
                return;
            }
            await loadTomlFileList(file);
            setTomlStatus('ok', res.message || '配置已移动到分组');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    function getMovableTomlGroups(currentGroupId = '') {
        return reorderTomlFileGroups(tomlFileGroups)
            .filter((group) => isTrainingTomlGroup(group) && group.movable && !group.locked && !group.user_group_locked && group.id !== currentGroupId);
    }

    function deleteTomlGroupButtonTitle(group) {
        if (!group) return '配置分组不可用';
        if (group.user_group_locked) return '该分组已锁定，请先解除分组锁定后再删除';
        if (!group.deletable) return '系统固定分组或只读分组不能删除';
        const count = (group.files || []).length;
        return count > 0
            ? `删除当前分组“${group.label || group.id}”；不会删除其中 ${count} 个 TOML 文件`
            : `删除当前空分组“${group.label || group.id}”`;
    }

    function canDeleteTomlGroup(group) {
        return Boolean(group?.deletable && !group.user_group_locked);
    }

    function showMoveTomlDialog(file, meta, groups) {
        const wrap = document.createElement('div');
        wrap.className = 'toml-move-dialog-body';

        const current = document.createElement('p');
        current.className = 'toml-move-current';
        current.textContent = `当前配置: ${file}`;
        wrap.appendChild(current);

        const list = document.createElement('div');
        list.className = 'toml-move-option-list';
        const radios = [];
        for (const group of groups) {
            const label = document.createElement('label');
            label.className = 'toml-move-option';

            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'toml-move-target-group';
            input.value = group.id;
            input.checked = group.id !== meta?.group && !radios.some((item) => item.checked);
            radios.push(input);

            const text = document.createElement('span');
            const title = document.createElement('strong');
            title.textContent = group.label || group.id;
            const detail = document.createElement('small');
            const count = (group.files || []).length;
            detail.textContent = `${count} 个配置`;
            text.append(title, detail);

            label.append(input, text);
            list.appendChild(label);
        }
        wrap.appendChild(list);

        return showHistoryTaskDialog({
            title: '移动配置',
            description: '选择目标分组后确认，配置文件路径不会改变，只调整右侧分组归属。',
            body: wrap,
            confirmText: '移动到分组',
            onOpen: () => {
                const checked = radios.find((item) => item.checked) || radios[0];
                checked?.focus();
            },
            getValue: () => {
                const checked = wrap.querySelector('input[name="toml-move-target-group"]:checked');
                return checked?.value || '';
            },
        });
    }

    async function deleteTomlGroup(group) {
        if (!canDeleteTomlGroup(group)) {
            setTomlStatus('error', deleteTomlGroupButtonTitle(group));
            return;
        }
        const count = (group.files || []).length;
        const ok = await showHistoryTaskConfirmDialog({
            title: '删除配置分组',
            description: group.label || group.id,
            message: count > 0
                ? `只删除这个分组，不删除其中 ${count} 个 TOML 文件；这些文件会回到导入配置或数据集配置等默认分组。`
                : '只删除这个分组，不会删除任何 TOML 文件。',
            confirmText: '删除分组',
            danger: true,
        });
        if (!ok) return;
        const reallyOk = await showHistoryTaskConfirmDialog({
            title: '你真的确认吗？',
            description: group.label || group.id,
            message: count > 0
                ? `确认后会删除这个分组，分组内 ${count} 个 TOML 文件会回到默认分组。`
                : '确认后会删除这个空分组。',
            confirmText: '我确认',
            cancelText: '我觉得不对',
            cancelPrimary: true,
            danger: true,
        });
        if (!reallyOk) return;
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '删除分组失败');
                return;
            }
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已删除');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    async function deleteTomlFile() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再删除');
            updateTomlActionState(file);
            return;
        }
        if (!meta) {
            await handleDeletedTomlSelection(file, '当前配置已不在列表中，已刷新配置列表');
            return;
        }
        if (meta.locked) {
            setTomlStatus('error', `${tomlLockLabel(meta) || '只读'}配置不能删除`);
            updateTomlActionState(file);
            return;
        }

        if (tomlDeleteConfirmFile !== file) {
            armTomlDeleteConfirm(file);
            return;
        }
        resetTomlDeleteConfirm({ update: false });

        try {
            const res = await api(`/api/config/raw?file=${encodeURIComponent(file)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                if (isMissingTomlFileResponse(res)) {
                    await handleDeletedTomlSelection(file, res.error || '配置文件不存在或已被删除');
                    return;
                }
                setTomlStatus('error', res.error || '删除失败');
                return;
            }

            await handleDeletedTomlSelection(file, `已删除配置: ${file}`, { ok: true });
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    function isMissingTomlFileResponse(res) {
        return String(res?.error || '').includes('不存在') || String(res?.error || '').includes('已被删除');
    }

    async function handleDeletedTomlSelection(file, message, options = {}) {
        if (currentTrainingSource.file === file) {
            setCurrentTrainingSourceFromVariant(val('variant-select') || 'lora');
        }
        delete tomlFileMeta[file];
        tomlFiles = tomlFiles.filter((item) => item !== file);
        clearCurrentTomlSelection();
        await loadTomlFileList('', { skipDefaultLoad: true });
        clearCurrentTomlSelection();
        updateTomlDirtyState();
        setTomlStatus(options.ok ? 'ok' : 'error', message, { persist: true });
    }

    function clearCurrentTomlSelection() {
        resetTomlDeleteConfirm({ update: false });
        resetTomlSaveConfirm({ update: false });
        currentTomlFile = '';
        tomlSavedContent = '';
        const editor = document.getElementById('toml-editor');
        if (editor) {
            editor.value = '';
            editor.readOnly = false;
            editor.title = '';
        }
        const select = document.getElementById('toml-file-select');
        if (select) select.value = '';
        updateTomlSelectionUI('');
        applyTomlLockState('');
    }

    async function restoreSystemTomlPresets() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再还原系统预设');
            updateTomlActionState(file);
            return;
        }

        const currentHint = meta?.restorable ? `\n当前文件 ${file} 也会一起还原。` : '';
        const ok = await showAppConfirmDialog({
            title: '还原系统预设',
            description: 'base、presets、methods、gui-methods',
            message: `还原会覆盖系统预设文件，但会先自动备份当前内容。用户导入/副本和数据集配置不会被还原。${currentHint}`,
            confirmText: '还原系统预设',
            danger: true,
        });
        if (!ok) return;

        try {
            const res = await api('/api/config/restore-system', {
                method: 'POST',
                body: JSON.stringify({}),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '还原失败');
                return;
            }

            const preferredFile = file && tomlFiles.includes(file) ? file : '';
            await loadTomlFileList(preferredFile);
            const restoredCount = res.restored?.length || 0;
            const skippedCount = res.skipped?.length || 0;
            const backupText = res.backup_dir ? `，备份在 ${res.backup_dir}` : '';
            setTomlStatus('ok', `已还原 ${restoredCount} 个系统预设，跳过 ${skippedCount} 个${backupText}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    // ── 训练控制 ──
    async function startTraining() {
        const selectedTrainingConfigFile = currentTrainingConfigFile();
        if (tomlManagerMode !== 'output' || !outputRunState.file) {
            if (hasPendingConfigChanges(currentTomlFile)) {
                setTomlStatus('error', '当前配置有未保存修改，请先保存更新当前选中配置或另存新配置，再开始训练');
                updateTomlActionState(currentTomlFile);
                document.querySelector('[data-tab="config"]')?.click();
                return;
            }
        }
        if (!selectedTrainingConfigFile) {
            const message = tomlManagerMode === 'output' && outputRunState.selectedRun
                ? '这个训练输出没有可直接继续训练的 config.runtime.toml，请先另存原始配置或选择其他运行目录'
                : '请选择要训练的配置文件';
            setTomlStatus('error', message);
            return;
        }
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant) return alert('请选择变体');
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            const message = 'SPD 是 CLI 实验配置，只能通过 tasks.py exp-spd / scripts/distill_spd.py 运行；Web 普通训练入口已拦截，避免误用 train.py。';
            setTomlStatus('error', message, { persist: true });
            alert(message);
            return;
        }
        if (continueTrainingSource && !(await refreshContinueTrainingSourceCompatibility())) {
            setTomlStatus('error', continueTrainingSource.message || '继续训练权重与当前配置不兼容', { persist: true });
            return;
        }
        const preflight = await runPreflight(variant, preset, methodsSubdir);
        if (!preflight) {
            if (isPreflightDialogOpen()) await waitForPreflightDialogClose();
            return;
        }
        const willAutoPreprocess = !currentTrainingConfigIsRuntime();
        if (!preflight.ok) {
            const action = await showPreflightDialog(preflight, false, { willAutoPreprocess });
            if (action === 'preprocess') {
                await startPreprocessFromPreflight(preflight);
            }
            return;
        }
        const action = await showPreflightDialog(preflight, true, { willAutoPreprocess });
        if (action === 'preprocess') {
            await startPreprocessFromPreflight(preflight);
            return;
        }
        if (action !== 'continue') return;
        await startTrainingUnchecked(variant, preset, methodsSubdir, { willAutoPreprocess });
    }

    async function queueCurrentTrainingFromConfig() {
        return ensureQueueFeature().queueCurrentTrainingFromConfig();
    }

    async function runPreflight(variant, preset, methodsSubdir) {
        const pending = showPreflightPendingDialog({
            title: '训练前预检测',
            message: '正在检查模型路径、数据集路径和预处理启动环境...',
            detail: '这一步可能需要几秒钟；窗口保持打开表示仍在检查。',
        });
        try {
            const res = await api('/api/training/preflight', {
                method: 'POST',
                signal: pending.signal,
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                }),
            });
            pending.resolve();
            return res;
        } catch (e) {
            pending.resolve();
            if (e.name === 'AbortError') {
                return null;
            }
            showPreflightRequestError(`预检测请求失败: ${e.message}`);
            return null;
        }
    }

    function isCliOnlySpdSource(variant, methodsSubdir) {
        return String(methodsSubdir || '') === 'methods' && String(variant || '') === 'spd';
    }

    function currentTrainingConfigIsRuntime() {
        return currentTrainingConfigFile().replace(/\\/g, '/').endsWith('/config.runtime.toml');
    }

    async function chooseTrainingLaunchMode(options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const isRunning = trainingRuntime.state === 'running' || trainingRuntime.state === 'compiling';
        const sourceDetail = continueTrainingSource
            ? `\n\n训练来源: 继续训练 ${continueTrainingSource.kind} · ${continueTrainingSource.name}\n基于权重: ${continueTrainingSource.abs_path}`
            : '\n\n训练来源: 从零开始';
        if (isRunning) {
            const ok = await showAppConfirmDialog({
                title: '加入训练队列',
                description: '当前已有任务在运行',
                message: `确认后会冻结当前配置，并加入队列等待自动执行。${sourceDetail}`,
                confirmText: '加入队列',
                cancelText: '取消',
            });
            return ok ? 'queue' : 'cancel';
        }
        const startNow = await showAppConfirmDialog({
            title: willAutoPreprocess ? '最终确认：预处理并训练' : '最终确认：开始训练',
            description: '可以立即启动，也可以先加入队列',
            message: willAutoPreprocess
                ? `确认后会立即创建本次运行目录并启动预处理。${sourceDetail}`
                : `确认后会立即创建本次运行目录并启动训练进程。${sourceDetail}`,
            confirmText: willAutoPreprocess ? '立即预处理并训练' : '立即开始训练',
            cancelText: '不立即启动',
        });
        if (startNow) return 'start';
        const queue = await showAppConfirmDialog({
            title: '加入训练队列',
            description: '冻结当前配置并等待手动继续',
            message: `确认后会创建独立运行配置并加入队列；队列会保持暂停，等待你手动继续。${sourceDetail}`,
            confirmText: '加入队列',
            cancelText: '取消',
        });
        return queue ? 'queue' : 'cancel';
    }

    async function confirmTrainingLaunch(options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const sourceDetail = continueTrainingSource
            ? `\n\n训练来源: 继续训练 ${continueTrainingSource.kind} · ${continueTrainingSource.name}\n基于权重: ${continueTrainingSource.abs_path}`
            : '\n\n训练来源: 从零开始';
        return showAppConfirmDialog({
            title: willAutoPreprocess ? '最终确认：预处理并训练' : '最终确认：开始训练',
            description: '训练启动前的最后一步',
            message: willAutoPreprocess
                ? `确认后会立即创建本次运行目录并启动预处理；预处理完成后会自动开始训练。${sourceDetail}`
                : `确认后会立即创建本次运行目录并启动训练进程。${sourceDetail}`,
            confirmText: willAutoPreprocess ? '确认预处理并训练' : '确认开始训练',
            cancelText: '返回检查',
        });
    }

    async function startTrainingUnchecked(variant, preset, methodsSubdir, options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const mode = await chooseTrainingLaunchMode({ willAutoPreprocess });
        if (mode === 'cancel') return;
        if (mode === 'queue') {
            await enqueueTrainingFromConfig(variant, preset, methodsSubdir, { willAutoPreprocess });
            return;
        }
        renderPreflightPending({
            title: willAutoPreprocess ? '启动预处理后训练' : '启动训练',
            message: willAutoPreprocess
                ? '正在创建运行目录并启动预处理...'
                : '正在创建运行目录并启动训练...',
            detail: willAutoPreprocess
                ? '预处理完成后会自动开始训练；成功后会自动切换到训练页。'
                : '后端正在准备训练进程；启动成功后会自动切换到训练页。',
        });
        try {
            const res = await api('/api/training/start', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                    extra_args: [],
                    gpu_whitelist: selectedGpuPayload(),
                    confirmed: true,
                    confirm_preprocess: willAutoPreprocess,
                    ...continueTrainingRequestPayload(),
                }),
            });
            if (res.ok) {
                const dialog = document.getElementById('preflight-dialog');
                if (dialog?.open) dialog.close('training-started');
                enterLiveTrainingForNewRun();
                appendLog(`[状态] ${res.message || '任务已启动'}`);
            } else {
                if (res.preflight) {
                    const action = await showPreflightDialog(res.preflight, false);
                    if (action === 'preprocess') {
                        await startPreprocessFromPreflight(res.preflight);
                    }
                } else {
                    showPreflightRequestError(res.error || '启动失败');
                }
            }
        } catch (e) {
            showPreflightRequestError('请求失败: ' + e.message);
        }
    }

    async function enqueueTrainingFromConfig(variant, preset, methodsSubdir, options = {}) {
        return ensureQueueFeature().enqueueTrainingFromConfig(variant, preset, methodsSubdir, options);
    }

    async function enqueueTrainingQueueRequest(options = {}) {
        return ensureQueueFeature().enqueueTrainingQueueRequest(options);
    }

    function enterLiveTrainingForNewRun() {
        returnToLiveTraining({ refresh: false });
        document.querySelector('[data-tab="training"]')?.click();
        pollStatus();
        replayTrainingLogs();
    }

    function showPreflightDialog(result, allowContinue, options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog) {
            if (!allowContinue) return Promise.resolve('cancel');
            const confirmText = options.willAutoPreprocess ? '确认预处理并训练' : '确认开始训练';
            return showAppConfirmDialog({
                title: '训练前预检测',
                description: '检测到训练前提示',
                message: `${preflightPlainText(result)}\n\n是否继续训练？`,
                confirmText,
            }).then((ok) => ok ? 'continue' : 'cancel');
        }
        renderPreflightResult(result, allowContinue, options);
        if (!dialog.open) dialog.showModal();
        return new Promise((resolve) => {
            dialog.addEventListener('close', () => {
                resolve(dialog.returnValue || 'cancel');
            }, { once: true });
        });
    }

    function showPreflightPendingDialog(options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const controller = new AbortController();
        if (!dialog) {
            return { signal: controller.signal, resolve: () => {} };
        }
        renderPreflightPending(options);
        let settled = false;
        const cleanup = () => {
            dialog.removeEventListener('close', handleClose);
        };
        const handleClose = () => {
            cleanup();
            if (!settled) {
                controller.abort();
            }
        };
        dialog.addEventListener('close', handleClose);
        if (!dialog.open) {
            try {
                dialog.showModal();
            } catch (e) {
                dialog.setAttribute('open', 'open');
            }
        }
        return {
            signal: controller.signal,
            resolve: () => {
                settled = true;
                cleanup();
            },
        };
    }

    function renderPreflightPending(options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const heading = dialog?.querySelector('.preflight-header h2');
        const summary = document.getElementById('preflight-summary');
        const list = document.getElementById('preflight-results');
        const continueBtn = document.getElementById('btn-preflight-continue');
        const preprocessBtn = document.getElementById('btn-preflight-preprocess');
        const cancelBtn = document.getElementById('btn-preflight-cancel');
        if (heading) heading.textContent = options.title || '训练前预检测';
        if (summary) {
            summary.className = 'preflight-summary pending';
            summary.setAttribute('aria-live', 'polite');
            summary.textContent = options.message || '正在预检测...';
        }
        if (list) {
            list.innerHTML = '';
            const row = document.createElement('div');
            row.className = 'preflight-item pending';
            row.setAttribute('aria-busy', 'true');

            const badge = document.createElement('span');
            badge.className = 'preflight-badge preflight-spinner';
            badge.setAttribute('aria-label', '正在检查');
            row.appendChild(badge);

            const body = document.createElement('div');
            body.className = 'preflight-body';
            const title = document.createElement('div');
            title.className = 'preflight-message';
            title.textContent = options.detail || '正在连接后端并执行轻量检查...';
            const path = document.createElement('div');
            path.className = 'preflight-path';
            path.textContent = '请稍等，预检测返回后会在这里显示每一项结果。';
            body.append(title, path);
            row.appendChild(body);
            list.appendChild(row);
        }
        if (preprocessBtn) {
            preprocessBtn.hidden = true;
            preprocessBtn.disabled = true;
        }
        if (continueBtn) {
            continueBtn.hidden = false;
            continueBtn.disabled = true;
            continueBtn.textContent = '正在检查...';
        }
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.textContent = '取消';
        }
    }

    function showPreflightRequestError(message) {
        const result = {
            ok: false,
            summary: { errors: 1, warnings: 0, checks: 1 },
            checks: [{
                level: 'error',
                key: 'preflight',
                message,
            }],
            errors: [{
                level: 'error',
                key: 'preflight',
                message,
            }],
            warnings: [],
        };
        const dialog = document.getElementById('preflight-dialog');
        if (dialog) {
            renderPreflightResult(result, false);
            if (!dialog.open) dialog.showModal();
        } else {
            alert(message);
        }
    }

    function isPreflightDialogOpen() {
        const dialog = document.getElementById('preflight-dialog');
        return Boolean(dialog?.open);
    }

    function waitForPreflightDialogClose() {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog?.open) return Promise.resolve();
        return new Promise((resolve) => {
            dialog.addEventListener('close', resolve, { once: true });
        });
    }

    function renderPreflightResult(result, allowContinue, options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const heading = dialog?.querySelector('.preflight-header h2');
        const summary = document.getElementById('preflight-summary');
        const list = document.getElementById('preflight-results');
        const continueBtn = document.getElementById('btn-preflight-continue');
        const preprocessBtn = document.getElementById('btn-preflight-preprocess');
        const cancelBtn = document.getElementById('btn-preflight-cancel');
        const errors = result.summary?.errors || 0;
        const warnings = result.summary?.warnings || 0;
        const checks = result.summary?.checks || 0;
        const canPreprocess = preflightCanStartPreprocess(result);
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);

        if (heading) heading.textContent = '训练前预检测';
        summary.className = `preflight-summary ${errors ? 'error' : warnings ? 'warning' : 'ok'}`;
        summary.removeAttribute('aria-live');
        if (errors && canPreprocess) {
            summary.textContent = `发现 ${errors} 个错误：当前数据需要先预处理。点击下方按钮后，还会出现最终确认；确认后才会启动预处理并在完成后训练。`;
        } else {
            summary.textContent = errors
                ? `发现 ${errors} 个错误，已阻止训练。`
                : warnings
                    ? (willAutoPreprocess
                        ? `通过基础检查，但有 ${warnings} 个警告。点击下方按钮后，还需要最终确认才会预处理并训练。`
                        : `通过基础检查，但有 ${warnings} 个警告。点击下方按钮后，还需要最终确认才会开始训练。`)
                    : willAutoPreprocess
                        ? `预检测通过，共 ${checks} 项。点击下方按钮后，还需要最终确认才会创建运行目录、预处理并自动训练。`
                        : `预检测通过，共 ${checks} 项。点击下方按钮后，还需要最终确认才会开始训练。`;
        }

        list.innerHTML = '';
        for (const item of result.checks || []) {
            const row = document.createElement('div');
            row.className = `preflight-item ${item.level}`;

            const badge = document.createElement('span');
            badge.className = 'preflight-badge';
            badge.textContent = item.level === 'ok' ? '通过' :
                item.level === 'warning' ? '警告' : '错误';
            row.appendChild(badge);

            const body = document.createElement('div');
            body.className = 'preflight-body';
            const title = document.createElement('div');
            title.className = 'preflight-message';
            title.textContent = `${FIELD_LABEL_ZH[item.key] || item.key}: ${item.message}`;
            body.appendChild(title);
            if (item.path) {
                const path = document.createElement('div');
                path.className = 'preflight-path';
                path.textContent = item.path;
                body.appendChild(path);
            }
            row.appendChild(body);
            list.appendChild(row);
        }

        preprocessBtn.hidden = !canPreprocess;
        preprocessBtn.disabled = !canPreprocess;
        continueBtn.hidden = !allowContinue;
        continueBtn.disabled = !allowContinue;
        continueBtn.textContent = warnings
            ? (willAutoPreprocess ? '查看最终确认' : '查看最终确认')
            : (willAutoPreprocess ? '下一步：最终确认' : '下一步：最终确认');
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.textContent = '取消';
        }
    }

    function preflightCanStartPreprocess(result) {
        const checks = result.checks || [];
        const errors = result.errors || [];
        const allowedErrorKeys = new Set(['training_images', 'resized_image_dir']);
        if (errors.some((item) => !allowedErrorKeys.has(item.key))) return false;
        const sourceOk = checks.some((item) => item.key === 'source_image_dir' && item.level === 'ok');
        if (!sourceOk) return false;
        return checks.some((item) =>
            ['training_images', 'resized_image_dir', 'lora_cache_dir', 'latent_cache', 'text_cache'].includes(item.key)
            && ['error', 'warning'].includes(item.level)
        );
    }

    async function startPreprocessFromPreflight(result) {
        const variant = result.variant || currentTrainingSource.method || val('variant-select');
        const preset = result.preset || val('preset-select');
        const methodsSubdir = result.methods_subdir || currentTrainingSource.methods_subdir || 'gui-methods';
        if (continueTrainingSource && !(await refreshContinueTrainingSourceCompatibility())) {
            showPreflightRequestError(continueTrainingSource.message || '继续训练权重与当前配置不兼容');
            return;
        }
        const mode = await chooseTrainingLaunchMode({ willAutoPreprocess: true });
        if (mode === 'cancel') return;
        if (mode === 'queue') {
            await enqueueTrainingFromConfig(variant, preset, methodsSubdir, { willAutoPreprocess: true });
            return;
        }
        renderPreflightPending({
            title: '启动预处理',
            message: '正在创建运行目录并启动预处理...',
            detail: '正在把任务交给后端；成功后会自动切换到训练页。',
        });
        try {
            const res = await api('/api/training/preprocess', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                    extra_args: [],
                    train_after: true,
                    confirmed: true,
                    confirm_train_after: true,
                    confirm_preprocess: true,
                    gpu_whitelist: selectedGpuPayload(),
                    ...continueTrainingRequestPayload(),
                }),
            });
            if (!res.ok) {
                showPreflightRequestError(res.error || '预处理启动失败');
                return;
            }
            const dialog = document.getElementById('preflight-dialog');
            if (dialog?.open) dialog.close('preprocess-started');
            enterLiveTrainingForNewRun();
            appendLog(`[状态] ${res.message || '预处理已启动'}`);
        } catch (e) {
            showPreflightRequestError('预处理请求失败: ' + e.message);
        }
    }

    function currentTrainingConfigFile() {
        if (tomlManagerMode === 'output') {
            return outputRunRuntimeFile();
        }
        return currentTrainingSource.file || currentTomlFile || val('toml-file-select') || '';
    }

    function preflightPlainText(result) {
        return (result.checks || [])
            .map((item) => `[${item.level}] ${item.key}: ${item.message}${item.path ? ` (${item.path})` : ''}`)
            .join('\n');
    }

    async function stopTraining() {
        const ok = await showAppConfirmDialog({
            title: '停止训练',
            description: '当前运行中的训练任务',
            message: '确定要停止训练吗？停止后当前训练过程会立即中断。',
            confirmText: '停止训练',
            danger: true,
        });
        if (!ok) return;
        await api('/api/training/stop', { method: 'POST' });
    }

    // ── WebSocket ──
    function connectWebSocket() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        setLogStatus('连接中', 'warning');
        ws = new WebSocket(`${proto}//${location.host}/ws/training`);
        ws.onopen = () => {
            setLogStatus('已连接', 'ok');
            replayTrainingLogs();
        };
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            handleWsMessage(msg);
        };
        ws.onclose = () => {
            setLogStatus('已断开，准备重连', 'warning');
            setTimeout(connectWebSocket, 3000);
        };
        ws.onerror = () => {
            setLogStatus('连接异常', 'error');
            ws.close();
        };
    }

    function handleWsMessage(msg) {
        switch (msg.type) {
            case 'log':
                if (isHistoryReviewMode()) break;
                markTrainingActivity(msg.ts);
                appendLogRecord(msg);
                break;
            case 'progress':
                if (isHistoryReviewMode()) break;
                updateProgress(msg);
                break;
            case 'metrics':
                if (isHistoryReviewMode()) break;
                updateMetrics(msg);
                break;
            case 'status':
                if (isHistoryReviewMode()) {
                    loadTrainingHistoryList();
                    renderResumePanelState();
                    break;
                }
                updateStatus(msg);
                loadTrainingQueue();
                loadTrainingHistoryList();
                break;
            case 'queue':
                updateTrainingQueueFromPayload(msg);
                loadTrainingHistoryList();
                break;
            case 'system':
                if (isHistoryReviewMode()) break;
                updateSystem(msg);
                break;
        }
    }

    function appendLog(line) {
        appendLogRecord({ line });
    }

    function appendLogRecord(record) {
        if (record?.id && record.id <= trainingRuntime.lastLogId) return;
        if (record?.id) trainingRuntime.lastLogId = record.id;

        const line = record?.line ?? '';
        const prefix = record?.kind === 'progress' ? '[进度] ' : '';
        trainingRuntime.logBuffer.push(prefix + line);
        trainingRuntime.logLineCount += 1;
        scheduleLogFlush();
    }

    function renderLogOutputLines(lines) {
        const el = document.getElementById('log-output');
        if (!el) return;
        const normalized = (lines || [])
            .map((line) => String(line || ''))
            .filter((line) => line.length);
        const fragment = document.createDocumentFragment();
        for (const line of normalized) {
            const span = document.createElement('span');
            span.className = `log-line ${logLineTone(line)}`;
            span.textContent = line;
            fragment.append(span, document.createTextNode('\n'));
        }
        el.replaceChildren(fragment);
    }

    function currentLogOutputLines() {
        const el = document.getElementById('log-output');
        if (!el) return [];
        return el.textContent.split('\n').filter(Boolean);
    }

    function logLineTone(line) {
        const text = String(line || '').toLowerCase();
        if (text.includes('traceback') || text.includes('exception') || text.includes('error') || text.includes('错误') || text.includes('异常') || text.includes('失败')) {
            return 'error';
        }
        if (text.includes('warn') || text.includes('warning') || text.includes('警告') || text.includes('跳过')) {
            return 'warning';
        }
        if (text.startsWith('[进度]') || text.includes('progress')) {
            return 'progress';
        }
        if (text.startsWith('[状态]') || text.startsWith('[提示]')) {
            return 'status';
        }
        return 'info';
    }

    function scheduleLogFlush() {
        if (trainingRuntime.logFlushPending) return;
        trainingRuntime.logFlushPending = true;
        const schedule = window.requestAnimationFrame
            ? (fn) => window.requestAnimationFrame(fn)
            : (fn) => window.setTimeout(fn, 16);
        schedule(flushLogBuffer);
    }

    function flushLogBuffer() {
        trainingRuntime.logFlushPending = false;
        if (!trainingRuntime.logBuffer.length) return;
        const el = document.getElementById('log-output');
        const nextLines = [...currentLogOutputLines(), ...trainingRuntime.logBuffer];
        trainingRuntime.logBuffer = [];
        const lines = nextLines.filter(Boolean).slice(-MAX_LOG_LINES);
        renderLogOutputLines(lines);
        trainingRuntime.logLineCount = lines.length;
        el.scrollTop = el.scrollHeight;
        updateLogStatusText();
    }

    async function replayTrainingLogs() {
        if (isHistoryReviewMode()) return;
        try {
            const payload = await api(`/api/training/logs?after=${trainingRuntime.lastLogId}&limit=1000`);
            for (const record of payload.records || []) {
                if (record.ts) markTrainingActivity(record.ts);
                appendLogRecord(record);
                replayMetricsFromLogRecord(record);
            }
            await replayMetricsHistory();
            updateLogStatusText();
        } catch (e) {
            setLogStatus('日志回放失败', 'error');
        }
    }

    async function replayMetricsHistory() {
        if (isHistoryReviewMode()) return;
        try {
            const records = await api('/api/training/metrics');
            for (const record of records || []) {
                updateMetrics(record);
            }
        } catch (e) {
            // 历史指标不是训练控制关键路径，失败时保留日志回放。
        }
    }

    function replayMetricsFromLogRecord(record) {
        const line = record?.line || '';
        const parsed = parseMetricsFromProgressLine(line);
        if (!parsed || parsed.loss === undefined) return;
        updateMetrics({ ...parsed, ts: record.ts });
    }

    function setLogStatus(text, state = '') {
        const el = document.getElementById('log-status');
        if (!el) return;
        el.textContent = text;
        el.className = `log-status ${state}`.trim();
    }

    function updateLogStatusText() {
        const state = ws?.readyState === WebSocket.OPEN ? 'ok' : 'warning';
        const text = ws?.readyState === WebSocket.OPEN
            ? `已连接 · ${trainingRuntime.logLineCount} 行`
            : `${trainingRuntime.logLineCount} 行`;
        setLogStatus(text, state);
    }

    function updateProgress(msg) {
        if (isHistoryReviewMode()) return;
        markTrainingActivity(msg.ts);
        const previousCurrent = Number(trainingRuntime.progressCurrent || 0);
        const previousUpdatedAt = Number(trainingRuntime.progressUpdatedAt || 0);
        const now = Date.now();
        const pct = msg.total > 0 ? (msg.current / msg.total * 100) : 0;
        trainingRuntime.progressCurrent = Number(msg.current) || 0;
        trainingRuntime.progressTotal = Number(msg.total) || 0;
        trainingRuntime.progressLabel = msg.label || '';
        trainingRuntime.progressRate = msg.rate || '';
        const rateSeconds = parseProgressRateSeconds(msg.rate);
        if (rateSeconds !== null) {
            trainingRuntime.progressSecondsPerStep = rateSeconds;
        } else if (previousUpdatedAt && trainingRuntime.progressCurrent > previousCurrent) {
            const elapsedSeconds = (now - previousUpdatedAt) / 1000;
            const stepDelta = trainingRuntime.progressCurrent - previousCurrent;
            const inferredSeconds = elapsedSeconds / stepDelta;
            if (Number.isFinite(inferredSeconds) && inferredSeconds > 0) {
                trainingRuntime.progressSecondsPerStep = inferredSeconds;
            }
        }
        trainingRuntime.progressUpdatedAt = now;
        document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
        let text = `${msg.label}: ${msg.current}/${msg.total} (${pct.toFixed(1)}%)`;
        if (msg.rate) text += ` — ${msg.rate}`;
        document.getElementById('progress-text').textContent = text;
        updateDashboardProgressIdleState(true);
        setMetricText('metric-step', msg.current);
        if (msg.rate) setMetricText('metric-rate', msg.rate);
        renderLiveTrainingDashboard();
        refreshQueueRunningProgressViews();
    }

    function updateMetrics(msg) {
        if (isHistoryReviewMode()) return;
        markTrainingActivity(msg.ts);
        const lrText = msg.lr !== undefined ? formatLr(msg.lr) : '';
        const lrNumber = msg.lr !== undefined ? Number(msg.lr) : null;
        if (msg.loss !== undefined) {
            const loss = Number(msg.loss);
            if (!Number.isFinite(loss)) return;
            setMetricText('metric-loss', loss.toFixed(5));
            const step = msg.step || ++stepCounter;
            const metadata = { rawStep: msg.step ?? step };
            if (Number.isFinite(lrNumber)) metadata.lr = lrNumber;
            lossChart?.push(step, loss, metadata);
            syncLossChartEmptyState();
        }
        if (msg.lr !== undefined) {
            setMetricText('metric-lr', lrText);
            if (msg.loss === undefined && Number.isFinite(lrNumber)) {
                lossChart?.updatePointMetadata?.(msg.step, { lr: lrNumber });
            }
        }
        if (msg.step !== undefined) {
            setMetricText('metric-step', msg.step);
        }
        if (msg.rate) {
            trainingRuntime.progressRate = msg.rate;
            const rateSeconds = parseProgressRateSeconds(msg.rate);
            if (rateSeconds !== null) trainingRuntime.progressSecondsPerStep = rateSeconds;
            setMetricText('metric-rate', msg.rate);
        }
        renderLiveChartPanel();
        renderLiveTrainingDashboard();
    }

    function updateStatus(msg) {
        if (isHistoryReviewMode()) return;
        const dot = document.querySelector('.dot');
        const text = document.getElementById('status-text');
        const stopBtn = document.getElementById('btn-stop-training');

        dot.className = 'dot ' + msg.state;
        const stateMap = { idle: '空闲', running: '训练中', error: '错误', compiling: '编译中' };
        const jobLabel = msg.job === 'preprocess' ? '预处理中' : (stateMap[msg.state] || msg.state);
        text.textContent = msg.state === 'running' ? jobLabel : (stateMap[msg.state] || msg.state);
        updateTrainingToolbarState(msg.state, text.textContent);
        trainingRuntime.state = msg.state;
        trainingRuntime.job = msg.job || trainingRuntime.job || '';
        trainingRuntime.variant = msg.variant || trainingRuntime.variant || '';
        trainingRuntime.preset = msg.preset || trainingRuntime.preset || '';
        trainingRuntime.methodsSubdir = msg.methods_subdir || trainingRuntime.methodsSubdir || '';
        if (msg.last_output_at) {
            markTrainingActivity(msg.last_output_at);
        }
        if (msg.state !== 'running' && msg.state !== 'compiling') {
            trainingRuntime.lastOutputAt = 0;
            trainingRuntime.lastUiActivityAt = 0;
            resetLiveSystemPeaks();
        }
        if (msg.output_dir !== undefined) {
            trainingRuntime.outputDir = msg.output_dir || '';
        }
        if (msg.sample_dir !== undefined) {
            trainingRuntime.sampleDir = msg.sample_dir || '';
            ensurePreviewFeature().updateRuntimeSampleState({ sampleDir: trainingRuntime.sampleDir });
        }
        if (msg.sample_config !== undefined) {
            trainingRuntime.sampleConfig = msg.sample_config || null;
            ensurePreviewFeature().updateRuntimeSampleState({ sampleConfig: trainingRuntime.sampleConfig });
        }
        applyRuntimeInfoToState(msg);

        stopBtn.disabled = msg.state !== 'running';
        stopBtn.classList.toggle('is-emergency', msg.state === 'running');

        if (msg.variant) document.getElementById('train-variant').textContent = msg.variant;
        if (msg.preset) document.getElementById('train-preset').textContent = msg.preset;

        if (msg.message) appendLog(`[状态] ${msg.message}`);

        if (msg.state === 'idle' || msg.state === 'error') {
            document.getElementById('progress-bar').style.width = '0%';
            trainingRuntime.progressCurrent = 0;
            trainingRuntime.progressTotal = 0;
            trainingRuntime.progressLabel = '';
            trainingRuntime.progressRate = '';
            trainingRuntime.progressSecondsPerStep = null;
            trainingRuntime.progressUpdatedAt = 0;
            document.getElementById('progress-text').textContent = '暂无正在运行的任务目录...';
            updateDashboardProgressIdleState(false);
            trainingRuntime.quietHintShown = false;
            trainingRuntime.job = '';
            refreshQueueRunningProgressViews();
            if (!msg.output_dir) {
                clearRuntimeInfo();
            }
        }
        renderCurrentRuntimePaths();
        renderLiveTrainingDashboard();
        refreshTrainingHealth();
    }

    function resetLiveSystemPeaks() {
        trainingRuntime.lastGpuUtil = null;
        trainingRuntime.lastGpuTemp = null;
        trainingRuntime.lastVramUsedGb = null;
        trainingRuntime.lastVramTotalGb = null;
        trainingRuntime.peakGpuUtil = null;
        trainingRuntime.peakGpuTemp = null;
        trainingRuntime.peakVramUsedGb = null;
        resetLiveMetricPlaceholders({ primary: false });
    }

    function clearRuntimeInfo() {
        trainingRuntime.runDir = '';
        trainingRuntime.runtimeConfigFile = '';
        trainingRuntime.originalConfigFile = '';
        trainingRuntime.datasetConfigFile = '';
        trainingRuntime.modelCacheDir = '';
        trainingRuntime.datasetCacheDir = '';
        trainingRuntime.trainingOutputDir = '';
        trainingRuntime.logsDir = '';
    }

    function applyRuntimeInfoToState(msg) {
        const fields = {
            run_dir: 'runDir',
            runtime_config_file: 'runtimeConfigFile',
            original_config_file: 'originalConfigFile',
            dataset_config_file: 'datasetConfigFile',
            model_cache_dir: 'modelCacheDir',
            dataset_cache_dir: 'datasetCacheDir',
            training_output_dir: 'trainingOutputDir',
            logs_dir: 'logsDir',
        };
        for (const [wireKey, stateKey] of Object.entries(fields)) {
            if (msg[wireKey] !== undefined) {
                trainingRuntime[stateKey] = msg[wireKey] || '';
            }
        }
    }

    function renderCurrentRuntimePaths() {
        if (isHistoryReviewMode()) return;
        const configPanel = document.getElementById('history-config-panel');
        const configTitle = document.getElementById('history-config-title');
        const configOutput = document.getElementById('history-config-output');
        const task = currentRuntimeTaskInfo();
        const hasRuntimePaths = runtimePathItems(task, { includeHistory: false }).length > 0;
        if (configPanel) configPanel.hidden = !hasRuntimePaths;
        if (!hasRuntimePaths) {
            const paths = document.getElementById('history-paths');
            if (paths) paths.innerHTML = '';
            if (configOutput) configOutput.textContent = '';
            return;
        }
        if (configTitle) {
            configTitle.textContent = trainingRuntime.job === 'preprocess'
                ? '当前预处理运行目录'
                : '当前任务运行目录';
        }
        if (configOutput) {
            configOutput.textContent = [
                task.runtime_config_file ? `实际运行配置: ${task.runtime_config_file}` : '',
                task.original_config_file ? `原始配置: ${task.original_config_file}` : '',
            ].filter(Boolean).join('\n');
        }
        renderHistoryPaths(task, { includeHistory: false });
    }

    function currentRuntimeTaskInfo() {
        return {
            run_dir: trainingRuntime.runDir,
            runtime_config_file: trainingRuntime.runtimeConfigFile,
            original_config_file: trainingRuntime.originalConfigFile,
            dataset_config_file: trainingRuntime.datasetConfigFile,
            model_cache_dir: trainingRuntime.modelCacheDir,
            dataset_cache_dir: trainingRuntime.datasetCacheDir,
            training_output_dir: trainingRuntime.trainingOutputDir,
            logs_dir: trainingRuntime.logsDir,
            output_dir: trainingRuntime.outputDir,
            sample_dir: trainingRuntime.sampleDir,
        };
    }

    function updateSystem(msg) {
        if (isHistoryReviewMode()) return;
        if (msg.last_output_at) {
            markTrainingActivity(msg.last_output_at);
        }
        if (msg.vram_used_gb !== undefined) {
            trainingRuntime.lastVramUsedGb = Number(msg.vram_used_gb);
            trainingRuntime.lastVramTotalGb = Number(msg.vram_total_gb);
            if (Number.isFinite(trainingRuntime.lastVramUsedGb)) {
                trainingRuntime.peakVramUsedGb = Math.max(
                    trainingRuntime.peakVramUsedGb ?? 0,
                    trainingRuntime.lastVramUsedGb
                );
            }
            setMetricText('metric-vram', formatRuntimeVram(
                trainingRuntime.lastVramUsedGb,
                trainingRuntime.lastVramTotalGb
            ));
            setMetricText('metric-vram-peak', formatRuntimeVram(
                trainingRuntime.peakVramUsedGb,
                trainingRuntime.lastVramTotalGb
            ));
        }
        if (msg.gpu_util !== undefined) {
            trainingRuntime.lastGpuUtil = Number(msg.gpu_util);
            if (Number.isFinite(trainingRuntime.lastGpuUtil)) {
                trainingRuntime.peakGpuUtil = Math.max(trainingRuntime.peakGpuUtil ?? 0, trainingRuntime.lastGpuUtil);
            }
            setMetricText('metric-gpu', formatSystemPercent(trainingRuntime.lastGpuUtil));
            setMetricText('metric-gpu-peak', formatSystemPercent(trainingRuntime.peakGpuUtil));
        }
        if (msg.gpu_temp !== undefined) {
            trainingRuntime.lastGpuTemp = Number(msg.gpu_temp);
            if (Number.isFinite(trainingRuntime.lastGpuTemp)) {
                trainingRuntime.peakGpuTemp = Math.max(trainingRuntime.peakGpuTemp ?? 0, trainingRuntime.lastGpuTemp);
            }
            setMetricText('metric-temp', formatSystemTemperature(trainingRuntime.lastGpuTemp));
            setMetricText('metric-temp-peak', formatSystemTemperature(trainingRuntime.peakGpuTemp));
        }
        renderLiveTrainingDashboard();
        refreshTrainingHealth();
    }

    function formatRuntimeVram(used, total) {
        const usedNumber = numberOrNull(used);
        if (usedNumber === null) return '-';
        const usedText = formatCompactNumber(usedNumber);
        if (usedText === '-') return '-';
        const totalNumber = numberOrNull(total);
        const totalText = totalNumber === null ? '-' : formatCompactNumber(totalNumber);
        return totalText === '-' ? `${usedText} GB` : `${usedText} / ${totalText} GB`;
    }

    function renderLiveTrainingDashboard() {
        if (isHistoryReviewMode()) return;
        const stateMap = { idle: '空闲', running: '运行中', error: '错误', compiling: '编译中' };
        const jobLabel = trainingRuntime.job === 'preprocess' ? '预处理' : '训练';
        const stateText = trainingRuntime.state === 'running'
            ? `${jobLabel}中`
            : (stateMap[trainingRuntime.state] || trainingRuntime.state || '空闲');
        setText('training-run-state', stateText);
        const stateEl = document.getElementById('training-run-state');
        if (stateEl) stateEl.className = `training-run-state ${trainingRuntime.state || 'idle'}`;
        updateTrainingToolbarState(trainingRuntime.state || 'idle', stateText);
        updateDashboardProgressIdleState(trainingRuntime.state === 'running' || trainingRuntime.state === 'compiling');
        setText('training-run-title', trainingRuntime.state === 'running' ? `当前${jobLabel}` : '当前监控');
        setText('training-run-meta', [
            trainingRuntime.methodsSubdir ? `方法目录 ${trainingRuntime.methodsSubdir}` : '',
            trainingRuntime.variant ? `配置 ${trainingRuntime.variant}` : '',
            trainingRuntime.preset ? `预设 ${trainingRuntime.preset}` : '',
        ].filter(Boolean).join(' · ') || '等待训练任务启动。');
        setText('training-run-summary', [
            trainingRuntime.runDir ? `运行目录: ${trainingRuntime.runDir}` : '',
            trainingRuntime.runtimeConfigFile ? `实际配置: ${trainingRuntime.runtimeConfigFile}` : '',
            trainingRuntime.outputDir ? `输出: ${trainingRuntime.outputDir}` : '',
            trainingRuntime.sampleDir ? `样张: ${trainingRuntime.sampleDir}` : '',
        ].filter(Boolean).join(' · ') || '运行目录和配置快照会在任务启动后显示。');
        setEtaMetricText(trainingEtaMetricInfo());
    }

    function trainingEtaMetricInfo() {
        const isRunning = trainingRuntime.state === 'running' || trainingRuntime.state === 'compiling';
        if (!isRunning) {
            return { text: '待计算', empty: true, title: '训练开始并收到进度后显示预计完成时间。' };
        }
        const current = Number(trainingRuntime.progressCurrent || 0);
        const total = Number(trainingRuntime.progressTotal || 0);
        if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
            return { text: '待计算', empty: true, title: '等待进度总数。' };
        }
        const remaining = Math.max(0, total - current);
        if (remaining <= 0) {
            return { text: '即将完成', empty: false, title: '当前进度已到达总步数。' };
        }
        const secondsPerStep = trainingRuntime.progressSecondsPerStep ?? parseProgressRateSeconds(trainingRuntime.progressRate);
        if (!Number.isFinite(secondsPerStep) || secondsPerStep <= 0) {
            return { text: '待计算', empty: true, title: '等待速度数据后计算预计完成时间。' };
        }
        const remainingSeconds = Math.ceil(remaining * secondsPerStep);
        if (!Number.isFinite(remainingSeconds) || remainingSeconds <= 0) {
            return { text: '即将完成', empty: false, title: '按当前速度估算，剩余不足 1 秒。' };
        }
        const eta = new Date(Date.now() + remainingSeconds * 1000);
        return {
            text: formatEtaClock(eta),
            empty: false,
            title: `按当前速度估算，剩余约 ${formatDuration(remainingSeconds)}。`,
        };
    }

    function parseProgressRateSeconds(value) {
        const text = String(value || '').trim().toLowerCase();
        if (!text) return null;
        const compact = text.replace(/\s+/g, '');
        const match = compact.match(/([\d.]+)(ms\/it|s\/it|s\/step|it\/s)/);
        if (!match) return null;
        const amount = Number(match[1]);
        if (!Number.isFinite(amount) || amount <= 0) return null;
        const unit = match[2];
        if (unit === 'it/s') return 1 / amount;
        if (unit === 'ms/it') return amount / 1000;
        return amount;
    }

    function formatEtaClock(date) {
        const pad = (value) => String(value).padStart(2, '0');
        const time = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
        const now = new Date();
        if (isSameDate(date, now)) return time;
        const tomorrow = new Date(now);
        tomorrow.setDate(now.getDate() + 1);
        if (isSameDate(date, tomorrow)) return `明日 ${time}`;
        return `${date.getMonth() + 1}/${date.getDate()} ${time}`;
    }

    function isSameDate(a, b) {
        return a.getFullYear() === b.getFullYear()
            && a.getMonth() === b.getMonth()
            && a.getDate() === b.getDate();
    }

    function markTrainingActivity(ts) {
        const value = Number(ts);
        const ms = value > 100000000000 ? value : value * 1000;
        if (Number.isFinite(ms) && ms > 0) {
            trainingRuntime.lastOutputAt = Math.max(trainingRuntime.lastOutputAt, ms);
        } else {
            trainingRuntime.lastOutputAt = Date.now();
        }
        trainingRuntime.lastUiActivityAt = Date.now();
        trainingRuntime.quietHintShown = false;
    }

    function refreshTrainingHealth() {
        const el = document.getElementById('training-health');
        const ageEl = document.getElementById('metric-log-age');
        if (!el || !ageEl) return;

        if (isHistoryReviewMode()) {
            el.className = 'training-health';
            return;
        }

        const isRunning = trainingRuntime.state === 'running' || trainingRuntime.state === 'compiling';
        if (!isRunning) {
            setMetricText('metric-log-age', 'N/A');
            setEtaMetricText({ text: '待计算', empty: true, title: '训练开始并收到进度后显示预计完成时间。' });
            el.className = 'training-health';
            el.textContent = '未运行任务。';
            return;
        }

        const ageSeconds = trainingRuntime.lastOutputAt
            ? Math.max(0, Math.floor((Date.now() - trainingRuntime.lastOutputAt) / 1000))
            : null;
        setMetricText('metric-log-age', ageSeconds == null ? 'N/A' : formatDuration(ageSeconds));
        setEtaMetricText(trainingEtaMetricInfo());

        const jobName = trainingRuntime.job === 'preprocess' ? '预处理' : '训练';

        const gpu = trainingRuntime.lastGpuUtil;
        const gpuActive = gpu != null && gpu >= 15;
        if (ageSeconds == null) {
            el.className = 'training-health';
            el.textContent = gpuActive
                ? `${jobName}运行中，GPU ${gpu}% 活跃，等待第一条日志。`
                : `${jobName}运行中，等待日志和系统指标。`;
            return;
        }

        if (ageSeconds >= 180 && gpuActive) {
            el.className = 'training-health warning';
            el.textContent = `已有 ${formatDuration(ageSeconds)} 没有新日志，但 GPU ${gpu}% 仍在工作；通常是单步较慢或任务脚本未输出进度。`;
            if (!trainingRuntime.quietHintShown) {
                appendLog(`[提示] ${el.textContent}`);
                trainingRuntime.quietHintShown = true;
            }
            return;
        }

        if (ageSeconds >= 180) {
            el.className = 'training-health error';
            el.textContent = `已有 ${formatDuration(ageSeconds)} 没有新日志，且 GPU 活跃度不高；建议观察进程或检查终端输出。`;
            return;
        }

        el.className = 'training-health ok';
        el.textContent = gpu == null
            ? `${jobName}运行中，最近 ${formatDuration(ageSeconds)} 前收到输出。`
            : `${jobName}运行中，最近 ${formatDuration(ageSeconds)} 前收到输出，GPU ${gpu}%。`;
    }

    function parseMetricsFromProgressLine(line) {
        const text = String(line || '');
        const stepMatch = text.match(/\|\s*(\d+)\/\d+\s*\[/) || text.match(/step[=:/\s]+(\d+)/i);
        const lossMatch = text.match(/(?:avr_)?loss[=:/\s]+([\d.eE\-+]+)/i);
        const lrMatch = text.match(/(?:^|[\s,])(?:lr|learning_rate)[=:/\s]+([\d.eE\-+]+)/i);
        const rateMatch = text.match(/([\d.]+\s*(?:s\/it|it\/s|s\/step))/i);
        const out = {};
        if (stepMatch) out.step = Number(stepMatch[1]);
        if (lossMatch) out.loss = Number(lossMatch[1]);
        if (lrMatch) out.lr = Number(lrMatch[1]);
        if (rateMatch) out.rate = rateMatch[1].replace(/\s+/g, '');
        if (Object.keys(out).length === 0) return null;
        if (out.step !== undefined && !Number.isFinite(out.step)) delete out.step;
        if (out.loss !== undefined && !Number.isFinite(out.loss)) delete out.loss;
        if (out.lr !== undefined && !Number.isFinite(out.lr)) delete out.lr;
        return Object.keys(out).length ? out : null;
    }

    function lastValue(records, key) {
        for (let i = records.length - 1; i >= 0; i -= 1) {
            const value = records[i]?.[key];
            if (value !== undefined && value !== null && value !== '') return value;
        }
        return undefined;
    }

    function readConfigNumber(configText, key) {
        const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const match = String(configText || '').match(new RegExp(`^\\s*${escapedKey}\\s*=\\s*([^\\n#]+)`, 'm'));
        if (!match) return undefined;
        const value = Number(match[1].trim().replace(/^["']|["']$/g, ''));
        return Number.isFinite(value) ? value : undefined;
    }

    function formatLr(value) {
        if (value === undefined || value === null || value === '') return '-';
        const n = Number(value);
        return Number.isFinite(n) ? n.toExponential(2) : '-';
    }

    function formatDuration(totalSeconds) {
        return ctx.format.formatDuration(totalSeconds);
    }

    // ── 全局设置 ──
    async function loadGlobalSettings() {
        if (location.protocol === 'file:') return;
        try {
            const data = await api('/api/settings/global');
            if (!data.ok) throw new Error(data.error || '读取全局设置失败');
            globalSettings = data;
            applyGlobalSettingsToInputs(data);
            updateChoiceGuide();
            setGlobalSettingsStatus('', '');
            if (tomlManagerMode === 'output') {
                await loadOutputRuns({ keepSelection: true });
            }
        } catch (e) {
            setGlobalSettingsStatus('读取全局设置失败: ' + e.message, 'error');
        }
    }

    async function saveGlobalSettings() {
        try {
            const payload = collectGlobalSettingsPayload();
            const res = await api('/api/settings/global', {
                method: 'PUT',
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                setGlobalSettingsStatus(res.error || '保存失败', 'error');
                return;
            }
            globalSettings = {
                ...(globalSettings || {}),
                ...res,
            };
            applyGlobalSettingsToInputs(globalSettings);
            updateChoiceGuide();
            setGlobalSettingsStatus(res.message || '全局设置已保存', 'ok');
        } catch (e) {
            setGlobalSettingsStatus('保存失败: ' + e.message, 'error');
        }
    }

    async function resetGlobalSettings() {
        applyGlobalSettingsToInputs({
            defaults: globalSettings?.defaults || {},
            output_root: globalSettings?.defaults?.output_root || 'output/runs',
            ...Object.fromEntries(GLOBAL_MODEL_PATH_FIELDS.map(([key]) => [key, globalSettings?.defaults?.[key] || ''])),
        });
        await saveGlobalSettings();
    }

    function setGlobalSettingsStatus(text, state = '') {
        const el = document.getElementById('global-settings-status');
        if (!el) return;
        el.textContent = text;
        el.className = `preview-status ${state}`.trim();
    }

    function applyGlobalSettingsToInputs(data) {
        const snapshot = data || globalSettings || {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            if (!input) continue;
            const fallback = snapshot?.defaults?.[key] || '';
            input.value = snapshot?.[key] ?? fallback;
        }
    }

    function collectGlobalSettingsPayload() {
        const payload = {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            payload[key] = input ? input.value : (globalSettings?.[key] || '');
        }
        return payload;
    }

    function getGlobalModelPathOverrides() {
        const overrides = {};
        const source = globalSettings || {};
        for (const [key] of GLOBAL_MODEL_PATH_FIELDS) {
            const value = source[key] ?? source.defaults?.[key] ?? '';
            if (String(value || '').trim()) {
                overrides[key] = String(value).trim();
            }
        }
        return overrides;
    }

    function toggleGlobalSettingHelp(button) {
        if (!button) return;
        const helpId = button.getAttribute('aria-controls');
        const help = helpId ? document.getElementById(helpId) : null;
        if (!help) return;
        const visible = help.classList.toggle('visible');
        button.classList.toggle('active', visible);
        button.setAttribute('aria-expanded', visible ? 'true' : 'false');
    }

    // ── 预览图 ──
    async function loadPreviewSettings() {
        return ensurePreviewFeature().loadPreviewSettings();
    }

    async function savePreviewSettings() {
        return ensurePreviewFeature().savePreviewSettings();
    }

    async function resetPreviewSettings() {
        return ensurePreviewFeature().resetPreviewSettings();
    }

    async function loadPreviewImages() {
        return ensurePreviewFeature().loadPreviewImages();
    }

    async function loadPreviewWeights() {
        return ensurePreviewFeature().loadPreviewWeights();
    }

    function setPreviewSource(source) {
        return ensurePreviewFeature().setPreviewSource(source);
    }

    async function openTrainingPreview(options = {}) {
        return ensurePreviewFeature().openTrainingPreview(options);
    }

    function openCurrentTrainingPreview(event) {
        return ensurePreviewFeature().openCurrentTrainingPreview(event);
    }

    async function openHistoryConfigGroupPreview(group) {
        return ensurePreviewFeature().openHistoryConfigGroupPreview(group);
    }

    function normalizePreviewGroup(group) {
        return ensurePreviewFeature().normalizePreviewGroup(group);
    }

    function renderPreviewTaskSelect() {
        return ensurePreviewFeature().renderPreviewTaskSelect();
    }

    async function changePreviewTask(taskId) {
        return ensurePreviewFeature().changePreviewTask(taskId);
    }

    function togglePreviewWeightSort() {
        return ensurePreviewFeature().togglePreviewWeightSort();
    }

    function openPreviewDialog(image) {
        return ensurePreviewFeature().openPreviewDialog(image);
    }

    function closePreviewImageDialog() {
        return ensurePreviewFeature().closePreviewImageDialog();
    }

    function openPreviewPanel() {
        return ensurePreviewFeature().openPreviewPanel();
    }

    function closePreviewPanel() {
        return ensurePreviewFeature().closePreviewPanel();
    }

    function restorePreviewWorkspaceAfterPanelClose() {
        return ensurePreviewFeature().restorePreviewWorkspaceAfterPanelClose();
    }

    function setPreviewStatus(text, state = '') {
        return ensurePreviewFeature().setPreviewStatus(text, state);
    }

    function createPreviewDetailRow(label, value) {
        return ensurePreviewFeature().createPreviewDetailRow(label, value);
    }

    function createPreviewDetailBlock(label, value, preformatted = false) {
        return ensurePreviewFeature().createPreviewDetailBlock(label, value, preformatted);
    }

    function renderDatasetImageDialogDetails(box, image, dims) {
        const caption = image.caption || {};
        const rows = [
            ['文件时间', image.mtime_text || '-'],
            ['尺寸', dims],
            ['长', image.height ? `${image.height} px` : '-'],
            ['宽', image.width ? `${image.width} px` : '-'],
            ['总像素', formatTotalPixels(image.total_pixels)],
            ['文件大小', formatBytes(image.size_bytes)],
        ];
        for (const [label, value] of rows) {
            box.appendChild(createPreviewDetailRow(label, value));
        }
        box.appendChild(createPreviewDetailBlock('文件路径', image.file || '-'));
        box.appendChild(createPreviewDetailBlock('标注文件', caption.file || '未找到同名标注文件'));
        const captionText = caption.ok ? (caption.text || '(空标注)') : '未找到同名 caption 文件';
        box.appendChild(createPreviewDetailBlock('标注内容', captionText, true));
    }

    function formatTotalPixels(totalPixels) {
        const count = Number(totalPixels);
        if (!Number.isFinite(count) || count <= 0) return '-';
        return `${count.toLocaleString('zh-CN')} px (${(count / 1000000).toFixed(2)} MP)`;
    }

    async function copyText(text) {
        return ctx.dom.copyText(text);
    }

    function formatBytes(bytes) {
        return ctx.format.formatBytes(bytes);
    }

    // ── 训练队列 ──
    async function loadTrainingQueue() {
        return ensureQueueFeature().loadTrainingQueue();
    }

    function updateTrainingQueueFromPayload(payload = {}) {
        return ensureQueueFeature().updateTrainingQueueFromPayload(payload);
    }

    function renderTrainingQueue() {
        return ensureQueueFeature().renderTrainingQueue();
    }

    function refreshQueueRunningProgressViews() {
        return ensureQueueFeature().updateRunningQueueProgress();
    }

    function showTrainingView(mode) {
        trainingViewMode = ['live', 'queue', 'history'].includes(mode) ? mode : 'live';
        renderTrainingViewMode();
    }

    function renderTrainingViewMode() {
        const queueView = document.getElementById('training-queue-manager');
        const monitorView = document.getElementById('training-monitor-view');
        const historyManager = document.getElementById('training-history-manager');
        const historyPlaceholder = document.getElementById('training-history-placeholder');
        const workspace = document.querySelector('#tab-training .training-workspace');
        const isQueue = trainingViewMode === 'queue';
        const isHistory = trainingViewMode === 'history';
        const mainWide = isQueue || isHistory;
        if (queueView) queueView.hidden = !isQueue;
        if (historyManager) historyManager.hidden = !isHistory;
        if (monitorView) monitorView.hidden = isQueue || isHistory;
        if (historyPlaceholder) historyPlaceholder.hidden = true;
        const trainingRoot = document.getElementById('tab-training');
        if (trainingRoot) {
            trainingRoot.classList.toggle('history-mode', isHistory);
            trainingRoot.classList.toggle('queue-mode', isQueue);
            trainingRoot.classList.toggle('live-mode', !isQueue && !isHistory);
        }
        if (workspace) {
            workspace.classList.toggle('main-wide', mainWide);
            workspace.classList.toggle('history-mode', isHistory);
        }
        document.querySelectorAll('.training-view-tab').forEach((btn) => {
            const active = btn.dataset.trainingView === trainingViewMode;
            btn.classList.toggle('active', active);
            btn.setAttribute('aria-selected', String(active));
        });
        if (isHistory) {
            renderHistoryManager();
        }
    }

    // ── 状态轮询 ──
    async function pollStatus() {
        if (isHistoryReviewMode()) return;
        try {
            const status = await api('/api/training/status');
            updateStatus({
                state: status.status,
                variant: status.variant,
                preset: status.preset,
                methods_subdir: status.methods_subdir,
                job: status.job,
                last_output_at: status.last_output_at,
                last_log_id: status.last_log_id,
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
            if ((status.last_log_id || 0) > trainingRuntime.lastLogId) {
                await replayTrainingLogs();
            }
        } catch (e) { /* ignore */ }
    }

    async function loadTrainingHistoryList() {
        if (location.protocol === 'file:') return;
        try {
            const params = new URLSearchParams();
            params.set('include_archived', '1');
            const suffix = params.toString() ? `?${params.toString()}` : '';
            const payload = await api(`/api/training/history${suffix}`);
            historyTasks = payload.tasks || [];
            await loadHistoryCollectionSettings();
            renderTrainingHistoryList();
            renderHistoryManager();
            renderPreviewTaskSelect();
            setPreviewStatus('', '');
        } catch (e) {
            const list = document.getElementById('task-history-list');
            if (list) list.textContent = '读取任务列表失败';
            const managerList = document.getElementById('history-manager-list');
            if (managerList) managerList.textContent = '读取历史任务失败';
            renderPreviewTaskSelect();
            setPreviewStatus('读取训练任务列表失败: ' + e.message, 'error');
        }
    }

    async function loadHistoryCollectionSettings() {
        if (location.protocol === 'file:') return;
        try {
            const payload = await api('/api/training/history/collections/settings');
            if (payload.ok !== false) {
                historyCollectionSettings = normalizeHistoryCollectionSettings(payload);
            }
        } catch (e) {
            appendLog(`[状态] 读取历史集合设置失败: ${e.message}`);
        }
    }

    async function saveHistoryCollectionSettings(nextSettings) {
        historyCollectionSettings = normalizeHistoryCollectionSettings(nextSettings);
        if (location.protocol === 'file:') {
            renderHistoryManager();
            return historyCollectionSettings;
        }
        try {
            const payload = await api('/api/training/history/collections/settings', {
                method: 'PUT',
                body: JSON.stringify(historyCollectionSettings),
            });
            if (payload.ok === false) throw new Error(payload.error || '保存集合设置失败');
            historyCollectionSettings = normalizeHistoryCollectionSettings(payload);
            renderHistoryManager();
            return historyCollectionSettings;
        } catch (e) {
            appendLog(`[状态] 保存历史集合设置失败: ${e.message}`);
            renderHistoryManager();
            return historyCollectionSettings;
        }
    }

    function normalizeHistoryCollectionSettings(payload = {}) {
        return {
            collection_order: uniqueStringList(payload.collection_order),
            config_group_order: normalizeHistoryConfigGroupOrder(payload.config_group_order),
        };
    }

    function uniqueStringList(value) {
        const list = Array.isArray(value) ? value : [];
        const out = [];
        const seen = new Set();
        for (const item of list) {
            const text = String(item || '').trim();
            if (!text || seen.has(text)) continue;
            out.push(text);
            seen.add(text);
        }
        return out;
    }

    function normalizeHistoryConfigGroupOrder(value) {
        if (!value || typeof value !== 'object') return {};
        const out = {};
        for (const [key, order] of Object.entries(value)) {
            const cleanKey = String(key || '').trim();
            const cleanOrder = uniqueStringList(order);
            if (cleanKey && cleanOrder.length) out[cleanKey] = cleanOrder;
        }
        return out;
    }

    function renderTrainingHistoryList() {
        const list = document.getElementById('task-history-list');
        if (!list) return;
        list.innerHTML = '';
        const recentTasks = recentTrainingSidebarTasks();
        if (!recentTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'task-history-empty';
            empty.textContent = historyTasks.length
                ? '最近没有未归档训练任务；归档和预处理请到历史任务大界面查看。'
                : '暂无历史任务。下一次训练启动后会自动记录。';
            list.appendChild(empty);
            return;
        }
        for (const task of recentTasks) {
            list.appendChild(createHistoryTaskItem(task));
        }
    }

    function recentTrainingSidebarTasks() {
        return historyTasks
            .filter((task) => task.job === 'training' && !historyTaskIsArchived(task))
            .sort((a, b) => {
                const aTime = Number(a.started_at || a.updated_at || 0);
                const bTime = Number(b.started_at || b.updated_at || 0);
                return (bTime - aTime) || String(b.id || '').localeCompare(String(a.id || ''), 'zh-CN');
            })
            .slice(0, 6);
    }

    function renderHistoryManager() {
        const panel = document.getElementById('training-history-manager');
        if (!panel) return;
        syncHistorySelectionWithTasks();
        renderHistoryManagerStats();
        const status = document.getElementById('history-manager-status');
        const list = document.getElementById('history-manager-list');
        const tablePanel = list?.closest('.history-table-panel');
        const selectAll = document.getElementById('history-select-all');
        const mergeBtn = document.getElementById('btn-history-manager-merge');
        if (!list) return;
        const baseVisible = historyManagerBaseFilteredTasks();
        const visible = historyManagerVisibleTasks(baseVisible);
        historyCurrentVisibleTaskIds = historyTaskIds(visible);
        if (tablePanel) tablePanel.classList.add('collections-mode');
        if (status) {
            const archivedCount = historyTasks.filter(historyTaskIsArchived).length;
            status.textContent = [
                `共 ${historyTasks.length} 条记录`,
                `当前分组 ${visible.length} 条`,
                `筛选后 ${baseVisible.length} 条`,
                `归档 ${archivedCount} 条`,
                historyDropFeedback.message,
            ].filter(Boolean).join(' · ');
            status.dataset.feedbackTone = historyDropFeedback.tone || '';
        }
        list.innerHTML = '';
        if (!historyTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'history-manager-empty';
            empty.textContent = '暂无历史任务。';
            list.appendChild(empty);
        } else {
            renderHistoryManagerItems(list, baseVisible);
        }
        if (selectAll) {
            const visibleIds = historyCurrentVisibleTaskIds;
            const selectedVisible = visibleIds.filter((id) => selectedHistoryTaskIds.has(id)).length;
            selectAll.checked = visibleIds.length > 0 && selectedVisible === visibleIds.length;
            selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleIds.length;
        }
        renderHistoryBulkBar();
        if (mergeBtn) {
            mergeBtn.disabled = selectedHistoryTasks().filter((task) => task.job === 'training').length === 0;
        }
        renderHistoryDetailDialog();
    }

    function renderHistoryManagerItems(list, visible) {
        list.dataset.groupMode = 'collections';
        renderHistoryCollectionsWorkbench(list, visible);
    }

    function resetTrainingExpandedStateOnLeave() {
        if (trainingViewMode === 'history') {
            renderHistoryManager();
        }
    }

    function renderHistoryCollectionsWorkbench(list, visible) {
        const workbench = document.createElement('div');
        workbench.className = 'history-collections-workbench compact';
        if (historyDragState.active) workbench.classList.add('dragging');
        if (historyDragState.pending) workbench.classList.add('drop-pending');
        if (historyCollectionDragState.active) workbench.classList.add('collection-reordering');

        const allCollections = historyCollectionsForWorkbench(visible);
        if (historyCollectionWorkbenchTarget && !allCollections.some((item) => item.value === historyCollectionWorkbenchTarget)) {
            historyCollectionWorkbenchTarget = '';
        }
        const smartSearch = historySmartSearchTerms();
        const collectionSearchTerms = historySearchTerms(historyCollectionSearch, smartSearch.collection);
        const configSearchTerms = historySearchTerms(historyConfigGroupSearch, smartSearch.config);
        const configSearch = configSearchTerms.join(' ');
        const visibleCollections = visibleHistoryCollectionsForSearch(allCollections, collectionSearchTerms);
        const selectedCollection = selectedHistoryCollectionForWorkbench(allCollections, collectionSearchTerms);
        const scopedTasks = selectedCollection.tasks || [];
        const configGroups = sortedHistoryConfigGroups(
            groupHistoryTasks(scopedTasks).map(sortHistoryManagerGroupTasks),
            historyCollectionStorageKey(selectedCollection),
        );
        const visibleConfigGroups = configGroups.filter((group) =>
            historySearchTextMatches(historyConfigGroupSearchText(group), configSearchTerms)
            || (configSearch && historyConfigGroupSearchText(group).includes(configSearch))
        );
        const currentVisibleTasks = uniqueHistoryTasks(visibleConfigGroups.flatMap((group) => group.tasks || []));
        historyCurrentVisibleTaskIds = historyTaskIds(currentVisibleTasks);
        const selectedTasks = currentVisibleTasks.filter((task) => task.id && selectedHistoryTaskIds.has(task.id));
        const selectedGroups = selectedHistoryConfigGroups(visibleConfigGroups);

        const head = document.createElement('div');
        head.className = 'history-collections-head';
        const title = document.createElement('div');
        title.className = 'history-collections-title';
        const heading = document.createElement('strong');
        heading.textContent = '历史分组';
        const desc = document.createElement('span');
        desc.textContent = `左侧: ${selectedCollection.is_ungrouped ? '未分类' : selectedCollection.label} · 右侧切换/拖拽归类`;
        title.append(heading, desc);

        const stats = document.createElement('div');
        stats.className = 'history-collections-stats';
        [
            ['分组', allCollections.filter((item) => !item.is_ungrouped).length],
            ['当前任务', currentVisibleTasks.length],
            ['配置组', visibleConfigGroups.length],
            ['已选分组', selectedGroups.length],
        ].forEach(([label, value]) => {
            const item = document.createElement('div');
            item.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
            stats.appendChild(item);
        });
        head.append(title, stats);
        workbench.appendChild(head);

        const toolbar = document.createElement('div');
        toolbar.className = 'history-collections-toolbar';
        const target = document.createElement('span');
        target.textContent = [
            `当前: ${selectedCollection.label}`,
            historyCollectionWorkbenchTarget ? `目标: ${historyCollectionWorkbenchTarget}` : '',
            selectedTasks.length ? `已选: ${selectedTasks.length}` : '未选',
        ].filter(Boolean).join(' · ');
        toolbar.appendChild(target);
        toolbar.append(
            createHistoryCollectionsToolbarButton('设置分组', () => groupSelectedHistoryTasks(), !selectedTasks.length),
            createHistoryCollectionsToolbarButton('清除分组', () => clearSelectedHistoryCollection(), !selectedTasks.length),
        );
        if (historyCollectionWorkbenchTarget) {
            toolbar.appendChild(createHistoryCollectionsToolbarButton(
                '加入目标',
                () => applySelectedHistoryTasksToCollection(historyCollectionWorkbenchTarget),
                !selectedTasks.length,
            ));
        }
        workbench.appendChild(toolbar);

        const body = document.createElement('div');
        body.className = 'history-collections-body';

        const configPanel = document.createElement('section');
        configPanel.className = 'history-collections-panel current-content history-current-group-content';
        configPanel.appendChild(historyCollectionsPanelTitle(
            selectedCollection.is_ungrouped ? '未分类任务' : `${selectedCollection.label} 内的任务`,
            `${visibleConfigGroups.length}/${configGroups.length} 组 · ${currentVisibleTasks.length} 条`,
        ));
        const configList = document.createElement('div');
        configList.className = 'history-config-group-card-list';
        const splitCollections = historyConfigGroupCollectionMap(visible);
        if (!visibleConfigGroups.length) {
            const empty = document.createElement('div');
            empty.className = 'history-current-group-empty';
            empty.textContent = selectedCollection.is_ungrouped ? '未分类暂无任务。' : '该分组暂无任务。';
            configList.appendChild(empty);
        } else {
            for (const group of visibleConfigGroups) {
                configList.appendChild(createHistoryConfigGroupWorkbenchCard(group, splitCollections, {
                    groups: configGroups,
                    collection: selectedCollection,
                }));
            }
        }
        configPanel.appendChild(configList);

        const collectionPanel = document.createElement('section');
        collectionPanel.className = 'history-collections-panel collection-nav history-collection-nav';
        const collectionPanelHead = document.createElement('div');
        collectionPanelHead.className = 'history-collection-nav-head';
        collectionPanelHead.appendChild(historyCollectionsPanelTitle('分组导航', `${visibleCollections.length}/${allCollections.length} 组`));
        const createBtn = createHistoryCollectionsToolbarButton('新建分组', (event) => openHistoryNewCollectionPopover(event, []));
        createBtn.classList.add('history-collection-create-btn');
        collectionPanelHead.appendChild(createBtn);
        collectionPanel.appendChild(collectionPanelHead);
        const collectionList = document.createElement('div');
        collectionList.className = 'history-collection-card-list';
        for (const collection of visibleCollections) {
            collectionList.appendChild(createHistoryCollectionWorkbenchCard(collection, selectedTasks.length, allCollections));
        }
        collectionPanel.appendChild(collectionList);

        body.append(configPanel, collectionPanel);
        workbench.appendChild(body);
        renderHistoryDropPopover(workbench);
        list.appendChild(workbench);
    }

    function renderHistoryManagerStats() {
        const el = document.getElementById('history-manager-stats');
        if (!el) return;
        const counts = {
            total: historyTasks.length,
            training: historyTasks.filter((task) => task.job === 'training').length,
            preprocess: historyTasks.filter((task) => task.job === 'preprocess').length,
            error: historyTasks.filter((task) => ['error', 'interrupted'].includes(task.state)).length,
            archived: historyTasks.filter(historyTaskIsArchived).length,
            queue: historyTasks.filter((task) => task.from_queue || task.queue_item_id).length,
        };
        el.innerHTML = '';
        [
            ['全部', counts.total, 'all'],
            ['训练', counts.training, 'training'],
            ['预处理', counts.preprocess, 'preprocess'],
            ['异常/中断', counts.error, 'error'],
            ['归档', counts.archived, 'archived'],
            ['来自队列', counts.queue, 'queue'],
        ].forEach(([label, value, state]) => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = ['history-manager-stat', state, historyStatFilterIsActive(state) ? 'active' : ''].filter(Boolean).join(' ');
            item.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
            item.addEventListener('click', () => applyHistoryStatFilter(state));
            el.appendChild(item);
        });
    }

    function applyHistoryStatFilter(state) {
        historyCollectionSearch = '';
        historyConfigGroupSearch = '';
        const next = {
            search: '',
            kind: 'all',
            state: 'all',
            archived: 'all',
            source: 'all',
            sort: historyManagerFilters.sort || 'newest',
        };
        if (state === 'training' || state === 'preprocess') {
            next.kind = state;
        } else if (state === 'error') {
            next.state = 'error';
        } else if (state === 'archived') {
            next.archived = 'archived';
        } else if (state === 'queue') {
            next.source = 'queue';
        }
        historyManagerFilters = next;
        syncHistoryFilterControls();
        renderHistoryManager();
    }

    function historyStatFilterIsActive(state) {
        const searchEmpty =
            !String(historyManagerFilters.search || '').trim() &&
            !String(historyCollectionSearch || '').trim() &&
            !String(historyConfigGroupSearch || '').trim();
        const base =
            searchEmpty &&
            Boolean(historyManagerFilters.sort || 'newest') &&
            (state === 'archived'
                ? historyManagerFilters.archived === 'archived'
                : (historyManagerFilters.archived || 'active') === 'all');
        if (!base) return false;
        if (state === 'all') {
            return historyManagerFilters.kind === 'all' &&
                historyManagerFilters.state === 'all' &&
                historyManagerFilters.source === 'all';
        }
        if (state === 'training' || state === 'preprocess') {
            return historyManagerFilters.kind === state &&
                historyManagerFilters.state === 'all' &&
                historyManagerFilters.source === 'all';
        }
        if (state === 'error') {
            return historyManagerFilters.kind === 'all' &&
                historyManagerFilters.state === 'error' &&
                historyManagerFilters.source === 'all';
        }
        if (state === 'archived') {
            return historyManagerFilters.kind === 'all' &&
                historyManagerFilters.state === 'all' &&
                historyManagerFilters.archived === 'archived' &&
                historyManagerFilters.source === 'all';
        }
        if (state === 'queue') {
            return historyManagerFilters.kind === 'all' &&
                historyManagerFilters.state === 'all' &&
                historyManagerFilters.source === 'queue';
        }
        return false;
    }

    function historyManagerFilteredTasks() {
        return historyManagerVisibleTasks(historyManagerBaseFilteredTasks());
    }

    function historyManagerBaseFilteredTasks() {
        const search = historySmartSearchTerms().global;
        const visible = historyTasks.filter((task) => {
            if (historyManagerFilters.kind !== 'all' && task.job !== historyManagerFilters.kind) return false;
            if (historyManagerFilters.state !== 'all') {
                if (historyManagerFilters.state === 'error') {
                    if (!['error', 'interrupted'].includes(task.state)) return false;
                } else if (task.state !== historyManagerFilters.state) {
                    return false;
                }
            }
            const archived = historyTaskIsArchived(task);
            if (historyManagerFilters.archived === 'active' && archived) return false;
            if (historyManagerFilters.archived === 'archived' && !archived) return false;
            if (!historyTaskMatchesSourceFilter(task, historyManagerFilters.source)) return false;
            if (search && !historyTaskSearchText(task).includes(search)) return false;
            return true;
        });
        visible.sort(historyTaskSortComparator(historyManagerFilters.sort));
        return visible;
    }

    function historyManagerVisibleTasks(baseTasks) {
        const base = baseTasks || [];
        const smartSearch = historySmartSearchTerms();
        const collectionSearchTerms = historySearchTerms(historyCollectionSearch, smartSearch.collection);
        const configSearchTerms = historySearchTerms(historyConfigGroupSearch, smartSearch.config);
        const collections = historyCollectionsForWorkbench(base);
        const selectedCollection = selectedHistoryCollectionForWorkbench(collections, collectionSearchTerms);
        const visibleGroups = (selectedCollection.groups || [])
            .filter((group) => historySearchTextMatches(historyConfigGroupSearchText(group), configSearchTerms));
        return uniqueHistoryTasks(visibleGroups.flatMap((group) => group.tasks || []));
    }

    function uniqueHistoryTasks(tasks) {
        const seen = new Set();
        const out = [];
        for (const task of tasks || []) {
            const key = task?.id || `${task?.history_dir || ''}:${out.length}`;
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(task);
        }
        out.sort(historyTaskSortComparator(historyManagerFilters.sort));
        return out;
    }

    function historyTaskMatchesSourceFilter(task, filter) {
        if (!filter || filter === 'all') return true;
        if (filter === 'queue') return Boolean(task.from_queue || task.queue_item_id);
        if (filter === 'resume') return Boolean(task.resume_from?.source_task_id);
        if (filter === 'continue') return task.training_mode === 'continue_lora';
        return true;
    }

    function historyTaskSearchText(task) {
        return [
            task.id,
            historyTaskDisplayName(task),
            task.name,
            task.group,
            task.history_group_label,
            task.history_source_config_file,
            task.history_run_label,
            task.methods_subdir,
            task.variant,
            task.preset,
            task.run_dir,
            task.training_output_dir,
            task.output_dir,
            task.message,
        ].filter(Boolean).join('\n').toLowerCase();
    }

    function historyTaskMatchesCollectionSearch(task, search) {
        return [
            historyTaskCollectionLabel(task),
            historyTaskCollectionValue(task),
            historyTaskSearchText(task),
        ].filter(Boolean).join('\n').toLowerCase().includes(search);
    }

    function historySmartSearchTerms() {
        const raw = String(historyManagerFilters.search || '').trim();
        const terms = { global: '', collection: '', config: '' };
        const match = raw.match(/^([^:：]+)\s*[:：]\s*(.*)$/);
        if (!match) {
            terms.global = raw.toLowerCase();
            return terms;
        }
        const prefix = match[1].trim().toLowerCase();
        const value = match[2].trim().toLowerCase();
        if (!value) return terms;
        if (['组', '集合', 'group', 'collection'].includes(prefix)) {
            terms.collection = value;
        } else if (['配置', '配置组', 'config'].includes(prefix)) {
            terms.config = value;
        } else {
            terms.global = raw.toLowerCase();
        }
        return terms;
    }

    function historySearchTerms(...values) {
        return values.map((value) => String(value || '').trim().toLowerCase()).filter(Boolean);
    }

    function historySearchTextMatches(text, terms) {
        const haystack = String(text || '').toLowerCase();
        return terms.every((term) => haystack.includes(term));
    }

    function historyCollectionMatchesSearch(collection, terms) {
        if (!terms.length) return true;
        const text = historyCollectionSearchText(collection);
        const phrase = terms.join(' ');
        return historySearchTextMatches(text, terms) || Boolean(phrase && text.includes(phrase));
    }

    function visibleHistoryCollectionsForSearch(collections, terms) {
        return (collections || []).filter((collection) => historyCollectionMatchesSearch(collection, terms));
    }

    function selectedHistoryCollectionForWorkbench(collections, collectionSearchTerms = []) {
        const allCollections = collections || [];
        const visibleCollections = visibleHistoryCollectionsForSearch(allCollections, collectionSearchTerms);
        if (collectionSearchTerms.length && !visibleCollections.length) {
            selectedHistoryCollectionKey = 'collection:__search_empty__';
            return createHistoryCollectionSearchEmptyCollection();
        }
        const candidates = collectionSearchTerms.length ? visibleCollections : allCollections;
        const selected = historyCollectionByKey(candidates, selectedHistoryCollectionKey)
            || (collectionSearchTerms.length ? candidates[0] : null)
            || historyCollectionByKey(allCollections, selectedHistoryCollectionKey)
            || historyCollectionByKey(allCollections, HISTORY_UNGROUPED_COLLECTION_KEY)
            || createEmptyHistoryCollection();
        selectedHistoryCollectionKey = selected.key;
        return selected;
    }

    function historyTaskSortComparator(mode) {
        return (a, b) => {
            if (mode === 'oldest') return (Number(a.started_at || 0) - Number(b.started_at || 0));
            if (mode === 'loss') return (Number(b.metric_count || 0) - Number(a.metric_count || 0)) || (Number(b.started_at || 0) - Number(a.started_at || 0));
            if (mode === 'logs') return (Number(b.log_count || 0) - Number(a.log_count || 0)) || (Number(b.started_at || 0) - Number(a.started_at || 0));
            if (mode === 'name') return historyTaskDisplayName(a).localeCompare(historyTaskDisplayName(b), 'zh-CN');
            return (Number(b.started_at || 0) - Number(a.started_at || 0));
        };
    }

    function createHistoryManagerRow(task) {
        const row = document.createElement('article');
        row.className = 'history-manager-row';
        if (viewingHistoryTaskId === task.id && isHistoryDetailDialogOpen()) row.classList.add('active');
        if (historyTaskIsArchived(task)) row.classList.add('archived');

        const select = document.createElement('label');
        select.className = 'history-row-select';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = selectedHistoryTaskIds.has(task.id);
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) selectedHistoryTaskIds.add(task.id);
            else selectedHistoryTaskIds.delete(task.id);
            renderHistoryManager();
        });
        select.appendChild(checkbox);

        const main = document.createElement('button');
        main.type = 'button';
        main.className = 'history-row-main';
        main.addEventListener('click', () => loadHistoryTask(task.id));
        const title = document.createElement('strong');
        title.textContent = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
        title.title = title.textContent;
        const meta = document.createElement('span');
        meta.className = 'history-compact-meta';
        meta.textContent = [
            compactHistoryPathLabel(task.history_source_config_file || `${task.methods_subdir || '-'} / ${task.variant || '-'}`),
            compactHistoryQueueLabel(task),
            compactHistoryContinueLabel(task),
            compactHistoryResumeLabel(task),
        ].filter(Boolean).join(' · ');
        meta.title = [
            task.history_source_config_file || `${task.methods_subdir || '-'} / ${task.variant || '-'}`,
            historyQueueLabel(task),
            historyContinueLabel(task),
            historyResumeLabel(task),
        ].filter(Boolean).join(' · ');
        main.append(title, meta);

        const state = document.createElement('div');
        state.className = ['history-row-state', task.state || 'unknown'].join(' ');
        state.textContent = [
            task.job === 'preprocess' ? '预处理' : '训练',
            historyStateLabel(task.state),
            historyTaskIsArchived(task) ? '已归档' : '',
        ].filter(Boolean).join(' · ');

        const time = document.createElement('div');
        time.className = 'history-row-time';
        time.textContent = `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`;

        const data = document.createElement('div');
        data.className = 'history-row-data';
        data.textContent = `${task.metric_count || 0} loss / ${task.log_count || 0} log`;

        const actions = document.createElement('div');
        actions.className = 'history-row-actions';
        if (task.job === 'training') {
            actions.append(
                createHistoryTaskPreviewButton(task),
            );
        }
        actions.append(
            createHistoryActionButton('查看', () => loadHistoryTask(task.id)),
            createHistoryMoreActions([
                createHistoryTaskConfigButton(task),
                createHistoryActionButton(historyTaskIsArchived(task) ? '取消归档' : '归档', () => archiveHistoryTask(task)),
                createHistoryActionButton('删除', () => deleteHistoryTask(task), 'danger'),
            ]),
        );

        row.append(select, main, state, time, data, actions);
        return row;
    }

    function compactHistoryPathLabel(value) {
        const text = String(value || '').trim();
        if (!text) return '';
        return runLabelFromPath(text) || text;
    }

    function compactHistoryQueueLabel(task) {
        if (!Boolean(task?.from_queue) && !String(task?.queue_item_id || '').trim()) return '';
        const attempt = Number(task?.queue_attempt || 1);
        return attempt > 1 ? `队列#${attempt}` : '队列';
    }

    function compactHistoryContinueLabel(task) {
        if (task?.training_mode !== 'continue_lora') return '';
        const kind = String(task.continue_from_weight_kind || 'LoRA').trim() || 'LoRA';
        const name = compactHistoryPathLabel(task.continue_from_weight_name || '');
        return name ? `续训 ${kind}:${name}` : `续训 ${kind}`;
    }

    function compactHistoryResumeLabel(task) {
        const resume = task?.resume_from || {};
        if (!resume || typeof resume !== 'object') return '';
        const checkpoint = compactHistoryPathLabel(resume.checkpoint_name || '');
        const step = resume.checkpoint_step !== undefined && resume.checkpoint_step !== null
            ? String(resume.checkpoint_step).trim()
            : '';
        if (checkpoint && step) return `恢复 ${checkpoint}@${step}`;
        if (checkpoint) return `恢复 ${checkpoint}`;
        if (step) return `恢复 step ${step}`;
        return resume.source_task_id ? '恢复' : '';
    }

    function createHistoryMoreActions(buttons) {
        const menu = document.createElement('details');
        menu.className = 'history-more-actions';
        menu.addEventListener('click', (event) => event.stopPropagation());
        const summary = document.createElement('summary');
        summary.textContent = '...';
        summary.title = '更多历史任务操作';
        summary.setAttribute('aria-label', '更多历史任务操作');
        const body = document.createElement('div');
        body.className = 'history-more-actions-menu';
        for (const button of buttons.filter(Boolean)) {
            body.appendChild(button);
        }
        menu.append(summary, body);
        return menu;
    }

    function selectedHistoryConfigGroups(groups) {
        return (groups || []).filter((group) => historyTaskIds(group.tasks).some((id) => selectedHistoryTaskIds.has(id)));
    }

    function historyCollectionSearchText(collection) {
        return [
            collection.label,
            collection.value,
            ...collection.groups.map((group) => historyGroupDisplayLabel(group)),
            ...collection.groups.map((group) => group.source_label || group.history_source_config_file || ''),
            ...collection.tasks.map((task) => historyTaskSearchText(task)),
        ].filter(Boolean).join('\n').toLowerCase();
    }

    function historyConfigGroupSearchText(group) {
        return [
            historyGroupDisplayLabel(group),
            group.source_label,
            group.history_source_config_file,
            group.fallback_group_label,
            ...group.tasks.map((task) => historyTaskSearchText(task)),
        ].filter(Boolean).join('\n').toLowerCase();
    }

    function createEmptyHistoryCollection(value = '') {
        const clean = String(value || '').trim();
        return enrichHistoryCollection({
            key: clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY,
            label: clean || '未分类',
            value: clean,
            is_ungrouped: !clean,
            tasks: [],
        });
    }

    function createHistoryCollectionSearchEmptyCollection() {
        return enrichHistoryCollection({
            key: 'collection:__search_empty__',
            label: '无匹配分组',
            value: '__search_empty__',
            is_ungrouped: false,
            tasks: [],
        });
    }

    function normalizeHistoryCollectionForWorkbench(collection) {
        const clean = String(collection?.value || '').trim();
        return enrichHistoryCollection({
            ...(collection || {}),
            key: clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY,
            label: clean || '未分类',
            value: clean,
            is_ungrouped: !clean,
            tasks: collection?.tasks || [],
        });
    }

    function historyCollectionsForWorkbench(tasks) {
        const byKey = new Map();
        for (const collection of groupHistoryTasksByCollection(tasks || [])) {
            const normalized = normalizeHistoryCollectionForWorkbench(collection);
            byKey.set(normalized.key, normalized);
        }
        if (!byKey.has(HISTORY_UNGROUPED_COLLECTION_KEY)) {
            const ungrouped = createEmptyHistoryCollection();
            byKey.set(ungrouped.key, ungrouped);
        }
        for (const value of uniqueStringList(historyCollectionSettings.collection_order || [])) {
            const clean = String(value || '').trim();
            const key = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;
            if (clean && !byKey.has(key)) {
                byKey.set(key, createEmptyHistoryCollection(clean));
            }
        }
        return Array.from(byKey.values()).sort(historyCollectionComparator);
    }

    function historyCollectionSelectOptions() {
        const collections = historyCollectionsForWorkbench(historyTasks);
        return collections.map((collection) => ({
            key: collection.key,
            label: collection.label,
            value: collection.value || '',
            task_count: collection.tasks.length,
            group_count: collection.groups.length,
            search_text: historyCollectionSearchText(collection),
        }));
    }

    function historyCollectionOptionSearchText(item) {
        return [
            item.label,
            item.value,
            item.search_text,
        ].filter(Boolean).join('\n').toLowerCase();
    }

    function historyCollectionsPanelTitle(titleText, metaText) {
        const title = document.createElement('div');
        title.className = 'history-collections-panel-title';
        const titleEl = document.createElement('strong');
        titleEl.textContent = titleText;
        const meta = document.createElement('span');
        meta.textContent = metaText;
        title.append(titleEl, meta);
        return title;
    }

    function createHistoryCollectionsToolbarButton(label, handler, disabled = false, tone = '') {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = ['task-history-action', tone].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.disabled = Boolean(disabled);
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            if (!btn.disabled) handler(event);
        });
        return btn;
    }

    function stopHistoryGroupButtonPropagation(button) {
        button.addEventListener('click', (event) => event.stopPropagation(), { capture: true });
        return button;
    }

    function historyDragTaskIdsForGroup(group) {
        const groupIds = historyTaskIds(group?.tasks || []);
        const visible = new Set(historyCurrentVisibleTaskIds);
        const selectedIds = Array.from(selectedHistoryTaskIds)
            .filter((id) => id && (!visible.size || visible.has(id)));
        if (selectedIds.some((id) => groupIds.includes(id))) {
            return selectedIds;
        }
        return groupIds;
    }

    function createHistoryDragImage(count) {
        removeHistoryDragImage();
        const image = document.createElement('div');
        image.className = 'history-drag-image';
        image.textContent = `${count} 条任务`;
        document.body.appendChild(image);
        historyDragImageElement = image;
        return image;
    }

    function removeHistoryDragImage() {
        if (historyDragImageElement?.parentNode) {
            historyDragImageElement.parentNode.removeChild(historyDragImageElement);
        }
        historyDragImageElement = null;
    }

    function canBeginHistoryConfigGroupDrag(group) {
        return Boolean(!historyDragState.pending && !historyConfigGroupSortState.pending && historyTaskIds(group?.tasks || []).length);
    }

    function beginHistoryConfigGroupDrag(event, group, options = {}) {
        if (historyDragState.pending) {
            event.preventDefault();
            return;
        }
        const taskIds = uniqueStringList(historyDragTaskIdsForGroup(group));
        if (!taskIds.length) {
            event.preventDefault();
            return;
        }
        closeHistoryDropPopover(false);
        historyDragState = {
            ...historyDragState,
            active: true,
            taskIds,
            sourceGroupKey: configGroupKey(group),
            activeDropTarget: '',
            popover: {
                open: false,
                x: 0,
                y: 0,
                taskIds: [],
                defaultName: '',
            },
        };
        const payload = JSON.stringify(taskIds);
        const groupKey = configGroupKey(group);
        const collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        historyConfigGroupSortState = {
            active: Boolean(groupKey),
            sourceKey: groupKey,
            collectionKey,
            activeDropTarget: '',
            dropPosition: 'after',
            pending: false,
        };
        if (event.dataTransfer) {
            event.dataTransfer.setData(HISTORY_TASK_DRAG_MIME, payload);
            event.dataTransfer.setData(HISTORY_CONFIG_GROUP_DRAG_MIME, JSON.stringify({ groupKey, collectionKey }));
            event.dataTransfer.setData('text/plain', payload);
            event.dataTransfer.effectAllowed = 'move';
            const dragImage = createHistoryDragImage(taskIds.length);
            event.dataTransfer.setDragImage(dragImage, 18, 18);
        }
        document.querySelector('.history-collections-workbench')?.classList.add('dragging');
    }

    function finishHistoryDrag() {
        removeHistoryDragImage();
        removeHistoryConfigGroupDropPreview();
        historyDragState.active = false;
        historyDragState.taskIds = [];
        historyDragState.sourceGroupKey = '';
        historyDragState.activeDropTarget = '';
        historyConfigGroupSortState = {
            active: false,
            sourceKey: '',
            collectionKey: '',
            activeDropTarget: '',
            dropPosition: 'after',
            pending: false,
        };
        document.querySelectorAll('.history-collection-card.drop-active').forEach((item) => {
            item.classList.remove('drop-active');
        });
        document.querySelectorAll('.history-config-group-card.config-sort-active, .history-config-group-card.config-sort-source').forEach((item) => {
            item.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after', 'config-sort-source');
        });
        document.querySelector('.history-collections-workbench')?.classList.remove('dragging');
    }

    function readHistoryDraggedConfigGroup(event) {
        try {
            const raw = event?.dataTransfer?.getData(HISTORY_CONFIG_GROUP_DRAG_MIME);
            if (raw) {
                const parsed = JSON.parse(raw);
                return {
                    groupKey: String(parsed?.groupKey || '').trim(),
                    collectionKey: historyCollectionStorageKey(parsed?.collectionKey || '__all__'),
                };
            }
        } catch (e) {
            /* DataTransfer 在部分浏览器只能在 drop 阶段读取。 */
        }
        return {
            groupKey: historyConfigGroupSortState.sourceKey || '',
            collectionKey: historyConfigGroupSortState.collectionKey || '__all__',
        };
    }

    function historyConfigGroupDropPosition(event, element) {
        const rect = element?.getBoundingClientRect?.();
        if (!rect) return 'after';
        return Number(event?.clientY || 0) < rect.top + (rect.height / 2) ? 'before' : 'after';
    }

    function removeHistoryConfigGroupDropPreview() {
        if (historyConfigGroupDropPreviewElement?.parentNode) {
            historyConfigGroupDropPreviewElement.parentNode.removeChild(historyConfigGroupDropPreviewElement);
        }
        historyConfigGroupDropPreviewElement = null;
    }

    function ensureHistoryConfigGroupDropPreview() {
        if (historyConfigGroupDropPreviewElement?.isConnected) return historyConfigGroupDropPreviewElement;
        const preview = document.createElement('div');
        preview.className = 'history-config-group-drop-preview';
        preview.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.textContent = '释放后插入到这里';
        preview.appendChild(label);
        historyConfigGroupDropPreviewElement = preview;
        return preview;
    }

    function placeHistoryConfigGroupDropPreview(element, position) {
        const parent = element?.parentElement;
        if (!parent) return;
        const preview = ensureHistoryConfigGroupDropPreview();
        const placement = position === 'before' ? 'before' : 'after';
        const parentStyle = window.getComputedStyle(parent);
        const gap = Number.parseFloat(parentStyle.rowGap || parentStyle.gap || '0') || 0;
        const top = placement === 'before'
            ? Math.max(0, element.offsetTop - (gap / 2))
            : element.offsetTop + element.offsetHeight + (gap / 2);
        preview.dataset.position = placement;
        preview.style.top = `${top}px`;
        if (preview.parentElement !== parent) parent.appendChild(preview);
    }

    function setHistoryConfigGroupSortTarget(targetKey, position, element) {
        historyConfigGroupSortState.activeDropTarget = `config-sort:${targetKey || ''}`;
        historyConfigGroupSortState.dropPosition = position === 'before' ? 'before' : 'after';
        document.querySelectorAll('.history-config-group-card.config-sort-active').forEach((item) => {
            if (item !== element) item.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
        });
        element?.classList.add(
            'config-sort-active',
            historyConfigGroupSortState.dropPosition === 'before' ? 'config-sort-before' : 'config-sort-after',
        );
        placeHistoryConfigGroupDropPreview(element, historyConfigGroupSortState.dropPosition);
    }

    function clearHistoryConfigGroupSortTarget(targetKey, element) {
        if (historyConfigGroupSortState.activeDropTarget === `config-sort:${targetKey || ''}`) {
            historyConfigGroupSortState.activeDropTarget = '';
            removeHistoryConfigGroupDropPreview();
        }
        element?.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
    }

    function clearHistoryConfigGroupSortIndicators() {
        historyConfigGroupSortState.activeDropTarget = '';
        removeHistoryConfigGroupDropPreview();
        document.querySelectorAll('.history-config-group-card.config-sort-active').forEach((item) => {
            item.classList.remove('config-sort-active', 'config-sort-before', 'config-sort-after');
        });
    }

    function historyConfigGroupOrderDragEnter(event, group, element, options = {}) {
        if (!historyConfigGroupSortState.active || historyConfigGroupSortState.pending) return false;
        const source = readHistoryDraggedConfigGroup(event);
        const targetKey = configGroupKey(group);
        const collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        if (!source.groupKey || !targetKey || source.groupKey === targetKey || source.collectionKey !== collectionKey) return false;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        const position = historyConfigGroupDropPosition(event, element);
        setHistoryConfigGroupSortTarget(targetKey, position, element);
        return true;
    }

    function historyConfigGroupOrderDragLeave(event, group, element) {
        if (!historyConfigGroupSortState.active) return false;
        if (element?.contains(event.relatedTarget)) return true;
        if (historyConfigGroupDropPreviewElement?.contains(event.relatedTarget)) return true;
        if (event.relatedTarget instanceof Element && event.relatedTarget.closest('.history-config-group-card-list')) return true;
        clearHistoryConfigGroupSortTarget(configGroupKey(group), element);
        return true;
    }

    function historyConfigGroupForPointerCard(card, groups = []) {
        if (!card) return null;
        const key = String(card.dataset.configGroupKey || '').trim();
        return (groups || []).find((group) => configGroupKey(group) === key) || null;
    }

    function historyConfigGroupPointerTargetForCard(card, x, y, groups = [], collection = null) {
        const group = historyConfigGroupForPointerCard(card, groups);
        if (!group) return null;
        return {
            element: card,
            group,
            key: configGroupKey(group),
            collectionKey: historyCollectionStorageKey(collection || '__all__'),
            position: historyConfigGroupDropPosition({ clientY: y }, card),
        };
    }

    function nearestHistoryConfigGroupPointerTarget(x, y, groups = [], collection = null) {
        let best = null;
        document.querySelectorAll('.history-config-group-card').forEach((card) => {
            if (!card?.isConnected) return;
            const rect = card.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
            const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
            const distance = Math.hypot(dx, dy);
            const maxDistance = Math.max(24, Math.min(76, rect.height * 0.9));
            if (distance > maxDistance || (best && distance >= best.distance)) return;
            const target = historyConfigGroupPointerTargetForCard(card, x, y, groups, collection);
            if (target) best = { ...target, distance };
        });
        if (!best) return null;
        const { distance, ...target } = best;
        return target;
    }

    function historyConfigGroupPointerTargetFromPoint(x, y, groups = [], collection = null) {
        const origin = document.elementFromPoint(x, y);
        const card = origin instanceof Element ? origin.closest('.history-config-group-card') : null;
        return historyConfigGroupPointerTargetForCard(card, x, y, groups, collection)
            || nearestHistoryConfigGroupPointerTarget(x, y, groups, collection);
    }

    function historyCollectionDropTargetFromPoint(x, y) {
        const origin = document.elementFromPoint(x, y);
        return origin instanceof Element ? origin.closest('.history-collection-card.nav-card') : null;
    }

    function cleanupHistoryConfigGroupPointerDrag() {
        const drag = historyConfigGroupPointerDrag;
        if (!drag) return null;
        document.removeEventListener('pointermove', drag.onMove);
        document.removeEventListener('pointerup', drag.onUp);
        document.removeEventListener('pointercancel', drag.onCancel);
        document.removeEventListener('mousemove', drag.onMouseMove);
        document.removeEventListener('mouseup', drag.onMouseUp);
        document.removeEventListener('touchmove', drag.onTouchMove);
        document.removeEventListener('touchend', drag.onTouchEnd);
        document.removeEventListener('touchcancel', drag.onTouchCancel);
        document.removeEventListener('keydown', drag.onKeydown);
        try {
            if (drag.pointerId !== null && drag.pointerId !== undefined) {
                drag.handle?.releasePointerCapture?.(drag.pointerId);
            }
        } catch (e) {
            /* 指针可能已被浏览器释放，忽略即可。 */
        }
        removeHistoryDragImage();
        drag.handle?.classList.remove('dragging');
        document.body.classList.remove('history-config-group-pointer-drag-active');
        historyConfigGroupPointerDrag = null;
        return drag;
    }

    async function finishHistoryConfigGroupPointerDrag(commit = false) {
        const drag = cleanupHistoryConfigGroupPointerDrag();
        if (!drag) return;
        const target = commit && drag.active ? drag.currentDrop : null;
        const collectionTarget = commit && drag.active ? drag.currentCollectionDrop : null;
        if (collectionTarget && drag.taskIds.length) {
            await dropHistoryTasksToCollectionLikePointer(collectionTarget, drag.taskIds);
            return;
        }
        if (!target || !drag.sourceKey) {
            if (drag.active) finishHistoryDrag();
            return;
        }
        if (target.key === drag.sourceKey) {
            setHistoryDropFeedback('配置分组顺序未变化。', 'ok');
            finishHistoryDrag();
            return;
        }
        historyConfigGroupSortState.pending = true;
        try {
            const changed = await reorderHistoryConfigGroupValue(
                drag.sourceKey,
                target.key,
                target.position,
                drag.groups,
                drag.collection,
            );
            setHistoryDropFeedback(changed ? '已调整配置分组顺序。' : '配置分组顺序未变化。', 'ok');
        } catch (e) {
            setHistoryDropFeedback(`调整配置分组顺序失败: ${e.message}`, 'error');
        } finally {
            finishHistoryDrag();
            renderHistoryManager();
        }
    }

    async function dropHistoryTasksToCollectionLikePointer(targetCard, taskIds) {
        const groupValue = String(targetCard.dataset.collectionValue || '').trim();
        const label = targetCard.querySelector('.history-collection-card-title strong')?.textContent || groupValue || '未分类';
        const clean = groupValue;
        if (!taskIds.length) {
            setHistoryDropFeedback('没有可移动的历史任务。', 'error');
            finishHistoryDrag();
            return;
        }
        if (historyDraggedTasksAlreadyInCollection(taskIds, clean)) {
            setHistoryDropFeedback(`已在${clean ? `分组「${clean}」` : '未分类'}中。`, 'ok');
            finishHistoryDrag();
            return;
        }
        historyDragState.pending = true;
        document.querySelector('.history-collections-workbench')?.classList.add('drop-pending');
        try {
            const res = await applyHistoryTaskIdsToCollection(taskIds, clean, { clearSelection: true });
            if (res === null) {
                setHistoryDropFeedback('移动失败，列表未更改。', 'error');
            } else {
                selectedHistoryCollectionKey = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;
                setHistoryDropFeedback(`${taskIds.length} 条任务已移动到${clean ? `「${label || clean}」` : '未分类'}。`, 'ok');
            }
        } catch (e) {
            setHistoryDropFeedback(`移动失败: ${e.message}`, 'error');
        } finally {
            historyDragState.pending = false;
            finishHistoryDrag();
            renderHistoryManager();
        }
    }

    function startHistoryConfigGroupPointerDrag(event, group, options = {}, handle = null, fallback = { pointer: true }) {
        const usePointer = fallback.pointer !== false && 'pointerId' in event;
        if (historyConfigGroupPointerDrag) return;
        if ((usePointer || fallback.mouse) && 'button' in event && event.button !== 0) return;
        if (usePointer && event.isPrimary === false) return;
        if (!canBeginHistoryConfigGroupDrag(group)) {
            event.preventDefault();
            event.stopPropagation();
            return;
        }
        const startPoint = historyCollectionEventPoint(event);
        if (!startPoint) return;
        event.stopPropagation();
        if (fallback.touch) event.preventDefault();
        const taskIds = uniqueStringList(historyDragTaskIdsForGroup(group));
        const sourceKey = configGroupKey(group);
        const collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        const dragHandle = handle || event.currentTarget;
        const pointerId = usePointer ? event.pointerId : null;
        const drag = {
            sourceKey,
            collectionKey,
            taskIds,
            groups: options.groups || [],
            collection: options.collection || null,
            handle: dragHandle,
            pointerId,
            startX: startPoint.x,
            startY: startPoint.y,
            active: false,
            image: null,
            currentDrop: null,
            currentCollectionDrop: null,
        };
        const moveDrag = (moveEvent) => {
            const point = historyCollectionEventPoint(moveEvent);
            if (!point) return;
            const distance = Math.hypot(point.x - drag.startX, point.y - drag.startY);
            if (!drag.active) {
                if (distance < 5) return;
                closeHistoryDropPopover(false);
                historyDragState = {
                    ...historyDragState,
                    active: true,
                    taskIds,
                    sourceGroupKey: sourceKey,
                    activeDropTarget: '',
                    popover: {
                        open: false,
                        x: 0,
                        y: 0,
                        taskIds: [],
                        defaultName: '',
                    },
                };
                historyConfigGroupSortState = {
                    active: Boolean(sourceKey),
                    sourceKey,
                    collectionKey,
                    activeDropTarget: '',
                    dropPosition: 'after',
                    pending: false,
                };
                drag.active = true;
                drag.image = createHistoryDragImage(taskIds.length);
                dragHandle?.classList.add('dragging');
                dragHandle?.closest('.history-config-group-card')?.classList.add('config-sort-source');
                document.body.classList.add('history-config-group-pointer-drag-active');
                document.querySelector('.history-collections-workbench')?.classList.add('dragging');
            }
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            moveHistoryCollectionPointerDragImage(drag.image, point.x, point.y);
            autoScrollHistoryCollectionPointerDrag(point.x, point.y);
            drag.currentCollectionDrop = historyCollectionDropTargetFromPoint(point.x, point.y);
            if (drag.currentCollectionDrop) {
                drag.currentDrop = null;
                setHistoryDropTarget(`collection:${drag.currentCollectionDrop.dataset.collectionValue || '__ungrouped__'}`, drag.currentCollectionDrop);
                clearHistoryConfigGroupSortIndicators();
                return;
            }
            clearHistoryDropIndicators();
            drag.currentDrop = historyConfigGroupPointerTargetFromPoint(point.x, point.y, drag.groups, drag.collection);
            if (drag.currentDrop && drag.currentDrop.key !== drag.sourceKey) {
                setHistoryConfigGroupSortTarget(drag.currentDrop.key, drag.currentDrop.position, drag.currentDrop.element);
            } else {
                drag.currentDrop = null;
                clearHistoryConfigGroupSortIndicators();
            }
        };
        drag.onMove = (moveEvent) => {
            if (moveEvent.pointerId !== pointerId) return;
            moveDrag(moveEvent);
        };
        drag.onUp = (upEvent) => {
            if (upEvent.pointerId !== pointerId) return;
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishHistoryConfigGroupPointerDrag(true);
        };
        drag.onCancel = (cancelEvent) => {
            if (cancelEvent.pointerId !== pointerId) return;
            finishHistoryConfigGroupPointerDrag(false);
        };
        drag.onMouseMove = (moveEvent) => moveDrag(moveEvent);
        drag.onMouseUp = (upEvent) => {
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishHistoryConfigGroupPointerDrag(true);
        };
        drag.onTouchMove = (moveEvent) => moveDrag(moveEvent);
        drag.onTouchEnd = (touchEvent) => {
            touchEvent.preventDefault();
            touchEvent.stopPropagation();
            finishHistoryConfigGroupPointerDrag(true);
        };
        drag.onTouchCancel = () => finishHistoryConfigGroupPointerDrag(false);
        drag.onKeydown = (keyEvent) => {
            if (keyEvent.key === 'Escape') finishHistoryConfigGroupPointerDrag(false);
        };
        historyConfigGroupPointerDrag = drag;
        if (usePointer) {
            try {
                dragHandle?.setPointerCapture?.(pointerId);
            } catch (e) {
                /* 某些浏览器会让原生拖拽抢占捕获，文档级监听仍作为兜底。 */
            }
            document.addEventListener('pointermove', drag.onMove, { passive: false });
            document.addEventListener('pointerup', drag.onUp, { passive: false });
            document.addEventListener('pointercancel', drag.onCancel, { passive: false });
        } else if (fallback.touch) {
            document.addEventListener('touchmove', drag.onTouchMove, { passive: false });
            document.addEventListener('touchend', drag.onTouchEnd, { passive: false });
            document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false });
        } else {
            document.addEventListener('mousemove', drag.onMouseMove, { passive: false });
            document.addEventListener('mouseup', drag.onMouseUp, { passive: false });
        }
        document.addEventListener('keydown', drag.onKeydown);
    }

    function startHistoryConfigGroupMouseDrag(event, group, options = {}, handle = null) {
        startHistoryConfigGroupPointerDrag(event, group, options, handle, { pointer: false, mouse: true });
    }

    function startHistoryConfigGroupTouchDrag(event, group, options = {}, handle = null) {
        startHistoryConfigGroupPointerDrag(event, group, options, handle, { pointer: false, touch: true });
    }

    async function reorderHistoryConfigGroupValue(sourceKey, targetKey, position, groups = [], collection = null) {
        const source = String(sourceKey || '').trim();
        const target = String(targetKey || '').trim();
        if (!source || !target) return false;
        const collectionKey = historyCollectionStorageKey(collection || '__all__');
        const currentOrder = configGroupOrderValues(groups, collection);
        const nextOrder = moveItemNearList(currentOrder, source, target, position);
        if (nextOrder.length === currentOrder.length && nextOrder.every((value, idx) => value === currentOrder[idx])) {
            return false;
        }
        await saveHistoryCollectionSettings({
            ...historyCollectionSettings,
            config_group_order: {
                ...(historyCollectionSettings.config_group_order || {}),
                [collectionKey]: nextOrder,
            },
        });
        return true;
    }

    async function dropHistoryConfigGroupToSort(event, targetGroup, options = {}) {
        if (!historyConfigGroupSortState.active) return false;
        const source = readHistoryDraggedConfigGroup(event);
        const targetKey = configGroupKey(targetGroup);
        const collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        if (!source.groupKey || !targetKey || source.collectionKey !== collectionKey) return false;
        event.preventDefault();
        event.stopPropagation();
        const position = historyConfigGroupSortState.dropPosition || historyConfigGroupDropPosition(event, event.currentTarget);
        clearHistoryConfigGroupSortTarget(targetKey, event.currentTarget);
        if (source.groupKey === targetKey) {
            setHistoryDropFeedback('配置分组顺序未变化。', 'ok');
            finishHistoryDrag();
            return true;
        }
        historyConfigGroupSortState.pending = true;
        try {
            const changed = await reorderHistoryConfigGroupValue(
                source.groupKey,
                targetKey,
                position,
                options.groups || [],
                options.collection || null,
            );
            setHistoryDropFeedback(changed ? '已调整配置分组顺序。' : '配置分组顺序未变化。', 'ok');
        } catch (e) {
            setHistoryDropFeedback(`调整配置分组顺序失败: ${e.message}`, 'error');
        } finally {
            finishHistoryDrag();
            renderHistoryManager();
        }
        return true;
    }

    function readHistoryDraggedTaskIds(event) {
        const fallback = historyDragState.taskIds || [];
        const transfer = event?.dataTransfer;
        const sources = [];
        try {
            sources.push(transfer?.getData(HISTORY_TASK_DRAG_MIME));
            sources.push(transfer?.getData('text/plain'));
        } catch (e) {
            /* 某些浏览器只允许在 drop 事件中读取 DataTransfer。 */
        }
        for (const raw of sources) {
            if (!raw) continue;
            try {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed)) return uniqueStringList(parsed);
            } catch (e) {
                const text = String(raw || '').trim();
                if (text) return uniqueStringList([text]);
            }
        }
        return uniqueStringList(fallback);
    }

    function setHistoryDropTarget(id, element) {
        if (!historyDragState.active || historyDragState.pending) return;
        historyDragState.activeDropTarget = id;
        document.querySelectorAll('.history-collection-card.drop-active').forEach((item) => {
            if (item !== element) item.classList.remove('drop-active');
        });
        element?.classList.add('drop-active');
    }

    function clearHistoryDropTarget(id, element) {
        if (historyDragState.activeDropTarget === id) {
            historyDragState.activeDropTarget = '';
        }
        element?.classList.remove('drop-active');
    }

    function clearHistoryDropIndicators() {
        historyDragState.activeDropTarget = '';
        document.querySelectorAll('.history-collection-card.drop-active').forEach((item) => {
            item.classList.remove('drop-active');
        });
    }

    function historyTasksByIds(ids) {
        const taskMap = new Map(historyTasks.map((task) => [task.id, task]));
        return uniqueStringList(ids).map((id) => taskMap.get(id)).filter(Boolean);
    }

    function historyDraggedTasksAlreadyInCollection(ids, groupValue) {
        const clean = String(groupValue || '').trim();
        const taskIds = uniqueStringList(ids);
        const tasks = historyTasksByIds(taskIds);
        return tasks.length === taskIds.length && tasks.every((task) => historyTaskCollectionValue(task) === clean);
    }

    function historyDropTargetDragEnter(event, targetId, element) {
        if (!historyDragState.active || historyDragState.pending) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        setHistoryDropTarget(targetId, element);
    }

    function historyDropTargetDragLeave(event, targetId, element) {
        if (element?.contains(event.relatedTarget)) return;
        clearHistoryDropTarget(targetId, element);
    }

    function canBeginHistoryCollectionSort(collection) {
        const value = String(collection?.value || '').trim();
        return Boolean(value && !collection?.is_ungrouped && !historyCollectionDragState.pending && !historyDragState.pending);
    }

    function beginHistoryCollectionDrag(event, collection) {
        const value = String(collection?.value || '').trim();
        if (!canBeginHistoryCollectionSort(collection)) {
            event.preventDefault();
            return;
        }
        closeHistoryDropPopover(false);
        finishHistoryDrag();
        historyCollectionDragState = {
            active: true,
            sourceValue: value,
            activeDropTarget: '',
            dropPosition: 'after',
            pending: false,
        };
        if (event.dataTransfer) {
            event.dataTransfer.setData(HISTORY_COLLECTION_DRAG_MIME, value);
            event.dataTransfer.setData('text/plain', JSON.stringify({ type: 'history-collection', value }));
            event.dataTransfer.effectAllowed = 'move';
        }
        event.currentTarget?.closest('.history-collection-card')?.classList.add('sort-source');
        document.querySelector('.history-collections-workbench')?.classList.add('collection-reordering');
    }

    function finishHistoryCollectionDrag() {
        historyCollectionDragState = {
            active: false,
            sourceValue: '',
            activeDropTarget: '',
            dropPosition: 'after',
            pending: false,
        };
        document.querySelectorAll('.history-collection-card.sort-active, .history-collection-card.sort-source').forEach((item) => {
            item.classList.remove('sort-active', 'sort-before', 'sort-after', 'sort-source');
        });
        document.querySelector('.history-collections-workbench')?.classList.remove('collection-reordering');
    }

    function clearHistoryCollectionSortIndicators() {
        historyCollectionDragState.activeDropTarget = '';
        document.querySelectorAll('.history-collection-card.sort-active').forEach((item) => {
            item.classList.remove('sort-active', 'sort-before', 'sort-after');
        });
    }

    function createHistoryCollectionPointerDragImage(label) {
        removeHistoryDragImage();
        const image = document.createElement('div');
        image.className = 'history-drag-image history-collection-drag-image-pointer';
        image.textContent = label || '历史分组';
        document.body.appendChild(image);
        historyDragImageElement = image;
        return image;
    }

    function moveHistoryCollectionPointerDragImage(image, x, y) {
        if (!image) return;
        image.style.left = `${x + 14}px`;
        image.style.top = `${y + 14}px`;
    }

    function historyCollectionForPointerCard(card, allCollections = []) {
        if (!card) return null;
        const key = String(card.dataset.collectionKey || '').trim();
        const value = String(card.dataset.collectionValue || '').trim();
        return (allCollections || []).find((collection) => collection.key === key)
            || (allCollections || []).find((collection) => String(collection.value || '').trim() === value)
            || null;
    }

    function historyCollectionPointerTargetForCard(card, x, y, allCollections = []) {
        const collection = historyCollectionForPointerCard(card, allCollections);
        if (!collection) return null;
        return {
            element: card,
            collection,
            value: String(collection.value || '').trim(),
            position: historyCollectionDropPosition({ clientY: y }, card, collection),
        };
    }

    function nearestHistoryCollectionPointerTarget(x, y, allCollections = []) {
        let best = null;
        document.querySelectorAll('.history-collection-card.nav-card').forEach((card) => {
            if (!card?.isConnected) return;
            const rect = card.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0;
            const dy = y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
            const distance = Math.hypot(dx, dy);
            const maxDistance = Math.max(22, Math.min(72, rect.height * 0.9));
            if (distance > maxDistance || (best && distance >= best.distance)) return;
            const target = historyCollectionPointerTargetForCard(card, x, y, allCollections);
            if (target) best = { ...target, distance };
        });
        if (!best) return null;
        const { distance, ...target } = best;
        return target;
    }

    function historyCollectionPointerTargetFromPoint(x, y, allCollections = []) {
        const origin = document.elementFromPoint(x, y);
        const card = origin instanceof Element ? origin.closest('.history-collection-card.nav-card') : null;
        return historyCollectionPointerTargetForCard(card, x, y, allCollections)
            || nearestHistoryCollectionPointerTarget(x, y, allCollections);
    }

    function findHistoryCollectionPointerScroller(origin) {
        let node = origin instanceof Element ? origin : null;
        while (node && node !== document.body) {
            const style = window.getComputedStyle(node);
            if (/(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight) {
                return node;
            }
            node = node.parentElement;
        }
        return document.scrollingElement;
    }

    function autoScrollHistoryCollectionPointerDrag(x, y) {
        const origin = document.elementFromPoint(x, y);
        const scroller = findHistoryCollectionPointerScroller(origin);
        if (!scroller) return;
        const rect = scroller === document.scrollingElement
            ? { top: 0, bottom: window.innerHeight }
            : scroller.getBoundingClientRect();
        const margin = 42;
        let delta = 0;
        if (y < rect.top + margin) delta = -14;
        else if (y > rect.bottom - margin) delta = 14;
        if (delta) scroller.scrollBy({ top: delta, behavior: 'auto' });
    }

    function cleanupHistoryCollectionPointerDrag() {
        const drag = historyCollectionPointerDrag;
        if (!drag) return null;
        document.removeEventListener('pointermove', drag.onMove);
        document.removeEventListener('pointerup', drag.onUp);
        document.removeEventListener('pointercancel', drag.onCancel);
        document.removeEventListener('mousemove', drag.onMouseMove);
        document.removeEventListener('mouseup', drag.onMouseUp);
        document.removeEventListener('touchmove', drag.onTouchMove);
        document.removeEventListener('touchend', drag.onTouchEnd);
        document.removeEventListener('touchcancel', drag.onTouchCancel);
        document.removeEventListener('keydown', drag.onKeydown);
        try {
            if (drag.pointerId !== null && drag.pointerId !== undefined) {
                drag.handle?.releasePointerCapture?.(drag.pointerId);
            }
        } catch (e) {
            /* 指针可能已被浏览器释放，忽略即可。 */
        }
        removeHistoryDragImage();
        drag.handle.classList.remove('dragging');
        document.body.classList.remove('history-collection-pointer-drag-active');
        historyCollectionPointerDrag = null;
        return drag;
    }

    function historyCollectionEventPoint(event) {
        const touch = event.changedTouches?.[0] || event.touches?.[0];
        const x = touch?.clientX ?? event.clientX;
        const y = touch?.clientY ?? event.clientY;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        return { x, y };
    }

    async function finishHistoryCollectionPointerDrag(commit = false) {
        const drag = cleanupHistoryCollectionPointerDrag();
        if (!drag) return;
        const source = drag.sourceValue;
        const target = commit && drag.active ? drag.currentDrop : null;
        if (!target || !source) {
            if (drag.active) finishHistoryCollectionDrag();
            return;
        }
        if (source === target.value) {
            setHistoryDropFeedback('分组顺序未变化。', 'ok');
            finishHistoryCollectionDrag();
            return;
        }
        historyCollectionDragState.pending = true;
        try {
            const changed = await reorderHistoryCollectionValue(source, target.value, target.position, drag.allCollections);
            setHistoryDropFeedback(changed ? `已调整「${source}」的位置。` : '分组顺序未变化。', 'ok');
        } catch (e) {
            setHistoryDropFeedback(`调整分组顺序失败: ${e.message}`, 'error');
        } finally {
            finishHistoryCollectionDrag();
            renderHistoryManager();
        }
    }

    function startHistoryCollectionPointerDrag(event, collection, allCollections = [], handle = null, options = { pointer: true }) {
        const usePointer = options.pointer !== false && 'pointerId' in event;
        if (historyCollectionPointerDrag) return;
        if ((usePointer || options.mouse) && 'button' in event && event.button !== 0) return;
        if (usePointer && event.isPrimary === false) return;
        if (!canBeginHistoryCollectionSort(collection)) {
            event.preventDefault();
            event.stopPropagation();
            return;
        }
        const startPoint = historyCollectionEventPoint(event);
        if (!startPoint) return;
        event.preventDefault();
        event.stopPropagation();
        const sourceValue = String(collection?.value || '').trim();
        const pointerId = usePointer ? event.pointerId : null;
        const dragHandle = handle || event.currentTarget;
        const drag = {
            sourceValue,
            allCollections,
            handle: dragHandle,
            pointerId,
            startX: startPoint.x,
            startY: startPoint.y,
            active: false,
            image: null,
            currentDrop: null,
        };
        const moveDrag = (moveEvent) => {
            const point = historyCollectionEventPoint(moveEvent);
            if (!point) return;
            const distance = Math.hypot(point.x - drag.startX, point.y - drag.startY);
            if (!drag.active) {
                if (distance < 5) return;
                closeHistoryDropPopover(false);
                finishHistoryDrag();
                historyCollectionDragState = {
                    active: true,
                    sourceValue,
                    activeDropTarget: '',
                    dropPosition: 'after',
                    pending: false,
                };
                drag.active = true;
                drag.image = createHistoryCollectionPointerDragImage(collection.label || sourceValue);
                dragHandle?.closest('.history-collection-card')?.classList.add('sort-source');
                dragHandle?.classList.add('dragging');
                document.body.classList.add('history-collection-pointer-drag-active');
                document.querySelector('.history-collections-workbench')?.classList.add('collection-reordering');
            }
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            moveHistoryCollectionPointerDragImage(drag.image, point.x, point.y);
            autoScrollHistoryCollectionPointerDrag(point.x, point.y);
            drag.currentDrop = historyCollectionPointerTargetFromPoint(point.x, point.y, allCollections);
            if (drag.currentDrop) {
                setHistoryCollectionSortTarget(drag.currentDrop.value, drag.currentDrop.position, drag.currentDrop.element);
            } else {
                clearHistoryCollectionSortIndicators();
            }
        };
        drag.onMove = (moveEvent) => {
            if (moveEvent.pointerId !== pointerId) return;
            moveDrag(moveEvent);
        };
        drag.onUp = (upEvent) => {
            if (upEvent.pointerId !== pointerId) return;
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishHistoryCollectionPointerDrag(true);
        };
        drag.onCancel = (cancelEvent) => {
            if (cancelEvent.pointerId !== pointerId) return;
            finishHistoryCollectionPointerDrag(false);
        };
        drag.onMouseMove = (moveEvent) => moveDrag(moveEvent);
        drag.onMouseUp = (upEvent) => {
            upEvent.preventDefault();
            upEvent.stopPropagation();
            finishHistoryCollectionPointerDrag(true);
        };
        drag.onTouchMove = (moveEvent) => moveDrag(moveEvent);
        drag.onTouchEnd = (touchEvent) => {
            touchEvent.preventDefault();
            touchEvent.stopPropagation();
            finishHistoryCollectionPointerDrag(true);
        };
        drag.onTouchCancel = () => finishHistoryCollectionPointerDrag(false);
        drag.onKeydown = (keyEvent) => {
            if (keyEvent.key === 'Escape') finishHistoryCollectionPointerDrag(false);
        };
        historyCollectionPointerDrag = drag;
        if (usePointer) {
            try {
                dragHandle?.setPointerCapture?.(pointerId);
            } catch (e) {
                /* 某些浏览器会让原生拖拽抢占捕获，文档级监听仍作为兜底。 */
            }
            document.addEventListener('pointermove', drag.onMove, { passive: false });
            document.addEventListener('pointerup', drag.onUp, { passive: false });
            document.addEventListener('pointercancel', drag.onCancel, { passive: false });
        } else if (options.touch) {
            document.addEventListener('touchmove', drag.onTouchMove, { passive: false });
            document.addEventListener('touchend', drag.onTouchEnd, { passive: false });
            document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false });
        } else {
            document.addEventListener('mousemove', drag.onMouseMove, { passive: false });
            document.addEventListener('mouseup', drag.onMouseUp, { passive: false });
        }
        document.addEventListener('keydown', drag.onKeydown);
    }

    function startHistoryCollectionMouseDrag(event, collection, allCollections = [], handle = null) {
        startHistoryCollectionPointerDrag(event, collection, allCollections, handle, { pointer: false, mouse: true });
    }

    function startHistoryCollectionTouchDrag(event, collection, allCollections = [], handle = null) {
        startHistoryCollectionPointerDrag(event, collection, allCollections, handle, { pointer: false, touch: true });
    }

    function readHistoryDraggedCollectionValue(event) {
        const fallback = historyCollectionDragState.sourceValue || '';
        try {
            const direct = event?.dataTransfer?.getData(HISTORY_COLLECTION_DRAG_MIME);
            if (direct) return String(direct || '').trim();
        } catch (e) {
            /* 某些浏览器只允许在 drop 事件中读取 DataTransfer。 */
        }
        return String(fallback || '').trim();
    }

    function historyCollectionDropPosition(event, element, collection) {
        if (collection?.is_ungrouped) return 'after';
        const rect = element?.getBoundingClientRect?.();
        if (!rect) return 'after';
        return Number(event?.clientY || 0) < rect.top + (rect.height / 2) ? 'before' : 'after';
    }

    function setHistoryCollectionSortTarget(targetValue, position, element) {
        historyCollectionDragState.activeDropTarget = `collection-sort:${targetValue || '__ungrouped__'}`;
        historyCollectionDragState.dropPosition = position === 'before' ? 'before' : 'after';
        document.querySelectorAll('.history-collection-card.sort-active').forEach((item) => {
            if (item !== element) item.classList.remove('sort-active', 'sort-before', 'sort-after');
        });
        element?.classList.add('sort-active', historyCollectionDragState.dropPosition === 'before' ? 'sort-before' : 'sort-after');
    }

    function clearHistoryCollectionSortTarget(targetValue, element) {
        if (historyCollectionDragState.activeDropTarget === `collection-sort:${targetValue || '__ungrouped__'}`) {
            historyCollectionDragState.activeDropTarget = '';
        }
        element?.classList.remove('sort-active', 'sort-before', 'sort-after');
    }

    function historyCollectionOrderDragEnter(event, collection, element) {
        if (!historyCollectionDragState.active || historyCollectionDragState.pending) return false;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        const targetValue = String(collection?.value || '').trim();
        const position = historyCollectionDropPosition(event, element, collection);
        setHistoryCollectionSortTarget(targetValue, position, element);
        return true;
    }

    function historyCollectionOrderDragLeave(event, collection, element) {
        if (!historyCollectionDragState.active) return false;
        if (element?.contains(event.relatedTarget)) return true;
        clearHistoryCollectionSortTarget(collection?.value || '', element);
        return true;
    }

    function moveItemNearList(list, sourceValue, targetValue, position = 'after') {
        const out = uniqueStringList(list);
        const source = String(sourceValue || '').trim();
        const target = String(targetValue || '').trim();
        if (!source || !out.includes(source)) return out;
        const original = [...out];
        out.splice(out.indexOf(source), 1);
        let index = 0;
        if (target) {
            const targetIndex = out.indexOf(target);
            index = targetIndex < 0 ? out.length : targetIndex + (position === 'after' ? 1 : 0);
        }
        out.splice(Math.max(0, Math.min(out.length, index)), 0, source);
        return out.length === original.length && out.every((value, idx) => value === original[idx]) ? original : out;
    }

    async function reorderHistoryCollectionValue(sourceValue, targetValue, position, allCollections = null) {
        const source = String(sourceValue || '').trim();
        if (!source) return false;
        const collections = allCollections || historyCollectionsForWorkbench(historyTasks);
        const currentOrder = collectionOrderValues(collections);
        const nextOrder = moveItemNearList(currentOrder, source, targetValue, position);
        if (nextOrder.length === currentOrder.length && nextOrder.every((value, idx) => value === currentOrder[idx])) {
            return false;
        }
        await saveHistoryCollectionSettings({
            ...historyCollectionSettings,
            collection_order: nextOrder,
        });
        return true;
    }

    async function dropHistoryCollectionToSort(event, targetCollection, allCollections = []) {
        if (!historyCollectionDragState.active) return false;
        event.preventDefault();
        event.stopPropagation();
        const source = readHistoryDraggedCollectionValue(event);
        const target = String(targetCollection?.value || '').trim();
        const position = historyCollectionDragState.dropPosition || historyCollectionDropPosition(event, event.currentTarget, targetCollection);
        clearHistoryCollectionSortTarget(target, event.currentTarget);
        if (!source) {
            setHistoryDropFeedback('没有可排序的分组。', 'error');
            finishHistoryCollectionDrag();
            return true;
        }
        if (source === target) {
            setHistoryDropFeedback('分组顺序未变化。', 'ok');
            finishHistoryCollectionDrag();
            return true;
        }
        historyCollectionDragState.pending = true;
        try {
            const changed = await reorderHistoryCollectionValue(source, target, position, allCollections);
            setHistoryDropFeedback(changed ? `已调整「${source}」的位置。` : '分组顺序未变化。', 'ok');
        } catch (e) {
            setHistoryDropFeedback(`调整分组顺序失败: ${e.message}`, 'error');
        } finally {
            finishHistoryCollectionDrag();
            renderHistoryManager();
        }
        return true;
    }

    async function dropHistoryTasksToCollection(event, groupValue, label) {
        event.preventDefault();
        event.stopPropagation();
        const taskIds = readHistoryDraggedTaskIds(event);
        const clean = String(groupValue || '').trim();
        clearHistoryDropTarget(historyDragState.activeDropTarget, event.currentTarget);
        if (!taskIds.length) {
            setHistoryDropFeedback('没有可移动的历史任务。', 'error');
            finishHistoryDrag();
            return;
        }
        if (historyDraggedTasksAlreadyInCollection(taskIds, clean)) {
            setHistoryDropFeedback(`已在${clean ? `分组「${clean}」` : '未分类'}中。`, 'ok');
            finishHistoryDrag();
            return;
        }
        historyDragState.pending = true;
        document.querySelector('.history-collections-workbench')?.classList.add('drop-pending');
        try {
            const res = await applyHistoryTaskIdsToCollection(taskIds, clean, { clearSelection: true });
            if (res === null) {
                setHistoryDropFeedback('移动失败，列表未更改。', 'error');
            } else {
                selectedHistoryCollectionKey = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;
                setHistoryDropFeedback(`${taskIds.length} 条任务已移动到${clean ? `「${label || clean}」` : '未分类'}。`, 'ok');
            }
        } catch (e) {
            setHistoryDropFeedback(`移动失败: ${e.message}`, 'error');
        } finally {
            historyDragState.pending = false;
            finishHistoryDrag();
            renderHistoryManager();
        }
    }

    function defaultHistoryCollectionName() {
        const now = new Date();
        const yyyy = String(now.getFullYear());
        const mm = String(now.getMonth() + 1).padStart(2, '0');
        const dd = String(now.getDate()).padStart(2, '0');
        return uniqueHistoryCollectionName(`未分配_${yyyy}${mm}${dd}`);
    }

    function uniqueHistoryCollectionName(base) {
        const cleanBase = String(base || '').trim().slice(0, 48) || '未分配';
        const existing = new Set([
            ...historyCollectionSelectOptions().map((item) => String(item.value || '').trim()).filter(Boolean),
            ...uniqueStringList(historyCollectionSettings.collection_order || []),
        ]);
        if (!existing.has(cleanBase)) return cleanBase;
        for (let index = 2; index < 1000; index += 1) {
            const suffix = `_${index}`;
            const candidate = `${cleanBase.slice(0, Math.max(1, 48 - suffix.length))}${suffix}`;
            if (!existing.has(candidate)) return candidate;
        }
        return cleanBase;
    }

    function openHistoryNewCollectionPopover(event = null, taskIds = []) {
        const rect = event?.currentTarget?.getBoundingClientRect?.();
        const x = Number(event?.clientX || 0) || (rect ? Math.round(rect.left + rect.width / 2) : Math.round(window.innerWidth / 2));
        const y = Number(event?.clientY || 0) || (rect ? Math.round(rect.bottom + 8) : Math.round(window.innerHeight / 2));
        historyDragState.active = false;
        historyDragState.activeDropTarget = '';
        historyDragState.popover = {
            open: true,
            x,
            y,
            taskIds: uniqueStringList(taskIds),
            defaultName: defaultHistoryCollectionName(),
        };
        finishHistoryDrag();
        renderHistoryManager();
    }

    function renderHistoryDropPopover(workbench) {
        if (historyDropPopoverOutsideHandler) {
            document.removeEventListener('mousedown', historyDropPopoverOutsideHandler);
            historyDropPopoverOutsideHandler = null;
        }
        const state = historyDragState.popover;
        if (!state.open) return;

        const popover = document.createElement('form');
        popover.className = 'history-drop-popover';
        popover.noValidate = true;
        const width = 320;
        const height = 152;
        const left = Math.min(Math.max(8, state.x), Math.max(8, window.innerWidth - width - 8));
        const top = Math.min(Math.max(8, state.y), Math.max(8, window.innerHeight - height - 8));
        popover.style.left = `${left}px`;
        popover.style.top = `${top}px`;

        const label = document.createElement('label');
        const title = document.createElement('span');
        title.textContent = state.taskIds.length ? `${state.taskIds.length} 条任务归入新分组` : '新建分组';
        const input = document.createElement('input');
        input.type = 'text';
        input.maxLength = 48;
        input.value = state.defaultName || defaultHistoryCollectionName();
        input.placeholder = '分组名称';
        input.disabled = historyDragState.pending;
        label.append(title, input);

        const actions = document.createElement('div');
        actions.className = 'history-drop-popover-actions';
        const cancel = document.createElement('button');
        cancel.type = 'button';
        cancel.className = 'task-history-action';
        cancel.textContent = '取消';
        cancel.disabled = historyDragState.pending;
        cancel.addEventListener('click', () => closeHistoryDropPopover());
        const submit = document.createElement('button');
        submit.type = 'submit';
        submit.className = 'task-history-action primary';
        submit.textContent = historyDragState.pending
            ? '保存中...'
            : (state.taskIds.length ? '新建并移动' : '新建分组');
        submit.disabled = historyDragState.pending || !input.value.trim();
        actions.append(cancel, submit);

        const updateSubmitState = () => {
            submit.disabled = historyDragState.pending || !input.value.trim();
        };
        input.addEventListener('input', updateSubmitState);
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                event.preventDefault();
                closeHistoryDropPopover();
            }
            if (event.key === 'Enter') {
                event.preventDefault();
                if (!submit.disabled) submitHistoryDropPopover(input.value);
            }
        });
        popover.addEventListener('submit', (event) => {
            event.preventDefault();
            if (!submit.disabled) submitHistoryDropPopover(input.value);
        });
        popover.append(label, actions);
        workbench.appendChild(popover);

        requestAnimationFrame(() => {
            input.focus();
            input.select();
        });
        const outsideHandler = (event) => {
            if (!popover.contains(event.target)) closeHistoryDropPopover();
        };
        historyDropPopoverOutsideHandler = outsideHandler;
        setTimeout(() => {
            if (historyDropPopoverOutsideHandler === outsideHandler) {
                document.addEventListener('mousedown', outsideHandler);
            }
        }, 0);
    }

    function closeHistoryDropPopover(render = true) {
        if (historyDropPopoverOutsideHandler) {
            document.removeEventListener('mousedown', historyDropPopoverOutsideHandler);
            historyDropPopoverOutsideHandler = null;
        }
        document.querySelectorAll('.history-drop-popover').forEach((popover) => popover.remove());
        historyDragState.pending = false;
        historyDragState.popover = {
            open: false,
            x: 0,
            y: 0,
            taskIds: [],
            defaultName: '',
        };
        if (render) renderHistoryManager();
    }

    async function submitHistoryDropPopover(name) {
        const clean = String(name || '').trim().slice(0, 48);
        const taskIds = uniqueStringList(historyDragState.popover.taskIds);
        if (!clean || historyDragState.pending) return;
        historyDragState.popover.defaultName = clean;
        historyDragState.pending = true;
        renderHistoryManager();
        try {
            if (taskIds.length) {
                const res = await applyHistoryTaskIdsToCollection(taskIds, clean, { clearSelection: true });
                if (res === null) {
                    historyDragState.pending = false;
                    setHistoryDropFeedback('新建分组失败，列表未更改。', 'error');
                    renderHistoryManager();
                    return;
                }
                setHistoryDropFeedback(`${taskIds.length} 条任务已移动到「${clean}」。`, 'ok');
            } else {
                await ensureHistoryCollectionOrderValue(clean);
                setHistoryDropFeedback(`已新建分组「${clean}」。`, 'ok');
            }
            selectedHistoryCollectionKey = `collection:${clean}`;
            closeHistoryDropPopover(false);
        } catch (e) {
            historyDragState.pending = false;
            setHistoryDropFeedback(`新建分组失败: ${e.message}`, 'error');
        } finally {
            renderHistoryManager();
        }
    }

    function setHistoryDropFeedback(message, tone = '') {
        historyDropFeedback = { message: String(message || ''), tone: String(tone || '') };
        if (historyDropFeedbackTimer) clearTimeout(historyDropFeedbackTimer);
        const status = document.getElementById('history-manager-status');
        if (status && historyDropFeedback.message) {
            const visible = historyManagerFilteredTasks();
            const archivedCount = historyTasks.filter(historyTaskIsArchived).length;
            status.textContent = [
                `共 ${historyTasks.length} 条记录`,
                `当前分组 ${visible.length} 条`,
                `归档 ${archivedCount} 条`,
                historyDropFeedback.message,
            ].filter(Boolean).join(' · ');
            status.dataset.feedbackTone = historyDropFeedback.tone;
        }
        historyDropFeedbackTimer = setTimeout(() => {
            historyDropFeedback = { message: '', tone: '' };
            historyDropFeedbackTimer = null;
            if (trainingViewMode === 'history') renderHistoryManager();
        }, 2600);
    }

    function createHistoryCollectionWorkbenchCard(collection, selectedTaskCount = 0, allCollections = []) {
        const card = document.createElement('article');
        card.className = ['history-collection-card', 'nav-card', collection.is_ungrouped ? 'ungrouped' : ''].filter(Boolean).join(' ');
        const dropTargetId = `collection:${collection.value || '__ungrouped__'}`;
        const canSortCollection = !collection.is_ungrouped && Boolean(collection.value);
        card.dataset.collectionKey = collection.key || '';
        card.dataset.collectionValue = collection.value || '';
        if (canSortCollection) {
            card.classList.add('sortable');
        }
        if (selectedHistoryCollectionKey === collection.key) {
            card.classList.add('active');
        }
        if (historyCollectionWorkbenchTarget && collection.value === historyCollectionWorkbenchTarget) {
            card.classList.add('target');
        }
        if (historyDragState.activeDropTarget === dropTargetId) {
            card.classList.add('drop-active');
        }
        if (historyCollectionDragState.sourceValue && historyCollectionDragState.sourceValue === collection.value) {
            card.classList.add('sort-source');
        }
        if (
            historyCollectionDragState.active &&
            historyCollectionDragState.activeDropTarget === `collection-sort:${collection.value || '__ungrouped__'}`
        ) {
            card.classList.add('sort-active', historyCollectionDragState.dropPosition === 'before' ? 'sort-before' : 'sort-after');
        }
        card.tabIndex = 0;
        card.setAttribute('role', 'button');
        card.setAttribute('aria-pressed', selectedHistoryCollectionKey === collection.key ? 'true' : 'false');
        card.addEventListener('dragenter', (event) => {
            if (historyCollectionOrderDragEnter(event, collection, card)) return;
            historyDropTargetDragEnter(event, dropTargetId, card);
        });
        card.addEventListener('dragover', (event) => {
            if (historyCollectionOrderDragEnter(event, collection, card)) return;
            historyDropTargetDragEnter(event, dropTargetId, card);
        });
        card.addEventListener('dragleave', (event) => {
            if (historyCollectionOrderDragLeave(event, collection, card)) return;
            historyDropTargetDragLeave(event, dropTargetId, card);
        });
        card.addEventListener('drop', async (event) => {
            if (await dropHistoryCollectionToSort(event, collection, allCollections)) return;
            dropHistoryTasksToCollection(event, collection.value || '', collection.label);
        });
        card.addEventListener('click', () => {
            selectedHistoryCollectionKey = collection.key;
            renderHistoryManager();
        });
        card.addEventListener('keydown', (event) => {
            if (event.target !== card) return;
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            selectedHistoryCollectionKey = collection.key;
            renderHistoryManager();
        });

        const head = document.createElement('div');
        head.className = 'history-collection-card-head';
        const title = document.createElement('div');
        title.className = 'history-collection-card-title';
        const titleRow = document.createElement('div');
        titleRow.className = 'history-collection-card-title-row';
        const passiveHandle = document.createElement('span');
        passiveHandle.className = 'history-drag-handle history-drag-handle-passive';
        passiveHandle.textContent = '⋮⋮';
        passiveHandle.title = collection.is_ungrouped ? '未分类分组不可排序' : '拖拽分组调整顺序';
        passiveHandle.setAttribute('aria-hidden', 'true');
        if (canSortCollection) {
            const dragHandle = document.createElement('button');
            dragHandle.type = 'button';
            dragHandle.className = 'history-drag-handle history-collection-drag-handle';
            dragHandle.textContent = '⋮⋮';
            dragHandle.title = '拖拽分组调整顺序';
            dragHandle.setAttribute('aria-label', '拖拽分组调整顺序');
            dragHandle.draggable = true;
            dragHandle.addEventListener('click', (event) => event.stopPropagation());
            dragHandle.addEventListener('pointerdown', (event) => startHistoryCollectionPointerDrag(event, collection, allCollections, dragHandle));
            dragHandle.addEventListener('mousedown', (event) => {
                event.stopPropagation();
                if (!('PointerEvent' in window)) startHistoryCollectionMouseDrag(event, collection, allCollections, dragHandle);
            });
            dragHandle.addEventListener('touchstart', (event) => {
                event.stopPropagation();
                if (!('PointerEvent' in window)) startHistoryCollectionTouchDrag(event, collection, allCollections, dragHandle);
            }, { passive: false });
            dragHandle.addEventListener('dragstart', (event) => {
                if (historyCollectionPointerDrag) finishHistoryCollectionPointerDrag(false);
                beginHistoryCollectionDrag(event, collection);
            });
            dragHandle.addEventListener('dragend', () => finishHistoryCollectionDrag());
            titleRow.appendChild(dragHandle);
        } else {
            titleRow.appendChild(passiveHandle);
        }
        const name = document.createElement('strong');
        name.textContent = collection.is_ungrouped ? '未分类' : collection.label;
        name.title = name.textContent;
        titleRow.appendChild(name);
        const meta = document.createElement('span');
        meta.className = 'history-compact-meta';
        meta.textContent = historyCompactGroupMetaParts(collection.tasks, [
            `${collection.groups.length} 组`,
        ]).join(' · ');
        title.append(titleRow, meta);

        const actions = document.createElement('div');
        actions.className = 'history-collection-card-actions';
        const joinSelectedBtn = createHistoryManagerGroupButton(
            collection.is_ungrouped ? '未分类' : '移入',
            () => applySelectedHistoryTasksToCollection(collection.value),
        );
        if (selectedTaskCount > 0) actions.append(joinSelectedBtn);
        if (!collection.is_ungrouped) {
            actions.append(
                createHistoryManagerGroupButton(
                    historyCollectionWorkbenchTarget === collection.value ? '取消目标' : '目标',
                    () => {
                        historyCollectionWorkbenchTarget = historyCollectionWorkbenchTarget === collection.value ? '' : collection.value;
                        renderHistoryManager();
                    },
                ),
                createHistoryMoreActions([
                    createHistoryManagerGroupButton('置顶', () => moveHistoryCollection(collection, 'top', allCollections)),
                    createHistoryManagerGroupButton('上移', () => moveHistoryCollection(collection, 'up', allCollections)),
                    createHistoryManagerGroupButton('下移', () => moveHistoryCollection(collection, 'down', allCollections)),
                    createHistoryManagerGroupButton('置底', () => moveHistoryCollection(collection, 'bottom', allCollections)),
                    createHistoryManagerGroupButton('重命名', () => renameHistoryCollection(collection)),
                    createHistoryManagerGroupButton('清空', () => clearHistoryCollection(collection)),
                ]),
            );
        }
        head.append(title, actions);
        card.appendChild(head);
        return card;
    }

    function createHistoryConfigGroupWorkbenchCard(group, splitCollections, options = {}) {
        const card = document.createElement('article');
        card.className = ['history-config-group-card', historyTasksAllSelected(group.tasks) ? 'selected' : ''].filter(Boolean).join(' ');
        card.classList.add('draggable');
        const groupKey = configGroupKey(group);
        card.dataset.configGroupKey = groupKey;
        card.dataset.collectionKey = historyCollectionStorageKey(options.collection || '__all__');
        if (historyConfigGroupSortState.sourceKey && historyConfigGroupSortState.sourceKey === groupKey) {
            card.classList.add('config-sort-source');
        }
        if (historyConfigGroupSortState.active && historyConfigGroupSortState.activeDropTarget === `config-sort:${groupKey}`) {
            card.classList.add(
                'config-sort-active',
                historyConfigGroupSortState.dropPosition === 'before' ? 'config-sort-before' : 'config-sort-after',
            );
            requestAnimationFrame(() => {
                if (card.isConnected && historyConfigGroupSortState.activeDropTarget === `config-sort:${groupKey}`) {
                    placeHistoryConfigGroupDropPreview(card, historyConfigGroupSortState.dropPosition);
                }
            });
        }
        card.addEventListener('dragenter', (event) => {
            historyConfigGroupOrderDragEnter(event, group, card, options);
        });
        card.addEventListener('dragover', (event) => {
            historyConfigGroupOrderDragEnter(event, group, card, options);
        });
        card.addEventListener('dragleave', (event) => {
            historyConfigGroupOrderDragLeave(event, group, card);
        });
        card.addEventListener('drop', async (event) => {
            if (await dropHistoryConfigGroupToSort(event, group, options)) return;
        });
        const ids = historyTaskIds(group.tasks);
        const selectedCount = ids.filter((id) => selectedHistoryTaskIds.has(id)).length;

        const select = document.createElement('label');
        select.className = 'history-config-group-select';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = ids.length > 0 && selectedCount === ids.length;
        checkbox.indeterminate = selectedCount > 0 && selectedCount < ids.length;
        checkbox.addEventListener('change', () => toggleHistoryTaskSelection(group.tasks));
        select.append(checkbox, document.createTextNode('选择分组'));

        const handle = document.createElement('button');
        handle.type = 'button';
        handle.className = 'history-drag-handle history-config-group-drag-handle';
        handle.textContent = '⋮⋮';
        handle.title = '拖拽配置分组调整顺序或移到右侧分组';
        handle.setAttribute('aria-label', '拖拽配置分组调整顺序或移到右侧分组');
        handle.draggable = true;
        handle.addEventListener('click', (event) => event.stopPropagation());
        handle.addEventListener('pointerdown', (event) => startHistoryConfigGroupPointerDrag(event, group, options, handle));
        handle.addEventListener('mousedown', (event) => {
            event.stopPropagation();
            if (!('PointerEvent' in window)) startHistoryConfigGroupMouseDrag(event, group, options, handle);
        });
        handle.addEventListener('touchstart', (event) => {
            event.stopPropagation();
            if (!('PointerEvent' in window)) startHistoryConfigGroupTouchDrag(event, group, options, handle);
        }, { passive: false });
        handle.addEventListener('dragstart', (event) => {
            if (historyConfigGroupPointerDrag) finishHistoryConfigGroupPointerDrag(false);
            beginHistoryConfigGroupDrag(event, group, options);
        });
        handle.addEventListener('dragend', () => finishHistoryDrag());

        if ((group.tasks || []).length === 1) {
            const task = group.tasks[0];
            card.classList.add('single-task');

            const main = document.createElement('button');
            main.type = 'button';
            main.className = 'history-config-group-card-main history-single-task-main';
            main.addEventListener('click', () => loadHistoryTask(task.id));

            const titleRow = document.createElement('span');
            titleRow.className = 'history-single-task-title-row';
            const title = document.createElement('strong');
            title.textContent = historyGroupDisplayLabel(group);
            title.title = title.textContent;
            const state = document.createElement('span');
            state.className = ['history-row-state', task.state || 'unknown'].join(' ');
            state.textContent = [
                task.job === 'preprocess' ? '预处理' : '训练',
                historyStateLabel(task.state),
                historyTaskIsArchived(task) ? '已归档' : '',
            ].filter(Boolean).join(' · ');
            titleRow.append(title, state);

            const taskName = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
            const sourceLabel = compactHistoryPathLabel(group.source_label || group.fallback_group_label || group.label);
            const timeText = `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`;
            const dataText = `${task.metric_count || 0} loss / ${task.log_count || 0} log`;
            const meta = document.createElement('span');
            meta.className = 'history-compact-meta';
            meta.textContent = [
                taskName && taskName !== title.textContent ? taskName : '',
                sourceLabel && sourceLabel !== title.textContent ? `源:${sourceLabel}` : '',
                compactHistoryQueueLabel(task),
                compactHistoryContinueLabel(task),
                compactHistoryResumeLabel(task),
                timeText,
                dataText,
            ].filter(Boolean).join(' · ');
            meta.title = [
                `任务: ${taskName}`,
                group.source_label ? `源配置: ${group.source_label}` : `配置组: ${group.fallback_group_label || group.label}`,
                historyQueueLabel(task),
                historyContinueLabel(task),
                historyResumeLabel(task),
                timeText,
                dataText,
            ].filter(Boolean).join(' · ');
            titleRow.appendChild(meta);
            main.appendChild(titleRow);

            const actions = document.createElement('div');
            actions.className = 'history-config-group-card-actions history-single-task-actions';
            if (historyCollectionWorkbenchTarget) {
                actions.append(createHistoryManagerGroupButton('目标', () => setHistoryCollectionForTasksDirect(group.tasks, historyCollectionWorkbenchTarget)));
            }
            if (task.job === 'training') {
                actions.append(createHistoryTaskPreviewButton(task));
            }
            actions.append(
                createHistoryActionButton('查看', () => loadHistoryTask(task.id)),
                createHistoryMoreActions([
                    createHistoryManagerGroupButton('置顶', () => moveHistoryConfigGroup(group, 'top', options.groups, options.collection)),
                    createHistoryManagerGroupButton('上移', () => moveHistoryConfigGroup(group, 'up', options.groups, options.collection)),
                    createHistoryManagerGroupButton('下移', () => moveHistoryConfigGroup(group, 'down', options.groups, options.collection)),
                    createHistoryManagerGroupButton('置底', () => moveHistoryConfigGroup(group, 'bottom', options.groups, options.collection)),
                    createHistoryTaskConfigButton(task),
                    createHistoryConfigGroupMergeButton(group),
                    createHistoryManagerGroupButton('设置分组', () => setHistoryCollectionForTasks(group.tasks, commonHistoryCollectionValue(group.tasks), historyGroupDisplayLabel(group))),
                    createHistoryManagerGroupButton('清除分组', () => clearHistoryCollectionForTasks(group.tasks, historyGroupDisplayLabel(group))),
                    createHistoryActionButton(historyTaskIsArchived(task) ? '取消归档' : '归档', () => archiveHistoryTask(task)),
                    createHistoryActionButton('删除', () => deleteHistoryTask(task), 'danger'),
                ]),
            );

            card.append(select, handle, main, actions);
            return card;
        }

        const main = document.createElement('div');
        main.className = 'history-config-group-card-main';
        const title = document.createElement('strong');
        title.textContent = historyGroupDisplayLabel(group);
        title.title = title.textContent;
        const collections = historyCollectionNamesForTasks(group.tasks);
        const split = splitCollections?.get(configGroupKey(group));
        const meta = document.createElement('span');
        meta.className = 'history-compact-meta';
        const sourceLabel = compactHistoryPathLabel(group.source_label || group.fallback_group_label || group.label);
        meta.textContent = historyCompactGroupMetaParts(group.tasks, [
            sourceLabel && sourceLabel !== title.textContent ? `源:${sourceLabel}` : '',
            split && split.size > 1 ? `跨 ${split.size} 组` : '',
        ]).join(' · ');
        meta.title = historyManagerGroupMetaParts(group.tasks, [
            group.source_label ? `源配置: ${group.source_label}` : `配置组: ${group.fallback_group_label || group.label}`,
            collections.length ? `当前分组: ${collections.join(' / ')}` : '当前分组: 未分类',
            split && split.size > 1 ? `分布在 ${split.size} 个分组` : '',
        ]).join(' · ');
        main.append(title, meta);

        const actions = document.createElement('div');
        actions.className = 'history-config-group-card-actions';
        const trainingCount = group.tasks.filter((task) => task.job === 'training').length;
        if (historyCollectionWorkbenchTarget) {
            actions.append(createHistoryManagerGroupButton('目标', () => setHistoryCollectionForTasksDirect(group.tasks, historyCollectionWorkbenchTarget)));
        }
        if (trainingCount) {
            actions.append(createHistoryConfigGroupPreviewButton(group));
        }
        actions.append(createHistoryMoreActions([
            createHistoryManagerGroupButton('置顶', () => moveHistoryConfigGroup(group, 'top', options.groups, options.collection)),
            createHistoryManagerGroupButton('上移', () => moveHistoryConfigGroup(group, 'up', options.groups, options.collection)),
            createHistoryManagerGroupButton('下移', () => moveHistoryConfigGroup(group, 'down', options.groups, options.collection)),
            createHistoryManagerGroupButton('置底', () => moveHistoryConfigGroup(group, 'bottom', options.groups, options.collection)),
            createHistoryConfigGroupMergeButton(group),
            createHistoryManagerGroupButton('设置分组', () => setHistoryCollectionForTasks(group.tasks, commonHistoryCollectionValue(group.tasks), historyGroupDisplayLabel(group))),
            createHistoryManagerGroupButton('清除分组', () => clearHistoryCollectionForTasks(group.tasks, historyGroupDisplayLabel(group))),
        ]));

        card.append(select, handle, main, actions);
        const taskList = document.createElement('div');
        taskList.className = 'history-config-group-task-list';
        for (const task of group.tasks) {
            taskList.appendChild(createHistoryManagerRow(task));
        }
        card.appendChild(taskList);
        return card;
    }

    function historyCollectionNamesForTasks(tasks) {
        const names = Array.from(new Set((tasks || []).map(historyTaskCollectionLabel).filter(Boolean)));
        return names.length ? names : ['未分类'];
    }

    function moveItemInList(list, value, direction) {
        const out = uniqueStringList(list);
        const item = String(value || '').trim();
        const index = out.indexOf(item);
        if (!item || index < 0) return out;
        out.splice(index, 1);
        if (direction === 'top') out.unshift(item);
        else if (direction === 'bottom') out.push(item);
        else if (direction === 'up') out.splice(Math.max(0, index - 1), 0, item);
        else if (direction === 'down') out.splice(Math.min(out.length, index + 1), 0, item);
        else out.splice(index, 0, item);
        return out;
    }

    function collectionOrderValues(collections) {
        const available = (collections || []).filter((collection) => !collection.is_ungrouped && collection.value).map((collection) => collection.value);
        const out = historyCollectionSettings.collection_order.filter((value) => available.includes(value));
        for (const value of available) {
            if (!out.includes(value)) out.push(value);
        }
        return out;
    }

    async function moveHistoryCollection(collection, direction, allCollections = []) {
        if (!collection || collection.is_ungrouped || !collection.value) return;
        await moveHistoryCollectionValue(collection.value, direction, allCollections);
    }

    async function moveHistoryCollectionValue(value, direction, allCollections = null) {
        const group = String(value || '').trim();
        if (!group) return;
        const collections = allCollections || historyCollectionsForWorkbench(historyTasks);
        const order = moveItemInList(collectionOrderValues(collections), group, direction);
        await saveHistoryCollectionSettings({
            ...historyCollectionSettings,
            collection_order: order,
        });
    }

    async function ensureHistoryCollectionOrderValue(value) {
        const group = String(value || '').trim();
        if (!group || historyCollectionSettings.collection_order.includes(group)) return;
        await saveHistoryCollectionSettings({
            ...historyCollectionSettings,
            collection_order: [...historyCollectionSettings.collection_order, group],
        });
    }

    function configGroupOrderValues(groups, collection) {
        const key = historyCollectionStorageKey(collection || '__all__');
        const available = (groups || []).map(configGroupKey).filter(Boolean);
        const saved = historyCollectionSettings.config_group_order?.[key] || [];
        const out = saved.filter((value) => available.includes(value));
        for (const value of available) {
            if (!out.includes(value)) out.push(value);
        }
        return out;
    }

    async function moveHistoryConfigGroup(group, direction, groups = [], collection = null) {
        const groupKey = configGroupKey(group);
        if (!groupKey) return;
        const collectionKey = historyCollectionStorageKey(collection || '__all__');
        const order = moveItemInList(configGroupOrderValues(groups, collection), groupKey, direction);
        await saveHistoryCollectionSettings({
            ...historyCollectionSettings,
            config_group_order: {
                ...(historyCollectionSettings.config_group_order || {}),
                [collectionKey]: order,
            },
        });
    }

    function groupHistoryTasksByCollection(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const key = historyTaskCollectionKey(task);
            if (!map.has(key)) {
                map.set(key, {
                    key,
                    label: historyTaskCollectionLabel(task),
                    value: historyTaskCollectionValue(task),
                    is_ungrouped: !historyTaskCollectionValue(task),
                    tasks: [],
                });
            }
            map.get(key).tasks.push(task);
        }
        return Array.from(map.values())
            .map(enrichHistoryCollection)
            .sort(historyCollectionComparator);
    }

    function historyCollectionComparator(a, b) {
        if (a.is_ungrouped !== b.is_ungrouped) return a.is_ungrouped ? -1 : 1;
        const order = historyCollectionSettings.collection_order || [];
        const aIndex = a.value ? order.indexOf(a.value) : -1;
        const bIndex = b.value ? order.indexOf(b.value) : -1;
        if (aIndex !== bIndex) {
            if (aIndex < 0) return 1;
            if (bIndex < 0) return -1;
            return aIndex - bIndex;
        }
        return (b.latest_started_at - a.latest_started_at) || a.label.localeCompare(b.label, 'zh-CN');
    }

    function historyCollectionStorageKey(collection) {
        if (!collection) return '__all__';
        if (typeof collection === 'string') {
            if (!collection || collection === 'collection:__all__') return '__all__';
            if (collection === HISTORY_UNGROUPED_COLLECTION_KEY) return '__ungrouped__';
            return collection.startsWith('collection:') ? collection.slice('collection:'.length) : collection;
        }
        if (collection.is_ungrouped) return '__ungrouped__';
        return String(collection.value || '').trim() || '__ungrouped__';
    }

    function historyCollectionByKey(collections, key) {
        return (collections || []).find((collection) => collection.key === key) || null;
    }

    function sortedHistoryConfigGroups(groups, collectionKey = '__all__') {
        const storageKey = historyCollectionStorageKey(collectionKey);
        const order = historyCollectionSettings.config_group_order?.[storageKey] || [];
        return [...(groups || [])].sort((a, b) => {
            const aKey = configGroupKey(a);
            const bKey = configGroupKey(b);
            const aIndex = order.indexOf(aKey);
            const bIndex = order.indexOf(bKey);
            if (aIndex !== bIndex) {
                if (aIndex < 0) return 1;
                if (bIndex < 0) return -1;
                return aIndex - bIndex;
            }
            const aTime = Math.max(0, ...a.tasks.map((task) => Number(task.started_at || 0)));
            const bTime = Math.max(0, ...b.tasks.map((task) => Number(task.started_at || 0)));
            return (bTime - aTime) || historyGroupDisplayLabel(a).localeCompare(historyGroupDisplayLabel(b), 'zh-CN');
        });
    }

    function enrichHistoryCollection(collection) {
        const tasks = [...(collection.tasks || [])].sort(historyTaskSortComparator(historyManagerFilters.sort));
        const groups = sortedHistoryConfigGroups(
            groupHistoryTasks(tasks).map(sortHistoryManagerGroupTasks),
            historyCollectionStorageKey(collection),
        );
        return {
            ...collection,
            tasks,
            groups,
            latest_started_at: Math.max(0, ...tasks.map((task) => Number(task.started_at || 0))),
        };
    }

    function sortHistoryManagerGroupTasks(group) {
        return {
            ...group,
            tasks: [...(group.tasks || [])].sort(historyTaskSortComparator(historyManagerFilters.sort)),
        };
    }

    function historyTaskCollectionValue(task) {
        return String(task?.group || '').trim();
    }

    function historyTaskCollectionLabel(task) {
        return historyTaskCollectionValue(task) || '未分类';
    }

    function historyTaskCollectionKey(task) {
        const value = historyTaskCollectionValue(task);
        return value ? `collection:${value}` : HISTORY_UNGROUPED_COLLECTION_KEY;
    }

    function historyConfigGroupCollectionMap(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const group = historyConfigGroupFromTask(task);
            const key = configGroupKey(group);
            if (!map.has(key)) map.set(key, new Set());
            map.get(key).add(historyTaskCollectionLabel(task));
        }
        return map;
    }

    function historyTaskIds(tasks) {
        return (tasks || []).map((task) => task.id).filter(Boolean);
    }

    function historyTasksAllSelected(tasks) {
        const ids = historyTaskIds(tasks);
        return ids.length > 0 && ids.every((id) => selectedHistoryTaskIds.has(id));
    }

    function toggleHistoryTaskSelection(tasks) {
        const ids = historyTaskIds(tasks);
        const selected = ids.every((id) => selectedHistoryTaskIds.has(id));
        ids.forEach((id) => {
            if (selected) selectedHistoryTaskIds.delete(id);
            else selectedHistoryTaskIds.add(id);
        });
        renderHistoryManager();
    }

    function historyManagerGroupMetaParts(tasks, extra = []) {
        const trainingCount = tasks.filter((task) => task.job === 'training').length;
        const preprocessCount = tasks.filter((task) => task.job === 'preprocess').length;
        const errorCount = tasks.filter((task) => ['error', 'interrupted'].includes(task.state)).length;
        const archivedCount = tasks.filter(historyTaskIsArchived).length;
        const queueCount = tasks.filter((task) => task.from_queue || task.queue_item_id).length;
        return [
            `${tasks.length} 条任务`,
            trainingCount ? `${trainingCount} 次训练` : '',
            preprocessCount ? `${preprocessCount} 个预处理` : '',
            errorCount ? `${errorCount} 个异常/中断` : '',
            archivedCount ? `${archivedCount} 个归档` : '',
            queueCount ? `${queueCount} 个队列来源` : '',
            ...extra,
        ].filter(Boolean);
    }

    function historyCompactGroupMetaParts(tasks, extra = []) {
        const trainingCount = tasks.filter((task) => task.job === 'training').length;
        const preprocessCount = tasks.filter((task) => task.job === 'preprocess').length;
        const errorCount = tasks.filter((task) => ['error', 'interrupted'].includes(task.state)).length;
        const archivedCount = tasks.filter(historyTaskIsArchived).length;
        const queueCount = tasks.filter((task) => task.from_queue || task.queue_item_id).length;
        return [
            `${tasks.length} 条`,
            trainingCount ? `${trainingCount} 训` : '',
            preprocessCount ? `${preprocessCount} 预` : '',
            errorCount ? `${errorCount} 异常` : '',
            archivedCount ? `${archivedCount} 归档` : '',
            queueCount ? `${queueCount} 队列` : '',
            ...extra,
        ].filter(Boolean);
    }

    function commonHistoryCollectionValue(tasks) {
        const values = Array.from(new Set((tasks || []).map(historyTaskCollectionValue).filter(Boolean)));
        return values.length === 1 ? values[0] : '';
    }

    function createHistoryManagerGroupButton(label, handler, tone = '') {
        const btn = createHistoryActionButton(label, handler, tone);
        btn.classList.add('history-manager-group-action');
        return btn;
    }

    function createHistoryConfigGroupMergeButton(group) {
        const btn = createHistoryManagerGroupButton('查看', () => loadConfigGroupTimeline(group, { skipSelectionDialog: true }));
        btn.title = '查阅这个自动配置分组内的训练日志、Loss 曲线和任务明细';
        return btn;
    }

    function createHistoryConfigGroupPreviewButton(group) {
        const btn = createHistoryManagerGroupButton('预览', () => openHistoryConfigGroupPreview(group));
        btn.title = '汇总查看这个配置分组下所有训练任务的样张和权重';
        return btn;
    }

    function canPreviewHistoryConfigGroup(group) {
        return Boolean(group && group.methods_subdir && group.variant && group.methods_subdir !== '手动选择');
    }

    async function setHistoryCollectionForTasks(tasks, value = '', description = '') {
        const ids = historyTaskIds(tasks);
        if (!ids.length) return;
        const group = await showHistoryCollectionSelectDialog({
            title: value ? '重命名集合' : '设置集合',
            description: `${ids.length} 条历史任务${description ? ` · ${description}` : ''}`,
            value,
            confirmText: '保存集合',
        });
        if (group === null) return;
        await applyHistoryTaskIdsToCollection(ids, group.trim());
    }

    async function renameHistoryCollection(collection) {
        const oldValue = String(collection?.value || '').trim();
        if (!oldValue || collection?.is_ungrouped) return;
        const nextValue = await showHistoryCollectionSelectDialog({
            title: '重命名集合',
            description: collection.label || oldValue,
            value: oldValue,
            confirmText: '保存集合',
        });
        if (nextValue === null) return;
        const clean = String(nextValue || '').trim();
        if (!clean || clean === oldValue) return;
        const oldKey = `collection:${oldValue}`;
        const newKey = `collection:${clean}`;
        const collectionOrder = renameHistoryCollectionOrderValue(oldValue, clean);
        const configGroupOrder = renameHistoryConfigGroupOrderKey(oldValue, clean);
        const ids = historyTaskIds(collection.tasks || []);
        if (ids.length) {
            const res = await applyHistoryTaskIdsBatchAction(ids, 'set_group', { group: clean });
            if (res === null) return;
        }
        historyCollectionSettings = normalizeHistoryCollectionSettings({
            ...historyCollectionSettings,
            collection_order: collectionOrder,
            config_group_order: configGroupOrder,
        });
        await saveHistoryCollectionSettings(historyCollectionSettings);
        selectedHistoryCollectionKey = newKey;
        if (historyCollectionWorkbenchTarget === oldValue) historyCollectionWorkbenchTarget = clean;
        renderHistoryManager();
        setHistoryDropFeedback(`已重命名分组「${oldValue}」为「${clean}」。`, 'ok');
    }

    async function clearHistoryCollection(collection) {
        const value = String(collection?.value || '').trim();
        if (!value || collection?.is_ungrouped) return;
        const ids = historyTaskIds(collection.tasks || []);
        const ok = await showHistoryTaskConfirmDialog({
            title: ids.length ? '清空集合' : '删除空集合',
            description: `${collection.label || value} · ${ids.length} 条历史任务`,
            message: ids.length
                ? '只会移除这些历史任务的集合名称，并删除右侧空集合入口；同配置文件自动分组、日志、权重和运行目录都会保留。'
                : '这个集合没有任务，会从右侧分组导航中移除。',
            confirmText: ids.length ? '清空集合' : '删除空集合',
        });
        if (!ok) return;
        if (ids.length) {
            const res = await applyHistoryTaskIdsBatchAction(ids, 'set_group', { group: '' });
            if (res === null) return;
        }
        await removeHistoryCollectionSettingValue(value);
        if (selectedHistoryCollectionKey === `collection:${value}`) selectedHistoryCollectionKey = HISTORY_UNGROUPED_COLLECTION_KEY;
        if (historyCollectionWorkbenchTarget === value) historyCollectionWorkbenchTarget = '';
        renderHistoryManager();
        setHistoryDropFeedback(ids.length ? `已清空分组「${collection.label || value}」。` : `已删除空分组「${collection.label || value}」。`, 'ok');
    }

    function renameHistoryCollectionOrderValue(oldValue, newValue) {
        const oldClean = String(oldValue || '').trim();
        const newClean = String(newValue || '').trim();
        const out = [];
        const seen = new Set();
        for (const value of uniqueStringList(historyCollectionSettings.collection_order || [])) {
            const next = value === oldClean ? newClean : value;
            if (!next || seen.has(next)) continue;
            out.push(next);
            seen.add(next);
        }
        if (newClean && !seen.has(newClean)) out.push(newClean);
        return out;
    }

    function renameHistoryConfigGroupOrderKey(oldValue, newValue) {
        const oldKey = historyCollectionStorageKey(oldValue);
        const newKey = historyCollectionStorageKey(newValue);
        const current = normalizeHistoryConfigGroupOrder(historyCollectionSettings.config_group_order);
        if (!oldKey || !newKey || oldKey === newKey) return current;
        const oldOrder = current[oldKey] || [];
        const newOrder = uniqueStringList([...(current[newKey] || []), ...oldOrder]);
        delete current[oldKey];
        if (newOrder.length) current[newKey] = newOrder;
        return current;
    }

    async function removeHistoryCollectionSettingValue(value) {
        const clean = String(value || '').trim();
        if (!clean) return historyCollectionSettings;
        const settings = {
            ...historyCollectionSettings,
            collection_order: uniqueStringList(historyCollectionSettings.collection_order || []).filter((item) => item !== clean),
            config_group_order: Object.fromEntries(
                Object.entries(normalizeHistoryConfigGroupOrder(historyCollectionSettings.config_group_order))
                    .filter(([key]) => key !== historyCollectionStorageKey(clean)),
            ),
        };
        return saveHistoryCollectionSettings(settings);
    }

    async function setHistoryCollectionForTasksDirect(tasks, value) {
        const ids = historyTaskIds(tasks);
        const group = String(value || '').trim();
        if (!ids.length || !group) return;
        await applyHistoryTaskIdsToCollection(ids, group);
    }

    async function applySelectedHistoryTasksToCollection(value) {
        const ids = historyTaskIds(selectedHistoryTasks());
        if (!ids.length) return;
        await applyHistoryTaskIdsToCollection(ids, String(value || '').trim(), { clearSelection: true });
    }

    async function applyHistoryTaskIdsToCollection(ids, group, options = {}) {
        const clean = String(group || '').trim();
        if (clean) await ensureHistoryCollectionOrderValue(clean);
        return applyHistoryTaskIdsBatchAction(ids, 'set_group', { group: clean }, options);
    }

    async function clearSelectedHistoryCollection() {
        const tasks = selectedHistoryTasks();
        if (!tasks.length) return;
        const ok = await showHistoryTaskConfirmDialog({
            title: '清除已选集合',
            description: `${tasks.length} 条历史任务`,
            message: '只会移除已选任务的集合名称；同配置文件自动分组、历史详情、日志和运行目录都会保留。',
            confirmText: '清除集合',
        });
        if (!ok) return;
        await applyHistoryTaskIdsBatchAction(historyTaskIds(tasks), 'set_group', { group: '' }, { clearSelection: true });
    }

    async function clearHistoryCollectionForTasks(tasks, description = '') {
        const ids = historyTaskIds(tasks);
        if (!ids.length) return;
        const ok = await showHistoryTaskConfirmDialog({
            title: '清空集合',
            description: `${ids.length} 条历史任务${description ? ` · ${description}` : ''}`,
            message: '只会移除这些历史任务的集合名称；同配置文件自动分组、日志、权重和运行目录都会保留。',
            confirmText: '清空集合',
        });
        if (!ok) return;
        await applyHistoryTaskIdsBatchAction(ids, 'set_group', { group: '' });
    }

    async function archiveHistoryTasksByIds(tasks, archived, description = '') {
        const ids = historyTaskIds(tasks);
        if (!ids.length) return;
        const ok = await showHistoryTaskConfirmDialog({
            title: archived ? '批量归档' : '批量取消归档',
            description: `${ids.length} 条历史任务${description ? ` · ${description}` : ''}`,
            message: archived ? '归档后默认会隐藏这些任务。' : '取消归档后这些任务会重新出现在默认列表中。',
            confirmText: archived ? '归档' : '取消归档',
        });
        if (!ok) return;
        await applyHistoryTaskIdsBatchAction(ids, archived ? 'archive' : 'unarchive');
    }

    async function deleteHistoryTasksByIds(tasks) {
        await deleteHistoryTasksThorough(historyTaskIds(tasks));
    }

    function syncHistorySelectionWithTasks() {
        const valid = new Set(historyTasks.map((task) => task.id).filter(Boolean));
        selectedHistoryTaskIds = new Set(Array.from(selectedHistoryTaskIds).filter((id) => valid.has(id)));
    }

    function selectedHistoryTasks() {
        const ids = selectedHistoryTaskIds;
        const visible = new Set(historyCurrentVisibleTaskIds);
        if (!visible.size) return [];
        return historyTasks.filter((task) => ids.has(task.id) && visible.has(task.id));
    }

    function renderHistoryBulkBar() {
        const bar = document.getElementById('history-bulk-bar');
        const summary = document.getElementById('history-bulk-summary');
        if (!bar || !summary) return;
        const tasks = selectedHistoryTasks();
        bar.hidden = tasks.length === 0;
        summary.textContent = `已选 ${tasks.length} 项`;
    }

    function syncHistoryFilterControls() {
        const controls = {
            'history-manager-search': 'search',
            'history-filter-kind': 'kind',
            'history-filter-state': 'state',
            'history-filter-archived': 'archived',
            'history-filter-source': 'source',
            'history-sort-mode': 'sort',
        };
        for (const [id, key] of Object.entries(controls)) {
            const el = document.getElementById(id);
            if (el) el.value = historyManagerFilters[key] || historyManagerFilterDefault(key);
        }
        const collectionSearch = document.getElementById('history-collection-search');
        if (collectionSearch) collectionSearch.value = historyCollectionSearch || '';
        const configGroupSearch = document.getElementById('history-config-group-search');
        if (configGroupSearch) configGroupSearch.value = historyConfigGroupSearch || '';
    }

    function historyManagerFilterDefault(key) {
        if (key === 'search') return '';
        if (key === 'archived') return 'active';
        if (key === 'sort') return 'newest';
        return 'all';
    }

    function openHistoryCollectionsWorkbench() {
        syncHistoryFilterControls();
        showTrainingView('history');
        renderHistoryManager();
    }

    function groupHistoryTasks(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const group = historyConfigGroupFromTask(task);
            if (!map.has(group.key)) {
                map.set(group.key, { ...group, tasks: [] });
            }
            map.get(group.key).tasks.push(task);
        }
        return Array.from(map.values())
            .map(enrichHistoryGroup)
            .sort((a, b) => {
                const aTime = Math.max(...a.tasks.map((task) => Number(task.started_at || 0)));
                const bTime = Math.max(...b.tasks.map((task) => Number(task.started_at || 0)));
                return (bTime - aTime) || historyGroupDisplayLabel(a).localeCompare(historyGroupDisplayLabel(b), 'zh-CN');
            });
    }

    function historyConfigGroupFromTask(task) {
        const methodsSubdir = String(task.methods_subdir || '-');
        const variant = String(task.variant || '-');
        const preset = String(task.preset || 'default');
        const legacyLabel = `${methodsSubdir} / ${variant} / ${preset}`;
        const historyKey = String(task.history_group_key || '').trim();
        const sourceConfig = String(task.history_source_config_file || '').trim();
        const groupLabel = String(task.history_group_label || '').trim() || sourceConfig || legacyLabel;
        return {
            key: historyKey || [methodsSubdir, variant, preset].join('\u0001'),
            history_group_key: historyKey || '',
            history_group_label: groupLabel,
            history_source_config_file: sourceConfig,
            methods_subdir: methodsSubdir,
            variant,
            preset,
            label: groupLabel,
            legacy_label: legacyLabel,
        };
    }

    function configGroupKey(group) {
        if (group?.key) return group.key;
        if (group?.history_group_key) return group.history_group_key;
        return [group.methods_subdir || '-', group.variant || '-', group.preset || 'default'].join('\u0001');
    }

    function enrichHistoryGroup(group) {
        const tasks = [...(group.tasks || [])].sort((a, b) => {
            const aTime = Number(a.started_at || 0);
            const bTime = Number(b.started_at || 0);
            return (bTime - aTime) || String(b.id || '').localeCompare(String(a.id || ''), 'zh-CN');
        });
        const latestTask = tasks[0] || {};
        const runDirs = new Set(tasks.map((task) => historyTaskRunPath(task)).filter(Boolean));
        return {
            ...group,
            tasks,
            latestTask,
            run_count: runDirs.size,
            display_label: historyTaskDisplayName(latestTask) || group.label,
            source_label: group.history_source_config_file || '',
            fallback_group_label: group.history_group_label || group.legacy_label || group.label,
        };
    }

    function historyTaskDisplayName(task) {
        if (!task) return '';
        const customName = String(task.name || '').trim();
        if (task.training_mode === 'continue_lora') {
            const kind = String(task.continue_from_weight_kind || 'LoRA').trim() || 'LoRA';
            const name = String(task.continue_from_weight_name || '').trim();
            const continueName = `继续训练 ${kind}${name ? ` · ${name}` : ''}`;
            return customName && customName !== continueName ? customName : continueName;
        }
        if (customName) return customName;
        const defaultName = String(
            task.history_run_label
            || runLabelFromPath(task.run_dir || task.training_output_dir || task.output_dir)
            || task.id
            || ''
        ).trim();
        return defaultName;
    }

    function historyTaskIsArchived(task) {
        if (Boolean(task?.archived)) return true;
        return task?.job === 'preprocess' && !task?.updated_at;
    }

    function historyTaskRunPath(task) {
        return String(task?.run_dir || task?.training_output_dir || task?.output_dir || '').trim();
    }

    function historyResumeLabel(task) {
        const resume = task?.resume_from || {};
        if (!resume || typeof resume !== 'object') return '';
        const checkpoint = String(resume.checkpoint_name || '').trim();
        const step = resume.checkpoint_step !== undefined && resume.checkpoint_step !== null
            ? String(resume.checkpoint_step).trim()
            : '';
        if (checkpoint && step) return `从检查点恢复: ${checkpoint} · step ${step}`;
        if (checkpoint) return `从检查点恢复: ${checkpoint}`;
        if (step) return `从检查点恢复: step ${step}`;
        return resume.source_task_id ? '从检查点恢复' : '';
    }

    function historyQueueLabel(task) {
        const queueId = String(task?.queue_item_id || '').trim();
        if (!Boolean(task?.from_queue) && !queueId) return '';
        const attempt = Number(task?.queue_attempt || 1);
        return attempt > 1 ? `来自队列 · 第 ${attempt} 次尝试` : '来自队列';
    }

    function historyContinueLabel(task) {
        if (task?.training_mode !== 'continue_lora') return '';
        const kind = String(task.continue_from_weight_kind || 'LoRA').trim() || 'LoRA';
        const name = String(task.continue_from_weight_name || '').trim();
        return `继续训练 ${kind}${name ? `: ${name}` : ''}`;
    }

    function historyContinuePathLabel(task) {
        if (task?.training_mode !== 'continue_lora') return '';
        const path = String(task.continue_from_weight_abs_path || '').trim();
        return path ? `基于: ${path}` : '';
    }

    function runLabelFromPath(value) {
        const text = String(value || '').replace(/\\/g, '/').trim();
        if (!text) return '';
        const parts = text.split('/').filter(Boolean);
        if (!parts.length) return text;
        if (parts[parts.length - 1] === 'training_output' && parts.length > 1) {
            return parts[parts.length - 2];
        }
        return parts[parts.length - 1];
    }

    function historyGroupDisplayLabel(group) {
        return String(group?.display_label || group?.history_run_label || group?.label || configGroupLabel(group) || '').trim();
    }

    function createHistoryGroupHeading(group) {
        const heading = document.createElement('div');
        heading.className = 'task-history-group-title';
        const trainingCount = group.tasks.filter((task) => task.job === 'training').length;
        const preprocessCount = group.tasks.filter((task) => task.job === 'preprocess').length;

        const title = document.createElement('span');
        const name = document.createElement('strong');
        name.textContent = historyGroupDisplayLabel(group);
        const meta = document.createElement('em');
        meta.textContent = [
            group.source_label ? `源配置: ${group.source_label}` : `配置分组: ${group.fallback_group_label || group.label}`,
            `${trainingCount} 次训练`,
            preprocessCount ? `${preprocessCount} 个预处理` : '',
            group.run_count ? `${group.run_count} 个运行目录` : '',
        ].filter(Boolean).join(' · ');
        title.append(name, meta);
        heading.appendChild(title);

        if (trainingCount) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'task-history-group-action';
            btn.textContent = '查阅分组详情';
            btn.title = '查阅这个自动配置分组内的训练日志、Loss 曲线和任务明细';
            btn.addEventListener('click', () => loadConfigGroupTimeline(group, { skipSelectionDialog: true }));
            heading.appendChild(btn);
            const previewBtn = document.createElement('button');
            previewBtn.type = 'button';
            previewBtn.className = 'task-history-group-action';
            previewBtn.textContent = '分组预览';
            previewBtn.title = '汇总查看这个配置分组下所有训练任务的样张和权重';
            previewBtn.addEventListener('click', () => openHistoryConfigGroupPreview(group));
            heading.appendChild(previewBtn);
        }
        return heading;
    }

    function createHistoryTaskItem(task) {
        const card = document.createElement('article');
        card.className = 'task-history-item';
        if (task.id === viewingHistoryTaskId && isHistoryDetailDialogOpen()) card.classList.add('active');
        const archived = historyTaskIsArchived(task);
        if (archived) card.classList.add('archived');

        const main = document.createElement('button');
        main.type = 'button';
        main.className = 'task-history-main';
        main.addEventListener('click', () => loadHistoryTask(task.id));

        const title = document.createElement('strong');
        title.className = 'task-history-title';
        title.textContent = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
        const meta = document.createElement('span');
        meta.className = 'task-history-meta';
        meta.textContent = [
            task.job === 'preprocess' ? '预处理' : '训练',
            historyQueueLabel(task),
            historyContinueLabel(task),
            historyResumeLabel(task),
            historyStateLabel(task.state),
            task.started_at_text || task.id,
            archived ? '已归档' : '',
        ].filter(Boolean).join(' · ');
        const pathValue = task.run_dir || task.training_output_dir || task.output_dir || task.history_dir || task.id;
        const paths = document.createElement('em');
        paths.className = 'task-history-path';
        paths.title = [
            pathValue ? `目录: ${pathValue}` : '',
            historyContinuePathLabel(task),
        ].filter(Boolean).join(' · ');
        const pathLabel = document.createElement('span');
        pathLabel.textContent = '目录';
        const pathText = document.createElement('code');
        pathText.textContent = compactPathLabel(pathValue);
        paths.append(pathLabel, pathText);
        const continuePath = historyContinuePathLabel(task);
        if (continuePath) {
            const continueText = document.createElement('code');
            continueText.textContent = compactPathLabel(continuePath.replace(/^基于:\s*/, ''));
            paths.appendChild(continueText);
        }
        const counts = document.createElement('em');
        counts.className = 'task-history-counts';
        counts.textContent = `${task.metric_count || 0} loss点 / ${task.log_count || 0} 日志`;
        main.append(title, meta, paths, counts);

        const actions = document.createElement('div');
        actions.className = 'task-history-actions';
        if (task.job === 'training') {
            actions.append(
                createHistoryTaskPreviewButton(task),
            );
        }
        actions.append(
            createHistoryActionButton('查看', () => loadHistoryTask(task.id)),
        );

        card.append(main, actions);
        return card;
    }

    function compactPathLabel(value) {
        const text = String(value || '').replace(/\\/g, '/').trim();
        if (!text) return '-';
        const parts = text.split('/').filter(Boolean);
        if (!parts.length) return text;
        const name = parts[parts.length - 1];
        const parent = parts[parts.length - 2] || '';
        if (name === 'training_output' && parent) return `.../${parent}/training_output`;
        return parts.length > 1 ? `.../${name}` : name;
    }

    function createHistoryActionButton(label, handler, tone = '') {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = ['task-history-action', tone].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            handler();
        });
        return btn;
    }

    function createHistoryTaskPreviewButton(task) {
        const btn = createHistoryActionButton('预览', () => loadHistoryTask(task.id, { detailTab: 'preview' }));
        btn.title = '只查看这一次训练任务的样张和权重；会在任务详情中打开。';
        return btn;
    }

    function createHistoryTaskConfigButton(task) {
        const btn = createHistoryActionButton('配置', () => loadHistoryTask(task.id, { detailTab: 'config_files' }));
        btn.title = '打开这条历史任务的配置快照';
        return btn;
    }

    async function applyHistoryTaskIdsBatchAction(taskIds, action, extra = {}, options = {}) {
        const ids = (taskIds || []).filter(Boolean);
        if (!ids.length) return null;
        const res = await api('/api/training/history/batch', {
            method: 'POST',
            body: JSON.stringify({ action, task_ids: ids, ...extra }),
        });
        if (!res.ok) {
            await showHistoryTaskMessageDialog({
                title: '批量操作失败',
                message: res.error || '批量操作失败',
                tone: 'error',
            });
            return null;
        }
        if (options.clearSelection) {
            ids.forEach((id) => selectedHistoryTaskIds.delete(id));
        }
        await loadTrainingHistoryList();
        return res;
    }

    async function applyHistoryBatchAction(action, extra = {}) {
        const taskIds = historyTaskIds(selectedHistoryTasks());
        if (!taskIds.length) return null;
        return applyHistoryTaskIdsBatchAction(taskIds, action, extra, { clearSelection: true });
    }

    async function archiveSelectedHistoryTasks(archived) {
        const tasks = selectedHistoryTasks();
        if (!tasks.length) return;
        const ok = await showHistoryTaskConfirmDialog({
            title: archived ? '批量归档' : '批量取消归档',
            description: `${tasks.length} 条历史任务`,
            message: archived ? '归档后默认会隐藏这些任务。' : '取消归档后这些任务会重新出现在默认列表中。',
            confirmText: archived ? '归档' : '取消归档',
        });
        if (!ok) return;
        await applyHistoryBatchAction(archived ? 'archive' : 'unarchive');
    }

    async function groupSelectedHistoryTasks() {
        const tasks = selectedHistoryTasks();
        if (!tasks.length) return;
        const group = await showHistoryCollectionSelectDialog({
            title: '批量设置集合',
            description: `${tasks.length} 条历史任务`,
            value: '',
            confirmText: '保存集合',
        });
        if (group === null) return;
        await applyHistoryTaskIdsToCollection(historyTaskIds(selectedHistoryTasks()), group.trim(), { clearSelection: true });
    }

    async function deleteSelectedHistoryTasks() {
        const taskIds = historyTaskIds(selectedHistoryTasks());
        await deleteHistoryTasksThorough(taskIds);
    }

    async function mergeSelectedHistoryTasks() {
        const taskIds = selectedHistoryTasks()
            .filter((task) => task.job === 'training')
            .map((task) => task.id)
            .filter(Boolean);
        if (!taskIds.length) {
            await showHistoryTaskMessageDialog({
                title: '无法合并查看',
                message: '请先选择至少一个训练任务。',
                tone: 'warning',
            });
            return;
        }
        await loadConfigGroupTimeline(
            { methods_subdir: '手动选择', variant: `${taskIds.length} 个已选训练任务`, preset: 'selected' },
            { taskIds, skipSelectionDialog: true },
        );
    }

    function historyBatchDeleteUnavailable(res) {
        const message = String(res?.error || res?.message || '').trim();
        return /\b405\b|method not allowed/i.test(message);
    }

    async function deleteHistoryTasksWithLegacyEndpoint(taskIds, options = {}) {
        const ids = uniqueStringList(taskIds || []).filter(Boolean);
        if (!ids.length) return;
        if (!options.confirmed) {
            const ok = await showHistoryTaskConfirmDialog({
                title: '兼容删除历史任务',
                description: `${ids.length} 条历史记录`,
                message: '当前服务未提供批量删除预览，将使用兼容删除接口逐条删除历史记录；不会清理 WebUI 运行目录、权重、样张和缓存。',
                confirmText: '删除历史记录',
                danger: true,
            });
            if (!ok) return;
        }
        const deletedIds = [];
        const failures = [];
        for (const id of ids) {
            try {
                const res = await api(`/api/training/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
                if (!res.ok) {
                    failures.push(`${id}: ${res.error || res.message || '删除失败'}`);
                    continue;
                }
                deletedIds.push(...(res.deleted_task_ids || [id]).filter(Boolean));
            } catch (e) {
                failures.push(`${id}: ${e.message || '删除失败'}`);
            }
        }
        const touchedIds = uniqueStringList(deletedIds);
        touchedIds.forEach((id) => selectedHistoryTaskIds.delete(id));
        if (touchedIds.includes(viewingHistoryTaskId)) {
            clearHistoryManagerDetail();
        }
        if (deletedIds.length) {
            await loadTrainingHistoryList();
        }
        if (failures.length) {
            await showHistoryTaskMessageDialog({
                title: deletedIds.length ? '部分历史任务删除失败' : '删除失败',
                message: deletedIds.length
                    ? '部分历史记录已删除，其余项目未能删除。'
                    : '兼容删除接口也未能删除这些历史记录。',
                detailLines: failures,
                tone: deletedIds.length ? 'warning' : 'error',
            });
            return;
        }
        await showHistoryTaskMessageDialog({
            title: '已删除历史记录',
            message: '已通过兼容接口删除历史记录；运行目录、权重、样张和缓存没有被清理。',
            tone: 'ok',
        });
    }

    async function deleteHistoryTasksThorough(taskIds) {
        const ids = uniqueStringList(taskIds || []).filter(Boolean);
        if (!ids.length) return;
        let preview;
        try {
            preview = await api('/api/training/history/batch', {
                method: 'POST',
                body: JSON.stringify({
                    action: 'delete',
                    task_ids: ids,
                    delete_runtime_dirs: true,
                    dry_run: true,
                }),
            });
        } catch (e) {
            await showHistoryTaskMessageDialog({
                title: '读取删除预览失败',
                message: e.message,
                tone: 'error',
            });
            return;
        }
        if (!preview.ok) {
            if (historyBatchDeleteUnavailable(preview)) {
                await deleteHistoryTasksWithLegacyEndpoint(ids);
                return;
            }
            await showHistoryTaskMessageDialog({
                title: '读取删除预览失败',
                message: preview.error || '读取删除预览失败',
                tone: 'error',
            });
            return;
        }
        if ((preview.blocked || []).length) {
            await showHistoryTaskMessageDialog({
                title: '存在不能删除的任务或运行目录',
                message: '请先处理以下阻止项，再重新执行删除。',
                detailLines: preview.blocked.map((item) => `${item.id || item.path || '-'}: ${item.reason || '-'}`),
                tone: 'error',
            });
            return;
        }
        const confirmText = await showHistoryDeletePreviewDialog(preview);
        if (confirmText !== '彻底删除') return;
        try {
            const res = await api('/api/training/history/batch', {
                method: 'POST',
                body: JSON.stringify({
                    action: 'delete',
                    task_ids: ids,
                    delete_runtime_dirs: true,
                    confirm_text: confirmText,
                }),
            });
            if (!res.ok) {
                if (historyBatchDeleteUnavailable(res)) {
                    await deleteHistoryTasksWithLegacyEndpoint(ids, { confirmed: true });
                    return;
                }
                await showHistoryTaskMessageDialog({
                    title: '删除失败',
                    message: res.error || '删除失败',
                    tone: 'error',
                });
                return;
            }
            selectedHistoryTaskIds.clear();
            if (ids.includes(viewingHistoryTaskId)) {
                clearHistoryManagerDetail();
            }
            await loadTrainingHistoryList();
        } catch (e) {
            await showHistoryTaskMessageDialog({
                title: '删除失败',
                message: e.message,
                tone: 'error',
            });
        }
    }

    function showHistoryDeletePreviewDialog(preview) {
        const wrap = document.createElement('div');
        wrap.className = 'history-delete-preview';
        const summary = document.createElement('div');
        summary.className = 'history-task-dialog-message';
        const strong = document.createElement('strong');
        strong.textContent = `${preview.task_count || 0} 条历史记录 · ${preview.runtime_dir_count || 0} 个运行目录`;
        const p = document.createElement('p');
        p.textContent = '会删除历史记录目录，并删除对应运行目录内的权重、样张、日志、缓存和 runtime 配置。';
        summary.append(strong, p);
        wrap.appendChild(summary);

        const taskList = document.createElement('pre');
        taskList.textContent = [
            '# 历史记录',
            ...(preview.tasks || []).map((item) => `${item.id} · ${item.name || item.job || '-'}`),
            '',
            '# 运行目录',
            ...((preview.runtime_dirs || []).map((item) => `${item.path} · ${item.status || 'ready'}`)),
        ].join('\n');
        wrap.appendChild(taskList);

        const label = document.createElement('label');
        label.className = 'history-task-dialog-field';
        const span = document.createElement('span');
        span.textContent = '输入“彻底删除”确认';
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'history-task-dialog-input';
        input.placeholder = '彻底删除';
        label.append(span, input);
        wrap.appendChild(label);

        const sync = () => {
            const confirmBtn = document.getElementById('history-task-dialog-confirm');
            if (confirmBtn) confirmBtn.disabled = input.value.trim() !== '彻底删除';
        };
        input.addEventListener('input', sync);

        return showHistoryTaskDialog({
            title: '彻底删除历史任务',
            description: '',
            body: wrap,
            confirmText: '彻底删除',
            danger: true,
            onOpen: () => {
                input.focus();
                sync();
            },
            getValue: () => input.value.trim(),
        });
    }

    async function renameHistoryTask(task) {
        const fallback = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
        const name = await showHistoryTaskInputDialog({
            title: '重命名任务',
            description: '只修改任务列表中的显示名称，不会改动磁盘目录。',
            label: '任务名称',
            value: fallback,
            placeholder: '例如：肋骨女神 5.14 第一次训练',
            confirmText: '保存名称',
        });
        if (name === null) return;
        await updateHistoryTaskMeta(task.id, { name: name.trim() });
    }

    async function regroupHistoryTask(task) {
        const group = await showHistoryTaskInputDialog({
            title: '设置任务分组',
            description: '相同分组名的任务会在左侧任务列表中归到一起。留空表示未分组。',
            label: '分组名称',
            value: task.group || '',
            placeholder: '例如：肋骨女神 / 测试组 / 正式训练',
            confirmText: '保存分组',
        });
        if (group === null) return;
        await updateHistoryTaskMeta(task.id, { group: group.trim() });
    }

    async function archiveHistoryTask(task) {
        const archived = historyTaskIsArchived(task);
        const ok = await showHistoryTaskConfirmDialog({
            title: archived ? '取消归档任务' : '归档任务',
            description: historyTaskLabel(task),
            message: archived
                ? '取消归档后，这个任务会重新出现在默认任务列表中。'
                : '归档后默认会隐藏这个任务，可勾选“显示归档”再次查看。',
            confirmText: archived ? '取消归档' : '确认归档',
        });
        if (!ok) return;
        await updateHistoryTaskMeta(task.id, { archived: !archived });
    }

    async function deleteHistoryTask(task) {
        await deleteHistoryTasksThorough([task.id]);
    }

    async function updateHistoryTaskMeta(taskId, patch) {
        try {
            const res = await api(`/api/training/history/${encodeURIComponent(taskId)}`, {
                method: 'PATCH',
                body: JSON.stringify(patch),
            });
            if (!res.ok) {
                await showHistoryTaskMessageDialog({
                    title: '更新任务失败',
                    message: res.error || '更新任务失败',
                    tone: 'error',
                });
                return;
            }
            await loadTrainingHistoryList();
            if (viewingHistoryTaskId === taskId) {
                const payload = await api(`/api/training/history/${encodeURIComponent(taskId)}`);
                if (payload.ok) {
                    currentHistoryTaskForResume = payload.task || null;
                    renderHistoryManagerDetail(payload, { open: isHistoryDetailDialogOpen() });
                }
            }
        } catch (e) {
            await showHistoryTaskMessageDialog({
                title: '更新任务失败',
                message: e.message,
                tone: 'error',
            });
        }
    }

    function historyTaskLabel(task) {
        return historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || task.id}`;
    }

    function showHistoryTaskInputDialog(options) {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = options.value || '';
        input.placeholder = options.placeholder || '';
        input.className = 'history-task-dialog-input';

        const label = document.createElement('label');
        label.className = 'history-task-dialog-field';
        const span = document.createElement('span');
        span.textContent = options.label || '输入内容';
        label.append(span, input);

        return showHistoryTaskDialog({
            title: options.title,
            description: options.description,
            body: label,
            confirmText: options.confirmText || '确认',
            onOpen: () => {
                input.focus();
                input.select();
            },
            getValue: () => input.value,
        });
    }

    function showHistoryCollectionSelectDialog(options) {
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

    function showHistoryTaskConfirmDialog(options) {
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

    function showHistoryTaskMessageDialog(options = {}) {
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

    function showHistoryTaskDialog(options) {
        const parts = sharedHistoryTaskDialogParts();
        if (!parts) {
            return Promise.resolve(null);
        }
        const { dialog, title, desc, body, cancelBtn, confirmBtn, closeBtn, form } = parts;
        if (sharedDialogBusy || sharedHistoryTaskDialogIsOpen(dialog)) {
            return Promise.resolve(null);
        }
        sharedDialogBusy = true;

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
                sharedDialogBusy = false;
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

    function normalizeHistoryDetailTab(tab) {
        return ensureHistoryDetailFeature().normalizeHistoryDetailTab(tab);
    }

    function renderHistoryManagerDetail(payload = ensureHistoryDetailFeature().getCurrentPayload(), options = {}) {
        return ensureHistoryDetailFeature().renderHistoryManagerDetail(payload, options);
    }

    function renderHistoryDetailDialog(payload = ensureHistoryDetailFeature().getCurrentPayload(), options = {}) {
        return ensureHistoryDetailFeature().renderHistoryDetailDialog(payload, options);
    }

    function closeHistoryDetailDialog() {
        return ensureHistoryDetailFeature().closeHistoryDetailDialog();
    }

    function isHistoryDetailDialogOpen() {
        return ensureHistoryDetailFeature().isHistoryDetailDialogOpen();
    }

    function shouldRenderInlineResumePanel() {
        return historyViewMode !== 'live' && trainingViewMode === 'live';
    }

    function clearViewingHistoryTaskContext(payload = null) {
        if (payload?.mode === 'config_group') return;
        viewingHistoryTaskId = '';
        currentHistoryTaskForResume = null;
        if (historyViewMode !== 'config_group') {
            historyViewMode = 'live';
            currentHistoryConfigGroup = null;
            currentHistoryTimelineSelection = [];
        }
        renderResumePanelState();
    }

    function handleHistoryDetailWindowKeydown(event) {
        return ensureHistoryDetailFeature().handleHistoryDetailWindowKeydown(event);
    }

    function restorePreviewWorkspaceFromHistoryDetail() {
        return ensurePreviewFeature().restorePreviewWorkspaceFromHistoryDetail();
    }

    function activateHistoryDetailPreview(payload) {
        return ensurePreviewFeature().activateHistoryDetailPreview(payload);
    }

    function clearHistoryManagerDetail() {
        viewingHistoryTaskId = '';
        historyViewMode = 'live';
        currentHistoryTaskForResume = null;
        currentHistoryConfigGroup = null;
        currentHistoryTimelineSelection = [];
        ensureHistoryDetailFeature().clearHistoryDetailState();
        closeHistoryDetailDialog();
        clearResumeOptions();
        renderTrainingHistoryList();
        renderHistoryManager();
    }

    function selectedHistoryManagerResumeCheckpoint() {
        return ensureHistoryDetailFeature().selectedHistoryManagerResumeCheckpoint();
    }

    async function resumeTrainingFromHistoryDetail(queueMode) {
        return ensureHistoryDetailFeature().resumeTrainingFromHistoryDetail(queueMode);
    }

    async function loadHistoryTask(taskId, options = {}) {
        return ensureHistoryDetailFeature().loadHistoryTask(taskId, options);
    }

    async function refreshHistoryView() {
        if (historyViewMode === 'config_group' && currentHistoryConfigGroup) {
            await loadConfigGroupTimeline(currentHistoryConfigGroup, {
                taskIds: currentHistoryTimelineSelection,
                skipSelectionDialog: true,
            });
            return;
        }
        if (!viewingHistoryTaskId) return;
        await loadHistoryTask(viewingHistoryTaskId);
    }

    async function loadConfigGroupTimeline(group, options = {}) {
        if (!group?.history_group_key && (!group?.methods_subdir || !group?.variant)) return;
        const taskIds = Array.isArray(options.taskIds) ? options.taskIds.filter(Boolean) : [];
        const query = new URLSearchParams({
            methods_subdir: group.methods_subdir || '',
            variant: group.variant || '',
            preset: group.preset || 'default',
            include_archived: showArchivedHistory ? '1' : '0',
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
            historyViewMode = 'config_group';
            viewingHistoryTaskId = '';
            showTrainingView('history');
            currentHistoryConfigGroup = payload.group || group;
            currentHistoryTimelineSelection = (payload.summary?.selected_task_ids || taskIds || []).filter(Boolean);
            currentHistoryTaskForResume = null;
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

    function historyTaskStepOffset(task) {
        const resume = task?.resume_from || {};
        const step = Number(resume.checkpoint_step || 0);
        return Number.isFinite(step) && step > 0 ? step : 0;
    }

    function historyLossChartPoints(lossPoints, task) {
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

    function renderHistoryTask(payload) {
        const task = payload.task || {};
        currentHistoryTaskForResume = task;
        const banner = document.getElementById('history-view-banner');
        const bannerTitle = document.getElementById('history-view-title');
        if (banner) banner.hidden = false;
        if (bannerTitle) {
            bannerTitle.textContent = `历史任务: ${historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`} · ${historyStateLabel(task.state)}`;
        }
        setText('training-run-state', '历史');
        const stateEl = document.getElementById('training-run-state');
        if (stateEl) stateEl.className = 'training-run-state history';
        updateTrainingToolbarState('history', '历史');
        setText('training-run-title', historyTaskDisplayName(task) || '历史任务');
        setText('training-run-meta', [
            task.methods_subdir ? `方法目录 ${task.methods_subdir}` : '',
            task.variant ? `配置 ${task.variant}` : '',
            task.preset ? `预设 ${task.preset}` : '',
        ].filter(Boolean).join(' · ') || '历史任务记录');
        setText('training-run-summary', [
            task.run_dir ? `运行目录: ${task.run_dir}` : '',
            task.output_dir ? `输出: ${task.output_dir}` : '',
            task.sample_dir ? `样张: ${task.sample_dir}` : '',
        ].filter(Boolean).join(' · ') || '该任务没有记录运行目录。');
        document.getElementById('train-variant').textContent = task.variant || '-';
        document.getElementById('train-preset').textContent = task.preset || '-';
        document.getElementById('progress-bar').style.width = task.state === 'idle' ? '100%' : '0%';
        document.getElementById('progress-text').textContent = `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`;
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
        lossChart?.setXLabel?.('step');
        lossChart?.setScaleMode?.('step', {
            xRange: {
                min: chartPoints[0]?.step,
                max: chartPoints[chartPoints.length - 1]?.step,
            },
        });
        lossChart?.setData(chartPoints, { keepAll: true });
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
        renderLogOutputLines(logs.map((record) => `${record.kind === 'progress' ? '[进度] ' : ''}${record.line || ''}`));
        logEl.scrollTop = logEl.scrollHeight;
        setLogStatus(`历史 · ${(payload.logs || []).length} 行`, 'warning');

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
        if (trainingViewMode === 'history') renderHistoryManagerDetail(payload);
    }

    function renderConfigGroupTimeline(payload) {
        const group = payload.group || {};
        const summary = payload.summary || {};
        const banner = document.getElementById('history-view-banner');
        const bannerTitle = document.getElementById('history-view-title');
        if (banner) banner.hidden = false;
        if (bannerTitle) {
            bannerTitle.textContent = `合并查看: ${configGroupLabel(group)} · ${summary.task_count || 0} 次训练`;
        }
        setText('training-run-state', '合并');
        const stateEl = document.getElementById('training-run-state');
        if (stateEl) stateEl.className = 'training-run-state history';
        updateTrainingToolbarState('history', '合并');
        setText('training-run-title', `合并查看: ${configGroupLabel(group)}`);
        setText('training-run-meta', [
            group.methods_subdir ? `方法目录 ${group.methods_subdir}` : '',
            group.variant ? `配置 ${group.variant}` : '',
            group.preset ? `预设 ${group.preset}` : '',
        ].filter(Boolean).join(' · ') || '配置组训练结果');
        setText('training-run-summary', [
            `${summary.task_count || 0} 次训练`,
            `${summary.loss_count || 0} 个 Loss 点`,
            `${summary.log_count || 0} 行日志`,
            `时间: ${summary.started_at_text || '-'} → ${summary.finished_at_text || '未结束'}`,
        ].filter(Boolean).join(' · '));

        document.getElementById('train-variant').textContent = group.variant || '-';
        document.getElementById('train-preset').textContent = group.preset || '-';
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-text').textContent =
            `${summary.started_at_text || '-'} → ${summary.finished_at_text || '持续/未结束'}`;
        setMetricText('metric-vram', 'N/A');
        setMetricText('metric-vram-peak', 'N/A');
        setMetricText('metric-gpu', 'N/A');
        setMetricText('metric-gpu-peak', 'N/A');
        setMetricText('metric-temp', 'N/A');
        setMetricText('metric-temp-peak', 'N/A');
        setMetricText('metric-log-age', '分组合并');
        setMetricText('metric-rate', 'N/A');
        setEtaMetricText({ text: '分组合并', empty: false, title: '合并视图不计算单次训练预计完成时间。' });

        const metrics = payload.metrics || [];
        const lossPoints = metrics.filter((item) => item.loss !== undefined);
        lossChart?.setXLabel?.('step');
        lossChart?.setScaleMode?.('step', {
            xRange: {
                min: summary.start_display_step ?? lossPoints[0]?.display_step ?? lossPoints[0]?.step,
                max: summary.end_display_step ?? lossPoints[lossPoints.length - 1]?.display_step ?? lossPoints[lossPoints.length - 1]?.step,
            },
        });
        lossChart?.setData(lossPoints.map((item) => ({
            step: item.display_step || item.step || 0,
            loss: item.loss,
            lr: item.lr,
            rawStep: item.step,
            displayStepOffset: item.display_step_offset || 0,
            sourceTaskLabel: item.source_task_label || '',
            sourceTaskIndex: item.source_task_index || 0,
            stageBreakBefore: Boolean(item.stage_break_before),
            stageLabel: item.stage_break_before ? `任务${item.source_task_index || ''}` : '',
        })), { keepAll: true });
        syncLossChartEmptyState();
        const lastMetric = metrics[metrics.length - 1] || {};
        const lastLossMetric = lossPoints[lossPoints.length - 1] || {};
        setMetricText('metric-loss',
            lastMetric.loss !== undefined ? Number(lastMetric.loss).toFixed(5) : 'N/A');
        setMetricText('metric-lr', formatLr(lastValue(metrics, 'lr')));
        setMetricText('metric-step',
            lastMetric.display_step ?? lastMetric.step ?? lastLossMetric.display_step ?? lastLossMetric.step ?? 'N/A');

        const logs = payload.logs || [];
        const logEl = document.getElementById('log-output');
        renderLogOutputLines(logs.map(formatGroupTimelineLogRecord));
        logEl.scrollTop = logEl.scrollHeight;
        setLogStatus(`手动合并 · ${logs.length} 行日志 · ${summary.loss_count || 0} Loss 点 · 已隐藏 ${summary.progress_count || 0} 条进度记录`, 'warning');

        const health = document.getElementById('training-health');
        health.className = 'training-health ok';
        health.textContent = [
            `已手动合并 ${summary.task_count || 0} 次训练`,
            `${summary.loss_count || 0} 个 Loss 点`,
            `${summary.log_count || 0} 行日志`,
            `${summary.progress_count || 0} 条进度记录未显示`,
            summary.include_archived ? '包含归档任务' : '',
        ].filter(Boolean).join(' · ');

        const configPanel = document.getElementById('history-config-panel');
        const configTitle = document.getElementById('history-config-title');
        const configOutput = document.getElementById('history-config-output');
        if (configPanel) configPanel.hidden = false;
        if (configTitle) configTitle.textContent = '分组训练明细';
        if (configOutput) configOutput.textContent = configGroupTimelineSummary(payload);
        renderConfigGroupPaths(payload);
        renderResumePanelState();
        if (trainingViewMode === 'history') renderHistoryManagerDetail(payload);
    }

    function formatGroupTimelineLogRecord(record) {
        const taskPrefix = record.source_task_index ? `[任务${record.source_task_index}] ` : '';
        const kindPrefix = record.kind === 'progress' ? '[进度] ' : '';
        return `${taskPrefix}${kindPrefix}${record.line || ''}`;
    }

    function configGroupTimelineSummary(payload) {
        const group = payload.group || {};
        const lines = [`# 手动合并查看: ${configGroupLabel(group)}`, ''];
        for (const segment of payload.segments || []) {
            const task = segment.task || {};
            const segmentLines = [
                `任务 ${segment.index}: ${task.label || task.id || '-'}`,
                `  ID: ${task.id || '-'}`,
                `  状态: ${historyStateLabel(task.state)}`,
                `  时间: ${task.started_at_text || '-'} -> ${task.finished_at_text || '未结束'}`,
                `  输出目录: ${task.output_dir || '-'}`,
                `  真实 Step: ${formatStepRange(segment.start_display_step, segment.end_display_step)}`,
                segment.display_step_offset ? `  续训偏移: +${segment.display_step_offset}` : '',
                `  日志: ${segment.log_count || 0} 行`,
                `  进度记录: ${segment.progress_count || 0} 条`,
                `  Loss/指标: ${segment.metric_count || 0} 条`,
                '',
            ].filter(Boolean);
            lines.push(...segmentLines);
        }
        return lines.join('\n');
    }

    function formatStepRange(start, end) {
        if (start === undefined || start === null || end === undefined || end === null) return '-';
        return `${start} -> ${end}`;
    }

    function renderConfigGroupPaths(payload) {
        const group = payload.group || {};
        const summary = payload.summary || {};
        const el = document.getElementById('history-paths');
        if (!el) return;
        el.innerHTML = '';
        const items = [
            ['配置文件', configGroupLabel(group)],
            ['源配置', group.history_source_config_file || '-'],
            ['合并训练数', `${summary.task_count || 0}`],
            ['时间范围', `${summary.started_at_text || '-'} -> ${summary.finished_at_text || '未结束'}`],
            ['真实步数', formatStepRange(summary.start_display_step, summary.end_display_step)],
            ['归档任务', summary.include_archived ? '已包含' : '未包含'],
        ];
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            row.append(key, valEl);
            el.appendChild(row);
        }
    }

    function configGroupLabel(group) {
        if (group.methods_subdir === '手动选择') {
            return group.variant || '手动选择';
        }
        return group.history_run_label || group.history_group_label || group.label || `${group.methods_subdir || '-'} / ${group.variant || '-'} / ${group.preset || 'default'}`;
    }

    function metricsWithProgressFallback(metrics, logs) {
        const out = [...metrics];
        const seen = new Set(out.map(metricIdentity));
        for (const record of logs || []) {
            if (record.kind !== 'progress') continue;
            const parsed = parseMetricsFromProgressLine(record.line);
            if (!parsed) continue;
            const item = { ...parsed, ts: record.ts };
            const key = metricIdentity(item);
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(item);
        }
        out.sort((a, b) => (Number(a.ts || 0) - Number(b.ts || 0)) || (Number(a.step || 0) - Number(b.step || 0)));
        return out;
    }

    function metricIdentity(item) {
        return [
            item.step ?? '',
            item.loss != null ? Number(item.loss).toFixed(8) : '',
            item.lr != null ? Number(item.lr).toFixed(12) : '',
        ].join('|');
    }

    function returnToLiveTraining(options = {}) {
        const refresh = options.refresh !== false;
        showTrainingView('live');
        viewingHistoryTaskId = '';
        historyViewMode = 'live';
        currentHistoryConfigGroup = null;
        currentHistoryTimelineSelection = [];
        currentHistoryTaskForResume = null;
        ensureHistoryDetailFeature().clearHistoryDetailState();
        closeHistoryDetailDialog();
        clearResumeOptions();
        const banner = document.getElementById('history-view-banner');
        if (banner) banner.hidden = true;
        const resumePanel = document.getElementById('history-resume-panel');
        if (resumePanel) resumePanel.hidden = true;
        const configPanel = document.getElementById('history-config-panel');
        if (configPanel) configPanel.hidden = true;
        const configOutput = document.getElementById('history-config-output');
        if (configOutput) configOutput.textContent = '';
        const paths = document.getElementById('history-paths');
        if (paths) paths.innerHTML = '';
        document.getElementById('log-output').textContent = '';
        trainingRuntime.logBuffer = [];
        trainingRuntime.logFlushPending = false;
        trainingRuntime.lastLogId = 0;
        trainingRuntime.logLineCount = 0;
        trainingRuntime.progressCurrent = 0;
        trainingRuntime.progressTotal = 0;
        trainingRuntime.progressLabel = '';
        trainingRuntime.progressRate = '';
        trainingRuntime.progressSecondsPerStep = null;
        trainingRuntime.progressUpdatedAt = 0;
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-text').textContent = '暂无正在运行的任务目录...';
        resetLiveMetricPlaceholders();
        resetLiveSystemPeaks();
        renderLiveTrainingDashboard();
        stepCounter = 0;
        lossChart?.clear();
        syncLossChartEmptyState();
        lossChart?.setXLabel?.('step');
        lossChart?.setScaleMode?.('index');
        renderTrainingHistoryList();
        if (refresh) {
            pollStatus();
            replayTrainingLogs();
        }
    }

    async function loadResumeOptionsForTask(taskId = viewingHistoryTaskId) {
        return ensureHistoryDetailFeature().loadResumeOptionsForTask(taskId);
    }

    function clearResumeOptions() {
        return ensureHistoryDetailFeature().clearResumeOptions();
    }

    function renderResumePanelState() {
        return ensureHistoryDetailFeature().renderResumePanelState();
    }

    function selectedResumeCheckpoint() {
        return ensureHistoryDetailFeature().selectedResumeCheckpoint();
    }

    async function resumeTrainingFromCheckpoint() {
        return ensureHistoryDetailFeature().resumeTrainingFromCheckpoint();
    }

    async function queueResumeTrainingFromCheckpoint() {
        return ensureQueueFeature().queueResumeTrainingFromCheckpoint();
    }

    function setResumeStatus(text, state = '') {
        return ensureHistoryDetailFeature().setResumeStatus(text, state);
    }

    function renderHistoryPaths(task, options = {}) {
        const el = document.getElementById('history-paths');
        if (!el) return;
        el.innerHTML = '';
        const items = runtimePathItems(task, options);
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            row.append(key, valEl);
            el.appendChild(row);
        }
    }

    function runtimePathItems(task, options = {}) {
        const includeHistory = options.includeHistory !== false;
        return [
            includeHistory ? ['历史目录', task.history_dir_abs || task.history_dir] : null,
            task.training_mode === 'continue_lora' ? ['基于权重', task.continue_from_weight_abs_path] : null,
            ['本次运行目录', task.run_dir],
            ['实际运行配置', task.runtime_config_file, 'runtime-config'],
            ['原始配置副本', task.original_config_file, 'original-config'],
            ['运行时数据集配置', task.dataset_config_file, 'dataset-config'],
            ['模型缓存目录', task.model_cache_dir],
            ['数据集缓存目录', task.dataset_cache_dir],
            ['训练结果目录', task.training_output_dir || task.output_dir],
            ['样张目录', task.sample_dir],
            ['日志目录', task.logs_dir],
            includeHistory ? ['历史日志文件', task.logs_path, 'logs'] : null,
            includeHistory ? ['历史指标文件', task.metrics_path, 'metrics'] : null,
            includeHistory ? ['系统指标文件', task.system_path, 'system'] : null,
            includeHistory ? ['历史 TOML 快照', task.config_snapshot, 'config-snapshot'] : null,
        ].filter((item) => item && item[1]);
    }

    function historyArtifactUrl(task, artifactKey, options = {}) {
        const taskId = String(task?.id || '').trim();
        const key = String(artifactKey || '').trim();
        if (!taskId || !key) return '#';
        const params = new URLSearchParams();
        if (options.download) params.set('download', '1');
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return `/api/training/history/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(key)}${suffix}`;
    }

    function historyStateLabel(state) {
        return {
            running: '运行中',
            idle: '完成',
            error: '异常',
            interrupted: '已中断',
        }[state] || state || '未知';
    }

    // ── 事件绑定 ──
    function setupEventListeners() {
        const on = (id, eventName, handler) => {
            document.getElementById(id)?.addEventListener(eventName, handler);
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
        document.getElementById('btn-start-from-config').addEventListener('click', startTraining);
        document.getElementById('btn-queue-from-config').addEventListener('click', queueCurrentTrainingFromConfig);
        on('live-chart-toggle-lr', 'change', (event) => {
            liveChartState.showLr = Boolean(event.target.checked);
            renderLiveChartPanel();
        });
        on('live-chart-range', 'change', (event) => {
            liveChartState.rangeMode = event.target.value || 'all';
            renderLiveChartPanel();
        });
        document.querySelectorAll('[data-sticky-config-category]').forEach((btn) => {
            btn.addEventListener('click', () => selectConfigCategory(btn.dataset.stickyConfigCategory, { scrollToForm: true }));
        });
        window.addEventListener('resize', () => requestAnimationFrame(updateConfigStickyPlacement));
        on('btn-open-continue-lora-dialog', 'click', openContinueLoraDialog);
        on('btn-clear-continue-lora-source', 'click', clearContinueTrainingSource);
        on('btn-inspect-continue-lora-path', 'click', () => {
            selectContinueLoraWeight(document.getElementById('continue-lora-path-input')?.value || '');
        });
        on('continue-lora-history-task', 'change', (event) => {
            continueLoraDialogState.taskId = event.target.value || '';
            loadContinueLoraWeights();
        });
        on('btn-refresh-continue-lora-weights', 'click', loadContinueLoraWeights);
        on('btn-open-tutorial', 'click', openTutorialDialog);
        on('btn-stop-training', 'click', stopTraining);
        on('btn-open-queue-manager', 'click', () => showTrainingView('queue'));
        on('btn-training-queue-view', 'click', () => showTrainingView('queue'));
        on('btn-training-history-view', 'click', () => showTrainingView('history'));
        document.getElementById('btn-open-history-manager').addEventListener('click', () => showTrainingView('history'));
        ensureQueueFeature().bindQueueEvents();
        document.getElementById('btn-apply-toml').addEventListener('click', applyTomlToConfig);
        document.getElementById('btn-move-toml-group').addEventListener('click', moveCurrentTomlToGroup);
        document.getElementById('btn-create-blank-preset').addEventListener('click', createBlankPresetFromLoraTemplate);
        document.getElementById('btn-save-toml').addEventListener('click', saveTomlFile);
        document.getElementById('btn-toggle-toml-editor').addEventListener('click', toggleTomlEditorPanel);
        document.getElementById('btn-copy-toml').addEventListener('click', copyTomlEditorContent);
        document.getElementById('btn-save-toml-direct').addEventListener('click', () => saveTomlFile({ mode: 'editor' }));
        document.getElementById('btn-import-toml').addEventListener('click', importTomlFile);
        document.getElementById('btn-export-toml').addEventListener('click', exportTomlFile);
        document.getElementById('btn-save-as-toml').addEventListener('click', saveTomlAs);
        document.getElementById('btn-lock-toml').addEventListener('click', toggleTomlUserLock);
        document.getElementById('btn-delete-toml').addEventListener('click', deleteTomlFile);
        document.getElementById('btn-restore-system-toml').addEventListener('click', restoreSystemTomlPresets);
        document.getElementById('toml-import-input').addEventListener('change', handleTomlImport);
        document.getElementById('btn-toml-mode-project').addEventListener('click', () => switchTomlManagerMode('project'));
        document.getElementById('btn-toml-mode-output').addEventListener('click', () => switchTomlManagerMode('output'));
        document.getElementById('btn-refresh-output-runs').addEventListener('click', () => loadOutputRuns({ keepSelection: true }));
        document.getElementById('btn-copy-output-config').addEventListener('click', copyOutputRunConfigContent);
        document.getElementById('btn-export-output-config').addEventListener('click', exportOutputRunConfig);
        document.getElementById('btn-save-output-config-as').addEventListener('click', openOutputRunSaveAs);
        document.getElementById('btn-confirm-output-config-save-as').addEventListener('click', confirmOutputRunSaveAs);
        document.getElementById('btn-cancel-output-config-save-as').addEventListener('click', closeOutputRunSaveAs);
        document.getElementById('output-run-search').addEventListener('input', (event) => {
            outputRunState = { ...outputRunState, search: event.target.value || '' };
            renderOutputRunList();
        });
        document.getElementById('btn-new-dataset-preset').addEventListener('click', createNewDatasetPreset);
        document.getElementById('btn-copy-dataset-preset').addEventListener('click', copyDatasetPreset);
        document.getElementById('btn-rename-dataset-preset').addEventListener('click', renameDatasetPreset);
        document.getElementById('btn-import-dataset-preset').addEventListener('click', importDatasetPreset);
        document.getElementById('dataset-import-input').addEventListener('change', handleDatasetPresetImport);
        document.getElementById('btn-export-dataset-preset').addEventListener('click', exportDatasetPreset);
        document.getElementById('btn-delete-dataset-preset').addEventListener('click', deleteDatasetPreset);
        document.getElementById('btn-save-dataset-preset').addEventListener('click', saveDatasetPresetEditor);
        document.getElementById('btn-create-dataset-preset-group').addEventListener('click', createDatasetPresetGroup);
        document.getElementById('btn-refresh-dataset-presets').addEventListener('click', () => loadDatasetPresets({ selectCurrent: false, manage: true }));
        document.getElementById('dataset-preset-search').addEventListener('input', (event) => {
            datasetPresetState.search = event.target.value || '';
            renderDatasetPresetList();
        });
        document.getElementById('btn-refresh-dataset-preview').addEventListener('click', loadDatasetPreviewImages);
        document.getElementById('btn-config-dataset-dialog-refresh').addEventListener('click', () => loadDatasetPresets({ selectCurrent: false, manage: false }));
        document.getElementById('btn-config-dataset-dialog-manage').addEventListener('click', () => {
            closeConfigDatasetPickerDialog();
            document.querySelector('[data-tab="datasets"]')?.click();
        });
        document.getElementById('btn-reload-toml').addEventListener('click', async () => {
            const file = currentTomlFile || val('toml-file-select');
            if (file && (await confirmDiscardTomlChanges('当前 TOML 有未保存修改，重新读取文件会丢失这些修改。是否继续？'))) {
                loadTomlFile(file, { force: true });
            }
        });
        document.getElementById('toml-file-select').addEventListener('change', (e) => {
            selectAndApplyTomlFile(e.target.value);
        });
        document.getElementById('toml-editor').addEventListener('input', updateTomlDirtyState);
        document.getElementById('btn-clear-log').addEventListener('click', () => {
            if (isHistoryReviewMode()) return;
            document.getElementById('log-output').textContent = '';
            trainingRuntime.logBuffer = [];
            trainingRuntime.logFlushPending = false;
            trainingRuntime.logLineCount = 0;
            updateLogStatusText();
        });
        document.getElementById('btn-refresh-history').addEventListener('click', loadTrainingHistoryList);
        document.getElementById('btn-preview-training-results').addEventListener('click', openCurrentTrainingPreview);
        document.getElementById('btn-history-manager-refresh').addEventListener('click', loadTrainingHistoryList);
        document.getElementById('btn-history-collections-workbench').addEventListener('click', openHistoryCollectionsWorkbench);
        document.getElementById('btn-history-manager-merge').addEventListener('click', mergeSelectedHistoryTasks);
        document.getElementById('btn-history-bulk-archive').addEventListener('click', () => archiveSelectedHistoryTasks(true));
        document.getElementById('btn-history-bulk-unarchive').addEventListener('click', () => archiveSelectedHistoryTasks(false));
        document.getElementById('btn-history-bulk-group').addEventListener('click', groupSelectedHistoryTasks);
        document.getElementById('btn-history-bulk-delete').addEventListener('click', deleteSelectedHistoryTasks);
        document.getElementById('history-select-all').addEventListener('change', (event) => {
            const visible = historyCurrentVisibleTaskIds;
            if (event.target.checked) {
                visible.forEach((id) => selectedHistoryTaskIds.add(id));
            } else {
                visible.forEach((id) => selectedHistoryTaskIds.delete(id));
            }
            renderHistoryManager();
        });
        const historyFilterMap = {
            'history-manager-search': 'search',
            'history-filter-kind': 'kind',
            'history-filter-state': 'state',
            'history-filter-archived': 'archived',
            'history-filter-source': 'source',
            'history-sort-mode': 'sort',
        };
        for (const [id, key] of Object.entries(historyFilterMap)) {
            document.getElementById(id).addEventListener(id === 'history-manager-search' ? 'input' : 'change', (event) => {
                const value = event.target.value || historyManagerFilterDefault(key);
                historyManagerFilters[key] = value;
                renderHistoryManager();
            });
        }
        document.getElementById('history-collection-search').addEventListener('input', (event) => {
            historyCollectionSearch = event.target.value || '';
            renderHistoryManager();
        });
        document.getElementById('history-config-group-search').addEventListener('input', (event) => {
            historyConfigGroupSearch = event.target.value || '';
            renderHistoryManager();
        });
        ensureHistoryDetailFeature().bindHistoryDetailEvents();
        document.getElementById('btn-live-training').addEventListener('click', returnToLiveTraining);
        document.getElementById('btn-refresh-history-view').addEventListener('click', refreshHistoryView);
        document.getElementById('btn-close-history').addEventListener('click', returnToLiveTraining);
        document.getElementById('btn-refresh-resume-options').addEventListener('click', () => loadResumeOptionsForTask());
        document.getElementById('btn-resume-training').addEventListener('click', resumeTrainingFromCheckpoint);
        document.getElementById('btn-queue-resume-training').addEventListener('click', queueResumeTrainingFromCheckpoint);
        document.getElementById('resume-checkpoint-select').addEventListener('change', renderResumePanelState);
        document.getElementById('history-show-archived').addEventListener('change', (e) => {
            showArchivedHistory = e.target.checked;
            loadTrainingHistoryList();
        });
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            btn.addEventListener('click', () => setPreviewSource(btn.dataset.previewSource));
        });
        document.getElementById('btn-refresh-preview').addEventListener('click', loadPreviewImages);
        document.getElementById('btn-refresh-weights').addEventListener('click', loadPreviewWeights);
        document.getElementById('btn-sort-weights').addEventListener('click', togglePreviewWeightSort);
        document.getElementById('btn-save-preview-settings').addEventListener('click', savePreviewSettings);
        document.getElementById('btn-reset-preview-settings').addEventListener('click', resetPreviewSettings);
        document.getElementById('btn-close-preview-panel').addEventListener('click', closePreviewPanel);
        document.getElementById('preview-panel-dialog').addEventListener('click', (event) => {
            if (event.target === event.currentTarget) closePreviewPanel();
        });
        document.getElementById('preview-panel-dialog').addEventListener('close', restorePreviewWorkspaceAfterPanelClose);
        document.getElementById('btn-save-global-settings').addEventListener('click', saveGlobalSettings);
        document.getElementById('btn-reset-global-settings').addEventListener('click', resetGlobalSettings);
        document.querySelectorAll('.global-setting-help-toggle').forEach((btn) => {
            btn.addEventListener('click', () => toggleGlobalSettingHelp(btn));
        });
        document.getElementById('preview-training-task').addEventListener('change', (e) => changePreviewTask(e.target.value));

        setTomlManagerMode('project');
    }

    function installBeginnerTooltips() {
        const tips = {
            'method-select': '选择训练方法家族。新手通常选择 lora；LoKr、Hydra、ReFT 等属于进阶或实验方法。',
            'variant-select': '选择具体训练配置文件。它决定默认学习率、rank、缓存、方法开关等实际训练参数。',
            'preset-select': '选择预设覆盖项。default 最稳；低显存或快速试跑时再选择其他预设。',
            'gpu-picker-toggle': '选择训练时允许使用的 GPU 白名单。默认“全部 GPU”表示不限制；选择会保存在本机浏览器。',
            'btn-load-config': '重新读取当前方法、变体和预设合并后的配置；不会启动训练，也不会保存当前未保存修改。',
            'btn-start-from-config': '进入训练前预检测。预检测通过后可选择立即启动或加入队列。',
            'btn-queue-from-config': '跳过立即启动选择，把当前左侧表单对应的训练配置冻结后直接加入队列。',
            'btn-sticky-config-required': '底部配置目录快捷入口，切换到模型路径和数据集必填项。',
            'btn-sticky-config-common': '底部配置目录快捷入口，切换到训练轮数、学习率和输出等常用项。',
            'btn-sticky-config-preview': '底部配置目录快捷入口，切换到训练中样张设置。',
            'btn-open-continue-lora-dialog': '选择已有 LoRA、LoHa 或 LoKr safetensors 权重作为新训练任务的初始化来源。',
            'btn-clear-continue-lora-source': '清除继续训练来源，下一次启动会按从零开始训练。',
            'btn-inspect-continue-lora-path': '检查这个 safetensors 是否为 LoRA/LoHa/LoKr，并确认是否兼容当前变体。',
            'continue-lora-history-task': '从历史训练任务中选择一个输出目录，读取其中保存的权重文件。',
            'btn-refresh-continue-lora-weights': '重新扫描所选历史训练任务的 safetensors 权重。',
            'btn-open-tutorial': '打开基础教程，按顺序了解全局设置、数据集、配置保存、预处理和训练启动。',
            'btn-stop-training': '停止当前正在运行的训练或预处理任务；已经写出的日志、样张和权重文件会保留。',
            'btn-open-queue-manager': '打开完整队列管理视图，可筛选、调序、重试、批量取消和清理记录。',
            'btn-training-queue-view': '打开训练队列管理视图，查看等待、运行、异常、完成和已取消任务。',
            'btn-training-history-view': '切换到历史任务管理台，可筛选、批量归档、分组、彻底删除，并查看任务详情。',
            'btn-open-history-manager': '打开完整历史任务管理台，查看、筛选和管理全部训练记录。',
            'btn-refresh-queue': '重新读取训练队列状态，包括等待、运行、异常和已取消任务。',
            'btn-manager-refresh-queue': '重新读取完整训练队列状态。',
            'btn-toggle-queue-pause': '暂停或继续队列自动启动。暂停不会停止当前正在运行的任务。',
            'btn-manager-toggle-queue-pause': '暂停或继续队列自动启动。失败策略为暂停时，异常任务会自动触发暂停。',
            'btn-cancel-all-queue': '停止当前运行中的队列任务并取消所有等待任务；不会删除历史记录或运行文件。',
            'btn-cancel-waiting-queue': '取消所有等待中的队列任务，不影响运行中任务和历史记录。',
            'btn-clear-completed-queue': '只清理队列里的已完成记录；不会删除历史、运行目录、缓存或实际文件。',
            'btn-clear-canceled-queue': '只清理队列里的已取消记录；不会删除历史、运行目录、缓存或实际文件。',
            'training-queue-failure-policy': '选择队列任务失败后的默认策略。建议暂停队列，确认后再继续。',
            'btn-refresh-resume-options': '重新扫描这个历史任务输出目录里的训练状态目录，例如 output_name-checkpoint-state。',
            'btn-resume-training': '从选中的训练状态目录恢复训练。它不是加载普通权重热启动，而是恢复 optimizer、scheduler、随机状态和步数。',
            'btn-queue-resume-training': '把选中的续训状态加入队列，等待当前任务结束后自动启动。',
            'resume-checkpoint-select': '只有包含 train_state.json 的状态目录才会出现在这里；普通 safetensors 权重不能完整恢复训练进度。',
            'btn-import-toml': '从本地选择 TOML 文件导入到 WebUI 管理区；导入后仍需要加载或保存为配置才能训练。',
            'btn-export-toml': '导出当前选中的单个 TOML 内容；如果编辑器里有未保存修改，会按当前编辑器内容导出。',
            'btn-save-toml': '保存左侧表单或当前 TOML 的未保存修改。保存后，“开始训练”才会使用这些新值。',
            'btn-toggle-toml-editor': '展开二级界面，查看、复制或直接编辑当前 TOML；适合批量改字段。',
            'btn-save-as-toml': '把当前配置另存为新 TOML，适合从系统预设复制出自己的可编辑版本。',
            'btn-apply-toml': '把右侧选中的 TOML 加载到左侧表单，并设为当前训练入口。',
            'btn-move-toml-group': '移动右侧配置文件所在分组，只改变列表归类，不改变 TOML 内容。',
            'btn-create-blank-preset': '以 LoRA 标准训练变体 lora.toml 为模板，并套用全局基础模型路径创建新的可编辑项目预设。',
            'btn-fill-global-model-paths': '用全局设置里的基础 DiT、Qwen3、VAE 路径覆盖当前配置表单；覆盖前会要求确认。',
            'btn-reload-toml': '从磁盘重新读取当前 TOML；会丢弃未保存编辑。',
            'btn-copy-toml': '复制当前编辑器里的 TOML 内容，方便备份或排查。',
            'btn-save-toml-direct': '保存直接编辑器里的 TOML 文本。需要连续点击两次确认写入。',
            'btn-lock-toml': '锁定当前配置文件，防止误改；系统预设或分组锁定的文件可能无法手动解锁。',
            'btn-delete-toml': '删除当前选中的可编辑 TOML。需要二次确认；不会删除训练输出目录。',
            'btn-restore-system-toml': '把项目内置系统预设恢复到项目版本。会自动备份，但不影响用户导入配置。',
            'btn-toml-mode-project': '管理 configs 下的项目预设，可编辑、另存、分组和锁定。',
            'btn-toml-mode-output': '查看全局输出文件夹里的训练快照配置；只读，可复制为新的项目预设后编辑。',
            'btn-refresh-output-runs': '重新扫描全局输出文件夹下的训练运行目录。',
            'btn-copy-output-config': '复制当前只读训练快照 TOML 内容。',
            'btn-export-output-config': '导出当前只读训练快照 TOML。',
            'btn-save-output-config-as': '把当前运行目录的原始配置复制到项目预设中，再切回项目预设编辑。',
            'output-run-search': '按运行目录名、时间或配置文件路径筛选训练输出配置。',
            'btn-live-training': '从历史任务视图回到当前正在监控的训练/预处理状态。',
            'btn-refresh-history': '重新读取训练任务历史列表，包括日志、loss、输出目录和样张目录记录。',
            'btn-preview-training-results': '打开当前训练或最新运行目录预览，查看样张和权重；历史列表中的“任务预览”和“分组预览”会进入详情里的样张与权重页。',
            'btn-history-manager-refresh': '重新读取历史任务管理台数据。',
            'btn-history-collections-workbench': '打开集合管理大界面，以配置分组为单位整理历史任务。',
            'btn-history-manager-merge': '合并查看选中的训练任务，预处理任务不会参与合并。',
            'btn-history-bulk-archive': '把选中的历史任务批量归档，默认列表会隐藏它们。',
            'btn-history-bulk-unarchive': '把选中的历史任务批量取消归档。',
            'btn-history-bulk-group': '给选中的历史任务设置同一个集合名称；同配置文件自动分组会继续保留。',
            'btn-history-bulk-delete': '彻底删除选中的历史记录和 WebUI 运行目录，需要输入确认文本。',
            'history-manager-search': '按任务名、配置、目录、消息等字段搜索历史任务。',
            'history-filter-kind': '按训练或预处理任务筛选。',
            'history-filter-state': '按完成、异常、中断或运行状态筛选。',
            'history-filter-archived': '筛选未归档、已归档或全部历史任务。',
            'history-filter-source': '按队列、续训或继续训练来源筛选。',
            'history-sort-mode': '调整历史任务排序方式。',
            'btn-refresh-history-view': '重新读取当前正在查看的历史日志和 Loss；适合训练仍在写日志时手动更新。',
            'btn-merge-config-group-history': '按同一个配置文件分组合并查看训练日志和 Loss 曲线；预处理任务不会参与合并。',
            'btn-clear-log': '清空当前页面显示的日志文本；不会删除磁盘上的历史日志。',
            'history-show-archived': '显示已归档任务。归档只是隐藏列表项，不会删除训练记录。',
            'btn-refresh-preview': '重新扫描当前预览来源目录，读取最新生成的样张图片。',
            'btn-refresh-weights': '重新扫描选中训练任务的权重文件，显示保存轮次和步数。',
            'btn-sort-weights': '按 Epoch/Step 切换权重文件正序或反序排列。',
            'btn-save-preview-settings': '保存预览图路径设置，只影响预览结果工作区读取目录，不会改训练配置。',
            'btn-reset-preview-settings': '恢复预览图目录默认值，例如旧版训练样张兼容目录 output/ckpt/sample。',
            'btn-save-global-settings': '保存 Web 训练输出根目录。每次训练都会在这里创建独立运行目录。',
            'btn-reset-global-settings': '恢复 Web 训练输出根目录默认值 output/runs。',
            'global-output-root': 'Web 训练输出根目录。支持项目相对路径或绝对路径。',
            'global-pretrained-model-path': '新建空白预设时默认写入的基础 DiT 模型路径；单个配置仍可覆盖。',
            'global-qwen3-path': '新建空白预设时默认写入的 Qwen3 文本编码器路径；单个配置仍可覆盖。',
            'global-vae-path': '新建空白预设时默认写入的 VAE 路径；单个配置仍可覆盖。',
            'preview-training-task': '选择一个历史训练任务后，预览图会读取该任务记录的 sample_dir；不选时会优先看当前任务和最新运行目录。',
            'preview-training-dir': '训练中采样的兼容兜底目录；新 Web 运行通常会优先读取全局输出根目录下的最新运行目录。',
            'preview-inference-dir': '推理预览来源目录，通常存放手动推理或测试生成的图片。',
            'preview-custom-dir': '自定义预览目录。填任意项目内或绝对路径后，可在“自定义路径”来源中查看图片。',
            'btn-new-dataset-preset': '新建一个 configs/datasets 下的数据集预设。',
            'btn-copy-dataset-preset': '把当前数据集预设复制成可编辑的新文件。',
            'btn-rename-dataset-preset': '重命名当前数据集预设，会保留图片和缓存目录不变。',
            'btn-import-dataset-preset': '导入外部 TOML 为新的数据集预设。',
            'btn-export-dataset-preset': '导出当前数据集预设 TOML。',
            'btn-delete-dataset-preset': '只删除当前数据集预设 TOML，不删除图片或缓存目录。',
            'btn-save-dataset-preset': '保存当前数据集预设编辑器中的路径和蓝图参数。',
            'btn-create-dataset-preset-group': '新建一个左侧数据集分组，只改变管理归类，不改变 TOML 内容。',
            'btn-refresh-dataset-presets': '重新读取 configs/datasets 和左侧分组状态。',
            'dataset-preset-search': '按预设名称、路径或第一组原始目录过滤左侧数据集预设。',
            'btn-refresh-dataset-preview': '重新扫描当前数据集路径，读取最新图片和同名 caption 标注。',
            'btn-config-dataset-dialog-refresh': '重新读取可选的数据集预设列表，并保留当前配置页的选择状态。',
            'btn-config-dataset-dialog-manage': '切换到数据集页，编辑或新增可复用的数据集预设。',
        };
        for (const [id, title] of Object.entries(tips)) {
            const el = document.getElementById(id);
            if (el && !el.title) el.title = title;
        }
        document.querySelectorAll('.tab-btn').forEach((btn) => {
            const labels = {
                config: '配置页：选择方法/变体/预设，编辑训练参数并引用数据集预设。',
                datasets: '数据集页：管理可复用的多数据集预设。',
                training: '训练页：查看当前任务、历史任务、loss 曲线、日志和显存状态。',
                settings: '全局设置页：设置 Web 训练输出根目录和新建预设默认模型路径。',
            };
            const key = btn.dataset.tab;
            if (labels[key]) btn.title = labels[key];
        });
        document.querySelectorAll('.preview-source-btn').forEach((btn) => {
            const labels = {
                training: '读取训练任务的 sample_dir，或优先读取最新 Web 运行目录里的训练样张。',
                inference: '读取推理预览目录，适合查看手动测试生成图。',
                custom: '读取你填写的自定义目录，适合临时检查任意图片文件夹。',
            };
            const key = btn.dataset.previewSource;
            if (labels[key]) btn.title = labels[key];
        });
    }

    // ── 工具函数 ──
    async function api(url, opts = {}) {
        return ctx.api(url, opts);
    }

    function val(id) {
        return ctx.dom.val(id);
    }

    function populateSelect(id, items, preferred = '') {
        ctx.dom.populateSelect(id, items, preferred);
    }
    })();
}
