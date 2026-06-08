/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;


    // ── 状态 ──
    globalThis.fieldHelp = {};
    globalThis.currentConfig = {};
    globalThis.ws = null;
    globalThis.lossChart = null;
    globalThis.stepCounter = 0;
    globalThis.trainingStatusPollFailures = 0;
    globalThis.tomlStatusTimer = null;
    globalThis.tomlFiles = [];
    globalThis.tomlFileGroups = [];
    globalThis.tomlFileMeta = {};
    globalThis.currentTomlFile = '';
    globalThis.tomlSavedContent = '';
    globalThis.tomlDeleteConfirmFile = '';
    globalThis.tomlDeleteConfirmTimer = null;
    globalThis.tomlSaveConfirmFile = '';
    globalThis.tomlSaveConfirmTimer = null;
    globalThis.tomlManagerMode = 'project';
    globalThis.configSwitchToastTimer = null;
    globalThis.sharedDialogBusy = false;
    globalThis.tomlGroupActionBusy = false;
    globalThis.fileGroupDragState = null;
    globalThis.fileGroupPointerDrag = null;
    globalThis.fileGroupDropPreviewElement = null;
    globalThis.fileGroupActiveDropTargetNode = null;
    globalThis.fileGroupActiveDropPosition = '';
    globalThis.datasetEditorDragState = null;
    globalThis.datasetEditorPointerDrag = null;
    globalThis.fileGroupDropTargets = new WeakMap();
    globalThis.fileGroupDropTargetNodes = new Set();
    globalThis.FILE_GROUP_DROP_TARGET_ATTR = 'data-file-group-drop-target';
    globalThis.configLoadSeq = 0;
    globalThis.datasetLoadSeq = 0;
    globalThis.stepEstimateSeq = 0;
    globalThis.samplePromptsLoadSeq = 0;
    globalThis.datasetPresetLoadSeq = 0;
    globalThis.datasetPreviewLoadSeq = 0;
    globalThis.configGroupHintSeq = 0;
    globalThis.configFormState = {
        activeCategory: 'required',
        showAdvanced: false,
        search: '',
        expandedGroups: new Set(),
        collapsedGroups: new Set(),
        draftValues: new Map(),
    };
    globalThis.RESOURCE_QUICK_PRESETS = [
        {
            id: 'gpu_full',
            label: '全 GPU',
            note: '显存充足优先；最快，不做 block swap。',
            values: {
                blocks_to_swap: 0,
                block_swap_transfer_dtype: 'bf16',
                selective_checkpoint: 'off',
                block_swap_profile_jsonl: 'off',
                memory_probe_jsonl: 'off',
                memory_probe_max_steps: 2,
                gradient_checkpointing: false,
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
                memory_probe_jsonl: 'off',
                memory_probe_max_steps: 2,
                gradient_checkpointing: false,
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
                memory_probe_jsonl: 'auto',
                memory_probe_max_steps: 2,
                gradient_checkpointing: false,
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
                gradient_checkpointing: false,
                unsloth_offload_checkpointing: false,
                torch_compile: true,
            },
        },
        {
            id: 'lokr_16g_rescue',
            label: 'LoKr 16G',
            note: 'LoKr 专用；实测交换 23 块，分组 8 作为当前速度默认。',
            values: {
                blocks_to_swap: 23,
                block_swap_transfer_dtype: 'bf16',
                selective_checkpoint: 'off',
                block_swap_profile_jsonl: 'auto',
                memory_probe_jsonl: 'auto',
                memory_probe_max_steps: 3,
                lokr_factor_group_size: 8,
                lokr_project_chunk_bytes: 4194304,
                gradient_checkpointing: false,
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
                gradient_checkpointing: false,
                unsloth_offload_checkpointing: false,
                torch_compile: true,
            },
            merge: {
                blocks_to_swap: 'max',
                selective_checkpoint: 'checkpoint_strength_max',
            },
        },
    ];
    globalThis.SELECTIVE_CHECKPOINT_STRENGTH = new Map([
        ['off', 0],
        ['peak_blocks_mlp_layer1', 1],
        ['mlp_layer1_only', 2],
        ['peak_blocks_mlp', 3],
        ['mlp_only', 4],
        ['every_other', 5],
    ]);
    globalThis.datasetCaptionSourceHelpSeq = 0;
    globalThis.choiceGuideHintSeq = 0;
    globalThis.selectionSnapshot = {
        method: '',
        variant: '',
        preset: '',
    };
    globalThis.currentStepEstimate = null;
    globalThis.datasetEditorState = {
        loading: false,
        loaded: false,
        dirty: false,
        dataset_config: '',
        datasets: [],
        defaults: {},
        error: '',
    };
    globalThis.datasetPresetState = {
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
    globalThis.datasetPreviewState = {
        datasetIndex: 0,
        source: 'source',
        payload: null,
    };
    globalThis.HIDDEN_DATASET_PRESET_FILES = new Set([
        'configs/datasets/easycontrol.toml',
        'configs/datasets/ip_adapter.toml',
    ]);
    globalThis.DATASET_PRESET_REQUEST_TIMEOUT_MS = 15000;
    globalThis.DATASET_PRESET_GROUP_STATE_KEY = 'anima_lora_dataset_preset_groups_v2';
    globalThis.selectedConfigDatasetFile = '';
    globalThis.selectedConfigDatasetSummary = null;
    globalThis.outputRunState = {
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
    globalThis.configDatasetPickerSearch = '';
    globalThis.configDatasetPreviewRequestSeq = 0;
    globalThis.configDatasetPreviewState = {
        file: '',
        loading: false,
        payload: null,
        error: '',
    };
    globalThis.DEFAULT_SAMPLE_PROMPTS_PATH = 'configs/sample_prompts.txt';
    globalThis.samplePromptsPath = DEFAULT_SAMPLE_PROMPTS_PATH;
    globalThis.samplePromptsContent = '';
    globalThis.samplePromptsMode = 'editor-inline';
    globalThis.viewingHistoryTaskId = '';
    globalThis.historyViewMode = 'live';
    globalThis.currentHistoryTaskForResume = null;
    globalThis.currentHistoryConfigGroup = null;
    globalThis.currentHistoryTimelineSelection = [];
    globalThis.STAGE_RESOLUTION_STEPS_PER_EPOCH = 1000;
    globalThis.stageResolutionState = {
        enabled: false,
        selectedIndex: 0,
        stages: [
            { name: 'EP1', epochs: 1, maxSide: 1024, downRange: 256, manualRepeats: false, repeats: 1 },
            { name: 'EP2', epochs: 1, maxSide: 1536, downRange: 512, manualRepeats: false, repeats: 1 },
        ],
    };
    globalThis.liveChartState = {
        showLr: true,
        rangeMode: 'all',
    };
    globalThis.continueTrainingSource = null;
    globalThis.continueLoraDialogState = {
        loading: false,
        taskId: '',
        weights: [],
        error: '',
        message: '',
    };
    globalThis.trainingViewMode = 'live';
    globalThis.historyTasks = [];
    globalThis.showArchivedHistory = false;
    globalThis.selectedHistoryTaskIds = new Set();
    globalThis.historyManagerFilters = {
        search: '',
        kind: 'all',
        state: 'all',
        archived: 'active',
        source: 'all',
        sort: 'newest',
    };
    globalThis.historyCollectionWorkbenchTarget = '';
    globalThis.historyCollectionSettings = {
        collection_order: [],
        config_group_order: {},
    };
    globalThis.historyCollectionSearch = '';
    globalThis.historyConfigGroupSearch = '';
    globalThis.HISTORY_UNGROUPED_COLLECTION_KEY = 'collection:__ungrouped__';
    globalThis.selectedHistoryCollectionKey = HISTORY_UNGROUPED_COLLECTION_KEY;
    globalThis.historyCurrentVisibleTaskIds = [];
    globalThis.HISTORY_TASK_DRAG_MIME = 'application/x-anima-history-task-ids';
    globalThis.HISTORY_COLLECTION_DRAG_MIME = 'application/x-anima-history-collection';
    globalThis.HISTORY_CONFIG_GROUP_DRAG_MIME = 'application/x-anima-history-config-group';
    globalThis.historyDragState = {
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
    globalThis.historyCollectionDragState = {
        active: false,
        sourceValue: '',
        activeDropTarget: '',
        dropPosition: 'after',
        pending: false,
    };
    globalThis.historyConfigGroupSortState = {
        active: false,
        sourceKey: '',
        collectionKey: '',
        activeDropTarget: '',
        dropPosition: 'after',
        pending: false,
    };
    globalThis.historyConfigGroupPointerDrag = null;
    globalThis.historyCollectionPointerDrag = null;
    globalThis.historyDragImageElement = null;
    globalThis.historyConfigGroupDropPreviewElement = null;
    globalThis.historyDropPopoverOutsideHandler = null;
    globalThis.historyDropFeedback = { message: '', tone: '' };
    globalThis.historyDropFeedbackTimer = null;
    globalThis.THEME_STORAGE_KEY = 'anima_lora_theme';
    globalThis.GPU_WHITELIST_STORAGE_KEY = 'anima_lora_gpu_whitelist';
    globalThis.currentTrainingSource = {
        method: 'lora',
        methods_subdir: 'gui-methods',
        file: 'configs/gui-methods/lora.toml',
    };
    Object.assign(globalThis, ctx.catalog);
    globalThis.LOSS_WEIGHTING_DEPENDENT_FIELDS = new Map([
        ['min_snr_gamma', 'min_snr'],
        ['p2_gamma', 'p2'],
        ['p2_k', 'p2'],
    ]);
    globalThis.datasetExperimentalScopeSelections = new Map();
    globalThis.trainingRuntime = {
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
        lastTerminalMessage: '',
        lastTerminalHint: '',
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

    globalThis.isLiveRunningState = function isLiveRunningState(state = trainingRuntime.state) {
        return state === 'running' || state === 'compiling';
    }
    globalThis.globalSettings = null;
    globalThis.previewFeature = null;
    globalThis.queueFeature = null;
    globalThis.historyDetailFeature = null;
    globalThis.weightAnalysisFeature = null;
    globalThis.ensureWeightAnalysisFeature = function ensureWeightAnalysisFeature() {
        if (weightAnalysisFeature) return weightAnalysisFeature;
        weightAnalysisFeature = createWeightAnalysisFeature(ctx);
        return weightAnalysisFeature;
    }

    globalThis.ensureQueueFeature = function ensureQueueFeature() {
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
            selectedGpuPayload: () => gpuPicker.selectedGpuPayload(),
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
    globalThis.ensurePreviewFeature = function ensurePreviewFeature() {
        if (previewFeature) return previewFeature;
        previewFeature = createPreviewFeature(ctx, {
            getHistoryTasks: () => historyTasks,
            getShowArchivedHistory: () => showArchivedHistory,
            loadTrainingHistoryList,
            loadHistoryTask,
            loadConfigGroupTimeline,
            showTrainingView,
            getTrainingViewMode: () => trainingViewMode,
            getViewingHistoryTaskId: () => viewingHistoryTaskId,
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

    globalThis.makeHistoryArtifactUrl = function makeHistoryArtifactUrl(task, artifactKey, options = {}) {
        const taskId = String(task?.id || '').trim();
        const key = String(artifactKey || '').trim();
        if (!taskId || !key) return '#';
        const params = new URLSearchParams();
        if (options.download) params.set('download', '1');
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return `/api/training/history/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(key)}${suffix}`;
    }
