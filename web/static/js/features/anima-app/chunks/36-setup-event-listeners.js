/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;
const SETUP_EVENT_DOM_CONTRACT = Object.freeze({
    required: Object.freeze([
        'method-select',
        'variant-select',
        'preset-select',
        'btn-load-config',
        'btn-start-from-config',
        'btn-queue-from-config',
        'btn-save-toml',
        'toml-file-select',
        'toml-editor',
    ]),
    optional: Object.freeze([
        'live-chart-toggle-lr',
        'live-chart-range',
        'btn-open-continue-lora-dialog',
        'btn-clear-continue-lora-source',
        'config-full-resume-task-select',
        'config-full-resume-checkpoint-select',
        'btn-refresh-config-full-resume',
        'btn-inspect-continue-lora-path',
        'continue-lora-history-task',
        'btn-refresh-continue-lora-weights',
        'btn-open-tutorial',
        'btn-stop-training',
        'btn-open-queue-manager',
        'btn-training-queue-view',
        'btn-training-history-view',
        'btn-open-history-manager',
        'btn-apply-toml',
        'btn-move-toml-group',
        'btn-create-blank-preset',
        'btn-toggle-toml-editor',
        'btn-copy-toml',
        'btn-save-toml-direct',
        'btn-import-toml',
        'btn-export-toml',
        'btn-save-as-toml',
        'btn-lock-toml',
        'btn-delete-toml',
        'btn-restore-system-toml',
        'toml-import-input',
        'btn-toml-mode-project',
        'btn-toml-mode-output',
        'btn-refresh-output-runs',
        'btn-copy-output-config',
        'btn-export-output-config',
        'btn-save-output-config-as',
        'btn-confirm-output-config-save-as',
        'btn-cancel-output-config-save-as',
        'output-run-search',
        'btn-new-dataset-preset',
        'btn-copy-dataset-preset',
        'btn-rename-dataset-preset',
        'btn-import-dataset-preset',
        'dataset-import-input',
        'btn-export-dataset-preset',
        'btn-delete-dataset-preset',
        'btn-save-dataset-preset',
        'btn-create-dataset-preset-group',
        'btn-refresh-dataset-presets',
        'dataset-preset-search',
        'btn-refresh-dataset-preview',
        'btn-config-dataset-dialog-refresh',
        'btn-config-dataset-dialog-manage',
        'btn-reload-toml',
        'btn-clear-log',
        'btn-refresh-history',
        'btn-preview-training-results',
        'btn-live-sampling-preview',
        'btn-history-manager-refresh',
        'btn-history-collections-workbench',
        'btn-history-manager-merge',
        'btn-history-bulk-archive',
        'btn-history-bulk-unarchive',
        'btn-history-bulk-group',
        'btn-history-bulk-delete',
        'history-select-all',
        'history-manager-search',
        'history-filter-kind',
        'history-filter-state',
        'history-filter-archived',
        'history-filter-source',
        'history-sort-mode',
        'history-collection-search',
        'history-config-group-search',
        'btn-live-training',
        'btn-refresh-history-view',
        'btn-close-history',
        'btn-refresh-resume-options',
        'btn-resume-training',
        'btn-queue-resume-training',
        'resume-checkpoint-select',
        'history-show-archived',
        'btn-refresh-preview',
        'btn-refresh-weights',
        'btn-sort-weights',
        'btn-save-preview-settings',
        'btn-reset-preview-settings',
        'btn-close-preview-panel',
        'preview-panel-dialog',
        'btn-save-global-settings',
        'btn-reset-global-settings',
        'global-ui-scale',
        'preview-training-task',
    ]),
});
const REQUIRED_SETUP_EVENT_DOM_IDS = new Set(SETUP_EVENT_DOM_CONTRACT.required);
globalThis.SETUP_EVENT_DOM_CONTRACT = SETUP_EVENT_DOM_CONTRACT;

    globalThis.setupEventListeners = function setupEventListeners() {
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
            continueLoraDialogState.taskId = event.target.value || '';
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
        ensureWeightAnalysisFeature().bindWeightAnalysisEvents();
        ensureEnvironmentCheckFeature().bindEnvironmentCheckEvents();
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
            outputRunState = { ...outputRunState, search: event.target.value || '' };
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
            datasetPresetState.search = event.target.value || '';
            renderDatasetPresetList();
        });
        on('btn-refresh-dataset-preview', 'click', loadDatasetPreviewImages);
        on('btn-config-dataset-dialog-refresh', 'click', () => loadDatasetPresets({ selectCurrent: false, manage: false }));
        on('btn-config-dataset-dialog-manage', 'click', () => {
            closeConfigDatasetPickerDialog();
            document.querySelector('[data-tab="datasets"]')?.click();
        });
        on('btn-reload-toml', 'click', async () => {
            const file = currentTomlFile || val('toml-file-select');
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
            trainingRuntime.logBuffer = [];
            trainingRuntime.logFlushPending = false;
            trainingRuntime.logLineCount = 0;
            updateLogStatusText();
        });
        on('btn-refresh-history', 'click', loadTrainingHistoryList);
        on('btn-preview-training-results', 'click', openCurrentTrainingPreview);
        on('btn-live-sampling-preview', 'click', openLiveSamplingPreview);
        on('btn-history-manager-refresh', 'click', loadTrainingHistoryList);
        on('btn-history-collections-workbench', 'click', openHistoryCollectionsWorkbench);
        on('btn-history-manager-merge', 'click', mergeSelectedHistoryTasks);
        on('btn-history-bulk-archive', 'click', () => archiveSelectedHistoryTasks(true));
        on('btn-history-bulk-unarchive', 'click', () => archiveSelectedHistoryTasks(false));
        on('btn-history-bulk-group', 'click', groupSelectedHistoryTasks);
        on('btn-history-bulk-delete', 'click', deleteSelectedHistoryTasks);
        on('history-select-all', 'change', (event) => {
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
            on(id, id === 'history-manager-search' ? 'input' : 'change', (event) => {
                const value = event.target.value || historyManagerFilterDefault(key);
                historyManagerFilters[key] = value;
                renderHistoryManager();
            });
        }
        on('history-collection-search', 'input', (event) => {
            historyCollectionSearch = event.target.value || '';
            renderHistoryManager();
        });
        on('history-config-group-search', 'input', (event) => {
            historyConfigGroupSearch = event.target.value || '';
            renderHistoryManager();
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
            showArchivedHistory = e.target.checked;
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

    globalThis.installBeginnerTooltips = function installBeginnerTooltips() {
        const tips = {
            'method-select': '选择训练方法家族。新手通常选择 lora；LoKr、Hydra、ReFT 等属于进阶或实验方法。',
            'variant-select': '选择具体训练配置文件。它决定默认学习率、rank、缓存、方法开关等实际训练参数。',
            'preset-select': '选择预设覆盖项。default 最稳；低显存或快速试跑时再选择其他预设。',
            'gpu-picker-toggle': '选择训练时允许使用的 GPU 白名单。默认“全部 GPU”表示不限制；选择会保存在本机浏览器。',
            'btn-load-config': '重新读取当前方法、变体和预设合并后的配置；不会启动训练，也不会保存当前未保存修改。',
            'btn-start-from-config': '按当前训练来源方案启动。完整续训会使用历史任务冻结配置，权重热启动会先审查权重。',
            'btn-queue-from-config': '按当前训练来源方案加入队列。完整续训会在真正启动前再次检查 checkpoint-state。',
            'btn-sticky-config-required': '底部配置目录快捷入口，切换到模型路径和数据集必填项。',
            'btn-sticky-config-common': '底部配置目录快捷入口，切换到训练轮数、学习率和输出等常用项。',
            'btn-sticky-config-preview': '底部配置目录快捷入口，切换到训练中样张设置。',
            'btn-sticky-config-optimization': '底部配置目录快捷入口，切换到显存、速度、block swap 和 LoKr 专用优化项。',
            'btn-open-continue-lora-dialog': '选择已有 LoRA、LoHa、LoKr 或 GLoRA safetensors 权重作为权重热启动来源。',
            'btn-clear-continue-lora-source': '清除权重热启动来源，下一次启动会按从零训练。',
            'btn-inspect-continue-lora-path': '检查这个 safetensors 是否为 LoRA/LoHa/LoKr/GLoRA，并确认是否兼容当前变体。',
            'config-full-resume-task-select': '选择一个历史训练任务作为完整续训来源；完整续训只使用它的冻结配置快照。',
            'config-full-resume-checkpoint-select': '选择包含 train_state.json 的 checkpoint-state；已达到目标步数的检查点不能启动。',
            'btn-refresh-config-full-resume': '重新审查历史任务的 checkpoint-state、train_state.json 和剩余步数。',
            'continue-lora-history-task': '从历史训练任务中选择一个输出目录，读取其中保存的权重文件。',
            'btn-refresh-continue-lora-weights': '重新扫描所选历史训练任务的 safetensors 权重。',
            'btn-open-tutorial': '打开基础教程，按顺序了解全局设置、数据集、配置保存、预处理和训练启动。',
            'weight-analysis-select': '从训练输出目录或全局输出目录选择 .safetensors 权重，只做静态 ΔW 分析。',
            'weight-analysis-path': '也可以手填权重路径；为安全起见，只允许训练输出目录或全局输出目录下的 .safetensors。',
            'weight-analysis-dropzone': '拖入 .safetensors 文件会临时上传到后端做 CPU 静态分析；拖入 file:// 或文本路径会自动填入路径框。',
            'weight-analysis-compare-path': '对比模式下的第二个权重路径；会与主权重 A 做 B - A 静态能量差值。',
            'weight-analysis-compare-dropzone': '拖入第二个 .safetensors 权重作为 B，不写入权重目录。',
            'btn-toggle-weight-compare': '开启后可分析两个 LoRA 权重的层类型和 block 静态能量差异。',
            'btn-export-weight-analysis': '打开浏览器打印导出，可在系统对话框中保存为 PDF 报告。',
            'btn-export-weight-analysis-json': '把当前 ΔW 分析结果导出为机器易读 JSON 报告；对比模式会额外包含 B - A 差异摘要。',
            'btn-refresh-analysis-weights': '重新扫描当前可读取的训练权重列表，不会加载模型。',
            'btn-run-weight-analysis': '在 CPU 上读取 safetensors 并计算 ΔW 范数，不跑图、不占 GPU。',
            'btn-refresh-image-test-status': '重新读取生图测试状态、最新日志以及 output/tests 的图片结果。',
            'btn-refresh-image-test-weights': '重新扫描训练输出目录中的 .safetensors，方便选当前 LoRA 直接试图。',
            'image-test-weight-select': '从训练输出目录中挑一个权重；也可以不选，直接跑基础模型。',
            'image-test-weight-path': '手填权重路径时，会走和 ΔW 分析相同的安全路径校验。',
            'image-test-runtime-dtype': '切换 DiT、latent 和 VAE 的运行精度；默认跟随“优化”里的精度倾向。',
            'image-test-text-encoder-dtype': '切换 Qwen3 文本编码器精度；“跟随推理精度”表示与上方保持一致。',
            'image-test-gpu-index': '选择本次生图测试实际使用的单张 GPU；会写入子进程 CUDA_VISIBLE_DEVICES，并持久化保存在本机浏览器。',
            'image-test-history-filter': '右侧历史图区默认只看近 7 天结果；切到近 14 天、近 30 天或全部后，会按日期自动分组。',
            'btn-image-test-export-merged': '把当前选中的多张结果图拼成一张对比网格图，方便发群、做对比或归档。',
            'btn-image-test-export-originals': '把当前选中的结果图直接打包成一个 zip 下载，适合批量带走原图原件。',
            'btn-image-test-clear-selection': '清空右侧图库里的当前选择，不会删除任何图片文件。',
            'btn-open-image-test-layer-dialog': '打开 anima 分层编辑器，在弹窗里逐层控制 LoRA 是否启用以及每层倍率。',
            'image-test-layer-dialog': '居中的层编辑弹窗；这里的层位倍率会真实进入本次 GPU 推理加载链。',
            'image-test-layer-enable': '打开后会按下方层位过滤 LoRA safetensors，再进入真实推理加载链。',
            'btn-image-test-layer-layout-toggle': '切换层位列表的排布密度，在一行一列和一行二列之间切换；双列更紧凑，单列更便于逐层精调。',
            'image-test-layer-preset': '先用预设快速铺一组初始层倍率；手动改任意一层后会自动切成 Custom。',
            'image-test-layer-selection': '这里是 39 个层位的逐层编辑区，每层都能独立启停，并用滑条或手输设置倍率。',
            'image-test-layer-io-text': '这里可以粘贴或微调分层参数 JSON；应用导入后会立刻回写到下方各层控件。',
            'btn-image-test-layer-export': '把当前分层启用状态、预设和每层倍率导出成 JSON，并复制到剪切板。',
            'btn-image-test-layer-import': '把上方文本框里的 JSON 参数导回弹窗，适合复用或手动微调层倍率。',
            'btn-start-image-test': '按左侧当前参数启动一次轻量推理，结果固定写到 output/tests。',
            'btn-stop-image-test': '停止当前生图测试子进程；不会删除已经生成的图片。',
            'btn-refresh-environment-check': '重新检测项目文件、Python 依赖、系统工具、CUDA 和 Web 运行目录。',
            'btn-copy-environment-report': '把当前环境检测报告复制为纯文本，方便排查依赖或安装问题。',
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
            'btn-live-sampling-preview': '打开当前训练或最新运行目录的途中采样弹窗，只显示样张和权重列表。',
            'btn-history-manager-refresh': '重新读取历史任务管理台数据。',
            'btn-history-collections-workbench': '打开集合管理大界面，以配置分组为单位整理历史任务。',
            'btn-history-manager-merge': '合并查看选中的训练任务，预处理任务不会参与合并。',
            'btn-history-bulk-archive': '把选中的历史任务批量归档，默认列表会隐藏它们。',
            'btn-history-bulk-unarchive': '把选中的历史任务批量取消归档。',
            'btn-history-bulk-group': '给选中的历史任务设置同一个集合名称；同配置文件自动分组会继续保留。',
            'btn-history-bulk-delete': '彻底删除选中的历史记录和 WebUI 运行目录，需要连续两次按钮确认。',
            'history-manager-search': '按任务名、配置、目录、消息等字段搜索历史任务。',
            'history-filter-kind': '按训练或预处理任务筛选。',
            'history-filter-state': '按完成、异常、中断或运行状态筛选。',
            'history-filter-archived': '筛选未归档、已归档或全部历史任务。',
            'history-filter-source': '按队列、完整续训或权重热启动来源筛选。',
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
                'weight-analysis': 'ΔW 分析页：读取 safetensors 静态权重能量，不跑图、不占 GPU。',
                settings: '全局设置页：设置 Web 训练输出根目录和新建预设默认模型路径。',
                environment: '环境检测页：检查 Windows/Linux 运行前置、Python 依赖、CUDA 和项目文件。',
                'image-test': '生图测试页：复用当前配置和 preview 目录，快速做单次推理试图。',
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
    globalThis.api = async function api(url, opts = {}) {
        return ctx.api(url, opts);
    }

    globalThis.datasetPresetApi = async function datasetPresetApi(url, opts = {}) {
        const timeoutMs = Number(opts.timeoutMs || DATASET_PRESET_REQUEST_TIMEOUT_MS);
        const requestOpts = { ...opts };
        delete requestOpts.timeoutMs;
        let timeoutId = null;
        try {
            return await Promise.race([
                api(url, requestOpts),
                new Promise((_, reject) => {
                    timeoutId = window.setTimeout(() => {
                        reject(new Error('数据集预设请求超时，请查看终端日志或刷新预设列表'));
                    }, timeoutMs);
                }),
            ]);
        } finally {
            if (timeoutId !== null) {
                window.clearTimeout(timeoutId);
            }
        }
    }

    globalThis.val = function val(id) {
        return ctx.dom.val(id);
    }

    globalThis.populateSelect = function populateSelect(id, items, preferred = '') {
        ctx.dom.populateSelect(id, items, preferred);
    }
