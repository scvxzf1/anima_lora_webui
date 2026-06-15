/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.setupEventListeners = function setupEventListeners() {
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
        document.getElementById('btn-open-history-manager').addEventListener('click', () => showTrainingView('history'));
        ensureQueueFeature().bindQueueEvents();
        ensureWeightAnalysisFeature().bindWeightAnalysisEvents();
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
        document.getElementById('btn-live-sampling-preview').addEventListener('click', openLiveSamplingPreview);
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
        bindTrainingViewTabKeyboard();
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
            'btn-refresh-analysis-weights': '重新扫描当前可读取的训练权重列表，不会加载模型。',
            'btn-run-weight-analysis': '在 CPU 上读取 safetensors 并计算 ΔW 范数，不跑图、不占 GPU。',
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
