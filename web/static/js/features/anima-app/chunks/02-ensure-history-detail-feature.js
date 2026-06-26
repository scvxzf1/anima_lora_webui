/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.ensureHistoryDetailFeature = function ensureHistoryDetailFeature() {
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
            selectedGpuPayload: () => gpuPicker.selectedGpuPayload(),
            inspectContinueLoraWeight: (path) => (
                globalThis.requestContinueLoraInspection?.(path)
                || Promise.resolve({ ok: false, error: '权重审查入口未初始化' })
            ),
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
    globalThis.themeController = null;
    globalThis.uiScaleController = null;
    globalThis.gpuPicker = null;
    globalThis.tabController = null;

    globalThis.startAnimaApp = async function startAnimaApp() {
        themeController = createThemeController({
            storageKey: THEME_STORAGE_KEY,
            getLossChart: () => lossChart,
            chartTheme,
        });
        uiScaleController = createUIScaleController();
        gpuPicker = createGpuPicker({
            storageKey: GPU_WHITELIST_STORAGE_KEY,
            api,
        });
        tabController = createTabController({
            loadDatasetPresets,
            loadGlobalSettings,
            ensureWeightAnalysisFeature,
            ensureEnvironmentCheckFeature,
            resetTrainingExpandedStateOnLeave,
            resizeLiveChart: () => lossChart?.resize?.(),
            auditConfigTrainingSourceOnEnter,
        });

        const boot = async () => {
            themeController.initThemeToggle();
            uiScaleController.initUIScale();
            tabController.setupTabs();
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
    globalThis.chartTheme = function chartTheme() {
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

    globalThis.isHistoryReviewMode = function isHistoryReviewMode() {
        return historyViewMode !== 'live';
    }

    globalThis.openTutorialDialog = function openTutorialDialog() {
        const dialog = document.getElementById('tutorial-dialog');
        if (!dialog) return;
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
    }

    // ── 加载初始数据 ──
    globalThis.loadInitialData = async function loadInitialData() {
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
            fieldHelp = help;
            populateSelect('method-select', methods, 'lora');
            populateSelect('preset-select', presets, 'default');
            await gpuPicker.loadGpuOptions();
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

    globalThis.showStandaloneWarning = function showStandaloneWarning() {
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

    globalThis.loadVariants = async function loadVariants({ reset = false } = {}) {
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

    globalThis.loadConfig = async function loadConfig() {
        const requestSeq = ++configLoadSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        if (!variant) return;
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        const params = new URLSearchParams({ variant, preset, methods_subdir: methodsSubdir });
        const configFile = currentTrainingConfigFile();
        if (configFile) params.set('config_file', configFile);
        const data = await api(`/api/config/merged?${params.toString()}`);
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
        scheduleStepEstimatePanelRefresh();
        const compatibilityPatch = applyConfigCompatibilityDrafts();
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
            await loadTomlFile(tomlFile, { force: true });
        }
        if (Object.keys(compatibilityPatch).length > 0) {
            updateTomlDirtyState();
            setTomlStatus('error', '已自动修正 CAME optimizer_args 的 betas 格式，请保存当前配置后再训练。', { persist: true });
        }
    }

    globalThis.reloadCurrentConfig = async function reloadCurrentConfig() {
        if (!(await confirmDiscardTomlChanges('当前配置有未保存修改，刷新会重新读取表单和数据集设置并丢弃这些修改。是否继续？'))) {
            return;
        }
        await loadConfig();
        rememberSelectionSnapshot();
    }

    // ── 配置表单渲染 ──
    globalThis.resetConfigFormDraft = function resetConfigFormDraft() {
        configFormState.draftValues.clear();
    }

    globalThis.applyConfigCompatibilityDrafts = function applyConfigCompatibilityDrafts() {
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

    globalThis.syncConfigDraftFromForm = function syncConfigDraftFromForm(options = {}) {
        document.querySelectorAll('#config-form .field-input[data-key]').forEach((input) => {
            updateConfigDraftFromInput(input, options);
        });
    }

    globalThis.updateConfigDraftFromInput = function updateConfigDraftFromInput(input, options = {}) {
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

    globalThis.originalConfigFieldValue = function originalConfigFieldValue(key) {
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

    globalThis.displayConfigFieldValue = function displayConfigFieldValue(key, value) {
        if (key === 'lora_adapter_kind') {
            return configFormState.draftValues.has(key)
                ? configFormState.draftValues.get(key)
                : loraAdapterKindFromConfig(currentConfig);
        }
        return configFormState.draftValues.has(key)
            ? configFormState.draftValues.get(key)
            : value;
    }

    globalThis.configDraftValueChanged = function configDraftValueChanged(key, next, original = originalConfigFieldValue(key), options = {}) {
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

    globalThis.renderConfigForm = function renderConfigForm(config) {
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

    globalThis.shouldRenderConfigSection = function shouldRenderConfigSection(section, config = currentConfig) {
        if (!section?.method) return true;
        return activeMethodKey(config) === section.method;
    }

    globalThis.shouldSkipConfigFormField = function shouldSkipConfigFormField(key, config = currentConfig) {
        if (CONFIG_FORM_MERGED_FIELDS?.has?.(key)) return true;
        if (DEPRECATED_CONFIG_FORM_FIELDS.has(key)) return true;
        if (RETIRED_CONFIG_FORM_FIELDS.has(key)) return true;
        const scopedFamilies = METHOD_SCOPED_CONFIG_FORM_FIELDS.get(key);
        if (!scopedFamilies) return false;
        return !scopedFamilies.has(activeMethodKey(config));
    }

    globalThis.shouldExposeUiDefaultField = function shouldExposeUiDefaultField(key, config, fieldsByKey = {}) {
        if (key in fieldsByKey) return true;
        if (NETWORK_ARG_FIELD_MAP.has(key)) return false;
        const family = activeMethodKey(config);
        if (SPD_UI_DEFAULT_FIELDS.has(key)) return family === 'spd';
        if (CHIMERA_UI_DEFAULT_FIELDS.has(key)) return family === 'chimera';
        if (IP_ADAPTER_UI_DEFAULT_FIELDS.has(key)) return family === 'ip_adapter';
        if (SOFT_TOKENS_UI_DEFAULT_FIELDS.has(key)) return family === 'soft_tokens';
        return true;
    }

    globalThis.applyNetworkArgFields = function applyNetworkArgFields(fieldsByKey, config) {
        const specs = activeNetworkArgSpecs(config);
        if (!specs.length) return;
        const argMap = parseNetworkArgMap(config?.network_args);
        for (const spec of specs) {
            const rawValue = argMap.has(spec.arg) ? argMap.get(spec.arg) : spec.default;
            fieldsByKey[spec.key] = coerceNetworkArgValue(rawValue, spec);
        }
    }

    globalThis.isActiveNetworkArgFieldKey = function isActiveNetworkArgFieldKey(key, config = currentConfig) {
        return activeNetworkArgSpecs(config).some((spec) => spec.key === key);
    }

    globalThis.collectSectionFields = function collectSectionFields(fieldsByKey, orderedKeys, consumed) {
        const fields = [];
        for (const key of orderedKeys) {
            if (consumed.has(key) || !(key in fieldsByKey)) continue;
            fields.push([key, fieldsByKey[key]]);
            consumed.add(key);
        }
        return fields;
    }

    globalThis.activeNetworkArgSpecs = function activeNetworkArgSpecs(config = currentConfig) {
        const families = activeNetworkArgFamilies(config);
        const argMap = parseNetworkArgMap(config?.network_args);
        return NETWORK_ARG_FIELD_SPECS.filter((spec) =>
            families.has(spec.family) || argMap.has(spec.arg)
        );
    }

    globalThis.activeNetworkArgFamilies = function activeNetworkArgFamilies(config = currentConfig) {
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

    globalThis.parseNetworkArgMap = function parseNetworkArgMap(networkArgs) {
        const map = new Map();
        for (const raw of normalizeNetworkArgArray(networkArgs)) {
            const parsed = parseNetworkArgEntry(raw);
            if (parsed) map.set(parsed.arg, parsed.value);
        }
        return map;
    }

    globalThis.normalizeNetworkArgArray = function normalizeNetworkArgArray(networkArgs) {
        if (Array.isArray(networkArgs)) return networkArgs.map((item) => String(item));
        if (typeof networkArgs === 'string' && networkArgs.trim()) return parseArrayValue(networkArgs).map((item) => String(item));
        return [];
    }
