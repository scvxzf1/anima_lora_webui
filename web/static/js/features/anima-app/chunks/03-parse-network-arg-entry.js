/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.parseNetworkArgEntry = function parseNetworkArgEntry(raw) {
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

    globalThis.stripNetworkArgQuotes = function stripNetworkArgQuotes(value) {
        const text = String(value || '').trim();
        if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
            return text.slice(1, -1);
        }
        return text;
    }

    globalThis.coerceNetworkArgValue = function coerceNetworkArgValue(value, spec) {
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

    globalThis.parseBooleanNetworkArg = function parseBooleanNetworkArg(value, fallback = false) {
        if (typeof value === 'boolean') return value;
        if (value === 1 || value === 0) return Boolean(value);
        const text = String(value ?? '').trim().toLowerCase();
        if (['1', 'true', 'yes', 'on'].includes(text)) return true;
        if (['0', 'false', 'no', 'off'].includes(text)) return false;
        return Boolean(fallback);
    }

    globalThis.loadStepEstimate = async function loadStepEstimate(parentSeq = configLoadSeq) {
        const requestSeq = ++stepEstimateSeq;
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant || location.protocol === 'file:') return;
        stepEstimateStatus = { loading: true, error: '' };
        currentStepEstimate = null;
        scheduleStepEstimatePanelRefresh();
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            stepEstimateStatus = { loading: false, error: 'SPD CLI 实验配置不使用 Web 步数估算。' };
            currentStepEstimate = null;
            scheduleStepEstimatePanelRefresh();
            return;
        }
        try {
            const params = new URLSearchParams({ variant, preset, methods_subdir: methodsSubdir });
            const configFile = currentTrainingConfigFile();
            if (configFile) params.set('config_file', configFile);
            const datasetConfigOverride = selectedDatasetConfigOverride();
            if (datasetConfigOverride !== null) params.set('dataset_config', datasetConfigOverride);
            const data = await api(`/api/config/steps?${params.toString()}`);
            if (parentSeq !== configLoadSeq || requestSeq !== stepEstimateSeq) return;
            stepEstimateStatus = { loading: false, error: data?.ok === false ? (data.error || '步数估算失败') : '' };
            currentStepEstimate = data?.ok === false ? null : data;
        } catch (error) {
            if (parentSeq !== configLoadSeq || requestSeq !== stepEstimateSeq) return;
            stepEstimateStatus = { loading: false, error: error?.message || '步数估算失败' };
            currentStepEstimate = null;
        }
        scheduleStepEstimatePanelRefresh();
    }

    globalThis.loadDatasetEditor = async function loadDatasetEditor(parentSeq = configLoadSeq) {
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
            const params = new URLSearchParams({ variant, preset, methods_subdir: methodsSubdir });
            const configFile = currentTrainingConfigFile();
            if (configFile) params.set('config_file', configFile);
            const datasetConfigOverride = selectedDatasetConfigOverride();
            if (datasetConfigOverride !== null) params.set('dataset_config', datasetConfigOverride);
            const data = await api(`/api/config/datasets?${params.toString()}`);
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

    globalThis.loadDatasetPresets = async function loadDatasetPresets(options = {}) {
        if (location.protocol === 'file:') return false;
        const requestSeq = ++datasetPresetLoadSeq;
        const managePresets = options.manage === true || (options.manage !== false && isDatasetTabActive());
        if (managePresets) {
            datasetPresetState.loading = true;
            renderDatasetPresetList();
        }
        try {
            const data = await datasetPresetApi('/api/config/dataset-presets');
            if (requestSeq !== datasetPresetLoadSeq) return false;
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
                return true;
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
            return true;
        } catch (e) {
            if (requestSeq !== datasetPresetLoadSeq) return false;
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
            if (options.throwOnError) {
                throw e;
            }
            return false;
        }
    }

    globalThis.loadDatasetPreset = async function loadDatasetPreset(file) {
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
            const data = await datasetPresetApi(`/api/config/dataset-presets/read?file=${encodeURIComponent(file)}`);
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

    globalThis.createStepEstimatePanel = function createStepEstimatePanel() {
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

    globalThis.scheduleStepEstimatePanelRefresh = function scheduleStepEstimatePanelRefresh() {
        updateStepEstimatePanel();
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(updateStepEstimatePanel);
            return;
        }
        setTimeout(updateStepEstimatePanel, 0);
    }

    globalThis.updateStepEstimatePanel = function updateStepEstimatePanel() {
        const panel = document.getElementById('step-estimate-panel');
        if (!panel) return;
        if (!currentStepEstimate) {
            setText('step-dataset-count', stepEstimateStatus?.loading ? '计算中' : '-');
            setText('step-train-images', '-');
            setText('step-repeated-images', '-');
            setText('step-effective-batch', '-');
            setText('step-per-epoch', '-');
            setText('step-max-train-epochs', '-');
            setText('step-total', '-');
            renderStepDatasetBreakdown([]);
            const note = stepEstimateStatus?.loading
                ? '正在重新读取训练配置、数据集配置和图片数量。'
                : (stepEstimateStatus?.error || '选择训练配置后会自动估算步数。');
            setText('step-estimate-note', note);
            return;
        }

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

    globalThis.liveDatasetRowsForEstimate = function liveDatasetRowsForEstimate() {
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

    globalThis.renderStepDatasetBreakdown = function renderStepDatasetBreakdown(datasets) {
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

    globalThis.readLiveNumber = function readLiveNumber(key, fallback) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return Number(fallback) || 0;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const n = Number(raw);
        return Number.isFinite(n) && n > 0 ? n : (Number(fallback) || 0);
    }

    globalThis.readNonnegativeLiveNumber = function readNonnegativeLiveNumber(key, fallback = 0) {
        const fallbackNumber = Math.max(0, Number(fallback) || 0);
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return fallbackNumber;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const trimmed = String(raw).trim();
        if (!trimmed) return fallbackNumber;
        const n = Number(trimmed);
        return Number.isFinite(n) && n >= 0 ? n : fallbackNumber;
    }

    globalThis.readOptionalLiveNumber = function readOptionalLiveNumber(key) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) return null;
        const raw = input.type === 'checkbox' ? input.checked : input.value;
        const trimmed = String(raw).trim();
        if (!trimmed) return null;
        const n = Number(trimmed);
        return Number.isFinite(n) && n > 0 ? n : null;
    }

    globalThis.setText = function setText(id, text) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = text;
        if (el.classList.contains('metric-value')) {
            const empty = metricValueIsEmpty(text);
            el.classList.toggle('metric-empty', empty);
            el.title = empty ? '' : String(text);
            el.closest('.metric-item')?.classList.toggle('is-empty', empty);
        }
    }

    globalThis.metricValueIsEmpty = function metricValueIsEmpty(value) {
        const text = String(value ?? '').trim();
        return !text || text === '-' || text.toUpperCase() === 'N/A';
    }

    globalThis.setMetricText = function setMetricText(id, value) {
        const text = metricValueIsEmpty(value) ? 'N/A' : String(value);
        setText(id, text);
    }

    globalThis.setEtaMetricText = function setEtaMetricText(info = {}) {
        const el = document.getElementById('metric-eta');
        if (!el) return;
        const text = String(info.text || '').trim() || '待计算';
        el.textContent = text;
        el.title = info.title || '';
        const empty = info.empty !== undefined ? Boolean(info.empty) : (text === '待计算' || metricValueIsEmpty(text));
        el.classList.toggle('metric-empty', empty);
        el.closest('.metric-item')?.classList.toggle('is-empty', empty);
    }

    globalThis.resetLiveMetricPlaceholders = function resetLiveMetricPlaceholders(options = {}) {
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

    globalThis.updateDashboardProgressIdleState = function updateDashboardProgressIdleState(active = null) {
        const wrap = document.querySelector('#tab-training .training-dashboard-progress');
        const head = document.querySelector('#tab-training .training-dashboard-head');
        const text = document.getElementById('progress-text');
        if (!wrap) return;
        const hasProgress = active !== null
            ? Boolean(active)
            : Number(trainingRuntime.progressTotal || 0) > 0;
        wrap.classList.toggle('is-idle', !hasProgress);
        head?.classList.toggle('is-idle', !hasProgress);
        if (!hasProgress && text) {
            text.textContent = '暂无正在运行的任务目录...';
        }
    }

    globalThis.setTrainingDashboardHeadState = function setTrainingDashboardHeadState(state = 'idle') {
        const head = document.querySelector('#tab-training .training-dashboard-head');
        if (!head) return;
        head.classList.remove('is-idle', 'is-running', 'is-compiling', 'is-error', 'is-history');
        head.classList.add(`is-${state || 'idle'}`);
    }

    globalThis.syncLossChartEmptyState = function syncLossChartEmptyState() {
        const shell = document.getElementById('loss-chart-shell');
        if (!shell) return;
        const pointCount = Array.isArray(lossChart?.data) ? lossChart.data.length : 0;
        shell.classList.toggle('is-empty', pointCount < 2);
        renderLiveChartPanel();
    }

    globalThis.syncLiveChartControls = function syncLiveChartControls() {
        const lrToggle = document.getElementById('live-chart-toggle-lr');
        if (lrToggle) lrToggle.checked = liveChartState.showLr;
        const rangeSelect = document.getElementById('live-chart-range');
        if (rangeSelect) rangeSelect.value = liveChartState.rangeMode;
    }

    globalThis.liveChartVisiblePoints = function liveChartVisiblePoints(points = []) {
        const all = Array.isArray(points) ? points : [];
        const match = String(liveChartState.rangeMode || 'all').match(/^last(\d+)$/);
        if (!match) return all;
        const count = Number(match[1]);
        return Number.isFinite(count) && count > 0 ? all.slice(-count) : all;
    }

    globalThis.renderLiveChartPanel = function renderLiveChartPanel() {
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

    globalThis.setLiveChartStat = function setLiveChartStat(id, value, empty = null) {
        const el = document.getElementById(id);
        if (!el) return;
        const text = metricValueIsEmpty(value) ? 'N/A' : String(value);
        el.textContent = text;
        const isEmpty = empty === null ? metricValueIsEmpty(text) : Boolean(empty);
        el.closest('.live-chart-stat')?.classList.toggle('is-empty', isEmpty);
    }

    globalThis.liveChartStepRangeText = function liveChartStepRangeText(points = []) {
        if (!points.length) return 'N/A';
        const first = points[0]?.step;
        const last = points[points.length - 1]?.step;
        return `${formatStepLabel(first)} - ${formatStepLabel(last)}`;
    }

    globalThis.formatStepLabel = function formatStepLabel(value) {
        const number = Number(value);
        return Number.isFinite(number) ? String(Math.round(number)) : '-';
    }

    globalThis.updateTrainingToolbarState = function updateTrainingToolbarState(state, label) {
        const safeState = state || 'idle';
        const stateEl = document.getElementById('training-toolbar-state');
        const textEl = document.getElementById('training-toolbar-state-text');
        if (stateEl) stateEl.className = `training-toolbar-state ${safeState}`;
        if (textEl) textEl.textContent = label || '空闲';
    }
