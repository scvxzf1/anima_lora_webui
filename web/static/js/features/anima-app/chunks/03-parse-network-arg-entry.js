/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { formatLossValue } from '../../history-detail/curve/data.js?v=module-bootstrap-20260711-ir1';
import { formatCompactNumber, numberOrNull } from '../../history-detail/ui.js?v=module-bootstrap-20260711-ir1';
import { formatLr } from '../../live-training/index.js?v=module-bootstrap-20260711-ir1';
import { HIDDEN_DATASET_PRESET_FILES } from '../helpers/app-constants.js?v=module-bootstrap-20260711-ir1';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../helpers/dataset-values.js?v=module-bootstrap-20260711-ir1';
import {
    datasetPresetSummaryByFile,
    orderDatasetPresetsForGroups,
    selectedDatasetConfigOverride,
    sortDatasetPresetGroups,
} from '../helpers/dataset-presets.js?v=module-bootstrap-20260711-ir1';
import {
    renderConfigDatasetPickerDialog,
    renderDatasetEditor,
    renderDatasetPresetHeader,
    renderDatasetPresetList,
    isDatasetTabActive,
} from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir1';
import { readLiveNumber, readNonnegativeLiveNumber, readOptionalLiveNumber } from '../helpers/live-form-values.js?v=module-bootstrap-20260711-ir1';
import { isCliOnlySpdSource } from '../helpers/training-launch-bridge.js?v=module-bootstrap-20260711-ir1';
import { confirmUnsavedDiscard } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir1';
import { api, datasetPresetApi, val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { currentTrainingConfigFile } from '../helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260711-ir1';
import { escapeHtml } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260711-ir1';
import { isConfigDatasetPickerDialogOpen, renderConfigDatasetPicker } from './06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260711-ir1';

const configState = getConfigState();
const datasetState = getDatasetState();
const trainingState = getTrainingState();

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

    export async function loadStepEstimate(parentSeq = configState.configLoadSeq) {
        const requestSeq = ++configState.stepEstimateSeq;
        const currentTrainingSource = currentTrainingSourceState();
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant || location.protocol === 'file:') return;
        configState.stepEstimateStatus = { loading: true, error: '' };
        configState.currentStepEstimate = null;
        scheduleStepEstimatePanelRefresh();
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            configState.stepEstimateStatus = { loading: false, error: 'SPD CLI 实验配置不使用 Web 步数估算。' };
            configState.currentStepEstimate = null;
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
            if (parentSeq !== configState.configLoadSeq || requestSeq !== configState.stepEstimateSeq) return;
            configState.stepEstimateStatus = { loading: false, error: data?.ok === false ? (data.error || '步数估算失败') : '' };
            configState.currentStepEstimate = data?.ok === false ? null : data;
        } catch (error) {
            if (parentSeq !== configState.configLoadSeq || requestSeq !== configState.stepEstimateSeq) return;
            configState.stepEstimateStatus = { loading: false, error: error?.message || '步数估算失败' };
            configState.currentStepEstimate = null;
        }
        scheduleStepEstimatePanelRefresh();
    }

    export async function loadDatasetEditor(parentSeq = configState.configLoadSeq) {
        const requestSeq = ++datasetState.datasetLoadSeq;
        const currentTrainingSource = currentTrainingSourceState();
        const datasetEditorState = currentDatasetEditorState();
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant || location.protocol === 'file:') return;
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            datasetState.datasetEditorState = {
                ...datasetEditorState,
                loading: false,
                loaded: false,
                error: 'SPD 是 CLI 实验配置，不使用 Web 数据集编辑器。',
            };
            renderDatasetEditor();
            return;
        }
        datasetState.datasetEditorState.loading = true;
        datasetState.datasetEditorState.error = '';
        renderDatasetEditor();
        try {
            const params = new URLSearchParams({ variant, preset, methods_subdir: methodsSubdir });
            const configFile = currentTrainingConfigFile();
            if (configFile) params.set('config_file', configFile);
            const datasetConfigOverride = selectedDatasetConfigOverride();
            if (datasetConfigOverride !== null) params.set('dataset_config', datasetConfigOverride);
            const data = await api(`/api/config/datasets?${params.toString()}`);
            if (parentSeq !== configState.configLoadSeq || requestSeq !== datasetState.datasetLoadSeq) return;
            if (!data.ok) {
                throw new Error(data.error || '读取数据集配置失败');
            }
            datasetState.datasetEditorState = {
                loading: false,
                loaded: true,
                dirty: false,
                dataset_config: data.dataset_config || '',
                datasets: normalizeDatasetEditorRows(data.datasets || []),
                defaults: normalizeDatasetDefaults(data.defaults || {}),
                error: '',
            };
        } catch (e) {
            if (parentSeq !== configState.configLoadSeq || requestSeq !== datasetState.datasetLoadSeq) return;
            datasetState.datasetEditorState = {
                ...datasetEditorState,
                loading: false,
                loaded: false,
                defaults: normalizeDatasetDefaults(datasetEditorState.defaults || {}),
                error: e.message || '读取数据集配置失败',
            };
        }
        renderDatasetEditor();
    }

    export async function loadDatasetPresets(options = {}) {
        if (location.protocol === 'file:') return false;
        const requestSeq = ++datasetState.datasetPresetLoadSeq;
        const datasetPresetState = currentDatasetPresetState();
        const managePresets = options.manage === true || (options.manage !== false && isDatasetTabActive());
        if (managePresets) {
            datasetState.datasetPresetState.loading = true;
            renderDatasetPresetList();
        }
        try {
            const data = await datasetPresetApi('/api/config/dataset-presets');
            if (requestSeq !== datasetState.datasetPresetLoadSeq) return false;
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
            datasetState.datasetPresetState.presets = orderDatasetPresetsForGroups(presets, sortedGroups);
            datasetState.datasetPresetState.groups = sortedGroups;
            if (managePresets) {
                datasetState.datasetPresetState.loading = false;
            }
            datasetState.datasetPresetState.error = '';
            datasetState.selectedConfigDatasetSummary = datasetPresetSummaryByFile(datasetState.selectedConfigDatasetFile);
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
                datasetState.datasetPresetState.selectedFile = '';
            }
            if (!preserveDirtySelection && options.selectCurrent !== false && datasetState.selectedConfigDatasetFile && !datasetPresetState.selectedFile && presets.some((preset) => preset.path === datasetState.selectedConfigDatasetFile)) {
                datasetState.datasetPresetState.selectedFile = datasetState.selectedConfigDatasetFile;
            }
            if (!preserveDirtySelection && !datasetPresetState.selectedFile && presets.length) {
                datasetState.datasetPresetState.selectedFile = presets[0].path;
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
            if (requestSeq !== datasetState.datasetPresetLoadSeq) return false;
            if (managePresets) {
                datasetState.datasetPresetState.loading = false;
            }
            datasetState.datasetPresetState.error = e.message || '读取数据集预设失败';
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

    export async function loadDatasetPreset(file) {
        const datasetPresetState = currentDatasetPresetState();
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
            datasetState.datasetPresetState = {
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
            datasetState.datasetPresetState = {
                ...datasetPresetState,
                loading: false,
                error: e.message || '读取数据集预设失败',
            };
        }
        renderDatasetPresetList();
        renderDatasetPresetHeader();
        renderDatasetEditor();
    }

    export function createStepEstimatePanel() {
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

    export function scheduleStepEstimatePanelRefresh() {
        updateStepEstimatePanel();
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(updateStepEstimatePanel);
            return;
        }
        setTimeout(updateStepEstimatePanel, 0);
    }

    export function updateStepEstimatePanel() {
        const currentStepEstimate = configState.currentStepEstimate;
        const stepEstimateStatus = configState.stepEstimateStatus || { loading: false, error: '' };
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

    function liveDatasetRowsForEstimate() {
        const currentStepEstimate = configState.currentStepEstimate;
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

    export function setText(id, text) {
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

    export function metricValueIsEmpty(value) {
        const text = String(value ?? '').trim();
        return !text || text === '-' || text.toUpperCase() === 'N/A';
    }

    export function setMetricText(id, value) {
        const text = metricValueIsEmpty(value) ? 'N/A' : String(value);
        setText(id, text);
    }

    export function setEtaMetricText(info = {}) {
        const el = document.getElementById('metric-eta');
        if (!el) return;
        const text = String(info.text || '').trim() || '待计算';
        el.textContent = text;
        el.title = info.title || '';
        const empty = info.empty !== undefined ? Boolean(info.empty) : (text === '待计算' || metricValueIsEmpty(text));
        el.classList.toggle('metric-empty', empty);
        el.closest('.metric-item')?.classList.toggle('is-empty', empty);
    }

    export function resetLiveMetricPlaceholders(options = {}) {
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

    export function updateDashboardProgressIdleState(active = null) {
        const trainingRuntime = trainingState.trainingRuntime;
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

    export function setTrainingDashboardHeadState(state = 'idle') {
        const head = document.querySelector('#tab-training .training-dashboard-head');
        if (!head) return;
        head.classList.remove('is-idle', 'is-running', 'is-compiling', 'is-error', 'is-history');
        head.classList.add(`is-${state || 'idle'}`);
    }

    export function syncLossChartEmptyState() {
        const lossChart = trainingState.lossChart;
        const shell = document.getElementById('loss-chart-shell');
        if (!shell) return;
        const pointCount = Array.isArray(lossChart?.data) ? lossChart.data.length : 0;
        shell.classList.toggle('is-empty', pointCount < 2);
        renderLiveChartPanel();
    }

    export function syncLiveChartControls() {
        const liveChartState = trainingState.liveChartState;
        const lrToggle = document.getElementById('live-chart-toggle-lr');
        if (lrToggle) lrToggle.checked = liveChartState.showLr;
        const rangeSelect = document.getElementById('live-chart-range');
        if (rangeSelect) rangeSelect.value = liveChartState.rangeMode;
    }

    function liveChartVisiblePoints(points = []) {
        const liveChartState = trainingState.liveChartState;
        const all = Array.isArray(points) ? points : [];
        const match = String(liveChartState.rangeMode || 'all').match(/^last(\d+)$/);
        if (!match) return all;
        const count = Number(match[1]);
        return Number.isFinite(count) && count > 0 ? all.slice(-count) : all;
    }

    export function renderLiveChartPanel() {
        const lossChart = trainingState.lossChart;
        const liveChartState = trainingState.liveChartState;
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

    export function updateTrainingToolbarState(state, label) {
        const safeState = state || 'idle';
        const stateEl = document.getElementById('training-toolbar-state');
        const textEl = document.getElementById('training-toolbar-state-text');
        if (stateEl) stateEl.className = `training-toolbar-state ${safeState}`;
        if (textEl) textEl.textContent = label || '空闲';
    }
