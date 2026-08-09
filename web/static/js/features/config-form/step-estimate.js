/**
 * Config-form step estimate panel and API loaders.
 * Extracted from anima-app chunk 03.
 */
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { selectedDatasetConfigOverride } from '../anima-app/helpers/dataset-presets.js?v=module-bootstrap-20260809-nf4-v2';
import { readLiveNumber, readNonnegativeLiveNumber, readOptionalLiveNumber } from '../anima-app/helpers/live-form-values.js?v=module-bootstrap-20260809-nf4-v2';
import { isCliOnlySpdSource } from '../anima-app/helpers/training-launch-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { api, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { currentTrainingConfigFile } from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { escapeHtml } from './field-input.js?v=module-bootstrap-20260809-nf4-v2';
import { setText } from '../live-training/dashboard-ui.js?v=module-bootstrap-20260809-nf4-v2';

const configState = getConfigState();
const trainingState = getTrainingState();

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
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
