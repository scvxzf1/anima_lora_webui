import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

const api = createApiClient();
const ESTIMATE_KEYS = new Set([
    'max_train_epochs',
    'max_train_steps',
    'train_batch_size',
    'gradient_accumulation_steps',
    'sample_ratio',
]);

export function renderDatasetConfigField(value, block = null) {
    const path = String(value || '');
    const blockClass = block ? ' dragon-config-block' : '';
    const blockAttributes = block
        ? ` data-field-span="${block.span}" data-config-tag="${escapeHtml(block.chapterId)}" data-config-tone="${escapeHtml(block.tone)}" data-control-kind="dataset" data-required="${block.required}" data-experimental="false" data-path-field="true"`
        : '';
    return `
        <div class="dragon-config-dataset-card${blockClass}" data-config-field-key="dataset_config" data-config-tool="dataset"
             data-dirty="false" data-search-text="dataset_config 数据集 数据集预设 训练图片 ${escapeHtml(path.toLocaleLowerCase())}"${blockAttributes}>
            <input type="hidden" name="config_dataset_config" data-key="dataset_config" data-baseline-value="${escapeHtml(path)}" value="${escapeHtml(path)}">
            <div class="dragon-config-dataset-card-head">
                <div>
                    <span class="dragon-eyebrow">训练数据</span>
                    <div class="dragon-config-dataset-title-line">
                        <strong data-config-dataset-title>${escapeHtml(datasetFilename(path) || '沿用当前训练配置')}</strong>
                        <span class="dragon-config-label-key"> | dataset_config</span>
                    </div>
                </div>
                <span class="dragon-config-dataset-state" data-config-dataset-state data-state="loading">读取中</span>
            </div>
            <code class="dragon-config-dataset-path" data-config-dataset-path title="${escapeHtml(path)}">${escapeHtml(path || '沿用当前训练配置中的数据集字段')}</code>
            <div class="dragon-config-dataset-meta" data-config-dataset-meta>
                <span>数据集 -</span><span>训练图片 -</span><span>重复 -</span>
            </div>
            <div class="dragon-config-dataset-actions dragon-field-label-actions">
                <button class="dragon-btn dragon-btn-primary dragon-btn-sm" type="button" data-config-dataset-action="open">${renderIcon('folder', 'dragon-btn-icon')}<span>${path ? '更换预设' : '选择预设'}</span></button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-config-dataset-action="preview" ${path ? '' : 'disabled'}>${renderIcon('eye', 'dragon-btn-icon')}<span>预览</span></button>
                <a class="dragon-btn dragon-btn-secondary dragon-btn-sm" href="#dataset-editor">${renderIcon('edit', 'dragon-btn-icon')}<span>管理数据集</span></a>
                <button class="dragon-field-reset" type="button" data-config-reset-field="dataset_config" hidden aria-label="撤销数据集预设修改" title="撤销此项修改">${renderIcon('refresh')}</button>
            </div>
        </div>`;
}

export function renderStepEstimatePanel() {
    return `
        <section class="dragon-step-estimate" data-step-estimate aria-labelledby="dragon-step-estimate-title">
            <div class="dragon-step-estimate-head">
                <div><span class="dragon-eyebrow">训练量校验</span><h3 id="dragon-step-estimate-title">预计训练步数</h3></div>
                <span class="dragon-step-estimate-status" data-step-estimate-status>计算中</span>
            </div>
            <div class="dragon-step-estimate-grid" data-step-estimate-grid>
                ${renderEstimateMetric('训练图片', 'trainImages')}
                ${renderEstimateMetric('重复后样本', 'repeatedImages')}
                ${renderEstimateMetric('有效批大小', 'effectiveBatch')}
                ${renderEstimateMetric('每轮步数', 'stepsPerEpoch')}
                ${renderEstimateMetric('最大轮数', 'epochs')}
                ${renderEstimateMetric('总步数', 'totalSteps', true)}
            </div>
            <details class="dragon-step-estimate-details">
                <summary>查看数据集明细与计算公式</summary>
                <div data-step-estimate-breakdown></div>
                <p data-step-estimate-note>正在读取数据集图片数量。</p>
            </details>
        </section>`;
}

export function renderDatasetPickerDialog() {
    return `
        <dialog class="dragon-dataset-picker-dialog" data-config-dataset-dialog aria-labelledby="dragon-dataset-picker-title">
            <div class="dragon-dataset-picker-shell">
                <div class="dragon-dataset-picker-head">
                    <div><span class="dragon-eyebrow">数据集预设库</span><h2 id="dragon-dataset-picker-title">选择训练数据</h2><p>选中预设后先核对首图与标注，再应用到当前训练配置。</p></div>
                    <div>
                        <button class="dragon-icon-button" type="button" data-config-dataset-action="refresh" aria-label="刷新数据集预设" title="刷新">${renderIcon('refresh')}</button>
                        <button class="dragon-icon-button" type="button" data-config-dataset-action="close" aria-label="关闭数据集选择" title="关闭">${renderIcon('x')}</button>
                    </div>
                </div>
                <div class="dragon-dataset-picker-status" data-config-dataset-dialog-status role="status" aria-live="polite"></div>
                <div class="dragon-dataset-picker-workspace">
                    <aside class="dragon-dataset-picker-library" aria-label="数据集预设列表">
                        <label><span class="visually-hidden">搜索数据集预设</span><input class="dragon-input" type="search" autocomplete="off" data-config-dataset-search placeholder="搜索名称、路径或图片目录…"></label>
                        <div data-config-dataset-list></div>
                    </aside>
                    <section class="dragon-dataset-picker-preview" data-config-dataset-preview aria-label="数据集预览"></section>
                </div>
                <footer class="dragon-dataset-picker-footer">
                    <div data-config-dataset-selection><span>当前选择</span><strong>未选择</strong></div>
                    <button class="dragon-btn dragon-btn-primary" type="button" data-config-dataset-action="apply" disabled>${renderIcon('check', 'dragon-btn-icon')}<span>应用此预设</span></button>
                </footer>
            </div>
        </dialog>`;
}

export function calculateStepEstimate(payload = {}, live = {}) {
    const batchSize = positiveNumber(live.train_batch_size, payload.train_batch_size || 1);
    const gradAccum = positiveNumber(live.gradient_accumulation_steps, payload.gradient_accumulation_steps || 1);
    const sampleRatio = positiveNumber(live.sample_ratio, payload.sample_ratio || 1);
    const epochs = optionalPositiveNumber(live.max_train_epochs, payload.max_train_epochs);
    const maxSteps = nonnegativeNumber(live.max_train_steps, payload.max_train_steps || 0);
    const weightedImages = Math.max(0, Number(payload.weighted_image_count || 0));
    const effectiveBatch = Math.max(1, batchSize * gradAccum);
    const repeatedImages = Math.floor(weightedImages * sampleRatio);
    const stepsPerEpoch = repeatedImages ? Math.ceil(repeatedImages / effectiveBatch) : 0;
    const durationMode = epochs !== null ? 'epochs' : (maxSteps > 0 ? 'steps' : 'unset');
    return {
        trainImages: Math.max(0, Number(payload.train_image_count || 0)),
        repeatedImages,
        weightedImages,
        sampleRatio,
        batchSize,
        gradAccum,
        effectiveBatch,
        stepsPerEpoch,
        epochs,
        maxSteps,
        durationMode,
        totalSteps: durationMode === 'epochs' ? stepsPerEpoch * epochs : maxSteps,
    };
}

export function groupDatasetPresets(payload = {}, query = '') {
    const presets = Array.isArray(payload.presets) ? payload.presets : [];
    const byPath = new Map(presets.map((preset) => [preset.path, preset]));
    const needle = String(query || '').trim().toLocaleLowerCase();
    const matches = (preset) => !needle || [
        preset.label,
        preset.filename,
        preset.path,
        preset.summary?.source_dir,
        preset.summary?.image_dir,
    ].some((value) => String(value || '').toLocaleLowerCase().includes(needle));
    const groups = Array.isArray(payload.groups) && payload.groups.length
        ? payload.groups
        : [{ id: 'all', label: '全部预设', files: presets }];
    return groups.map((group) => ({
        ...group,
        files: (group.files || []).map((item) => byPath.get(item.path) || item).filter(matches),
    })).filter((group) => group.files.length);
}

export function bindTrainingDataTools(root, { trainingContext, getDraftValue } = {}) {
    const card = root.querySelector('[data-config-field-key="dataset_config"]');
    const estimatePanel = root.querySelector('[data-step-estimate]');
    const dialog = root.querySelector('[data-config-dataset-dialog]');
    if (!card && !estimatePanel && !dialog) return null;
    const state = createTrainingDataState(trainingContext, getDraftValue);
    const cleanup = [];

    bindDatasetCard(root, card, dialog, state, cleanup);
    bindEstimateInputs(root, estimatePanel, state, cleanup);
    loadEstimate(root, state);

    const savedHandler = () => {
        const input = datasetInput(root);
        if (input) input.dataset.baselineValue = input.value;
        renderDatasetSummary(root, state);
    };
    root.addEventListener('dragon-config-saved', savedHandler);
    cleanup.push(() => root.removeEventListener('dragon-config-saved', savedHandler));
    return () => {
        state.disposed = true;
        cleanup.forEach((dispose) => dispose());
    };
}

function createTrainingDataState(trainingContext, getDraftValue) {
    return {
        context: trainingContext || {},
        getDraftValue: typeof getDraftValue === 'function' ? getDraftValue : () => '',
        library: { presets: [], groups: [] },
        highlightedFile: '',
        estimate: null,
        estimateError: '',
        estimateLoading: true,
        estimateSequence: 0,
        librarySequence: 0,
        previewSequence: 0,
        previewPayload: null,
        previewError: '',
        previewLoading: false,
        disposed: false,
    };
}

function bindDatasetCard(root, card, dialog, state, cleanup) {
    if (!card || !dialog) return;
    const open = async (previewOnly = false) => {
        state.highlightedFile = datasetValue(root, state);
        if (!dialog.open) dialog.showModal();
        renderDatasetDialog(dialog, state);
        if (!state.library.presets.length) await loadPresetLibrary(root, dialog, state);
        if (!state.highlightedFile) state.highlightedFile = state.library.presets[0]?.path || '';
        renderDatasetDialog(dialog, state);
        if (state.highlightedFile) await loadPresetPreview(dialog, state);
        if (!previewOnly) dialog.querySelector('[data-config-dataset-search]')?.focus({ preventScroll: true });
    };
    card.querySelector('[data-config-dataset-action="open"]')?.addEventListener('click', () => open(false));
    card.querySelector('[data-config-dataset-action="preview"]')?.addEventListener('click', () => open(true));
    const refreshAfterReset = () => queueMicrotask(() => {
        renderDatasetSummary(root, state);
        loadEstimate(root, state);
    });
    const fieldReset = card.querySelector('[data-config-reset-field="dataset_config"]');
    const formReset = root.querySelector('#dragon-config-reset');
    fieldReset?.addEventListener('click', refreshAfterReset);
    formReset?.addEventListener('click', refreshAfterReset);
    cleanup.push(() => {
        fieldReset?.removeEventListener('click', refreshAfterReset);
        formReset?.removeEventListener('click', refreshAfterReset);
    });
    bindDatasetDialog(root, dialog, state, cleanup);
}

function bindDatasetDialog(root, dialog, state, cleanup) {
    const close = () => dialog.close('cancel');
    dialog.querySelector('[data-config-dataset-action="close"]')?.addEventListener('click', close);
    dialog.querySelector('[data-config-dataset-action="refresh"]')?.addEventListener('click', () => loadPresetLibrary(root, dialog, state, true));
    dialog.querySelector('[data-config-dataset-action="apply"]')?.addEventListener('click', () => applyHighlightedDataset(root, dialog, state));
    dialog.querySelector('[data-config-dataset-search]')?.addEventListener('input', () => renderDatasetList(dialog, state));
    dialog.querySelector('[data-config-dataset-list]')?.addEventListener('click', (event) => selectDatasetFromEvent(dialog, state, event));
    dialog.addEventListener('click', (event) => { if (event.target === dialog) close(); });
    const input = datasetInput(root);
    const onDatasetChange = () => {
        renderDatasetSummary(root, state);
        loadEstimate(root, state);
    };
    input?.addEventListener('input', onDatasetChange);
    cleanup.push(() => input?.removeEventListener('input', onDatasetChange));
}

function bindEstimateInputs(root, panel, state, cleanup) {
    if (!panel) return;
    const schedule = () => renderStepEstimate(root, state);
    root.querySelectorAll('[data-key]').forEach((input) => {
        if (!ESTIMATE_KEYS.has(input.dataset.key)) return;
        input.addEventListener('input', schedule);
        input.addEventListener('change', schedule);
        cleanup.push(() => {
            input.removeEventListener('input', schedule);
            input.removeEventListener('change', schedule);
        });
    });
}

async function loadPresetLibrary(root, dialog, state, preserveSelection = false) {
    const sequence = ++state.librarySequence;
    setDialogStatus(dialog, '正在读取数据集预设…', 'info');
    try {
        const { loadDatasetPresetLibrary } = await import('./dataset-editor-presets.js?v=dragon-ui-20260824v71');
        const library = await loadDatasetPresetLibrary(api);
        if (state.disposed || sequence !== state.librarySequence) return;
        state.library = library;
        const current = datasetValue(root, state);
        const previousExists = library.presets.some((item) => item.path === state.highlightedFile);
        state.highlightedFile = (preserveSelection && previousExists ? state.highlightedFile : current)
            || library.presets[0]?.path
            || '';
        setDialogStatus(dialog, '', 'info');
        renderDatasetSummary(root, state);
        renderDatasetDialog(dialog, state);
        if (dialog?.open && state.highlightedFile) await loadPresetPreview(dialog, state);
    } catch (error) {
        if (state.disposed || sequence !== state.librarySequence) return;
        setDialogStatus(dialog, error.message || '读取数据集预设失败', 'error');
    }
}

async function loadEstimate(root, state) {
    const sequence = ++state.estimateSequence;
    state.estimateLoading = true;
    state.estimateError = '';
    renderStepEstimate(root, state);
    renderDatasetSummary(root, state);
    try {
        const payload = await api(stepEstimateUrl(state.context, datasetValue(root, state)));
        if (payload?.ok === false) throw new Error(payload.error || '步数估算失败');
        if (state.disposed || sequence !== state.estimateSequence) return;
        state.estimate = payload;
    } catch (error) {
        if (state.disposed || sequence !== state.estimateSequence) return;
        state.estimate = null;
        state.estimateError = error.message || '步数估算失败';
    } finally {
        if (state.disposed || sequence !== state.estimateSequence) return;
        state.estimateLoading = false;
        renderStepEstimate(root, state);
        renderDatasetSummary(root, state);
    }
}

function renderDatasetSummary(root, state) {
    const card = root.querySelector('[data-config-field-key="dataset_config"]');
    if (!card) return;
    const input = datasetInput(root);
    const path = datasetValue(root, state);
    const preset = state.library.presets.find((item) => item.path === path);
    const summary = preset?.summary || {};
    const dirty = input ? input.value !== (input.dataset.baselineValue || '') : false;
    setContent(card, '[data-config-dataset-title]', preset?.label || datasetFilename(path) || '沿用当前训练配置');
    const pathNode = card.querySelector('[data-config-dataset-path]');
    if (pathNode) { pathNode.textContent = path || '沿用当前训练配置中的数据集字段'; pathNode.title = path; }
    const stateNode = card.querySelector('[data-config-dataset-state]');
    if (stateNode) {
        stateNode.dataset.state = dirty ? 'dirty' : (path ? 'synced' : 'legacy');
        stateNode.textContent = dirty ? '未保存' : (path ? '已同步' : '当前配置');
    }
    const meta = card.querySelector('[data-config-dataset-meta]');
    if (meta) meta.innerHTML = [
        `<span>数据集 <strong>${Number(summary.dataset_count ?? state.estimate?.dataset_count ?? 0)}</strong></span>`,
        `<span>训练图片 <strong>${state.estimateLoading ? '…' : Number(state.estimate?.train_image_count || 0)}</strong></span>`,
        `<span>重复 <strong>${Number(summary.repeat_total ?? state.estimate?.dataset_num_repeats ?? 0)}</strong></span>`,
    ].join('');
    const openLabel = card.querySelector('[data-config-dataset-action="open"] span');
    if (openLabel) openLabel.textContent = path ? '更换预设' : '选择预设';
    const preview = card.querySelector('[data-config-dataset-action="preview"]');
    if (preview) preview.disabled = !path;
}

function renderStepEstimate(root, state) {
    const panel = root.querySelector('[data-step-estimate]');
    if (!panel) return;
    const status = panel.querySelector('[data-step-estimate-status]');
    if (state.estimateLoading) {
        if (status) { status.dataset.tone = 'info'; status.textContent = '计算中'; }
        return;
    }
    if (!state.estimate) {
        if (status) { status.dataset.tone = 'error'; status.textContent = '无法估算'; }
        setContent(panel, '[data-step-estimate-note]', state.estimateError || '没有可用的数据集信息。');
        return;
    }
    const result = calculateStepEstimate(state.estimate, readLiveEstimateValues(root, state));
    if (status) {
        status.dataset.tone = result.durationMode === 'unset' ? 'warning' : 'success';
        status.textContent = result.durationMode === 'unset' ? '未设置训练时长' : '已根据当前表单更新';
    }
    Object.entries(result).forEach(([key, value]) => setContent(panel, `[data-step-metric="${key}"]`, metricValue(key, value)));
    renderDatasetBreakdown(panel, state.estimate.datasets || []);
    setContent(panel, '[data-step-estimate-note]', estimateNote(result));
}

function renderDatasetDialog(dialog, state) {
    if (!dialog) return;
    renderDatasetList(dialog, state);
    renderDatasetPreview(dialog, state);
    const preset = highlightedPreset(state);
    const selection = dialog.querySelector('[data-config-dataset-selection]');
    if (selection) selection.innerHTML = `<span>当前选择</span><strong>${escapeHtml(preset?.label || preset?.filename || '未选择')}</strong>`;
    const apply = dialog.querySelector('[data-config-dataset-action="apply"]');
    if (apply) apply.disabled = !preset;
}

function renderDatasetList(dialog, state) {
    const list = dialog?.querySelector('[data-config-dataset-list]');
    if (!list) return;
    const query = dialog.querySelector('[data-config-dataset-search]')?.value || '';
    const groups = groupDatasetPresets(state.library, query);
    if (!groups.length) {
        list.innerHTML = '<div class="dragon-dataset-picker-empty">没有匹配的数据集预设</div>';
        return;
    }
    list.innerHTML = groups.map((group) => `
        <section class="dragon-dataset-picker-group">
            <div><strong>${escapeHtml(group.label || group.id || '数据集预设')}</strong><span>${group.files.length}</span></div>
            ${group.files.map((preset) => renderDatasetOption(preset, preset.path === state.highlightedFile)).join('')}
        </section>`).join('');
}

function renderDatasetOption(preset, active) {
    const summary = preset.summary || {};
    return `<button class="dragon-dataset-picker-item" type="button" data-config-dataset-file="${escapeHtml(preset.path)}" data-active="${active}" aria-pressed="${active}">
        <span><strong>${escapeHtml(preset.label || preset.filename || preset.path)}</strong><small>${escapeHtml(preset.path)}</small></span>
        <em>${Number(summary.dataset_count || 0)}组 · ${Number(summary.repeat_total || 0)} Reps</em>
    </button>`;
}

async function selectDatasetFromEvent(dialog, state, event) {
    if (!(event.target instanceof Element)) return;
    const item = event.target.closest('[data-config-dataset-file]');
    if (!item) return;
    state.highlightedFile = item.dataset.configDatasetFile || '';
    state.previewPayload = null;
    state.previewError = '';
    renderDatasetDialog(dialog, state);
    await loadPresetPreview(dialog, state);
}

async function loadPresetPreview(dialog, state) {
    const file = state.highlightedFile;
    if (!file) return;
    const sequence = ++state.previewSequence;
    state.previewLoading = true;
    state.previewError = '';
    renderDatasetPreview(dialog, state);
    try {
        const params = new URLSearchParams({ file, dataset_index: '0', source: 'source', limit: '1' });
        const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
        if (payload?.ok === false) throw new Error(payload.error || '读取数据集预览失败');
        if (state.disposed || sequence !== state.previewSequence) return;
        state.previewPayload = payload;
    } catch (error) {
        if (state.disposed || sequence !== state.previewSequence) return;
        state.previewPayload = null;
        state.previewError = error.message || '读取数据集预览失败';
    } finally {
        if (state.disposed || sequence !== state.previewSequence) return;
        state.previewLoading = false;
        renderDatasetDialog(dialog, state);
    }
}

function renderDatasetPreview(dialog, state) {
    const preview = dialog?.querySelector('[data-config-dataset-preview]');
    if (!preview) return;
    const preset = highlightedPreset(state);
    if (!preset) {
        preview.innerHTML = '<div class="dragon-dataset-picker-empty"><strong>请选择数据集预设</strong></div>';
        return;
    }
    if (state.previewLoading) {
        preview.innerHTML = '<div class="dragon-dataset-picker-empty"><strong>正在读取首图与标注…</strong></div>';
        return;
    }
    if (state.previewError) {
        preview.innerHTML = `<div class="dragon-dataset-picker-empty" data-tone="error"><strong>预览读取失败</strong><span>${escapeHtml(state.previewError)}</span></div>`;
        return;
    }
    preview.innerHTML = renderPreviewPayload(preset, state.previewPayload);
}

function renderPreviewPayload(preset, payload) {
    const summary = preset.summary || {};
    const image = Array.isArray(payload?.images) ? payload.images[0] : null;
    const caption = image?.caption || {};
    return `
        <div class="dragon-dataset-picker-preview-head">
            <div><span class="dragon-eyebrow">第 1 组首图</span><h3>${escapeHtml(preset.label || preset.filename || preset.path)}</h3></div>
            <span>${Number(summary.dataset_count || 0)} 组 · ${Number(summary.repeat_total || 0)} Reps</span>
        </div>
        <code title="${escapeHtml(summary.source_dir || preset.path)}">${escapeHtml(summary.source_dir || preset.path)}</code>
        <div class="dragon-dataset-picker-image" data-empty="${image ? 'false' : 'true'}">
            ${image ? `<img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.name || '数据集首图')}" width="640" height="420">` : '<span>当前目录没有可预览图片</span>'}
        </div>
        <div class="dragon-dataset-picker-caption" data-missing="${caption.ok ? 'false' : 'true'}">
            <span>${caption.ok ? `标注 · ${escapeHtml(caption.format_label || caption.extension || '')}` : '未找到标注'}</span>
            <p>${escapeHtml(caption.ok ? (caption.text || '(空标注)') : (payload?.caption_summary || '当前标注来源没有对应 caption'))}</p>
        </div>`;
}

function applyHighlightedDataset(root, dialog, state) {
    const preset = highlightedPreset(state);
    const input = datasetInput(root);
    if (!preset || !input) return;
    input.value = preset.path;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    dialog.close('apply');
}

function renderDatasetBreakdown(panel, datasets) {
    const target = panel.querySelector('[data-step-estimate-breakdown]');
    if (!target) return;
    target.innerHTML = datasets.length ? datasets.map((row, index) => {
        const count = Number(row.train_image_count || 0);
        const repeats = Number(row.num_repeats || 1);
        return `<div class="dragon-step-estimate-row"><strong>第 ${Number(row.index || index + 1)} 组</strong><span>${count} 张 x 重复 ${repeats} = ${count * repeats} 样本</span><code title="${escapeHtml(row.source_dir || row.image_dir || '')}">${escapeHtml(row.source_dir || row.image_dir || '-')}</code></div>`;
    }).join('') : '<div class="dragon-step-estimate-row"><span>还没有可估算的数据集。</span></div>';
}

function estimateNote(result) {
    if (result.durationMode === 'epochs') {
        return `向上取整(${result.repeatedImages} / ${result.effectiveBatch}) = ${result.stepsPerEpoch} 步/轮；${result.stepsPerEpoch} x ${result.epochs} 轮 = ${result.totalSteps} 总步数。max_train_epochs 已设置，max_train_steps 不生效。`;
    }
    if (result.durationMode === 'steps') return `当前未设置 max_train_epochs，训练直接使用 max_train_steps=${result.maxSteps}。`;
    return '尚未设置最大训练轮数，且 max_train_steps=0；启动训练前必须配置训练时长。';
}

function readLiveEstimateValues(root, state) {
    const values = {};
    ESTIMATE_KEYS.forEach((key) => {
        const input = root.querySelector(`[data-key="${key}"]`);
        values[key] = input ? input.value : state.getDraftValue(key);
    });
    return values;
}

function stepEstimateUrl(context, datasetConfig) {
    const params = new URLSearchParams({
        variant: context.variant || 'lora',
        preset: context.preset || 'default',
        methods_subdir: context.methodsSubdir || 'gui-methods',
    });
    if (context.configFile) params.set('config_file', context.configFile);
    params.set('dataset_config', datasetConfig || '');
    return `/api/config/steps?${params.toString()}`;
}

function datasetInput(root) {
    return root.querySelector('[data-key="dataset_config"]');
}

function datasetValue(root, state) {
    const input = datasetInput(root);
    return String(input ? input.value : state.getDraftValue('dataset_config') || '').trim();
}

function highlightedPreset(state) {
    return state.library.presets.find((item) => item.path === state.highlightedFile) || null;
}

function datasetFilename(path) {
    return String(path || '').replace(/\\/g, '/').split('/').filter(Boolean).pop() || '';
}

function renderEstimateMetric(label, key, primary = false) {
    return `<div data-primary="${primary}"><span>${label}</span><strong data-step-metric="${key}">-</strong></div>`;
}

function metricValue(key, value) {
    if (key === 'epochs') return value === null ? '未设置' : String(value);
    return Number.isFinite(Number(value)) ? Number(value).toLocaleString('zh-CN') : '-';
}

function positiveNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : Number(fallback || 1);
}

function optionalPositiveNumber(value, fallback) {
    if (value === '' || value === null || value === undefined) {
        const fallbackNumber = Number(fallback);
        return Number.isFinite(fallbackNumber) && fallbackNumber > 0 ? fallbackNumber : null;
    }
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : null;
}

function nonnegativeNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? number : Math.max(0, Number(fallback || 0));
}

function setDialogStatus(dialog, message, tone) {
    const target = dialog?.querySelector('[data-config-dataset-dialog-status]');
    if (!target) return;
    target.textContent = message;
    target.dataset.tone = tone;
}

function setContent(root, selector, content) {
    const node = root?.querySelector(selector);
    if (node) node.textContent = content;
}
