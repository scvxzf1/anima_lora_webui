/**
 * Stage schedule chart, dialog shell, and inline summary.
 *
 * stage_schedule is owned by the *dataset preset* TOML
 * (configs/datasets/*.toml). Applying a dataset or preparing runtime injects
 * the schedule into the training config so the trainer still sees it.
 */
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { datasetPresetApi } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    activeStageScheduleDatasetState,
    hydrateStageScheduleFromConfig,
    hydrateStageScheduleFromDatasetPreset,
    listSubsetOptions,
    resolveStageScheduleSource,
    stageResolutionMetrics,
    stageSchedulePayload,
    normalizeRawStages,
} from './stage-resolution-model.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    createStageResolutionSummary,
    createStageResolutionChartPanel,
    createStageResolutionEditor,
    createStageResolutionTable,
    syncStageResolutionEditorInputs,
} from './stage-resolution-ui-widgets.js?v=module-bootstrap-20260714-stage-dataset5';
import { registerStageResolutionRenderer } from './stage-resolution-ui-render.js?v=module-bootstrap-20260714-stage-dataset5';
import { datasetRowsForPayload, normalizeDatasetDefaults } from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260714-stage-dataset5';

const configState = getConfigState();
const stageResolutionState = configState.stageResolutionState;

export function drawStageResolutionChart() {
    const canvas = document.getElementById('stage-resolution-chart');
    if (!canvas) return;
    const metrics = stageResolutionMetrics();
    const stages = metrics.stages;
    const ctx2d = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.floor(rect.width || 720));
    const height = Math.max(140, Math.floor(rect.height || 160));
    const ratio = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx2d.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx2d.clearRect(0, 0, width, height);

    const styles = getComputedStyle(document.documentElement);
    const text = styles.getPropertyValue('--text-dim').trim() || '#8892a4';
    const grid = styles.getPropertyValue('--chart-grid').trim() || '#2a3a5e';
    const pad = { top: 28, right: 16, bottom: 28, left: 16 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    if (!stages.length || plotW <= 0 || plotH <= 0) return;

    ctx2d.fillStyle = grid;
    ctx2d.globalAlpha = 0.25;
    ctx2d.fillRect(pad.left, pad.top + plotH * 0.25, plotW, plotH * 0.5);
    ctx2d.globalAlpha = 1;

    const bands = [];
    stages.forEach((stage) => {
        const x0 = pad.left + stage.start_pct * plotW;
        const x1 = pad.left + stage.end_pct * plotW;
        const w = Math.max(2, x1 - x0);
        ctx2d.fillStyle = stage.color;
        ctx2d.globalAlpha = stage.index === stageResolutionState.selectedIndex ? 0.92 : 0.72;
        ctx2d.fillRect(x0, pad.top + plotH * 0.15, w, plotH * 0.7);
        ctx2d.globalAlpha = 1;
        ctx2d.fillStyle = '#0b1220';
        ctx2d.font = 'bold 12px monospace';
        ctx2d.textAlign = 'center';
        const label = stage.resolution != null ? String(stage.resolution) : stage.name;
        if (w > 36) {
            ctx2d.fillText(label, x0 + w / 2, pad.top + plotH * 0.55);
        }
        bands.push({ index: stage.index, x0, x1 });
    });

    ctx2d.fillStyle = text;
    ctx2d.font = '10px monospace';
    ctx2d.textAlign = 'left';
    ctx2d.fillText('0%', pad.left, height - 10);
    ctx2d.textAlign = 'right';
    ctx2d.fillText('100%', width - pad.right, height - 10);
    if (metrics.totalSteps) {
        ctx2d.textAlign = 'center';
        ctx2d.fillText(`S = ${metrics.totalSteps} steps`, width / 2, 16);
    } else {
        ctx2d.textAlign = 'center';
        ctx2d.fillText('请设定 max_train_steps 以换算绝对步数', width / 2, 16);
    }
    canvas._stageResolutionBands = bands;
}

function stageScheduleDraftDirty() {
    const draft = configState.configFormState?.draftValues;
    return Boolean(draft?.has?.('stage_schedule') || draft?.has?.('stage_schedule_enabled'));
}

function shortConfigName(file) {
    const text = String(file || '').trim();
    if (!text) return '';
    const parts = text.split(/[\\/]/).filter(Boolean);
    return parts[parts.length - 1] || text;
}

function uniqueNonEmpty(items) {
    const out = [];
    const seen = new Set();
    for (const raw of items) {
        const file = String(raw || '').trim();
        if (!file || seen.has(file)) continue;
        seen.add(file);
        out.push(file);
    }
    return out;
}

function stageScheduleFileMeta(file) {
    const presets = getDatasetState()?.datasetPresetState?.presets || [];
    return presets.find((item) => item?.path === file) || null;
}

function isDatasetPresetPath(file) {
    return String(file || '').includes('/datasets/');
}

function isStageScheduleFileWritable(file) {
    if (!file || !isDatasetPresetPath(file)) return false;
    const datasetState = getDatasetState();
    const activeDataset = activeStageScheduleDatasetState(datasetState);
    if (activeDataset?.readonly) return false;
    const meta = stageScheduleFileMeta(file);
    if (meta && (meta.locked || meta.system_locked || meta.readonly)) return false;
    return true;
}

/**
 * Resolve the file that stage_schedule should write to.
 *
 * Priority (dataset-first):
 * 1) currently selected dataset preset (configs/datasets/*.toml)
 * 2) selectedConfigDatasetFile linked to current training config
 * 3) writable training config fallback (legacy / no dataset selected)
 */
export function listStageScheduleTargetCandidates() {
    const datasetState = getDatasetState();
    const activeDataset = activeStageScheduleDatasetState(datasetState);
    const activeTargets = uniqueNonEmpty([
        activeDataset.selectedFile,
        activeDataset.dataset_config,
    ]).filter((file) => isDatasetPresetPath(file));
    if (activeTargets.length) return activeTargets;
    return uniqueNonEmpty([
        datasetState?.selectedConfigDatasetFile,
        configState.currentConfig?.dataset_config,
    ]).filter((file) => isDatasetPresetPath(file));
}

export function resolveStageScheduleTargetFile(options = {}) {
    const preferWritable = options.preferWritable !== false;
    const candidates = listStageScheduleTargetCandidates();
    if (!candidates.length) return '';
    if (preferWritable) {
        return candidates.find((file) => isStageScheduleFileWritable(file)) || '';
    }
    return candidates[0] || '';
}

function describeStageScheduleTargetContext(targetFile) {
    const bits = [];
    if (targetFile) {
        bits.push(`数据集配置目标：${targetFile}`);
    } else {
        bits.push('保存目标：未选中数据集预设');
    }
    return bits.join(' · ');
}

function stageScheduleFeedbackText(kind = 'idle') {
    if (kind === 'saved') {
        return '已写入数据集配置文件。';
    }
    if (kind === 'draft') {
        return '当前只在浏览器草稿里，还没写入数据集配置文件；刷新页面会丢失。请点“写入并保存配置”。';
    }
    if (kind === 'error') {
        return '保存失败。';
    }
    return '分阶段调度绑定数据集配置。修改后请点“写入并保存配置”。';
}

function setStageScheduleDialogFeedback(kind = 'idle', detail = '') {
    const el = document.getElementById('stage-resolution-feedback');
    if (!el) return;
    const tone = kind === 'saved' ? 'ok' : (kind === 'error' ? 'error' : 'warning');
    el.className = `stage-resolution-status stage-resolution-feedback ${tone}`;
    const base = stageScheduleFeedbackText(kind);
    el.textContent = detail ? `${base} ${detail}` : base;
}

function suggestSaveAsNameFromLocked(file) {
    const base = shortConfigName(file).replace(/\.toml$/i, '') || 'stage_schedule';
    return `${base}_copy`;
}

async function saveStageScheduleViaSaveAs(payload, lockedFile) {
    const datasetState = getDatasetState();
    const activeDataset = activeStageScheduleDatasetState(datasetState);
    const suggested = suggestSaveAsNameFromLocked(lockedFile || '数据集.toml');
    const inputName = window.prompt(
        '当前数据集配置不可直接写入。\n请输入新的数据集配置名称（保存到 configs/datasets/）：',
        suggested,
    );
    if (inputName === null) {
        return { ok: false, cancelled: true, error: '已取消另存' };
    }
    const name = String(inputName || '').trim();
    if (!name) {
        return { ok: false, error: '另存名称无效' };
    }
    const rows = datasetRowsForPayload(activeDataset.datasets || []);
    if (!rows.length || rows.some((row) => !String(row.source_dir || '').trim())) {
        return { ok: false, error: '当前数据集配置没有可用子集路径' };
    }
    const saved = await datasetPresetApi('/api/config/dataset-presets/save-as', {
        method: 'POST',
        body: JSON.stringify({
            name,
            datasets: rows,
            defaults: normalizeDatasetDefaults(activeDataset.defaults || {}),
            stage_schedule_enabled: payload.stage_schedule_enabled,
            stage_schedule: payload.stage_schedule,
        }),
    });
    if (!saved?.ok) {
        return { ok: false, error: saved?.error || '另存数据集配置失败' };
    }
    activeDataset.selectedFile = saved.file;
    activeDataset.stage_schedule_enabled = saved.stage_schedule_enabled ?? payload.stage_schedule_enabled;
    activeDataset.stage_schedule = Array.isArray(saved.stage_schedule)
        ? saved.stage_schedule
        : payload.stage_schedule;
    activeDataset.datasets = Array.isArray(saved.datasets) ? saved.datasets : rows;
    activeDataset.defaults = saved.defaults || activeDataset.defaults || {};
    activeDataset.readonly = false;
    activeDataset.dirty = false;
    activeDataset.status = saved.message || '已另存数据集配置';
    return { ok: true, file: saved.file };
}

export function renderStageResolutionDialog() {
    const body = document.getElementById('stage-resolution-dialog-body');
    if (!body) return;
    const metrics = stageResolutionMetrics();
    body.innerHTML = '';
    body.appendChild(createStageResolutionSummary(metrics));

    const workspace = document.createElement('div');
    workspace.className = 'stage-resolution-workspace';
    workspace.appendChild(createStageResolutionChartPanel());
    workspace.appendChild(createStageResolutionEditor(metrics.selected, metrics));
    body.appendChild(workspace);
    body.appendChild(createStageResolutionTable(metrics.stages));

    const candidates = listStageScheduleTargetCandidates();
    const writableTarget = resolveStageScheduleTargetFile({ preferWritable: true });
    const rawTarget = resolveStageScheduleTargetFile({ preferWritable: false });
    const targetFile = writableTarget || rawTarget;
    const targetMeta = document.createElement('div');
    targetMeta.className = 'stage-resolution-status stage-resolution-target';

    if (!targetFile) {
        targetMeta.classList.add('error');
        targetMeta.textContent = '未选中数据集预设。请先在数据集页左侧选择一个可编辑数据集配置，再保存分阶段调度。';
    } else if (!isStageScheduleFileWritable(targetFile)) {
        targetMeta.classList.add('error');
        targetMeta.textContent = [
            isDatasetPresetPath(targetFile)
                ? `当前数据集预设只读：${shortConfigName(targetFile)}。`
                : `当前目标不可写：${shortConfigName(targetFile)}。`,
            '分阶段调度绑定数据集配置。',
            '请选择可编辑数据集预设，或复制后编辑。',
        ].filter(Boolean).join(' ');
    } else {
        targetMeta.classList.add('ok');
        targetMeta.textContent = describeStageScheduleTargetContext(targetFile);
    }
    body.appendChild(targetMeta);

    if (candidates.length > 1) {
        const cand = document.createElement('div');
        cand.className = 'stage-resolution-status stage-resolution-target-candidates';
        cand.textContent = `候选保存目标：${candidates.map(shortConfigName).join(' / ')}`;
        body.appendChild(cand);
    }

    const feedback = document.createElement('div');
    feedback.id = 'stage-resolution-feedback';
    feedback.className = 'stage-resolution-status stage-resolution-feedback warning';
    feedback.textContent = stageScheduleFeedbackText(stageScheduleDraftDirty() ? 'draft' : 'idle');
    body.appendChild(feedback);

    const foot = document.createElement('div');
    foot.className = 'stage-resolution-actions';
    foot.style.justifyContent = 'flex-end';
    foot.style.flexWrap = 'wrap';
    foot.style.gap = '8px';

    const draftBtn = document.createElement('button');
    draftBtn.type = 'button';
    draftBtn.className = 'btn btn-small';
    draftBtn.textContent = '仅写入草稿';
    draftBtn.title = '只放进当前页面草稿，不写文件；刷新会丢';
    draftBtn.addEventListener('click', () => {
        applyStageScheduleToDraft({ save: false });
    });

    const saveBtn = document.createElement('button');
    saveBtn.type = 'button';
    saveBtn.className = 'btn btn-primary';
    saveBtn.textContent = '写入并保存配置';
    saveBtn.title = '写入当前数据集预设（configs/datasets/*.toml）';
    saveBtn.disabled = !writableTarget;
    saveBtn.addEventListener('click', () => {
        applyStageScheduleToDraft({ save: true });
    });

    const saveAsBtn = document.createElement('button');
    saveAsBtn.type = 'button';
    saveAsBtn.className = 'btn btn-small';
    saveAsBtn.textContent = '另存数据集配置';
    saveAsBtn.title = '复制当前数据集内容和分阶段调度到新的 configs/datasets/*.toml';
    saveAsBtn.addEventListener('click', () => {
        applyStageScheduleToDraft({ save: true, forceSaveAs: true });
    });

    const activeDataset = activeStageScheduleDatasetState(getDatasetState());
    foot.append(draftBtn);
    if (Object.prototype.hasOwnProperty.call(activeDataset, 'selectedFile')) {
        foot.append(saveAsBtn);
    }
    foot.append(saveBtn);
    body.appendChild(foot);
    requestAnimationFrame(drawStageResolutionChart);
}

registerStageResolutionRenderer(renderStageResolutionDialog);

function refreshStageScheduleUiAfterDraft() {
    const summaryHost = document.querySelector('#config-form .stage-schedule-inline-summary');
    if (summaryHost?.parentElement) {
        summaryHost.replaceWith(createStageScheduleInlineSummary());
    }
    try {
        const root = globalThis;
        root.updateTomlDirtyState?.();
        root.updateConfigEditingIdentity?.();
    } catch (_) {
        // optional during partial loads
    }
}

function clearStageScheduleDraftMarkers() {
    const draft = configState.configFormState?.draftValues;
    if (draft && typeof draft.delete === 'function') {
        draft.delete('stage_schedule');
        draft.delete('stage_schedule_enabled');
    }
}

function writeStageScheduleDraft(payload) {
    const datasetState = getDatasetState();
    const activeDataset = activeStageScheduleDatasetState(datasetState);
    if (activeDataset && typeof activeDataset === 'object') {
        activeDataset.stage_schedule_enabled = payload.stage_schedule_enabled;
        activeDataset.stage_schedule = payload.stage_schedule;
        activeDataset.dirty = true;
        activeDataset.status = '有未保存的数据集修改（含分阶段调度）';
    }
    const draft = configState.configFormState?.draftValues;
    if (draft && typeof draft.set === 'function') {
        draft.set('stage_schedule_enabled', payload.stage_schedule_enabled);
        draft.set('stage_schedule', payload.stage_schedule);
    }
    if (configState.currentConfig && typeof configState.currentConfig === 'object') {
        configState.currentConfig.stage_schedule_enabled = payload.stage_schedule_enabled;
        configState.currentConfig.stage_schedule = payload.stage_schedule;
    }
    refreshStageScheduleUiAfterDraft();
    return { ok: true };
}

async function applyStageScheduleToDraft(options = {}) {
    const save = Boolean(options.save);
    const forceSaveAs = Boolean(options.forceSaveAs);
    syncStageResolutionEditorInputs();
    const metrics = stageResolutionMetrics();
    if (metrics.enabled && metrics.problemCount) {
        const msg = `分阶段调度有 ${metrics.problemCount} 项错误，请先修正`;
        setTomlStatus('error', msg);
        setStageScheduleDialogFeedback('error', msg);
        return false;
    }
    const payload = stageSchedulePayload();
    const draftResult = writeStageScheduleDraft(payload);
    if (!draftResult.ok) {
        setTomlStatus('error', draftResult.error);
        setStageScheduleDialogFeedback('error', draftResult.error);
        return false;
    }

    const segmentText = payload.stage_schedule_enabled
        ? `${payload.stage_schedule.length} 段`
        : '已关闭';

    if (!save) {
        setStageScheduleDialogFeedback('draft', `草稿状态：${segmentText}。`);
        setTomlStatus(
            'pending',
            payload.stage_schedule_enabled
                ? `分阶段调度已写入草稿（${payload.stage_schedule.length} 段），尚未保存到数据集配置；刷新会丢失`
                : '已关闭分阶段调度（仅草稿，尚未保存到数据集配置；刷新会丢失）',
            { persist: true },
        );
        return true;
    }

    const rawTarget = resolveStageScheduleTargetFile({ preferWritable: false });
    if (forceSaveAs) {
        setStageScheduleDialogFeedback('draft', '正在另存数据集配置…');
        try {
            const result = await saveStageScheduleViaSaveAs(payload, rawTarget);
            if (result.cancelled) {
                setStageScheduleDialogFeedback('error', '已取消另存');
                return false;
            }
            if (!result.ok) {
                setStageScheduleDialogFeedback('error', result.error || '另存失败');
                setTomlStatus('error', result.error || '另存失败');
                return false;
            }
            clearStageScheduleDraftMarkers();
            setTomlStatus('ok', `✓ 已另存数据集配置并写入分阶段调度：${result.file}`, { persist: true });
            renderStageResolutionDialog();
            setStageScheduleDialogFeedback('saved', `已落盘到 ${shortConfigName(result.file)}（${segmentText}）。`);
            return true;
        } catch (err) {
            const msg = `另存失败：${err?.message || err}`;
            setStageScheduleDialogFeedback('error', msg);
            setTomlStatus('error', msg);
            return false;
        }
    }

    const file = rawTarget;
    if (!file) {
        const msg = '请先在数据集页选中一个可编辑数据集预设，再保存分阶段调度';
        setTomlStatus('error', msg);
        setStageScheduleDialogFeedback('error', msg);
        return false;
    }
    if (!isStageScheduleFileWritable(file)) {
        const msg = isDatasetPresetPath(file)
            ? `数据集预设只读：${shortConfigName(file)}。请复制后再编辑分阶段调度`
            : `配置不可写：${shortConfigName(file)}。请选中可编辑数据集预设`;
        setTomlStatus('error', msg);
        setStageScheduleDialogFeedback('error', msg);
        return false;
    }

    // Dataset config ownership path.
    if (isDatasetPresetPath(file)) {
        setStageScheduleDialogFeedback('draft', `正在保存到数据集预设 ${shortConfigName(file)} …`);
        setTomlStatus('pending', `正在保存分阶段调度到 ${shortConfigName(file)} …`, { persist: true });
        try {
            const datasetState = getDatasetState();
            const presetState = activeStageScheduleDatasetState(datasetState);
            const rows = datasetRowsForPayload(presetState.datasets || []);
            if (!rows.length || rows.some((row) => !String(row.source_dir || '').trim())) {
                const msg = '当前数据集预设没有可用子集路径，无法保存分阶段调度';
                setTomlStatus('error', msg);
                setStageScheduleDialogFeedback('error', msg);
                return false;
            }
            const res = await datasetPresetApi('/api/config/dataset-presets', {
                method: 'PUT',
                body: JSON.stringify({
                    file,
                    datasets: rows,
                    defaults: normalizeDatasetDefaults(presetState.defaults || {}),
                    overwrite: true,
                    stage_schedule_enabled: payload.stage_schedule_enabled,
                    stage_schedule: payload.stage_schedule,
                }),
            });
            if (!res?.ok) {
                const msg = res?.error || '保存数据集预设失败';
                setTomlStatus('error', msg);
                setStageScheduleDialogFeedback('error', msg);
                return false;
            }
            if (presetState && typeof presetState === 'object') {
                if ('selectedFile' in presetState) presetState.selectedFile = res.file || file;
                if ('dataset_config' in presetState) presetState.dataset_config = res.file || file;
                presetState.stage_schedule_enabled = (
                    res.stage_schedule_enabled ?? payload.stage_schedule_enabled
                );
                presetState.stage_schedule = (
                    Array.isArray(res.stage_schedule) ? res.stage_schedule : payload.stage_schedule
                );
                presetState.dirty = false;
                presetState.status = res.message || '已保存分阶段调度';
                if (Array.isArray(res.datasets)) {
                    presetState.datasets = res.datasets;
                }
                if (res.defaults) {
                    presetState.defaults = res.defaults;
                }
            }
            // Saved to dataset file: stage draft no longer dirty.
            clearStageScheduleDraftMarkers();
            setStageScheduleDialogFeedback('saved', `已落盘到 ${shortConfigName(res.file || file)}（${segmentText}）。`);
            setTomlStatus(
                'ok',
                payload.stage_schedule_enabled
                    ? `✓ 分阶段调度已保存到数据集预设 ${shortConfigName(res.file || file)}（${payload.stage_schedule.length} 段）`
                    : `✓ 已关闭分阶段调度，并保存到数据集预设 ${shortConfigName(res.file || file)}`,
                { persist: true },
            );
            try {
                renderStageResolutionDialog();
                setStageScheduleDialogFeedback('saved', `已落盘到 ${shortConfigName(res.file || file)}（${segmentText}）。`);
            } catch (_) {
                // dialog may be mid-close
            }
            return true;
        } catch (err) {
            const msg = `保存失败：${err?.message || err}`;
            setTomlStatus('error', msg);
            setStageScheduleDialogFeedback('error', msg);
            return false;
        }
    }
    return false;
}

export function createOpenStageResolutionDialogButton() {
    const btn = document.createElement('button');
    btn.id = 'btn-open-stage-resolution-dialog';
    btn.type = 'button';
    btn.className = 'btn btn-small config-group-title-action';
    btn.textContent = '分阶段调度';
    btn.title = '按总训练步数百分比切换数据集子集（写入当前数据集配置）';
    btn.addEventListener('click', openStageResolutionDialog);
    return btn;
}

export function openStageResolutionDialog() {
    const dialog = document.getElementById('stage-resolution-dialog');
    if (!dialog) {
        setTomlStatus('error', '找不到分阶段调度对话框，请刷新页面');
        return;
    }
    // Prefer dataset preset ownership; fall back to training config draft.
    const datasetState = getDatasetState();
    const activeDataset = activeStageScheduleDatasetState(datasetState);
    if (activeDataset?.loading) {
        setTomlStatus('pending', '当前数据集配置仍在读取，请稍候再编辑分阶段调度');
        return;
    }
    if (
        activeDataset?.selectedFile
        || activeDataset?.dataset_config
        || Array.isArray(activeDataset?.stage_schedule)
        || activeDataset?.stage_schedule_enabled != null
    ) {
        hydrateStageScheduleFromDatasetPreset(activeDataset);
    } else {
        hydrateStageScheduleFromConfig(configState.currentConfig || {});
    }
    renderStageResolutionDialog();
    if (dialog.showModal && !dialog.open) dialog.showModal();
    else if (!dialog.open) dialog.setAttribute('open', 'open');
    requestAnimationFrame(drawStageResolutionChart);
    if (stageScheduleDraftDirty()) {
        setStageScheduleDialogFeedback('draft');
    }
}

export function createStageScheduleInlineSummary() {
    const source = resolveStageScheduleSource(configState.currentConfig || {});
    const enabled = Boolean(source.stage_schedule_enabled);
    const stages = normalizeRawStages(source.stage_schedule);
    const draft = configState.configFormState?.draftValues;
    const dirty = Boolean(
        draft?.has?.('stage_schedule')
        || draft?.has?.('stage_schedule_enabled'),
    );
    const wrap = document.createElement('div');
    wrap.className = [
        'stage-schedule-inline-summary',
        enabled ? 'is-enabled' : 'is-disabled',
        dirty ? 'is-dirty' : '',
    ].filter(Boolean).join(' ');
    wrap.dataset.searchText = '分阶段调度 stage_schedule dataset 数据集 步数区间';
    const title = document.createElement('strong');
    title.textContent = enabled ? '分阶段调度已启用' : '分阶段调度未启用';
    const detail = document.createElement('span');
    if (enabled && stages.length) {
        const options = listSubsetOptions();
        const optionByIndex = new Map(options.map((item) => [item.index, item]));
        const resChain = stages.map((stage) => {
            const opt = optionByIndex.get(stage.subset_index);
            if (opt?.resolution != null) return `${opt.resolution}px`;
            return `S${Number(stage.subset_index) + 1}`;
        }).join('→');
        const subsetChain = stages.map((stage) => Number(stage.subset_index) + 1).join('→');
        detail.textContent = `${stages.length} 段 · subset ${subsetChain} · ${resChain}${dirty ? ' · 未保存' : ''}`;
    } else {
        detail.textContent = dirty ? '草稿已修改，尚未写入数据集配置' : '按总步数百分比切换数据集子集（写入数据集配置）';
    }
    const openBtn = document.createElement('button');
    openBtn.type = 'button';
    openBtn.className = 'btn btn-small';
    openBtn.textContent = dirty ? '继续编辑（未保存）' : '编辑步数区间';
    openBtn.addEventListener('click', openStageResolutionDialog);
    wrap.append(title, detail, openBtn);
    return wrap;
}
