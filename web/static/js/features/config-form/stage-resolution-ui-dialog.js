/**
 * Stage schedule chart, dialog shell, and inline summary.
 */
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    hydrateStageScheduleFromConfig,
    listSubsetOptions,
    resolveStageScheduleSource,
    stageResolutionMetrics,
    stageSchedulePayload,
    normalizeRawStages,
} from './stage-resolution-model.js?v=module-bootstrap-20260711-ir1';
import {
    createStageResolutionSummary,
    createStageResolutionChartPanel,
    createStageResolutionEditor,
    createStageResolutionTable,
} from './stage-resolution-ui-widgets.js?v=module-bootstrap-20260711-ir1';

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

    // Track background.
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

    // Axis labels.
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

    const foot = document.createElement('div');
    foot.className = 'stage-resolution-actions';
    foot.style.justifyContent = 'flex-end';
    const note = document.createElement('span');
    note.className = 'stage-resolution-status';
    note.style.marginRight = 'auto';
    note.textContent = '预处理须在开训前完成各子集缓存；训练中只切换索引。';
    const applyBtn = document.createElement('button');
    applyBtn.type = 'button';
    applyBtn.className = 'btn btn-primary';
    applyBtn.textContent = '应用到配置草稿';
    applyBtn.addEventListener('click', applyStageScheduleToDraft);
    foot.append(note, applyBtn);
    body.appendChild(foot);
    requestAnimationFrame(drawStageResolutionChart);
}

function applyStageScheduleToDraft() {
    const metrics = stageResolutionMetrics();
    if (metrics.enabled && metrics.problemCount) {
        setTomlStatus('error', `阶段课表有 ${metrics.problemCount} 项错误，请先修正`);
        return;
    }
    const payload = stageSchedulePayload();
    const draft = configState.configFormState.draftValues;
    draft.set('stage_schedule_enabled', payload.stage_schedule_enabled);
    draft.set('stage_schedule', payload.stage_schedule);
    // Refresh inline summary + dirty badges without full form rebuild when possible.
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
    setTomlStatus(
        'ok',
        payload.stage_schedule_enabled
            ? `已写入分阶段调度（${payload.stage_schedule.length} 段），保存 TOML 后生效`
            : '已关闭分阶段调度（保存 TOML 后生效）',
    );
}

export function createOpenStageResolutionDialogButton() {
    const btn = document.createElement('button');
    btn.id = 'btn-open-stage-resolution-dialog';
    btn.type = 'button';
    btn.className = 'btn btn-small config-group-title-action';
    btn.textContent = '分阶段调度';
    btn.title = '按总训练步数百分比切换数据集子集';
    btn.addEventListener('click', openStageResolutionDialog);
    return btn;
}

export function openStageResolutionDialog() {
    const dialog = document.getElementById('stage-resolution-dialog');
    if (!dialog) {
        setTomlStatus('error', '找不到阶段课表对话框，请刷新页面');
        return;
    }
    // Always re-sync from draft || currentConfig so switching configs cannot leak stages.
    hydrateStageScheduleFromConfig(configState.currentConfig || {});
    renderStageResolutionDialog();
    if (dialog.showModal && !dialog.open) dialog.showModal();
    else if (!dialog.open) dialog.setAttribute('open', 'open');
    requestAnimationFrame(drawStageResolutionChart);
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
    wrap.dataset.searchText = '分阶段调度 stage_schedule dataset 数据集 课表';
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
        detail.textContent = dirty ? '草稿已修改，保存 TOML 后生效' : '按总步数百分比切换数据集子集';
    }
    const openBtn = document.createElement('button');
    openBtn.type = 'button';
    openBtn.className = 'btn btn-small';
    openBtn.textContent = '编辑课表';
    openBtn.addEventListener('click', openStageResolutionDialog);
    wrap.append(title, detail, openBtn);
    return wrap;
}
