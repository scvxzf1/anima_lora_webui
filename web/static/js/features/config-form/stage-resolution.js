/**
 * Percent-based multi-dataset stage schedule UI (分阶段调度课表).
 *
 * Stages bind to subset indices from the active dataset editor and cover
 * [start_pct, end_pct) of max_train_steps. Caches must be prebuilt.
 */
import { getAppContext } from '../anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260707-93';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260707-93';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260707-93';

const ctx = getAppContext();
const configState = getConfigState();
const stageResolutionState = configState.stageResolutionState;

const STAGE_COLORS = ['#5B8DEF', '#2DD4BF', '#A78BFA', '#F59E0B', '#F472B6', '#34D399'];

function clamp01(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(1, n));
}

function toFraction(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    return n > 1 ? clamp01(n / 100) : clamp01(n);
}

function pctLabel(fraction) {
    return `${Math.round(clamp01(fraction) * 1000) / 10}%`;
}

function readTotalSteps() {
    const draft = configState.configFormState?.draftValues;
    const current = configState.currentConfig || {};
    const raw = draft?.has?.('max_train_steps')
        ? draft.get('max_train_steps')
        : current.max_train_steps;
    const steps = Math.round(Number(raw) || 0);
    return steps > 0 ? steps : 0;
}

function listSubsetOptions() {
    const datasetState = getDatasetState();
    const rows = datasetState.datasetEditorState?.datasets
        || datasetState.datasetPresetState?.datasets
        || [];
    if (!Array.isArray(rows) || !rows.length) {
        return [{ index: 0, label: 'SUBSET 1（当前数据集）', resolution: 1024 }];
    }
    return rows.map((row, index) => {
        const settings = row?.settings || {};
        const resolution = Number(settings.resolution) || 1024;
        const path = String(row?.image_dir || row?.path || '').trim();
        const short = path ? path.split(/[\\/]/).filter(Boolean).slice(-2).join('/') : `子集 ${index + 1}`;
        return {
            index,
            label: `SUBSET ${index + 1} · ${resolution}px · ${short || '未命名'}`,
            resolution,
        };
    });
}

export function defaultStageScheduleStages() {
    return [
        { name: '阶段1', subset_index: 0, start_pct: 0, end_pct: 0.5 },
        { name: '阶段2', subset_index: 1, start_pct: 0.5, end_pct: 1 },
    ];
}

export function resolveStageScheduleSource(config = configState.currentConfig || {}) {
    const draft = configState.configFormState?.draftValues;
    const hasDraft = (key) => Boolean(draft?.has?.(key));
    return {
        stage_schedule_enabled: hasDraft('stage_schedule_enabled')
            ? draft.get('stage_schedule_enabled')
            : config.stage_schedule_enabled,
        stage_schedule: hasDraft('stage_schedule')
            ? draft.get('stage_schedule')
            : config.stage_schedule,
    };
}

export function hydrateStageScheduleFromConfig(config = {}) {
    const source = resolveStageScheduleSource(config);
    const enabled = Boolean(source.stage_schedule_enabled);
    let stages = normalizeRawStages(source.stage_schedule);
    if (!stages.length) stages = defaultStageScheduleStages();
    stageResolutionState.enabled = enabled;
    stageResolutionState.stages = stages;
    stageResolutionState.selectedIndex = 0;
    stageResolutionState._hydratedFromConfig = true;
}

function normalizeRawStages(raw) {
    if (!Array.isArray(raw)) return [];
    return raw.map((stage, index) => ({
        name: String(stage?.name || `阶段${index + 1}`).trim() || `阶段${index + 1}`,
        subset_index: Math.max(0, Math.round(Number(stage?.subset_index ?? stage?.subsetIndex ?? index) || 0)),
        start_pct: toFraction(stage?.start_pct ?? stage?.startPct ?? 0),
        end_pct: toFraction(stage?.end_pct ?? stage?.endPct ?? 1),
    }));
}

export function normalizedStageResolutionStages() {
    if (!Array.isArray(stageResolutionState.stages) || !stageResolutionState.stages.length) {
        stageResolutionState.stages = defaultStageScheduleStages();
    }
    // Keep order; re-normalize pct and names only.
    stageResolutionState.stages = stageResolutionState.stages.map((stage, index) => ({
        name: String(stage.name || `阶段${index + 1}`).trim() || `阶段${index + 1}`,
        subset_index: Math.max(0, Math.round(Number(stage.subset_index) || 0)),
        start_pct: toFraction(stage.start_pct),
        end_pct: toFraction(stage.end_pct),
    }));
    // Snap cover: first start 0, last end 1, adjacent seams.
    if (stageResolutionState.stages.length) {
        stageResolutionState.stages[0].start_pct = 0;
        stageResolutionState.stages[stageResolutionState.stages.length - 1].end_pct = 1;
        for (let i = 1; i < stageResolutionState.stages.length; i += 1) {
            const prev = stageResolutionState.stages[i - 1];
            const cur = stageResolutionState.stages[i];
            // Prefer previous end as seam when close; otherwise leave and flag in metrics.
            if (Math.abs(prev.end_pct - cur.start_pct) < 1e-6) {
                cur.start_pct = prev.end_pct;
            }
        }
    }
    stageResolutionState.selectedIndex = Math.max(
        0,
        Math.min(stageResolutionState.selectedIndex || 0, stageResolutionState.stages.length - 1),
    );
    return stageResolutionState.stages;
}

export function stageSchedulePayload() {
    const stages = normalizedStageResolutionStages().map((stage) => ({
        name: stage.name,
        subset_index: stage.subset_index,
        start_pct: stage.start_pct,
        end_pct: stage.end_pct,
    }));
    return {
        stage_schedule_enabled: Boolean(stageResolutionState.enabled),
        stage_schedule: stages,
    };
}

export function stageResolutionMetrics() {
    stageResolutionState.enabled = Boolean(stageResolutionState.enabled);
    const stages = normalizedStageResolutionStages();
    const totalSteps = readTotalSteps();
    const options = listSubsetOptions();
    const optionByIndex = new Map(options.map((item) => [item.index, item]));
    const ranges = stages.map((stage, index) => {
        const problems = [];
        const warnings = [];
        if (!(stage.end_pct > stage.start_pct + 1e-9)) problems.push('区间为空');
        if (!optionByIndex.has(stage.subset_index) && options.length) {
            warnings.push('子集索引可能超出当前数据集');
        }
        if (index > 0) {
            const prev = stages[index - 1];
            if (Math.abs(prev.end_pct - stage.start_pct) > 1e-6) {
                if (stage.start_pct < prev.end_pct - 1e-6) problems.push('与上一段重叠');
                else problems.push('与上一段未贴齐');
            }
        }
        if (index === 0 && Math.abs(stage.start_pct) > 1e-6) problems.push('须从 0% 开始');
        if (index === stages.length - 1 && Math.abs(stage.end_pct - 1) > 1e-6) problems.push('须到 100%');
        const opt = optionByIndex.get(stage.subset_index);
        const startStep = totalSteps ? Math.floor(totalSteps * stage.start_pct) : null;
        const endStep = totalSteps ? Math.floor(totalSteps * stage.end_pct) : null;
        return {
            ...stage,
            index,
            resolution: opt?.resolution ?? null,
            subsetLabel: opt?.label || `SUBSET ${stage.subset_index + 1}`,
            startStep,
            endStep,
            steps: startStep != null && endStep != null ? Math.max(0, endStep - startStep) : null,
            problems,
            warnings,
            color: STAGE_COLORS[index % STAGE_COLORS.length],
        };
    });
    const problemCount = ranges.filter((item) => item.problems.length).length;
    const warningCount = ranges.filter((item) => item.warnings.length).length;
    return {
        enabled: stageResolutionState.enabled,
        stages: ranges,
        totalSteps,
        problemCount,
        warningCount,
        selected: ranges[stageResolutionState.selectedIndex] || ranges[0],
        subsetOptions: options,
    };
}

export function stageResolutionStatus(stage) {
    if (stage.problems?.length) return { tone: 'error', text: stage.problems[0] };
    if (stage.warnings?.length) return { tone: 'warning', text: stage.warnings[0] };
    return { tone: 'ok', text: '就绪' };
}

export function createStageResolutionSummary(metrics) {
    const wrap = document.createElement('div');
    wrap.className = 'stage-resolution-summary';
    wrap.appendChild(createStageResolutionEnableControl(metrics.enabled));
    const rows = [
        ['阶段数', `${metrics.stages.length}`],
        ['总步数 S', metrics.totalSteps ? String(metrics.totalSteps) : '未锁定'],
        ['覆盖', metrics.problemCount ? '有问题' : '0–100%'],
        ['检查', metrics.problemCount ? `${metrics.problemCount} 项错误` : (metrics.warningCount ? `${metrics.warningCount} 项提示` : '就绪')],
        ['说明', '按总步数 % 切换子集'],
    ];
    rows.forEach(([label, value]) => {
        const item = document.createElement('div');
        item.className = 'stage-resolution-summary-item';
        const strong = document.createElement('strong');
        strong.textContent = value;
        const span = document.createElement('span');
        span.textContent = label;
        item.append(strong, span);
        wrap.appendChild(item);
    });
    return wrap;
}

function createStageResolutionEnableControl(enabled) {
    const item = document.createElement('label');
    item.className = 'stage-resolution-summary-item stage-resolution-enable-control';
    const input = document.createElement('input');
    input.id = 'stage-resolution-enable-toggle';
    input.type = 'checkbox';
    input.checked = enabled;
    input.addEventListener('change', (event) => {
        stageResolutionState.enabled = Boolean(event.target.checked);
        renderStageResolutionDialog();
    });
    const copy = document.createElement('span');
    const strong = document.createElement('strong');
    strong.textContent = '启用分阶段调度';
    const hint = document.createElement('span');
    hint.textContent = enabled
        ? '训练将按总步数百分比切换数据集子集'
        : '关闭时保持全程使用当前数据集（现有行为）';
    copy.append(strong, hint);
    item.append(input, copy);
    return item;
}

export function createStageResolutionChartPanel() {
    const panel = document.createElement('section');
    panel.className = 'stage-resolution-chart-panel';
    const header = document.createElement('div');
    header.className = 'stage-resolution-panel-head';
    const title = document.createElement('div');
    title.innerHTML = '<strong>时间轴课表</strong><span>色带宽度 = 进度百分比；段数可变（2/3/N）。</span>';
    const actions = document.createElement('div');
    actions.className = 'stage-resolution-actions';
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn btn-small';
    addBtn.textContent = '新增阶段';
    addBtn.addEventListener('click', addStageResolutionPoint);
    const twoBtn = document.createElement('button');
    twoBtn.type = 'button';
    twoBtn.className = 'btn btn-small';
    twoBtn.textContent = '两段模板';
    twoBtn.addEventListener('click', () => applyStageTemplate(2));
    const threeBtn = document.createElement('button');
    threeBtn.type = 'button';
    threeBtn.className = 'btn btn-small';
    threeBtn.textContent = '三段模板';
    threeBtn.addEventListener('click', () => applyStageTemplate(3));
    actions.append(twoBtn, threeBtn, addBtn);
    header.append(title, actions);
    const canvas = document.createElement('canvas');
    canvas.id = 'stage-resolution-chart';
    canvas.width = 720;
    canvas.height = 160;
    canvas.addEventListener('click', selectStageResolutionPointFromCanvas);
    panel.append(header, canvas);
    return panel;
}

export function createStageResolutionEditor(stage, metrics) {
    const aside = document.createElement('aside');
    aside.className = 'stage-resolution-editor';
    const head = document.createElement('div');
    head.className = 'stage-resolution-panel-head';
    const title = document.createElement('div');
    title.innerHTML = '<strong>当前阶段</strong><span>绑定数据集子集，编辑起止 %。</span>';
    head.appendChild(title);
    aside.appendChild(head);
    if (!stage) return aside;

    const fields = document.createElement('div');
    fields.className = 'stage-resolution-fields';
    fields.append(
        createTextField('阶段名', 'name', stage.name),
        createSubsetSelect(stage, metrics.subsetOptions),
        createNumberField('起始 %', 'start_pct', Math.round(stage.start_pct * 1000) / 10),
        createNumberField('结束 %', 'end_pct', Math.round(stage.end_pct * 1000) / 10),
        createReadonlyField('分辨率', stage.resolution != null ? `${stage.resolution}px` : '—'),
        createReadonlyField(
            '步数区间',
            stage.startStep != null
                ? `${stage.startStep}–${stage.endStep}（约 ${stage.steps} 步）`
                : '请先设定 max_train_steps',
        ),
    );
    aside.appendChild(fields);
    return aside;
}

function createTextField(labelText, key, value) {
    const label = document.createElement('label');
    label.className = 'stage-resolution-field';
    const span = document.createElement('span');
    span.textContent = labelText;
    const input = document.createElement('input');
    input.type = 'text';
    input.value = value;
    input.dataset.stageField = key;
    input.addEventListener('change', updateSelectedStageResolutionField);
    label.append(span, input);
    return label;
}

function createNumberField(labelText, key, value) {
    const label = document.createElement('label');
    label.className = 'stage-resolution-field';
    const span = document.createElement('span');
    span.textContent = labelText;
    const input = document.createElement('input');
    input.type = 'number';
    input.min = '0';
    input.max = '100';
    input.step = '0.1';
    input.value = value;
    input.dataset.stageField = key;
    input.addEventListener('change', updateSelectedStageResolutionField);
    label.append(span, input);
    return label;
}

function createReadonlyField(labelText, value) {
    const label = document.createElement('label');
    label.className = 'stage-resolution-field';
    const span = document.createElement('span');
    span.textContent = labelText;
    const output = document.createElement('output');
    output.textContent = value;
    label.append(span, output);
    return label;
}

function createSubsetSelect(stage, options) {
    const label = document.createElement('label');
    label.className = 'stage-resolution-field';
    const span = document.createElement('span');
    span.textContent = '数据集子集';
    const select = document.createElement('select');
    select.dataset.stageField = 'subset_index';
    (options || []).forEach((opt) => {
        const option = document.createElement('option');
        option.value = String(opt.index);
        option.textContent = opt.label;
        if (opt.index === stage.subset_index) option.selected = true;
        select.appendChild(option);
    });
    if (!(options || []).some((opt) => opt.index === stage.subset_index)) {
        const option = document.createElement('option');
        option.value = String(stage.subset_index);
        option.textContent = `SUBSET ${stage.subset_index + 1}（配置中的索引）`;
        option.selected = true;
        select.appendChild(option);
    }
    select.addEventListener('change', updateSelectedStageResolutionField);
    label.append(span, select);
    return label;
}

export function createStageResolutionTable(stages) {
    const section = document.createElement('section');
    section.className = 'stage-resolution-table-panel';
    const head = document.createElement('div');
    head.className = 'stage-resolution-panel-head';
    const title = document.createElement('div');
    title.innerHTML = '<strong>阶段表</strong><span>每行绑定一个子集；N 可变。</span>';
    head.appendChild(title);
    const tableWrap = document.createElement('div');
    tableWrap.className = 'stage-resolution-table-wrap';
    const table = document.createElement('table');
    table.className = 'stage-resolution-table';
    table.innerHTML = '<thead><tr><th>阶段</th><th>子集</th><th>分辨率</th><th>起%</th><th>止%</th><th>步数</th><th>状态</th><th>操作</th></tr></thead>';
    const tbody = document.createElement('tbody');
    stages.forEach((stage) => tbody.appendChild(createStageResolutionTableRow(stage)));
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    section.append(head, tableWrap);
    return section;
}

function createStageResolutionTableRow(stage) {
    const tr = document.createElement('tr');
    tr.className = stage.index === stageResolutionState.selectedIndex ? 'selected' : '';
    tr.dataset.stageIndex = String(stage.index);
    tr.addEventListener('click', () => selectStageResolutionPoint(stage.index));
    const status = stageResolutionStatus(stage);
    tr.append(
        cell(stage.name),
        cell(`#${stage.subset_index}`),
        cell(stage.resolution != null ? String(stage.resolution) : '—'),
        cell(pctLabel(stage.start_pct)),
        cell(pctLabel(stage.end_pct)),
        cell(stage.steps != null ? String(stage.steps) : '—'),
        statusCell(status),
        actionCell(stage),
    );
    return tr;
}

function cell(text) {
    const td = document.createElement('td');
    td.textContent = text;
    return td;
}

function statusCell(status) {
    const td = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = `stage-resolution-status ${status.tone}`;
    badge.textContent = status.text;
    td.appendChild(badge);
    return td;
}

function actionCell(stage) {
    const td = document.createElement('td');
    td.className = 'stage-resolution-actions';
    const up = actionButton('↑', () => moveStageResolutionPoint(stage.index, -1));
    const down = actionButton('↓', () => moveStageResolutionPoint(stage.index, 1));
    const del = actionButton('删', () => deleteStageResolutionPoint(stage.index));
    up.disabled = stage.index <= 0;
    down.disabled = stage.index >= stageResolutionState.stages.length - 1;
    del.disabled = stageResolutionState.stages.length <= 1;
    td.append(up, down, del);
    return td;
}

function actionButton(text, handler) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-small';
    btn.textContent = text;
    btn.addEventListener('click', (event) => {
        event.stopPropagation();
        handler();
    });
    return btn;
}

function updateSelectedStageResolutionField(event) {
    const key = event.target.dataset.stageField;
    let value = event.target.value;
    if (key === 'subset_index') value = Math.max(0, Math.round(Number(value) || 0));
    if (key === 'start_pct' || key === 'end_pct') value = toFraction(Number(value));
    updateStageResolutionStage(stageResolutionState.selectedIndex, { [key]: value });
}

function updateStageResolutionStage(index, patch) {
    const stages = normalizedStageResolutionStages();
    if (!stages[index]) return;
    const next = { ...stages[index], ...patch };
    // Keep adjacent seams when editing end/start.
    if ('end_pct' in patch && stages[index + 1]) {
        stages[index + 1] = { ...stages[index + 1], start_pct: next.end_pct };
    }
    if ('start_pct' in patch && stages[index - 1]) {
        stages[index - 1] = { ...stages[index - 1], end_pct: next.start_pct };
    }
    stages[index] = next;
    stageResolutionState.stages = stages;
    renderStageResolutionDialog();
}

function applyStageTemplate(count) {
    const n = Math.max(1, Math.min(12, Math.round(Number(count) || 2)));
    const options = listSubsetOptions();
    const stages = [];
    for (let i = 0; i < n; i += 1) {
        stages.push({
            name: `阶段${i + 1}`,
            subset_index: Math.min(i, Math.max(0, options.length - 1)),
            start_pct: i / n,
            end_pct: (i + 1) / n,
        });
    }
    stageResolutionState.stages = stages;
    stageResolutionState.selectedIndex = 0;
    renderStageResolutionDialog();
}

function addStageResolutionPoint() {
    const stages = normalizedStageResolutionStages();
    const last = stages[stages.length - 1];
    if (!last) {
        stageResolutionState.stages = defaultStageScheduleStages();
        renderStageResolutionDialog();
        return;
    }
    const mid = (last.start_pct + last.end_pct) / 2;
    last.end_pct = mid;
    stages.push({
        name: `阶段${stages.length + 1}`,
        subset_index: last.subset_index,
        start_pct: mid,
        end_pct: 1,
    });
    stageResolutionState.stages = stages;
    stageResolutionState.selectedIndex = stages.length - 1;
    renderStageResolutionDialog();
}

function deleteStageResolutionPoint(index) {
    const stages = normalizedStageResolutionStages();
    if (stages.length <= 1) return;
    stages.splice(index, 1);
    stages[0].start_pct = 0;
    stages[stages.length - 1].end_pct = 1;
    // Re-equalize seams lightly.
    for (let i = 1; i < stages.length; i += 1) {
        if (stages[i].start_pct < stages[i - 1].end_pct) {
            stages[i].start_pct = stages[i - 1].end_pct;
        }
    }
    stageResolutionState.stages = stages;
    stageResolutionState.selectedIndex = Math.max(0, Math.min(index, stages.length - 1));
    renderStageResolutionDialog();
}

function moveStageResolutionPoint(index, direction) {
    const stages = normalizedStageResolutionStages();
    const nextIndex = index + direction;
    if (nextIndex < 0 || nextIndex >= stages.length) return;
    [stages[index], stages[nextIndex]] = [stages[nextIndex], stages[index]];
    // After reorder, redistribute equal slices to keep a valid cover.
    const n = stages.length;
    stages.forEach((stage, i) => {
        stage.start_pct = i / n;
        stage.end_pct = (i + 1) / n;
    });
    stageResolutionState.stages = stages;
    stageResolutionState.selectedIndex = nextIndex;
    renderStageResolutionDialog();
}

function selectStageResolutionPoint(index) {
    stageResolutionState.selectedIndex = index;
    renderStageResolutionDialog();
}

function selectStageResolutionPointFromCanvas(event) {
    const canvas = event.currentTarget;
    const bands = canvas._stageResolutionBands || [];
    if (!bands.length) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const hit = bands.find((band) => x >= band.x0 && x <= band.x1) || bands[0];
    selectStageResolutionPoint(hit.index);
}

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
        const cover = stages.reduce((sum, stage) => {
            const start = Math.max(0, Math.min(1, Number(stage.start_pct) || 0));
            const end = Math.max(0, Math.min(1, Number(stage.end_pct) || 0));
            return sum + Math.max(0, end - start);
        }, 0);
        detail.textContent = `${stages.length} 段 · 覆盖约 ${Math.round(cover * 1000) / 10}%${dirty ? ' · 未保存' : ''}`;
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

// Compatibility exports used by old chunk imports / quick-preset panel co-location.
export {
    createFillGlobalModelPathsButton,
    createResourceQuickPresetsButton,
    createResourceQuickPresetPanel,
    createNoDatasetRegularizationQuickPresetsButton,
    createNoDatasetRegularizationQuickPresetPanel,
} from './stage-resolution-presets.js?v=module-bootstrap-20260707-93';
