/**
 * Stage schedule widgets: summary, editor, table, and stage mutations.
 */
import { getAppContext } from '../anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260711-ir1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    STAGE_COLORS,
    clamp01,
    defaultStageScheduleStages,
    hydrateStageScheduleFromConfig,
    listSubsetOptions,
    normalizedStageResolutionStages,
    pctLabel,
    pickDatasetRows,
    readTotalSteps,
    resolveStageScheduleSource,
    stageResolutionMetrics,
    stageResolutionStatus,
    stageSchedulePayload,
    toFraction,
    normalizeRawStages,
} from './stage-resolution-model.js?v=module-bootstrap-20260711-ir1';

const ctx = getAppContext();
const configState = getConfigState();
const stageResolutionState = configState.stageResolutionState;

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
    const equalBtn = document.createElement('button');
    equalBtn.type = 'button';
    equalBtn.className = 'btn btn-small';
    equalBtn.textContent = '均分当前段';
    equalBtn.addEventListener('click', () => {
        applyStageTemplate(Math.max(1, stageResolutionState.stages.length || 2));
    });
    actions.append(twoBtn, threeBtn, equalBtn, addBtn);
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

export function selectStageResolutionPointFromCanvas(event) {
    const canvas = event.currentTarget;
    const bands = canvas._stageResolutionBands || [];
    if (!bands.length) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const hit = bands.find((band) => x >= band.x0 && x <= band.x1) || bands[0];
    selectStageResolutionPoint(hit.index);
}
