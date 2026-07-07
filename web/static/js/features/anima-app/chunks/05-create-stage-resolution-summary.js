/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { NO_DATASET_REGULARIZATION_QUICK_PRESETS, RESOURCE_QUICK_PRESETS } from '../helpers/app-constants.js?v=module-bootstrap-20260707-93';
import { setFieldInputValue } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260707-93';
import { handleFormFieldChange } from './14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260707-93';
import { normalizedStageResolutionStages, renderStageResolutionDialog, stageResolutionMetrics, stageResolutionStatus } from './04-create-config-group-entry.js?v=module-bootstrap-20260707-93';
import { fillGlobalModelPathsIntoConfigForm, resourceQuickCurrentValue, strongerSelectiveCheckpointValue } from './06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260707-93';
import {
    setTomlStatus,
} from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260707-93';
import { getAppContext } from '../helpers/app-context-bridge.js?v=module-bootstrap-20260707-93';

const ctx = getAppContext();

    export function createStageResolutionSummary(metrics) {
        const wrap = document.createElement('div');
        wrap.className = 'stage-resolution-summary';
        const rows = [
            ['调度状态', metrics.enabled ? '已启用' : '未启用'],
            ['阶段数', `${metrics.stages.length}`],
            ['预计 steps', `${metrics.totalSteps}`],
            ['配置检查', metrics.problemCount ? `${metrics.problemCount} 项错误` : (metrics.warningCount ? `${metrics.warningCount} 项提示` : '就绪')],
            ['图片统计', '待接入'],
        ];
        wrap.appendChild(createStageResolutionEnableControl(metrics.enabled));
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
            setStageResolutionEnabled(event.target.checked);
        });
        const copy = document.createElement('span');
        const strong = document.createElement('strong');
        strong.textContent = '启用阶段调度';
        const hint = document.createElement('span');
        hint.textContent = enabled ? '将用于阶段方案' : '草稿，不影响训练';
        copy.append(strong, hint);
        item.append(input, copy);
        return item;
    }

    function setStageResolutionEnabled(enabled) {
        stageResolutionState.enabled = Boolean(enabled);
        renderStageResolutionDialog();
    }

    export function createStageResolutionChartPanel() {
        const panel = document.createElement('section');
        panel.className = 'stage-resolution-chart-panel';
        const header = document.createElement('div');
        header.className = 'stage-resolution-panel-head';
        const title = document.createElement('div');
        title.innerHTML = '<strong>阶段折线</strong><span>点表示阶段，阴影表示该阶段的单边范围。</span>';
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small';
        addBtn.textContent = '新增阶段';
        addBtn.addEventListener('click', addStageResolutionPoint);
        header.append(title, addBtn);
        const canvas = document.createElement('canvas');
        canvas.id = 'stage-resolution-chart';
        canvas.width = 720;
        canvas.height = 280;
        canvas.addEventListener('click', selectStageResolutionPointFromCanvas);
        panel.append(header, canvas);
        return panel;
    }

    export function createStageResolutionEditor(stage) {
        const aside = document.createElement('aside');
        aside.className = 'stage-resolution-editor';
        const head = document.createElement('div');
        head.className = 'stage-resolution-panel-head';
        const title = document.createElement('div');
        title.innerHTML = '<strong>当前点编辑器</strong><span>修改后立即同步折线图和阶段表。</span>';
        head.appendChild(title);
        aside.appendChild(head);
        if (!stage) return aside;

        const fields = document.createElement('div');
        fields.className = 'stage-resolution-fields';
        fields.append(
            createStageResolutionInput('阶段名', 'name', stage.name, 'text'),
            createStageResolutionInput('epochs', 'epochs', stage.epochs, 'number'),
            createStageResolutionInput('单边最大值', 'maxSide', stage.maxSide, 'number'),
            createStageResolutionInput('向下波动', 'downRange', stage.downRange, 'number'),
            createStageResolutionReadonly('单边最小值', `${Math.max(0, stage.minSide || 0)}`),
            createStageResolutionReadonly('预计图片数', '待统计'),
            createStageResolutionRepeats(stage)
        );
        aside.appendChild(fields);
        return aside;
    }

    function createStageResolutionInput(labelText, key, value, type) {
        const label = document.createElement('label');
        label.className = 'stage-resolution-field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = type;
        input.value = value;
        input.dataset.stageField = key;
        if (type === 'number') {
            input.min = key === 'epochs' ? '1' : '0';
            input.step = '1';
        }
        input.addEventListener('input', updateSelectedStageResolutionField);
        label.append(span, input);
        return label;
    }

    function createStageResolutionReadonly(labelText, value) {
        const label = document.createElement('label');
        label.className = 'stage-resolution-field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const output = document.createElement('output');
        output.textContent = value;
        label.append(span, output);
        return label;
    }

    function createStageResolutionRepeats(stage) {
        const wrap = document.createElement('div');
        wrap.className = 'stage-resolution-field stage-resolution-repeat-field';
        const label = document.createElement('label');
        const check = document.createElement('input');
        check.type = 'checkbox';
        check.checked = stage.manualRepeats;
        check.addEventListener('change', (event) => {
            updateStageResolutionStage(stage.index, { manualRepeats: event.target.checked });
        });
        label.append(check, document.createTextNode('手动 repeats'));
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '1';
        input.step = '1';
        input.value = stage.autoRepeats;
        input.disabled = !stage.manualRepeats;
        input.addEventListener('input', (event) => {
            updateStageResolutionStage(stage.index, { repeats: Math.max(1, Math.round(Number(event.target.value) || 1)) });
        });
        wrap.append(label, input);
        return wrap;
    }

    export function createStageResolutionTable(stages) {
        const section = document.createElement('section');
        section.className = 'stage-resolution-table-panel';
        const head = document.createElement('div');
        head.className = 'stage-resolution-panel-head';
        const title = document.createElement('div');
        title.innerHTML = '<strong>阶段表</strong><span>每行对应一个阶段点。</span>';
        head.appendChild(title);
        const tableWrap = document.createElement('div');
        tableWrap.className = 'stage-resolution-table-wrap';
        const table = document.createElement('table');
        table.className = 'stage-resolution-table';
        table.innerHTML = '<thead><tr><th>阶段</th><th>epochs</th><th>单边最大</th><th>向下波动</th><th>step 范围</th><th>分辨率范围</th><th>图片</th><th>repeats</th><th>状态</th><th>操作</th></tr></thead>';
        const tbody = document.createElement('tbody');
        stages.forEach((stage) => tbody.appendChild(createStageResolutionTableRow(stage)));
        table.appendChild(tbody);
        tableWrap.appendChild(table);
        section.append(head, tableWrap);
        return section;
    }

    function createStageResolutionTableRow(stage) {
        const tr = document.createElement('tr');
        const selected = stage.index === stageResolutionState.selectedIndex;
        const status = stageResolutionStatus(stage);
        tr.className = selected ? 'selected' : '';
        tr.dataset.stageIndex = String(stage.index);
        tr.append(
            stageResolutionTableInputCell(stage, 'name', stage.name, 'text'),
            stageResolutionTableInputCell(stage, 'epochs', stage.epochs, 'number'),
            stageResolutionTableInputCell(stage, 'maxSide', stage.maxSide, 'number'),
            stageResolutionTableInputCell(stage, 'downRange', stage.downRange, 'number'),
            stageResolutionTableCell(`${stage.startStep}-${stage.endStep}`),
            stageResolutionTableCell(`${Math.max(0, stage.minSide)}-${stage.maxSide}`),
            stageResolutionTableCell('待统计'),
            stageResolutionTableCell(`${stage.autoRepeats}`),
            stageResolutionStatusCell(status),
            stageResolutionActionCell(stage)
        );
        tr.addEventListener('click', (event) => {
            if (event.target.closest('button')) return;
            selectStageResolutionPoint(stage.index);
        });
        return tr;
    }

    function stageResolutionTableInputCell(stage, key, value, type) {
        const td = document.createElement('td');
        const input = document.createElement('input');
        input.className = 'stage-resolution-table-input';
        input.type = type;
        input.value = value;
        if (type === 'number') {
            input.min = key === 'epochs' ? '1' : '0';
            input.step = '1';
        }
        input.addEventListener('click', (event) => event.stopPropagation());
        input.addEventListener('input', (event) => {
            const next = key === 'name' ? event.target.value : Number(event.target.value);
            updateStageResolutionStage(stage.index, { [key]: next });
        });
        td.appendChild(input);
        return td;
    }

    function stageResolutionTableCell(text) {
        const td = document.createElement('td');
        td.textContent = text;
        return td;
    }

    function stageResolutionStatusCell(status) {
        const td = document.createElement('td');
        const badge = document.createElement('span');
        badge.className = `stage-resolution-status ${status.tone}`;
        badge.textContent = status.text;
        td.appendChild(badge);
        return td;
    }

    function stageResolutionActionCell(stage) {
        const td = document.createElement('td');
        td.className = 'stage-resolution-actions';
        const up = stageResolutionActionButton('↑', '上移', () => moveStageResolutionPoint(stage.index, -1));
        const down = stageResolutionActionButton('↓', '下移', () => moveStageResolutionPoint(stage.index, 1));
        const del = stageResolutionActionButton('删', '删除', () => deleteStageResolutionPoint(stage.index));
        up.disabled = stage.index <= 0;
        down.disabled = stage.index >= stageResolutionState.stages.length - 1;
        del.disabled = stageResolutionState.stages.length <= 1;
        td.append(up, down, del);
        return td;
    }

    function stageResolutionActionButton(text, title, handler) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-small';
        btn.textContent = text;
        btn.title = title;
        btn.addEventListener('click', handler);
        return btn;
    }

    function updateSelectedStageResolutionField(event) {
        const key = event.target.dataset.stageField;
        const value = key === 'name' ? event.target.value : Number(event.target.value);
        updateStageResolutionStage(stageResolutionState.selectedIndex, { [key]: value });
    }

    function updateStageResolutionStage(index, patch) {
        const stages = normalizedStageResolutionStages();
        if (!stages[index]) return;
        stageResolutionState.stages[index] = { ...stages[index], ...patch };
        renderStageResolutionDialog();
    }

    function addStageResolutionPoint() {
        const stages = normalizedStageResolutionStages();
        const last = stages[stages.length - 1] || { maxSide: 1024, downRange: 256 };
        stages.push({
            name: `EP${stages.length + 1}`,
            epochs: 1,
            maxSide: Math.max(256, Number(last.maxSide || 1024) + 512),
            downRange: Math.max(64, Number(last.downRange || 256)),
            manualRepeats: false,
            repeats: 1,
        });
        stageResolutionState.selectedIndex = stages.length - 1;
        renderStageResolutionDialog();
    }

    function deleteStageResolutionPoint(index) {
        const stages = normalizedStageResolutionStages();
        if (stages.length <= 1) return;
        stages.splice(index, 1);
        stageResolutionState.selectedIndex = Math.max(0, Math.min(index, stages.length - 1));
        renderStageResolutionDialog();
    }

    function moveStageResolutionPoint(index, direction) {
        const stages = normalizedStageResolutionStages();
        const nextIndex = index + direction;
        if (nextIndex < 0 || nextIndex >= stages.length) return;
        [stages[index], stages[nextIndex]] = [stages[nextIndex], stages[index]];
        stageResolutionState.selectedIndex = nextIndex;
        renderStageResolutionDialog();
    }

    function selectStageResolutionPoint(index) {
        stageResolutionState.selectedIndex = index;
        renderStageResolutionDialog();
    }

    function selectStageResolutionPointFromCanvas(event) {
        const canvas = event.currentTarget;
        const points = canvas._stageResolutionPoints || [];
        if (!points.length) return;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        let nearest = points[0];
        for (const point of points) {
            if (Math.abs(point.x - x) < Math.abs(nearest.x - x)) nearest = point;
        }
        selectStageResolutionPoint(nearest.index);
    }

    export function drawStageResolutionChart() {
        const canvas = document.getElementById('stage-resolution-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const metrics = stageResolutionMetrics();
        const stages = metrics.stages;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(320, Math.floor(rect.width || 720));
        const height = Math.max(220, Math.floor(rect.height || 280));
        const ratio = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);

        const styles = getComputedStyle(document.documentElement);
        const accent = styles.getPropertyValue('--accent').trim() || '#4fc3f7';
        const warning = styles.getPropertyValue('--warning').trim() || '#f0c36a';
        const danger = styles.getPropertyValue('--danger').trim() || '#ef5350';
        const grid = styles.getPropertyValue('--chart-grid').trim() || '#2a3a5e';
        const text = styles.getPropertyValue('--text-dim').trim() || '#8892a4';
        const success = styles.getPropertyValue('--success').trim() || '#22c55e';
        const pad = { top: 24, right: 38, bottom: 38, left: 46 };
        const plotW = width - pad.left - pad.right;
        const plotH = height - pad.top - pad.bottom;
        if (!stages.length || plotW <= 0 || plotH <= 0) return;

        const values = stages.flatMap((stage) => [stage.maxSide, Math.max(0, stage.minSide)]);
        let minY = Math.min(...values);
        let maxY = Math.max(...values);
        if (minY === maxY) {
            minY -= 128;
            maxY += 128;
        }
        minY = Math.max(0, Math.floor(minY / 128) * 128);
        maxY = Math.ceil(maxY / 128) * 128;
        const yFor = (value) => pad.top + (1 - ((value - minY) / Math.max(1, maxY - minY))) * plotH;
        const xFor = (index) => stages.length === 1
            ? pad.left + plotW / 2
            : pad.left + (plotW * index / (stages.length - 1));

        ctx.strokeStyle = grid;
        ctx.lineWidth = 0.5;
        ctx.fillStyle = text;
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i += 1) {
            const y = pad.top + (plotH * i / 4);
            const value = maxY - ((maxY - minY) * i / 4);
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(width - pad.right, y);
            ctx.stroke();
            ctx.fillText(String(Math.round(value)), pad.left - 8, y + 3);
        }

        const points = [];
        stages.forEach((stage, index) => {
            const x = xFor(index);
            const yMax = yFor(stage.maxSide);
            const yMin = yFor(Math.max(0, stage.minSide));
            const status = stageResolutionStatus(stage);
            const color = status.tone === 'error' ? danger : (status.tone === 'warning' ? warning : accent);
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.16;
            ctx.fillRect(x - 24, yMax, 48, Math.max(2, yMin - yMax));
            ctx.globalAlpha = 1;
            points.push({ x, y: yMax, index });
        });

        ctx.strokeStyle = accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((point, index) => {
            if (index === 0) ctx.moveTo(point.x, point.y);
            else ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();

        points.forEach((point) => {
            const stage = stages[point.index];
            const status = stageResolutionStatus(stage);
            const selected = point.index === stageResolutionState.selectedIndex;
            const color = status.tone === 'error' ? danger : (status.tone === 'warning' ? warning : success);
            ctx.fillStyle = color;
            ctx.strokeStyle = selected ? warning : accent;
            ctx.lineWidth = selected ? 3 : 1.5;
            ctx.beginPath();
            ctx.arc(point.x, point.y, selected ? 6 : 4.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = text;
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`${stage.startStep}-${stage.endStep}`, point.x, height - 18);
        });
        canvas._stageResolutionPoints = points;
    }

    export function createFillGlobalModelPathsButton() {
        const btn = document.createElement('button');
        btn.id = 'btn-fill-global-model-paths';
        btn.type = 'button';
        btn.className = 'btn btn-small config-group-title-action';
        btn.textContent = '填写全局路径配置';
        btn.title = '用全局设置里的三项基础模型路径覆盖当前配置表单';
        btn.addEventListener('click', () => {
            fillGlobalModelPathsIntoConfigForm().catch((e) => {
                setTomlStatus('error', '填写全局路径配置失败: ' + e.message);
            });
        });
        return btn;
    }

    function createConfigQuickPresetsButton(options, content, collapseBtn) {
        const groupName = options.groupName || '';
        const btn = document.createElement('button');
        if (options.id) btn.id = options.id;
        btn.type = 'button';
        btn.className = ['btn btn-small config-group-title-action config-quick-preset-toggle', options.className || ''].filter(Boolean).join(' ');
        btn.textContent = options.text || '快速填写';
        btn.title = options.showTitle || `显示${groupName}预设，一键填写当前表单`;
        btn.setAttribute('aria-expanded', 'false');
        btn.addEventListener('click', () => {
            const panel = content.querySelector(`.${options.panelClass}`);
            if (!panel) return;
            const nextVisible = panel.hidden;
            panel.hidden = !nextVisible;
            btn.classList.toggle('active', nextVisible);
            btn.setAttribute('aria-expanded', String(nextVisible));
            btn.title = nextVisible
                ? (options.hideTitle || `收起${groupName}快速预设`)
                : (options.showTitle || `显示${groupName}预设，一键填写当前表单`);
            if (nextVisible && content.hidden) {
                content.hidden = false;
                collapseBtn.textContent = '收起';
                collapseBtn.setAttribute('aria-expanded', 'true');
                collapseBtn.title = '收起这个配置区';
                if (groupName) {
                    configFormState.expandedGroups.add(groupName);
                    configFormState.collapsedGroups.delete(groupName);
                }
            }
        });
        return btn;
    }

    function createConfigQuickPresetPanel(options) {
        const panel = document.createElement('div');
        panel.className = ['config-quick-presets', options.panelClass || ''].filter(Boolean).join(' ');
        panel.hidden = true;
        panel.setAttribute('aria-label', options.ariaLabel || `${options.groupName || '配置'}快速预设`);

        const label = document.createElement('span');
        label.className = 'config-quick-label';
        label.textContent = options.label || '快速预设';
        panel.appendChild(label);

        for (const preset of options.presets || []) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'btn btn-small config-quick-preset-btn';
            if (options.datasetKey) btn.dataset[options.datasetKey] = preset.id;
            btn.textContent = preset.label;
            btn.title = preset.note;
            btn.addEventListener('click', () => options.applyPreset(preset));
            panel.appendChild(btn);
        }
        return panel;
    }

    export function createResourceQuickPresetsButton(content, collapseBtn) {
        return createConfigQuickPresetsButton({
            id: 'btn-resource-quick-presets',
            className: 'config-resource-quick-toggle',
            panelClass: 'config-resource-quick-presets',
            groupName: '显存与速度优化',
            showTitle: '显示显存与速度优化预设，一键填写当前表单',
            hideTitle: '收起显存与速度优化快速预设',
        }, content, collapseBtn);
    }

    export function createResourceQuickPresetPanel() {
        return createConfigQuickPresetPanel({
            panelClass: 'config-resource-quick-presets',
            groupName: '显存与速度优化',
            ariaLabel: '显存与速度优化快速预设',
            label: '快速预设',
            presets: RESOURCE_QUICK_PRESETS,
            datasetKey: 'resourcePreset',
            applyPreset: applyResourceQuickPreset,
        });
    }

    function applyResourceQuickPreset(preset) {
        for (const [key, value] of Object.entries(resourceQuickPresetPatch(preset))) {
            setFieldInputValue(key, value);
        }
        handleFormFieldChange();
        setTomlStatus('ok', `已填写显存与速度优化预设: ${preset.label}`);
    }

    function resourceQuickPresetPatch(preset) {
        const patch = {};
        for (const [key, value] of Object.entries(preset?.values || {})) {
            patch[key] = resourceQuickPresetValue(preset, key, value);
        }

        const selectiveCheckpoint = String(
            patch.selective_checkpoint ?? resourceQuickCurrentValue('selective_checkpoint') ?? 'off'
        ).trim();
        if (selectiveCheckpoint && selectiveCheckpoint !== 'off') {
            patch.gradient_checkpointing = false;
            patch.cpu_offload_checkpointing = false;
            patch.unsloth_offload_checkpointing = false;
        }

        const blocksToSwap = Number(patch.blocks_to_swap ?? resourceQuickCurrentValue('blocks_to_swap'));
        if (Number.isFinite(blocksToSwap) && blocksToSwap > 0) {
            patch.cpu_offload_checkpointing = false;
            patch.unsloth_offload_checkpointing = false;
        }
        return patch;
    }

    function resourceQuickPresetValue(preset, key, value) {
        const strategy = preset?.merge?.[key] || '';
        if (strategy === 'max') {
            const current = Number(resourceQuickCurrentValue(key));
            const next = Number(value);
            if (Number.isFinite(current) && Number.isFinite(next)) {
                return Math.max(current, next);
            }
        }
        if (strategy === 'checkpoint_strength_max') {
            return strongerSelectiveCheckpointValue(resourceQuickCurrentValue(key), value);
        }
        return value;
    }

    export function createNoDatasetRegularizationQuickPresetsButton(content, collapseBtn) {
        return createConfigQuickPresetsButton({
            id: 'btn-no-dataset-regularization-quick-presets',
            className: 'config-no-dataset-regularization-quick-toggle',
            panelClass: 'config-no-dataset-regularization-quick-presets',
            groupName: '无数据集正则化',
            showTitle: '显示无数据集正则化预设，一键填写当前表单',
            hideTitle: '收起无数据集正则化快速预设',
        }, content, collapseBtn);
    }

    export function createNoDatasetRegularizationQuickPresetPanel() {
        return createConfigQuickPresetPanel({
            panelClass: 'config-no-dataset-regularization-quick-presets',
            groupName: '无数据集正则化',
            ariaLabel: '无数据集正则化快速预设',
            label: '快速填写',
            presets: NO_DATASET_REGULARIZATION_QUICK_PRESETS,
            datasetKey: 'noDatasetRegularizationPreset',
            applyPreset: applyNoDatasetRegularizationQuickPreset,
        });
    }

    function applyNoDatasetRegularizationQuickPreset(preset) {
        for (const [key, value] of Object.entries(preset.values || {})) {
            setFieldInputValue(key, value);
        }
        handleFormFieldChange();
        const extra = preset.id === 'dop_roles' ? '，还需要填写泛化类别作为 DOP 类提示，例如 woman / character，并重新生成文本缓存' : '';
        setTomlStatus('ok', `已填写无数据集正则化预设: ${preset.label}${extra}`);
    }
