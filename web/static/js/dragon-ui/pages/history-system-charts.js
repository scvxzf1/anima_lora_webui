/* Interactive single-GPU resource chart for training history details. */

import { areaSvgPath, smoothSvgPath } from './trend-utils.js?v=dragon-ui-20260825v1';
import { bindLatestPointerMove, pointerEventName } from '../pointer-frame.js?v=dragon-ui-20260826v1';

const WIDTH = 900;
const HEIGHT = 300;
const PADDING = { top: 44, right: 74, bottom: 34, left: 58 };
const SYSTEM_SERIES = Object.freeze([
    { id: 'vram', label: 'VRAM 占用', shortLabel: 'VRAM', key: 'vramUsed', unit: 'gb', axis: 'vram', color: 'var(--dragon-accent)' },
    { id: 'gpu', label: 'GPU 占用', shortLabel: 'GPU', key: 'gpuUtil', unit: 'percent', axis: 'percent', color: 'var(--dragon-success)' },
    { id: 'temp', label: '核心温度', shortLabel: '温度', key: 'gpuTemp', unit: 'temperature', axis: 'percent', color: 'var(--dragon-warning)' },
]);

export function renderHistorySystemCharts(records = [], limits = {}, options = {}) {
    const rows = normalizeSystemRecords(records);
    const state = normalizeOptions(options);
    return `<section class="dragon-history-panel dragon-history-system-section dragon-reveal" aria-labelledby="dragon-history-system-title">
        <div class="dragon-history-panel-head dragon-history-chart-head">
            <div><span class="dragon-eyebrow">单卡资源监控</span><h2 id="dragon-history-system-title">VRAM 与 GPU 状态</h2></div>
            ${rows.length ? `<span>${systemSampleLabel(rows.length, limits)}</span>` : ''}
        </div>
        ${rows.length ? `${renderSystemSummary(rows)}${renderSystemControls(state)}
            <div class="dragon-history-system-chart-shell" data-history-system-chart-container>${renderSystemChart(rows, state)}</div>`
        : '<div class="dragon-history-inline-empty"><p>当前任务没有可用的 GPU 系统采样。</p></div>'}
    </section>`;
}

export function bindHistorySystemCharts(root, records = []) {
    const section = root?.querySelector('.dragon-history-system-section');
    const container = section?.querySelector('[data-history-system-chart-container]');
    const rows = normalizeSystemRecords(records);
    if (!section || !container || !rows.length) return () => {};

    const state = normalizeOptions();
    let unbindHover = () => {};
    const render = () => {
        unbindHover();
        container.innerHTML = renderSystemChart(rows, state);
        unbindHover = bindSystemChartHover(container.querySelector('[data-history-system-chart]'), rows, state);
    };
    const listeners = [];
    section.querySelectorAll('[data-history-system-toggle]').forEach((input) => {
        const onChange = () => {
            state[input.dataset.historySystemToggle] = input.checked;
            render();
        };
        input.addEventListener('change', onChange);
        listeners.push(() => input.removeEventListener('change', onChange));
    });
    render();
    return () => {
        unbindHover();
        listeners.forEach((remove) => remove());
    };
}

function renderSystemSummary(rows) {
    return `<div class="dragon-history-system-summary" aria-label="GPU 资源摘要">
        ${SYSTEM_SERIES.map((spec) => {
            const values = rows.map((row) => row[spec.key]).filter((value) => value != null);
            const latest = latestValue(rows, spec.key);
            const peak = values.length ? Math.max(...values) : null;
            return `<div data-tone="${spec.id}"><span>${spec.label}</span><strong>${formatSystemValue(latest, spec.unit)}</strong><small>峰值 ${formatSystemValue(peak, spec.unit)}</small></div>`;
        }).join('')}
    </div>`;
}

function renderSystemControls(state) {
    return `<div class="dragon-history-chart-controls dragon-history-system-controls" aria-label="GPU 曲线显示配置">
        <span class="dragon-history-chart-control-label">曲线</span>
        ${SYSTEM_SERIES.map((spec) => `<label><input type="checkbox" data-history-system-toggle="${spec.id}" ${state[spec.id] ? 'checked' : ''}><i class="dragon-chart-swatch" data-tone="${spec.id}"></i><span>${spec.label}</span></label>`).join('')}
        <span class="dragon-history-system-scale-note">左轴 GB · 右轴 % / °C</span>
    </div>`;
}

function renderSystemChart(rows, state) {
    const visible = SYSTEM_SERIES.filter((spec) => state[spec.id]);
    const layout = chartLayout(rows);
    const vramDomain = vramChartDomain(rows);
    const grid = renderGrid(layout, vramDomain);
    const series = visible.map((spec) => renderSeries(rows, spec, layout, vramDomain)).join('');
    const markers = visible.map((spec) => `<circle data-history-system-hover-point="${spec.id}" r="4.5" fill="${spec.color}" stroke="var(--dragon-bg-secondary)" stroke-width="1.5" hidden/>`).join('');
    return `<div class="dragon-chart-glass-tooltip dragon-history-system-tooltip" data-history-system-tooltip hidden></div>
        <svg class="dragon-history-system-chart" data-history-system-chart viewBox="0 0 ${WIDTH} ${HEIGHT}" preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby="dragon-history-system-chart-title">
            <title id="dragon-history-system-chart-title">单卡 VRAM、GPU 占用率与温度趋势，移动指针可查看采样详情</title>
            ${grid}${series}
            <g data-history-system-hover hidden>
                <line data-history-system-hover-guide y1="${PADDING.top}" y2="${HEIGHT - PADDING.bottom}" class="dragon-history-system-hover-guide"/>
                ${markers}
            </g>
            <rect x="${PADDING.left}" y="${PADDING.top}" width="${layout.plotW}" height="${layout.plotH}" class="dragon-history-system-hitarea"/>
        </svg>`;
}

function renderGrid(layout, vramDomain) {
    return [0, .25, .5, .75, 1].map((ratio) => {
        const y = PADDING.top + ratio * layout.plotH;
        const vramValue = vramDomain.max - ratio * (vramDomain.max - vramDomain.min);
        const percentValue = Math.round(100 - ratio * 100);
        return `<line x1="${PADDING.left}" y1="${y}" x2="${WIDTH - PADDING.right}" y2="${y}" class="dragon-history-system-grid-line"/>
            <text x="${PADDING.left - 9}" y="${y + 4}" class="dragon-history-system-axis-label" text-anchor="end">${formatAxisGb(vramValue)}</text>
            <text x="${WIDTH - PADDING.right + 9}" y="${y + 4}" class="dragon-history-system-axis-label dragon-history-system-axis-label-right">${percentValue}</text>`;
    }).join('');
}

function renderSeries(rows, spec, layout, vramDomain) {
    const points = seriesPoints(rows, spec);
    const y = spec.axis === 'vram'
        ? (value) => axisY(value, vramDomain.min, vramDomain.max, layout.plotH)
        : (value) => axisY(value, 0, 100, layout.plotH);
    const pairs = points.map((point) => [layout.x(point.x), y(point.value)]);
    const line = smoothSvgPath(pairs, .2);
    const area = spec.id === 'vram' ? areaSvgPath(line, pairs, HEIGHT - PADDING.bottom) : '';
    return `${area ? `<path d="${area}" class="dragon-history-system-area" data-history-system-series="vram-area"/>` : ''}
        <path d="${line}" class="dragon-history-system-line" data-tone="${spec.id}" data-history-system-series="${spec.id}"/>`;
}

function bindSystemChartHover(svg, rows, state) {
    if (!svg || !rows.length) return () => {};
    const layout = chartLayout(rows);
    const vramDomain = vramChartDomain(rows);
    const hitarea = svg.querySelector('.dragon-history-system-hitarea');
    const hover = svg.querySelector('[data-history-system-hover]');
    const guide = svg.querySelector('[data-history-system-hover-guide]');
    const tooltip = svg.parentElement?.querySelector('[data-history-system-tooltip]');
    const points = new Map(SYSTEM_SERIES.map((spec) => [
        spec.id,
        svg.querySelector(`[data-history-system-hover-point="${spec.id}"]`),
    ]));
    const elements = { hover, guide, tooltip, points, lastRow: null };
    const onMove = (event) => updateSystemHover(svg, event, rows, state, layout, vramDomain, elements);
    const hide = () => {
        elements.lastRow = null;
        hover?.setAttribute('hidden', '');
        if (tooltip) tooltip.hidden = true;
    };
    const unbindMove = bindLatestPointerMove(hitarea, onMove);
    const leaveEvent = pointerEventName(svg, 'pointerleave', 'mouseleave');
    hitarea?.addEventListener(leaveEvent, hide);
    svg.addEventListener(leaveEvent, hide);
    return () => {
        unbindMove();
        hitarea?.removeEventListener(leaveEvent, hide);
        svg.removeEventListener(leaveEvent, hide);
    };
}

function updateSystemHover(svg, event, rows, state, layout, vramDomain, elements) {
    const viewX = clientPointToSvg(svg, event.clientX, event.clientY).x;
    const ratio = clamp((viewX - PADDING.left) / layout.plotW, 0, 1);
    const targetX = layout.minX + ratio * Math.max(1, layout.maxX - layout.minX);
    const row = nearestRow(rows, targetX);
    if (!row || row === elements.lastRow) return;
    elements.lastRow = row;
    const x = layout.x(row.x);
    elements.guide?.setAttribute('x1', String(x));
    elements.guide?.setAttribute('x2', String(x));
    const visible = SYSTEM_SERIES.filter((spec) => state[spec.id] && row[spec.key] != null);
    if (!visible.length) {
        elements.hover?.setAttribute('hidden', '');
        if (elements.tooltip) elements.tooltip.hidden = true;
        return;
    }
    visible.forEach((spec) => positionHoverPoint(elements.points.get(spec.id), spec, row, x, layout, vramDomain));
    if (elements.tooltip) {
        elements.tooltip.innerHTML = renderHoverTooltip(row, visible);
        elements.tooltip.hidden = false;
    }
    elements.hover?.removeAttribute('hidden');
}

function positionHoverPoint(point, spec, row, x, layout, vramDomain) {
    if (!point) return;
    const y = spec.axis === 'vram'
        ? axisY(row[spec.key], vramDomain.min, vramDomain.max, layout.plotH)
        : axisY(row[spec.key], 0, 100, layout.plotH);
    point.setAttribute('cx', String(x));
    point.setAttribute('cy', String(y));
    point.removeAttribute('hidden');
}

function renderHoverTooltip(row, visible) {
    const values = visible.map((spec) => `<span data-tone="${spec.id}"><i></i>${spec.shortLabel}<strong>${formatSystemValue(row[spec.key], spec.unit)}</strong></span>`).join('');
    const total = row.vramTotal != null && visible.some((spec) => spec.id === 'vram')
        ? `<small>总显存 ${formatSystemValue(row.vramTotal, 'gb')}</small>`
        : '';
    return `<time>${formatSystemTime(row.ts)}</time><div>${values}</div>${total}`;
}

function normalizeSystemRecords(records) {
    return (Array.isArray(records) ? records : []).map((record, index) => ({
        index,
        x: numeric(record?.ts) ?? index,
        ts: numeric(record?.ts),
        vramUsed: numeric(record?.vram_used_gb),
        vramTotal: numeric(record?.vram_total_gb),
        gpuUtil: numeric(record?.gpu_util),
        gpuTemp: numeric(record?.gpu_temp),
    })).filter((row) => row.vramUsed != null || row.gpuUtil != null || row.gpuTemp != null);
}

function normalizeOptions(options = {}) {
    return {
        vram: options.vram !== false,
        gpu: options.gpu !== false,
        temp: options.temp !== false,
    };
}

function seriesPoints(rows, spec) {
    return rows.map((row) => ({ x: row.x, value: row[spec.key] })).filter((point) => point.value != null);
}

function vramChartDomain(rows) {
    const totals = rows.map((row) => row.vramTotal).filter((value) => value != null && value > 0);
    const used = rows.map((row) => row.vramUsed).filter((value) => value != null);
    const peak = used.length ? Math.max(...used) : 1;
    return { min: 0, max: Math.max(totals.length ? Math.max(...totals) : 0, Math.ceil(peak * 1.15), 1) };
}

function chartLayout(rows) {
    const minX = Math.min(...rows.map((row) => row.x));
    const maxX = Math.max(...rows.map((row) => row.x));
    const plotW = WIDTH - PADDING.left - PADDING.right;
    const plotH = HEIGHT - PADDING.top - PADDING.bottom;
    return {
        minX,
        maxX,
        plotW,
        plotH,
        x: (value) => PADDING.left + ((value - minX) / Math.max(1, maxX - minX)) * plotW,
    };
}

function axisY(value, min, max, plotH) {
    return PADDING.top + ((max - value) / Math.max(.000001, max - min)) * plotH;
}

function nearestRow(rows, targetX) {
    let nearest = null;
    let distance = Infinity;
    for (const row of rows) {
        const nextDistance = Math.abs(row.x - targetX);
        if (nextDistance < distance) {
            nearest = row;
            distance = nextDistance;
        }
    }
    return nearest;
}

function latestValue(rows, key) {
    for (let index = rows.length - 1; index >= 0; index -= 1) {
        if (rows[index][key] != null) return rows[index][key];
    }
    return null;
}

function clientPointToSvg(svg, clientX, clientY) {
    const matrix = svg.getScreenCTM?.();
    if (matrix && typeof DOMPoint === 'function') {
        try { return new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse()); } catch { /* use viewBox fallback */ }
    }
    const rect = svg.getBoundingClientRect();
    const scale = Math.min(rect.width / WIDTH, rect.height / HEIGHT) || 1;
    const renderedWidth = WIDTH * scale;
    return { x: (clientX - rect.left - (rect.width - renderedWidth) / 2) / scale };
}

function systemSampleLabel(returned, limits) {
    const total = Number(limits?.system_total);
    return Number.isFinite(total) && total > returned ? `最近 ${returned} / ${total} 个采样` : `${returned} 个采样`;
}

function formatSystemValue(value, unit) {
    if (!Number.isFinite(Number(value))) return '-';
    if (unit === 'gb') return `${formatCompact(value)} GB`;
    if (unit === 'temperature') return `${Math.round(value)}°C`;
    return `${Math.round(value)}%`;
}

function formatAxisGb(value) {
    return Number(value).toFixed(value >= 10 ? 0 : 1).replace(/\.0$/, '');
}

function formatCompact(value) {
    const number = Number(value);
    return Math.abs(number - Math.round(number)) < .01 ? String(Math.round(number)) : number.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
}

function formatSystemTime(value) {
    if (!Number.isFinite(Number(value))) return '未记录时间';
    return new Date(Number(value) * 1000).toLocaleTimeString('zh-CN', { hour12: false });
}

function numeric(value) {
    if (value == null || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
