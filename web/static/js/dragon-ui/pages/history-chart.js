/* Interactive loss / learning-rate chart used by training history details. */

import { areaSvgPath, emaValues, smoothSvgPath } from './trend-utils.js?v=dragon-ui-20260825v1';
import { bindLatestPointerMove, pointerEventName } from '../pointer-frame.js?v=dragon-ui-20260826v1';
import { createFrameScheduler } from '../frame-scheduler.js?v=dragon-ui-20260826v1';

const WIDTH = 900;
const HEIGHT = 300;
const PADDING = { top: 34, right: 72, bottom: 32, left: 54 };
const LOSS_COLOR = 'var(--dragon-accent)';
const LR_COLOR = '#f59e0b';

export function renderHistoryMetricsChart(metrics = [], options = {}) {
    const rows = normalizeMetrics(metrics);
    if (!rows.length) return '<div class="dragon-empty-state"><p>暂无训练数据</p></div>';
    const displayRows = smoothMetricRows(rows, options.smoothing ?? .2);
    const showLoss = options.lossCurve !== false;
    const showLr = options.lrCurve !== false;
    const showLossValue = options.lossValue !== false;
    const showLrValue = options.lrValue !== false;
    const innerW = WIDTH - PADDING.left - PADDING.right;
    const innerH = HEIGHT - PADDING.top - PADDING.bottom;
    const lossDomain = domain(rows.map((row) => row.loss));
    const lrDomain = domain(rows.map((row) => row.lr));
    const x = (index) => PADDING.left + (index / Math.max(rows.length - 1, 1)) * innerW;
    const lossY = (value) => PADDING.top + (1 - (value - lossDomain[0]) / (lossDomain[1] - lossDomain[0])) * innerH;
    const lrY = (value) => PADDING.top + (1 - (value - lrDomain[0]) / (lrDomain[1] - lrDomain[0])) * innerH;
    const grid = [0, .25, .5, .75, 1].map((ratio) => {
        const y = PADDING.top + ratio * innerH;
        const value = (lossDomain[1] - ratio * (lossDomain[1] - lossDomain[0])).toFixed(3);
        return `<line x1="${PADDING.left}" y1="${y}" x2="${PADDING.left + innerW}" y2="${y}" stroke="var(--dragon-divider)" stroke-width="0.7"/><text x="${PADDING.left - 9}" y="${y + 4}" text-anchor="end" fill="var(--dragon-text-quaternary)" font-size="11">${value}</text>`;
    }).join('');
    const rightTicks = showLr && rows.some((row) => row.lr != null) ? [0, .5, 1].map((ratio) => {
        const y = PADDING.top + ratio * innerH;
        const value = formatLr(lrDomain[1] - ratio * (lrDomain[1] - lrDomain[0]));
        return `<text x="${WIDTH - PADDING.right + 9}" y="${y + 4}" fill="${LR_COLOR}" font-size="10">${value}</text>`;
    }).join('') : '';
    const lossPoints = showLoss ? pointPairs(displayRows, x, lossY, 'loss') : [];
    const lrPoints = showLr ? pointPairs(displayRows, x, lrY, 'lr') : [];
    const lossPath = smoothSvgPath(lossPoints, .2);
    const lrPath = smoothSvgPath(lrPoints, .2);
    const lossArea = areaSvgPath(lossPath, lossPoints, PADDING.top + innerH);
    return `<div class="dragon-chart-glass-tooltip" data-history-chart-tooltip hidden></div><svg class="dragon-loss-chart dragon-history-metrics-chart" viewBox="0 0 ${WIDTH} ${HEIGHT}" preserveAspectRatio="xMidYMid meet" role="img" aria-labelledby="dragon-history-chart-title" data-history-chart>
        <title id="dragon-history-chart-title">训练损失与学习率变化曲线</title>
        <defs><linearGradient id="dragon-history-loss-area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${LOSS_COLOR}" stop-opacity=".15"/><stop offset="1" stop-color="${LOSS_COLOR}" stop-opacity="0"/></linearGradient></defs>
        ${grid}${rightTicks}
        ${lossArea ? `<path d="${lossArea}" fill="url(#dragon-history-loss-area)" data-history-series="loss-area"/>` : ''}
        ${lossPath ? `<path d="${lossPath}" fill="none" stroke="${LOSS_COLOR}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" data-history-series="loss"/>` : ''}
        ${lrPath ? `<path d="${lrPath}" fill="none" stroke="${LR_COLOR}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" data-history-series="lr"/>` : ''}
        <g class="dragon-history-chart-values" data-history-chart-values hidden>
            <text x="${WIDTH - PADDING.right}" y="17" text-anchor="end" fill="${LOSS_COLOR}" font-size="12" data-history-hover-loss ${showLossValue ? '' : 'hidden'}>Loss: —</text>
            <text x="${WIDTH - PADDING.right}" y="31" text-anchor="end" fill="${LR_COLOR}" font-size="12" data-history-hover-lr ${showLrValue ? '' : 'hidden'}>LR: —</text>
        </g>
        <g data-history-hover hidden>
            <line data-history-hover-guide y1="${PADDING.top}" y2="${PADDING.top + innerH}" stroke="#aeb4bf" stroke-width="0.8" stroke-dasharray="3 3"/>
            <circle data-history-hover-loss-point r="4.5" fill="${LOSS_COLOR}" stroke="white" stroke-width="1.5" hidden/>
            <circle data-history-hover-lr-point r="4.5" fill="${LR_COLOR}" stroke="white" stroke-width="1.5" hidden/>
        </g>
        <rect class="dragon-history-chart-hitarea" x="${PADDING.left}" y="${PADDING.top}" width="${innerW}" height="${innerH}" fill="transparent"/>
    </svg>`;
}

export function bindHistoryChart(root, metrics = []) {
    const container = root.querySelector('[data-history-chart-container]');
    if (!container) return () => {};
    const state = { lossCurve: true, lrCurve: true, lossValue: true, lrValue: true, smoothing: .2 };
    let unbindHover = () => {};
    const render = () => {
        unbindHover();
        container.innerHTML = renderHistoryMetricsChart(metrics, state);
        unbindHover = bindHover(container.querySelector('[data-history-chart]'), metrics, state);
    };
    const renderFrame = createFrameScheduler(render, root.ownerDocument?.defaultView);
    const listeners = [];
    root.querySelectorAll('[data-history-chart-toggle]').forEach((input) => {
        const onChange = () => { state[input.dataset.historyChartToggle] = input.checked; render(); };
        input.checked = true;
        input.addEventListener('change', onChange);
        listeners.push(() => input.removeEventListener('change', onChange));
    });
    const smoothingInput = root.querySelector('[data-history-chart-smoothing]');
    const smoothingOutput = root.querySelector('[data-history-chart-smoothing-value]');
    const onSmoothing = () => {
        state.smoothing = Math.min(.99, Math.max(0, Number(smoothingInput?.value || 0) / 100));
        if (smoothingOutput) smoothingOutput.textContent = `${Math.round(state.smoothing * 100)}%`;
        renderFrame.schedule();
    };
    smoothingInput?.addEventListener('input', onSmoothing);
    listeners.push(() => smoothingInput?.removeEventListener('input', onSmoothing));
    render();
    return () => { renderFrame.cancel(); unbindHover(); listeners.forEach((remove) => remove()); };
}

function bindHover(svg, metrics, options) {
    if (!svg) return () => {};
    const hitarea = svg.querySelector('.dragon-history-chart-hitarea');
    const hover = svg.querySelector('[data-history-hover]');
    const values = svg.querySelector('[data-history-chart-values]');
    const lossPoint = svg.querySelector('[data-history-hover-loss-point]');
    const lrPoint = svg.querySelector('[data-history-hover-lr-point]');
    const guide = svg.querySelector('[data-history-hover-guide]');
    const lossText = svg.querySelector('[data-history-hover-loss]');
    const lrText = svg.querySelector('[data-history-hover-lr]');
    const tooltip = svg.parentElement?.querySelector('[data-history-chart-tooltip]');
    const rows = normalizeMetrics(metrics);
    const displayRows = smoothMetricRows(rows, options.smoothing ?? .2);
    const lossDomain = domain(rows.map((row) => row.loss));
    const lrDomain = domain(rows.map((row) => row.lr));
    const innerW = WIDTH - PADDING.left - PADDING.right;
    const innerH = HEIGHT - PADDING.top - PADDING.bottom;
    const x = (index) => PADDING.left + (index / Math.max(rows.length - 1, 1)) * innerW;
    const lossY = (value) => PADDING.top + (1 - (value - lossDomain[0]) / (lossDomain[1] - lossDomain[0])) * innerH;
    const lrY = (value) => PADDING.top + (1 - (value - lrDomain[0]) / (lrDomain[1] - lrDomain[0])) * innerH;
    let lastIndex = -1;
    const onMove = (event) => {
        const svgX = clientPointToSvg(svg, event.clientX, event.clientY).x;
        const index = Math.max(0, Math.min(rows.length - 1, Math.round((svgX - PADDING.left) / innerW * Math.max(rows.length - 1, 1))));
        if (index === lastIndex) return;
        lastIndex = index;
        const row = rows[index];
        if (!row) return;
        const cx = x(index);
        const showLossPoint = options.lossCurve !== false && row.loss != null;
        const showLrPoint = options.lrCurve !== false && row.lr != null;
        const showLossText = options.lossValue !== false && row.loss != null;
        const showLrText = options.lrValue !== false && row.lr != null;
        const displayRow = displayRows[index] || row;
        positionHoverPoint(lossPoint, cx, displayRow.loss == null ? null : lossY(displayRow.loss), showLossPoint);
        positionHoverPoint(lrPoint, cx, displayRow.lr == null ? null : lrY(displayRow.lr), showLrPoint);
        guide.setAttribute('x1', String(cx)); guide.setAttribute('x2', String(cx));
        if (lossText) lossText.textContent = `Loss: ${row.loss == null ? '—' : row.loss.toFixed(5)}`;
        if (lrText) lrText.textContent = `LR: ${row.lr == null ? '—' : formatLr(row.lr)}`;
        if (tooltip) {
            tooltip.innerHTML = `<span>STEP ${index + 1}</span><strong>${row.loss == null ? 'Loss —' : `Loss ${row.loss.toFixed(5)}`}</strong><small>${row.lr == null ? 'LR —' : `LR ${formatLr(row.lr)}`}</small>`;
            tooltip.hidden = false;
        }
        setSvgHidden(values, !(showLossText || showLrText));
        setSvgHidden(hover, !(showLossPoint || showLrPoint || showLossText || showLrText));
    };
    const hide = () => {
        lastIndex = -1;
        setSvgHidden(hover, true);
        setSvgHidden(values, true);
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

function positionHoverPoint(point, x, y, visible) {
    if (!point) return;
    setSvgHidden(point, !visible);
    if (!visible) return;
    point.setAttribute('cx', String(x));
    point.setAttribute('cy', String(y));
}

function setSvgHidden(element, hidden) {
    element?.toggleAttribute('hidden', hidden);
}

function clientPointToSvg(svg, clientX, clientY) {
    const screenMatrix = svg.getScreenCTM?.();
    if (screenMatrix && typeof DOMPoint === 'function') {
        try {
            const point = new DOMPoint(clientX, clientY).matrixTransform(screenMatrix.inverse());
            if (Number.isFinite(point.x) && Number.isFinite(point.y)) return point;
        } catch {
            // Fall back to the viewBox calculation for incomplete SVG DOM implementations.
        }
    }
    const rect = svg.getBoundingClientRect();
    const scale = Math.min(rect.width / WIDTH, rect.height / HEIGHT) || 1;
    const renderedWidth = WIDTH * scale;
    const renderedHeight = HEIGHT * scale;
    return {
        x: (clientX - rect.left - (rect.width - renderedWidth) / 2) / scale,
        y: (clientY - rect.top - (rect.height - renderedHeight) / 2) / scale,
    };
}

function normalizeMetrics(metrics) {
    return (Array.isArray(metrics) ? metrics : []).map((item) => ({
        loss: numeric(item?.loss),
        lr: numeric(item?.lr ?? item?.learning_rate),
    })).filter((row) => row.loss != null || row.lr != null);
}

function pointPairs(rows, x, y, key) {
    return rows.map((row, index) => row[key] == null ? null : [x(index), y(row[key])]).filter(Boolean);
}

function smoothMetricRows(rows, smoothing) {
    const lossIndices = rows.flatMap((row, index) => row.loss == null ? [] : [index]);
    const smoothed = emaValues(lossIndices.map((index) => rows[index].loss), smoothing);
    const lossByIndex = new Map(lossIndices.map((index, position) => [index, smoothed[position]]));
    return rows.map((row, index) => ({ ...row, loss: lossByIndex.has(index) ? lossByIndex.get(index) : null }));
}

function numeric(value) {
    if (value == null || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}
function domain(values) {
    const valid = values.filter((value) => value != null);
    if (!valid.length) return [0, 1];
    let min = Math.min(...valid); let max = Math.max(...valid);
    if (min === max) { const pad = Math.abs(min) * .05 || 1; min -= pad; max += pad; }
    return [min, max];
}
function formatLr(value) { return Number(value).toExponential(2); }
