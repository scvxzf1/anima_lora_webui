import {
    historyDetailEmptyText,
    numberOrNull,
    svgCircle,
    svgGroup,
    svgLine,
    svgPolyline,
    svgText,
} from '../ui.js?v=module-bootstrap-20260705-3';
import {
    formatHistoryCurveAxisValue,
    historyCurveMetricRange,
    historyCurveMetricVisible,
    historyCurveStatsWithHover,
    historyCurveVisibleSegments,
} from './data.js?v=module-bootstrap-20260705-3';

export function createHistoryCurveChart({
    historyCurveState,
    metrics,
    formatLr,
    hover,
    renderHistoryDetailContent,
}) {
    function renderHistoryCurveMainChart(payload, points, smoothPoints, stats, originalCount = points.length) {
        const shell = document.createElement('div');
        shell.className = 'history-curve-chart-shell';
        if (!points.length) {
            shell.appendChild(historyDetailEmptyText('当前范围没有可绘制的 Loss 或学习率点。请调整范围筛选。'));
            return shell;
        }
        if (originalCount > points.length) {
            const note = document.createElement('p');
            note.className = 'history-detail-limit-note';
            note.textContent = `绘图已降采样为 ${points.length} 个点；上方统计仍基于当前范围的 ${originalCount} 个数据点。`;
            shell.appendChild(note);
        }
        shell.appendChild(hover.renderHistoryCurveHoverReadout(stats));
        const svgScroll = document.createElement('div');
        svgScroll.className = 'history-curve-svg-scroll';
        svgScroll.appendChild(createHistoryCurveSvg(payload, points, smoothPoints, stats));
        shell.appendChild(svgScroll);
        return shell;
    }

    function createHistoryCurveSvg(payload, points, smoothPoints, stats) {
        const width = 1100;
        const height = 430;
        const pad = { top: 44, right: 94, bottom: 46, left: 66 };
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        svg.setAttribute('role', 'img');
        svg.classList.add('history-curve-svg', 'dual-metric');
        const xs = points.map((point) => point.step);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const plotW = width - pad.left - pad.right;
        const plotH = height - pad.top - pad.bottom;
        const xScale = (value) => pad.left + ((value - minX) / Math.max(1, maxX - minX)) * plotW;
        const lossRange = historyCurveMetricVisible(historyCurveState, metrics.loss)
            ? historyCurveMetricRange(points, smoothPoints, metrics.loss)
            : null;
        const lrRange = historyCurveMetricVisible(historyCurveState, metrics.lr)
            ? historyCurveMetricRange(points, smoothPoints, metrics.lr)
            : null;
        const lossScale = lossRange ? historyCurveYScale(lossRange, pad, plotH) : null;
        const lrScale = lrRange ? historyCurveYScale(lrRange, pad, plotH) : null;

        for (let i = 0; i <= 4; i += 1) {
            const y = pad.top + (plotH * i / 4);
            const line = svgLine(pad.left, y, width - pad.right, y, 'history-curve-grid-line');
            svg.appendChild(line);
            if (lossRange) {
                const val = lossRange.max - ((lossRange.max - lossRange.min) * i / 4);
                svg.appendChild(svgText(pad.left - 11, y + 3, formatHistoryCurveAxisValue(val, metrics.loss, formatLr), 'history-curve-axis-label loss-axis'));
            }
            if (lrRange) {
                const val = lrRange.max - ((lrRange.max - lrRange.min) * i / 4);
                svg.appendChild(svgText(width - pad.right + 11, y + 3, formatHistoryCurveAxisValue(val, metrics.lr, formatLr), 'history-curve-axis-label lr-axis'));
            }
        }
        if (lossRange) {
            svg.appendChild(svgText(pad.left, 18, 'Loss', 'history-curve-axis-title loss-axis'));
        }
        if (lrRange) {
            svg.appendChild(svgText(width - pad.right, 18, '学习率', 'history-curve-axis-title lr-axis'));
        }
        for (const x of [minX, maxX]) {
            svg.appendChild(svgText(xScale(x), height - 12, `step ${Math.round(x)}`, 'history-curve-axis-label history-curve-x-label'));
        }

        const segments = historyCurveVisibleSegments(payload, points);
        for (const segment of segments) {
            const x = xScale(segment.step);
            svg.appendChild(svgLine(x, pad.top, x, height - pad.bottom, 'history-curve-segment-line'));
            const label = svgText(Math.min(x + 6, width - pad.right - 88), pad.top + 14, segment.label, 'history-curve-segment-label');
            svg.appendChild(label);
        }

        const showRaw = historyCurveState.showRaw || !historyCurveState.showSmooth;
        if (showRaw && lossScale) {
            appendHistoryCurveLineSegments(svg, points, metrics.loss.key, xScale, lossScale, 'history-curve-line raw loss');
        }
        if (showRaw && lrScale) {
            appendHistoryCurveLineSegments(svg, points, metrics.lr.key, xScale, lrScale, 'history-curve-line raw lr');
        }
        if (historyCurveState.showSmooth && lossScale) {
            appendHistoryCurveLineSegments(svg, smoothPoints, metrics.loss.smoothKey, xScale, lossScale, 'history-curve-line smooth loss');
        }
        if (historyCurveState.showSmooth && lrScale) {
            appendHistoryCurveLineSegments(svg, smoothPoints, metrics.lr.smoothKey, xScale, lrScale, 'history-curve-line smooth lr');
        }

        drawHistoryCurveMetricPoints(svg, stats.loss, xScale, lossScale);
        drawHistoryCurveMetricPoints(svg, stats.lr, xScale, lrScale);

        const hoverLayer = svgGroup('history-curve-hover-layer');
        const hoverLine = svgLine(0, pad.top, 0, height - pad.bottom, 'history-curve-hover-line');
        const lossHoverPoint = svgCircle(0, 0, 5.5, 'history-curve-point hover loss');
        const lrHoverPoint = svgCircle(0, 0, 5.5, 'history-curve-point hover lr');
        const lrHoverLabel = hover.createHistoryCurveHoverLabel('history-curve-hover-label lr');
        const lossHoverLabel = hover.createHistoryCurveHoverLabel('history-curve-hover-label loss');
        const stepHoverLabel = hover.createHistoryCurveHoverLabel('history-curve-hover-label step');
        hoverLayer.append(
            hoverLine,
            lossHoverPoint,
            lrHoverPoint,
            lrHoverLabel.group,
            lossHoverLabel.group,
            stepHoverLabel.group,
        );
        svg.appendChild(hoverLayer);
        const hoverLayout = {
            hoverLayer,
            hoverLine,
            lossHoverPoint,
            lrHoverPoint,
            lossHoverLabel,
            lrHoverLabel,
            stepHoverLabel,
            width,
            height,
            pad,
            xScale,
            lossScale,
            lrScale,
        };
        hover.updateHistoryCurveHoverLayer(hoverLayout, stats);

        const overlay = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        overlay.setAttribute('x', `${pad.left}`);
        overlay.setAttribute('y', `${pad.top}`);
        overlay.setAttribute('width', `${plotW}`);
        overlay.setAttribute('height', `${plotH}`);
        overlay.setAttribute('class', 'history-curve-overlay');
        let pendingHoverStep = null;
        let hoverFrame = null;
        const applyHoverStep = (step) => {
            const nextStats = historyCurveStatsWithHover(stats, step, metrics);
            hover.updateHistoryCurveHoverLayer(hoverLayout, nextStats);
            const inspector = svg.closest('.history-curve-workbench')?.querySelector('.history-curve-inspector');
            if (inspector) hover.renderHistoryCurveInspectorRows(inspector, nextStats);
            const readout = svg.closest('.history-curve-chart-shell')?.querySelector('.history-curve-hover-readout');
            if (readout) hover.renderHistoryCurveHoverReadoutRows(readout, nextStats);
        };
        const scheduleHoverStep = (step) => {
            pendingHoverStep = step;
            if (hoverFrame) return;
            hoverFrame = requestAnimationFrame(() => {
                hoverFrame = null;
                historyCurveState.hoverStep = pendingHoverStep;
                applyHoverStep(pendingHoverStep);
            });
        };
        overlay.addEventListener('mousemove', (event) => {
            const svgPoint = svgClientPoint(svg, event);
            const viewX = svgPoint?.x ?? (() => {
                const rect = svg.getBoundingClientRect();
                const ratioX = (event.clientX - rect.left) / Math.max(1, rect.width);
                return ratioX * width;
            })();
            const step = minX + ((viewX - pad.left) / Math.max(1, plotW)) * (maxX - minX);
            scheduleHoverStep(Math.max(minX, Math.min(maxX, step)));
        });
        overlay.addEventListener('mouseleave', () => {
            if (hoverFrame) {
                cancelAnimationFrame(hoverFrame);
                hoverFrame = null;
            }
            historyCurveState.hoverStep = null;
            pendingHoverStep = null;
            applyHoverStep(null);
        });
        svg.appendChild(overlay);
        return svg;
    }

    function renderHistoryCurveSegments(payload, points) {
        const segments = historyCurveVisibleSegments(payload, points);
        if (payload.mode !== 'config_group' || !segments.length) {
            const note = document.createElement('p');
            note.className = 'history-curve-note';
            note.textContent = payload.mode === 'config_group'
                ? '当前范围内没有任务分界。'
                : '单任务曲线没有任务分界。';
            return note;
        }
        const box = document.createElement('div');
        box.className = 'history-curve-segments';
        for (const segment of segments) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'btn btn-small';
            btn.textContent = `${segment.label} · step ${segment.step}`;
            btn.addEventListener('click', () => {
                historyCurveState.rangeMode = 'custom';
                historyCurveState.customStart = String(segment.step);
                historyCurveState.customEnd = '';
                historyCurveState.hoverStep = segment.step;
                renderHistoryDetailContent();
            });
            box.appendChild(btn);
        }
        return box;
    }

    function appendHistoryCurveLineSegments(svg, points, valueKey, xScale, yScale, className) {
        let segment = [];
        const flush = () => {
            if (segment.length > 1) svg.appendChild(svgPolyline(segment, className));
            segment = [];
        };
        for (const point of points || []) {
            const value = numberOrNull(point[valueKey]);
            if (value === null) {
                flush();
                continue;
            }
            if (point.stageBreakBefore) flush();
            segment.push([xScale(point.step), yScale(value)]);
        }
        flush();
    }

    function drawHistoryCurveMetricPoints(svg, metricStats, xScale, yScale) {
        if (!metricStats?.metric || !yScale) return;
        const { metric } = metricStats;
        if (metricStats.minPoint) {
            svg.appendChild(svgCircle(xScale(metricStats.minPoint.step), yScale(metricStats.minPoint[metric.key]), 4.5, `history-curve-point min ${metric.className}`));
        }
        if (metricStats.latest) {
            svg.appendChild(svgCircle(xScale(metricStats.latest.step), yScale(metricStats.latest[metric.key]), 4.5, `history-curve-point latest ${metric.className}`));
        }
    }

    return { renderHistoryCurveMainChart, renderHistoryCurveSegments };
}

function historyCurveYScale(range, pad, plotH) {
    return (value) => pad.top + ((range.max - Number(value)) / Math.max(0.000001, range.max - range.min)) * plotH;
}

function svgClientPoint(svg, event) {
    if (!svg?.createSVGPoint || !svg.getScreenCTM) return null;
    const matrix = svg.getScreenCTM();
    if (!matrix) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(matrix.inverse());
}
