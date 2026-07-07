import { formatHistorySystemTime } from '../system.js?v=module-bootstrap-20260707-93';
import {
    clampNumber,
    historyDetailEmptyText,
    historyDetailRow,
    numberOrNull,
    svgGroup,
    svgRect,
    svgText,
} from '../ui.js?v=module-bootstrap-20260707-93';
import { formatLossValue } from './data.js?v=module-bootstrap-20260707-93';

export function createHistoryCurveHover({ historyCurveState, metrics, formatLr }) {
    function renderHistoryCurveHoverReadout(stats) {
        const readout = document.createElement('div');
        readout.className = 'history-curve-hover-readout';
        renderHistoryCurveHoverReadoutRows(readout, stats);
        return readout;
    }

    function renderHistoryCurveHoverReadoutRows(readout, stats) {
        const point = stats.hoverPoint || stats.latest;
        readout.replaceChildren();
        if (!point) {
            readout.appendChild(historyDetailEmptyText('悬停曲线查看 Step、Loss 和学习率。'));
            return;
        }
        readout.append(
            historyCurveHoverReadoutItem('当前查看', `Step ${Math.round(point.step)}`, 'step'),
            historyCurveMetricReadoutItem(stats.loss.hoverPoint || stats.loss.latest, metrics.loss),
            historyCurveMetricReadoutItem(stats.lr.hoverPoint || stats.lr.latest, metrics.lr),
        );
    }

    function historyCurveMetricReadoutItem(point, metric) {
        const { value, label } = historyCurveHoverValue(point, metric);
        return historyCurveHoverReadoutItem(label, value === null ? '-' : metric.format(value), metric.className);
    }

    function historyCurveHoverReadoutItem(label, value, className = '') {
        const item = document.createElement('div');
        item.className = `history-curve-hover-readout-item ${className}`.trim();
        const labelNode = document.createElement('span');
        labelNode.textContent = label;
        const valueNode = document.createElement('strong');
        valueNode.textContent = String(value === undefined || value === null || value === '' ? '-' : value);
        item.append(labelNode, valueNode);
        return item;
    }

    function createHistoryCurveHoverLabel(className) {
        const group = svgGroup(className);
        const rect = svgRect(0, 0, 1, 1, 'history-curve-hover-label-bg');
        const text = svgText(0, 0, '', 'history-curve-hover-label-text');
        group.append(rect, text);
        return { group, rect, text };
    }

    function updateHistoryCurveHoverLayer(layout, stats) {
        const point = stats.hoverPoint || stats.latest;
        if (!point) {
            setSvgVisibility(layout.hoverLayer, false);
            return;
        }
        const x = layout.xScale(point.step);
        layout.hoverLine.setAttribute('x1', `${x}`);
        layout.hoverLine.setAttribute('x2', `${x}`);
        layout.hoverLine.setAttribute('y1', `${layout.pad.top}`);
        layout.hoverLine.setAttribute('y2', `${layout.height - layout.pad.bottom}`);
        updateHistoryCurveHoverPoint(
            layout.lossHoverPoint,
            stats.loss.hoverPoint,
            layout.xScale,
            layout.lossScale,
            metrics.loss,
        );
        updateHistoryCurveHoverPoint(
            layout.lrHoverPoint,
            stats.lr.hoverPoint,
            layout.xScale,
            layout.lrScale,
            metrics.lr,
        );
        setSvgVisibility(layout.lossHoverLabel.group, false);
        setSvgVisibility(layout.lrHoverLabel.group, false);
        positionHistoryCurveHoverLabel(
            layout.stepHoverLabel,
            x,
            layout.height - layout.pad.bottom,
            `step ${Math.round(point.step)}`,
            layout,
            'bottom',
        );
        setSvgVisibility(layout.hoverLayer, true);
    }

    function updateHistoryCurveHoverPoint(circle, point, xScale, yScale, metric) {
        const { value } = historyCurveHoverValue(point, metric);
        if (!point || !yScale || value === null) {
            setSvgVisibility(circle, false);
            return;
        }
        circle.setAttribute('cx', `${xScale(point.step)}`);
        circle.setAttribute('cy', `${yScale(value)}`);
        setSvgVisibility(circle, true);
    }

    function historyCurveHoverValue(point, metric) {
        const raw = numberOrNull(point?.[metric.key]);
        const smooth = numberOrNull(point?.[metric.smoothKey]);
        const preferSmooth = historyCurveState.showSmooth && !historyCurveState.showRaw;
        if (preferSmooth && smooth !== null) {
            return { value: smooth, label: `平滑${metric.label}` };
        }
        if (raw !== null) {
            return { value: raw, label: metric.label };
        }
        if (smooth !== null) {
            return { value: smooth, label: `平滑${metric.label}` };
        }
        return { value: null, label: metric.label };
    }

    function positionHistoryCurveHoverLabel(label, x, y, text, layout, placement = 'above') {
        const value = String(text);
        const paddingX = 7;
        const labelH = 21;
        const labelW = Math.max(54, estimateSvgTextWidth(value) + paddingX * 2);
        const minX = layout.pad.left + 4;
        const maxX = layout.width - layout.pad.right - labelW - 4;
        const left = clampNumber(x - labelW / 2, minX, Math.max(minX, maxX));
        const top = placement === 'bottom'
            ? clampNumber(layout.height - layout.pad.bottom + 8, layout.height - layout.pad.bottom + 4, layout.height - labelH - 4)
            : clampNumber(y - labelH - 10, layout.pad.top + 4, layout.height - layout.pad.bottom - labelH - 4);

        label.rect.setAttribute('x', `${left}`);
        label.rect.setAttribute('y', `${top}`);
        label.rect.setAttribute('width', `${labelW}`);
        label.rect.setAttribute('height', `${labelH}`);
        label.text.setAttribute('x', `${left + labelW / 2}`);
        label.text.setAttribute('y', `${top + 14}`);
        label.text.textContent = value;
        setSvgVisibility(label.group, true);
    }

    function renderHistoryCurveInspector(stats) {
        const box = document.createElement('div');
        box.className = 'history-curve-inspector';
        renderHistoryCurveInspectorRows(box, stats);
        return box;
    }

    function renderHistoryCurveInspectorRows(box, stats) {
        const point = stats.hoverPoint || stats.latest;
        box.replaceChildren();
        if (!point) {
            box.appendChild(historyDetailEmptyText('悬停曲线查看 Step 明细。'));
            return;
        }
        const lossPoint = stats.loss.hoverPoint || stats.loss.latest || {};
        const lrPoint = stats.lr.hoverPoint || stats.lr.latest || {};
        const anchor = document.createElement('div');
        anchor.className = 'history-curve-inspector-anchor';
        const title = document.createElement('strong');
        title.textContent = `当前查看：Step ${Math.round(point.step)}`;
        const meta = document.createElement('span');
        meta.textContent = [
            point.sourceTaskLabel || point.sourceTaskId || '',
            formatHistorySystemTime(point.ts),
        ].filter((value) => value && value !== '-').join(' · ') || '悬停曲线可切换查看点';
        anchor.append(title, meta);
        const grid = document.createElement('div');
        grid.className = 'history-curve-inspector-grid';
        [
            ['Step', point.step],
            ['原始 Step', point.rawStep ?? '-'],
            ['Loss', formatLossValue(lossPoint.loss)],
            ['平滑 Loss', formatLossValue(lossPoint.smoothLoss)],
            ['学习率', formatLr(lrPoint.lr)],
            ['平滑学习率', formatLr(lrPoint.smoothLr)],
            ['时间', formatHistorySystemTime(point.ts)],
            ['来源任务', point.sourceTaskLabel || point.sourceTaskId || '-'],
        ].forEach(([label, value]) => grid.appendChild(historyDetailRow(label, value)));
        box.append(anchor, grid);
    }

    return {
        createHistoryCurveHoverLabel,
        renderHistoryCurveHoverReadout,
        renderHistoryCurveHoverReadoutRows,
        renderHistoryCurveInspector,
        renderHistoryCurveInspectorRows,
        updateHistoryCurveHoverLayer,
    };
}

function estimateSvgTextWidth(text) {
    return String(text).length * 7.4;
}

function setSvgVisibility(node, visible) {
    node.setAttribute('visibility', visible ? 'visible' : 'hidden');
}
