import { numberOrNull } from '../ui.js?v=module-bootstrap-20260608-3';

export const HISTORY_CURVE_RENDER_POINT_LIMIT = 1600;

export function createHistoryCurveMetrics(formatLr) {
    const HISTORY_CURVE_METRICS = {
        loss: {
            key: 'loss',
            smoothKey: 'smoothLoss',
            className: 'loss',
            label: 'Loss',
            pointLabel: 'Loss 点',
            format: formatLossValue,
            formatSigned: formatSignedLoss,
        },
        lr: {
            key: 'lr',
            smoothKey: 'smoothLr',
            className: 'lr',
            label: '学习率',
            pointLabel: '学习率点',
            format: formatLr,
            formatSigned: formatSignedLr,
        },
    };
    return HISTORY_CURVE_METRICS;
}

export function historyCurvePoints(payload, deps) {
    const { historyLossChartPoints, metricsWithProgressFallback } = deps;
    if (payload.mode === 'config_group') {
        return (payload.metrics || [])
            .map((item, index) => historyCurveNormalizePoint(item, index, true))
            .filter(historyCurvePointHasAnyMetric)
            .filter(Boolean);
    }
    const task = payload.task || {};
    const metricPoints = metricsWithProgressFallback(payload.metrics || [], payload.logs || [])
        .filter(historyCurveRawPointHasAnyMetric)
        .map(historyCurveNormalizeRawMetricPoint);
    return historyLossChartPoints(metricPoints, task)
        .map((item, index) => historyCurveNormalizePoint(item, index, false))
        .filter(historyCurvePointHasAnyMetric)
        .filter(Boolean);
}

export function historyCurveNormalizePoint(item, index, merged) {
    const step = Number(merged ? (item.display_step ?? item.step) : item.step);
    if (!Number.isFinite(step)) return null;
    const loss = numberOrNull(item.loss);
    const lr = numberOrNull(item.lr ?? item.learningRate ?? item.learning_rate);
    if (loss === null && lr === null) return null;
    return {
        ...item,
        index,
        step,
        loss,
        rawStep: Number.isFinite(Number(item.rawStep ?? item.step)) ? Number(item.rawStep ?? item.step) : step,
        lr,
        ts: numberOrNull(item.ts),
        sourceTaskId: strOrEmpty(item.source_task_id || item.sourceTaskId),
        sourceTaskIndex: Number(item.source_task_index || item.sourceTaskIndex || 0),
        sourceTaskLabel: strOrEmpty(item.source_task_label || item.sourceTaskLabel),
        stageBreakBefore: Boolean(item.stage_break_before || item.stageBreakBefore),
        displayStepOffset: Number(item.display_step_offset || item.displayStepOffset || 0),
    };
}

export function historyCurvePointHasAnyMetric(point) {
    return Boolean(point)
        && (numberOrNull(point.loss) !== null || numberOrNull(point.lr) !== null);
}

export function historyCurveRawPointHasAnyMetric(point) {
    return Boolean(point)
        && (
            numberOrNull(point.loss) !== null
            || numberOrNull(point.lr ?? point.learningRate ?? point.learning_rate) !== null
        );
}

export function historyCurveNormalizeRawMetricPoint(point) {
    const lr = numberOrNull(point.lr ?? point.learningRate ?? point.learning_rate);
    return {
        ...point,
        lr,
    };
}

export function historyCurveMetricStateKey(metric) {
    return (typeof metric === 'string' ? metric : metric?.key) === 'lr' ? 'showLr' : 'showLoss';
}

export function historyCurveMetricVisible(historyCurveState, metric) {
    return historyCurveState[historyCurveMetricStateKey(metric)] !== false;
}

export function historyCurveFilteredPoints(points, historyCurveState) {
    const sorted = [...points].sort((a, b) => a.step - b.step || a.index - b.index);
    if (!sorted.length) return [];
    if (historyCurveState.rangeMode === 'last100') {
        return sorted.slice(-100);
    }
    if (historyCurveState.rangeMode === 'last25') {
        return sorted.slice(-Math.max(1, Math.ceil(sorted.length * 0.25)));
    }
    if (historyCurveState.rangeMode === 'custom') {
        const start = Number(historyCurveState.customStart);
        const end = Number(historyCurveState.customEnd);
        return sorted.filter((point) => {
            if (Number.isFinite(start) && point.step < start) return false;
            if (Number.isFinite(end) && point.step > end) return false;
            return true;
        });
    }
    return sorted;
}

export function historyCurveSmoothPoints(points, windowSize, metrics) {
    const window = Math.max(1, Number(windowSize) || 1);
    return points.map((point, index) => {
        const start = Math.max(0, index - window + 1);
        const slice = points.slice(start, index + 1);
        const out = { ...point };
        for (const metric of Object.values(metrics)) {
            if (numberOrNull(point[metric.key]) === null) continue;
            const values = slice
                .map((item) => numberOrNull(item[metric.key]))
                .filter((value) => value !== null);
            if (values.length) {
                out[metric.smoothKey] = values.reduce((sum, value) => sum + value, 0) / values.length;
            }
        }
        return out;
    });
}

export function historyCurveDisplayPoints(points) {
    const limit = HISTORY_CURVE_RENDER_POINT_LIMIT;
    if (!Array.isArray(points) || points.length <= limit) return points || [];
    const out = [];
    const used = new Set();
    const stride = (points.length - 1) / Math.max(1, limit - 1);
    for (let i = 0; i < limit; i += 1) {
        const index = Math.min(points.length - 1, Math.round(i * stride));
        if (used.has(index)) continue;
        used.add(index);
        out.push(points[index]);
    }
    return out;
}

export function historyCurveStats(points, smoothPoints, historyCurveState, metrics) {
    if (!points.length) {
        return {
            count: 0,
            smoothPoints,
            stepRange: '-',
            latest: null,
            hoverPoint: null,
            loss: historyCurveMetricStats([], [], metrics.loss, historyCurveState),
            lr: historyCurveMetricStats([], [], metrics.lr, historyCurveState),
        };
    }
    const latest = points[points.length - 1];
    const first = points[0];
    const hoverPoint = historyCurveNearestPoint(smoothPoints, historyCurveState.hoverStep) || smoothPoints[smoothPoints.length - 1] || latest;
    return {
        count: points.length,
        smoothPoints,
        first,
        latest,
        stepRange: `${first.step} → ${latest.step}`,
        hoverPoint,
        loss: historyCurveMetricStats(points, smoothPoints, metrics.loss, historyCurveState),
        lr: historyCurveMetricStats(points, smoothPoints, metrics.lr, historyCurveState),
    };
}

export function historyCurveMetricStats(points, smoothPoints, metric, historyCurveState) {
    const metricPoints = (points || []).filter((point) => numberOrNull(point[metric.key]) !== null);
    const metricSmoothPoints = (smoothPoints || []).filter((point) => numberOrNull(point[metric.key]) !== null);
    if (!metricPoints.length) {
        return {
            metric,
            count: 0,
            first: null,
            latest: null,
            minPoint: null,
            maxPoint: null,
            activeLatest: null,
            recentAverage: null,
            trend: null,
            delta: null,
            hoverPoint: null,
        };
    }
    const first = metricPoints[0];
    const latest = metricPoints[metricPoints.length - 1];
    const minPoint = metricPoints.reduce((best, point) => numberOrNull(point[metric.key]) < numberOrNull(best[metric.key]) ? point : best, metricPoints[0]);
    const maxPoint = metricPoints.reduce((best, point) => numberOrNull(point[metric.key]) > numberOrNull(best[metric.key]) ? point : best, metricPoints[0]);
    const activeMetricPoints = historyCurveMetricActivePoints(metricPoints, metric);
    const activeLatest = activeMetricPoints[activeMetricPoints.length - 1] || latest;
    const recent = metricPoints.slice(-50);
    const recentAverage = recent.reduce((sum, point) => sum + numberOrNull(point[metric.key]), 0) / Math.max(1, recent.length);
    const trendStart = metricPoints[Math.max(0, metricPoints.length - 50)];
    const trend = numberOrNull(latest[metric.key]) - (numberOrNull(trendStart?.[metric.key]) ?? numberOrNull(first[metric.key]));
    const delta = numberOrNull(latest[metric.key]) - numberOrNull(first[metric.key]);
    const hoverPoint = historyCurveNearestMetricPoint(metricSmoothPoints, historyCurveState.hoverStep, metric.key)
        || historyCurveNearestMetricPoint(metricSmoothPoints, latest.step, metric.key)
        || latest;
    return {
        metric,
        count: metricPoints.length,
        first,
        latest,
        minPoint,
        maxPoint,
        activeLatest,
        recentAverage,
        trend,
        delta,
        hoverPoint,
    };
}

function historyCurveMetricActivePoints(metricPoints, metric) {
    if (metric.key !== 'lr') return metricPoints;
    const positive = metricPoints.filter((point) => {
        const value = numberOrNull(point[metric.key]);
        return value !== null && value > 0;
    });
    return positive.length ? positive : metricPoints;
}

export function historyCurveStatsWithHover(stats, step, metrics) {
    const target = numberOrNull(step);
    const smoothPoints = stats.smoothPoints || [];
    const fallbackPoint = smoothPoints[smoothPoints.length - 1] || stats.latest;
    const hoverPoint = target !== null
        ? (historyCurveNearestPoint(smoothPoints, target) || fallbackPoint)
        : fallbackPoint;
    const hoverStep = hoverPoint?.step ?? target;
    return {
        ...stats,
        hoverPoint,
        loss: {
            ...stats.loss,
            hoverPoint: hoverStep !== null
                ? (historyCurveNearestMetricPoint(smoothPoints, hoverStep, metrics.loss.key) || stats.loss.latest)
                : stats.loss.latest,
        },
        lr: {
            ...stats.lr,
            hoverPoint: hoverStep !== null
                ? (historyCurveNearestMetricPoint(smoothPoints, hoverStep, metrics.lr.key) || stats.lr.latest)
                : stats.lr.latest,
        },
    };
}

export function historyCurveNearestPoint(points, step) {
    const target = Number(step);
    if (!Number.isFinite(target) || !points.length) return null;
    let left = 0;
    let right = points.length - 1;
    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (Number(points[mid].step) < target) left = mid + 1;
        else right = mid;
    }
    const after = points[left];
    const before = points[Math.max(0, left - 1)];
    if (!before) return after;
    if (!after) return before;
    return Math.abs(Number(before.step) - target) <= Math.abs(Number(after.step) - target) ? before : after;
}

export function historyCurveNearestMetricPoint(points, step, key) {
    const metricPoints = (points || []).filter((point) => numberOrNull(point[key]) !== null);
    return historyCurveNearestPoint(metricPoints, step);
}

export function historyCurveMetricRange(points, smoothPoints, metric) {
    const values = (points || []).flatMap((point, index) => [
        point[metric.key],
        smoothPoints[index]?.[metric.smoothKey],
    ]).map(numberOrNull).filter((value) => value !== null);
    if (!values.length) return null;
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
        const pad = metric.key === 'lr'
            ? Math.max(Math.abs(min) * 0.08, 1e-12)
            : Math.max(Math.abs(min) * 0.08, 0.01);
        min -= pad;
        max += pad;
    }
    return { min, max };
}

export function historyCurveVisibleSegments(payload, points) {
    if (!points.length) return [];
    const minStep = points[0].step;
    const maxStep = points[points.length - 1].step;
    const out = [];
    const byMetric = points
        .filter((point) => point.stageBreakBefore)
        .map((point) => ({
            step: point.step,
            sourceTaskId: point.sourceTaskId,
            label: `任务${point.sourceTaskIndex || ''}`,
        }));
    for (const segmentPoint of byMetric) {
        if (segmentPoint.step >= minStep && segmentPoint.step <= maxStep) out.push(segmentPoint);
    }
    for (const segment of payload.segments || []) {
        const step = Number(segment.start_display_step);
        if (!Number.isFinite(step) || step < minStep || step > maxStep) continue;
        if (out.some((item) => Math.abs(item.step - step) < 0.001)) continue;
        out.push({ step, label: `任务${segment.index || ''}`, sourceTaskId: segment.task?.id || '' });
    }
    return out.sort((a, b) => a.step - b.step);
}

export function formatHistoryCurveAxisValue(value, metric, formatLr) {
    return metric.key === 'lr' ? formatLr(value) : Number(value).toFixed(4);
}

export function formatLossValue(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(5) : '-';
}

export function formatSignedLoss(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '-';
    const sign = n > 0 ? '+' : '';
    return `${sign}${n.toFixed(5)}`;
}

export function formatSignedLr(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '-';
    const sign = n > 0 ? '+' : '';
    return `${sign}${n.toExponential(2)}`;
}

function strOrEmpty(value) {
    return value === undefined || value === null ? '' : String(value);
}
