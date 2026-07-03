import { historyDetailEmptyText, historyDetailSection } from '../ui.js?v=module-bootstrap-20260703-9';
import { createHistoryCurveChart } from './chart.js?v=module-bootstrap-20260703-9';
import {
    createHistoryCurveMetrics,
    historyCurveDisplayPoints,
    historyCurveFilteredPoints,
    historyCurvePoints,
    historyCurveSmoothPoints,
    historyCurveStats,
} from './data.js?v=module-bootstrap-20260703-9';
import { createHistoryCurveHover } from './hover.js?v=module-bootstrap-20260703-9';
import { createHistoryCurveToolbar } from './toolbar.js?v=module-bootstrap-20260703-9';

export function createHistoryCurveRenderer({ state, deps, renderHistoryDetailContent, renderHistoryDetailSystem }) {
    const { curve: historyCurveState } = state;
    const metrics = createHistoryCurveMetrics(deps.formatLr);
    const hover = createHistoryCurveHover({
        historyCurveState,
        metrics,
        formatLr: deps.formatLr,
    });
    const chart = createHistoryCurveChart({
        historyCurveState,
        metrics,
        formatLr: deps.formatLr,
        hover,
        renderHistoryDetailContent,
    });
    const toolbar = createHistoryCurveToolbar({ historyCurveState, renderHistoryDetailContent });

    function renderHistoryDetailChart(payload) {
        const box = document.createElement('div');
        box.className = 'history-curve-workbench';
        const allPoints = historyCurvePoints(payload, deps);
        if (!allPoints.length) {
            box.appendChild(historyDetailEmptyText('没有可绘制的 Loss 或学习率数据。'));
            return box;
        }
        const filteredPoints = historyCurveFilteredPoints(allPoints, historyCurveState);
        const smoothPoints = historyCurveSmoothPoints(filteredPoints, historyCurveState.smoothWindow, metrics);
        const stats = historyCurveStats(filteredPoints, smoothPoints, historyCurveState, metrics);
        const chartPoints = historyCurveDisplayPoints(filteredPoints);
        const chartSmoothPoints = historyCurveSmoothPoints(chartPoints, historyCurveState.smoothWindow, metrics);
        box.append(
            renderHistoryCurveStats(stats, payload),
            toolbar.renderHistoryCurveToolbar(allPoints),
            chart.renderHistoryCurveMainChart(payload, chartPoints, chartSmoothPoints, stats, filteredPoints.length),
            hover.renderHistoryCurveInspector(stats),
            chart.renderHistoryCurveSegments(payload, allPoints),
        );
        return box;
    }

    function renderHistoryDetailAnalysis(payload) {
        const box = document.createElement('div');
        box.className = 'history-detail-analysis';
        box.append(
            historyDetailSection('Loss / 学习率曲线', renderHistoryDetailChart(payload), 'history-detail-section chart'),
            renderHistoryDetailSystem(payload),
        );
        return box;
    }

    function renderHistoryCurveStats(stats, payload = {}) {
        const grid = document.createElement('div');
        grid.className = 'history-curve-stat-grid';
        const task = payload.task || {};
        grid.append(
            historyCurveStatGroup('Loss 组', [
                ['Loss 点', stats.loss.count],
                ['最新 Loss', formatLossValue(stats.loss.latest?.loss)],
                ['最低 Loss', formatLossValue(stats.loss.minPoint?.loss)],
                ['Loss 近50均值', formatLossValue(stats.loss.recentAverage)],
            ], 'loss'),
            historyCurveStatGroup('学习率组', [
                ['学习率点', stats.lr.count],
                ['最后有效学习率', deps.formatLr(stats.lr.activeLatest?.lr)],
                ['峰值学习率', deps.formatLr(stats.lr.maxPoint?.lr)],
                ['学习率变化', formatSignedLr(stats.lr.delta)],
            ], 'lr'),
            historyCurveStatGroup('Step 组', [
                ['Step 范围', stats.stepRange],
                ['最新 Step', stats.latest ? `Step ${Math.round(stats.latest.step)}` : '-'],
                ['原始 Step', stats.latest?.rawStep ?? '-'],
                ['平滑窗口', historyCurveState.smoothWindow],
            ], 'step'),
            historyCurveStatGroup('速度组', [
                ['平均速度', formatHistoryAverageSpeed(task)],
                ['采样范围', formatAverageSpeedStepRange(task)],
                ['采样数', task.average_step_sample_count || '-'],
                ['来源', task.average_step_source || '-'],
            ], 'speed'),
        );
        return grid;
    }

    function historyCurveStatGroup(title, rows, className = '') {
        const group = document.createElement('div');
        group.className = `history-curve-stat-group ${className}`.trim();
        const head = document.createElement('div');
        head.className = 'history-curve-stat-group-head';
        const strong = document.createElement('strong');
        strong.textContent = title;
        const span = document.createElement('span');
        span.textContent = rows.map(([label]) => label).slice(0, 2).join(' / ');
        head.append(strong, span);
        const body = document.createElement('div');
        body.className = 'history-curve-stat-items';
        rows.forEach(([label, value]) => {
            const item = document.createElement('div');
            item.className = 'history-curve-stat';
            const valueNode = document.createElement('strong');
            valueNode.textContent = String(value === undefined || value === null || value === '' ? '-' : value);
            const labelNode = document.createElement('span');
            labelNode.textContent = label;
            item.append(valueNode, labelNode);
            body.appendChild(item);
        });
        group.append(head, body);
        return group;
    }

    function formatLossValue(value) {
        const text = String(value ?? '').trim();
        if (/^[+\-]?nan$/i.test(text)) return 'NaN';
        if (/^[+\-]?inf(?:inity)?$/i.test(text)) return text.startsWith('-') ? '-Infinity' : 'Infinity';
        const n = Number(value);
        return Number.isFinite(n) ? n.toFixed(5) : '-';
    }

    function formatSignedLr(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return '-';
        const sign = n > 0 ? '+' : '';
        return `${sign}${n.toExponential(2)}`;
    }

    function formatHistoryAverageSpeed(task) {
        const rate = String(task?.average_step_rate || '').trim();
        if (rate) return rate;
        const seconds = Number(task?.average_step_seconds);
        return Number.isFinite(seconds) && seconds > 0 ? `${seconds.toFixed(2)}s/step` : '-';
    }

    function formatAverageSpeedStepRange(task) {
        const start = Number(task?.average_step_start_step);
        const end = Number(task?.average_step_end_step);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return '-';
        return `${Math.round(start)} → ${Math.round(end)}`;
    }

    return { renderHistoryDetailChart, renderHistoryDetailAnalysis };
}
