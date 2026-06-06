import {
    clampNumber,
    formatCompactNumber,
    historyDetailEmptyText,
    historyDetailSection,
    maxRecordBy,
    maxValue,
    numberOrNull,
    svgCircle,
    svgGroup,
    svgLine,
    svgRect,
    svgText,
} from './ui.js?v=module-bootstrap-20260604-10';

const HISTORY_SYSTEM_TABLE_RENDER_LIMIT = 500;

export function createHistorySystemRenderer() {
    function renderHistoryDetailSystem(payload) {
        const box = document.createElement('div');
        box.className = 'history-detail-system';
        if (payload.mode === 'config_group') {
            box.appendChild(historyDetailEmptyText('合并视图暂不汇总系统指标。请打开单个历史任务查看 VRAM、GPU 占用和温度采样。'));
            return box;
        }
        const records = historySystemRecords(payload);
        const summary = historySystemSummary(payload);
        if (!records.length) {
            box.appendChild(historyDetailEmptyText('无系统采样记录。历史 GPU 信息来自该任务的 system.jsonl，旧任务或采样未启动时这里会为空。'));
            return box;
        }

        const statGrid = document.createElement('div');
        statGrid.className = 'history-detail-stat-grid history-system-summary';
        [
            { label: '采样数', value: records.length },
            { label: '最后 VRAM', value: formatSystemVram(summary.last), progress: historySystemVramPercent(summary.last), tone: 'vram' },
            { label: '峰值 VRAM', value: formatSystemVram(summary.peakVramRecord), progress: historySystemVramPercent(summary.peakVramRecord), tone: 'vram' },
            { label: '最后 GPU', value: formatSystemPercent(summary.last?.gpuUtil), progress: summary.last?.gpuUtil, tone: 'gpu' },
            { label: '峰值 GPU', value: formatSystemPercent(summary.peakGpu), progress: summary.peakGpu, tone: 'gpu' },
            { label: '最后温度', value: formatSystemTemperature(summary.last?.gpuTemp), progress: summary.last?.gpuTemp, tone: 'temp' },
            { label: '峰值温度', value: formatSystemTemperature(summary.peakTemp), progress: summary.peakTemp, tone: 'temp' },
            { label: '最后输出', value: formatHistorySystemTime(summary.last?.lastOutputAt) },
        ].forEach((item) => statGrid.appendChild(historySystemSummaryStat(item)));

        const trends = document.createElement('div');
        trends.className = 'history-system-trends';
        trends.append(
            historySystemTrendCard('VRAM', records, 'vramUsed', formatSystemGigabytes, {
                yMin: 0,
                yMax: historySystemVramAxisMax(records),
                tone: 'vram',
            }),
            historySystemTrendCard('GPU 占用', records, 'gpuUtil', formatSystemPercent, {
                yMin: 0,
                yMax: 100,
                tone: 'gpu',
            }),
            historySystemTrendCard('温度', records, 'gpuTemp', formatSystemTemperature, {
                yMin: 0,
                yMax: 100,
                tone: 'temp',
            }),
        );

        const table = document.createElement('div');
        table.className = 'history-system-table';
        const tableRecords = records.slice(-HISTORY_SYSTEM_TABLE_RENDER_LIMIT);
        const tableNotice = historyDetailLimitNotice(payload, 'system', '系统采样');
        const head = document.createElement('div');
        head.className = 'history-system-table-row history-system-table-head';
        ['时间', 'VRAM', 'GPU 占用', '温度', 'last_output_at'].forEach((label) => {
            const cell = document.createElement('span');
            cell.textContent = label;
            head.appendChild(cell);
        });
        table.appendChild(head);
        if (tableNotice) table.appendChild(tableNotice);
        if (records.length > tableRecords.length && !tableNotice) {
            const localNotice = document.createElement('p');
            localNotice.className = 'history-detail-limit-note';
            localNotice.textContent = `明细仅渲染最近 ${tableRecords.length} 条采样，摘要和趋势基于当前返回数据计算。`;
            table.appendChild(localNotice);
        }
        for (const record of tableRecords) {
            const row = document.createElement('div');
            row.className = 'history-system-table-row';
            [
                formatHistorySystemTime(record.ts),
                formatSystemVram(record),
                formatSystemPercent(record.gpuUtil),
                formatSystemTemperature(record.gpuTemp),
                formatHistorySystemTime(record.lastOutputAt),
            ].forEach((value) => {
                const cell = document.createElement('span');
                cell.textContent = value;
                row.appendChild(cell);
            });
            table.appendChild(row);
        }

        box.append(
            historyDetailSection('系统摘要', statGrid, 'history-detail-section system-summary'),
            historyDetailSection('系统趋势', trends, 'history-detail-section system-trends'),
            historyDetailSection('系统采样明细', table, 'history-detail-section system-table-section'),
        );
        return box;
    }

    return { renderHistoryDetailSystem };
}

export function historyDetailLimitNotice(payload, key, label) {
    const limits = payload?.limits || {};
    const total = Number(limits[`${key}_total`]);
    const returned = Number(limits[`${key}_returned`]);
    const truncated = Boolean(limits[`${key}_truncated`]);
    if (!truncated || !Number.isFinite(total) || !Number.isFinite(returned)) return null;
    const note = document.createElement('p');
    note.className = 'history-detail-limit-note';
    note.textContent = `仅显示最近 ${returned} 条${label}记录，共 ${total} 条；完整内容仍保留在历史目录文件中。`;
    return note;
}

export function historySystemRecords(payload) {
    return (payload.system || [])
        .map((record) => {
            const vramUsed = numberOrNull(record.vram_used_gb);
            const vramTotal = numberOrNull(record.vram_total_gb);
            const gpuUtil = numberOrNull(record.gpu_util);
            const gpuTemp = numberOrNull(record.gpu_temp);
            return {
                ...record,
                ts: numberOrNull(record.ts),
                lastOutputAt: numberOrNull(record.last_output_at),
                vramUsed,
                vramTotal,
                gpuUtil,
                gpuTemp,
                hasValue: vramUsed !== null || gpuUtil !== null || gpuTemp !== null,
            };
        })
        .filter((record) => record.hasValue);
}

export function historySystemSummary(payload) {
    const records = historySystemRecords(payload);
    const last = records[records.length - 1] || null;
    const peakVramRecord = maxRecordBy(records, 'vramUsed');
    return {
        hasSystem: records.length > 0,
        records,
        last,
        peakVramRecord,
        peakGpu: maxValue(records, 'gpuUtil'),
        peakTemp: maxValue(records, 'gpuTemp'),
    };
}

function historySystemSummaryStat({ label, value, progress = null, tone = '' }) {
    const item = document.createElement('div');
    item.className = `history-detail-stat history-system-stat ${tone}`.trim();
    const strong = document.createElement('strong');
    strong.textContent = String(value === undefined || value === null || value === '' ? '-' : value);
    const span = document.createElement('span');
    span.textContent = label;
    item.append(strong, span);
    const pct = numberOrNull(progress);
    if (pct !== null) {
        const bar = document.createElement('div');
        bar.className = 'history-system-progress';
        bar.setAttribute('aria-label', `${label} ${Math.round(clampNumber(pct, 0, 100))}%`);
        const fill = document.createElement('i');
        fill.style.width = `${clampNumber(pct, 0, 100)}%`;
        bar.appendChild(fill);
        item.appendChild(bar);
    }
    return item;
}

function historySystemVramPercent(record) {
    const used = numberOrNull(record?.vramUsed);
    const total = numberOrNull(record?.vramTotal);
    if (used === null || total === null || total <= 0) return null;
    return (used / total) * 100;
}

function historySystemVramAxisMax(records) {
    const totals = (records || []).map((record) => numberOrNull(record.vramTotal)).filter((value) => value !== null && value > 0);
    const peak = maxValue(records || [], 'vramUsed');
    return Math.max(1, totals.length ? Math.max(...totals) : 0, Math.ceil((peak || 1) * 1.15));
}

function formatSystemGigabytes(value) {
    const n = numberOrNull(value);
    return n === null ? '-' : `${formatCompactNumber(n)} GB`;
}

export function formatSystemVram(record) {
    if (!record || record.vramUsed === null || record.vramUsed === undefined) return '-';
    const used = `${formatCompactNumber(record.vramUsed)} GB`;
    if (record.vramTotal === null || record.vramTotal === undefined) return used;
    return `${used} / ${formatCompactNumber(record.vramTotal)} GB`;
}

export function formatSystemPercent(value) {
    const n = Number(value);
    return Number.isFinite(n) ? `${Math.round(n)}%` : '-';
}

export function formatSystemTemperature(value) {
    const n = Number(value);
    return Number.isFinite(n) ? `${Math.round(n)}°C` : '-';
}

export function formatHistorySystemTime(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return '-';
    try {
        return new Date(n * 1000).toLocaleString('zh-CN', { hour12: false });
    } catch (e) {
        return '-';
    }
}

function historySystemTrendCard(label, records, key, formatter, options = {}) {
    const card = document.createElement('div');
    card.className = `history-system-trend-card ${options.tone || ''}`.trim();
    const head = document.createElement('div');
    head.className = 'history-system-trend-head';
    const title = document.createElement('strong');
    title.textContent = label;
    const points = records
        .map((record, index) => ({
            x: Number.isFinite(record.ts) ? record.ts : index,
            y: numberOrNull(record[key]),
            ts: record.ts,
            index,
        }))
        .filter((point) => point.y !== null);
    const latest = points[points.length - 1]?.y;
    const summary = document.createElement('span');
    summary.textContent = `最后 ${formatter(latest)} · 峰值 ${formatter(maxValue(records, key))}`;
    head.append(title, summary);
    card.append(head, createHistorySystemSparkline(points, {
        ...options,
        label,
        formatter,
    }));
    return card;
}

function createHistorySystemSparkline(points, options = {}) {
    const width = 420;
    const height = 152;
    const pad = { top: 14, right: 14, bottom: 18, left: 48 };
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `${options.label || '系统指标'}趋势图，悬停查看采样值`);
    svg.classList.add('history-system-sparkline', options.tone || 'metric');
    if (!points.length) return svg;
    const gradientId = `history-system-area-${options.tone || 'metric'}-${Math.random().toString(36).slice(2, 9)}`;
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient.setAttribute('id', gradientId);
    gradient.setAttribute('x1', '0');
    gradient.setAttribute('x2', '0');
    gradient.setAttribute('y1', '0');
    gradient.setAttribute('y2', '1');
    const topStop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    topStop.setAttribute('offset', '0%');
    topStop.setAttribute('stop-color', historySystemTrendColor(options.tone));
    topStop.setAttribute('stop-opacity', '0.24');
    const bottomStop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    bottomStop.setAttribute('offset', '100%');
    bottomStop.setAttribute('stop-color', historySystemTrendColor(options.tone));
    bottomStop.setAttribute('stop-opacity', '0.02');
    gradient.append(topStop, bottomStop);
    defs.appendChild(gradient);
    svg.appendChild(defs);
    const xs = points.map((item, index) => Number.isFinite(item.x) ? item.x : index);
    const ys = points.map((item) => Number(item.y));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    let minY = numberOrNull(options.yMin) ?? Math.min(...ys);
    let maxY = numberOrNull(options.yMax) ?? Math.max(...ys);
    if (minY === maxY) {
        const extra = Math.max(Math.abs(maxY) * 0.08, 1);
        minY -= extra;
        maxY += extra;
    }
    if (maxY < minY) [minY, maxY] = [maxY, minY];
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const xScale = (value) => pad.left + ((value - minX) / Math.max(1, maxX - minX)) * plotW;
    const yScale = (value) => pad.top + ((maxY - Number(value)) / Math.max(0.000001, maxY - minY)) * plotH;
    const tickCount = Number(options.ticks) || 4;
    for (let i = 0; i <= tickCount; i += 1) {
        const y = pad.top + (plotH * i / tickCount);
        const value = maxY - ((maxY - minY) * i / tickCount);
        svg.appendChild(svgLine(pad.left, y, width - pad.right, y, 'history-system-grid-line'));
        svg.appendChild(svgText(pad.left - 8, y + 3, options.formatter ? options.formatter(value) : formatCompactNumber(value), 'history-system-axis-label'));
    }
    svg.appendChild(svgLine(pad.left, pad.top, pad.left, height - pad.bottom, 'history-system-axis-line'));
    svg.appendChild(svgLine(pad.left, height - pad.bottom, width - pad.right, height - pad.bottom, 'history-system-axis-line baseline'));
    const linePoints = points.map((item, index) => [xScale(xs[index]), yScale(Number(item.y))]);
    const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    area.setAttribute('d', historySystemAreaPath(linePoints, height - pad.bottom));
    area.setAttribute('class', 'history-system-area');
    area.style.fill = `url(#${gradientId})`;
    svg.appendChild(area);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    line.setAttribute('points', linePoints.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' '));
    line.setAttribute('class', 'history-sparkline-line');
    svg.appendChild(line);
    appendHistorySystemSparklineHover(svg, points, {
        width,
        height,
        pad,
        plotW,
        plotH,
        minX,
        maxX,
        xScale,
        yScale,
        label: options.label || '指标',
        formatter: options.formatter || formatCompactNumber,
    });
    return svg;
}

function appendHistorySystemSparklineHover(svg, points, layout) {
    const hoverLayer = svgGroup('history-system-hover-layer');
    const hoverLine = svgLine(0, layout.pad.top, 0, layout.height - layout.pad.bottom, 'history-system-hover-line');
    const hoverPoint = svgCircle(0, 0, 4.4, 'history-system-hover-point');
    const hoverLabel = svgGroup('history-system-hover-label');
    const hoverLabelBg = svgRect(0, 0, 1, 1, 'history-system-hover-label-bg');
    const hoverLabelValue = svgText(0, 0, '', 'history-system-hover-label-value');
    const hoverLabelTime = svgText(0, 0, '', 'history-system-hover-label-time');
    hoverLabel.append(hoverLabelBg, hoverLabelValue, hoverLabelTime);
    hoverLayer.append(hoverLine, hoverPoint, hoverLabel);
    setSvgVisibility(hoverLayer, false);
    svg.appendChild(hoverLayer);

    const overlay = svgRect(layout.pad.left, layout.pad.top, layout.plotW, layout.plotH, 'history-system-hover-overlay');
    const updateHover = (event) => {
        const svgPoint = svgClientPoint(svg, event);
        const viewX = svgPoint?.x ?? (() => {
            const rect = svg.getBoundingClientRect();
            const ratioX = (event.clientX - rect.left) / Math.max(1, rect.width);
            return ratioX * layout.width;
        })();
        const targetX = layout.minX
            + ((clampNumber(viewX, layout.pad.left, layout.width - layout.pad.right) - layout.pad.left) / Math.max(1, layout.plotW))
            * (layout.maxX - layout.minX);
        const nearest = nearestHistorySystemPoint(points, targetX);
        if (!nearest) {
            setSvgVisibility(hoverLayer, false);
            return;
        }
        positionHistorySystemHover(layout, nearest, hoverLine, hoverPoint, hoverLabelBg, hoverLabelValue, hoverLabelTime);
        hideSiblingHistorySystemHovers(svg, hoverLayer);
        setSvgVisibility(hoverLayer, true);
    };
    overlay.addEventListener('mousemove', updateHover);
    overlay.addEventListener('mouseenter', updateHover);
    overlay.addEventListener('mouseleave', () => setSvgVisibility(hoverLayer, false));
    svg.appendChild(overlay);
}

function hideSiblingHistorySystemHovers(svg, activeLayer) {
    svg.closest('.history-system-trends')
        ?.querySelectorAll('.history-system-hover-layer')
        .forEach((layer) => {
            if (layer !== activeLayer) setSvgVisibility(layer, false);
        });
}

function nearestHistorySystemPoint(points, targetX) {
    let best = null;
    let bestDistance = Infinity;
    for (const point of points) {
        const distance = Math.abs(Number(point.x) - targetX);
        if (distance < bestDistance) {
            best = point;
            bestDistance = distance;
        }
    }
    return best;
}

function positionHistorySystemHover(layout, point, hoverLine, hoverPoint, labelBg, labelValue, labelTime) {
    const x = layout.xScale(point.x);
    const y = layout.yScale(point.y);
    hoverLine.setAttribute('x1', `${x}`);
    hoverLine.setAttribute('x2', `${x}`);
    hoverPoint.setAttribute('cx', `${x}`);
    hoverPoint.setAttribute('cy', `${y}`);

    const valueText = `${layout.label} ${layout.formatter(point.y)}`;
    const timeText = formatHistorySystemTime(point.ts) || `采样 ${point.index + 1}`;
    const labelW = Math.max(116, Math.min(210, Math.max(valueText.length * 7.4, timeText.length * 6.2) + 18));
    const labelH = 38;
    const labelX = clampNumber(x + 10, layout.pad.left + 4, layout.width - layout.pad.right - labelW - 4);
    const preferTop = y > layout.pad.top + labelH + 14;
    const labelY = preferTop
        ? y - labelH - 10
        : clampNumber(y + 10, layout.pad.top + 4, layout.height - layout.pad.bottom - labelH - 4);
    labelBg.setAttribute('x', `${labelX}`);
    labelBg.setAttribute('y', `${labelY}`);
    labelBg.setAttribute('width', `${labelW}`);
    labelBg.setAttribute('height', `${labelH}`);
    labelValue.setAttribute('x', `${labelX + 9}`);
    labelValue.setAttribute('y', `${labelY + 15}`);
    labelValue.textContent = valueText;
    labelTime.setAttribute('x', `${labelX + 9}`);
    labelTime.setAttribute('y', `${labelY + 30}`);
    labelTime.textContent = timeText;
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

function setSvgVisibility(node, visible) {
    node.setAttribute('visibility', visible ? 'visible' : 'hidden');
}

function historySystemAreaPath(points, baselineY) {
    if (!points.length) return '';
    const [firstX] = points[0];
    const [lastX] = points[points.length - 1];
    const line = points.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`).join('');
    return `M${firstX.toFixed(2)} ${baselineY.toFixed(2)}${line.replace(/^M/, 'L')}L${lastX.toFixed(2)} ${baselineY.toFixed(2)}Z`;
}

function historySystemTrendColor(tone) {
    if (tone === 'gpu') return 'var(--success)';
    if (tone === 'temp') return 'var(--warning)';
    return 'var(--accent-strong)';
}
