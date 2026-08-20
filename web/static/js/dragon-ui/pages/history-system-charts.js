/* Independent VRAM, GPU-utilization, and temperature charts for history details. */

const WIDTH = 420;
const HEIGHT = 180;
const PADDING = { top: 30, right: 14, bottom: 18, left: 54 };

const SYSTEM_CHARTS = Object.freeze([
    { id: 'vram', label: 'VRAM', key: 'vramUsed', tone: 'vram', unit: 'gb' },
    { id: 'gpu', label: 'GPU 占用', key: 'gpuUtil', tone: 'gpu', unit: 'percent', max: 100 },
    { id: 'temp', label: '温度', key: 'gpuTemp', tone: 'temp', unit: 'temperature', max: 100 },
]);

export function renderHistorySystemCharts(records = [], limits = {}) {
    const rows = normalizeSystemRecords(records);
    return `<section class="dragon-history-system-section dragon-reveal" aria-labelledby="dragon-history-system-title">
        <div class="dragon-history-system-head">
            <div><span class="dragon-eyebrow">资源监控</span><h2 id="dragon-history-system-title">系统趋势</h2></div>
            ${rows.length ? `<span>${systemSampleLabel(rows.length, limits)}</span>` : ''}
        </div>
        ${rows.length ? `<div class="dragon-history-system-grid">
            ${SYSTEM_CHARTS.map((spec) => renderSystemChartCard(rows, spec)).join('')}
        </div>` : '<div class="dragon-history-inline-empty"><p>当前任务没有可用的 GPU 系统采样。</p></div>'}
    </section>`;
}

export function bindHistorySystemCharts(root, records = []) {
    const rows = normalizeSystemRecords(records);
    const cleanups = SYSTEM_CHARTS.map((spec) => {
        const svg = root?.querySelector(`[data-history-system-chart="${spec.id}"]`);
        return bindSystemChartHover(svg, seriesPoints(rows, spec), rows, spec);
    });
    return () => cleanups.forEach((cleanup) => cleanup());
}

function renderSystemChartCard(rows, spec) {
    const points = seriesPoints(rows, spec);
    const latest = points[points.length - 1]?.value;
    const peak = points.length ? Math.max(...points.map((point) => point.value)) : null;
    const summary = `最后 ${formatSystemValue(latest, spec.unit)} · 峰值 ${formatSystemValue(peak, spec.unit)}`;
    return `<article class="dragon-history-system-card" data-tone="${spec.tone}">
        <div class="dragon-history-system-card-head"><strong>${spec.label}</strong><span>${summary}</span></div>
        ${renderSystemChart(points, rows, spec)}
    </article>`;
}

function renderSystemChart(points, rows, spec) {
    if (!points.length) return '<div class="dragon-history-system-chart-empty">无可用采样</div>';
    const domain = chartDomain(points, rows, spec);
    const layout = chartLayout(points, domain);
    const grid = [0, .25, .5, .75, 1].map((ratio) => {
        const y = PADDING.top + ratio * layout.plotH;
        const value = domain.max - ratio * (domain.max - domain.min);
        return `<line x1="${PADDING.left}" y1="${y}" x2="${WIDTH - PADDING.right}" y2="${y}" class="dragon-history-system-grid-line"/><text x="${PADDING.left - 8}" y="${y + 3}" class="dragon-history-system-axis-label">${formatSystemValue(value, spec.unit)}</text>`;
    }).join('');
    const line = points.map((point) => `${layout.x(point.x).toFixed(2)},${layout.y(point.value).toFixed(2)}`).join(' ');
    const area = systemAreaPath(points.map((point) => [layout.x(point.x), layout.y(point.value)]), HEIGHT - PADDING.bottom);
    return `<svg class="dragon-history-system-chart" data-history-system-chart="${spec.id}" viewBox="0 0 ${WIDTH} ${HEIGHT}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${spec.label}趋势图，悬停查看采样值">
        ${grid}<path d="${area}" class="dragon-history-system-area"/><polyline points="${line}" class="dragon-history-system-line"/>
        <g data-history-system-hover hidden><line data-history-system-hover-guide y1="${PADDING.top}" y2="${HEIGHT - PADDING.bottom}" class="dragon-history-system-hover-guide"/><circle data-history-system-hover-point r="4.2" class="dragon-history-system-hover-point"/><text x="${WIDTH - PADDING.right}" y="17" text-anchor="end" class="dragon-history-system-hover-text" data-history-system-hover-text></text></g>
        <rect x="${PADDING.left}" y="${PADDING.top}" width="${layout.plotW}" height="${layout.plotH}" class="dragon-history-system-hitarea"/>
    </svg>`;
}

function bindSystemChartHover(svg, points, rows, spec) {
    if (!svg || !points.length) return () => {};
    const domain = chartDomain(points, rows, spec);
    const layout = chartLayout(points, domain);
    const overlay = svg.querySelector('.dragon-history-system-hitarea');
    const hover = svg.querySelector('[data-history-system-hover]');
    const guide = svg.querySelector('[data-history-system-hover-guide]');
    const point = svg.querySelector('[data-history-system-hover-point]');
    const text = svg.querySelector('[data-history-system-hover-text]');
    const onMove = (event) => updateSystemHover(svg, event, points, spec, layout, { hover, guide, point, text });
    const hide = () => hover?.setAttribute('hidden', '');
    overlay?.addEventListener('pointermove', onMove);
    overlay?.addEventListener('mousemove', onMove);
    overlay?.addEventListener('pointerleave', hide);
    overlay?.addEventListener('mouseleave', hide);
    return () => {
        overlay?.removeEventListener('pointermove', onMove);
        overlay?.removeEventListener('mousemove', onMove);
        overlay?.removeEventListener('pointerleave', hide);
        overlay?.removeEventListener('mouseleave', hide);
    };
}

function updateSystemHover(svg, event, points, spec, layout, elements) {
    const viewX = clientPointToSvg(svg, event.clientX, event.clientY).x;
    const ratio = clamp((viewX - PADDING.left) / layout.plotW, 0, 1);
    const targetX = layout.minX + ratio * (layout.maxX - layout.minX);
    const sample = nearestPoint(points, targetX);
    if (!sample) return;
    const x = layout.x(sample.x);
    const y = layout.y(sample.value);
    elements.guide?.setAttribute('x1', String(x));
    elements.guide?.setAttribute('x2', String(x));
    elements.point?.setAttribute('cx', String(x));
    elements.point?.setAttribute('cy', String(y));
    if (elements.text) elements.text.textContent = `${formatSystemValue(sample.value, spec.unit)} · ${formatSystemTime(sample.ts)}`;
    elements.hover?.removeAttribute('hidden');
}

function normalizeSystemRecords(records) {
    return (Array.isArray(records) ? records : []).map((record, index) => ({
        index,
        ts: numeric(record?.ts),
        vramUsed: numeric(record?.vram_used_gb),
        vramTotal: numeric(record?.vram_total_gb),
        gpuUtil: numeric(record?.gpu_util),
        gpuTemp: numeric(record?.gpu_temp),
    })).filter((row) => row.vramUsed != null || row.gpuUtil != null || row.gpuTemp != null);
}

function seriesPoints(rows, spec) {
    return rows.map((row, index) => ({
        x: row.ts ?? index,
        ts: row.ts,
        value: row[spec.key],
    })).filter((point) => point.value != null);
}

function chartDomain(points, rows, spec) {
    if (spec.id === 'vram') {
        const totals = rows.map((row) => row.vramTotal).filter((value) => value != null && value > 0);
        const peak = Math.max(...points.map((point) => point.value), 1);
        return { min: 0, max: Math.max(totals.length ? Math.max(...totals) : 0, Math.ceil(peak * 1.15), 1) };
    }
    return { min: 0, max: spec.max || Math.max(...points.map((point) => point.value), 1) };
}

function chartLayout(points, domain) {
    const minX = Math.min(...points.map((point) => point.x));
    const maxX = Math.max(...points.map((point) => point.x));
    const plotW = WIDTH - PADDING.left - PADDING.right;
    const plotH = HEIGHT - PADDING.top - PADDING.bottom;
    return {
        minX,
        maxX,
        plotW,
        plotH,
        x: (value) => PADDING.left + ((value - minX) / Math.max(1, maxX - minX)) * plotW,
        y: (value) => PADDING.top + ((domain.max - value) / Math.max(.000001, domain.max - domain.min)) * plotH,
    };
}

function nearestPoint(points, targetX) {
    let nearest = null;
    let distance = Infinity;
    for (const point of points) {
        const nextDistance = Math.abs(point.x - targetX);
        if (nextDistance < distance) { nearest = point; distance = nextDistance; }
    }
    return nearest;
}

function clientPointToSvg(svg, clientX, clientY) {
    const matrix = svg.getScreenCTM?.();
    if (matrix && typeof DOMPoint === 'function') {
        try { return new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse()); } catch { /* use viewBox fallback */ }
    }
    const rect = svg.getBoundingClientRect();
    return { x: ((clientX - rect.left) / Math.max(1, rect.width)) * WIDTH, y: ((clientY - rect.top) / Math.max(1, rect.height)) * HEIGHT };
}

function systemAreaPath(points, baseline) {
    if (!points.length) return '';
    const [firstX] = points[0];
    const [lastX] = points[points.length - 1];
    const line = points.map(([x, y]) => `L${x.toFixed(2)} ${y.toFixed(2)}`).join('');
    return `M${firstX.toFixed(2)} ${baseline}${line}L${lastX.toFixed(2)} ${baseline}Z`;
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
