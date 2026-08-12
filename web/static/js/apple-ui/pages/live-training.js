/* Live training page: loss chart, step progress, system metrics.
 * Polls /api/training/status and /api/training/metrics.
 */

import { createApiClient } from '../../shared/api.js?v=apple-ui-20260812v33';

const api = createApiClient();
let pollTimer = null;

export async function loadLiveTraining() {
    let status = {};
    let metrics = {};
    try { status = await api('/api/training/status'); } catch { /* server unavailable */ }
    try { metrics = await api('/api/training/metrics'); } catch { /* server unavailable */ }

    const progress = status?.latest_progress || {};
    const systemStats = status?.latest_system || {};
    const state = status?.status || 'idle';
    const step = progress.step || 0;
    const total = progress.total_steps || progress.total || 0;
    const loss = progress.loss;
    const lr = progress.lr;
    const epoch = progress.epoch || 0;
    const vram = systemStats.gpu_memory_used || systemStats.vram_used;
    const gpuTemp = systemStats.gpu_temp;
    const gpuUtil = systemStats.gpu_util;
    const lossHistory = Array.isArray(metrics?.metrics) ? metrics.metrics : [];

    const lossChart = renderLossChart(lossHistory);
    const progressPct = total > 0 ? Math.min(100, (step / total) * 100) : 0;

    return `
        <div class="apple-page apple-page-wide">
            <div class="apple-page-hero apple-reveal">
                <div class="apple-status-badge" data-state="${state}">
                    <span class="apple-nav-status-dot" data-state="${state}"></span>
                    ${stateText(state)}
                </div>
                <h1>实时训练</h1>
                <p>${total > 0 ? `步数 ${step} / ${total}（${progressPct.toFixed(1)}%）` : '训练未运行'}</p>
            </div>

            <div class="apple-progress-bar-wrapper apple-reveal" data-stagger="1">
                <div class="apple-progress-bar">
                    <div class="apple-progress-bar-fill" style="width: ${progressPct}%"></div>
                </div>
                <div class="apple-progress-bar-meta">
                    <span>${step} / ${total} 步</span>
                    <span>轮数 ${epoch}</span>
                </div>
            </div>

            <div class="apple-metrics-grid apple-reveal" data-stagger="2">
                ${metricTile('损失值', loss != null ? Number(loss).toFixed(4) : '-')}
                ${metricTile('学习率', lr != null ? Number(lr).toExponential(3) : '-')}
                ${metricTile('显存', vram != null ? Number(vram).toFixed(1) + ' GB' : '-')}
                ${metricTile('GPU 利用率', gpuUtil != null ? gpuUtil + '%' : '-')}
                ${metricTile('GPU 温度', gpuTemp != null ? gpuTemp + '\u00b0C' : '-')}
                ${metricTile('轮数', epoch || '-')}
            </div>

            <div class="apple-section apple-reveal" data-stagger="3">
                <h2 class="apple-section-title">损失曲线</h2>
                <p class="apple-section-desc">最近 ${lossHistory.length} 条记录。</p>
                <div class="apple-chart-container">
                    ${lossChart}
                </div>
            </div>
        </div>
    `;
}

export function renderLossChart(metrics) {
    if (!metrics.length) {
        return '<div class="apple-empty-state"><p>暂无训练数据</p></div>';
    }

    const width = 800;
    const height = 240;
    const padding = { top: 20, right: 20, bottom: 30, left: 50 };
    const innerW = width - padding.left - padding.right;
    const innerH = height - padding.top - padding.bottom;

    const losses = metrics.map((m) => Number(m.loss)).filter((n) => !isNaN(n));
    if (!losses.length) return '<div class="apple-empty-state"><p>暂无损失数据</p></div>';

    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const range = maxLoss - minLoss || 1;

    const points = losses.map((loss, i) => {
        const x = padding.left + (i / (losses.length - 1 || 1)) * innerW;
        const y = padding.top + (1 - (loss - minLoss) / range) * innerH;
        return `${x},${y}`;
    }).join(' ');

    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
        const y = padding.top + t * innerH;
        const val = (maxLoss - t * range).toFixed(3);
        return `<line x1="${padding.left}" y1="${y}" x2="${padding.left + innerW}" y2="${y}" stroke="var(--apple-divider)" stroke-width="0.5"/>
                 <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" fill="var(--apple-text-quaternary)" font-size="11">${val}</text>`;
    }).join('');

    return `
        <svg class="apple-loss-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
            ${yTicks}
            <polyline points="${points}" fill="none" stroke="var(--apple-accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
        </svg>
    `;
}

function metricTile(label, value) {
    return `
        <div class="apple-metric-tile">
            <div class="apple-metric-value">${value}</div>
            <div class="apple-metric-label">${label}</div>
        </div>
    `;
}

function stateText(state) {
    const map = {
        idle: '空闲', running: '训练中', training: '训练中',
        queued: '排队中', completed: '已完成', error: '错误',
        stopped: '已停止', unknown: '未知',
    };
    return map[state] || state;
}
