/* Presentational helpers for the dedicated live training workspace. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { renderStatusRegion, renderToolButton } from './tool-page.js?v=dragon-ui-20260814v43';
import { isRunningState, logRecordText, stateText, visualState } from './live-training-state.js?v=dragon-ui-20260825v46';
import { highlightedLogHtml } from './log-highlighter.js?v=dragon-ui-20260825v1';
import {
    hardwarePercent,
    liveWorkspaceMode,
    lossDelta,
    renderLiveSidebar,
    renderLiveStatePanel,
} from './live-training-workspace.js?v=dragon-ui-20260825v46';

export function renderLiveTrainingPage(model, renderLossChart) {
    const running = isRunningState(model.state);
    const mode = liveWorkspaceMode(model.state);
    const smoothing = Number.isFinite(Number(model.chartSmoothing)) ? Number(model.chartSmoothing) : .2;
    const delta = lossDelta(model.metrics);
    const badge = `<div class="dragon-status-badge" data-live-state data-state="${visualState(model.state)}"><span class="dragon-nav-status-dot" data-state="${visualState(model.state)}"></span><span data-live-state-text>${stateText(model.state)}</span></div>`;
    const actions = [
        renderToolButton('list', '查看队列', 'queue'),
        renderToolButton('stop', '停止训练', 'stop', 'dragon-btn-secondary dragon-btn-danger', running ? '' : 'hidden disabled'),
    ].join('');
    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-live-training-page" data-live-training-root data-live-mode="${mode}">
            <div class="dragon-live-shell">
                ${renderLiveSidebar(model)}
                <main class="dragon-live-workspace">
                    <header class="dragon-live-workbench-header">
                        <div class="dragon-live-workbench-copy"><span class="dragon-eyebrow">训练监控</span><div><h1 data-live-header-title>${escapeHtml(headerTitle(model, mode))}</h1>${badge}</div><p data-live-header-meta>${escapeHtml(headerMeta(model, mode))}</p></div>
                        <div class="dragon-live-workbench-actions">${actions}</div>
                    </header>
                    ${renderStatusRegion('data-live-feedback')}
                    ${renderLiveStatePanel(model, 'idle')}
                    ${renderLiveStatePanel(model, 'error')}
                    <div class="dragon-live-running-workspace" data-live-section="running" ${mode === 'running' ? '' : 'hidden'}>
                        <section class="dragon-live-core-grid" aria-label="训练核心指标">
                            ${coreMetric('实时损失', formatLoss(model.loss), 'live-loss', delta.text, delta.tone)}
                            ${coreMetric('实时学习率', formatLr(model.lr), 'live-lr', '实时调度值')}
                            ${progressMetric(model)}
                            ${coreMetric('速度与 ETA', formatRate(model.rate), 'live-rate', formatEta(model), 'neutral', 'live-eta')}
                        </section>
                        <div class="dragon-live-control-grid">
                            <div class="dragon-live-telemetry-column">
                                ${renderChartPanel(model, smoothing, renderLossChart)}
                                ${renderHardwarePanel(model)}
                            </div>
                            ${renderConsole(model)}
                        </div>
                    </div>
                </main>
            </div>
            ${renderStopDialog()}
        </div>`;
}

function renderChartPanel(model, smoothing, renderLossChart) {
    return `<section class="dragon-tool-panel dragon-live-chart-panel"><div class="dragon-tool-panel-head dragon-live-chart-head"><div><span class="dragon-eyebrow">LOSS TREND</span><h2>损失趋势</h2></div><div class="dragon-live-chart-tools"><label><span>平滑度</span><input type="range" min="0" max="99" value="${Math.round(smoothing * 100)}" data-live-chart-smoothing aria-label="损失曲线平滑度"><output data-live-chart-smoothing-value>${Math.round(smoothing * 100)}%</output></label><span class="dragon-tool-note" data-live-chart-count>最近 ${model.metrics.length} 条记录</span></div></div><div class="dragon-chart-container" data-live-chart>${renderLossChart(model.metrics, smoothing)}</div></section>`;
}

function renderHardwarePanel(model) {
    return `<section class="dragon-tool-panel dragon-live-hardware-panel" aria-label="硬件监控"><div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">SYSTEM HEALTH</span><h2>硬件监控</h2></div></div><div class="dragon-live-hardware-grid">
        ${hardwareCard('memory', '显存', formatVram(model.vram, null), `Max ${formatVram(model.vramTotal, null)} · ${formatPeakMetric(formatVram(model.peakVram, null))}`, 'live-vram', hardwarePercent(model.vram, model.vramTotal))}
        ${hardwareCard('cpu', 'GPU 算力', formatPercent(model.gpuUtil), formatPeakMetric(formatPercent(model.peakGpuUtil)), 'live-gpu-util', Number(model.gpuUtil) || 0)}
        ${hardwareCard('thermometer', '核心温度', formatTemperature(model.gpuTemp), temperatureDetail(model.gpuTemp, model.peakGpuTemp), 'live-gpu-temp', Number(model.gpuTemp) || 0, Number(model.gpuTemp) >= 80 ? 'warning' : 'normal')}
    </div></section>`;
}

function renderConsole(model) {
    return `<section class="dragon-live-console dragon-live-log-panel"><div class="dragon-live-console-head"><div><span class="dragon-eyebrow">LIVE OUTPUT</span><h2>实时控制台</h2></div><div class="dragon-live-console-status" data-live-connection data-tone="info" role="status" aria-live="polite"><span class="dragon-live-connection-dot" aria-hidden="true"></span><strong data-live-connection-label>正在连接</strong><span class="visually-hidden" data-live-connection-detail>正在建立实时推送连接。</span></div></div><div class="dragon-live-log-toolbar" role="toolbar" aria-label="实时日志工具"><label class="dragon-live-log-search">${renderIcon('search')}<span class="visually-hidden">搜索实时日志</span><input type="search" name="live_log_search" autocomplete="off" placeholder="搜索日志" data-live-log-search></label><div class="dragon-live-log-actions">${logToolButton('copy', '复制日志', 'copy')}${logToolButton('download', '下载日志', 'download')}${logToolButton('clear', '清空视图', 'trash')}${logToolButton('pause', '自动滚屏', 'activity', 'aria-pressed="true"', false)}</div></div><div class="dragon-live-log-meta"><span data-live-log-visible-count>${model.logs.length} 条可见</span><span data-live-log-count>${model.logs.length} 条</span></div><p class="dragon-live-log-feedback" data-live-log-feedback role="status" aria-live="polite"></p><div class="dragon-log-stream dragon-live-log-stream" data-live-log tabindex="0" aria-label="实时训练日志" aria-live="off">${renderLogs(model.logs)}</div></section>`;
}

function coreMetric(label, value, key, detail, tone = 'neutral', detailKey = '') {
    return `<article class="dragon-live-core-metric" data-live-metric="${key}" aria-label="${key === 'live-rate' ? '训练速度与预计剩余（ETA）' : label}"><span>${label}</span><strong>${formatMetricHtml(value)}</strong><small ${detailKey ? `data-live-metric-detail="${detailKey}" aria-label="预计剩余（ETA）"` : ''} data-tone="${tone}">${escapeHtml(detail)}</small></article>`;
}

function progressMetric(model) {
    return `<article class="dragon-live-core-metric dragon-live-progress-metric" data-live-metric="live-progress"><span>步数与进度</span><strong data-live-progress-text>${model.total > 0 ? `${model.progressPct.toFixed(1)}%` : '-'}</strong><small><span data-live-step-text>${model.step} / ${model.total} 步</span><span data-live-epoch-text>第 ${formatEpoch(model.epoch)} 轮</span></small><div class="dragon-progress-bar" role="progressbar" aria-label="训练进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(model.progressPct)}" data-live-progress><div class="dragon-progress-bar-fill" data-live-progress-fill style="width:${model.progressPct}%"></div></div></article>`;
}

function hardwareCard(icon, label, value, detail, key, percent, tone = 'normal') {
    return `<article class="dragon-live-hardware-card" data-live-metric="${key}" data-tone="${tone}" aria-label="${key === 'live-vram' ? '显存与峰值显存' : label}"><span class="dragon-live-hardware-icon">${renderIcon(icon)}</span><div><span>${label}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small><div class="dragon-live-meter" aria-hidden="true"><i style="width:${Math.max(0, Math.min(100, percent))}%"></i></div></div></article>`;
}

function renderStopDialog() {
    return `<dialog class="dragon-live-stop-dialog" data-live-stop-dialog aria-labelledby="dragon-live-stop-title"><form method="dialog"><span class="dragon-live-stop-symbol">${renderIcon('stop')}</span><div><span class="dragon-eyebrow">停止训练</span><h2 id="dragon-live-stop-title">确认停止当前训练？</h2><p>训练进程会收到停止请求；已生成的日志、样张和权重会保留。</p></div><div class="dragon-live-stop-actions"><button class="dragon-btn dragon-btn-secondary" type="submit" value="cancel">继续训练</button><button class="dragon-btn dragon-btn-danger" type="button" data-live-stop-confirm>停止训练</button></div></form></dialog>`;
}

export function renderLogs(records, options = {}) {
    if (!records.length) {
        const message = options.cleared ? '当前视图已清空，新日志到达后会继续显示。' : (options.filtering ? '没有匹配的日志，请调整搜索关键词。' : '训练开始后显示实时日志。');
        return `<div class="dragon-empty-state"><p>${message}</p></div>`;
    }
    return records.slice(-300).map((record) => {
        const level = record.level || record.kind || '';
        const timestamp = formatLogTimestamp(record.ts);
        const prefix = [timestamp, level ? String(level).toUpperCase() : ''].filter(Boolean).join(' ');
        return `<div class="dragon-log-line" data-level="${escapeHtml(level)}">${prefix ? `<span class="dragon-log-prefix">${escapeHtml(prefix)}</span>` : ''}<span class="dragon-log-message">${highlightedLogHtml(logRecordText(record))}</span></div>`;
    }).join('');
}

function formatMetricHtml(value) { return escapeHtml(value).replace(/(e[+-]?\d+)$/i, '<span class="dragon-metric-exponent">$1</span>'); }
export function formatLoss(value) { return finite(value) ? Number(value).toFixed(4) : '-'; }
export function formatLr(value) { return finite(value) ? Number(value).toExponential(3) : '-'; }
export function formatPercent(value) { return finite(value) ? `${Math.round(Number(value))}%` : '-'; }
export function formatTemperature(value) { return finite(value) ? `${Math.round(Number(value))}°C` : '-'; }
export function formatVram(value, total) { if (!finite(value)) return '-'; return finite(total) ? `${Number(value).toFixed(1)} / ${Number(total).toFixed(1)} GB` : `${Number(value).toFixed(1)} GB`; }
export function formatRate(value) { return String(value || '').trim() || '-'; }

export function formatEta(model) {
    if (!isRunningState(model.state) || !finite(model.total) || Number(model.total) <= 0) return '待计算';
    const secondsPerStep = parseRateSeconds(model.rate);
    if (!finite(secondsPerStep) || Number(secondsPerStep) <= 0) return '待计算';
    const remaining = Math.max(0, Number(model.total) - Number(model.step || 0));
    return remaining ? formatDuration(Math.ceil(remaining * Number(secondsPerStep))) : '即将完成';
}

function parseRateSeconds(value) {
    const match = String(value || '').replace(/\s+/g, '').toLowerCase().match(/([\d.]+)(ms\/it|s\/it|s\/step|it\/s)/);
    if (!match) return null;
    const amount = Number(match[1]);
    if (!Number.isFinite(amount) || amount <= 0) return null;
    if (match[2] === 'it/s') return 1 / amount;
    return match[2] === 'ms/it' ? amount / 1000 : amount;
}

function formatDuration(seconds) {
    if (seconds < 60) return `${Math.max(1, seconds)} 秒`;
    const minutes = Math.ceil(seconds / 60);
    if (minutes < 60) return `约 ${minutes} 分钟`;
    const hours = Math.floor(minutes / 60);
    const remainder = minutes % 60;
    return remainder ? `约 ${hours} 小时 ${remainder} 分` : `约 ${hours} 小时`;
}

function logToolButton(action, label, icon, attributes = '', iconOnly = true) {
    return `<button class="dragon-live-log-action${action === 'clear' ? ' dragon-live-log-danger' : ''}${iconOnly ? ' dragon-live-log-icon-action' : ''}" type="button" data-live-log-action="${action}" ${attributes} aria-label="${label}" title="${label}">${renderIcon(icon, 'dragon-btn-icon')}${iconOnly ? '' : `<span>${label}</span>`}</button>`;
}

function headerTitle(model, mode) { return mode === 'running' ? `正在训练：${model.currentTask}` : (mode === 'error' ? '训练异常' : '训练监控工作台'); }
function headerMeta(model, mode) { return mode === 'running' ? `Step ${model.step} / ${model.total}` : (mode === 'error' ? model.lastActivity : '等待训练任务'); }
function temperatureDetail(current, peak) { return Number(current) >= 80 ? `高温预警 · 峰值 ${formatTemperature(peak)}` : `峰值 ${formatTemperature(peak)}`; }
function formatPeakMetric(value) { return value === '-' ? '峰值等待采样' : `峰值 ${value}`; }
function formatLogTimestamp(value) { const seconds = Number(value); return Number.isFinite(seconds) && seconds > 0 ? new Date(seconds * 1000).toLocaleTimeString('zh-CN', { hour12: false }) : ''; }
function formatEpoch(value) { return finite(value) ? String(value) : '-'; }
function finite(value) { return value !== '' && value != null && Number.isFinite(Number(value)); }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
