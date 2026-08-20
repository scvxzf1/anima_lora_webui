/* Presentational helpers for the dedicated live training workspace. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { renderStatusRegion, renderToolButton, renderToolHero } from './tool-page.js?v=dragon-ui-20260814v43';
import { isRunningState, logRecordText, stateText, visualState } from './live-training-state.js?v=dragon-ui-20260814v43';

export function renderLiveTrainingPage(model, renderLossChart) {
    const running = isRunningState(model.state);
    const badge = `
        <div class="dragon-status-badge" data-live-state data-state="${visualState(model.state)}">
            <span class="dragon-nav-status-dot" data-state="${visualState(model.state)}"></span>
            <span data-live-state-text>${stateText(model.state)}</span>
        </div>
    `;
    const actions = [
        renderToolButton('list', '查看队列', 'queue'),
        renderToolButton('stop', '停止训练', 'stop', 'dragon-btn-secondary dragon-btn-danger', running ? '' : 'hidden disabled'),
    ].join('');
    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-live-training-page" data-live-training-root>
            ${renderToolHero({
                eyebrow: '训练监控',
                title: '当前监控',
                description: '专注当前任务：检查训练上下文、关键指标、损失趋势和完整实时日志。',
                badge,
                actions,
            })}

            ${renderStatusRegion('data-live-feedback')}

            <div class="dragon-live-connection" data-live-connection data-tone="info" role="status" aria-live="polite">
                <span class="dragon-live-connection-dot" aria-hidden="true"></span>
                <div><strong data-live-connection-label>正在连接实时通道…</strong><span data-live-connection-detail>状态接口可用，正在建立日志和指标推送连接。</span></div>
            </div>

            <section class="dragon-live-context dragon-reveal" data-stagger="1" aria-label="当前训练上下文">
                ${contextItem('activity', '当前任务', model.currentTask, 'task')}
                ${contextItem('layers', '训练配置', model.configLabel, 'config')}
                ${contextItem('folder', '运行目录', model.runDir, 'run-dir')}
                ${contextItem('terminal', '最新活动', model.lastActivity, 'activity')}
            </section>

            <section class="dragon-tool-panel dragon-live-progress-panel dragon-reveal" data-stagger="2">
                <div class="dragon-tool-panel-head">
                    <div><span class="dragon-eyebrow">当前进度</span><h2>训练步数</h2></div>
                    <strong data-live-progress-text>${model.total > 0 ? `${model.progressPct.toFixed(1)}%` : '等待训练'}</strong>
                </div>
                <div class="dragon-progress-bar" role="progressbar" aria-label="训练进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(model.progressPct)}" data-live-progress>
                    <div class="dragon-progress-bar-fill" data-live-progress-fill style="width: ${model.progressPct}%"></div>
                </div>
                <div class="dragon-progress-bar-meta">
                    <span data-live-step-text>${model.step} / ${model.total} 步</span>
                    <span data-live-epoch-text>第 ${formatEpoch(model.epoch)} 轮</span>
                </div>
            </section>

            <section class="dragon-stat-grid dragon-stat-grid-compact dragon-reveal" data-stagger="3" aria-label="训练关键指标">
                ${metricTile('损失值', formatLoss(model.loss), 'live-loss')}
                ${metricTile('学习率', formatLr(model.lr), 'live-lr')}
                ${metricTile('训练速度', formatRate(model.rate), 'live-rate')}
                ${metricTile('预计剩余（ETA）', formatEta(model), 'live-eta')}
                ${metricTile('当前显存', formatVram(model.vram, model.vramTotal), 'live-vram')}
                ${metricTile('峰值显存', formatVram(model.peakVram, model.vramTotal), 'live-vram-peak')}
                ${metricTile('GPU 利用率', formatPercent(model.gpuUtil), 'live-gpu-util', formatPeak('峰值', formatPercent(model.peakGpuUtil)))}
                ${metricTile('GPU 温度', formatTemperature(model.gpuTemp), 'live-gpu-temp', formatPeak('峰值', formatTemperature(model.peakGpuTemp)))}
            </section>

            <div class="dragon-tool-split dragon-reveal" data-stagger="4">
                <section class="dragon-tool-panel">
                    <div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">趋势</span><h2>损失曲线</h2></div><span class="dragon-tool-note" data-live-chart-count>最近 ${model.metrics.length} 条记录</span></div>
                    <div class="dragon-chart-container" data-live-chart>${renderLossChart(model.metrics)}</div>
                </section>
                <section class="dragon-tool-panel dragon-live-log-panel">
                    <div class="dragon-tool-panel-head"><div><span class="dragon-eyebrow">输出</span><h2>实时日志</h2></div><span class="dragon-tool-note" data-live-log-count>${model.logs.length} 条</span></div>
                    <div class="dragon-live-log-toolbar" role="toolbar" aria-label="实时日志工具">
                        <label class="dragon-live-log-search"><span class="visually-hidden">搜索实时日志</span><input class="dragon-input" type="search" name="live_log_search" autocomplete="off" placeholder="搜索当前日志…" data-live-log-search></label>
                        <div class="dragon-live-log-actions">
                            ${logToolButton('copy', '复制', 'copy')}
                            ${logToolButton('download', '下载', 'download')}
                            ${logToolButton('pause', '暂停滚动', 'stop', 'aria-pressed="false"')}
                            ${logToolButton('clear', '清空视图', 'trash')}
                        </div>
                    </div>
                    <div class="dragon-live-log-meta"><span data-live-log-visible-count>${model.logs.length} 条可见</span><span>清空只影响当前浏览器视图</span></div>
                    <p class="dragon-live-log-feedback" data-live-log-feedback role="status" aria-live="polite"></p>
                    <div class="dragon-log-stream dragon-live-log-stream" data-live-log tabindex="0" aria-label="实时训练日志" aria-live="off">${renderLogs(model.logs)}</div>
                </section>
            </div>
        </div>
    `;
}

function contextItem(icon, label, value, key) {
    return `<div class="dragon-live-context-item"><span class="dragon-live-context-icon">${renderIcon(icon)}</span><div><span>${label}</span><strong data-live-context="${key}">${escapeHtml(value)}</strong></div></div>`;
}

function metricTile(label, value, key, detail = '') {
    return `<div class="dragon-stat-tile" data-live-metric="${key}"><span>${label}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></div>`;
}

export function renderLogs(records, options = {}) {
    if (!records.length) {
        const message = options.cleared
            ? '当前视图已清空，新日志到达后会继续显示。'
            : (options.filtering ? '没有匹配的日志，请调整搜索关键词。' : '训练开始后显示实时日志。');
        return `<div class="dragon-empty-state"><p>${message}</p></div>`;
    }
    return records.slice(-300).map((record) => {
        const level = record.level || record.kind || '';
        return `<div class="dragon-log-line" data-level="${escapeHtml(level)}">${escapeHtml(logRecordText(record))}</div>`;
    }).join('');
}

export function formatLoss(value) { return finite(value) ? Number(value).toFixed(4) : '-'; }
export function formatLr(value) { return finite(value) ? Number(value).toExponential(3) : '-'; }
export function formatPercent(value) { return finite(value) ? `${Math.round(Number(value))}%` : '-'; }
export function formatTemperature(value) { return finite(value) ? `${Math.round(Number(value))}°C` : '-'; }
export function formatVram(value, total) {
    if (!finite(value)) return '-';
    return finite(total) ? `${Number(value).toFixed(1)} / ${Number(total).toFixed(1)} GB` : `${Number(value).toFixed(1)} GB`;
}

export function formatRate(value) {
    const text = String(value || '').trim();
    return text || '-';
}

export function formatEta(model) {
    if (!isRunningState(model.state) || !finite(model.total) || Number(model.total) <= 0) return '待计算';
    const secondsPerStep = parseRateSeconds(model.rate);
    if (!finite(secondsPerStep) || Number(secondsPerStep) <= 0) return '待计算';
    const remaining = Math.max(0, Number(model.total) - Number(model.step || 0));
    if (!remaining) return '即将完成';
    return formatDuration(Math.ceil(remaining * Number(secondsPerStep)));
}

function parseRateSeconds(value) {
    const match = String(value || '').replace(/\s+/g, '').toLowerCase().match(/([\d.]+)(ms\/it|s\/it|s\/step|it\/s)/);
    if (!match) return null;
    const amount = Number(match[1]);
    if (!Number.isFinite(amount) || amount <= 0) return null;
    if (match[2] === 'it/s') return 1 / amount;
    if (match[2] === 'ms/it') return amount / 1000;
    return amount;
}

function formatDuration(seconds) {
    if (seconds < 60) return `${Math.max(1, seconds)} 秒`;
    const minutes = Math.ceil(seconds / 60);
    if (minutes < 60) return `约 ${minutes} 分钟`;
    const hours = Math.floor(minutes / 60);
    const remainder = minutes % 60;
    return remainder ? `约 ${hours} 小时 ${remainder} 分` : `约 ${hours} 小时`;
}

function formatEpoch(value) { return finite(value) ? String(value) : '-'; }
function formatPeak(label, value) { return value === '-' ? '峰值等待采样' : `${label} ${value}`; }

function logToolButton(action, label, icon, attributes = '') {
    return `<button class="dragon-live-log-action${action === 'clear' ? ' dragon-live-log-danger' : ''}" type="button" data-live-log-action="${action}" ${attributes} aria-label="${label}">${renderIcon(icon, 'dragon-btn-icon')}<span>${label}</span></button>`;
}

function finite(value) { return value !== '' && value != null && Number.isFinite(Number(value)); }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
