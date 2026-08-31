/* Presentational helpers for Dragon training history. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { escapeHtml, formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';
import {
    normalizeHistoryDetailTab,
    renderHistoryDetailTabs,
} from './history-detail-tabs.js?v=dragon-ui-20260816v2';
import { renderHistorySampleDialog } from './history-sample-dialog-view.js?v=dragon-ui-20260826v1';
import { renderHistoryPathsPanel } from './history-paths.js?v=dragon-ui-20260824v1';
import { stateText, taskDisplayName, taskGroup } from './history-model.js?v=dragon-ui-20260828v3';

export function renderHistoryDetailPage(model) {
    const { taskId, payload, images = {}, weights = {}, resume = {}, lossChart, systemCharts = '' } = model;
    const task = payload.task || {};
    const metrics = Array.isArray(payload.metrics) ? payload.metrics : [];
    const logs = Array.isArray(payload.logs) ? payload.logs : [];
    const activeTab = normalizeHistoryDetailTab(model.activeTab);
    const state = task.state || task.status || 'unknown';
    const tabCounts = historyDetailTabCounts(task, payload, images, weights);
    return `
        <div class="dragon-page dragon-page-wide dragon-history-detail-page" data-history-detail="${escapeAttribute(taskId)}" data-history-detail-active-tab="${activeTab}">
            <header class="dragon-history-detail-hero dragon-reveal">
                <div class="dragon-history-detail-toolbar">
                    ${renderHistoryBackButton(model.returnNavigation)}
                    <div class="dragon-history-detail-actions">
                        ${resumeShortcut(taskId, task, resume)}
                        <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-detail-refresh>${renderIcon('refresh', 'dragon-btn-icon')}<span>刷新详情</span></button>
                    </div>
                </div>
                <div class="dragon-history-detail-heading">
                    <div><span class="dragon-eyebrow">${escapeHtml(taskGroup(task))}</span><h1>${escapeHtml(taskDisplayName(task))}</h1><p>${escapeHtml(task.started_at_text || formatTimestamp(task.started_at) || taskId)}</p></div>
                    <span class="dragon-history-item-state" data-state="${escapeAttribute(state)}">${escapeHtml(stateText(state))}</span>
                </div>
            </header>
            ${renderHistoryDetailTabs(taskId, activeTab, tabCounts)}
            <div class="dragon-history-detail-content" data-history-detail-content>
                ${historyDetailPanel('overview', activeTab, renderHistoryOverview(taskId, task, metrics, resume))}
                ${historyDetailPanel('metrics', activeTab, renderHistoryMetrics(metrics, lossChart, systemCharts))}
                ${historyDetailPanel('artifacts', activeTab, renderPreviewSection(images, weights))}
                ${historyDetailPanel('config', activeTab, renderHistoryArtifacts(taskId, task, payload.config_toml || ''))}
                ${historyDetailPanel('logs', activeTab, renderLogsSection(taskId, logs, task, payload.limits))}
            </div>
            ${renderHistorySampleDialog()}
        </div>
    `;
}

function renderHistoryOverview(taskId, task, metrics, resume) {
    const lastLoss = lastMetricValue(metrics, 'loss');
    return `
        <section class="dragon-history-stat-grid dragon-reveal" data-stagger="1">
            ${metricTile('最终损失', formatLoss(lastLoss ?? task.final_loss))}
            ${metricTile('最后步数', lastMetricValue(metrics, 'step') ?? task.total_steps ?? '-')}
            ${metricTile('学习率', formatLr(lastMetricValue(metrics, 'lr')))}
            ${metricTile('曲线数据', `${metrics.length || task.metric_count || 0} 点`)}
        </section>
        <div class="dragon-history-detail-grid dragon-reveal" data-stagger="2">
            ${renderRunInformation(taskId, task)}
            ${renderResumePanel(task, resume)}
        </div>`;
}

function renderRunInformation(taskId, task) {
    return `<section class="dragon-history-panel">
        <div class="dragon-history-panel-head"><div><span class="dragon-eyebrow">任务上下文</span><h2>运行信息</h2></div>${linkedTaskLink(task)}</div>
        <dl class="dragon-history-detail-kv">
            ${detailRow('任务 ID', taskId)}
            ${detailRow('用户分组', task.group)}
            ${detailRow('配置组', task.history_group_label || task.history_source_config_file)}
            ${detailRow('方法', [task.methods_subdir, task.variant, task.preset].filter(Boolean).join(' / '))}
            ${detailRow('开始时间', task.started_at_text || formatTimestamp(task.started_at))}
            ${detailRow('结束时间', task.finished_at_text || formatTimestamp(task.finished_at))}
            ${detailRow('运行目录', task.run_dir)}
            ${detailRow('输出目录', task.output_dir || task.training_output_dir)}
            ${detailRow('样张目录', task.sample_dir)}
        </dl>
    </section>`;
}

function renderHistoryMetrics(metrics, lossChart, systemCharts) {
    return `<section class="dragon-history-panel dragon-reveal" data-stagger="1">
        <div class="dragon-history-panel-head dragon-history-chart-head">
            <div><span class="dragon-eyebrow">训练指标</span><h2>损失与学习率曲线</h2></div>
            <span>${metrics.length} 个数据点</span>
        </div>
        <div class="dragon-history-chart-controls" aria-label="曲线显示配置">
            <span class="dragon-history-chart-control-label">曲线</span>
            <label><input type="checkbox" data-history-chart-toggle="lossCurve" checked><i class="dragon-chart-swatch dragon-chart-swatch-loss"></i><span>Loss 曲线</span></label>
            <label><input type="checkbox" data-history-chart-toggle="lrCurve" checked><i class="dragon-chart-swatch dragon-chart-swatch-lr"></i><span>学习率曲线</span></label>
            <span class="dragon-history-chart-control-label dragon-history-chart-values-label">悬停数值</span>
            <label><input type="checkbox" data-history-chart-toggle="lrValue" checked><span>显示学习率</span></label>
            <label><input type="checkbox" data-history-chart-toggle="lossValue" checked><span>显示 Loss</span></label>
            <label class="dragon-history-smoothing"><span>Smoothing</span><input type="range" min="0" max="99" value="20" data-history-chart-smoothing aria-label="损失曲线平滑度"><output data-history-chart-smoothing-value>20%</output></label>
        </div>
        <div class="dragon-chart-container" data-history-chart-container>${lossChart || '<div class="dragon-empty-state"><p>正在加载图表…</p></div>'}</div>
    </section>
    <div data-history-system-host>${systemCharts}</div>`;
}

function historyDetailPanel(tabId, activeTab, content) {
    const selected = tabId === activeTab;
    return `<div class="dragon-history-detail-subview${selected ? ' active' : ''}" data-history-detail-panel="${tabId}" role="region" aria-labelledby="dragon-history-tab-${tabId}" ${selected ? '' : 'hidden'}>${content}</div>`;
}

function historyDetailTabCounts(task, payload, imagesPayload, weightsPayload) {
    const images = Array.isArray(imagesPayload.images) ? imagesPayload.images : [];
    const weights = Array.isArray(weightsPayload.weights) ? weightsPayload.weights : [];
    const configCount = [
        task.config_snapshot || payload.config_toml,
        task.runtime_config_file,
        task.original_config_file,
        task.dataset_config_file,
    ].filter(Boolean).length;
    return {
        metrics: Array.isArray(payload.metrics) ? payload.metrics.length : Number(task.metric_count || 0),
        artifacts: images.length + weights.length,
        config: configCount,
        logs: Number(payload.limits?.logs_total) || (Array.isArray(payload.logs) ? payload.logs.length : Number(task.log_count || 0)),
    };
}

function resumeShortcut(taskId, task, resume) {
    if (task.job !== 'training') return '';
    const count = Array.isArray(resume.checkpoints)
        ? resume.checkpoints.filter((item) => item.resume_available !== false).length
        : 0;
    if (!count) return '';
    return `<a class="dragon-btn dragon-btn-primary dragon-btn-sm" href="#history/${encodeURIComponent(taskId)}/overview" data-history-resume-shortcut>${renderIcon('activity', 'dragon-btn-icon')}<span>续训设置</span></a>`;
}

export function renderHistoryDetailError(taskId, error, returnNavigation = null) {
    return `
        <div class="dragon-page dragon-history-detail-page">
            <div class="dragon-history-detail-toolbar dragon-reveal">${renderHistoryBackButton(returnNavigation)}<button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-detail-refresh>重试</button></div>
            <div class="dragon-history-empty dragon-reveal"><span>${renderIcon('history')}</span><h1>无法加载训练记录</h1><p>${escapeHtml(error)}。请确认任务仍存在，然后重试。</p><code>${escapeHtml(taskId)}</code></div>
        </div>
    `;
}

function renderHistoryBackButton(navigation = null) {
    const label = navigation?.label || '返回历史';
    const icon = navigation?.icon || 'history';
    return `<button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-back aria-label="${escapeAttribute(label)}">${renderIcon(icon, 'dragon-btn-icon')}<span>${escapeHtml(label)}</span></button>`;
}

function renderResumePanel(task, payload = {}) {
    if (task.job !== 'training') return '';
    const checkpoints = Array.isArray(payload.checkpoints) ? payload.checkpoints : [];
    const available = checkpoints.filter((item) => item.resume_available !== false);
    const selected = payload.default_checkpoint || available[0]?.path || '';
    const body = available.length ? `
        <label class="dragon-history-resume-field"><span>训练状态目录</span><select class="dragon-select dragon-text-mono" name="history_resume_checkpoint" data-history-resume-checkpoint>${available.map((item) => resumeOption(item, selected)).join('')}</select></label>
        <div class="dragon-history-resume-actions"><button class="dragon-btn dragon-btn-primary" type="button" data-history-resume-mode="direct">${renderIcon('activity', 'dragon-btn-icon')}<span>立即续训</span></button><button class="dragon-btn dragon-btn-secondary" type="button" data-history-resume-mode="queue">${renderIcon('list', 'dragon-btn-icon')}<span>加入队列</span></button></div>
    ` : `<div class="dragon-history-inline-empty"><p>${escapeHtml(payload.error || payload.message || payload.diagnostic?.reason || '没有找到包含训练状态的可续训目录。')}</p></div>`;
    return `
        <section class="dragon-history-panel dragon-history-resume-panel">
            <div class="dragon-history-panel-head"><div><span class="dragon-eyebrow">继续训练</span><h2>从检查点恢复</h2></div><span>${available.length} 个可用项</span></div>
            <p class="dragon-history-panel-copy">恢复优化器、学习率调度器和已完成步数；启动后会生成新的历史记录。</p>
            ${body}
            <p class="dragon-config-feedback dragon-status-region" data-history-resume-status role="status" aria-live="polite"></p>
        </section>
    `;
}

function renderHistoryArtifacts(taskId, task, configToml) {
    const artifactLinks = [
        task.config_snapshot || configToml ? artifactLink(taskId, 'config-snapshot', '下载配置快照', true) : '',
        task.runtime_config_file ? artifactLink(taskId, 'runtime-config', '运行时配置') : '',
        task.original_config_file ? artifactLink(taskId, 'original-config', '原始配置') : '',
        task.dataset_config_file ? artifactLink(taskId, 'dataset-config', '数据集配置') : '',
    ].filter(Boolean).join('');
    return `
        <section class="dragon-history-panel dragon-reveal" data-stagger="4">
            <div class="dragon-history-panel-head"><div><span class="dragon-eyebrow">可复现性</span><h2>配置快照</h2></div><div class="dragon-history-artifact-actions">${artifactLinks}</div></div>
            ${renderConfigSummary(task, configToml)}
            <div class="dragon-history-snapshot-meta"><span>快照路径</span><strong class="dragon-text-mono">${escapeHtml(task.config_snapshot || '未记录')}</strong></div>
            <details class="dragon-history-config-snapshot" ${configToml ? '' : 'open'}><summary>${configToml ? '展开 TOML 内容' : 'TOML 内容不可用'}</summary><pre><code>${highlightToml(configToml || '当前历史记录没有可读取的配置快照内容。')}</code></pre></details>
        </section>
        ${renderHistoryPathsPanel(task)}
    `;
}

function renderPreviewSection(imagesPayload = {}, weightsPayload = {}) {
    const images = Array.isArray(imagesPayload.images) ? imagesPayload.images.slice(0, 8) : [];
    const weights = Array.isArray(weightsPayload.weights) ? weightsPayload.weights : [];
    return `
        <section class="dragon-history-panel dragon-reveal" data-stagger="5">
            <div class="dragon-history-panel-head"><div><span class="dragon-eyebrow">训练产物</span><h2>样张与权重</h2></div><span>${images.length} 张样图 · ${weights.length} 个权重</span></div>
            ${imagesPayload.ok === false ? inlineError(imagesPayload.error) : renderImages(images, imagesPayload.message)}
            ${weightsPayload.ok === false ? inlineError(weightsPayload.error) : renderWeights(weights, weightsPayload.message)}
        </section>
    `;
}

function renderImages(images, message) {
    if (!images.length) return `<div class="dragon-history-inline-empty"><p>${escapeHtml(message || '这个任务还没有可显示的训练样张。')}</p></div>`;
    return `<div class="dragon-history-preview-grid">${images.map((image, index) => {
        const sample = image.sample || {};
        const title = sample.step != null ? `Step ${sample.step}` : (image.name || '训练样张');
        return `<figure><button class="dragon-history-preview-open" type="button" data-history-sample-open="${index}" aria-label="查看 ${escapeAttribute(title)} 的生成参数" title="查看生成参数与提示词"><img src="${escapeAttribute(image.url || '')}" alt="${escapeAttribute(sample.prompt ? `${title}：${sample.prompt}` : title)}" width="${escapeAttribute(image.width || 1)}" height="${escapeAttribute(image.height || 1)}" loading="lazy"><span class="dragon-history-preview-step">${escapeHtml(title)}</span></button><figcaption><span>${escapeHtml(image.mtime_text || image.name || '')}</span></figcaption></figure>`;
    }).join('')}</div>`;
}

function renderWeights(weights, message) {
    if (!weights.length) return `<div class="dragon-history-inline-empty"><p>${escapeHtml(message || '这个任务还没有可下载的训练权重。')}</p></div>`;
    const fallbackFinalIndex = weights.findIndex((item) => item?.is_final === true || item?.final === true)
        < 0 ? weights.findIndex((item) => item?.steps == null && /\.safetensors$/i.test(item?.name || '')) : -1;
    return `<div class="dragon-history-weight-list">${weights.map((item, index) => {
        const path = item.abs_path || item.file || '';
        const name = item.name || '未命名权重';
        const finalTag = isFinalWeight(item, name, index, fallbackFinalIndex) ? '<span class="dragon-history-final-model">Final Model</span>' : '';
        return `<article><span class="dragon-history-weight-icon">${renderIcon('layers')}</span><div><div class="dragon-history-weight-title"><strong>${escapeHtml(name)}</strong>${finalTag}</div><span>${escapeHtml([item.steps != null ? `Step ${item.steps}` : '', item.mtime_text, item.size_bytes != null ? formatBytes(item.size_bytes) : ''].filter(Boolean).join(' · '))}</span></div><div class="dragon-history-weight-actions"><button class="dragon-icon-button" type="button" data-history-weight-copy="${escapeAttribute(path)}" aria-label="复制 ${escapeAttribute(name)} 的本地路径" title="复制本地路径">${renderIcon('copy')}<span class="visually-hidden">复制路径</span></button><a class="dragon-icon-button" href="${escapeAttribute(item.download_url || '')}" download="${escapeAttribute(item.name || 'weight.safetensors')}" aria-label="下载 ${escapeAttribute(name)}" title="下载权重">${renderIcon('download')}<span class="visually-hidden">下载</span></a></div></article>`;
    }).join('')}</div>`;
}

function isFinalWeight(item, name, index, fallbackFinalIndex) {
    return item?.is_final === true || item?.final === true || index === fallbackFinalIndex || /(?:^|[_\-.])(final|last)(?:[_\-.]|$)/i.test(name);
}

function renderConfigSummary(task, configToml) {
    const parsed = parseTomlScalars(configToml);
    const cards = [
        ['Base Model', firstValue(task.model_family, parsed.model_family, parsed.pretrained_model_name_or_path, parsed.dit)],
        ['Batch Size', firstValue(task.batch_size, parsed.train_batch_size, parsed.batch_size)],
        ['Learning Rate', firstValue(task.learning_rate, parsed.learning_rate, parsed.unet_lr)],
        ['Dim / Alpha', joinPair(firstValue(parsed.network_dim, parsed.dim), firstValue(parsed.network_alpha, parsed.alpha))],
        ['Optimizer', firstValue(task.optimizer_type, parsed.optimizer_type, parsed.optimizer)],
    ];
    return `<section class="dragon-history-config-summary" aria-label="关键超参数">${cards.map(([label, value]) => `<div><span>${label}</span><strong>${formatMetricHtml(value || '未记录')}</strong></div>`).join('')}</section>`;
}

function parseTomlScalars(source) {
    const result = {};
    String(source || '').split(/\r?\n/).forEach((line) => {
        const match = line.match(/^\s*([A-Za-z0-9_.-]+)\s*=\s*(.+?)\s*(?:#.*)?$/);
        if (!match) return;
        result[match[1].split('.').at(-1)] = match[2].replace(/^(["'])(.*)\1$/, '$2');
    });
    return result;
}

function highlightToml(source) {
    return String(source || '').split(/\r?\n/).map((line) => {
        const section = line.match(/^(\s*)(\[+[^\]]+\]+)(\s*(?:#.*)?)$/);
        if (section) return `${escapeHtml(section[1])}<span class="toml-section">${escapeHtml(section[2])}</span><span class="toml-comment">${escapeHtml(section[3])}</span>`;
        const assignment = line.match(/^(\s*)([A-Za-z0-9_.-]+)(\s*=\s*)(.*)$/);
        if (!assignment) return `<span class="toml-comment">${escapeHtml(line)}</span>`;
        const valueParts = assignment[4].match(/^(.*?)(\s+#.*)?$/) || [];
        return `${escapeHtml(assignment[1])}<span class="toml-key">${escapeHtml(assignment[2])}</span>${escapeHtml(assignment[3])}<span class="toml-value">${escapeHtml(valueParts[1] || '')}</span><span class="toml-comment">${escapeHtml(valueParts[2] || '')}</span>`;
    }).join('\n');
}

function formatMetricHtml(value) {
    return escapeHtml(value).replace(/(e[+-]?\d+)$/i, '<span class="dragon-metric-exponent">$1</span>');
}

function joinPair(left, right) { return [left, right].filter((value) => value != null && value !== '').join(' / '); }
function firstValue(...values) { return values.find((value) => value != null && String(value).trim() !== ''); }

function renderLogsSection(taskId, logs, task, limits = {}) {
    const returned = logs.length;
    const reportedTotal = Math.max(returned, Number(limits?.logs_total) || 0);
    const countLabel = limits?.logs_paged
        ? `${reportedTotal} 行`
        : (reportedTotal > returned ? `最近 ${returned} / ${reportedTotal} 行` : `${returned} 行`);
    const initialOffset = Math.max(0, Number(limits?.logs_offset) || (reportedTotal - returned));
    return `
        <section class="dragon-history-panel dragon-history-log-workspace dragon-reveal" data-stagger="6">
            <div class="dragon-history-panel-head"><div><span class="dragon-eyebrow">运行输出</span><h2>完整日志</h2></div><div class="dragon-history-log-head-actions"><span>${countLabel}</span><a class="dragon-btn dragon-btn-secondary dragon-btn-sm" href="/api/training/history/${encodeURIComponent(taskId)}/logs/download" download>${renderIcon('download', 'dragon-btn-icon')}<span>下载日志</span></a></div></div>
            ${task.message ? `<p class="dragon-history-task-message">${escapeHtml(task.message)}</p>` : ''}
            <div class="dragon-history-log-search" role="search" aria-label="搜索历史日志">
                <label class="dragon-history-log-search-field">
                    <span class="visually-hidden">搜索历史日志</span>
                    ${renderIcon('search', 'dragon-history-log-search-icon')}
                    <input class="dragon-input" type="search" autocomplete="off" placeholder="搜索日志…" data-history-log-search>
                </label>
                <span class="dragon-history-log-search-status" data-history-log-search-status role="status" aria-live="polite">0 个匹配</span>
                <div class="dragon-history-log-search-actions">
                    <button class="dragon-icon-button" type="button" data-history-log-search-previous aria-label="上一个匹配项" title="上一个匹配项" disabled>${renderIcon('chevronUp')}</button>
                    <button class="dragon-icon-button" type="button" data-history-log-search-next aria-label="下一个匹配项" title="下一个匹配项" disabled>${renderIcon('chevronDown')}</button>
                </div>
            </div>
            <div class="dragon-history-log-window-meta"><span data-history-log-window-status>${reportedTotal ? `第 ${initialOffset + 1}–${Math.min(reportedTotal, initialOffset + returned)} / ${reportedTotal} 行` : '0 行'}</span></div>
            <div class="dragon-log-panel dragon-history-log-panel" data-history-log-viewer tabindex="0" role="list" aria-label="历史训练日志">${reportedTotal ? '' : '<div class="dragon-history-inline-empty"><p>暂无日志。</p></div>'}</div>
        </section>
    `;
}

function resumeOption(item, selected) {
    const label = [item.name || basename(item.path), item.step != null ? `Step ${item.step}` : '', item.mtime_text || ''].filter(Boolean).join(' · ');
    return `<option value="${escapeAttribute(item.path || '')}" ${item.path === selected ? 'selected' : ''}>${escapeHtml(label || item.path || '训练状态目录')}</option>`;
}

function linkedTaskLink(task) {
    const linked = task.linked_preprocess_task;
    return linked?.id ? `<a class="dragon-btn dragon-btn-ghost dragon-btn-sm" href="#history/${encodeURIComponent(linked.id)}">查看关联预处理</a>` : '';
}

function artifactLink(taskId, key, label, download = false) {
    const suffix = download ? '?download=1' : '';
    return `<a class="dragon-btn dragon-btn-secondary dragon-btn-sm" href="/api/training/history/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(key)}${suffix}" ${download ? 'download' : 'target="_blank" rel="noopener"'}>${escapeHtml(label)}</a>`;
}

function detailRow(label, value) {
    if (value == null || value === '') return '';
    return `<div class="dragon-history-detail-row"><dt>${escapeHtml(label)}</dt><dd class="dragon-text-mono">${escapeHtml(value)}</dd></div>`;
}

function metricTile(label, value) {
    return `<article><span>${escapeHtml(label)}</span><strong>${formatMetricHtml(value ?? '-')}</strong></article>`;
}

function inlineError(message) {
    return `<div class="dragon-history-inline-error" role="alert"><p>${escapeHtml(message || '读取训练产物失败')}。请刷新详情后重试。</p></div>`;
}

function lastMetricValue(metrics, key) {
    for (let index = metrics.length - 1; index >= 0; index -= 1) {
        const value = metrics[index]?.[key];
        if (value != null && value !== '') return value;
    }
    return null;
}

function formatLoss(value) {
    if (value == null || value === '') return '-';
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(5) : '-';
}

function formatLr(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toExponential(3) : '-';
}

function formatTimestamp(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) return '';
    return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'medium' }).format(new Date(numeric * 1000));
}

function basename(value) { return String(value || '').replace(/\\/g, '/').split('/').filter(Boolean).pop() || ''; }
function escapeAttribute(value) { return escapeHtml(value); }
