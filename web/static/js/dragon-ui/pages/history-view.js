/* Presentational helpers for Dragon training history. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260816v36';
import { escapeHtml, formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';
import {
    normalizeHistoryDetailTab,
    renderHistoryDetailTabs,
} from './history-detail-tabs.js?v=dragon-ui-20260816v2';
import { renderHistorySampleDialog } from './history-sample-dialog.js?v=dragon-ui-20260819v2';

export function renderHistoryPage(model = {}) {
    const filters = normalizeHistoryFilters(model.filters);
    const result = renderHistoryResults(model.tasks || [], filters);
    const error = model.error || '';
    return `
        <div class="dragon-page dragon-page-wide dragon-history-page" data-history-page>
            <div class="dragon-history-layout">
                <aside class="dragon-history-sidebar">
                    <header class="dragon-history-hero dragon-reveal">
                        <div class="dragon-history-hero-copy">
                            <span class="dragon-eyebrow">HISTORY FORGE</span>
                            <h1>历史任务</h1>
                            <p>按名称、配置分组和状态定位训练记录。</p>
                        </div>
                        <div class="dragon-history-hero-side">
                            <span class="dragon-history-count" data-history-count>${result.visibleCount} / ${(model.tasks || []).length} 条记录</span>
                            <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-refresh>${renderIcon('refresh', 'dragon-btn-icon')}<span>刷新</span></button>
                        </div>
                        <div class="dragon-history-sidebar-actions">
                            <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-classic>合并查看</button>
                        </div>
                    </header>
                    <p class="dragon-history-summary" data-history-summary>${renderHistorySummary(model.tasks || [], filters)}</p>
                    <p class="dragon-config-feedback dragon-status-region${error ? ' dragon-config-feedback-visible' : ''}" data-history-status-region data-tone="${error ? 'error' : ''}" role="status" aria-live="polite">${escapeHtml(error ? `${error}。请检查 WebUI 服务后重试。` : '')}</p>
                    <div class="dragon-history-stats" aria-label="历史任务统计">${renderHistoryStats(model.tasks || [], filters)}</div>
                    <section class="dragon-history-controls dragon-reveal" data-stagger="1" aria-label="训练历史筛选">
                        ${historyFilterControls(filters, model.tasks || [])}
                    </section>
                </aside>
                <main class="dragon-history-content">
                    <div data-history-results aria-live="polite">${model.resultsHtml ?? result.html}</div>
                </main>
            </div>
        </div>
    `;
}

export function renderHistoryResults(tasks = [], queryOrFilters = '', status = 'all') {
    const filters = normalizeHistoryFilters(
        queryOrFilters && typeof queryOrFilters === 'object'
            ? queryOrFilters
            : { search: queryOrFilters, status },
    );
    const visible = filterHistoryTasks(tasks, filters);
    if (!visible.length) {
        const hasFilters = hasActiveHistoryFilters(filters);
        return {
            visibleCount: 0,
            html: `<div class="dragon-history-empty"><span>${renderIcon(hasFilters ? 'eye' : 'history')}</span><h2>${hasFilters ? '没有匹配的记录' : '暂无训练记录'}</h2><p>${hasFilters ? '调整搜索关键词或任务状态后再试。' : '完成或中断的训练任务会显示在这里。'}</p></div>`,
        };
    }

    const groups = groupHistoryTasks(visible);
    const html = [...groups.entries()].map(([group, items], index) => `
        <section class="dragon-history-group dragon-reveal" data-stagger="${Math.min(index + 2, 6)}">
            <div class="dragon-history-group-head"><div><span class="dragon-eyebrow">配置分组</span><h2>${escapeHtml(group)}</h2></div><span>${items.length} 个任务</span></div>
            <ul class="dragon-history-list">${items.map(renderHistoryItem).join('')}</ul>
        </section>
    `).join('');
    return { visibleCount: visible.length, html };
}

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
                    <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-back>${renderIcon('history', 'dragon-btn-icon')}<span>返回历史</span></button>
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
        </div>
        <div class="dragon-chart-container" data-history-chart-container>${lossChart}</div>
    </section>
    ${systemCharts}`;
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
        logs: Array.isArray(payload.logs) ? payload.logs.length : Number(task.log_count || 0),
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

export function renderHistoryDetailError(taskId, error) {
    return `
        <div class="dragon-page dragon-history-detail-page">
            <div class="dragon-history-detail-toolbar dragon-reveal"><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-back>返回历史</button><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-detail-refresh>重试</button></div>
            <div class="dragon-history-empty dragon-reveal"><span>${renderIcon('history')}</span><h1>无法加载训练记录</h1><p>${escapeHtml(error)}。请确认任务仍存在，然后重试。</p><code>${escapeHtml(taskId)}</code></div>
        </div>
    `;
}

function renderHistoryItem(task) {
    const state = task.state || task.status || 'unknown';
    const time = task.started_at_text || formatTimestamp(task.started_at) || task.created_at_text || '';
    const loss = task.final_loss != null ? formatLoss(task.final_loss) : '';
    const steps = task.total_steps ?? task.steps ?? '';
    const method = [task.variant, task.preset].filter(Boolean).join(' / ');
    return `
        <li><a class="dragon-history-item" href="#history/${encodeURIComponent(task.id || '')}">
            <span class="dragon-history-item-main"><strong class="dragon-history-item-name">${escapeHtml(taskDisplayName(task))}</strong><span class="dragon-history-item-sub">${escapeHtml([task.job === 'preprocess' ? '预处理' : '训练', method].filter(Boolean).join(' · '))}</span></span>
            <span class="dragon-history-item-numbers">${loss ? `<span>Loss <strong>${loss}</strong></span>` : ''}${steps !== '' ? `<span><strong>${escapeHtml(steps)}</strong> 步</span>` : ''}</span>
            <span class="dragon-history-item-state" data-state="${escapeAttribute(state)}">${escapeHtml(stateText(state))}</span>
            <time class="dragon-history-item-meta">${escapeHtml(time || '时间未记录')}</time>
        </a></li>
    `;
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
            <div class="dragon-history-snapshot-meta"><span>快照路径</span><strong class="dragon-text-mono">${escapeHtml(task.config_snapshot || '未记录')}</strong></div>
            <details class="dragon-history-config-snapshot" ${configToml ? '' : 'open'}><summary>${configToml ? '展开 TOML 内容' : 'TOML 内容不可用'}</summary><pre><code>${escapeHtml(configToml || '当前历史记录没有可读取的配置快照内容。')}</code></pre></details>
        </section>
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
        return `<figure><button class="dragon-history-preview-open" type="button" data-history-sample-open="${index}" aria-label="查看 ${escapeAttribute(title)} 的生成参数" title="查看生成参数与提示词"><img src="${escapeAttribute(image.url || '')}" alt="${escapeAttribute(sample.prompt ? `${title}：${sample.prompt}` : title)}" width="${escapeAttribute(image.width || 1)}" height="${escapeAttribute(image.height || 1)}" loading="lazy"><span class="dragon-history-preview-open-label">${renderIcon('zap')}<span>参数</span></span></button><figcaption><strong>${escapeHtml(title)}</strong><span>${escapeHtml(image.mtime_text || image.name || '')}</span></figcaption></figure>`;
    }).join('')}</div>`;
}

function renderWeights(weights, message) {
    if (!weights.length) return `<div class="dragon-history-inline-empty"><p>${escapeHtml(message || '这个任务还没有可下载的训练权重。')}</p></div>`;
    return `<div class="dragon-history-weight-list">${weights.map((item) => {
        const path = item.abs_path || item.file || '';
        const name = item.name || '未命名权重';
        return `<article><span class="dragon-history-weight-icon">${renderIcon('layers')}</span><div><strong>${escapeHtml(name)}</strong><span>${escapeHtml([item.steps != null ? `Step ${item.steps}` : '', item.mtime_text, item.size_bytes != null ? formatBytes(item.size_bytes) : ''].filter(Boolean).join(' · '))}</span></div><div class="dragon-history-weight-actions"><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-weight-copy="${escapeAttribute(path)}" aria-label="复制 ${escapeAttribute(name)} 的本地路径" title="复制本地路径">${renderIcon('copy', 'dragon-btn-icon')}<span>复制路径</span></button><a class="dragon-btn dragon-btn-secondary dragon-btn-sm" href="${escapeAttribute(item.download_url || '')}" download="${escapeAttribute(item.name || 'weight.safetensors')}">${renderIcon('download', 'dragon-btn-icon')}<span>下载</span></a></div></article>`;
    }).join('')}</div>`;
}

function renderLogsSection(taskId, logs, task, limits = {}) {
    const returned = logs.length;
    const reportedTotal = Math.max(returned, Number(limits?.logs_total) || 0);
    const countLabel = reportedTotal > returned ? `最近 ${returned} / ${reportedTotal} 行` : `${returned} 行`;
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
            <div class="dragon-history-log-window-meta"><span data-history-log-window-status>${returned ? `第 ${Math.max(1, reportedTotal - returned + 1)}–${reportedTotal} / ${reportedTotal} 行` : '0 行'}</span></div>
            <div class="dragon-log-panel dragon-history-log-panel" data-history-log-viewer tabindex="0" role="list" aria-label="历史训练日志">${returned ? '' : '<div class="dragon-history-inline-empty"><p>暂无日志。</p></div>'}</div>
        </section>
    `;
}

export function taskDisplayName(task = {}) {
    return String(task.name || task.output_name || task.task_name || task.history_run_label || task.id || '未命名任务').trim();
}

export function stateText(state) {
    const key = String(state || 'unknown').toLowerCase();
    return {
        idle: '已完成', completed: '已完成', done: '已完成',
        running: '训练中', training: '训练中', compiling: '编译中', queued: '排队中',
        interrupted: '已中断', stopped: '已停止', canceled: '已取消', cancelled: '已取消',
        error: '失败', failed: '失败', unknown: '未知',
    }[key] || state;
}

export function filterHistoryTasks(tasks, filters = {}) {
    const normalized = normalizeHistoryFilters(filters);
    const search = parseHistorySearch(normalized.search);
    return tasks.filter((task) => {
        if (normalized.kind !== 'all' && task.job !== normalized.kind) return false;
        if (normalized.status !== 'all' && !historyStatusMatches(task, normalized.status)) return false;
        const archived = Boolean(task.archived);
        if (normalized.archived === 'active' && archived) return false;
        if (normalized.archived === 'archived' && !archived) return false;
        if (!historySourceMatches(task, normalized.source)) return false;
        if (!historyChipMatches(task, normalized)) return false;
        if (search.global) {
            const haystack = historySearchText(task);
            if (!haystack.includes(search.global)) return false;
        }
        if (search.collection && ![task.group, task.collection, task.history_collection].filter(Boolean).join(' ').toLocaleLowerCase().includes(search.collection)) return false;
        if (search.config && ![task.history_group_label, task.history_source_config_file, task.config_group, task.variant, task.preset].filter(Boolean).join(' ').toLocaleLowerCase().includes(search.config)) return false;
        return true;
    }).sort(historyTaskComparator(normalized.sort));
}

function parseHistorySearch(value) {
    const raw = String(value || '').trim().toLocaleLowerCase();
    if (!raw) return { global: '', collection: '', config: '' };
    const match = raw.match(/^([^:：]+)\s*[:：]\s*(.*)$/);
    if (!match || !match[2].trim()) return { global: raw, collection: '', config: '' };
    const prefix = match[1].trim();
    const query = match[2].trim();
    if (['组', '集合', 'group', 'collection'].includes(prefix)) return { global: '', collection: query, config: '' };
    if (['配置', '配置组', 'config'].includes(prefix)) return { global: '', collection: '', config: query };
    return { global: raw, collection: '', config: '' };
}

function historySearchText(task) {
    return [taskDisplayName(task), task.id, task.group, task.history_group_label, task.history_source_config_file, task.variant, task.training_variant, task.preset, task.methods_subdir, task.run_dir, task.output_dir, task.message, stateText(task.state || task.status)].filter(Boolean).join(' ').toLocaleLowerCase();
}

function normalizeHistoryFilters(filters = {}) {
    return {
        search: String(filters.search ?? '').trim(), kind: filters.kind || 'all', status: filters.status || 'all',
        archived: filters.archived || 'active', source: filters.source || 'all',
        trainingVariant: filters.trainingVariant || 'all', preprocessPrecision: filters.preprocessPrecision || 'all',
        blockSwapPrecision: filters.blockSwapPrecision || 'all', baseCompute: filters.baseCompute || 'all',
        precisionPreference: filters.precisionPreference || 'all', sort: filters.sort || 'newest',
    };
}

function hasActiveHistoryFilters(filters) {
    return Object.entries(normalizeHistoryFilters(filters)).some(([key, value]) => key === 'search' ? Boolean(value) : value !== HISTORY_FILTER_DEFAULTS[key]);
}

function historyStatusMatches(task, status) {
    const state = stateCategory(task.state || task.status);
    return status === 'error' ? ['error', 'interrupted'].includes(state) : state === status;
}

function historySourceMatches(task, source) {
    if (source === 'queue') return Boolean(task.from_queue || task.queue_item_id);
    if (source === 'resume') return Boolean(task.resume_from?.source_task_id || task.resume_source_task_id);
    if (source === 'continue') return task.training_mode === 'continue_lora';
    return source === 'all';
}

function historyChipMatches(task, filters) {
    const values = [
        ['trainingVariant', task.training_variant || task.variant],
        ['preprocessPrecision', task.preprocess_precision],
        ['blockSwapPrecision', task.block_swap_precision],
        ['baseCompute', task.base_compute],
        ['precisionPreference', task.precision_preference],
    ];
    return values.every(([key, value]) => filters[key] === 'all' || String(value || '').trim().toLowerCase() === String(filters[key]).toLowerCase());
}

function historyTaskComparator(sort) {
    return (a, b) => {
        if (sort === 'oldest' || sort === 'newest') {
            const delta = Number(a.started_at || a.updated_at || 0) - Number(b.started_at || b.updated_at || 0);
            return sort === 'oldest' ? delta : -delta;
        }
        if (sort === 'loss') return Number(b.metric_count || 0) - Number(a.metric_count || 0);
        if (sort === 'logs') return Number(b.log_count || 0) - Number(a.log_count || 0);
        return taskDisplayName(a).localeCompare(taskDisplayName(b), 'zh-CN');
    };
}

function stateCategory(state) {
    const key = String(state || '').toLowerCase();
    if (['idle', 'completed', 'done'].includes(key)) return 'completed';
    if (['running', 'training', 'compiling'].includes(key)) return 'running';
    if (['interrupted', 'stopped'].includes(key)) return 'interrupted';
    if (['canceled', 'cancelled'].includes(key)) return 'canceled';
    if (['error', 'failed'].includes(key)) return 'error';
    if (key === 'queued') return 'queued';
    return 'unknown';
}

function groupHistoryTasks(tasks) {
    const groups = new Map();
    for (const task of tasks) {
        const group = taskGroup(task);
        if (!groups.has(group)) groups.set(group, []);
        groups.get(group).push(task);
    }
    return groups;
}

function taskGroup(task = {}) {
    return String(task.group || task.history_group_label || task.config_group || task.variant || '未分组').trim();
}

function statusOptions() {
    return [['all', '全部状态'], ['running', '运行中 / 编译中'], ['queued', '排队中'], ['completed', '已完成'], ['interrupted', '已中断'], ['canceled', '已取消'], ['error', '失败'], ['unknown', '其他状态']].map(([value, label]) => `<option value="${value}">${label}</option>`).join('');
}

const HISTORY_FILTER_DEFAULTS = Object.freeze({
    search: '', kind: 'all', status: 'all', archived: 'active', source: 'all',
    trainingVariant: 'all', preprocessPrecision: 'all', blockSwapPrecision: 'all',
    baseCompute: 'all', precisionPreference: 'all', sort: 'newest',
});

export function renderHistoryStats(tasks, filters) {
    const counts = {
        all: tasks.length,
        training: tasks.filter((task) => task.job === 'training').length,
        preprocess: tasks.filter((task) => task.job === 'preprocess').length,
        error: tasks.filter((task) => ['error', 'interrupted'].includes(stateCategory(task.state || task.status))).length,
        archived: tasks.filter((task) => Boolean(task.archived)).length,
        queue: tasks.filter((task) => Boolean(task.from_queue || task.queue_item_id)).length,
    };
    const labels = [['all', '全部'], ['training', '训练'], ['preprocess', '预处理'], ['error', '异常/中断'], ['archived', '归档'], ['queue', '来自队列']];
    return labels.map(([key, label]) => `<button type="button" class="dragon-history-stat${historyStatActive(filters, key) ? ' active' : ''}${key === 'error' ? ' error' : ''}" data-history-stat="${key}"><strong>${counts[key]}</strong><span>${label}</span></button>`).join('');
}

export function renderHistorySummary(tasks, filters) {
    const all = tasks || [];
    const visible = filterHistoryTasks(all, filters).length;
    const archived = all.filter((task) => Boolean(task.archived)).length;
    return `共 ${all.length} 条记录 · 当前筛选 ${visible} 条 · 归档 ${archived} 条`;
}

function historyStatActive(filters, key) {
    const current = normalizeHistoryFilters(filters);
    const base = Object.entries(current).every(([name, value]) => ['kind', 'status', 'archived', 'source'].includes(name) || value === HISTORY_FILTER_DEFAULTS[name]);
    if (!base) return false;
    if (key === 'all') return current.kind === 'all' && current.status === 'all' && current.archived === 'active' && current.source === 'all';
    if (key === 'training' || key === 'preprocess') return current.kind === key && current.status === 'all' && current.archived === 'active' && current.source === 'all';
    if (key === 'error') return current.status === 'error' && current.kind === 'all' && current.archived === 'active' && current.source === 'all';
    if (key === 'archived') return current.archived === 'archived' && current.kind === 'all' && current.status === 'all' && current.source === 'all';
    if (key === 'queue') return current.source === 'queue' && current.kind === 'all' && current.status === 'all' && current.archived === 'active';
    return false;
}

function historyFilterControls(filters, tasks = []) {
    const select = (key, label, options) => `<label><span>${label}</span><select class="dragon-select" data-history-filter="${key}"${key === 'status' ? ' data-history-status' : ''} name="history_${key}">${options.map(([value, text]) => {
        const disabled = !historyFilterOptionAvailable(tasks, key, value);
        return `<option value="${value}"${filters[key] === value ? ' selected' : ''}${disabled ? ' disabled' : ''}>${text}</option>`;
    }).join('')}</select></label>`;
    const all = [['all', '全部']];
    return [
        `<label class="dragon-history-search"><span>全局搜索</span><input class="dragon-input" type="search" name="history_search" autocomplete="off" data-history-filter="search" placeholder="任务 / 配置 / 目录，支持 组:关键词 或 配置:lora" value="${escapeAttribute(filters.search)}"></label>`,
        select('kind', '类型', [...all, ['training', '训练'], ['preprocess', '预处理']]),
        select('status', '状态', [['all', '全部'], ['running', '运行中'], ['completed', '已完成'], ['queued', '排队中'], ['error', '异常/失败'], ['interrupted', '已中断'], ['canceled', '已取消'], ['unknown', '其他']]), /* name="history_status" */
        select('archived', '归档', [['active', '未归档'], ['all', '全部'], ['archived', '已归档']]),
        select('source', '来源', [['all', '全部'], ['queue', '来自队列'], ['resume', '续训'], ['continue', '权重热启动']]),
        select('trainingVariant', '训练变体', [...all, ...['lora', 'lokr', 'loha', 'vera', 'glora', 'dora', 'hydralora', 'reft', 'tlora', 'ortholora', 'chimera', 'soft_tokens', 'ip_adapter', 'easycontrol'].map((value) => [value, value])]),
        select('preprocessPrecision', '预处理精度', [...all, ['bf16', 'bf16'], ['fp16', 'fp16'], ['fp32', 'fp32']]),
        select('blockSwapPrecision', '块交换精度', [...all, ['bf16', 'bf16'], ['fp8_e4m3', 'fp8_e4m3']]),
        select('baseCompute', '底模计算路径', [...all, ['bf16', 'bf16'], ['nf4', 'nf4'], ['w8a16_convrot', 'w8a16_convrot'], ['w8a8_convrot', 'w8a8_convrot']]),
        select('precisionPreference', '精度倾向', [...all, ['bf16', 'bf16'], ['fp16', 'fp16'], ['fp32', 'fp32']]),
        select('sort', '排序', [['newest', '最新优先'], ['oldest', '最早优先'], ['loss', 'Loss 点数'], ['logs', '日志行数'], ['name', '名称']]),
    ].join('');
}

function historyFilterOptionAvailable(tasks, key, value) {
    if (value === 'all' || key === 'sort' || key === 'archived') return true;
    return (tasks || []).some((task) => {
        if (key === 'kind') return task.job === value;
        if (key === 'status') return historyStatusMatches(task, value);
        if (key === 'source') return historySourceMatches(task, value);
        const fields = {
            trainingVariant: task.training_variant || task.variant,
            preprocessPrecision: task.preprocess_precision,
            blockSwapPrecision: task.block_swap_precision,
            baseCompute: task.base_compute,
            precisionPreference: task.precision_preference,
        };
        return String(fields[key] || '').trim().toLowerCase() === String(value).toLowerCase();
    });
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
    return `<article><strong>${escapeHtml(value ?? '-')}</strong><span>${escapeHtml(label)}</span></article>`;
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
