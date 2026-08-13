/* Presentational helpers for Dragon training history. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { escapeHtml, formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';

export function renderHistoryPage(model = {}) {
    const result = renderHistoryResults(model.tasks || [], model.query || '', model.status || 'all');
    const error = model.error || '';
    return `
        <div class="dragon-page dragon-page-wide dragon-history-page" data-history-page>
            <header class="dragon-history-hero dragon-reveal">
                <div class="dragon-history-hero-copy">
                    <span class="dragon-eyebrow">训练监控 · 归档</span>
                    <h1>训练历史</h1>
                    <p>按名称、配置分组和状态定位训练记录，查看配置快照、曲线、样张与权重。</p>
                </div>
                <div class="dragon-history-hero-side">
                    <span class="dragon-history-count" data-history-count>${result.visibleCount} / ${(model.tasks || []).length} 条记录</span>
                    <button class="dragon-btn dragon-btn-secondary" type="button" data-history-refresh>${renderIcon('refresh', 'dragon-btn-icon')}<span>刷新</span></button>
                </div>
            </header>
            <p class="dragon-config-feedback dragon-status-region${error ? ' dragon-config-feedback-visible' : ''}" data-history-status-region data-tone="${error ? 'error' : ''}" role="status" aria-live="polite">${escapeHtml(error ? `${error}。请检查 WebUI 服务后重试。` : '')}</p>
            <section class="dragon-history-controls dragon-reveal" data-stagger="1" aria-label="训练历史筛选">
                <label class="dragon-history-search"><span>搜索记录</span><input class="dragon-input" type="search" name="history_search" autocomplete="off" data-history-search placeholder="例如：任务名、配置组或任务 ID…"></label>
                <label class="dragon-history-status-filter"><span>任务状态</span><select class="dragon-select" name="history_status" data-history-status>${statusOptions()}</select></label>
            </section>
            <div data-history-results aria-live="polite">${result.html}</div>
        </div>
    `;
}

export function renderHistoryResults(tasks = [], query = '', status = 'all') {
    const visible = filterHistoryTasks(tasks, query, status);
    if (!visible.length) {
        const hasFilters = Boolean(String(query || '').trim()) || status !== 'all';
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
    const { taskId, payload, images, weights, resume, lossChart } = model;
    const task = payload.task || {};
    const metrics = Array.isArray(payload.metrics) ? payload.metrics : [];
    const logs = Array.isArray(payload.logs) ? payload.logs : [];
    const lastLoss = lastMetricValue(metrics, 'loss');
    const state = task.state || task.status || 'unknown';
    return `
        <div class="dragon-page dragon-page-wide dragon-history-detail-page" data-history-detail="${escapeAttribute(taskId)}">
            <header class="dragon-history-detail-hero dragon-reveal">
                <div class="dragon-history-detail-toolbar">
                    <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-back>${renderIcon('history', 'dragon-btn-icon')}<span>返回历史</span></button>
                    <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-detail-refresh>${renderIcon('refresh', 'dragon-btn-icon')}<span>刷新详情</span></button>
                </div>
                <div class="dragon-history-detail-heading">
                    <div><span class="dragon-eyebrow">${escapeHtml(taskGroup(task))}</span><h1>${escapeHtml(taskDisplayName(task))}</h1><p>${escapeHtml(task.started_at_text || formatTimestamp(task.started_at) || taskId)}</p></div>
                    <span class="dragon-history-item-state" data-state="${escapeAttribute(state)}">${escapeHtml(stateText(state))}</span>
                </div>
            </header>

            <section class="dragon-history-stat-grid dragon-reveal" data-stagger="1">
                ${metricTile('最终损失', formatLoss(lastLoss ?? task.final_loss))}
                ${metricTile('最后步数', lastMetricValue(metrics, 'step') ?? task.total_steps ?? '-')}
                ${metricTile('学习率', formatLr(lastMetricValue(metrics, 'lr')))}
                ${metricTile('曲线数据', `${metrics.length || task.metric_count || 0} 点`)}
            </section>

            <div class="dragon-history-detail-grid dragon-reveal" data-stagger="2">
                <section class="dragon-history-panel">
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
                </section>
                ${renderResumePanel(task, resume)}
            </div>

            <section class="dragon-history-panel dragon-reveal" data-stagger="3">
                <div class="dragon-history-panel-head"><div><span class="dragon-eyebrow">训练指标</span><h2>损失曲线</h2></div><span>${metrics.length} 个数据点</span></div>
                <div class="dragon-chart-container">${lossChart}</div>
            </section>

            ${renderHistoryArtifacts(taskId, task, payload.config_toml || '')}
            ${renderPreviewSection(images, weights)}
            ${renderLogsSection(taskId, logs, task)}
        </div>
    `;
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
    return `<div class="dragon-history-preview-grid">${images.map((image) => {
        const sample = image.sample || {};
        const title = sample.step != null ? `Step ${sample.step}` : (image.name || '训练样张');
        return `<figure><a href="${escapeAttribute(image.url || '')}" target="_blank" rel="noopener"><img src="${escapeAttribute(image.url || '')}" alt="${escapeAttribute(sample.prompt ? `${title}：${sample.prompt}` : title)}" width="${escapeAttribute(image.width || 1)}" height="${escapeAttribute(image.height || 1)}" loading="lazy"></a><figcaption><strong>${escapeHtml(title)}</strong><span>${escapeHtml(image.mtime_text || image.name || '')}</span></figcaption></figure>`;
    }).join('')}</div>`;
}

function renderWeights(weights, message) {
    if (!weights.length) return `<div class="dragon-history-inline-empty"><p>${escapeHtml(message || '这个任务还没有可下载的训练权重。')}</p></div>`;
    return `<div class="dragon-history-weight-list">${weights.map((item) => `<article><span class="dragon-history-weight-icon">${renderIcon('layers')}</span><div><strong>${escapeHtml(item.name || '未命名权重')}</strong><span>${escapeHtml([item.steps != null ? `Step ${item.steps}` : '', item.mtime_text, item.size_bytes != null ? formatBytes(item.size_bytes) : ''].filter(Boolean).join(' · '))}</span></div><a class="dragon-btn dragon-btn-secondary dragon-btn-sm" href="${escapeAttribute(item.download_url || '')}" download="${escapeAttribute(item.name || 'weight.safetensors')}">${renderIcon('download', 'dragon-btn-icon')}<span>下载</span></a></article>`).join('')}</div>`;
}

function renderLogsSection(taskId, logs, task) {
    const lines = logs.slice(-12).map((record) => {
        const line = record.line || record.message || record.text || JSON.stringify(record);
        return `<div class="dragon-log-line"${record.level ? ` data-level="${escapeAttribute(record.level)}"` : ''}>${escapeHtml(line)}</div>`;
    }).join('');
    return `
        <section class="dragon-history-panel dragon-reveal" data-stagger="6">
            <div class="dragon-history-panel-head"><div><span class="dragon-eyebrow">运行输出</span><h2>最近日志</h2></div><a class="dragon-btn dragon-btn-secondary dragon-btn-sm" href="/api/training/history/${encodeURIComponent(taskId)}/logs/download">${renderIcon('download', 'dragon-btn-icon')}<span>下载日志</span></a></div>
            ${task.message ? `<p class="dragon-history-task-message">${escapeHtml(task.message)}</p>` : ''}
            <div class="dragon-log-panel dragon-history-log-panel">${lines || '<div class="dragon-history-inline-empty"><p>暂无日志。</p></div>'}</div>
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

function filterHistoryTasks(tasks, query, status) {
    const needle = String(query || '').trim().toLocaleLowerCase();
    return tasks.filter((task) => {
        if (status !== 'all' && stateCategory(task.state || task.status) !== status) return false;
        if (!needle) return true;
        const haystack = [taskDisplayName(task), task.id, task.group, task.history_group_label, task.history_source_config_file, task.variant, task.preset, task.methods_subdir, stateText(task.state || task.status)].join(' ').toLocaleLowerCase();
        return haystack.includes(needle);
    });
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
