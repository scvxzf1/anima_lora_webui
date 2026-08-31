import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import {
    filterHistoryTasks,
    hasActiveHistoryFilters,
    HISTORY_FILTER_DEFAULTS,
    historySourceMatches,
    historyStatusMatches,
    normalizeHistoryFilters,
    stateCategory,
    stateText,
    taskDisplayName,
    taskGroup,
} from './history-model.js?v=dragon-ui-20260828v3';

export function renderHistoryPage(model = {}) {
    const filters = normalizeHistoryFilters(model.filters);
    const result = renderHistoryResults(model.tasks || [], filters);
    const visibleCount = Number.isFinite(model.visibleCount)
        ? model.visibleCount
        : filterHistoryTasks(model.tasks || [], filters).length;
    const error = model.error || '';
    return `
        <div class="dragon-page dragon-page-wide dragon-history-page" data-history-page>
            <div class="dragon-history-layout">
                <aside class="dragon-history-sidebar">
                    <header class="dragon-history-hero dragon-reveal">
                        <div class="dragon-history-hero-copy"><span class="dragon-eyebrow">HISTORY FORGE</span><h1>历史任务</h1><p>按名称、配置分组和状态定位训练记录。</p></div>
                        <div class="dragon-history-hero-side"><span class="dragon-history-count" data-history-count>${visibleCount} / ${(model.tasks || []).length} 条记录</span><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-refresh>${renderIcon('refresh', 'dragon-btn-icon')}<span>刷新</span></button></div>
                        <div class="dragon-history-sidebar-actions"><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-classic>合并查看</button></div>
                    </header>
                    <p class="dragon-history-summary" data-history-summary>${renderHistorySummary(model.tasks || [], filters)}</p>
                    <p class="dragon-config-feedback dragon-status-region${error ? ' dragon-config-feedback-visible' : ''}" data-history-status-region data-tone="${error ? 'error' : ''}" role="status" aria-live="polite">${escapeHtml(error ? `${error}。请检查 WebUI 服务后重试。` : '')}</p>
                    <div class="dragon-history-stats" aria-label="历史任务统计">${renderHistoryStats(model.tasks || [], filters)}</div>
                    <section class="dragon-history-controls dragon-reveal" data-stagger="1" aria-label="训练历史筛选">${historyFilterControls(filters, model.tasks || [])}</section>
                </aside>
                <main class="dragon-history-content"><div data-history-results aria-live="polite">${model.resultsHtml ?? result.html}</div></main>
            </div>
        </div>`;
}

export function renderHistoryResults(tasks = [], queryOrFilters = '', status = 'all') {
    const filters = normalizeHistoryFilters(queryOrFilters && typeof queryOrFilters === 'object' ? queryOrFilters : { search: queryOrFilters, status });
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
        </section>`).join('');
    return { visibleCount: visible.length, html };
}

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
    return labels.map(([key, label]) => `<button type="button" class="dragon-history-stat${historyStatActive(filters, key) ? ' active' : ''}${key === 'error' ? ' error' : ''}" data-history-stat="${key}" data-state="${key}" data-count-state="${counts[key] > 0 ? 'nonzero' : 'zero'}"><strong>${counts[key]}</strong><span>${label}</span></button>`).join('');
}

export function renderHistorySummary(tasks, filters, visibleCount = null) {
    const all = tasks || [];
    const visible = Number.isFinite(visibleCount) ? visibleCount : filterHistoryTasks(all, filters).length;
    return `共 ${all.length} 条记录 · 当前筛选 ${visible} 条 · 归档 ${all.filter((task) => Boolean(task.archived)).length} 条`;
}

export function activeHistoryStat(filters) {
    return ['all', 'training', 'preprocess', 'error', 'archived', 'queue']
        .find((key) => historyStatActive(filters, key)) || '';
}

function renderHistoryItem(task) {
    const state = task.state || task.status || 'unknown';
    const time = task.started_at_text || formatTimestamp(task.started_at) || task.created_at_text || '';
    const loss = task.final_loss != null ? formatLoss(task.final_loss) : '';
    const steps = task.total_steps ?? task.steps ?? '';
    const method = [task.variant, task.preset].filter(Boolean).join(' / ');
    return `<li><a class="dragon-history-item" href="#history/${encodeURIComponent(task.id || '')}"><span class="dragon-history-item-main"><strong class="dragon-history-item-name">${escapeHtml(taskDisplayName(task))}</strong><span class="dragon-history-item-sub">${escapeHtml([task.job === 'preprocess' ? '预处理' : '训练', method].filter(Boolean).join(' · '))}</span></span><span class="dragon-history-item-numbers">${loss ? `<span>Loss <strong>${loss}</strong></span>` : ''}${steps !== '' ? `<span><strong>${escapeHtml(steps)}</strong> 步</span>` : ''}</span><span class="dragon-history-item-state" data-state="${escapeAttribute(state)}">${escapeHtml(stateText(state))}</span><time class="dragon-history-item-meta">${escapeHtml(time || '时间未记录')}</time></a></li>`;
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
    const select = (key, label, options) => `<label><span>${label}</span><select class="dragon-select" data-history-filter="${key}"${key === 'status' ? ' data-history-status' : ''} name="history_${key}">${options.map(([value, text]) => `<option value="${value}"${filters[key] === value ? ' selected' : ''}${historyFilterOptionAvailable(tasks, key, value) ? '' : ' disabled'}>${text}</option>`).join('')}</select></label>`;
    const all = [['all', '全部']];
    const basic = [
        `<label class="dragon-history-search"><span>全局搜索</span><input class="dragon-input" type="search" name="history_search" autocomplete="off" data-history-filter="search" placeholder="任务 / 配置 / 目录，支持 组:关键词 或 配置:lora" value="${escapeAttribute(filters.search)}"></label>`,
        select('kind', '类型', [...all, ['training', '训练'], ['preprocess', '预处理']]),
        select('status', '状态', [['all', '全部'], ['running', '运行中'], ['completed', '已完成'], ['queued', '排队中'], ['error', '异常/失败'], ['interrupted', '已中断'], ['canceled', '已取消'], ['unknown', '其他']]),
        select('sort', '排序', [['newest', '最新优先'], ['oldest', '最早优先'], ['loss', 'Loss 点数'], ['logs', '日志行数'], ['name', '名称']]),
    ].join('');
    const advanced = [
        select('modelFamily', '基座模型', [...all, ['anima', 'Anima'], ['krea2_raw', 'Krea-2'], ['z_image', 'Z-Image']]),
        select('trainingVariant', '训练变体', [...all, ...['lora', 'lokr', 'loha', 'vera', 'glora', 'dora', 'hydralora', 'reft', 'tlora', 'ortholora', 'chimera', 'soft_tokens', 'ip_adapter', 'easycontrol'].map((value) => [value, value])]),
        select('archived', '归档', [['active', '未归档'], ['all', '全部'], ['archived', '已归档']]),
        select('source', '来源', [['all', '全部'], ['queue', '来自队列'], ['resume', '续训'], ['continue', '权重热启动']]),
        select('preprocessPrecision', '预处理精度', [...all, ['bf16', 'bf16'], ['fp16', 'fp16'], ['fp32', 'fp32']]),
        select('blockSwapPrecision', '块交换精度', [...all, ['bf16', 'bf16'], ['fp8_e4m3', 'fp8_e4m3']]),
        select('baseCompute', '底模计算路径', [...all, ['bf16', 'bf16'], ['nf4', 'nf4'], ['w8a16_convrot', 'w8a16_convrot'], ['w8a8_convrot', 'w8a8_convrot']]),
        select('precisionPreference', '精度倾向', [...all, ['bf16', 'bf16'], ['fp16', 'fp16'], ['fp32', 'fp32']]),
    ].join('');
    const advancedActive = ['modelFamily', 'trainingVariant', 'archived', 'source', 'preprocessPrecision', 'blockSwapPrecision', 'baseCompute', 'precisionPreference'].some((key) => filters[key] !== HISTORY_FILTER_DEFAULTS[key]);
    return `${basic}<details class="dragon-history-advanced"${advancedActive ? ' open' : ''}><summary>高级筛选</summary><div>${advanced}</div></details>`;
}

function historyFilterOptionAvailable(tasks, key, value) {
    if (value === 'all' || key === 'sort' || key === 'archived') return true;
    return (tasks || []).some((task) => {
        if (key === 'kind') return task.job === value;
        if (key === 'status') return historyStatusMatches(task, value);
        if (key === 'source') return historySourceMatches(task, value);
        const fields = { modelFamily: task.model_family, trainingVariant: task.training_variant || task.variant, preprocessPrecision: task.preprocess_precision, blockSwapPrecision: task.block_swap_precision, baseCompute: task.base_compute, precisionPreference: task.precision_preference };
        return String(fields[key] || '').trim().toLowerCase() === String(value).toLowerCase();
    });
}

function formatLoss(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(5) : '-';
}

function formatTimestamp(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) return '';
    return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'medium' }).format(new Date(numeric * 1000));
}

function escapeAttribute(value) { return escapeHtml(value); }
