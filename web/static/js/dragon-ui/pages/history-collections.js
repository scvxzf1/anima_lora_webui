/* Dragon history collection workbench: pure rendering and collection helpers. */

import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { filterHistoryTasks, stateText, taskDisplayName } from './history-view.js?v=dragon-ui-20260819v92';

export const HISTORY_COLLECTION_ALL = '__all__';
export const HISTORY_COLLECTION_UNGROUPED = '__ungrouped__';

export function createHistoryCollectionWorkspace(settings = {}) {
    return {
        activeKey: HISTORY_COLLECTION_UNGROUPED,
        selectedTaskIds: new Set(),
        expandedConfigKeys: new Set(),
        dragTaskIds: [],
        draggedCollection: '',
        historyDragFrame: 0,
        historyPendingDrop: null,
        historyDropTarget: null,
        initializedExpansion: false,
        settings: normalizeCollectionSettings(settings),
    };
}

export function normalizeCollectionSettings(settings = {}) {
    return {
        collection_order: uniqueStrings(settings.collection_order),
        config_group_order: settings.config_group_order && typeof settings.config_group_order === 'object'
            ? settings.config_group_order
            : {},
    };
}

export function historyTaskCollection(task = {}) {
    return String(task.group || task.collection || task.history_collection || '').trim();
}

export function collectionKey(value) {
    const clean = String(value || '').trim();
    return clean ? `collection:${clean}` : HISTORY_COLLECTION_UNGROUPED;
}

export function collectionValue(key) {
    return String(key || '').startsWith('collection:') ? String(key).slice(11) : '';
}

export function renderHistoryCollectionWorkbench(tasks = [], filters = {}, workspace) {
    const filtered = filterHistoryTasks(tasks, filters);
    const collections = buildCollections(filtered, workspace.settings);
    const active = resolveActiveCollection(collections, workspace.activeKey);
    workspace.activeKey = active.key;
    const visible = active.key === HISTORY_COLLECTION_ALL
        ? filtered
        : filtered.filter((task) => collectionKey(historyTaskCollection(task)) === active.key);
    const visibleIds = new Set(visible.map((task) => task.id).filter(Boolean));
    workspace.selectedTaskIds = new Set(
        [...workspace.selectedTaskIds].filter((taskId) => visibleIds.has(taskId)),
    );
    const selected = visible.filter((task) => workspace.selectedTaskIds.has(task.id));
    const configGroups = groupByConfig(visible);
    if (!workspace.initializedExpansion) {
        if (configGroups[0]?.key) workspace.expandedConfigKeys.add(configGroups[0].key);
        workspace.initializedExpansion = true;
    }
    const assignedCount = filtered.filter((task) => historyTaskCollection(task)).length;
    const ungroupedCount = filtered.length - assignedCount;

    return `
        <section class="dragon-history-workbench" data-history-workbench>
            <header class="dragon-history-workbench-hero">
                <div><span class="dragon-eyebrow">COLLECTION WORKBENCH</span><h2>历史分组</h2><p>选择任务或拖动配置组，在右侧导航中整理训练记录。</p></div>
                <div class="dragon-history-workbench-metrics" aria-label="历史分组统计">
                    ${metric(collections.filter((item) => item.value).length, '分组')}
                    ${metric(filtered.length, '当前任务')}
                    ${metric(configGroups.length, '任务集')}
                    ${metric(selected.length, '已选任务')}
                </div>
            </header>
            <div class="dragon-history-workbench-toolbar">
                <strong>当前：${escapeHtml(active.label)}</strong>
                <span>${visible.length} 条任务 · ${selected.length} 条已选</span>
                <div>
                    <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-history-collection-action="assign"${selected.length ? '' : ' disabled'}>设置分组</button>
                    <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-history-collection-action="clear-selected"${selected.length ? '' : ' disabled'}>清除分组</button>
                </div>
            </div>
            <div class="dragon-history-workbench-grid">
                <section class="dragon-history-collection-tasks">
                    <div class="dragon-history-workbench-section-head"><div><span class="dragon-eyebrow">任务工作区</span><h3>${escapeHtml(active.label)}任务</h3></div><span>${configGroups.length} 个任务集 · ${visible.length} 条</span></div>
                    ${renderCollectionTaskBody(configGroups, active, workspace)}
                </section>
                <aside class="dragon-history-collection-nav dragon-dataset-library" aria-label="历史分组库">
                    <div class="dragon-dataset-library-head"><div><span class="dragon-eyebrow">分组导航</span><h2>历史分组</h2></div><button class="dragon-icon-button" type="button" data-history-collection-action="refresh" aria-label="刷新历史分组">↻</button></div>
                    <div class="dragon-dataset-library-actions dragon-history-collection-library-actions"><button class="dragon-btn dragon-btn-secondary dragon-btn-sm dragon-dataset-new-group" type="button" data-history-collection-action="new"><span aria-hidden="true">□</span><span>新建历史分组</span></button></div>
                    <div class="dragon-dataset-library-meta"><span>${filtered.length} 条当前任务</span><span>${collections.filter((item) => item.value).length} 个分组</span></div>
                    <div class="dragon-dataset-preset-list dragon-history-collection-nav-list">
                        <section class="dragon-dataset-preset-group">
                            <header><div class="dragon-dataset-preset-group-title"><span>历史分组</span><small>${collections.length + 1}</small></div></header>
                            <div>
                                ${renderCollectionNavItem({ key: HISTORY_COLLECTION_ALL, label: '全部记录', tasks: filtered }, active.key, workspace)}
                                ${collections.map((item) => renderCollectionNavItem(item, active.key, workspace)).join('')}
                                <div class="dragon-dataset-preset-dropzone" data-history-collection-dropzone data-empty="false">拖到分组末尾</div>
                            </div>
                        </section>
                    </div>
                    <p class="dragon-history-collection-hint">拖动配置组或已选任务到分组卡片可归类；拖动自定义分组可调整顺序。</p>
                    <div class="dragon-history-collection-summary"><span>${ungroupedCount} 条未分类</span><span>${assignedCount} 条已归类</span></div>
                </aside>
            </div>
        </section>
    `;
}

function buildCollections(tasks, settings) {
    const map = new Map();
    for (const task of tasks) {
        const value = historyTaskCollection(task);
        const key = collectionKey(value);
        if (!map.has(key)) map.set(key, { key, value, label: value || '未分类', tasks: [] });
        map.get(key).tasks.push(task);
    }
    if (!map.has(HISTORY_COLLECTION_UNGROUPED)) {
        map.set(HISTORY_COLLECTION_UNGROUPED, { key: HISTORY_COLLECTION_UNGROUPED, value: '', label: '未分类', tasks: [] });
    }
    for (const value of settings.collection_order || []) {
        const key = collectionKey(value);
        if (!map.has(key)) map.set(key, { key, value, label: value, tasks: [] });
    }
    const ordered = [];
    ordered.push(map.get(HISTORY_COLLECTION_UNGROUPED));
    const values = uniqueStrings([
        ...(settings.collection_order || []),
        ...[...map.values()].map((item) => item.value).filter(Boolean).sort((a, b) => a.localeCompare(b, 'zh-CN')),
    ]);
    for (const value of values) ordered.push(map.get(collectionKey(value)));
    return ordered.filter(Boolean);
}

function resolveActiveCollection(collections, key) {
    if (key === HISTORY_COLLECTION_ALL) return { key, value: '', label: '全部记录', tasks: collections.flatMap((item) => item.tasks) };
    return collections.find((item) => item.key === key) || collections[0];
}

function groupByConfig(tasks) {
    const groups = new Map();
    for (const task of tasks) {
        const key = task.history_group_key || task.history_source_config_file || [task.methods_subdir, task.variant, task.preset].join(':');
        if (!groups.has(key)) groups.set(key, { key, label: task.history_group_label || task.history_source_config_file || task.variant || '未命名配置', tasks: [] });
        groups.get(key).tasks.push(task);
    }
    return [...groups.values()];
}

function renderCollectionTaskBody(configGroups, active, workspace) {
    if (!configGroups.length) return renderCollectionEmpty(active);
    return configGroups.map((group) => renderConfigGroup(group, workspace)).join('');
}

function renderConfigGroup(group, workspace) {
    const ids = group.tasks.map((task) => task.id).filter(Boolean);
    const allSelected = ids.length > 0 && ids.every((id) => workspace.selectedTaskIds.has(id));
    const expanded = workspace.expandedConfigKeys.has(group.key);
    const training = group.tasks.filter((task) => task.job === 'training').length;
    const preprocess = group.tasks.length - training;
    const errors = group.tasks.filter((task) => ['error', 'failed', 'interrupted', 'stopped'].includes(String(task.state || task.status).toLowerCase())).length;
    const queue = group.tasks.filter((task) => task.from_queue || task.queue_item_id).length;
    return `
        <article class="dragon-history-config-card dragon-dataset-preset-group${expanded ? ' expanded' : ''}" draggable="true" data-history-drag-task-ids="${escapeHtml(ids.join(','))}">
            <div class="dragon-history-config-card-head">
                <input type="checkbox" aria-label="选择配置组 ${escapeHtml(group.label)}" data-history-select-group="${escapeHtml(ids.join(','))}"${allSelected ? ' checked' : ''}>
                <button class="dragon-history-config-toggle" type="button" data-history-collection-action="toggle-config" data-history-config-key="${escapeHtml(group.key)}" aria-expanded="${expanded}">
                    <span aria-hidden="true">${expanded ? '−' : '+'}</span><span><strong>${escapeHtml(group.label)}</strong><small>${group.tasks.length} 条 · ${training} 训${preprocess ? ` · ${preprocess} 预` : ''}${errors ? ` · ${errors} 异常` : ''}${queue ? ` · ${queue} 队列` : ''}</small></span>
                </button>
            </div>
            ${expanded ? `<ul class="dragon-history-config-task-list">${group.tasks.map((task) => renderWorkbenchTask(task, workspace)).join('')}</ul>` : `<button class="dragon-history-config-collapsed" type="button" data-history-collection-action="toggle-config" data-history-config-key="${escapeHtml(group.key)}">已折叠 ${group.tasks.length} 条任务 · 点击展开查看</button>`}
        </article>
    `;
}

function renderWorkbenchTask(task, workspace) {
    const checked = workspace.selectedTaskIds.has(task.id);
    const time = task.started_at_text || task.created_at_text || '时间未记录';
    return `<li class="dragon-dataset-preset-row" draggable="true" data-history-drag-task-ids="${escapeHtml(task.id || '')}">
        <input type="checkbox" aria-label="选择任务 ${escapeHtml(taskDisplayName(task))}" data-history-select-task="${escapeHtml(task.id || '')}"${checked ? ' checked' : ''}>
        <a class="dragon-dataset-preset-item" href="#history/${encodeURIComponent(task.id || '')}"><span><strong>${escapeHtml(taskDisplayName(task))}</strong><small>${escapeHtml([task.variant, task.preset, historyTaskCollection(task) || '未分类'].filter(Boolean).join(' · '))}</small></span></a>
        <span class="dragon-history-item-state" data-state="${escapeHtml(task.state || task.status || 'unknown')}">${escapeHtml(stateText(task.state || task.status))}</span>
        <time>${escapeHtml(time)}</time>
        <span class="dragon-history-workbench-counts">${Number(task.metric_count || 0)} loss / ${Number(task.log_count || 0)} log</span>
    </li>`;
}

function renderCollectionNavItem(item, activeKey, workspace) {
    const active = item.key === activeKey;
    const training = item.tasks.filter((task) => task.job === 'training').length;
    const errors = item.tasks.filter((task) => ['error', 'failed', 'interrupted', 'stopped'].includes(String(task.state || task.status).toLowerCase())).length;
    const queue = item.tasks.filter((task) => task.from_queue || task.queue_item_id).length;
    const removable = Boolean(item.value);
    const dropTarget = item.key === HISTORY_COLLECTION_ALL
        ? ''
        : ` data-history-drop-collection="${escapeHtml(item.value || '')}"`;
    const draggable = removable ? ` draggable="true" data-history-drag-collection="${escapeHtml(item.value)}"` : '';
    return `<article class="dragon-dataset-preset-row dragon-history-collection-nav-card${active ? ' active' : ''}"${dropTarget}${draggable}>
        <button class="dragon-dataset-preset-item" type="button" data-history-collection-action="select" data-history-collection-key="${escapeHtml(item.key)}"><span><strong>${escapeHtml(item.label)}</strong><small>${item.tasks.length} 条 · ${training} 训${errors ? ` · ${errors} 异常` : ''}${queue ? ` · ${queue} 队列` : ''}</small></span><span class="dragon-dataset-preset-item-meta"><em>${item.tasks.length} 条</em></span></button>
        ${removable ? `<div class="dragon-history-collection-row-actions"><button type="button" title="重命名分组" aria-label="重命名 ${escapeHtml(item.label)}" data-history-collection-action="rename" data-history-collection-value="${escapeHtml(item.value)}">✎</button><button type="button" title="清空并删除分组" aria-label="删除 ${escapeHtml(item.label)}" data-history-collection-action="delete" data-history-collection-value="${escapeHtml(item.value)}">×</button></div>` : ''}
    </article>`;
}

function renderCollectionEmpty(active) {
    return `<div class="dragon-history-collection-empty"><strong>${escapeHtml(active.label)}中还没有任务</strong><p>从“全部记录”选择任务，或直接把配置组拖到右侧分组卡片。</p></div>`;
}

function metric(value, label) {
    return `<div><strong>${Number(value || 0)}</strong><span>${label}</span></div>`;
}

function uniqueStrings(values) {
    return [...new Set((Array.isArray(values) ? values : []).map((item) => String(item || '').trim()).filter(Boolean))];
}
