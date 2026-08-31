export const HISTORY_FILTER_DEFAULTS = Object.freeze({
    search: '', kind: 'all', status: 'all', archived: 'active', source: 'all',
    modelFamily: 'all', trainingVariant: 'all', preprocessPrecision: 'all', blockSwapPrecision: 'all',
    baseCompute: 'all', precisionPreference: 'all', sort: 'newest',
});

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

export function createHistorySearchIndex(tasks = []) {
    const index = new WeakMap();
    tasks.forEach((task) => index.set(task, historySearchRecord(task)));
    return index;
}

export function filterHistoryTasks(tasks, filters = {}, searchIndex = null) {
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
        const indexed = searchIndex?.get(task);
        if (search.global && !(indexed?.global ?? historySearchText(task)).includes(search.global)) return false;
        if (search.collection && !(indexed?.collection ?? historyCollectionSearchText(task)).includes(search.collection)) return false;
        if (search.config && !(indexed?.config ?? historyConfigSearchText(task)).includes(search.config)) return false;
        return true;
    }).sort(historyTaskComparator(normalized.sort));
}

export function normalizeHistoryFilters(filters = {}) {
    return {
        search: String(filters.search ?? '').trim(), kind: filters.kind || 'all', status: filters.status || 'all',
        archived: filters.archived || 'active', source: filters.source || 'all',
        modelFamily: filters.modelFamily || 'all', trainingVariant: filters.trainingVariant || 'all', preprocessPrecision: filters.preprocessPrecision || 'all',
        blockSwapPrecision: filters.blockSwapPrecision || 'all', baseCompute: filters.baseCompute || 'all',
        precisionPreference: filters.precisionPreference || 'all', sort: filters.sort || 'newest',
    };
}

export function hasActiveHistoryFilters(filters) {
    return Object.entries(normalizeHistoryFilters(filters)).some(([key, value]) => (
        key === 'search' ? Boolean(value) : value !== HISTORY_FILTER_DEFAULTS[key]
    ));
}

export function historyStatusMatches(task, status) {
    const state = stateCategory(task.state || task.status);
    return status === 'error' ? ['error', 'interrupted'].includes(state) : state === status;
}

export function historySourceMatches(task, source) {
    if (source === 'queue') return Boolean(task.from_queue || task.queue_item_id);
    if (source === 'resume') return Boolean(task.resume_from?.source_task_id || task.resume_source_task_id);
    if (source === 'continue') return task.training_mode === 'continue_lora';
    return source === 'all';
}

export function stateCategory(state) {
    const key = String(state || '').toLowerCase();
    if (['idle', 'completed', 'done'].includes(key)) return 'completed';
    if (['running', 'training', 'compiling'].includes(key)) return 'running';
    if (['interrupted', 'stopped'].includes(key)) return 'interrupted';
    if (['canceled', 'cancelled'].includes(key)) return 'canceled';
    if (['error', 'failed'].includes(key)) return 'error';
    if (key === 'queued') return 'queued';
    return 'unknown';
}

export function taskGroup(task = {}) {
    return String(task.group || task.history_group_label || task.config_group || task.variant || '未分组').trim();
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
    return [taskDisplayName(task), task.id, task.group, task.history_group_label, task.history_source_config_file, task.model_family, task.variant, task.training_variant, task.preset, task.methods_subdir, task.run_dir, task.output_dir, task.message, stateText(task.state || task.status)].filter(Boolean).join(' ').toLocaleLowerCase();
}

function historySearchRecord(task) {
    return {
        global: historySearchText(task),
        collection: historyCollectionSearchText(task),
        config: historyConfigSearchText(task),
    };
}

function historyCollectionSearchText(task) {
    return [task.group, task.collection, task.history_collection].filter(Boolean).join(' ').toLocaleLowerCase();
}

function historyConfigSearchText(task) {
    return [task.history_group_label, task.history_source_config_file, task.config_group, task.variant, task.preset].filter(Boolean).join(' ').toLocaleLowerCase();
}

function historyChipMatches(task, filters) {
    const values = [
        ['modelFamily', task.model_family],
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
