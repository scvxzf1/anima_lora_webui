export function createQueueState() {
    return {
        queue: {
            loading: false,
            paused: false,
            failurePolicy: 'pause',
            items: [],
            error: '',
            status: '',
            currentItemId: '',
            summary: {},
        },
        filter: 'actionable',
        feedback: {
            message: '',
            tone: '',
            busyAction: '',
            busyItemId: '',
            flashItemId: '',
            updatedAt: 0,
        },
    };
}

export function setQueueLoading(state) {
    state.queue = { ...state.queue, loading: true, error: '' };
}

export function setQueueError(state, message) {
    state.queue = { ...state.queue, loading: false, error: message || '' };
}

export function updateQueueStateFromPayload(state, payload = {}) {
    const previous = state.queue || {};
    const hasItems = Array.isArray(payload.items);
    const hasSummary = payload.summary && typeof payload.summary === 'object';
    const isErrorOnly = payload.ok === false && !hasItems;
    state.queue = {
        loading: false,
        paused: payload.paused === undefined ? Boolean(previous.paused) : Boolean(payload.paused),
        failurePolicy: payload.failure_policy || previous.failurePolicy || 'pause',
        items: hasItems ? payload.items : (isErrorOnly ? (previous.items || []) : []),
        error: payload.ok === false ? (payload.error || '队列状态异常') : '',
        status: payload.status === undefined
            ? (isErrorOnly ? (previous.status || '') : '')
            : String(payload.status || ''),
        currentItemId: payload.current_item_id === undefined
            ? (isErrorOnly ? (previous.currentItemId || '') : '')
            : String(payload.current_item_id || ''),
        summary: hasSummary ? payload.summary : (isErrorOnly ? (previous.summary || {}) : {}),
    };
    return state.queue;
}

/**
 * Stable signature for full queue UI rebuilds.
 * Running progress is patched separately via updateRunningQueueProgress, so
 * progress/percent fields are intentionally excluded to avoid full re-renders.
 */
export function queueRenderSignature(state) {
    const queue = state?.queue || {};
    const feedback = state?.feedback || {};
    const items = Array.isArray(queue.items) ? queue.items : [];
    const itemSig = items.map((item) => [
        item?.id || '',
        item?.state || '',
        item?.title || item?.name || item?.variant || '',
        item?.config_file || item?.runtime_config_file || '',
        item?.job || '',
        item?.error || '',
        item?.message || '',
    ].join('')).join('');
    const summary = queue.summary || {};
    return [
        state?.filter || '',
        queue.loading ? '1' : '0',
        queue.paused ? '1' : '0',
        queue.failurePolicy || '',
        queue.error || '',
        queue.status || '',
        queue.currentItemId || '',
        summary.total || 0,
        summary.queued || 0,
        summary.running || 0,
        summary.done || 0,
        summary.error || 0,
        summary.canceled || 0,
        feedback.message || '',
        feedback.tone || '',
        feedback.busyAction || '',
        feedback.busyItemId || '',
        feedback.flashItemId || '',
        itemSig,
    ].join('');
}

export function setQueueFilter(state, filter) {
    state.filter = filter || 'actionable';
}

export function setQueueFeedback(state, feedback = {}) {
    state.feedback = {
        message: '',
        tone: '',
        busyAction: '',
        busyItemId: '',
        flashItemId: '',
        updatedAt: Date.now(),
        ...feedback,
    };
}

export function queueSummaryCounts(state) {
    const queue = state.queue || {};
    const base = { total: 0, queued: 0, running: 0, done: 0, error: 0, canceled: 0 };
    const source = queue.summary || {};
    for (const key of Object.keys(base)) {
        base[key] = Number(source[key] || 0);
    }
    if (!source.total && Array.isArray(queue.items) && queue.items.length) {
        for (const item of queue.items) {
            base.total += 1;
            const itemState = String(item.state || '');
            if (Object.prototype.hasOwnProperty.call(base, itemState)) base[itemState] += 1;
        }
    }
    return base;
}

export function queueManagerSections(state) {
    const items = state.queue.items || [];
    const running = items.filter((item) => item.state === 'running');
    const queued = items.filter((item) => item.state === 'queued');
    const error = items.filter((item) => item.state === 'error');
    const done = items.filter((item) => item.state === 'done');
    const canceled = items.filter((item) => item.state === 'canceled');
    const sections = {
        running: { key: 'running', title: '运行中', description: '当前正在占用训练进程的任务', items: running },
        queued: { key: 'queued', title: '等待执行', description: '空闲且队列未暂停时会按顺序启动', items: queued },
        error: { key: 'error', title: '异常待处理', description: '建议先确认原因，再重新入队或移除列表记录', items: error },
        done: { key: 'done', title: '完成记录', description: '已完成的队列任务；清理已取消不会影响这里', items: done, collapsed: true },
        canceled: { key: 'canceled', title: '已取消记录', description: '已取消的队列任务；清理已完成不会影响这里', items: canceled, collapsed: true },
    };
    if (state.filter === 'all') return [sections.running, sections.queued, sections.error, sections.done, sections.canceled];
    if (state.filter === 'running') return [sections.running];
    if (state.filter === 'queued') return [sections.queued];
    if (state.filter === 'error') return [sections.error];
    if (state.filter === 'done') {
        return [{ ...sections.done, collapsed: false }];
    }
    if (state.filter === 'canceled') {
        return [{ ...sections.canceled, collapsed: false }];
    }
    return [sections.running, sections.queued, sections.error];
}
