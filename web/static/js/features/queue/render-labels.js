/**
 * Pure labels and small text helpers for training queue render.
 */
export function queueFilterLabel(value) {
    return {
        actionable: '待处理',
        all: '全部',
        queued: '等待',
        running: '运行',
        error: '异常',
        done: '完成',
        canceled: '已取消',
    }[value] || '当前';
}

export function queueDetailsSummaryText(item) {
    if (item?.state === 'error') return '查看异常上下文和任务路径';
    if (item?.state === 'running') return '查看运行配置和关联信息';
    if (item?.state === 'queued') return '查看排队配置和创建时间';
    return '查看任务记录和路径';
}

export function queueFailurePolicyLabel(value) {
    return value === 'continue' ? '继续下一个' : '暂停队列';
}

export function queueFilterCount(value, counts) {
    if (value === 'all') return counts.total;
    if (value === 'actionable') return counts.running + counts.queued + counts.error;
    return Number(counts[value] || 0);
}

export function queueEmptyStateText(filter, counts) {
    if (!counts.total) return '队列为空。可以回到配置页，用「加入队列」把训练任务排进来。';
    if (filter === 'actionable') return '当前没有待处理任务。完成或取消记录可以切到「全部」「完成」「已取消」查看。';
    return `「${queueFilterLabel(filter)}」视图暂无任务。可以切换筛选或刷新队列状态。`;
}

export function queueKindLabel(item) {
    if (item?.requires_preprocess) return '预处理后训练';
    if (item?.kind === 'resume') return '续训';
    return '训练';
}

export function queueAttemptLabel(item) {
    const attempt = Number(item?.attempt || 1);
    return attempt > 1 ? `第 ${attempt} 次尝试` : '';
}

export function queueHistoryLabel(item) {
    const list = Array.isArray(item?.history_task_ids) ? item.history_task_ids : [];
    return list.join(', ');
}

export function queueShortId(value) {
    const text = String(value || '');
    return text.length > 12 ? `${text.slice(0, 6)}…${text.slice(-4)}` : text;
}

export function queueGpuLabel(value) {
    const list = Array.isArray(value) ? value : [];
    return list.length ? list.join(',') : '全部';
}

export function queueStateLabel(itemState) {
    return {
        queued: '等待',
        running: '运行中',
        done: '完成',
        error: '异常',
        canceled: '已取消',
    }[itemState] || itemState || '未知';
}

export function queueManagerStatusText(counts, state) {
    const policy = queueFailurePolicyLabel(state.queue.failurePolicy);
    if (state.queue.paused) return `队列已暂停 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
    if (counts.running) return `队列运行中 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
    if (counts.queued) return `空闲时会自动启动 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
    return counts.total ? `没有等待任务 · 异常 ${counts.error} 个 · 失败后${policy}` : `队列为空 · 失败后${policy}`;
}
