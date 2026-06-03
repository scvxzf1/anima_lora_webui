import {
    cancelAllTrainingQueue,
    cancelWaitingTrainingQueue,
    clearCanceledTrainingQueue,
    clearCompletedTrainingQueue,
    deleteTrainingQueueItem,
    moveTrainingQueueItem,
    retryTrainingQueueItem,
    updateTrainingQueueSettingsRequest,
} from './api.js?v=module-bootstrap-20260603-6';
import { queueSummaryCounts, setQueueFilter } from './state.js?v=module-bootstrap-20260603-6';

export function createQueueActions({ ctx, state, deps, updateTrainingQueueFromPayload, renderTrainingQueue, queueItemTitle }) {
    async function moveQueueItem(itemId, direction) {
        try {
            const payload = await moveTrainingQueueItem(ctx, itemId, direction);
            updateTrainingQueueFromPayload(payload);
            if (!payload.ok) deps.appendLog(`[状态] ${payload.error || '移动队列任务失败'}`);
        } catch (e) {
            deps.appendLog(`[状态] 移动队列任务失败: ${e.message}`);
        }
    }

    async function cancelQueueItem(itemId) {
        const item = state.queue.items.find((entry) => entry.id === itemId);
        const running = item?.state === 'running';
        const ok = await deps.showAppConfirmDialog({
            title: running ? '停止队列任务' : '取消队列任务',
            description: queueItemTitle(item || {}),
            message: running
                ? '会停止当前正在运行的进程，并自动暂停队列，避免立刻启动下一项。'
                : '等待中的任务会从自动调度中移除；已创建的运行目录会保留。',
            confirmText: running ? '停止任务' : '取消任务',
            danger: true,
        });
        if (!ok) return;
        try {
            const payload = await deleteTrainingQueueItem(ctx, itemId);
            updateTrainingQueueFromPayload(payload);
            if (!payload.ok) deps.appendLog(`[状态] ${payload.error || '取消队列任务失败'}`);
        } catch (e) {
            deps.appendLog(`[状态] 取消队列任务失败: ${e.message}`);
        }
    }

    async function removeQueueItemFromList(itemId) {
        const item = state.queue.items.find((entry) => entry.id === itemId);
        const ok = await deps.showAppConfirmDialog({
            title: '移除列表',
            description: queueItemTitle(item || {}),
            message: '只会将这条记录从队列界面移除，不会删除运行缓存、历史任务、权重、日志或其他文件。',
            confirmText: '移除列表',
            danger: true,
        });
        if (!ok) return;
        try {
            await removeQueueItemRecord(itemId);
        } catch (e) {
            deps.appendLog(`[状态] 移除队列列表记录失败: ${e.message}`);
        }
    }

    async function removeQueueItemRecord(itemId) {
        const payload = await deleteTrainingQueueItem(ctx, itemId);
        updateTrainingQueueFromPayload(payload);
        if (!payload.ok) deps.appendLog(`[状态] ${payload.error || '移除队列列表记录失败'}`);
        return payload;
    }

    async function retryQueueItem(itemId) {
        const item = state.queue.items.find((entry) => entry.id === itemId);
        const ok = await deps.showAppConfirmDialog({
            title: '重新加入队列',
            description: queueItemTitle(item || {}),
            message: '会从该任务冻结的 runtime 配置克隆出新的运行目录，不会读取当前已修改的源 TOML。',
            confirmText: '重新入队',
        });
        if (!ok) return;
        try {
            const payload = await retryTrainingQueueItem(ctx, itemId);
            updateTrainingQueueFromPayload(payload);
            if (!payload.ok) deps.appendLog(`[状态] ${payload.error || '重新入队失败'}`);
            if (payload.ok !== false && item?.state === 'error') {
                const cleanup = await deps.showAppConfirmDialog({
                    title: '新任务已加入队列',
                    description: queueItemTitle(item || {}),
                    message: '是否移除原异常记录？只会将原记录从队列界面移除，不会删除运行缓存、历史任务、权重、日志或其他文件。',
                    confirmText: '移除原记录',
                    cancelText: '保留记录',
                    danger: true,
                });
                if (cleanup) await removeQueueItemRecord(itemId);
            }
        } catch (e) {
            deps.appendLog(`[状态] 重新入队失败: ${e.message}`);
        }
    }

    async function cancelWaitingQueueItems() {
        const count = queueSummaryCounts(state).queued;
        if (!count) return;
        const ok = await deps.showAppConfirmDialog({
            title: '取消全部等待任务',
            description: `${count} 个等待任务`,
            message: '只会取消尚未启动的队列项；运行中任务、历史记录和运行目录都会保留。',
            confirmText: '取消全部等待',
            danger: true,
        });
        if (!ok) return;
        try {
            updateTrainingQueueFromPayload(await cancelWaitingTrainingQueue(ctx));
        } catch (e) {
            deps.appendLog(`[状态] 批量取消队列失败: ${e.message}`);
        }
    }

    async function cancelAllQueueItems() {
        const counts = queueSummaryCounts(state);
        const count = counts.queued + counts.running;
        if (!count) return;
        const ok = await deps.showAppConfirmDialog({
            title: '取消全部队列任务',
            description: `${count} 个队列任务`,
            message: counts.running
                ? '会停止当前运行中的队列任务，并取消所有等待任务；队列会自动暂停。不会删除训练历史、缓存、权重、日志或其他文件。'
                : '会取消所有等待中的队列任务；不会删除训练历史、缓存、权重、日志或其他文件。',
            confirmText: '取消全部队列',
            danger: true,
        });
        if (!ok) return;
        try {
            updateTrainingQueueFromPayload(await cancelAllTrainingQueue(ctx));
        } catch (e) {
            deps.appendLog(`[状态] 一键取消队列失败: ${e.message}`);
        }
    }

    async function clearQueueItemsByTerminalState(itemState) {
        const counts = queueSummaryCounts(state);
        const options = {
            done: {
                count: counts.done,
                title: '清理已完成记录',
                description: `${counts.done} 条已完成记录`,
                message: '只会清理队列文件里的已完成记录；不会删除训练历史、日志、样张、权重、运行目录、缓存或任何实际文件。',
                confirmText: '清理已完成',
                request: () => clearCompletedTrainingQueue(ctx),
                errorText: '清理已完成队列记录失败',
            },
            canceled: {
                count: counts.canceled,
                title: '清理已取消记录',
                description: `${counts.canceled} 条已取消记录`,
                message: '只会清理队列文件里的已取消记录；不会删除训练历史、日志、样张、权重、运行目录、缓存或任何实际文件。',
                confirmText: '清理已取消',
                request: () => clearCanceledTrainingQueue(ctx),
                errorText: '清理已取消队列记录失败',
            },
        };
        const option = options[itemState];
        if (!option) return;
        const count = option.count;
        if (!count) return;
        const ok = await deps.showAppConfirmDialog({
            title: option.title,
            description: option.description,
            message: option.message,
            confirmText: option.confirmText,
        });
        if (!ok) return;
        try {
            const payload = await option.request();
            updateTrainingQueueFromPayload(payload);
            focusQueueFilterAfterTerminalClear(itemState);
            const kept = itemState === 'canceled' ? queueSummaryCounts(state).done : queueSummaryCounts(state).canceled;
            const keptText = itemState === 'canceled' ? '已完成记录已保留' : '已取消记录已保留';
            deps.appendLog(`[状态] ${payload.message || option.confirmText}${kept ? `；${keptText} ${kept} 条` : ''}`);
        } catch (e) {
            deps.appendLog(`[状态] ${option.errorText}: ${e.message}`);
        }
    }

    function focusQueueFilterAfterTerminalClear(itemState) {
        const counts = queueSummaryCounts(state);
        const nextFilter = itemState === 'canceled'
            ? (counts.done > 0 ? 'done' : 'all')
            : (counts.canceled > 0 ? 'canceled' : 'all');
        if (state.filter !== nextFilter) {
            setQueueFilter(state, nextFilter);
            renderTrainingQueue();
        }
    }

    async function clearCompletedQueueItems() {
        await clearQueueItemsByTerminalState('done');
    }

    async function clearCanceledQueueItems() {
        await clearQueueItemsByTerminalState('canceled');
    }

    async function updateTrainingQueueSettings(patch) {
        try {
            const payload = await updateTrainingQueueSettingsRequest(ctx, patch);
            updateTrainingQueueFromPayload(payload);
        } catch (e) {
            deps.appendLog(`[状态] 更新队列设置失败: ${e.message}`);
            deps.loadTrainingQueue();
        }
    }

    async function toggleTrainingQueuePause() {
        await updateTrainingQueueSettings({ paused: !state.queue.paused });
    }

    function bindQueueEvents() {
        const on = (id, eventName, handler) => {
            document.getElementById(id)?.addEventListener(eventName, handler);
        };
        on('btn-refresh-queue', 'click', deps.loadTrainingQueue);
        on('btn-manager-refresh-queue', 'click', deps.loadTrainingQueue);
        on('btn-toggle-queue-pause', 'click', toggleTrainingQueuePause);
        on('btn-manager-toggle-queue-pause', 'click', toggleTrainingQueuePause);
        on('btn-cancel-all-queue', 'click', cancelAllQueueItems);
        on('btn-cancel-waiting-queue', 'click', cancelWaitingQueueItems);
        on('btn-clear-completed-queue', 'click', clearCompletedQueueItems);
        on('btn-clear-canceled-queue', 'click', clearCanceledQueueItems);
        document.querySelectorAll('.training-queue-filter').forEach((btn) => {
            btn.addEventListener('click', () => {
                setQueueFilter(state, btn.dataset.queueFilter || 'actionable');
                renderTrainingQueue();
            });
        });
        document.getElementById('training-queue-failure-policy')?.addEventListener('change', (e) => {
            updateTrainingQueueSettings({ failure_policy: e.target.value || 'pause' });
        });
    }

    return {
        moveQueueItem,
        cancelQueueItem,
        removeQueueItemFromList,
        retryQueueItem,
        cancelWaitingQueueItems,
        cancelAllQueueItems,
        clearCompletedQueueItems,
        clearCanceledQueueItems,
        updateTrainingQueueSettings,
        toggleTrainingQueuePause,
        bindQueueEvents,
    };
}
