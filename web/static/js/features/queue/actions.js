import {
    abortTrainingQueueAfterCurrent,
    cancelAllTrainingQueue,
    cancelWaitingTrainingQueue,
    clearCanceledTrainingQueue,
    clearCompletedTrainingQueue,
    deleteTrainingQueueItem,
    forceAbortTrainingQueue,
    moveTrainingQueueItem,
    retryTrainingQueueItem,
    updateTrainingQueueSettingsRequest,
} from './api.js?v=module-bootstrap-20260711-ir6';
import { queueSummaryCounts, setQueueFeedback, setQueueFilter } from './state.js?v=module-bootstrap-20260711-ir6';

export function createQueueActions({ ctx, state, deps, updateTrainingQueueFromPayload, renderTrainingQueue, queueItemTitle }) {
    function showQueueFeedback(feedback = {}) {
        setQueueFeedback(state, feedback);
        renderTrainingQueue();
    }

    function beginQueueFeedback(message, options = {}) {
        showQueueFeedback({
            message,
            tone: 'busy',
            busyAction: options.action || '',
            busyItemId: options.itemId || '',
        });
    }

    function finishQueueFeedback(message, tone = 'ok', options = {}) {
        showQueueFeedback({
            message,
            tone,
            flashItemId: options.itemId || '',
        });
    }

    async function moveQueueItem(itemId, direction) {
        const item = state.queue.items.find((entry) => entry.id === itemId);
        const action = `move-${itemId}-${direction}`;
        const directionLabel = queueMoveDirectionLabel(direction);
        beginQueueFeedback(`正在${directionLabel}：${queueItemTitle(item || {})}`, { action, itemId });
        try {
            const payload = await moveTrainingQueueItem(ctx, itemId, direction);
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || '移动队列任务失败';
                finishQueueFeedback(message, 'error', { itemId });
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            finishQueueFeedback(`已${directionLabel}：${queueItemTitle(item || {})}`, 'ok', { itemId });
        } catch (e) {
            const message = `移动队列任务失败: ${e.message}`;
            finishQueueFeedback(message, 'error', { itemId });
            deps.appendLog(`[状态] ${message}`);
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
        beginQueueFeedback(running ? `正在停止：${queueItemTitle(item || {})}` : `正在取消：${queueItemTitle(item || {})}`, { action: `cancel-${itemId}`, itemId });
        try {
            const payload = await deleteTrainingQueueItem(ctx, itemId);
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || '取消队列任务失败';
                finishQueueFeedback(message, 'error', { itemId });
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            finishQueueFeedback(running ? '停止请求已发送，队列会暂停。' : '等待任务已取消。', 'ok', { itemId });
        } catch (e) {
            const message = `取消队列任务失败: ${e.message}`;
            finishQueueFeedback(message, 'error', { itemId });
            deps.appendLog(`[状态] ${message}`);
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
        beginQueueFeedback(`正在移除列表记录：${queueItemTitle(item || {})}`, { action: `remove-${itemId}`, itemId });
        try {
            await removeQueueItemRecord(itemId);
            finishQueueFeedback('队列列表记录已移除，运行文件保持不变。', 'ok', { itemId });
        } catch (e) {
            const message = `移除队列列表记录失败: ${e.message}`;
            finishQueueFeedback(message, 'error', { itemId });
            deps.appendLog(`[状态] ${message}`);
        }
    }

    async function removeQueueItemRecord(itemId) {
        const payload = await deleteTrainingQueueItem(ctx, itemId);
        updateTrainingQueueFromPayload(payload);
        if (!payload.ok) {
            const message = payload.error || '移除队列列表记录失败';
            finishQueueFeedback(message, 'error', { itemId });
            deps.appendLog(`[状态] ${message}`);
        }
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
        beginQueueFeedback(`正在重新入队：${queueItemTitle(item || {})}`, { action: `retry-${itemId}`, itemId });
        try {
            const payload = await retryTrainingQueueItem(ctx, itemId);
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || '重新入队失败';
                finishQueueFeedback(message, 'error', { itemId });
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            finishQueueFeedback('新任务已加入队列。', 'ok', { itemId });
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
            const message = `重新入队失败: ${e.message}`;
            finishQueueFeedback(message, 'error', { itemId });
            deps.appendLog(`[状态] ${message}`);
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
        beginQueueFeedback(`正在取消 ${count} 个等待任务`, { action: 'cancel-waiting' });
        try {
            const payload = await cancelWaitingTrainingQueue(ctx);
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || '批量取消队列失败';
                finishQueueFeedback(message, 'error');
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            finishQueueFeedback(`已取消 ${count} 个等待任务。`);
        } catch (e) {
            const message = `批量取消队列失败: ${e.message}`;
            finishQueueFeedback(message, 'error');
            deps.appendLog(`[状态] ${message}`);
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
        beginQueueFeedback(`正在取消 ${count} 个队列任务`, { action: 'cancel-all' });
        try {
            const payload = await cancelAllTrainingQueue(ctx);
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || '一键取消队列失败';
                finishQueueFeedback(message, 'error');
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            finishQueueFeedback(`已发送 ${count} 个队列任务的取消请求。`);
        } catch (e) {
            const message = `一键取消队列失败: ${e.message}`;
            finishQueueFeedback(message, 'error');
            deps.appendLog(`[状态] ${message}`);
        }
    }

    async function abortQueueAfterCurrent() {
        const counts = queueSummaryCounts(state);
        const count = counts.queued;
        if (!count) return;
        const ok = await deps.showAppConfirmDialog({
            title: '中止后续队列',
            description: `${count} 个等待任务`,
            message: counts.running
                ? '当前正在运行的任务会继续执行到完成；确认后会暂停队列并取消所有等待任务，当前任务完成后不会继续下一项。'
                : '确认后会暂停队列并取消所有等待任务，不会影响已完成记录、历史、缓存、权重、日志或运行目录。',
            confirmText: '中止后续队列',
            danger: true,
        });
        if (!ok) return;
        beginQueueFeedback(`正在中止 ${count} 个后续等待任务`, { action: 'abort-after-current' });
        try {
            const payload = await abortTrainingQueueAfterCurrent(ctx);
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || '中止后续队列失败';
                finishQueueFeedback(message, 'error');
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            finishQueueFeedback(payload.message || '已中止后续队列。');
            deps.appendLog(`[状态] ${payload.message || '已中止后续队列'}`);
        } catch (e) {
            const message = `中止后续队列失败: ${e.message}`;
            finishQueueFeedback(message, 'error');
            deps.appendLog(`[状态] ${message}`);
        }
    }

    async function forceAbortQueue() {
        const counts = queueSummaryCounts(state);
        const hasActiveProcess = state.queue.status === 'running' || deps.getTrainingRuntime()?.state === 'running';
        const count = counts.queued + counts.running + (hasActiveProcess && !counts.running ? 1 : 0);
        if (!count) return;
        const ok = await deps.showAppConfirmDialog({
            title: '强制中止队列',
            description: `${count} 个运行或等待任务`,
            message: counts.running || hasActiveProcess
                ? '会立即停止当前正在运行的训练/预处理进程，并取消所有等待任务；队列会保持暂停。不会删除训练历史、缓存、权重、日志或运行目录。'
                : '会取消所有等待任务并暂停队列。不会删除训练历史、缓存、权重、日志或运行目录。',
            confirmText: '强制中止',
            danger: true,
        });
        if (!ok) return;
        beginQueueFeedback(`正在强制中止 ${count} 个队列任务`, { action: 'force-abort' });
        try {
            const payload = await forceAbortTrainingQueue(ctx);
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || '强制中止队列失败';
                finishQueueFeedback(message, 'error');
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            finishQueueFeedback(payload.message || '已强制中止队列。');
            deps.appendLog(`[状态] ${payload.message || '已强制中止队列'}`);
        } catch (e) {
            const message = `强制中止队列失败: ${e.message}`;
            finishQueueFeedback(message, 'error');
            deps.appendLog(`[状态] ${message}`);
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
        beginQueueFeedback(`正在${option.confirmText} ${count} 条记录`, { action: itemState === 'done' ? 'clear-done' : 'clear-canceled' });
        try {
            const payload = await option.request();
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || option.errorText;
                finishQueueFeedback(message, 'error');
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            focusQueueFilterAfterTerminalClear(itemState);
            const kept = itemState === 'canceled' ? queueSummaryCounts(state).done : queueSummaryCounts(state).canceled;
            const keptText = itemState === 'canceled' ? '已完成记录已保留' : '已取消记录已保留';
            const message = `${payload.message || option.confirmText}${kept ? `；${keptText} ${kept} 条` : ''}`;
            finishQueueFeedback(message);
            deps.appendLog(`[状态] ${message}`);
        } catch (e) {
            const message = `${option.errorText}: ${e.message}`;
            finishQueueFeedback(message, 'error');
            deps.appendLog(`[状态] ${message}`);
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
        const policyPatch = Object.prototype.hasOwnProperty.call(patch, 'failure_policy');
        const pausedPatch = Object.prototype.hasOwnProperty.call(patch, 'paused');
        const action = policyPatch ? 'settings-policy' : pausedPatch ? 'settings-paused' : 'settings';
        beginQueueFeedback(policyPatch ? '正在保存失败处理策略' : patch.paused ? '正在暂停队列' : '正在继续队列', { action });
        try {
            const payload = await updateTrainingQueueSettingsRequest(ctx, patch);
            updateTrainingQueueFromPayload(payload);
            if (payload.ok === false) {
                const message = payload.error || '更新队列设置失败';
                finishQueueFeedback(message, 'error');
                deps.appendLog(`[状态] ${message}`);
                return;
            }
            finishQueueFeedback(policyPatch ? '失败处理策略已保存。' : patch.paused ? '队列已暂停。' : '队列已继续。');
        } catch (e) {
            const message = `更新队列设置失败: ${e.message}`;
            finishQueueFeedback(message, 'error');
            deps.appendLog(`[状态] ${message}`);
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
        const refreshQueue = async () => {
            beginQueueFeedback('正在刷新队列状态', { action: 'refresh' });
            await deps.loadTrainingQueue();
            if (state.queue.error) {
                finishQueueFeedback(state.queue.error, 'error');
                return;
            }
            finishQueueFeedback('队列状态已刷新。');
        };
        on('btn-refresh-queue', 'click', refreshQueue);
        on('btn-manager-refresh-queue', 'click', refreshQueue);
        on('btn-toggle-queue-pause', 'click', toggleTrainingQueuePause);
        on('btn-manager-toggle-queue-pause', 'click', toggleTrainingQueuePause);
        on('btn-cancel-all-queue', 'click', cancelAllQueueItems);
        on('btn-abort-queue-after-current', 'click', abortQueueAfterCurrent);
        on('btn-force-abort-queue', 'click', forceAbortQueue);
        on('btn-cancel-waiting-queue', 'click', cancelWaitingQueueItems);
        on('btn-clear-completed-queue', 'click', clearCompletedQueueItems);
        on('btn-clear-canceled-queue', 'click', clearCanceledQueueItems);
        document.querySelectorAll('.training-queue-filter').forEach((btn) => {
            btn.addEventListener('click', () => {
                const nextFilter = btn.dataset.queueFilter || 'actionable';
                setQueueFilter(state, nextFilter);
                finishQueueFeedback(`已切换到「${queueFilterFeedbackLabel(nextFilter)}」视图`, 'info');
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
        abortQueueAfterCurrent,
        forceAbortQueue,
        clearCompletedQueueItems,
        clearCanceledQueueItems,
        updateTrainingQueueSettings,
        toggleTrainingQueuePause,
        bindQueueEvents,
    };

    function queueMoveDirectionLabel(direction) {
        return {
            top: '置顶',
            up: '上移',
            down: '下移',
            bottom: '置底',
        }[direction] || '移动';
    }

    function queueFilterFeedbackLabel(value) {
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
}
