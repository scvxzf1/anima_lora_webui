/* Full training queue controller backed by the existing queue control routes. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { connectWebSocket, disconnectWebSocket, onClose, onMessage, onOpen } from '../ws.js?v=dragon-ui-20260812v35';
import { normalizeQueueSnapshot, queueItemTitle, renderQueuePage } from './queue-view.js?v=dragon-ui-20260814v43';

const api = createApiClient();
const ACTIVE_FALLBACK_INTERVAL_MS = 5000;
const IDLE_FALLBACK_INTERVAL_MS = 60000;

export async function loadQueue() {
    const initial = await readQueueSnapshot();
    const state = {
        model: initial.model,
        draft: createPolicyDraft(initial.model),
        filter: readQueueFilter(),
        feedback: initial.error
            ? { message: `${initial.error} 请检查 WebUI 服务后重试。`, tone: 'error' }
            : { message: '', tone: '' },
        busy: false,
        settingsDirty: false,
        hasRendered: false,
        root: null,
    };
    let cleanup = null;
    return {
        html: renderQueuePage(state.model, state),
        onMount: (root) => { cleanup = mountQueue(root, state); },
        beforeLeave: () => !state.settingsDirty || window.confirm('队列策略尚未保存，确认离开吗？'),
        onUnmount: () => cleanup?.(),
    };
}

async function readQueueSnapshot() {
    try {
        const payload = await api('/api/training/queue');
        if (payload.ok === false) throw new Error(payload.error || '读取训练队列失败');
        return { model: normalizeQueueSnapshot(payload), error: '' };
    } catch (error) {
        return { model: normalizeQueueSnapshot({}), error: error.message || '读取训练队列失败' };
    }
}

function mountQueue(root, state) {
    state.root = root;
    state.hasRendered = true;
    const clickHandler = (event) => handleQueueClick(event, state);
    const submitHandler = (event) => handleQueueSubmit(event, state);
    const settingsHandler = (event) => capturePolicyDraft(event.target, state);
    root.addEventListener('click', clickHandler);
    root.addEventListener('submit', submitHandler);
    root.addEventListener('input', settingsHandler);
    root.addEventListener('change', settingsHandler);

    let refreshTimer = null;
    let wsConnected = false;
    const scheduleRefresh = () => {
        if (refreshTimer) window.clearTimeout(refreshTimer);
        refreshTimer = null;
        if (document.hidden || !state.root) return;
        const active = Number(state.model.summary?.queued || 0) > 0
            || Number(state.model.summary?.running || 0) > 0;
        const delay = wsConnected || !active
            ? IDLE_FALLBACK_INTERVAL_MS
            : ACTIVE_FALLBACK_INTERVAL_MS;
        refreshTimer = window.setTimeout(async () => {
            await refreshQueue(state, { quiet: true });
            scheduleRefresh();
        }, delay);
    };
    const applyQueueEvent = (payload) => {
        if (state.busy || state.settingsDirty || !state.root) return;
        const nextModel = normalizeQueueSnapshot(payload);
        if (queueSnapshotsEqual(state.model, nextModel)) return;
        state.model = nextModel;
        state.draft = createPolicyDraft(nextModel);
        renderQueue(state);
        scheduleRefresh();
    };
    const subscriptions = [
        onMessage('queue', applyQueueEvent),
        onOpen(() => { wsConnected = true; scheduleRefresh(); }),
        onClose((event = {}) => {
            if (event.intentional) return;
            wsConnected = false;
            scheduleRefresh();
        }),
    ];
    const visibilityHandler = () => {
        if (document.hidden) {
            if (refreshTimer) window.clearTimeout(refreshTimer);
            refreshTimer = null;
            return;
        }
        refreshQueue(state, { quiet: true }).finally(scheduleRefresh);
    };
    document.addEventListener('visibilitychange', visibilityHandler);
    connectWebSocket();
    scheduleRefresh();
    return () => {
        if (refreshTimer) window.clearTimeout(refreshTimer);
        subscriptions.forEach((unsubscribe) => unsubscribe());
        disconnectWebSocket();
        document.removeEventListener('visibilitychange', visibilityHandler);
        root.removeEventListener('click', clickHandler);
        root.removeEventListener('submit', submitHandler);
        root.removeEventListener('input', settingsHandler);
        root.removeEventListener('change', settingsHandler);
        state.root = null;
    };
}

async function handleQueueClick(event, state) {
    const filterButton = event.target.closest('[data-queue-filter]');
    if (filterButton) {
        state.filter = filterButton.dataset.queueFilter || 'active';
        writeQueueFilter(state.filter);
        renderQueue(state);
        return;
    }

    const itemButton = event.target.closest('[data-item-action]');
    if (itemButton) {
        await handleItemAction(itemButton, state);
        return;
    }

    const actionButton = event.target.closest('[data-queue-action]');
    if (!actionButton) return;
    await handleQueueAction(actionButton.dataset.queueAction, state);
}

async function handleQueueSubmit(event, state) {
    const form = event.target.closest('[data-queue-settings-form]');
    if (!form) return;
    event.preventDefault();
    if (state.busy) return;
    capturePolicyDraft(form, state);
    const payload = {
        failure_policy: state.draft.failure_policy,
        auto_retry: state.draft.auto_retry,
        max_attempts: Number(state.draft.max_attempts),
        retry_backoff_sec: Number(state.draft.retry_backoff_sec),
    };
    if (!Number.isInteger(payload.max_attempts) || payload.max_attempts < 1 || payload.max_attempts > 10) {
        setFeedback(state, '最大尝试次数必须是 1–10 的整数。', 'error');
        return;
    }
    if (!Number.isFinite(payload.retry_backoff_sec) || payload.retry_backoff_sec < 0 || payload.retry_backoff_sec > 3600) {
        setFeedback(state, '重试等待必须在 0–3600 秒之间。', 'error');
        return;
    }
    await performRequest(state, {
        message: '正在保存队列策略…',
        success: '队列策略已保存。',
        request: () => api('/api/training/queue/settings', { method: 'POST', body: JSON.stringify(payload) }),
        onSuccess: () => {
            state.settingsDirty = false;
            state.draft = createPolicyDraft(state.model);
        },
    });
}

async function handleQueueAction(action, state) {
    if (state.busy) return;
    if (action === 'refresh') {
        await refreshQueue(state);
        return;
    }
    if (action === 'toggle-pause') {
        await performRequest(state, {
            message: state.model.paused ? '正在继续队列…' : '正在暂停队列…',
            success: state.model.paused ? '队列已继续。' : '队列已暂停。',
            request: () => api('/api/training/queue/settings', {
                method: 'POST',
                body: JSON.stringify({ paused: !state.model.paused }),
            }),
        });
        return;
    }

    const operation = bulkOperation(action, state.model);
    if (!operation) return;
    if (!window.confirm(operation.confirm)) return;
    await performRequest(state, operation);
}

async function handleItemAction(button, state) {
    if (state.busy) return;
    const itemElement = button.closest('[data-item-id]');
    const itemId = itemElement?.dataset.itemId;
    const item = state.model.items.find((entry) => String(entry.id || '') === itemId);
    if (!itemId || !item) return;
    const action = button.dataset.itemAction;

    if (action === 'move') {
        const direction = button.dataset.direction;
        await performRequest(state, {
            message: `正在调整顺序：${queueItemTitle(item)}…`,
            success: `已调整顺序：${queueItemTitle(item)}。`,
            request: () => api(`/api/training/queue/${encodeURIComponent(itemId)}/move`, {
                method: 'POST', body: JSON.stringify({ direction }),
            }),
        });
        return;
    }

    if (action === 'retry') {
        const confirmed = window.confirm(`重新加入“${queueItemTitle(item)}”吗？\n\n将从该任务冻结的运行配置克隆新任务，不读取当前已修改的源 TOML。`);
        if (!confirmed) return;
        await performRequest(state, {
            message: `正在重新入队：${queueItemTitle(item)}…`,
            success: '新任务已加入队列。',
            request: () => api(`/api/training/queue/${encodeURIComponent(itemId)}/retry`, { method: 'POST' }),
        });
        return;
    }

    const running = item.state === 'running';
    const terminal = ['done', 'error', 'canceled'].includes(item.state);
    const prompt = terminal
        ? `将“${queueItemTitle(item)}”移出队列列表吗？\n\n只删除队列记录，不删除训练目录、日志、权重或历史任务。`
        : running
            ? `立即停止“${queueItemTitle(item)}”吗？\n\n已生成的运行文件会保留。`
            : `取消等待任务“${queueItemTitle(item)}”吗？\n\n已创建的运行目录会保留。`;
    if (!window.confirm(prompt)) return;
    await performRequest(state, {
        message: terminal ? '正在移出队列记录…' : (running ? '正在停止任务…' : '正在取消任务…'),
        success: terminal ? '队列记录已移出，运行文件保持不变。' : (running ? '停止请求已发送。' : '等待任务已取消。'),
        request: () => api(`/api/training/queue/${encodeURIComponent(itemId)}`, { method: 'DELETE' }),
    });
}

function bulkOperation(action, model) {
    const active = model.summary.queued + model.summary.running + (model.status === 'running' && !model.summary.running ? 1 : 0);
    const operations = {
        'cancel-waiting': {
            confirm: `取消全部 ${model.summary.queued} 个等待任务吗？\n\n运行中任务和所有训练文件都会保留。`,
            message: '正在取消全部等待任务…', success: '等待任务已取消。', endpoint: 'cancel-waiting',
        },
        'cancel-all': {
            confirm: `取消全部 ${active} 个运行或等待任务吗？\n\n当前训练会停止，队列会暂停；训练文件不会删除。`,
            message: '正在取消全部队列任务…', success: '队列取消请求已完成。', endpoint: 'cancel-all',
        },
        'abort-after-current': {
            confirm: `当前任务完成后停止队列，并取消 ${model.summary.queued} 个后续等待任务吗？`,
            message: '正在中止后续队列…', success: '后续队列已中止，当前任务不受影响。', endpoint: 'abort-after-current',
        },
        'force-abort': {
            confirm: `立即强制中止 ${active} 个运行或等待任务吗？\n\n当前训练会立即停止，队列保持暂停；训练文件不会删除。`,
            message: '正在强制中止队列…', success: '队列已强制中止。', endpoint: 'force-abort',
        },
        'clear-completed': {
            confirm: `清理 ${model.summary.done} 条已完成队列记录吗？\n\n训练历史、运行目录、日志和权重不会删除。`,
            message: '正在清理已完成记录…', success: '已完成记录已清理。', endpoint: 'clear-completed',
        },
        'clear-canceled': {
            confirm: `清理 ${model.summary.canceled} 条已取消队列记录吗？\n\n训练历史、运行目录、日志和权重不会删除。`,
            message: '正在清理已取消记录…', success: '已取消记录已清理。', endpoint: 'clear-canceled',
        },
    };
    const operation = operations[action];
    if (!operation) return null;
    return {
        ...operation,
        request: () => api(`/api/training/queue/${operation.endpoint}`, { method: 'POST' }),
    };
}

async function performRequest(state, options) {
    state.busy = true;
    setFeedback(state, options.message, 'busy');
    try {
        const payload = await options.request();
        if (payload.ok === false) throw new Error(payload.error || '队列操作失败');
        state.model = normalizeQueueSnapshot(payload);
        options.onSuccess?.(payload);
        if (!state.settingsDirty) state.draft = createPolicyDraft(state.model);
        state.feedback = { message: payload.message || options.success, tone: 'success' };
    } catch (error) {
        state.feedback = { message: `${error.message || '队列操作失败'} 请检查当前队列状态后重试。`, tone: 'error' };
    } finally {
        state.busy = false;
        renderQueue(state);
    }
}

async function refreshQueue(state, options = {}) {
    if (state.busy || !state.root || !document.contains(state.root)) return;
    if (options.quiet && state.settingsDirty) return;
    try {
        const payload = await api('/api/training/queue');
        if (payload.ok === false) throw new Error(payload.error || '刷新训练队列失败');
        const nextModel = normalizeQueueSnapshot(payload);
        // The background poll is deliberately silent when the server has not
        // changed anything. Replacing the whole queue tree here would reset
        // focus/scroll and replay the page entrance animation every 5 seconds.
        if (options.quiet && queueSnapshotsEqual(state.model, nextModel)) return;
        state.model = nextModel;
        if (!state.settingsDirty) state.draft = createPolicyDraft(state.model);
        if (!options.quiet) state.feedback = { message: '队列状态已刷新。', tone: 'success' };
        renderQueue(state);
    } catch (error) {
        if (!options.quiet) {
            state.feedback = { message: `${error.message || '刷新训练队列失败'} 请稍后重试。`, tone: 'error' };
            renderQueue(state);
        }
    }
}

function setFeedback(state, message, tone) {
    state.feedback = { message, tone };
    renderQueue(state);
}

function renderQueue(state) {
    if (!state.root || !document.contains(state.root)) return;
    const scrollTop = window.scrollY;
    const active = document.activeElement;
    const focusKey = active && state.root.contains(active)
        ? active.getAttribute('name') || active.getAttribute('data-queue-filter') || active.getAttribute('data-queue-action')
        : '';
    if (state.hasRendered) state.root.dataset.queueLiveUpdate = 'true';
    state.root.innerHTML = renderQueuePage(state.model, state);
    state.root.querySelectorAll('.dragon-reveal').forEach((element) => {
        element.classList.add('dragon-in-view');
    });
    if (state.busy) {
        state.root.querySelectorAll('button, input, select').forEach((control) => { control.disabled = true; });
    }
    window.scrollTo({ top: scrollTop, behavior: 'auto' });
    if (focusKey) {
        const selector = `[name="${CSS.escape(focusKey)}"], [data-queue-filter="${CSS.escape(focusKey)}"], [data-queue-action="${CSS.escape(focusKey)}"]`;
        state.root.querySelector(selector)?.focus({ preventScroll: true });
    }
    state.hasRendered = true;
}

function queueSnapshotsEqual(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
}

function capturePolicyDraft(target, state) {
    const form = target.closest?.('[data-queue-settings-form]');
    if (!form) return;
    const formData = new FormData(form);
    state.draft = {
        failure_policy: String(formData.get('failure_policy') || 'pause'),
        auto_retry: formData.get('auto_retry') === 'on',
        max_attempts: String(formData.get('max_attempts') ?? ''),
        retry_backoff_sec: String(formData.get('retry_backoff_sec') ?? ''),
    };
    state.settingsDirty = true;
}

function createPolicyDraft(model) {
    return {
        failure_policy: model.failure_policy,
        auto_retry: model.auto_retry,
        max_attempts: String(model.max_attempts),
        retry_backoff_sec: String(model.retry_backoff_sec),
    };
}

function readQueueFilter() {
    const params = new URLSearchParams(window.location.search);
    const value = params.get('queue_filter');
    return ['active', 'all', 'queued', 'running', 'error', 'done', 'canceled'].includes(value) ? value : 'active';
}

function writeQueueFilter(filter) {
    const url = new URL(window.location.href);
    if (filter === 'active') url.searchParams.delete('queue_filter');
    else url.searchParams.set('queue_filter', filter);
    window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
}
