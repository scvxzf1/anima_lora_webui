const HISTORY_RETURN_STATE_KEY = 'dragonHistoryReturn';

const DEFAULT_HISTORY_RETURN = Object.freeze({
    hash: '#history',
    label: '返回历史',
    icon: 'history',
});

let activeHistoryReturn = null;

export function trackHistoryDetailEntry(previousHash, nextHash) {
    const nextTaskId = historyDetailTaskId(nextHash);
    if (!nextTaskId) {
        activeHistoryReturn = null;
        return;
    }

    const previousTaskId = historyDetailTaskId(previousHash);
    const stored = storedHistoryReturn(nextTaskId);
    const navigation = previousTaskId
        ? (activeHistoryReturn?.taskId === nextTaskId ? activeHistoryReturn.navigation : stored)
        : navigationForSource(previousHash);

    activeHistoryReturn = { taskId: nextTaskId, navigation };
    persistHistoryReturn(nextTaskId, navigation);
}

export function resolveHistoryReturnNavigation(taskId) {
    const normalizedTaskId = String(taskId || '');
    if (activeHistoryReturn?.taskId === normalizedTaskId) return activeHistoryReturn.navigation;
    return storedHistoryReturn(normalizedTaskId);
}

function navigationForSource(sourceHash) {
    const hash = normalizedSourceHash(sourceHash);
    if (hash === '#history') return DEFAULT_HISTORY_RETURN;
    if (hash === '#page/live-training' || hash === '#live-training') {
        return { hash, label: '返回当前监控', icon: 'activity' };
    }
    if (hash === '#page/queue' || hash === '#queue') {
        return { hash, label: '返回任务队列', icon: 'list' };
    }
    if (hash === '#dashboard' || hash === '') {
        return { hash: hash || '#dashboard', label: '返回首页', icon: 'home' };
    }
    return { hash, label: '返回上一页', icon: 'history' };
}

function historyDetailTaskId(hash) {
    const parts = String(hash || '').replace(/^#/, '').split('/');
    if (parts[0] !== 'history' || !parts[1]) return '';
    try {
        return decodeURIComponent(parts[1]);
    } catch {
        return parts[1];
    }
}

function normalizedSourceHash(hash) {
    const value = String(hash || '');
    if (!value || value === '#') return '';
    if (!value.startsWith('#') || historyDetailTaskId(value)) return '#history';
    return value;
}

function storedHistoryReturn(taskId) {
    const stored = window.history.state?.[HISTORY_RETURN_STATE_KEY];
    if (!stored || stored.taskId !== taskId) return DEFAULT_HISTORY_RETURN;
    if (typeof stored.hash !== 'string' || !stored.hash.startsWith('#') || historyDetailTaskId(stored.hash)) {
        return DEFAULT_HISTORY_RETURN;
    }
    return {
        hash: stored.hash,
        label: String(stored.label || DEFAULT_HISTORY_RETURN.label),
        icon: String(stored.icon || DEFAULT_HISTORY_RETURN.icon),
    };
}

function persistHistoryReturn(taskId, navigation) {
    const state = window.history.state && typeof window.history.state === 'object'
        ? window.history.state
        : {};
    window.history.replaceState({
        ...state,
        [HISTORY_RETURN_STATE_KEY]: { taskId, ...navigation },
    }, '', window.location.href);
}
