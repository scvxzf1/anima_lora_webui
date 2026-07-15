/**
 * Training queue API wrappers and training-view mode helpers.
 */
import { ensureQueueFeature } from '../anima-app/helpers/feature-ensurers.js?v=module-bootstrap-20260714-stage-dataset5';
import { returnToLiveTraining } from '../anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { renderHistoryManager } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

const trainingState = getTrainingState();

export async function loadTrainingQueue() {
    return ensureQueueFeature().loadTrainingQueue();
}

export function updateTrainingQueueFromPayload(payload = {}) {
    return ensureQueueFeature().updateTrainingQueueFromPayload(payload);
}

export function renderTrainingQueue() {
    return ensureQueueFeature().renderTrainingQueue();
}

export function refreshQueueRunningProgressViews() {
    return ensureQueueFeature().updateRunningQueueProgress();
}

export function showTrainingView(mode) {
    trainingState.trainingViewMode = ['live', 'queue', 'history'].includes(mode) ? mode : 'live';
    renderTrainingViewMode();
}

export function trainingViewTabs() {
    return Array.from(document.querySelectorAll('#tab-training .training-view-tab'));
}

export function focusTrainingViewTab(mode = trainingState.trainingViewMode) {
    const target = trainingViewTabs().find((btn) => btn.dataset.trainingView === mode);
    target?.focus({ preventScroll: true });
}

export function activateTrainingViewTabButton(button) {
    const nextMode = button?.dataset.trainingView || 'live';
    if (nextMode === 'live' && typeof returnToLiveTraining === 'function') {
        returnToLiveTraining({ refresh: false });
    } else {
        showTrainingView(nextMode);
    }
    focusTrainingViewTab(nextMode);
}

export function moveTrainingViewTabFocus(currentButton, offset = 0) {
    const tabs = trainingViewTabs();
    if (!tabs.length) return;
    const currentIndex = Math.max(0, tabs.indexOf(currentButton));
    const nextIndex = (currentIndex + offset + tabs.length) % tabs.length;
    activateTrainingViewTabButton(tabs[nextIndex]);
}

export function bindTrainingViewTabKeyboard() {
    renderTrainingViewMode();
    trainingViewTabs().forEach((btn) => {
        if (btn.dataset.trainingKeyboardBound === '1') return;
        btn.dataset.trainingKeyboardBound = '1';
        btn.addEventListener('keydown', (event) => {
            const key = event.key;
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key)) return;
            event.preventDefault();
            const tabs = trainingViewTabs();
            if (!tabs.length) return;
            if (key === 'Home') return activateTrainingViewTabButton(tabs[0]);
            if (key === 'End') return activateTrainingViewTabButton(tabs[tabs.length - 1]);
            moveTrainingViewTabFocus(btn, key === 'ArrowRight' ? 1 : -1);
        });
    });
}

export function renderTrainingViewMode() {
    const queueView = document.getElementById('training-queue-manager');
    const monitorView = document.getElementById('training-monitor-view');
    const historyManager = document.getElementById('training-history-manager');
    const historyPlaceholder = document.getElementById('training-history-placeholder');
    const workspace = document.querySelector('#tab-training .training-workspace');
    const isQueue = trainingState.trainingViewMode === 'queue';
    const isHistory = trainingState.trainingViewMode === 'history';
    const mainWide = isQueue || isHistory;
    if (queueView) queueView.hidden = !isQueue;
    if (historyManager) historyManager.hidden = !isHistory;
    if (monitorView) monitorView.hidden = isQueue || isHistory;
    if (historyPlaceholder) historyPlaceholder.hidden = true;
    const trainingRoot = document.getElementById('tab-training');
    if (trainingRoot) {
        trainingRoot.classList.toggle('history-mode', isHistory);
        trainingRoot.classList.toggle('queue-mode', isQueue);
        trainingRoot.classList.toggle('live-mode', !isQueue && !isHistory);
    }
    if (workspace) {
        workspace.classList.toggle('main-wide', mainWide);
        workspace.classList.toggle('history-mode', isHistory);
    }
    document.querySelectorAll('.training-view-tab').forEach((btn) => {
        const active = btn.dataset.trainingView === trainingState.trainingViewMode;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-selected', String(active));
        btn.tabIndex = active ? 0 : -1;
    });
    if (isHistory) {
        renderHistoryManager();
    }
}
