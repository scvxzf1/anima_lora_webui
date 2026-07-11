/**
 * Training queue renderer facade.
 * Labels: render-labels.js · Item builders: render-items.js
 */
import { queueManagerSections, queueSummaryCounts } from './state.js?v=module-bootstrap-20260711-ir6';
import {
    queueEmptyStateText,
    queueFailurePolicyLabel,
    queueFilterCount,
    queueFilterLabel,
    queueManagerStatusText,
} from './render-labels.js?v=module-bootstrap-20260711-ir6';
import { createQueueItemRenderers } from './render-items.js?v=module-bootstrap-20260711-ir6';

export function createQueueRenderer({ state, deps, actions }) {
    const items = createQueueItemRenderers({ state, deps, actions });
    const queueItemTitle = items.queueItemTitle;

    function renderTrainingQueue() {
        renderTrainingQueueSummary();
        renderTrainingQueueManager();
        deps.renderTrainingViewMode();
    }
    function renderTrainingQueueSummary() {
        const list = document.getElementById('training-queue-list');
        const summary = document.getElementById('training-queue-summary');
        const pauseBtn = document.getElementById('btn-toggle-queue-pause');
        const managerPauseBtn = document.getElementById('btn-manager-toggle-queue-pause');
        const refreshBtn = document.getElementById('btn-refresh-queue');
        const managerRefreshBtn = document.getElementById('btn-manager-refresh-queue');
        const badge = document.getElementById('training-queue-tab-badge');
        const cancelAllBtn = document.getElementById('btn-cancel-all-queue');
        const abortAfterCurrentBtn = document.getElementById('btn-abort-queue-after-current');
        const forceAbortBtn = document.getElementById('btn-force-abort-queue');
        const cancelWaitingBtn = document.getElementById('btn-cancel-waiting-queue');
        const clearCompletedBtn = document.getElementById('btn-clear-completed-queue');
        const clearCanceledBtn = document.getElementById('btn-clear-canceled-queue');
        if (!list || !summary) return;
        list.innerHTML = '';
        const counts = queueSummaryCounts(state);
        const activeWorkCount = counts.queued + counts.running + (items.queueBackendRunning() && !counts.running ? 1 : 0);
        const running = state.queue.items.find((item) => item.state === 'running');
        const nextQueued = state.queue.items.find((item) => item.state === 'queued');
        summary.className = [
            'training-queue-summary',
            state.queue.paused ? 'paused' : '',
            running ? 'running' : '',
            counts.error ? 'error' : '',
        ].filter(Boolean).join(' ');
        if (state.queue.loading) {
            summary.textContent = '正在读取队列...';
        } else if (state.queue.error) {
            summary.textContent = state.queue.error;
        } else if (running) {
            summary.textContent = `正在运行：${items.queueItemTitle(running)} · 等待 ${counts.queued} 个`;
        } else if (counts.queued) {
            summary.textContent = state.queue.paused
                ? `队列已暂停 · 等待 ${counts.queued} 个任务`
                : `空闲时会自动启动 · 等待 ${counts.queued} 个任务`;
        } else if (counts.error) {
            summary.textContent = `有 ${counts.error} 个异常任务，队列${state.queue.paused ? '已暂停' : '可继续'}。`;
        } else {
            summary.textContent = state.queue.paused ? '队列已暂停，暂无等待任务。' : '暂无等待任务。';
        }
        for (const btn of [pauseBtn, managerPauseBtn]) {
            if (!btn) continue;
            btn.textContent = state.queue.paused && counts.queued ? '继续队列' : (state.queue.paused ? '继续' : '暂停');
            btn.disabled = state.queue.loading;
            updateQueueStaticActionButton(btn, 'settings-paused', btn.textContent, '处理中');
        }
        updateQueueStaticActionButton(refreshBtn, 'refresh', '刷新', '刷新中');
        updateQueueStaticActionButton(managerRefreshBtn, 'refresh', '刷新', '刷新中');
        if (cancelAllBtn) cancelAllBtn.disabled = state.queue.loading || (counts.queued + counts.running) <= 0;
        if (abortAfterCurrentBtn) abortAfterCurrentBtn.disabled = state.queue.loading || counts.queued <= 0;
        if (forceAbortBtn) forceAbortBtn.disabled = state.queue.loading || activeWorkCount <= 0;
        if (cancelWaitingBtn) cancelWaitingBtn.disabled = state.queue.loading || counts.queued <= 0;
        if (clearCompletedBtn) clearCompletedBtn.disabled = state.queue.loading || counts.done <= 0;
        if (clearCanceledBtn) clearCanceledBtn.disabled = state.queue.loading || counts.canceled <= 0;
        updateQueueActionHints({
            cancelAllBtn,
            abortAfterCurrentBtn,
            forceAbortBtn,
            cancelWaitingBtn,
            clearCompletedBtn,
            clearCanceledBtn,
            counts,
            activeWorkCount,
        });
        if (badge) {
            const active = counts.queued + counts.running;
            badge.hidden = active <= 0;
            badge.textContent = String(active);
        }
        const visible = [running, nextQueued].filter(Boolean);
        if (!visible.length) {
            const empty = document.createElement('div');
            empty.className = 'task-history-empty';
            empty.textContent = counts.total ? '没有正在运行或等待的任务。' : '从配置页开始训练时，可以选择加入队列。';
            list.appendChild(empty);
            return;
        }
        for (const item of visible) {
            list.appendChild(items.createTrainingQueueItem(item));
        }
    }
    function renderTrainingQueueManager() {
        const status = document.getElementById('training-queue-manager-status');
        const overview = document.getElementById('training-queue-manager-overview');
        const feedback = document.getElementById('training-queue-feedback');
        const stats = document.getElementById('training-queue-stats');
        const list = document.getElementById('training-queue-manager-list');
        const policy = document.getElementById('training-queue-failure-policy');
        if (!status || !stats || !list) return;
        const counts = queueSummaryCounts(state);
        status.textContent = state.queue.loading
            ? '正在读取队列...'
            : state.queue.error
                ? state.queue.error
                : queueManagerStatusText(counts, state);
        renderQueueManagerOverview(overview, counts);
        renderQueueFeedback(feedback);
        if (policy && policy.value !== state.queue.failurePolicy) {
            policy.value = state.queue.failurePolicy || 'pause';
        }
        if (policy) {
            const policyBusy = items.queueFeedbackBusyAction('settings-policy');
            policy.disabled = policyBusy;
            policy.setAttribute('aria-busy', policyBusy ? 'true' : 'false');
            policy.title = policyBusy ? '正在保存失败处理策略' : '设置队列失败后的处理方式';
        }
        stats.innerHTML = '';
        [
            ['运行中', counts.running, 'running'],
            ['等待中', counts.queued, 'queued'],
            ['异常', counts.error, 'error'],
            ['完成', counts.done, 'done'],
            ['已取消', counts.canceled, 'canceled'],
            ['总计', counts.total, 'total'],
        ].forEach(([label, value, itemState]) => {
            const item = document.createElement('div');
            item.className = ['training-queue-stat', itemState, itemState === 'error' && value > 0 ? 'active-alert' : ''].filter(Boolean).join(' ');
            item.title = `${label}: ${value}`;
            item.setAttribute('aria-label', `${label}: ${value}`);
            const valueEl = document.createElement('strong');
            valueEl.textContent = String(value);
            const labelEl = document.createElement('span');
            labelEl.textContent = label;
            item.append(valueEl, labelEl);
            stats.appendChild(item);
        });
        document.querySelectorAll('.training-queue-filter').forEach((btn) => {
            updateQueueFilterButton(btn, counts);
        });
        list.innerHTML = '';
        const sections = queueManagerSections(state);
        const visibleCount = sections.reduce((sum, section) => sum + section.items.length, 0);
        if (!visibleCount) {
            const empty = document.createElement('div');
            empty.className = 'training-queue-manager-empty';
            empty.textContent = queueEmptyStateText(state.filter, counts);
            list.appendChild(empty);
            return;
        }
        for (const section of sections) {
            if (!section.items.length) continue;
            list.appendChild(items.createTrainingQueueSection(section));
        }
    }
    function updateQueueActionHints({ cancelAllBtn, abortAfterCurrentBtn, forceAbortBtn, cancelWaitingBtn, clearCompletedBtn, clearCanceledBtn, counts, activeWorkCount }) {
        const pairs = [
            [cancelAllBtn, counts.queued + counts.running, 'cancel-all', '取消所有等待和运行中的队列任务', '没有可取消的运行或等待任务'],
            [abortAfterCurrentBtn, counts.queued, 'abort-after-current', '当前任务完成后不再继续，取消所有等待任务', '没有等待中的后续任务'],
            [forceAbortBtn, activeWorkCount, 'force-abort', '立即停止运行中任务并取消所有等待任务', '没有可中止的运行或等待任务'],
            [cancelWaitingBtn, counts.queued, 'cancel-waiting', '取消所有尚未启动的等待任务', '没有等待中的任务'],
            [clearCompletedBtn, counts.done, 'clear-done', '清理已完成队列记录，不删除训练文件', '没有已完成记录可清理'],
            [clearCanceledBtn, counts.canceled, 'clear-canceled', '清理已取消队列记录，不删除训练文件', '没有已取消记录可清理'],
        ];
        for (const [btn, count, action, enabledTitle, disabledTitle] of pairs) {
            if (!btn) continue;
            const busy = items.queueFeedbackBusyAction(action);
            if (busy) btn.disabled = true;
            btn.classList.toggle('queue-action-busy', busy);
            btn.setAttribute('aria-busy', busy ? 'true' : 'false');
            btn.title = busy ? '请求已发送，正在等待队列返回' : state.queue.loading ? '正在读取队列状态' : (count > 0 ? enabledTitle : disabledTitle);
            btn.setAttribute('aria-disabled', btn.disabled ? 'true' : 'false');
        }
    }
    function updateQueueStaticActionButton(btn, action, defaultLabel, busyLabel) {
        if (!btn) return;
        const busy = items.queueFeedbackBusyAction(action);
        btn.textContent = busy ? busyLabel : defaultLabel;
        if (busy) btn.disabled = true;
        btn.classList.toggle('queue-action-busy', busy);
        btn.setAttribute('aria-busy', busy ? 'true' : 'false');
        btn.setAttribute('aria-disabled', btn.disabled ? 'true' : 'false');
    }
    function renderQueueFeedback(feedback) {
        if (!feedback) return;
        const current = state.feedback || {};
        const message = current.message || '';
        feedback.hidden = !message;
        feedback.textContent = message;
        feedback.dataset.feedbackTone = current.tone || '';
        feedback.setAttribute('aria-busy', current.busyAction ? 'true' : 'false');
    }
    function updateQueueFilterButton(btn, counts) {
        const key = btn.dataset.queueFilter || 'actionable';
        const active = key === state.filter;
        const count = queueFilterCount(key, counts);
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        btn.classList.toggle('queue-filter-feedback', state.feedback?.busyAction === `filter-${key}`);
        btn.setAttribute('aria-busy', state.feedback?.busyAction === `filter-${key}` ? 'true' : 'false');
        btn.title = `${queueFilterLabel(key)}：${count} 项`;
        btn.innerHTML = '';
        const label = document.createElement('span');
        label.textContent = queueFilterLabel(key);
        const badge = document.createElement('b');
        badge.textContent = String(count);
        btn.append(label, badge);
    }
    function renderQueueManagerOverview(overview, counts) {
        if (!overview) return;
        overview.innerHTML = '';
        const running = state.queue.items.find((item) => item.state === 'running');
        const nextQueued = state.queue.items.find((item) => item.state === 'queued');
        const policy = queueFailurePolicyLabel(state.queue.failurePolicy);
        const activeText = running
            ? `正在运行：${items.queueItemTitle(running)}`
            : nextQueued
                ? `下一项：${items.queueItemTitle(nextQueued)}`
                : counts.total
                    ? '当前没有可执行任务'
                    : '队列为空';
        const filterText = `视图：${queueFilterLabel(state.filter)} · ${counts.total} 项`;
        const riskText = counts.error
            ? `${counts.error} 个异常 · 失败后${policy}`
            : `无异常 · 失败后${policy}`;
        [
            [state.queue.paused ? '队列暂停' : running ? '自动执行中' : '调度就绪', activeText, running ? 'running' : state.queue.paused ? 'paused' : 'ready'],
            ['筛选范围', filterText, 'filter'],
            ['异常处理', riskText, counts.error ? 'error' : 'policy'],
        ].forEach(([label, value, tone]) => {
            const item = document.createElement('div');
            item.className = ['training-queue-overview-item', tone].join(' ');
            item.title = `${label}: ${value}`;
            const labelEl = document.createElement('span');
            labelEl.textContent = label;
            const valueEl = document.createElement('strong');
            valueEl.textContent = value;
            valueEl.title = value;
            item.append(labelEl, valueEl);
            overview.appendChild(item);
        });
    }
    return {
        renderTrainingQueue,
        updateRunningQueueProgress: items.updateRunningQueueProgress,
        queueItemTitle,
    };
}
