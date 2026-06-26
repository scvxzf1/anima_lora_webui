import { queueManagerSections, queueSummaryCounts } from './state.js?v=module-bootstrap-20260627-2';

export function createQueueRenderer({ state, deps, actions }) {
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
        const activeWorkCount = counts.queued + counts.running + (queueBackendRunning() && !counts.running ? 1 : 0);
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
            summary.textContent = `正在运行：${queueItemTitle(running)} · 等待 ${counts.queued} 个`;
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
            list.appendChild(createTrainingQueueItem(item));
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
                : queueManagerStatusText(counts);
        renderQueueManagerOverview(overview, counts);
        renderQueueFeedback(feedback);
        if (policy && policy.value !== state.queue.failurePolicy) {
            policy.value = state.queue.failurePolicy || 'pause';
        }
        if (policy) {
            const policyBusy = queueFeedbackBusyAction('settings-policy');
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
            list.appendChild(createTrainingQueueSection(section));
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
            const busy = queueFeedbackBusyAction(action);
            if (busy) btn.disabled = true;
            btn.classList.toggle('queue-action-busy', busy);
            btn.setAttribute('aria-busy', busy ? 'true' : 'false');
            btn.title = busy ? '请求已发送，正在等待队列返回' : state.queue.loading ? '正在读取队列状态' : (count > 0 ? enabledTitle : disabledTitle);
            btn.setAttribute('aria-disabled', btn.disabled ? 'true' : 'false');
        }
    }

    function updateQueueStaticActionButton(btn, action, defaultLabel, busyLabel) {
        if (!btn) return;
        const busy = queueFeedbackBusyAction(action);
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

    function queueFeedbackBusyAction(action) {
        return Boolean(action && state.feedback?.busyAction === action);
    }

    function queueFeedbackItemState(item) {
        if (!item?.id) return '';
        if (state.feedback?.busyItemId === item.id) return 'pending';
        if (state.feedback?.flashItemId === item.id) return state.feedback?.tone === 'error' ? 'failed' : 'updated';
        return '';
    }

    function queueBackendRunning() {
        return state.queue.status === 'running' || deps.getTrainingRuntime()?.state === 'running';
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

    function queueFilterCount(value, counts) {
        if (value === 'all') return counts.total;
        if (value === 'actionable') return counts.running + counts.queued + counts.error;
        return Number(counts[value] || 0);
    }

    function queueEmptyStateText(filter, counts) {
        if (!counts.total) return '队列为空。可以回到配置页，用「加入队列」把训练任务排进来。';
        if (filter === 'actionable') return '当前没有待处理任务。完成或取消记录可以切到「全部」「完成」「已取消」查看。';
        return `「${queueFilterLabel(filter)}」视图暂无任务。可以切换筛选或刷新队列状态。`;
    }

    function renderQueueManagerOverview(overview, counts) {
        if (!overview) return;
        overview.innerHTML = '';
        const running = state.queue.items.find((item) => item.state === 'running');
        const nextQueued = state.queue.items.find((item) => item.state === 'queued');
        const policy = queueFailurePolicyLabel(state.queue.failurePolicy);
        const activeText = running
            ? `正在运行：${queueItemTitle(running)}`
            : nextQueued
                ? `下一项：${queueItemTitle(nextQueued)}`
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

    function queueManagerStatusText(counts) {
        const policy = queueFailurePolicyLabel(state.queue.failurePolicy);
        if (state.queue.paused) return `队列已暂停 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
        if (counts.running) return `队列运行中 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
        if (counts.queued) return `空闲时会自动启动 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
        return counts.total ? `没有等待任务 · 异常 ${counts.error} 个 · 失败后${policy}` : `队列为空 · 失败后${policy}`;
    }

    function queueFilterLabel(value) {
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

    function queueDetailsSummaryText(item) {
        if (item?.state === 'error') return '查看异常上下文和任务路径';
        if (item?.state === 'running') return '查看运行配置和关联信息';
        if (item?.state === 'queued') return '查看排队配置和创建时间';
        return '查看任务记录和路径';
    }

    function queueFailurePolicyLabel(value) {
        return value === 'continue' ? '继续下一个' : '暂停队列';
    }

    function createTrainingQueueSection(section) {
        const node = document.createElement(section.collapsed ? 'details' : 'section');
        node.className = ['training-queue-section', section.key || '', section.collapsed ? 'collapsible' : ''].filter(Boolean).join(' ');
        if (section.collapsed) node.open = false;

        const title = document.createElement(section.collapsed ? 'summary' : 'div');
        title.className = 'training-queue-section-head';
        const label = document.createElement('strong');
        label.textContent = section.title;
        const count = document.createElement('span');
        count.textContent = `${section.items.length} 个`;
        title.append(label, count);
        node.appendChild(title);

        const description = document.createElement('p');
        description.className = 'training-queue-section-description';
        description.textContent = section.description;
        node.appendChild(description);

        const list = document.createElement('div');
        list.className = 'training-queue-section-list';
        for (const item of section.items) {
            list.appendChild(createTrainingQueueManagerItem(item));
        }
        node.appendChild(list);
        return node;
    }

    function createTrainingQueueItem(item) {
        const card = document.createElement('article');
        const feedbackState = queueFeedbackItemState(item);
        card.className = ['training-queue-item', item.state || 'queued', feedbackState ? `queue-feedback-${feedbackState}` : ''].filter(Boolean).join(' ');
        card.dataset.queueItemId = item.id || '';
        if (feedbackState) card.setAttribute('aria-busy', feedbackState === 'pending' ? 'true' : 'false');
        applyQueueProgressStyle(card, item);
        const itemState = document.createElement('span');
        itemState.className = ['training-queue-state', item.state || 'queued'].join(' ');
        itemState.textContent = queueStateLabel(item.state);
        const main = document.createElement('div');
        main.className = 'training-queue-main';
        const title = document.createElement('strong');
        title.textContent = queueItemTitle(item);
        const meta = createQueueMetaTags(item, { compact: true });
        const facts = createQueueFactRow(item, { compact: true });
        const message = document.createElement('em');
        message.textContent = [
            item.message || '',
            item.created_at_text ? `入队: ${item.created_at_text}` : '',
        ].filter(Boolean).join(' · ');
        main.append(title, meta);
        if (facts) main.appendChild(facts);
        const progress = createQueueRunningProgress(item);
        if (progress) main.appendChild(progress);
        main.appendChild(message);
        card.append(itemState, main);
        return card;
    }

    function createTrainingQueueManagerItem(item) {
        const card = document.createElement('article');
        const feedbackState = queueFeedbackItemState(item);
        card.className = ['training-queue-manager-item', item.state || 'queued', feedbackState ? `queue-feedback-${feedbackState}` : ''].filter(Boolean).join(' ');
        card.dataset.queueItemId = item.id || '';
        if (feedbackState) card.setAttribute('aria-busy', feedbackState === 'pending' ? 'true' : 'false');
        applyQueueProgressStyle(card, item);
        const head = document.createElement('div');
        head.className = 'training-queue-manager-item-head';
        const itemState = document.createElement('span');
        itemState.className = ['training-queue-state', item.state || 'queued'].join(' ');
        itemState.textContent = queueStateLabel(item.state);
        const titleWrap = document.createElement('div');
        titleWrap.className = 'training-queue-manager-title';
        const title = document.createElement('strong');
        title.textContent = queueItemTitle(item);
        const meta = createQueueMetaTags(item);
        const facts = createQueueFactRow(item);
        titleWrap.append(title, meta);
        if (facts) titleWrap.appendChild(facts);
        head.append(itemState, titleWrap);

        const actionsEl = document.createElement('div');
        actionsEl.className = 'training-queue-manager-item-actions';
        if (item.state === 'queued') {
            actionsEl.append(
                createQueueIconActionButton('置顶', '⇈', () => actions.moveQueueItem(item.id, 'top'), '', { action: `move-${item.id}-top`, itemId: item.id }),
                createQueueIconActionButton('上移', '↑', () => actions.moveQueueItem(item.id, 'up'), '', { action: `move-${item.id}-up`, itemId: item.id }),
                createQueueIconActionButton('下移', '↓', () => actions.moveQueueItem(item.id, 'down'), '', { action: `move-${item.id}-down`, itemId: item.id }),
                createQueueIconActionButton('置底', '⇊', () => actions.moveQueueItem(item.id, 'bottom'), '', { action: `move-${item.id}-bottom`, itemId: item.id }),
            );
            actionsEl.appendChild(createQueueItemMoreMenu([
                createQueueActionButton('取消等待', () => actions.cancelQueueItem(item.id), 'danger', { action: `cancel-${item.id}`, itemId: item.id, busyLabel: '取消中' }),
            ]));
        } else if (item.state === 'running') {
            actionsEl.append(createQueueActionButton('停止', () => actions.cancelQueueItem(item.id), 'danger queue-stop-action', { action: `cancel-${item.id}`, itemId: item.id, busyLabel: '停止中' }));
        } else {
            actionsEl.append(
                createQueueActionButton('重新入队', () => actions.retryQueueItem(item.id), '', { action: `retry-${item.id}`, itemId: item.id, busyLabel: '入队中' }),
                createQueueActionButton('移除列表', () => actions.removeQueueItemFromList(item.id), 'danger', { action: `remove-${item.id}`, itemId: item.id, busyLabel: '移除中' }),
            );
        }
        head.appendChild(actionsEl);
        const progress = createQueueRunningProgress(item);
        if (progress) titleWrap.appendChild(progress);

        const details = document.createElement('details');
        details.className = 'training-queue-details';
        const summary = document.createElement('summary');
        summary.textContent = item.message || queueDetailsSummaryText(item);
        details.appendChild(summary);
        const grid = document.createElement('div');
        grid.className = 'training-queue-detail-grid';
        [
            ['任务 ID', item.id],
            ['运行配置', item.runtime_config_file],
            ['源配置', item.source_config_file],
            ['创建时间', item.created_at_text],
            ['开始时间', item.started_at_text],
            ['结束时间', item.finished_at_text],
            ['关联历史', queueHistoryLabel(item)],
            ['重试来源', item.retry_of || ''],
        ].forEach(([label, value]) => {
            if (!value) return;
            grid.appendChild(queueDetailRow(label, value));
        });
        details.appendChild(grid);
        card.append(head, details);
        return card;
    }

    function createQueueMetaTags(item, options = {}) {
        const meta = document.createElement('div');
        meta.className = ['training-queue-meta-tags', options.compact ? 'compact' : ''].filter(Boolean).join(' ');
        const parts = [
            ['类型', queueKindLabel(item), 'kind'],
            ['方法', item.methods_subdir || '-', 'method'],
            ['配置', item.variant || '-', 'variant'],
            ['预设', item.preset || 'default', 'preset'],
            ['GPU', queueGpuLabel(item.gpu_whitelist), 'gpu'],
            ['尝试', queueAttemptLabel(item), 'attempt'],
        ];
        for (const [label, value, tone] of parts) {
            if (!value) continue;
            if (options.compact && ['preset', 'attempt'].includes(tone)) continue;
            const tag = document.createElement('span');
            tag.className = ['training-queue-meta-tag', tone].join(' ');
            tag.textContent = `${label}: ${value}`;
            meta.appendChild(tag);
        }
        return meta;
    }

    function createQueueFactRow(item, options = {}) {
        const facts = [
            ['ID', queueShortId(item?.id)],
            ['创建', item?.created_at_text],
            ['开始', item?.started_at_text],
            ['结束', item?.finished_at_text],
            ['历史', queueHistoryLabel(item)],
            ['来源', deps.runLabelFromPath(item?.source_config_file || '')],
        ].filter(([, value]) => value);
        const visibleFacts = options.compact
            ? facts.filter(([label]) => !['ID', '历史', '来源'].includes(label)).slice(0, 2)
            : facts.slice(0, 5);
        if (!visibleFacts.length) return null;
        const row = document.createElement('div');
        row.className = ['training-queue-facts', options.compact ? 'compact' : ''].filter(Boolean).join(' ');
        for (const [label, value] of visibleFacts) {
            const fact = document.createElement('span');
            fact.className = 'training-queue-fact';
            const key = document.createElement('b');
            key.textContent = label;
            const val = document.createElement('span');
            val.textContent = String(value);
            fact.append(key, val);
            row.appendChild(fact);
        }
        return row;
    }

    function queueRunningProgress() {
        const runtime = deps.getTrainingRuntime();
        const current = Number(runtime.progressCurrent || 0);
        const total = Number(runtime.progressTotal || 0);
        if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) return null;
        const pct = Math.max(0, Math.min(100, current / total * 100));
        return { current, total, pct };
    }

    function applyQueueProgressStyle(card, item) {
        if (item?.state !== 'running') return;
        const progress = queueRunningProgress();
        if (!progress) return;
        card.style.setProperty('--queue-progress', `${progress.pct.toFixed(1)}%`);
    }

    function createQueueRunningProgress(item) {
        if (item?.state !== 'running') return null;
        const progress = queueRunningProgress();
        if (!progress) return null;
        const wrap = document.createElement('div');
        wrap.className = 'training-queue-running-progress';
        const text = document.createElement('span');
        wrap.appendChild(text);
        updateQueueRunningProgressElement(wrap, progress);
        return wrap;
    }

    function queueRunningProgressText(progress) {
        const runtime = deps.getTrainingRuntime();
        const label = runtime.progressLabel || 'Step';
        return [
            `${label}: ${progress.current}/${progress.total}`,
            `${progress.pct.toFixed(1)}%`,
            runtime.progressRate || '',
        ].filter(Boolean).join(' · ');
    }

    function updateQueueRunningProgressElement(el, progress) {
        el.style.setProperty('--queue-progress', `${progress.pct.toFixed(1)}%`);
        const text = el.querySelector('span') || document.createElement('span');
        text.textContent = queueRunningProgressText(progress);
        if (!text.parentNode) el.appendChild(text);
    }

    function updateRunningQueueProgress() {
        const progress = queueRunningProgress();
        document.querySelectorAll('#tab-training .training-queue-item.running, #tab-training .training-queue-manager-item.running').forEach((card) => {
            const progressEl = card.querySelector('.training-queue-running-progress');
            if (!progress) {
                card.style.removeProperty('--queue-progress');
                if (progressEl) progressEl.remove();
                return;
            }
            card.style.setProperty('--queue-progress', `${progress.pct.toFixed(1)}%`);
            if (progressEl) {
                updateQueueRunningProgressElement(progressEl, progress);
                return;
            }
            const next = createQueueRunningProgress({ state: 'running' });
            if (!next) return;
            const titleWrap = card.querySelector('.training-queue-manager-title');
            if (titleWrap) {
                titleWrap.appendChild(next);
                return;
            }
            const main = card.querySelector('.training-queue-main');
            const message = main?.querySelector('em');
            if (main && message) main.insertBefore(next, message);
            else if (main) main.appendChild(next);
        });
    }

    function queueDetailRow(label, value) {
        const row = document.createElement('div');
        const key = document.createElement('span');
        key.textContent = label;
        const valEl = document.createElement('code');
        valEl.textContent = String(value || '-');
        row.append(key, valEl);
        return row;
    }

    function createQueueActionButton(label, handler, tone = '', options = {}) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = ['task-history-action', tone].filter(Boolean).join(' ');
        const actionBusy = queueFeedbackBusyAction(options.action);
        const itemBusy = Boolean(options.itemId && state.feedback?.busyItemId === options.itemId);
        btn.textContent = actionBusy ? (options.busyLabel || '处理中') : label;
        btn.disabled = actionBusy || (itemBusy && !actionBusy);
        btn.classList.toggle('queue-action-busy', actionBusy);
        btn.setAttribute('aria-busy', actionBusy ? 'true' : 'false');
        btn.setAttribute('aria-disabled', btn.disabled ? 'true' : 'false');
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (btn.disabled) return;
            const menu = btn.closest('.training-queue-item-more, .training-queue-more-menu');
            if (menu instanceof HTMLDetailsElement) menu.open = false;
            handler(event);
        });
        return btn;
    }

    function createQueueIconActionButton(label, icon, handler, tone = '', options = {}) {
        const btn = createQueueActionButton(label, handler, ['queue-icon-action', tone].filter(Boolean).join(' '), { ...options, busyLabel: '...' });
        if (!queueFeedbackBusyAction(options.action)) btn.textContent = icon;
        btn.title = queueFeedbackBusyAction(options.action) ? `${label}处理中` : label;
        btn.setAttribute('aria-label', label);
        return btn;
    }

    function createQueueItemMoreMenu(actions) {
        const menu = document.createElement('details');
        menu.className = 'training-queue-item-more';
        const summary = document.createElement('summary');
        summary.className = 'task-history-action';
        summary.textContent = '⋯';
        summary.title = '更多操作';
        summary.setAttribute('aria-label', '更多操作');
        const popover = document.createElement('div');
        popover.className = 'training-queue-item-more-popover';
        popover.addEventListener('click', (event) => event.stopPropagation());
        actions.forEach((action) => popover.appendChild(action));
        menu.append(summary, popover);
        return menu;
    }

    function queueItemTitle(item) {
        const resumeName = item?.resume_info?.checkpoint_name || '';
        const source = item?.source_config_file || item?.runtime_config_file || '';
        const fallback = deps.runLabelFromPath(source) || `${item?.variant || '训练'} / ${item?.preset || 'default'}`;
        return item?.kind === 'resume' && resumeName ? `续训 · ${resumeName}` : fallback;
    }

    function queueKindLabel(item) {
        if (item?.requires_preprocess) return '预处理后训练';
        if (item?.kind === 'resume') return '续训';
        return '训练';
    }

    function queueAttemptLabel(item) {
        const attempt = Number(item?.attempt || 1);
        return attempt > 1 ? `第 ${attempt} 次尝试` : '';
    }

    function queueHistoryLabel(item) {
        const list = Array.isArray(item?.history_task_ids) ? item.history_task_ids : [];
        return list.join(', ');
    }

    function queueShortId(value) {
        const text = String(value || '');
        return text.length > 12 ? `${text.slice(0, 6)}…${text.slice(-4)}` : text;
    }

    function queueGpuLabel(value) {
        const list = Array.isArray(value) ? value : [];
        return list.length ? list.join(',') : '全部';
    }

    function queueStateLabel(itemState) {
        return {
            queued: '等待',
            running: '运行中',
            done: '完成',
            error: '异常',
            canceled: '已取消',
        }[itemState] || itemState || '未知';
    }

    return {
        renderTrainingQueue,
        renderTrainingQueueSummary,
        renderTrainingQueueManager,
        queueItemTitle,
        updateRunningQueueProgress,
    };
}
