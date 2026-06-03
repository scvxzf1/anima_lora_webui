import { queueManagerSections, queueSummaryCounts } from './state.js?v=module-bootstrap-20260601-8';

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
        const badge = document.getElementById('training-queue-tab-badge');
        const cancelAllBtn = document.getElementById('btn-cancel-all-queue');
        const cancelWaitingBtn = document.getElementById('btn-cancel-waiting-queue');
        const clearCompletedBtn = document.getElementById('btn-clear-completed-queue');
        const clearCanceledBtn = document.getElementById('btn-clear-canceled-queue');
        if (!list || !summary) return;
        list.innerHTML = '';
        const counts = queueSummaryCounts(state);
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
        }
        if (cancelAllBtn) cancelAllBtn.disabled = state.queue.loading || (counts.queued + counts.running) <= 0;
        if (cancelWaitingBtn) cancelWaitingBtn.disabled = state.queue.loading || counts.queued <= 0;
        if (clearCompletedBtn) clearCompletedBtn.disabled = state.queue.loading || counts.done <= 0;
        if (clearCanceledBtn) clearCanceledBtn.disabled = state.queue.loading || counts.canceled <= 0;
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
        if (policy && policy.value !== state.queue.failurePolicy) {
            policy.value = state.queue.failurePolicy || 'pause';
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
            const valueEl = document.createElement('strong');
            valueEl.textContent = String(value);
            const labelEl = document.createElement('span');
            labelEl.textContent = label;
            item.append(valueEl, labelEl);
            stats.appendChild(item);
        });
        document.querySelectorAll('.training-queue-filter').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.queueFilter === state.filter);
        });
        list.innerHTML = '';
        const sections = queueManagerSections(state);
        const visibleCount = sections.reduce((sum, section) => sum + section.items.length, 0);
        if (!visibleCount) {
            const empty = document.createElement('div');
            empty.className = 'training-queue-manager-empty';
            empty.textContent = state.filter === 'all' ? '队列为空。' : '当前视图没有任务。';
            list.appendChild(empty);
            return;
        }
        for (const section of sections) {
            if (!section.items.length) continue;
            list.appendChild(createTrainingQueueSection(section));
        }
    }

    function queueManagerStatusText(counts) {
        const policy = queueFailurePolicyLabel(state.queue.failurePolicy);
        if (state.queue.paused) return `队列已暂停 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
        if (counts.running) return `队列运行中 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
        if (counts.queued) return `空闲时会自动启动 · 等待 ${counts.queued} 个 · 异常 ${counts.error} 个 · 失败后${policy}`;
        return counts.total ? `没有等待任务 · 异常 ${counts.error} 个 · 失败后${policy}` : `队列为空 · 失败后${policy}`;
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
        card.className = ['training-queue-item', item.state || 'queued'].join(' ');
        card.dataset.queueItemId = item.id || '';
        applyQueueProgressStyle(card, item);
        const itemState = document.createElement('span');
        itemState.className = ['training-queue-state', item.state || 'queued'].join(' ');
        itemState.textContent = queueStateLabel(item.state);
        const main = document.createElement('div');
        main.className = 'training-queue-main';
        const title = document.createElement('strong');
        title.textContent = queueItemTitle(item);
        const meta = createQueueMetaTags(item, { compact: true });
        const message = document.createElement('em');
        message.textContent = [
            item.message || '',
            item.created_at_text ? `入队: ${item.created_at_text}` : '',
        ].filter(Boolean).join(' · ');
        main.append(title, meta);
        const progress = createQueueRunningProgress(item);
        if (progress) main.appendChild(progress);
        main.appendChild(message);
        card.append(itemState, main);
        return card;
    }

    function createTrainingQueueManagerItem(item) {
        const card = document.createElement('article');
        card.className = ['training-queue-manager-item', item.state || 'queued'].join(' ');
        card.dataset.queueItemId = item.id || '';
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
        titleWrap.append(title, meta);
        head.append(itemState, titleWrap);

        const actionsEl = document.createElement('div');
        actionsEl.className = 'training-queue-manager-item-actions';
        if (item.state === 'queued') {
            actionsEl.append(
                createQueueIconActionButton('置顶', '⇈', () => actions.moveQueueItem(item.id, 'top')),
                createQueueIconActionButton('上移', '↑', () => actions.moveQueueItem(item.id, 'up')),
                createQueueIconActionButton('下移', '↓', () => actions.moveQueueItem(item.id, 'down')),
                createQueueIconActionButton('置底', '⇊', () => actions.moveQueueItem(item.id, 'bottom')),
            );
            actionsEl.appendChild(createQueueItemMoreMenu([
                createQueueActionButton('取消等待', () => actions.cancelQueueItem(item.id), 'danger'),
            ]));
        } else if (item.state === 'running') {
            actionsEl.append(createQueueActionButton('停止', () => actions.cancelQueueItem(item.id), 'danger queue-stop-action'));
        } else {
            actionsEl.append(
                createQueueActionButton('重新入队', () => actions.retryQueueItem(item.id)),
                createQueueActionButton('移除列表', () => actions.removeQueueItemFromList(item.id), 'danger'),
            );
        }
        head.appendChild(actionsEl);
        const progress = createQueueRunningProgress(item);
        if (progress) titleWrap.appendChild(progress);

        const details = document.createElement('details');
        details.className = 'training-queue-details';
        const summary = document.createElement('summary');
        summary.textContent = item.message || '查看任务详情';
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

    function createQueueActionButton(label, handler, tone = '') {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = ['task-history-action', tone].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.addEventListener('click', handler);
        return btn;
    }

    function createQueueIconActionButton(label, icon, handler, tone = '') {
        const btn = createQueueActionButton(label, handler, ['queue-icon-action', tone].filter(Boolean).join(' '));
        btn.textContent = icon;
        btn.title = label;
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
