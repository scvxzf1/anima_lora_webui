/**
 * Queue item/card builders and action controls.
 */
import {
    queueDetailsSummaryText,
    queueHistoryLabel,
    queueKindLabel,
    queueAttemptLabel,
    queueShortId,
    queueGpuLabel,
    queueStateLabel,
} from './render-labels.js?v=module-bootstrap-20260831-release-v1';

export function createQueueItemRenderers({ state, deps, actions }) {
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
    function queueFeedbackItemState(item) {
        if (!item?.id) return '';
        if (state.feedback?.busyItemId === item.id) return 'pending';
        if (state.feedback?.flashItemId === item.id) return state.feedback?.tone === 'error' ? 'failed' : 'updated';
        return '';
    }
    function queueFeedbackBusyAction(action) {
        return Boolean(action && state.feedback?.busyAction === action);
    }
    function queueBackendRunning() {
        return state.queue.status === 'running' || deps.getTrainingRuntime()?.state === 'running';
    }
    function queueItemTitle(item) {
        const resumeName = item?.resume_info?.checkpoint_name || '';
        const source = item?.source_config_file || item?.runtime_config_file || '';
        const fallback = deps.runLabelFromPath(source) || `${item?.variant || '训练'} / ${item?.preset || 'default'}`;
        return item?.kind === 'resume' && resumeName ? `续训 · ${resumeName}` : fallback;
    }
    return {
        createTrainingQueueSection,
        createTrainingQueueItem,
        createTrainingQueueManagerItem,
        updateRunningQueueProgress,
        queueItemTitle,
        queueFeedbackBusyAction,
        queueBackendRunning,
    };
}
