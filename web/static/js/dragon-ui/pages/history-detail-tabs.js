/* Route-backed secondary navigation for a single training-history task. */

export const HISTORY_DETAIL_TABS = Object.freeze([
    { id: 'overview', label: '概览' },
    { id: 'metrics', label: '指标' },
    { id: 'artifacts', label: '产物' },
    { id: 'config', label: '配置' },
    { id: 'logs', label: '日志' },
]);

const HISTORY_DETAIL_TAB_IDS = new Set(HISTORY_DETAIL_TABS.map((tab) => tab.id));

export function normalizeHistoryDetailTab(value) {
    const tabId = String(value || '').trim().toLowerCase();
    return HISTORY_DETAIL_TAB_IDS.has(tabId) ? tabId : 'overview';
}

export function renderHistoryDetailTabs(taskId, activeTab, counts = {}) {
    const current = normalizeHistoryDetailTab(activeTab);
    const encodedTaskId = encodeURIComponent(String(taskId || ''));
    return `<nav class="dragon-history-detail-tabs" aria-label="训练任务详情">
        ${HISTORY_DETAIL_TABS.map((tab) => {
            const selected = tab.id === current;
            const count = normalizeTabCount(counts[tab.id]);
            return `<a id="dragon-history-tab-${tab.id}" class="dragon-history-detail-tab" href="#history/${encodedTaskId}/${tab.id}" data-history-detail-tab="${tab.id}" ${selected ? 'aria-current="page"' : ''}><span>${tab.label}</span>${count == null ? '' : `<small>${count}</small>`}</a>`;
        }).join('')}
    </nav>`;
}

export function activateHistoryDetailTab(root, requestedTab) {
    const activeTab = normalizeHistoryDetailTab(requestedTab);
    root?.querySelectorAll('[data-history-detail-tab]').forEach((link) => {
        const selected = link.dataset.historyDetailTab === activeTab;
        link.classList.toggle('active', selected);
        if (selected) link.setAttribute('aria-current', 'page');
        else link.removeAttribute('aria-current');
    });
    root?.querySelectorAll('[data-history-detail-panel]').forEach((panel) => {
        const selected = panel.dataset.historyDetailPanel === activeTab;
        panel.hidden = !selected;
        if (selected) {
            panel.classList.add('dragon-in-view');
            panel.querySelectorAll('.dragon-reveal').forEach((element) => element.classList.add('dragon-in-view'));
        }
    });
    const activeLink = root?.querySelector(`[data-history-detail-tab="${activeTab}"]`);
    revealActiveTab(activeLink);
    if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(() => revealActiveTab(activeLink));
    }
    const page = root?.querySelector('[data-history-detail]');
    if (page) page.dataset.historyDetailActiveTab = activeTab;
    return activeTab;
}

function revealActiveTab(link) {
    const nav = link?.parentElement;
    if (!nav || !nav.clientWidth) return;
    const navRect = nav.getBoundingClientRect();
    const linkRect = link.getBoundingClientRect();
    if (linkRect.left < navRect.left) nav.scrollLeft -= navRect.left - linkRect.left;
    else if (linkRect.right > navRect.right) nav.scrollLeft += linkRect.right - navRect.right;
}

function normalizeTabCount(value) {
    const count = Number(value);
    return Number.isFinite(count) && count >= 0 ? Math.floor(count) : null;
}
