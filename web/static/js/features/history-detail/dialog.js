import { createHistoryAnalysisRenderer } from './analysis.js?v=module-bootstrap-20260603-6';
import { createHistoryConfigFilesRenderer } from './config-files.js?v=module-bootstrap-20260603-6';
import { createHistoryLogsRenderer } from './logs.js?v=module-bootstrap-20260603-6';
import { createHistoryOverviewRenderer } from './overview.js?v=module-bootstrap-20260603-6';
import { createHistoryResumeFeature } from './resume/index.js?v=module-bootstrap-20260603-6';
import { HISTORY_DETAIL_TABS, normalizeHistoryDetailTab, setHistoryDetailTab } from './state.js?v=module-bootstrap-20260603-6';
import { createHistoryDetailWorkspace } from './workspace.js?v=module-bootstrap-20260603-6';

export function createHistoryDetailDialog({ ctx, state, deps }) {
    const slots = {};
    const workspace = createHistoryDetailWorkspace({ deps });
    const resume = createHistoryResumeFeature({ ctx, state, deps, slots });
    const overview = createHistoryOverviewRenderer({
        ctx,
        state,
        deps,
        renderHistoryDetailResume: resume.renderHistoryDetailResume,
    });
    const analysis = createHistoryAnalysisRenderer({
        state,
        deps,
        renderHistoryDetailContent,
    });
    const logs = createHistoryLogsRenderer({ state, deps });
    const configFiles = createHistoryConfigFilesRenderer({ ctx, deps });

    function renderHistoryDetailTabs() {
        const tabs = document.querySelector('#history-detail-dialog .history-detail-tabs');
        if (!tabs) return;
        state.detailTab = normalizeHistoryDetailTab(state.detailTab);
        tabs.innerHTML = '';
        for (const item of HISTORY_DETAIL_TABS) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'history-detail-tab';
            btn.dataset.historyDetailTab = item.key;
            btn.textContent = item.label;
            btn.classList.toggle('active', item.key === state.detailTab);
            tabs.appendChild(btn);
        }
    }

    function renderHistoryManagerDetail(payload = state.currentPayload, options = {}) {
        renderHistoryDetailDialog(payload, options);
    }

    function renderHistoryDetailDialog(payload = state.currentPayload, options = {}) {
        const dialog = document.getElementById('history-detail-dialog');
        const title = document.getElementById('history-detail-title');
        const meta = document.getElementById('history-detail-meta');
        const actions = document.getElementById('history-detail-actions');
        const content = document.getElementById('history-detail-content');
        if (!dialog || !title || !meta || !actions || !content) return;
        state.currentPayload = payload;
        if (!payload) {
            title.textContent = '历史任务';
            meta.textContent = '选择一条历史任务查看详情。';
            actions.innerHTML = '';
            content.innerHTML = '';
            renderHistoryDetailTabs();
            return;
        }
        const task = payload.task || null;
        const group = payload.group || null;
        title.textContent = task
            ? deps.historyTaskDisplayName(task) || task.id || '历史任务'
            : `合并查看: ${deps.configGroupLabel(group || {})}`;
        meta.textContent = task
            ? [
                task.job === 'preprocess' ? '预处理' : '训练',
                deps.historyStateLabel(task.state),
                task.started_at_text || task.id,
                task.history_source_config_file || '',
            ].filter(Boolean).join(' · ')
            : [
                `${payload.summary?.task_count || 0} 次训练`,
                `${payload.summary?.loss_count || 0} Loss 点`,
                payload.summary?.started_at_text || '',
            ].filter(Boolean).join(' · ');
        actions.innerHTML = '';
        if (task) {
            if (task.job === 'training') {
                actions.append(
                    deps.createHistoryTaskPreviewButton(task),
                );
            }
            actions.append(
                deps.createHistoryActionButton('重命名', () => deps.renameHistoryTask(task)),
                deps.createHistoryActionButton(deps.historyTaskIsArchived(task) ? '取消归档' : '归档', () => deps.archiveHistoryTask(task)),
                deps.createHistoryActionButton('彻底删除', () => deps.deleteHistoryTask(task), 'danger'),
            );
        } else if (payload.mode === 'config_group' && deps.canPreviewHistoryConfigGroup(group)) {
            const previewBtn = deps.createHistoryActionButton('分组预览', () => {
                state.detailTab = 'preview';
                renderHistoryManagerDetail(state.currentPayload, { open: true });
            });
            previewBtn.title = '汇总查看这个配置分组下所有训练任务的样张和权重';
            actions.append(previewBtn);
        }
        renderHistoryDetailTabs();
        renderHistoryDetailContent();
        if (options.open) openHistoryDetailDialog();
    }

    function captureHistoryDetailReturnState() {
        const active = document.activeElement;
        const focus = active instanceof HTMLElement && !active.closest('#history-detail-dialog') ? active : null;
        const scrollTargets = [];
        [
            '#history-manager-list',
            '#task-history-list',
            '.history-config-group-card-list',
            '.history-collection-card-list',
        ].forEach((selector) => {
            document.querySelectorAll(selector).forEach((el, index) => {
                if (el.scrollTop || el.scrollLeft) {
                    scrollTargets.push({
                        selector,
                        index,
                        top: el.scrollTop,
                        left: el.scrollLeft,
                    });
                }
            });
        });
        return {
            focus,
            scrollX: window.scrollX,
            scrollY: window.scrollY,
            scrollTargets,
        };
    }

    function setHistoryDetailWindowOpen(open) {
        const dialog = document.getElementById('history-detail-dialog');
        if (!dialog) return;
        dialog.hidden = !open;
        dialog.classList.toggle('open', open);
        dialog.toggleAttribute('open', open);
        dialog.setAttribute('aria-hidden', open ? 'false' : 'true');
        document.body.classList.toggle('history-detail-window-open', open);
    }

    function openHistoryDetailDialog() {
        const dialog = document.getElementById('history-detail-dialog');
        if (!dialog || isHistoryDetailDialogOpen()) return;
        state.returnState = captureHistoryDetailReturnState();
        setHistoryDetailWindowOpen(true);
        requestAnimationFrame(() => {
            (document.getElementById('btn-close-history-detail') || dialog).focus({ preventScroll: true });
        });
    }

    function closeHistoryDetailDialog() {
        const dialog = document.getElementById('history-detail-dialog');
        if (!dialog || !isHistoryDetailDialogOpen()) return;
        deps.restorePreviewWorkspaceFromHistoryDetail();
        setHistoryDetailWindowOpen(false);
        restoreHistoryDetailReturnState();
        deps.clearViewingHistoryTaskContext?.(state.currentPayload);
    }

    function isHistoryDetailDialogOpen() {
        const dialog = document.getElementById('history-detail-dialog');
        return Boolean(dialog && !dialog.hidden && dialog.hasAttribute('open'));
    }

    function restoreHistoryDetailReturnState() {
        const returnState = state.returnState;
        state.returnState = null;
        if (!returnState) return;
        for (const target of returnState.scrollTargets || []) {
            const el = document.querySelectorAll(target.selector)[target.index];
            if (!el) continue;
            el.scrollTop = target.top || 0;
            el.scrollLeft = target.left || 0;
        }
        if (Number.isFinite(returnState.scrollX) && Number.isFinite(returnState.scrollY)) {
            window.scrollTo(returnState.scrollX, returnState.scrollY);
        }
        if (returnState.focus && document.contains(returnState.focus)) {
            returnState.focus.focus({ preventScroll: true });
        }
    }

    function handleHistoryDetailWindowKeydown(event) {
        if (event.key !== 'Escape' || !isHistoryDetailDialogOpen()) return;
        event.preventDefault();
        closeHistoryDetailDialog();
    }

    function renderHistoryDetailContent() {
        const content = document.getElementById('history-detail-content');
        if (!content) return;
        deps.restorePreviewWorkspaceFromHistoryDetail();
        content.innerHTML = '';
        const payload = state.currentPayload;
        if (!payload) {
            delete content.dataset.historyDetailTab;
            return;
        }
        state.detailTab = normalizeHistoryDetailTab(state.detailTab);
        content.dataset.historyDetailTab = state.detailTab;
        if (state.detailTab === 'overview') {
            content.appendChild(overview.renderHistoryDetailOverview(payload));
        } else if (state.detailTab === 'analysis') {
            content.appendChild(analysis.renderHistoryDetailAnalysis(payload));
        } else if (state.detailTab === 'preview') {
            content.appendChild(workspace.renderHistoryDetailPreview(payload));
            deps.activateHistoryDetailPreview(payload);
        } else if (state.detailTab === 'logs') {
            content.appendChild(logs.renderHistoryDetailLogs(payload));
        } else if (state.detailTab === 'config_files') {
            content.appendChild(configFiles.renderHistoryDetailConfigFiles(payload));
        }
    }

    function bindHistoryDetailEvents() {
        document.querySelector('#history-detail-dialog .history-detail-tabs')?.addEventListener('click', (event) => {
            const btn = event.target.closest('.history-detail-tab');
            if (!btn) return;
            setHistoryDetailTab(state, btn.dataset.historyDetailTab);
            renderHistoryManagerDetail(state.currentPayload);
        });
        document.getElementById('btn-close-history-detail')?.addEventListener('click', closeHistoryDetailDialog);
        document.getElementById('history-detail-dialog')?.addEventListener('click', (event) => {
            if (event.target === event.currentTarget) closeHistoryDetailDialog();
        });
        document.addEventListener('keydown', handleHistoryDetailWindowKeydown);
    }

    Object.assign(slots, {
        closeHistoryDetailDialog,
        isHistoryDetailDialogOpen,
        renderHistoryDetailContent,
    });

    return {
        ...resume,
        renderHistoryManagerDetail,
        renderHistoryDetailDialog,
        closeHistoryDetailDialog,
        isHistoryDetailDialogOpen,
        handleHistoryDetailWindowKeydown,
        renderHistoryDetailContent,
        renderHistoryDetailTabs,
        bindHistoryDetailEvents,
        getCurrentPayload: () => state.currentPayload,
        getActiveTab: () => state.detailTab,
        setActiveTab: (tab) => setHistoryDetailTab(state, tab),
    };
}
