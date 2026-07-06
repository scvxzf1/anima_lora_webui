import { fetchHistoryTask } from './api.js?v=module-bootstrap-20260706-1';
import { createHistoryAnalysisRenderer } from './analysis.js?v=module-bootstrap-20260706-1';
import { createHistoryConfigFilesRenderer } from './config-files.js?v=module-bootstrap-20260706-1';
import { createHistoryLogsRenderer } from './logs.js?v=module-bootstrap-20260706-1';
import { createHistoryOverviewRenderer } from './overview.js?v=module-bootstrap-20260706-1';
import { createHistoryResumeFeature } from './resume/index.js?v=module-bootstrap-20260706-1';
import { HISTORY_DETAIL_TABS, normalizeHistoryDetailTab, setHistoryDetailTab } from './state.js?v=module-bootstrap-20260706-1';
import { createHistoryDetailWorkspace } from './workspace.js?v=module-bootstrap-20260706-1';

export function createHistoryDetailDialog({ ctx, state, deps }) {
    const slots = {};
    const contentCache = {
        payloadKey: '',
        nodes: new Map(),
    };
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
        const visibleTabs = historyDetailTabsForPayload(state.currentPayload);
        state.detailTab = normalizeVisibleHistoryDetailTab(state.currentPayload, state.detailTab);
        tabs.innerHTML = '';
        for (const item of visibleTabs) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'history-detail-tab';
            btn.dataset.historyDetailTab = item.key;
            btn.textContent = item.label;
            btn.classList.toggle('active', item.key === state.detailTab);
            tabs.appendChild(btn);
        }
    }

    function historyDetailTabsForPayload(payload) {
        const task = payload?.task || null;
        if (task?.job === 'preprocess') {
            return HISTORY_DETAIL_TABS.filter((item) => ['overview', 'logs', 'config_files'].includes(item.key));
        }
        return HISTORY_DETAIL_TABS;
    }

    function normalizeVisibleHistoryDetailTab(payload, tab) {
        const normalized = normalizeHistoryDetailTab(tab);
        return historyDetailTabsForPayload(payload).some((item) => item.key === normalized) ? normalized : 'overview';
    }

    function renderHistoryManagerDetail(payload = state.currentPayload, options = {}) {
        renderHistoryDetailDialog(payload, options);
    }

    function historyDetailContentCacheKey(payload) {
        const task = payload?.task || null;
        if (task?.id) return `${task.job || 'task'}:${task.id}`;
        const group = payload?.group || null;
        if (payload?.mode === 'config_group') {
            return `config_group:${group?.key || group?.history_group_key || ''}:${payload.summary?.task_count || 0}`;
        }
        return payload ? String(payload.mode || 'history') : '';
    }

    function syncHistoryDetailContentCache(payload) {
        const key = historyDetailContentCacheKey(payload);
        if (contentCache.payloadKey === key) return;
        deps.restorePreviewWorkspaceFromHistoryDetail();
        contentCache.payloadKey = key;
        contentCache.nodes.clear();
    }

    function clearHistoryDetailContentCache() {
        deps.restorePreviewWorkspaceFromHistoryDetail();
        contentCache.payloadKey = '';
        contentCache.nodes.clear();
    }

    function selectHistoryDetailTab(tab) {
        setHistoryDetailTab(state, tab);
        renderHistoryDetailTabs();
        renderHistoryDetailContent({ reuseCached: true });
    }

    function renderHistoryDetailDialog(payload = state.currentPayload, options = {}) {
        const dialog = document.getElementById('history-detail-dialog');
        const title = document.getElementById('history-detail-title');
        const meta = document.getElementById('history-detail-meta');
        const actions = document.getElementById('history-detail-actions');
        const content = document.getElementById('history-detail-content');
        if (!dialog || !title || !meta || !actions || !content) return;
        state.currentPayload = payload;
        syncHistoryDetailContentCache(payload);
        if (!payload) {
            title.textContent = '历史任务';
            meta.textContent = '选择一条历史任务查看详情。';
            state.mainTaskReturn = null;
            actions.innerHTML = '';
            content.replaceChildren();
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
                const preprocessTask = task.linked_preprocess_task || null;
                if (preprocessTask?.id) {
                    const preprocessBtn = deps.createHistoryActionButton('查阅预处理', () => openLinkedPreprocessTask(task, preprocessTask));
                    preprocessBtn.title = '在当前详情中查看这次训练对应的预处理任务';
                    actions.append(preprocessBtn);
                }
                actions.append(
                    deps.createHistoryTaskPreviewButton(task),
                );
            } else if (task.job === 'preprocess' && state.mainTaskReturn?.taskId) {
                const returnBtn = deps.createHistoryActionButton('返回主项目', () => returnToMainHistoryTask());
                returnBtn.title = '返回进入预处理详情前查看的训练任务';
                actions.append(returnBtn);
            }
            actions.append(
                deps.createHistoryActionButton('重命名', () => deps.renameHistoryTask(task)),
                deps.createHistoryActionButton(deps.historyTaskIsArchived(task) ? '取消归档' : '归档', () => deps.archiveHistoryTask(task)),
                deps.createHistoryActionButton('彻底删除', () => deps.deleteHistoryTask(task), 'danger'),
            );
        } else if (payload.mode === 'config_group' && deps.canPreviewHistoryConfigGroup(group)) {
            const previewBtn = deps.createHistoryActionButton('分组预览', () => {
                selectHistoryDetailTab('preview');
            });
            previewBtn.title = '汇总查看这个配置分组下所有训练任务的样张和权重';
            actions.append(previewBtn);
        }
        renderHistoryDetailTabs();
        renderHistoryDetailContent();
        if (options.open) openHistoryDetailDialog();
    }

    async function openLinkedPreprocessTask(task, preprocessTask) {
        const preprocessTaskId = String(preprocessTask?.id || '').trim();
        const mainTaskId = String(task?.id || '').trim();
        if (!preprocessTaskId || !mainTaskId) return;
        state.mainTaskReturn = {
            taskId: mainTaskId,
            detailTab: state.detailTab,
        };
        await loadHistoryTaskInDetail(preprocessTaskId, { detailTab: 'overview' });
    }

    async function returnToMainHistoryTask() {
        const target = state.mainTaskReturn;
        if (!target?.taskId) return;
        state.mainTaskReturn = null;
        await loadHistoryTaskInDetail(target.taskId, { detailTab: target.detailTab || 'overview' });
    }

    async function loadHistoryTaskInDetail(taskId, options = {}) {
        try {
            const payload = await fetchHistoryTask(ctx, taskId);
            if (!payload.ok) {
                alert(payload.error || '读取历史任务失败');
                return;
            }
            if (options.detailTab) {
                state.detailTab = normalizeHistoryDetailTab(options.detailTab);
            }
            deps.setViewingHistoryTaskContext({
                taskId,
                viewMode: 'live',
                task: payload.task || null,
                configGroup: null,
                timelineSelection: [],
            });
            if (payload.task?.job === 'training') {
                dialogSetResumeLoadingForTask(taskId);
            } else {
                resume.clearResumeOptions();
            }
            state.curve.hoverStep = null;
            renderHistoryManagerDetail(payload, { open: true });
            deps.renderTrainingHistoryList();
            deps.renderHistoryManager();
            if (payload.task?.job === 'training') {
                await resume.loadResumeOptionsForTask(taskId);
            }
        } catch (e) {
            alert('读取历史任务失败: ' + e.message);
        }
    }

    function dialogSetResumeLoadingForTask(taskId) {
        resume.setResumeLoadingForTask(taskId);
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
        clearHistoryDetailContentCache();
        setHistoryDetailWindowOpen(false);
        restoreHistoryDetailReturnState();
        state.mainTaskReturn = null;
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

    function renderHistoryDetailContent(options = {}) {
        const content = document.getElementById('history-detail-content');
        if (!content) return;
        const payload = state.currentPayload;
        syncHistoryDetailContentCache(payload);
        deps.restorePreviewWorkspaceFromHistoryDetail();
        content.replaceChildren();
        if (!payload) {
            delete content.dataset.historyDetailTab;
            deps.applyHistoryDetailUIScale?.('overview');
            return;
        }
        state.detailTab = normalizeVisibleHistoryDetailTab(payload, state.detailTab);
        content.dataset.historyDetailTab = state.detailTab;
        deps.applyHistoryDetailUIScale?.(state.detailTab);
        const cached = options.reuseCached ? contentCache.nodes.get(state.detailTab) : null;
        if (cached) {
            content.appendChild(cached);
            if (state.detailTab === 'preview') deps.activateHistoryDetailPreview(payload);
            return;
        }

        let node = null;
        if (state.detailTab === 'overview') {
            node = overview.renderHistoryDetailOverview(payload);
        } else if (state.detailTab === 'analysis') {
            node = analysis.renderHistoryDetailAnalysis(payload);
        } else if (state.detailTab === 'preview') {
            node = workspace.renderHistoryDetailPreview(payload);
        } else if (state.detailTab === 'logs') {
            node = logs.renderHistoryDetailLogs(payload);
        } else if (state.detailTab === 'config_files') {
            node = configFiles.renderHistoryDetailConfigFiles(payload);
        }
        if (!node) return;
        contentCache.nodes.set(state.detailTab, node);
        content.appendChild(node);
        if (state.detailTab === 'preview') {
            deps.activateHistoryDetailPreview(payload);
        }
    }

    function bindHistoryDetailEvents() {
        document.querySelector('#history-detail-dialog .history-detail-tabs')?.addEventListener('click', (event) => {
            const btn = event.target.closest('.history-detail-tab');
            if (!btn) return;
            selectHistoryDetailTab(btn.dataset.historyDetailTab);
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
        clearHistoryDetailContentCache,
        getCurrentPayload: () => state.currentPayload,
        getActiveTab: () => state.detailTab,
        setActiveTab: (tab) => setHistoryDetailTab(state, tab),
    };
}
