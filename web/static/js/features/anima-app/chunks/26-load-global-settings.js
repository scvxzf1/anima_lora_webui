/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    GLOBAL_MODEL_PATH_FIELDS,
    GLOBAL_SETTING_INPUTS,
    GLOBAL_UI_OVERRIDE_FIELDS,
    help,
} from '../../../config/catalog.js?v=module-bootstrap-20260707-93';
import { ensurePreviewFeature, ensureQueueFeature } from '../helpers/feature-ensurers.js?v=module-bootstrap-20260707-93';
import { renderContinueTrainingSource } from '../helpers/training-source-bridge.js?v=module-bootstrap-20260707-93';
import { historyManagerBaseFilteredTasks, historyManagerVisibleTasks, historyTaskIds, historyTaskIsArchived, renderHistoryBulkBar, renderHistoryCollectionsWorkbench, renderHistoryManagerStats, selectedHistoryTasks, syncHistorySelectionWithTasks } from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260707-93';
import { createHistoryTaskItem, renderHistoryDetailDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260707-93';
import { returnToLiveTraining } from '../helpers/history-timeline-bridge.js?v=module-bootstrap-20260707-93';
import { loadOutputRuns } from '../helpers/toml-manager-bridge.js?v=module-bootstrap-20260707-93';
import { updateChoiceGuide } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260707-93';
import { getUiScaleController } from '../helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260707-93';
import { getAppShellState } from '../helpers/app-shell-state-bridge.js?v=module-bootstrap-20260707-93';
import { getAppContext } from '../helpers/app-context-bridge.js?v=module-bootstrap-20260707-93';
import { configureGlobalSettingsBridge } from '../helpers/global-settings-bridge.js?v=module-bootstrap-20260707-93';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260707-93';
import { getHistoryDetailFeature } from '../helpers/history-detail-bridge.js?v=module-bootstrap-20260707-93';
import { appendLog } from '../helpers/live-log-bridge.js?v=module-bootstrap-20260707-93';
import { configurePreviewViewBridge } from '../helpers/preview-view-bridge.js?v=module-bootstrap-20260707-93';
import { configureQueueViewBridge } from '../helpers/queue-view-bridge.js?v=module-bootstrap-20260707-93';
import { configureHistoryListBridge } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260707-93';
import { api } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260707-93';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260707-93';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260707-93';

const ctx = getAppContext();
const appShellState = getAppShellState();
const historyState = getHistoryState();
const tomlState = getTomlState();
const trainingState = getTrainingState();

    export function resolveGlobalUIScaleDefaultValue(snapshot = appShellState.globalSettings || {}) {
        const uiScaleController = getUiScaleController();
        const fallback = uiScaleController?.DEFAULT_SCALE || 100;
        const raw = snapshot?.ui_scale ?? snapshot?.defaults?.ui_scale ?? document.getElementById('global-ui-scale')?.value;
        const scale = Number.parseInt(String(raw ?? '').trim(), 10);
        if (!Number.isFinite(scale)) return fallback;
        return uiScaleController?.clampScale?.(scale) ?? fallback;
    }

    export function syncGlobalUIScaleOverrideField(field, options = {}) {
        const uiScaleController = getUiScaleController();
        if (!field?.inputId || !field?.followDefaultId) return;
        const input = document.getElementById(field.inputId);
        const followToggle = document.getElementById(field.followDefaultId);
        if (!input || !followToggle) return;
        const snapshot = options.snapshot ?? null;
        const defaultScale = resolveGlobalUIScaleDefaultValue(snapshot || appShellState.globalSettings || {});
        const row = input.closest('.global-ui-scale-row');
        let followDefault = Boolean(followToggle.checked);
        let nextValue = input.value;

        if (snapshot) {
            const raw = snapshot?.[field.key] ?? snapshot?.defaults?.[field.key] ?? '';
            followDefault = String(raw ?? '').trim() === '';
            nextValue = followDefault
                ? defaultScale
                : (uiScaleController?.clampScale?.(raw) ?? defaultScale);
        } else if (followDefault || !options.preserveCustom || !String(nextValue || '').trim()) {
            nextValue = defaultScale;
        } else {
            nextValue = uiScaleController?.clampScale?.(nextValue) ?? defaultScale;
        }

        followToggle.checked = followDefault;
        input.disabled = followDefault;
        input.placeholder = String(defaultScale);
        input.value = String(nextValue);
        row?.classList.toggle('is-follow-default', followDefault);
    }

    export function syncAllGlobalUIScaleOverrideFields(options = {}) {
        const snapshot = options.snapshot ?? null;
        const defaultScale = resolveGlobalUIScaleDefaultValue(snapshot || appShellState.globalSettings || {});
        for (const field of GLOBAL_UI_OVERRIDE_FIELDS) {
            syncGlobalUIScaleOverrideField(field, {
                ...options,
                snapshot,
                defaultScale,
            });
        }
    }

    export function applyGlobalUIScaleOverrideInputs(snapshot = appShellState.globalSettings || {}) {
        syncAllGlobalUIScaleOverrideFields({ snapshot });
    }

    export function collectGlobalUIScaleOverridePayload(payload) {
        const uiScaleController = getUiScaleController();
        const defaultScale = resolveGlobalUIScaleDefaultValue();
        for (const field of GLOBAL_UI_OVERRIDE_FIELDS) {
            const input = document.getElementById(field.inputId);
            const followToggle = document.getElementById(field.followDefaultId);
            if (!input || !followToggle) {
                payload[field.key] = appShellState.globalSettings?.[field.key] ?? '';
                continue;
            }
            if (followToggle.checked) {
                payload[field.key] = '';
                continue;
            }
            payload[field.key] = String(uiScaleController?.clampScale?.(input.value) ?? defaultScale);
        }
        return payload;
    }

    export async function loadGlobalSettings() {
        const uiScaleController = getUiScaleController();
        const historyDetailFeature = getHistoryDetailFeature();
        if (location.protocol === 'file:') return;
        try {
            const data = await api('/api/settings/global');
            if (!data.ok) throw new Error(data.error || '读取全局设置失败');
            appShellState.globalSettings = data;
            applyGlobalSettingsToInputs(data);
            uiScaleController?.applyScaleFromSettings?.(data, {
                activeHistoryDetailTab: historyDetailFeature?.getActiveTab?.(),
            });
            updateChoiceGuide();
            setGlobalSettingsStatus('', '');
            if (tomlState.tomlManagerMode === 'output') {
                await loadOutputRuns({ keepSelection: true });
            }
        } catch (e) {
            setGlobalSettingsStatus('读取全局设置失败: ' + e.message, 'error');
        }
    }

    export async function saveGlobalSettings() {
        const uiScaleController = getUiScaleController();
        const historyDetailFeature = getHistoryDetailFeature();
        try {
            const payload = collectGlobalSettingsPayload();
            const res = await api('/api/settings/global', {
                method: 'PUT',
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                setGlobalSettingsStatus(res.error || '保存失败', 'error');
                return;
            }
            appShellState.globalSettings = {
                ...(appShellState.globalSettings || {}),
                ...res,
            };
            applyGlobalSettingsToInputs(appShellState.globalSettings);
            uiScaleController?.applyScaleFromSettings?.(appShellState.globalSettings, {
                activeHistoryDetailTab: historyDetailFeature?.getActiveTab?.(),
            });
            updateChoiceGuide();
            setGlobalSettingsStatus(res.message || '全局设置已保存', 'ok');
        } catch (e) {
            setGlobalSettingsStatus('保存失败: ' + e.message, 'error');
        }
    }

    export async function resetGlobalSettings() {
        const defaults = appShellState.globalSettings?.defaults || {};
        applyGlobalSettingsToInputs({
            defaults,
            ...Object.fromEntries(GLOBAL_SETTING_INPUTS.map(([key]) => [key, defaults[key] ?? ''])),
            ...Object.fromEntries(GLOBAL_UI_OVERRIDE_FIELDS.map(({ key }) => [key, defaults[key] ?? ''])),
        });
        await saveGlobalSettings();
    }

    export function setGlobalSettingsStatus(text, state = '') {
        const el = document.getElementById('global-settings-status');
        if (!el) return;
        el.textContent = text;
        el.className = `preview-status ${state}`.trim();
    }

    export function applyGlobalSettingsToInputs(data) {
        const snapshot = data || appShellState.globalSettings || {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            if (!input) continue;
            const fallback = snapshot?.defaults?.[key] || '';
            input.value = snapshot?.[key] ?? fallback;
        }
        applyGlobalUIScaleOverrideInputs(snapshot);
    }

    export function collectGlobalSettingsPayload() {
        const payload = {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            payload[key] = input ? input.value : (appShellState.globalSettings?.[key] || '');
        }
        return collectGlobalUIScaleOverridePayload(payload);
    }

    export function getGlobalModelPathOverrides() {
        const overrides = {};
        const source = appShellState.globalSettings || {};
        for (const [key] of GLOBAL_MODEL_PATH_FIELDS) {
            const value = source[key] ?? source.defaults?.[key] ?? '';
            if (String(value || '').trim()) {
                overrides[key] = String(value).trim();
            }
        }
        return overrides;
    }

    export function toggleGlobalSettingHelp(button) {
        if (!button) return;
        const helpId = button.getAttribute('aria-controls');
        const help = helpId ? document.getElementById(helpId) : null;
        if (!help) return;
        const visible = help.classList.toggle('visible');
        button.classList.toggle('active', visible);
        button.setAttribute('aria-expanded', visible ? 'true' : 'false');
    }

    // ── 预览图 ──
    export async function loadPreviewSettings() {
        return ensurePreviewFeature().loadPreviewSettings();
    }

    export async function savePreviewSettings() {
        return ensurePreviewFeature().savePreviewSettings();
    }

    export async function resetPreviewSettings() {
        return ensurePreviewFeature().resetPreviewSettings();
    }

    export async function loadPreviewImages() {
        return ensurePreviewFeature().loadPreviewImages();
    }

    export async function loadPreviewWeights() {
        return ensurePreviewFeature().loadPreviewWeights();
    }

    export function setPreviewSource(source) {
        return ensurePreviewFeature().setPreviewSource(source);
    }

    export async function openTrainingPreview(options = {}) {
        return ensurePreviewFeature().openTrainingPreview(options);
    }

    export function openCurrentTrainingPreview(event) {
        return ensurePreviewFeature().openCurrentTrainingPreview(event);
    }

    export function openLiveSamplingPreview(event) {
        return ensurePreviewFeature().openLiveSamplingPreview(event);
    }

    export async function openHistoryConfigGroupPreview(group) {
        return ensurePreviewFeature().openHistoryConfigGroupPreview(group);
    }

    export function normalizePreviewGroup(group) {
        return ensurePreviewFeature().normalizePreviewGroup(group);
    }

    export function renderPreviewTaskSelect() {
        return ensurePreviewFeature().renderPreviewTaskSelect();
    }

    export async function changePreviewTask(taskId) {
        return ensurePreviewFeature().changePreviewTask(taskId);
    }

    export function togglePreviewWeightSort() {
        return ensurePreviewFeature().togglePreviewWeightSort();
    }

    export function openPreviewDialog(image) {
        return ensurePreviewFeature().openPreviewDialog(image);
    }

    export function closePreviewImageDialog() {
        return ensurePreviewFeature().closePreviewImageDialog();
    }

    export function openPreviewPanel() {
        return ensurePreviewFeature().openPreviewPanel();
    }

    export function closePreviewPanel() {
        return ensurePreviewFeature().closePreviewPanel();
    }

    export function restorePreviewWorkspaceAfterPanelClose() {
        return ensurePreviewFeature().restorePreviewWorkspaceAfterPanelClose();
    }

    export function setPreviewStatus(text, state = '') {
        return ensurePreviewFeature().setPreviewStatus(text, state);
    }

    export function createPreviewDetailRow(label, value) {
        return ensurePreviewFeature().createPreviewDetailRow(label, value);
    }

    export function createPreviewDetailBlock(label, value, preformatted = false) {
        return ensurePreviewFeature().createPreviewDetailBlock(label, value, preformatted);
    }

    export function renderDatasetImageDialogDetails(box, image, dims) {
        const caption = image.caption || {};
        const rows = [
            ['文件时间', image.mtime_text || '-'],
            ['尺寸', dims],
            ['长', image.height ? `${image.height} px` : '-'],
            ['宽', image.width ? `${image.width} px` : '-'],
            ['总像素', formatTotalPixels(image.total_pixels)],
            ['文件大小', formatBytes(image.size_bytes)],
        ];
        for (const [label, value] of rows) {
            box.appendChild(createPreviewDetailRow(label, value));
        }
        box.appendChild(createPreviewDetailBlock('文件路径', image.file || '-'));
        box.appendChild(createPreviewDetailBlock('标注文件', caption.file || '未找到同名标注文件'));
        const captionText = caption.ok ? (caption.text || '(空标注)') : '未找到同名 caption 文件';
        box.appendChild(createPreviewDetailBlock('标注内容', captionText, true));
    }

    export function formatTotalPixels(totalPixels) {
        const count = Number(totalPixels);
        if (!Number.isFinite(count) || count <= 0) return '-';
        return `${count.toLocaleString('zh-CN')} px (${(count / 1000000).toFixed(2)} MP)`;
    }

    export async function copyText(text) {
        return ctx.dom.copyText(text);
    }

    export function formatBytes(bytes) {
        return ctx.format.formatBytes(bytes);
    }

    // ── 训练队列 ──
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

    export async function loadTrainingHistoryList() {
        if (location.protocol === 'file:') return;
        try {
            const params = new URLSearchParams();
            params.set('include_archived', '1');
            const suffix = params.toString() ? `?${params.toString()}` : '';
            const payload = await api(`/api/training/history${suffix}`);
            historyState.historyTasks = payload.tasks || [];
            await loadHistoryCollectionSettings();
            renderTrainingHistoryList();
            renderHistoryManager();
            renderPreviewTaskSelect();
            renderContinueTrainingSource();
            setPreviewStatus('', '');
        } catch (e) {
            const list = document.getElementById('task-history-list');
            if (list) list.textContent = '读取任务列表失败';
            const managerList = document.getElementById('history-manager-list');
            if (managerList) managerList.textContent = '读取历史任务失败';
            renderPreviewTaskSelect();
            setPreviewStatus('读取训练任务列表失败: ' + e.message, 'error');
        }
    }

    export async function loadHistoryCollectionSettings() {
        if (location.protocol === 'file:') return;
        try {
            const payload = await api('/api/training/history/collections/settings');
            if (payload.ok !== false) {
                historyState.historyCollectionSettings = normalizeHistoryCollectionSettings(payload);
            }
        } catch (e) {
            appendLog(`[状态] 读取历史集合设置失败: ${e.message}`);
        }
    }

    export async function saveHistoryCollectionSettings(nextSettings) {
        historyState.historyCollectionSettings = normalizeHistoryCollectionSettings(nextSettings);
        if (location.protocol === 'file:') {
            renderHistoryManager();
            return historyState.historyCollectionSettings;
        }
        try {
            const payload = await api('/api/training/history/collections/settings', {
                method: 'PUT',
                body: JSON.stringify(historyState.historyCollectionSettings),
            });
            if (payload.ok === false) throw new Error(payload.error || '保存集合设置失败');
            historyState.historyCollectionSettings = normalizeHistoryCollectionSettings(payload);
            renderHistoryManager();
            return historyState.historyCollectionSettings;
        } catch (e) {
            appendLog(`[状态] 保存历史集合设置失败: ${e.message}`);
            renderHistoryManager();
            return historyState.historyCollectionSettings;
        }
    }

    export function normalizeHistoryCollectionSettings(payload = {}) {
        return {
            collection_order: uniqueStringList(payload.collection_order),
            config_group_order: normalizeHistoryConfigGroupOrder(payload.config_group_order),
        };
    }

    export function uniqueStringList(value) {
        const list = Array.isArray(value) ? value : [];
        const out = [];
        const seen = new Set();
        for (const item of list) {
            const text = String(item || '').trim();
            if (!text || seen.has(text)) continue;
            out.push(text);
            seen.add(text);
        }
        return out;
    }

    export function normalizeHistoryConfigGroupOrder(value) {
        if (!value || typeof value !== 'object') return {};
        const out = {};
        for (const [key, order] of Object.entries(value)) {
            const cleanKey = String(key || '').trim();
            const cleanOrder = uniqueStringList(order);
            if (cleanKey && cleanOrder.length) out[cleanKey] = cleanOrder;
        }
        return out;
    }

    export function renderTrainingHistoryList() {
        const list = document.getElementById('task-history-list');
        if (!list) return;
        list.innerHTML = '';
        const recentTasks = recentTrainingSidebarTasks();
        if (!recentTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'task-history-empty';
            empty.textContent = historyState.historyTasks.length
                ? '最近没有未归档训练任务；归档和预处理请到历史任务大界面查看。'
                : '暂无历史任务。下一次训练启动后会自动记录。';
            list.appendChild(empty);
            return;
        }
        for (const task of recentTasks) {
            list.appendChild(createHistoryTaskItem(task));
        }
    }

    export function recentTrainingSidebarTasks() {
        return historyState.historyTasks
            .filter((task) => task.job === 'training' && !historyTaskIsArchived(task))
            .sort((a, b) => {
                const aTime = Number(a.started_at || a.updated_at || 0);
                const bTime = Number(b.started_at || b.updated_at || 0);
                return (bTime - aTime) || String(b.id || '').localeCompare(String(a.id || ''), 'zh-CN');
            })
            .slice(0, 6);
    }

    export function renderHistoryManager() {
        const panel = document.getElementById('training-history-manager');
        if (!panel) return;
        syncHistorySelectionWithTasks();
        renderHistoryManagerStats();
        const status = document.getElementById('history-manager-status');
        const list = document.getElementById('history-manager-list');
        const tablePanel = list?.closest('.history-table-panel');
        const selectAll = document.getElementById('history-select-all');
        const mergeBtn = document.getElementById('btn-history-manager-merge');
        if (!list) return;
        const baseVisible = historyManagerBaseFilteredTasks();
        const visible = historyManagerVisibleTasks(baseVisible);
        historyState.historyCurrentVisibleTaskIds = historyTaskIds(visible);
        if (tablePanel) tablePanel.classList.add('collections-mode');
        if (status) {
            const archivedCount = historyState.historyTasks.filter(historyTaskIsArchived).length;
            status.textContent = [
                `共 ${historyState.historyTasks.length} 条记录`,
                `当前分组 ${visible.length} 条`,
                `筛选后 ${baseVisible.length} 条`,
                `归档 ${archivedCount} 条`,
                historyState.historyDropFeedback.message,
            ].filter(Boolean).join(' · ');
            status.dataset.feedbackTone = historyState.historyDropFeedback.tone || '';
        }
        list.innerHTML = '';
        if (!historyState.historyTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'history-manager-empty';
            empty.textContent = '暂无历史任务。';
            list.appendChild(empty);
        } else {
            renderHistoryManagerItems(list, baseVisible);
        }
        if (selectAll) {
            const visibleIds = historyState.historyCurrentVisibleTaskIds;
            const selectedVisible = visibleIds.filter((id) => historyState.selectedHistoryTaskIds.has(id)).length;
            selectAll.checked = visibleIds.length > 0 && selectedVisible === visibleIds.length;
            selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleIds.length;
        }
        renderHistoryBulkBar();
        if (mergeBtn) {
            mergeBtn.disabled = selectedHistoryTasks().filter((task) => task.job === 'training').length === 0;
        }
        renderHistoryDetailDialog();
    }

    export function renderHistoryManagerItems(list, visible) {
        list.dataset.groupMode = 'collections';
        renderHistoryCollectionsWorkbench(list, visible);
    }

    export function resetTrainingExpandedStateOnLeave() {
        if (trainingState.trainingViewMode === 'history') {
            renderHistoryManager();
        }
    }

configureGlobalSettingsBridge({
    resolveGlobalUIScaleDefaultValue,
    syncGlobalUIScaleOverrideField,
    syncAllGlobalUIScaleOverrideFields,
    applyGlobalUIScaleOverrideInputs,
    collectGlobalUIScaleOverridePayload,
    loadGlobalSettings,
    saveGlobalSettings,
    resetGlobalSettings,
    setGlobalSettingsStatus,
    applyGlobalSettingsToInputs,
    collectGlobalSettingsPayload,
    getGlobalModelPathOverrides,
    toggleGlobalSettingHelp,
});

configurePreviewViewBridge({
    loadPreviewSettings,
    savePreviewSettings,
    resetPreviewSettings,
    loadPreviewImages,
    loadPreviewWeights,
    setPreviewSource,
    openTrainingPreview,
    openCurrentTrainingPreview,
    openLiveSamplingPreview,
    openHistoryConfigGroupPreview,
    normalizePreviewGroup,
    renderPreviewTaskSelect,
    changePreviewTask,
    togglePreviewWeightSort,
    openPreviewDialog,
    closePreviewImageDialog,
    openPreviewPanel,
    closePreviewPanel,
    restorePreviewWorkspaceAfterPanelClose,
    setPreviewStatus,
    createPreviewDetailRow,
    createPreviewDetailBlock,
    renderDatasetImageDialogDetails,
    formatTotalPixels,
    copyText,
    formatBytes,
});

configureQueueViewBridge({
    loadTrainingQueue,
    updateTrainingQueueFromPayload,
    renderTrainingQueue,
    refreshQueueRunningProgressViews,
    showTrainingView,
    trainingViewTabs,
    focusTrainingViewTab,
    activateTrainingViewTabButton,
    moveTrainingViewTabFocus,
    bindTrainingViewTabKeyboard,
    renderTrainingViewMode,
    resetTrainingExpandedStateOnLeave,
});

configureHistoryListBridge({
    loadTrainingHistoryList,
    loadHistoryCollectionSettings,
    saveHistoryCollectionSettings,
    normalizeHistoryCollectionSettings,
    uniqueStringList,
    normalizeHistoryConfigGroupOrder,
    renderTrainingHistoryList,
    recentTrainingSidebarTasks,
    renderHistoryManager,
    renderHistoryManagerItems,
});
