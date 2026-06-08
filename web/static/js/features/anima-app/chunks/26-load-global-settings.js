/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.loadGlobalSettings = async function loadGlobalSettings() {
        if (location.protocol === 'file:') return;
        try {
            const data = await api('/api/settings/global');
            if (!data.ok) throw new Error(data.error || '读取全局设置失败');
            globalSettings = data;
            applyGlobalSettingsToInputs(data);
            updateChoiceGuide();
            setGlobalSettingsStatus('', '');
            if (tomlManagerMode === 'output') {
                await loadOutputRuns({ keepSelection: true });
            }
        } catch (e) {
            setGlobalSettingsStatus('读取全局设置失败: ' + e.message, 'error');
        }
    }

    globalThis.saveGlobalSettings = async function saveGlobalSettings() {
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
            globalSettings = {
                ...(globalSettings || {}),
                ...res,
            };
            applyGlobalSettingsToInputs(globalSettings);
            updateChoiceGuide();
            setGlobalSettingsStatus(res.message || '全局设置已保存', 'ok');
        } catch (e) {
            setGlobalSettingsStatus('保存失败: ' + e.message, 'error');
        }
    }

    globalThis.resetGlobalSettings = async function resetGlobalSettings() {
        applyGlobalSettingsToInputs({
            defaults: globalSettings?.defaults || {},
            output_root: globalSettings?.defaults?.output_root || 'output/runs',
            ...Object.fromEntries(GLOBAL_MODEL_PATH_FIELDS.map(([key]) => [key, globalSettings?.defaults?.[key] || ''])),
        });
        await saveGlobalSettings();
    }

    globalThis.setGlobalSettingsStatus = function setGlobalSettingsStatus(text, state = '') {
        const el = document.getElementById('global-settings-status');
        if (!el) return;
        el.textContent = text;
        el.className = `preview-status ${state}`.trim();
    }

    globalThis.applyGlobalSettingsToInputs = function applyGlobalSettingsToInputs(data) {
        const snapshot = data || globalSettings || {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            if (!input) continue;
            const fallback = snapshot?.defaults?.[key] || '';
            input.value = snapshot?.[key] ?? fallback;
        }
    }

    globalThis.collectGlobalSettingsPayload = function collectGlobalSettingsPayload() {
        const payload = {};
        for (const [key, id] of GLOBAL_SETTING_INPUTS) {
            const input = document.getElementById(id);
            payload[key] = input ? input.value : (globalSettings?.[key] || '');
        }
        return payload;
    }

    globalThis.getGlobalModelPathOverrides = function getGlobalModelPathOverrides() {
        const overrides = {};
        const source = globalSettings || {};
        for (const [key] of GLOBAL_MODEL_PATH_FIELDS) {
            const value = source[key] ?? source.defaults?.[key] ?? '';
            if (String(value || '').trim()) {
                overrides[key] = String(value).trim();
            }
        }
        return overrides;
    }

    globalThis.toggleGlobalSettingHelp = function toggleGlobalSettingHelp(button) {
        if (!button) return;
        const helpId = button.getAttribute('aria-controls');
        const help = helpId ? document.getElementById(helpId) : null;
        if (!help) return;
        const visible = help.classList.toggle('visible');
        button.classList.toggle('active', visible);
        button.setAttribute('aria-expanded', visible ? 'true' : 'false');
    }

    // ── 预览图 ──
    globalThis.loadPreviewSettings = async function loadPreviewSettings() {
        return ensurePreviewFeature().loadPreviewSettings();
    }

    globalThis.savePreviewSettings = async function savePreviewSettings() {
        return ensurePreviewFeature().savePreviewSettings();
    }

    globalThis.resetPreviewSettings = async function resetPreviewSettings() {
        return ensurePreviewFeature().resetPreviewSettings();
    }

    globalThis.loadPreviewImages = async function loadPreviewImages() {
        return ensurePreviewFeature().loadPreviewImages();
    }

    globalThis.loadPreviewWeights = async function loadPreviewWeights() {
        return ensurePreviewFeature().loadPreviewWeights();
    }

    globalThis.setPreviewSource = function setPreviewSource(source) {
        return ensurePreviewFeature().setPreviewSource(source);
    }

    globalThis.openTrainingPreview = async function openTrainingPreview(options = {}) {
        return ensurePreviewFeature().openTrainingPreview(options);
    }

    globalThis.openCurrentTrainingPreview = function openCurrentTrainingPreview(event) {
        return ensurePreviewFeature().openCurrentTrainingPreview(event);
    }

    globalThis.openLiveSamplingPreview = function openLiveSamplingPreview(event) {
        return ensurePreviewFeature().openLiveSamplingPreview(event);
    }

    globalThis.openHistoryConfigGroupPreview = async function openHistoryConfigGroupPreview(group) {
        return ensurePreviewFeature().openHistoryConfigGroupPreview(group);
    }

    globalThis.normalizePreviewGroup = function normalizePreviewGroup(group) {
        return ensurePreviewFeature().normalizePreviewGroup(group);
    }

    globalThis.renderPreviewTaskSelect = function renderPreviewTaskSelect() {
        return ensurePreviewFeature().renderPreviewTaskSelect();
    }

    globalThis.changePreviewTask = async function changePreviewTask(taskId) {
        return ensurePreviewFeature().changePreviewTask(taskId);
    }

    globalThis.togglePreviewWeightSort = function togglePreviewWeightSort() {
        return ensurePreviewFeature().togglePreviewWeightSort();
    }

    globalThis.openPreviewDialog = function openPreviewDialog(image) {
        return ensurePreviewFeature().openPreviewDialog(image);
    }

    globalThis.closePreviewImageDialog = function closePreviewImageDialog() {
        return ensurePreviewFeature().closePreviewImageDialog();
    }

    globalThis.openPreviewPanel = function openPreviewPanel() {
        return ensurePreviewFeature().openPreviewPanel();
    }

    globalThis.closePreviewPanel = function closePreviewPanel() {
        return ensurePreviewFeature().closePreviewPanel();
    }

    globalThis.restorePreviewWorkspaceAfterPanelClose = function restorePreviewWorkspaceAfterPanelClose() {
        return ensurePreviewFeature().restorePreviewWorkspaceAfterPanelClose();
    }

    globalThis.setPreviewStatus = function setPreviewStatus(text, state = '') {
        return ensurePreviewFeature().setPreviewStatus(text, state);
    }

    globalThis.createPreviewDetailRow = function createPreviewDetailRow(label, value) {
        return ensurePreviewFeature().createPreviewDetailRow(label, value);
    }

    globalThis.createPreviewDetailBlock = function createPreviewDetailBlock(label, value, preformatted = false) {
        return ensurePreviewFeature().createPreviewDetailBlock(label, value, preformatted);
    }

    globalThis.renderDatasetImageDialogDetails = function renderDatasetImageDialogDetails(box, image, dims) {
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

    globalThis.formatTotalPixels = function formatTotalPixels(totalPixels) {
        const count = Number(totalPixels);
        if (!Number.isFinite(count) || count <= 0) return '-';
        return `${count.toLocaleString('zh-CN')} px (${(count / 1000000).toFixed(2)} MP)`;
    }

    globalThis.copyText = async function copyText(text) {
        return ctx.dom.copyText(text);
    }

    globalThis.formatBytes = function formatBytes(bytes) {
        return ctx.format.formatBytes(bytes);
    }

    // ── 训练队列 ──
    globalThis.loadTrainingQueue = async function loadTrainingQueue() {
        return ensureQueueFeature().loadTrainingQueue();
    }

    globalThis.updateTrainingQueueFromPayload = function updateTrainingQueueFromPayload(payload = {}) {
        return ensureQueueFeature().updateTrainingQueueFromPayload(payload);
    }

    globalThis.renderTrainingQueue = function renderTrainingQueue() {
        return ensureQueueFeature().renderTrainingQueue();
    }

    globalThis.refreshQueueRunningProgressViews = function refreshQueueRunningProgressViews() {
        return ensureQueueFeature().updateRunningQueueProgress();
    }

    globalThis.showTrainingView = function showTrainingView(mode) {
        trainingViewMode = ['live', 'queue', 'history'].includes(mode) ? mode : 'live';
        renderTrainingViewMode();
    }

    globalThis.trainingViewTabs = function trainingViewTabs() {
        return Array.from(document.querySelectorAll('#tab-training .training-view-tab'));
    }

    globalThis.focusTrainingViewTab = function focusTrainingViewTab(mode = trainingViewMode) {
        const target = trainingViewTabs().find((btn) => btn.dataset.trainingView === mode);
        target?.focus({ preventScroll: true });
    }

    globalThis.activateTrainingViewTabButton = function activateTrainingViewTabButton(button) {
        const nextMode = button?.dataset.trainingView || 'live';
        if (nextMode === 'live' && typeof returnToLiveTraining === 'function') {
            returnToLiveTraining({ refresh: false });
        } else {
            showTrainingView(nextMode);
        }
        focusTrainingViewTab(nextMode);
    }

    globalThis.moveTrainingViewTabFocus = function moveTrainingViewTabFocus(currentButton, offset = 0) {
        const tabs = trainingViewTabs();
        if (!tabs.length) return;
        const currentIndex = Math.max(0, tabs.indexOf(currentButton));
        const nextIndex = (currentIndex + offset + tabs.length) % tabs.length;
        activateTrainingViewTabButton(tabs[nextIndex]);
    }

    globalThis.bindTrainingViewTabKeyboard = function bindTrainingViewTabKeyboard() {
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

    globalThis.renderTrainingViewMode = function renderTrainingViewMode() {
        const queueView = document.getElementById('training-queue-manager');
        const monitorView = document.getElementById('training-monitor-view');
        const historyManager = document.getElementById('training-history-manager');
        const historyPlaceholder = document.getElementById('training-history-placeholder');
        const workspace = document.querySelector('#tab-training .training-workspace');
        const isQueue = trainingViewMode === 'queue';
        const isHistory = trainingViewMode === 'history';
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
            const active = btn.dataset.trainingView === trainingViewMode;
            btn.classList.toggle('active', active);
            btn.setAttribute('aria-selected', String(active));
            btn.tabIndex = active ? 0 : -1;
        });
        if (isHistory) {
            renderHistoryManager();
        }
    }

    // ── 状态轮询 ──
    globalThis.pollStatus = async function pollStatus(options = {}) {
        if (isHistoryReviewMode()) return;
        try {
            const status = await api('/api/training/status');
            if (status.ok === false) throw new Error(status.error || '读取训练状态失败');
            if (trainingStatusPollFailures) {
                trainingStatusPollFailures = 0;
                updateLogStatusText();
            }
            updateStatus({
                state: status.status,
                variant: status.variant,
                preset: status.preset,
                methods_subdir: status.methods_subdir,
                job: status.job,
                last_output_at: status.last_output_at,
                last_log_id: status.last_log_id,
                last_log_line: status.last_log_line,
                error_hint: status.error_hint,
                output_dir: status.output_dir,
                sample_dir: status.sample_dir,
                sample_config: status.sample_config,
                run_dir: status.run_dir,
                runtime_config_file: status.runtime_config_file,
                original_config_file: status.original_config_file,
                dataset_config_file: status.dataset_config_file,
                model_cache_dir: status.model_cache_dir,
                dataset_cache_dir: status.dataset_cache_dir,
                training_output_dir: status.training_output_dir,
                logs_dir: status.logs_dir,
            });
            applyStatusSnapshotFallbacks(status);
            if ((status.last_log_id || 0) > trainingRuntime.lastLogId) {
                await replayTrainingLogs();
            } else if (options.forceReplayMetrics || isLiveRunningState()) {
                await replayMetricsHistory();
            }
        } catch (e) {
            trainingStatusPollFailures += 1;
            if (trainingStatusPollFailures < 3) return;
            const message = `训练状态轮询连续失败 ${trainingStatusPollFailures} 次: ${e.message}`;
            setLogStatus('状态轮询失败', 'error');
            setTrainingHealthNotice(message, 'error');
            if (trainingStatusPollFailures === 3) appendLog(`[状态] ${message}`);
        }
    }

    globalThis.applyStatusSnapshotFallbacks = function applyStatusSnapshotFallbacks(status = {}) {
        if (!isLiveRunningState(status.status)) return;
        if (hasStatusPayload(status.latest_progress)) {
            updateProgress(status.latest_progress, { replay: true });
        }
        if (hasStatusPayload(status.latest_metric)) {
            updateMetrics(status.latest_metric, { replay: true });
        }
        if (hasStatusPayload(status.latest_system)) {
            updateSystem(status.latest_system, { replay: true });
        }
    }

    globalThis.hasStatusPayload = function hasStatusPayload(value) {
        return value && typeof value === 'object' && Object.keys(value).length > 0;
    }

    globalThis.loadTrainingHistoryList = async function loadTrainingHistoryList() {
        if (location.protocol === 'file:') return;
        try {
            const params = new URLSearchParams();
            params.set('include_archived', '1');
            const suffix = params.toString() ? `?${params.toString()}` : '';
            const payload = await api(`/api/training/history${suffix}`);
            historyTasks = payload.tasks || [];
            await loadHistoryCollectionSettings();
            renderTrainingHistoryList();
            renderHistoryManager();
            renderPreviewTaskSelect();
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

    globalThis.loadHistoryCollectionSettings = async function loadHistoryCollectionSettings() {
        if (location.protocol === 'file:') return;
        try {
            const payload = await api('/api/training/history/collections/settings');
            if (payload.ok !== false) {
                historyCollectionSettings = normalizeHistoryCollectionSettings(payload);
            }
        } catch (e) {
            appendLog(`[状态] 读取历史集合设置失败: ${e.message}`);
        }
    }

    globalThis.saveHistoryCollectionSettings = async function saveHistoryCollectionSettings(nextSettings) {
        historyCollectionSettings = normalizeHistoryCollectionSettings(nextSettings);
        if (location.protocol === 'file:') {
            renderHistoryManager();
            return historyCollectionSettings;
        }
        try {
            const payload = await api('/api/training/history/collections/settings', {
                method: 'PUT',
                body: JSON.stringify(historyCollectionSettings),
            });
            if (payload.ok === false) throw new Error(payload.error || '保存集合设置失败');
            historyCollectionSettings = normalizeHistoryCollectionSettings(payload);
            renderHistoryManager();
            return historyCollectionSettings;
        } catch (e) {
            appendLog(`[状态] 保存历史集合设置失败: ${e.message}`);
            renderHistoryManager();
            return historyCollectionSettings;
        }
    }

    globalThis.normalizeHistoryCollectionSettings = function normalizeHistoryCollectionSettings(payload = {}) {
        return {
            collection_order: uniqueStringList(payload.collection_order),
            config_group_order: normalizeHistoryConfigGroupOrder(payload.config_group_order),
        };
    }

    globalThis.uniqueStringList = function uniqueStringList(value) {
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

    globalThis.normalizeHistoryConfigGroupOrder = function normalizeHistoryConfigGroupOrder(value) {
        if (!value || typeof value !== 'object') return {};
        const out = {};
        for (const [key, order] of Object.entries(value)) {
            const cleanKey = String(key || '').trim();
            const cleanOrder = uniqueStringList(order);
            if (cleanKey && cleanOrder.length) out[cleanKey] = cleanOrder;
        }
        return out;
    }

    globalThis.renderTrainingHistoryList = function renderTrainingHistoryList() {
        const list = document.getElementById('task-history-list');
        if (!list) return;
        list.innerHTML = '';
        const recentTasks = recentTrainingSidebarTasks();
        if (!recentTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'task-history-empty';
            empty.textContent = historyTasks.length
                ? '最近没有未归档训练任务；归档和预处理请到历史任务大界面查看。'
                : '暂无历史任务。下一次训练启动后会自动记录。';
            list.appendChild(empty);
            return;
        }
        for (const task of recentTasks) {
            list.appendChild(createHistoryTaskItem(task));
        }
    }

    globalThis.recentTrainingSidebarTasks = function recentTrainingSidebarTasks() {
        return historyTasks
            .filter((task) => task.job === 'training' && !historyTaskIsArchived(task))
            .sort((a, b) => {
                const aTime = Number(a.started_at || a.updated_at || 0);
                const bTime = Number(b.started_at || b.updated_at || 0);
                return (bTime - aTime) || String(b.id || '').localeCompare(String(a.id || ''), 'zh-CN');
            })
            .slice(0, 6);
    }

    globalThis.renderHistoryManager = function renderHistoryManager() {
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
        historyCurrentVisibleTaskIds = historyTaskIds(visible);
        if (tablePanel) tablePanel.classList.add('collections-mode');
        if (status) {
            const archivedCount = historyTasks.filter(historyTaskIsArchived).length;
            status.textContent = [
                `共 ${historyTasks.length} 条记录`,
                `当前分组 ${visible.length} 条`,
                `筛选后 ${baseVisible.length} 条`,
                `归档 ${archivedCount} 条`,
                historyDropFeedback.message,
            ].filter(Boolean).join(' · ');
            status.dataset.feedbackTone = historyDropFeedback.tone || '';
        }
        list.innerHTML = '';
        if (!historyTasks.length) {
            const empty = document.createElement('div');
            empty.className = 'history-manager-empty';
            empty.textContent = '暂无历史任务。';
            list.appendChild(empty);
        } else {
            renderHistoryManagerItems(list, baseVisible);
        }
        if (selectAll) {
            const visibleIds = historyCurrentVisibleTaskIds;
            const selectedVisible = visibleIds.filter((id) => selectedHistoryTaskIds.has(id)).length;
            selectAll.checked = visibleIds.length > 0 && selectedVisible === visibleIds.length;
            selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visibleIds.length;
        }
        renderHistoryBulkBar();
        if (mergeBtn) {
            mergeBtn.disabled = selectedHistoryTasks().filter((task) => task.job === 'training').length === 0;
        }
        renderHistoryDetailDialog();
    }

    globalThis.renderHistoryManagerItems = function renderHistoryManagerItems(list, visible) {
        list.dataset.groupMode = 'collections';
        renderHistoryCollectionsWorkbench(list, visible);
    }

    globalThis.resetTrainingExpandedStateOnLeave = function resetTrainingExpandedStateOnLeave() {
        if (trainingViewMode === 'history') {
            renderHistoryManager();
        }
    }
