/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { HISTORY_UNGROUPED_COLLECTION_KEY } from '../helpers/app-constants.js?v=module-bootstrap-20260707-93';
import { setHistoryDropFeedback } from '../helpers/history-collection-drag-bridge.js?v=module-bootstrap-20260707-93';
import {
    configureHistoryCollectionsBridge,
    ensureHistoryCollectionOrderValue,
    historyCollectionStorageKey,
    historyTaskCollectionValue,
} from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260707-93';
import {
    applyHistoryTaskIdsBatchAction,
    createHistoryActionButton,
    deleteHistoryTasksThorough,
    loadConfigGroupTimeline,
    showHistoryCollectionSelectDialog,
    showHistoryTaskConfirmDialog,
} from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260707-93';
import { configGroupLabel } from '../helpers/history-timeline-bridge.js?v=module-bootstrap-20260707-93';
import { showTrainingView } from '../helpers/queue-view-bridge.js?v=module-bootstrap-20260707-93';
import { openHistoryConfigGroupPreview } from '../helpers/preview-view-bridge.js?v=module-bootstrap-20260707-93';
import { renderHistoryManager, saveHistoryCollectionSettings, normalizeHistoryCollectionSettings, uniqueStringList, normalizeHistoryConfigGroupOrder } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260707-93';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260707-93';

const historyState = getHistoryState();

    export function historyTaskCollectionLabel(task) {
        return historyTaskCollectionValue(task) || '未分类';
    }

    export function historyTaskCollectionKey(task) {
        const value = historyTaskCollectionValue(task);
        return value ? `collection:${value}` : HISTORY_UNGROUPED_COLLECTION_KEY;
    }

    export function historyConfigGroupCollectionMap(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const group = historyConfigGroupFromTask(task);
            const key = configGroupKey(group);
            if (!map.has(key)) map.set(key, new Set());
            map.get(key).add(historyTaskCollectionLabel(task));
        }
        return map;
    }

    export function historyTaskIds(tasks) {
        return (tasks || []).map((task) => task.id).filter(Boolean);
    }

    export function historyTasksAllSelected(tasks) {
        const ids = historyTaskIds(tasks);
        return ids.length > 0 && ids.every((id) => historyState.selectedHistoryTaskIds.has(id));
    }

    export function toggleHistoryTaskSelection(tasks) {
        const ids = historyTaskIds(tasks);
        const selected = ids.every((id) => historyState.selectedHistoryTaskIds.has(id));
        ids.forEach((id) => {
            if (selected) historyState.selectedHistoryTaskIds.delete(id);
            else historyState.selectedHistoryTaskIds.add(id);
        });
        renderHistoryManager();
    }

    export function historyManagerGroupMetaParts(tasks, extra = []) {
        const trainingCount = tasks.filter((task) => task.job === 'training').length;
        const preprocessCount = tasks.filter((task) => task.job === 'preprocess').length;
        const errorCount = tasks.filter((task) => ['error', 'interrupted'].includes(task.state)).length;
        const archivedCount = tasks.filter(historyTaskIsArchived).length;
        const queueCount = tasks.filter((task) => task.from_queue || task.queue_item_id).length;
        return [
            `${tasks.length} 条任务`,
            trainingCount ? `${trainingCount} 次训练` : '',
            preprocessCount ? `${preprocessCount} 个预处理` : '',
            errorCount ? `${errorCount} 个异常/中断` : '',
            archivedCount ? `${archivedCount} 个归档` : '',
            queueCount ? `${queueCount} 个队列来源` : '',
            ...extra,
        ].filter(Boolean);
    }

    export function historyCompactGroupMetaParts(tasks, extra = []) {
        const trainingCount = tasks.filter((task) => task.job === 'training').length;
        const preprocessCount = tasks.filter((task) => task.job === 'preprocess').length;
        const errorCount = tasks.filter((task) => ['error', 'interrupted'].includes(task.state)).length;
        const archivedCount = tasks.filter(historyTaskIsArchived).length;
        const queueCount = tasks.filter((task) => task.from_queue || task.queue_item_id).length;
        return [
            `${tasks.length} 条`,
            trainingCount ? `${trainingCount} 训` : '',
            preprocessCount ? `${preprocessCount} 预` : '',
            errorCount ? `${errorCount} 异常` : '',
            archivedCount ? `${archivedCount} 归档` : '',
            queueCount ? `${queueCount} 队列` : '',
            ...extra,
        ].filter(Boolean);
    }

    export function commonHistoryCollectionValue(tasks) {
        const values = Array.from(new Set((tasks || []).map(historyTaskCollectionValue).filter(Boolean)));
        return values.length === 1 ? values[0] : '';
    }

    export function createHistoryManagerGroupButton(label, handler, tone = '') {
        const btn = createHistoryActionButton(label, handler, tone);
        btn.classList.add('history-manager-group-action');
        return btn;
    }

    export function createHistoryConfigGroupMergeButton(group) {
        const btn = createHistoryManagerGroupButton('查看', () => loadConfigGroupTimeline(group, { skipSelectionDialog: true }));
        btn.title = '查阅这个自动配置分组内的训练日志、Loss 曲线和任务明细';
        return btn;
    }

    export function createHistoryConfigGroupPreviewButton(group) {
        const btn = createHistoryManagerGroupButton('预览', () => openHistoryConfigGroupPreview(group));
        btn.title = '汇总查看这个配置分组下所有训练任务的样张和权重';
        return btn;
    }

    export function canPreviewHistoryConfigGroup(group) {
        return Boolean(group && group.methods_subdir && group.variant && group.methods_subdir !== '手动选择');
    }

    export async function setHistoryCollectionForTasks(tasks, value = '', description = '') {
        const ids = historyTaskIds(tasks);
        if (!ids.length) return;
        const group = await showHistoryCollectionSelectDialog({
            title: value ? '重命名集合' : '设置集合',
            description: `${ids.length} 条历史任务${description ? ` · ${description}` : ''}`,
            value,
            confirmText: '保存集合',
        });
        if (group === null) return;
        await applyHistoryTaskIdsToCollection(ids, group.trim());
    }

    export async function renameHistoryCollection(collection) {
        const oldValue = String(collection?.value || '').trim();
        if (!oldValue || collection?.is_ungrouped) return;
        const nextValue = await showHistoryCollectionSelectDialog({
            title: '重命名集合',
            description: collection.label || oldValue,
            value: oldValue,
            confirmText: '保存集合',
        });
        if (nextValue === null) return;
        const clean = String(nextValue || '').trim();
        if (!clean || clean === oldValue) return;
        const oldKey = `collection:${oldValue}`;
        const newKey = `collection:${clean}`;
        const collectionOrder = renameHistoryCollectionOrderValue(oldValue, clean);
        const configGroupOrder = renameHistoryConfigGroupOrderKey(oldValue, clean);
        const ids = historyTaskIds(collection.tasks || []);
        if (ids.length) {
            const res = await applyHistoryTaskIdsBatchAction(ids, 'set_group', { group: clean });
            if (res === null) return;
        }
        historyState.historyCollectionSettings = normalizeHistoryCollectionSettings({
            ...historyState.historyCollectionSettings,
            collection_order: collectionOrder,
            config_group_order: configGroupOrder,
        });
        await saveHistoryCollectionSettings(historyState.historyCollectionSettings);
        historyState.selectedHistoryCollectionKey = newKey;
        if (historyState.historyCollectionWorkbenchTarget === oldValue) historyState.historyCollectionWorkbenchTarget = clean;
        renderHistoryManager();
        setHistoryDropFeedback(`已重命名分组「${oldValue}」为「${clean}」。`, 'ok');
    }

    export async function clearHistoryCollection(collection) {
        const value = String(collection?.value || '').trim();
        if (!value || collection?.is_ungrouped) return;
        const ids = historyTaskIds(collection.tasks || []);
        const ok = await showHistoryTaskConfirmDialog({
            title: ids.length ? '清空集合' : '删除空集合',
            description: `${collection.label || value} · ${ids.length} 条历史任务`,
            message: ids.length
                ? '只会移除这些历史任务的集合名称，并删除右侧空集合入口；同配置文件自动分组、日志、权重和运行目录都会保留。'
                : '这个集合没有任务，会从右侧分组导航中移除。',
            confirmText: ids.length ? '清空集合' : '删除空集合',
        });
        if (!ok) return;
        if (ids.length) {
            const res = await applyHistoryTaskIdsBatchAction(ids, 'set_group', { group: '' });
            if (res === null) return;
        }
        await removeHistoryCollectionSettingValue(value);
        if (historyState.selectedHistoryCollectionKey === `collection:${value}`) {
            historyState.selectedHistoryCollectionKey = HISTORY_UNGROUPED_COLLECTION_KEY;
        }
        if (historyState.historyCollectionWorkbenchTarget === value) historyState.historyCollectionWorkbenchTarget = '';
        renderHistoryManager();
        setHistoryDropFeedback(ids.length ? `已清空分组「${collection.label || value}」。` : `已删除空分组「${collection.label || value}」。`, 'ok');
    }

    export function renameHistoryCollectionOrderValue(oldValue, newValue) {
        const oldClean = String(oldValue || '').trim();
        const newClean = String(newValue || '').trim();
        const out = [];
        const seen = new Set();
        for (const value of uniqueStringList(historyState.historyCollectionSettings.collection_order || [])) {
            const next = value === oldClean ? newClean : value;
            if (!next || seen.has(next)) continue;
            out.push(next);
            seen.add(next);
        }
        if (newClean && !seen.has(newClean)) out.push(newClean);
        return out;
    }

    export function renameHistoryConfigGroupOrderKey(oldValue, newValue) {
        const oldKey = historyCollectionStorageKey(oldValue);
        const newKey = historyCollectionStorageKey(newValue);
        const current = normalizeHistoryConfigGroupOrder(historyState.historyCollectionSettings.config_group_order);
        if (!oldKey || !newKey || oldKey === newKey) return current;
        const oldOrder = current[oldKey] || [];
        const newOrder = uniqueStringList([...(current[newKey] || []), ...oldOrder]);
        delete current[oldKey];
        if (newOrder.length) current[newKey] = newOrder;
        return current;
    }

    export async function removeHistoryCollectionSettingValue(value) {
        const clean = String(value || '').trim();
        if (!clean) return historyState.historyCollectionSettings;
        const settings = {
            ...historyState.historyCollectionSettings,
            collection_order: uniqueStringList(historyState.historyCollectionSettings.collection_order || []).filter((item) => item !== clean),
            config_group_order: Object.fromEntries(
                Object.entries(normalizeHistoryConfigGroupOrder(historyState.historyCollectionSettings.config_group_order))
                    .filter(([key]) => key !== historyCollectionStorageKey(clean)),
            ),
        };
        return saveHistoryCollectionSettings(settings);
    }

    export async function setHistoryCollectionForTasksDirect(tasks, value) {
        const ids = historyTaskIds(tasks);
        const group = String(value || '').trim();
        if (!ids.length || !group) return;
        await applyHistoryTaskIdsToCollection(ids, group);
    }

    export async function applySelectedHistoryTasksToCollection(value) {
        const ids = historyTaskIds(selectedHistoryTasks());
        if (!ids.length) return;
        await applyHistoryTaskIdsToCollection(ids, String(value || '').trim(), { clearSelection: true });
    }

    export async function applyHistoryTaskIdsToCollection(ids, group, options = {}) {
        const clean = String(group || '').trim();
        if (clean) await ensureHistoryCollectionOrderValue(clean);
        return applyHistoryTaskIdsBatchAction(ids, 'set_group', { group: clean }, options);
    }

    export async function clearSelectedHistoryCollection() {
        const tasks = selectedHistoryTasks();
        if (!tasks.length) return;
        const ok = await showHistoryTaskConfirmDialog({
            title: '清除已选集合',
            description: `${tasks.length} 条历史任务`,
            message: '只会移除已选任务的集合名称；同配置文件自动分组、历史详情、日志和运行目录都会保留。',
            confirmText: '清除集合',
        });
        if (!ok) return;
        await applyHistoryTaskIdsBatchAction(historyTaskIds(tasks), 'set_group', { group: '' }, { clearSelection: true });
    }

    export async function clearHistoryCollectionForTasks(tasks, description = '') {
        const ids = historyTaskIds(tasks);
        if (!ids.length) return;
        const ok = await showHistoryTaskConfirmDialog({
            title: '清空集合',
            description: `${ids.length} 条历史任务${description ? ` · ${description}` : ''}`,
            message: '只会移除这些历史任务的集合名称；同配置文件自动分组、日志、权重和运行目录都会保留。',
            confirmText: '清空集合',
        });
        if (!ok) return;
        await applyHistoryTaskIdsBatchAction(ids, 'set_group', { group: '' });
    }

    export async function archiveHistoryTasksByIds(tasks, archived, description = '') {
        const ids = historyTaskIds(tasks);
        if (!ids.length) return;
        const ok = await showHistoryTaskConfirmDialog({
            title: archived ? '批量归档' : '批量取消归档',
            description: `${ids.length} 条历史任务${description ? ` · ${description}` : ''}`,
            message: archived ? '归档后默认会隐藏这些任务。' : '取消归档后这些任务会重新出现在默认列表中。',
            confirmText: archived ? '归档' : '取消归档',
        });
        if (!ok) return;
        await applyHistoryTaskIdsBatchAction(ids, archived ? 'archive' : 'unarchive');
    }

    export async function deleteHistoryTasksByIds(tasks) {
        await deleteHistoryTasksThorough(historyTaskIds(tasks));
    }

    export function syncHistorySelectionWithTasks() {
        const valid = new Set(historyState.historyTasks.map((task) => task.id).filter(Boolean));
        historyState.selectedHistoryTaskIds = new Set(
            Array.from(historyState.selectedHistoryTaskIds).filter((id) => valid.has(id))
        );
    }

    export function selectedHistoryTasks() {
        const ids = historyState.selectedHistoryTaskIds;
        const visible = new Set(historyState.historyCurrentVisibleTaskIds);
        if (!visible.size) return [];
        return historyState.historyTasks.filter((task) => ids.has(task.id) && visible.has(task.id));
    }

    export function renderHistoryBulkBar() {
        const bar = document.getElementById('history-bulk-bar');
        const summary = document.getElementById('history-bulk-summary');
        if (!bar || !summary) return;
        const tasks = selectedHistoryTasks();
        bar.hidden = tasks.length === 0;
        summary.textContent = `已选 ${tasks.length} 项`;
    }

    export function syncHistoryFilterControls() {
        const controls = {
            'history-manager-search': 'search',
            'history-filter-kind': 'kind',
            'history-filter-state': 'state',
            'history-filter-archived': 'archived',
            'history-filter-source': 'source',
            'history-sort-mode': 'sort',
        };
        for (const [id, key] of Object.entries(controls)) {
            const el = document.getElementById(id);
            if (el) el.value = historyState.historyManagerFilters[key] || historyManagerFilterDefault(key);
        }
        const collectionSearch = document.getElementById('history-collection-search');
        if (collectionSearch) collectionSearch.value = historyState.historyCollectionSearch || '';
        const configGroupSearch = document.getElementById('history-config-group-search');
        if (configGroupSearch) configGroupSearch.value = historyState.historyConfigGroupSearch || '';
    }

    export function historyManagerFilterDefault(key) {
        if (key === 'search') return '';
        if (key === 'archived') return 'active';
        if (key === 'sort') return 'newest';
        return 'all';
    }

    export function openHistoryCollectionsWorkbench() {
        syncHistoryFilterControls();
        showTrainingView('history');
        renderHistoryManager();
    }

    export function groupHistoryTasks(tasks) {
        const map = new Map();
        for (const task of tasks) {
            const group = historyConfigGroupFromTask(task);
            if (!map.has(group.key)) {
                map.set(group.key, { ...group, tasks: [] });
            }
            map.get(group.key).tasks.push(task);
        }
        return Array.from(map.values())
            .map(enrichHistoryGroup)
            .sort((a, b) => {
                const aTime = Math.max(...a.tasks.map((task) => Number(task.started_at || 0)));
                const bTime = Math.max(...b.tasks.map((task) => Number(task.started_at || 0)));
                return (bTime - aTime) || historyGroupDisplayLabel(a).localeCompare(historyGroupDisplayLabel(b), 'zh-CN');
            });
    }

    export function historyConfigGroupFromTask(task) {
        const methodsSubdir = String(task.methods_subdir || '-');
        const variant = String(task.variant || '-');
        const preset = String(task.preset || 'default');
        const legacyLabel = `${methodsSubdir} / ${variant} / ${preset}`;
        const historyKey = String(task.history_group_key || '').trim();
        const sourceConfig = String(task.history_source_config_file || '').trim();
        const groupLabel = String(task.history_group_label || '').trim() || sourceConfig || legacyLabel;
        return {
            key: historyKey || [methodsSubdir, variant, preset].join('\u0001'),
            history_group_key: historyKey || '',
            history_group_label: groupLabel,
            history_source_config_file: sourceConfig,
            methods_subdir: methodsSubdir,
            variant,
            preset,
            label: groupLabel,
            legacy_label: legacyLabel,
        };
    }

    export function configGroupKey(group) {
        if (group?.key) return group.key;
        if (group?.history_group_key) return group.history_group_key;
        return [group.methods_subdir || '-', group.variant || '-', group.preset || 'default'].join('\u0001');
    }

    export function enrichHistoryGroup(group) {
        const tasks = [...(group.tasks || [])].sort((a, b) => {
            const aTime = Number(a.started_at || 0);
            const bTime = Number(b.started_at || 0);
            return (bTime - aTime) || String(b.id || '').localeCompare(String(a.id || ''), 'zh-CN');
        });
        const latestTask = tasks[0] || {};
        const runDirs = new Set(tasks.map((task) => historyTaskRunPath(task)).filter(Boolean));
        return {
            ...group,
            tasks,
            latestTask,
            run_count: runDirs.size,
            display_label: historyTaskDisplayName(latestTask) || group.label,
            source_label: group.history_source_config_file || '',
            fallback_group_label: group.history_group_label || group.legacy_label || group.label,
        };
    }

    export function historyTaskDisplayName(task) {
        if (!task) return '';
        const customName = String(task.name || '').trim();
        if (task.training_mode === 'continue_lora') {
            const kind = String(task.continue_from_weight_kind || 'LoRA').trim() || 'LoRA';
            const name = String(task.continue_from_weight_name || '').trim();
            const continueName = `权重热启动 ${kind}${name ? ` · ${name}` : ''}`;
            return customName && customName !== continueName ? customName : continueName;
        }
        if (customName) return customName;
        const defaultName = String(
            task.history_run_label
            || runLabelFromPath(task.run_dir || task.training_output_dir || task.output_dir)
            || task.id
            || ''
        ).trim();
        return defaultName;
    }

    export function historyTaskIsArchived(task) {
        if (Boolean(task?.archived)) return true;
        return task?.job === 'preprocess' && !task?.updated_at;
    }

    export function historyTaskRunPath(task) {
        return String(task?.run_dir || task?.training_output_dir || task?.output_dir || '').trim();
    }

    export function historyResumeLabel(task) {
        const resume = task?.resume_from || {};
        if (!resume || typeof resume !== 'object') return '';
        const checkpoint = String(resume.checkpoint_name || '').trim();
        const step = resume.checkpoint_step !== undefined && resume.checkpoint_step !== null
            ? String(resume.checkpoint_step).trim()
            : '';
        if (checkpoint && step) return `从检查点恢复: ${checkpoint} · step ${step}`;
        if (checkpoint) return `从检查点恢复: ${checkpoint}`;
        if (step) return `从检查点恢复: step ${step}`;
        return resume.source_task_id ? '从检查点恢复' : '';
    }

    export function historyQueueLabel(task) {
        const queueId = String(task?.queue_item_id || '').trim();
        if (!Boolean(task?.from_queue) && !queueId) return '';
        const attempt = Number(task?.queue_attempt || 1);
        return attempt > 1 ? `来自队列 · 第 ${attempt} 次尝试` : '来自队列';
    }

    export function historyContinueLabel(task) {
        if (task?.training_mode !== 'continue_lora') return '';
        const kind = String(task.continue_from_weight_kind || 'LoRA').trim() || 'LoRA';
        const name = String(task.continue_from_weight_name || '').trim();
        return `权重热启动 ${kind}${name ? `: ${name}` : ''}`;
    }

    export function historyContinuePathLabel(task) {
        if (task?.training_mode !== 'continue_lora') return '';
        const path = String(task.continue_from_weight_abs_path || '').trim();
        return path ? `基于: ${path}` : '';
    }

    export function runLabelFromPath(value) {
        const text = String(value || '').replace(/\\/g, '/').trim();
        if (!text) return '';
        const parts = text.split('/').filter(Boolean);
        if (!parts.length) return text;
        if (parts[parts.length - 1] === 'training_output' && parts.length > 1) {
            return parts[parts.length - 2];
        }
        return parts[parts.length - 1];
    }

    export function historyGroupDisplayLabel(group) {
        return String(group?.display_label || group?.history_run_label || group?.label || configGroupLabel(group) || '').trim();
    }

    export function createHistoryGroupHeading(group) {
        const heading = document.createElement('div');
        heading.className = 'task-history-group-title';
        const trainingCount = group.tasks.filter((task) => task.job === 'training').length;
        const preprocessCount = group.tasks.filter((task) => task.job === 'preprocess').length;

        const title = document.createElement('span');
        const name = document.createElement('strong');
        name.textContent = historyGroupDisplayLabel(group);
        const meta = document.createElement('em');
        meta.textContent = [
            group.source_label ? `源配置: ${group.source_label}` : `配置分组: ${group.fallback_group_label || group.label}`,
            `${trainingCount} 次训练`,
            preprocessCount ? `${preprocessCount} 个预处理` : '',
            group.run_count ? `${group.run_count} 个运行目录` : '',
        ].filter(Boolean).join(' · ');
        title.append(name, meta);
        heading.appendChild(title);

        if (trainingCount) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'task-history-group-action';
            btn.textContent = '查阅分组详情';
            btn.title = '查阅这个自动配置分组内的训练日志、Loss 曲线和任务明细';
            btn.addEventListener('click', () => loadConfigGroupTimeline(group, { skipSelectionDialog: true }));
            heading.appendChild(btn);
            const previewBtn = document.createElement('button');
            previewBtn.type = 'button';
            previewBtn.className = 'task-history-group-action';
            previewBtn.textContent = '分组预览';
            previewBtn.title = '汇总查看这个配置分组下所有训练任务的样张和权重';
            previewBtn.addEventListener('click', () => openHistoryConfigGroupPreview(group));
            heading.appendChild(previewBtn);
        }
        return heading;
    }

configureHistoryCollectionsBridge({
    historyTaskCollectionLabel,
    historyTaskCollectionKey,
    historyConfigGroupCollectionMap,
    historyTaskIds,
    historyTasksAllSelected,
    toggleHistoryTaskSelection,
    historyManagerGroupMetaParts,
    historyCompactGroupMetaParts,
    commonHistoryCollectionValue,
    createHistoryManagerGroupButton,
    createHistoryConfigGroupMergeButton,
    createHistoryConfigGroupPreviewButton,
    canPreviewHistoryConfigGroup,
    setHistoryCollectionForTasks,
    renameHistoryCollection,
    clearHistoryCollection,
    renameHistoryCollectionOrderValue,
    renameHistoryConfigGroupOrderKey,
    removeHistoryCollectionSettingValue,
    setHistoryCollectionForTasksDirect,
    applySelectedHistoryTasksToCollection,
    applyHistoryTaskIdsToCollection,
    clearSelectedHistoryCollection,
    clearHistoryCollectionForTasks,
    archiveHistoryTasksByIds,
    deleteHistoryTasksByIds,
    syncHistorySelectionWithTasks,
    selectedHistoryTasks,
    renderHistoryBulkBar,
    syncHistoryFilterControls,
    historyManagerFilterDefault,
    openHistoryCollectionsWorkbench,
    groupHistoryTasks,
    historyConfigGroupFromTask,
    configGroupKey,
    enrichHistoryGroup,
    historyTaskDisplayName,
    historyTaskIsArchived,
    historyTaskRunPath,
    historyResumeLabel,
    historyQueueLabel,
    historyContinueLabel,
    historyContinuePathLabel,
    runLabelFromPath,
    historyGroupDisplayLabel,
    createHistoryGroupHeading,
});
