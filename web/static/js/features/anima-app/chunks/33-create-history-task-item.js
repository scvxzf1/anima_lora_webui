/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    applyHistoryTaskIdsToCollection,
    historyContinueLabel,
    historyContinuePathLabel,
    historyQueueLabel,
    historyResumeLabel,
    historyTaskDisplayName,
    historyTaskIds,
    historyTaskIsArchived,
    selectedHistoryTasks,
} from '../helpers/history-collections-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { clearHistoryManagerDetail, configureHistoryTaskActionsBridge, isHistoryDetailDialogOpen, loadConfigGroupTimeline, loadHistoryTask, openSidebarHistoryTask, renderHistoryManagerDetail, showHistoryCollectionSelectDialog, showHistoryTaskConfirmDialog, showHistoryTaskDialog, showHistoryTaskMessageDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { isHistoryReviewMode } from '../helpers/history-detail-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { historyStateLabel } from '../helpers/history-timeline-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { loadTrainingHistoryList, uniqueStringList } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { api } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';

const historyState = getHistoryState();

    export function createHistoryTaskItem(task) {
        const card = document.createElement('article');
        card.className = 'task-history-item';
        card.dataset.taskId = String(task.id || '');
        card.dataset.historyTaskId = String(task.id || '');
        if (task.id === historyState.viewingHistoryTaskId && isHistoryReviewMode()) card.classList.add('active');
        const archived = historyTaskIsArchived(task);
        if (archived) card.classList.add('archived');

        const main = document.createElement('button');
        main.type = 'button';
        main.className = 'task-history-main';
        main.addEventListener('click', () => openSidebarHistoryTask(task.id));

        const title = document.createElement('strong');
        title.className = 'task-history-title';
        title.textContent = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
        const meta = document.createElement('span');
        meta.className = 'task-history-meta';
        meta.textContent = [
            task.job === 'preprocess' ? '预处理' : '训练',
            historyQueueLabel(task),
            historyContinueLabel(task),
            historyResumeLabel(task),
            historyStateLabel(task.state),
            task.started_at_text || task.id,
            archived ? '已归档' : '',
        ].filter(Boolean).join(' · ');
        const pathValue = task.run_dir || task.training_output_dir || task.output_dir || task.history_dir || task.id;
        const paths = document.createElement('em');
        paths.className = 'task-history-path';
        paths.title = [
            pathValue ? `目录: ${pathValue}` : '',
            historyContinuePathLabel(task),
        ].filter(Boolean).join(' · ');
        const pathLabel = document.createElement('span');
        pathLabel.textContent = '目录';
        const pathText = document.createElement('code');
        pathText.textContent = compactPathLabel(pathValue);
        if (pathValue) pathText.title = String(pathValue);
        paths.append(pathLabel, pathText);
        const continuePath = historyContinuePathLabel(task);
        if (continuePath) {
            const continueFull = continuePath.replace(/^基于:\s*/, '');
            const continueText = document.createElement('code');
            continueText.textContent = compactPathLabel(continueFull);
            if (continueFull) continueText.title = String(continueFull);
            paths.appendChild(continueText);
        }
        const counts = document.createElement('em');
        counts.className = 'task-history-counts';
        counts.dataset.liveHistoryCounts = 'sidebar';
        counts.textContent = `${task.metric_count || 0} loss点 / ${task.log_count || 0} 日志`;
        main.append(title, meta, paths, counts);

        const actions = document.createElement('div');
        actions.className = 'task-history-actions';
        if (task.job === 'training') {
            actions.append(
                createHistoryTaskPreviewButton(task),
            );
        }
        actions.append(
            createHistoryActionButton('查看', () => openSidebarHistoryTask(task.id)),
        );

        card.append(main, actions);
        return card;
    }

    export function compactPathLabel(value) {
        const text = String(value || '').replace(/\\/g, '/').trim();
        if (!text) return '-';
        const parts = text.split('/').filter(Boolean);
        if (!parts.length) return text;
        const name = parts[parts.length - 1];
        const parent = parts[parts.length - 2] || '';
        if (name === 'training_output' && parent) return `.../${parent}/training_output`;
        return parts.length > 1 ? `.../${name}` : name;
    }

    export function createHistoryActionButton(label, handler, tone = '') {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = ['task-history-action', tone].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            handler();
        });
        return btn;
    }

    export function createHistoryTaskPreviewButton(task) {
        const btn = createHistoryActionButton('预览', () => loadHistoryTask(task.id, { detailTab: 'preview' }));
        btn.title = '只查看这一次训练任务的样张和权重；会在任务详情中打开。';
        return btn;
    }

    export function createHistoryTaskConfigButton(task) {
        const btn = createHistoryActionButton('配置', () => loadHistoryTask(task.id, { detailTab: 'config_files' }));
        btn.title = '打开这条历史任务的配置快照';
        return btn;
    }

    export async function applyHistoryTaskIdsBatchAction(taskIds, action, extra = {}, options = {}) {
        const ids = (taskIds || []).filter(Boolean);
        if (!ids.length) return null;
        const res = await api('/api/training/history/batch', {
            method: 'POST',
            body: JSON.stringify({ action, task_ids: ids, ...extra }),
        });
        if (!res.ok) {
            await showHistoryTaskMessageDialog({
                title: '批量操作失败',
                message: res.error || '批量操作失败',
                tone: 'error',
            });
            return null;
        }
        if (options.clearSelection) {
            ids.forEach((id) => historyState.selectedHistoryTaskIds.delete(id));
        }
        await loadTrainingHistoryList();
        return res;
    }

    export async function applyHistoryBatchAction(action, extra = {}) {
        const taskIds = historyTaskIds(selectedHistoryTasks());
        if (!taskIds.length) return null;
        return applyHistoryTaskIdsBatchAction(taskIds, action, extra, { clearSelection: true });
    }

    export async function archiveSelectedHistoryTasks(archived) {
        const tasks = selectedHistoryTasks();
        if (!tasks.length) return;
        const ok = await showHistoryTaskConfirmDialog({
            title: archived ? '批量归档' : '批量取消归档',
            description: `${tasks.length} 条历史任务`,
            message: archived ? '归档后默认会隐藏这些任务。' : '取消归档后这些任务会重新出现在默认列表中。',
            confirmText: archived ? '归档' : '取消归档',
        });
        if (!ok) return;
        await applyHistoryBatchAction(archived ? 'archive' : 'unarchive');
    }

    export async function groupSelectedHistoryTasks() {
        const tasks = selectedHistoryTasks();
        if (!tasks.length) return;
        const group = await showHistoryCollectionSelectDialog({
            title: '批量设置集合',
            description: `${tasks.length} 条历史任务`,
            value: '',
            confirmText: '保存集合',
        });
        if (group === null) return;
        await applyHistoryTaskIdsToCollection(historyTaskIds(selectedHistoryTasks()), group.trim(), { clearSelection: true });
    }

    export async function deleteSelectedHistoryTasks() {
        const taskIds = historyTaskIds(selectedHistoryTasks());
        await deleteHistoryTasksThorough(taskIds);
    }

    export async function mergeSelectedHistoryTasks() {
        const taskIds = selectedHistoryTasks()
            .filter((task) => task.job === 'training')
            .map((task) => task.id)
            .filter(Boolean);
        if (!taskIds.length) {
            await showHistoryTaskMessageDialog({
                title: '无法合并查看',
                message: '请先选择至少一个训练任务。',
                tone: 'warning',
            });
            return;
        }
        await loadConfigGroupTimeline(
            { methods_subdir: '手动选择', variant: `${taskIds.length} 个已选训练任务`, preset: 'selected' },
            { taskIds, skipSelectionDialog: true },
        );
    }

    export function historyBatchDeleteUnavailable(res) {
        const message = String(res?.error || res?.message || '').trim();
        return /\b405\b|method not allowed/i.test(message);
    }

    export async function deleteHistoryTasksWithLegacyEndpoint(taskIds, options = {}) {
        const ids = uniqueStringList(taskIds || []).filter(Boolean);
        if (!ids.length) return;
        if (!options.confirmed) {
            const ok = await showHistoryTaskConfirmDialog({
                title: '兼容删除历史任务',
                description: `${ids.length} 条历史记录`,
                message: '当前服务未提供批量删除预览，将使用兼容删除接口逐条删除历史记录；不会清理 WebUI 运行目录、权重、样张和缓存。',
                confirmText: '删除历史记录',
                danger: true,
            });
            if (!ok) return;
        }
        const deletedIds = [];
        const failures = [];
        for (const id of ids) {
            try {
                const res = await api(`/api/training/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
                if (!res.ok) {
                    failures.push(`${id}: ${res.error || res.message || '删除失败'}`);
                    continue;
                }
                deletedIds.push(...(res.deleted_task_ids || [id]).filter(Boolean));
            } catch (e) {
                failures.push(`${id}: ${e.message || '删除失败'}`);
            }
        }
        const touchedIds = uniqueStringList(deletedIds);
        touchedIds.forEach((id) => historyState.selectedHistoryTaskIds.delete(id));
        if (touchedIds.includes(historyState.viewingHistoryTaskId)) {
            clearHistoryManagerDetail();
        }
        if (deletedIds.length) {
            await loadTrainingHistoryList();
        }
        if (failures.length) {
            await showHistoryTaskMessageDialog({
                title: deletedIds.length ? '部分历史任务删除失败' : '删除失败',
                message: deletedIds.length
                    ? '部分历史记录已删除，其余项目未能删除。'
                    : '兼容删除接口也未能删除这些历史记录。',
                detailLines: failures,
                tone: deletedIds.length ? 'warning' : 'error',
            });
            return;
        }
        await showHistoryTaskMessageDialog({
            title: '已删除历史记录',
            message: '已通过兼容接口删除历史记录；运行目录、权重、样张和缓存没有被清理。',
            tone: 'ok',
        });
    }

    export async function deleteHistoryTasksThorough(taskIds) {
        const ids = uniqueStringList(taskIds || []).filter(Boolean);
        if (!ids.length) return;
        let preview;
        try {
            preview = await api('/api/training/history/batch', {
                method: 'POST',
                body: JSON.stringify({
                    action: 'delete',
                    task_ids: ids,
                    delete_runtime_dirs: true,
                    dry_run: true,
                }),
            });
        } catch (e) {
            await showHistoryTaskMessageDialog({
                title: '读取删除预览失败',
                message: e.message,
                tone: 'error',
            });
            return;
        }
        if (!preview.ok) {
            if (historyBatchDeleteUnavailable(preview)) {
                await deleteHistoryTasksWithLegacyEndpoint(ids);
                return;
            }
            await showHistoryTaskMessageDialog({
                title: '读取删除预览失败',
                message: preview.error || '读取删除预览失败',
                tone: 'error',
            });
            return;
        }
        if ((preview.blocked || []).length) {
            await showHistoryTaskMessageDialog({
                title: '存在不能删除的任务或运行目录',
                message: '请先处理以下阻止项，再重新执行删除。',
                detailLines: preview.blocked.map((item) => `${item.id || item.path || '-'}: ${item.reason || '-'}`),
                tone: 'error',
            });
            return;
        }
        const confirmed = await showHistoryDeletePreviewDialog(preview);
        if (!confirmed) return;
        try {
            const res = await api('/api/training/history/batch', {
                method: 'POST',
                body: JSON.stringify({
                    action: 'delete',
                    task_ids: ids,
                    delete_runtime_dirs: true,
                    confirmed: true,
                }),
            });
            if (!res.ok) {
                if (historyBatchDeleteUnavailable(res)) {
                    await deleteHistoryTasksWithLegacyEndpoint(ids, { confirmed: true });
                    return;
                }
                await showHistoryTaskMessageDialog({
                    title: '删除失败',
                    message: res.error || '删除失败',
                    tone: 'error',
                });
                return;
            }
            const cleanupErrors = Object.entries({ ...(res.cleanup_errors || {}), ...(res.runtime_cleanup_errors || {}) }).map(([path, error]) => `${path}: ${error}`);
            historyState.selectedHistoryTaskIds.clear();
            if (ids.includes(historyState.viewingHistoryTaskId)) clearHistoryManagerDetail();
            await loadTrainingHistoryList();
            if (cleanupErrors.length) await showHistoryTaskMessageDialog({ title: '历史记录已删除，部分文件未清理', message: '以下运行目录或历史记录残留需要手动检查。', detailLines: cleanupErrors, tone: 'warning' });
        } catch (e) {
            await showHistoryTaskMessageDialog({
                title: '删除失败',
                message: e.message,
                tone: 'error',
            });
        }
    }

    export async function showHistoryDeletePreviewDialog(preview) {
        const wrap = document.createElement('div');
        wrap.className = 'history-delete-preview';
        const summary = document.createElement('div');
        summary.className = 'history-task-dialog-message';
        const strong = document.createElement('strong');
        strong.textContent = `${preview.task_count || 0} 条历史记录 · ${preview.runtime_dir_count || 0} 个运行目录`;
        const p = document.createElement('p');
        p.textContent = '会删除历史记录目录，并删除对应运行目录内的权重、样张、日志、缓存和 runtime 配置。';
        summary.append(strong, p);
        wrap.appendChild(summary);

        const taskList = document.createElement('pre');
        taskList.textContent = [
            '# 历史记录',
            ...(preview.tasks || []).map((item) => `${item.id} · ${item.name || item.job || '-'}`),
            '',
            '# 运行目录',
            ...((preview.runtime_dirs || []).map((item) => `${item.path} · ${item.status || 'ready'}`)),
        ].join('\n');
        wrap.appendChild(taskList);

        const firstConfirmed = await showHistoryTaskDialog({
            title: '彻底删除历史任务',
            description: '第一步确认：请检查删除范围。',
            body: wrap,
            confirmText: '彻底删除',
            danger: true,
            getValue: () => true,
        });
        if (!firstConfirmed) return false;

        const finalConfirmed = await showHistoryTaskConfirmDialog({
            title: '确认要删吗',
            description: `${preview.task_count || 0} 条历史记录 · ${preview.runtime_dir_count || 0} 个运行目录`,
            message: '这是最后一次确认。确认后会立即删除历史记录目录和对应运行目录内的权重、样张、日志、缓存及 runtime 配置。',
            confirmText: '确认要删吗',
            danger: true,
        });
        return Boolean(finalConfirmed);
    }

    export async function renameHistoryTask(task) {
        const fallback = historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || '-'}`;
        const name = await showHistoryTaskInputDialog({
            title: '重命名任务',
            description: '只修改任务列表中的显示名称，不会改动磁盘目录。',
            label: '任务名称',
            value: fallback,
            placeholder: '例如：肋骨女神 5.14 第一次训练',
            confirmText: '保存名称',
        });
        if (name === null) return;
        await updateHistoryTaskMeta(task.id, { name: name.trim() });
    }

    export async function regroupHistoryTask(task) {
        const group = await showHistoryTaskInputDialog({
            title: '设置任务分组',
            description: '相同分组名的任务会在左侧任务列表中归到一起。留空表示未分组。',
            label: '分组名称',
            value: task.group || '',
            placeholder: '例如：肋骨女神 / 测试组 / 正式训练',
            confirmText: '保存分组',
        });
        if (group === null) return;
        await updateHistoryTaskMeta(task.id, { group: group.trim() });
    }

    export async function archiveHistoryTask(task) {
        const archived = historyTaskIsArchived(task);
        const ok = await showHistoryTaskConfirmDialog({
            title: archived ? '取消归档任务' : '归档任务',
            description: historyTaskLabel(task),
            message: archived
                ? '取消归档后，这个任务会重新出现在默认任务列表中。'
                : '归档后默认会隐藏这个任务，可勾选“显示归档”再次查看。',
            confirmText: archived ? '取消归档' : '确认归档',
        });
        if (!ok) return;
        await updateHistoryTaskMeta(task.id, { archived: !archived });
    }

    export async function deleteHistoryTask(task) {
        await deleteHistoryTasksThorough([task.id]);
    }

    export async function updateHistoryTaskMeta(taskId, patch) {
        try {
            const res = await api(`/api/training/history/${encodeURIComponent(taskId)}`, {
                method: 'PATCH',
                body: JSON.stringify(patch),
            });
            if (!res.ok) {
                await showHistoryTaskMessageDialog({
                    title: '更新任务失败',
                    message: res.error || '更新任务失败',
                    tone: 'error',
                });
                return;
            }
            await loadTrainingHistoryList();
            if (historyState.viewingHistoryTaskId === taskId) {
                const payload = await api(`/api/training/history/${encodeURIComponent(taskId)}`);
                if (payload.ok) {
                    historyState.currentHistoryTaskForResume = payload.task || null;
                    renderHistoryManagerDetail(payload, { open: isHistoryDetailDialogOpen() });
                }
            }
        } catch (e) {
            await showHistoryTaskMessageDialog({
                title: '更新任务失败',
                message: e.message,
                tone: 'error',
            });
        }
    }

    export function historyTaskLabel(task) {
        return historyTaskDisplayName(task) || `${task.methods_subdir || '-'} / ${task.variant || task.id}`;
    }

    export function showHistoryTaskInputDialog(options) {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = options.value || '';
        input.placeholder = options.placeholder || '';
        input.className = 'history-task-dialog-input';

        const label = document.createElement('label');
        label.className = 'history-task-dialog-field';
        const span = document.createElement('span');
        span.textContent = options.label || '输入内容';
        label.append(span, input);

        return showHistoryTaskDialog({
            title: options.title,
            description: options.description,
            body: label,
            confirmText: options.confirmText || '确认',
            onOpen: () => {
                input.focus();
                input.select();
            },
            getValue: () => input.value,
        });
    }

configureHistoryTaskActionsBridge({
    createHistoryTaskItem,
    compactPathLabel,
    createHistoryActionButton,
    createHistoryTaskPreviewButton,
    createHistoryTaskConfigButton,
    applyHistoryTaskIdsBatchAction,
    applyHistoryBatchAction,
    archiveSelectedHistoryTasks,
    groupSelectedHistoryTasks,
    deleteSelectedHistoryTasks,
    mergeSelectedHistoryTasks,
    historyBatchDeleteUnavailable,
    deleteHistoryTasksWithLegacyEndpoint,
    deleteHistoryTasksThorough,
    showHistoryDeletePreviewDialog,
    renameHistoryTask,
    regroupHistoryTask,
    archiveHistoryTask,
    deleteHistoryTask,
    updateHistoryTaskMeta,
    historyTaskLabel,
    showHistoryTaskInputDialog,
});
