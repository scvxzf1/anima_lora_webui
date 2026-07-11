/**
 * TOML group drag/drop, render, export, and queue helpers.
 * Moved out of anima-app mechanical chunks.
 */
import { readTomlGroupState, writeTomlGroupState } from './group-state.js?v=module-bootstrap-20260711-ir1';
import {
    createFileGroupDragHandle,
    setupFileGroupHeaderDropTarget,
    setupFileGroupListDropTarget,
    setupFileGroupRowDropTarget,
} from './file-group-drag.js?v=module-bootstrap-20260711-ir1';
import { setupConfigGroupDropTarget } from './config-group-drop.js?v=module-bootstrap-20260711-ir1';
import { showHistoryTaskDialog } from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadTomlFileList, updateConfigPageSummary } from '../anima-app/helpers/toml-manager-bridge.js?v=module-bootstrap-20260711-ir1';
import { canDeleteTomlGroup, deleteTomlGroup, deleteTomlGroupButtonTitle } from '../anima-app/helpers/toml-actions-bridge.js?v=module-bootstrap-20260711-ir1';
import { enqueueTrainingQueueBatchRequest, isCliOnlySpdSource, showPreflightDialog } from '../anima-app/helpers/training-launch-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    createTomlGroup,
    renameTomlGroup,
    setTomlStatus,
    tomlFileDisplayName,
    tomlLockLabel,
    toggleTomlGroupLock,
    updateTomlActionState,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { api, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';
import { hasPendingConfigChanges, updateTomlSelectionUI } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir1';
import { createTomlZipBlob, downloadBlob, getSortableTomlGroups, isTomlGroupDraggable, isTrainingTomlGroup, shouldShowTomlGroup } from '../anima-app/helpers/toml-io-bridge.js?v=module-bootstrap-20260711-ir1';
import { renderPreflightPending, showPreflightRequestError } from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260711-ir1';
import { appendLog } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir1';
import { configureTomlDragBridge } from '../anima-app/helpers/toml-drag-bridge.js?v=module-bootstrap-20260711-ir1';
import { showTrainingView, updateTrainingQueueFromPayload } from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260711-ir1';
import { selectAndApplyTomlFile } from '../anima-app/helpers/output-run-bridge.js?v=module-bootstrap-20260711-ir1';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir1';

const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

    export function canDropTomlFileToGroup(group) {
        return Boolean(
            group?.id &&
            isTrainingTomlGroup(group) &&
            group.movable &&
            !group.locked &&
            !group.user_group_locked
        );
    }

    export function isTomlFileDraggable(item) {
        return Boolean(item?.path && !item.locked && !hasPendingConfigChanges(tomlState.currentTomlFile));
    }

    export function createTomlGroupDragHandle(group, details) {
        const disabled = !isTomlGroupDraggable(group);
        return createFileGroupDragHandle({
            target: 'group',
            scope: 'training',
            groupId: group.id,
            sourceElement: details,
            canDrag: () => isTomlGroupDraggable(group),
            blockedMessage: () => setTomlStatus('error', '该配置分组不能拖动排序'),
        }, {
            disabled,
            label: `拖动配置分组 ${group.label || group.id}`,
            title: disabled ? '该配置分组不能拖动排序' : '拖动调整配置分组顺序',
        });
    }

    export async function placeTomlGroup(payload, index) {
        const groupId = payload?.groupId;
        if (!groupId) return;
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'group', group: groupId, scope: 'training', index }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '调整分组位置失败');
                return;
            }
            await loadTomlFileList(tomlState.currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组位置已更新');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export async function placeTomlFile(payload, groupId, index) {
        const file = payload?.file;
        if (!file || !groupId) return;
        if (hasPendingConfigChanges(tomlState.currentTomlFile)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再拖动排序');
            updateTomlActionState(tomlState.currentTomlFile);
            return;
        }
        try {
            const res = await api('/api/config/file-groups/place', {
                method: 'POST',
                body: JSON.stringify({ target: 'file', file, group: groupId, index }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '配置位置调整失败');
                return;
            }
            await loadTomlFileList(tomlState.currentTomlFile || file);
            setTomlStatus('ok', res.message || '配置位置已更新');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export function tomlFileDragOptions() {
        return {
            scope: 'training',
            rowSelector: '.toml-file-row-wrap',
            canDropToGroup: canDropTomlFileToGroup,
            onDrop: placeTomlFile,
        };
    }

    export function tomlGroupDragOptions() {
        return {
            scope: 'training',
            getSortableGroups: () => getSortableTomlGroups(),
            canDropOnGroup: (group) => isTomlGroupDraggable(group),
            onDrop: placeTomlGroup,
        };
    }

    export function populateTomlFileSelect(groups) {
        const sel = document.getElementById('toml-file-select');
        const prev = sel.value;
        sel.innerHTML = '';
        for (const group of groups) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = group.label || group.id || '配置文件';
            for (const item of group.files || []) {
                const opt = document.createElement('option');
                opt.value = item.path;
                opt.textContent = [tomlLockLabel(item), tomlFileDisplayName(item)].filter(Boolean).join(' / ');
                opt.dataset.locked = item.locked ? '1' : '0';
                optgroup.appendChild(opt);
            }
            sel.appendChild(optgroup);
        }
        if (tomlState.tomlFiles.includes(prev)) {
            sel.value = prev;
        }
        renderTomlFileGroups(groups);
        updateConfigPageSummary('project');
    }

    export function renderTomlFileGroups(groups) {
        const container = document.getElementById('toml-file-groups');
        if (!container) return;
        container.innerHTML = '';
        const stored = readTomlGroupState();
        const fragment = document.createDocumentFragment();

        const toolbar = document.createElement('div');
        toolbar.className = 'toml-group-toolbar';
        const createBtn = document.createElement('button');
        createBtn.type = 'button';
        createBtn.className = 'toml-group-action-btn';
        createBtn.textContent = '新建分组';
        createBtn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            runTomlGroupAction(createTomlGroup, createBtn);
        });
        toolbar.appendChild(createBtn);
        fragment.appendChild(toolbar);

        const visibleGroups = (groups || []).filter(shouldShowTomlGroup);
        if (visibleGroups.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'toml-file-group-empty';
            empty.textContent = '系统分组已隐藏。可点击“新建分组”创建自己的配置分组。';
            fragment.appendChild(empty);
        }

        for (const group of visibleGroups) {
            const details = document.createElement('details');
            details.className = 'toml-file-group';
            if (group.locked) details.classList.add('readonly');
            details.dataset.groupId = group.id;
            details.open = stored[group.id] ?? Boolean(group.open);

            const summary = document.createElement('summary');
            const groupHandle = createTomlGroupDragHandle(group, details);
            if (groupHandle) summary.appendChild(groupHandle);
            const title = document.createElement('span');
            title.className = 'toml-group-title';
            title.textContent = `${group.label || group.id} (${(group.files || []).length})`;
            summary.appendChild(title);
            const actions = createTomlGroupActions(group);
            if (actions) summary.appendChild(actions);
            if (group.lockable) {
                const groupLockBtn = document.createElement('button');
                groupLockBtn.type = 'button';
                groupLockBtn.className = 'toml-group-lock-btn';
                groupLockBtn.textContent = group.user_group_locked ? '解除分组锁定' : '锁定分组';
                groupLockBtn.title = group.user_group_locked
                    ? '解除该分组的用户锁定'
                    : '锁定该分组内所有文件，防止误保存';
                groupLockBtn.addEventListener('click', (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();
                    runTomlGroupAction(() => toggleTomlGroupLock(group), groupLockBtn);
                });
                summary.appendChild(groupLockBtn);
            }
            if (group.locked) {
                const badge = document.createElement('em');
                badge.textContent = group.user_group_locked ? '分组锁定' : '锁定';
                summary.appendChild(badge);
            }
            setupFileGroupHeaderDropTarget(summary, group, tomlFileDragOptions());
            details.appendChild(summary);

            const list = document.createElement('div');
            list.className = 'toml-file-list';
            setupFileGroupListDropTarget(list, group, tomlFileDragOptions());
            const files = group.files || [];
            const renderGroupFiles = () => renderTomlFileGroupList(list, group, files);
            if (details.open) renderGroupFiles();
            details.addEventListener('toggle', () => {
                const next = readTomlGroupState();
                next[group.id] = details.open;
                writeTomlGroupState(next);
                if (details.open) {
                    renderGroupFiles();
                    updateTomlSelectionUI(tomlState.currentTomlFile);
                }
            });
            details.appendChild(list);
            setupConfigGroupDropTarget(details, group, tomlGroupDragOptions());
            fragment.appendChild(details);
        }
        container.appendChild(fragment);
        updateTomlSelectionUI(tomlState.currentTomlFile);
    }

    export function renderTomlFileGroupList(list, group, files = group?.files || []) {
        if (!list || list.dataset.rendered === '1') return;
        list.dataset.rendered = '1';
        const fragment = document.createDocumentFragment();
        if (!files.length) {
            const empty = document.createElement('div');
            empty.className = 'toml-file-group-empty';
            empty.textContent = group?.user_managed ? '空分组，可使用“移动”放入当前配置。' : '暂无配置文件。';
            fragment.appendChild(empty);
        }
        files.forEach((item, index) => {
            fragment.appendChild(createTomlFileButton(item, group, index, files.length));
        });
        list.appendChild(fragment);
    }

    export function createTomlGroupActions(group) {
        const wrap = document.createElement('span');
        wrap.className = 'toml-group-actions';

        const queueableFiles = queueableTomlGroupFiles(group);
        wrap.appendChild(createTomlGroupActionButton('加入队列', () => enqueueTomlGroupToQueue(group), {
            title: queueableFiles.length
                ? `将该分组内 ${queueableFiles.length} 个可训练配置加入训练队列`
                : '该分组没有可加入队列的训练配置',
            disabled: !queueableFiles.length,
            variant: 'queue',
        }));
        if (group.renamable) {
            wrap.appendChild(createTomlGroupActionButton('重命名', () => renameTomlGroup(group), {
                title: '重命名这个配置分组',
            }));
        }
        const exportableFiles = exportableTomlGroupFiles(group);
        wrap.appendChild(createTomlGroupActionButton('导出分组', () => exportTomlGroup(group), {
            title: exportableFiles.length
                ? `将该分组内 ${exportableFiles.length} 个配置导出为一个 zip，内部保留独立 TOML 文件`
                : '该分组没有可导出的 TOML 文件',
            disabled: !exportableFiles.length,
            variant: 'export',
        }));
        wrap.appendChild(createTomlGroupActionButton('删除分组', () => deleteTomlGroup(group), {
            title: deleteTomlGroupButtonTitle(group),
            danger: true,
            disabled: !canDeleteTomlGroup(group),
        }));
        return wrap;
    }

    export function exportableTomlGroupFiles(group) {
        return (group?.files || [])
            .filter((item) => item?.path && String(item.path).toLowerCase().endsWith('.toml'));
    }

    export async function exportTomlGroup(group) {
        const files = exportableTomlGroupFiles(group);
        if (!files.length) {
            setTomlStatus('error', '该分组没有可导出的 TOML 文件');
            return;
        }
        if (hasPendingConfigChanges(tomlState.currentTomlFile)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再导出分组');
            updateTomlActionState(tomlState.currentTomlFile);
            return;
        }

        const filename = `${exportTomlGroupFilename(group)}.zip`;
        setTomlStatus('pending', `正在读取分组“${group.label || group.id}”中的 ${files.length} 个配置...`, { persist: true });

        try {
            const entries = await Promise.all(files.map(async (item) => {
                const path = String(item.path || '');
                const data = await api(`/api/config/raw?file=${encodeURIComponent(path)}`);
                if (data?.ok === false) {
                    throw new Error(`${path}: ${data.error || '读取失败'}`);
                }
                return {
                    name: item.filename || path,
                    content: data.content || '',
                };
            }));
            const blob = createTomlZipBlob(entries);
            downloadBlob(blob, filename);
            setTomlStatus('ok', `已导出分组“${group.label || group.id}”：1 个 zip，内含 ${files.length} 个独立 TOML 文件`, { persist: true });
        } catch (e) {
            setTomlStatus('error', `导出分组失败: ${e.message || e}`, { persist: true });
        }
    }

    export function exportTomlGroupFilename(group) {
        const raw = String(group?.label || group?.id || 'toml-group').trim();
        const safe = raw.replace(/[\\/:*?"<>|\r\n\t]+/g, '_').replace(/\s+/g, '_').replace(/^[._]+|[._]+$/g, '');
        return safe || 'toml-group';
    }

    export function queueableTomlGroupFiles(group) {
        return (group?.files || [])
            .filter((item) => item?.path && item.trainable)
            .filter((item) => !String(item.path || '').replace(/\\/g, '/').startsWith('configs/datasets/'));
    }

    export function tomlItemQueueVariant(item) {
        if (item?.method) return item.method;
        const filename = String(item?.filename || item?.path || '').split('/').pop() || '';
        return filename.toLowerCase().endsWith('.toml') ? filename.slice(0, -5) : filename;
    }

    export function tomlItemQueueEntry(item, preset = '') {
        const path = String(item?.path || '').trim();
        const methodsSubdir = String(item?.methods_subdir || '').trim() || 'imported';
        const label = tomlFileDisplayName(item);
        return {
            variant: tomlItemQueueVariant(item),
            preset: preset || val('preset-select') || 'default',
            methods_subdir: methodsSubdir,
            config_file: path,
            filename: item?.filename || (path ? path.split('/').pop() : ''),
            label: label === '未命名配置文件' ? '' : label,
            confirm_preprocess: true,
        };
    }

    export function tomlGroupQueueFailureLabel(item, failure = {}, index = -1) {
        const path = String(item?.path || item?.config_file || failure.config_file || '').trim();
        if (path) return path;
        const failureLabel = String(failure.label || failure.filename || item?.label || item?.filename || '').trim();
        if (failureLabel) return failureLabel;
        const label = tomlFileDisplayName(item);
        if (label && label !== '未命名配置文件') return label;
        const fallbackIndex = Number.isFinite(index) && index >= 0 ? index + 1 : Number(failure.index || 0) + 1;
        return fallbackIndex > 0 ? `第 ${fallbackIndex} 个配置` : '批量请求';
    }

    export async function showTomlGroupQueueConfirmDialog(group, files) {
        const wrap = document.createElement('div');
        wrap.className = 'history-task-dialog-message toml-group-queue-dialog';

        const strong = document.createElement('strong');
        strong.textContent = `${group.label || group.id || '配置分组'} · ${files.length} 个配置`;
        const message = document.createElement('p');
        message.textContent = `确认后会按当前 GPU 选择和当前预设 ${val('preset-select') || 'default'}，把该分组内可训练配置逐个冻结并加入队列；队列会保持暂停，等待你手动继续。`;

        const list = document.createElement('div');
        list.className = 'toml-group-queue-list';
        files.slice(0, 12).forEach((item) => {
            const row = document.createElement('code');
            row.textContent = item.path || tomlFileDisplayName(item);
            list.appendChild(row);
        });
        if (files.length > 12) {
            const more = document.createElement('span');
            more.textContent = `还有 ${files.length - 12} 个配置...`;
            list.appendChild(more);
        }

        wrap.append(strong, message, list);
        return showHistoryTaskDialog({
            title: '批量加入训练队列',
            description: '这会创建独立运行配置，不会修改原 TOML 文件。',
            body: wrap,
            confirmText: '确认加入队列',
            cancelText: '取消',
            getValue: () => true,
        });
    }

    export async function enqueueTomlGroupToQueue(group) {
        const files = queueableTomlGroupFiles(group);
        if (!files.length) {
            setTomlStatus('error', '该分组没有可加入队列的训练配置');
            return;
        }
        if (hasPendingConfigChanges(tomlState.currentTomlFile)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再批量加入队列');
            updateTomlActionState(tomlState.currentTomlFile);
            return;
        }
        const confirmed = await showTomlGroupQueueConfirmDialog(group, files);
        if (!confirmed) return;

        const preset = val('preset-select') || 'default';
        const entries = files.map((item) => tomlItemQueueEntry(item, preset));
        const invalidIndex = entries.findIndex((entry) => !entry.variant || isCliOnlySpdSource(entry.variant, entry.methods_subdir));
        if (invalidIndex >= 0) {
            const item = files[invalidIndex];
            const entry = entries[invalidIndex];
            const message = entry.variant ? 'SPD CLI 实验配置不能通过 Web 队列启动' : '配置缺少可训练变体名称';
            setTomlStatus('error', `批量加入队列已停止：${item.path || tomlFileDisplayName(item)}，${message}`, { persist: true });
            showPreflightRequestError(message);
            return;
        }

        renderPreflightPending({
            title: '批量加入训练队列',
            message: `正在冻结 ${files.length} 个配置并加入队列...`,
            detail: '后端会逐个预检测并创建独立运行配置；如果某个配置失败，会返回停止位置和失败原因。',
        });
        let res;
        try {
            res = await enqueueTrainingQueueBatchRequest({
                items: entries,
                preset,
                startPaused: true,
            });
        } catch (e) {
            const message = `批量加入队列失败: ${e.message || e}`;
            setTomlStatus('error', message, { persist: true });
            showPreflightRequestError(message);
            return;
        }
        const queued = Number(res.queued_count || 0);
        updateTrainingQueueFromPayload(res);

        if (queued > 0) {
            document.querySelector('[data-tab="training"]')?.click();
            showTrainingView('queue');
            appendLog(`[状态] 已将分组“${group.label || group.id}”中的 ${queued} 个配置加入训练队列`);
        }
        if (!res.ok) {
            const failure = (res.failures || [])[0] || {};
            const failedIndex = Number.isInteger(res.failed_index) ? res.failed_index : Number(failure.index ?? -1);
            const item = files[failedIndex] || res.failed_item || {};
            const fileLabel = tomlGroupQueueFailureLabel(item, failure, failedIndex);
            const error = res.error || failure.error || '加入队列失败';
            const message = `批量加入队列已停止：${fileLabel}，${error}`;
            setTomlStatus('error', queued ? `已加入 ${queued} 个配置；${message}` : message, { persist: true });
            if (res.preflight) {
                showPreflightDialog(res.preflight, false, { willAutoPreprocess: true });
            } else {
                showPreflightRequestError(message);
            }
            return;
        }

        const dialog = document.getElementById('preflight-dialog');
        if (dialog?.open) dialog.close('queued-group');
        setTomlStatus('ok', `已将 ${queued} 个配置加入训练队列`, { persist: true });
    }

    export function createTomlGroupActionButton(label, handler, options = {}) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'toml-group-action-btn',
            options.variant ? `toml-group-action-btn-${options.variant}` : '',
            options.danger ? 'danger' : '',
        ].filter(Boolean).join(' ');
        btn.textContent = label;
        btn.disabled = Boolean(options.disabled);
        btn.title = options.title || label;
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            if (!btn.disabled) runTomlGroupAction(handler, btn);
        });
        return btn;
    }

    export function runTomlGroupAction(handler, button = null) {
        if (tomlState.tomlGroupActionBusy) return;
        tomlState.tomlGroupActionBusy = true;
        if (button) button.disabled = true;
        Promise.resolve()
            .then(handler)
            .catch((e) => {
                setTomlStatus('error', '分组操作失败: ' + e.message);
            })
            .finally(() => {
                tomlState.tomlGroupActionBusy = false;
                if (button?.isConnected) button.disabled = false;
            });
    }

    export function createTomlFileButton(item, group = null) {
        const row = document.createElement('div');
        row.className = 'toml-file-row-wrap';
        row.dataset.file = item.path;
        row.dataset.groupId = group?.id || item.group || '';
        setupFileGroupRowDropTarget(row, group, item.path, tomlFileDragOptions());

        const dragHandle = createFileGroupDragHandle({
            target: 'file',
            scope: 'training',
            file: item.path,
            groupId: group?.id || item.group || '',
            sourceElement: row,
            canDrag: () => isTomlFileDraggable(item),
            blockedMessage: () => {
                const message = hasPendingConfigChanges(tomlState.currentTomlFile)
                    ? '当前配置尚未保存，请先保存或放弃修改后再拖动排序'
                    : '该配置文件不能拖动排序';
                setTomlStatus('error', message);
            },
        }, {
            disabled: !isTomlFileDraggable(item),
            label: `拖动配置文件 ${tomlFileDisplayName(item)}`,
            title: isTomlFileDraggable(item)
                ? '拖动调整配置文件位置或移动到其他分组'
                : '当前配置文件不能拖动',
        });
        row.appendChild(dragHandle);

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'toml-file-item';
        if (item.locked) btn.classList.add('readonly');
        btn.dataset.file = item.path;
        btn.title = tomlFileDisplayName(item);
        btn.addEventListener('click', () => selectAndApplyTomlFile(item.path));

        const name = document.createElement('span');
        name.className = 'toml-file-name';
        name.textContent = item.label || item.path;
        btn.appendChild(name);

        const meta = document.createElement('span');
        meta.className = 'toml-file-meta';
        const tags = [];
        if (item.filename && item.filename !== item.label) tags.push(item.filename);
        if (currentTrainingSourceState().file === item.path) tags.push('当前训练');
        const lockLabel = tomlLockLabel(item);
        if (lockLabel) tags.push(lockLabel);
        tags.push(item.trainable ? '可训练' : '非训练');
        tags.push(item.path);
        meta.textContent = tags.join(' / ');
        btn.appendChild(meta);
        row.appendChild(btn);
        return row;
    }

configureTomlDragBridge({
    canDropTomlFileToGroup,
    isTomlFileDraggable,
    createTomlGroupDragHandle,
    placeTomlGroup,
    placeTomlFile,
    tomlFileDragOptions,
    tomlGroupDragOptions,
    populateTomlFileSelect,
    renderTomlFileGroups,
    renderTomlFileGroupList,
    createTomlGroupActions,
    exportableTomlGroupFiles,
    exportTomlGroup,
    exportTomlGroupFilename,
    queueableTomlGroupFiles,
    tomlItemQueueVariant,
    tomlItemQueueEntry,
    tomlGroupQueueFailureLabel,
    showTomlGroupQueueConfirmDialog,
    enqueueTomlGroupToQueue,
    createTomlGroupActionButton,
    runTomlGroupAction,
    createTomlFileButton,
});
