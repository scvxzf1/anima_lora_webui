/**
 * TOML group list rendering and file/group row UI.
 */
import { readTomlGroupState, writeTomlGroupState } from './group-state.js?v=module-bootstrap-20260831-release-v1';
import {
    createFileGroupDragHandle,
    setupFileGroupHeaderDropTarget,
    setupFileGroupListDropTarget,
    setupFileGroupRowDropTarget,
} from './file-group-drag.js?v=module-bootstrap-20260831-release-v1';
import { setupConfigGroupDropTarget } from './config-group-drop.js?v=module-bootstrap-20260831-release-v1';
import { updateConfigPageSummary } from '../anima-app/helpers/toml-manager-bridge.js?v=module-bootstrap-20260831-release-v1';
import { canDeleteTomlGroup, deleteTomlGroup, deleteTomlGroupButtonTitle } from '../anima-app/helpers/toml-actions-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    createTomlGroup,
    renameTomlGroup,
    setTomlStatus,
    tomlFileDisplayName,
    tomlLockLabel,
    toggleTomlGroupLock,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { hasPendingConfigChanges, updateTomlSelectionUI } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260831-release-v1';
import { shouldShowTomlGroup } from '../anima-app/helpers/toml-io-bridge.js?v=module-bootstrap-20260831-release-v1';
import { selectAndApplyTomlFile } from '../anima-app/helpers/output-run-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    createTomlGroupDragHandle,
    isTomlFileDraggable,
    tomlFileDragOptions,
    tomlGroupDragOptions,
} from './drag-core.js?v=module-bootstrap-20260831-release-v1';
import {
    createTomlGroupActionButton,
    enqueueTomlGroupToQueue,
    exportTomlGroup,
    exportableTomlGroupFiles,
    queueableTomlGroupFiles,
    runTomlGroupAction,
} from './drag-actions.js?v=module-bootstrap-20260831-release-v1';

const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
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
        fragment.appendChild(createTomlFileButton(item, group));
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
