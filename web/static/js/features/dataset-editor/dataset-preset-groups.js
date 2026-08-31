/**
 * Dataset preset group list rendering, drag/drop, and header action state.
 */

import {
    deleteDatasetPresetGroup,
    placeDatasetPresetFile,
    placeDatasetPresetGroup,
    renameDatasetPresetGroup,
    setDatasetPresetStatus,
} from '../anima-app/helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    datasetPresetByFile,
    isUnfiledDatasetGroup,
} from '../anima-app/helpers/dataset-presets.js?v=module-bootstrap-20260831-release-v1';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { loadDatasetPreset } from './load.js?v=module-bootstrap-20260831-release-v1';
import { updateDatasetPresetPageSummary } from './preset-page.js?v=module-bootstrap-20260831-release-v1';
import {
    clearFileGroupDropTarget,
    configGroupDropIndex,
    createFileGroupDragHandle,
    fileGroupContainsRelatedTarget,
    finishFileGroupDrag,
    markFileGroupDropTarget,
    registerFileGroupDropTarget,
    setupFileGroupHeaderDropTarget,
    setupFileGroupListDropTarget,
    setupFileGroupRowDropTarget,
} from '../toml-manager/file-group-drag.js?v=module-bootstrap-20260831-release-v1';
import { setupConfigGroupDropTarget } from '../toml-manager/config-group-drop-target.js?v=module-bootstrap-20260831-release-v1';
import { escapeHtml } from '../config-form/field-input.js?v=module-bootstrap-20260831-release-v1';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

function currentFileGroupDragState() {
    return datasetState.fileGroupDragState || null;
}

export function createDatasetPresetGroupNode(group) {
    const datasetPresetState = currentDatasetPresetState();
    const files = group.files || [];
    const details = document.createElement('details');
    details.className = ['dataset-preset-group', !files.length ? 'empty' : '', group.locked ? 'readonly' : ''].filter(Boolean).join(' ');
    details.dataset.groupId = group.id || '';
    // 分组默认折叠；只有搜索时强制展开，方便逐项过目。切换界面重新渲染后回到默认折叠。
    details.open = Boolean(datasetPresetState.search.trim());

    const summary = document.createElement('summary');
    const groupHandle = createDatasetPresetGroupDragHandle(group, details);
    if (groupHandle) summary.appendChild(groupHandle);
    const title = document.createElement('span');
    title.className = 'dataset-preset-group-title';
    title.textContent = `${group.label || group.id || '数据集分组'} (${(group.files || []).length})`;
    summary.appendChild(title);
    const actions = createDatasetPresetGroupActions(group);
    if (actions) summary.appendChild(actions);
    if (group.locked || group.user_group_locked) {
        const badge = document.createElement('em');
        badge.textContent = group.user_group_locked ? '分组锁定' : '只读';
        summary.appendChild(badge);
    }
    setupFileGroupHeaderDropTarget(summary, group, datasetPresetDragOptions());
    details.appendChild(summary);

    const list = document.createElement('div');
    list.className = 'dataset-preset-group-list';
    setupFileGroupListDropTarget(list, group, datasetPresetDragOptions());
    if (!files.length) {
        const empty = document.createElement('div');
        empty.className = 'dataset-preset-empty dataset-preset-empty-state';
        empty.textContent = datasetPresetState.search.trim() ? '此分组没有匹配项。' : '空分组，可将数据集预设移动到这里。';
        list.appendChild(empty);
    }
    files.forEach((preset) => {
        list.appendChild(createDatasetPresetGroupFileRow(preset, group));
    });
    details.appendChild(list);
    setupConfigGroupDropTarget(details, group, datasetPresetGroupDragOptions());
    return details;
}

function getSortableDatasetPresetGroups() {
    const datasetPresetState = currentDatasetPresetState();
    return (datasetPresetState.groups || [])
        .filter((group) => group.id && !group.system_locked && !group.locked && !group.user_group_locked && !isUnfiledDatasetGroup(group));
}

function createDatasetPresetGroupDragHandle(group, details) {
    const disabled = !isDatasetPresetGroupDraggable(group);
    return createFileGroupDragHandle({
        target: 'group',
        scope: 'dataset',
        groupId: group.id,
        sourceElement: details,
        canDrag: () => isDatasetPresetGroupDraggable(group),
        blockedMessage: () => setDatasetPresetStatus('该数据集分组不能拖动排序', 'error'),
    }, {
        disabled,
        label: `拖动数据集分组 ${group.label || group.id}`,
        title: disabled ? '该数据集分组不能拖动排序' : '拖动调整数据集分组顺序',
    });
}

function isDatasetPresetGroupDraggable(group) {
    const datasetPresetState = currentDatasetPresetState();
    return Boolean(group?.id && !datasetPresetState.search.trim() && !group.system_locked && !group.locked && !group.user_group_locked && !isUnfiledDatasetGroup(group));
}

function isDatasetPresetFileDraggable(preset, group) {
    const datasetPresetState = currentDatasetPresetState();
    return Boolean(preset?.path && group?.id && !datasetPresetState.search.trim() && !preset.readonly);
}

function datasetPresetCanDropToGroup(group, payload) {
    const datasetPresetState = currentDatasetPresetState();
    return Boolean(
        payload?.file &&
        group?.id &&
        !datasetPresetState.search.trim() &&
        (group.kind === 'dataset' || group.id === 'datasets' || group.id === 'unfiled_datasets') &&
        group.movable &&
        !group.locked &&
        !group.user_group_locked
    );
}

function datasetPresetDragOptions() {
    return {
        scope: 'dataset',
        rowSelector: '.dataset-preset-row',
        canDropToGroup: datasetPresetCanDropToGroup,
        onDrop: placeDatasetPresetFile,
    };
}

function datasetPresetGroupDragOptions() {
    return {
        scope: 'dataset',
        getSortableGroups: () => getSortableDatasetPresetGroups(),
        canDropOnGroup: (group) => isDatasetPresetGroupDraggable(group),
        onDrop: placeDatasetPresetGroup,
    };
}

function createDatasetPresetGroupActions(group) {
    const wrap = document.createElement('span');
    wrap.className = 'dataset-preset-group-actions';
    if (group.renamable) {
        wrap.appendChild(createDatasetPresetGroupActionButton('重命名', () => renameDatasetPresetGroup(group), {
            title: '重命名这个数据集分组',
        }));
    }
    if (group.deletable) {
        wrap.appendChild(createDatasetPresetGroupActionButton('删除分组', () => deleteDatasetPresetGroup(group), {
            title: `删除分组“${group.label || group.id}”；不会删除其中的 TOML 文件`,
            danger: true,
        }));
    }
    return wrap.childElementCount ? wrap : null;
}

function createDatasetPresetGroupActionButton(label, handler, options = {}) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = [
        'dataset-preset-group-action-btn',
        options.variant ? `dataset-preset-group-action-btn-${options.variant}` : '',
        options.danger ? 'danger' : '',
    ].filter(Boolean).join(' ');
    btn.textContent = label;
    btn.disabled = Boolean(options.disabled);
    btn.title = options.title || label;
    btn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        if (!btn.disabled) handler();
    });
    return btn;
}

function createDatasetPresetGroupFileRow(preset, group) {
    const datasetPresetState = currentDatasetPresetState();
    const row = document.createElement('div');
    row.className = 'dataset-preset-row';
    row.dataset.file = preset.path;
    row.dataset.groupId = group?.id || '';
    setupFileGroupRowDropTarget(row, group, preset.path, datasetPresetDragOptions());

    const dragHandle = createFileGroupDragHandle({
        target: 'file',
        scope: 'dataset',
        file: preset.path,
        groupId: group?.id || '',
        sourceElement: row,
        canDrag: () => isDatasetPresetFileDraggable(preset, group),
        blockedMessage: () => {
            const message = datasetPresetState.search.trim()
                ? '筛选数据集预设时不能拖动排序，请先清空搜索'
                : '该数据集预设不能拖动排序';
            setDatasetPresetStatus(message, 'error');
        },
    }, {
        disabled: !isDatasetPresetFileDraggable(preset, group),
        label: `拖动数据集预设 ${preset.label || preset.filename || preset.path}`,
        title: isDatasetPresetFileDraggable(preset, group)
            ? '拖动调整数据集预设位置或移动到其他分组'
            : '当前数据集预设不能拖动',
    });
    row.appendChild(dragHandle);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = [
        'dataset-preset-item',
        preset.path === datasetPresetState.selectedFile ? 'active' : '',
        preset.readonly ? 'readonly' : '',
    ].filter(Boolean).join(' ');
    btn.dataset.file = preset.path;
    const summary = preset.summary || {};
    btn.innerHTML = [
        `<strong>${escapeHtml(preset.label || preset.filename || preset.path)}</strong>`,
        `<span>${escapeHtml(preset.path)}</span>`,
        `<small>${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}${preset.readonly ? ' · 只读' : ''}</small>`,
    ].join('');
    btn.addEventListener('click', () => loadDatasetPreset(preset.path));
    row.appendChild(btn);

    return row;
}

export function renderDatasetPresetHeader() {
    const datasetPresetState = currentDatasetPresetState();
    const header = document.getElementById('dataset-preset-header');
    updateDatasetPresetPageSummary();
    if (!header) return;
    const file = datasetPresetState.selectedFile;
    const preset = datasetPresetByFile(file);
    const summary = preset?.summary || {};
    header.innerHTML = '';
    const title = document.createElement('div');
    title.className = 'dataset-preset-title-block';
    const datasetCount = Number(summary.dataset_count || datasetPresetState.datasets.length || 0);
    const repeatTotal = Number(summary.repeat_total || datasetPresetState.datasets.reduce((sum, row) => sum + Number(row.num_repeats || 1), 0) || 0);
    const status = datasetPresetState.dirty ? '未保存' : (datasetPresetState.readonly ? '只读' : '已同步');
    title.innerHTML = [
        '<span class="dataset-preset-breadcrumb">CONFIGS / DATASETS</span>',
        `<strong>${escapeHtml(preset?.label || preset?.filename || file || '新数据集预设')}</strong>`,
        `<span class="dataset-preset-file-path">${escapeHtml(file || '尚未保存')}</span>`,
    ].join('');
    const meta = document.createElement('div');
    meta.className = 'dataset-preset-meta';
    [
        ['状态', status, datasetPresetState.dirty ? 'warn' : (datasetPresetState.readonly ? 'lock' : 'ok')],
        ['子集', datasetCount],
        ['重复', repeatTotal],
        ['分辨率', summary.resolution ? `${summary.resolution}px` : '-'],
    ].forEach(([label, value, tone]) => {
        const stat = document.createElement('span');
        stat.className = ['dataset-preset-stat', tone ? `dataset-preset-stat-${tone}` : ''].filter(Boolean).join(' ');
        stat.innerHTML = `<small>${escapeHtml(String(label))}</small><strong>${escapeHtml(String(value))}</strong>`;
        meta.appendChild(stat);
    });
    header.append(title, meta);
    if (datasetPresetState.status) {
        const statusEl = document.createElement('span');
        statusEl.className = 'dataset-preset-status dataset-preset-inline-status';
        statusEl.textContent = datasetPresetState.status;
        header.appendChild(statusEl);
    }
    updateDatasetPresetActionState();
}

function updateDatasetPresetActionState() {
    const datasetPresetState = currentDatasetPresetState();
    const saveBtn = document.getElementById('btn-save-dataset-preset');
    if (saveBtn) {
        saveBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile || !datasetPresetState.dirty;
        saveBtn.title = datasetPresetState.readonly
            ? '系统数据集预设只读，请复制后编辑'
            : (datasetPresetState.dirty ? '保存当前数据集预设' : '当前数据集预设没有未保存修改');
    }
    const deleteBtn = document.getElementById('btn-delete-dataset-preset');
    if (deleteBtn) {
        deleteBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile;
        deleteBtn.title = datasetPresetState.readonly ? '系统数据集预设不能删除' : '只删除 TOML 预设，不删除图片或缓存目录';
    }
    const renameBtn = document.getElementById('btn-rename-dataset-preset');
    if (renameBtn) {
        renameBtn.disabled = datasetPresetState.readonly || !datasetPresetState.selectedFile;
    }
    const copyBtn = document.getElementById('btn-copy-dataset-preset');
    if (copyBtn) copyBtn.disabled = !datasetPresetState.selectedFile;
    const exportBtn = document.getElementById('btn-export-dataset-preset');
    if (exportBtn) exportBtn.disabled = !datasetPresetState.selectedFile;
}
