/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { captureDatasetExperimentalOpenStates } from './10a-dataset-inline-help.js?v=module-bootstrap-20260711-ir1';
import { help } from '../../../config/catalog.js?v=module-bootstrap-20260711-ir1';
import { DATASET_PRESET_GROUP_STATE_KEY } from '../helpers/app-constants.js?v=module-bootstrap-20260711-ir1';
import { datasetConfigLabel, datasetConfigValue } from '../helpers/dataset-config-fields.js?v=module-bootstrap-20260711-ir1';
import { createHelpContent } from '../helpers/config-field-ui-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    deleteDatasetPresetGroup,
    placeDatasetPresetFile,
    placeDatasetPresetGroup,
    renameDatasetPresetGroup,
    setDatasetPresetStatus,
} from '../helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    datasetPresetByFile,
    isUnfiledDatasetGroup,
} from '../helpers/dataset-presets.js?v=module-bootstrap-20260711-ir1';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { configureDatasetRenderBridge } from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../helpers/dataset-values.js?v=module-bootstrap-20260711-ir1';
import { loadDatasetPreset } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260711-ir1';
import {
    registerFileGroupDropTarget,
    updateDatasetPresetPageSummary,
} from './07-render-config-dataset-picker-dialog.js?v=module-bootstrap-20260711-ir1';
import {
    clearFileGroupDropTarget,
    configGroupDropIndex,
    createFileGroupDragHandle,
    fileGroupContainsRelatedTarget,
    finishFileGroupDrag,
    markFileGroupDropTarget,
    setupFileGroupHeaderDropTarget,
    setupFileGroupListDropTarget,
    setupFileGroupRowDropTarget,
} from './08-origin-closest.js?v=module-bootstrap-20260711-ir1';
import {
    createDatasetConfigInput,
    createDatasetEditorItem,
} from './10-create-dataset-config-input.js?v=module-bootstrap-20260711-ir1';
import { addDatasetEditorRow, datasetValidTargetIndices, escapeHtml } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260711-ir1';
import { createDatasetEditorToolbarActions } from '../../dataset-editor/toolbar.js?v=module-bootstrap-20260711-ir1';

const configState = getConfigState();
const datasetState = getDatasetState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

function currentFileGroupDragState() {
    return datasetState.fileGroupDragState || null;
}


    export function setupConfigGroupDropTarget(node, group, options) {
        registerFileGroupDropTarget(node, ({ payload, y }) => {
            if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return null;
            if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return null;
            const rect = node.getBoundingClientRect();
            const placeAfter = y > rect.top + rect.height / 2;
            const position = placeAfter ? 'after' : 'before';
            return {
                position,
                drop: async () => {
                    const index = configGroupDropIndex(options.getSortableGroups(), group.id, placeAfter, payload.groupId);
                    await options.onDrop(payload, index);
                },
            };
        });
        const updateDropTarget = (event) => {
            const payload = currentFileGroupDragState();
            if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return;
            if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            const rect = node.getBoundingClientRect();
            const placeAfter = event.clientY > rect.top + rect.height / 2;
            node.dataset.dropPosition = placeAfter ? 'after' : 'before';
            markFileGroupDropTarget(node, placeAfter ? 'after' : 'before');
        };
        node.addEventListener('dragenter', updateDropTarget);
        node.addEventListener('dragover', updateDropTarget);
        node.addEventListener('dragleave', (event) => {
            if (fileGroupContainsRelatedTarget(node, event)) return;
            clearFileGroupDropTarget(node);
        });
        node.addEventListener('drop', async (event) => {
            const payload = currentFileGroupDragState();
            if (!payload || payload.target !== 'group' || payload.scope !== options.scope) return;
            if (payload.groupId === group?.id || !options.canDropOnGroup(group)) return;
            event.preventDefault();
            event.stopPropagation();
            const placeAfter = node.dataset.dropPosition === 'after';
            const index = configGroupDropIndex(options.getSortableGroups(), group.id, placeAfter, payload.groupId);
            await options.onDrop(payload, index);
            finishFileGroupDrag();
        });
    }

    function createDatasetPresetGroupNode(group, stored) {
        const datasetPresetState = currentDatasetPresetState();
        const files = group.files || [];
        const details = document.createElement('details');
        details.className = ['dataset-preset-group', !files.length ? 'empty' : '', group.locked ? 'readonly' : ''].filter(Boolean).join(' ');
        details.dataset.groupId = group.id || '';
        const containsSelected = files.some((preset) => preset.path === datasetPresetState.selectedFile);
        const shouldForceOpen = containsSelected || Boolean(datasetPresetState.search.trim());
        const defaultOpen = isUnfiledDatasetGroup(group);
        details.open = shouldForceOpen || (stored[group.id] ?? defaultOpen);
        details.addEventListener('toggle', () => {
            const next = readDatasetPresetGroupState();
            next[group.id] = details.open;
            writeDatasetPresetGroupState(next);
        });

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

    function readDatasetPresetGroupState() {
        try {
            return JSON.parse(localStorage.getItem(DATASET_PRESET_GROUP_STATE_KEY) || '{}') || {};
        } catch (_) {
            return {};
        }
    }

    function writeDatasetPresetGroupState(state) {
        try {
            localStorage.setItem(DATASET_PRESET_GROUP_STATE_KEY, JSON.stringify(state || {}));
        } catch (_) {
            // 忽略本地存储不可用；分组折叠状态不是关键数据。
        }
    }

    function renderDatasetPresetHeader() {
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

    function renderDatasetEditor(existingPanel = null) {
        const panel = existingPanel || document.getElementById('dataset-editor');
        if (!panel) return;
        captureDatasetExperimentalOpenStates(panel);
        panel.innerHTML = '';
        const state = datasetEditorStateForActivePanel();

        const header = document.createElement('div');
        header.className = 'dataset-editor-header';
        const title = document.createElement('div');
        title.innerHTML = '<strong>多数据集路径</strong><span>每一行是一组数据：填写原始图路径、重复次数和分桶参数；缩放图与 LoRA 缓存会在训练运行目录中自动生成。</span>';
        const actions = document.createElement('div');
        actions.className = 'dataset-editor-actions';
        const toolbar = createDatasetEditorToolbarActions();
        const addBtn = document.createElement('button');
        addBtn.type = 'button';
        addBtn.className = 'btn btn-small';
        addBtn.textContent = '添加数据集';
        addBtn.title = '新增一组数据集路径。适合把多个角色、画风或批次一起训练，并给每组设置独立重复次数。';
        addBtn.addEventListener('click', addDatasetEditorRow);
        actions.append(toolbar, addBtn);
        header.append(title, actions);
        panel.appendChild(header);

        if (state.loading) {
            const loading = document.createElement('p');
            loading.className = 'dataset-editor-message';
            loading.textContent = '正在读取数据集配置...';
            panel.appendChild(loading);
            return;
        }
        if (state.error) {
            const error = document.createElement('p');
            error.className = 'dataset-editor-message error';
            error.textContent = state.error;
            panel.appendChild(error);
        }

        const rows = state.datasets.length
            ? state.datasets
            : normalizeDatasetEditorRows([{
                source_dir: currentConfigState().source_image_dir || '',
                image_dir: currentConfigState().resized_image_dir || '',
                cache_dir: currentConfigState().lora_cache_dir || '',
                num_repeats: 1,
                settings: normalizeDatasetDefaults(state.defaults || {}),
            }]);
        if (!state.datasets.length) {
            setActiveDatasetRows(rows);
        }

        panel.appendChild(createDatasetDefaultsEditor());

        const list = document.createElement('div');
        list.className = 'dataset-editor-list';
        rows.forEach((row, index) => {
            list.appendChild(createDatasetEditorItem(row, index));
        });
        panel.appendChild(list);

        const footer = document.createElement('div');
        footer.className = 'dataset-editor-footer';
        const configPath = document.createElement('code');
        configPath.textContent = activeDatasetFileLabel();
        const dirty = document.createElement('span');
        dirty.className = activeDatasetDirty() ? 'dataset-editor-dirty active' : 'dataset-editor-dirty';
        dirty.textContent = activeDatasetDirty() ? '有未保存的数据集修改' : '数据集路径已同步';
        footer.append(configPath, dirty);
        panel.appendChild(footer);
        if (isDatasetTabActive()) {
            renderDatasetPresetHeader();
        }
    }

    function refreshDatasetEditorItem(index) {
        const panel = document.getElementById('dataset-editor');
        if (!panel) return false;
        const state = datasetEditorStateForActivePanel();
        if (state.loading || state.error) return false;
        const rows = normalizeDatasetEditorRows(state.datasets);
        const row = rows[index];
        if (!row) return false;
        const list = panel.querySelector('.dataset-editor-list');
        const currentItem = list?.querySelector(`.dataset-editor-item[data-index="${index}"]`);
        if (!list || !currentItem) return false;
        const nextItem = createDatasetEditorItem(row, index);
        list.replaceChild(nextItem, currentItem);
        return true;
    }

    function refreshDatasetEditorItems(indices) {
        const targets = datasetValidTargetIndices(indices, normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets).length);
        if (!targets.length) return false;
        let updated = false;
        for (const index of targets) {
            updated = refreshDatasetEditorItem(index) || updated;
        }
        return updated;
    }

    function datasetEditorStateForActivePanel() {
        return isDatasetTabActive() ? currentDatasetPresetState() : currentDatasetEditorState();
    }

    function isDatasetTabActive() {
        return Boolean(document.getElementById('tab-datasets')?.classList.contains('active'));
    }

    function setActiveDatasetRows(rows) {
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.datasets = rows;
        } else {
            datasetState.datasetEditorState.datasets = rows;
        }
    }

    function activeDatasetFileLabel() {
        if (isDatasetTabActive()) {
            return currentDatasetPresetState().selectedFile || '保存后生成 configs/datasets/<名称>.toml';
        }
        return currentDatasetEditorState().dataset_config || currentConfigState().dataset_config || '保存后自动生成 configs/datasets/<当前配置>.toml';
    }

    function activeDatasetDirty() {
        return isDatasetTabActive() ? currentDatasetPresetState().dirty : currentDatasetEditorState().dirty;
    }

    function createDatasetDefaultsEditor() {
        const state = datasetEditorStateForActivePanel();
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.defaults = defaults;
        } else {
            datasetState.datasetEditorState.defaults = defaults;
        }
        const wrap = document.createElement('div');
        wrap.className = 'dataset-defaults-list';

        const heading = document.createElement('div');
        heading.className = 'dataset-defaults-heading';
        heading.innerHTML = '<strong>通用标注设置</strong><span>这里只保留 keep_tokens；文本标注扩展名等兼容项在每组数据集的高级区配置。</span>';
        wrap.appendChild(heading);

        const fields = [
            ['keep_tokens', 'number'],
        ];

        for (const [key, type, layout] of fields) {
            const row = document.createElement('div');
            row.className = [
                'dataset-config-field',
                layout === 'wide' ? 'wide' : '',
                layout === 'switch' ? 'switch' : '',
            ].filter(Boolean).join(' ');
            row.dataset.key = key;

            const label = document.createElement('label');
            label.className = 'dataset-config-label';
            const nameSpan = document.createElement('span');
            nameSpan.className = 'field-name';
            nameSpan.textContent = datasetConfigLabel(key);
            nameSpan.title = key;
            label.appendChild(nameSpan);

            const btn = document.createElement('button');
            btn.className = 'info-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '查看填写建议、好处、代价、风险和推荐';
            btn.addEventListener('click', () => {
                btn.classList.toggle('active');
                row.querySelector('.field-help')?.classList.toggle('visible');
            });
            label.appendChild(btn);
            row.appendChild(label);

            const input = createDatasetConfigInput(key, type, defaults);
            row.appendChild(input);

            const helpDiv = document.createElement('div');
            helpDiv.className = 'field-help';
            helpDiv.appendChild(createHelpContent(key, datasetConfigValue(key, defaults)));
            row.appendChild(helpDiv);
            wrap.appendChild(row);
        }
        return wrap;
    }

configureDatasetRenderBridge({
    createDatasetPresetGroupNode,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    readDatasetPresetGroupState,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    renderDatasetPresetHeader,
});
