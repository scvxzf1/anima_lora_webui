/* Complete Dragon dataset workspace backed by dataset editor and preset APIs. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import {
    clearActiveOrderedDropTarget,
    clearOrderedDropTargetIf,
    clearOrderedDropTargets,
    scheduleOrderedRowDropTarget,
    setOrderedDropTarget,
} from '../ordered-drag-target.js?v=dragon-ui-20260816v1';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import {
    collectDatasetFields,
    collectDatasetRows,
    createEmptyDatasetRow,
    DATASET_SETTING_KEYS,
    escapeAttribute,
    escapeHtml,
    hydrateDatasetFieldA11y,
    renderDatasetDefaults,
    renderDatasetRow,
    validateDatasetEditor,
} from './dataset-editor-fields.js?v=dragon-ui-20260828v54';
import {
    createDatasetEditorBindings,
    disableDatasetPreviews,
    renderDatasetDirtyState,
    updateDatasetRowSummaries,
    updateDatasetRowSummaryForControl,
} from './dataset-editor-runtime.js?v=dragon-ui-20260826v1';
import { bindDatasetPathTools, refreshDatasetPathStatus } from './dataset-editor-paths.js?v=dragon-ui-20260824v3';
import {
    applyDatasetPreset,
    createDatasetPresetGroup,
    datasetPresetPathFromName,
    deleteDatasetPreset,
    deleteDatasetPresetGroup,
    exportDatasetPreset,
    importDatasetPreset,
    loadDatasetPresetLibrary,
    placeDatasetPreset,
    readDatasetPreset,
    renameDatasetPresetGroup,
    renderDatasetPresetLibrary,
    renderPresetGroups,
    saveDatasetPreset,
    saveDatasetPresetAs,
} from './dataset-editor-presets.js?v=dragon-ui-20260824v71';
import { createDatasetPreviewController } from './dataset-preview-controller.js?v=dragon-ui-20260831v7';
import { loadTrainingContext, mergedConfigUrl } from './training-controls.js?v=dragon-ui-20260824v114';
import { writeTaggingPrefill } from './tagging-context.js?v=dragon-ui-20260831v2';

const api = createApiClient();
let activeEditor = null;
let legacyStageDatasetState = null;
let legacyStageConfigState = null;

export async function loadDatasetEditor() {
    const context = await loadTrainingContext({ includeGpus: false });
    const [editorResult, libraryResult] = await Promise.allSettled([
        loadLinkedDataset(context),
        loadDatasetPresetLibrary(api),
    ]);
    const linked = editorResult.status === 'fulfilled' ? editorResult.value : { datasets: [], defaults: {}, dataset_config: '', error: editorResult.reason?.message || '读取当前数据集配置失败' };
    const library = libraryResult.status === 'fulfilled' ? libraryResult.value : { presets: [], groups: [], error: libraryResult.reason?.message || '读取数据集预设失败' };
    const selectedPreset = library.presets.find((item) => item.path === linked.dataset_config) || null;
    const state = {
        context,
        trainingConfig: null,
        trainingConfigLoading: null,
        presets: library.presets,
        groups: library.groups,
        search: '',
        draggedPresetFile: '',
        presetDropTarget: null,
        presetDragFrame: 0,
        presetPendingDrop: null,
        presetAutoScrollFrame: 0,
        presetAutoScrollClientY: null,
        presetDragRecovery: null,
        presetSuppressClickUntil: 0,
        presetSearchTimer: null,
        selectedFile: linked.dataset_config || '',
        datasetConfig: linked.dataset_config || '',
        rows: linked.datasets?.length ? linked.datasets : [createEmptyDatasetRow(linked.defaults)],
        defaults: linked.defaults || {},
        stageScheduleEnabled: Boolean(linked.stage_schedule_enabled),
        stageSchedule: Array.isArray(linked.stage_schedule) ? linked.stage_schedule : [],
        readonly: Boolean(selectedPreset?.readonly),
        dirty: false,
        isNew: !linked.dataset_config,
        loading: false,
        fatalError: linked.error || '',
        libraryError: library.error || '',
        previewIndex: 0,
        beforeUnload: null,
        stageDialogBound: false,
        stageDialogHandler: null,
        stageScheduleHandler: null,
    };
    activeEditor = state;
    return {
        html: renderPage(state),
        onMount: (root) => bindEditor(root, state),
        beforeLeave: () => shouldLeaveEditor(state),
        onUnmount: () => cleanupEditor(state),
    };
}

async function loadLinkedDataset(context) {
    const params = new URLSearchParams({ variant: context.variant, preset: context.preset, methods_subdir: context.methodsSubdir, config_file: context.configFile });
    const payload = await api(`/api/config/datasets?${params.toString()}`);
    if (payload.ok === false) throw new Error(payload.error || '读取数据集配置失败');
    return payload;
}

function renderPage(state) {
    return `
        <div class="dragon-page dragon-page-wide dragon-dataset-page" data-dataset-page>
            <header class="dragon-dataset-hero dragon-reveal">
                <div><h1>数据集蓝图</h1></div>
                <div class="dragon-dataset-hero-actions">
                    <button class="dragon-btn dragon-btn-secondary" type="button" data-workspace-action="stage">${renderIcon('layers', 'dragon-btn-icon')}<span>分阶段调度</span></button>
                    <button class="dragon-btn dragon-btn-secondary" type="button" data-workspace-action="apply" title="把已保存的数据集预设关联到当前训练配置" ${state.selectedFile ? '' : 'disabled'}>${renderIcon('check', 'dragon-btn-icon')}<span>应用到训练</span></button>
                    <button class="dragon-btn dragon-btn-primary" type="button" data-workspace-action="tagging" title="使用外部视觉模型生成候选 caption" ${state.selectedFile && !state.fatalError ? '' : 'disabled'}>${renderIcon('wand', 'dragon-btn-icon')}<span>打开打标工作台</span></button>
                </div>
            </header>
            <section class="dragon-dataset-context dragon-reveal" data-stagger="1" aria-label="当前训练上下文">
                <div><span>训练配置</span><strong title="${escapeAttribute(state.context.configFile || '')}">${escapeHtml(state.context.configFile || '未选择')}</strong></div>
                <div><span>运行预设</span><strong>${escapeHtml(state.context.preset || '默认')}</strong></div>
                <div><span>数据集文件</span><strong class="dragon-text-mono" data-dataset-path title="${escapeAttribute(state.datasetConfig || '')}">${escapeHtml(state.datasetConfig || '未关联配置')}</strong><span class="dragon-status-badge" data-dataset-link-state data-state="${state.datasetConfig ? 'saved' : 'unsaved'}"><i aria-hidden="true"></i><b>${state.datasetConfig ? '已关联训练配置' : '尚未保存'}</b></span></div>
                <div data-dataset-sync-card><span>编辑状态</span><strong class="dragon-status-badge" data-dataset-dirty data-state="${state.dirty ? 'dirty' : 'synced'}"><i aria-hidden="true"></i><b data-dataset-dirty-text>${state.dirty ? '有未保存更改' : (state.selectedFile === state.datasetConfig ? '已同步至配置' : '已同步至预设')}</b></strong></div>
            </section>
            <div class="dragon-dataset-workspace" data-stagger="2">
                <div class="dragon-dataset-editor-panel">
                    ${renderEditorPanel(state)}
                    <div class="dragon-dataset-savebar" data-dataset-savebar>
                        <div><strong data-savebar-title>${state.selectedFile ? escapeHtml(shortName(state.selectedFile)) : '未命名数据集预设'}</strong><span data-savebar-status>${state.readonly ? '系统预设只读，请复制后编辑。' : '修改会保留在当前页面，保存后写入 TOML。'}</span></div>
                        <div class="dragon-dataset-savebar-actions">
                            <button class="dragon-btn dragon-btn-secondary" type="button" data-workspace-action="reload">重新加载</button>
                            <button class="dragon-btn dragon-btn-secondary" type="button" data-workspace-action="save-as">${state.readonly ? '复制后编辑' : '另存为'}</button>
                            <button class="dragon-btn dragon-btn-primary" type="button" data-workspace-action="save" title="将当前草稿持久化到数据集 TOML" ${state.readonly || state.fatalError ? 'disabled' : ''}>${renderIcon('save', 'dragon-btn-icon')}<span>保存数据集预设</span></button>
                        </div>
                        <span class="dragon-config-feedback" data-dataset-feedback role="status" aria-live="polite"></span>
                    </div>
                </div>
                ${renderDatasetPresetLibrary(state)}
            </div>
        </div>
    `;
}

function renderEditorPanel(state) {
    if (state.fatalError) return `<div class="dragon-dataset-blocking-error"><strong>当前训练配置的数据集读取失败</strong><p>${escapeHtml(state.fatalError)}</p><button class="dragon-btn dragon-btn-secondary" type="button" data-workspace-action="reload">重新读取</button></div>`;
    return `
        <div class="dragon-dataset-editor-head">
            <div><span class="dragon-eyebrow">${state.readonly ? '只读预设' : (state.isNew ? '新预设' : '当前预设')}</span><h2 data-editor-title>${escapeHtml(state.selectedFile ? shortName(state.selectedFile) : '未命名数据集预设')}</h2><p data-editor-path>${escapeHtml(state.selectedFile || '保存后生成 configs/datasets/<名称>.toml')}</p></div>
            <div class="dragon-dataset-editor-tools">
                <button class="dragon-icon-button" type="button" data-workspace-action="export" aria-label="导出当前数据集预设" ${state.selectedFile ? '' : 'disabled'}>${renderIcon('download')}</button>
                <button class="dragon-icon-button" type="button" data-workspace-action="rename" aria-label="重命名当前数据集预设" ${state.selectedFile && !state.readonly ? '' : 'disabled'}>${renderIcon('edit')}</button>
                <button class="dragon-icon-button dragon-icon-button-danger" type="button" data-workspace-action="delete" aria-label="删除当前数据集预设" title="${state.selectedFile === state.datasetConfig ? '当前训练配置正在使用，需先应用其他预设' : '删除当前数据集预设'}" ${state.selectedFile && !state.readonly && state.selectedFile !== state.datasetConfig ? '' : 'disabled'}>${renderIcon('trash')}</button>
            </div>
        </div>
        ${state.libraryError ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="error">${escapeHtml(state.libraryError)}；编辑器仍可使用。</div>` : ''}
        <form class="dragon-dataset-form" data-dataset-form novalidate>
            <section class="dragon-dataset-section dragon-dataset-defaults" data-dataset-defaults>
                <div class="dragon-section-header-row"><div><span class="dragon-eyebrow">通用规则</span><h2 class="dragon-section-title">训练与标注基线</h2><p class="dragon-section-desc">新建数据组会使用这些值；修改后可明确同步到现有组。</p></div><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-dataset-apply-defaults ${state.readonly ? 'disabled' : ''}>同步到所有组</button></div>
                <div class="dragon-dataset-defaults-content" data-dataset-defaults-content>${renderDatasetDefaults(state.defaults, { readonly: state.readonly })}</div>
            </section>
            <section class="dragon-dataset-section dragon-dataset-groups">
                <div class="dragon-section-header-row"><div><span class="dragon-eyebrow">数据来源</span><h2 class="dragon-section-title">数据集分组</h2><p class="dragon-section-desc">每组对应一个原始图片目录，可独立设置重复次数与高级规则。</p></div><button class="dragon-btn dragon-btn-secondary" type="button" data-dataset-add ${state.readonly ? 'disabled' : ''}>添加数据集组</button></div>
                <div class="dragon-dataset-rows" data-dataset-rows>${state.rows.map((row, index) => renderDatasetRow(row, index, state.defaults, { readonly: state.readonly, canPreview: Boolean(state.selectedFile) && !state.dirty, totalRows: state.rows.length })).join('')}</div>
            </section>
        </form>
    `;
}

function bindEditor(root, state) {
    state.root = root;
    state.ui = createDatasetEditorBindings(root);
    root.classList.add('dragon-dataset-page-host');
    state.beforeUnload = (event) => { if (!state.dirty) return; event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', state.beforeUnload);
    state.previewController?.dispose();
    state.previewController = createDatasetPreviewController(api, state);
    bindStageScheduleEvents(root, state);
    bindWorkspace(root, state);
    bindPresetLibrary(root, state);
    bindForm(root, state);
    syncLegacyDatasetState(state);
}


function bindWorkspace(root, state) {
    root.querySelectorAll('[data-workspace-action]').forEach((button) => {
        if (button.dataset.dragonDatasetBound === 'true') return;
        button.dataset.dragonDatasetBound = 'true';
        button.addEventListener('click', async () => {
        const action = button.dataset.workspaceAction;
        if (action === 'save') return saveCurrentPreset(root, state);
        if (action === 'save-as') return saveAsCurrentPreset(root, state);
        if (action === 'apply') return applyCurrentPreset(root, state);
        if (action === 'reload') return reloadCurrentPreset(root, state);
        if (action === 'delete') return removeCurrentPreset(root, state);
        if (action === 'export') return exportCurrentPreset(root, state);
        if (action === 'rename') return renameCurrentPreset(root, state);
        if (action === 'stage') return openStageSchedule(root, state);
        if (action === 'tagging') return openTaggingWorkspace(root, state);
        });
    });
}

function bindPresetLibrary(root, state) {
    root.querySelector('[data-preset-search]')?.addEventListener('input', (event) => {
        state.search = event.target.value;
        if (state.presetSearchTimer) window.clearTimeout(state.presetSearchTimer);
        state.presetSearchTimer = window.setTimeout(() => {
            state.presetSearchTimer = null;
            refreshPresetList(root, state);
        }, 100);
    });
    root.querySelectorAll('[data-preset-action]').forEach((button) => button.addEventListener('click', async () => {
        if (button.dataset.presetAction === 'refresh') return refreshLibrary(root, state);
        if (button.dataset.presetAction === 'new') return startNewPreset(root, state);
        if (button.dataset.presetAction === 'new-group') return createPresetGroup(root, state);
        if (button.dataset.presetAction === 'export') return exportCurrentPreset(root, state);
        if (button.dataset.presetAction === 'import') root.querySelector('[data-preset-import]')?.click();
    }));
    root.querySelector('[data-preset-import]')?.addEventListener('change', (event) => importPresetFile(root, state, event));
    bindPresetListActions(root, state);
    bindPresetButtons(root, state);
}

function bindPresetListActions(root, state) {
    const list = root.querySelector('[data-preset-list]');
    if (!list || list.dataset.dragonPresetActionsBound === 'true') return;
    list.dataset.dragonPresetActionsBound = 'true';
    const activate = (item) => {
        if (!item || performance.now() < state.presetSuppressClickUntil) return;
        selectPreset(root, state, item.dataset.presetFile);
    };
    list.addEventListener('click', (event) => {
        if (event.target.closest('.dragon-dataset-preset-drag-handle')) return;
        const button = event.target.closest('[data-preset-group-action]');
        if (!button) return activate(event.target.closest('.dragon-dataset-preset-item[data-preset-file]'));
        const group = state.groups.find((item) => item.id === button.dataset.groupId);
        if (!group) return;
        if (button.dataset.presetGroupAction === 'rename') return renamePresetGroup(root, state, group);
        if (button.dataset.presetGroupAction === 'delete') return removePresetGroup(root, state, group);
    });
    list.addEventListener('keydown', (event) => {
        if (!['Enter', ' '].includes(event.key)) return;
        const item = event.target.closest('.dragon-dataset-preset-item[data-preset-file]');
        if (!item) return;
        event.preventDefault();
        activate(item);
    });
}

function bindPresetButtons(root, state) {
    if (!String(state.search || '').trim()) bindPresetDragAndDrop(root, state);
}

function bindPresetDragAndDrop(root, state) {
    bindPresetDragRecovery(root, state);
    const list = root.querySelector('[data-preset-list]');
    if (list && list.dataset.dragonPresetDragBound !== 'true') {
        list.dataset.dragonPresetDragBound = 'true';
        list.addEventListener('dragover', (event) => {
            if (state.draggedPresetFile) schedulePresetListAutoScroll(state, list, event.clientY);
        }, true);
        list.addEventListener('dragleave', (event) => {
            if (!event.relatedTarget || !list.contains(event.relatedTarget)) stopPresetListAutoScroll(state);
        });
    }
    root.querySelectorAll('[data-preset-row]').forEach((row) => {
        const handle = row.querySelector('.dragon-dataset-preset-drag-handle[draggable="true"]');
        handle?.addEventListener('click', (event) => event.stopPropagation());
        handle?.addEventListener('dragstart', (event) => {
            finishPresetDrag(root, state);
            const file = row.dataset.presetRow || '';
            if (!file) return;
            state.draggedPresetFile = file;
            state.presetSuppressClickUntil = performance.now() + 300;
            root.classList.add('dragon-dataset-dragging');
            row.classList.add('dragon-dataset-preset-dragging');
            event.dataTransfer?.setData('text/plain', file);
            if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
        });
        handle?.addEventListener('dragend', () => finishPresetDrag(root, state, row));
        row.addEventListener('dragover', (event) => {
            const groupId = row.closest('[data-preset-group]')?.dataset.presetGroup || '';
            if (row.dataset.presetRow === state.draggedPresetFile) return;
            if (!canDropPresetToGroup(state, state.draggedPresetFile, groupId)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            schedulePresetRowDropTarget(state, row, event.clientY);
        });
        row.addEventListener('drop', async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const groupId = row.closest('[data-preset-group]')?.dataset.presetGroup || '';
            const file = state.draggedPresetFile || event.dataTransfer?.getData('text/plain') || '';
            const rect = row.getBoundingClientRect();
            const position = event.clientY < rect.top + rect.height / 2 ? 'before' : 'after';
            finishPresetDrag(root, state);
            if (row.dataset.presetRow === file) return;
            if (!canDropPresetToGroup(state, file, groupId)) return;
            await placePresetAt(root, state, file, groupId, row.dataset.presetRow || '', position);
        });
    });
    root.querySelectorAll('[data-preset-dropzone]').forEach((dropzone) => {
        dropzone.addEventListener('dragover', (event) => {
            const groupId = dropzone.dataset.presetDropzone || '';
            if (!canDropPresetToGroup(state, state.draggedPresetFile, groupId)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            setPresetDropTarget(state, dropzone, 'dropzone');
        });
        dropzone.addEventListener('dragleave', (event) => {
            if (event.relatedTarget && dropzone.contains(event.relatedTarget)) return;
            clearPresetDropTargetIf(state, dropzone);
        });
        dropzone.addEventListener('drop', async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const file = state.draggedPresetFile || event.dataTransfer?.getData('text/plain') || '';
            const groupId = dropzone.dataset.presetDropzone || '';
            finishPresetDrag(root, state);
            if (!canDropPresetToGroup(state, file, groupId)) return;
            await placePresetAt(root, state, file, groupId, '', 'after');
        });
    });
    root.querySelectorAll('[data-preset-drop-group]').forEach((groupNode) => {
        groupNode.addEventListener('dragover', (event) => {
            if (event.target.closest?.('[data-preset-row], [data-preset-dropzone]')) return;
            if (!canDropPresetToGroup(state, state.draggedPresetFile, groupNode.dataset.presetDropGroup)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
            setPresetDropTarget(state, groupNode, 'group');
        });
        groupNode.addEventListener('dragleave', (event) => {
            if (event.relatedTarget && groupNode.contains(event.relatedTarget)) return;
            clearPresetDropTargetIf(state, groupNode);
        });
        groupNode.addEventListener('drop', async (event) => {
            event.preventDefault();
            if (event.target.closest?.('[data-preset-row], [data-preset-dropzone]')) return;
            const file = state.draggedPresetFile || event.dataTransfer?.getData('text/plain') || '';
            const groupId = groupNode.dataset.presetDropGroup || '';
            finishPresetDrag(root, state);
            if (!canDropPresetToGroup(state, file, groupId)) return;
            await placePresetAt(root, state, file, groupId, '', 'after');
        });
    });
}

function presetListAutoScrollDelta(list, clientY) {
    const rect = list.getBoundingClientRect();
    const edge = Math.min(84, rect.height * 0.22);
    if (clientY < rect.top + edge) {
        const intensity = Math.min(1, (rect.top + edge - clientY) / edge);
        return -Math.ceil(4 + 18 * intensity);
    }
    if (clientY > rect.bottom - edge) {
        const intensity = Math.min(1, (clientY - (rect.bottom - edge)) / edge);
        return Math.ceil(4 + 18 * intensity);
    }
    return 0;
}

function schedulePresetListAutoScroll(state, list, clientY) {
    state.presetAutoScrollClientY = clientY;
    if (state.presetAutoScrollFrame) return;
    const tick = () => {
        state.presetAutoScrollFrame = 0;
        if (!state.draggedPresetFile || state.presetAutoScrollClientY == null) return;
        const delta = presetListAutoScrollDelta(list, state.presetAutoScrollClientY);
        if (!delta) return;
        list.scrollTop += delta;
        state.presetAutoScrollFrame = window.requestAnimationFrame(tick);
    };
    state.presetAutoScrollFrame = window.requestAnimationFrame(tick);
}

function stopPresetListAutoScroll(state) {
    if (state.presetAutoScrollFrame) window.cancelAnimationFrame(state.presetAutoScrollFrame);
    state.presetAutoScrollFrame = 0;
    state.presetAutoScrollClientY = null;
}

const PRESET_DRAG_TARGET_OPTIONS = Object.freeze({
    frameKey: 'presetDragFrame',
    pendingKey: 'presetPendingDrop',
    targetKey: 'presetDropTarget',
    rowClassPrefix: 'dragon-dataset-preset-drop',
    groupTargetClass: 'dragon-dataset-preset-drop-target',
});

function schedulePresetRowDropTarget(state, row, clientY) {
    scheduleOrderedRowDropTarget(state, row, clientY, PRESET_DRAG_TARGET_OPTIONS);
}

function setPresetDropTarget(state, node, kind, position = '') {
    setOrderedDropTarget(state, node, kind, position, PRESET_DRAG_TARGET_OPTIONS);
}

function clearActivePresetDropTarget(state) {
    clearActiveOrderedDropTarget(state, PRESET_DRAG_TARGET_OPTIONS);
}

function clearPresetDropTargetIf(state, node) {
    clearOrderedDropTargetIf(state, node, PRESET_DRAG_TARGET_OPTIONS);
}

function clearPresetDropTargets(state) {
    clearOrderedDropTargets(state, PRESET_DRAG_TARGET_OPTIONS);
}

function finishPresetDrag(root, state, row = null) {
    stopPresetListAutoScroll(state);
    state.draggedPresetFile = '';
    root.classList.remove('dragon-dataset-dragging');
    row?.classList.remove('dragon-dataset-preset-dragging');
    root.querySelectorAll('.dragon-dataset-preset-dragging').forEach((node) => node.classList.remove('dragon-dataset-preset-dragging'));
    clearPresetDropTargets(state);
}

function bindPresetDragRecovery(root, state) {
    if (state.presetDragRecovery) return;
    const finish = () => finishPresetDrag(root, state);
    const finishAfterDrop = () => window.requestAnimationFrame(() => {
        if (state.draggedPresetFile || root.classList.contains('dragon-dataset-dragging')) finish();
    });
    const cancelOnEscape = (event) => { if (event.key === 'Escape') finish(); };
    window.addEventListener('dragend', finish, true);
    window.addEventListener('drop', finishAfterDrop, true);
    window.addEventListener('blur', finish);
    document.addEventListener('keydown', cancelOnEscape);
    state.presetDragRecovery = { finish, finishAfterDrop, cancelOnEscape };
}

function disposePresetDragRecovery(root, state) {
    const recovery = state.presetDragRecovery;
    if (!recovery) return;
    window.removeEventListener('dragend', recovery.finish, true);
    window.removeEventListener('drop', recovery.finishAfterDrop, true);
    window.removeEventListener('blur', recovery.finish);
    document.removeEventListener('keydown', recovery.cancelOnEscape);
    state.presetDragRecovery = null;
    finishPresetDrag(root, state);
}

function canDropPresetToGroup(state, file, groupId) {
    if (!file || !groupId) return false;
    const preset = state.presets.find((item) => item.path === file);
    const group = state.groups.find((item) => item.id === groupId);
    const sourceGroup = state.groups.find((item) => (item.files || []).some((itemFile) => itemFile.path === file));
    return Boolean(
        !String(state.search || '').trim() && preset && !preset.readonly && group?.kind === 'dataset' && group.movable
        && !group.locked && !group.group_locked && !group.user_group_locked && !group.system_locked
        && (!sourceGroup || (sourceGroup.movable && !sourceGroup.locked && !sourceGroup.group_locked && !sourceGroup.user_group_locked && !sourceGroup.system_locked))
    );
}

function bindForm(root, state) {
    hydrateDatasetFieldA11y(root.querySelector('[data-dataset-form]'));
    const form = root.querySelector('[data-dataset-form]');
    form?.addEventListener('input', (event) => {
        markDirty(root, state);
        updateDatasetRowSummaryForControl(event.target);
    });
    form?.addEventListener('change', (event) => {
        markDirty(root, state);
        updateDatasetRowSummaryForControl(event.target);
    });
    root.querySelector('[data-dataset-add]')?.addEventListener('click', () => {
        syncStateFromForm(root, state);
        state.rows.push(createEmptyDatasetRow(state.defaults));
        state.dirty = true;
        refreshRows(root, state);
        updateDirtyUi(root, state);
    });
    root.querySelector('[data-dataset-apply-defaults]')?.addEventListener('click', () => applyDefaultsToRows(root, state));
    bindRowActions(root, state);
}

function applyDefaultsToRows(root, state) {
    syncStateFromForm(root, state);
    const shared = Object.fromEntries(DATASET_SETTING_KEYS.filter((key) => key in state.defaults).map((key) => [key, state.defaults[key]]));
    state.rows = state.rows.map((row) => ({ ...row, settings: { ...(row.settings || {}), ...shared } }));
    state.dirty = true;
    refreshRows(root, state);
    updateDirtyUi(root, state);
    showFeedback(root, `已把通用基线同步到 ${state.rows.length} 个数据集组`, 'success');
}

function bindRowActions(root, state) {
    state.rowPathCleanup?.();
    state.rowPathCleanup = bindDatasetPathTools(api, root, { onFeedback: (message, tone) => showFeedback(root, message, tone) });
    root.querySelectorAll('[data-dataset-remove]').forEach((button) => button.addEventListener('click', () => {
        const rows = [...root.querySelectorAll('[data-dataset-row]')];
        if (rows.length <= 1) return showFeedback(root, '至少需要保留 1 个数据集组', 'error');
        button.closest('[data-dataset-row]')?.remove();
        syncStateFromForm(root, state);
        state.dirty = true;
        refreshRows(root, state);
        updateDirtyUi(root, state);
    }));
    root.querySelectorAll('[data-dataset-move]').forEach((button) => button.addEventListener('click', () => {
        syncStateFromForm(root, state);
        const row = button.closest('[data-dataset-row]');
        const index = Number(row?.dataset.index || 0);
        const next = button.dataset.datasetMove === 'up' ? index - 1 : index + 1;
        if (next < 0 || next >= state.rows.length) return;
        [state.rows[index], state.rows[next]] = [state.rows[next], state.rows[index]];
        state.dirty = true;
        refreshRows(root, state);
        updateDirtyUi(root, state);
    }));
    root.querySelectorAll('[data-dataset-row]').forEach((row) => {
        row.addEventListener('keydown', (event) => {
            if (!event.altKey || !['ArrowUp', 'ArrowDown'].includes(event.key)) return;
            if (event.target.closest('input, select, textarea')) return;
            const direction = event.key === 'ArrowUp' ? 'up' : 'down';
            const button = row.querySelector(`[data-dataset-move="${direction}"]`);
            if (!button || button.disabled) return;
            event.preventDefault();
            button.click();
        });
    });
    root.querySelectorAll('[data-dataset-suggest]').forEach((button) => button.addEventListener('click', () => suggestDirectories(root, state, button)));
    root.querySelectorAll('[data-dataset-preview]').forEach((button) => button.addEventListener('click', async () => {
        try { await state.previewController.open(Number(button.closest('[data-dataset-row]')?.dataset.index || 0)); } catch (error) { showFeedback(root, error.message, 'error'); }
    }));
}

async function selectPreset(root, state, file) {
    if (file === state.selectedFile) return;
    if (!confirmDiscard(state, '切换预设')) return;
    await loadPresetIntoState(root, state, file);
}

async function loadPresetIntoState(root, state, file) {
    setLoading(root, state, true, '正在读取预设…');
    try {
        const payload = await readDatasetPreset(api, file);
        hydratePreset(state, payload);
        refreshEditor(root, state);
        refreshPresetList(root, state);
        showFeedback(root, `已加载 ${shortName(file)}`, 'success');
    } catch (error) {
        showFeedback(root, error.message, 'error');
    } finally { setLoading(root, state, false); }
}

function hydratePreset(state, payload) {
    state.selectedFile = payload.file || '';
    state.rows = Array.isArray(payload.datasets) && payload.datasets.length ? payload.datasets : [createEmptyDatasetRow(payload.defaults || {})];
    state.defaults = payload.defaults || {};
    state.readonly = Boolean(payload.readonly);
    state.stageScheduleEnabled = Boolean(payload.stage_schedule_enabled);
    state.stageSchedule = Array.isArray(payload.stage_schedule) ? payload.stage_schedule : [];
    state.fatalError = '';
    state.previewIndex = Math.max(0, Math.min(Number(state.previewIndex || 0), state.rows.length - 1));
    state.dirty = false;
    state.isNew = false;
    clearLegacyStageDraftMarkers(state);
    syncLegacyDatasetState(state);
}

function startNewPreset(root, state) {
    if (!confirmDiscard(state, '新建预设')) return;
    state.selectedFile = '';
    state.defaults = {};
    state.rows = [createEmptyDatasetRow()];
    state.readonly = false;
    state.isNew = true;
    state.dirty = true;
    state.stageScheduleEnabled = false;
    state.stageSchedule = [];
    state.previewIndex = 0;
    clearLegacyStageDraftMarkers(state);
    refreshEditor(root, state);
    refreshPresetList(root, state);
}

async function refreshLibrary(root, state) {
    setLoading(root, state, true, '正在刷新预设库…');
    try {
        await reloadPresetLibrary(root, state);
        showFeedback(root, '预设库已刷新', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally { setLoading(root, state, false); }
}

async function reloadPresetLibrary(root, state) {
    const library = await loadDatasetPresetLibrary(api);
    state.presets = library.presets;
    state.groups = library.groups;
    state.libraryError = '';
    refreshPresetList(root, state);
    syncLegacyDatasetState(state);
    return library;
}

async function createPresetGroup(root, state) {
    const label = window.prompt('请输入新的数据集预设分组名称：', '');
    if (label === null) return;
    if (!String(label).trim()) return showFeedback(root, '分组名称不能为空', 'error');
    setLoading(root, state, true, '正在创建分组…');
    try {
        const result = await createDatasetPresetGroup(api, String(label).trim());
        await reloadPresetLibrary(root, state);
        showFeedback(root, result.message || '数据集分组已创建', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally { setLoading(root, state, false); }
}

async function renamePresetGroup(root, state, group) {
    if (!group?.id || !group.renamable) return;
    const label = window.prompt('请输入新的数据集预设分组名称：', group.label || group.id);
    if (label === null) return;
    if (!String(label).trim()) return showFeedback(root, '分组名称不能为空', 'error');
    setLoading(root, state, true, '正在重命名分组…');
    try {
        const result = await renameDatasetPresetGroup(api, group.id, String(label).trim());
        await reloadPresetLibrary(root, state);
        showFeedback(root, result.message || '数据集分组已重命名', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally { setLoading(root, state, false); }
}

async function removePresetGroup(root, state, group) {
    if (!group?.id || !group.deletable) return;
    const count = Number(group.files?.length || 0);
    const detail = count
        ? `只删除分组，不删除其中 ${count} 个 TOML；这些预设会回到其他可见数据集分组。`
        : '只删除这个空分组，不会删除任何 TOML。';
    if (!window.confirm(`确认删除分组“${group.label || group.id}”吗？\n${detail}`)) return;
    setLoading(root, state, true, '正在删除分组…');
    try {
        const result = await deleteDatasetPresetGroup(api, group.id);
        await reloadPresetLibrary(root, state);
        showFeedback(root, result.message || '数据集分组已删除，TOML 已保留', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally { setLoading(root, state, false); }
}

function presetOrderForDrop(state, file, groupId, anchorFile = '', position = 'after') {
    const targetGroup = state.groups.find((item) => item.id === groupId);
    const paths = (targetGroup?.files || []).map((item) => item.path).filter((path) => path !== file);
    const anchorIndex = anchorFile ? paths.indexOf(anchorFile) : -1;
    const insertIndex = anchorIndex < 0 ? paths.length : anchorIndex + (position === 'before' ? 0 : 1);
    paths.splice(insertIndex, 0, file);
    return paths;
}

async function placePresetAt(root, state, file, groupId, anchorFile, position) {
    finishPresetDrag(root, state);
    const order = presetOrderForDrop(state, file, groupId, anchorFile, position);
    const sourceGroup = state.groups.find((item) => (item.files || []).some((preset) => preset.path === file));
    const currentOrder = (sourceGroup?.files || []).map((preset) => preset.path);
    if (sourceGroup?.id === groupId && order.length === currentOrder.length && order.every((path, index) => path === currentOrder[index])) return;
    setLoading(root, state, true, '正在移动预设…');
    try {
        const result = await placeDatasetPreset(api, file, groupId, order);
        await reloadPresetLibrary(root, state);
        showFeedback(root, result.message || '数据集预设位置已更新', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally {
        finishPresetDrag(root, state);
        setLoading(root, state, false);
    }
}

async function saveCurrentPreset(root, state) {
    if (!state.selectedFile) return saveAsCurrentPreset(root, state);
    if (state.readonly) return showFeedback(root, '系统预设只读，请使用“复制后编辑”', 'error');
    if (!prepareForSave(root, state)) return false;
    setLoading(root, state, true, '正在保存…');
    try {
        const result = await saveDatasetPreset(api, presetPayload(state, { file: state.selectedFile, overwrite: true }));
        hydratePreset(state, { ...result, readonly: false });
        await refreshLibrary(root, state);
        refreshEditor(root, state);
        updateContextUi(root, state);
        showFeedback(root, result.message || '数据集预设已保存', 'success');
        return true;
    } catch (error) { showFeedback(root, error.message, 'error'); return false; } finally { setLoading(root, state, false); }
}

async function saveAsCurrentPreset(root, state, suggestedName = '') {
    if (!prepareForSave(root, state)) return false;
    const name = window.prompt('请输入新的数据集预设名称（保存到 configs/datasets/）：', suggestedName || `${shortName(state.selectedFile || 'dataset').replace(/\.toml$/i, '')}${state.readonly ? '_copy' : ''}`);
    if (name === null) return false;
    if (!String(name).trim()) return showFeedback(root, '预设名称不能为空', 'error');
    setLoading(root, state, true, '正在另存…');
    try {
        const result = await saveDatasetPresetAs(api, presetPayload(state, { name }));
        hydratePreset(state, { ...result, readonly: false });
        await refreshLibrary(root, state);
        refreshEditor(root, state);
        updateContextUi(root, state);
        showFeedback(root, result.message || '已另存数据集预设', 'success');
        return true;
    } catch (error) { showFeedback(root, error.message, 'error'); return false; } finally { setLoading(root, state, false); }
}

async function applyCurrentPreset(root, state) {
    if (state.dirty && !(await saveCurrentPreset(root, state))) return;
    if (!state.selectedFile) return showFeedback(root, '请先选择或保存一个数据集预设', 'error');
    setLoading(root, state, true, '正在应用到训练配置…');
    try {
        const result = await applyDatasetPreset(api, state.selectedFile, state.context.configFile);
        state.datasetConfig = result.dataset_config || state.selectedFile;
        refreshEditor(root, state);
        updateContextUi(root, state);
        showFeedback(root, result.message || '已应用到当前训练配置', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally { setLoading(root, state, false); }
}

async function reloadCurrentPreset(root, state) {
    if (!confirmDiscard(state, '重新加载')) return;
    if (state.selectedFile) {
        state.dirty = false;
        return loadPresetIntoState(root, state, state.selectedFile);
    }
    state.dirty = false;
    clearLegacyStageDraftMarkers(state);
    window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
}

async function removeCurrentPreset(root, state) {
    if (!state.selectedFile || state.readonly) return;
    if (state.selectedFile === state.datasetConfig) return showFeedback(root, '当前训练配置正在使用这个预设，请先应用其他数据集预设再删除', 'error');
    if (!window.confirm(`确认删除 ${shortName(state.selectedFile)} 吗？\n只会删除 TOML 预设，不会删除图片或缓存目录。`)) return;
    setLoading(root, state, true, '正在删除预设…');
    try {
        await deleteDatasetPreset(api, state.selectedFile);
        state.dirty = false;
        state.selectedFile = '';
        state.isNew = true;
        state.rows = [createEmptyDatasetRow(state.defaults)];
        state.stageScheduleEnabled = false;
        state.stageSchedule = [];
        state.previewIndex = 0;
        clearLegacyStageDraftMarkers(state);
        await refreshLibrary(root, state);
        refreshEditor(root, state);
        showFeedback(root, '数据集预设已删除，图片和缓存目录未受影响', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally { setLoading(root, state, false); }
}

async function renameCurrentPreset(root, state) {
    if (!state.selectedFile || state.readonly) return;
    syncStateFromForm(root, state);
    const oldFile = state.selectedFile;
    const name = window.prompt('输入新的预设名称：', shortName(oldFile).replace(/\.toml$/i, ''));
    if (name === null || !String(name).trim()) return;
    const target = datasetPresetPathFromName(name);
    if (target === oldFile) return;
    setLoading(root, state, true, '正在重命名…');
    try {
        const result = await saveDatasetPresetAs(api, presetPayload(state, { name }));
        const linkedToTraining = state.datasetConfig === oldFile;
        if (linkedToTraining) {
            try {
                await applyDatasetPreset(api, result.file, state.context.configFile);
                state.datasetConfig = result.file;
            } catch (error) {
                await reloadPresetLibrary(root, state);
                showFeedback(root, `新预设已保存，但当前训练配置更新失败；旧预设未删除：${error.message}`, 'error');
                return;
            }
        }
        try {
            await deleteDatasetPreset(api, oldFile);
        } catch (error) {
            hydratePreset(state, { ...result, readonly: false });
            await refreshLibrary(root, state);
            refreshEditor(root, state);
            updateContextUi(root, state);
            showFeedback(root, `新预设已保存${linkedToTraining ? '并已更新当前训练配置' : ''}，但旧预设删除失败：${error.message}`, 'error');
            return;
        }
        hydratePreset(state, { ...result, readonly: false });
        await refreshLibrary(root, state);
        refreshEditor(root, state);
        updateContextUi(root, state);
        showFeedback(root, '数据集预设已重命名', 'success');
    } catch (error) { showFeedback(root, `重命名失败：${error.message}`, 'error'); } finally { setLoading(root, state, false); }
}

async function exportCurrentPreset(root, state) {
    if (!state.selectedFile) return;
    try { await exportDatasetPreset(api, state.selectedFile); showFeedback(root, '已导出数据集预设', 'success'); } catch (error) { showFeedback(root, error.message, 'error'); }
}

async function importPresetFile(root, state, event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!confirmDiscard(state, '导入预设')) { event.target.value = ''; return; }
    const name = window.prompt('请输入导入后的预设名称：', file.name.replace(/\.toml$/i, ''));
    if (name === null) { event.target.value = ''; return; }
    setLoading(root, state, true, '正在导入预设…');
    try {
        const result = await importDatasetPreset(api, name, await file.text());
        hydratePreset(state, { ...result, readonly: false });
        await refreshLibrary(root, state);
        refreshEditor(root, state);
        showFeedback(root, '已导入数据集预设', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally { event.target.value = ''; setLoading(root, state, false); }
}

async function suggestDirectories(root, state, button) {
    const row = button.closest('[data-dataset-row]');
    const source = row?.querySelector('[data-field="source_dir"]')?.value || '';
    if (!source.trim()) return showFeedback(root, '请先填写原始图片目录', 'error');
    button.disabled = true;
    try {
        const result = await api('/api/config/datasets/suggest', { method: 'POST', body: JSON.stringify({ source_dirs: [source] }) });
        if (result.ok === false) throw new Error(result.error || '推导目录失败');
        const item = result.datasets?.[0];
        if (!item) throw new Error('后端没有返回处理目录');
        row.querySelector('[data-field="source_dir"]').value = item.source_dir || source;
        row.querySelector('[data-field="image_dir"]').value = item.image_dir || '';
        row.querySelector('[data-field="cache_dir"]').value = item.cache_dir || '';
        markDirty(root, state);
        refreshDatasetPathStatus(api, row);
        showFeedback(root, '已推导缩放图片与 LoRA 缓存目录', 'success');
    } catch (error) { showFeedback(root, error.message, 'error'); } finally { button.disabled = false; }
}

async function openStageSchedule(root, state) {
    syncStateFromForm(root, state);
    if (!state.selectedFile) return showFeedback(root, '请先保存或选择一个数据集预设', 'error');
    try {
        await ensureTrainingConfig(state);
    } catch (error) {
        showFeedback(root, `分阶段调度需要读取当前训练配置：${error.message}`, 'error');
        return;
    }
    syncLegacyDatasetState(state);
    try {
        const [{ configureAppContextBridge }, { configureDatasetStateBridge }, { configureConfigStateBridge }, { configureRuntimeBridge }, { configureTomlActionStateBridge }, { createDatasetState }, { createConfigState }] = await Promise.all([
            import('../../features/anima-app/helpers/app-context-bridge.js?v=module-bootstrap-20260831-release-v1'),
            import('../../features/anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260831-release-v1'),
            import('../../features/anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260831-release-v1'),
            import('../../features/anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260831-release-v1'),
            import('../../features/anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260831-release-v1'),
            import('../../features/anima-app/state/dataset-state.js?v=module-bootstrap-20260831-release-v1'),
            import('../../features/anima-app/state/config-state.js?v=module-bootstrap-20260831-release-v1'),
        ]);
        if (!legacyStageDatasetState) {
            legacyStageDatasetState = createDatasetState();
            legacyStageConfigState = createConfigState();
            const runtimeApi = (...args) => api(...args);
            runtimeApi.datasetPresetApi = (...args) => api(...args);
            configureRuntimeBridge({ api: runtimeApi, dom: {} });
        }
        configureTomlActionStateBridge({ setTomlStatus: (_tone, message) => showFeedback(root, message, _tone === 'error' ? 'error' : (_tone === 'ok' ? 'success' : 'info')) });
        state.legacyDatasetState = legacyStageDatasetState;
        state.legacyConfigState = legacyStageConfigState;
        configureAppContextBridge({ api });
        configureDatasetStateBridge(state.legacyDatasetState);
        configureConfigStateBridge(state.legacyConfigState);
        syncLegacyDatasetState(state);
        bindStageDialogSync(root, state);
        const { openStageResolutionDialog } = await import('../../features/config-form/stage-resolution-ui.js?v=module-bootstrap-20260831-release-v1');
        openStageResolutionDialog();
    } catch (error) { showFeedback(root, `分阶段调度加载失败：${error.message}`, 'error'); }
}

async function ensureTrainingConfig(state) {
    if (state.trainingConfig) return state.trainingConfig;
    if (!state.trainingConfigLoading) {
        state.trainingConfigLoading = api(mergedConfigUrl(state.context)).then((payload) => {
            if (!payload || payload.ok === false || payload.error) {
                throw new Error(payload?.error || '当前训练配置读取失败');
            }
            state.trainingConfig = payload.config || payload;
            return state.trainingConfig;
        }).finally(() => { state.trainingConfigLoading = null; });
    }
    return state.trainingConfigLoading;
}

function bindStageDialogSync(root, state) {
    if (state.stageDialogBound) return;
    const dialog = document.getElementById('stage-resolution-dialog');
    if (!dialog) return;
    state.stageDialogBound = true;
    state.stageDialogHandler = () => syncStageScheduleFromLegacy(root, state);
    dialog.addEventListener('close', state.stageDialogHandler);
}

function bindStageScheduleEvents(root, state) {
    state.stageScheduleHandler = (event) => applyStageScheduleEvent(root, state, event.detail || {});
    window.addEventListener('anima-stage-schedule-change', state.stageScheduleHandler);
}

async function applyStageScheduleEvent(root, state, detail) {
    if (!root.isConnected || activeEditor !== state) return;
    if (!detail || !['draft', 'saved', 'saved-as'].includes(detail.phase)) return;
    state.stageScheduleEnabled = Boolean(detail.stage_schedule_enabled);
    state.stageSchedule = Array.isArray(detail.stage_schedule)
        ? detail.stage_schedule.map((stage) => ({ ...stage }))
        : [];
    if (detail.file) state.selectedFile = detail.file;
    if (Array.isArray(detail.datasets) && detail.datasets.length) state.rows = detail.datasets;
    if (detail.defaults) state.defaults = detail.defaults;
    if (detail.readonly != null) state.readonly = Boolean(detail.readonly);
    state.dirty = detail.phase === 'draft' || Boolean(detail.dirty);
    syncLegacyDatasetState(state);
    if (detail.phase === 'saved-as') await refreshLibrary(root, state);
    refreshEditor(root, state);
    refreshPresetList(root, state);
}

function syncStageScheduleFromLegacy(root, state) {
    if (!root.isConnected || activeEditor !== state) return;
    const legacy = state.legacyDatasetState?.datasetEditorState;
    if (!legacy) return;
    const nextFile = legacy.selectedFile || legacy.dataset_config || state.selectedFile;
    const schedule = Array.isArray(legacy.stage_schedule) ? legacy.stage_schedule : [];
    state.selectedFile = nextFile;
    state.stageScheduleEnabled = Boolean(legacy.stage_schedule_enabled);
    state.stageSchedule = schedule;
    state.rows = Array.isArray(legacy.datasets) && legacy.datasets.length ? legacy.datasets : state.rows;
    state.defaults = legacy.defaults || state.defaults;
    state.readonly = Boolean(legacy.readonly);
    state.dirty = Boolean(legacy.dirty);
    syncLegacyDatasetState(state);
    refreshEditor(root, state);
    refreshPresetList(root, state);
    showFeedback(root, state.dirty ? '分阶段调度已写入当前草稿，请保存数据集预设' : '分阶段调度已同步', state.dirty ? 'info' : 'success');
}

function syncLegacyDatasetState(state) {
    if (!state.legacyDatasetState || !state.legacyConfigState) return;
    state.legacyDatasetState.datasetEditorState = {
        ...state.legacyDatasetState.datasetEditorState,
        loading: state.loading,
        dirty: state.dirty,
        dataset_config: state.selectedFile,
        selectedFile: state.selectedFile,
        datasets: state.rows,
        defaults: state.defaults,
        readonly: state.readonly,
        stage_schedule_enabled: state.stageScheduleEnabled,
        stage_schedule: state.stageSchedule,
    };
    state.legacyDatasetState.datasetPresetState = {
        ...state.legacyDatasetState.datasetPresetState,
        presets: state.presets,
        groups: state.groups,
        selectedFile: state.selectedFile,
        datasets: state.rows,
        defaults: state.defaults,
        readonly: state.readonly,
        dirty: state.dirty,
        stage_schedule_enabled: state.stageScheduleEnabled,
        stage_schedule: state.stageSchedule,
    };
    state.legacyDatasetState.selectedConfigDatasetFile = state.datasetConfig;
    state.legacyConfigState.currentConfig = {
        ...state.legacyConfigState.currentConfig,
        ...(state.trainingConfig || {}),
        dataset_config: state.datasetConfig,
        stage_schedule_enabled: state.stageScheduleEnabled,
        stage_schedule: state.stageSchedule.map((stage) => ({ ...stage })),
    };
}

function clearLegacyStageDraftMarkers(state) {
    const draft = state.legacyConfigState?.configFormState?.draftValues;
    if (!draft || typeof draft.delete !== 'function') return;
    draft.delete('stage_schedule_enabled');
    draft.delete('stage_schedule');
}

function prepareForSave(root, state) {
    const errors = validateDatasetEditor(root, { stageScheduleEnabled: state.stageScheduleEnabled });
    if (errors.length) { showFeedback(root, errors[0].message, 'error'); return false; }
    syncStateFromForm(root, state);
    return true;
}

function presetPayload(state, extra = {}) {
    return {
        ...extra,
        datasets: state.rows,
        defaults: state.defaults,
        stage_schedule_enabled: state.stageScheduleEnabled,
        stage_schedule: state.stageSchedule,
    };
}

function syncStateFromForm(root, state) {
    state.defaults = collectDatasetFields(root.querySelector('[data-dataset-defaults]'), state.defaults);
    state.rows = collectDatasetRows(root);
    syncLegacyDatasetState(state);
}

function markDirty(root, state) {
    if (state.dirty) return;
    state.dirty = true;
    updateDirtyUi(root, state);
    disableDatasetPreviews(root);
    syncLegacyDatasetState(state);
}

function updateDirtyUi(root, state) {
    state.ui ||= createDatasetEditorBindings(root);
    renderDatasetDirtyState(state.ui, state);
}

function updateContextUi(root, state) {
    const path = root.querySelector('[data-dataset-path]');
    if (path) { path.textContent = state.datasetConfig || '未关联配置'; path.title = state.datasetConfig || ''; }
    const linkState = root.querySelector('[data-dataset-link-state]');
    if (linkState) {
        linkState.dataset.state = state.datasetConfig ? 'saved' : 'unsaved';
        const text = linkState.querySelector('b');
        if (text) text.textContent = state.datasetConfig ? '已关联训练配置' : '尚未保存';
    }
    updateDirtyUi(root, state);
}

function refreshRows(root, state) {
    const rows = root.querySelector('[data-dataset-rows]');
    if (!rows) return;
    rows.innerHTML = state.rows.map((row, index) => renderDatasetRow(row, index, state.defaults, { readonly: state.readonly, canPreview: Boolean(state.selectedFile) && !state.dirty, totalRows: state.rows.length })).join('');
    bindRowActions(root, state);
    updateDatasetRowSummaries(root);
}

function refreshEditor(root, state) {
    const panel = root.querySelector('.dragon-dataset-editor-panel');
    if (!panel) return;
    panel.innerHTML = renderEditorPanel(state);
    bindForm(root, state);
    bindWorkspace(root, state);
    const title = root.querySelector('[data-savebar-title]');
    if (title) title.textContent = state.selectedFile ? shortName(state.selectedFile) : '未命名数据集预设';
    updateContextUi(root, state);
}

function refreshPresetList(root, state) {
    const list = root.querySelector('[data-preset-list]');
    if (!list) return;
    list.innerHTML = renderPresetGroups(state);
    bindPresetButtons(root, state);
}

function confirmDiscard(state, action) {
    return !state.dirty || window.confirm(`当前数据集有未保存修改。${action}会丢弃这些修改，是否继续？`);
}

function openTaggingWorkspace(root, state) {
    const datasetFile = state.selectedFile || state.datasetConfig;
    if (!datasetFile) {
        showFeedback(root, '请先选择或保存一个数据集预设', 'error');
        return;
    }
    if (!shouldLeaveEditor(state)) return;
    writeTaggingPrefill({
        dataset_file: datasetFile,
        dataset_index: state.previewIndex,
        source: 'source',
    });
    window.location.hash = '#page/captioning';
}

function shouldLeaveEditor(state) {
    const allowed = confirmDiscard(state, '离开页面');
    if (allowed) {
        state.dirty = false;
        clearLegacyStageDraftMarkers(state);
    }
    return allowed;
}

function cleanupEditor(state) {
    if (state.presetSearchTimer) window.clearTimeout(state.presetSearchTimer);
    state.presetSearchTimer = null;
    if (state.root) disposePresetDragRecovery(state.root, state);
    if (state.beforeUnload) window.removeEventListener('beforeunload', state.beforeUnload);
    const dialog = document.getElementById('stage-resolution-dialog');
    if (dialog && state.stageDialogHandler) dialog.removeEventListener('close', state.stageDialogHandler);
    if (state.stageScheduleHandler) window.removeEventListener('anima-stage-schedule-change', state.stageScheduleHandler);
    state.previewController?.dispose();
    state.previewController = null;
    state.rowPathCleanup?.();
    state.rowPathCleanup = null;
    state.ui = null;
    if (activeEditor === state) activeEditor = null;
}

function setLoading(root, state, loading, message = '') {
    state.loading = loading;
    root.querySelectorAll('button').forEach((button) => {
        if (loading && button.dataset.wasDisabled == null) button.dataset.wasDisabled = String(button.disabled);
        if (loading) button.disabled = true;
        else if (button.dataset.wasDisabled != null) { button.disabled = button.dataset.wasDisabled === 'true'; delete button.dataset.wasDisabled; }
    });
    if (message) showFeedback(root, message, 'info');
}

function showFeedback(root, message, tone) {
    const feedback = root.querySelector('[data-dataset-feedback]');
    if (!feedback) return;
    feedback.textContent = message;
    feedback.dataset.tone = tone;
    feedback.classList.add('dragon-config-feedback-visible');
}

function shortName(file) {
    return String(file || '').split(/[\\/]/).filter(Boolean).pop() || 'dataset.toml';
}
