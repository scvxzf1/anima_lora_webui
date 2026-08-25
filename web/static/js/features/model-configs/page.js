import { loadGlobalSettings } from '../anima-app/helpers/global-settings-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { showAppConfirmDialog } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { fetchModelConfigLibrary, saveModelConfigLibrary } from './api.js?v=module-bootstrap-20260809-nf4-v2';
import {
    MODEL_CONFIG_PATH_FIELDS,
    cleanModelConfigItem,
    modelConfigValidationError,
    modelFamilyLabel,
    moveModelConfig,
    moveModelConfigByOffset,
} from './model-config-data.js?v=module-bootstrap-20260824-zimage-v1';

const state = {
    items: [],
    savedItems: [],
    defaultId: '',
    revision: '',
    selectedId: '',
    search: '',
    dirty: false,
    manage: false,
    loaded: false,
    draggingId: '',
};

function selectedItem() {
    return state.items.find((item) => item.id === state.selectedId) || null;
}

function setStatus(text = '', tone = '') {
    const status = document.getElementById('model-config-status');
    if (!status) return;
    status.textContent = text;
    status.className = `model-config-status ${tone}`.trim();
}

function applyLibrary(payload, preferredId = '') {
    state.items = payload.items;
    state.savedItems = payload.items.map((item) => ({ ...item }));
    state.defaultId = payload.defaultId;
    state.revision = payload.revision;
    state.selectedId = state.items.some((item) => item.id === preferredId)
        ? preferredId
        : (state.items.some((item) => item.id === state.selectedId) ? state.selectedId : state.defaultId);
    state.loaded = true;
    state.dirty = false;
}

function readEditorItem() {
    const current = selectedItem() || {};
    const item = {
        ...current,
        name: document.getElementById('model-config-name')?.value || '',
        model_family: document.getElementById('model-config-family')?.value || 'anima',
    };
    for (const field of MODEL_CONFIG_PATH_FIELDS) {
        item[field.key] = document.getElementById(field.inputId)?.value || '';
    }
    return cleanModelConfigItem(item);
}

function writeEditorItem(item) {
    const editor = document.getElementById('model-config-editor-form');
    const empty = document.getElementById('model-config-editor-empty');
    if (editor) editor.hidden = !item;
    if (empty) empty.hidden = Boolean(item);
    if (!item) return;
    document.getElementById('model-config-name').value = item.name;
    document.getElementById('model-config-family').value = item.model_family;
    for (const field of MODEL_CONFIG_PATH_FIELDS) {
        document.getElementById(field.inputId).value = item[field.key] || '';
    }
    const badge = document.getElementById('model-config-editor-family');
    if (badge) badge.textContent = modelFamilyLabel(item.model_family);
    const defaultButton = document.getElementById('btn-model-config-set-default');
    if (defaultButton) {
        const isDefault = item.id === state.defaultId;
        defaultButton.disabled = isDefault;
        defaultButton.textContent = isDefault ? '当前默认' : '设为默认';
    }
}

function createListItem(item, index) {
    const row = document.createElement('div');
    row.className = `model-config-list-row${state.manage ? ' managing' : ''}`;
    row.dataset.modelConfigId = item.id;
    row.draggable = state.manage;

    const handle = document.createElement('span');
    handle.className = 'model-config-drag-handle';
    handle.textContent = '⋮⋮';
    handle.title = '拖动排序';
    handle.setAttribute('aria-hidden', 'true');
    handle.hidden = !state.manage;

    const select = document.createElement('button');
    select.type = 'button';
    select.className = `model-config-list-item${item.id === state.selectedId ? ' active' : ''}`;
    select.setAttribute('aria-pressed', String(item.id === state.selectedId));
    const title = document.createElement('strong');
    title.textContent = item.name;
    const meta = document.createElement('span');
    meta.textContent = modelFamilyLabel(item.model_family);
    const path = document.createElement('small');
    path.textContent = item.pretrained_model_name_or_path || '路径未填写';
    select.append(title, meta, path);
    if (item.id === state.defaultId) {
        const badge = document.createElement('i');
        badge.className = 'model-config-default-badge';
        badge.textContent = '默认';
        select.appendChild(badge);
    }
    select.addEventListener('click', () => selectModelConfig(item.id));

    const actions = document.createElement('div');
    actions.className = 'model-config-row-actions';
    actions.hidden = !state.manage;
    actions.append(
        createMoveButton(item.id, -1, index === 0, '上移', '↑'),
        createMoveButton(item.id, 1, index === state.items.length - 1, '下移', '↓'),
        createDeleteButton(item),
    );
    row.append(handle, select, actions);
    bindRowDragEvents(row, item.id);
    return row;
}

function createMoveButton(itemId, offset, disabled, label, glyph) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'model-config-icon-btn';
    button.textContent = glyph;
    button.title = label;
    button.setAttribute('aria-label', label);
    button.disabled = disabled;
    button.addEventListener('click', () => reorderByOffset(itemId, offset));
    return button;
}

function createDeleteButton(item) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'model-config-icon-btn danger';
    button.textContent = '×';
    button.title = '删除配置';
    button.setAttribute('aria-label', `删除 ${item.name}`);
    button.disabled = state.items.length <= 1 || item.id === state.defaultId;
    button.addEventListener('click', () => deleteModelConfig(item));
    return button;
}

function bindRowDragEvents(row, itemId) {
    row.addEventListener('dragstart', (event) => {
        if (!state.manage) return event.preventDefault();
        state.draggingId = itemId;
        row.classList.add('dragging');
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', itemId);
    });
    row.addEventListener('dragover', (event) => {
        if (!state.draggingId || state.draggingId === itemId) return;
        event.preventDefault();
        clearDropMarkers();
        const position = event.clientY >= row.getBoundingClientRect().top + row.offsetHeight / 2 ? 'after' : 'before';
        row.classList.add(`drop-${position}`);
        row.dataset.dropPosition = position;
    });
    row.addEventListener('drop', (event) => {
        event.preventDefault();
        const position = row.dataset.dropPosition || 'before';
        reorderByDrop(state.draggingId, itemId, position);
    });
    row.addEventListener('dragend', finishDragging);
}

function clearDropMarkers() {
    document.querySelectorAll('.model-config-list-row').forEach((row) => {
        row.classList.remove('drop-before', 'drop-after');
        delete row.dataset.dropPosition;
    });
}

function finishDragging() {
    state.draggingId = '';
    document.querySelectorAll('.model-config-list-row').forEach((row) => row.classList.remove('dragging'));
    clearDropMarkers();
}

function renderList() {
    const list = document.getElementById('model-config-list');
    if (!list) return;
    list.innerHTML = '';
    const query = state.search.trim().toLocaleLowerCase();
    const visible = state.items.filter((item) => (
        !query
        || item.name.toLocaleLowerCase().includes(query)
        || modelFamilyLabel(item.model_family).toLocaleLowerCase().includes(query)
        || item.pretrained_model_name_or_path.toLocaleLowerCase().includes(query)
    ));
    visible.forEach((item) => list.appendChild(createListItem(item, state.items.indexOf(item))));
    if (!visible.length) {
        const empty = document.createElement('p');
        empty.className = 'model-config-list-empty';
        empty.textContent = '没有匹配的模型配置';
        list.appendChild(empty);
    }
}

function renderPage() {
    renderList();
    writeEditorItem(selectedItem());
    const count = document.getElementById('model-config-count');
    if (count) count.textContent = String(state.items.length);
    const defaultName = document.getElementById('model-config-default-name');
    if (defaultName) defaultName.textContent = state.items.find((item) => item.id === state.defaultId)?.name || '-';
    const manage = document.getElementById('btn-model-config-manage');
    if (manage) {
        manage.classList.toggle('active', state.manage);
        manage.setAttribute('aria-pressed', String(state.manage));
        manage.textContent = state.manage ? '完成管理' : '管理';
    }
    document.getElementById('model-config-unsaved').hidden = !state.dirty;
    const hasSelection = Boolean(selectedItem());
    document.getElementById('btn-model-config-save').disabled = !hasSelection;
    document.getElementById('btn-model-config-save-footer').disabled = !hasSelection;
    document.getElementById('btn-model-config-revert').disabled = !hasSelection || !state.dirty;
}

async function confirmDiscardDraft() {
    if (!state.dirty) return true;
    return showAppConfirmDialog({
        title: '放弃未保存修改',
        description: '切换模型配置',
        message: '当前模型配置尚未保存，继续会放弃这些修改。',
        confirmText: '放弃修改',
        cancelText: '继续编辑',
    });
}

function restoreWorkingItems() {
    state.items = state.savedItems.map((item) => ({ ...item }));
    state.dirty = false;
}

async function selectModelConfig(itemId) {
    if (itemId === state.selectedId) return;
    if (!(await confirmDiscardDraft())) return;
    restoreWorkingItems();
    state.selectedId = itemId;
    setStatus();
    renderPage();
}

function uniqueDraftName() {
    const base = '新建模型配置';
    let index = 1;
    let name = base;
    const names = new Set(state.items.map((item) => item.name.toLocaleLowerCase()));
    while (names.has(name.toLocaleLowerCase())) name = `${base} ${++index}`;
    return name;
}

async function createModelConfig() {
    if (!(await confirmDiscardDraft())) return;
    restoreWorkingItems();
    const id = `model-${Date.now().toString(36)}`;
    state.items.push(cleanModelConfigItem({ id, name: uniqueDraftName(), model_family: 'anima' }));
    state.selectedId = id;
    state.search = '';
    state.dirty = true;
    document.getElementById('model-config-search').value = '';
    setStatus('请填写三项模型路径后保存', 'pending');
    renderPage();
    document.getElementById('model-config-name')?.focus();
}

async function persistState(message, preferredId = state.selectedId) {
    const payload = await saveModelConfigLibrary(state);
    applyLibrary(payload, preferredId);
    await loadGlobalSettings();
    setStatus(message || payload.message || '全局模型配置已保存', 'ok');
    renderPage();
}

async function saveSelectedModelConfig({ makeDefault = false } = {}) {
    const item = readEditorItem();
    const error = modelConfigValidationError(item, state.items);
    if (error) return setStatus(error, 'error');
    state.items = state.items.map((candidate) => candidate.id === item.id ? item : candidate);
    if (makeDefault) state.defaultId = item.id;
    try {
        await persistState(makeDefault ? '默认模型配置已更新' : '模型配置已保存', item.id);
    } catch (error_) {
        setStatus(error_.message, 'error');
    }
}

async function deleteModelConfig(item) {
    const confirmed = await showAppConfirmDialog({
        title: '删除模型配置',
        description: item.name,
        message: '只删除这条全局模型配置，不会删除任何模型文件。',
        confirmText: '删除',
        cancelText: '取消',
    });
    if (!confirmed) return;
    const next = state.items.filter((candidate) => candidate.id !== item.id);
    const selectedId = item.id === state.selectedId ? next[0]?.id || '' : state.selectedId;
    state.items = next;
    try {
        await persistState('模型配置已删除', selectedId);
    } catch (error) {
        setStatus(error.message, 'error');
        await loadModelConfigsPage({ force: true });
    }
}

async function saveReorderedItems(items, selectedId = state.selectedId) {
    state.items = items;
    renderPage();
    try {
        await persistState('模型配置顺序已保存', selectedId);
    } catch (error) {
        setStatus(error.message, 'error');
        await loadModelConfigsPage({ force: true });
    }
}

function reorderByOffset(itemId, offset) {
    if (!state.manage || state.dirty) return;
    saveReorderedItems(moveModelConfigByOffset(state.items, itemId, offset), itemId);
}

function reorderByDrop(sourceId, targetId, position) {
    finishDragging();
    if (!state.manage || state.dirty || !sourceId) return;
    saveReorderedItems(moveModelConfig(state.items, sourceId, targetId, position), sourceId);
}

function markEditorDirty() {
    if (!selectedItem()) return;
    state.dirty = true;
    document.getElementById('model-config-unsaved').hidden = false;
    const family = document.getElementById('model-config-family')?.value || 'anima';
    document.getElementById('model-config-editor-family').textContent = modelFamilyLabel(family);
    setStatus('有未保存修改', 'pending');
}

export async function loadModelConfigsPage({ force = false } = {}) {
    if (state.loaded && !force) return renderPage();
    setStatus('正在读取模型配置...', 'pending');
    try {
        const payload = await fetchModelConfigLibrary();
        applyLibrary(payload, state.selectedId);
        setStatus(payload.migrated ? '已载入原全局路径，保存后会建立模型配置库' : '', payload.migrated ? 'pending' : '');
        renderPage();
    } catch (error) {
        setStatus(error.message, 'error');
    }
}

export function bindModelConfigEvents() {
    const root = document.getElementById('tab-model-config');
    if (!root || root.dataset.eventsBound === '1') return;
    root.dataset.eventsBound = '1';
    document.getElementById('btn-model-config-create')?.addEventListener('click', createModelConfig);
    document.getElementById('btn-model-config-manage')?.addEventListener('click', () => {
        state.manage = !state.manage;
        renderPage();
    });
    document.getElementById('btn-model-config-save')?.addEventListener('click', () => saveSelectedModelConfig());
    document.getElementById('btn-model-config-save-footer')?.addEventListener('click', () => saveSelectedModelConfig());
    document.getElementById('btn-model-config-revert')?.addEventListener('click', () => loadModelConfigsPage({ force: true }));
    document.getElementById('btn-model-config-set-default')?.addEventListener('click', () => saveSelectedModelConfig({ makeDefault: true }));
    document.getElementById('model-config-search')?.addEventListener('input', (event) => {
        state.search = event.target.value || '';
        renderList();
    });
    root.querySelectorAll('#model-config-editor-form input, #model-config-editor-form select').forEach((input) => {
        input.addEventListener('input', markEditorDirty);
        input.addEventListener('change', markEditorDirty);
    });
}
