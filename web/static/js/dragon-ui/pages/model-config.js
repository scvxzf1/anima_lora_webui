/* Ordered model configuration library backed by /api/settings/model-configs. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260814v43';
import {
    bindModelConfigDrag,
    disposeModelConfigDrag,
    renderModelConfigList,
} from './model-config-library.js?v=dragon-ui-20260817v2';
import {
    cleanModelGroup,
    cleanModelItem,
    cloneModelGroups,
    cloneModelItems,
    familyLabel,
    modelGroupForItem,
    moveModelGroup,
    moveModelItemInGroups,
    normalizeModelGroups,
    placeModelGroup,
    placeModelItem,
    removeModelGroup,
    serializeModelState,
    uniqueDraftName,
    validateModelGroups,
    validateModelItems,
} from './model-config-state.js?v=dragon-ui-20260824-zimage-v1';

const api = createApiClient();
const PATH_FIELDS = [
    ['pretrained_model_name_or_path', '基础 DiT 模型', 'models/diffusion_models/model.safetensors'],
    ['qwen3', 'Qwen3 文本编码器', 'models/text_encoders/qwen.safetensors'],
    ['vae', 'VAE 模型', 'models/vae/vae.safetensors'],
];

export async function loadModelConfig() {
    let payload;
    try {
        payload = await fetchLibrary();
    } catch (error) {
        return renderError(error.message || '读取全局模型配置失败');
    }

    const state = createState(payload);
    return {
        html: renderPage(state),
        onMount: (root) => bindPage(root, state),
        beforeLeave: () => confirmLeave(state),
        onUnmount: () => cleanupPage(state),
    };
}

async function fetchLibrary() {
    const payload = await api('/api/settings/model-configs');
    if (payload?.ok === false) {
        const error = new Error(payload.error || '读取全局模型配置失败');
        error.status = Number(payload.status || payload.status_code || 0);
        throw error;
    }
    return payload || {};
}

function createState(payload) {
    const items = Array.isArray(payload.items) ? payload.items.map(cleanModelItem) : [];
    const groups = normalizeModelGroups(Array.isArray(payload.groups) ? payload.groups : [], items);
    const defaultId = String(payload.default_id || items[0]?.id || '');
    const state = {
        items,
        savedItems: cloneModelItems(items),
        groups,
        savedGroups: cloneModelGroups(groups),
        defaultId,
        savedDefaultId: defaultId,
        revision: String(payload.revision || ''),
        migrated: Boolean(payload.migrated),
        groupsMigrated: Boolean(payload.groups_migrated),
        selectedId: String(defaultId || items[0]?.id || ''),
        search: '',
        dirty: false,
        saving: false,
        beforeUnload: null,
    };
    state.savedSignature = serializeModelState(state.savedItems, state.savedDefaultId, state.savedGroups);
    return state;
}

function renderPage(state) {
    const selected = selectedItem(state);
    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-model-config-page">
            <header class="dragon-tool-hero dragon-reveal">
                <div class="dragon-tool-hero-copy">
                    <span class="dragon-eyebrow">模型与系统</span>
                    <h1>全局模型配置</h1>
                    <p>集中管理训练与推理共用的模型组合。默认项会同步到全局模型路径。</p>
                </div>
                <dl class="dragon-tool-summary" aria-label="模型配置摘要">
                    <div><dt>配置数量</dt><dd data-model-count>${state.items.length}</dd></div>
                    <div><dt>分组数量</dt><dd data-model-group-count>${state.groups.length}</dd></div>
                    <div><dt>当前默认</dt><dd data-model-default-name>${escapeHtml(defaultItem(state)?.name || '未设置')}</dd></div>
                </dl>
            </header>

            <div class="dragon-model-config-layout dragon-reveal" data-stagger="1">
                <aside class="dragon-model-config-sidebar" aria-label="模型配置库">
                    <div class="dragon-model-config-sidebar-head">
                        <div><span class="dragon-eyebrow">配置库</span><h2>模型组合</h2></div>
                        <div class="dragon-model-config-sidebar-actions">
                            <button class="dragon-icon-button" type="button" data-model-action="add-group" aria-label="新建模型配置分组" title="新建分组">${renderIcon('folder')}</button>
                            <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-model-action="add">${renderIcon('filePlus', 'dragon-btn-icon')}<span>新建配置</span></button>
                        </div>
                    </div>
                    <label class="dragon-compact-field dragon-model-config-search">
                        <span class="visually-hidden">搜索模型配置</span>
                        <input class="dragon-input" type="search" name="model_config_search" autocomplete="off" data-model-search placeholder="搜索名称、模型族或路径…">
                    </label>
                    <div class="dragon-model-config-list" data-model-list role="region" aria-label="模型配置分组列表">${renderModelConfigList(state)}</div>
                </aside>

                <form class="dragon-model-config-editor" data-model-form novalidate>
                    <div class="dragon-model-config-editor-body" data-model-editor-body>
                        ${renderEditor(selected, state)}
                    </div>
                    <div class="dragon-savebar dragon-model-config-savebar" data-dirty="false">
                        <div class="dragon-savebar-status">
                            <strong data-model-dirty-label>所有修改已保存</strong>
                            <span data-model-feedback role="status" aria-live="polite">${payloadHint(state)}</span>
                        </div>
                        <div class="dragon-savebar-actions">
                            <button class="dragon-btn dragon-btn-secondary" type="button" data-model-action="revert" disabled>还原修改</button>
                            <button class="dragon-btn dragon-btn-secondary" type="button" data-model-action="default" ${selected?.id === state.defaultId ? 'disabled' : ''}>设为默认</button>
                            <button class="dragon-btn dragon-btn-danger" type="button" data-model-action="remove" ${state.items.length <= 1 || selected?.id === state.defaultId ? 'disabled' : ''}>删除配置</button>
                            <button class="dragon-btn dragon-btn-primary" type="button" data-model-action="save">保存模型配置</button>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    `;
}

function payloadHint(state) {
    if (state.migrated) return '已载入旧版全局模型路径；保存一次后会建立独立模型配置库。';
    if (state.groupsMigrated) return '已为旧版模型配置库建立“未分组”；保存一次后持久化。';
    return state.items.length ? '更改会在保存后应用到训练与推理。' : '请先新建模型配置。';
}

function renderEditor(item, state) {
    if (!item) return '<div class="dragon-empty-state"><p>请新建 1 个模型配置。</p></div>';
    return `
        <section class="dragon-model-editor-section">
            <header class="dragon-model-editor-header">
                <div><span class="dragon-eyebrow">当前配置</span><h2 data-model-editor-title>${escapeHtml(item.name || '未命名配置')}</h2></div>
                <span class="dragon-inline-badge" data-model-family-badge>${familyLabel(item.model_family)}</span>
            </header>
            <p class="dragon-model-editor-desc">配置名称、模型格式与 3 个必需组件路径。路径支持项目相对路径、绝对路径和环境变量。</p>
            <div class="dragon-field-grid-2 dragon-model-identity-fields">
                ${textField('name', '配置名称', item.name, '例如：Krea-2 主模型')}
                ${selectField('model_family', '模型格式', item.model_family, [['anima', 'Anima'], ['krea2_raw', 'Krea-2'], ['z_image', 'Z-Image']])}
            </div>
        </section>
        <section class="dragon-model-editor-section">
            <header class="dragon-model-editor-header"><div><span class="dragon-eyebrow">模型组件</span><h2>基础权重路径</h2></div></header>
            <div class="dragon-model-path-fields">${PATH_FIELDS.map(([key, label, placeholder]) => textField(key, label, item[key], placeholder)).join('')}</div>
        </section>
        <input type="hidden" data-model-field="id" value="${escapeAttribute(item.id)}">
        <input type="hidden" data-model-revision value="${escapeAttribute(state.revision)}">
    `;
}

function textField(key, label, value, placeholder) {
    return `<label class="dragon-field" data-model-field-wrap="${key}"><span class="dragon-field-label-text">${label}</span><input class="dragon-input" type="text" name="model_config_${key}" autocomplete="off" spellcheck="false" data-model-field="${key}" value="${escapeAttribute(value)}" placeholder="${escapeAttribute(placeholder)}"><small class="dragon-field-error" data-model-error="${key}" aria-live="polite"></small></label>`;
}

function selectField(key, label, value, options) {
    return `<label class="dragon-field" data-model-field-wrap="${key}"><span class="dragon-field-label-text">${label}</span><select class="dragon-select" name="model_config_${key}" autocomplete="off" data-model-field="${key}">${options.map(([option, text]) => `<option value="${option}" ${option === value ? 'selected' : ''}>${text}</option>`).join('')}</select><small class="dragon-field-error" data-model-error="${key}" aria-live="polite"></small></label>`;
}

function bindPage(root, state) {
    state.root = root;
    state.beforeUnload = (event) => {
        if (!state.dirty) return;
        event.preventDefault();
        event.returnValue = '';
    };
    window.addEventListener('beforeunload', state.beforeUnload);

    bindDynamicEvents(root, state);
    root.querySelector('[data-model-search]')?.addEventListener('input', (event) => {
        state.search = event.target.value || '';
        refreshList(root, state);
    });
    root.querySelector('[data-model-action="add"]')?.addEventListener('click', () => addItem(root, state));
    root.querySelector('[data-model-action="add-group"]')?.addEventListener('click', () => addGroup(root, state));
    root.querySelector('[data-model-action="remove"]')?.addEventListener('click', () => removeItem(root, state));
    root.querySelector('[data-model-action="default"]')?.addEventListener('click', () => setDefault(root, state));
    root.querySelector('[data-model-action="revert"]')?.addEventListener('click', () => restoreSaved(root, state));
    root.querySelector('[data-model-action="save"]')?.addEventListener('click', () => saveLibrary(root, state));
    bindLibraryEvents(root, state);
    updatePageState(root, state);
}

function bindDynamicEvents(root, state) {
    root.querySelectorAll('.dragon-model-config-list-item[data-model-id]').forEach((button) => {
        if (button.dataset.bound === 'true') return;
        button.dataset.bound = 'true';
        button.addEventListener('click', () => {
            if (performance.now() < (state.modelSuppressClickUntil || 0)) return;
            switchItem(root, state, button.dataset.modelId);
        });
    });
    root.querySelectorAll('[data-model-field]').forEach((field) => {
        if (field.type === 'hidden' || field.dataset.bound === 'true') return;
        field.dataset.bound = 'true';
        const update = () => {
            syncEditor(root, state);
            clearFieldError(root, field.dataset.modelField);
            markDirty(root, state);
        };
        field.addEventListener('input', update);
        field.addEventListener('change', update);
    });
}

function bindLibraryEvents(root, state) {
    root.querySelectorAll('[data-model-item-move]').forEach((button) => {
        button.addEventListener('click', () => moveItem(root, state, button.dataset.modelId, Number(button.dataset.modelItemMove)));
    });
    root.querySelectorAll('[data-model-group-move]').forEach((button) => {
        button.addEventListener('click', () => moveGroup(root, state, button.dataset.groupId, Number(button.dataset.modelGroupMove)));
    });
    root.querySelectorAll('[data-model-group-action]').forEach((button) => {
        button.addEventListener('click', () => {
            const group = state.groups.find((candidate) => candidate.id === button.dataset.groupId);
            if (button.dataset.modelGroupAction === 'rename') renameGroup(root, state, group);
            if (button.dataset.modelGroupAction === 'delete') deleteGroup(root, state, group);
        });
    });
    bindModelConfigDrag(root, state, {
        onPlaceItem: (itemId, groupId, anchorId, position) => placeItemAt(root, state, itemId, groupId, anchorId, position),
        onPlaceGroup: (groupId, index) => placeGroupAt(root, state, groupId, index),
    });
}

function switchItem(root, state, itemId) {
    if (!itemId || itemId === state.selectedId) return;
    syncEditor(root, state);
    state.selectedId = itemId;
    refreshWorkspace(root, state);
}

function addItem(root, state) {
    syncEditor(root, state);
    const source = selectedItem(state) || {};
    const id = `model-${Date.now().toString(36)}`;
    state.items.push(cleanModelItem({ ...source, id, name: uniqueDraftName(state.items) }));
    const targetGroup = modelGroupForItem(state.groups, state.selectedId) || state.groups[0];
    targetGroup?.item_ids.push(id);
    state.selectedId = id;
    state.search = '';
    const search = root.querySelector('[data-model-search]');
    if (search) search.value = '';
    markDirty(root, state);
    refreshWorkspace(root, state);
    root.querySelector('[data-model-field="name"]')?.select();
}

function addGroup(root, state) {
    syncEditor(root, state);
    const label = window.prompt('请输入新的模型配置分组名称：', '新建分组');
    if (label === null) return;
    const cleaned = String(label).trim();
    if (!cleaned) return showFeedback(root, '分组名称不能为空', 'error');
    if (state.groups.some((group) => group.label.toLocaleLowerCase() === cleaned.toLocaleLowerCase())) {
        return showFeedback(root, `分组名称“${cleaned}”已存在`, 'error');
    }
    state.groups.push(cleanModelGroup({ id: uniqueGroupId(state), label: cleaned, item_ids: [] }));
    markDirty(root, state);
    refreshList(root, state);
    showFeedback(root, '分组已创建，保存后生效', 'info');
}

function renameGroup(root, state, group) {
    if (!group) return;
    const label = window.prompt('请输入新的模型配置分组名称：', group.label);
    if (label === null) return;
    const cleaned = String(label).trim();
    if (!cleaned) return showFeedback(root, '分组名称不能为空', 'error');
    if (state.groups.some((candidate) => candidate.id !== group.id && candidate.label.toLocaleLowerCase() === cleaned.toLocaleLowerCase())) {
        return showFeedback(root, `分组名称“${cleaned}”已存在`, 'error');
    }
    group.label = cleaned;
    markDirty(root, state);
    refreshList(root, state);
}

function deleteGroup(root, state, group) {
    if (!group || state.groups.length <= 1) return showFeedback(root, '至少需要保留 1 个模型配置分组', 'error');
    const count = group.item_ids.length;
    const detail = count ? `其中 ${count} 个模型配置会移到相邻分组。` : '该分组当前为空。';
    if (!window.confirm(`确认删除分组“${group.label}”吗？\n${detail}\n不会删除模型配置或模型文件。`)) return;
    state.groups = removeModelGroup(state.groups, group.id);
    markDirty(root, state);
    refreshList(root, state);
}

function removeItem(root, state) {
    syncEditor(root, state);
    const item = selectedItem(state);
    if (!item || state.items.length <= 1) return showFeedback(root, '至少需要保留 1 个模型配置', 'error');
    if (item.id === state.defaultId) return showFeedback(root, '默认配置不能删除，请先将其他配置设为默认', 'error');
    if (!window.confirm(`确认删除“${item.name || '未命名配置'}”吗？\n只会删除这条配置，不会删除任何模型文件。`)) return;
    const index = state.items.findIndex((candidate) => candidate.id === item.id);
    state.items = state.items.filter((candidate) => candidate.id !== item.id);
    state.groups.forEach((group) => { group.item_ids = group.item_ids.filter((itemId) => itemId !== item.id); });
    state.selectedId = state.items[Math.min(index, state.items.length - 1)]?.id || '';
    markDirty(root, state);
    refreshWorkspace(root, state);
}

function setDefault(root, state) {
    syncEditor(root, state);
    if (!selectedItem(state)) return;
    state.defaultId = state.selectedId;
    markDirty(root, state);
    refreshWorkspace(root, state);
    showFeedback(root, '已标记为默认，保存后同步到全局模型路径', 'info');
}

function moveItem(root, state, itemId, offset) {
    syncEditor(root, state);
    const next = moveModelItemInGroups(state.groups, itemId, offset);
    if (serializeGroups(next) === serializeGroups(state.groups)) return;
    state.groups = next;
    state.selectedId = itemId;
    markDirty(root, state);
    refreshList(root, state);
}

function moveGroup(root, state, groupId, offset) {
    syncEditor(root, state);
    const next = moveModelGroup(state.groups, groupId, offset);
    if (serializeGroups(next) === serializeGroups(state.groups)) return;
    state.groups = next;
    markDirty(root, state);
    refreshList(root, state);
}

function placeItemAt(root, state, itemId, groupId, anchorId, position) {
    syncEditor(root, state);
    const next = placeModelItem(state.groups, itemId, groupId, anchorId, position);
    if (serializeGroups(next) === serializeGroups(state.groups)) return;
    state.groups = next;
    state.selectedId = itemId;
    markDirty(root, state);
    refreshList(root, state);
    showFeedback(root, '模型配置位置已更新，保存后生效', 'info');
}

function placeGroupAt(root, state, groupId, index) {
    syncEditor(root, state);
    const next = placeModelGroup(state.groups, groupId, index);
    if (serializeGroups(next) === serializeGroups(state.groups)) return;
    state.groups = next;
    markDirty(root, state);
    refreshList(root, state);
}

function restoreSaved(root, state) {
    if (!state.dirty || !window.confirm('确认还原本页的所有未保存修改吗？')) return;
    state.items = cloneModelItems(state.savedItems);
    state.groups = cloneModelGroups(state.savedGroups);
    state.defaultId = state.savedDefaultId;
    state.selectedId = state.items.some((item) => item.id === state.selectedId) ? state.selectedId : state.defaultId;
    state.search = '';
    const search = root.querySelector('[data-model-search]');
    if (search) search.value = '';
    state.dirty = false;
    refreshWorkspace(root, state);
    showFeedback(root, '已还原到上次保存的版本', 'info');
}

function refreshWorkspace(root, state) {
    refreshList(root, state);
    const editorBody = root.querySelector('[data-model-editor-body]');
    if (editorBody) editorBody.innerHTML = renderEditor(selectedItem(state), state);
    bindDynamicEvents(root, state);
    updatePageState(root, state);
}

function refreshList(root, state) {
    const list = root.querySelector('[data-model-list]');
    if (list) list.innerHTML = renderModelConfigList(state);
    bindDynamicEvents(root, state);
    bindLibraryEvents(root, state);
    updatePageState(root, state);
}

function syncEditor(root, state) {
    const item = selectedItem(state);
    if (!item) return;
    root.querySelectorAll('[data-model-field]').forEach((field) => {
        item[field.dataset.modelField] = String(field.value || '').trim();
    });
    const title = root.querySelector('[data-model-editor-title]');
    if (title) title.textContent = item.name || '未命名配置';
    const badge = root.querySelector('[data-model-family-badge]');
    if (badge) badge.textContent = familyLabel(item.model_family);
}

function markDirty(root, state) {
    state.dirty = serializeModelState(state.items, state.defaultId, state.groups) !== state.savedSignature;
    updatePageState(root, state);
}

function updatePageState(root, state) {
    const selected = selectedItem(state);
    const savebar = root.querySelector('.dragon-model-config-savebar');
    if (savebar) savebar.dataset.dirty = String(state.dirty);
    setText(root, '[data-model-dirty-label]', state.dirty ? '有未保存修改' : '所有修改已保存');
    setText(root, '[data-model-count]', state.items.length);
    setText(root, '[data-model-group-count]', state.groups.length);
    setText(root, '[data-model-default-name]', defaultItem(state)?.name || '未设置');
    setDisabled(root, '[data-model-action="revert"]', !state.dirty || state.saving);
    setDisabled(root, '[data-model-action="default"]', !selected || selected.id === state.defaultId || state.saving);
    setDisabled(root, '[data-model-action="remove"]', !selected || state.items.length <= 1 || selected.id === state.defaultId || state.saving);
    setDisabled(root, '[data-model-action="save"]', state.saving || (!state.dirty && !migrationPending(state)));
}

async function saveLibrary(root, state) {
    if (state.saving) return;
    syncEditor(root, state);
    clearAllErrors(root);
    const validation = validateModelItems(state.items, state.defaultId);
    if (validation) {
        state.selectedId = validation.itemId || state.selectedId;
        refreshWorkspace(root, state);
        showFieldError(root, validation.field, validation.message);
        return showFeedback(root, validation.message, 'error');
    }
    const groupValidation = validateModelGroups(state.groups, state.items);
    if (groupValidation) return showFeedback(root, groupValidation.message, 'error');
    state.saving = true;
    updatePageState(root, state);
    showFeedback(root, '正在保存…', 'info');
    try {
        const payload = await api('/api/settings/model-configs', {
            method: 'PUT',
            body: JSON.stringify({
                revision: state.revision,
                default_id: state.defaultId,
                items: state.items.map(cleanModelItem),
                groups: state.groups.map(cleanModelGroup),
            }),
        });
        if (payload?.ok === false) {
            const error = new Error(payload.error || '保存模型配置失败');
            error.status = Number(payload.status || payload.status_code || 0);
            throw error;
        }
        applyPayload(state, payload);
        refreshWorkspace(root, state);
        showFeedback(root, payload.message || '模型配置已保存', 'success');
    } catch (error) {
        if (error.status === 409) {
            await recoverConflict(root, state, error);
        } else {
            showFeedback(root, `${error.message || '保存模型配置失败'}。请检查配置后重试。`, 'error');
        }
    } finally {
        state.saving = false;
        updatePageState(root, state);
    }
}

async function recoverConflict(root, state, error) {
    if (!window.confirm(`${error.message || '模型配置已在其他页面更新。'}\n是否重新载入服务器版本？本页未保存修改会丢失。`)) {
        showFeedback(root, '保留了本页草稿；请复制需要的路径后再重新载入', 'error');
        return;
    }
    try {
        const payload = await fetchLibrary();
        applyPayload(state, payload);
        refreshWorkspace(root, state);
        showFeedback(root, '已载入服务器上的最新模型配置', 'info');
    } catch (reloadError) {
        showFeedback(root, `${reloadError.message || '重新载入失败'}。请刷新页面后重试。`, 'error');
    }
}

function applyPayload(state, payload) {
    state.items = Array.isArray(payload.items) ? payload.items.map(cleanModelItem) : [];
    state.savedItems = cloneModelItems(state.items);
    state.groups = normalizeModelGroups(Array.isArray(payload.groups) ? payload.groups : [], state.items);
    state.savedGroups = cloneModelGroups(state.groups);
    state.defaultId = String(payload.default_id || state.items[0]?.id || '');
    state.savedDefaultId = state.defaultId;
    state.revision = String(payload.revision || '');
    state.migrated = Boolean(payload.migrated);
    state.groupsMigrated = Boolean(payload.groups_migrated);
    state.selectedId = state.items.some((item) => item.id === state.selectedId) ? state.selectedId : state.defaultId;
    state.savedSignature = serializeModelState(state.savedItems, state.savedDefaultId, state.savedGroups);
    state.dirty = false;
}

function confirmLeave(state) {
    return !state.dirty || window.confirm('全局模型配置有未保存修改，离开会丢失这些修改。是否继续？');
}

function migrationPending(state) {
    return state.migrated || state.groupsMigrated;
}

function cleanupPage(state) {
    if (state.beforeUnload) window.removeEventListener('beforeunload', state.beforeUnload);
    if (state.root) disposeModelConfigDrag(state.root, state);
}

function uniqueGroupId(state) {
    const existing = new Set(state.groups.map((group) => group.id));
    const base = `model-group-${Date.now().toString(36)}`;
    let candidate = base;
    let suffix = 1;
    while (existing.has(candidate)) candidate = `${base}-${suffix++}`;
    return candidate;
}

function serializeGroups(groups) {
    return JSON.stringify(groups.map(cleanModelGroup));
}

function selectedItem(state) {
    return state.items.find((item) => item.id === state.selectedId) || state.items[0] || null;
}

function defaultItem(state) {
    return state.items.find((item) => item.id === state.defaultId) || null;
}

function showFieldError(root, field, message) {
    if (!field) return;
    const error = root.querySelector(`[data-model-error="${field}"]`);
    const input = root.querySelector(`[data-model-field="${field}"]`);
    if (error) error.textContent = message;
    if (input) {
        input.setAttribute('aria-invalid', 'true');
        input.focus();
    }
}

function clearFieldError(root, field) {
    const error = root.querySelector(`[data-model-error="${field}"]`);
    const input = root.querySelector(`[data-model-field="${field}"]`);
    if (error) error.textContent = '';
    input?.removeAttribute('aria-invalid');
}

function clearAllErrors(root) {
    root.querySelectorAll('[data-model-error]').forEach((error) => { error.textContent = ''; });
    root.querySelectorAll('[aria-invalid="true"]').forEach((input) => input.removeAttribute('aria-invalid'));
}

function showFeedback(root, message, tone = '') {
    const feedback = root.querySelector('[data-model-feedback]');
    if (!feedback) return;
    feedback.textContent = message;
    feedback.dataset.tone = tone;
}

function setText(root, selector, value) {
    const element = root.querySelector(selector);
    if (element) element.textContent = String(value ?? '');
}

function setDisabled(root, selector, disabled) {
    const element = root.querySelector(selector);
    if (element) element.disabled = Boolean(disabled);
}

function renderError(message) {
    return `<div class="dragon-page dragon-tool-page"><header class="dragon-tool-hero"><div><span class="dragon-eyebrow">模型与系统</span><h1>全局模型配置</h1><p>无法读取模型配置库。</p></div></header><div class="dragon-empty-state"><p>${escapeHtml(message)}</p><p>请检查服务连接后刷新页面。</p></div></div>`;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
