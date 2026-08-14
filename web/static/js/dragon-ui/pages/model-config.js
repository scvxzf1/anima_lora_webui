/* Ordered model configuration library backed by /api/settings/model-configs. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import {
    cleanModelItem,
    cloneModelItems,
    familyLabel,
    filterModelItems,
    moveModelItem,
    serializeModelState,
    uniqueDraftName,
    validateModelItems,
} from './model-config-state.js?v=dragon-ui-20260814v43';

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
    const defaultId = String(payload.default_id || items[0]?.id || '');
    const state = {
        items,
        savedItems: cloneModelItems(items),
        defaultId,
        savedDefaultId: defaultId,
        revision: String(payload.revision || ''),
        migrated: Boolean(payload.migrated),
        selectedId: String(defaultId || items[0]?.id || ''),
        search: '',
        dirty: false,
        saving: false,
        beforeUnload: null,
    };
    state.savedSignature = serializeModelState(state.savedItems, state.savedDefaultId);
    return state;
}

function renderPage(state) {
    const selected = selectedItem(state);
    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-model-config-page">
            <header class="dragon-tool-hero dragon-reveal">
                <div>
                    <span class="dragon-eyebrow">模型与系统</span>
                    <h1>全局模型配置</h1>
                    <p>集中管理训练与推理共用的模型组合。默认项会同步到全局模型路径。</p>
                </div>
                <dl class="dragon-tool-summary" aria-label="模型配置摘要">
                    <div><dt>配置数量</dt><dd data-model-count>${state.items.length}</dd></div>
                    <div><dt>当前默认</dt><dd data-model-default-name>${escapeHtml(defaultItem(state)?.name || '未设置')}</dd></div>
                </dl>
            </header>

            <div class="dragon-model-config-layout dragon-reveal" data-stagger="1">
                <aside class="dragon-model-config-sidebar" aria-label="模型配置库">
                    <div class="dragon-model-config-sidebar-head">
                        <div><span class="dragon-eyebrow">配置库</span><h2>模型组合</h2></div>
                        <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-model-action="add">新建配置</button>
                    </div>
                    <label class="dragon-compact-field dragon-model-config-search">
                        <span class="visually-hidden">搜索模型配置</span>
                        <input class="dragon-input" type="search" name="model_config_search" autocomplete="off" data-model-search placeholder="搜索名称、模型族或路径…">
                    </label>
                    <div class="dragon-model-config-list" data-model-list>${renderList(state)}</div>
                </aside>

                <form class="dragon-model-config-editor" data-model-form novalidate>
                    ${renderEditor(selected, state)}
                </form>
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
        </div>
    `;
}

function payloadHint(state) {
    if (state.migrated) return '已载入旧版全局模型路径；保存一次后会建立独立模型配置库。';
    return state.items.length ? '更改会在保存后应用到训练与推理。' : '请先新建模型配置。';
}

function renderList(state) {
    const visible = filterModelItems(state.items, state.search);
    if (!visible.length) return '<div class="dragon-model-config-list-empty">没有匹配的模型配置</div>';
    return visible.map((item) => {
        const index = state.items.findIndex((candidate) => candidate.id === item.id);
        const path = item.pretrained_model_name_or_path || '基础模型路径未填写';
        return `
            <div class="dragon-model-config-row" data-model-row="${escapeAttribute(item.id)}">
                <button class="dragon-model-config-list-item" type="button" data-model-id="${escapeAttribute(item.id)}" data-active="${item.id === state.selectedId}" aria-pressed="${item.id === state.selectedId}" ${item.id === state.selectedId ? 'aria-current="true"' : ''}>
                    <span><strong>${escapeHtml(item.name || '未命名配置')}</strong><small>${familyLabel(item.model_family)} · ${escapeHtml(path)}</small></span>
                    ${item.id === state.defaultId ? '<em>默认</em>' : ''}
                </button>
                <div class="dragon-model-config-row-actions" aria-label="调整配置顺序">
                    ${orderButton(item, -1, index === 0, '上移')}
                    ${orderButton(item, 1, index === state.items.length - 1, '下移')}
                </div>
            </div>
        `;
    }).join('');
}

function orderButton(item, offset, disabled, label) {
    const glyph = offset < 0 ? '↑' : '↓';
    return `<button class="dragon-icon-btn" type="button" data-model-move="${offset}" data-model-id="${escapeAttribute(item.id)}" aria-label="${label} ${escapeAttribute(item.name)}" title="${label}" ${disabled ? 'disabled' : ''}>${glyph}</button>`;
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
                ${selectField('model_family', '模型格式', item.model_family, [['anima', 'Anima'], ['krea2_raw', 'Krea-2']])}
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
    root.querySelector('[data-model-action="remove"]')?.addEventListener('click', () => removeItem(root, state));
    root.querySelector('[data-model-action="default"]')?.addEventListener('click', () => setDefault(root, state));
    root.querySelector('[data-model-action="revert"]')?.addEventListener('click', () => restoreSaved(root, state));
    root.querySelector('[data-model-action="save"]')?.addEventListener('click', () => saveLibrary(root, state));
    updatePageState(root, state);
}

function bindDynamicEvents(root, state) {
    root.querySelectorAll('[data-model-id]').forEach((button) => {
        if (button.dataset.bound === 'true') return;
        button.dataset.bound = 'true';
        button.addEventListener('click', () => switchItem(root, state, button.dataset.modelId));
    });
    root.querySelectorAll('[data-model-move]').forEach((button) => {
        if (button.dataset.bound === 'true') return;
        button.dataset.bound = 'true';
        button.addEventListener('click', () => moveItem(root, state, button.dataset.modelId, Number(button.dataset.modelMove)));
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
    state.selectedId = id;
    state.search = '';
    const search = root.querySelector('[data-model-search]');
    if (search) search.value = '';
    markDirty(root, state);
    refreshWorkspace(root, state);
    root.querySelector('[data-model-field="name"]')?.select();
}

function removeItem(root, state) {
    syncEditor(root, state);
    const item = selectedItem(state);
    if (!item || state.items.length <= 1) return showFeedback(root, '至少需要保留 1 个模型配置', 'error');
    if (item.id === state.defaultId) return showFeedback(root, '默认配置不能删除，请先将其他配置设为默认', 'error');
    if (!window.confirm(`确认删除“${item.name || '未命名配置'}”吗？\n只会删除这条配置，不会删除任何模型文件。`)) return;
    const index = state.items.findIndex((candidate) => candidate.id === item.id);
    state.items = state.items.filter((candidate) => candidate.id !== item.id);
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
    const next = moveModelItem(state.items, itemId, offset);
    if (next.every((item, index) => item.id === state.items[index]?.id)) return;
    state.items = next;
    state.selectedId = itemId;
    markDirty(root, state);
    refreshList(root, state);
}

function restoreSaved(root, state) {
    if (!state.dirty || !window.confirm('确认还原本页的所有未保存修改吗？')) return;
    state.items = cloneModelItems(state.savedItems);
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
    const editor = root.querySelector('[data-model-form]');
    if (editor) editor.innerHTML = renderEditor(selectedItem(state), state);
    bindDynamicEvents(root, state);
    updatePageState(root, state);
}

function refreshList(root, state) {
    const list = root.querySelector('[data-model-list]');
    if (list) list.innerHTML = renderList(state);
    bindDynamicEvents(root, state);
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
    state.dirty = serializeModelState(state.items, state.defaultId) !== state.savedSignature;
    updatePageState(root, state);
}

function updatePageState(root, state) {
    const selected = selectedItem(state);
    const savebar = root.querySelector('.dragon-model-config-savebar');
    if (savebar) savebar.dataset.dirty = String(state.dirty);
    setText(root, '[data-model-dirty-label]', state.dirty ? '有未保存修改' : '所有修改已保存');
    setText(root, '[data-model-count]', state.items.length);
    setText(root, '[data-model-default-name]', defaultItem(state)?.name || '未设置');
    setDisabled(root, '[data-model-action="revert"]', !state.dirty || state.saving);
    setDisabled(root, '[data-model-action="default"]', !selected || selected.id === state.defaultId || state.saving);
    setDisabled(root, '[data-model-action="remove"]', !selected || state.items.length <= 1 || selected.id === state.defaultId || state.saving);
    setDisabled(root, '[data-model-action="save"]', state.saving || (!state.dirty && !state.migrated));
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
    state.saving = true;
    updatePageState(root, state);
    showFeedback(root, '正在保存…', 'info');
    try {
        const payload = await api('/api/settings/model-configs', {
            method: 'PUT',
            body: JSON.stringify({ revision: state.revision, default_id: state.defaultId, items: state.items.map(cleanModelItem) }),
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
    state.defaultId = String(payload.default_id || state.items[0]?.id || '');
    state.savedDefaultId = state.defaultId;
    state.revision = String(payload.revision || '');
    state.migrated = Boolean(payload.migrated);
    state.selectedId = state.items.some((item) => item.id === state.selectedId) ? state.selectedId : state.defaultId;
    state.savedSignature = serializeModelState(state.savedItems, state.savedDefaultId);
    state.dirty = false;
}

function confirmLeave(state) {
    return !state.dirty || window.confirm('全局模型配置有未保存修改，离开会丢失这些修改。是否继续？');
}

function cleanupPage(state) {
    if (state.beforeUnload) window.removeEventListener('beforeunload', state.beforeUnload);
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
