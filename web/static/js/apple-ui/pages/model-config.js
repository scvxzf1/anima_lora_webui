/* Ordered model configuration library backed by /api/settings/model-configs. */

import { createApiClient } from '../../shared/api.js?v=apple-ui-20260812v33';

const api = createApiClient();
const PATH_FIELDS = [
    ['pretrained_model_name_or_path', '基础 DiT 模型', 'models/diffusion_models/model.safetensors'],
    ['qwen3', 'Qwen3 文本编码器', 'models/text_encoders/qwen.safetensors'],
    ['vae', 'VAE 模型', 'models/vae/vae.safetensors'],
];

export async function loadModelConfig() {
    let payload;
    try {
        payload = await api('/api/settings/model-configs');
        if (payload.ok === false) throw new Error(payload.error || '读取全局模型配置失败');
    } catch (error) {
        return renderError(error.message || '读取全局模型配置失败');
    }

    const state = {
        items: Array.isArray(payload.items) ? payload.items.map(cleanItem) : [],
        defaultId: String(payload.default_id || ''),
        revision: String(payload.revision || ''),
        selectedId: String(payload.default_id || payload.items?.[0]?.id || ''),
    };
    return { html: renderPage(state), onMount: (root) => bindPage(root, state) };
}

function renderPage(state) {
    return `
        <div class="apple-page apple-page-wide apple-model-config-page">
            <div class="apple-page-hero apple-reveal">
                <span class="apple-eyebrow">模型与系统</span>
                <h1>全局模型配置</h1>
                <p>管理训练与推理共用的模型组件。默认配置会同步到全局模型路径。</p>
            </div>
            <div class="apple-model-config-layout apple-reveal" data-stagger="1">
                <aside class="apple-model-config-sidebar">
                    <div class="apple-section-header-row"><div><span class="apple-eyebrow">配置库</span><h2 class="apple-section-title">模型组合</h2></div><button class="apple-btn apple-btn-secondary apple-btn-sm" type="button" data-model-action="add">新建</button></div>
                    <div class="apple-model-config-list" data-model-list>${renderList(state)}</div>
                </aside>
                <form class="apple-model-config-editor" data-model-form>${renderEditor(selectedItem(state), state)}</form>
            </div>
            <div class="apple-config-actions apple-config-actions-sticky">
                <button class="apple-btn apple-btn-secondary" type="button" data-model-action="default">设为默认</button>
                <button class="apple-btn apple-btn-secondary" type="button" data-model-action="remove">删除当前配置</button>
                <button class="apple-btn apple-btn-primary" type="button" data-model-action="save">保存模型配置</button>
                <span class="apple-config-feedback" data-model-feedback role="status" aria-live="polite"></span>
            </div>
        </div>
    `;
}

function renderList(state) {
    return state.items.map((item) => `
        <button class="apple-model-config-list-item" type="button" data-model-id="${escapeAttribute(item.id)}" data-active="${item.id === state.selectedId}">
            <span><strong>${escapeHtml(item.name)}</strong><small>${familyLabel(item.model_family)}</small></span>
            ${item.id === state.defaultId ? '<em>默认</em>' : ''}
        </button>
    `).join('');
}

function renderEditor(item, state) {
    if (!item) return '<div class="apple-empty-state"><p>请新建一个模型配置</p></div>';
    return `
        <div class="apple-config-section">
            <div class="apple-config-section-header"><span class="apple-eyebrow">当前配置</span><h2 class="apple-config-section-title">${escapeHtml(item.name)}</h2><p class="apple-config-section-desc">配置名称、模型格式与三个必需组件路径。</p></div>
            <div class="apple-field-grid-2">
                ${textField('name', '配置名称', item.name, '例如：Krea-2 主模型')}
                ${selectField('model_family', '模型格式', item.model_family, [['anima', 'Anima'], ['krea2_raw', 'Krea-2']])}
            </div>
        </div>
        <div class="apple-config-section">
            <div class="apple-config-section-header"><span class="apple-eyebrow">模型组件</span><h2 class="apple-config-section-title">基础权重路径</h2><p class="apple-config-section-desc">支持项目相对路径、绝对路径和环境变量。</p></div>
            <div class="apple-model-path-fields">${PATH_FIELDS.map(([key, label, placeholder]) => textField(key, label, item[key], placeholder)).join('')}</div>
        </div>
        <input type="hidden" data-model-field="id" value="${escapeAttribute(item.id)}">
        <input type="hidden" data-model-revision value="${escapeAttribute(state.revision)}">
    `;
}

function bindPage(root, state) {
    bindList(root, state);
    root.querySelector('[data-model-action="add"]')?.addEventListener('click', () => {
        const id = `model-${Date.now().toString(36)}`;
        state.items.push(cleanItem({ ...selectedItem(state), id, name: '新模型配置' }));
        state.selectedId = id;
        refreshWorkspace(root, state);
    });
    root.querySelector('[data-model-action="remove"]')?.addEventListener('click', () => {
        syncEditor(root, state);
        if (state.items.length <= 1) return showFeedback(root, '至少需要保留一个模型配置', 'error');
        const removedId = state.selectedId;
        state.items = state.items.filter((item) => item.id !== removedId);
        if (state.defaultId === removedId) state.defaultId = state.items[0].id;
        state.selectedId = state.items[0].id;
        refreshWorkspace(root, state);
    });
    root.querySelector('[data-model-action="default"]')?.addEventListener('click', () => {
        syncEditor(root, state);
        state.defaultId = state.selectedId;
        refreshWorkspace(root, state);
        showFeedback(root, '已标记为默认，保存后生效', 'info');
    });
    root.querySelector('[data-model-action="save"]')?.addEventListener('click', () => saveLibrary(root, state));
}

function bindList(root, state) {
    root.querySelectorAll('[data-model-id]').forEach((button) => {
        button.addEventListener('click', () => {
            syncEditor(root, state);
            state.selectedId = button.dataset.modelId;
            refreshWorkspace(root, state);
        });
    });
}

function refreshWorkspace(root, state) {
    const list = root.querySelector('[data-model-list]');
    const editor = root.querySelector('[data-model-form]');
    if (list) list.innerHTML = renderList(state);
    if (editor) editor.innerHTML = renderEditor(selectedItem(state), state);
    bindList(root, state);
}

function syncEditor(root, state) {
    const item = selectedItem(state);
    if (!item) return;
    root.querySelectorAll('[data-model-field]').forEach((field) => {
        item[field.dataset.modelField] = String(field.value || '').trim();
    });
}

async function saveLibrary(root, state) {
    syncEditor(root, state);
    const error = validateState(state);
    if (error) return showFeedback(root, error, 'error');
    const button = root.querySelector('[data-model-action="save"]');
    button.disabled = true;
    try {
        const payload = await api('/api/settings/model-configs', {
            method: 'PUT',
            body: JSON.stringify({ revision: state.revision, default_id: state.defaultId, items: state.items }),
        });
        if (payload.ok === false) throw new Error(payload.error || '保存模型配置失败');
        state.items = payload.items.map(cleanItem);
        state.defaultId = payload.default_id;
        state.revision = payload.revision;
        refreshWorkspace(root, state);
        showFeedback(root, payload.message || '模型配置已保存', 'success');
    } catch (error) {
        showFeedback(root, error.message || '保存模型配置失败', 'error');
    } finally {
        button.disabled = false;
    }
}

function validateState(state) {
    if (!state.items.length) return '至少需要保留一个模型配置';
    const names = new Set();
    for (const item of state.items) {
        if (!item.name) return '配置名称不能为空';
        const name = item.name.toLocaleLowerCase();
        if (names.has(name)) return '配置名称不能重复';
        names.add(name);
        const missing = PATH_FIELDS.find(([key]) => !item[key]);
        if (missing) return `“${item.name}”缺少${missing[1]}`;
    }
    return '';
}

function selectedItem(state) { return state.items.find((item) => item.id === state.selectedId) || state.items[0] || null; }
function cleanItem(item = {}) { return { id: String(item.id || ''), name: String(item.name || ''), model_family: item.model_family === 'krea2_raw' ? 'krea2_raw' : 'anima', ...Object.fromEntries(PATH_FIELDS.map(([key]) => [key, String(item[key] || '')])) }; }
function familyLabel(value) { return value === 'krea2_raw' ? 'Krea-2' : 'Anima'; }
function textField(key, label, value, placeholder) { return `<label class="apple-field"><span class="apple-field-label-text">${label}</span><input class="apple-input" type="text" data-model-field="${key}" value="${escapeAttribute(value)}" placeholder="${escapeAttribute(placeholder)}"></label>`; }
function selectField(key, label, value, options) { return `<label class="apple-field"><span class="apple-field-label-text">${label}</span><select class="apple-select" data-model-field="${key}">${options.map(([option, text]) => `<option value="${option}" ${option === value ? 'selected' : ''}>${text}</option>`).join('')}</select></label>`; }
function showFeedback(root, message, tone) { const el = root.querySelector('[data-model-feedback]'); if (el) { el.textContent = message; el.dataset.tone = tone; el.classList.add('apple-config-feedback-visible'); } }
function renderError(message) { return `<div class="apple-page"><div class="apple-page-hero"><h1>全局模型配置</h1></div><div class="apple-empty-state"><p>${escapeHtml(message)}</p></div></div>`; }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
function escapeAttribute(value) { return escapeHtml(value); }
