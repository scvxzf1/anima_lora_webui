import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260817v44';
import {
    cleanModelItem,
    familyLabel,
    MODEL_PATH_FIELDS,
    normalizeModelGroups,
} from './model-config-state.js?v=dragon-ui-20260824-zimage-v1';

const api = createApiClient();

export const MODEL_QUICK_PATH_KEYS = Object.freeze(MODEL_PATH_FIELDS.map(([key]) => key));

export function orderedModelGroups(payload = {}) {
    const items = Array.isArray(payload.items) ? payload.items.map(cleanModelItem) : [];
    const groups = normalizeModelGroups(Array.isArray(payload.groups) ? payload.groups : [], items);
    const byId = new Map(items.map((item) => [item.id, item]));
    return groups.map((group) => ({
        ...group,
        items: group.item_ids.map((itemId) => byId.get(itemId)).filter(Boolean),
    }));
}

export function renderModelQuickPickerTrigger() {
    return `<button class="dragon-btn dragon-btn-secondary dragon-model-quick-trigger" type="button" data-model-quick-action="open">${renderIcon('layers', 'dragon-btn-icon')}<span>快速配置模型</span></button>`;
}

export function renderModelQuickPickerDialog() {
    return `
        <dialog class="dragon-model-quick-dialog" data-model-quick-dialog aria-labelledby="dragon-model-quick-title">
            <div class="dragon-model-quick-shell">
                <div class="dragon-model-quick-head">
                    <div>
                        <span class="dragon-eyebrow">模型配置库</span>
                        <h2 id="dragon-model-quick-title">快速配置模型</h2>
                        <p>全局模型配置 · 分组顺序同步</p>
                    </div>
                    <div class="dragon-model-quick-head-actions">
                        <button class="dragon-icon-button" type="button" data-model-quick-action="refresh" aria-label="刷新模型配置" title="刷新">${renderIcon('refresh')}</button>
                        <button class="dragon-icon-button" type="button" data-model-quick-action="close" aria-label="关闭快速配置" title="关闭">${renderIcon('x')}</button>
                    </div>
                </div>
                <div class="dragon-model-quick-status" data-model-quick-status role="status" aria-live="polite"></div>
                <div class="dragon-model-quick-workspace">
                    <aside class="dragon-model-quick-library" aria-label="模型配置列表">
                        <label class="dragon-model-quick-search">
                            <span class="visually-hidden">搜索模型配置</span>
                            <input class="dragon-input" type="search" autocomplete="off" data-model-quick-search placeholder="搜索名称、模型族或路径…">
                        </label>
                        <div class="dragon-model-quick-groups" data-model-quick-list></div>
                    </aside>
                    <section class="dragon-model-quick-preview" data-model-quick-preview aria-label="模型路径预览"></section>
                </div>
                <div class="dragon-model-quick-footer">
                    <div class="dragon-model-quick-selection" data-model-quick-selection></div>
                    <button class="dragon-btn dragon-btn-primary" type="button" data-model-quick-action="apply">应用此配置</button>
                </div>
            </div>
        </dialog>
    `;
}

function allItems(state) {
    return state.groups.flatMap((group) => group.items);
}

function selectedItem(state) {
    return allItems(state).find((item) => item.id === state.selectedId) || null;
}

function itemComplete(item) {
    return Boolean(item) && MODEL_PATH_FIELDS.every(([key]) => Boolean(item[key]));
}

function itemMatches(item, query) {
    const needle = String(query || '').trim().toLocaleLowerCase();
    if (!needle) return true;
    return [item.name, familyLabel(item.model_family), ...MODEL_PATH_FIELDS.map(([key]) => item[key])]
        .some((value) => String(value || '').toLocaleLowerCase().includes(needle));
}

function matchingCurrentItem(root, state) {
    return allItems(state).find((item) => MODEL_PATH_FIELDS.every(([key]) => (
        String(root.querySelector(`[data-key="${key}"]`)?.value || '').trim() === item[key]
    ))) || null;
}

function renderList(dialog, state) {
    const list = dialog.querySelector('[data-model-quick-list]');
    if (!list) return;
    const query = String(dialog.querySelector('[data-model-quick-search]')?.value || '');
    const groups = state.groups.map((group) => ({
        ...group,
        visibleItems: group.items.filter((item) => itemMatches(item, query)),
    })).filter((group) => !query.trim() || group.visibleItems.length);
    const visibleCount = groups.reduce((total, group) => total + group.visibleItems.length, 0);

    if (!visibleCount) {
        list.innerHTML = '<div class="dragon-model-quick-empty">没有匹配的模型配置</div>';
        return;
    }

    list.innerHTML = groups.map((group) => `
        <section class="dragon-model-quick-group">
            <div class="dragon-model-quick-group-title"><strong>${escapeHtml(group.label)}</strong><span>${group.visibleItems.length}</span></div>
            <div class="dragon-model-quick-items">
                ${group.visibleItems.map((item) => `
                    <button class="dragon-model-quick-item" type="button" data-model-quick-item="${escapeHtml(item.id)}" data-active="${item.id === state.selectedId}" aria-pressed="${item.id === state.selectedId}">
                        <span><strong>${escapeHtml(item.name || '未命名配置')}</strong><small>${escapeHtml(item.pretrained_model_name_or_path || '基础模型路径未填写')}</small></span>
                        <em>${escapeHtml(familyLabel(item.model_family))}</em>
                    </button>
                `).join('')}
            </div>
        </section>
    `).join('');
}

function renderPreview(dialog, state) {
    const preview = dialog.querySelector('[data-model-quick-preview]');
    const selection = dialog.querySelector('[data-model-quick-selection]');
    const apply = dialog.querySelector('[data-model-quick-action="apply"]');
    const item = selectedItem(state);
    const complete = itemComplete(item);
    if (apply) apply.disabled = !complete || state.loading;

    if (!item) {
        if (preview) preview.innerHTML = '<div class="dragon-model-quick-empty"><strong>未选择模型配置</strong></div>';
        if (selection) selection.innerHTML = '<span>当前选择</span><strong>未选择</strong>';
        return;
    }

    if (preview) preview.innerHTML = `
        <div class="dragon-model-quick-preview-head">
            <div><span class="dragon-eyebrow">路径预览</span><h3>${escapeHtml(item.name || '未命名配置')}</h3></div>
            <div class="dragon-model-quick-badges">
                <span>${escapeHtml(familyLabel(item.model_family))}</span>
                ${item.id === state.defaultId ? '<span data-default="true">默认</span>' : ''}
            </div>
        </div>
        <dl class="dragon-model-quick-paths">
            ${MODEL_PATH_FIELDS.map(([key, label]) => `
                <div data-empty="${item[key] ? 'false' : 'true'}">
                    <dt>${escapeHtml(label)}</dt>
                    <dd><code title="${escapeHtml(item[key] || '未配置')}">${escapeHtml(item[key] || '未配置')}</code></dd>
                </div>
            `).join('')}
        </dl>
    `;
    if (selection) selection.innerHTML = `<span>当前选择</span><strong>${escapeHtml(item.name || '未命名配置')}</strong>`;
}

function renderState(dialog, state) {
    renderList(dialog, state);
    renderPreview(dialog, state);
}

async function loadLibrary(root, dialog, state, { preserveSelection = false } = {}) {
    const requestId = ++state.loadSequence;
    state.loading = true;
    const status = dialog.querySelector('[data-model-quick-status]');
    const refresh = dialog.querySelector('[data-model-quick-action="refresh"]');
    if (status) {
        status.dataset.tone = 'info';
        status.textContent = '正在读取模型配置…';
    }
    if (refresh) refresh.disabled = true;
    renderPreview(dialog, state);

    try {
        const payload = await api('/api/settings/model-configs');
        if (requestId !== state.loadSequence) return;
        if (payload?.ok === false) throw new Error(payload.error || '读取模型配置失败');
        state.groups = orderedModelGroups(payload || {});
        state.defaultId = String(payload?.default_id || '');
        const current = matchingCurrentItem(root, state);
        const items = allItems(state);
        const previous = items.find((item) => item.id === state.selectedId);
        state.selectedId = current?.id
            || (preserveSelection ? previous?.id : '')
            || state.defaultId
            || items[0]?.id
            || '';
        if (status) status.textContent = '';
        renderState(dialog, state);
    } catch (error) {
        if (requestId !== state.loadSequence) return;
        state.groups = [];
        state.selectedId = '';
        if (status) {
            status.dataset.tone = 'error';
            status.textContent = error.message || '读取模型配置失败';
        }
        renderState(dialog, state);
    } finally {
        if (requestId !== state.loadSequence) return;
        state.loading = false;
        if (refresh) refresh.disabled = false;
        renderPreview(dialog, state);
    }
}

export function bindModelQuickPicker(root, { onApply } = {}) {
    const dialog = root.querySelector('[data-model-quick-dialog]');
    const trigger = root.querySelector('[data-model-quick-action="open"]');
    if (!dialog || !trigger || trigger.dataset.bound === 'true') return;
    trigger.dataset.bound = 'true';
    const state = { groups: [], defaultId: '', selectedId: '', loading: false, loadSequence: 0 };

    trigger.addEventListener('click', async () => {
        const search = dialog.querySelector('[data-model-quick-search]');
        if (search) search.value = '';
        if (!dialog.open) dialog.showModal();
        await loadLibrary(root, dialog, state);
        search?.focus({ preventScroll: true });
    });
    dialog.querySelector('[data-model-quick-action="close"]')?.addEventListener('click', () => dialog.close('cancel'));
    dialog.querySelector('[data-model-quick-action="refresh"]')?.addEventListener('click', () => (
        loadLibrary(root, dialog, state, { preserveSelection: true })
    ));
    dialog.querySelector('[data-model-quick-action="apply"]')?.addEventListener('click', () => {
        const item = selectedItem(state);
        if (!itemComplete(item)) return;
        onApply?.(item);
        dialog.close('apply');
    });
    dialog.querySelector('[data-model-quick-search]')?.addEventListener('input', () => renderList(dialog, state));
    dialog.querySelector('[data-model-quick-list]')?.addEventListener('click', (event) => {
        if (!(event.target instanceof Element)) return;
        const button = event.target.closest('[data-model-quick-item]');
        if (!button) return;
        state.selectedId = button.dataset.modelQuickItem || '';
        renderState(dialog, state);
    });
    dialog.addEventListener('click', (event) => {
        if (event.target === dialog) dialog.close('cancel');
    });
}
