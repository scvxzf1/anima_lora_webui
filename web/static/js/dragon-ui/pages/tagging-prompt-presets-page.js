/* Dedicated prompt-preset manager for the tagging workflow. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import {
    createPromptPreset,
    deletePromptPreset,
    loadPromptPresets,
    loadTaggingSettings,
    updatePromptPreset,
} from './tagging-api.js?v=dragon-ui-20260901v6';
import {
    readTaggingWorkspaceState,
    returnToTaggingWorkspace,
    updateTaggingPromptDraft,
} from './tagging-workspace-state.js?v=dragon-ui-20260831v4';

const api = createApiClient();

export async function loadTaggingPromptPresetsPage() {
    const [settingsResult, presetsResult] = await Promise.all([
        loadTaggingSettings(api),
        loadPromptPresets(api),
    ]);
    const workspace = readTaggingWorkspaceState();
    const presets = Array.isArray(presetsResult.presets) ? presetsResult.presets : [];
    const selected = presets.find((item) => item.id === workspace.currentPresetId) || presets[0] || null;
    const state = {
        settings: settingsResult.settings || settingsResult,
        presets,
        selectedId: selected?.id || '',
        draft: selected ? presetDraft(selected) : {
            name: '',
            system_prompt: workspace.systemPrompt || settingsResult.system_prompt || '',
            user_prompt: workspace.userPrompt || '',
        },
        dirty: false,
        saving: false,
        allowLeave: false,
        error: '',
        notice: '',
        root: null,
        cleanup: null,
        active: true,
        operationId: 0,
        draftVersion: 0,
    };
    return {
        html: renderPage(state),
        onMount: (root) => mountPage(root, state),
        beforeLeave: () => state.allowLeave || !state.dirty || window.confirm('提示词预设有未保存修改，仍要离开吗？'),
        onUnmount: () => {
            state.active = false;
            state.operationId += 1;
            state.cleanup?.();
        },
    };
}

function mountPage(root, state) {
    state.root = root;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    root.addEventListener('click', (event) => handleClick(state, event), options);
    root.addEventListener('input', (event) => handleInput(state, event), options);
    root.addEventListener('submit', (event) => handleSubmit(state, event), options);
    state.cleanup = () => controller.abort();
}

function renderPage(state) {
    return `<div class="dragon-page dragon-page-wide dragon-caption-page dragon-tagging-tool-page" data-tagging-prompt-presets-page>
        <header class="dragon-tagging-tool-header">
            <div><button class="dragon-icon-button" type="button" data-prompt-back aria-label="返回打标工作台" title="返回">${renderIcon('chevronDown')}</button><span><span class="dragon-eyebrow">PROMPT LIBRARY</span><h1>提示词预设</h1></span></div>
            <button class="dragon-btn dragon-btn-primary" type="button" data-prompt-new>${renderIcon('filePlus', 'dragon-btn-icon')}<span>新建预设</span></button>
        </header>
        ${feedback(state)}
        <div class="dragon-tagging-library-layout">
            <aside class="dragon-tagging-preset-library" aria-label="提示词预设列表">
                <header><strong>预设</strong><span>${state.presets.length}</span></header>
                <div data-prompt-preset-list>${state.presets.length ? state.presets.map((preset) => renderPresetItem(preset, state.selectedId)).join('') : '<div class="dragon-tagging-library-empty">暂无预设</div>'}</div>
            </aside>
            <form class="dragon-tagging-preset-editor" data-prompt-form>
                <header><div><span class="dragon-eyebrow">${state.selectedId ? (state.draft.builtin ? 'BUILT-IN' : 'EDIT') : 'NEW'}</span><h2>${state.selectedId ? '编辑预设' : '新建预设'}</h2></div>${state.draft.builtin ? '<button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-prompt-clone>复制为新预设</button>' : ''}${state.dirty ? '<span class="dragon-tagging-dirty-state">未保存</span>' : ''}</header>
                ${state.draft.builtin ? '<p class="dragon-tagging-preset-note">这是内置模板，不能直接修改或删除。复制后可保存为自己的预设。</p>' : ''}
                <label class="dragon-field"><span>预设名称</span><input class="dragon-input" type="text" name="name" maxlength="80" value="${escapeAttribute(state.draft.name)}" ${state.draft.builtin ? 'disabled' : ''} required></label>
                <label class="dragon-field"><span>系统提示词</span><textarea class="dragon-textarea" name="system_prompt" rows="8" maxlength="10000" ${state.draft.builtin ? 'disabled' : ''} required>${escapeHtml(state.draft.system_prompt)}</textarea></label>
                <label class="dragon-field"><span>用户提示词</span><textarea class="dragon-textarea" name="user_prompt" rows="8" maxlength="10000" ${state.draft.builtin ? 'disabled' : ''} required>${escapeHtml(state.draft.user_prompt)}</textarea></label>
                <footer><button class="dragon-btn dragon-btn-danger" type="button" data-prompt-delete ${state.selectedId && !state.draft.builtin && !state.saving ? '' : 'disabled'}>${renderIcon('trash', 'dragon-btn-icon')}<span>删除</span></button><span></span><button class="dragon-btn dragon-btn-secondary" type="button" data-prompt-apply ${validDraft(state.draft) ? '' : 'disabled'}>${renderIcon('check', 'dragon-btn-icon')}<span>应用并返回</span></button><button class="dragon-btn dragon-btn-primary" type="submit" ${state.saving || state.draft.builtin || !validDraft(state.draft) ? 'disabled' : ''}>${renderIcon('save', 'dragon-btn-icon')}<span>${state.saving ? '保存中…' : '保存预设'}</span></button></footer>
            </form>
        </div>
    </div>`;
}

function renderPresetItem(preset, selectedId) {
    return `<button type="button" data-prompt-select="${escapeAttribute(preset.id)}" data-active="${preset.id === selectedId}"><span><strong>${escapeHtml(preset.name)}${preset.builtin ? ' <small class="dragon-tagging-preset-badge">内置</small>' : ''}</strong><small>${escapeHtml(excerpt(preset.user_prompt))}</small></span>${renderIcon('chevronDown')}</button>`;
}

function handleClick(state, event) {
    const target = event.target.closest?.('[data-prompt-back], [data-prompt-new], [data-prompt-select], [data-prompt-delete], [data-prompt-apply], [data-prompt-clone]');
    if (!target) return;
    if (target.matches('[data-prompt-back]')) return leavePage(state);
    if (target.matches('[data-prompt-new]')) return startNew(state);
    if (target.matches('[data-prompt-clone]')) return clonePreset(state);
    if (target.matches('[data-prompt-select]')) return selectPreset(state, target.dataset.promptSelect);
    if (target.matches('[data-prompt-delete]')) return run(() => removePreset(state));
    if (target.matches('[data-prompt-apply]')) return applyPreset(state);
}

function handleInput(state, event) {
    const input = event.target;
    if (!input.form?.matches('[data-prompt-form]') || !['name', 'system_prompt', 'user_prompt'].includes(input.name)) return;
    state.draft[input.name] = input.value;
    state.dirty = true;
    state.draftVersion += 1;
    syncDirtyUi(state);
}

function handleSubmit(state, event) {
    if (!event.target.matches('[data-prompt-form]')) return;
    event.preventDefault();
    run(() => savePreset(state));
}

function selectPreset(state, presetId) {
    if (state.dirty && !window.confirm('放弃当前未保存修改并切换预设吗？')) return;
    const preset = state.presets.find((item) => item.id === presetId);
    if (!preset) return;
    state.operationId += 1;
    state.draftVersion += 1;
    state.saving = false;
    state.selectedId = preset.id;
    state.draft = presetDraft(preset);
    state.dirty = false;
    rerender(state, { preserveListScroll: true });
}

function startNew(state) {
    if (state.dirty && !window.confirm('放弃当前未保存修改并新建预设吗？')) return;
    const workspace = readTaggingWorkspaceState();
    state.operationId += 1;
    state.draftVersion += 1;
    state.saving = false;
    state.selectedId = '';
    state.draft = {
        name: '',
        system_prompt: workspace.systemPrompt || state.settings.system_prompt || '',
        user_prompt: workspace.userPrompt || '',
    };
    state.dirty = false;
    rerender(state, { preserveListScroll: true, focusName: true });
}

function clonePreset(state) {
    const source = state.presets.find((item) => item.id === state.selectedId);
    if (!source) return startNew(state);
    state.operationId += 1;
    state.draftVersion += 1;
    state.saving = false;
    state.selectedId = '';
    state.draft = {
        name: `${source.name}（副本）`.slice(0, 80),
        system_prompt: source.system_prompt || '',
        user_prompt: source.user_prompt || '',
        builtin: false,
    };
    state.dirty = true;
    rerender(state, { preserveListScroll: true, focusName: true });
}

async function savePreset(state) {
    if (state.saving || !validDraft(state.draft)) return;
    const operationId = ++state.operationId;
    const draftVersion = state.draftVersion;
    const selectedId = state.selectedId;
    const draft = { ...state.draft };
    state.saving = true;
    rerender(state, { preserveListScroll: true });
    try {
        const payload = state.selectedId
            ? await updatePromptPreset(api, selectedId, draft)
            : await createPromptPreset(api, draft);
        if (!state.active || state.operationId !== operationId) return;
        if (state.draftVersion !== draftVersion || state.selectedId !== selectedId) {
            state.saving = false;
            rerender(state, { preserveListScroll: true });
            return;
        }
        state.presets = payload.presets || state.presets;
        state.selectedId = payload.preset?.id || state.selectedId;
        state.draft = presetDraft(payload.preset || state.draft);
        state.dirty = false;
        state.notice = '提示词预设已保存。';
        state.error = '';
    } catch (error) {
        if (state.active && state.operationId === operationId) state.error = error.message || '保存提示词预设失败';
    } finally {
        if (state.active && state.operationId === operationId) {
            state.saving = false;
            rerender(state, { preserveListScroll: true });
        }
    }
}

async function removePreset(state) {
    if (!state.selectedId || !window.confirm('删除这个提示词预设吗？')) return;
    const operationId = ++state.operationId;
    const selectedId = state.selectedId;
    state.saving = true;
    try {
        const payload = await deletePromptPreset(api, selectedId);
        if (!state.active || state.operationId !== operationId || state.selectedId !== selectedId) return;
        state.presets = payload.presets || [];
        const next = state.presets[0] || null;
        state.selectedId = next?.id || '';
        state.draft = next ? presetDraft(next) : { name: '', system_prompt: state.settings.system_prompt || '', user_prompt: '' };
        state.dirty = false;
        state.notice = '提示词预设已删除。';
        state.error = '';
    } catch (error) {
        if (state.active && state.operationId === operationId) state.error = error.message || '删除提示词预设失败';
    } finally {
        if (state.active && state.operationId === operationId) state.saving = false;
    }
    if (state.active && state.operationId === operationId) rerender(state);
}

function applyPreset(state) {
    if (!validDraft(state.draft)) return;
    updateTaggingPromptDraft({
        currentPresetId: state.selectedId,
        systemPrompt: state.draft.system_prompt,
        userPrompt: state.draft.user_prompt,
    });
    state.allowLeave = true;
    returnToTaggingWorkspace();
}

function leavePage(state) {
    if (state.dirty && !window.confirm('提示词预设有未保存修改，仍要返回吗？')) return;
    state.allowLeave = true;
    returnToTaggingWorkspace();
}

function syncDirtyUi(state) {
    const marker = state.root?.querySelector('.dragon-tagging-dirty-state');
    if (!marker && state.dirty) {
        state.root?.querySelector('.dragon-tagging-preset-editor > header')?.insertAdjacentHTML('beforeend', '<span class="dragon-tagging-dirty-state">未保存</span>');
    }
    const valid = validDraft(state.draft);
    const save = state.root?.querySelector('[data-prompt-form] > footer button[type="submit"]');
    const apply = state.root?.querySelector('[data-prompt-apply]');
    const remove = state.root?.querySelector('[data-prompt-delete]');
    if (save) save.disabled = state.saving || !valid;
    if (apply) apply.disabled = !valid;
    if (remove) remove.disabled = !state.selectedId || state.draft.builtin || state.saving;
}

function rerender(state, { preserveListScroll = false, focusName = false } = {}) {
    if (!state.root) return;
    const scrollTop = preserveListScroll ? state.root.querySelector('[data-prompt-preset-list]')?.scrollTop || 0 : 0;
    state.root.innerHTML = renderPage(state);
    const list = state.root.querySelector('[data-prompt-preset-list]');
    if (list) list.scrollTop = scrollTop;
    if (focusName) state.root.querySelector('input[name="name"]')?.focus();
}

function feedback(state) {
    return `${state.error ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="error" role="alert">${escapeHtml(state.error)}</div>` : ''}${state.notice ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="success" role="status">${escapeHtml(state.notice)}</div>` : ''}`;
}

function presetDraft(preset) {
    return {
        name: String(preset?.name || ''),
        system_prompt: String(preset?.system_prompt || ''),
        user_prompt: String(preset?.user_prompt || ''),
        builtin: preset?.builtin === true,
    };
}

function validDraft(draft) {
    return Boolean(draft.name.trim() && draft.system_prompt.trim() && draft.user_prompt.trim());
}

function excerpt(value) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    return text.length > 52 ? `${text.slice(0, 52)}…` : text;
}

function run(fn) {
    Promise.resolve().then(fn).catch((error) => console.error('[dragon-tagging-prompts]', error));
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
