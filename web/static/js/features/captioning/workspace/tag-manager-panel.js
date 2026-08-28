import { captioningApi, escapeAttribute, escapeHtml, feedback, groupOptions, jsonOptions, panelShell, selectedGroup } from './shared.js?v=dragon-ui-20260829v11';

export function renderTagManagerPanel(state) {
    const items = state.workspaceData.tagItems || [];
    const selected = items.find((item) => item.name === state.workspaceData.tagItemName);
    const query = state.workspaceData.tagQuery || '';
    const visible = items.filter((item) => `${item.name} ${item.text}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
    const loading = state.workspaceData.tagLoading === true;
    return panelShell('TAG MANAGER', 'Tag 管理', '', `<div class="dragon-caption-tag-toolbar">
        <label><span>目录组</span><select class="dragon-select" data-tag-group ${loading ? 'disabled' : ''}><option value="">选择目录组</option>${groupOptions(state, state.workspaceData.tagGroupId)}</select></label>
        <button class="dragon-btn dragon-btn-secondary" type="button" data-tag-load ${loading ? 'disabled' : ''}>${loading ? '加载中…' : '重新加载'}</button>
        <label><span>查找</span><input class="dragon-input" data-tag-find value="${escapeAttribute(state.workspaceData.tagFind || '')}"></label><label><span>替换</span><input class="dragon-input" data-tag-replace value="${escapeAttribute(state.workspaceData.tagReplace || '')}"></label>
        <button class="dragon-btn dragon-btn-secondary" type="button" data-tag-replace-all ${items.length && !loading ? '' : 'disabled'}>全局替换并保存</button>
    </div><div class="dragon-caption-split-list dragon-caption-tag-manager"><aside><input class="dragon-input" data-tag-search aria-label="搜索图片或 Tag" placeholder="搜索图片或 Tag" value="${escapeAttribute(query)}"><small data-tag-count>${visible.length}/${items.length} 项</small><div>${items.map((item) => `<button type="button" data-tag-item="${escapeAttribute(item.name)}" data-active="${item.name === state.workspaceData.tagItemName}" ${visible.includes(item) ? '' : 'hidden'}><strong>${escapeHtml(item.name)}</strong><small>${item.text_length} 字符</small></button>`).join('')}</div>${items.length && !visible.length ? '<div class="dragon-empty-state" data-tag-empty><p>没有匹配的图片或 Tag。</p></div>' : ''}</aside><section>${state.workspaceData.tagNotice ? `<div class="dragon-caption-inline-status" role="status">${escapeHtml(state.workspaceData.tagNotice)}</div>` : ''}${selected ? renderEditor(selected, state) : `<div class="dragon-empty-state"><p>${items.length ? '选择一项开始编辑。' : '选择目录组并加载标注。'}</p></div>`}</section></div>`);
}

export function bindTagManagerPanel(root, state) {
    const groupSelect = root.querySelector('[data-tag-group]');
    groupSelect?.addEventListener('change', () => {
        if (!discardDirty(state)) { groupSelect.value = state.workspaceData.tagGroupId || ''; return; }
        state.workspaceData.tagGroupId = groupSelect.value; state.workspaceData.tagItems = []; state.workspaceData.tagItemName = ''; state.workspaceData.tagNotice = '目录组已切换，请重新加载。'; state.suiteRender();
    });
    root.querySelector('[data-tag-load]')?.addEventListener('click', async () => load(root, state));
    root.querySelectorAll('[data-tag-item]').forEach((button) => button.addEventListener('click', () => { if (!discardDirty(state)) return; state.workspaceData.tagItemName = button.dataset.tagItem; state.workspaceData.tagNotice = ''; state.suiteRender(); }));
    root.querySelector('[data-tag-search]')?.addEventListener('input', (event) => {
        const query = event.target.value.toLocaleLowerCase();
        state.workspaceData.tagQuery = event.target.value; let visible = 0;
        root.querySelectorAll('[data-tag-item]').forEach((button) => { button.hidden = !button.textContent.toLocaleLowerCase().includes(query); if (!button.hidden) visible += 1; });
        const count = root.querySelector('[data-tag-count]'); if (count) count.textContent = `${visible}/${state.workspaceData.tagItems.length} 项`;
    });
    root.querySelector('[data-tag-find]')?.addEventListener('input', (event) => { state.workspaceData.tagFind = event.target.value; });
    root.querySelector('[data-tag-replace]')?.addEventListener('input', (event) => { state.workspaceData.tagReplace = event.target.value; });
    root.querySelector('[data-tag-editor]')?.addEventListener('input', (event) => { state.workspaceData.tagDrafts ||= {}; state.workspaceData.tagDrafts[state.workspaceData.tagItemName] = event.target.value; state.workspaceData.tagDirtyName = state.workspaceData.tagItemName; const save = root.querySelector('[data-tag-save]'); if (save) save.textContent = '保存修改'; });
    root.querySelector('[data-tag-save]')?.addEventListener('click', async (event) => {
        const item = selectedItem(state); const group = selectedGroup(state, state.workspaceData.tagGroupId);
        if (!item || !group) return feedback(root, '当前图片或目录组已失效，请重新加载', 'error');
        event.currentTarget.disabled = true;
        try {
            const payload = await captioningApi('/workspace/tag', jsonOptions('PATCH', {directory: group.path, image_name: item.name, text: root.querySelector('[data-tag-editor]').value, expected: item.text}));
            item.text = payload.text; item.text_length = payload.text.length; delete state.workspaceData.tagDrafts?.[item.name]; state.workspaceData.tagDirtyName = ''; state.workspaceData.tagNotice = '当前标注已保存'; state.suiteRender();
        } catch (error) { event.currentTarget.disabled = false; feedback(root, error.message, 'error'); }
    });
    root.querySelector('[data-tag-undo]')?.addEventListener('click', () => { const item = selectedItem(state); if (!item) return; root.querySelector('[data-tag-editor]').value = item.text; delete state.workspaceData.tagDrafts?.[item.name]; state.workspaceData.tagDirtyName = ''; });
    root.querySelector('[data-tag-replace-all]')?.addEventListener('click', async () => {
        const button = root.querySelector('[data-tag-replace-all]'); const find = root.querySelector('[data-tag-find]').value; const replace = root.querySelector('[data-tag-replace]').value;
        const group = selectedGroup(state, state.workspaceData.tagGroupId); if (!find) return feedback(root, '查找内容不能为空', 'error'); if (!group) return feedback(root, '请选择并加载目录组', 'error');
        const matches = state.workspaceData.tagItems.filter((item) => item.text.includes(find));
        if (!matches.length) return feedback(root, '没有标注命中查找内容', 'error');
        if (!window.confirm(`将在当前目录组的 ${matches.length} 个标注中替换“${find}”，是否继续？`)) return;
        button.disabled = true; button.textContent = `替换中 0/${matches.length}`;
        let changed = 0, failed = 0;
        for (const item of matches) {
            const next = item.text.split(find).join(replace); if (next === item.text) continue;
            try { await captioningApi('/workspace/tag', jsonOptions('PATCH', {directory: group.path, image_name: item.name, text: next, expected: item.text})); item.text = next; item.text_length = next.length; changed += 1; }
            catch { failed += 1; }
            button.textContent = `替换中 ${changed + failed}/${matches.length}`;
        }
        state.workspaceData.tagNotice = `替换 ${changed} 项，失败 ${failed} 项`; state.suiteRender();
    });
}

async function load(root, state) {
    const group = selectedGroup(state, state.workspaceData.tagGroupId || root.querySelector('[data-tag-group]').value);
    if (!group) return feedback(root, '请选择目录组', 'error');
    if (!discardDirty(state) || state.workspaceData.tagLoading) return;
    state.workspaceData.tagLoading = true; state.workspaceData.tagNotice = ''; state.suiteRender();
    try { const payload = await captioningApi('/workspace/tags', jsonOptions('POST', {directory: group.path})); state.workspaceData.tagItems = payload.results; state.workspaceData.tagGroupId = group.id; state.workspaceData.tagItemName = payload.results[0]?.name || ''; state.workspaceData.tagNotice = payload.results.length ? `已加载 ${payload.results.length} 项` : '该目录没有可编辑标注'; }
    catch (error) { state.workspaceData.tagNotice = `加载失败：${error.message}`; }
    finally { state.workspaceData.tagLoading = false; state.suiteRender(); }
}
function selectedItem(state) { return state.workspaceData.tagItems.find((item) => item.name === state.workspaceData.tagItemName); }
function renderEditor(item, state) { const draft = state.workspaceData.tagDrafts?.[item.name]; const dirty = state.workspaceData.tagDirtyName === item.name; return `<header><strong>${escapeHtml(item.name)}</strong><span>${dirty ? '未保存 · ' : ''}${String(draft ?? item.text).length} 字符</span></header><textarea class="dragon-textarea" data-tag-editor rows="18">${escapeHtml(draft ?? item.text)}</textarea><div class="dragon-caption-form-actions"><button class="dragon-btn dragon-btn-primary" type="button" data-tag-save>${dirty ? '保存修改' : '保存'}</button><button class="dragon-btn dragon-btn-secondary" type="button" data-tag-undo>撤回</button></div>`; }
function discardDirty(state) { if (!state.workspaceData.tagDirtyName) return true; if (!window.confirm('当前标注有未保存修改，放弃并继续？')) return false; delete state.workspaceData.tagDrafts?.[state.workspaceData.tagDirtyName]; state.workspaceData.tagDirtyName = ''; return true; }
