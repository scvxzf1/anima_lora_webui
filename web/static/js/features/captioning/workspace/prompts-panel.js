import { escapeAttribute, escapeHtml, feedback, panelShell, saveWorkspace, uid } from './shared.js?v=dragon-ui-20260829v11';

export function renderPromptsPanel(state) {
    const kind = state.workspaceData.promptKind || 'system';
    const prompts = state.workspace?.prompts?.[kind] || [];
    const query = state.workspaceData.promptQuery || '';
    const visible = prompts.filter((prompt) => `${prompt.name} ${prompt.content}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
    const selected = prompts.find((prompt) => prompt.id === state.workspaceData.promptId) || prompts[0];
    return panelShell('PROMPTS', '提示词预设', `<div class="dragon-segmented"><button type="button" data-prompt-kind="system" data-active="${kind === 'system'}">系统提示词</button><button type="button" data-prompt-kind="user" data-active="${kind === 'user'}">用户提示词</button></div>`, `
        <div class="dragon-caption-split-list">
            <aside><div class="dragon-caption-list-toolbar"><input class="dragon-input" data-prompt-search aria-label="搜索提示词" placeholder="搜索提示词" value="${escapeAttribute(query)}"><button class="dragon-icon-button" type="button" data-prompt-add title="添加提示词" aria-label="添加提示词">＋</button></div><small data-prompt-count>${visible.length}/${prompts.length} 个预设</small>
                <div data-prompt-list role="listbox">${prompts.map((prompt) => `<button type="button" role="option" aria-selected="${selected?.id === prompt.id}" data-prompt-id="${escapeAttribute(prompt.id)}" data-active="${selected?.id === prompt.id}" ${visible.includes(prompt) ? '' : 'hidden'}><strong>${escapeHtml(prompt.name)}</strong><small>${prompt.builtin ? '内置只读' : kind === 'system' ? '系统' : '用户'}</small></button>`).join('')}${prompts.length && !visible.length ? '<div class="dragon-empty-state" data-prompt-empty><p>没有匹配的提示词。</p></div>' : ''}</div>
            </aside>
            <form data-prompt-form>${state.workspaceData.promptNotice ? `<div class="dragon-caption-inline-status" role="status">${escapeHtml(state.workspaceData.promptNotice)}</div>` : ''}${renderEditor(selected, state)}</form>
        </div>`);
}

export function bindPromptsPanel(root, state) {
    root.querySelectorAll('[data-prompt-kind]').forEach((button) => button.addEventListener('click', () => { if (!discardPromptDraft(state)) return; state.workspaceData.promptKind = button.dataset.promptKind; state.workspaceData.promptId = ''; state.workspaceData.promptNotice = ''; state.suiteRender(); }));
    root.querySelectorAll('[data-prompt-id]').forEach((button) => button.addEventListener('click', () => { if (!discardPromptDraft(state)) return; state.workspaceData.promptId = button.dataset.promptId; state.workspaceData.promptNotice = ''; state.suiteRender(); }));
    root.querySelector('[data-prompt-search]')?.addEventListener('input', (event) => {
        const query = event.target.value.toLocaleLowerCase();
        state.workspaceData.promptQuery = event.target.value; let visible = 0;
        root.querySelectorAll('[data-prompt-id]').forEach((button) => { button.hidden = !button.textContent.toLocaleLowerCase().includes(query); if (!button.hidden) visible += 1; });
        const count = root.querySelector('[data-prompt-count]'); if (count) count.textContent = `${visible}/${root.querySelectorAll('[data-prompt-id]').length} 个预设`;
    });
    root.querySelector('[data-prompt-add]')?.addEventListener('click', () => {
        if (!discardPromptDraft(state)) return;
        const kind = state.workspaceData.promptKind || 'system';
        const prompt = {id: uid('prompt'), name: '新提示词', content: '', kind, builtin: false};
        state.workspace.prompts[kind].push(prompt); state.workspaceData.promptId = prompt.id; state.workspaceData.promptDirtyId = prompt.id; state.workspaceData.promptNewId = prompt.id; state.workspaceData.promptNotice = '新预设尚未保存'; state.suiteRender();
    });
    root.querySelector('[data-prompt-form]')?.addEventListener('input', (event) => { if (!event.target.matches('[name="name"], [name="content"]')) return; const id = root.querySelector('[data-prompt-editor-id]')?.value; if (id) { state.workspaceData.promptDirtyId = id; state.workspaceData.promptNotice = '当前预设有未保存修改'; } });
    root.querySelector('[data-prompt-form]')?.addEventListener('submit', async (event) => {
        event.preventDefault();
        const kind = state.workspaceData.promptKind || 'system';
        const prompt = state.workspace.prompts[kind].find((item) => item.id === event.currentTarget.querySelector('[data-prompt-editor-id]')?.value);
        if (!prompt || prompt.builtin) return;
        const name = event.currentTarget.elements.name.value.trim(); const content = event.currentTarget.elements.content.value.trim();
        if (!name || !content) return feedback(root, '名称和内容不能为空', 'error');
        prompt.name = name; prompt.content = content;
        try { await saveWorkspace(state); state.workspaceData.promptDirtyId = ''; state.workspaceData.promptNewId = ''; state.workspaceData.promptNotice = '提示词已保存'; state.suiteRender(); }
        catch (error) { feedback(root, error.message, 'error'); }
    });
    root.querySelector('[data-prompt-remove]')?.addEventListener('click', async () => {
        const kind = state.workspaceData.promptKind || 'system';
        const id = state.workspaceData.promptId; const original = [...state.workspace.prompts[kind]];
        const prompt = original.find((item) => item.id === id); if (!prompt || prompt.builtin) return;
        if (!window.confirm(`删除提示词“${prompt.name}”？`)) return;
        state.workspace.prompts[kind] = original.filter((item) => item.id !== id); state.workspaceData.promptId = '';
        try { await saveWorkspace(state); state.workspaceData.promptDirtyId = ''; state.workspaceData.promptNewId = ''; state.workspaceData.promptNotice = '提示词已删除'; state.suiteRender(); }
        catch (error) { state.workspace.prompts[kind] = original; state.workspaceData.promptId = id; feedback(root, error.message, 'error'); }
    });
}

function renderEditor(prompt, state) {
    if (!prompt) return '<div class="dragon-empty-state"><p>选择或添加一个提示词。</p></div>';
    const dirty = state.workspaceData.promptDirtyId === prompt.id;
    return `<input type="hidden" data-prompt-editor-id value="${escapeAttribute(prompt.id)}">${prompt.builtin ? '<div class="dragon-caption-inline-status">内置预设只读；可新建自定义版本。</div>' : ''}<label><span>名称${dirty ? ' · 未保存' : ''}</span><input class="dragon-input" name="name" value="${escapeAttribute(prompt.name)}" ${prompt.builtin ? 'readonly' : ''}></label><label><span>内容</span><textarea class="dragon-textarea" name="content" rows="16" ${prompt.builtin ? 'readonly' : ''}>${escapeHtml(prompt.content)}</textarea></label><div class="dragon-caption-form-actions">${prompt.builtin ? '' : '<button class="dragon-btn dragon-btn-primary" type="submit">保存</button><button class="dragon-btn dragon-btn-secondary" type="button" data-prompt-remove>删除</button>'}</div>`;
}

function discardPromptDraft(state) {
    const id = state.workspaceData.promptDirtyId; if (!id) return true;
    if (!window.confirm('当前提示词有未保存修改，放弃并继续？')) return false;
    if (state.workspaceData.promptNewId === id) { const kind = state.workspaceData.promptKind || 'system'; state.workspace.prompts[kind] = state.workspace.prompts[kind].filter((prompt) => prompt.id !== id); }
    state.workspaceData.promptDirtyId = ''; state.workspaceData.promptNewId = ''; return true;
}
