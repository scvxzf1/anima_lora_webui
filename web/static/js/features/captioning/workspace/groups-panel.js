import { escapeAttribute, escapeHtml, feedback, panelShell, saveWorkspace, uid } from './shared.js?v=dragon-ui-20260829v12';

export function renderGroupsPanel(state) {
    const groups = state.workspace?.groups || [];
    const scans = state.workspaceData.groupScans || {};
    const dirty = state.workspaceData.groupsDirty === true;
    return panelShell('DIRECTORIES', '文件目录管理', '<button class="dragon-btn dragon-btn-primary" type="button" data-group-add>＋ 添加目录组</button>', `
        <div class="dragon-caption-table" data-group-list>
            <div class="dragon-caption-table-head"><span>名称</span><span>目录</span><span>备注</span><span>图片</span><span></span></div>
            ${groups.map((group, index) => `<div class="dragon-caption-table-row" data-group-index="${index}" data-group-id="${escapeAttribute(group.id)}">
                <label><span>名称</span><input class="dragon-input" value="${escapeAttribute(group.name)}" data-group-field="name" aria-label="目录组名称"></label>
                <label><span>目录</span><input class="dragon-input" value="${escapeAttribute(group.path)}" data-group-field="path" aria-label="目录路径"></label>
                <label><span>备注</span><input class="dragon-input" value="${escapeAttribute(group.note)}" data-group-field="note" aria-label="目录组备注"></label>
                <span data-group-count>${scanLabel(scans[group.id])}</span>
                <div><button class="dragon-icon-button" type="button" data-group-pick title="选择目录" aria-label="选择目录">…</button><button class="dragon-icon-button" type="button" data-group-scan title="扫描图片" aria-label="扫描图片" ${scans[group.id]?.status === 'scanning' ? 'disabled' : ''}>↻</button><button class="dragon-icon-button" type="button" data-group-remove title="删除目录组" aria-label="删除目录组">×</button></div>
            </div>`).join('') || '<div class="dragon-empty-state"><p>尚未添加目录组。</p><button class="dragon-btn dragon-btn-primary" type="button" data-group-add-empty>添加第一个目录组</button></div>'}
        </div>
        <div class="dragon-caption-form-actions"><button class="dragon-btn dragon-btn-primary" type="button" data-group-save ${dirty ? '' : 'disabled'}>${dirty ? '保存目录组（未保存）' : '目录组已保存'}</button></div>`);
}

export function bindGroupsPanel(root, state) {
    const addGroup = () => {
        state.workspace.groups.push({id: uid('group'), name: '新目录组', path: '', note: ''});
        state.workspaceData.groupsDirty = true;
        state.suiteRender();
    };
    root.querySelector('[data-group-add]')?.addEventListener('click', addGroup);
    root.querySelector('[data-group-add-empty]')?.addEventListener('click', addGroup);
    root.querySelectorAll('[data-group-field]').forEach((input) => input.addEventListener('input', () => {
        const index = Number(input.closest('[data-group-index]').dataset.groupIndex);
        state.workspace.groups[index][input.dataset.groupField] = input.value;
        state.workspaceData.groupsDirty = true;
        const save = root.querySelector('[data-group-save]'); if (save) { save.disabled = false; save.textContent = '保存目录组（未保存）'; }
    }));
    root.querySelectorAll('[data-group-remove]').forEach((button) => button.addEventListener('click', () => {
        const index = Number(button.closest('[data-group-index]').dataset.groupIndex);
        const group = state.workspace.groups[index];
        if ((group.path || group.note) && !window.confirm(`删除目录组“${group.name}”？此操作将在保存后生效。`)) return;
        state.workspace.groups.splice(index, 1); state.workspaceData.groupsDirty = true;
        state.suiteRender();
    }));
    root.querySelectorAll('[data-group-pick]').forEach((button) => button.addEventListener('click', async () => {
        const row = button.closest('[data-group-index]'); const index = Number(row.dataset.groupIndex);
        try {
            const {captioningApi} = await import('./shared.js?v=dragon-ui-20260829v12');
            const payload = await captioningApi('/workspace/select-folder');
            if (payload.path) { state.workspace.groups[index].path = payload.path; state.workspaceData.groupsDirty = true; row.querySelector('[data-group-field="path"]').value = payload.path; const save = root.querySelector('[data-group-save]'); if (save) { save.disabled = false; save.textContent = '保存目录组（未保存）'; } }
        } catch (error) { feedback(root, error.message, 'error'); }
    }));
    root.querySelectorAll('[data-group-scan]').forEach((button) => button.addEventListener('click', async () => {
        const row = button.closest('[data-group-index]');
        const group = state.workspace.groups[Number(row.dataset.groupIndex)];
        if (!String(group.path || '').trim()) return feedback(root, '请先填写目录路径', 'error');
        state.workspaceData.groupScans ||= {}; state.workspaceData.groupScans[group.id] = {status: 'scanning'}; state.suiteRender();
        try {
            const {captioningApi, jsonOptions} = await import('./shared.js?v=dragon-ui-20260829v12');
            const payload = await captioningApi('/workspace/images', jsonOptions('POST', {directory: group.path}));
            state.workspaceData.groupScans[group.id] = {status: 'ready', count: payload.images.length};
        } catch (error) { state.workspaceData.groupScans[group.id] = {status: 'error', error: error.message}; }
        finally { state.suiteRender(); }
    }));
    root.querySelector('[data-group-save]')?.addEventListener('click', async (event) => {
        event.currentTarget.disabled = true;
        try { await saveWorkspace(state); state.workspaceData.groupsDirty = false; event.currentTarget.textContent = '目录组已保存'; event.currentTarget.disabled = true; feedback(root, `已保存 ${state.workspace.groups.length} 个目录组`, 'success'); }
        catch (error) { feedback(root, error.message, 'error'); }
        finally { event.currentTarget.disabled = false; }
    });
}

function scanLabel(scan) {
    if (scan?.status === 'scanning') return '扫描中…';
    if (scan?.status === 'error') return `<span title="${escapeAttribute(scan.error || '')}">扫描失败</span>`;
    if (scan?.status === 'ready') return `${scan.count || 0} 张`;
    return '未扫描';
}
