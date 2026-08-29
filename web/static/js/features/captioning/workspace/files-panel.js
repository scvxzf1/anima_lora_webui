import { captioningApi, escapeAttribute, escapeHtml, feedback, groupOptions, jsonOptions, panelShell, selectedGroup } from './shared.js?v=dragon-ui-20260829v12';

export function renderFilesPanel(state) {
    const images = state.workspaceData.fileImages || [];
    const query = state.workspaceData.fileQuery || '';
    const visibleImages = images.filter((item) => item.name.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
    const selected = state.workspaceData.fileSelected || images[0]?.name || '';
    const group = selectedGroup(state, state.workspaceData.fileGroupId);
    const scanning = state.workspaceData.fileScanning === true;
    const scanned = state.workspaceData.fileScanned === true;
    const preview = group && selected ? `/api/captioning/workspace/image?directory=${encodeURIComponent(group.path)}&name=${encodeURIComponent(selected)}` : '';
    return panelShell('FILES', '目录图片浏览', '', `${state.workspaceData.fileNotice ? `<div class="dragon-caption-inline-status" role="status">${escapeHtml(state.workspaceData.fileNotice)}</div>` : ''}<div class="dragon-caption-list-toolbar">
        <select class="dragon-select" data-file-group aria-label="目录组" ${scanning ? 'disabled' : ''}><option value="">选择目录组</option>${groupOptions(state, state.workspaceData.fileGroupId)}</select>
        <input class="dragon-input" data-file-search aria-label="搜索文件名" placeholder="搜索文件名" value="${escapeAttribute(query)}" ${images.length ? '' : 'disabled'}>
        <button class="dragon-btn dragon-btn-secondary" type="button" data-file-scan ${scanning ? 'disabled' : ''}>${scanning ? '扫描中…' : '扫描目录'}</button>
        <button class="dragon-btn dragon-btn-primary" type="button" data-file-open-workbench ${group ? '' : 'disabled'}>在审阅台打标</button>
        <span class="dragon-caption-list-summary" data-file-count>${visibleImages.length}/${images.length} 张</span>
    </div><div class="dragon-caption-file-browser" aria-busy="${scanning}">
        <div class="dragon-caption-file-grid" data-file-grid aria-label="目录图片">${images.map((item) => `<button type="button" data-file-item="${escapeAttribute(item.name)}" data-active="${item.name === selected}" aria-pressed="${item.name === selected}" aria-label="查看 ${escapeAttribute(item.name)}" title="${escapeAttribute(item.name)}" ${visibleImages.includes(item) ? '' : 'hidden'}><img src="/api/captioning/workspace/image?directory=${encodeURIComponent(group?.path || '')}&name=${encodeURIComponent(item.name)}" alt="${escapeAttribute(item.name)}"><span>${escapeHtml(item.name)}</span></button>`).join('') || renderEmptyState({group, scanned, scanning, error: state.workspaceData.fileError})}<div class="dragon-empty-state" data-file-search-empty ${images.length && !visibleImages.length ? '' : 'hidden'}><p>没有匹配“${escapeHtml(query)}”的图片。</p></div></div>
        <div class="dragon-caption-file-preview">${preview ? `<img src="${escapeAttribute(preview)}" alt="${escapeAttribute(selected)}"><strong>${escapeHtml(selected)}</strong>` : '<div class="dragon-empty-state"><p>选择图片查看预览。</p></div>'}</div>
    </div>`);
}

export function bindFilesPanel(root, state) {
    const groupSelect = root.querySelector('[data-file-group]');
    groupSelect?.addEventListener('change', () => { state.workspaceData.fileGroupId = groupSelect.value; state.workspaceData.fileImages = []; state.workspaceData.fileSelected = ''; state.workspaceData.fileQuery = ''; state.workspaceData.fileScanned = false; state.workspaceData.fileError = ''; state.workspaceData.fileNotice = groupSelect.value ? '目录组已切换，请点击“扫描目录”加载图片。' : ''; state.suiteRender(); });
    root.querySelector('[data-file-scan]')?.addEventListener('click', async () => {
        const group = selectedGroup(state, groupSelect.value);
        if (!group) return feedback(root, '请选择目录组', 'error');
        if (state.workspaceData.fileScanning) return;
        state.workspaceData.fileScanning = true; state.workspaceData.fileError = ''; state.suiteRender();
        try {
            const payload = await captioningApi('/workspace/images', jsonOptions('POST', {directory: group.path}));
            state.workspaceData.fileGroupId = group.id; state.workspaceData.fileImages = payload.images; state.workspaceData.fileSelected = payload.images[0]?.name || ''; state.workspaceData.fileScanned = true; state.workspaceData.fileNotice = payload.images.length ? `已扫描 ${payload.images.length} 张图片` : '扫描完成，但目录中没有支持的图片。';
        } catch (error) { state.workspaceData.fileError = error.message; }
        finally { state.workspaceData.fileScanning = false; state.suiteRender(); }
    });
    const grid = root.querySelector('[data-file-grid]');
    grid?.addEventListener('click', (event) => { const item = event.target.closest('[data-file-item]'); if (item) { state.workspaceData.fileSelected = item.dataset.fileItem; state.suiteRender(); } });
    grid?.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        const visible = [...grid.querySelectorAll('[data-file-item]:not([hidden])')]; const current = visible.indexOf(event.target.closest('[data-file-item]')); if (current < 0) return;
        event.preventDefault(); const next = visible[Math.max(0, Math.min(visible.length - 1, current + (event.key === 'ArrowRight' ? 1 : -1)))]; state.workspaceData.fileSelected = next.dataset.fileItem; state.suiteRender(); requestAnimationFrame(() => root.querySelector(`[data-file-item="${CSS.escape(next.dataset.fileItem)}"]`)?.focus());
    });
    root.querySelector('[data-file-search]')?.addEventListener('input', (event) => {
        const query = event.target.value.toLocaleLowerCase(); let visible = 0;
        state.workspaceData.fileQuery = event.target.value;
        root.querySelectorAll('[data-file-item]').forEach((item) => { item.hidden = query && !item.dataset.fileItem.toLocaleLowerCase().includes(query); if (!item.hidden) visible += 1; });
        const empty = root.querySelector('[data-file-search-empty]'); if (empty) { empty.hidden = visible > 0; const message = empty.querySelector('p'); if (message) message.textContent = `没有匹配“${event.target.value}”的图片。`; }
        const count = root.querySelector('[data-file-count]'); if (count) count.textContent = `${visible}/${state.workspaceData.fileImages.length} 张`;
    });
    root.querySelector('[data-file-open-workbench]')?.addEventListener('click', () => { const group = selectedGroup(state, groupSelect.value); if (!group) return; state.workspaceData.workbenchDirectory = group.path; state.activePanel = 'workbench'; state.suiteRender(); });
    root.querySelector('[data-file-open-groups]')?.addEventListener('click', () => { state.activePanel = 'groups'; state.suiteRender(); });
}

function renderEmptyState({group, scanned, scanning, error}) {
    if (scanning) return '<div class="dragon-empty-state"><p>正在扫描目录图片…</p></div>';
    if (error) return `<div class="dragon-empty-state" data-tone="error"><p>扫描失败：${escapeHtml(error)}</p></div>`;
    if (group && scanned) return '<div class="dragon-empty-state"><p>该目录没有支持的图片。</p></div>';
    return '<div class="dragon-empty-state"><p>选择目录组并扫描图片。</p><button class="dragon-btn dragon-btn-secondary dragon-caption-empty-action" type="button" data-file-open-groups>管理目录组</button></div>';
}
