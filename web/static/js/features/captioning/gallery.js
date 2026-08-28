import { escapeAttribute, escapeHtml, stateLabel, statusTone } from './utils.js?v=dragon-ui-20260829v12';

export function renderGallery(state) {
    const job = state.selectedJob;
    if (!job) return `<aside class="dragon-caption-gallery"><div class="dragon-empty-state"><p>创建或选择任务后显示图片。</p></div></aside>`;
    const selected = new Set(state.workspaceData.gallerySelected || []);
    const results = visibleResults(state);
    return `<aside class="dragon-caption-gallery" aria-label="缩略图画廊">
        <header><div><span class="dragon-eyebrow">GALLERY</span><h2>${escapeHtml(job.directory.split(/[\\/]/).pop() || job.directory)}</h2></div><span>${job.total} 张</span></header>
        <div class="dragon-caption-gallery-tools"><select class="dragon-select" data-gallery-filter><option value="all">全部</option><option value="exportable">可导出</option><option value="failed">失败</option><option value="parse">解析失败</option><option value="selected">已选择</option></select><input class="dragon-input" data-gallery-search value="${escapeAttribute(state.workspaceData.gallerySearch || '')}" placeholder="搜索图片或标签"><button class="dragon-btn dragon-btn-secondary" type="button" data-gallery-select-visible>选择可见项</button></div>
        <div class="dragon-caption-gallery-grid" data-caption-gallery-grid>${results.map((item) => renderTile(item, job.results.indexOf(item), state.selectedItemId, selected.has(item.id))).join('') || '<div class="dragon-empty-state"><p>没有符合筛选条件的图片。</p></div>'}</div>
    </aside>`;
}

export function bindGallery(root, state, onSelect) {
    root.querySelector('[data-caption-gallery-grid]')?.addEventListener('click', (event) => {
        const checkbox = event.target.closest('[data-caption-select-item]');
        if (checkbox) {
            const values = new Set(state.workspaceData.gallerySelected || []);
            checkbox.checked ? values.add(checkbox.dataset.captionSelectItem) : values.delete(checkbox.dataset.captionSelectItem);
            state.workspaceData.gallerySelected = [...values];
            return;
        }
        const tile = event.target.closest('[data-caption-item]');
        if (tile) onSelect(tile.dataset.captionItem);
    });
    root.querySelector('[data-caption-gallery-grid]')?.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const delta = event.key === 'ArrowRight' ? 1 : -1;
        onSelect(adjacentItemId(state, delta));
    });
    const filter = root.querySelector('[data-gallery-filter]');
    if (filter) filter.value = state.workspaceData.galleryFilter || 'all';
    filter?.addEventListener('change', () => { state.workspaceData.galleryFilter = filter.value; state.actions.renderWorkspace(); });
    root.querySelector('[data-gallery-search]')?.addEventListener('input', (event) => { state.workspaceData.gallerySearch = event.target.value; state.actions.renderWorkspace(); });
    root.querySelector('[data-gallery-select-visible]')?.addEventListener('click', () => { state.workspaceData.gallerySelected = visibleResults(state).map((item) => item.id); state.actions.renderWorkspace(); });
}

export function adjacentItemId(state, delta) {
    const results = visibleResults(state);
    if (!results.length) return '';
    const index = Math.max(0, results.findIndex((item) => item.id === state.selectedItemId));
    return results[(index + delta + results.length) % results.length].id;
}

function renderTile(item, index, selectedItemId, checked) {
    return `<div class="dragon-caption-thumb-wrap"><label class="dragon-caption-thumb-select" title="选择"><input type="checkbox" data-caption-select-item="${escapeAttribute(item.id)}" ${checked ? 'checked' : ''}></label><button class="dragon-caption-thumb" type="button" data-caption-item="${escapeAttribute(item.id)}" data-active="${item.id === selectedItemId}" data-tone="${statusTone(item.state)}" title="${escapeAttribute(item.name)} · ${escapeAttribute(stateLabel(item.state))}">
        <img src="${escapeAttribute(item.image_url)}" alt="" loading="lazy"><span class="dragon-caption-thumb-status" aria-label="${escapeAttribute(stateLabel(item.state))}"></span><b>${String(index + 1).padStart(2, '0')}</b>
    </button></div>`;
}

function visibleResults(state) {
    const filter = state.workspaceData.galleryFilter || 'all';
    const query = String(state.workspaceData.gallerySearch || '').trim().toLocaleLowerCase();
    const selected = new Set(state.workspaceData.gallerySelected || []);
    return (state.selectedJob?.results || []).filter((item) => {
        if (filter === 'exportable' && !['ready', 'committed'].includes(item.state)) return false;
        if (filter === 'failed' && item.state !== 'failed') return false;
        if (filter === 'parse' && item.failure_kind !== 'parse') return false;
        if (filter === 'selected' && !selected.has(item.id)) return false;
        return !query || `${item.name} ${item.proposed_caption || ''} ${item.error || ''}`.toLocaleLowerCase().includes(query);
    });
}
