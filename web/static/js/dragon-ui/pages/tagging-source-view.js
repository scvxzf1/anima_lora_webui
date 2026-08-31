/* Dataset selection panel with paged image rendering. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

export function renderTaggingSource(state, { jobBusy = false } = {}) {
    return `<details class="dragon-tagging-source dragon-section" data-tagging-source-details ${state.sourceExpanded ? 'open' : ''}>
        <summary class="dragon-tagging-panel-summary">
            <span class="dragon-tagging-summary-icon">${renderIcon('panels')}</span>
            <span><span class="dragon-eyebrow">SOURCE</span><strong>选择图片</strong><small data-tagging-source-status>${sourceStatus(state)}</small></span>
            <span class="dragon-tagging-summary-chevron">${renderIcon('chevronDown')}</span>
        </summary>
        ${renderTaggingSourceBody(state, { jobBusy })}
    </details>`;
}

export function renderTaggingSourceBody(state, { jobBusy = false } = {}) {
    const loaded = state.images.length;
    const selected = state.selectedFiles.size;
    const total = Number(state.total || loaded);
    return `<div class="dragon-tagging-source-body">
        <div class="dragon-tagging-dataset-controls">
            <label class="dragon-field"><span>数据集预设</span><select class="dragon-select" data-tagging-dataset ${state.loadingPreset || jobBusy ? 'disabled' : ''}>${datasetOptions(state)}</select></label>
            <label class="dragon-field"><span>图片组</span><select class="dragon-select" data-tagging-index ${state.rows.length && !jobBusy ? '' : 'disabled'}>${rowOptions(state)}</select></label>
            <label class="dragon-field"><span>读取目录</span><select class="dragon-select" data-tagging-source ${jobBusy ? 'disabled' : ''}><option value="source" ${state.source === 'source' ? 'selected' : ''}>原始图目录</option><option value="training" ${state.source === 'training' ? 'selected' : ''}>训练图目录</option></select></label>
        </div>
        <div class="dragon-tagging-source-toolbar">
            <span class="dragon-tagging-directory" title="${escapeAttribute(state.directory || '')}">${escapeHtml(state.directory || '未选择图片目录')}</span>
            <div>
                <button class="dragon-icon-button" type="button" data-tagging-refresh aria-label="刷新图片" title="刷新图片" ${state.loadingImages ? 'disabled' : ''}>${renderIcon('refresh')}</button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-tagging-select-all ${total && !jobBusy && !state.selectingAll ? '' : 'disabled'}>${state.selectingAll ? '<span class="dragon-spinner" aria-hidden="true"></span>' : renderIcon('check', 'dragon-btn-icon')}<span>全选</span></button>
                <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-tagging-clear ${selected && !jobBusy ? '' : 'disabled'}><span>清空</span></button>
            </div>
        </div>
        <div class="dragon-tagging-selection-status" data-tagging-selection-status role="status" aria-live="polite">${selectionStatus(state)}</div>
        <div class="dragon-tagging-image-grid" data-tagging-image-grid aria-live="polite">${renderImageGrid(state)}</div>
    </div>`;
}

export function syncTaggingSource(root, state, { jobBusy = isJobBusy(state) } = {}) {
    const body = root?.querySelector?.('.dragon-tagging-source-body');
    if (!body) return false;
    const currentGrid = body.querySelector?.('[data-tagging-image-grid]');
    const scrollTop = Number(currentGrid?.scrollTop || state.gridScrollTop || 0);
    body.innerHTML = renderTaggingSourceBody(state, { jobBusy });
    const nextGrid = body.querySelector?.('[data-tagging-image-grid]');
    if (nextGrid) nextGrid.scrollTop = scrollTop;
    syncTaggingSourceState(root, state);
    return true;
}

export function appendTaggingImageCards(root, state, images) {
    const grid = root?.querySelector?.('[data-tagging-image-grid]');
    if (!grid || !Array.isArray(images) || !images.length) return false;
    const placeholder = grid.querySelector('[data-tagging-grid-placeholder]');
    placeholder?.remove();
    const sentinel = grid.querySelector('[data-tagging-load-sentinel]');
    const markup = images.map((image) => renderImageCard(image, state.selectedFiles)).join('');
    if (sentinel) sentinel.insertAdjacentHTML('beforebegin', markup);
    else grid.insertAdjacentHTML('beforeend', markup);
    syncTaggingSourceState(root, state);
    return true;
}

export function syncTaggingSourceState(root, state) {
    root?.querySelectorAll?.('[data-tagging-image]').forEach((input) => {
        const selected = state.selectedFiles.has(input.dataset.file || '');
        input.checked = selected;
        input.closest('.dragon-tagging-image-card')?.setAttribute('data-selected', String(selected));
    });
    root?.querySelectorAll?.('[data-tagging-selected-count]').forEach((node) => {
        node.textContent = String(state.selectedFiles.size);
    });
    const status = root?.querySelector?.('[data-tagging-selection-status]');
    if (status) status.textContent = selectionStatus(state);
    const sourceSummary = root?.querySelector?.('[data-tagging-source-status]');
    if (sourceSummary) sourceSummary.textContent = sourceStatus(state);
    const imageLoadStatus = root?.querySelector?.('[data-tagging-image-load-status]');
    if (imageLoadStatus) imageLoadStatus.textContent = imageLoadStatusText(state);
    const clear = root?.querySelector?.('[data-tagging-clear]');
    if (clear) clear.disabled = !state.selectedFiles.size || isJobBusy(state);
    updateLoadSentinel(root, state);
}

export function updateLoadSentinel(root, state) {
    const sentinel = root?.querySelector?.('[data-tagging-load-sentinel]');
    if (!sentinel) return;
    sentinel.dataset.loading = String(Boolean(state.loadingMore));
    sentinel.hidden = !state.hasMore;
    sentinel.innerHTML = state.loadingMore
        ? '<span class="dragon-spinner" aria-hidden="true"></span><span>正在加载…</span>'
        : `<button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-tagging-load-more>${renderIcon('chevronDown', 'dragon-btn-icon')}<span>加载更多</span></button>`;
}

export function observeTaggingImageSentinel(root, onVisible) {
    const sentinel = root?.querySelector?.('[data-tagging-load-sentinel]');
    if (!sentinel || typeof IntersectionObserver !== 'function') return () => {};
    const scrollRoot = root.querySelector('[data-tagging-image-grid]');
    const observer = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) onVisible?.();
    }, { root: scrollRoot, rootMargin: '280px 0px', threshold: 0.01 });
    observer.observe(sentinel);
    return () => observer.disconnect();
}

function renderImageGrid(state) {
    if (state.loadingImages && !state.images.length) {
        return '<div class="dragon-tagging-grid-state" data-tagging-grid-placeholder><span class="dragon-spinner" aria-hidden="true"></span><strong>正在扫描图片…</strong></div>';
    }
    if (!state.images.length) {
        const label = state.error || (state.directory ? '当前目录没有可用图片' : '先选择数据集预设');
        return `<div class="dragon-tagging-grid-state" data-tagging-grid-placeholder><span class="dragon-empty-state-icon">${renderIcon('folder')}</span><strong>${escapeHtml(label)}</strong></div>`;
    }
    const cards = state.images.map((image) => renderImageCard(image, state.selectedFiles)).join('');
    return `${cards}<div class="dragon-tagging-load-sentinel" data-tagging-load-sentinel ${state.hasMore ? '' : 'hidden'}>${state.loadingMore ? '<span class="dragon-spinner" aria-hidden="true"></span><span>正在加载…</span>' : `<button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-tagging-load-more>${renderIcon('chevronDown', 'dragon-btn-icon')}<span>加载更多</span></button>`}</div>`;
}

function renderImageCard(image, selectedFiles) {
    const file = String(image.file || '');
    const selected = selectedFiles.has(file);
    const caption = image.caption || {};
    return `<label class="dragon-tagging-image-card" data-selected="${selected}"><input type="checkbox" data-tagging-image data-file="${escapeAttribute(file)}" ${selected ? 'checked' : ''}><span class="dragon-tagging-image-media"><img src="${escapeAttribute(image.thumbnail_url || image.url || '')}" alt="${escapeAttribute(image.name || '数据集图片')}" width="320" height="180" loading="lazy" decoding="async"></span><span class="dragon-tagging-image-name" title="${escapeAttribute(file)}">${escapeHtml(image.name || file)}</span><small>${caption.ok ? '已有标注' : '缺少标注'}</small></label>`;
}

function selectionStatus(state) {
    const total = Number(state.total || state.images.length);
    const selected = state.selectedFiles.size;
    const limitSuffix = total > 500 ? ' · 单次任务上限 500' : '';
    return `${selected} / ${total} 已选${limitSuffix}`;
}

function sourceStatus(state) {
    const loaded = state.images.length;
    const selected = state.selectedFiles.size;
    const total = Number(state.total || loaded);
    const progress = state.imagesLoaded ? `${loaded}/${total} 已加载` : state.loadingImages ? '正在加载…' : '展开后加载';
    return `${selected} 已选 · ${progress}`;
}

function imageLoadStatusText(state) {
    if (state.loadingImages) return '正在扫描…';
    return state.imagesLoaded ? `${Number(state.total || 0)} 张` : '展开后扫描';
}

function datasetOptions(state) {
    if (!state.presets.length) return '<option value="">没有可用预设</option>';
    return state.presets.map((item) => {
        const file = item.path || item.file || '';
        return `<option value="${escapeAttribute(file)}" ${file === state.datasetFile ? 'selected' : ''}>${escapeHtml(item.name || shortName(file))}</option>`;
    }).join('');
}

function rowOptions(state) {
    if (!state.rows.length) return '<option value="0">-</option>';
    return state.rows.map((_row, index) => `<option value="${index}" ${index === Number(state.datasetIndex) ? 'selected' : ''}>第 ${index + 1} 组</option>`).join('');
}

function isJobBusy(state) {
    return state.submitting || ['queued', 'running'].includes(state.job?.state);
}

function shortName(value) {
    const clean = String(value || '').replaceAll('\\', '/');
    return clean.split('/').pop() || clean;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
