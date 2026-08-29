import { captioningApi, jsonOptions } from './api.js?v=dragon-ui-20260829v12';
import { adjacentItemId } from './gallery.js?v=dragon-ui-20260829v12';
import { escapeAttribute, escapeHtml, showFeedback, splitTags, stateLabel, withBusy } from './utils.js?v=dragon-ui-20260829v12';

export function renderInspector(job, item, editorMode = 'pills', zoom = 1) {
    if (!job || !item) return `<section class="dragon-caption-inspector"><div class="dragon-empty-state"><p>从画廊选择一张图片开始审阅。</p></div></section>`;
    const tagMode = job.output_mode === 'tags' || job.output_mode === 'style_trigger_json' || (job.output_mode === 'three_format' && job.output_variant === 'tag');
    const alertText = item.error || '';
    return `<section class="dragon-caption-inspector" aria-label="审阅与标注编辑器">
        <header><div><span class="dragon-eyebrow">INSPECTOR</span><h2>${escapeHtml(item.name)}</h2></div><div class="dragon-caption-inspector-status"><span data-tone="${item.state === 'failed' ? 'error' : 'success'}">${escapeHtml(stateLabel(item.state))}</span>${alertText ? `<button type="button" class="dragon-caption-alert" data-caption-open-attempts aria-label="查看错误详情">${escapeHtml(alertText)}</button>` : ''}</div></header>
        <div class="dragon-caption-image-stage" data-caption-image-stage><img src="${escapeAttribute(item.image_url)}" alt="${escapeAttribute(item.name)}" style="--caption-zoom:${zoom}" data-caption-preview></div>
        <div class="dragon-caption-zoom"><button type="button" data-caption-zoom="out" aria-label="缩小">−</button><output>${Math.round(zoom * 100)}%</output><button type="button" data-caption-zoom="in" aria-label="放大">＋</button><button type="button" data-caption-zoom="reset">适配</button></div>
        <div class="dragon-caption-editor-head"><div class="dragon-segmented"><button type="button" data-caption-editor-mode="pills" data-active="${editorMode === 'pills'}" ${tagMode ? '' : 'disabled'}>标签胶囊</button><button type="button" data-caption-editor-mode="raw" data-active="${editorMode === 'raw'}">纯文本</button></div><small>← → 快速切图</small></div>
        ${editorMode === 'pills' && tagMode ? renderPills(item) : `<textarea class="dragon-textarea dragon-caption-raw" data-caption-raw rows="7">${escapeHtml(item.proposed_caption || item.original_caption || '')}</textarea>`}
        ${renderVariants(item)}
        ${renderAttemptLog(item)}
        <footer><button class="dragon-btn dragon-btn-primary" type="button" data-caption-save-next ${editable(item) ? '' : 'disabled'}>保存并下一张</button><button class="dragon-btn dragon-btn-secondary" type="button" data-caption-retry-item ${['ready', 'failed'].includes(item.state) ? '' : 'disabled'}>重新请求此图</button><button class="dragon-btn dragon-btn-secondary" type="button" data-caption-save ${editable(item) ? '' : 'disabled'}>仅保存候选</button></footer>
    </section>`;
}

export function bindInspector(root, state, actions) {
    root.querySelectorAll('[data-caption-editor-mode]').forEach((button) => button.addEventListener('click', () => { state.editorMode = button.dataset.captionEditorMode; actions.renderInspector(); }));
    root.querySelectorAll('[data-caption-zoom]').forEach((button) => button.addEventListener('click', () => changeZoom(state, actions, button.dataset.captionZoom)));
    root.querySelector('[data-caption-image-stage]')?.addEventListener('wheel', (event) => { event.preventDefault(); changeZoom(state, actions, event.deltaY < 0 ? 'in' : 'out'); }, {passive: false});
    root.querySelector('[data-caption-pills]')?.addEventListener('click', (event) => { const remove = event.target.closest('[data-caption-remove-tag]'); if (remove) remove.closest('[data-caption-pill]').remove(); });
    root.querySelector('[data-caption-add-tag]')?.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ',') { event.preventDefault(); addTag(root, event.currentTarget); } });
    bindDrag(root);
    root.querySelector('[data-caption-save]')?.addEventListener('click', (event) => saveCurrent(event.currentTarget, root, state, actions, false));
    root.querySelector('[data-caption-save-next]')?.addEventListener('click', (event) => saveCurrent(event.currentTarget, root, state, actions, true));
    root.querySelector('[data-caption-retry-item]')?.addEventListener('click', (event) => retryItem(event.currentTarget, root, state, actions));
    root.querySelectorAll('[data-caption-use-variant]').forEach((button) => button.addEventListener('click', () => useVariant(button, root, state)));
    root.querySelector('[data-caption-open-attempts]')?.addEventListener('click', () => {
        const details = root.querySelector('[data-caption-attempts]');
        if (details) details.open = true;
    });
}

export function currentEditorValue(root) {
    const raw = root.querySelector('[data-caption-raw]');
    if (raw) return raw.value.trim();
    return [...root.querySelectorAll('[data-caption-pill]')].map((pill) => pill.dataset.captionPill).join(', ');
}

async function saveCurrent(button, root, state, actions, next) {
    await withBusy(button, async () => {
        try {
            const itemId = state.selectedItemId;
            await captioningApi(`/jobs/${encodeURIComponent(state.selectedJobId)}/items/${encodeURIComponent(itemId)}`, jsonOptions('PATCH', {proposed_caption: currentEditorValue(root)}));
            if (next) {
                const commit = await captioningApi(`/jobs/${encodeURIComponent(state.selectedJobId)}/commit`, jsonOptions('POST', {item_ids: [itemId], write_mode: 'replace', all: false}));
                if (commit.conflicts) throw new Error('标注文件已被外部修改，当前候选未写回');
            }
            const nextId = adjacentItemId(state, 1);
            await actions.refresh(true);
            if (next) actions.selectItem(nextId);
            showFeedback(root, next ? '已写回并切换到下一张' : '候选标注已保存', 'success');
        } catch (error) { showFeedback(root, error.message, 'error'); }
    });
}

async function retryItem(button, root, state, actions) {
    await withBusy(button, async () => {
        try {
            await captioningApi(`/jobs/${encodeURIComponent(state.selectedJobId)}/items/${encodeURIComponent(state.selectedItemId)}/retry`, jsonOptions('POST', {}));
            await actions.refresh(true);
        } catch (error) { showFeedback(root, error.message, 'error'); }
    });
}

function renderPills(item) {
    return `<div class="dragon-caption-pill-editor" data-caption-pills>${splitTags(item.proposed_caption || item.original_caption).map((tag) => `<span class="dragon-caption-pill" draggable="true" data-caption-pill="${escapeAttribute(tag)}"><span>${escapeHtml(tag)}</span><button type="button" data-caption-remove-tag aria-label="删除 ${escapeAttribute(tag)}">×</button></span>`).join('')}<input data-caption-add-tag placeholder="添加标签"></div>`;
}

function renderVariants(item) {
    const entries = Object.entries(item.caption_variants || {});
    if (entries.length <= 1) return '';
    return `<details class="dragon-caption-variants"><summary>模型返回的全部格式（${entries.length}）</summary>${entries.map(([type, caption]) => `<section><header><strong>${escapeHtml(type)}</strong><button class="dragon-btn dragon-btn-secondary" type="button" data-caption-use-variant="${escapeAttribute(type)}">采用</button></header><p>${escapeHtml(caption)}</p></section>`).join('')}</details>`;
}

function renderAttemptLog(item) {
    const entries = item.attempt_log || [];
    if (!entries.length) return '';
    return `<details class="dragon-caption-attempts" data-caption-attempts><summary>API 尝试记录（${entries.length}）</summary>${entries.map((entry) => `<p data-tone="${entry.ok ? 'success' : 'error'}"><strong>${escapeHtml(entry.channel || `步骤 ${entry.step}`)}</strong> · ${escapeHtml(entry.model || '')} · 第 ${escapeHtml(entry.attempt || 1)} 次${entry.status ? ` · HTTP ${escapeHtml(entry.status)}` : ''}${entry.error ? `<br>${escapeHtml(entry.error)}` : ''}</p>`).join('')}</details>`;
}

function useVariant(button, root, state) {
    const item = state.selectedJob?.results?.find((entry) => entry.id === state.selectedItemId);
    const caption = item?.caption_variants?.[button.dataset.captionUseVariant];
    const raw = root.querySelector('[data-caption-raw]');
    if (raw && caption) raw.value = caption;
    else if (caption) {
        state.editorMode = 'raw';
        item.proposed_caption = caption;
        state.actions.renderInspector();
    }
}

function addTag(root, input) {
    const tag = input.value.replace(/,$/, '').trim();
    if (!tag) return;
    input.insertAdjacentHTML('beforebegin', `<span class="dragon-caption-pill" draggable="true" data-caption-pill="${escapeAttribute(tag)}"><span>${escapeHtml(tag)}</span><button type="button" data-caption-remove-tag aria-label="删除 ${escapeAttribute(tag)}">×</button></span>`);
    input.value = '';
}

function bindDrag(root) {
    let dragged = null;
    root.querySelectorAll('[data-caption-pill]').forEach((pill) => {
        pill.addEventListener('dragstart', () => { dragged = pill; });
        pill.addEventListener('dragover', (event) => { event.preventDefault(); if (dragged && dragged !== pill) pill.before(dragged); });
    });
}

function editable(item) { return ['ready', 'committed'].includes(item.state); }
function changeZoom(state, actions, direction) { state.zoom = direction === 'reset' ? 1 : Math.max(0.5, Math.min(4, state.zoom + (direction === 'in' ? 0.25 : -0.25))); actions.renderInspector(); }
