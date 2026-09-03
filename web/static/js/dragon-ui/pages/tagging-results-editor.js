import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

export function splitCaptionTags(value) {
    return String(value || '').split(/[\n,]+/).map((tag) => tag.trim()).filter(Boolean);
}

export function joinCaptionTags(tags) {
    return (Array.isArray(tags) ? tags : []).map((tag) => String(tag || '').trim()).filter(Boolean).join(', ');
}

export function replaceCaptionTag(value, index, nextTag) {
    const tags = splitCaptionTags(value);
    if (index < 0 || index >= tags.length) return joinCaptionTags(tags);
    const clean = String(nextTag || '').trim();
    if (clean) tags[index] = clean;
    else tags.splice(index, 1);
    return joinCaptionTags(tags);
}

export function removeCaptionTag(value, index) {
    const tags = splitCaptionTags(value);
    if (index >= 0 && index < tags.length) tags.splice(index, 1);
    return joinCaptionTags(tags);
}

export function appendCaptionTag(value, nextTag) {
    const tags = splitCaptionTags(value);
    const clean = String(nextTag || '').trim().replace(/,+$/g, '').trim();
    if (clean) tags.push(clean);
    return joinCaptionTags(tags);
}

export function moveCaptionTag(value, fromIndex, toIndex) {
    const tags = splitCaptionTags(value);
    if (fromIndex < 0 || fromIndex >= tags.length || toIndex < 0 || toIndex >= tags.length || fromIndex === toIndex) {
        return joinCaptionTags(tags);
    }
    const [tag] = tags.splice(fromIndex, 1);
    tags.splice(toIndex, 0, tag);
    return joinCaptionTags(tags);
}

export function renderCaptionEditor({ itemId, text, mode, busy, saving }) {
    if (mode === 'raw') {
        return `<label class="dragon-field dragon-tagging-raw-editor"><span>原始文本</span><textarea class="dragon-textarea" rows="7" data-result-caption data-item-id="${escapeAttribute(itemId)}" ${busy ? 'disabled' : ''}>${escapeHtml(text)}</textarea></label>`;
    }
    const tags = splitCaptionTags(text);
    return `<div class="dragon-tagging-chip-editor" data-result-chip-editor data-item-id="${escapeAttribute(itemId)}" data-disabled="${busy}">
        <div class="dragon-tagging-chip-list" data-result-chip-list role="list" aria-label="标签顺序">
            ${tags.length ? tags.map((tag, index) => renderTagChip(itemId, tag, index, busy)).join('') : '<span class="dragon-tagging-chip-empty">暂无标签</span>'}
        </div>
        <div class="dragon-tagging-chip-add"><input class="dragon-input" type="text" data-result-tag-add data-item-id="${escapeAttribute(itemId)}" placeholder="添加 tag" ${busy ? 'disabled' : ''}><button class="dragon-icon-button" type="button" data-result-tag-add-button data-item-id="${escapeAttribute(itemId)}" aria-label="添加标签" title="添加标签" ${busy ? 'disabled' : ''}>${renderIcon('plus')}</button></div>
        <input type="hidden" data-result-caption-value data-item-id="${escapeAttribute(itemId)}" value="${escapeAttribute(text)}">
        ${saving ? '<span class="dragon-tagging-editor-saving">正在保存…</span>' : ''}
    </div>`;
}

function renderTagChip(itemId, tag, index, busy) {
    return `<span class="dragon-tagging-chip" role="listitem" draggable="${busy ? 'false' : 'true'}" data-result-tag data-item-id="${escapeAttribute(itemId)}" data-tag-index="${index}">
        <span class="dragon-tagging-chip-grip" aria-hidden="true">${renderIcon('grip')}</span>
        <span class="dragon-tagging-chip-text" contenteditable="${busy ? 'false' : 'plaintext-only'}" spellcheck="false" data-result-tag-text>${escapeHtml(tag)}</span>
        <button class="dragon-tagging-chip-remove" type="button" data-result-tag-remove data-item-id="${escapeAttribute(itemId)}" data-tag-index="${index}" aria-label="删除 ${escapeAttribute(tag)}" title="删除标签" ${busy ? 'disabled' : ''}>${renderIcon('x')}</button>
    </span>`;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
