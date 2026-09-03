import { createDatasetPreviewDetailController } from './dataset-preview-detail.js?v=dragon-ui-20260902v4';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v36';
import { renderCaptionEditor } from './tagging-results-editor.js?v=dragon-ui-20260902v1';

export function renderTaggingResultImageDialog() {
    return `<dialog class="preview-dialog dataset-preview-dialog dragon-tagging-result-preview-dialog" data-results-image-dialog>
        <form method="dialog">
            <div class="preview-dialog-header">
                <div><h2>图片预览</h2><p>查看原图与当前候选标注。</p></div>
                <div class="dataset-preview-dialog-actions"><button class="btn btn-small" value="close" type="submit">关闭</button></div>
            </div>
            <div class="dataset-preview-dialog-body" hidden aria-hidden="true"></div>
        </form>
    </dialog>`;
}

export function mountTaggingResultImagePreview(root, { getImage, getEditor } = {}) {
    const dialog = root?.querySelector?.('[data-results-image-dialog]');
    if (!dialog) return createNoopPreview();
    let activeItemId = '';
    let activeTrigger = null;
    const detailOptions = {
        resolveImage: (image) => {
            const current = typeof getImage === 'function' ? getImage(image?.id || '') : null;
            return current || image;
        },
    };
    if (typeof getEditor === 'function') {
        detailOptions.editor = {
            render: (image) => renderResultPreviewEditor(image, getEditor),
            sync: (view, image) => syncResultPreviewEditor(view, image, getEditor),
        };
    }
    const detail = createDatasetPreviewDetailController(dialog, detailOptions);

    const restore = () => detail.restore({ focus: true, defer: false });
    const close = () => {
        if (dialog.open && typeof dialog.close === 'function') dialog.close();
        else {
            dialog.hidden = true;
            dialog.removeAttribute('open');
            restore();
        }
    };
    const handleClick = (event) => {
        if (!event.target?.closest?.('[data-dataset-preview-detail-back]')) return;
        event.preventDefault();
        event.stopPropagation();
        close();
    };
    const handleKeydown = (event) => {
        if (event.key !== 'Escape') return;
        event.preventDefault();
        event.stopPropagation();
        close();
    };
    const handleClose = () => {
        restore();
        activeItemId = '';
        activeTrigger = null;
    };

    dialog.addEventListener('click', handleClick, true);
    dialog.addEventListener('keydown', handleKeydown, true);
    dialog.addEventListener('close', handleClose);

    const open = (itemId, trigger = null) => {
        const image = typeof getImage === 'function' ? getImage(itemId) : null;
        if (!image?.url) return false;
        if (dialog.open) close();
        activeItemId = String(itemId || '');
        activeTrigger = trigger || null;
        dialog.hidden = false;
        if (typeof dialog.showModal === 'function') dialog.showModal();
        else dialog.setAttribute('open', '');
        const opened = detail.open(image, trigger);
        const backLabel = dialog.querySelector('[data-dataset-preview-detail-back] span');
        if (backLabel) backLabel.textContent = '关闭预览';
        return opened;
    };

    const dispose = () => {
        if (dialog.open && typeof dialog.close === 'function') dialog.close();
        else restore();
        dialog.removeEventListener('click', handleClick, true);
        dialog.removeEventListener('keydown', handleKeydown, true);
        dialog.removeEventListener('close', handleClose);
        detail.dispose();
        activeItemId = '';
        activeTrigger = null;
    };

    return {
        open,
        close,
        refresh: () => detail.refresh(),
        sync: () => detail.sync(),
        isOpen: () => detail.isOpen(),
        getActiveItemId: () => activeItemId,
        getActiveTrigger: () => activeTrigger,
        dispose,
    };
}

function renderResultPreviewEditor(image, getEditor) {
    const config = typeof getEditor === 'function' ? getEditor(image) || {} : {};
    const itemId = String(config.itemId || image?.id || '');
    if (!itemId) return '';
    const text = String(config.text ?? image?.caption?.text ?? '');
    const mode = config.mode === 'raw' ? 'raw' : 'tags';
    const busy = Boolean(config.busy);
    const saving = Boolean(config.saving);
    const translating = Boolean(config.translating);
    const dirty = Boolean(config.dirty);
    const language = config.language === 'zh' ? 'zh' : 'en';
    return `<div class="dragon-tagging-result-preview-editor" data-result-preview-editor data-item-id="${escapeAttribute(itemId)}">
        <div class="dragon-tagging-result-preview-mode-row"><span>编辑标注</span><div class="dragon-segmented" role="group" aria-label="预览标注查看模式"><button type="button" data-results-mode="tags" data-active="${mode === 'tags'}">${renderIcon('tags')}<span>Tag</span></button><button type="button" data-results-mode="raw" data-active="${mode === 'raw'}">${renderIcon('list')}<span>原文</span></button></div></div>
        <div data-result-preview-caption-editor>${renderCaptionEditor({ itemId, text, mode, busy, saving })}</div>
        <footer class="dragon-tagging-result-preview-actions"><div><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-result-translate data-item-id="${escapeAttribute(itemId)}" ${busy || translating ? 'disabled' : ''}>${renderIcon('languages', 'dragon-btn-icon')}<span data-result-translate-label>${translating ? '翻译中…' : language === 'en' ? '中文' : 'EN'}</span></button><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-result-save data-item-id="${escapeAttribute(itemId)}" ${busy || saving || !dirty ? 'disabled' : ''}>${renderIcon('save', 'dragon-btn-icon')}<span data-result-save-label>${saving ? '保存中…' : '保存修改'}</span></button></div></footer>
    </div>`;
}

function syncResultPreviewEditor(view, image, getEditor) {
    const config = typeof getEditor === 'function' ? getEditor(image) || {} : {};
    const itemId = String(config.itemId || image?.id || '');
    if (!itemId) return;
    const busy = Boolean(config.busy);
    const saving = Boolean(config.saving);
    const translating = Boolean(config.translating);
    const dirty = Boolean(config.dirty);
    const editorHost = view.querySelector('[data-result-preview-caption-editor]');
    if (editorHost && !editorHost.contains(globalThis.document?.activeElement)) {
        editorHost.innerHTML = renderCaptionEditor({
            itemId,
            text: String(config.text ?? image?.caption?.text ?? ''),
            mode: config.mode === 'raw' ? 'raw' : 'tags',
            busy,
            saving,
        });
    }
    view.querySelectorAll('[data-result-save]').forEach((button) => {
        if (button.dataset.itemId !== itemId) return;
        button.disabled = busy || saving || !dirty;
        const label = button.querySelector('[data-result-save-label]');
        if (label) label.textContent = saving ? '保存中…' : '保存修改';
    });
    view.querySelectorAll('[data-result-translate]').forEach((button) => {
        if (button.dataset.itemId !== itemId) return;
        button.disabled = busy || translating;
        const label = button.querySelector('[data-result-translate-label]');
        if (label) label.textContent = translating ? '翻译中…' : config.language === 'zh' ? 'EN' : '中文';
    });
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}

function createNoopPreview() {
    return {
        open: () => false,
        close: () => {},
        refresh: () => false,
        sync: () => false,
        isOpen: () => false,
        getActiveItemId: () => '',
        getActiveTrigger: () => null,
        dispose: () => {},
    };
}
