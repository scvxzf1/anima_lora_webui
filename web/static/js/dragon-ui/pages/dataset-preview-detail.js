/* Detail view for a dataset preview image.
 * The list stays mounted while this view is open so its loaded pages and
 * scroll position can be restored without another request.
 */

import { copyText } from '../../shared/dom.js?v=dragon-ui-20260812v35';
import { formatBytes } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { escapeHtml } from './dataset-editor-fields.js?v=dragon-ui-20260828v54';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

export function createDatasetPreviewDetailController(dialog = document.getElementById('dataset-preview-dialog')) {
    const listView = dialog?.querySelector('.dataset-preview-dialog-body');
    const form = dialog?.querySelector('form');
    if (!dialog || !listView || !form) return createNoopController();

    let detailView = null;
    let returnState = null;
    let activeImage = null;
    let restoreFrame = 0;

    const ensureView = () => {
        if (detailView) return detailView;
        detailView = document.createElement('section');
        detailView.className = 'dataset-preview-detail-view';
        detailView.dataset.datasetPreviewDetail = '';
        detailView.hidden = true;
        detailView.setAttribute('aria-labelledby', 'dataset-preview-detail-title');
        detailView.innerHTML = `
            <div class="dataset-preview-detail-toolbar">
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-dataset-preview-detail-back>
                    ${renderIcon('chevronUp', 'dragon-btn-icon dataset-preview-detail-back-icon')}
                    <span>返回列表</span>
                </button>
                <span class="dataset-preview-detail-position" data-dataset-preview-detail-position></span>
            </div>
            <div class="dataset-preview-detail-content">
                <div class="dataset-preview-detail-image-wrap">
                    <img data-dataset-preview-detail-image alt="数据集图片" decoding="async">
                </div>
                <aside class="dataset-preview-detail-info">
                    <div class="dataset-preview-detail-section">
                        <h3 id="dataset-preview-detail-title">图片信息</h3>
                        <dl data-dataset-preview-detail-meta></dl>
                    </div>
                    <section class="dataset-preview-detail-section dataset-preview-detail-caption">
                        <div class="dataset-preview-detail-section-head">
                            <h3>标注结果</h3>
                            <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-dataset-preview-detail-copy>
                                ${renderIcon('copy', 'dragon-btn-icon')}<span>复制标注</span>
                            </button>
                        </div>
                        <p class="dataset-preview-detail-caption-source" data-dataset-preview-detail-caption-source></p>
                        <pre data-dataset-preview-detail-caption-text></pre>
                    </section>
                </aside>
            </div>
        `;
        form.append(detailView);

        detailView.addEventListener('click', (event) => {
            const back = event.target?.closest?.('[data-dataset-preview-detail-back]');
            if (back) {
                restore();
                return;
            }
            const copy = event.target?.closest?.('[data-dataset-preview-detail-copy]');
            if (copy) void copyCaption(copy);
        });
        detailView.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape') return;
            event.preventDefault();
            restore();
        });
        return detailView;
    };

    const render = (image) => {
        const view = ensureView();
        const caption = image.caption && typeof image.caption === 'object' ? image.caption : {};
        const imageElement = view.querySelector('[data-dataset-preview-detail-image]');
        if (imageElement) {
            imageElement.src = String(image.url || '');
            imageElement.alt = String(image.name || '数据集图片');
            if (image.width) imageElement.width = Number(image.width);
            if (image.height) imageElement.height = Number(image.height);
        }

        const meta = view.querySelector('[data-dataset-preview-detail-meta]');
        if (meta) {
            meta.innerHTML = [
                detailMetaRow('文件名', image.name || '-'),
                detailMetaRow('完整路径', image.file || '-'),
                detailMetaRow('尺寸', image.width && image.height ? `${image.width} × ${image.height}px` : '-'),
                detailMetaRow('文件大小', image.size_bytes != null ? formatBytes(image.size_bytes) : '-'),
                detailMetaRow('修改时间', image.mtime_text || '-'),
            ].join('');
        }

        const source = view.querySelector('[data-dataset-preview-detail-caption-source]');
        if (source) {
            source.textContent = caption.ok
                ? [caption.format_label || caption.extension || '已识别标注', caption.file || ''].filter(Boolean).join(' · ')
                : `未找到标注 · ${caption.source_label || '自动识别'}`;
        }
        const captionText = view.querySelector('[data-dataset-preview-detail-caption-text]');
        if (captionText) captionText.textContent = caption.ok ? (caption.text || '(空标注)') : '未按当前标注来源找到 caption 文件';
        const copy = view.querySelector('[data-dataset-preview-detail-copy]');
        if (copy) copy.hidden = !caption.ok;

        const position = view.querySelector('[data-dataset-preview-detail-position]');
        if (position) position.textContent = String(image.name || '');
    };

    const open = (image, trigger = null) => {
        if (!image) return false;
        if (!returnState) {
            returnState = {
                top: Math.max(0, Number(listView.scrollTop) || 0),
                left: Math.max(0, Number(listView.scrollLeft) || 0),
                focus: trigger || (document.activeElement !== dialog ? document.activeElement : null),
            };
        }
        activeImage = image;
        render(image);
        listView.hidden = true;
        listView.setAttribute('aria-hidden', 'true');
        const view = ensureView();
        view.hidden = false;
        view.setAttribute('aria-hidden', 'false');
        view.querySelector('[data-dataset-preview-detail-back]')?.focus?.({ preventScroll: true });
        return true;
    };

    const restore = ({ focus = true, defer = true } = {}) => {
        if (!detailView || detailView.hidden) return false;
        const saved = returnState;
        returnState = null;
        activeImage = null;
        if (restoreFrame) {
            cancelFrame(restoreFrame);
            restoreFrame = 0;
        }
        detailView.hidden = true;
        detailView.setAttribute('aria-hidden', 'true');
        listView.hidden = false;
        listView.removeAttribute('aria-hidden');
        const apply = () => {
            restoreFrame = 0;
            if (saved) {
                if (focus && saved.focus && document.contains(saved.focus)) {
                    saved.focus.focus?.({ preventScroll: true });
                }
                // Focus restoration may scroll the nearest container on browsers
                // that ignore preventScroll; apply the captured coordinates last.
                applyScrollPosition(saved);
                if (defer && typeof window.requestAnimationFrame === 'function') {
                    restoreFrame = window.requestAnimationFrame(() => {
                        restoreFrame = 0;
                        applyScrollPosition(saved);
                    });
                }
            }
        };
        if (defer && typeof window.requestAnimationFrame === 'function') {
            restoreFrame = window.requestAnimationFrame(apply);
        } else {
            apply();
        }
        return true;
    };

    const applyScrollPosition = (saved) => {
        listView.scrollTop = saved.top;
        listView.scrollLeft = saved.left;
    };

    const handleCancel = (event) => {
        if (!isOpen()) return false;
        event.preventDefault();
        restore();
        return true;
    };

    const isOpen = () => Boolean(detailView && !detailView.hidden);

    const copyCaption = async (button) => {
        const text = activeImage?.caption?.ok ? String(activeImage.caption.text || '') : '';
        const label = button.querySelector('span');
        const original = label?.textContent || '复制标注';
        button.disabled = true;
        try {
            await copyText(text);
            if (label) label.textContent = '已复制';
        } catch (_error) {
            if (label) label.textContent = '复制失败';
        } finally {
            window.setTimeout(() => {
                if (!button.isConnected) return;
                if (label) label.textContent = original;
                button.disabled = false;
            }, 1000);
        }
    };

    const dispose = () => {
        if (restoreFrame) cancelFrame(restoreFrame);
        restoreFrame = 0;
        detailView?.remove();
        detailView = null;
        returnState = null;
        activeImage = null;
    };

    return { open, restore, isOpen, handleCancel, dispose };
}

function detailMetaRow(label, value) {
    return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`;
}

function cancelFrame(frame) {
    if (typeof window.cancelAnimationFrame === 'function') window.cancelAnimationFrame(frame);
}

function createNoopController() {
    return {
        open: () => false,
        restore: () => false,
        isOpen: () => false,
        handleCancel: () => false,
        dispose: () => {},
    };
}
