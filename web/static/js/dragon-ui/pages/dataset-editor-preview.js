/* Image and caption preview for the Dragon dataset workspace. */

import { escapeAttribute, escapeHtml } from './dataset-editor-fields.js?v=dragon-ui-20260814v43';

export async function openDatasetPreview(api, state, datasetIndex) {
    const file = state.selectedFile || state.datasetConfig;
    if (!file) throw new Error('请先保存或选择一个数据集预设，再打开图片预览');
    if (state.dirty) throw new Error('当前数据集有未保存修改，请先保存后再预览');
    const dialog = document.getElementById('dataset-preview-dialog');
    if (!dialog) throw new Error('找不到数据集预览窗口，请刷新页面');
    const params = new URLSearchParams({ file, dataset_index: String(datasetIndex), source: 'source', limit: '120' });
    renderPreview({ loading: true, datasetIndex });
    if (dialog.showModal && !dialog.open) dialog.showModal();
    const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
    if (payload.ok === false) throw new Error(payload.error || '读取数据集预览失败');
    state.previewIndex = datasetIndex;
    renderPreview({ payload, datasetIndex });
}

export function bindDatasetPreviewRefresh(api, state) {
    const button = document.getElementById('btn-refresh-dataset-preview');
    if (!button) return;
    if (button._dragonDatasetPreviewHandler) {
        button.removeEventListener('click', button._dragonDatasetPreviewHandler);
    }
    button._dragonDatasetPreviewHandler = async () => {
        button.disabled = true;
        try { await openDatasetPreview(api, state, state.previewIndex || 0); } catch (error) { renderPreview({ error: error.message, datasetIndex: state.previewIndex || 0 }); } finally { button.disabled = false; }
    };
    button.addEventListener('click', button._dragonDatasetPreviewHandler);
}

function renderPreview({ payload = null, loading = false, error = '', datasetIndex = 0 }) {
    const title = document.getElementById('dataset-preview-dialog-title');
    const meta = document.getElementById('dataset-preview-dialog-meta');
    const details = document.getElementById('dataset-preview-details');
    const grid = document.getElementById('dataset-preview-grid');
    const empty = document.getElementById('dataset-preview-empty');
    if (!title || !meta || !details || !grid || !empty) return;
    title.textContent = `第 ${datasetIndex + 1} 组数据集预览`;
    grid.innerHTML = '';
    details.innerHTML = '';
    if (loading) {
        meta.textContent = '正在读取图片与标注…';
        empty.hidden = false;
        empty.textContent = '正在扫描数据集目录…';
        return;
    }
    if (error) {
        meta.textContent = error;
        empty.hidden = false;
        empty.textContent = error;
        return;
    }
    const images = Array.isArray(payload?.images) ? payload.images : [];
    meta.textContent = `${payload.source_label || '原始图目录'} · ${payload.directory || '-'} · ${images.length}/${payload.total || 0} 张 · ${payload.caption_summary || '未识别到标注'}`;
    const row = payload.row || {};
    const settings = payload.settings || row.settings || {};
    [
        ['数据集文件', payload.file || '-'],
        ['当前目录', payload.directory || '-'],
        ['原始路径', row.source_dir || '-'],
        ['重复次数', row.num_repeats ?? '-'],
        ['分辨率', settings.resolution ? `${settings.resolution}px` : '-'],
        ['标注来源', payload.caption_source_label || '-'],
    ].forEach(([label, value]) => { details.insertAdjacentHTML('beforeend', `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`); });
    if (!images.length) {
        empty.hidden = false;
        empty.textContent = payload.message || '当前目录没有可预览图片。';
        return;
    }
    empty.hidden = true;
    grid.innerHTML = images.map(renderPreviewCard).join('');
    grid.querySelectorAll('[data-caption-copy]').forEach((button) => button.addEventListener('click', async () => {
        const text = button.closest('.dataset-preview-caption')?.querySelector('pre')?.textContent || '';
        await navigator.clipboard.writeText(text);
        const original = button.textContent;
        button.textContent = '已复制';
        window.setTimeout(() => { button.textContent = original; }, 1000);
    }));
}

function renderPreviewCard(image) {
    const caption = image.caption || {};
    return `
        <article class="dataset-preview-card">
            <div class="dataset-preview-image-btn"><img src="${escapeAttribute(image.url)}" alt="${escapeAttribute(image.name || '数据集图片')}" width="320" height="240" loading="lazy"></div>
            <div class="dataset-preview-card-body">
                <strong>${escapeHtml(image.name || '-')}</strong><span>${escapeHtml(image.file || '')}</span>
                <div class="dataset-preview-caption${caption.ok ? '' : ' missing'}">
                    <div><span>${caption.ok ? `标注 ${escapeHtml(caption.format_label || caption.extension || '')}` : `缺少标注 · ${escapeHtml(caption.source_label || '自动识别')}`}</span>${caption.ok ? '<button class="btn btn-small" type="button" data-caption-copy>复制标注</button>' : ''}</div>
                    <pre>${escapeHtml(caption.ok ? (caption.text || '(空标注)') : '未按当前标注来源找到 caption 文件')}</pre>
                </div>
            </div>
        </article>
    `;
}
