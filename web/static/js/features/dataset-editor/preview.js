/**
 * Dataset subset preview dialog and image loading.
 * Extracted from former chunk 12 / row-fields.
 */
import { setDatasetPresetStatus } from '../anima-app/helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    datasetPreviewImageToPreviewImage,
    datasetPreviewValidationText,
} from '../anima-app/helpers/dataset-preview.js?v=module-bootstrap-20260711-ir1';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { datasetPresetApi } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';
import { openPreviewDialog, createPreviewDetailRow, copyText } from '../anima-app/helpers/preview-view-bridge.js?v=module-bootstrap-20260711-ir1';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentDatasetPreviewState() {
    return datasetState.datasetPreviewState || {};
}

export async function openDatasetPreview(index) {
        if (!currentDatasetPresetState().selectedFile) {
            setDatasetPresetStatus('请先选择一个数据集预设', 'error');
            return;
        }
        if (currentDatasetPresetState().dirty) {
            setDatasetPresetStatus('请先保存当前数据集预设，再打开预览', 'error');
            return;
        }
        const previewState = currentDatasetPreviewState();
        previewState.datasetIndex = index;
        previewState.source = 'source';
        previewState.payload = null;
        const dialog = document.getElementById('dataset-preview-dialog');
        renderDatasetPreviewDialog({ loading: true });
        if (dialog?.showModal && !dialog.open) {
            dialog.showModal();
        }
        await loadDatasetPreviewImages();
    }

export async function loadDatasetPreviewImages() {
        const file = currentDatasetPresetState().selectedFile;
        if (!file) return;
        const requestSeq = ++datasetState.datasetPreviewLoadSeq;
        renderDatasetPreviewDialog({ loading: true });
        try {
            const previewState = currentDatasetPreviewState();
            const params = new URLSearchParams({
                file,
                dataset_index: String(previewState.datasetIndex || 0),
                source: 'source',
                limit: '120',
            });
            const payload = await datasetPresetApi(`/api/config/dataset-presets/images?${params.toString()}`);
            if (requestSeq !== datasetState.datasetPreviewLoadSeq) return;
            if (!payload.ok) throw new Error(payload.error || '读取数据集预览失败');
            previewState.payload = payload;
            renderDatasetPreviewDialog();
        } catch (e) {
            if (requestSeq !== datasetState.datasetPreviewLoadSeq) return;
            currentDatasetPreviewState().payload = {
                ok: false,
                error: e.message || '读取数据集预览失败',
                images: [],
            };
            renderDatasetPreviewDialog();
        }
    }

    function renderDatasetPreviewDialog(options = {}) {
        const title = document.getElementById('dataset-preview-dialog-title');
        const meta = document.getElementById('dataset-preview-dialog-meta');
        const grid = document.getElementById('dataset-preview-grid');
        const details = document.getElementById('dataset-preview-details');
        const empty = document.getElementById('dataset-preview-empty');
        if (!title || !meta || !grid || !details || !empty) return;

        const previewState = currentDatasetPreviewState();
        const datasetNo = Number(previewState.datasetIndex || 0) + 1;
        title.textContent = `第 ${datasetNo} 组数据集预览`;
        if (options.loading) {
            meta.textContent = '正在读取图片和同名标注...';
            grid.innerHTML = '';
            details.innerHTML = '';
            empty.textContent = '正在读取数据集图片...';
            empty.hidden = false;
            return;
        }

        const payload = previewState.payload || {};
        if (payload.error) {
            meta.textContent = payload.error;
            grid.innerHTML = '';
            details.innerHTML = '';
            empty.textContent = payload.error;
            empty.hidden = false;
            return;
        }

        const countText = `${payload.count || 0}/${payload.total || 0} 张`;
        const sourceLabel = payload.caption_source_label || captionSourceModeLabel(payload.caption_source_mode || 'auto');
        const detectedSummary = payload.caption_summary ? ` · 识别 ${payload.caption_summary}` : '';
        meta.textContent = `${payload.source_label || '原始图目录'} · ${payload.directory || '-'} · ${countText} · 标注来源 ${sourceLabel}${detectedSummary}`;
        renderDatasetPreviewDetails(payload);
        grid.innerHTML = '';
        const images = Array.isArray(payload.images) ? payload.images : [];
        if (!images.length) {
            empty.textContent = payload.message || '当前目录没有可预览图片。';
            empty.hidden = false;
            return;
        }
        empty.hidden = true;
        for (const image of images) {
            grid.appendChild(createDatasetPreviewCard(image));
        }
    }

    function renderDatasetPreviewDetails(payload) {
        const details = document.getElementById('dataset-preview-details');
        if (!details) return;
        details.innerHTML = '';
        const row = payload.row || {};
        const settings = normalizeDatasetDefaults(payload.settings || row.settings || {});
        const items = [
            ['数据集文件', payload.file || currentDatasetPresetState().selectedFile || '-'],
            ['当前目录', payload.directory || '-'],
            ['原始路径', row.source_dir || '-'],
            ['重复次数', row.num_repeats ?? '-'],
            ['分辨率', settings.resolution || '-'],
            ['分桶', settings.enable_bucket ? `${settings.min_bucket_reso}-${settings.max_bucket_reso}/${settings.bucket_reso_steps}` : '关闭'],
            ['验证集', datasetPreviewValidationText(settings)],
            ['标注来源', payload.caption_source_label || captionSourceModeLabel(settings.caption_source_mode || 'auto')],
            ['识别结果', payload.caption_summary || '-'],
        ];
        for (const [label, value] of items) {
            details.appendChild(createPreviewDetailRow(label, String(value)));
        }
    }

    function createDatasetPreviewCard(image) {
        const card = document.createElement('article');
        card.className = 'dataset-preview-card';
        const imageWrap = document.createElement('button');
        imageWrap.type = 'button';
        imageWrap.className = 'dataset-preview-image-btn';
        imageWrap.title = '点击在大图预览中查看。';
        imageWrap.addEventListener('click', () => openPreviewDialog(datasetPreviewImageToPreviewImage(image)));

        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name;
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            card.classList.add('dataset-preview-card-error');
            img.alt = '图片加载失败';
        });
        imageWrap.appendChild(img);

        const body = document.createElement('div');
        body.className = 'dataset-preview-card-body';
        const name = document.createElement('strong');
        name.textContent = image.name || '-';
        const file = document.createElement('span');
        file.textContent = image.file || '';
        body.append(name, file);

        const caption = image.caption || {};
        const captionBox = document.createElement('div');
        captionBox.className = ['dataset-preview-caption', caption.ok ? '' : 'missing'].filter(Boolean).join(' ');
        const captionHead = document.createElement('div');
        const captionTitle = document.createElement('span');
        const captionCount = Number(caption.caption_count || 0);
        const formatLabel = caption.format_label || caption.extension || '';
        captionTitle.textContent = caption.ok
            ? `标注 ${formatLabel}${captionCount > 1 ? ` · ${captionCount} 条` : ''}`
            : `缺少标注 · ${caption.source_label || captionSourceModeLabel(caption.source_mode || 'auto')}`;
        captionHead.appendChild(captionTitle);
        if (caption.file) {
            const copyBtn = document.createElement('button');
            copyBtn.type = 'button';
            copyBtn.className = 'btn btn-small';
            copyBtn.textContent = '复制标注';
            copyBtn.addEventListener('click', () => copyDatasetCaptionText(caption.text || '', copyBtn));
            captionHead.appendChild(copyBtn);
        }
        const pre = document.createElement('pre');
        pre.textContent = caption.ok ? (caption.text || '(空标注)') : '未按当前标注来源找到 caption 文件';
        captionBox.append(captionHead, pre);
        body.appendChild(captionBox);

        card.append(imageWrap, body);
        return card;
    }

    async function copyDatasetCaptionText(text, button) {
        try {
            await copyText(text || '');
            const original = button.textContent;
            button.textContent = '已复制';
            button.classList.add('btn-primary');
            setTimeout(() => {
                button.textContent = original;
                button.classList.remove('btn-primary');
            }, 1000);
        } catch (e) {
            alert('复制标注失败: ' + e.message);
        }
    }
