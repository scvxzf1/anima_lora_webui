/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    CAPTION_SOURCE_MODE_OPTIONS,
    help,
} from '../../../config/catalog.js?v=module-bootstrap-20260707-93';
import {
    captionSourceModeLabel,
    normalizeCaptionSourceMode,
} from '../helpers/caption-source.js?v=module-bootstrap-20260707-93';
import { datasetConfigValue } from '../helpers/dataset-config-fields.js?v=module-bootstrap-20260707-93';
import { setDatasetPresetStatus } from '../helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260707-93';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../helpers/dataset-values.js?v=module-bootstrap-20260707-93';
import {
    datasetPreviewImageToPreviewImage,
    datasetPreviewValidationText,
} from '../helpers/dataset-preview.js?v=module-bootstrap-20260707-93';
import {
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
} from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260707-93';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import { datasetPresetApi } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260707-93';
import { updateStepEstimatePanel } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260707-93';
import { markDatasetEditorDirty, setFieldInputValue, updateDatasetEditorRowsSettingValue } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260707-93';
import { openPreviewDialog, createPreviewDetailRow, copyText } from '../helpers/preview-view-bridge.js?v=module-bootstrap-20260707-93';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentDatasetPreviewState() {
    return datasetState.datasetPreviewState || {};
}


    export function createDatasetRowCaptionSourceModeEditor(settings, index) {
        const current = normalizeCaptionSourceMode(settings.caption_source_mode, settings.prefer_json_caption);
        const panel = document.createElement('div');
        panel.className = 'dataset-caption-source';
        panel.dataset.mode = current;

        const head = document.createElement('div');
        head.className = 'dataset-caption-source-head';
        const copy = document.createElement('div');
        copy.className = 'dataset-caption-source-copy';
        const titleRow = document.createElement('div');
        titleRow.className = 'dataset-caption-source-title-row';
        const title = document.createElement('strong');
        title.textContent = '标注来源 / caption_source_mode';
        const helpId = `dataset-caption-source-notes-${++datasetState.datasetCaptionSourceHelpSeq}`;
        const helpBtn = document.createElement('button');
        helpBtn.className = 'info-toggle dataset-caption-source-help-toggle';
        helpBtn.type = 'button';
        helpBtn.textContent = '?';
        helpBtn.title = '展开标注来源注释';
        helpBtn.setAttribute('aria-label', '标注来源格式注释');
        helpBtn.setAttribute('aria-controls', helpId);
        helpBtn.setAttribute('aria-expanded', 'false');
        titleRow.append(title, helpBtn);
        const desc = document.createElement('span');
        desc.textContent = '默认 auto 自动识别；保存后预览和训练前预检测都会显示识别结果，也可以强制指定格式。';
        copy.append(titleRow, desc);
        const state = document.createElement('span');
        state.className = 'dataset-caption-source-state';
        state.textContent = captionSourceModeLabel(current);
        head.append(copy, state);

        const controls = document.createElement('div');
        controls.className = 'dataset-caption-source-options';
        CAPTION_SOURCE_MODE_OPTIONS.forEach((option) => {
            const label = document.createElement('label');
            label.className = ['dataset-caption-source-option', option.value === current ? 'selected' : ''].filter(Boolean).join(' ');
            const input = document.createElement('input');
            input.type = 'radio';
            input.name = `dataset-caption-source-${index}`;
            input.value = option.value;
            input.checked = option.value === current;
            input.setAttribute('aria-label', `${option.label} ${option.detail}`);
            input.addEventListener('change', () => {
                if (!input.checked) return;
                updateDatasetEditorRowsSettingValue(
                    [index],
                    'caption_source_mode',
                    option.value,
                    { render: 'item' },
                );
            });
            const labelText = document.createElement('span');
            labelText.textContent = option.label;
            const detail = document.createElement('small');
            detail.textContent = option.detail;
            label.append(input, labelText, detail);
            controls.appendChild(label);
        });

        const notes = document.createElement('ul');
        notes.className = 'dataset-caption-source-notes';
        notes.id = helpId;
        notes.hidden = true;
        [
            '"1.png+1.txt"*n = sd-scripts格式标注',
            '"1.png+1.json"*n = AnimaLoraToolkit格式标注',
            '"png*n"+captions.json = DiffPipeForge格式标注',
            'caption_extension 仅影响 txt 来源或 auto 回退到文本标注；json / captions.json 模式会忽略它。',
        ].forEach((text) => {
            const item = document.createElement('li');
            item.textContent = text;
            notes.appendChild(item);
        });
        helpBtn.addEventListener('click', () => {
            const nextVisible = notes.hidden;
            notes.hidden = !nextVisible;
            helpBtn.classList.toggle('active', nextVisible);
            helpBtn.setAttribute('aria-expanded', String(nextVisible));
            helpBtn.title = nextVisible ? '收起标注来源注释' : '展开标注来源注释';
        });

        panel.append(head, controls, notes);
        return panel;
    }

    export function createDatasetRowSettingInput(index, key, type, settings) {
        let input;
        if (type === 'select') {
            input = document.createElement('select');
            const options = key === 'enable_bucket'
                ? [[true, '启用'], [false, '关闭']]
                : [[false, '允许放大'], [true, '不放大小图']];
            const current = Boolean(settings[key]);
            for (const [value, label] of options) {
                const opt = document.createElement('option');
                opt.value = value ? 'true' : 'false';
                opt.textContent = label;
                opt.selected = value === current;
                input.appendChild(opt);
            }
        } else {
            input = document.createElement('input');
            input.type = type;
            input.value = datasetConfigValue(key, settings);
            if (type === 'number') {
                input.min = '0';
                input.step = key === 'validation_split' ? '0.001' : (key === 'resolution' || key.endsWith('_reso') || key === 'bucket_reso_steps' ? '16' : '1');
            }
        }
        input.className = 'field-input dataset-row-setting-input';
        input.addEventListener('input', () => updateDatasetEditorRowSetting(index, key, input));
        input.addEventListener('change', () => updateDatasetEditorRowSetting(index, key, input));
        return input;
    }

    export function createDatasetPathField(index, key, label, value, placeholder) {
        const field = document.createElement('label');
        field.className = 'dataset-path-field';
        field.dataset.key = key;
        const text = document.createElement('span');
        text.textContent = label;
        const titles = {
            source_dir: '原始图片和 caption 所在目录。预处理从这里读图；缩放图和 LoRA 缓存会写入本次训练运行目录。',
            image_dir: '缩放图目录。预处理会把图片按分辨率/分桶规则写到这里；训练从这里枚举训练图片。',
            cache_dir: 'LoRA 缓存目录。VAE latent、文本编码器缓存、PE 特征缓存会写到这里；训练用它加速。',
        };
        text.title = titles[key] || label;
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input dataset-path-input';
        input.value = value || '';
        input.placeholder = placeholder;
        input.title = titles[key] || '';
        input.addEventListener('input', () => updateDatasetEditorRow(index, key, input.value));
        field.append(text, input);
        return field;
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

    export function updateDatasetDefault(key, input) {
        const state = datasetEditorStateForActivePanel();
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        if (input.type === 'checkbox') {
            defaults[key] = input.checked;
        } else if (input.tagName === 'SELECT') {
            defaults[key] = input.value === 'true';
        } else if (input.type === 'number') {
            defaults[key] = key === 'validation_split' || key === 'prior_loss_weight'
                ? Math.max(0, Number(input.value) || 0)
                : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            defaults[key] = input.value;
        }
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.defaults = defaults;
        } else {
            datasetState.datasetEditorState.defaults = defaults;
        }
        markDatasetEditorDirty();
    }

    export function updateDatasetEditorRow(index, key, value) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (!rows[index]) return;
        if (key === 'source_dir' && rows[index].source_dir !== value) {
            rows[index].image_dir = '';
            rows[index].cache_dir = '';
        }
        rows[index][key] = key === 'num_repeats'
            ? Math.max(1, Number.parseInt(value || '1', 10) || 1)
            : value;
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.datasets = rows;
        } else {
            datasetState.datasetEditorState.datasets = rows;
        }
        if (!isDatasetTabActive() && index === 0 && key === 'source_dir') {
            setFieldInputValue('source_image_dir', value);
        }
        markDatasetEditorDirty();
        if (key === 'num_repeats') {
            updateStepEstimatePanel();
        }
    }

    function updateDatasetEditorRowSetting(index, key, input) {
        let value;
        if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.tagName === 'SELECT') {
            value = input.value === 'true';
        } else if (input.type === 'number') {
            value = key === 'validation_split' || key === 'prior_loss_weight'
                ? Math.max(0, Number(input.value) || 0)
                : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            value = input.value;
        }
        updateDatasetEditorRowSettingValue(index, key, value);
    }

    export function updateDatasetEditorRowSettingValue(index, key, value) {
        updateDatasetEditorRowsSettingValue([index], key, value);
    }
