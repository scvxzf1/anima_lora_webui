/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.createDatasetRowCaptionSourceModeEditor = function createDatasetRowCaptionSourceModeEditor(settings, index) {
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
        const helpId = `dataset-caption-source-notes-${++datasetCaptionSourceHelpSeq}`;
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

    globalThis.createDatasetRowSettingInput = function createDatasetRowSettingInput(index, key, type, settings) {
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

    globalThis.createDatasetPathField = function createDatasetPathField(index, key, label, value, placeholder) {
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

    globalThis.openDatasetPreview = async function openDatasetPreview(index) {
        if (!datasetPresetState.selectedFile) {
            setDatasetPresetStatus('请先选择一个数据集预设', 'error');
            return;
        }
        if (datasetPresetState.dirty) {
            setDatasetPresetStatus('请先保存当前数据集预设，再打开预览', 'error');
            return;
        }
        datasetPreviewState.datasetIndex = index;
        datasetPreviewState.source = 'source';
        datasetPreviewState.payload = null;
        const dialog = document.getElementById('dataset-preview-dialog');
        renderDatasetPreviewDialog({ loading: true });
        if (dialog?.showModal && !dialog.open) {
            dialog.showModal();
        }
        await loadDatasetPreviewImages();
    }

    globalThis.loadDatasetPreviewImages = async function loadDatasetPreviewImages() {
        const file = datasetPresetState.selectedFile;
        if (!file) return;
        const requestSeq = ++datasetPreviewLoadSeq;
        renderDatasetPreviewDialog({ loading: true });
        try {
            const params = new URLSearchParams({
                file,
                dataset_index: String(datasetPreviewState.datasetIndex || 0),
                source: 'source',
                limit: '120',
            });
            const payload = await datasetPresetApi(`/api/config/dataset-presets/images?${params.toString()}`);
            if (requestSeq !== datasetPreviewLoadSeq) return;
            if (!payload.ok) throw new Error(payload.error || '读取数据集预览失败');
            datasetPreviewState.payload = payload;
            renderDatasetPreviewDialog();
        } catch (e) {
            if (requestSeq !== datasetPreviewLoadSeq) return;
            datasetPreviewState.payload = {
                ok: false,
                error: e.message || '读取数据集预览失败',
                images: [],
            };
            renderDatasetPreviewDialog();
        }
    }

    globalThis.renderDatasetPreviewDialog = function renderDatasetPreviewDialog(options = {}) {
        const title = document.getElementById('dataset-preview-dialog-title');
        const meta = document.getElementById('dataset-preview-dialog-meta');
        const grid = document.getElementById('dataset-preview-grid');
        const details = document.getElementById('dataset-preview-details');
        const empty = document.getElementById('dataset-preview-empty');
        if (!title || !meta || !grid || !details || !empty) return;

        const datasetNo = Number(datasetPreviewState.datasetIndex || 0) + 1;
        title.textContent = `第 ${datasetNo} 组数据集预览`;
        if (options.loading) {
            meta.textContent = '正在读取图片和同名标注...';
            grid.innerHTML = '';
            details.innerHTML = '';
            empty.textContent = '正在读取数据集图片...';
            empty.hidden = false;
            return;
        }

        const payload = datasetPreviewState.payload || {};
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

    globalThis.renderDatasetPreviewDetails = function renderDatasetPreviewDetails(payload) {
        const details = document.getElementById('dataset-preview-details');
        if (!details) return;
        details.innerHTML = '';
        const row = payload.row || {};
        const settings = normalizeDatasetDefaults(payload.settings || row.settings || {});
        const items = [
            ['数据集文件', payload.file || datasetPresetState.selectedFile || '-'],
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

    globalThis.datasetPreviewValidationText = function datasetPreviewValidationText(settings) {
        const validationNum = Number(settings.validation_split_num || 0);
        if (validationNum > 0) return `固定 ${validationNum} 张`;
        const validationSplit = Number(settings.validation_split ?? 0);
        if (validationSplit > 0) return `比例 ${validationSplit}`;
        return '关闭';
    }

    globalThis.createDatasetPreviewCard = function createDatasetPreviewCard(image) {
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

    globalThis.datasetPreviewImageToPreviewImage = function datasetPreviewImageToPreviewImage(image) {
        return {
            ...image,
            detailContext: 'dataset',
            sample: {},
            source_task: null,
        };
    }

    globalThis.copyDatasetCaptionText = async function copyDatasetCaptionText(text, button) {
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

    globalThis.normalizeNlTagMix = function normalizeNlTagMix(raw) {
        const source = raw && typeof raw === 'object' ? raw : {};
        const enabled = source.enabled === true || source.enabled === 'true';
        const parsedRatio = Number(source.tag_ratio ?? source.tagRatio ?? DEFAULT_NL_TAG_MIX.tag_ratio);
        const tagRatio = Number.isFinite(parsedRatio)
            ? Math.min(1, Math.max(0, parsedRatio > 1 ? parsedRatio / 100 : parsedRatio))
            : DEFAULT_NL_TAG_MIX.tag_ratio;
        return {
            enabled,
            tag_ratio: tagRatio,
        };
    }

    globalThis.nlTagMixSummary = function nlTagMixSummary(mix) {
        const normalized = normalizeNlTagMix(mix);
        const tagPercent = Math.round(normalized.tag_ratio * 100);
        return `${tagPercent}% tag + ${100 - tagPercent}% nl`;
    }

    globalThis.normalizeTriggerClone = function normalizeTriggerClone(raw) {
        const source = raw && typeof raw === 'object' ? raw : {};
        return {
            enabled: source.enabled === true || source.enabled === 'true',
            prompt: String(source.prompt || '').trim(),
            num_repeats: Math.max(1, Number.parseInt(source.num_repeats || 1, 10) || 1),
        };
    }

    globalThis.normalizeDatasetEditorRows = function normalizeDatasetEditorRows(rows) {
        return (rows || [])
            .filter((row) => row && typeof row === 'object')
            .map((row) => ({
                source_dir: String(row.source_dir || row.source_image_dir || ''),
                image_dir: String(row.image_dir || row.resized_image_dir || ''),
                cache_dir: String(row.cache_dir || row.lora_cache_dir || ''),
                num_repeats: Math.max(1, Number.parseInt(row.num_repeats || 1, 10) || 1),
                recursive: row.recursive !== false && row.recursive !== 'false',
                path_pattern: String(row.path_pattern || '*').trim() || '*',
                is_reg: row.is_reg === true,
                nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix),
                trigger_clone: normalizeTriggerClone(row.trigger_clone),
                settings: normalizeDatasetRowSettings(row),
            }));
    }

    globalThis.datasetRowsForPayload = function datasetRowsForPayload(rows) {
        return normalizeDatasetEditorRows(rows).map((row) => ({
            source_dir: row.source_dir,
            image_dir: row.image_dir,
            cache_dir: row.cache_dir,
            num_repeats: row.num_repeats,
            recursive: row.recursive,
            path_pattern: row.path_pattern,
            is_reg: row.is_reg,
            nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix),
            trigger_clone: normalizeTriggerClone(row.trigger_clone),
            settings: normalizeDatasetDefaults(row.settings || {}),
        }));
    }

    globalThis.normalizeDatasetRowSettings = function normalizeDatasetRowSettings(row) {
        if (row.settings && typeof row.settings === 'object') {
            return normalizeDatasetDefaults(row.settings);
        }
        if ([...DATASET_SETTING_KEYS].some((key) => key in row)) {
            return normalizeDatasetDefaults(row);
        }
        return {};
    }

    globalThis.normalizeDatasetDefaults = function normalizeDatasetDefaults(defaults) {
        const raw = defaults && typeof defaults === 'object' ? defaults : {};
        const preferJson = raw.prefer_json_caption === true || raw.prefer_json_caption === 'true';
        const captionSourceMode = normalizeCaptionSourceMode(raw.caption_source_mode, preferJson);
        const validationSeed = Number.parseInt(raw.validation_seed ?? 42, 10);
        const priorLossWeight = Number(raw.prior_loss_weight ?? 1.0);
        return {
            resolution: Math.max(1, Number.parseInt(raw.resolution || 1024, 10) || 1024),
            prior_loss_weight: Number.isFinite(priorLossWeight) ? Math.max(0, priorLossWeight) : 1.0,
            enable_bucket: raw.enable_bucket !== false && raw.enable_bucket !== 'false',
            min_bucket_reso: Math.max(1, Number.parseInt(raw.min_bucket_reso || 256, 10) || 256),
            max_bucket_reso: Math.max(1, Number.parseInt(raw.max_bucket_reso || 1024, 10) || 1024),
            bucket_reso_steps: Math.max(1, Number.parseInt(raw.bucket_reso_steps || 64, 10) || 64),
            bucket_no_upscale: raw.bucket_no_upscale === true || raw.bucket_no_upscale === 'true',
            validation_split: Math.max(0, Number(raw.validation_split ?? 0) || 0),
            validation_split_num: Math.max(0, Number.parseInt(raw.validation_split_num || 0, 10) || 0),
            validation_seed: Number.isFinite(validationSeed) ? Math.max(0, validationSeed) : 42,
            caption_extension: String(raw.caption_extension || '.txt'),
            keep_tokens: Math.max(0, Number.parseInt(raw.keep_tokens || 3, 10) || 0),
            prefer_json_caption: preferJson,
            caption_source_mode: captionSourceMode,
        };
    }

    globalThis.updateDatasetDefault = function updateDatasetDefault(key, input) {
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
            datasetPresetState.defaults = defaults;
        } else {
            datasetEditorState.defaults = defaults;
        }
        markDatasetEditorDirty();
    }

    globalThis.updateDatasetEditorRow = function updateDatasetEditorRow(index, key, value) {
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
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        if (!isDatasetTabActive() && index === 0 && key === 'source_dir') {
            setFieldInputValue('source_image_dir', value);
        }
        markDatasetEditorDirty();
        if (key === 'num_repeats') {
            updateStepEstimatePanel();
        }
    }

    globalThis.updateDatasetEditorRowSetting = function updateDatasetEditorRowSetting(index, key, input) {
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

    globalThis.updateDatasetEditorRowSettingValue = function updateDatasetEditorRowSettingValue(index, key, value) {
        updateDatasetEditorRowsSettingValue([index], key, value);
    }
