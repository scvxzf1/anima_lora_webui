/* Field and row rendering helpers for the Dragon dataset workspace. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

export const CAPTION_MODES = [
    ['auto', '自动识别'],
    ['txt', '同名 TXT'],
    ['json', 'JSON 标注'],
    ['captions_json', 'captions.json'],
];

export const DATASET_SETTING_KEYS = [
    'resolution',
    'batch_size',
    'prior_loss_weight',
    'enable_bucket',
    'min_bucket_reso',
    'max_bucket_reso',
    'bucket_reso_steps',
    'bucket_no_upscale',
    'validation_split',
    'validation_split_num',
    'validation_seed',
    'caption_extension',
    'prefer_json_caption',
    'caption_source_mode',
];

export function createEmptyDatasetRow(defaults = {}) {
    return {
        source_dir: '',
        image_dir: '',
        cache_dir: '',
        num_repeats: 1,
        is_reg: false,
        recursive: true,
        path_pattern: '*',
        settings: { ...defaults },
        nl_tag_mix: { enabled: false, tag_ratio: 0.7 },
        trigger_clone: { enabled: false, prompt: '', num_repeats: 1 },
    };
}

export function renderDatasetDefaults(defaults = {}, options = {}) {
    const locked = { disabled: Boolean(options.readonly) };
    return `
        <div class="dragon-dataset-settings-sections">
            ${settingsGroup('训练尺寸', 'ruler', [
                numberField('resolution', '训练分辨率', defaults.resolution ?? 1024, { ...locked, min: 64, step: 64 }),
                numberField('batch_size', '数据集批次大小', defaults.batch_size ?? 1, { ...locked, min: 1, step: 1 }),
                numberField('prior_loss_weight', '先验损失权重', defaults.prior_loss_weight ?? 1, { ...locked, min: 0, step: 0.05, help: '正则化图像损失相对普通训练图像的权重。' }),
            ])}
            ${settingsGroup('验证集', 'shieldCheck', [
                numberField('validation_split', '验证集比例', defaults.validation_split ?? 0, { ...locked, min: 0, max: 1, step: 0.01, help: '按比例从每组数据中划出验证样本，0 表示不划分。' }),
                numberField('validation_split_num', '验证集数量', defaults.validation_split_num ?? 0, { ...locked, min: 0, step: 1 }),
                numberField('validation_seed', '验证随机种子', defaults.validation_seed ?? 42, { ...locked, min: 0, step: 1 }),
            ])}
            ${settingsGroup('分桶规则', 'layers', [
                selectField('enable_bucket', '启用分桶', Boolean(defaults.enable_bucket ?? true), [['true', '开启'], ['false', '关闭']], locked),
                numberField('min_bucket_reso', '最小桶尺寸', defaults.min_bucket_reso ?? 256, { ...locked, min: 64, step: 64 }),
                numberField('max_bucket_reso', '最大桶尺寸', defaults.max_bucket_reso ?? 1024, { ...locked, min: 64, step: 64 }),
                numberField('bucket_reso_steps', '桶尺寸步长', defaults.bucket_reso_steps ?? 64, { ...locked, min: 1, step: 1 }),
                selectField('bucket_no_upscale', '禁止放大', Boolean(defaults.bucket_no_upscale), [['true', '开启'], ['false', '关闭']], { ...locked, help: '开启后只缩小过大的图像，不放大小于目标桶的图像。' }),
            ])}
            ${settingsGroup('标注读取', 'tags', [
                textField('caption_extension', '标注扩展名', defaults.caption_extension ?? '.txt', { ...locked, placeholder: '例如：.txt' }),
                numberField('keep_tokens', '保留 Token 数', defaults.keep_tokens ?? 3, { ...locked, min: 0, step: 1, help: '打乱标注时固定保留在开头、不参与随机排序的 Token 数量。' }),
                selectField('prefer_json_caption', '优先 JSON 标注', Boolean(defaults.prefer_json_caption), [['true', '开启'], ['false', '关闭']], locked),
                selectField('caption_source_mode', '标注来源', defaults.caption_source_mode || 'auto', CAPTION_MODES, locked),
            ])}
        </div>
    `;
}

export function renderDatasetRow(row, index, defaults, options = {}) {
    const settings = { ...defaults, ...(row.settings || {}) };
    const mix = row.nl_tag_mix || {};
    const clone = row.trigger_clone || {};
    const readonly = Boolean(options.readonly);
    const lockAttr = readonly ? 'disabled' : '';
    const summary = `${row.is_reg ? '正则数据' : '训练数据'} · 重复 ${Number(row.num_repeats || 1)} · ${Number(settings.resolution || 1024)}px`;
    return `
        <article class="dragon-dataset-row" data-dataset-row data-index="${index}" tabindex="0" aria-label="数据集组 ${index + 1}；按 Alt 加上下方向键可排序">
            <header class="dragon-dataset-row-head">
                <div class="dragon-dataset-row-identity">
                    <span class="dragon-dataset-row-number">${index + 1}</span>
                    <div><span class="dragon-eyebrow">数据集组 ${index + 1}</span><h3>训练来源与子集规则</h3><p data-row-summary>${escapeHtml(summary)}</p></div>
                </div>
                <div class="dragon-dataset-row-actions">
                    <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-dataset-move="up" ${index === 0 || readonly ? 'disabled' : ''} aria-label="上移数据集组 ${index + 1}">↑</button>
                    <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-dataset-move="down" ${index >= Number(options.totalRows || 1) - 1 || readonly ? 'disabled' : ''} aria-label="下移数据集组 ${index + 1}">↓</button>
                    <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-dataset-preview ${options.canPreview ? '' : 'disabled'}>${renderIcon('eye', 'dragon-btn-icon')}<span>预览</span></button>
                    <button class="dragon-btn dragon-btn-ghost dragon-btn-sm dragon-btn-danger" type="button" data-dataset-remove ${readonly || Number(options.totalRows || 1) <= 1 ? 'disabled' : ''}>移除</button>
                </div>
            </header>
            <div class="dragon-dataset-path-panel">
                ${sourceDirectoryField(row.source_dir || '', index, { disabled: readonly, lockAttr })}
                <div class="dragon-dataset-field-grid dragon-dataset-path-grid">
                    ${textField('image_dir', '缩放图片目录', row.image_dir || '', { placeholder: '保存预处理后的训练图片', disabled: readonly })}
                    ${textField('cache_dir', 'LoRA 缓存目录', row.cache_dir || '', { placeholder: '保存 latent 与文本缓存', disabled: readonly })}
                </div>
            </div>
            <div class="dragon-dataset-field-grid dragon-dataset-row-primary">
                ${numberField('num_repeats', '重复次数', row.num_repeats ?? 1, { min: 1, step: 1, disabled: readonly })}
                ${selectField('is_reg', '数据用途', Boolean(row.is_reg), [['false', '普通训练数据'], ['true', '正则化数据']], { disabled: readonly })}
                ${selectField('recursive', '目录扫描', row.recursive !== false, [['true', '包含子目录'], ['false', '仅当前目录']], { disabled: readonly })}
                ${textField('path_pattern', '路径筛选', row.path_pattern || '*', { placeholder: '例如：*.png', disabled: readonly })}
            </div>
            <details class="dragon-dataset-advanced">
                <summary><span>高级规则</span><small>分辨率覆盖、验证集、标签混合与触发词复制</small></summary>
                <div class="dragon-dataset-advanced-body">
                    ${settingsGroup('本组覆盖', '', [
                        numberField('resolution', '本组分辨率', settings.resolution ?? 1024, { min: 64, step: 64, disabled: readonly }),
                        numberField('batch_size', '本组批次大小', settings.batch_size ?? 1, { min: 1, step: 1, disabled: readonly }),
                        numberField('prior_loss_weight', '本组先验损失权重', settings.prior_loss_weight ?? 1, { min: 0, step: 0.05, disabled: readonly }),
                        selectField('enable_bucket', '本组启用分桶', Boolean(settings.enable_bucket ?? true), [['true', '开启'], ['false', '关闭']], { disabled: readonly }),
                        numberField('min_bucket_reso', '本组最小桶尺寸', settings.min_bucket_reso ?? 256, { min: 64, step: 64, disabled: readonly }),
                        numberField('max_bucket_reso', '本组最大桶尺寸', settings.max_bucket_reso ?? 1024, { min: 64, step: 64, disabled: readonly }),
                        numberField('bucket_reso_steps', '本组桶步长', settings.bucket_reso_steps ?? 64, { min: 1, step: 1, disabled: readonly }),
                        selectField('bucket_no_upscale', '本组禁止放大', Boolean(settings.bucket_no_upscale), [['true', '开启'], ['false', '关闭']], { disabled: readonly }),
                    ])}
                    ${settingsGroup('标注与验证', '', [
                        textField('caption_extension', '本组标注扩展名', settings.caption_extension ?? '.txt', { placeholder: '例如：.txt', disabled: readonly }),
                        selectField('caption_source_mode', '本组标注来源', settings.caption_source_mode || 'auto', CAPTION_MODES, { disabled: readonly }),
                        selectField('prefer_json_caption', '本组优先 JSON', Boolean(settings.prefer_json_caption), [['true', '开启'], ['false', '关闭']], { disabled: readonly }),
                        numberField('validation_split', '本组验证集比例', settings.validation_split ?? 0, { min: 0, max: 1, step: 0.01, disabled: readonly }),
                        numberField('validation_split_num', '本组验证集数量', settings.validation_split_num ?? 0, { min: 0, step: 1, disabled: readonly }),
                        numberField('validation_seed', '本组验证随机种子', settings.validation_seed ?? 42, { min: 0, step: 1, disabled: readonly }),
                    ])}
                    ${settingsGroup('实验规则', '', [
                        selectField('nl_tag_mix.enabled', '标签混合', Boolean(mix.enabled), [['true', '开启'], ['false', '关闭']], { disabled: readonly }),
                        numberField('nl_tag_mix.tag_ratio', '标签混合比例', mix.tag_ratio ?? 0.7, { min: 0, max: 1, step: 0.01, disabled: readonly }),
                        selectField('trigger_clone.enabled', '触发词复制', Boolean(clone.enabled), [['true', '开启'], ['false', '关闭']], { disabled: readonly }),
                        textField('trigger_clone.prompt', '触发词', clone.prompt || '', { placeholder: '例如：my_style', disabled: readonly }),
                        numberField('trigger_clone.num_repeats', '触发词复制次数', clone.num_repeats ?? 1, { min: 1, step: 1, disabled: readonly }),
                    ])}
                </div>
            </details>
        </article>
    `;
}

function settingsGroup(title, icon, fields) {
    const count = fields.length;
    const columns = count <= 3 ? count : (count === 4 ? 2 : 3);
    return `<section class="dragon-dataset-settings-group" data-field-count="${count}"><header>${icon ? renderIcon(icon, 'dragon-dataset-settings-icon') : ''}<h3>${escapeHtml(title)}</h3></header><div class="dragon-dataset-field-grid" style="--dataset-field-columns:${columns}">${fields.join('')}</div></section>`;
}

function sourceDirectoryField(value, index, options = {}) {
    const fieldId = `dataset-source-dir-${index}`;
    const errorId = `${fieldId}-error`;
    return `
        <div class="dragon-field dragon-field-span dragon-dataset-source-field" data-field-shell="source_dir">
            <label class="dragon-field-label-text" for="${fieldId}">原始图片目录</label>
            <div class="dragon-dataset-path-input">
                <input id="${fieldId}" class="dragon-input" type="text" name="source_dir" autocomplete="off" data-field="source_dir" value="${escapeAttribute(value)}" placeholder="例如：image_dataset…" required aria-describedby="${errorId}" ${options.disabled ? 'disabled' : ''}>
                <div class="dragon-dataset-path-input-actions">
                    <button class="dragon-path-icon-button" type="button" data-dataset-browse aria-label="选择原始图片目录" title="选择原始图片目录" ${options.lockAttr || ''}>${renderIcon('folder')}</button>
                    <button class="dragon-path-icon-button" type="button" data-dataset-copy aria-label="复制原始图片目录" title="复制原始图片目录" ${value ? '' : 'disabled'}>${renderIcon('copy')}</button>
                </div>
            </div>
            <div class="dragon-dataset-path-meta">
                <button class="dragon-dataset-path-suggest" type="button" data-dataset-suggest ${options.lockAttr || ''}>${renderIcon('wand', 'dragon-btn-icon')}<span>根据原始目录补全缓存路径</span></button>
                <span class="dragon-dataset-path-status" data-dataset-path-status data-state="idle" role="status" aria-live="polite"><span aria-hidden="true"></span><span>等待检测</span></span>
            </div>
            <span id="${errorId}" class="dragon-field-error" data-error-key="source_dir" data-field-error role="alert" hidden></span>
        </div>`;
}

function textField(key, label, value, options = {}) {
    return fieldShell(key, label, `<input class="dragon-input" type="text" name="${escapeAttribute(key)}" autocomplete="off" data-field="${escapeAttribute(key)}" value="${escapeAttribute(value)}" ${options.placeholder ? `placeholder="${escapeAttribute(options.placeholder)}…"` : ''} ${options.required ? 'required' : ''} ${options.disabled ? 'disabled' : ''}>`, options);
}

function numberField(key, label, value, options = {}) {
    const step = options.step ?? 1;
    return fieldShell(key, label, `<input class="dragon-input" type="number" inputmode="decimal" name="${escapeAttribute(key)}" autocomplete="off" step="${escapeAttribute(step)}" ${options.min != null ? `min="${escapeAttribute(options.min)}"` : ''} ${options.max != null ? `max="${escapeAttribute(options.max)}"` : ''} data-field="${escapeAttribute(key)}" value="${escapeAttribute(value)}" ${options.disabled ? 'disabled' : ''}>`, options);
}

function selectField(key, label, value, options, fieldOptions = {}) {
    const selected = typeof value === 'boolean' ? String(value) : String(value ?? '');
    return fieldShell(key, label, `<select class="dragon-select" name="${escapeAttribute(key)}" autocomplete="off" data-field="${escapeAttribute(key)}" ${fieldOptions.disabled ? 'disabled' : ''}>${options.map(([option, text]) => `<option value="${escapeAttribute(option)}" ${String(option) === selected ? 'selected' : ''}>${escapeHtml(text)}</option>`).join('')}</select>`, fieldOptions);
}

function fieldShell(key, label, control, options = {}) {
    const fieldId = String(key).replace(/[^A-Za-z0-9_-]+/g, '-');
    const errorId = `dataset-field-error-${fieldId}`;
    const describedBy = [options.help ? `${errorId}-hint` : '', errorId].filter(Boolean).join(' ');
    const accessibleControl = control.replace(/(<(?:input|select)\b)([^>]*>)/, `$1 aria-describedby="${escapeAttribute(describedBy)}"$2`);
    const tooltip = options.help
        ? `<span class="dragon-field-help" tabindex="0" data-hint-key="${escapeAttribute(fieldId)}" data-tooltip="${escapeAttribute(options.help)}" aria-label="${escapeAttribute(label)}说明">${renderIcon('circleHelp')}<span class="visually-hidden">${escapeHtml(options.help)}</span></span>`
        : '';
    return `<label class="dragon-field${options.wide ? ' dragon-field-span' : ''}" data-field-shell="${escapeAttribute(key)}"><span class="dragon-field-label-text">${escapeHtml(label)}${tooltip}</span>${accessibleControl}<span class="dragon-field-error" data-error-key="${escapeAttribute(fieldId)}" data-field-error role="alert" hidden></span></label>`;
}

export function hydrateDatasetFieldA11y(root) {
    root?.querySelectorAll('[data-field-shell]')?.forEach((shell, index) => {
        const field = shell.querySelector('[data-field]');
        const error = shell.querySelector('[data-field-error]');
        const hint = shell.querySelector('[data-hint-key]');
        if (!field || !error) return;
        const key = error.dataset.errorKey || `field-${index}`;
        const rowIndex = field.closest('[data-dataset-row]')?.dataset.index;
        const suffix = rowIndex == null ? 'defaults' : `row-${rowIndex}`;
        error.id = `dataset-${suffix}-${key}-error`;
        if (hint) hint.id = `dataset-${suffix}-${key}-hint`;
        field.setAttribute('aria-describedby', [hint?.id, error.id].filter(Boolean).join(' '));
    });
}

export function collectDatasetFields(root, defaults = {}) {
    const values = {};
    root?.querySelectorAll('[data-field]')?.forEach((field) => {
        let value = field.value;
        if (field.tagName === 'SELECT') {
            if (value === 'true') value = true;
            else if (value === 'false') value = false;
        } else if (field.type === 'number') {
            value = value === '' ? undefined : Number(value);
        }
        values[field.dataset.field] = value;
    });
    return { ...defaults, ...values };
}

export function collectDatasetRows(root, defaults) {
    return [...root.querySelectorAll('[data-dataset-row]')].map((row) => {
        const values = collectDatasetFields(row);
        const settings = {};
        DATASET_SETTING_KEYS.forEach((key) => { if (values[key] !== undefined) settings[key] = values[key]; });
        return {
            source_dir: String(values.source_dir || '').trim(),
            image_dir: String(values.image_dir || '').trim(),
            cache_dir: String(values.cache_dir || '').trim(),
            num_repeats: values.num_repeats,
            is_reg: values.is_reg,
            recursive: values.recursive,
            path_pattern: values.path_pattern,
            settings,
            nl_tag_mix: { enabled: values['nl_tag_mix.enabled'], tag_ratio: values['nl_tag_mix.tag_ratio'] },
            trigger_clone: {
                enabled: values['trigger_clone.enabled'],
                prompt: values['trigger_clone.prompt'] || '',
                num_repeats: values['trigger_clone.num_repeats'] ?? 1,
            },
        };
    });
}

export function validateDatasetEditor(root) {
    root.querySelectorAll('[data-field-error]').forEach((item) => { item.hidden = true; item.textContent = ''; });
    root.querySelectorAll('[data-field-shell]').forEach((item) => {
        delete item.dataset.invalid;
        item.querySelector('[data-field]')?.removeAttribute('aria-invalid');
    });
    const errors = [];
    const rows = [...root.querySelectorAll('[data-dataset-row]')];
    rows.forEach((row, index) => {
        const source = row.querySelector('[data-field="source_dir"]');
        if (!String(source?.value || '').trim()) errors.push({ field: source, message: `数据集组 ${index + 1} 需要填写原始图片目录` });
        const cloneEnabled = row.querySelector('[data-field="trigger_clone.enabled"]')?.value === 'true';
        const clonePrompt = row.querySelector('[data-field="trigger_clone.prompt"]');
        if (cloneEnabled && !String(clonePrompt?.value || '').trim()) {
            errors.push({ field: clonePrompt, message: `数据集组 ${index + 1} 启用触发词复制后必须填写触发词` });
        }
    });
    if (rows.length && rows.every((row) => row.querySelector('[data-field="is_reg"]')?.value === 'true')) {
        errors.push({ field: rows[0].querySelector('[data-field="is_reg"]'), message: '至少保留 1 组普通训练数据；正则化数据只能作为辅助数据' });
    }
    root.querySelectorAll('[data-dataset-defaults], [data-dataset-row]').forEach((scope) => validateBucketFields(scope, errors));
    root.querySelectorAll('input[type="number"][data-field]').forEach((field) => validateNumberField(field, errors));
    errors.forEach(({ field, message }) => {
        const shell = field?.closest('[data-field-shell]');
        if (!shell) return;
        shell.dataset.invalid = 'true';
        field.setAttribute('aria-invalid', 'true');
        const error = shell.querySelector('[data-field-error]');
        if (error) { error.textContent = message; error.hidden = false; }
    });
    errors[0]?.field?.focus();
    return errors;
}

function validateBucketFields(scope, errors) {
    const resolution = scope.querySelector('[data-field="resolution"]');
    const minBucket = scope.querySelector('[data-field="min_bucket_reso"]');
    const maxBucket = scope.querySelector('[data-field="max_bucket_reso"]');
    if (Number(minBucket?.value || 0) > Number(maxBucket?.value || 0)) {
        errors.push({ field: minBucket, message: '最小桶尺寸不能大于最大桶尺寸' });
    }
    if (Number(maxBucket?.value || 0) < Number(resolution?.value || 0)) {
        errors.push({ field: maxBucket, message: '最大桶尺寸不能小于训练分辨率，否则预处理会失败' });
    }
}

function validateNumberField(field, errors) {
    if (field.value === '') return;
    const value = Number(field.value);
    const min = field.min === '' ? null : Number(field.min);
    const max = field.max === '' ? null : Number(field.max);
    if (!Number.isFinite(value)) {
        errors.push({ field, message: '请输入有效数字' });
    } else if (min != null && value < min) {
        errors.push({ field, message: `数值不能小于 ${min}` });
    } else if (max != null && value > max) {
        errors.push({ field, message: `数值不能大于 ${max}` });
    }
}

export function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

export function escapeAttribute(value) {
    return escapeHtml(value);
}
