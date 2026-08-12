/* Dataset editor page backed by /api/config/datasets. */

import { createApiClient } from '../../shared/api.js?v=apple-ui-20260812v33';
import { loadTrainingContext } from './training-controls.js?v=apple-ui-20260812v33';

const api = createApiClient();
const CAPTION_MODES = [
    ['auto', '自动识别'],
    ['txt', '同名 TXT'],
    ['json', 'JSON 标注'],
    ['captions_json', 'captions.json'],
];

export async function loadDatasetEditor() {
    const context = await loadTrainingContext();
    const params = new URLSearchParams({
        variant: context.variant,
        preset: context.preset,
        methods_subdir: context.methodsSubdir,
        config_file: context.configFile,
    });
    let data = {};
    let error = '';
    try {
        data = await api(`/api/config/datasets?${params.toString()}`);
        if (data.ok === false) error = data.error || '读取数据集配置失败';
    } catch (cause) {
        error = cause.message || '读取数据集配置失败';
    }

    const rows = Array.isArray(data.datasets) ? data.datasets : [];
    const defaults = data.defaults && typeof data.defaults === 'object' ? data.defaults : {};
    const editor = {
        context,
        rows: rows.length ? rows : [emptyRow(defaults)],
        defaults,
        datasetConfig: data.dataset_config || '',
        feedback: error,
    };
    return {
        html: renderPage(editor),
        onMount: (root) => bindEditor(root, editor),
    };
}

function renderPage(editor) {
    const { context, rows, defaults, datasetConfig, feedback } = editor;
    return `
        <div class="apple-page apple-page-wide apple-dataset-page">
            <div class="apple-page-hero apple-reveal">
                <span class="apple-eyebrow">数据与标注</span>
                <h1>数据集蓝图</h1>
                <p>配置当前训练来源的图片路径、分桶、验证集和标注规则。保存后会同步写入数据集配置与训练配置。</p>
            </div>
            <section class="apple-dataset-context apple-reveal" data-stagger="1">
                <div><span>训练配置</span><strong>${escapeHtml(context.configFile || '未选择')}</strong></div>
                <div><span>运行预设</span><strong>${escapeHtml(context.preset || '默认')}</strong></div>
                <div><span>数据集文件</span><strong class="apple-text-mono">${escapeHtml(datasetConfig || '保存时自动生成')}</strong></div>
            </section>
            ${feedback ? `<div class="apple-config-feedback apple-config-feedback-visible" data-tone="error">${escapeHtml(feedback)}</div>` : ''}
            <form class="apple-dataset-form" data-dataset-form>
                <section class="apple-section apple-dataset-defaults apple-reveal" data-stagger="2">
                    <div class="apple-section-header-row">
                        <div><span class="apple-eyebrow">默认规则</span><h2 class="apple-section-title">通用数据设置</h2></div>
                        <span class="apple-section-desc">每组数据集未单独设置时使用这里的值。</span>
                    </div>
                    ${renderDefaultFields(defaults)}
                </section>
                <section class="apple-section apple-dataset-groups apple-reveal" data-stagger="3">
                    <div class="apple-section-header-row">
                        <div><span class="apple-eyebrow">数据来源</span><h2 class="apple-section-title">数据集分组</h2></div>
                        <button class="apple-btn apple-btn-secondary" type="button" data-dataset-add>添加数据集组</button>
                    </div>
                    <div data-dataset-rows>${rows.map((row, index) => renderRow(row, index, defaults)).join('')}</div>
                </section>
                <div class="apple-config-actions apple-config-actions-sticky">
                    <button class="apple-btn apple-btn-primary" type="submit" data-dataset-save>保存数据集配置</button>
                    <span class="apple-config-feedback" data-dataset-feedback role="status" aria-live="polite"></span>
                </div>
            </form>
        </div>
    `;
}

function renderDefaultFields(defaults) {
    return `<div class="apple-dataset-field-grid">${[
        numberField('resolution', '分辨率', defaults.resolution ?? 1024),
        numberField('batch_size', '批次大小', defaults.batch_size ?? 1),
        numberField('prior_loss_weight', '先验损失权重', defaults.prior_loss_weight ?? 1, '0.05'),
        selectField('enable_bucket', '启用分桶', Boolean(defaults.enable_bucket ?? true), [['true', '开启'], ['false', '关闭']]),
        numberField('min_bucket_reso', '最小桶尺寸', defaults.min_bucket_reso ?? 256),
        numberField('max_bucket_reso', '最大桶尺寸', defaults.max_bucket_reso ?? 1024),
        numberField('bucket_reso_steps', '桶尺寸步长', defaults.bucket_reso_steps ?? 64),
        selectField('bucket_no_upscale', '禁止放大', Boolean(defaults.bucket_no_upscale), [['true', '开启'], ['false', '关闭']]),
        numberField('validation_split', '验证集比例', defaults.validation_split ?? 0, '0.01'),
        numberField('validation_split_num', '验证集数量', defaults.validation_split_num ?? 0),
        numberField('validation_seed', '验证随机种子', defaults.validation_seed ?? 42),
        textField('caption_extension', '标注扩展名', defaults.caption_extension ?? '.txt'),
        numberField('keep_tokens', '保留 Token 数', defaults.keep_tokens ?? 3),
        selectField('prefer_json_caption', '优先 JSON 标注', Boolean(defaults.prefer_json_caption), [['true', '开启'], ['false', '关闭']]),
        selectField('caption_source_mode', '标注来源', defaults.caption_source_mode || 'auto', CAPTION_MODES),
    ].join('')}</div>`;
}

function renderRow(row, index, defaults) {
    const settings = { ...defaults, ...(row.settings || {}) };
    const mix = row.nl_tag_mix || {};
    const clone = row.trigger_clone || {};
    return `
        <article class="apple-dataset-row" data-dataset-row data-index="${index}">
            <header class="apple-dataset-row-head">
                <div><span class="apple-eyebrow">数据集组 ${index + 1}</span><h3>训练来源与子集规则</h3></div>
                <button class="apple-btn apple-btn-ghost apple-btn-sm" type="button" data-dataset-remove ${index === 0 ? 'disabled' : ''}>移除</button>
            </header>
            <div class="apple-dataset-field-grid apple-dataset-path-grid">
                ${textField('source_dir', '原始图片目录', row.source_dir || '')}
                ${textField('image_dir', '缩放图片目录', row.image_dir || '')}
                ${textField('cache_dir', 'LoRA 缓存目录', row.cache_dir || '')}
                ${numberField('num_repeats', '重复次数', row.num_repeats ?? 1)}
                ${selectField('is_reg', '正则数据集', Boolean(row.is_reg), [['true', '开启'], ['false', '关闭']])}
                ${selectField('recursive', '递归扫描', row.recursive !== false, [['true', '开启'], ['false', '关闭']])}
                ${textField('path_pattern', '路径筛选', row.path_pattern || '*')}
            </div>
            <details class="apple-dataset-advanced">
                <summary>展开本组高级规则</summary>
                <div class="apple-dataset-field-grid">
                    ${numberField('resolution', '本组分辨率', settings.resolution ?? 1024)}
                    ${numberField('batch_size', '本组批次大小', settings.batch_size ?? 1)}
                    ${numberField('prior_loss_weight', '本组先验损失权重', settings.prior_loss_weight ?? 1, '0.05')}
                    ${selectField('enable_bucket', '本组启用分桶', Boolean(settings.enable_bucket ?? true), [['true', '开启'], ['false', '关闭']])}
                    ${numberField('min_bucket_reso', '本组最小桶尺寸', settings.min_bucket_reso ?? 256)}
                    ${numberField('max_bucket_reso', '本组最大桶尺寸', settings.max_bucket_reso ?? 1024)}
                    ${numberField('bucket_reso_steps', '本组桶步长', settings.bucket_reso_steps ?? 64)}
                    ${selectField('bucket_no_upscale', '本组禁止放大', Boolean(settings.bucket_no_upscale), [['true', '开启'], ['false', '关闭']])}
                    ${textField('caption_extension', '本组标注扩展名', settings.caption_extension ?? '.txt')}
                    ${numberField('keep_tokens', '本组保留 Token 数', settings.keep_tokens ?? 3)}
                    ${selectField('caption_source_mode', '本组标注来源', settings.caption_source_mode || 'auto', CAPTION_MODES)}
                    ${selectField('prefer_json_caption', '本组优先 JSON', Boolean(settings.prefer_json_caption), [['true', '开启'], ['false', '关闭']])}
                    ${numberField('validation_split', '本组验证集比例', settings.validation_split ?? 0, '0.01')}
                    ${numberField('validation_split_num', '本组验证集数量', settings.validation_split_num ?? 0)}
                    ${numberField('validation_seed', '本组验证随机种子', settings.validation_seed ?? 42)}
                    ${selectField('nl_tag_mix.enabled', '标签混合', Boolean(mix.enabled), [['true', '开启'], ['false', '关闭']])}
                    ${numberField('nl_tag_mix.tag_ratio', '标签混合比例', mix.tag_ratio ?? 0.7, '0.01')}
                    ${selectField('trigger_clone.enabled', '触发词复制', Boolean(clone.enabled), [['true', '开启'], ['false', '关闭']])}
                    ${textField('trigger_clone.prompt', '触发词', clone.prompt || '')}
                    ${numberField('trigger_clone.num_repeats', '触发词复制次数', clone.num_repeats ?? 1)}
                </div>
            </details>
        </article>
    `;
}

function textField(key, label, value) {
    return fieldShell(key, label, `<input class="apple-input" type="text" data-field="${key}" value="${escapeAttribute(value)}">`);
}

function numberField(key, label, value, step = '1') {
    return fieldShell(key, label, `<input class="apple-input" type="number" step="${step}" data-field="${key}" value="${escapeAttribute(value)}">`);
}

function selectField(key, label, value, options) {
    const selected = typeof value === 'boolean' ? String(value) : String(value ?? '');
    return fieldShell(key, label, `<select class="apple-select" data-field="${key}">${options.map(([option, text]) => `<option value="${escapeAttribute(option)}" ${option === selected ? 'selected' : ''}>${text}</option>`).join('')}</select>`);
}

function fieldShell(key, label, control) {
    return `<label class="apple-field" data-field-shell="${key}"><span class="apple-field-label-text">${label}</span>${control}</label>`;
}

function emptyRow(defaults) {
    return { source_dir: '', image_dir: '', cache_dir: '', num_repeats: 1, is_reg: false, recursive: true, path_pattern: '*', settings: { ...defaults }, nl_tag_mix: { enabled: false, tag_ratio: 0.7 }, trigger_clone: { enabled: false, prompt: '', num_repeats: 1 } };
}

function bindEditor(root, editor) {
    const rowsRoot = root.querySelector('[data-dataset-rows]');
    root.querySelector('[data-dataset-add]')?.addEventListener('click', () => {
        const index = rowsRoot?.querySelectorAll('[data-dataset-row]').length || 0;
        rowsRoot?.insertAdjacentHTML('beforeend', renderRow(emptyRow(editor.defaults), index, editor.defaults));
        bindRowRemoval(root);
    });
    bindRowRemoval(root);
    root.querySelector('[data-dataset-form]')?.addEventListener('submit', async (event) => {
        event.preventDefault();
        await saveEditor(root, editor);
    });
}

function bindRowRemoval(root) {
    root.querySelectorAll('[data-dataset-remove]').forEach((button) => {
        if (button.dataset.bound) return;
        button.dataset.bound = 'true';
        button.addEventListener('click', () => {
            const rows = root.querySelectorAll('[data-dataset-row]');
            if (rows.length <= 1) return;
            button.closest('[data-dataset-row]')?.remove();
            root.querySelectorAll('[data-dataset-row]').forEach((row, index) => {
                row.dataset.index = String(index);
                const eyebrow = row.querySelector('.apple-dataset-row-head .apple-eyebrow');
                if (eyebrow) eyebrow.textContent = `数据集组 ${index + 1}`;
            });
        });
    });
}

async function saveEditor(root, editor) {
    const button = root.querySelector('[data-dataset-save]');
    const feedback = root.querySelector('[data-dataset-feedback]');
    button.disabled = true;
    setFeedback(feedback, '正在保存...', 'info');
    try {
        const payload = {
            variant: editor.context.variant,
            preset: editor.context.preset,
            methods_subdir: editor.context.methodsSubdir,
            datasets: collectRows(root, editor.defaults),
            defaults: collectFields(root.querySelector('.apple-dataset-defaults'), editor.defaults),
            config_values: {},
            train_file: editor.context.configFile,
            prefer_existing_dataset_config: true,
        };
        const result = await api('/api/config/datasets', { method: 'PUT', body: JSON.stringify(payload) });
        if (result.ok === false) throw new Error(result.error || '保存数据集配置失败');
        editor.datasetConfig = result.dataset_config || editor.datasetConfig;
        setFeedback(feedback, result.message || '数据集配置已保存', 'success');
    } catch (error) {
        setFeedback(feedback, error.message || '保存数据集配置失败', 'error');
    } finally {
        button.disabled = false;
    }
}

function collectRows(root, defaults) {
    return [...root.querySelectorAll('[data-dataset-row]')].map((row) => {
        const values = collectFields(row, defaults);
        const settings = {};
        const flatKeys = ['resolution', 'batch_size', 'prior_loss_weight', 'enable_bucket', 'min_bucket_reso', 'max_bucket_reso', 'bucket_reso_steps', 'bucket_no_upscale', 'validation_split', 'validation_split_num', 'validation_seed', 'caption_extension', 'keep_tokens', 'prefer_json_caption', 'caption_source_mode'];
        flatKeys.forEach((key) => { if (values[key] !== undefined) settings[key] = values[key]; });
        return {
            source_dir: values.source_dir,
            image_dir: values.image_dir,
            cache_dir: values.cache_dir,
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

function collectFields(root, defaults = {}) {
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

function setFeedback(element, message, tone) {
    if (!element) return;
    element.textContent = message;
    element.dataset.tone = tone;
    element.classList.add('apple-config-feedback-visible');
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
