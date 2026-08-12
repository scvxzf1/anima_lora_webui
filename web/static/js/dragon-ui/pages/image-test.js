/* Inference test page backed by ImageTestService. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { mergedConfigUrl, loadTrainingContext } from './training-controls.js?v=dragon-ui-20260812v35';

const api = createApiClient();
let pollTimer = null;

const SAMPLER_OPTIONS = [['euler', 'Euler'], ['er_sde', 'ER-SDE'], ['lcm', 'LCM']];
const DTYPE_OPTIONS = [['bf16', 'BF16'], ['fp16', 'FP16'], ['fp32', 'FP32']];
const TEXT_DTYPE_OPTIONS = [['same', '跟随推理精度'], ...DTYPE_OPTIONS];
const ATTN_OPTIONS = [
    ['flash', 'FlashAttention'], ['torch', 'PyTorch 原生'], ['sageattn', 'SageAttention'],
    ['flex', 'FlexAttention'], ['xformers', 'xFormers'], ['sdpa', 'SDPA'],
];
const LORA_PRESETS = [
    ['default', '全部层位'], ['all_off', '全部关闭'], ['half_strength', '全部半强度'],
    ['main_blocks_only', '仅主干层'], ['llm_adapter_only', '仅文本适配层'], ['late_main', '仅后段主干层'],
    ['mid_late_main', '仅中后段主干层'], ['evens_only', '仅偶数层'], ['odds_only', '仅奇数层'], ['custom', '自定义'],
];
const LORA_BLOCKS = [
    ...Array.from({ length: 28 }, (_, index) => [`block_${index}`, `主干层 ${index}`]),
    ...Array.from({ length: 6 }, (_, index) => [`llm_adapter_${index}`, `文本适配层 ${index}`]),
    ['llm_adapter_io', '文本适配输入输出'], ['final_layer', '最终输出层'], ['t_embedder', '时间步嵌入层'],
    ['x_embedder', '图像嵌入层'], ['other_weights', '其余权重'],
];

export async function loadImageTest() {
    const context = await loadTrainingContext();
    const [status, weights, gpus, config, settings, imagesData] = await Promise.all([
        readApi('/api/image-test/status', {}),
        readApi('/api/analysis/weights', {}),
        readApi('/api/training/gpus', {}),
        readApi(mergedConfigUrl(context), {}),
        readApi('/api/settings/global', {}),
        readApi('/api/preview/images?source=inference&limit=24', {}),
    ]);
    const running = Boolean(status.running || status.status === 'running');
    const images = Array.isArray(imagesData.images) ? imagesData.images : [];
    const weightItems = Array.isArray(weights.weights) ? weights.weights : Array.isArray(weights.items) ? weights.items : [];
    const state = { context, config: config.config || config, settings, status, running, weightItems, gpus: Array.isArray(gpus.gpus) ? gpus.gpus : [] };
    return { html: renderPage(state, images), onMount: (root) => bindPage(root, state) };
}

function renderPage(state, images) {
    const cfg = state.config || {};
    const status = state.status || {};
    const weightOptions = state.weightItems.map((item) => {
        const path = item.abs_path || item.path || item.file || item.absolute_path || '';
        return path ? `<option value="${escapeAttribute(path)}">${escapeHtml(item.name || path)}</option>` : '';
    }).join('');
    const family = String(state.settings?.model_family || cfg.model_family || '').toLowerCase();
    const attnOptions = family === 'krea2_raw' ? ATTN_OPTIONS.filter(([value]) => ['torch', 'flash'].includes(value)) : ATTN_OPTIONS;
    const gpuOptions = state.gpus.map((gpu) => {
        const index = gpu.index ?? gpu.gpu_index ?? '';
        const label = gpu.label || gpu.name || `GPU ${index}`;
        return `<option value="${escapeAttribute(index)}">${escapeHtml(label)}</option>`;
    }).join('');
    const savePath = state.settings?.image_test_save_root || status.output_dir || 'output/tests';
    return `
        <div class="dragon-page dragon-page-wide dragon-image-test-page">
            <div class="dragon-page-hero dragon-reveal"><span class="dragon-eyebrow">模型验证</span><h1>生图测试</h1><p>使用当前训练配置和已生成权重运行一次真实推理。</p></div>
            <form class="dragon-image-test-form dragon-reveal" data-image-test-form data-stagger="1">
                <div class="dragon-image-test-primary">
                    ${textareaField('prompt', '正向提示词', '', '描述要生成的图像内容')}
                    ${textareaField('negative_prompt', '反向提示词', '', '可选')}
                </div>
                <div class="dragon-dataset-field-grid">
                    ${numberField('width', '宽度', cfg.resolution || 1024)}
                    ${numberField('height', '高度', cfg.resolution || 1024)}
                    ${numberField('infer_steps', '采样步数', cfg.sample_steps || 28)}
                    ${numberField('guidance_scale', '引导强度', cfg.guidance_scale || 4, '0.1')}
                    ${numberField('flow_shift', 'Flow Shift', cfg.flow_shift ?? cfg.discrete_flow_shift ?? 1, '0.1')}
                    ${numberField('seed', '随机种子', '', '1')}
                    ${numberField('lora_multiplier', 'LoRA 强度', 1, '0.05')}
                    ${selectField('sampler', '采样器', normalizeChoice(cfg.sample_sampler, SAMPLER_OPTIONS, 'euler'), SAMPLER_OPTIONS)}
                    ${selectField('attn_mode', '注意力后端', normalizeChoice(cfg.attn_mode, attnOptions, family === 'krea2_raw' ? 'torch' : 'flash'), attnOptions)}
                    ${selectField('runtime_dtype', '推理精度', normalizeChoice(cfg.precision_preference, DTYPE_OPTIONS, 'bf16'), DTYPE_OPTIONS)}
                    ${selectField('text_encoder_dtype', '文本编码器精度', 'same', TEXT_DTYPE_OPTIONS)}
                    <label class="dragon-field"><span class="dragon-field-label-text">计算设备</span><select class="dragon-select" data-image-field="gpu_index"><option value="">自动选择</option>${gpuOptions}</select></label>
                    <label class="dragon-field"><span class="dragon-field-label-text">权重文件</span><select class="dragon-select" data-image-field="weight_path"><option value="">使用基础模型</option>${weightOptions}</select></label>
                    ${textField('save_path', '输出目录', savePath)}
                </div>
                ${renderSelectiveLora()}
                <div class="dragon-config-actions"><button class="dragon-btn dragon-btn-primary" type="submit" data-image-action="start" ${state.running ? 'disabled' : ''}>${state.running ? '推理进行中' : '开始推理'}</button>${state.running ? '<button class="dragon-btn dragon-btn-secondary" type="button" data-image-action="stop">停止推理</button>' : ''}</div>
                <p class="dragon-config-feedback" data-image-feedback role="status" aria-live="polite">${escapeHtml(status.error || '')}</p>
            </form>
            <section class="dragon-section dragon-reveal" data-stagger="2"><div class="dragon-section-header-row"><div><span class="dragon-eyebrow">输出结果</span><h2 class="dragon-section-title">推理预览</h2></div><span class="dragon-section-desc">${images.length} 张图片</span></div>${images.length ? `<div class="dragon-image-grid">${images.map(renderImage).join('')}</div>` : '<div class="dragon-empty-state"><p>暂无生成图片</p></div>'}</section>
        </div>
    `;
}

function bindPage(root, state) {
    root.querySelector('[data-image-test-form]')?.addEventListener('submit', async (event) => {
        event.preventDefault();
        await startInference(root, state);
    });
    root.querySelector('[data-image-action="stop"]')?.addEventListener('click', async () => {
        try {
            const data = await api('/api/image-test/stop', { method: 'POST' });
            if (data.ok === false) throw new Error(data.error || '停止推理失败');
            window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
        } catch (error) { showFeedback(root, error.message || '停止推理失败', 'error'); }
    });
    root.querySelector('[data-image-field="anima_selective_preset"]')?.addEventListener('change', (event) => applyLoraPreset(root, event.target.value));
    root.querySelector('[data-image-field="anima_selective_lora"]')?.addEventListener('change', (event) => {
        root.querySelector('[data-selective-lora-fields]')?.toggleAttribute('hidden', !event.target.checked);
    });
    if (state.running) {
        window.clearTimeout(pollTimer);
        pollTimer = window.setTimeout(() => window.dispatchEvent(new CustomEvent('dragon-refresh-route')), 3000);
    }
}

async function startInference(root, state) {
    const button = root.querySelector('[data-image-action="start"]');
    button.disabled = true;
    try {
        const payload = {
            prompt: root.querySelector('[data-image-field="prompt"]')?.value || '',
            negative_prompt: root.querySelector('[data-image-field="negative_prompt"]')?.value || '',
            width: readNumber(root, 'width'), height: readNumber(root, 'height'),
            infer_steps: readNumber(root, 'infer_steps'), guidance_scale: readNumber(root, 'guidance_scale'),
            flow_shift: readNumber(root, 'flow_shift'), seed: readNumber(root, 'seed'),
            lora_multiplier: readNumber(root, 'lora_multiplier'),
            sampler: readValue(root, 'sampler'),
            attn_mode: readValue(root, 'attn_mode'),
            runtime_dtype: readValue(root, 'runtime_dtype'),
            text_encoder_dtype: readValue(root, 'text_encoder_dtype'),
            gpu_index: readValue(root, 'gpu_index'),
            save_path: readValue(root, 'save_path'),
            weight_path: root.querySelector('[data-image-field="weight_path"]')?.value || '',
            ...collectSelectiveLora(root),
            config: state.config || {},
        };
        const data = await api('/api/image-test/start', { method: 'POST', body: JSON.stringify(payload) });
        if (data.ok === false) throw new Error(data.error || '启动推理失败');
        window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    } catch (error) {
        showFeedback(root, error.message || '启动推理失败', 'error');
        button.disabled = false;
    }
}

async function readApi(url, fallback) {
    try { const data = await api(url); return data.ok === false ? fallback : data; } catch { return fallback; }
}

function readNumber(root, key) {
    const value = root.querySelector(`[data-image-field="${key}"]`)?.value || '';
    return value === '' ? '' : Number(value);
}

function readValue(root, key) {
    return root.querySelector(`[data-image-field="${key}"]`)?.value || '';
}

function renderSelectiveLora() {
    return `<details class="dragon-dataset-advanced dragon-image-lora-details"><summary>分层 LoRA 加载</summary><div class="dragon-selective-lora-head"><label class="dragon-check-field"><input type="checkbox" data-image-field="anima_selective_lora"><span>启用分层加载</span></label>${selectField('anima_selective_preset', '层位预设', 'default', LORA_PRESETS)}</div><div data-selective-lora-fields hidden><p class="dragon-section-desc">每个层位可独立关闭或设置 0 至 2 倍强度。</p><div class="dragon-lora-block-grid">${LORA_BLOCKS.map(([key, label]) => `<label class="dragon-lora-block"><span>${label}</span><input class="dragon-input" type="number" min="0" max="2" step="0.05" value="1" data-lora-block="${key}"></label>`).join('')}</div></div></details>`;
}

function applyLoraPreset(root, preset) {
    root.querySelectorAll('[data-lora-block]').forEach((input, index) => {
        const mainIndex = index < 28 ? index : -1;
        const adapterIndex = index >= 28 && index < 34 ? index - 28 : -1;
        let value = 1;
        if (preset === 'all_off') value = 0;
        if (preset === 'half_strength') value = 0.5;
        if (preset === 'main_blocks_only') value = mainIndex >= 0 || [35, 36, 37, 38].includes(index) ? 1 : 0;
        if (preset === 'llm_adapter_only') value = adapterIndex >= 0 || [34, 38].includes(index) ? 1 : 0;
        if (preset === 'late_main') value = mainIndex >= 20 || [35, 36, 37, 38].includes(index) ? 1 : 0;
        if (preset === 'mid_late_main') value = mainIndex >= 14 || [35, 36, 37, 38].includes(index) ? 1 : 0;
        if (preset === 'evens_only') value = (mainIndex >= 0 && mainIndex % 2 === 0) || (adapterIndex >= 0 && adapterIndex % 2 === 0) ? 1 : 0;
        if (preset === 'odds_only') value = (mainIndex >= 0 && mainIndex % 2 === 1) || (adapterIndex >= 0 && adapterIndex % 2 === 1) ? 1 : 0;
        if (preset !== 'custom') input.value = String(value);
    });
}

function collectSelectiveLora(root) {
    const enabled = Boolean(root.querySelector('[data-image-field="anima_selective_lora"]')?.checked);
    const strengths = Object.fromEntries(Array.from(root.querySelectorAll('[data-lora-block]')).map((input) => [input.dataset.loraBlock, Number(input.value || 0)]));
    return {
        anima_selective_lora: enabled,
        anima_selective_preset: readValue(root, 'anima_selective_preset') || 'default',
        anima_selective_strength: 1,
        anima_selective_blocks: enabled ? Object.entries(strengths).filter(([, value]) => value > 0).map(([key]) => key) : [],
        anima_selective_block_strengths: enabled ? strengths : {},
    };
}

function numberField(key, label, value, step = '1') { return `<label class="dragon-field"><span class="dragon-field-label-text">${label}</span><input class="dragon-input" type="number" step="${step}" data-image-field="${key}" value="${escapeAttribute(value)}"></label>`; }
function textField(key, label, value) { return `<label class="dragon-field"><span class="dragon-field-label-text">${label}</span><input class="dragon-input" type="text" data-image-field="${key}" value="${escapeAttribute(value)}"></label>`; }
function selectField(key, label, value, options) { return `<label class="dragon-field"><span class="dragon-field-label-text">${label}</span><select class="dragon-select" data-image-field="${key}">${options.map(([option, text]) => `<option value="${escapeAttribute(option)}" ${String(option) === String(value) ? 'selected' : ''}>${escapeHtml(text)}</option>`).join('')}</select></label>`; }
function textareaField(key, label, value, placeholder) { return `<label class="dragon-field dragon-field-wide"><span class="dragon-field-label-text">${label}</span><textarea class="dragon-textarea" data-image-field="${key}" placeholder="${placeholder}">${escapeHtml(value)}</textarea></label>`; }
function normalizeChoice(value, options, fallback) { return options.some(([option]) => String(option) === String(value)) ? String(value) : fallback; }
function renderImage(image) { return `<figure class="dragon-image-card"><img src="${escapeAttribute(image.url || '')}" alt="${escapeAttribute(image.name || '')}" loading="lazy"><figcaption>${escapeHtml(image.prompt || image.name || '')}</figcaption></figure>`; }
function showFeedback(root, message, tone) { const el = root.querySelector('[data-image-feedback]'); if (el) { el.textContent = message; el.dataset.tone = tone; el.classList.add('dragon-config-feedback-visible'); } }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
function escapeAttribute(value) { return escapeHtml(value); }
