/* Inference test page backed by ImageTestService. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { mergedConfigUrl, loadTrainingContext } from './training-controls.js?v=dragon-ui-20260824v114';
import { renderImageTestPage } from './image-test-view.js?v=dragon-ui-20260814v43';

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
    const [statusResult, weightsResult, gpusResult, configResult, settingsResult] = await Promise.all([
        readApiResult('/api/image-test/status'),
        readApiResult('/api/analysis/weights'),
        readApiResult('/api/training/gpus'),
        readApiResult(mergedConfigUrl(context)),
        readApiResult('/api/settings/global'),
    ]);
    const statusSnapshotAvailable = Boolean(statusResult.data && statusResult.data.status);
    const blockingErrors = [
        ...(!statusResult.ok && !statusSnapshotAvailable ? [statusResult] : []),
        ...[configResult, settingsResult].filter((result) => !result.ok),
    ].map((result) => result.error);
    const status = statusResult.data || {};
    const weights = weightsResult.data || {};
    const gpus = gpusResult.data || {};
    const config = configResult.data || {};
    const settings = settingsResult.data || {};
    const imagesResult = { ok: true, data: { images: status.output_files || [] }, error: '' };
    const imagesData = imagesResult.data;
    const running = Boolean(status.running || status.status === 'running');
    const images = Array.isArray(imagesData.images) ? imagesData.images : [];
    const weightItems = Array.isArray(weights.weights) ? weights.weights : Array.isArray(weights.items) ? weights.items : [];
    const state = {
        context,
        config: config.config || config,
        settings,
        status,
        running,
        blockingError: blockingErrors.join('；'),
        warning: [
            ...(!statusResult.ok && statusSnapshotAvailable ? [statusResult] : []),
            ...[weightsResult, gpusResult, imagesResult].filter((result) => !result.ok),
        ].map((result) => result.error).join('；'),
        weightItems,
        gpus: Array.isArray(gpus.gpus) ? gpus.gpus : [],
    };
    let cleanup = null;
    return {
        html: renderImageTestPage(state, images, {
            sampler: SAMPLER_OPTIONS,
            dtype: DTYPE_OPTIONS,
            textDtype: TEXT_DTYPE_OPTIONS,
            attention: ATTN_OPTIONS,
            loraPresets: LORA_PRESETS,
            loraBlocks: LORA_BLOCKS,
        }),
        onMount: (root) => { cleanup = bindPage(root, state); },
        onUnmount: () => cleanup?.(),
    };
}

function bindPage(root, state) {
    const form = root.querySelector('[data-image-test-form]');
    const submitHandler = async (event) => {
        event.preventDefault();
        await startInference(root, state);
    };
    const clickHandler = async (event) => {
        const action = event.target.closest('[data-image-action]')?.dataset.imageAction;
        if (action === 'refresh') {
            window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
            return;
        }
        if (action !== 'stop') return;
        if (!window.confirm('确认停止当前推理吗？已经生成并保存的图片不会删除。')) return;
        try {
            const data = await api('/api/image-test/stop', { method: 'POST' });
            if (data.ok === false) throw new Error(data.error || '停止推理失败');
            window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
        } catch (error) { showFeedback(root, error.message || '停止推理失败', 'error'); }
    };
    form?.addEventListener('submit', submitHandler);
    root.addEventListener('click', clickHandler);
    root.querySelector('[data-image-field="anima_selective_preset"]')?.addEventListener('change', (event) => applyLoraPreset(root, event.target.value));
    root.querySelector('[data-image-field="anima_selective_lora"]')?.addEventListener('change', (event) => {
        root.querySelector('[data-selective-lora-fields]')?.toggleAttribute('hidden', !event.target.checked);
    });
    root.querySelector('[data-image-action="resolve-weight"]')?.addEventListener('click', () => resolveWeight(root));
    bindWeightDrop(root);
    if (state.running) {
        window.clearTimeout(pollTimer);
        pollTimer = window.setTimeout(() => window.dispatchEvent(new CustomEvent('dragon-refresh-route')), 3000);
    }
    return () => {
        window.clearTimeout(pollTimer);
        pollTimer = null;
        form?.removeEventListener('submit', submitHandler);
        root.removeEventListener('click', clickHandler);
    };
}

function bindWeightDrop(root) {
    const input = root.querySelector('[data-image-field="weight_path"]');
    const row = input?.closest('.dragon-image-weight-row');
    if (!input || !row) return;
    const prevent = (event) => {
        event.preventDefault();
        event.stopPropagation();
    };
    row.addEventListener('dragenter', (event) => { prevent(event); row.dataset.dragging = 'true'; });
    row.addEventListener('dragover', (event) => { prevent(event); row.dataset.dragging = 'true'; });
    row.addEventListener('dragleave', (event) => {
        prevent(event);
        if (!row.contains(event.relatedTarget)) delete row.dataset.dragging;
    });
    row.addEventListener('drop', (event) => {
        prevent(event);
        delete row.dataset.dragging;
        const path = droppedWeightPath(event.dataTransfer);
        if (!path) return showFeedback(root, '未能从拖入内容中读取权重路径。请改为粘贴完整路径。', 'error');
        input.value = path;
        resolveWeight(root);
    });
}

function droppedWeightPath(dataTransfer) {
    const file = dataTransfer?.files?.[0];
    if (file?.path) return file.path;
    const uriList = String(dataTransfer?.getData?.('text/uri-list') || '').split(/\r?\n/).find((line) => line && !line.startsWith('#')) || '';
    const plain = String(dataTransfer?.getData?.('text/plain') || '').trim();
    const candidate = uriList || plain;
    if (!candidate) return file?.name || '';
    try {
        const url = new URL(candidate);
        if (url.protocol === 'file:') return decodeURIComponent(url.pathname || '');
    } catch { /* plain filesystem path */ }
    return candidate.replace(/^['"]|['"]$/g, '').trim();
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

async function resolveWeight(root) {
    const input = root.querySelector('[data-image-field="weight_path"]');
    const path = String(input?.value || '').trim();
    if (!path) return showFeedback(root, '请先填写或选择一个权重路径。', 'error');
    try {
        const payload = await api('/api/image-test/resolve-weight', { method: 'POST', body: JSON.stringify({ path }) });
        if (payload?.ok === false) throw new Error(payload.error || '解析权重路径失败');
        if (input) input.value = payload.weight_path || payload.display_path || path;
        showFeedback(root, `已找到权重：${payload.display_path || payload.name || path}`, 'success');
    } catch (error) {
        showFeedback(root, error.message || '解析权重路径失败', 'error');
    }
}

async function readApiResult(url) {
    try {
        const data = await api(url);
        if (data?.ok === false) return { ok: false, data, error: data.error || '服务请求失败' };
        return { ok: true, data, error: '' };
    } catch (error) {
        return { ok: false, data: null, error: error.message || '服务请求失败' };
    }
}

function readNumber(root, key) {
    const value = root.querySelector(`[data-image-field="${key}"]`)?.value || '';
    return value === '' ? '' : Number(value);
}

function readValue(root, key) {
    return root.querySelector(`[data-image-field="${key}"]`)?.value || '';
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

function showFeedback(root, message, tone) { const el = root.querySelector('[data-image-feedback]'); if (el) { el.textContent = message; el.dataset.tone = tone; el.classList.add('dragon-config-feedback-visible'); } }
