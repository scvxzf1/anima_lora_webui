import {
    fetchImageTestImages,
    fetchImageTestStatus,
    fetchImageTestWeights,
    startImageTestRequest,
    stopImageTestRequest,
} from './api.js?v=module-bootstrap-20260702-1';
import { createImageTestRenderer } from './render.js?v=module-bootstrap-20260702-1';
import { createImageTestState, IMAGE_TEST_DEFAULTS } from './state.js?v=module-bootstrap-20260702-1';

const IMAGE_TEST_IMAGE_LIMIT = 24;
const IMAGE_TEST_STATUS_POLL_MS = 3000;

export function createImageTestFeature(ctx, deps) {
    const state = createImageTestState();
    const renderer = createImageTestRenderer({
        ctx,
        state,
        openPreviewDialog: deps.openPreviewDialog,
    });

    function bindImageTestEvents() {
        if (state.initialized) return;
        state.initialized = true;
        renderer.initStaticUI();
        document.getElementById('btn-refresh-image-test-status')?.addEventListener('click', () => {
            loadImageTestPage({ force: true });
        });
        document.getElementById('btn-refresh-image-test-weights')?.addEventListener('click', () => {
            loadImageTestWeights({ force: true });
        });
        document.getElementById('btn-start-image-test')?.addEventListener('click', startImageTest);
        document.getElementById('btn-stop-image-test')?.addEventListener('click', stopImageTest);
        document.getElementById('image-test-weight-select')?.addEventListener('change', (event) => {
            const path = event.target.value || '';
            if (path) {
                document.getElementById('image-test-weight-path').value = path;
            }
        });
    }

    async function loadImageTestPage(options = {}) {
        bindImageTestEvents();
        if (!state.syncReady) {
            syncFromCurrentConfig({ force: true });
        }
        if (location.protocol === 'file:') {
            renderer.setImageTestStatus('静态打开没有后端 API，无法运行生图测试。', 'error');
            renderer.setImageEmpty('静态打开模式下无法读取 output/tests。');
            return;
        }
        await Promise.all([
            loadImageTestStatus({ refreshImages: true, force: options.force }),
            loadImageTestWeights({ force: options.force }),
        ]);
    }

    function syncFromCurrentConfig(options = {}) {
        const cfg = deps.getCurrentConfig();
        if (!cfg || !Object.keys(cfg).length) {
            renderer.setImageTestStatus('当前还没有加载配置，先回到“配置”页读取一个变体。', 'error');
            renderer.syncButtons(state.lastStatus || {});
            return false;
        }
        state.configSnapshot = deepClone(cfg);
        setInputValue('image-test-width', resolvePositiveInt(cfg.resolution, IMAGE_TEST_DEFAULTS.width), options.force);
        setInputValue('image-test-height', resolvePositiveInt(cfg.resolution, IMAGE_TEST_DEFAULTS.height), options.force);
        setInputValue('image-test-infer-steps', resolvePositiveInt(cfg.sample_steps ?? cfg.infer_steps, IMAGE_TEST_DEFAULTS.inferSteps), options.force);
        setInputValue('image-test-guidance-scale', resolveNumber(cfg.guidance_scale ?? cfg.cfg_scale, IMAGE_TEST_DEFAULTS.guidanceScale), options.force);
        setInputValue('image-test-flow-shift', resolveNumber(cfg.flow_shift ?? cfg.discrete_flow_shift, IMAGE_TEST_DEFAULTS.flowShift), options.force);
        setInputValue('image-test-seed', cfg.seed ?? '', options.force);
        setInputValue('image-test-sampler', normalizeSampler(cfg.sample_sampler), options.force);
        setInputValue('image-test-attn-mode', normalizeAttnMode(cfg.attn_mode), options.force);
        setInputValue('image-test-runtime-dtype', normalizeRuntimeDtype(cfg.precision_preference), options.force);
        setInputValue('image-test-text-encoder-dtype', normalizeTextEncoderDtype(), options.force);
        state.syncReady = true;
        renderer.syncButtons(state.lastStatus || {});
        return true;
    }

    async function loadImageTestStatus(options = {}) {
        if (state.loadingStatus && !options.force) return state.lastStatus;
        state.loadingStatus = true;
        try {
            const payload = await fetchImageTestStatus(ctx);
            state.lastStatus = payload;
            renderer.renderRuntime(payload);
            if (options.refreshImages !== false) {
                await loadImageTestImages();
            }
            scheduleStatusPoll(payload);
            return payload;
        } catch (e) {
            renderer.setImageTestStatus('读取生图测试状态失败: ' + e.message, 'error');
            return null;
        } finally {
            state.loadingStatus = false;
        }
    }

    async function loadImageTestWeights(options = {}) {
        if (state.loadingWeights && !options.force) return;
        state.loadingWeights = true;
        try {
            renderer.renderWeightOptions(await fetchImageTestWeights(ctx));
        } catch (e) {
            renderer.renderWeightOptions({ ok: false, weights: [], message: '读取权重列表失败: ' + e.message });
        } finally {
            state.loadingWeights = false;
        }
    }

    async function loadImageTestImages() {
        const requestSeq = ++state.imageRequestSeq;
        state.loadingImages = true;
        renderer.setImageLoading();
        try {
            const payload = await fetchImageTestImages(ctx, IMAGE_TEST_IMAGE_LIMIT);
            if (requestSeq !== state.imageRequestSeq) return;
            if (payload.ok === false) {
                renderer.setImageEmpty(payload.error || '读取 output/tests 失败');
                return;
            }
            renderer.renderImages(payload);
        } catch (e) {
            if (requestSeq !== state.imageRequestSeq) return;
            renderer.setImageEmpty('读取 output/tests 失败: ' + e.message);
        } finally {
            if (requestSeq === state.imageRequestSeq) {
                state.loadingImages = false;
            }
        }
    }

    async function startImageTest() {
        if (!syncFromCurrentConfig({ force: false })) return;
        const payload = collectRequestPayload();
        if (!payload.prompt) {
            renderer.setImageTestStatus('请输入正向提示词。', 'error');
            return;
        }
        state.starting = true;
        renderer.syncButtons(state.lastStatus || {});
        renderer.setImageTestStatus('正在启动生图测试…', '');
        try {
            const res = await startImageTestRequest(ctx, payload);
            if (res.ok === false) {
                renderer.setImageTestStatus(res.error || '启动失败', 'error');
                return;
            }
            state.lastStatus = res;
            renderer.renderRuntime(res);
            await loadImageTestImages();
            scheduleStatusPoll(res);
        } catch (e) {
            renderer.setImageTestStatus('启动失败: ' + e.message, 'error');
        } finally {
            state.starting = false;
            renderer.syncButtons(state.lastStatus || {});
        }
    }

    async function stopImageTest() {
        state.stopping = true;
        renderer.syncButtons(state.lastStatus || {});
        try {
            const res = await stopImageTestRequest(ctx);
            if (res.ok === false) {
                renderer.setImageTestStatus(res.error || '停止失败', 'error');
                return;
            }
            state.lastStatus = res;
            renderer.renderRuntime(res);
            await loadImageTestImages();
        } catch (e) {
            renderer.setImageTestStatus('停止失败: ' + e.message, 'error');
        } finally {
            state.stopping = false;
            renderer.syncButtons(state.lastStatus || {});
        }
    }

    function collectRequestPayload() {
        return {
            prompt: readValue('image-test-prompt'),
            negative_prompt: readValue('image-test-negative-prompt'),
            width: readValue('image-test-width'),
            height: readValue('image-test-height'),
            infer_steps: readValue('image-test-infer-steps'),
            guidance_scale: readValue('image-test-guidance-scale'),
            flow_shift: readValue('image-test-flow-shift'),
            sampler: readValue('image-test-sampler'),
            attn_mode: readValue('image-test-attn-mode'),
            runtime_dtype: readValue('image-test-runtime-dtype'),
            text_encoder_dtype: readValue('image-test-text-encoder-dtype'),
            seed: readValue('image-test-seed'),
            weight_path: readValue('image-test-weight-path') || readValue('image-test-weight-select'),
            lora_multiplier: readValue('image-test-lora-multiplier'),
            config: deepClone(state.configSnapshot || deps.getCurrentConfig() || {}),
        };
    }

    function scheduleStatusPoll(status = {}) {
        if (state.pollTimer) {
            window.clearTimeout(state.pollTimer);
            state.pollTimer = null;
        }
        if (!status?.running) return;
        state.pollTimer = window.setTimeout(() => {
            loadImageTestStatus({ refreshImages: true, force: true });
        }, IMAGE_TEST_STATUS_POLL_MS);
    }

    function setInputValue(id, value, force = false) {
        const input = document.getElementById(id);
        if (!input) return;
        if (!force && String(input.value || '').trim()) return;
        input.value = value ?? '';
    }

    function readValue(id) {
        return String(document.getElementById(id)?.value || '').trim();
    }

    function deepClone(value) {
        return JSON.parse(JSON.stringify(value || {}));
    }

    function resolvePositiveInt(value, fallback) {
        const num = Number.parseInt(String(value ?? '').trim(), 10);
        return Number.isFinite(num) && num > 0 ? num : fallback;
    }

    function resolveNumber(value, fallback) {
        const num = Number.parseFloat(String(value ?? '').trim());
        return Number.isFinite(num) ? num : fallback;
    }

    function normalizeSampler(value) {
        const normalized = String(value || '').trim().toLowerCase();
        return ['euler', 'er_sde', 'lcm'].includes(normalized)
            ? normalized
            : IMAGE_TEST_DEFAULTS.sampler;
    }

    function normalizeAttnMode(value) {
        const normalized = String(value || '').trim().toLowerCase();
        return ['flash', 'torch', 'sageattn', 'flex', 'xformers', 'sdpa'].includes(normalized)
            ? normalized
            : IMAGE_TEST_DEFAULTS.attnMode;
    }

    function normalizeRuntimeDtype(value) {
        const normalized = String(value || '').trim().toLowerCase();
        return ['bf16', 'fp16', 'fp32'].includes(normalized)
            ? normalized
            : IMAGE_TEST_DEFAULTS.runtimeDtype;
    }

    function normalizeTextEncoderDtype(value = '') {
        const normalized = String(value || '').trim().toLowerCase();
        return ['same', 'bf16', 'fp16', 'fp32'].includes(normalized)
            ? normalized
            : IMAGE_TEST_DEFAULTS.textEncoderDtype;
    }

    return {
        bindImageTestEvents,
        loadImageTestPage,
        syncFromCurrentConfig,
    };
}
