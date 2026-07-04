import {
    fetchImageTestGpus,
    fetchImageTestImages,
    fetchImageTestStatus,
    fetchImageTestWeights,
    resolveImageTestWeightPathRequest,
    startImageTestRequest,
    stopImageTestRequest,
} from './api.js?v=module-bootstrap-20260704-2';
import { createImageTestRenderer } from './render.js?v=module-bootstrap-20260704-2';
import { createImageTestSelectiveLoraController } from './selective-lora.js?v=module-bootstrap-20260704-2';
import { createImageTestState, IMAGE_TEST_DEFAULTS } from './state.js?v=module-bootstrap-20260704-2';
import { createImageTestUiStorage } from './storage.js?v=module-bootstrap-20260704-2';

const IMAGE_TEST_IMAGE_LIMIT = 500;
const IMAGE_TEST_STATUS_POLL_MS = 3000;

export function createImageTestFeature(ctx, deps) {
    const state = createImageTestState();
    const draftStore = createImageTestUiStorage();
    const selectiveLora = createImageTestSelectiveLoraController();
    const renderer = createImageTestRenderer({
        ctx,
        state,
        openPreviewDialog: deps.openPreviewDialog,
        initialHistoryFilter: draftStore.storedHistoryRange(),
        requestHistoryReload: (nextRange) => {
            draftStore.persistFromDom({ history_range: nextRange });
            void loadImageTestImages({ force: true, historyRange: nextRange });
        },
    });

    function bindImageTestEvents() {
        if (state.initialized) return;
        state.initialized = true;
        renderer.initStaticUI();
        selectiveLora.init();
        state.restoredDraftFieldIds = draftStore.restoreToDom();
        state.hasPersistedDraft = draftStore.hasStoredDraft();
        draftStore.bind((fieldId) => {
            state.restoredDraftFieldIds.add(fieldId);
        });
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
            state.restoredDraftFieldIds.add('image-test-weight-select');
            state.restoredDraftFieldIds.add('image-test-weight-path');
            draftStore.persistFromDom({ history_range: renderer.currentHistoryFilter() });
        });
        bindWeightDropTargetEvents();
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
            loadImageTestGpus({ force: options.force }),
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
        setInputValue('image-test-width', resolvePositiveInt(cfg.resolution, IMAGE_TEST_DEFAULTS.width), { force: options.force });
        setInputValue('image-test-height', resolvePositiveInt(cfg.resolution, IMAGE_TEST_DEFAULTS.height), { force: options.force });
        setInputValue('image-test-infer-steps', resolvePositiveInt(cfg.sample_steps ?? cfg.infer_steps, IMAGE_TEST_DEFAULTS.inferSteps), { force: options.force });
        setInputValue('image-test-guidance-scale', resolveNumber(cfg.guidance_scale ?? cfg.cfg_scale, IMAGE_TEST_DEFAULTS.guidanceScale), { force: options.force });
        setInputValue('image-test-flow-shift', resolveNumber(cfg.flow_shift ?? cfg.discrete_flow_shift, IMAGE_TEST_DEFAULTS.flowShift), { force: options.force });
        setInputValue('image-test-seed', cfg.seed ?? '', { force: options.force });
        setInputValue('image-test-sampler', normalizeSampler(cfg.sample_sampler), { force: options.force });
        setInputValue('image-test-attn-mode', normalizeAttnMode(cfg.attn_mode), { force: options.force });
        setInputValue('image-test-runtime-dtype', normalizeRuntimeDtype(cfg.precision_preference), { force: options.force });
        setInputValue('image-test-text-encoder-dtype', normalizeTextEncoderDtype(), { force: options.force });
        state.syncReady = true;
        draftStore.persistFromDom({ history_range: renderer.currentHistoryFilter() });
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
            draftStore.restoreDeferredField('image-test-weight-select');
            resolveWeightPathFromCandidates();
        } catch (e) {
            renderer.renderWeightOptions({ ok: false, weights: [], message: '读取权重列表失败: ' + e.message });
        } finally {
            state.loadingWeights = false;
        }
    }

    async function loadImageTestGpus(options = {}) {
        if (state.loadingGpus && !options.force) return;
        state.loadingGpus = true;
        try {
            renderer.renderGpuOptions(await fetchImageTestGpus(ctx));
            draftStore.restoreDeferredField('image-test-gpu-index');
        } catch (e) {
            renderer.renderGpuOptions({ ok: false, gpus: [], message: '读取 GPU 列表失败: ' + e.message });
        } finally {
            state.loadingGpus = false;
        }
    }

    async function loadImageTestImages(options = {}) {
        const requestSeq = ++state.imageRequestSeq;
        state.loadingImages = true;
        renderer.setImageLoading();
        try {
            const historyRange = options.historyRange || renderer.currentHistoryFilter();
            const payload = await fetchImageTestImages(ctx, IMAGE_TEST_IMAGE_LIMIT, historyRange);
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
        resolveWeightPathFromCandidates();
        const payload = collectRequestPayload();
        if (!payload.prompt) {
            renderer.setImageTestStatus('请输入正向提示词。', 'error');
            return;
        }
        const selectiveError = selectiveLora.validate(payload);
        if (selectiveError) {
            renderer.setImageTestStatus(selectiveError, 'error');
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
            gpu_index: readValue('image-test-gpu-index'),
            seed: readValue('image-test-seed'),
            weight_path: readValue('image-test-weight-path') || readValue('image-test-weight-select'),
            lora_multiplier: readValue('image-test-lora-multiplier'),
            ...selectiveLora.collectPayload(),
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

    function setInputValue(id, value, options = {}) {
        const input = document.getElementById(id);
        if (!input) return;
        if (state.restoredDraftFieldIds.has(id)) return;
        if (!options.force && String(input.value || '').trim()) return;
        input.value = value ?? '';
    }

    function readValue(id) {
        return String(document.getElementById(id)?.value || '').trim();
    }

    function bindWeightDropTargetEvents() {
        const dropTargets = [
            document.getElementById('image-test-weight-drop-target'),
            document.getElementById('image-test-weight-path'),
        ].filter(Boolean);
        if (!dropTargets.length) return;
        const prevent = (event) => {
            event.preventDefault();
            event.stopPropagation();
        };
        dropTargets.forEach((dropTarget) => {
            bindSingleWeightDropTarget(dropTarget, prevent);
        });
    }

    function bindSingleWeightDropTarget(dropTarget, prevent) {
        ['dragenter', 'dragover'].forEach((type) => {
            dropTarget.addEventListener(type, (event) => {
                prevent(event);
                if (event.dataTransfer) {
                    event.dataTransfer.dropEffect = 'copy';
                }
                dropTarget.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach((type) => {
            dropTarget.addEventListener(type, (event) => {
                prevent(event);
                if (type === 'dragleave') {
                    dropTarget.classList.remove('dragover');
                    return;
                }
                void handleWeightDrop(event);
            });
        });
    }

    function clearWeightDropTargetState() {
        ['image-test-weight-drop-target', 'image-test-weight-path'].forEach((id) => {
            document.getElementById(id)?.classList.remove('dragover');
        });
    }

    async function handleWeightDrop(event) {
        clearWeightDropTargetState();
        const droppedPath = droppedSafetensorsPath(event.dataTransfer);
        if (!droppedPath) {
            renderer.setImageTestStatus('没有识别到可用的 .safetensors 权重路径。', 'error');
            return;
        }
        await applyDroppedWeightPath(droppedPath);
    }

    async function applyDroppedWeightPath(path) {
        const normalizedPath = normalizeDroppedWeightPath(path);
        if (!isSafetensorsPath(normalizedPath)) {
            renderer.setImageTestStatus('请拖入 .safetensors 权重文件。', 'error');
            return;
        }
        const resolvedPath = await resolveDroppedWeightPath(normalizedPath);
        if (!resolvedPath) {
            return;
        }
        const pathInput = document.getElementById('image-test-weight-path');
        if (pathInput) {
            pathInput.value = resolvedPath;
        }
        syncWeightSelectWithPath(resolvedPath);
        state.restoredDraftFieldIds.add('image-test-weight-select');
        state.restoredDraftFieldIds.add('image-test-weight-path');
        draftStore.persistFromDom({ history_range: renderer.currentHistoryFilter() });
        renderer.setImageTestStatus(`已读取拖入权重：${fileNameFromPath(resolvedPath)}`, 'ok');
    }

    async function resolveDroppedWeightPath(path) {
        if (location.protocol === 'file:') {
            return path;
        }
        try {
            const payload = await resolveImageTestWeightPathRequest(ctx, { path });
            if (payload?.ok === false) {
                throw new Error(payload.error || '拖入权重解析失败');
            }
            const resolvedPath = String(payload?.weight_path || '').trim();
            if (!resolvedPath) {
                throw new Error(payload?.error || '后端没有返回可用的权重路径');
            }
            return resolvedPath;
        } catch (e) {
            renderer.setImageTestStatus(`拖入权重解析失败：${e.message}`, 'error');
            return '';
        }
    }

    function syncWeightSelectWithPath(path) {
        const select = document.getElementById('image-test-weight-select');
        if (!(select instanceof HTMLSelectElement)) return;
        const normalizedPath = String(path || '').trim();
        const candidates = Array.from(select.options).filter((option) => String(option.value || '').trim());
        const direct = candidates.find((option) => String(option.value || '').trim() === normalizedPath);
        if (direct) {
            select.value = direct.value;
            return;
        }
        if (!isBareSafetensorsFileName(normalizedPath) && isSafetensorsPath(normalizedPath)) {
            ensureWeightOption(select, normalizedPath);
            select.value = normalizedPath;
            return;
        }
        const nameMatched = candidates.filter((option) => fileNameFromPath(option.value) === fileNameFromPath(normalizedPath));
        select.value = nameMatched.length === 1 ? nameMatched[0].value : '';
    }

    function ensureWeightOption(select, path) {
        const normalizedPath = String(path || '').trim();
        if (!normalizedPath) return;
        const existing = Array.from(select.options).find((option) => String(option.value || '').trim() === normalizedPath);
        if (existing) return;
        const option = document.createElement('option');
        option.value = normalizedPath;
        option.textContent = `${fileNameFromPath(normalizedPath)} · 当前已选`;
        option.title = normalizedPath;
        select.appendChild(option);
    }

    function resolveWeightPathFromCandidates() {
        const pathInput = document.getElementById('image-test-weight-path');
        const select = document.getElementById('image-test-weight-select');
        if (!(pathInput instanceof HTMLInputElement) || !(select instanceof HTMLSelectElement)) return;
        const currentPath = String(pathInput.value || '').trim();
        const selectedPath = String(select.value || '').trim();
        if (selectedPath) {
            const selectedName = fileNameFromPath(selectedPath);
            if (!currentPath || isBareSafetensorsFileName(currentPath) || fileNameFromPath(currentPath) === selectedName) {
                if (currentPath !== selectedPath) {
                    pathInput.value = selectedPath;
                    state.restoredDraftFieldIds.add('image-test-weight-path');
                    draftStore.persistFromDom({ history_range: renderer.currentHistoryFilter() });
                }
                return;
            }
        }
        if (!isBareSafetensorsFileName(currentPath)) {
            syncWeightSelectWithPath(currentPath);
            return;
        }
        const resolved = resolvePreferredWeightOptionByName(currentPath);
        if (!resolved || resolved === currentPath) return;
        pathInput.value = resolved;
        syncWeightSelectWithPath(resolved);
        state.restoredDraftFieldIds.add('image-test-weight-select');
        state.restoredDraftFieldIds.add('image-test-weight-path');
        draftStore.persistFromDom({ history_range: renderer.currentHistoryFilter() });
    }

    function resolvePreferredWeightOptionByName(path) {
        const targetName = fileNameFromPath(path);
        if (!targetName) return '';
        const weights = Array.isArray(state.lastWeightsPayload?.weights) ? state.lastWeightsPayload.weights : [];
        const matches = weights.filter((item) => {
            const candidatePath = String(item?.abs_path || item?.file || '').trim();
            return candidatePath && fileNameFromPath(candidatePath) === targetName;
        });
        if (!matches.length) return '';
        matches.sort(comparePreferredWeightCandidate);
        return String(matches[0]?.abs_path || matches[0]?.file || '').trim();
    }

    function comparePreferredWeightCandidate(a, b) {
        const aCurrent = weightCandidatePriority(a);
        const bCurrent = weightCandidatePriority(b);
        if (aCurrent !== bCurrent) return bCurrent - aCurrent;
        const aMtime = Number(a?.mtime || 0);
        const bMtime = Number(b?.mtime || 0);
        if (aMtime !== bMtime) return bMtime - aMtime;
        const aPath = String(a?.abs_path || a?.file || '');
        const bPath = String(b?.abs_path || b?.file || '');
        return aPath.localeCompare(bPath);
    }

    function weightCandidatePriority(item) {
        const sourceTaskId = String(item?.source_task?.id || '').trim().toLowerCase();
        const scope = String(item?.scope || '').trim().toLowerCase();
        if (sourceTaskId === 'current') return 2;
        if (scope === 'task') return 1;
        return 0;
    }

    function droppedSafetensorsPath(dataTransfer) {
        const fileInfo = firstDroppedSafetensorsFileInfo(dataTransfer?.files);
        if (fileInfo.path) return fileInfo.path;
        const raw = dataTransfer?.getData('text/uri-list') || dataTransfer?.getData('text/plain') || '';
        const candidate = String(raw || '')
            .split(/\r?\n/)
            .map((line) => line.trim())
            .find((line) => line && !line.startsWith('#')) || '';
        return resolveDroppedPathCandidate(candidate, fileInfo.name);
    }

    function firstDroppedSafetensorsFileInfo(files) {
        const file = Array.from(files || []).find((item) => isSafetensorsPath(item?.name || item?.path || ''));
        if (!file) {
            return { path: '', name: '' };
        }
        return {
            path: String(file.path || '').trim(),
            name: String(file.name || '').trim(),
        };
    }

    function resolveDroppedPathCandidate(path, fileName = '') {
        const normalizedPath = normalizeDroppedWeightPath(path);
        const normalizedFileName = String(fileName || '').trim();
        if (!normalizedPath) {
            return normalizedFileName;
        }
        if (isSafetensorsPath(normalizedPath)) {
            return normalizedPath;
        }
        if (!normalizedFileName) {
            return normalizedPath;
        }
        const normalizedFileStem = stripSafetensorsExt(normalizedFileName);
        const normalizedPathName = fileNameFromPath(normalizedPath);
        if (normalizedPathName === normalizedFileStem) {
            return `${normalizedPath}.safetensors`;
        }
        return joinDroppedPath(normalizedPath, normalizedFileName);
    }

    function joinDroppedPath(basePath, fileName) {
        const normalizedBase = String(basePath || '').trim().replace(/[\\/]+$/, '');
        if (!normalizedBase) return fileName;
        const separator = normalizedBase.includes('\\') && !normalizedBase.includes('/') ? '\\' : '/';
        return `${normalizedBase}${separator}${fileName}`;
    }

    function normalizeDroppedWeightPath(path) {
        const raw = String(path || '').trim();
        if (!raw) return '';
        if (!raw.toLowerCase().startsWith('file://')) {
            return raw;
        }
        try {
            const url = new URL(raw);
            let normalized = decodeURIComponent(url.pathname || '');
            if (/^\/[A-Za-z]:\//.test(normalized)) {
                normalized = normalized.slice(1);
            }
            return normalized || raw;
        } catch (_) {
            return raw;
        }
    }

    function isSafetensorsPath(path) {
        return String(path || '').trim().toLowerCase().endsWith('.safetensors');
    }

    function isBareSafetensorsFileName(path) {
        const normalized = String(path || '').trim();
        return isSafetensorsPath(normalized) && !normalized.includes('/') && !normalized.includes('\\');
    }

    function stripSafetensorsExt(value) {
        return String(value || '').trim().replace(/\.safetensors$/i, '');
    }

    function fileNameFromPath(path) {
        const normalized = String(path || '').trim().replace(/\\/g, '/');
        return normalized.split('/').filter(Boolean).pop() || normalized || '未命名权重';
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
