import {
    IMAGE_TEST_ATTN_MODE_OPTIONS,
    IMAGE_TEST_DEFAULTS,
    IMAGE_TEST_RUNTIME_DTYPE_OPTIONS,
    IMAGE_TEST_SAMPLER_OPTIONS,
    IMAGE_TEST_TEXT_ENCODER_DTYPE_OPTIONS,
} from './state.js?v=module-bootstrap-20260705-3';
import { createImageTestGallery } from './gallery.js?v=module-bootstrap-20260705-3';

export function createImageTestRenderer({
    ctx,
    state,
    openPreviewDialog,
    requestHistoryReload,
    initialHistoryFilter,
}) {
    const { formatBytes } = ctx.format;
    const gallery = createImageTestGallery({
        formatBytes,
        openPreviewDialog,
        requestHistoryReload,
        initialFilterValue: initialHistoryFilter,
    });

    function initStaticUI() {
        populateSelect('image-test-sampler', IMAGE_TEST_SAMPLER_OPTIONS, IMAGE_TEST_DEFAULTS.sampler);
        populateSelect('image-test-attn-mode', IMAGE_TEST_ATTN_MODE_OPTIONS, IMAGE_TEST_DEFAULTS.attnMode);
        populateSelect('image-test-runtime-dtype', IMAGE_TEST_RUNTIME_DTYPE_OPTIONS, IMAGE_TEST_DEFAULTS.runtimeDtype);
        populateSelect('image-test-text-encoder-dtype', IMAGE_TEST_TEXT_ENCODER_DTYPE_OPTIONS, IMAGE_TEST_DEFAULTS.textEncoderDtype);
        renderGpuOptions({ ok: true, gpus: [] });
        setReadonlyText('image-test-output-dir', 'output/tests');
        gallery.init();
        renderRuntime(null);
        setImageEmpty('打开本页后会读取 output/tests 中的结果图。');
        renderWeightOptions({ ok: true, weights: [], message: '点击刷新后读取可选权重。' });
    }

    function renderRuntime(payload) {
        const status = payload || {};
        state.lastStatus = payload || null;
        const running = Boolean(status.running);
        const tone = running
            ? 'running'
            : status.status === 'done'
                ? 'ok'
                : status.status === 'error'
                    ? 'error'
                    : status.status === 'canceled'
                        ? 'warning'
                        : '';
        const label = running
            ? '运行中'
            : status.status === 'done'
                ? '已完成'
                : status.status === 'error'
                    ? '失败'
                    : status.status === 'canceled'
                        ? '已取消'
                        : '空闲';

        const badge = document.getElementById('image-test-run-badge');
        if (badge) {
            badge.className = `image-test-run-badge ${tone}`.trim();
            badge.textContent = label;
        }

        const statusBox = document.getElementById('image-test-run-summary');
        if (statusBox) {
            statusBox.innerHTML = '';
            statusBox.append(
                createMetric('状态', label),
                createMetric('输出张数', String(status.output_count ?? 0)),
                createMetric('开始时间', status.started_at_text || '-'),
                createMetric('结束时间', status.finished_at_text || '-'),
            );
        }

        setReadonlyText('image-test-output-dir', status.output_dir || 'output/tests');
        setImageTestStatus(runtimeStatusMessage(status), tone === 'running' ? '' : tone);
        renderLogs(status.logs || []);
        renderLastRequest(status.last_request || {});
        renderCommand(status.command || []);
        syncButtons(status);
    }

    function runtimeStatusMessage(status = {}) {
        if (status.error) return status.error;
        if (status.running) return '生图测试正在运行，会持续刷新右侧 output/tests。';
        if (status.status === 'done') return '生图测试已完成，右侧已切到最新推理结果。';
        if (status.status === 'canceled') return '生图测试已停止。';
        return '当前没有运行中的生图测试任务。';
    }

    function renderLogs(lines = []) {
        const box = document.getElementById('image-test-log');
        if (!box) return;
        box.textContent = Array.isArray(lines) && lines.length
            ? lines.join('\n')
            : '暂无运行日志。';
    }

    function renderLastRequest(request = {}) {
        const box = document.getElementById('image-test-last-request');
        if (!box) return;
        if (!request || !Object.keys(request).length) {
            box.innerHTML = '<div class="image-test-inline-empty">还没有提交过生图请求。</div>';
            return;
        }
        const rows = [
            ['采样器', request.sampler || '-'],
            ['注意力后端', request.attn_mode || '-'],
            ['推理精度', request.runtime_dtype || '-'],
            ['文本编码器精度', request.text_encoder_dtype || '-'],
            ['GPU', request.gpu_label || request.device || '自动'],
            ['尺寸', request.width && request.height ? `${request.width}x${request.height}` : '-'],
            ['步数', request.infer_steps ?? '-'],
            ['CFG', request.guidance_scale ?? '-'],
            ['Flow Shift', request.flow_shift ?? '-'],
            ['种子', request.seed ?? '随机'],
            ['LoRA 强度', request.lora_multiplier ?? '-'],
            ['分层加载', request.anima_selective_lora ? '开' : '关'],
        ];
        if (request.anima_selective_lora) {
            rows.push(
                ['分层预设', request.anima_selective_preset || '-'],
                ['启用层数', request.anima_selective_block_count ?? '-'],
            );
        }
        box.innerHTML = '';
        rows.forEach(([label, value]) => {
            const row = document.createElement('div');
            row.className = 'image-test-request-row';
            row.innerHTML = `<span>${label}</span><strong>${escapeText(String(value))}</strong>`;
            box.appendChild(row);
        });
        if (request.prompt) {
            box.appendChild(createBlock('正向提示词', request.prompt));
        }
        if (request.negative_prompt) {
            box.appendChild(createBlock('负面提示词', request.negative_prompt));
        }
        if (request.weight_path) {
            box.appendChild(createBlock('LoRA 权重', request.weight_path));
        }
        if (request.anima_selective_lora && Array.isArray(request.anima_selective_blocks) && request.anima_selective_blocks.length) {
            box.appendChild(createBlock(
                '分层层位',
                summarizeSelectiveBlocks(request.anima_selective_blocks, request.anima_selective_block_strengths || {}),
            ));
        }
    }

    function renderCommand(command = []) {
        const box = document.getElementById('image-test-command');
        if (!box) return;
        box.textContent = Array.isArray(command) && command.length
            ? command.join(' ')
            : '当前没有可展示的推理命令。';
    }

    function renderWeightOptions(payload = {}) {
        const select = document.getElementById('image-test-weight-select');
        if (!select) return;
        const previous = select.value;
        const preferredWeightPath = String(document.getElementById('image-test-weight-path')?.value || '').trim();
        const weights = Array.isArray(payload.weights) ? payload.weights : [];
        state.lastWeightsPayload = payload;
        select.innerHTML = '';
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = weights.length ? '从训练输出中选择权重…' : (payload.message || '没有扫描到可用权重');
        select.appendChild(placeholder);
        for (const item of weights) {
            const option = document.createElement('option');
            option.value = item.abs_path || item.file || '';
            option.textContent = [
                item.name || fileNameFromPath(option.value),
                item.epoch != null ? `ep ${item.epoch}` : '',
                item.steps != null ? `${item.steps} step` : '',
                item.scope_label || '',
            ].filter(Boolean).join(' · ');
            option.title = option.value;
            select.appendChild(option);
        }
        select.disabled = !weights.length;
        const preferred = previous || preferredWeightPath;
        if (preferred && Array.from(select.options).some((option) => option.value === preferred)) {
            select.value = preferred;
        }
    }

    function renderGpuOptions(payload = {}) {
        const select = document.getElementById('image-test-gpu-index');
        if (!select) return;
        const previous = String(select.value || '').trim();
        const preferredGpuIndex = String(document.getElementById('image-test-gpu-index')?.value || '').trim();
        const gpus = Array.isArray(payload.gpus) ? payload.gpus : [];
        state.lastGpusPayload = payload;
        select.innerHTML = '';

        const auto = document.createElement('option');
        auto.value = '';
        auto.textContent = gpus.length
            ? '自动（默认可见 GPU）'
            : (location.protocol === 'file:' ? '自动（静态模式不读取 GPU）' : '自动（未读取到 GPU 列表）');
        select.appendChild(auto);

        for (const gpu of gpus) {
            const option = document.createElement('option');
            const index = Number(gpu?.index);
            option.value = Number.isInteger(index) && index >= 0 ? String(index) : '';
            option.textContent = gpu?.label || (Number.isInteger(index) ? `GPU ${index}` : '未命名 GPU');
            option.title = gpu?.memory_total_gb
                ? `${option.textContent} · 显存 ${gpu.memory_total_gb} GB`
                : option.textContent;
            if (option.value) {
                select.appendChild(option);
            }
        }

        const preferred = previous || preferredGpuIndex || IMAGE_TEST_DEFAULTS.gpuIndex;
        select.value = Array.from(select.options).some((option) => option.value === preferred)
            ? preferred
            : '';
    }

    function renderImages(payload = {}) {
        state.lastImagesPayload = payload;
        gallery.render(payload);
    }

    function setImageLoading() {
        gallery.setLoading();
    }

    function setImageEmpty(message) {
        gallery.setEmpty(message);
    }

    function setImageTestStatus(text, tone = '') {
        const el = document.getElementById('image-test-status');
        if (!el) return;
        el.textContent = text;
        el.className = `preview-status ${tone}`.trim();
    }

    function syncButtons(status = {}) {
        const startBtn = document.getElementById('btn-start-image-test');
        const stopBtn = document.getElementById('btn-stop-image-test');
        const running = Boolean(status.running);
        if (startBtn) {
            startBtn.disabled = running || state.starting || !state.syncReady;
        }
        if (stopBtn) {
            stopBtn.disabled = !running || state.stopping;
        }
    }

    function createMetric(label, value) {
        const item = document.createElement('div');
        item.className = 'image-test-run-metric';
        item.innerHTML = `<span>${escapeText(label)}</span><strong>${escapeText(String(value || '-'))}</strong>`;
        return item;
    }

    function createBlock(label, value) {
        const block = document.createElement('div');
        block.className = 'image-test-request-block';
        block.innerHTML = `<span>${escapeText(label)}</span><p>${escapeText(String(value || '-'))}</p>`;
        return block;
    }

    function setReadonlyText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function populateSelect(id, options, preferred = '') {
        const select = document.getElementById(id);
        if (!select) return;
        const previous = select.value;
        select.innerHTML = '';
        options.forEach((item) => {
            const option = document.createElement('option');
            option.value = item.value;
            option.textContent = item.label;
            select.appendChild(option);
        });
        select.value = previous || preferred || options[0]?.value || '';
    }

    function fileNameFromPath(path) {
        const clean = String(path || '').trim().replace(/\\/g, '/');
        return clean.split('/').filter(Boolean).pop() || clean || '-';
    }

    function summarizeSelectiveBlocks(blocks = [], blockStrengths = {}) {
        if (!Array.isArray(blocks) || !blocks.length) {
            return '未选择层位';
        }
        const lines = blocks.map((blockId) => {
            const strength = Number(blockStrengths?.[blockId]);
            return Number.isFinite(strength)
                ? `${blockId} ${strength.toFixed(2).replace(/\.00$/, '')}x`
                : blockId;
        });
        if (lines.length <= 12) {
            return lines.join(', ');
        }
        return `${lines.slice(0, 12).join(', ')} ... 共 ${lines.length} 项`;
    }

    function escapeText(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    return {
        currentHistoryFilter: () => gallery.currentFilter(),
        initStaticUI,
        renderRuntime,
        renderGpuOptions,
        renderWeightOptions,
        renderImages,
        setImageLoading,
        setImageEmpty,
        setImageTestStatus,
        syncButtons,
    };
}
