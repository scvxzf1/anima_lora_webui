import {
    IMAGE_TEST_ATTN_MODE_OPTIONS,
    IMAGE_TEST_DEFAULTS,
    IMAGE_TEST_RUNTIME_DTYPE_OPTIONS,
    IMAGE_TEST_SAMPLER_OPTIONS,
    IMAGE_TEST_TEXT_ENCODER_DTYPE_OPTIONS,
} from './state.js?v=module-bootstrap-20260702-1';

export function createImageTestRenderer({ ctx, state, openPreviewDialog }) {
    const { formatBytes } = ctx.format;

    function initStaticUI() {
        populateSelect('image-test-sampler', IMAGE_TEST_SAMPLER_OPTIONS, IMAGE_TEST_DEFAULTS.sampler);
        populateSelect('image-test-attn-mode', IMAGE_TEST_ATTN_MODE_OPTIONS, IMAGE_TEST_DEFAULTS.attnMode);
        populateSelect('image-test-runtime-dtype', IMAGE_TEST_RUNTIME_DTYPE_OPTIONS, IMAGE_TEST_DEFAULTS.runtimeDtype);
        populateSelect('image-test-text-encoder-dtype', IMAGE_TEST_TEXT_ENCODER_DTYPE_OPTIONS, IMAGE_TEST_DEFAULTS.textEncoderDtype);
        setReadonlyText('image-test-output-dir', 'output/tests');
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
            ['尺寸', request.width && request.height ? `${request.width}x${request.height}` : '-'],
            ['步数', request.infer_steps ?? '-'],
            ['CFG', request.guidance_scale ?? '-'],
            ['Flow Shift', request.flow_shift ?? '-'],
            ['种子', request.seed ?? '随机'],
            ['LoRA 强度', request.lora_multiplier ?? '-'],
        ];
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
        if (previous && Array.from(select.options).some((option) => option.value === previous)) {
            select.value = previous;
        }
    }

    function renderImages(payload = {}) {
        const title = document.getElementById('image-test-title');
        const subtitle = document.getElementById('image-test-subtitle');
        const count = document.getElementById('image-test-count');
        const grid = document.getElementById('image-test-grid');
        const empty = document.getElementById('image-test-empty');
        if (!title || !subtitle || !count || !grid || !empty) return;

        state.lastImagesPayload = payload;
        title.textContent = payload.label || '推理预览';
        subtitle.textContent = payload.directory
            ? `目录: ${payload.directory}`
            : '尚未找到 output/tests 结果目录。';
        count.textContent = `${payload.count || 0} 张`;
        grid.innerHTML = '';

        if (!payload.images?.length) {
            setImageEmpty(payload.message || '还没有生图结果。');
            return;
        }

        empty.hidden = true;
        payload.images.forEach((image, index) => {
            grid.appendChild(createImageCard(image, index));
        });
    }

    function createImageCard(image, index = 0) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'preview-card image-test-card';
        button.title = `原图预览: ${image.name || image.file || '预览图'}`;
        button.addEventListener('click', () => openPreviewDialog(image));

        const imageWrap = document.createElement('div');
        imageWrap.className = 'preview-card-image';
        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name || '生图结果';
        img.loading = index < 8 ? 'eager' : 'lazy';
        const errorMessage = document.createElement('span');
        errorMessage.className = 'preview-card-error-message';
        errorMessage.textContent = '图片加载失败';
        errorMessage.hidden = true;
        img.addEventListener('load', () => {
            button.classList.remove('preview-card-error');
            errorMessage.hidden = true;
        });
        img.addEventListener('error', () => {
            button.classList.add('preview-card-error');
            errorMessage.hidden = false;
        });
        imageWrap.append(img, errorMessage);

        const meta = document.createElement('div');
        meta.className = 'preview-card-meta';
        const head = document.createElement('strong');
        head.textContent = image.name || '未命名结果';
        head.title = image.file || image.name || '';
        const file = document.createElement('span');
        file.className = 'preview-card-filename';
        file.textContent = image.file || image.name || '';
        file.title = image.file || image.name || '';
        const dims = image.width && image.height ? `${image.width}x${image.height}` : '尺寸未知';
        const sub = document.createElement('span');
        sub.textContent = [
            dims,
            image.sample?.parameters?.sample_steps ? `${image.sample.parameters.sample_steps} steps` : '',
            image.sample?.sampler || image.sample?.parameters?.sample_sampler || '',
            formatBytes(image.size_bytes || 0),
        ].filter(Boolean).join(' · ');
        meta.append(head, file, sub);

        button.append(imageWrap, meta);
        return button;
    }

    function setImageLoading() {
        const count = document.getElementById('image-test-count');
        const grid = document.getElementById('image-test-grid');
        if (count) count.textContent = '读取中';
        if (grid) grid.innerHTML = '';
        setImageEmpty('正在读取 output/tests 中的结果图...');
    }

    function setImageEmpty(message) {
        const empty = document.getElementById('image-test-empty');
        const grid = document.getElementById('image-test-grid');
        if (!empty || !grid) return;
        empty.textContent = message;
        empty.hidden = false;
        grid.innerHTML = '';
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

    function escapeText(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    return {
        initStaticUI,
        renderRuntime,
        renderWeightOptions,
        renderImages,
        setImageLoading,
        setImageEmpty,
        setImageTestStatus,
        syncButtons,
    };
}
