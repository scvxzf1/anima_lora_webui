import { previewSourceLabel } from './state.js?v=module-bootstrap-20260603-6';

export function createPreviewImages({ ctx, state, deps, openPreviewDialog, syncPreviewPanelSubtitle }) {
    const { formatBytes } = ctx.format;

    function renderPreviewImages(payload) {
        const grid = document.getElementById('preview-grid');
        const empty = document.getElementById('preview-empty');
        const title = document.getElementById('preview-title');
        const subtitle = document.getElementById('preview-subtitle');
        const count = document.getElementById('preview-count');

        title.textContent = payload.label || previewSourceLabel(state.source);
        subtitle.textContent = payload.directory
            ? `目录: ${payload.directory}${previewDirectoryHint(payload)}`
            : '尚未设置目录。';
        count.textContent = `${payload.count || 0} 张`;
        document.getElementById('preview-current-dir').textContent = payload.directory || '-';
        syncPreviewPanelSubtitle();

        grid.innerHTML = '';
        if (!payload.images?.length) {
            setPreviewEmpty(previewEmptyMessage(payload));
            return;
        }
        empty.hidden = true;
        for (const image of payload.images) {
            grid.appendChild(createPreviewCard(image));
        }
    }

    function createPreviewCard(image) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'preview-card';
        button.title = `原图预览: ${image.name || image.file || '预览图'}`;
        button.addEventListener('click', () => openPreviewDialog(image));

        const imageWrap = document.createElement('div');
        imageWrap.className = 'preview-card-image';
        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name;
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            button.classList.add('preview-card-error');
            img.alt = '图片加载失败';
        });
        const previewHint = document.createElement('span');
        previewHint.className = 'preview-card-preview-indicator';
        previewHint.setAttribute('aria-hidden', 'true');
        imageWrap.append(img, previewHint);

        const meta = document.createElement('div');
        meta.className = 'preview-card-meta';
        const title = document.createElement('strong');
        title.textContent = previewCardTitle(image);
        title.title = image.file || image.name || '';
        const dims = image.width && image.height ? `${image.width}x${image.height}` : '尺寸未知';
        const file = document.createElement('span');
        file.className = 'preview-card-filename';
        file.textContent = image.name || '';
        file.title = image.file || image.name || '';
        const sub = document.createElement('span');
        sub.textContent = previewCardSecondaryMeta(image, dims);
        meta.append(title, file, sub);
        if (image.source_task?.label) {
            const source = document.createElement('span');
            source.textContent = `来源: ${image.source_task.label}`;
            meta.appendChild(source);
        }

        button.append(imageWrap, meta);
        return button;
    }

    function setPreviewLoading() {
        document.getElementById('preview-count').textContent = '读取中';
        document.getElementById('preview-grid').innerHTML = '';
        setPreviewEmpty('正在读取预览图...');
        syncPreviewPanelSubtitle();
    }

    function setPreviewEmpty(message) {
        const empty = document.getElementById('preview-empty');
        if (!empty) return;
        empty.textContent = message;
        empty.hidden = false;
        document.getElementById('preview-grid').innerHTML = '';
        syncPreviewPanelSubtitle();
    }

    function previewEmptyMessage(payload) {
        const base = payload.message || '暂无预览图。';
        if (state.source !== 'training') return base;
        const runtime = deps.getTrainingRuntime();
        const cfg = payload.sample_config || state.trainingSampleState || runtime.sampleConfig || {};
        const msg = cfg.message || '';
        if (!msg || base.includes(msg)) return base;
        if (cfg.enabled) {
            const samplingDelayHint = '如果训练刚开始，可能还没到达采样频率。';
            return base.includes(samplingDelayHint) ? base : `${base} ${samplingDelayHint}`;
        }
        if (payload.preview_settings?.effective_training_source === 'latest_run') {
            const latestRunHint = '最新运行目录里还没有可显示的样张。';
            return base.includes(latestRunHint) ? base : `${base} ${latestRunHint}`;
        }
        return `${base} ${msg}。`;
    }

    function previewDirectoryHint(payload) {
        const source = payload.preview_settings?.effective_training_source || '';
        const latestRun = payload.preview_settings?.latest_run_dir || '';
        if (source === 'current_task') return ' · 当前任务';
        if (source === 'latest_run') {
            return latestRun ? ` · 最新运行 ${latestRun}` : ' · 最新运行';
        }
        return '';
    }

    function previewCardTitle(image) {
        const sample = image.sample || {};
        const parts = [];
        if (sample.epoch != null) parts.push(`Epoch ${sample.epoch}`);
        if (sample.step != null) parts.push(`Step ${sample.step}`);
        return parts.length ? parts.join(' · ') : (image.mtime_text || '无采样元信息');
    }

    function previewCardSecondaryMeta(image, dims) {
        const sample = image.sample || {};
        const params = sample.parameters || {};
        const renderSize = params.width && params.height ? `${params.width}x${params.height}` : dims;
        const steps = params.sample_steps ? `${params.sample_steps} steps` : '';
        const sampler = sample.sampler || params.sample_sampler || '';
        const seed = sample.seed != null ? `seed ${sample.seed}` : '';
        return [seed, renderSize, steps, sampler, formatBytes(image.size_bytes)].filter(Boolean).join(' · ');
    }

    return {
        renderPreviewImages,
        setPreviewLoading,
        setPreviewEmpty,
    };
}
