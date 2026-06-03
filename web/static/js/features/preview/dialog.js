export function createPreviewDialog({ ctx, deps }) {
    const { formatBytes } = ctx.format;

    function closePreviewImageDialog() {
        const dialog = document.getElementById('preview-dialog');
        if (dialog?.open) {
            dialog.close();
        }
    }

    function openPreviewDialog(image) {
        const dialog = document.getElementById('preview-dialog');
        const img = document.getElementById('preview-dialog-image');
        document.getElementById('preview-dialog-title').textContent = image.name;
        const dims = image.width && image.height ? `${image.width}x${image.height}` : '尺寸未知';
        document.getElementById('preview-dialog-meta').textContent =
            `${image.file} · ${dims} · ${formatBytes(image.size_bytes)} · ${image.mtime_text || ''}`;
        renderPreviewDialogDetails(image, dims);
        img.src = image.url;
        img.alt = image.name;
        if (dialog?.showModal) {
            dialog.showModal();
        }
    }

    function renderPreviewDialogDetails(image, dims) {
        const box = document.getElementById('preview-dialog-details');
        if (!box) return;
        box.innerHTML = '';
        if (image.detailContext === 'dataset') {
            deps.renderDatasetImageDialogDetails(box, image, dims);
            return;
        }
        const sample = image.sample || {};
        const params = sample.parameters || {};
        const promptNo = sample.prompt_index != null ? Number(sample.prompt_index) + 1 : null;

        const rows = [
            ['轮次', sample.epoch != null ? `Epoch ${sample.epoch}` : '-'],
            ['步数', sample.step != null ? `Step ${sample.step}` : '-'],
            ['来源任务', image.source_task?.label || '-'],
            ['任务时间', image.source_task?.started_at_text || '-'],
            ['提示词序号', promptNo ? `第 ${promptNo} 条` : '-'],
            ['生成时间', sample.generated_at_text || image.mtime_text || '-'],
            ['种子', sample.seed ?? params.seed ?? '-'],
            ['采样器', sample.sampler || params.sample_sampler || '-'],
            ['生成步数', params.sample_steps ?? '-'],
            ['CFG', params.guidance_scale ?? params.scale ?? '-'],
            ['Flow Shift', params.flow_shift ?? '-'],
            ['尺寸', params.width && params.height ? `${params.width}x${params.height}` : dims],
            ['文件大小', formatBytes(image.size_bytes)],
            ['提示词文件', sample.source?.prompt_file || '-'],
        ];
        for (const [label, value] of rows) {
            box.appendChild(createPreviewDetailRow(label, value));
        }
        if (sample.prompt) {
            box.appendChild(createPreviewDetailBlock('提示词', sample.prompt));
        }
        if (sample.negative_prompt) {
            box.appendChild(createPreviewDetailBlock('负面提示词', sample.negative_prompt));
        }
        if (sample.raw_prompt) {
            box.appendChild(createPreviewDetailBlock('原始参数行', sample.raw_prompt));
        }
        box.appendChild(createPreviewDetailBlock('文件路径', image.file || '-'));
    }

    return {
        closePreviewImageDialog,
        openPreviewDialog,
    };
}

export function createPreviewDetailRow(label, value) {
    const row = document.createElement('div');
    row.className = 'preview-detail-row';
    const key = document.createElement('span');
    key.textContent = label;
    const valEl = document.createElement('strong');
    valEl.textContent = value;
    row.append(key, valEl);
    return row;
}

export function createPreviewDetailBlock(label, value, preformatted = false) {
    const block = document.createElement('div');
    block.className = 'preview-detail-block';
    const key = document.createElement('span');
    key.textContent = label;
    const valEl = document.createElement('p');
    if (preformatted) valEl.className = 'preview-detail-preformatted';
    valEl.textContent = value;
    block.append(key, valEl);
    return block;
}
