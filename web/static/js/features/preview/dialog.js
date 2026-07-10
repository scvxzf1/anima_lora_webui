export function createPreviewDialog({ ctx, deps }) {
    const { formatBytes } = ctx.format;
    const state = {
        image: null,
        deleteHandler: null,
        deleteLabel: '从硬盘永久删除',
        deletePending: false,
    };
    const dialog = document.getElementById('preview-dialog');
    dialog?.addEventListener('close', () => {
        state.image = null;
        state.deleteHandler = null;
        state.deleteLabel = '从硬盘永久删除';
        state.deletePending = false;
        setPreviewDialogStatus('', '');
        syncPreviewDialogDeleteButton();
    });
    document.getElementById('btn-preview-dialog-delete')?.addEventListener('click', async (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (state.deletePending || typeof state.deleteHandler !== 'function' || !state.image) return;
        state.deletePending = true;
        setPreviewDialogStatus('正在从硬盘永久删除图片...', 'warning');
        syncPreviewDialogDeleteButton();
        try {
            const result = await state.deleteHandler(state.image);
            const deletedCount = Number(result?.deleted_count || 0);
            if (result?.ok === false && deletedCount <= 0) {
                setPreviewDialogStatus(result?.error || result?.message || '删除失败。', 'error');
                return;
            }
            closePreviewImageDialog();
        } catch (error) {
            setPreviewDialogStatus(`删除失败：${error?.message || '无法删除图片。'}`, 'error');
        } finally {
            state.deletePending = false;
            syncPreviewDialogDeleteButton();
        }
    });

    function closePreviewImageDialog() {
        if (dialog?.open) {
            dialog.close();
        }
    }

    function openPreviewDialog(image, options = {}) {
        const img = document.getElementById('preview-dialog-image');
        document.getElementById('preview-dialog-title').textContent = image.name;
        const dims = image.width && image.height ? `${image.width}x${image.height}` : '尺寸未知';
        document.getElementById('preview-dialog-meta').textContent =
            `${image.file} · ${dims} · ${formatBytes(image.size_bytes)} · ${image.mtime_text || ''}`;
        state.image = image;
        state.deleteHandler = typeof options.onDelete === 'function' ? options.onDelete : null;
        state.deleteLabel = String(options.deleteLabel || '从硬盘永久删除');
        state.deletePending = false;
        setPreviewDialogStatus('', '');
        syncPreviewDialogDeleteButton();
        renderPreviewDialogDetails(image, dims);
        img.src = image.url;
        img.alt = image.name;
        if (dialog?.showModal && !dialog.open) {
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

    function syncPreviewDialogDeleteButton() {
        const button = document.getElementById('btn-preview-dialog-delete');
        if (!(button instanceof HTMLButtonElement)) return;
        const enabled = typeof state.deleteHandler === 'function' && state.image?.detailContext !== 'dataset';
        button.hidden = !enabled;
        button.disabled = state.deletePending;
        button.textContent = state.deletePending ? '删除中...' : state.deleteLabel;
    }

    function setPreviewDialogStatus(text, tone = '') {
        const el = document.getElementById('preview-dialog-status');
        if (!el) return;
        if (!text) {
            el.hidden = true;
            el.textContent = '';
            el.className = 'preview-status';
            return;
        }
        el.hidden = false;
        el.textContent = text;
        el.className = `preview-status ${tone}`.trim();
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
