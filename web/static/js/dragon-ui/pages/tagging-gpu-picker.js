/* Single-GPU selection helpers for local tagging profiles. */

export function normalizeGpuIndex(value) {
    if (value == null || value === '') return '';
    const index = Number(value);
    return Number.isInteger(index) && index >= 0 ? String(index) : '';
}

export function gpuIndexPayload(device, value) {
    if (device !== 'cuda') return null;
    const normalized = normalizeGpuIndex(value);
    return normalized === '' ? null : Number(normalized);
}

export function renderTaggingGpuOptions(gpus, selectedValue) {
    const selected = normalizeGpuIndex(selectedValue);
    const values = Array.isArray(gpus)
        ? gpus.filter((gpu) => normalizeGpuIndex(gpu?.index) !== '')
        : [];
    const known = new Set(values.map((gpu) => normalizeGpuIndex(gpu.index)));
    const options = [
        `<option value="" ${selected === '' ? 'selected' : ''}>默认可见 GPU</option>`,
    ];
    if (selected && !known.has(selected)) {
        options.push(`<option value="${escapeAttribute(selected)}" selected>GPU ${escapeHtml(selected)}（当前不可用）</option>`);
    }
    for (const gpu of values) {
        const index = normalizeGpuIndex(gpu.index);
        const label = String(gpu.label || gpu.name || `GPU ${index}`);
        options.push(`<option value="${escapeAttribute(index)}" ${index === selected ? 'selected' : ''}>${escapeHtml(label)}</option>`);
    }
    return options.join('');
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
