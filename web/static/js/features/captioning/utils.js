export function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

export const escapeAttribute = escapeHtml;

export function stateLabel(value) {
    return ({
        queued: '排队中', running: '推理中', paused: '已暂停', completed: '已完成',
        completed_with_errors: '部分失败', canceled: '已取消', interrupted: '已中断',
        waiting: '待处理', ready: '已打标', committed: '已写入', failed: '失败',
    })[value] || String(value || '未知');
}

export function statusTone(value) {
    if (['ready', 'committed', 'completed'].includes(value)) return 'success';
    if (['running', 'queued'].includes(value)) return 'running';
    if (['failed', 'completed_with_errors', 'canceled'].includes(value)) return 'error';
    return 'idle';
}

export function splitTags(value) {
    return String(value || '').split(/[\n,]/).map((tag) => tag.trim().replace(/^Tags:\s*/i, '')).filter(Boolean);
}

export function showFeedback(root, message, tone = 'info') {
    const node = root.querySelector('[data-caption-feedback]');
    if (!node) return;
    node.textContent = message;
    node.dataset.tone = tone;
}

export async function withBusy(control, operation) {
    if (control?.disabled) return undefined;
    if (control) control.disabled = true;
    try { return await operation(); }
    finally { if (control) control.disabled = false; }
}
