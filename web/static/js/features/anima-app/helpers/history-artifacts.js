export function makeHistoryArtifactUrl(task, artifactKey, options = {}) {
    const taskId = String(task?.id || '').trim();
    const key = String(artifactKey || '').trim();
    if (!taskId || !key) return '#';
    const params = new URLSearchParams();
    if (options.download) params.set('download', '1');
    const suffix = params.toString() ? `?${params.toString()}` : '';
    return `/api/training/history/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(key)}${suffix}`;
}
