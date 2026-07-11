export function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

export function formatBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDuration(totalSeconds) {
    const seconds = Math.max(0, Number(totalSeconds) || 0);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const rest = seconds % 60;
    if (minutes < 60) return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const restMinutes = minutes % 60;
    return restMinutes ? `${hours}h ${restMinutes}m` : `${hours}h`;
}

function normalizePathSeparators(value) {
    return String(value ?? '').replace(/\\/g, '/');
}

function pathSegments(value) {
    return normalizePathSeparators(value).split('/').filter(Boolean);
}

export function formatPathLabel(path, options = {}) {
    const mode = options?.mode || 'length';
    const maxLength = Number.isFinite(Number(options?.maxLength))
        ? Math.max(1, Number(options.maxLength))
        : 64;
    const text = String(path ?? '');

    if (mode === 'basename') {
        const parts = pathSegments(text);
        return parts.length ? parts[parts.length - 1] : text;
    }

    if (mode === 'parent-basename') {
        const parts = pathSegments(text);
        if (parts.length >= 2) {
            return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
        }
        return parts.length ? parts[parts.length - 1] : text;
    }

    // mode: length (default) - middle-ellipsis for long paths
    if (text.length <= maxLength) return text;
    const head = Math.max(8, Math.floor((maxLength - 1) * 0.42));
    const tail = Math.max(8, maxLength - head - 1);
    return `${text.slice(0, head)}…${text.slice(-tail)}`;
}

export function compactPathLabel(value, maxLength = 64) {
    return formatPathLabel(value, { mode: 'length', maxLength });
}
