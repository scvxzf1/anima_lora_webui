/* Safe, dependency-free syntax accents for local training logs. */

const TOKEN_PATTERN = /(\b(?:ok|success|completed|done)\b|\b100(?:\.0+)?%|\[\s*\d+\s*\/\s*\d+\s*\]|\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b|\b(?:INFO|DEBUG|WARNING|WARN|ERROR|CRITICAL)\b)/gi;

export function highlightedLogHtml(value) {
    const text = String(value ?? '');
    let cursor = 0;
    let html = '';
    for (const match of text.matchAll(TOKEN_PATTERN)) {
        html += escapeHtml(text.slice(cursor, match.index));
        html += `<span class="dragon-log-token" data-token="${tokenKind(match[0])}">${escapeHtml(match[0])}</span>`;
        cursor = match.index + match[0].length;
    }
    return html + escapeHtml(text.slice(cursor));
}

export function appendHighlightedLog(target, value) {
    const template = document.createElement('template');
    template.innerHTML = highlightedLogHtml(value);
    target.appendChild(template.content);
}

function tokenKind(token) {
    if (/^\[/.test(token)) return 'step';
    if (/^\d{4}-/.test(token)) return 'time';
    if (/^(warning|warn)$/i.test(token)) return 'warning';
    if (/^(error|critical)$/i.test(token)) return 'error';
    if (/^(info|debug)$/i.test(token)) return 'meta';
    return 'success';
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[char]));
}
