export function optionNode(value, text) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    return option;
}

export function historyDetailSection(title, body, className = 'history-detail-section') {
    const section = document.createElement('section');
    section.className = className;
    const head = document.createElement('div');
    head.className = 'history-detail-section-title';
    const h4 = document.createElement('h4');
    h4.textContent = title;
    head.appendChild(h4);
    section.append(head, body);
    return section;
}

export function historyDetailRow(label, value, options = {}, helpers = {}) {
    const row = document.createElement('div');
    row.className = options.className || '';
    row.classList.toggle('is-muted', Boolean(options.muted));
    const key = document.createElement('span');
    key.textContent = label;
    const val = document.createElement('code');
    const rawValue = String(value === undefined || value === null || value === '' ? '-' : value);
    val.textContent = String(options.displayValue === undefined ? rawValue : options.displayValue);
    if (options.copyValue) {
        row.classList.add('has-copy-action');
        val.title = rawValue;
    }
    row.append(key, val);
    if (options.copyValue && helpers.copyButton) {
        row.appendChild(helpers.copyButton(options.copyValue, `${label}完整路径`));
    }
    return row;
}

export function historyDetailEmptyText(text) {
    const p = document.createElement('p');
    p.className = 'history-detail-empty-text';
    p.textContent = text;
    return p;
}

const HISTORY_DETAIL_ICONS = {
    'chevron-down': [['path', { d: 'm6 9 6 6 6-6' }]],
    'chevron-up': [['path', { d: 'm18 15-6-6-6 6' }]],
    copy: [
        ['rect', { x: '8', y: '8', width: '12', height: '12', rx: '2', ry: '2' }],
        ['path', { d: 'M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2' }],
    ],
    download: [
        ['path', { d: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4' }],
        ['path', { d: 'M7 10l5 5 5-5' }],
        ['path', { d: 'M12 15V3' }],
    ],
    eye: [
        ['path', { d: 'M2.1 12.3a1 1 0 0 1 0-.6C3.4 7.5 7.4 5 12 5s8.6 2.5 9.9 6.7a1 1 0 0 1 0 .6C20.6 16.5 16.6 19 12 19s-8.6-2.5-9.9-6.7Z' }],
        ['circle', { cx: '12', cy: '12', r: '3' }],
    ],
    search: [
        ['circle', { cx: '11', cy: '11', r: '8' }],
        ['path', { d: 'm21 21-4.3-4.3' }],
    ],
};

export function historyDetailIcon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    for (const [tag, attrs] of HISTORY_DETAIL_ICONS[name] || []) {
        const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
        svg.appendChild(node);
    }
    return svg;
}

export function historyDetailIconButton(label, iconName, handler, className = '') {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = ['history-detail-icon-btn', className].filter(Boolean).join(' ');
    btn.title = label;
    btn.setAttribute('aria-label', label);
    btn.appendChild(historyDetailIcon(iconName));
    btn.addEventListener('click', handler);
    return btn;
}

export function historyDetailIconLink(label, iconName, href, options = {}) {
    const link = document.createElement('a');
    link.className = 'history-detail-icon-btn history-detail-icon-link';
    link.href = href;
    link.title = label;
    link.setAttribute('aria-label', label);
    link.target = '_blank';
    link.rel = 'noopener';
    if (options.download) {
        link.download = '';
    }
    link.appendChild(historyDetailIcon(iconName));
    return link;
}

export function historyDetailFlashToolButton(btn, doneLabel, defaultLabel) {
    btn.classList.add('is-copied');
    btn.title = doneLabel;
    btn.setAttribute('aria-label', doneLabel);
    setTimeout(() => {
        btn.classList.remove('is-copied');
        btn.title = defaultLabel;
        btn.setAttribute('aria-label', defaultLabel);
    }, 1200);
}

export function createHistoryDetailCopyButton(copyText, value, label = '路径') {
    const btn = historyDetailIconButton(`复制${label}`, 'copy', async () => {
        try {
            await copyText(String(value || ''));
            historyDetailFlashToolButton(btn, `已复制${label}`, `复制${label}`);
        } catch (e) {
            alert('复制路径失败: ' + e.message);
        }
    }, 'history-detail-copy-btn');
    return btn;
}

export function fileNameFromPath(value) {
    const text = String(value || '').trim();
    if (!text) return '';
    return text.split(/[\\/]/).filter(Boolean).pop() || text;
}

export function historyDetailRunRoot(task) {
    return task.run_dir || '';
}

export function normalizedHistoryDetailPath(value) {
    const path = String(value || '').trim().replace(/\\/g, '/');
    if (path === '/') return path;
    return path.replace(/\/+$/, '');
}

export function relativeHistoryDetailPath(value, rootPath) {
    const path = normalizedHistoryDetailPath(value);
    const root = normalizedHistoryDetailPath(rootPath);
    if (!path) return '-';
    if (root && path === root) return '.';
    if (root && path.startsWith(`${root}/`)) return path.slice(root.length + 1);
    if (!path.startsWith('/') && !/^[A-Za-z]:\//.test(path)) return path;
    const parts = path.split('/').filter(Boolean);
    return parts.length > 3 ? `.../${parts.slice(-3).join('/')}` : path;
}

export function numberOrNull(value) {
    if (value === undefined || value === null || value === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

export function maxValue(records, key) {
    const values = records.map((record) => numberOrNull(record[key])).filter((value) => value !== null);
    return values.length ? Math.max(...values) : null;
}

export function maxRecordBy(records, key) {
    let best = null;
    for (const record of records) {
        const value = numberOrNull(record[key]);
        if (value === null) continue;
        if (!best || value > numberOrNull(best[key])) best = record;
    }
    return best;
}

export function clampNumber(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

export function formatCompactNumber(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '-';
    return n.toFixed(digits).replace(/\.?0+$/, '');
}

export function svgPolyline(points, className) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    line.setAttribute('points', points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' '));
    line.setAttribute('class', className);
    return line;
}

export function svgLine(x1, y1, x2, y2, className) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', `${x1}`);
    line.setAttribute('y1', `${y1}`);
    line.setAttribute('x2', `${x2}`);
    line.setAttribute('y2', `${y2}`);
    line.setAttribute('class', className);
    return line;
}

export function svgCircle(cx, cy, r, className) {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', `${cx}`);
    circle.setAttribute('cy', `${cy}`);
    circle.setAttribute('r', `${r}`);
    circle.setAttribute('class', className);
    return circle;
}

export function svgRect(x, y, width, height, className) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', `${x}`);
    rect.setAttribute('y', `${y}`);
    rect.setAttribute('width', `${width}`);
    rect.setAttribute('height', `${height}`);
    rect.setAttribute('rx', '7');
    rect.setAttribute('ry', '7');
    rect.setAttribute('class', className);
    return rect;
}

export function svgGroup(className) {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', className);
    return group;
}

export function svgText(x, y, text, className) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    node.setAttribute('x', `${x}`);
    node.setAttribute('y', `${y}`);
    node.setAttribute('class', className);
    node.textContent = String(text);
    return node;
}
