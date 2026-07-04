export function byId(id) {
    return document.getElementById(id);
}

export function optionalById(id) {
    return byId(id);
}

export function requireById(id, options = {}) {
    const el = byId(id);
    if (el) return el;
    const contract = options.contract ? ` (${options.contract})` : '';
    throw new Error(`[webui-dom-contract] missing required DOM node: #${id}${contract}`);
}

export function bindEvent(id, eventName, handler, options = {}) {
    const required = Boolean(options.required);
    const el = required ? requireById(id, options) : optionalById(id);
    if (!el) {
        if (options.warn !== false) {
            globalThis.console?.warn?.(`[webui-dom-contract] optional DOM node not found: #${id}`);
        }
        return false;
    }
    el.addEventListener(eventName, handler, options.listenerOptions);
    return true;
}

export function val(id) {
    return byId(id)?.value || '';
}

export function setText(id, text) {
    const el = byId(id);
    if (el) el.textContent = text;
}

export function setButtonDisabled(id, disabled) {
    const btn = byId(id);
    if (btn) btn.disabled = Boolean(disabled);
}

export function populateSelect(id, items, preferred = '') {
    const sel = byId(id);
    if (!sel) return;
    const choices = Array.isArray(items) ? items : [];
    const prev = sel.value;
    sel.innerHTML = '';
    sel.disabled = !choices.length;
    for (const item of choices) {
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        sel.appendChild(opt);
    }
    if (choices.includes(prev)) {
        sel.value = prev;
    } else if (preferred && choices.includes(preferred)) {
        sel.value = preferred;
    }
}

export async function copyText(text) {
    if (navigator.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return;
        } catch (_) {
            // 浏览器可能因权限或焦点拒绝 Clipboard API，继续使用 textarea 兜底。
        }
    }
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.left = '-9999px';
    document.body.appendChild(area);
    area.focus();
    area.select();
    try {
        if (!document.execCommand('copy')) {
            throw new Error('浏览器拒绝复制操作');
        }
    } finally {
        area.remove();
    }
}
