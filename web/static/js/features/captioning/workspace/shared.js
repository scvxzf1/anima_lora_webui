import { captioningApi, jsonOptions } from '../api.js?v=dragon-ui-20260829v12';
import { escapeAttribute, escapeHtml } from '../utils.js?v=dragon-ui-20260829v12';

export { captioningApi, escapeAttribute, escapeHtml, jsonOptions };

export function groupOptions(state, selected = '') {
    return (state.workspace?.groups || []).map((group) => `<option value="${escapeAttribute(group.id)}" ${group.id === selected ? 'selected' : ''}>${escapeHtml(group.name)}</option>`).join('');
}

export function scheduleOptions(state, selected = '') {
    const active = selected || state.routing?.default_schedule_id;
    return (state.routing?.schedules || []).map((schedule) => `<option value="${escapeAttribute(schedule.id)}" ${schedule.id === active ? 'selected' : ''}>${escapeHtml(schedule.name)}</option>`).join('');
}

export function promptOptions(state, kind, selected = '') {
    return (state.workspace?.prompts?.[kind] || []).map((prompt) => `<option value="${escapeAttribute(prompt.id)}" ${prompt.id === selected ? 'selected' : ''}>${escapeHtml(prompt.name)}</option>`).join('');
}

export function selectedGroup(state, id) {
    return (state.workspace?.groups || []).find((group) => group.id === id);
}

export function selectedPrompt(state, kind, id) {
    return (state.workspace?.prompts?.[kind] || []).find((prompt) => prompt.id === id);
}

export async function saveWorkspace(state) {
    const payload = JSON.parse(JSON.stringify(state.workspace));
    for (const kind of ['system', 'user']) {
        payload.prompts[kind] = (payload.prompts[kind] || []).filter((prompt) => prompt.builtin || String(prompt.content || '').trim());
    }
    state.workspace = await captioningApi('/workspace', jsonOptions('PUT', payload));
    return state.workspace;
}

export function feedback(root, message, tone = 'info', selector = '[data-workspace-feedback]') {
    const node = root.querySelector(selector);
    if (node) { node.textContent = message; node.dataset.tone = tone; }
}

export function panelShell(eyebrow, title, actions, body) {
    return `<section class="dragon-caption-workspace-panel"><header><div><span class="dragon-eyebrow">${eyebrow}</span><h2>${title}</h2></div><div class="dragon-caption-panel-actions">${actions || ''}</div></header>${body}<footer><span class="dragon-caption-feedback" data-workspace-feedback role="status" aria-live="polite"></span></footer></section>`;
}

export function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        if (!file || !file.type.startsWith('image/') || file.size > 32 * 1024 * 1024) return reject(new Error('请选择不超过 32MB 的图片'));
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ''));
        reader.onerror = () => reject(new Error('图片读取失败'));
        reader.readAsDataURL(file);
    });
}

export function uid(prefix) {
    return `${prefix}-${crypto.randomUUID()}`;
}
