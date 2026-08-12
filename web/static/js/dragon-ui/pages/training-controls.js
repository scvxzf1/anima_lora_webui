import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

const api = createApiClient();
const STORAGE_KEY = 'anima_dragon_training_context';
let cachedContext = null;

function readStoredContext() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch { return {}; }
}

function storeContext(context) {
    cachedContext = context;
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(context)); } catch { /* ignore */ }
}

export async function loadTrainingContext({ refresh = false } = {}) {
    if (cachedContext && !refresh) return cachedContext;
    const stored = readStoredContext();
    const [groupsPayload, presetsPayload, gpuPayload] = await Promise.all([
        api('/api/config/file-groups?kind=training'),
        api('/api/presets'),
        api('/api/training/gpus'),
    ]);
    const groups = Array.isArray(groupsPayload) ? groupsPayload : [];
    const files = groups.flatMap((group) => (group.files || [])
        .filter((file) => file.trainable && !file.locked)
        .map((file) => ({ ...file, methods_subdir: file.methods_subdir || group.methods_subdir })));
    const fallback = files.find((file) => file.path === 'configs/gui-methods/lora.toml') || files[0] || null;
    const selected = files.find((file) => file.path === stored.configFile) || fallback;
    const presets = Array.isArray(presetsPayload?.items) ? presetsPayload.items : [];
    const gpus = Array.isArray(gpuPayload?.gpus) ? gpuPayload.gpus : [];
    const context = {
        files,
        presets,
        gpus,
        configFile: selected?.path || '',
        variant: selected?.method || 'lora',
        methodsSubdir: selected?.methods_subdir || 'gui-methods',
        preset: presets.includes(stored.preset) ? stored.preset : (presets.includes('default') ? 'default' : presets[0] || 'default'),
        gpuWhitelist: sanitizeGpuSelection(stored.gpuWhitelist, gpus),
    };
    storeContext(context);
    return context;
}

function sanitizeGpuSelection(value, gpus) {
    const known = new Set(gpus.map((gpu) => Number(gpu.index)));
    return (Array.isArray(value) ? value : [])
        .map(Number)
        .filter((item, index, list) => Number.isInteger(item) && known.has(item) && list.indexOf(item) === index);
}

export function mergedConfigUrl(context) {
    const params = new URLSearchParams({
        variant: context.variant,
        preset: context.preset,
        methods_subdir: context.methodsSubdir,
        config_file: context.configFile,
    });
    return `/api/config/merged?${params.toString()}`;
}

export function renderTrainingControls(context) {
    const fileOptions = context.files.map((file) =>
        `<option value="${file.path}" ${file.path === context.configFile ? 'selected' : ''}>${file.label || file.filename}</option>`
    ).join('');
    const presetOptions = context.presets.map((preset) =>
        `<option value="${preset}" ${preset === context.preset ? 'selected' : ''}>${preset}</option>`
    ).join('');
    return `
        <section class="dragon-config-runbar dragon-reveal" aria-label="训练运行设置">
            <label class="dragon-runbar-field dragon-runbar-config"><span>当前训练配置</span><select class="dragon-select" data-training-context="file">${fileOptions}</select></label>
            <label class="dragon-runbar-field"><span>运行覆盖</span><select class="dragon-select" data-training-context="preset">${presetOptions}</select></label>
            ${renderGpuPicker(context)}
            <div class="dragon-runbar-actions">
                <button class="dragon-btn dragon-btn-secondary" type="button" data-training-action="preflight">${renderIcon('check', 'dragon-btn-icon')}<span>训练前检查</span></button>
                <button class="dragon-btn dragon-btn-primary" type="button" data-training-action="start">${renderIcon('activity', 'dragon-btn-icon')}<span>开始训练</span></button>
                <button class="dragon-btn dragon-btn-secondary" type="button" data-training-action="queue">${renderIcon('list', 'dragon-btn-icon')}<span>加入队列</span></button>
            </div>
        </section>
        <dialog class="dragon-training-dialog" data-training-dialog><div data-training-dialog-content></div></dialog>
    `;
}

function renderGpuPicker(context) {
    const selected = new Set(context.gpuWhitelist);
    const label = selected.size ? `已选 ${selected.size} 张 GPU` : '全部 GPU';
    const options = context.gpus.map((gpu) => `
        <label><input type="checkbox" value="${gpu.index}" ${selected.has(Number(gpu.index)) ? 'checked' : ''}><span>${gpu.label || `GPU ${gpu.index} · ${gpu.name}`}</span></label>
    `).join('');
    return `<div class="dragon-runbar-device"><span class="dragon-runbar-label">训练设备</span><details class="dragon-runbar-gpus" data-training-gpus><summary>${label}</summary><div class="dragon-runbar-gpu-list"><button type="button" data-gpu-all>使用全部 GPU</button>${options || '<span>未读取到 GPU</span>'}</div></details></div>`;
}

export function bindTrainingControls(root, context, { saveChanges } = {}) {
    root.querySelector('[data-training-context="file"]')?.addEventListener('change', (event) => {
        const file = context.files.find((item) => item.path === event.target.value);
        if (!file) return;
        storeContext({ ...context, configFile: file.path, variant: file.method, methodsSubdir: file.methods_subdir });
        window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    });
    root.querySelector('[data-training-context="preset"]')?.addEventListener('change', (event) => {
        storeContext({ ...context, preset: event.target.value });
        window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    });
    bindGpuSelection(root, context);
    root.querySelectorAll('[data-training-action]').forEach((button) => {
        button.addEventListener('click', () => runTrainingAction(root, context, button.dataset.trainingAction, saveChanges));
    });
}

function bindGpuSelection(root, context) {
    const details = root.querySelector('[data-training-gpus]');
    details?.querySelectorAll('input[type="checkbox"]').forEach((input) => {
        input.addEventListener('change', () => {
            const selected = [...details.querySelectorAll('input:checked')].map((item) => Number(item.value));
            storeContext({ ...context, gpuWhitelist: selected });
            details.querySelector('summary').textContent = selected.length ? `已选 ${selected.length} 张 GPU` : '全部 GPU';
        });
    });
    details?.querySelector('[data-gpu-all]')?.addEventListener('click', () => {
        details.querySelectorAll('input').forEach((input) => { input.checked = false; });
        storeContext({ ...context, gpuWhitelist: [] });
        details.querySelector('summary').textContent = '全部 GPU';
    });
}

async function runTrainingAction(root, context, action, saveChanges) {
    const button = root.querySelector(`[data-training-action="${action}"]`);
    if (button) button.disabled = true;
    try {
        if (saveChanges && !(await saveChanges())) return;
        const liveContext = cachedContext || context;
        const preflight = await api('/api/training/preflight', {
            method: 'POST',
            body: JSON.stringify(trainingPayload(liveContext)),
        });
        if (action === 'preflight' || !preflight.ok) {
            await showPreflightDialog(root, preflight, action === 'preflight');
            return;
        }
        const confirmed = await showLaunchConfirmation(root, action, preflight);
        if (!confirmed) return;
        const endpoint = action === 'queue' ? '/api/training/queue/start' : '/api/training/start';
        const payload = await api(endpoint, {
            method: 'POST',
            body: JSON.stringify({
                ...trainingPayload(liveContext),
                confirmed: true,
                confirm_preprocess: true,
                start_paused: action === 'queue',
            }),
        });
        await showResultDialog(root, payload);
        if (payload.ok) window.location.hash = action === 'queue' ? '#queue' : '#live-training';
    } finally {
        if (button) button.disabled = false;
    }
}

function trainingPayload(context) {
    return {
        variant: context.variant,
        preset: context.preset,
        methods_subdir: context.methodsSubdir,
        config_file: context.configFile,
        extra_args: [],
        gpu_whitelist: cachedContext?.gpuWhitelist || context.gpuWhitelist || [],
    };
}

function preflightLines(payload) {
    const checks = Array.isArray(payload?.checks) ? payload.checks : [];
    if (!checks.length) return `<p>${payload?.error || (payload?.ok ? '检查通过，可以开始训练。' : '没有收到检查结果。')}</p>`;
    return `<ul class="dragon-training-checks">${checks.map((item) => `<li data-level="${item.level || 'info'}">${item.message || item.key}</li>`).join('')}</ul>`;
}

function showPreflightDialog(root, payload, closeOnly) {
    return openDialog(root, {
        title: payload?.ok ? '训练前检查通过' : '训练前检查未通过',
        body: preflightLines(payload),
        confirmText: closeOnly ? '' : '返回修改',
    });
}

function showLaunchConfirmation(root, action, preflight) {
    return openDialog(root, {
        title: action === 'queue' ? '确认加入训练队列' : '确认开始训练',
        body: `${preflightLines(preflight)}<p>${action === 'queue' ? '配置会被冻结并加入暂停的队列。' : '需要预处理时会先完成预处理，再自动开始训练。'}</p>`,
        confirmText: action === 'queue' ? '加入队列' : '开始训练',
        cancelText: '取消',
    });
}

function showResultDialog(root, payload) {
    return openDialog(root, {
        title: payload?.ok ? '操作已提交' : '操作失败',
        body: `<p>${payload?.message || payload?.error || '后端未返回详细信息'}</p>`,
        confirmText: '关闭',
    });
}

function openDialog(root, { title, body, confirmText = '关闭', cancelText = '' }) {
    const dialog = root.querySelector('[data-training-dialog]');
    const content = dialog?.querySelector('[data-training-dialog-content]');
    if (!dialog || !content) return Promise.resolve(false);
    content.innerHTML = `<h2>${title}</h2><div>${body}</div><div class="dragon-training-dialog-actions">${cancelText ? `<button class="dragon-btn dragon-btn-secondary" value="cancel">${cancelText}</button>` : ''}${confirmText ? `<button class="dragon-btn dragon-btn-primary" value="confirm">${confirmText}</button>` : '<button class="dragon-btn dragon-btn-primary" value="cancel">关闭</button>'}</div>`;
    content.querySelectorAll('button').forEach((button) => button.addEventListener('click', () => dialog.close(button.value)));
    dialog.showModal();
    return new Promise((resolve) => dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), { once: true }));
}
