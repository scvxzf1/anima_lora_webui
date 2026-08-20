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

export async function loadTrainingContext({ refresh = false, includeGpus = true } = {}) {
    if (cachedContext && !refresh && (!includeGpus || cachedContext.gpusLoaded)) return cachedContext;
    const [groupsPayload, presetsPayload, gpuPayload] = await Promise.all([
        api('/api/config/file-groups?kind=training'),
        api('/api/presets'),
        includeGpus ? api('/api/training/gpus') : Promise.resolve(null),
    ]);
    // Read the committed selection after the inventory requests finish. A preset
    // library refresh may overlap a config switch; an early snapshot would let
    // the slower refresh overwrite the newly committed file/preset selection.
    const stored = readStoredContext();
    const groups = Array.isArray(groupsPayload) ? groupsPayload : [];
    const files = groups.flatMap((group) => (group.files || [])
        .filter((file) => file.trainable)
        .map((file) => ({ ...file, methods_subdir: file.methods_subdir || group.methods_subdir })));
    // gui-methods files are system read-only templates; the config editor can only
    // persist changes into editable (unlocked) files. Default to the first
    // editable file instead of the locked gui-methods/lora.toml template so a
    // first-time save actually reaches the backend.
    const editableFiles = files.filter((file) => !file.locked && !file.readonly);
    const fallback = editableFiles[0]
        || files.find((file) => file.path === 'configs/gui-methods/lora.toml')
        || files[0]
        || null;
    const selected = files.find((file) => file.path === stored.configFile) || fallback;
    const presets = Array.isArray(presetsPayload?.items) ? presetsPayload.items : [];
    const gpus = includeGpus
        ? (Array.isArray(gpuPayload?.gpus) ? gpuPayload.gpus : [])
        : (Array.isArray(cachedContext?.gpus) ? cachedContext.gpus : []);
    const context = {
        groups,
        files,
        editableFiles,
        presets,
        gpus,
        configFile: selected?.path || '',
        variant: selected?.method || 'lora',
        methodsSubdir: selected?.methods_subdir || 'gui-methods',
        preset: presets.includes(stored.preset) ? stored.preset : (presets.includes('default') ? 'default' : presets[0] || 'default'),
        gpuWhitelist: includeGpus
            ? sanitizeGpuSelection(stored.gpuWhitelist, gpus)
            : preserveGpuSelection(cachedContext?.gpuWhitelist || stored.gpuWhitelist),
        gpusLoaded: includeGpus || Boolean(cachedContext?.gpusLoaded),
    };
    storeContext(context);
    return context;
}

function preserveGpuSelection(value) {
    return (Array.isArray(value) ? value : [])
        .map(Number)
        .filter((item, index, list) => Number.isInteger(item) && item >= 0 && list.indexOf(item) === index);
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
    const fileOptions = context.files.map((file) => {
        const readOnly = Boolean(file.locked || file.readonly);
        return `<option value="${file.path}" ${file.path === context.configFile ? 'selected' : ''} ${readOnly ? 'data-locked="true"' : ''}>${file.label || file.filename}${readOnly ? '（只读）' : ''}</option>`;
    }).join('');
    const presetOptions = context.presets.map((preset) =>
        `<option value="${preset}" ${preset === context.preset ? 'selected' : ''}>${preset}</option>`
    ).join('');
    return `
        <section class="dragon-config-runbar dragon-reveal" aria-label="训练运行设置">
            <label class="dragon-runbar-field dragon-runbar-config"><span>当前训练配置</span><select class="dragon-select" name="training_config_file" autocomplete="off" data-training-context="file">${fileOptions}</select></label>
            <label class="dragon-runbar-field"><span>运行覆盖</span><select class="dragon-select" name="training_preset" autocomplete="off" data-training-context="preset">${presetOptions}</select></label>
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
        <label><input type="checkbox" name="training_gpu" value="${gpu.index}" ${selected.has(Number(gpu.index)) ? 'checked' : ''}><span>${gpu.label || `GPU ${gpu.index} · ${gpu.name}`}</span></label>
    `).join('');
    return `<div class="dragon-runbar-device"><span class="dragon-runbar-label">训练设备</span><details class="dragon-runbar-gpus" data-training-gpus><summary>${label}</summary><div class="dragon-runbar-gpu-list"><button type="button" data-gpu-all>使用全部 GPU</button>${options || '<span>未读取到 GPU</span>'}</div></details></div>`;
}

export function selectTrainingConfigFile(context, file, { notify = true, persist = true } = {}) {
    if (!file?.path) return null;
    const nextContext = {
        ...context,
        configFile: file.path,
        variant: file.method || context.variant || 'lora',
        methodsSubdir: file.methods_subdir || context.methodsSubdir || 'gui-methods',
    };
    if (persist) storeContext(nextContext);
    if (notify) window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    return nextContext;
}

export function selectTrainingPreset(context, preset, { notify = true, persist = true } = {}) {
    const nextContext = { ...context, preset };
    if (persist) storeContext(nextContext);
    if (notify) window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    return nextContext;
}

export function commitTrainingContext(context) {
    storeContext(context);
    return context;
}

export function isEditableConfigFile(context) {
    const file = Array.isArray(context?.files)
        ? context.files.find((item) => item.path === context.configFile)
        : null;
    // Unknown or empty config files fall through to the backend, which will
    // report the authoritative lock/path error.
    if (!file) return true;
    return !file.locked && !file.readonly;
}

export function bindTrainingControls(root, context, {
    saveChanges,
    beforeContextChange,
    onConfigFileChange,
    onPresetChange,
} = {}) {
    root.querySelector('[data-training-context="file"]')?.addEventListener('change', (event) => {
        if (beforeContextChange && beforeContextChange() === false) {
            event.target.value = context.configFile;
            return;
        }
        const file = context.files.find((item) => item.path === event.target.value);
        if (!file) return;
        if (onConfigFileChange) {
            event.target.value = context.configFile;
            onConfigFileChange(file);
        } else selectTrainingConfigFile(context, file);
    });
    root.querySelector('[data-training-context="preset"]')?.addEventListener('change', (event) => {
        if (beforeContextChange && beforeContextChange() === false) {
            event.target.value = context.preset;
            return;
        }
        if (onPresetChange) {
            const preset = event.target.value;
            event.target.value = context.preset;
            onPresetChange(preset);
        } else selectTrainingPreset(context, event.target.value);
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
        await showResultDialog(root, payload, action);
        // Adding to the queue is an in-place action on the configuration page.
        // Starting immediately still takes the user to the live monitor.
        if (payload.ok && action !== 'queue') window.location.hash = '#live-training';
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
        eyebrow: '训练前检查',
        title: payload?.ok ? '训练前检查通过' : '训练前检查未通过',
        body: preflightLines(payload),
        confirmText: closeOnly ? '' : '返回修改',
    });
}

function showLaunchConfirmation(root, action, preflight) {
    return openDialog(root, {
        eyebrow: action === 'queue' ? '训练队列' : '训练启动',
        title: action === 'queue' ? '确认加入训练队列' : '确认开始训练',
        body: `${preflightLines(preflight)}<p>${action === 'queue' ? '配置会被冻结并加入暂停的队列。' : '需要预处理时会先完成预处理，再自动开始训练。'}</p>`,
        confirmText: action === 'queue' ? '加入队列' : '开始训练',
        cancelText: '取消',
    });
}

function showResultDialog(root, payload, action) {
    const queued = action === 'queue' && payload?.ok;
    return openDialog(root, {
        eyebrow: queued ? '训练队列' : '训练结果',
        title: queued ? '已加入训练队列' : (payload?.ok ? '训练已启动' : '操作失败'),
        body: queued
            ? `<p>${payload?.message || '配置已加入暂停的训练队列。'}</p><p class="dragon-training-dialog-note">当前仍停留在训练配置页；可稍后从“训练队列”查看并启动任务。</p>`
            : `<p>${payload?.message || payload?.error || '后端未返回详细信息'}</p>`,
        confirmText: '关闭',
    });
}

function openDialog(root, { eyebrow = '训练配置', title, body, confirmText = '关闭', cancelText = '' }) {
    const dialog = root.querySelector('[data-training-dialog]');
    const content = dialog?.querySelector('[data-training-dialog-content]');
    if (!dialog || !content) return Promise.resolve(false);
    content.innerHTML = `<div class="dragon-training-dialog-shell"><header class="dragon-training-dialog-header"><span class="dragon-eyebrow">${eyebrow}</span><h2>${title}</h2></header><div class="dragon-training-dialog-body">${body}</div><footer class="dragon-training-dialog-actions">${cancelText ? `<button class="dragon-btn dragon-btn-secondary" value="cancel">${cancelText}</button>` : ''}${confirmText ? `<button class="dragon-btn dragon-btn-primary" value="confirm">${confirmText}</button>` : '<button class="dragon-btn dragon-btn-primary" value="cancel">关闭</button>'}</footer></div>`;
    content.querySelectorAll('button').forEach((button) => button.addEventListener('click', () => dialog.close(button.value)));
    dialog.showModal();
    return new Promise((resolve) => dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), { once: true }));
}
