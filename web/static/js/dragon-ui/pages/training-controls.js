import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

const api = createApiClient();
const STORAGE_KEY = 'anima_dragon_training_context';
let cachedContext = null;

const CHECK_LABELS = {
    pretrained_model_name_or_path: '基础 DiT 模型',
    qwen3: 'Qwen3 文本编码器',
    vae: 'VAE 模型',
    dataset_config: '数据集配置',
    source_image_dir: '源图像目录',
    training_images: '训练图像',
    resized_image_dir: '预处理图像',
    lora_cache_dir: '训练缓存',
    latent_cache: 'VAE 缓存',
    text_cache: '文本缓存',
    preprocess_environment: '预处理环境',
    runtime_preprocess: '运行预处理',
    output_name: '输出名称',
    schema: '配置规则',
    preflight: '训练前检查',
};

const LEVEL_META = {
    error: { label: '错误', icon: 'x' },
    warning: { label: '警告', icon: 'activity' },
    ok: { label: '通过', icon: 'check' },
    info: { label: '信息', icon: 'activity' },
};

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
        <dialog class="dragon-training-dialog" data-training-dialog aria-labelledby="dragon-training-dialog-title" aria-describedby="dragon-training-dialog-body"><div data-training-dialog-content></div></dialog>
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
    if (root.dataset.trainingBusy === 'true') return;
    const actionButtons = [...root.querySelectorAll('[data-training-action]')];
    const previousDisabled = actionButtons.map((button) => button.disabled);
    const button = root.querySelector(`[data-training-action="${action}"]`);
    root.dataset.trainingBusy = 'true';
    actionButtons.forEach((item) => { item.disabled = true; });
    button?.setAttribute('aria-busy', 'true');
    try {
        if (saveChanges && !(await saveChanges())) return;
        const liveContext = cachedContext || context;
        const preflight = await api('/api/training/preflight', {
            method: 'POST',
            body: JSON.stringify(trainingPayload(liveContext)),
        });
        if (action === 'preflight' || !preflight.ok) {
            await showPreflightDialog(root, preflight, action === 'preflight');
            if (!preflight.ok) focusFirstPreflightError(root, preflight);
            return;
        }
        const confirmed = await showLaunchConfirmation(root, action, preflight, liveContext);
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
    } catch (error) {
        await showResultDialog(root, {
            ok: false,
            error: `请求失败：${error?.message || error}`,
        }, action);
    } finally {
        delete root.dataset.trainingBusy;
        actionButtons.forEach((item, index) => { item.disabled = previousDisabled[index]; });
        button?.removeAttribute('aria-busy');
    }
}

function focusFirstPreflightError(root, payload) {
    const controls = [...root.querySelectorAll('[data-key]')];
    const checks = Array.isArray(payload?.checks) ? payload.checks : [];
    const item = checks.find((check) => normalizeLevel(check?.level) === 'error'
        && controls.some((control) => control.dataset.key === String(check?.key || '')));
    if (!item) return;
    const control = controls.find((candidate) => candidate.dataset.key === String(item.key));
    const field = control?.closest('.dragon-field');
    field?.removeAttribute('hidden');
    field?.closest('[data-config-filter-group]')?.removeAttribute('hidden');
    const details = field?.closest('details');
    if (details) details.open = true;
    field?.setAttribute('data-preflight-error', 'true');
    control?.setAttribute('aria-invalid', 'true');
    field?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    requestAnimationFrame(() => control?.focus());
    const clear = () => {
        field?.removeAttribute('data-preflight-error');
        control?.removeAttribute('aria-invalid');
    };
    control?.addEventListener('input', clear, { once: true });
    control?.addEventListener('change', clear, { once: true });
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

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function normalizeLevel(level) {
    return Object.hasOwn(LEVEL_META, level) ? level : 'info';
}

function preflightSummary(payload) {
    const checks = Array.isArray(payload?.checks) ? payload.checks : [];
    const summary = payload?.summary || {};
    return {
        checks,
        total: Number.isFinite(Number(summary.checks)) ? Number(summary.checks) : checks.length,
        errors: Number.isFinite(Number(summary.errors))
            ? Number(summary.errors)
            : checks.filter((item) => normalizeLevel(item.level) === 'error').length,
        warnings: Number.isFinite(Number(summary.warnings))
            ? Number(summary.warnings)
            : checks.filter((item) => normalizeLevel(item.level) === 'warning').length,
    };
}

function preflightTone(payload) {
    const summary = preflightSummary(payload);
    if (summary.errors) return 'error';
    if (summary.warnings) return 'warning';
    return payload?.ok ? 'success' : 'error';
}

function renderPreflightOverview(payload) {
    const summary = preflightSummary(payload);
    const tone = preflightTone(payload);
    const meta = tone === 'success' ? LEVEL_META.ok : LEVEL_META[tone];
    const headline = tone === 'success'
        ? '必要条件已经就绪'
        : tone === 'warning'
            ? '检查通过，但建议先确认警告'
            : '存在阻止启动的问题';
    const detail = tone === 'success'
        ? `共完成 ${summary.total} 项检查，可以继续下一步。`
        : tone === 'warning'
            ? `${summary.warnings} 项警告不会阻止操作，请确认风险后继续。`
            : `${summary.errors || 1} 项错误需要处理，当前不会创建训练任务。`;
    return `
        <section class="dragon-training-overview" data-tone="${tone}" aria-label="训练前检查摘要">
            <span class="dragon-training-overview-icon">${renderIcon(meta.icon, 'dragon-training-state-icon')}</span>
            <div class="dragon-training-overview-copy"><strong>${headline}</strong><span>${detail}</span></div>
            <dl class="dragon-training-metrics">
                <div><dt>检查</dt><dd>${summary.total}</dd></div>
                <div><dt>警告</dt><dd>${summary.warnings}</dd></div>
                <div><dt>错误</dt><dd>${summary.errors}</dd></div>
            </dl>
        </section>`;
}

function checkCopy(item) {
    const key = String(item?.key || 'preflight');
    const label = CHECK_LABELS[key] || key.replaceAll('_', ' ');
    const message = String(item?.message || label);
    const detail = message.startsWith(label) ? message.slice(label.length).trim() : message;
    return { label, detail: detail || message };
}

function renderPreflightChecks(payload) {
    const levelOrder = { error: 0, warning: 1, info: 2, ok: 3 };
    const checks = preflightSummary(payload).checks
        .map((item, index) => ({ item, index, level: normalizeLevel(item?.level) }))
        .sort((a, b) => levelOrder[a.level] - levelOrder[b.level] || a.index - b.index);
    if (!checks.length) {
        return `<p class="dragon-training-empty">${escapeHtml(payload?.error || (payload?.ok ? '检查通过，可以继续。' : '后端没有返回检查明细。'))}</p>`;
    }
    return `
        <section class="dragon-training-check-section" aria-label="检查明细">
            <header><h3>检查明细</h3><span>${checks.length} 项</span></header>
            <ul class="dragon-training-checks">${checks.map(({ item, level }) => {
                const meta = LEVEL_META[level];
                const copy = checkCopy(item);
                const path = item?.path ? `<code title="${escapeHtml(item.path)}">${escapeHtml(item.path)}</code>` : '';
                return `<li data-level="${level}"><span class="dragon-training-check-icon">${renderIcon(meta.icon, 'dragon-training-state-icon')}</span><div class="dragon-training-check-copy"><div><strong>${escapeHtml(copy.label)}</strong><span>${escapeHtml(copy.detail)}</span></div>${path}</div><span class="dragon-training-check-badge">${meta.label}</span></li>`;
            }).join('')}</ul>
        </section>`;
}

function renderLaunchContext(context, action) {
    const selected = context?.files?.find((item) => item.path === context.configFile);
    const configLabel = selected?.label || selected?.filename || context?.configFile || '未选择';
    const gpuIndexes = cachedContext?.gpuWhitelist || context?.gpuWhitelist || [];
    const gpuLabel = gpuIndexes.length ? gpuIndexes.map((index) => `GPU ${index}`).join('、') : '全部可用 GPU';
    return `
        <dl class="dragon-training-launch-context" aria-label="本次运行上下文">
            <div><dt>训练配置</dt><dd title="${escapeHtml(context?.configFile || '')}">${escapeHtml(configLabel)}</dd></div>
            <div><dt>运行覆盖</dt><dd>${escapeHtml(context?.preset || 'default')}</dd></div>
            <div><dt>训练设备</dt><dd>${escapeHtml(gpuLabel)}</dd></div>
            <div><dt>执行方式</dt><dd>${action === 'queue' ? '暂停入队' : '立即启动'}</dd></div>
        </dl>`;
}

function renderLaunchPlan(action, context) {
    const queueMode = action === 'queue';
    const runtimeConfig = String(context?.configFile || '').replaceAll('\\', '/').endsWith('/config.runtime.toml');
    const detail = queueMode
        ? '当前配置会冻结为独立快照并加入暂停队列，不会立即占用训练设备。'
        : runtimeConfig
            ? '将使用当前运行快照直接启动训练，并切换到实时训练页面。'
            : '将创建独立运行目录；需要预处理时会自动完成，并在结束后衔接训练。';
    return `<section class="dragon-training-plan" data-tone="${queueMode ? 'queue' : 'start'}"><span>${renderIcon(queueMode ? 'list' : 'activity', 'dragon-training-plan-icon')}</span><div><strong>${queueMode ? '冻结配置，暂停入队' : '立即执行本次训练'}</strong><p>${detail}</p></div></section>`;
}

function showPreflightDialog(root, payload, closeOnly) {
    const tone = preflightTone(payload);
    return openDialog(root, {
        eyebrow: '训练前检查',
        title: payload?.ok ? '训练前检查通过' : '训练前检查未通过',
        description: closeOnly ? '检查结果不会自动启动或加入任务。' : '请先处理错误，再重新发起操作。',
        body: `${renderPreflightOverview(payload)}${renderPreflightChecks(payload)}`,
        tone,
        icon: tone === 'success' ? 'check' : tone === 'warning' ? 'activity' : 'x',
        confirmText: closeOnly ? '关闭检查' : '返回配置',
        confirmIcon: closeOnly && tone === 'success' ? 'check' : 'x',
    });
}

function showLaunchConfirmation(root, action, preflight, context) {
    const queueMode = action === 'queue';
    const tone = preflightTone(preflight);
    return openDialog(root, {
        eyebrow: queueMode ? '训练队列' : '训练启动',
        title: queueMode ? '确认加入训练队列' : '确认开始训练',
        description: queueMode ? '检查本次运行上下文，然后冻结配置。' : '检查本次运行上下文，然后立即执行。',
        body: `${renderPreflightOverview(preflight)}${renderLaunchContext(context, action)}${renderLaunchPlan(action, context)}${renderPreflightChecks(preflight)}`,
        tone,
        icon: queueMode ? 'list' : 'activity',
        confirmText: queueMode ? '加入队列' : '开始训练',
        confirmIcon: queueMode ? 'list' : 'activity',
        cancelText: '取消',
    });
}

function showResultDialog(root, payload, action) {
    const queued = action === 'queue' && payload?.ok;
    return openDialog(root, {
        eyebrow: queued ? '训练队列' : '训练结果',
        title: queued ? '已加入训练队列' : (payload?.ok ? '训练已启动' : '操作失败'),
        description: queued ? '任务快照已经创建，队列保持暂停。' : (payload?.ok ? '运行目录已创建，正在进入实时训练。' : '没有创建新的训练任务。'),
        body: queued
            ? `<section class="dragon-training-result" data-tone="success"><span>${renderIcon('check', 'dragon-training-result-icon')}</span><div><strong>${escapeHtml(payload?.message || '配置已加入暂停的训练队列。')}</strong><p>当前仍停留在训练配置页；可稍后从「训练队列」查看并启动任务。</p></div></section>`
            : payload?.preflight
                ? `${renderPreflightOverview(payload.preflight)}${renderPreflightChecks(payload.preflight)}<p class="dragon-training-dialog-note">${escapeHtml(payload?.error || '训练启动条件已经发生变化，请返回配置页处理。')}</p>`
                : `<section class="dragon-training-result" data-tone="${payload?.ok ? 'success' : 'error'}"><span>${renderIcon(payload?.ok ? 'check' : 'x', 'dragon-training-result-icon')}</span><div><strong>${escapeHtml(payload?.message || payload?.error || '后端未返回详细信息')}</strong></div></section>`,
        tone: payload?.ok ? 'success' : 'error',
        icon: payload?.ok ? 'check' : 'x',
        confirmText: '关闭',
        confirmIcon: 'check',
    });
}

function openDialog(root, {
    eyebrow = '训练配置',
    title,
    description = '',
    body,
    tone = 'neutral',
    icon = 'activity',
    confirmText = '关闭',
    confirmIcon = 'check',
    cancelText = '',
}) {
    const dialog = root.querySelector('[data-training-dialog]');
    const content = dialog?.querySelector('[data-training-dialog-content]');
    if (!dialog || !content) return Promise.resolve(false);
    dialog.dataset.tone = tone;
    dialog.returnValue = '';
    content.innerHTML = `
        <div class="dragon-training-dialog-shell">
            <header class="dragon-training-dialog-header">
                <div class="dragon-training-dialog-heading">
                    <span class="dragon-training-dialog-symbol">${renderIcon(icon, 'dragon-training-dialog-symbol-icon')}</span>
                    <div><span class="dragon-eyebrow">${escapeHtml(eyebrow)}</span><h2 id="dragon-training-dialog-title">${escapeHtml(title)}</h2>${description ? `<p>${escapeHtml(description)}</p>` : ''}</div>
                </div>
                <button class="dragon-training-dialog-close" type="button" value="cancel" aria-label="关闭弹窗" title="关闭">${renderIcon('x', 'dragon-training-dialog-close-icon')}</button>
            </header>
            <div class="dragon-training-dialog-body" id="dragon-training-dialog-body" aria-live="polite">${body}</div>
            <footer class="dragon-training-dialog-actions">
                ${cancelText ? `<button class="dragon-btn dragon-btn-secondary" type="button" value="cancel" autofocus>${escapeHtml(cancelText)}</button>` : ''}
                <button class="dragon-btn dragon-btn-primary" type="button" value="confirm" ${cancelText ? '' : 'autofocus'}>${renderIcon(confirmIcon, 'dragon-btn-icon')}<span>${escapeHtml(confirmText || '关闭')}</span></button>
            </footer>
        </div>`;
    content.querySelectorAll('button').forEach((button) => button.addEventListener('click', () => dialog.close(button.value)));
    const closeFromBackdrop = (event) => {
        if (event.target === dialog) dialog.close('cancel');
    };
    dialog.addEventListener('click', closeFromBackdrop);
    if (!dialog.open) dialog.showModal();
    return new Promise((resolve) => dialog.addEventListener('close', () => {
        dialog.removeEventListener('click', closeFromBackdrop);
        resolve(dialog.returnValue === 'confirm');
    }, { once: true }));
}
