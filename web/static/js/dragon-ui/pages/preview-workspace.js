/* Dragon preview workspace controller: sources, paths, images, and weights. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import {
    effectiveDirectory,
    renderPreviewImages,
    renderPreviewWeights,
    renderPreviewWorkspacePage,
    sourceLabel,
} from './preview-workspace-view.js?v=dragon-ui-20260814v43';

const api = createApiClient();

export async function loadPreviewWorkspace() {
    const [settingsPayload, historyPayload] = await Promise.all([
        safeApi('/api/preview/settings'),
        safeApi('/api/training/history?limit=100'),
    ]);
    const settings = normalizeSettings(settingsPayload);
    const tasks = Array.isArray(historyPayload.tasks)
        ? historyPayload.tasks.filter((task) => task.job === 'training')
        : [];
    const groups = buildPreviewGroups(tasks);
    const state = { source: 'training', scope: 'latest', days: 'all', settings, tasks, groups, selectedTaskId: '', selectedGroupKey: '', images: [], requestSeq: 0, weightRequestSeq: 0, dirty: false, beforeUnload: null };
    return {
        html: renderPreviewWorkspacePage({
            settings,
            directory: effectiveDirectory(settings, 'training'),
            error: settings.error,
            tasks,
            groups,
        }),
        onMount(root) {
            const page = root.querySelector('[data-preview-root]') || root;
            bindPreviewWorkspace(page, state);
            loadWorkspaceContent(page, state);
        },
        beforeLeave: () => confirmPreviewDiscard(state),
        onUnmount() {
            state.requestSeq += 1;
            state.weightRequestSeq += 1;
            if (state.beforeUnload) window.removeEventListener('beforeunload', state.beforeUnload);
        },
    };
}

function bindPreviewWorkspace(root, state) {
    const page = root;
    page.querySelectorAll('[data-preview-source]').forEach((button) => {
        button.addEventListener('click', () => changeSource(root, state, button.dataset.previewSource));
    });
    page.querySelector('[data-preview-task]')?.addEventListener('change', (event) => {
        state.selectedTaskId = event.target.value || '';
        loadWorkspaceContent(root, state);
    });
    page.querySelector('[data-preview-group]')?.addEventListener('change', (event) => {
        state.selectedGroupKey = event.target.value || '';
        loadWorkspaceContent(root, state);
    });
    page.querySelector('[data-preview-scope]')?.addEventListener('change', (event) => {
        state.scope = event.target.value || 'latest';
        syncScopeControls(root, state);
        loadWorkspaceContent(root, state);
    });
    page.querySelector('[data-preview-days]')?.addEventListener('change', (event) => {
        state.days = event.target.value || 'all';
        loadWorkspaceContent(root, state);
    });
    page.querySelector('[data-tool-action="refresh-preview"]')?.addEventListener('click', () => loadWorkspaceContent(root, state));
    page.querySelector('[data-tool-action="toggle-settings"]')?.addEventListener('click', () => toggleSettings(root));
    const form = page.querySelector('[data-preview-settings-form]');
    form?.addEventListener('submit', (event) => {
        event.preventDefault();
        saveSettings(root, state);
    });
    form?.addEventListener('input', () => syncPreviewDirty(root, state));
    page.querySelector('[data-preview-action="restore-defaults"]')?.addEventListener('click', () => restoreDefaults(root, state));
    page.querySelector('[data-preview-action="delete-selected"]')?.addEventListener('click', () => deleteSelectedImages(root, state));
    page.addEventListener('change', (event) => {
        if (event.target.matches('[data-preview-image-select]')) syncImageSelection(root, state);
    });
    page.addEventListener('click', (event) => {
        const copy = event.target.closest('[data-preview-copy-weight]');
        if (copy) copyWeightPath(root, copy.dataset.previewCopyWeight, copy);
        const hotstart = event.target.closest('[data-preview-hotstart-weight]');
        if (hotstart) applyHotstartWeight(root, hotstart.dataset.previewHotstartWeight, hotstart);
    });
    syncScopeControls(root, state);
    state.beforeUnload = (event) => {
        if (!state.dirty) return;
        event.preventDefault();
        event.returnValue = '';
    };
    window.addEventListener('beforeunload', state.beforeUnload);
}

function changeSource(root, state, source) {
    if (!['training', 'inference', 'custom'].includes(source)) return;
    state.source = source;
    root.dataset.source = source;
    root.querySelectorAll('[data-preview-source]').forEach((button) => {
        button.setAttribute('aria-pressed', String(button.dataset.previewSource === source));
    });
    const label = root.querySelector('[data-preview-source-label]');
    if (label) label.textContent = sourceLabel(source);
    syncScopeControls(root, state);
    syncDirectory(root, state);
    loadWorkspaceContent(root, state);
}

function syncScopeControls(root, state) {
    const training = state.source === 'training';
    const taskWrap = root.querySelector('[data-preview-task-wrap]');
    const groupWrap = root.querySelector('[data-preview-group-wrap]');
    const scope = root.querySelector('[data-preview-scope]');
    const task = root.querySelector('[data-preview-task]');
    const group = root.querySelector('[data-preview-group]');
    if (scope) scope.disabled = !training;
    if (!training) state.scope = 'latest';
    if (taskWrap) taskWrap.hidden = !training || state.scope !== 'task';
    if (groupWrap) groupWrap.hidden = !training || state.scope !== 'group';
    if (task) task.disabled = !training || state.scope !== 'task';
    if (group) group.disabled = !training || state.scope !== 'group';
    syncImageSelection(root, state);
}

function toggleSettings(root) {
    const panel = root.querySelector('[data-preview-settings-panel]');
    const button = root.querySelector('[data-tool-action="toggle-settings"]');
    if (!panel) return;
    const open = panel.hidden;
    panel.hidden = !open;
    button?.setAttribute('aria-expanded', String(open));
    if (open) panel.querySelector('input')?.focus();
}

async function saveSettings(root, state) {
    const form = root.querySelector('[data-preview-settings-form]');
    const submit = form?.querySelector('[type="submit"]');
    if (!form || state.settings.error) {
        setStatus(root, '路径设置尚未成功读取，请先刷新页面再保存。', 'error');
        return;
    }
    if (submit) submit.disabled = true;
    setStatus(root, '正在保存预览路径…', 'info');
    try {
        const values = Object.fromEntries(new FormData(form).entries());
        const payload = await api('/api/preview/settings', { method: 'PUT', body: JSON.stringify(values) });
        if (payload?.ok === false) throw new Error(payload.error || '保存路径设置失败');
        const refreshed = await api('/api/preview/settings');
        if (refreshed?.ok === false) throw new Error(refreshed.error || '保存后重新读取路径设置失败');
        state.settings = normalizeSettings(refreshed);
        state.dirty = false;
        syncSettingsForm(root, state.settings);
        syncDirectory(root, state);
        setStatus(root, payload.message || '预览路径设置已保存。', 'success');
        await loadWorkspaceContent(root, state, { keepStatus: true });
    } catch (error) {
        setStatus(root, error.message || '保存路径设置失败', 'error');
    } finally {
        if (submit) submit.disabled = false;
    }
}

function restoreDefaults(root, state) {
    const defaults = state.settings.defaults || {};
    const form = root.querySelector('[data-preview-settings-form]');
    if (!form) return;
    for (const key of ['training_dir', 'inference_dir', 'custom_dir']) {
        const input = form.elements.namedItem(key);
        if (input) input.value = defaults[key] ?? (key === 'custom_dir' ? '' : input.value);
    }
    syncPreviewDirty(root, state);
    setStatus(root, '已恢复默认路径；点击“保存路径设置”后生效。', 'info');
}

function syncPreviewDirty(root, state) {
    const form = root.querySelector('[data-preview-settings-form]');
    if (!form) return;
    state.dirty = ['training_dir', 'inference_dir', 'custom_dir'].some((key) => {
        const input = form.elements.namedItem(key);
        return String(input?.value || '').trim() !== String(state.settings[key] || '').trim();
    });
}

function confirmPreviewDiscard(state) {
    if (!state.dirty) return true;
    const allowed = window.confirm('预览路径有未保存修改。离开页面会丢弃这些修改，是否继续？');
    if (allowed) state.dirty = false;
    return allowed;
}

async function loadWorkspaceContent(root, state, options = {}) {
    const requestSeq = ++state.requestSeq;
    const refresh = root.querySelector('[data-tool-action="refresh-preview"]');
    if (refresh) refresh.disabled = true;
    setGalleryLoading(root, state.source);
    if (!options.keepStatus) setStatus(root, `正在读取${sourceLabel(state.source)}…`, 'info');
    try {
        const payload = await api(previewImagesUrl(state));
        if (requestSeq !== state.requestSeq) return;
        if (payload?.ok === false) throw new Error(payload.error || '读取预览图失败');
        renderGallery(root, state, payload);
        if (!options.keepStatus) setStatus(root, payload.message || `已读取 ${Number(payload.count || 0)} 张图片。`, 'success');
    } catch (error) {
        if (requestSeq !== state.requestSeq) return;
        renderGalleryError(root, error.message || '读取预览图失败');
        setStatus(root, error.message || '读取预览图失败', 'error');
    } finally {
        if (requestSeq === state.requestSeq && refresh) refresh.disabled = false;
    }
    if (state.source === 'training') await loadWeights(root, state);
    else clearWeights(root, state);
}

async function loadWeights(root, state) {
    const requestSeq = ++state.weightRequestSeq;
    const panel = root.querySelector('[data-preview-weight-panel]');
    if (panel) panel.hidden = false;
    setWeightLoading(root);
    try {
        const payload = await api(previewWeightsUrl(state));
        if (requestSeq !== state.weightRequestSeq) return;
        if (payload?.ok === false) throw new Error(payload.error || '读取训练权重失败');
        const weights = Array.isArray(payload.weights) ? payload.weights : [];
        const list = root.querySelector('[data-preview-weight-list]');
        const empty = root.querySelector('[data-preview-weight-empty]');
        const meta = root.querySelector('[data-preview-weight-meta]');
        if (list) list.innerHTML = renderPreviewWeights(payload);
        if (empty) {
            empty.hidden = weights.length > 0;
            empty.querySelector('p').textContent = payload.message || '当前训练目录还没有权重文件。';
        }
        if (meta) meta.textContent = `${weights.length} 个权重${payload.directory ? ` · ${payload.directory}` : ''}`;
    } catch (error) {
        if (requestSeq !== state.weightRequestSeq) return;
        const empty = root.querySelector('[data-preview-weight-empty]');
        if (empty) {
            empty.hidden = false;
            empty.querySelector('p').textContent = error.message || '读取训练权重失败';
        }
    }
}

function renderGallery(root, state, payload) {
    const images = Array.isArray(payload.images) ? payload.images : [];
    const grid = root.querySelector('[data-preview-grid]');
    const empty = root.querySelector('[data-preview-empty]');
    const meta = root.querySelector('[data-preview-gallery-meta]');
    const count = root.querySelector('[data-preview-count]');
    state.images = images;
    if (grid) grid.innerHTML = renderPreviewImages(payload);
    if (empty) {
        empty.hidden = images.length > 0;
        empty.querySelector('p').textContent = payload.message || '当前目录暂无图片。';
    }
    if (meta) meta.textContent = `${Number(payload.count || images.length)} / ${Number(payload.total || images.length)} 张 · ${payload.directory || '目录未设置'}`;
    if (count) count.textContent = `${Number(payload.count || images.length)} 张图片`;
    const directory = root.querySelector('[data-preview-directory]');
    if (directory) directory.textContent = payload.directory || effectiveDirectory(state.settings, state.source) || '未设置';
    syncImageSelection(root, state);
}

function renderGalleryError(root, message) {
    const grid = root.querySelector('[data-preview-grid]');
    const empty = root.querySelector('[data-preview-empty]');
    if (grid) grid.innerHTML = '';
    if (empty) {
        empty.hidden = false;
        empty.querySelector('p').textContent = message;
    }
}

function syncImageSelection(root, state) {
    const selected = root.querySelectorAll('[data-preview-image-select]:checked').length;
    const button = root.querySelector('[data-preview-action="delete-selected"]');
    const note = root.querySelector('[data-preview-delete-note]');
    const grouped = state.source === 'training' && state.scope === 'group';
    root.querySelectorAll('[data-preview-image-select]').forEach((input) => { input.disabled = grouped; });
    if (note) note.hidden = !grouped;
    if (!button) return;
    button.disabled = grouped || selected === 0;
    button.textContent = grouped ? '分组视图只读' : (selected ? `删除所选（${selected}）` : '删除所选');
}

async function deleteSelectedImages(root, state) {
    if (state.source === 'training' && state.scope === 'group') {
        setStatus(root, '配置分组聚合了多个训练目录；请切换到单个任务后删除图片。', 'warning');
        return;
    }
    const files = Array.from(root.querySelectorAll('[data-preview-image-select]:checked')).map((input) => input.value).filter(Boolean);
    if (!files.length) return;
    const label = sourceLabel(state.source);
    if (!window.confirm(`确认永久删除所选 ${files.length} 张${label}吗？此操作无法撤销。`)) return;
    const button = root.querySelector('[data-preview-action="delete-selected"]');
    if (button) button.disabled = true;
    setStatus(root, '正在删除所选图片…', 'info');
    try {
        const payload = await api(previewDeleteUrl(state), { method: 'DELETE', body: JSON.stringify({ source: state.source, files }) });
        if (payload?.ok === false && !payload?.deleted_count) throw new Error(payload.error || payload.message || '删除图片失败');
        setStatus(root, payload.message || `已删除 ${Number(payload.deleted_count || files.length)} 张图片。`, payload.ok === false ? 'warning' : 'success');
        await loadWorkspaceContent(root, state, { keepStatus: true });
    } catch (error) {
        setStatus(root, error.message || '删除图片失败', 'error');
        syncImageSelection(root, state);
    }
}

async function copyWeightPath(root, path, button) {
    if (!path) return setStatus(root, '这个权重没有可复制的路径。', 'error');
    try {
        await navigator.clipboard.writeText(path);
        const original = button.textContent;
        button.textContent = '已复制';
        window.setTimeout(() => { button.textContent = original; }, 1200);
        setStatus(root, '权重路径已复制。', 'success');
    } catch {
        setStatus(root, `无法访问剪贴板，请手动复制：${path}`, 'error');
    }
}

async function applyHotstartWeight(root, path, button) {
    if (!path) return setStatus(root, '这个权重没有可用路径。', 'error');
    const context = readStoredTrainingContext();
    if (!context.configFile) return setStatus(root, '尚未选择可编辑的训练配置。请先打开训练配置页选择配置文件。', 'error');
    button.disabled = true;
    setStatus(root, '正在检查权重与当前训练配置的兼容性…', 'info');
    try {
        const payload = await api('/api/training/continue-lora/inspect', {
            method: 'POST',
            body: JSON.stringify({
                path,
                variant: context.variant || 'lora',
                preset: context.preset || 'default',
                methods_subdir: context.methodsSubdir || 'gui-methods',
                config_file: context.configFile || '',
            }),
        });
        if (payload?.ok === false || !payload?.compatible) throw new Error(payload?.message || payload?.error || '当前训练配置与这个权重不兼容');
        const patched = await api('/api/config/raw', {
            method: 'PATCH',
            body: JSON.stringify({ file: context.configFile, values: { network_weights: payload.abs_path || path, dim_from_weights: true } }),
        });
        if (patched?.ok === false) throw new Error(patched.error || '无法写入当前训练配置');
        setStatus(root, '已写入当前训练配置的热启动权重，即将打开适配器设置。', 'success');
        window.setTimeout(() => { window.location.hash = '#config/training-config/adapter-basics'; }, 500);
    } catch (error) {
        setStatus(root, `${error.message || '设置热启动权重失败'}。请确认当前配置可编辑后重试。`, 'error');
        button.disabled = false;
    }
}

function setGalleryLoading(root, source) {
    const grid = root.querySelector('[data-preview-grid]');
    const empty = root.querySelector('[data-preview-empty]');
    const meta = root.querySelector('[data-preview-gallery-meta]');
    if (grid) grid.innerHTML = '';
    if (empty) {
        empty.hidden = false;
        empty.querySelector('p').textContent = `正在读取${sourceLabel(source)}…`;
    }
    if (meta) meta.textContent = '正在读取…';
}

function setWeightLoading(root) {
    const list = root.querySelector('[data-preview-weight-list]');
    const empty = root.querySelector('[data-preview-weight-empty]');
    const meta = root.querySelector('[data-preview-weight-meta]');
    if (list) list.innerHTML = '';
    if (empty) {
        empty.hidden = false;
        empty.querySelector('p').textContent = '正在读取权重…';
    }
    if (meta) meta.textContent = '正在读取…';
}

function clearWeights(root, state) {
    state.weightRequestSeq += 1;
    const panel = root.querySelector('[data-preview-weight-panel]');
    const meta = root.querySelector('[data-preview-weight-meta]');
    const list = root.querySelector('[data-preview-weight-list]');
    const empty = root.querySelector('[data-preview-weight-empty]');
    if (panel) panel.hidden = true;
    if (meta) meta.textContent = `${sourceLabel(state.source)}不关联训练权重`;
    if (list) list.innerHTML = '';
    if (empty) empty.hidden = true;
}

function syncSettingsForm(root, settings) {
    const form = root.querySelector('[data-preview-settings-form]');
    if (!form) return;
    for (const key of ['training_dir', 'inference_dir', 'custom_dir']) {
        const input = form.elements.namedItem(key);
        if (input) input.value = settings[key] || '';
    }
}

function syncDirectory(root, state) {
    const directory = root.querySelector('[data-preview-directory]');
    if (directory) directory.textContent = effectiveDirectory(state.settings, state.source) || '未设置';
}

function setStatus(root, message, tone) {
    const status = root.querySelector('[data-preview-status]');
    if (!status) return;
    status.textContent = message;
    status.dataset.tone = tone;
    status.classList.toggle('dragon-config-feedback-visible', Boolean(message));
}

async function safeApi(url) {
    try { return await api(url); }
    catch (error) { return { ok: false, error: error.message || '服务请求失败' }; }
}

function normalizeSettings(payload = {}) {
    return {
        ...payload,
        training_dir: payload.training_dir || '',
        inference_dir: payload.inference_dir || '',
        custom_dir: payload.custom_dir || '',
        defaults: payload.defaults || { training_dir: 'output/ckpt/sample', inference_dir: 'output/tests', custom_dir: '' },
        error: payload.ok === false ? (payload.error || '读取路径设置失败') : '',
    };
}

function previewImagesUrl(state) {
    const params = new URLSearchParams({ source: state.source, limit: '500', days: state.days || 'all' });
    appendPreviewScope(params, state);
    return `/api/preview/images?${params.toString()}`;
}

function previewWeightsUrl(state) {
    const params = new URLSearchParams();
    appendPreviewScope(params, state);
    const query = params.toString();
    return `/api/preview/weights${query ? `?${query}` : ''}`;
}

function previewDeleteUrl(state) {
    const params = new URLSearchParams();
    if (state.source === 'training' && state.scope === 'task' && state.selectedTaskId) params.set('task_id', state.selectedTaskId);
    const query = params.toString();
    return `/api/preview/images${query ? `?${query}` : ''}`;
}

function appendPreviewScope(params, state) {
    if (state.source !== 'training') return;
    if (state.scope === 'task' && state.selectedTaskId) params.set('task_id', state.selectedTaskId);
    if (state.scope !== 'group' || !state.selectedGroupKey) return;
    const group = state.groups.find((item) => item.key === state.selectedGroupKey);
    if (!group) return;
    params.set('mode', 'config_group');
    if (group.historyGroupKey) params.set('group_key', group.historyGroupKey);
    params.set('methods_subdir', group.methodsSubdir);
    params.set('variant', group.variant);
    params.set('preset', group.preset);
}

function buildPreviewGroups(tasks) {
    const groups = new Map();
    for (const task of tasks) {
        const historyGroupKey = String(task.history_group_key || '').trim();
        const methodsSubdir = String(task.methods_subdir || '').trim();
        const variant = String(task.variant || '').trim();
        const preset = String(task.preset || 'default').trim() || 'default';
        const key = historyGroupKey || `${methodsSubdir}::${variant}::${preset}`;
        if (!key || (!historyGroupKey && (!methodsSubdir || !variant))) continue;
        const current = groups.get(key) || {
            key,
            historyGroupKey,
            methodsSubdir,
            variant,
            preset,
            label: task.history_group_label || task.history_source_config_file || [methodsSubdir, variant, preset].filter(Boolean).join(' / '),
            count: 0,
        };
        current.count += 1;
        groups.set(key, current);
    }
    return Array.from(groups.values()).sort((a, b) => a.label.localeCompare(b.label, 'zh-CN'));
}

function readStoredTrainingContext() {
    try { return JSON.parse(localStorage.getItem('anima_dragon_training_context') || '{}'); }
    catch { return {}; }
}
