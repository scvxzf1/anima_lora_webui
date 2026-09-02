import { escapeHtml } from '../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from './icons.js?v=dragon-ui-20260812v35';
import { loadDatasetPresetLibrary } from './pages/dataset-editor-presets.js?v=dragon-ui-20260824v71';

export function renderDatasetPresetPickerDialog({
    title = '选择数据集',
    description = '选中预设后核对首图与标注，再应用到当前页面。',
    applyLabel = '应用此预设',
} = {}) {
    return `<dialog class="dragon-dataset-picker-dialog" data-dataset-picker-dialog aria-labelledby="dragon-shared-dataset-picker-title">
        <div class="dragon-dataset-picker-shell">
            <div class="dragon-dataset-picker-head">
                <div><span class="dragon-eyebrow">数据集预设库</span><h2 id="dragon-shared-dataset-picker-title">${escapeHtml(title)}</h2><p>${escapeHtml(description)}</p></div>
                <div><button class="dragon-icon-button" type="button" data-dataset-picker-action="refresh" aria-label="刷新数据集预设" title="刷新">${renderIcon('refresh')}</button><button class="dragon-icon-button" type="button" data-dataset-picker-action="close" aria-label="关闭数据集选择" title="关闭">${renderIcon('x')}</button></div>
            </div>
            <div class="dragon-dataset-picker-status" data-dataset-picker-status role="status" aria-live="polite"></div>
            <div class="dragon-dataset-picker-workspace">
                <aside class="dragon-dataset-picker-library" aria-label="数据集预设列表">
                    <label><span class="visually-hidden">搜索数据集预设</span><input class="dragon-input" type="search" autocomplete="off" data-dataset-picker-search placeholder="搜索名称、路径或图片目录…"></label>
                    <div data-dataset-picker-list></div>
                </aside>
                <section class="dragon-dataset-picker-preview" data-dataset-picker-preview aria-label="数据集预览"></section>
            </div>
            <footer class="dragon-dataset-picker-footer">
                <div data-dataset-picker-selection><span>当前选择</span><strong>未选择</strong></div>
                <button class="dragon-btn dragon-btn-primary" type="button" data-dataset-picker-action="apply" disabled>${renderIcon('check', 'dragon-btn-icon')}<span>${escapeHtml(applyLabel)}</span></button>
            </footer>
        </div>
    </dialog>`;
}

export function mountDatasetPresetPicker(root, {
    api,
    getCurrentFile = () => '',
    onApply = () => {},
} = {}) {
    const dialog = root?.querySelector?.('[data-dataset-picker-dialog]');
    if (!dialog || typeof api !== 'function') return null;
    const state = {
        library: { presets: [], groups: [] },
        highlightedFile: '',
        preview: null,
        previewError: '',
        previewLoading: false,
        librarySequence: 0,
        previewSequence: 0,
        openSequence: 0,
        applying: false,
        disposed: false,
    };
    const controller = new AbortController();
    bindPicker(dialog, state, api, onApply, controller.signal);
    return {
        open: async () => {
            const sequence = ++state.openSequence;
            state.highlightedFile = String(getCurrentFile() || '');
            if (!dialog.open) dialog.showModal();
            renderPicker(dialog, state);
            const needsLibrary = !state.library.presets.length;
            if (needsLibrary) await loadLibrary(dialog, state, api, true);
            if (!isCurrentOpen(dialog, state, sequence)) return;
            if (!state.highlightedFile) state.highlightedFile = state.library.presets[0]?.path || '';
            renderPicker(dialog, state);
            if (!needsLibrary && state.highlightedFile) await loadPreview(dialog, state, api);
            if (!isCurrentOpen(dialog, state, sequence)) return;
            dialog.querySelector('[data-dataset-picker-search]')?.focus({ preventScroll: true });
        },
        dispose: () => {
            state.disposed = true;
            controller.abort();
            if (dialog.open) dialog.close('dispose');
        },
    };
}

function bindPicker(dialog, state, api, onApply, signal) {
    const options = { signal };
    dialog.addEventListener('click', (event) => {
        if (event.target === dialog) return dialog.close('cancel');
        const target = event.target.closest?.('[data-dataset-picker-action], [data-dataset-picker-file]');
        if (!target) return;
        const action = target.dataset.datasetPickerAction;
        if (action === 'close') return dialog.close('cancel');
        if (action === 'refresh') return run(() => loadLibrary(dialog, state, api, true));
        if (action === 'apply' && !state.applying) return run(async () => {
            const preset = highlightedPreset(state);
            if (!preset) return;
            state.applying = true;
            renderPicker(dialog, state);
            try {
                const applied = await onApply(preset.path, preset);
                if (applied !== false && !state.disposed && dialog.open) dialog.close('apply');
            } catch (error) {
                if (!state.disposed) setStatus(dialog, error.message || '应用数据集预设失败', 'error');
            } finally {
                state.applying = false;
                if (!state.disposed && dialog.open) renderPicker(dialog, state);
            }
        });
        if (target.dataset.datasetPickerFile) {
            state.highlightedFile = target.dataset.datasetPickerFile;
            state.preview = null;
            state.previewError = '';
            renderPicker(dialog, state);
            return run(() => loadPreview(dialog, state, api));
        }
    }, options);
    dialog.addEventListener('close', () => { state.openSequence += 1; }, options);
    dialog.querySelector('[data-dataset-picker-search]')?.addEventListener('input', () => renderList(dialog, state), options);
}

async function loadLibrary(dialog, state, api, preserveSelection = false) {
    const sequence = ++state.librarySequence;
    setStatus(dialog, '正在读取数据集预设…', 'info');
    try {
        const library = await loadDatasetPresetLibrary(api);
        if (state.disposed || sequence !== state.librarySequence) return;
        const previousExists = library.presets.some((item) => item.path === state.highlightedFile);
        state.library = library;
        if (!preserveSelection || !previousExists) state.highlightedFile = library.presets[0]?.path || '';
        setStatus(dialog, '', 'info');
        renderPicker(dialog, state);
        if (state.highlightedFile) await loadPreview(dialog, state, api);
    } catch (error) {
        if (!state.disposed && sequence === state.librarySequence) setStatus(dialog, error.message || '读取数据集预设失败', 'error');
    }
}

async function loadPreview(dialog, state, api) {
    const file = state.highlightedFile;
    if (!file) return;
    const sequence = ++state.previewSequence;
    state.previewLoading = true;
    state.previewError = '';
    renderPreview(dialog, state);
    try {
        const params = new URLSearchParams({ file, dataset_index: '0', source: 'source', limit: '1' });
        const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
        if (payload?.ok === false) throw new Error(payload.error || '读取数据集预览失败');
        if (state.disposed || sequence !== state.previewSequence) return;
        state.preview = payload;
    } catch (error) {
        if (state.disposed || sequence !== state.previewSequence) return;
        state.preview = null;
        state.previewError = error.message || '读取数据集预览失败';
    } finally {
        if (state.disposed || sequence !== state.previewSequence) return;
        state.previewLoading = false;
        renderPicker(dialog, state);
    }
}

function renderPicker(dialog, state) {
    renderList(dialog, state);
    renderPreview(dialog, state);
    const preset = highlightedPreset(state);
    const selection = dialog.querySelector('[data-dataset-picker-selection]');
    if (selection) selection.innerHTML = `<span>当前选择</span><strong>${escapeHtml(preset?.label || preset?.filename || '未选择')}</strong>`;
    const apply = dialog.querySelector('[data-dataset-picker-action="apply"]');
    if (apply) {
        apply.disabled = !preset || state.applying;
        apply.setAttribute('aria-busy', state.applying ? 'true' : 'false');
    }
}

function renderList(dialog, state) {
    const list = dialog.querySelector('[data-dataset-picker-list]');
    if (!list) return;
    const query = dialog.querySelector('[data-dataset-picker-search]')?.value || '';
    const groups = groupPresets(state.library, query);
    list.innerHTML = groups.length ? groups.map((group) => `
        <section class="dragon-dataset-picker-group">
            <div><strong>${escapeHtml(group.label || group.id || '数据集预设')}</strong><span>${group.files.length}</span></div>
            ${group.files.map((preset) => renderOption(preset, preset.path === state.highlightedFile)).join('')}
        </section>`).join('') : '<div class="dragon-dataset-picker-empty">没有匹配的数据集预设</div>';
}

function renderOption(preset, active) {
    const summary = preset.summary || {};
    return `<button class="dragon-dataset-picker-item" type="button" data-dataset-picker-file="${escapeHtml(preset.path)}" data-active="${active}" aria-pressed="${active}"><span><strong>${escapeHtml(preset.label || preset.filename || preset.path)}</strong><small>${escapeHtml(preset.path)}</small></span><em>${Number(summary.dataset_count || 0)}组 · ${Number(summary.repeat_total || 0)} Reps</em></button>`;
}

function renderPreview(dialog, state) {
    const host = dialog.querySelector('[data-dataset-picker-preview]');
    if (!host) return;
    const preset = highlightedPreset(state);
    if (!preset) return void (host.innerHTML = '<div class="dragon-dataset-picker-empty"><strong>请选择数据集预设</strong></div>');
    if (state.previewLoading) return void (host.innerHTML = '<div class="dragon-dataset-picker-empty"><strong>正在读取首图与标注…</strong></div>');
    if (state.previewError) return void (host.innerHTML = `<div class="dragon-dataset-picker-empty" data-tone="error"><strong>预览读取失败</strong><span>${escapeHtml(state.previewError)}</span></div>`);
    host.innerHTML = renderPreviewPayload(preset, state.preview);
}

function renderPreviewPayload(preset, payload) {
    const summary = preset.summary || {};
    const image = Array.isArray(payload?.images) ? payload.images[0] : null;
    const caption = image?.caption || {};
    return `<div class="dragon-dataset-picker-preview-head"><div><span class="dragon-eyebrow">第 1 组首图</span><h3>${escapeHtml(preset.label || preset.filename || preset.path)}</h3></div><span>${Number(summary.dataset_count || 0)} 组 · ${Number(summary.repeat_total || 0)} Reps</span></div><code title="${escapeHtml(summary.source_dir || preset.path)}">${escapeHtml(summary.source_dir || preset.path)}</code><div class="dragon-dataset-picker-image" data-empty="${image ? 'false' : 'true'}">${image ? `<img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.name || '数据集首图')}" width="640" height="420">` : '<span>当前目录没有可预览图片</span>'}</div><div class="dragon-dataset-picker-caption" data-missing="${caption.ok ? 'false' : 'true'}"><span>${caption.ok ? `标注 · ${escapeHtml(caption.format_label || caption.extension || '')}` : '未找到标注'}</span><p>${escapeHtml(caption.ok ? (caption.text || '(空标注)') : (payload?.caption_summary || '当前标注来源没有对应 caption'))}</p></div>`;
}

function groupPresets(library, query) {
    const needle = String(query || '').trim().toLocaleLowerCase();
    const byPath = new Map((library.presets || []).map((preset) => [preset.path, preset]));
    const matches = (preset) => !needle || [preset.label, preset.filename, preset.path, preset.summary?.source_dir, preset.summary?.image_dir].some((value) => String(value || '').toLocaleLowerCase().includes(needle));
    const groups = library.groups?.length ? library.groups : [{ id: 'all', label: '全部预设', files: library.presets || [] }];
    return groups.map((group) => ({ ...group, files: (group.files || []).map((item) => byPath.get(item.path) || item).filter(matches) })).filter((group) => group.files.length);
}

function highlightedPreset(state) {
    return state.library.presets.find((item) => item.path === state.highlightedFile) || null;
}

function isCurrentOpen(dialog, state, sequence) {
    return !state.disposed && dialog.open && state.openSequence === sequence;
}

function setStatus(dialog, message, tone) {
    const node = dialog.querySelector('[data-dataset-picker-status]');
    if (!node) return;
    node.textContent = message;
    node.dataset.tone = tone;
    node.hidden = !message;
}

function run(fn) {
    Promise.resolve().then(fn).catch((error) => console.error('[dragon-dataset-picker]', error));
}
