/* Training configuration preset library for Dragon config workspaces.
 * Manages user groups and persists cross-group / in-group file placement.
 */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { DRAGON_VIEWPORT_QUERIES, matchesDragonViewport } from '../responsive.js?v=dragon-ui-20260824v1';
import { loadTrainingContext, selectTrainingConfigFile } from './training-controls.js?v=dragon-ui-20260816v67';

const api = createApiClient();
const HIDDEN_TRAINING_GROUP_IDS = new Set(['gui_methods', 'presets']);

function visibleTrainingGroups(groups) {
    return (Array.isArray(groups) ? groups : []).filter((group) => (
        !HIDDEN_TRAINING_GROUP_IDS.has(group.id) && group.methods_subdir !== 'gui-methods'
    ));
}

export function renderTrainingPresetLibrary(context) {
    return `<aside class="dragon-training-preset-library" aria-label="训练配置预设管理" data-training-preset-library>
        ${renderLibraryContent(context)}
    </aside>`;
}

export function bindTrainingPresetLibrary(root, context, {
    beforeContextChange,
    onConfigFileChange,
    onSaveChanges,
} = {}) {
    const library = root.querySelector('[data-training-preset-library]');
    if (!library) return null;
    const state = { context, groups: context.groups || [], draggedFile: '', onConfigFileChange, onSaveChanges };
    const viewportLayout = bindTrainingPresetViewport(library);
    bindLibraryEvents(library, state, beforeContextChange);
    return {
        updateContext(nextContext) {
            state.context = {
                ...nextContext,
                groups: state.groups,
                files: state.context.files,
            };
            const scrollTop = library.querySelector('[data-training-preset-groups]')?.scrollTop || 0;
            library.innerHTML = renderLibraryContent(state.context);
            bindLibraryEvents(library, state, beforeContextChange);
            const groups = library.querySelector('[data-training-preset-groups]');
            if (groups) groups.scrollTop = scrollTop;
            viewportLayout.schedule();
        },
        destroy: () => viewportLayout.destroy(),
    };
}

function bindTrainingPresetViewport(library) {
    let frame = 0;
    let destroyed = false;
    const sync = () => {
        frame = 0;
        if (!matchesDragonViewport(DRAGON_VIEWPORT_QUERIES.trainingPresetSidebar)) {
            library.style.removeProperty('--dragon-training-preset-height');
            return;
        }
        const top = Math.max(0, library.getBoundingClientRect().top);
        const available = Math.max(240, window.innerHeight - top - 16);
        library.style.setProperty('--dragon-training-preset-height', `${available}px`);
    };
    const schedule = () => {
        if (destroyed) return;
        if (!frame) frame = window.requestAnimationFrame(sync);
    };
    window.addEventListener('resize', schedule, { passive: true });
    window.addEventListener('scroll', schedule, { passive: true });
    document.fonts?.ready.then(schedule).catch(() => {});
    schedule();
    return {
        schedule,
        destroy() {
            destroyed = true;
            if (frame) window.cancelAnimationFrame(frame);
            window.removeEventListener('resize', schedule);
            window.removeEventListener('scroll', schedule);
            library.style.removeProperty('--dragon-training-preset-height');
        },
    };
}

function renderLibraryContent(context) {
    const allGroups = Array.isArray(context.groups) ? context.groups : [];
    const groups = visibleTrainingGroups(allGroups);
    const files = groups.flatMap((group) => group.files || []);
    const selected = allGroups.flatMap((group) => group.files || [])
        .find((file) => file.path === context.configFile);
    return `<div class="dragon-training-preset-head">
            <div><span class="dragon-eyebrow">预设库</span><h2>训练配置</h2></div>
            <span>${files.length} 个配置</span>
        </div>
        <section class="dragon-training-current-preset" aria-label="当前训练配置">
            <span>当前文件</span>
            <strong>${escapeHtml(selected?.label || selected?.filename || selected?.path || '未选择')}</strong>
            <small>${escapeHtml(selected?.path || context.configFile || '—')}</small>
            <div><em>${selected?.trainable ? '可训练' : '配置文件'}</em>${selected?.readonly || selected?.locked ? '<em data-tone="locked">系统只读</em>' : ''}</div>
        </section>
        <div class="dragon-training-preset-toolbar">
            <div class="dragon-training-preset-toolbar-actions">
                <button class="dragon-btn dragon-btn-primary dragon-btn-sm" type="button" data-training-preset-action="save-updates" ${context.configFile && !selected?.readonly && !selected?.locked ? '' : 'disabled'}>${renderIcon('check', 'dragon-btn-icon')}<span>保存更新</span></button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-training-preset-action="new-group">${renderIcon('folder', 'dragon-btn-icon')}<span>新建分组</span></button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-training-preset-action="import">${renderIcon('upload', 'dragon-btn-icon')}<span>导入配置</span></button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-training-preset-action="save-as" ${context.configFile ? '' : 'disabled'}>${renderIcon('copy', 'dragon-btn-icon')}<span>另存为</span></button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-training-preset-action="export" ${context.configFile ? '' : 'disabled'}>${renderIcon('download', 'dragon-btn-icon')}<span>导出当前</span></button>
            </div>
            <button class="dragon-icon-button" type="button" data-training-preset-action="refresh" aria-label="刷新训练配置预设">${renderIcon('refresh')}</button>
            <input type="file" accept=".toml,text/plain,application/toml" data-training-preset-import-file hidden>
        </div>
        <p class="dragon-training-preset-feedback" data-training-preset-feedback role="status" aria-live="polite"></p>
        <p class="dragon-training-preset-drop-status" data-training-preset-drop-status aria-live="polite"></p>
        <div class="dragon-training-preset-groups" data-training-preset-groups>${renderGroups(groups, context.configFile)}</div>`;
}

function renderGroups(groups, selectedFile) {
    if (!groups.length) return '<div class="dragon-training-preset-empty">没有可用的训练配置分组。</div>';
    return groups.map((group) => {
        const files = group.files || [];
        return `<section class="dragon-training-preset-group" data-training-preset-group="${escapeHtml(group.id || '')}" data-training-preset-drop-group="${escapeHtml(group.id || '')}">
            <header>
                <div><strong>${escapeHtml(group.label || group.id || '配置分组')}</strong><span>${files.length}</span>${group.locked || group.system_locked ? '<em>只读</em>' : ''}</div>
                <div class="dragon-training-preset-group-actions">
                    ${(group.files || []).length ? `<a href="/api/config/file-groups/${encodeURIComponent(group.id)}/export?kind=training" download aria-label="导出分组 ${escapeHtml(group.label || group.id)}" title="导出分组 ZIP">${renderIcon('download')}</a>` : ''}
                    ${group.renamable ? `<button type="button" data-training-group-action="rename" data-group-id="${escapeHtml(group.id)}" aria-label="重命名 ${escapeHtml(group.label || group.id)}">${renderIcon('edit')}</button>` : ''}
                    ${group.deletable ? `<button type="button" data-training-group-action="delete" data-group-id="${escapeHtml(group.id)}" aria-label="删除 ${escapeHtml(group.label || group.id)}">${renderIcon('trash')}</button>` : ''}
                </div>
            </header>
            <div class="dragon-training-preset-list">
                ${files.map((file) => renderPresetRow(file, group, selectedFile)).join('')}
                <div class="dragon-training-preset-dropzone" data-training-preset-dropzone="${escapeHtml(group.id || '')}" data-empty="${files.length ? 'false' : 'true'}">${files.length ? '拖到此组末尾' : '空分组，可拖到此处'}</div>
            </div>
        </section>`;
    }).join('');
}

function renderPresetRow(file, group, selectedFile) {
    const movable = canMoveFromGroup(file, group);
    const name = file.label || file.filename || file.path || '未命名配置';
    return `<div class="dragon-training-preset-row" data-training-preset-row="${escapeHtml(file.path || '')}" data-selected="${file.path === selectedFile}" data-readonly="${Boolean(file.readonly || file.locked)}" draggable="${movable}">
        <span class="dragon-training-preset-drag-handle" aria-hidden="true">${renderIcon('grip')}</span>
        <button type="button" data-training-preset-select="${escapeHtml(file.path || '')}" ${file.trainable ? '' : 'disabled aria-disabled="true"'}>
            <strong>${escapeHtml(name)}</strong>
            <small>${escapeHtml(file.path || '')}</small>
        </button>
    </div>`;
}

function bindLibraryEvents(library, state, beforeContextChange) {
    library.querySelector('[data-training-preset-action="refresh"]')?.addEventListener('click', () => refreshLibrary(library, state, beforeContextChange));
    library.querySelector('[data-training-preset-action="save-updates"]')?.addEventListener('click', () => saveCurrentUpdates(library, state));
    library.querySelector('[data-training-preset-action="new-group"]')?.addEventListener('click', () => createGroup(library, state, beforeContextChange));
    const importInput = library.querySelector('[data-training-preset-import-file]');
    library.querySelector('[data-training-preset-action="import"]')?.addEventListener('click', () => {
        if (beforeContextChange && beforeContextChange() === false) return;
        importInput?.click();
    });
    importInput?.addEventListener('change', () => importTrainingConfig(library, state, importInput, beforeContextChange));
    library.querySelector('[data-training-preset-action="save-as"]')?.addEventListener('click', () => saveCurrentConfigAs(library, state, beforeContextChange));
    library.querySelector('[data-training-preset-action="export"]')?.addEventListener('click', () => exportCurrentConfig(library, state));
    library.querySelectorAll('[data-training-preset-select]').forEach((button) => button.addEventListener('click', () => {
        const file = state.context.files.find((item) => item.path === button.dataset.trainingPresetSelect);
        if (!file || file.path === state.context.configFile) return;
        if (beforeContextChange && beforeContextChange() === false) return;
        activateConfigFile(state, file.path);
    }));
    library.querySelectorAll('[data-training-group-action]').forEach((button) => button.addEventListener('click', () => {
        const group = state.groups.find((item) => item.id === button.dataset.groupId);
        if (!group) return;
        if (button.dataset.trainingGroupAction === 'rename') renameGroup(library, state, group, beforeContextChange);
        if (button.dataset.trainingGroupAction === 'delete') deleteGroup(library, state, group, beforeContextChange);
    }));
    bindDragAndDrop(library, state, beforeContextChange);
}

async function saveCurrentUpdates(library, state) {
    const button = library.querySelector('[data-training-preset-action="save-updates"]');
    if (!button || !state.onSaveChanges) return;
    button.disabled = true;
    setFeedback(library, '正在保存当前配置…');
    try {
        const saved = await state.onSaveChanges();
        if (saved === false) throw new Error('未能保存当前配置，请检查编辑区提示');
        setFeedback(library, '当前配置已更新');
    } catch (error) {
        setFeedback(library, error.message || '保存当前配置失败', true);
    } finally {
        button.disabled = false;
    }
}

function bindDragAndDrop(library, state, beforeContextChange) {
    library.querySelectorAll('[data-training-preset-row][draggable="true"]').forEach((row) => {
        row.addEventListener('dragstart', (event) => {
            state.draggedFile = row.dataset.trainingPresetRow || '';
            library.dataset.dragging = 'true';
            row.dataset.dragging = 'true';
            event.dataTransfer?.setData('text/plain', state.draggedFile);
            if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
        });
        row.addEventListener('dragend', () => clearDropPreview(library, state));
        row.addEventListener('dragover', (event) => {
            const file = state.draggedFile;
            const groupId = row.closest('[data-training-preset-group]')?.dataset.trainingPresetGroup || '';
            if (!file || row.dataset.trainingPresetRow === file || !canDropInto(state, file, groupId)) return;
            event.preventDefault();
            event.stopPropagation();
            const rect = row.getBoundingClientRect();
            const position = event.clientY < rect.top + rect.height / 2 ? 'before' : 'after';
            showRowDropPreview(library, row, groupId, position, state);
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        });
        row.addEventListener('drop', async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const file = state.draggedFile || event.dataTransfer?.getData('text/plain') || '';
            const groupId = row.closest('[data-training-preset-group]')?.dataset.trainingPresetGroup || '';
            const position = row.dataset.trainingPresetDropPosition || 'after';
            if (!canDropInto(state, file, groupId) || row.dataset.trainingPresetRow === file) return clearDropPreview(library, state);
            const order = targetOrder(state, file, groupId, row.dataset.trainingPresetRow || '', position);
            await placeFile(library, state, file, groupId, order, beforeContextChange);
        });
    });
    library.querySelectorAll('[data-training-preset-dropzone]').forEach((zone) => {
        zone.addEventListener('dragover', (event) => {
            const groupId = zone.dataset.trainingPresetDropzone || '';
            if (!canDropInto(state, state.draggedFile, groupId)) return;
            event.preventDefault();
            event.stopPropagation();
            clearDropMarkers(library);
            zone.dataset.over = 'true';
            setDropStatus(library, `将放置到「${groupLabel(state, groupId)}」末尾`);
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        });
        zone.addEventListener('drop', async (event) => {
            event.preventDefault();
            event.stopPropagation();
            const file = state.draggedFile || event.dataTransfer?.getData('text/plain') || '';
            const groupId = zone.dataset.trainingPresetDropzone || '';
            if (!canDropInto(state, file, groupId)) return clearDropPreview(library, state);
            await placeFile(library, state, file, groupId, targetOrder(state, file, groupId), beforeContextChange);
        });
    });
}

function showRowDropPreview(library, row, groupId, position, state) {
    clearDropMarkers(library);
    row.dataset.trainingPresetDropPosition = position;
    row.classList.add(`dragon-training-preset-drop-${position}`);
    const target = row.querySelector('strong')?.textContent || '目标配置';
    setDropStatus(library, `将放置到「${groupLabel(state, groupId)}」中，位于「${target}」${position === 'before' ? '之前' : '之后'}`);
}

function targetOrder(state, file, groupId, anchor = '', position = 'after') {
    const group = state.groups.find((item) => item.id === groupId);
    const paths = (group?.files || []).map((item) => item.path).filter((path) => path && path !== file);
    if (!anchor) return [...paths, file];
    const anchorIndex = Math.max(0, paths.indexOf(anchor));
    paths.splice(anchorIndex + (position === 'after' ? 1 : 0), 0, file);
    return paths;
}

async function placeFile(library, state, file, groupId, order, beforeContextChange) {
    try {
        setFeedback(library, '正在更新配置位置…');
        const payload = await api('/api/config/file-groups/place', {
            method: 'POST',
            body: JSON.stringify({ target: 'file', file, group: groupId, order }),
        });
        if (payload.ok === false) throw new Error(payload.error || '配置位置更新失败');
        await refreshLibrary(library, state, beforeContextChange, payload.message || '配置位置已更新');
    } catch (error) {
        setFeedback(library, error.message || '配置位置更新失败', true);
        clearDropPreview(library, state);
    }
}


async function exportCurrentConfig(library, state) {
    const file = state.context.configFile;
    if (!file) return;
    try {
        setFeedback(library, '正在导出当前配置…');
        const payload = await api(`/api/config/raw?file=${encodeURIComponent(file)}`);
        if (payload.ok === false) throw new Error(payload.error || '读取训练配置失败');
        downloadTextFile(payload.content || '', file.split('/').pop() || 'training-config.toml');
        setFeedback(library, '当前训练配置已导出');
    } catch (error) {
        setFeedback(library, error.message || '导出训练配置失败', true);
    }
}

async function importTrainingConfig(library, state, input, beforeContextChange) {
    const source = input.files?.[0];
    input.value = '';
    if (!source) return;
    try {
        if (!/\.toml$/i.test(source.name)) throw new Error('只能导入 TOML 配置文件');
        const target = promptImportedConfigPath('导入配置名称', source.name);
        if (!target) return;
        const content = await source.text();
        await saveNewTrainingConfig(library, state, target, content, beforeContextChange, '配置已导入');
    } catch (error) {
        setFeedback(library, error.message || '导入训练配置失败', true);
    }
}

async function saveCurrentConfigAs(library, state, beforeContextChange) {
    const source = state.context.configFile;
    if (!source) return;
    if (beforeContextChange && beforeContextChange() === false) return;
    try {
        const target = promptImportedConfigPath('另存为配置名称', defaultCopyFilename(source));
        if (!target) return;
        setFeedback(library, '正在读取当前配置…');
        const payload = await api(`/api/config/raw?file=${encodeURIComponent(source)}`);
        if (payload.ok === false) throw new Error(payload.error || '读取训练配置失败');
        await saveNewTrainingConfig(library, state, target, payload.content || '', beforeContextChange, '配置已另存并切换');
    } catch (error) {
        setFeedback(library, error.message || '另存训练配置失败', true);
    }
}

async function saveNewTrainingConfig(library, state, file, content, beforeContextChange, successMessage) {
    setFeedback(library, '正在保存新配置…');
    const payload = await api('/api/config/raw/save-as', {
        method: 'POST',
        body: JSON.stringify({ file, content }),
    });
    if (payload.ok === false) throw new Error(payload.error || '保存训练配置失败');
    await refreshLibrary(library, state, beforeContextChange);
    if (!activateConfigFile(state, file)) throw new Error('新配置已保存，但没有出现在预设库中，请刷新后重试');
    setFeedback(library, payload.message || successMessage);
}

function activateConfigFile(state, path) {
    const file = (state.context.files || []).find((item) => item.path === path);
    if (!file) return false;
    const nextContext = selectTrainingConfigFile(state.context, file, { notify: false, persist: false });
    if (!nextContext) return false;
    if (state.onConfigFileChange) state.onConfigFileChange(file, nextContext);
    else {
        state.context = nextContext;
        window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
    }
    return true;
}

function promptImportedConfigPath(label, defaultFilename) {
    const answer = window.prompt(label, defaultFilename);
    if (answer === null) return '';
    return `configs/imported/${normalizeImportedFilename(answer)}`;
}

function normalizeImportedFilename(value) {
    const basename = String(value || '').trim().split(/[\\/]/).pop()?.trim() || '';
    const stem = basename.replace(/\.toml$/i, '').trim();
    if (!stem || stem === '.' || stem === '..') throw new Error('请输入有效的配置文件名称');
    return `${stem}.toml`;
}

function defaultCopyFilename(path) {
    const basename = String(path || '').split('/').pop() || 'training-config.toml';
    const stem = basename.replace(/\.toml$/i, '') || 'training-config';
    return `${stem}-副本.toml`;
}

function downloadTextFile(content, filename) {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

async function createGroup(library, state, beforeContextChange) {
    const label = window.prompt('请输入新的训练配置分组名称');
    if (!label?.trim()) return;
    await mutateGroup(library, state, '/api/config/file-groups', { method: 'POST', body: JSON.stringify({ label: label.trim(), kind: 'training' }) }, beforeContextChange);
}

async function renameGroup(library, state, group, beforeContextChange) {
    const label = window.prompt('请输入新的分组名称', group.label || group.id);
    if (!label?.trim() || label.trim() === group.label) return;
    await mutateGroup(library, state, `/api/config/file-groups/${encodeURIComponent(group.id)}`, { method: 'PATCH', body: JSON.stringify({ label: label.trim() }) }, beforeContextChange);
}

async function deleteGroup(library, state, group, beforeContextChange) {
    if (!window.confirm(`确认删除分组“${group.label || group.id}”吗？配置文件不会被删除。`)) return;
    await mutateGroup(library, state, `/api/config/file-groups/${encodeURIComponent(group.id)}`, { method: 'DELETE' }, beforeContextChange);
}

async function mutateGroup(library, state, url, options, beforeContextChange) {
    try {
        const payload = await api(url, options);
        if (payload.ok === false) throw new Error(payload.error || '分组操作失败');
        await refreshLibrary(library, state, beforeContextChange, payload.message || '分组已更新');
    } catch (error) {
        setFeedback(library, error.message || '分组操作失败', true);
    }
}

async function refreshLibrary(library, state, beforeContextChange, message = '') {
    const inventory = await loadTrainingContext({ refresh: true, includeGpus: false });
    state.groups = inventory.groups || [];
    state.context = {
        ...inventory,
        configFile: state.context.configFile,
        variant: state.context.variant,
        methodsSubdir: state.context.methodsSubdir,
        preset: state.context.preset,
        groups: state.groups,
        files: inventory.files || [],
    };
    library.innerHTML = renderLibraryContent(state.context);
    bindLibraryEvents(library, state, beforeContextChange);
    if (message) setFeedback(library, message);
    clearDropPreview(library, state);
}

function canMoveFromGroup(file, group) {
    return Boolean(file?.path && !file.readonly && !file.locked && group?.movable && !group.locked && !group.group_locked && !group.user_group_locked && !group.system_locked);
}

function canDropInto(state, file, groupId) {
    const preset = state.context.files.find((item) => item.path === file);
    const target = state.groups.find((item) => item.id === groupId);
    const source = state.groups.find((group) => (group.files || []).some((item) => item.path === file));
    return Boolean(preset && target?.kind === 'training' && target.movable && !target.locked && !target.group_locked && !target.user_group_locked && !target.system_locked && (!source || canMoveFromGroup(preset, source)));
}

function groupLabel(state, groupId) {
    const group = state.groups.find((item) => item.id === groupId);
    return group?.label || groupId || '目标分组';
}

function setDropStatus(library, message) {
    const status = library.querySelector('[data-training-preset-drop-status]');
    if (status) status.textContent = message;
}

function setFeedback(library, message, error = false) {
    const node = library.querySelector('[data-training-preset-feedback]');
    if (!node) return;
    node.textContent = message;
    node.dataset.tone = error ? 'error' : 'success';
}

function clearDropMarkers(library) {
    library.querySelectorAll('.dragon-training-preset-drop-before, .dragon-training-preset-drop-after').forEach((node) => {
        node.classList.remove('dragon-training-preset-drop-before', 'dragon-training-preset-drop-after');
        delete node.dataset.trainingPresetDropPosition;
    });
    library.querySelectorAll('[data-training-preset-dropzone][data-over="true"]').forEach((node) => { node.dataset.over = 'false'; });
}

function clearDropPreview(library, state) {
    state.draggedFile = '';
    delete library.dataset.dragging;
    library.querySelectorAll('[data-dragging="true"]').forEach((node) => delete node.dataset.dragging);
    clearDropMarkers(library);
    setDropStatus(library, '');
}
