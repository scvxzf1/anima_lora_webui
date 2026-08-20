/* Preset library rendering and mutations for the Dragon dataset workspace. */

import { renderIcon } from '../icons.js?v=dragon-ui-20260814v43';
import { escapeAttribute, escapeHtml } from './dataset-editor-fields.js?v=dragon-ui-20260814v43';

export async function loadDatasetPresetLibrary(api) {
    const payload = await api('/api/config/dataset-presets');
    if (payload.ok === false) throw new Error(payload.error || '读取数据集预设失败');
    return {
        presets: Array.isArray(payload.presets) ? payload.presets : [],
        groups: Array.isArray(payload.groups) ? payload.groups : [],
    };
}

export function renderDatasetPresetLibrary(state) {
    return `
        <aside class="dragon-dataset-library" aria-label="数据集预设库">
            <div class="dragon-dataset-library-head">
                <div><span class="dragon-eyebrow">预设库</span><h2>数据集配置</h2></div>
                <button class="dragon-icon-button" type="button" data-preset-action="refresh" aria-label="刷新数据集预设">${renderIcon('refresh')}</button>
            </div>
            <label class="dragon-dataset-search"><span class="visually-hidden">搜索数据集预设</span><input class="dragon-input" type="search" name="dataset_preset_search" autocomplete="off" placeholder="搜索名称或路径…" value="${escapeAttribute(state.search || '')}" data-preset-search></label>
            <div class="dragon-dataset-library-actions">
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-preset-action="new">${renderIcon('filePlus', 'dragon-btn-icon')}<span>新建</span></button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-preset-action="import">${renderIcon('upload', 'dragon-btn-icon')}<span>导入</span></button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-preset-action="export" ${state.selectedFile ? '' : 'disabled'}>${renderIcon('download', 'dragon-btn-icon')}<span>导出</span></button>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm dragon-dataset-new-group" type="button" data-preset-action="new-group">${renderIcon('folder', 'dragon-btn-icon')}<span>新建预设分组</span></button>
                <input type="file" name="dataset_preset_import" accept=".toml,text/plain" data-preset-import hidden>
            </div>
            <div class="dragon-dataset-library-meta"><span>${state.presets.length} 个预设</span><span>${state.groups.length} 个分组</span></div>
            <div class="dragon-dataset-preset-list" data-preset-list role="region" aria-label="数据集配置列表" tabindex="0">${renderPresetGroups(state)}</div>
        </aside>
    `;
}

export function renderPresetGroups(state) {
    const keyword = String(state.search || '').trim().toLocaleLowerCase();
    const presetByPath = new Map(state.presets.map((item) => [item.path, item]));
    const sourceGroups = state.groups.length ? state.groups : [{ id: 'all', label: '全部预设', files: state.presets }];
    const groups = sourceGroups.map((group) => ({
        ...group,
        files: (group.files || []).map((item) => presetByPath.get(item.path) || item).filter((item) => presetMatches(item, keyword)),
    })).filter((group) => group.files.length || (!keyword && group.kind === 'dataset'));
    if (!groups.length) return `<div class="dragon-dataset-library-empty"><strong>${keyword ? '没有匹配的预设' : '还没有数据集预设'}</strong><span>${keyword ? '尝试搜索名称、文件路径或训练目录。' : '新建或导入一个 TOML 数据集配置。'}</span></div>`;
    return groups.map((group) => `
        <section class="dragon-dataset-preset-group" data-preset-group="${escapeAttribute(group.id || '')}" data-preset-drop-group="${escapeAttribute(group.id || '')}">
            <header>
                <div class="dragon-dataset-preset-group-title"><span title="${escapeAttribute(group.label || group.id || '数据集配置')}">${escapeHtml(group.label || group.id || '数据集配置')}</span><small>${group.files.length}</small>${renderGroupBadge(group)}</div>
                ${renderGroupActions(group)}
            </header>
            <div>
                ${group.files.map((preset) => renderPresetItem(preset, state.selectedFile, state.search)).join('')}
                <div class="dragon-dataset-preset-dropzone" data-preset-dropzone="${escapeAttribute(group.id || '')}" data-empty="${group.files.length ? 'false' : 'true'}">${group.files.length ? '拖到此组末尾' : '空分组，可拖到此处'}</div>
            </div>
        </section>
    `).join('');
}

function renderGroupBadge(group) {
    if (group.user_group_locked) return '<em>分组锁定</em>';
    if (group.locked || group.system_locked) return '<em>只读</em>';
    return '';
}

function renderGroupActions(group) {
    const actions = [];
    if ((group.files || []).length) {
        const href = `/api/config/file-groups/${encodeURIComponent(group.id)}/export?kind=dataset`;
        actions.push(`<a class="dragon-dataset-group-action" href="${escapeAttribute(href)}" download aria-label="导出分组 ${escapeAttribute(group.label || group.id)}" title="导出分组 ZIP">${renderIcon('download')}</a>`);
    }
    if (group.renamable) {
        actions.push(`<button class="dragon-dataset-group-action" type="button" data-preset-group-action="rename" data-group-id="${escapeAttribute(group.id)}" aria-label="重命名分组 ${escapeAttribute(group.label || group.id)}" title="重命名分组">${renderIcon('edit')}</button>`);
    }
    if (group.deletable) {
        actions.push(`<button class="dragon-dataset-group-action dragon-dataset-group-action-danger" type="button" data-preset-group-action="delete" data-group-id="${escapeAttribute(group.id)}" aria-label="删除分组 ${escapeAttribute(group.label || group.id)}" title="只删除分组，不删除其中的 TOML">${renderIcon('trash')}</button>`);
    }
    return actions.length ? `<div class="dragon-dataset-preset-group-actions">${actions.join('')}</div>` : '';
}

function presetMatches(preset, keyword) {
    if (!keyword) return true;
    const summary = preset.summary || {};
    return [preset.label, preset.filename, preset.path, summary.source_dir, summary.image_dir].some((value) => String(value || '').toLocaleLowerCase().includes(keyword));
}

function renderPresetItem(preset, selectedFile, search = '') {
    const summary = preset.summary || {};
    const active = preset.path === selectedFile;
    const draggable = !preset.readonly && !String(search || '').trim();
    return `
        <div class="dragon-dataset-preset-row dragon-dataset-preset-item" role="button" tabindex="0" data-preset-row="${escapeAttribute(preset.path)}" data-preset-file="${escapeAttribute(preset.path)}" data-preset-drag-source data-active="${active}" ${active ? 'aria-current="true"' : ''}>
            <span class="dragon-dataset-preset-drag-handle" draggable="${draggable ? 'true' : 'false'}" aria-hidden="true" title="拖动调整位置">${renderIcon('grip')}</span>
            <span><strong>${escapeHtml(preset.label || preset.filename || preset.path)}</strong><small title="${escapeAttribute(preset.path)}">${escapeHtml(preset.path)}</small></span>
            <span class="dragon-dataset-preset-item-meta"><em>${Number(summary.dataset_count || 0)} 组</em><em>重复 ${Number(summary.repeat_total || 0)}</em>${preset.readonly ? '<em>只读</em>' : ''}</span>
        </div>
    `;
}

export async function readDatasetPreset(api, file) {
    const payload = await api(`/api/config/dataset-presets/read?file=${encodeURIComponent(file)}`);
    if (payload.ok === false) throw new Error(payload.error || '读取数据集预设失败');
    return payload;
}

export async function saveDatasetPreset(api, payload) {
    const result = await api('/api/config/dataset-presets', { method: 'PUT', body: JSON.stringify(payload) });
    if (result.ok === false) throw new Error(result.error || '保存数据集预设失败');
    return result;
}

export async function saveDatasetPresetAs(api, payload) {
    const result = await api('/api/config/dataset-presets/save-as', { method: 'POST', body: JSON.stringify(payload) });
    if (result.ok === false) throw new Error(result.error || '另存数据集预设失败');
    return result;
}

export async function applyDatasetPreset(api, file, trainFile) {
    const result = await api('/api/config/dataset-presets/apply', {
        method: 'POST',
        body: JSON.stringify({ dataset_file: file, train_file: trainFile }),
    });
    if (result.ok === false) throw new Error(result.error || '应用数据集预设失败');
    return result;
}

export async function deleteDatasetPreset(api, file) {
    const result = await api(`/api/config/dataset-presets?file=${encodeURIComponent(file)}`, { method: 'DELETE' });
    if (result.ok === false) throw new Error(result.error || '删除数据集预设失败');
    return result;
}

export async function importDatasetPreset(api, name, content) {
    const result = await api('/api/config/dataset-presets/import', {
        method: 'POST',
        body: JSON.stringify({ name, content }),
    });
    if (result.ok === false) throw new Error(result.error || '导入数据集预设失败');
    return result;
}

export async function createDatasetPresetGroup(api, label) {
    const result = await api('/api/config/file-groups', {
        method: 'POST',
        body: JSON.stringify({ label, kind: 'dataset' }),
    });
    if (result.ok === false) throw new Error(result.error || '创建数据集分组失败');
    if (result.group?.kind !== 'dataset') throw new Error('WebUI 后端仍是旧版本，请重启服务后再创建数据集分组');
    return result;
}

export async function renameDatasetPresetGroup(api, groupId, label) {
    const result = await api(`/api/config/file-groups/${encodeURIComponent(groupId)}`, {
        method: 'PATCH',
        body: JSON.stringify({ label }),
    });
    if (result.ok === false) throw new Error(result.error || '重命名数据集分组失败');
    return result;
}

export async function deleteDatasetPresetGroup(api, groupId) {
    const result = await api(`/api/config/file-groups/${encodeURIComponent(groupId)}`, { method: 'DELETE' });
    if (result.ok === false) throw new Error(result.error || '删除数据集分组失败');
    return result;
}

export async function placeDatasetPreset(api, file, groupId, order) {
    const result = await api('/api/config/file-groups/place', {
        method: 'POST',
        body: JSON.stringify({ target: 'file', file, group: groupId, order }),
    });
    if (result.ok === false) throw new Error(result.error || '更新数据集预设顺序失败');
    return result;
}

export async function exportDatasetPreset(api, file) {
    const result = await readDatasetPreset(api, file);
    const blob = new Blob([result.content || ''], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = file.split('/').pop() || 'dataset.toml';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

export function datasetPresetPathFromName(name) {
    const stem = String(name || '')
        .replace(/\.toml$/i, '')
        .replace(/\\/g, '/')
        .split('/')
        .pop()
        .replace(/[^A-Za-z0-9_-]+/g, '_')
        .replace(/^_+|_+$/g, '') || 'dataset';
    return `configs/datasets/${stem}.toml`;
}
