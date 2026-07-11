/**
 * TOML group/file move-delete actions.
 * Moved out of anima-app mechanical chunks.
 */
import { showHistoryTaskConfirmDialog, showHistoryTaskDialog } from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260711-ir6';
import { configureTomlActionsBridge } from '../anima-app/helpers/toml-actions-bridge.js?v=module-bootstrap-20260711-ir6';
import { loadTomlFileList } from '../anima-app/helpers/toml-manager-bridge.js?v=module-bootstrap-20260711-ir6';
import { api, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir6';
import { setCurrentTrainingSourceFromVariant } from '../training-source/source-state.js?v=module-bootstrap-20260711-ir6';
import { hasPendingConfigChanges, showAppConfirmDialog, updateTomlDirtyState, updateTomlSelectionUI } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir6';
import { isTrainingTomlGroup, reorderTomlFileGroups } from '../anima-app/helpers/toml-io-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    applyTomlLockState,
    armTomlDeleteConfirm,
    resetTomlDeleteConfirm,
    resetTomlSaveConfirm,
    setTomlStatus,
    tomlLockLabel,
    updateTomlActionState,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir6';

const datasetState = getDatasetState();
const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function currentOutputRunState() {
    return datasetState.outputRunState || {};
}

    export async function moveCurrentTomlToGroup() {
        const file = tomlState.currentTomlFile || val('toml-file-select');
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再移动分组');
            updateTomlActionState(file);
            return;
        }
        const meta = tomlState.tomlFileMeta[file];
        if (meta?.locked) {
            setTomlStatus('error', `${tomlLockLabel(meta) || '只读'}配置不能移动分组`);
            return;
        }

        const groups = getMovableTomlGroups(meta?.group);
        if (!groups.length) {
            setTomlStatus('error', '当前没有其他可移入的分组，请先新建分组或解除目标分组锁定');
            return;
        }
        const targetGroupId = await showMoveTomlDialog(file, meta, groups);
        if (!targetGroupId) return;
        try {
            const res = await api('/api/config/file-groups/move-file', {
                method: 'POST',
                body: JSON.stringify({ file, group: targetGroupId }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '移动分组失败');
                return;
            }
            await loadTomlFileList(file);
            setTomlStatus('ok', res.message || '配置已移动到分组');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export function getMovableTomlGroups(currentGroupId = '') {
        return reorderTomlFileGroups(tomlState.tomlFileGroups)
            .filter((group) => isTrainingTomlGroup(group) && group.movable && !group.locked && !group.user_group_locked && group.id !== currentGroupId);
    }

    export function deleteTomlGroupButtonTitle(group) {
        if (!group) return '配置分组不可用';
        if (group.user_group_locked) return '该分组已锁定，请先解除分组锁定后再删除';
        if (!group.deletable) return '系统固定分组或只读分组不能删除';
        const count = (group.files || []).length;
        return count > 0
            ? `删除当前分组“${group.label || group.id}”；不会删除其中 ${count} 个 TOML 文件`
            : `删除当前空分组“${group.label || group.id}”`;
    }

    export function canDeleteTomlGroup(group) {
        return Boolean(group?.deletable && !group.user_group_locked);
    }

    export function showMoveTomlDialog(file, meta, groups) {
        const wrap = document.createElement('div');
        wrap.className = 'toml-move-dialog-body';

        const current = document.createElement('p');
        current.className = 'toml-move-current';
        current.textContent = `当前配置: ${file}`;
        wrap.appendChild(current);

        const list = document.createElement('div');
        list.className = 'toml-move-option-list';
        const radios = [];
        for (const group of groups) {
            const label = document.createElement('label');
            label.className = 'toml-move-option';

            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'toml-move-target-group';
            input.value = group.id;
            input.checked = group.id !== meta?.group && !radios.some((item) => item.checked);
            radios.push(input);

            const text = document.createElement('span');
            const title = document.createElement('strong');
            title.textContent = group.label || group.id;
            const detail = document.createElement('small');
            const count = (group.files || []).length;
            detail.textContent = `${count} 个配置`;
            text.append(title, detail);

            label.append(input, text);
            list.appendChild(label);
        }
        wrap.appendChild(list);

        return showHistoryTaskDialog({
            title: '移动配置',
            description: '选择目标分组后确认，配置文件路径不会改变，只调整右侧分组归属。',
            body: wrap,
            confirmText: '移动到分组',
            onOpen: () => {
                const checked = radios.find((item) => item.checked) || radios[0];
                checked?.focus();
            },
            getValue: () => {
                const checked = wrap.querySelector('input[name="toml-move-target-group"]:checked');
                return checked?.value || '';
            },
        });
    }

    export async function deleteTomlGroup(group) {
        if (!canDeleteTomlGroup(group)) {
            setTomlStatus('error', deleteTomlGroupButtonTitle(group));
            return;
        }
        const count = (group.files || []).length;
        const ok = await showHistoryTaskConfirmDialog({
            title: '删除配置分组',
            description: group.label || group.id,
            message: count > 0
                ? `只删除这个分组，不删除其中 ${count} 个 TOML 文件；这些文件会回到导入配置或数据集配置等默认分组。`
                : '只删除这个分组，不会删除任何 TOML 文件。',
            confirmText: '删除分组',
            danger: true,
        });
        if (!ok) return;
        const reallyOk = await showHistoryTaskConfirmDialog({
            title: '你真的确认吗？',
            description: group.label || group.id,
            message: count > 0
                ? `确认后会删除这个分组，分组内 ${count} 个 TOML 文件会回到默认分组。`
                : '确认后会删除这个空分组。',
            confirmText: '我确认',
            cancelText: '我觉得不对',
            cancelPrimary: true,
            danger: true,
        });
        if (!reallyOk) return;
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '删除分组失败');
                return;
            }
            await loadTomlFileList(tomlState.currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已删除');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export async function deleteTomlFile() {
        const file = tomlState.currentTomlFile || val('toml-file-select');
        const meta = tomlState.tomlFileMeta[file];
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再删除');
            updateTomlActionState(file);
            return;
        }
        if (!meta) {
            await handleDeletedTomlSelection(file, '当前配置已不在列表中，已刷新配置列表');
            return;
        }
        if (meta.locked) {
            setTomlStatus('error', `${tomlLockLabel(meta) || '只读'}配置不能删除`);
            updateTomlActionState(file);
            return;
        }

        if (tomlState.tomlDeleteConfirmFile !== file) {
            armTomlDeleteConfirm(file);
            return;
        }
        resetTomlDeleteConfirm({ update: false });

        try {
            const res = await api(`/api/config/raw?file=${encodeURIComponent(file)}`, {
                method: 'DELETE',
            });
            if (!res.ok) {
                if (isMissingTomlFileResponse(res)) {
                    await handleDeletedTomlSelection(file, res.error || '配置文件不存在或已被删除');
                    return;
                }
                setTomlStatus('error', res.error || '删除失败');
                return;
            }

            await handleDeletedTomlSelection(file, `已删除配置: ${file}`, { ok: true });
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export function isMissingTomlFileResponse(res) {
        return String(res?.error || '').includes('不存在') || String(res?.error || '').includes('已被删除');
    }

    export async function handleDeletedTomlSelection(file, message, options = {}) {
        if (currentTrainingSourceState().file === file) {
            setCurrentTrainingSourceFromVariant(val('variant-select') || 'lora');
        }
        delete tomlState.tomlFileMeta[file];
        tomlState.tomlFiles = tomlState.tomlFiles.filter((item) => item !== file);
        clearCurrentTomlSelection();
        await loadTomlFileList('', { skipDefaultLoad: true });
        clearCurrentTomlSelection();
        updateTomlDirtyState();
        setTomlStatus(options.ok ? 'ok' : 'error', message, { persist: true });
    }

    export function clearCurrentTomlSelection() {
        resetTomlDeleteConfirm({ update: false });
        resetTomlSaveConfirm({ update: false });
        tomlState.currentTomlFile = '';
        tomlState.tomlSavedContent = '';
        const editor = document.getElementById('toml-editor');
        if (editor) {
            editor.value = '';
            editor.readOnly = false;
            editor.title = '';
        }
        const select = document.getElementById('toml-file-select');
        if (select) select.value = '';
        updateTomlSelectionUI('');
        applyTomlLockState('');
    }

    export async function restoreSystemTomlPresets() {
        const file = tomlState.currentTomlFile || val('toml-file-select');
        const meta = tomlState.tomlFileMeta[file];
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再还原系统预设');
            updateTomlActionState(file);
            return;
        }

        const currentHint = meta?.restorable ? `\n当前文件 ${file} 也会一起还原。` : '';
        const ok = await showAppConfirmDialog({
            title: '还原系统预设',
            description: 'base、presets、methods、gui-methods',
            message: `还原会覆盖系统预设文件，但会先自动备份当前内容。用户导入/副本和数据集配置不会被还原。${currentHint}`,
            confirmText: '还原系统预设',
            danger: true,
        });
        if (!ok) return;

        try {
            const res = await api('/api/config/restore-system', {
                method: 'POST',
                body: JSON.stringify({}),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '还原失败');
                return;
            }

            const preferredFile = file && tomlState.tomlFiles.includes(file) ? file : '';
            await loadTomlFileList(preferredFile);
            const restoredCount = res.restored?.length || 0;
            const skippedCount = res.skipped?.length || 0;
            const backupText = res.backup_dir ? `，备份在 ${res.backup_dir}` : '';
            setTomlStatus('ok', `已还原 ${restoredCount} 个系统预设，跳过 ${skippedCount} 个${backupText}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }


configureTomlActionsBridge({
    moveCurrentTomlToGroup,
    getMovableTomlGroups,
    deleteTomlGroupButtonTitle,
    canDeleteTomlGroup,
    showMoveTomlDialog,
    deleteTomlGroup,
    deleteTomlFile,
    isMissingTomlFileResponse,
    handleDeletedTomlSelection,
    clearCurrentTomlSelection,
    restoreSystemTomlPresets,
});
