/**
 * TOML group/file drag placement helpers.
 */
import { createFileGroupDragHandle } from './file-group-drag.js?v=module-bootstrap-20260831-release-v1';
import { loadTomlFileList } from '../anima-app/helpers/toml-manager-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    setTomlStatus,
    updateTomlActionState,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { api } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260831-release-v1';
import { hasPendingConfigChanges } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getSortableTomlGroups, isTomlGroupDraggable, isTrainingTomlGroup } from '../anima-app/helpers/toml-io-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260831-release-v1';

const tomlState = getTomlState();

export function canDropTomlFileToGroup(group) {
    return Boolean(
        group?.id &&
        isTrainingTomlGroup(group) &&
        group.movable &&
        !group.locked &&
        !group.user_group_locked
    );
}

export function isTomlFileDraggable(item) {
    return Boolean(item?.path && !item.locked && !hasPendingConfigChanges(tomlState.currentTomlFile));
}

export function createTomlGroupDragHandle(group, details) {
    const disabled = !isTomlGroupDraggable(group);
    return createFileGroupDragHandle({
        target: 'group',
        scope: 'training',
        groupId: group.id,
        sourceElement: details,
        canDrag: () => isTomlGroupDraggable(group),
        blockedMessage: () => setTomlStatus('error', '该配置分组不能拖动排序'),
    }, {
        disabled,
        label: `拖动配置分组 ${group.label || group.id}`,
        title: disabled ? '该配置分组不能拖动排序' : '拖动调整配置分组顺序',
    });
}

export async function placeTomlGroup(payload, index) {
    const groupId = payload?.groupId;
    if (!groupId) return;
    try {
        const res = await api('/api/config/file-groups/place', {
            method: 'POST',
            body: JSON.stringify({ target: 'group', group: groupId, scope: 'training', index }),
        });
        if (!res.ok) {
            setTomlStatus('error', res.error || '调整分组位置失败');
            return;
        }
        await loadTomlFileList(tomlState.currentTomlFile || '');
        setTomlStatus('ok', res.message || '分组位置已更新');
    } catch (e) {
        setTomlStatus('error', '请求失败: ' + e.message);
    }
}

export async function placeTomlFile(payload, groupId, index, placeOptions = {}) {
    const file = payload?.file;
    if (!file || !groupId) return;
    if (hasPendingConfigChanges(tomlState.currentTomlFile)) {
        setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再拖动排序');
        updateTomlActionState(tomlState.currentTomlFile);
        return;
    }
    try {
        const res = await api('/api/config/file-groups/place', {
            method: 'POST',
            body: JSON.stringify({
                target: 'file',
                file,
                group: groupId,
                index,
                ...(placeOptions?.anchor ? {
                    anchor: placeOptions.anchor,
                    position: placeOptions.position === 'before' ? 'before' : 'after',
                } : {}),
                ...(Array.isArray(placeOptions?.order) && placeOptions.order.length ? {
                    order: placeOptions.order,
                } : {}),
            }),
        });
        if (!res.ok) {
            setTomlStatus('error', res.error || '配置位置调整失败');
            return;
        }
        await loadTomlFileList(tomlState.currentTomlFile || file);
        setTomlStatus('ok', res.message || '配置位置已更新');
    } catch (e) {
        setTomlStatus('error', '请求失败: ' + e.message);
    }
}

export function tomlFileDragOptions() {
    return {
        scope: 'training',
        rowSelector: '.toml-file-row-wrap',
        canDropToGroup: canDropTomlFileToGroup,
        onDrop: placeTomlFile,
    };
}

export function tomlGroupDragOptions() {
    return {
        scope: 'training',
        getSortableGroups: () => getSortableTomlGroups(),
        canDropOnGroup: (group) => isTomlGroupDraggable(group),
        onDrop: placeTomlGroup,
    };
}
