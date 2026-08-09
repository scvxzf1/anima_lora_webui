/**
 * TOML group export/queue action helpers.
 */
import { showHistoryTaskDialog } from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { enqueueTrainingQueueBatchRequest, isCliOnlySpdSource, showPreflightDialog } from '../anima-app/helpers/training-launch-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    setTomlStatus,
    tomlFileDisplayName,
    updateTomlActionState,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { hasPendingConfigChanges } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { api, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { createTomlZipBlob, downloadBlob } from '../anima-app/helpers/toml-io-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { renderPreflightPending, showPreflightRequestError } from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { appendLog } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { showTrainingView, updateTrainingQueueFromPayload } from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';

const tomlState = getTomlState();

export function exportableTomlGroupFiles(group) {
    return (group?.files || [])
        .filter((item) => item?.path && String(item.path).toLowerCase().endsWith('.toml'));
}

export async function exportTomlGroup(group) {
    const files = exportableTomlGroupFiles(group);
    if (!files.length) {
        setTomlStatus('error', '该分组没有可导出的 TOML 文件');
        return;
    }
    if (hasPendingConfigChanges(tomlState.currentTomlFile)) {
        setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再导出分组');
        updateTomlActionState(tomlState.currentTomlFile);
        return;
    }

    const filename = `${exportTomlGroupFilename(group)}.zip`;
    setTomlStatus('pending', `正在读取分组“${group.label || group.id}”中的 ${files.length} 个配置...`, { persist: true });

    try {
        const entries = await Promise.all(files.map(async (item) => {
            const path = String(item.path || '');
            const data = await api(`/api/config/raw?file=${encodeURIComponent(path)}`);
            if (data?.ok === false) {
                throw new Error(`${path}: ${data.error || '读取失败'}`);
            }
            return {
                name: item.filename || path,
                content: data.content || '',
            };
        }));
        const blob = createTomlZipBlob(entries);
        downloadBlob(blob, filename);
        setTomlStatus('ok', `已导出分组“${group.label || group.id}”：1 个 zip，内含 ${files.length} 个独立 TOML 文件`, { persist: true });
    } catch (e) {
        setTomlStatus('error', `导出分组失败: ${e.message || e}`, { persist: true });
    }
}

export function exportTomlGroupFilename(group) {
    const raw = String(group?.label || group?.id || 'toml-group').trim();
    const safe = raw.replace(/[\\/:*?"<>|\r\n\t]+/g, '_').replace(/\s+/g, '_').replace(/^[._]+|[._]+$/g, '');
    return safe || 'toml-group';
}

export function queueableTomlGroupFiles(group) {
    return (group?.files || [])
        .filter((item) => item?.path && item.trainable)
        .filter((item) => !String(item.path || '').replace(/\\/g, '/').startsWith('configs/datasets/'));
}

export function tomlItemQueueVariant(item) {
    if (item?.method) return item.method;
    const filename = String(item?.filename || item?.path || '').split('/').pop() || '';
    return filename.toLowerCase().endsWith('.toml') ? filename.slice(0, -5) : filename;
}

export function tomlItemQueueEntry(item, preset = '') {
    const path = String(item?.path || '').trim();
    const methodsSubdir = String(item?.methods_subdir || '').trim() || 'imported';
    const label = tomlFileDisplayName(item);
    return {
        variant: tomlItemQueueVariant(item),
        preset: preset || val('preset-select') || 'default',
        methods_subdir: methodsSubdir,
        config_file: path,
        filename: item?.filename || (path ? path.split('/').pop() : ''),
        label: label === '未命名配置文件' ? '' : label,
        confirm_preprocess: true,
    };
}

export function tomlGroupQueueFailureLabel(item, failure = {}, index = -1) {
    const path = String(item?.path || item?.config_file || failure.config_file || '').trim();
    if (path) return path;
    const failureLabel = String(failure.label || failure.filename || item?.label || item?.filename || '').trim();
    if (failureLabel) return failureLabel;
    const label = tomlFileDisplayName(item);
    if (label && label !== '未命名配置文件') return label;
    const fallbackIndex = Number.isFinite(index) && index >= 0 ? index + 1 : Number(failure.index || 0) + 1;
    return fallbackIndex > 0 ? `第 ${fallbackIndex} 个配置` : '批量请求';
}

export async function showTomlGroupQueueConfirmDialog(group, files) {
    const wrap = document.createElement('div');
    wrap.className = 'history-task-dialog-message toml-group-queue-dialog';

    const strong = document.createElement('strong');
    strong.textContent = `${group.label || group.id || '配置分组'} · ${files.length} 个配置`;
    const message = document.createElement('p');
    message.textContent = `确认后会按当前 GPU 选择和当前预设 ${val('preset-select') || 'default'}，把该分组内可训练配置逐个冻结并加入队列；队列会保持暂停，等待你手动继续。`;

    const list = document.createElement('div');
    list.className = 'toml-group-queue-list';
    files.slice(0, 12).forEach((item) => {
        const row = document.createElement('code');
        row.textContent = item.path || tomlFileDisplayName(item);
        list.appendChild(row);
    });
    if (files.length > 12) {
        const more = document.createElement('span');
        more.textContent = `还有 ${files.length - 12} 个配置...`;
        list.appendChild(more);
    }

    wrap.append(strong, message, list);
    return showHistoryTaskDialog({
        title: '批量加入训练队列',
        description: '这会创建独立运行配置，不会修改原 TOML 文件。',
        body: wrap,
        confirmText: '确认加入队列',
        cancelText: '取消',
        getValue: () => true,
    });
}

export async function enqueueTomlGroupToQueue(group) {
    const files = queueableTomlGroupFiles(group);
    if (!files.length) {
        setTomlStatus('error', '该分组没有可加入队列的训练配置');
        return;
    }
    if (hasPendingConfigChanges(tomlState.currentTomlFile)) {
        setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再批量加入队列');
        updateTomlActionState(tomlState.currentTomlFile);
        return;
    }
    const confirmed = await showTomlGroupQueueConfirmDialog(group, files);
    if (!confirmed) return;

    const preset = val('preset-select') || 'default';
    const entries = files.map((item) => tomlItemQueueEntry(item, preset));
    const invalidIndex = entries.findIndex((entry) => !entry.variant || isCliOnlySpdSource(entry.variant, entry.methods_subdir));
    if (invalidIndex >= 0) {
        const item = files[invalidIndex];
        const entry = entries[invalidIndex];
        const message = entry.variant ? 'SPD CLI 实验配置不能通过 Web 队列启动' : '配置缺少可训练变体名称';
        setTomlStatus('error', `批量加入队列已停止：${item.path || tomlFileDisplayName(item)}，${message}`, { persist: true });
        showPreflightRequestError(message);
        return;
    }

    renderPreflightPending({
        title: '批量加入训练队列',
        message: `正在冻结 ${files.length} 个配置并加入队列...`,
        detail: '后端会逐个预检测并创建独立运行配置；如果某个配置失败，会返回停止位置和失败原因。',
    });
    let res;
    try {
        res = await enqueueTrainingQueueBatchRequest({
            items: entries,
            preset,
            startPaused: true,
        });
    } catch (e) {
        const message = `批量加入队列失败: ${e.message || e}`;
        setTomlStatus('error', message, { persist: true });
        showPreflightRequestError(message);
        return;
    }
    const queued = Number(res.queued_count || 0);
    updateTrainingQueueFromPayload(res);

    if (queued > 0) {
        document.querySelector('[data-tab="training"]')?.click();
        showTrainingView('queue');
        appendLog(`[状态] 已将分组“${group.label || group.id}”中的 ${queued} 个配置加入训练队列`);
    }
    if (!res.ok) {
        const failure = (res.failures || [])[0] || {};
        const failedIndex = Number.isInteger(res.failed_index) ? res.failed_index : Number(failure.index ?? -1);
        const item = files[failedIndex] || res.failed_item || {};
        const fileLabel = tomlGroupQueueFailureLabel(item, failure, failedIndex);
        const error = res.error || failure.error || '加入队列失败';
        const message = `批量加入队列已停止：${fileLabel}，${error}`;
        setTomlStatus('error', queued ? `已加入 ${queued} 个配置；${message}` : message, { persist: true });
        if (res.preflight) {
            showPreflightDialog(res.preflight, false, { willAutoPreprocess: true });
        } else {
            showPreflightRequestError(message);
        }
        return;
    }

    const dialog = document.getElementById('preflight-dialog');
    if (dialog?.open) dialog.close('queued-group');
    setTomlStatus('ok', `已将 ${queued} 个配置加入训练队列`, { persist: true });
}

export function createTomlGroupActionButton(label, handler, options = {}) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = [
        'toml-group-action-btn',
        options.variant ? `toml-group-action-btn-${options.variant}` : '',
        options.danger ? 'danger' : '',
    ].filter(Boolean).join(' ');
    btn.textContent = label;
    btn.disabled = Boolean(options.disabled);
    btn.title = options.title || label;
    btn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        if (!btn.disabled) runTomlGroupAction(handler, btn);
    });
    return btn;
}

export function runTomlGroupAction(handler, button = null) {
    if (tomlState.tomlGroupActionBusy) return;
    tomlState.tomlGroupActionBusy = true;
    if (button) button.disabled = true;
    Promise.resolve()
        .then(handler)
        .catch((e) => {
            setTomlStatus('error', '分组操作失败: ' + e.message);
        })
        .finally(() => {
            tomlState.tomlGroupActionBusy = false;
            if (button?.isConnected) button.disabled = false;
        });
}
