/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { isLiveRunningState } from '../../live-training/index.js?v=module-bootstrap-20260831-release-v1';
import { VARIANT_METHOD_FAMILY } from '../../../config/catalog.js?v=module-bootstrap-20260831-release-v1';
import {
    configTrainingSourceMode,
    trainingSourceLaunchReadiness,
} from '../helpers/training-source-bridge.js?v=module-bootstrap-20260831-release-v1';
import { showHistoryTaskInputDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260831-release-v1';
import { loadTomlFileList } from '../helpers/toml-manager-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getMovableTomlGroups } from '../helpers/toml-actions-bridge.js?v=module-bootstrap-20260831-release-v1';
import { configureTomlActionStateBridge } from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { loadConfig } from '../helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260831-release-v1';
import { currentFormConfigFile, hasPendingConfigChanges, hasUnsavedFormChanges, isTomlDirty, showAppConfirmDialog, updateTomlDirtyState } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260831-release-v1';
import { api, populateSelect, val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { reorderTomlFileGroups } from '../helpers/toml-io-bridge.js?v=module-bootstrap-20260831-release-v1';
import { currentTrainingConfigFile } from '../helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260831-release-v1';
import { renderTomlFileGroups } from '../helpers/toml-drag-bridge.js?v=module-bootstrap-20260831-release-v1';
import { rememberSelectionSnapshot } from './13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260831-release-v1';

const tomlState = getTomlState();
const trainingState = getTrainingState();
const trainingRuntime = trainingState.trainingRuntime;

    export function updateTomlActionState(filePath) {
        const selectedFile = filePath || tomlState.currentTomlFile || val('toml-file-select') || '';
        const meta = tomlState.tomlFileMeta[selectedFile];
        const editorDirty = isTomlDirty();
        const formFile = currentFormConfigFile();
        const formDirty = hasUnsavedFormChanges(formFile);
        const dirty = editorDirty || formDirty;
        const saveFile = formDirty ? formFile : selectedFile;
        const saveMeta = tomlState.tomlFileMeta[saveFile] || (saveFile === selectedFile ? meta : undefined);
        const saveLocked = formDirty ? isTomlLocked(saveFile) : Boolean(saveMeta?.locked);
        const saveBtn = document.getElementById('btn-save-toml');
        if (saveBtn) {
            saveBtn.disabled = saveLocked || !saveFile || !dirty;
            saveBtn.textContent = formDirty && saveFile !== selectedFile
                ? '保存更新当前表单配置'
                : '保存更新当前选中配置';
            saveBtn.classList.remove('btn-confirm-danger');
            saveBtn.title = saveLocked
                ? '该配置文件已锁定，请使用新名称另存新配置后编辑'
                : (dirty
                    ? (formDirty
                        ? `把左侧表单、数据集预设选择和采样提示词等修改写回 ${saveFile}；保存后训练会使用这些新值。`
                        : '把直接编辑器里的 TOML 文本写回当前文件。')
                    : '当前配置没有未保存修改，不需要保存。');
        }
        updateTomlEditorPanelState(selectedFile);
        const applyBtn = document.getElementById('btn-apply-toml');
        if (applyBtn) {
            applyBtn.disabled = !meta?.trainable || dirty;
            applyBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : (meta?.trainable
                    ? '把右侧选中的 TOML 加载到左侧表单，并把它设为“开始训练”使用的配置。'
                    : '该文件不是完整训练配置，不能作为训练入口。');
        }
        const moveBtn = document.getElementById('btn-move-toml-group');
        if (moveBtn) {
            const canMove = Boolean(selectedFile && meta && !meta.locked && !dirty && getMovableTomlGroups(meta.group).length > 0);
            moveBtn.disabled = !canMove;
            moveBtn.title = dirty
                ? '当前配置尚未保存，请先保存或放弃修改后再移动分组位置'
                : (meta?.locked
                    ? `${tomlLockLabel(meta) || '只读'}配置不能移动分组位置`
                    : (canMove ? '只调整右侧配置文件列表里的分组归属，不会改 TOML 内容或磁盘路径。' : '当前没有其他可移入的分组'));
        }
        const reloadBtn = document.getElementById('btn-reload-toml');
        if (reloadBtn) {
            reloadBtn.disabled = !selectedFile;
            reloadBtn.title = '从磁盘重新读取当前配置文件；未保存的编辑会被丢弃，但不会切换训练入口。';
        }
        const lockBtn = document.getElementById('btn-lock-toml');
        if (lockBtn) {
            const hasFile = Boolean(selectedFile && meta);
            const isSystemOrGroupLocked = Boolean(meta?.system_locked || meta?.group_locked);
            lockBtn.disabled = !hasFile || isSystemOrGroupLocked || dirty;
            lockBtn.textContent = meta?.user_locked ? '解除锁定' : '锁定当前文件';
            lockBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : lockTomlButtonTitle(meta);
        }
        const deleteBtn = document.getElementById('btn-delete-toml');
        if (deleteBtn) {
            const canDelete = Boolean(selectedFile && meta && !meta.locked && !dirty);
            if (!canDelete) resetTomlDeleteConfirm({ update: false });
            const confirming = canDelete && tomlState.tomlDeleteConfirmFile === selectedFile;
            deleteBtn.disabled = !canDelete;
            deleteBtn.textContent = confirming ? '确认删除配置' : '删除当前配置';
            deleteBtn.classList.toggle('btn-confirm-danger', confirming);
            deleteBtn.title = dirty
                ? '当前配置尚未保存，请先保存或放弃修改后再删除'
                : (confirming ? '再次点击才会真正删除当前配置文件' : deleteTomlButtonTitle(meta));
        }
        const restoreBtn = document.getElementById('btn-restore-system-toml');
        if (restoreBtn) {
            restoreBtn.disabled = dirty;
            restoreBtn.title = dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : '从项目内置版本还原系统预设；会覆盖系统预设文件，还原前会自动备份。用户导入配置不会被还原。';
        }
        const startBtn = document.getElementById('btn-start-from-config');
        const trainingConfigFile = currentTrainingConfigFile();
        const sourceMode = typeof configTrainingSourceMode === 'function' ? configTrainingSourceMode() : 'fresh';
        const sourceReady = typeof trainingSourceLaunchReadiness === 'function'
            ? trainingSourceLaunchReadiness()
            : { ready: true, checking: false, reason: '' };
        const sourceBlockedTitle = sourceReady.checking ? '正在审查续接来源，审查完成前不能启动。' : (sourceReady.reason || '训练来源审查未通过');
        const canStart = sourceMode === 'full_resume'
            ? sourceReady.ready && !isLiveRunningState(trainingRuntime.state)
            : Boolean(trainingConfigFile) && !dirty && sourceReady.ready;
        if (startBtn) {
            startBtn.disabled = !canStart;
            startBtn.textContent = sourceMode === 'full_resume'
                ? '开始完整续训'
                : sourceMode === 'weight_hotstart'
                    ? '开始热启动训练'
                    : '开始训练';
            startBtn.title = !sourceReady.ready
                ? sourceBlockedTitle
                : sourceMode === 'full_resume' && isLiveRunningState(trainingRuntime.state)
                    ? '当前已有任务在运行，请改用加入队列。'
                    : sourceMode === 'full_resume'
                        ? '使用历史任务冻结配置快照启动完整续训，当前表单不会覆盖历史配置。'
                    : dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : (canStart ? '运行训练前预检测，通过后选择立即启动或加入队列。' : '请先选择可训练配置文件');
        }
        const queueBtn = document.getElementById('btn-queue-from-config');
        const canQueue = sourceMode === 'full_resume'
            ? sourceReady.ready
            : Boolean(trainingConfigFile) && !dirty && sourceReady.ready;
        if (queueBtn) {
            queueBtn.disabled = !canQueue;
            queueBtn.textContent = sourceMode === 'full_resume'
                ? '完整续训入队'
                : sourceMode === 'weight_hotstart'
                    ? '热启动入队'
                    : '加入队列';
            queueBtn.title = !sourceReady.ready
                ? sourceBlockedTitle
                : sourceMode === 'full_resume'
                    ? '使用历史任务冻结配置快照加入完整续训队列，启动前会再次检查 checkpoint-state。'
                : dirty
                ? '当前配置尚未保存，请先保存更新当前选中配置或另存新配置'
                : (canQueue ? '把当前训练配置直接冻结并加入训练队列。' : '请先选择可训练配置文件');
        }
    }

    export function isTomlLocked(filePath) {
        return Boolean(tomlState.tomlFileMeta[filePath]?.locked);
    }

    export function applyTomlLockState(filePath) {
        const locked = isTomlLocked(filePath);
        setTomlEditorLocked(locked);
        updateTomlActionState(filePath);
    }

    export function setTomlEditorLocked(locked) {
        const editor = document.getElementById('toml-editor');
        editor.readOnly = locked;
        editor.title = locked ? '该配置文件已锁定，只能导出或使用新名称另存新配置' : '';
    }

    export function updateTomlEditorPanelState(filePath = tomlState.currentTomlFile) {
        const panel = document.getElementById('toml-edit-panel');
        const manager = document.getElementById('config-project-workspace') || document.querySelector('.toml-manager');
        const directEditor = document.getElementById('config-direct-editor');
        const toggleBtn = document.getElementById('btn-toggle-toml-editor');
        const saveDirectBtn = document.getElementById('btn-save-toml-direct');
        const copyBtn = document.getElementById('btn-copy-toml');
        const meta = tomlState.tomlFileMeta[filePath];
        const editorDirty = isTomlDirty();
        const formFile = currentFormConfigFile();
        const formDirty = hasUnsavedFormChanges(formFile);
        const dirty = editorDirty || formDirty;
        const locked = Boolean(meta?.locked);
        const confirming = Boolean(filePath && tomlState.tomlSaveConfirmFile === filePath);
        if (toggleBtn) {
            const open = Boolean(panel && !panel.hidden && tomlState.tomlManagerMode === 'project');
            if (manager) manager.classList.toggle('toml-edit-open', open);
            if (directEditor) directEditor.hidden = !open;
            toggleBtn.disabled = !filePath;
            toggleBtn.textContent = open ? '收起配置文件编辑' : '直接编辑配置文件';
            toggleBtn.classList.toggle('active', open);
            toggleBtn.title = open
                ? '收起二级配置文件编辑界面；不会自动保存修改。'
                : '展开二级界面，查看、复制或直接编辑当前 TOML。适合批量改字段，保存时需要二次确认。';
        }
        if (saveDirectBtn) {
            saveDirectBtn.disabled = locked || !filePath || !editorDirty || formDirty;
            saveDirectBtn.textContent = confirming ? '确认保存配置文件' : '保存配置文件';
            saveDirectBtn.classList.toggle('btn-confirm-danger', confirming);
            saveDirectBtn.title = locked
                ? '该配置文件已锁定，请使用新名称另存新配置后编辑'
                : (formDirty
                    ? '左侧表单或数据集还有未保存修改，请先用“保存更新当前选中配置”保存。'
                    : (editorDirty
                    ? (confirming ? '再次点击才会真正写入磁盘；请确认 TOML 内容没有语法错误。' : '第一次点击进入确认，第二次点击保存，防止误覆盖配置文件。')
                    : '直接编辑器没有未保存的 TOML 文本修改'));
        }
        if (copyBtn) {
            copyBtn.disabled = !filePath && !document.getElementById('toml-editor')?.value;
            copyBtn.title = '复制当前编辑器里的 TOML 内容，方便备份、对比或发给别人排查。';
        }
    }

    export function toggleTomlEditorPanel() {
        const panel = document.getElementById('toml-edit-panel');
        if (!panel) return;
        if (!tomlState.currentTomlFile) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        panel.hidden = !panel.hidden;
        updateTomlEditorPanelState(tomlState.currentTomlFile);
        if (!panel.hidden) {
            document.getElementById('toml-editor')?.focus();
        }
    }

    export async function copyTomlEditorContent() {
        const editor = document.getElementById('toml-editor');
        if (!editor) return;
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(editor.value);
            } else {
                editor.focus();
                editor.select();
                document.execCommand('copy');
            }
            setTomlStatus('ok', '已复制当前配置内容');
        } catch (e) {
            setTomlStatus('error', '复制失败: ' + e.message);
        }
    }

    export function tomlLockLabel(meta) {
        if (!meta?.locked) return '';
        if (meta.system_locked) return '系统只读';
        if (meta.user_locked) return '用户锁定';
        if (meta.user_group_locked) return '分组锁定';
        if (meta.group_locked) return '分组只读';
        return meta.lock_reason_label || '只读';
    }

    export function tomlFileDisplayParts(fileOrMeta) {
        const meta = typeof fileOrMeta === 'string'
            ? (tomlState.tomlFileMeta[fileOrMeta] || { path: fileOrMeta })
            : (fileOrMeta || {});
        const path = meta.path || '';
        const filename = meta.filename || (path ? path.split('/').pop() : '');
        const label = meta.label || '';
        const parts = [];
        if (label && label !== filename && label !== path) parts.push(label);
        if (filename) parts.push(filename);
        if (path && path !== filename) parts.push(path);
        return parts;
    }

    export function tomlFileDisplayName(fileOrMeta) {
        const parts = tomlFileDisplayParts(fileOrMeta);
        return parts.length ? parts.join(' / ') : '未命名配置文件';
    }

    export function lockTomlButtonTitle(meta) {
        if (!meta) return '请先选择一个配置文件';
        if (meta.system_locked) return '系统预设已内置锁定，不能手动解除';
        if (meta.group_locked) return '该文件属于只读分组，不能手动解除';
        if (meta.user_group_locked) return '该文件所在分组已锁定，请在分组标题解除锁定';
        if (meta.user_locked) return '解除你为该文件设置的锁定';
        return '锁定当前文件，防止误保存';
    }

    export function deleteTomlButtonTitle(meta) {
        if (!meta) return '请先选择一个配置文件';
        if (meta.locked) return `${tomlLockLabel(meta) || '只读'}配置不能删除`;
        return '删除当前选中的配置文件';
    }

    export function resetTomlDeleteConfirm(options = {}) {
        if (tomlState.tomlDeleteConfirmTimer) {
            clearTimeout(tomlState.tomlDeleteConfirmTimer);
            tomlState.tomlDeleteConfirmTimer = null;
        }
        if (!tomlState.tomlDeleteConfirmFile) return;
        tomlState.tomlDeleteConfirmFile = '';
        if (options.update !== false) {
            updateTomlActionState(tomlState.currentTomlFile);
        }
    }

    export function armTomlDeleteConfirm(file) {
        resetTomlDeleteConfirm({ update: false });
        tomlState.tomlDeleteConfirmFile = file;
        tomlState.tomlDeleteConfirmTimer = setTimeout(() => {
            resetTomlDeleteConfirm();
            setTomlStatus('', '');
        }, 8000);
        updateTomlActionState(file);
        setTomlStatus('error', `再次点击“确认删除配置”才会删除: ${file}`);
    }

    export function resetTomlSaveConfirm(options = {}) {
        if (tomlState.tomlSaveConfirmTimer) {
            clearTimeout(tomlState.tomlSaveConfirmTimer);
            tomlState.tomlSaveConfirmTimer = null;
        }
        if (!tomlState.tomlSaveConfirmFile) return;
        tomlState.tomlSaveConfirmFile = '';
        if (options.update !== false) {
            updateTomlActionState(tomlState.currentTomlFile);
        }
    }

    export function armTomlSaveConfirm(file) {
        resetTomlSaveConfirm({ update: false });
        tomlState.tomlSaveConfirmFile = file;
        tomlState.tomlSaveConfirmTimer = setTimeout(() => {
            resetTomlSaveConfirm();
            setTomlStatus('', '');
        }, 8000);
        updateTomlActionState(file);
        setTomlStatus('error', `再次点击“确认保存”才会写入当前配置: ${file}`);
    }

    export function setTomlStatus(cls, text, options = {}) {
        const el = document.getElementById('toml-status');
        if (tomlState.tomlStatusTimer) {
            clearTimeout(tomlState.tomlStatusTimer);
            tomlState.tomlStatusTimer = null;
        }
        el.className = cls;
        el.textContent = text;
        if (cls === 'ok' && !options.persist) {
            tomlState.tomlStatusTimer = setTimeout(() => {
                el.textContent = '';
                tomlState.tomlStatusTimer = null;
            }, 3000);
        }
    }

    export async function applyTomlToConfig(options = {}) {
        const file = tomlState.currentTomlFile || val('toml-file-select');
        const meta = tomlState.tomlFileMeta[file];
        // Confirmed switch/save-as flows resolve the pending changes before reaching
        // this step, but their form drafts are only cleared by loadConfig below.
        if (!options.pendingChangesResolved && hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再加载选中配置');
            updateTomlActionState(file);
            return;
        }
        if (!meta?.trainable) {
            setTomlStatus('error', '该文件不是完整训练配置，不能加载选中配置');
            return;
        }

        trainingState.currentTrainingSource = {
            method: meta.method,
            methods_subdir: meta.methods_subdir || 'gui-methods',
            file: meta.path,
        };

        if (meta.methods_subdir === 'methods' && meta.method === 'spd') {
            const methodSelect = document.getElementById('method-select');
            if ([...methodSelect.options].some((opt) => opt.value === 'spd')) {
                methodSelect.value = 'spd';
            }
            const variantSelect = document.getElementById('variant-select');
            const variantsPayload = await api('/api/methods/spd/variants');
            const variants = Array.isArray(variantsPayload) ? variantsPayload : (variantsPayload?.items || []);
            populateSelect('variant-select', variants, 'spd');
        } else if (meta.methods_subdir === 'gui-methods') {
            const methodFamily = VARIANT_METHOD_FAMILY[meta.method] || meta.method || 'lora';
            const methodSelect = document.getElementById('method-select');
            if ([...methodSelect.options].some((opt) => opt.value === methodFamily)) {
                methodSelect.value = methodFamily;
            }
            const variantSelect = document.getElementById('variant-select');
            if (![...variantSelect.options].some((opt) => opt.value === meta.method)) {
                const variantsPayload = await api(`/api/methods/${encodeURIComponent(methodFamily)}/variants`);
                const variants = Array.isArray(variantsPayload) ? variantsPayload : (variantsPayload?.items || []);
                populateSelect('variant-select', variants, meta.method);
            }
            if ([...variantSelect.options].some((opt) => opt.value === meta.method)) {
                variantSelect.value = meta.method;
            }
        }

        await loadConfig();
        renderTomlFileGroups(reorderTomlFileGroups(tomlState.tomlFileGroups));
        updateTomlDirtyState();
        rememberSelectionSnapshot();
        if (!options.silent) {
            setTomlStatus('ok', `已应用 ${meta.path} 到表单`);
        }
    }

    export async function toggleTomlUserLock() {
        const file = tomlState.currentTomlFile || val('toml-file-select');
        const meta = tomlState.tomlFileMeta[file];
        if (!file || !meta) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存更新当前选中配置或另存新配置，再调整锁定');
            updateTomlActionState(file);
            return;
        }
        if (meta.system_locked) {
            setTomlStatus('error', '系统预设已内置锁定，不能手动解除');
            return;
        }
        if (meta.group_locked) {
            setTomlStatus('error', '该文件属于只读分组，不能手动解除');
            return;
        }
        if (meta.user_group_locked) {
            setTomlStatus('error', '该文件所在分组已锁定，请在分组标题解除锁定');
            return;
        }

        const nextLocked = !meta.user_locked;
        const message = nextLocked
            ? `锁定 ${file}？锁定后不能直接保存，仍可使用新名称另存新配置。`
            : `解除 ${file} 的用户锁定？解除后可以直接编辑保存。`;
        if (!(await showAppConfirmDialog({
            title: nextLocked ? '锁定配置文件' : '解除配置锁定',
            description: file,
            message,
            confirmText: nextLocked ? '锁定' : '解除锁定',
            danger: nextLocked,
        }))) return;

        try {
            const res = await api('/api/config/lock', {
                method: 'POST',
                body: JSON.stringify({ file, locked: nextLocked }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '锁定操作失败');
                if (res.meta) tomlState.tomlFileMeta[file] = res.meta;
                updateTomlDirtyState();
                return;
            }
            if (res.meta) tomlState.tomlFileMeta[file] = res.meta;
            await loadTomlFileList(file);
            applyTomlLockState(file);
            updateTomlDirtyState();
            setTomlStatus('ok', res.message || (nextLocked ? '已锁定当前文件' : '已解除用户锁定'));
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export async function toggleTomlGroupLock(groupOrId) {
        const group = typeof groupOrId === 'string'
            ? tomlState.tomlFileGroups.find((item) => item.id === groupOrId)
            : groupOrId;
        if (!group) {
            setTomlStatus('error', '分组不存在，请先刷新文件列表');
            return;
        }
        if (!group.lockable) {
            setTomlStatus('error', '该分组不能手动锁定或解锁');
            return;
        }

        const nextLocked = !group.user_group_locked;
        const sourceGroupIds = group.sourceGroupIds?.length ? group.sourceGroupIds : [group.id];
        const message = nextLocked
            ? `锁定分组“${group.label || group.id}”？该分组内文件将不能直接保存，仍可使用新名称另存新配置。`
            : `解除分组“${group.label || group.id}”的锁定？解除后该分组内文件可恢复编辑保存。`;
        if (!(await showAppConfirmDialog({
            title: nextLocked ? '锁定配置分组' : '解除分组锁定',
            description: group.label || group.id,
            message,
            confirmText: nextLocked ? '锁定分组' : '解除锁定',
            danger: nextLocked,
        }))) return;

        try {
            let lastResponse = null;
            for (const groupId of sourceGroupIds) {
                const res = await api('/api/config/group-lock', {
                    method: 'POST',
                    body: JSON.stringify({ group: groupId, locked: nextLocked }),
                });
                if (!res.ok) {
                    setTomlStatus('error', res.error || '分组锁定操作失败');
                    return;
                }
                lastResponse = res;
            }
            await loadTomlFileList(tomlState.currentTomlFile || '');
            applyTomlLockState(tomlState.currentTomlFile);
            updateTomlDirtyState();
            setTomlStatus('ok', lastResponse?.message || (nextLocked ? '已锁定当前分组' : '已解除分组锁定'));
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export async function createTomlGroup() {
        const label = await showHistoryTaskInputDialog({
            title: '新建配置分组',
            description: '用于整理右侧 TOML 配置文件。新分组默认可训练，可移入 imported 配置。',
            label: '分组名称',
            placeholder: '例如：角色配置 / 试验配置 / 正式配置',
            confirmText: '创建分组',
        });
        if (label === null) return;
        if (!label.trim()) {
            setTomlStatus('error', '分组名称不能为空');
            return;
        }
        try {
            const res = await api('/api/config/file-groups', {
                method: 'POST',
                body: JSON.stringify({ label: label.trim(), kind: 'training' }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '创建分组失败');
                return;
            }
            await loadTomlFileList(tomlState.currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已创建');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export async function renameTomlGroup(group) {
        const label = await showHistoryTaskInputDialog({
            title: '重命名配置分组',
            description: '只修改分组显示名称，不会改动配置文件路径。',
            label: '分组名称',
            value: group.label || group.id,
            placeholder: '例如：正式配置',
            confirmText: '保存名称',
        });
        if (label === null) return;
        if (!label.trim()) {
            setTomlStatus('error', '分组名称不能为空');
            return;
        }
        try {
            const res = await api(`/api/config/file-groups/${encodeURIComponent(group.id)}`, {
                method: 'PATCH',
                body: JSON.stringify({ label: label.trim() }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '重命名分组失败');
                return;
            }
            await loadTomlFileList(tomlState.currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已重命名');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

configureTomlActionStateBridge({
    updateTomlActionState,
    isTomlLocked,
    applyTomlLockState,
    setTomlEditorLocked,
    updateTomlEditorPanelState,
    toggleTomlEditorPanel,
    copyTomlEditorContent,
    tomlLockLabel,
    tomlFileDisplayParts,
    tomlFileDisplayName,
    lockTomlButtonTitle,
    deleteTomlButtonTitle,
    resetTomlDeleteConfirm,
    armTomlDeleteConfirm,
    resetTomlSaveConfirm,
    armTomlSaveConfirm,
    setTomlStatus,
    applyTomlToConfig,
    toggleTomlUserLock,
    toggleTomlGroupLock,
    createTomlGroup,
    renameTomlGroup,
});
