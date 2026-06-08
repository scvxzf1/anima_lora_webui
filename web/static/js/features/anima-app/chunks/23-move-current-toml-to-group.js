/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.moveCurrentTomlToGroup = async function moveCurrentTomlToGroup() {
        const file = currentTomlFile || val('toml-file-select');
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件');
            return;
        }
        if (hasPendingConfigChanges(file)) {
            setTomlStatus('error', '当前配置尚未保存，请先保存或放弃修改后再移动分组');
            updateTomlActionState(file);
            return;
        }
        const meta = tomlFileMeta[file];
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

    globalThis.getMovableTomlGroups = function getMovableTomlGroups(currentGroupId = '') {
        return reorderTomlFileGroups(tomlFileGroups)
            .filter((group) => isTrainingTomlGroup(group) && group.movable && !group.locked && !group.user_group_locked && group.id !== currentGroupId);
    }

    globalThis.deleteTomlGroupButtonTitle = function deleteTomlGroupButtonTitle(group) {
        if (!group) return '配置分组不可用';
        if (group.user_group_locked) return '该分组已锁定，请先解除分组锁定后再删除';
        if (!group.deletable) return '系统固定分组或只读分组不能删除';
        const count = (group.files || []).length;
        return count > 0
            ? `删除当前分组“${group.label || group.id}”；不会删除其中 ${count} 个 TOML 文件`
            : `删除当前空分组“${group.label || group.id}”`;
    }

    globalThis.canDeleteTomlGroup = function canDeleteTomlGroup(group) {
        return Boolean(group?.deletable && !group.user_group_locked);
    }

    globalThis.showMoveTomlDialog = function showMoveTomlDialog(file, meta, groups) {
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

    globalThis.deleteTomlGroup = async function deleteTomlGroup(group) {
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
            await loadTomlFileList(currentTomlFile || '');
            setTomlStatus('ok', res.message || '分组已删除');
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    globalThis.deleteTomlFile = async function deleteTomlFile() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
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

        if (tomlDeleteConfirmFile !== file) {
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

    globalThis.isMissingTomlFileResponse = function isMissingTomlFileResponse(res) {
        return String(res?.error || '').includes('不存在') || String(res?.error || '').includes('已被删除');
    }

    globalThis.handleDeletedTomlSelection = async function handleDeletedTomlSelection(file, message, options = {}) {
        if (currentTrainingSource.file === file) {
            setCurrentTrainingSourceFromVariant(val('variant-select') || 'lora');
        }
        delete tomlFileMeta[file];
        tomlFiles = tomlFiles.filter((item) => item !== file);
        clearCurrentTomlSelection();
        await loadTomlFileList('', { skipDefaultLoad: true });
        clearCurrentTomlSelection();
        updateTomlDirtyState();
        setTomlStatus(options.ok ? 'ok' : 'error', message, { persist: true });
    }

    globalThis.clearCurrentTomlSelection = function clearCurrentTomlSelection() {
        resetTomlDeleteConfirm({ update: false });
        resetTomlSaveConfirm({ update: false });
        currentTomlFile = '';
        tomlSavedContent = '';
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

    globalThis.restoreSystemTomlPresets = async function restoreSystemTomlPresets() {
        const file = currentTomlFile || val('toml-file-select');
        const meta = tomlFileMeta[file];
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

            const preferredFile = file && tomlFiles.includes(file) ? file : '';
            await loadTomlFileList(preferredFile);
            const restoredCount = res.restored?.length || 0;
            const skippedCount = res.skipped?.length || 0;
            const backupText = res.backup_dir ? `，备份在 ${res.backup_dir}` : '';
            setTomlStatus('ok', `已还原 ${restoredCount} 个系统预设，跳过 ${skippedCount} 个${backupText}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    // ── 训练控制 ──
    globalThis.startTraining = async function startTraining() {
        const selectedTrainingConfigFile = currentTrainingConfigFile();
        if (tomlManagerMode !== 'output' || !outputRunState.file) {
            if (hasPendingConfigChanges(currentTomlFile)) {
                setTomlStatus('error', '当前配置有未保存修改，请先保存更新当前选中配置或另存新配置，再开始训练');
                updateTomlActionState(currentTomlFile);
                document.querySelector('[data-tab="config"]')?.click();
                return;
            }
        }
        if (!selectedTrainingConfigFile) {
            const message = tomlManagerMode === 'output' && outputRunState.selectedRun
                ? '这个训练输出没有可直接继续训练的 config.runtime.toml，请先另存原始配置或选择其他运行目录'
                : '请选择要训练的配置文件';
            setTomlStatus('error', message);
            return;
        }
        const variant = currentTrainingSource.method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSource.methods_subdir || 'gui-methods';
        if (!variant) return alert('请选择变体');
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            const message = 'SPD 是 CLI 实验配置，只能通过 tasks.py exp-spd / scripts/distill_spd.py 运行；Web 普通训练入口已拦截，避免误用 train.py。';
            setTomlStatus('error', message, { persist: true });
            alert(message);
            return;
        }
        if (continueTrainingSource && !(await refreshContinueTrainingSourceCompatibility())) {
            setTomlStatus('error', continueTrainingSource.message || '继续训练权重与当前配置不兼容', { persist: true });
            return;
        }
        const preflight = await runPreflight(variant, preset, methodsSubdir);
        if (!preflight) {
            if (isPreflightDialogOpen()) await waitForPreflightDialogClose();
            return;
        }
        const willAutoPreprocess = !currentTrainingConfigIsRuntime();
        if (!preflight.ok) {
            const action = await showPreflightDialog(preflight, false, { willAutoPreprocess });
            if (action === 'preprocess') {
                await startPreprocessFromPreflight(preflight);
            }
            return;
        }
        const action = await showPreflightDialog(preflight, true, { willAutoPreprocess });
        if (action === 'preprocess') {
            await startPreprocessFromPreflight(preflight);
            return;
        }
        if (action !== 'continue') return;
        await startTrainingUnchecked(variant, preset, methodsSubdir, { willAutoPreprocess });
    }

    globalThis.queueCurrentTrainingFromConfig = async function queueCurrentTrainingFromConfig() {
        return ensureQueueFeature().queueCurrentTrainingFromConfig();
    }

    globalThis.runPreflight = async function runPreflight(variant, preset, methodsSubdir) {
        const pending = showPreflightPendingDialog({
            title: '训练前预检测',
            message: '正在检查模型路径、数据集路径和预处理启动环境...',
            detail: '这一步可能需要几秒钟；窗口保持打开表示仍在检查。',
        });
        try {
            const res = await api('/api/training/preflight', {
                method: 'POST',
                signal: pending.signal,
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                }),
            });
            pending.resolve();
            return res;
        } catch (e) {
            pending.resolve();
            if (e.name === 'AbortError') {
                return null;
            }
            showPreflightRequestError(`预检测请求失败: ${e.message}`);
            return null;
        }
    }

    globalThis.isCliOnlySpdSource = function isCliOnlySpdSource(variant, methodsSubdir) {
        return String(methodsSubdir || '') === 'methods' && String(variant || '') === 'spd';
    }

    globalThis.currentTrainingConfigIsRuntime = function currentTrainingConfigIsRuntime() {
        return currentTrainingConfigFile().replace(/\\/g, '/').endsWith('/config.runtime.toml');
    }

    globalThis.chooseTrainingLaunchMode = async function chooseTrainingLaunchMode(options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const isRunning = isLiveRunningState();
        const sourceDetail = continueTrainingSource
            ? `\n\n训练来源: 继续训练 ${continueTrainingSource.kind} · ${continueTrainingSource.name}\n基于权重: ${continueTrainingSource.abs_path}`
            : '\n\n训练来源: 从零开始';
        if (isRunning) {
            const ok = await showAppConfirmDialog({
                title: '加入训练队列',
                description: '当前已有任务在运行',
                message: `确认后会冻结当前配置，并加入队列等待自动执行。${sourceDetail}`,
                confirmText: '加入队列',
                cancelText: '取消',
            });
            return ok ? 'queue' : 'cancel';
        }
        const startNow = await showAppConfirmDialog({
            title: willAutoPreprocess ? '最终确认：预处理并训练' : '最终确认：开始训练',
            description: '可以立即启动，也可以先加入队列',
            message: willAutoPreprocess
                ? `确认后会立即创建本次运行目录并启动预处理。${sourceDetail}`
                : `确认后会立即创建本次运行目录并启动训练进程。${sourceDetail}`,
            confirmText: willAutoPreprocess ? '立即预处理并训练' : '立即开始训练',
            cancelText: '不立即启动',
        });
        if (startNow) return 'start';
        const queue = await showAppConfirmDialog({
            title: '加入训练队列',
            description: '冻结当前配置并等待手动继续',
            message: `确认后会创建独立运行配置并加入队列；队列会保持暂停，等待你手动继续。${sourceDetail}`,
            confirmText: '加入队列',
            cancelText: '取消',
        });
        return queue ? 'queue' : 'cancel';
    }

    globalThis.confirmTrainingLaunch = async function confirmTrainingLaunch(options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const sourceDetail = continueTrainingSource
            ? `\n\n训练来源: 继续训练 ${continueTrainingSource.kind} · ${continueTrainingSource.name}\n基于权重: ${continueTrainingSource.abs_path}`
            : '\n\n训练来源: 从零开始';
        return showAppConfirmDialog({
            title: willAutoPreprocess ? '最终确认：预处理并训练' : '最终确认：开始训练',
            description: '训练启动前的最后一步',
            message: willAutoPreprocess
                ? `确认后会立即创建本次运行目录并启动预处理；预处理完成后会自动开始训练。${sourceDetail}`
                : `确认后会立即创建本次运行目录并启动训练进程。${sourceDetail}`,
            confirmText: willAutoPreprocess ? '确认预处理并训练' : '确认开始训练',
            cancelText: '返回检查',
        });
    }

    globalThis.startTrainingUnchecked = async function startTrainingUnchecked(variant, preset, methodsSubdir, options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const mode = await chooseTrainingLaunchMode({ willAutoPreprocess });
        if (mode === 'cancel') return;
        if (mode === 'queue') {
            await enqueueTrainingFromConfig(variant, preset, methodsSubdir, { willAutoPreprocess });
            return;
        }
        renderPreflightPending({
            title: willAutoPreprocess ? '启动预处理后训练' : '启动训练',
            message: willAutoPreprocess
                ? '正在创建运行目录并启动预处理...'
                : '正在创建运行目录并启动训练...',
            detail: willAutoPreprocess
                ? '预处理完成后会自动开始训练；成功后会自动切换到训练页。'
                : '后端正在准备训练进程；启动成功后会自动切换到训练页。',
        });
        try {
            const res = await api('/api/training/start', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                    extra_args: [],
                    gpu_whitelist: gpuPicker.selectedGpuPayload(),
                    confirmed: true,
                    confirm_preprocess: willAutoPreprocess,
                    ...continueTrainingRequestPayload(),
                }),
            });
            if (res.ok) {
                const dialog = document.getElementById('preflight-dialog');
                if (dialog?.open) dialog.close('training-started');
                enterLiveTrainingForNewRun();
                appendLog(`[状态] ${res.message || '任务已启动'}`);
            } else {
                if (res.preflight) {
                    const action = await showPreflightDialog(res.preflight, false);
                    if (action === 'preprocess') {
                        await startPreprocessFromPreflight(res.preflight);
                    }
                } else {
                    showPreflightRequestError(res.error || '启动失败');
                }
            }
        } catch (e) {
            showPreflightRequestError('请求失败: ' + e.message);
        }
    }

    globalThis.enqueueTrainingFromConfig = async function enqueueTrainingFromConfig(variant, preset, methodsSubdir, options = {}) {
        return ensureQueueFeature().enqueueTrainingFromConfig(variant, preset, methodsSubdir, options);
    }

    globalThis.enqueueTrainingQueueRequest = async function enqueueTrainingQueueRequest(options = {}) {
        return ensureQueueFeature().enqueueTrainingQueueRequest(options);
    }

    globalThis.enqueueTrainingQueueBatchRequest = async function enqueueTrainingQueueBatchRequest(options = {}) {
        return ensureQueueFeature().enqueueTrainingQueueBatchRequest(options);
    }

    globalThis.enterLiveTrainingForNewRun = function enterLiveTrainingForNewRun() {
        returnToLiveTraining({ refresh: false });
        document.querySelector('[data-tab="training"]')?.click();
        recoverLiveTrainingState();
    }

    globalThis.showPreflightDialog = function showPreflightDialog(result, allowContinue, options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog) {
            if (!allowContinue) return Promise.resolve('cancel');
            const confirmText = options.willAutoPreprocess ? '确认预处理并训练' : '确认开始训练';
            return showAppConfirmDialog({
                title: '训练前预检测',
                description: '检测到训练前提示',
                message: `${preflightPlainText(result)}\n\n是否继续训练？`,
                confirmText,
            }).then((ok) => ok ? 'continue' : 'cancel');
        }
        renderPreflightResult(result, allowContinue, options);
        if (!dialog.open) dialog.showModal();
        return new Promise((resolve) => {
            dialog.addEventListener('close', () => {
                resolve(dialog.returnValue || 'cancel');
            }, { once: true });
        });
    }
