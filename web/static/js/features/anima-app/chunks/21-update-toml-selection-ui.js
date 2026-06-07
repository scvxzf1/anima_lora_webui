/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.updateTomlSelectionUI = function updateTomlSelectionUI(filePath) {
        document.querySelectorAll('.toml-file-item').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.file === filePath);
        });
        const label = document.getElementById('toml-current-file');
        if (label) label.textContent = filePath ? tomlFileDisplayName(filePath) : '未保存导入内容';
        const applyBtn = document.getElementById('btn-apply-toml');
        if (applyBtn) {
            const meta = tomlFileMeta[filePath];
            const dirty = hasPendingConfigChanges(filePath);
            applyBtn.disabled = !meta?.trainable || dirty;
            applyBtn.title = dirty
                ? '当前配置尚未保存，请先保存或另存为'
                : (meta?.trainable ? '将该配置作为当前表单和训练入口' : '该文件不是完整训练配置');
        }
        updateTomlBadges(filePath);
    }

    globalThis.isTomlDirty = function isTomlDirty() {
        const editor = document.getElementById('toml-editor');
        if (!editor) return false;
        return editor.value !== tomlSavedContent;
    }

    globalThis.hasUnsavedFormChanges = function hasUnsavedFormChanges(filePath = currentTomlFile) {
        if (!filePath || currentTrainingSource.file !== filePath) return false;
        if (!currentConfig || Object.keys(currentConfig).length === 0) return false;
        return datasetEditorState.dirty
            || selectedConfigDatasetFile !== (currentConfig.dataset_config || '')
            || Object.keys(collectChangedFormValues()).length > 0;
    }

    globalThis.hasPendingConfigChanges = function hasPendingConfigChanges(filePath = currentTomlFile) {
        return isTomlDirty() || hasUnsavedFormChanges(filePath);
    }

    globalThis.confirmDiscardTomlChanges = async function confirmDiscardTomlChanges(message) {
        if (!hasPendingConfigChanges(currentTomlFile)) return true;
        return confirmUnsavedDiscard(message);
    }

    globalThis.confirmUnsavedDiscard = function confirmUnsavedDiscard(message) {
        return showAppConfirmDialog({
            title: '未保存更改',
            description: '当前页面有尚未保存的修改',
            message,
            confirmText: '继续并丢弃',
            cancelText: '留在当前页面',
            danger: true,
        });
    }

    globalThis.collectPendingConfigChangeDetails = function collectPendingConfigChangeDetails(pending = pendingConfigSwitchState()) {
        const changes = [];
        if (pending.formDirty) {
            if (selectedConfigDatasetFile !== (currentConfig.dataset_config || '')) {
                changes.push({
                    label: '数据集预设 / dataset_config',
                    original: currentConfig.dataset_config || '未设置',
                    next: selectedConfigDatasetFile || '未设置',
                });
            }
            for (const [key, nextValue] of Object.entries(collectChangedFormValues())) {
                changes.push({
                    label: formatFieldName(key),
                    original: originalValueForChange(key),
                    next: nextValue,
                });
            }
            if (datasetEditorState.dirty) {
                changes.push({
                    label: '多数据集路径与参数',
                    original: currentConfig.dataset_config || '当前配置内的数据集字段',
                    next: summarizeDatasetEditorState(datasetEditorState),
                });
            }
        }
        if (pending.editorDirty) {
            const editorValue = document.getElementById('toml-editor')?.value || '';
            changes.push({
                label: '直接编辑 TOML',
                original: summarizeTextChange(tomlSavedContent),
                next: summarizeTextChange(editorValue),
            });
        }
        return changes;
    }

    globalThis.originalValueForChange = function originalValueForChange(key) {
        if (key === 'sample_prompts' && samplePromptsMode !== 'path') {
            return samplePromptsContent || '';
        }
        if (isActiveNetworkArgFieldKey(key)) {
            return networkArgFieldValueFromConfig(NETWORK_ARG_FIELD_MAP.get(key), currentConfig);
        }
        if (key in currentConfig) return currentConfig[key];
        return FORM_UI_DEFAULTS[key];
    }

    globalThis.summarizeDatasetEditorState = function summarizeDatasetEditorState(state) {
        const rows = normalizeDatasetEditorRows(state.datasets || []);
        const parts = rows.map((row, index) => {
            const settings = normalizeDatasetDefaults(row.settings || state.defaults || {});
            const triggerClone = normalizeTriggerClone(row.trigger_clone);
            return [
                `第 ${index + 1} 组`,
                row.source_dir || '未设置原始路径',
                `重复 ${row.num_repeats || 1}`,
                `${settings.resolution}px`,
                `标注 ${captionSourceModeLabel(settings.caption_source_mode)}`,
                triggerClone.enabled ? `触发克隆 x${triggerClone.num_repeats}` : '',
            ].filter(Boolean).join(' · ');
        });
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        parts.push(`通用 keep_tokens ${defaults.keep_tokens}`);
        return parts.join('\n');
    }

    globalThis.summarizeTextChange = function summarizeTextChange(text) {
        const value = String(text || '');
        const lines = value.split(/\r?\n/).length;
        const chars = value.length;
        const preview = value.split(/\r?\n/).find((line) => line.trim()) || '空内容';
        return `${lines} 行 / ${chars} 字符\n${preview}`;
    }

    globalThis.formatConfigChangeValue = function formatConfigChangeValue(value) {
        let text;
        if (typeof value === 'string') {
            text = value;
        } else {
            try {
                text = JSON.stringify(value, null, 2);
            } catch {
                text = String(value);
            }
        }
        if (text === '') text = '空';
        return text.length > 600 ? `${text.slice(0, 600)}\n...` : text;
    }

    globalThis.showConfigSwitchToast = function showConfigSwitchToast(filePath, stateText) {
        const toast = document.getElementById('config-switch-toast');
        if (!toast) return;
        if (configSwitchToastTimer) {
            clearTimeout(configSwitchToastTimer);
            configSwitchToastTimer = null;
        }
        const file = (filePath || currentTomlFile || '当前配置').split('/').pop() || '当前配置';
        toast.textContent = `${file}，${stateText}`;
        toast.hidden = false;
        configSwitchToastTimer = setTimeout(() => {
            toast.hidden = true;
            configSwitchToastTimer = null;
        }, 2000);
    }

    globalThis.handlePendingConfigSwitch = async function handlePendingConfigSwitch({ targetLabel = '' } = {}) {
        const pending = pendingConfigSwitchState();
        if (!pending.hasChanges) return true;
        const action = await showUnsavedConfigSwitchDialog({ pending, targetLabel });
        if (action === 'cancel') return false;
        if (action === 'discard') {
            showConfigSwitchToast(pendingToastLabel(pending), '更改未保存');
            return true;
        }
        const saved = await savePendingConfigSwitchChanges(pending);
        if (!saved) return false;
        showConfigSwitchToast(pendingToastLabel(pending), '更改保存完成');
        return true;
    }

    globalThis.pendingConfigSwitchState = function pendingConfigSwitchState() {
        const editorFile = currentTomlFile || val('toml-file-select') || '';
        const formFile = currentTrainingSource.file || '';
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(formFile);
        const dirtyFiles = [];
        if (editorDirty && editorFile) dirtyFiles.push(editorFile);
        if (formDirty && formFile && !dirtyFiles.includes(formFile)) dirtyFiles.push(formFile);
        const canSave = dirtyFiles.length > 0 && dirtyFiles.every((file) => file && !isTomlLocked(file));
        return {
            editorDirty,
            editorFile,
            formDirty,
            formFile,
            dirtyFiles,
            sourceFile: dirtyFiles[0] || formFile || editorFile || '',
            hasChanges: editorDirty || formDirty,
            canSave,
        };
    }

    globalThis.pendingToastLabel = function pendingToastLabel(pending) {
        const files = pending?.dirtyFiles || [];
        if (files.length > 1) {
            const first = files[0].split('/').pop() || files[0];
            return `${first} 等 ${files.length} 个配置`;
        }
        return pending?.sourceFile || currentTomlFile || '当前配置';
    }

    globalThis.sharedHistoryTaskDialogParts = function sharedHistoryTaskDialogParts() {
        const dialog = document.getElementById('history-task-dialog');
        const title = document.getElementById('history-task-dialog-title');
        const desc = document.getElementById('history-task-dialog-desc');
        const body = document.getElementById('history-task-dialog-body');
        const cancelBtn = document.getElementById('history-task-dialog-cancel');
        const confirmBtn = document.getElementById('history-task-dialog-confirm');
        const closeBtn = dialog?.querySelector('.history-task-dialog-header button[value="cancel"]');
        const form = dialog?.querySelector('form');
        if (!dialog || !title || !desc || !body || !cancelBtn || !confirmBtn) return null;
        return { dialog, title, desc, body, cancelBtn, confirmBtn, closeBtn, form };
    }

    globalThis.sharedHistoryTaskDialogIsOpen = function sharedHistoryTaskDialogIsOpen(dialog) {
        return Boolean(dialog?.open || dialog?.hasAttribute?.('open'));
    }

    globalThis.openSharedHistoryTaskDialog = function openSharedHistoryTaskDialog(dialog) {
        document.body.classList.remove('history-task-dialog-fallback-open');
        if (typeof dialog.showModal === 'function') {
            try {
                dialog.showModal();
                return;
            } catch (e) {
                if (sharedHistoryTaskDialogIsOpen(dialog)) return;
            }
        }
        dialog.setAttribute('open', 'open');
        dialog.setAttribute('role', 'dialog');
        dialog.setAttribute('aria-modal', 'true');
        document.body.classList.add('history-task-dialog-fallback-open');
    }

    globalThis.closeSharedHistoryTaskDialog = function closeSharedHistoryTaskDialog(dialog, value, fallbackClose) {
        dialog.returnValue = value || '';
        if (typeof dialog.close === 'function' && sharedHistoryTaskDialogIsOpen(dialog)) {
            try {
                dialog.close(dialog.returnValue);
                return;
            } catch (e) {
                /* 部分浏览器的 dialog close 可能在 fallback 状态下抛错，继续手动关闭。 */
            }
        }
        dialog.removeAttribute('open');
        fallbackClose();
    }

    globalThis.savePendingConfigSwitchChanges = async function savePendingConfigSwitchChanges(pending) {
        if (pending.editorDirty) {
            const savedEditor = await saveTomlFile({ skipConfirm: true, source: 'switch' });
            if (!savedEditor) return false;
        }
        if (pending.formDirty && (!pending.editorDirty || pending.formFile !== pending.editorFile)) {
            if (currentTomlFile !== pending.formFile) {
                await loadTomlFile(pending.formFile, { force: true });
            }
            const savedForm = await saveTomlFile({ skipConfirm: true, source: 'switch' });
            if (!savedForm) return false;
        }
        return true;
    }

    globalThis.showUnsavedConfigSwitchDialog = function showUnsavedConfigSwitchDialog({ pending = pendingConfigSwitchState(), targetLabel = '' } = {}) {
        const parts = sharedHistoryTaskDialogParts();
        if (!parts) {
            return Promise.resolve('cancel');
        }
        const { dialog, title, desc, body, cancelBtn, confirmBtn, closeBtn, form } = parts;
        if (sharedDialogBusy || sharedHistoryTaskDialogIsOpen(dialog)) {
            return Promise.resolve('cancel');
        }
        sharedDialogBusy = true;

        title.textContent = '有更改待保存';
        desc.textContent = targetLabel ? `即将切换到 ${targetLabel}` : '即将切换配置';
        body.innerHTML = '';
        body.appendChild(createConfigSwitchDialogBody(pending));
        cancelBtn.textContent = '放弃未保存的更改';
        cancelBtn.value = 'discard';
        confirmBtn.textContent = '保存更改并切换';
        confirmBtn.value = 'save';
        confirmBtn.disabled = !pending.canSave;
        confirmBtn.title = pending.canSave ? '' : '存在只读配置，不能直接保存；请先另存为可编辑配置，或放弃未保存更改后切换。';
        confirmBtn.classList.remove('btn-danger');
        confirmBtn.classList.add('btn-primary');
        cancelBtn.hidden = false;
        dialog.returnValue = '';

        return new Promise((resolve) => {
            let settled = false;
            const closeClick = (event) => {
                event.preventDefault();
                event.stopPropagation();
                closeSharedHistoryTaskDialog(dialog, event.currentTarget?.value || 'cancel', handleClose);
            };
            const submitDialog = (event) => {
                event.preventDefault();
                const value = event.submitter?.value || (confirmBtn.disabled ? 'cancel' : 'save');
                if (value === 'save' && confirmBtn.disabled) return;
                closeSharedHistoryTaskDialog(dialog, value, handleClose);
            };
            const keydownDialog = (event) => {
                if (event.key !== 'Escape') return;
                event.preventDefault();
                closeSharedHistoryTaskDialog(dialog, 'cancel', handleClose);
            };
            const cleanup = () => {
                dialog.removeEventListener('close', handleClose);
                form?.removeEventListener('submit', submitDialog);
                closeBtn?.removeEventListener('click', closeClick);
                cancelBtn.removeEventListener('click', closeClick);
                confirmBtn.removeEventListener('click', closeClick);
                dialog.removeEventListener('keydown', keydownDialog);
                document.body.classList.remove('history-task-dialog-fallback-open');
                sharedDialogBusy = false;
                cancelBtn.hidden = false;
                cancelBtn.value = 'cancel';
                confirmBtn.value = 'confirm';
                confirmBtn.title = '';
                if (closeBtn) closeBtn.value = 'cancel';
            };
            const handleClose = () => {
                if (settled) return;
                settled = true;
                const action = dialog.returnValue === 'save'
                    ? 'save'
                    : (dialog.returnValue === 'discard' ? 'discard' : 'cancel');
                cleanup();
                resolve(action);
            };
            dialog.addEventListener('close', handleClose);
            form?.addEventListener('submit', submitDialog);
            closeBtn?.addEventListener('click', closeClick);
            cancelBtn.addEventListener('click', closeClick);
            confirmBtn.addEventListener('click', closeClick);
            dialog.addEventListener('keydown', keydownDialog);
            try {
                openSharedHistoryTaskDialog(dialog);
            } catch {
                cleanup();
                resolve('cancel');
                return;
            }
            requestAnimationFrame(() => cancelBtn.focus());
        });
    }

    globalThis.createConfigSwitchDialogBody = function createConfigSwitchDialogBody(pending = pendingConfigSwitchState()) {
        const wrap = document.createElement('div');
        wrap.className = 'config-switch-dialog-body';

        const intro = document.createElement('p');
        intro.textContent = '当前配置有未保存修改。请先选择保存后切换，或放弃这些修改后继续切换。';
        wrap.appendChild(intro);

        const list = document.createElement('div');
        list.className = 'config-switch-change-list';
        const changes = collectPendingConfigChangeDetails(pending);
        if (!changes.length) {
            const empty = document.createElement('p');
            empty.textContent = '检测到未保存状态，但没有可展示的字段差异。';
            list.appendChild(empty);
        }
        for (const change of changes) {
            const item = document.createElement('article');
            item.className = 'config-switch-change-item';

            const label = document.createElement('strong');
            label.textContent = change.label;
            item.appendChild(label);

            const values = document.createElement('div');
            values.className = 'config-switch-change-values';
            values.appendChild(createConfigSwitchChangeValue('原始', change.original));
            values.appendChild(createConfigSwitchChangeValue('未保存的更改', change.next));
            item.appendChild(values);
            list.appendChild(item);
        }
        wrap.appendChild(list);
        return wrap;
    }

    globalThis.createConfigSwitchChangeValue = function createConfigSwitchChangeValue(labelText, value) {
        const box = document.createElement('div');
        box.className = 'config-switch-change-value';
        const label = document.createElement('span');
        label.textContent = labelText;
        const code = document.createElement('code');
        code.textContent = formatConfigChangeValue(value);
        box.append(label, code);
        return box;
    }

    globalThis.showAppConfirmDialog = function showAppConfirmDialog(options) {
        return showHistoryTaskConfirmDialog({
            title: options.title || '确认操作',
            description: options.description || '',
            message: options.message || '',
            confirmText: options.confirmText || '确认',
            cancelText: options.cancelText || '取消',
            danger: options.danger,
        }).then(Boolean);
    }

    globalThis.updateTomlDirtyState = function updateTomlDirtyState() {
        if (!hasPendingConfigChanges(currentTomlFile)) {
            resetTomlSaveConfirm({ update: false });
        }
        updateChangedFieldMarks();
        updateTomlBadges(currentTomlFile);
        updateTomlActionState(currentTomlFile);
    }

    globalThis.updateChangedFieldMarks = function updateChangedFieldMarks() {
        let changedCount = 0;
        document.querySelectorAll('#config-form .field-input[data-key]').forEach((input) => {
            const changed = configFieldInputChanged(input);
            input.closest('.field-row')?.classList.toggle('field-row-changed', changed);
            if (changed) changedCount += 1;
        });
        for (const [key, value] of configFormState.draftValues.entries()) {
            if (!document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`)
                && configDraftValueChanged(key, value)) {
                changedCount += 1;
            }
        }
        if (selectedConfigDatasetFile !== (currentConfig.dataset_config || '')) {
            changedCount += 1;
        }
        const count = document.getElementById('config-modified-count');
        if (count) count.textContent = String(changedCount);
    }

    globalThis.configFieldInputChanged = function configFieldInputChanged(input) {
        const key = input?.dataset?.key;
        if (!key || CONFIG_FORM_INTERNAL_KEYS.has(key)) return false;
        const original = originalConfigFieldValue(key);
        const next = readFieldInputValue(input, original);
        return configDraftValueChanged(key, next, original);
    }

    globalThis.updateTomlBadges = function updateTomlBadges(filePath) {
        const meta = tomlFileMeta[filePath];
        setBadge('toml-current-badge', Boolean(filePath && currentTrainingSource.file === filePath), '当前训练');
        setBadge('toml-trainable-badge', Boolean(filePath), meta?.trainable ? '可训练' : '非训练');
        setBadge('toml-lock-badge', Boolean(meta?.locked), tomlLockLabel(meta) || '只读');
        setBadge('toml-dirty-badge', hasPendingConfigChanges(filePath), '未保存');
    }

    globalThis.setBadge = function setBadge(id, visible, text) {
        const badge = document.getElementById(id);
        if (!badge) return;
        badge.hidden = !visible;
        badge.textContent = text;
    }
