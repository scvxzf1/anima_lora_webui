/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.loadOutputRunConfig = async function loadOutputRunConfig(runName, kind = 'original') {
        const run = outputRunState.runs.find((item) => item.name === runName);
        if (!run) {
            outputRunState = { ...outputRunState, selectedRun: '', content: '', file: '' };
            renderOutputRunManager();
            return;
        }
        const available = new Set((run.files || []).map((item) => item.kind));
        const selectedKind = available.has(kind) ? kind : preferredOutputRunKind(runName);
        outputRunState = {
            ...outputRunState,
            selectedRun: runName,
            selectedKind,
            content: '读取中...',
            file: '',
            saveAsOpen: false,
            error: '',
        };
        renderOutputRunManager();
        try {
            const data = await api(`/api/config/output-runs/read?run=${encodeURIComponent(runName)}&kind=${encodeURIComponent(selectedKind)}`);
            if (!data.ok) throw new Error(data.error || '读取运行配置失败');
            outputRunState = {
                ...outputRunState,
                selectedRun: data.run || runName,
                selectedKind: data.kind || selectedKind,
                content: data.content || '',
                file: data.file || '',
                error: '',
            };
            renderOutputRunManager();
            setTomlStatus('', '');
        } catch (e) {
            outputRunState = {
                ...outputRunState,
                content: '',
                file: '',
                error: e.message,
            };
            renderOutputRunManager();
            setTomlStatus('error', '读取运行配置失败: ' + e.message);
        }
    }

    globalThis.preferredOutputRunKind = function preferredOutputRunKind(runName = outputRunState.selectedRun) {
        const run = outputRunState.runs.find((item) => item.name === runName);
        const kinds = (run?.files || []).map((item) => item.kind);
        if (kinds.includes(outputRunState.selectedKind)) return outputRunState.selectedKind;
        if (kinds.includes('original')) return 'original';
        if (kinds.includes('runtime')) return 'runtime';
        if (kinds.includes('dataset')) return 'dataset';
        return 'original';
    }

    globalThis.renderOutputRunManager = function renderOutputRunManager() {
        renderOutputRunList();
        renderOutputRunDetail();
        updateOutputRunActionState();
        updateConfigPageSummary('output');
        if (tomlManagerMode === 'output') {
            updateOutputRunSelectionUI();
        }
    }

    globalThis.renderOutputRunList = function renderOutputRunList() {
        const container = document.getElementById('output-run-list');
        if (!container) return;
        container.innerHTML = '';
        if (outputRunState.loading) {
            const loading = document.createElement('div');
            loading.className = 'output-run-empty';
            loading.textContent = '正在读取全局输出文件夹...';
            container.appendChild(loading);
            return;
        }
        if (outputRunState.error) {
            const error = document.createElement('div');
            error.className = 'output-run-empty error';
            error.textContent = outputRunState.error;
            container.appendChild(error);
            return;
        }
        const runs = filteredOutputRuns();
        if (!runs.length) {
            const empty = document.createElement('div');
            empty.className = 'output-run-empty';
            empty.textContent = outputRunState.search
                ? '没有匹配的训练输出配置。'
                : `没有在 ${outputRunState.outputRoot || '输出文件夹'} 找到训练配置。`;
            container.appendChild(empty);
            return;
        }
        for (const run of runs) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'output-run-item';
            btn.classList.toggle('active', run.name === outputRunState.selectedRun);
            btn.dataset.run = run.name;
            btn.title = run.path || run.name;
            btn.addEventListener('click', () => loadOutputRunConfig(run.name, preferredOutputRunKind(run.name)));

            const name = document.createElement('strong');
            name.textContent = run.name;
            const meta = document.createElement('span');
            const fileLabels = (run.files || []).map((item) => item.label).join(' / ');
            meta.textContent = [run.mtime_text, fileLabels].filter(Boolean).join(' · ');
            const path = document.createElement('small');
            path.textContent = run.path || '';
            btn.append(name, meta, path);
            container.appendChild(btn);
        }
    }

    globalThis.renderOutputRunDetail = function renderOutputRunDetail() {
        const run = selectedOutputRun();
        const title = document.getElementById('output-run-title');
        const meta = document.getElementById('output-run-meta');
        const tabs = document.getElementById('output-run-kind-tabs');
        const viewer = document.getElementById('output-run-config-viewer');
        const saveAs = document.getElementById('output-run-save-as');
        if (title) title.textContent = run?.name || '未选择运行目录';
        if (meta) {
            meta.textContent = run
                ? [run.path, run.mtime_text].filter(Boolean).join(' · ')
                : `从 ${outputRunState.outputRoot || '全局输出文件夹'} 读取训练快照配置。`;
        }
        if (tabs) {
            tabs.innerHTML = '';
            const files = run?.files || [];
            if (!files.length) {
                const empty = document.createElement('span');
                empty.className = 'output-run-kind-empty';
                empty.textContent = '无可读 TOML';
                tabs.appendChild(empty);
            }
            for (const file of files) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'output-run-kind-btn';
                btn.classList.toggle('active', file.kind === outputRunState.selectedKind);
                btn.textContent = file.label;
                btn.title = file.file || file.filename;
                btn.addEventListener('click', () => loadOutputRunConfig(run.name, file.kind));
                tabs.appendChild(btn);
            }
        }
        if (viewer) {
            viewer.value = outputRunState.content || '';
            viewer.placeholder = run ? '这个运行目录没有可显示的配置内容。' : '选择左侧运行目录后查看配置。';
        }
        if (saveAs) {
            saveAs.hidden = !outputRunState.saveAsOpen;
        }
        renderOutputRunSaveAsControls();
    }

    globalThis.renderOutputRunSaveAsControls = function renderOutputRunSaveAsControls() {
        const input = document.getElementById('output-run-save-name');
        const select = document.getElementById('output-run-save-group');
        if (input && !input.value && outputRunState.saveAsOpen) {
            input.value = outputRunSaveAsDefaultName();
        }
        if (!select) return;
        const current = select.value || 'imported';
        select.innerHTML = '';
        const groups = saveAsTargetGroups();
        for (const group of groups) {
            const opt = document.createElement('option');
            opt.value = group.id;
            opt.textContent = group.label || group.id;
            select.appendChild(opt);
        }
        select.value = groups.some((group) => group.id === current) ? current : (groups[0]?.id || 'imported');
    }

    globalThis.filteredOutputRuns = function filteredOutputRuns() {
        const query = outputRunState.search.trim().toLowerCase();
        if (!query) return outputRunState.runs;
        return outputRunState.runs.filter((run) => {
            const haystack = [
                run.name,
                run.path,
                run.mtime_text,
                ...(run.files || []).map((item) => `${item.kind} ${item.label} ${item.file}`),
            ].join(' ').toLowerCase();
            return haystack.includes(query);
        });
    }

    globalThis.selectedOutputRun = function selectedOutputRun() {
        return outputRunState.runs.find((item) => item.name === outputRunState.selectedRun) || null;
    }

    globalThis.updateOutputRunSelectionUI = function updateOutputRunSelectionUI() {
        const label = document.getElementById('toml-current-file');
        if (label) {
            label.textContent = outputRunState.file || outputRunState.selectedRun || '训练输出配置';
        }
        setBadge('toml-current-badge', false, '当前训练');
        setBadge('toml-trainable-badge', Boolean(outputRunState.file), '只读快照');
        setBadge('toml-lock-badge', Boolean(outputRunState.file), '只读');
        setBadge('toml-dirty-badge', false, '未保存');
    }

    globalThis.updateOutputRunActionState = function updateOutputRunActionState() {
        const run = selectedOutputRun();
        const hasContent = Boolean(outputRunState.content && outputRunState.file);
        const hasOriginal = Boolean(run?.has_original);
        setButtonDisabled('btn-copy-output-config', !hasContent);
        setButtonDisabled('btn-export-output-config', !hasContent);
        const saveBtn = document.getElementById('btn-save-output-config-as');
        if (saveBtn) {
            saveBtn.disabled = !run || !hasOriginal;
            saveBtn.title = !run
                ? '请先选择一个训练运行目录'
                : (hasOriginal ? '把 config.original.toml 复制到 configs/imported，随后可在项目预设中编辑。' : '这个运行目录没有 config.original.toml，不能复制为项目预设。');
        }
    }

    globalThis.setButtonDisabled = function setButtonDisabled(id, disabled) {
        ctx.dom.setButtonDisabled(id, disabled);
    }

    globalThis.copyOutputRunConfigContent = async function copyOutputRunConfigContent() {
        const text = outputRunState.content || '';
        if (!text) return;
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
            } else {
                const viewer = document.getElementById('output-run-config-viewer');
                viewer?.focus();
                viewer?.select();
                document.execCommand('copy');
            }
            setTomlStatus('ok', '已复制训练输出配置内容');
        } catch (e) {
            setTomlStatus('error', '复制失败: ' + e.message);
        }
    }

    globalThis.exportOutputRunConfig = function exportOutputRunConfig() {
        if (!outputRunState.content) return;
        const filename = outputRunState.file
            ? outputRunState.file.split('/').pop()
            : `${outputRunState.selectedRun || 'output-run'}.${outputRunState.selectedKind}.toml`;
        const blob = new Blob([outputRunState.content], { type: 'application/toml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename || 'output-run-config.toml';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setTomlStatus('ok', `已导出 ${filename}`);
    }

    globalThis.openOutputRunSaveAs = function openOutputRunSaveAs() {
        const run = selectedOutputRun();
        if (!run) {
            setTomlStatus('error', '请先选择一个训练运行目录');
            return;
        }
        if (!run.has_original) {
            setTomlStatus('error', '这个运行目录没有 config.original.toml，不能复制为项目预设');
            return;
        }
        outputRunState = { ...outputRunState, saveAsOpen: true };
        renderOutputRunManager();
        const input = document.getElementById('output-run-save-name');
        if (input) {
            input.value = outputRunSaveAsDefaultName();
            input.focus();
            input.select();
        }
    }

    globalThis.closeOutputRunSaveAs = function closeOutputRunSaveAs() {
        outputRunState = { ...outputRunState, saveAsOpen: false };
        renderOutputRunManager();
    }

    globalThis.outputRunSaveAsDefaultName = function outputRunSaveAsDefaultName() {
        const run = selectedOutputRun();
        const stem = String(run?.name || 'output_run')
            .replace(/-\d{8}-\d{6}(?:-\d+)?$/i, '')
            .replace(/[^A-Za-z0-9_.-]+/g, '_')
            .replace(/^_+|_+$/g, '') || 'output_run';
        return `${stem}_from_output`;
    }

    globalThis.confirmOutputRunSaveAs = async function confirmOutputRunSaveAs() {
        const run = selectedOutputRun();
        if (!run) return;
        const name = val('output-run-save-name') || outputRunSaveAsDefaultName();
        const group = val('output-run-save-group') || 'imported';
        try {
            const res = await api('/api/config/output-runs/save-as', {
                method: 'POST',
                body: JSON.stringify({
                    run: run.name,
                    name,
                    target_group: group,
                }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '复制为项目预设失败');
                return;
            }
            outputRunState = { ...outputRunState, saveAsOpen: false };
            if (hasPendingConfigChanges(currentTomlFile)) {
                if (!(await confirmDiscardTomlChanges('复制完成后会回到项目预设并加载新文件，当前项目预设里已有未保存修改。是否继续？'))) {
                    return;
                }
            }
            await switchTomlManagerMode('project');
            await loadTomlFileList(res.file, { force: true });
            await loadTomlFile(res.file, { force: true });
            setTomlStatus('ok', `已复制为新项目预设: ${res.file}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '复制为项目预设失败: ' + e.message);
        }
    }

    globalThis.selectAndApplyTomlFile = async function selectAndApplyTomlFile(filePath) {
        if (!filePath) return false;
        const previousFile = currentTomlFile;
        const previousSelect = document.getElementById('toml-file-select');
        const targetLabel = tomlFileDisplayName(filePath);
        const canSwitch = await handlePendingConfigSwitch({ targetLabel });
        if (!canSwitch) {
            if (previousSelect) previousSelect.value = previousFile || '';
            updateTomlSelectionUI(previousFile);
            return false;
        }
        await loadTomlFile(filePath, { force: true });
        const meta = tomlFileMeta[filePath];
        if (!meta?.trainable) {
            setTomlStatus('error', '已打开该配置文件，但它不是完整训练配置，不能加载为当前训练入口');
            return false;
        }
        await applyTomlToConfig({ silent: true });
        rememberSelectionSnapshot();
        setTomlStatus('ok', `已加载选中配置: ${meta.path || filePath}`);
        return true;
    }

    globalThis.loadTomlFile = async function loadTomlFile(filePath, options = {}) {
        if (!options.force && !(await confirmDiscardTomlChanges('当前 TOML 有未保存修改，切换文件会丢失这些修改。是否继续？'))) {
            const select = document.getElementById('toml-file-select');
            if (select) select.value = currentTomlFile || '';
            return;
        }
        resetTomlDeleteConfirm();
        resetTomlSaveConfirm();
        const data = await api(`/api/config/raw?file=${encodeURIComponent(filePath)}`);
        if (data?.ok === false) {
            if (isMissingTomlFileResponse(data)) {
                await handleDeletedTomlSelection(filePath, data.error || '配置文件不存在或已被删除');
                return;
            }
            setTomlStatus('error', data.error || '读取配置文件失败');
            return;
        }
        currentTomlFile = filePath;
        document.getElementById('toml-file-select').value = filePath;
        tomlSavedContent = data.content || '';
        document.getElementById('toml-editor').value = tomlSavedContent;
        if (data.meta) tomlFileMeta[filePath] = data.meta;
        updateTomlSelectionUI(filePath);
        applyTomlLockState(filePath);
        updateTomlDirtyState();
        setTomlStatus('', '');
    }

    globalThis.saveTomlFile = async function saveTomlFile(options = {}) {
        const file = currentTomlFile || val('toml-file-select');
        if (!file) {
            setTomlStatus('error', '请先选择一个配置文件，或使用“另存新配置”保存导入内容');
            return false;
        }
        if (isTomlLocked(file)) {
            setTomlStatus('error', '该配置文件已锁定，请使用“另存新配置”创建可编辑配置');
            return false;
        }
        const editorDirty = isTomlDirty();
        const formDirty = hasUnsavedFormChanges(file);
        const directEditorSave = options.mode === 'editor';
        if (directEditorSave) {
            if (formDirty) {
                setTomlStatus('error', '左侧表单或数据集预设选择有未保存修改，请先使用“保存更新当前选中配置”处理后再直接保存 TOML');
                updateTomlActionState(file);
                return false;
            }
            if (!editorDirty) {
                setTomlStatus('error', '直接编辑器没有未保存的 TOML 文本修改');
                return false;
            }
            if (!options.skipConfirm && tomlSaveConfirmFile !== file) {
                armTomlSaveConfirm(file);
                return false;
            }
            resetTomlSaveConfirm({ update: false });
            return await saveRawTomlContent(file, document.getElementById('toml-editor').value, { reloadConfig: currentTrainingSource.file === file });
        }
        if (editorDirty && !formDirty && !options.skipConfirm && tomlSaveConfirmFile !== file) {
            armTomlSaveConfirm(file);
            return false;
        }
        resetTomlSaveConfirm({ update: false });
        if (currentTrainingSource.file === file) {
            const datasetApplied = await applySelectedDatasetPresetToCurrentConfig(file);
            if (!datasetApplied) return false;
            const datasetWasDirty = datasetEditorState.dirty;
            if (datasetWasDirty) {
                const datasetSaved = await saveDatasetEditor({ trainFile: file, reloadList: false });
                if (!datasetSaved) return false;
            }
            const changedValues = collectChangedFormValues({ persistDefaultFields: true });
            if (Object.keys(changedValues).length > 0) {
                return await saveFormPatchToToml(file, changedValues);
            }
            if (datasetApplied.applied || datasetWasDirty) {
                await loadConfig();
                await loadTomlFileList(file);
                updateTomlDirtyState();
                setTomlStatus('ok', datasetWasDirty ? '✓ 已保存数据集修改' : '✓ 已应用数据集预设');
                return true;
            }
        }
        const content = document.getElementById('toml-editor').value;
        return await saveRawTomlContent(file, content, { reloadConfig: currentTrainingSource.file === file });
    }

    globalThis.saveRawTomlContent = async function saveRawTomlContent(file, content, options = {}) {
        try {
            const res = await api('/api/config/raw', {
                method: 'PUT',
                body: JSON.stringify({ file, content }),
            });
            if (res.ok) {
                tomlSavedContent = content;
                resetTomlSaveConfirm({ update: false });
                updateTomlDirtyState();
                setTomlStatus('ok', '✓ 已保存');
                await loadTomlFileList(file);
                if (options.reloadConfig) {
                    await loadConfig(); // 仅当前训练源被保存时刷新左侧表单
                }
                return true;
            } else {
                setTomlStatus('error', res.error || '保存失败');
                return false;
            }
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
            return false;
        }
    }

    globalThis.saveFormPatchToToml = async function saveFormPatchToToml(file, values) {
        try {
            const content = document.getElementById('toml-editor').value;
            const preparedValues = await prepareFormPatchValues(values);
            const res = await api('/api/config/raw', {
                method: 'PATCH',
                body: JSON.stringify({ file, values: preparedValues, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '保存失败');
                return false;
            }

            if (typeof res.content === 'string') {
                document.getElementById('toml-editor').value = res.content;
                tomlSavedContent = res.content;
            }
            resetTomlSaveConfirm({ update: false });
            await loadConfig();
            await loadTomlFileList(file);
            updateTomlDirtyState();
            setTomlStatus('ok', `✓ 已保存 ${res.changed?.length || Object.keys(preparedValues).length} 个表单修改`);
            return true;
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
            return false;
        }
    }
