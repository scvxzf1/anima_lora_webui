/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    BLANK_PRESET_TEMPLATE_FILE,
    BLANK_PRESET_TEMPLATE_LABEL,
    FORM_UI_DEFAULTS,
} from '../../../config/catalog.js?v=module-bootstrap-20260831-release-v1';
import { DEFAULT_SAMPLE_PROMPTS_PATH } from '../helpers/app-constants.js?v=module-bootstrap-20260831-release-v1';
import { collectChangedFormValues, prepareFormPatchValues } from '../helpers/config-form-bridge.js?v=module-bootstrap-20260831-release-v1';
import { applySelectedDatasetPresetToCurrentConfig } from '../helpers/dataset-preset-actions-bridge.js?v=module-bootstrap-20260831-release-v1';
import { showHistoryTaskDialog } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260831-release-v1';
import { loadTomlFileList } from '../helpers/toml-manager-bridge.js?v=module-bootstrap-20260831-release-v1';
import { setSamplePromptsEditorContent } from './14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260831-release-v1';
import {
    applyTomlLockState,
    applyTomlToConfig,
    setTomlStatus,
    setTomlEditorLocked,
} from '../helpers/toml-action-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getGlobalModelPathOverrides } from '../helpers/global-settings-bridge.js?v=module-bootstrap-20260831-release-v1';
import { confirmDiscardTomlChanges, handlePendingConfigSwitch, updateTomlDirtyState, updateTomlSelectionUI } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getAppContext } from '../helpers/app-context-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTomlState } from '../helpers/toml-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { api, val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260831-release-v1';
import { configureSamplePromptsBridge } from '../helpers/sample-prompts-bridge.js?v=module-bootstrap-20260831-release-v1';
import { configureTomlIoBridge } from '../helpers/toml-io-bridge.js?v=module-bootstrap-20260831-release-v1';

const ctx = getAppContext();
const configState = getConfigState();
const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

    export function currentSamplePromptText(config) {
        const raw = typeof config.sample_prompts === 'string' ? config.sample_prompts.trim() : '';
        const previousMode = configState.samplePromptsMode;
        const previousPath = configState.samplePromptsPath;
        const previousContent = configState.samplePromptsContent;
        configState.samplePromptsPath = DEFAULT_SAMPLE_PROMPTS_PATH;
        configState.samplePromptsContent = '';

        if (!raw) {
            configState.samplePromptsMode = 'editor-inline';
            return FORM_UI_DEFAULTS.sample_prompts;
        }
        if (isEditableSamplePromptsTextFilePath(raw)) {
            const nextPath = normalizeSamplePromptsPath(raw);
            configState.samplePromptsMode = 'editor-file';
            configState.samplePromptsPath = nextPath;
            if (previousMode === 'editor-file' && previousPath === nextPath) {
                configState.samplePromptsContent = previousContent || '';
                return configState.samplePromptsContent || FORM_UI_DEFAULTS.sample_prompts;
            }
            return FORM_UI_DEFAULTS.sample_prompts;
        }
        if (isSamplePromptsFilePath(raw)) {
            configState.samplePromptsMode = 'path';
            return raw;
        }

        configState.samplePromptsMode = 'editor-inline';
        configState.samplePromptsContent = raw;
        return raw;
    }

    export function normalizeSamplePromptsPath(value) {
        return String(value || '').replace(/\\/g, '/').trim();
    }

    export function isEditableSamplePromptsTextFilePath(value) {
        const text = normalizeSamplePromptsPath(value);
        if (!text.toLowerCase().endsWith('.txt')) return false;
        if (!text.startsWith('configs/')) return false;
        return !text.split('/').includes('..');
    }

    export function isSamplePromptsFilePath(value) {
        const text = normalizeSamplePromptsPath(value).toLowerCase();
        return text.endsWith('.txt') || text.endsWith('.toml') || text.endsWith('.json');
    }

    export async function loadSamplePrompts(filePath = configState.samplePromptsPath, parentSeq = configState.configLoadSeq) {
        if (location.protocol === 'file:') return;
        const requestSeq = ++configState.samplePromptsLoadSeq;
        try {
            const data = await api(`/api/config/sample-prompts?file=${encodeURIComponent(filePath || configState.samplePromptsPath)}`);
            if (parentSeq !== configState.configLoadSeq || requestSeq !== configState.samplePromptsLoadSeq) return;
            if (data?.ok === false) {
                throw new Error(data.error || '读取预览提示词失败');
            }
            configState.samplePromptsPath = data.file || configState.samplePromptsPath;
            configState.samplePromptsContent = data.content || '';
            const input = document.querySelector('#config-form .field-input[data-key="sample_prompts"]');
            if (input) {
                if (configState.configFormState.draftValues.has('sample_prompts')) return;
                if (input.classList?.contains('sample-prompts-editor')) {
                    setSamplePromptsEditorContent(input, configState.samplePromptsContent);
                } else {
                    input.value = configState.samplePromptsContent;
                }
            }
        } catch (e) {
            console.warn('读取预览提示词失败:', e);
        }
    }

    export async function saveSamplePrompts(content) {
        const res = await api('/api/config/sample-prompts', {
            method: 'PUT',
            body: JSON.stringify({
                file: configState.samplePromptsPath,
                train_config_file: currentTrainingSourceState().file || tomlState.currentTomlFile || '',
                content,
            }),
        });
        if (!res.ok) {
            throw new Error(res.error || '保存预览提示词失败');
        }
        configState.samplePromptsPath = res.file || configState.samplePromptsPath;
        configState.samplePromptsContent = res.content || '';
        return res;
    }

    export async function importTomlFile() {
        if (!(await confirmDiscardTomlChanges('当前 TOML 有未保存修改，导入会覆盖编辑器内容。是否继续？'))) {
            return;
        }
        const input = document.getElementById('toml-import-input');
        if (!input) return;
        input.value = '';
        input.click();
    }

    export function handleTomlImport(event) {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = () => {
            tomlState.currentTomlFile = '';
            tomlState.tomlSavedContent = '';
            document.getElementById('toml-current-file').textContent = `未保存导入: ${file.name}`;
            document.getElementById('toml-file-select').value = '';
            document.getElementById('toml-editor').value = reader.result || '';
            setTomlEditorLocked(false);
            updateTomlSelectionUI('');
            applyTomlLockState('');
            updateTomlDirtyState();
            setTomlStatus('ok', `已导入 ${file.name}，点击保存或另存为写入项目`, { persist: true });
        };
        reader.onerror = () => {
            setTomlStatus('error', '导入失败: 无法读取本地文件');
        };
        reader.readAsText(file, 'utf-8');
    }

    export function exportTomlFile() {
        const content = document.getElementById('toml-editor').value;
        const file = tomlState.currentTomlFile || val('toml-file-select');
        const filename = exportTomlFilename(file);
        downloadTomlContent(content, filename);
        setTomlStatus('ok', `已导出 ${filename}`);
    }

    export function downloadTomlContent(content, filename) {
        ctx.download.downloadText(content, filename, 'application/toml;charset=utf-8');
    }

    export function triggerDownload(url, filename) {
        ctx.download.triggerDownload(url, filename);
    }

    export function downloadBlob(blob, filename) {
        ctx.download.downloadBlob(blob, filename);
    }

    export function createTomlZipBlob(entries) {
        return ctx.download.createZipBlob(entries, uniqueZipEntryName);
    }

    export function uniqueZipEntryName(name, usedNames) {
        const base = exportTomlFilename(name || 'config.toml');
        if (!usedNames.has(base)) {
            usedNames.add(base);
            return base;
        }
        const stem = base.replace(/\.toml$/i, '');
        let index = 2;
        let candidate = `${stem}-${index}.toml`;
        while (usedNames.has(candidate)) {
            index += 1;
            candidate = `${stem}-${index}.toml`;
        }
        usedNames.add(candidate);
        return candidate;
    }

    export async function saveTomlAs() {
        const editor = document.getElementById('toml-editor');
        const currentFile = tomlState.currentTomlFile;
        const target = await showTomlSaveAsDialog(currentFile);
        if (target === null) return;

        const file = normalizeTomlSaveAsPath(target?.name ?? target);
        const targetGroupId = target?.group || 'imported';
        if (!file) {
            setTomlStatus('error', '另存新配置失败: 请先输入新的配置名称');
            return;
        }
        if (file === currentFile) {
            setTomlStatus('error', '另存新配置失败: 新配置不能和当前选中文件同名');
            return;
        }
        if (tomlState.tomlFiles.includes(file)) {
            setTomlStatus('error', `${file} 已存在，请换一个新的配置名称`);
            return;
        }

        try {
            const baseContent = editor.value;
            const preparedValues = await prepareFormPatchValues(collectChangedFormValues({ persistDefaultFields: true }));
            const content = Object.keys(preparedValues).length
                ? await previewPatchedTomlContent(file, baseContent, preparedValues)
                : baseContent;
            const res = await api('/api/config/raw/save-as', {
                method: 'POST',
                body: JSON.stringify({ file, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '另存为失败');
                return;
            }

            trainingState.currentTrainingSource = {
                method: file.split('/').pop().replace(/\.toml$/i, ''),
                methods_subdir: 'imported',
                file,
            };
            tomlState.currentTomlFile = file;
            tomlState.tomlSavedContent = content;
            editor.value = content;
            const datasetApplied = await applySelectedDatasetPresetToCurrentConfig(file);
            if (!datasetApplied) {
                setTomlStatus('error', `新配置已创建: ${file}，但数据集预设应用失败，请修正后再次保存更新当前选中配置`, { persist: true });
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            if (datasetApplied.applied) {
                const editorAfterDataset = document.getElementById('toml-editor');
                tomlState.tomlSavedContent = editorAfterDataset?.value || tomlState.tomlSavedContent;
            }
            const moved = await moveTomlFileToGroup(file, targetGroupId);
            if (!moved) {
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            await loadTomlFileList(file);
            await applyTomlToConfig({ silent: true, pendingChangesResolved: true });
            updateTomlDirtyState();
            const groupLabel = saveAsTargetGroups().find((group) => group.id === targetGroupId)?.label || targetGroupId;
            setTomlStatus('ok', `已另存新配置: ${file} → ${groupLabel}`);
        } catch (e) {
            setTomlStatus('error', '请求失败: ' + e.message);
        }
    }

    export async function createBlankPresetFromLoraTemplate() {
        let templateContent = '';
        try {
            const data = await api(`/api/config/raw?file=${encodeURIComponent(BLANK_PRESET_TEMPLATE_FILE)}`);
            if (data?.ok === false) {
                setTomlStatus('error', data.error || '读取 LoRA 模板失败');
                return;
            }
            templateContent = typeof data.content === 'string' ? data.content : '';
        } catch (e) {
            setTomlStatus('error', '读取 LoRA 模板失败: ' + e.message);
            return;
        }
        if (!templateContent.trim()) {
            setTomlStatus('error', `读取 LoRA 模板失败: ${BLANK_PRESET_TEMPLATE_FILE} 内容为空或不存在`);
            return;
        }

        const target = await showTomlSaveAsDialog(BLANK_PRESET_TEMPLATE_FILE, {
            title: '创建空白预设配置',
            description: `以 ${BLANK_PRESET_TEMPLATE_LABEL} 为模板，并套用默认全局模型配置，创建一个新的可编辑项目预设。`,
            confirmText: '创建空白预设配置',
            hint: '新文件默认创建到 configs/imported/；分组只影响右侧列表归类。默认全局模型配置只作为初始值，创建后仍可在配置页覆盖。',
            currentText: `模板: ${BLANK_PRESET_TEMPLATE_LABEL} (${BLANK_PRESET_TEMPLATE_FILE})`,
        });
        if (target === null) return;

        const file = normalizeTomlSaveAsPath(target?.name ?? target);
        const targetGroupId = target?.group || 'imported';
        if (!file) {
            setTomlStatus('error', '创建空白预设配置失败: 请先输入新的配置名称');
            return;
        }
        if (file === BLANK_PRESET_TEMPLATE_FILE) {
            setTomlStatus('error', '创建空白预设配置失败: 不能覆盖 LoRA 标准模板');
            return;
        }
        if (tomlState.tomlFiles.includes(file)) {
            setTomlStatus('error', `${file} 已存在，请换一个新的配置名称`);
            return;
        }

        const canSwitch = await handlePendingConfigSwitch({ targetLabel: `新的空白预设配置 ${file.split('/').pop() || file}` });
        if (!canSwitch) return;

        try {
            // 空白预设先复用模板，再写入默认全局模型配置，减少新建后的重复填写。
            const globalModelPathOverrides = getGlobalModelPathOverrides();
            const content = Object.keys(globalModelPathOverrides).length
                ? await previewPatchedTomlContent(file, templateContent, globalModelPathOverrides)
                : templateContent;
            const res = await api('/api/config/raw/save-as', {
                method: 'POST',
                body: JSON.stringify({ file, content }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '创建空白预设配置失败');
                return;
            }

            const moved = await moveTomlFileToGroup(file, targetGroupId);
            if (!moved) {
                await loadTomlFileList(file, { force: true });
                updateTomlDirtyState();
                return;
            }
            await loadTomlFileList(file, { force: true });
            await applyTomlToConfig({ silent: true, pendingChangesResolved: true });
            updateTomlDirtyState();
            const groupLabel = saveAsTargetGroups().find((group) => group.id === targetGroupId)?.label || targetGroupId;
            setTomlStatus('ok', `已创建空白预设配置: ${file} → ${groupLabel}`, { persist: true });
        } catch (e) {
            setTomlStatus('error', '创建空白预设配置失败: ' + e.message);
        }
    }

    export async function previewPatchedTomlContent(file, content, values) {
        const res = await api('/api/config/raw/patch-preview', {
            method: 'POST',
            body: JSON.stringify({ file, content, values }),
        });
        if (!res.ok) {
            throw new Error(res.error || '应用表单修改失败');
        }
        return typeof res.content === 'string' ? res.content : content;
    }

    export async function showTomlSaveAsDialog(currentFile, options = {}) {
        const wrap = document.createElement('div');
        wrap.className = 'toml-save-as-dialog-body';

        const label = document.createElement('label');
        label.className = 'history-task-dialog-field';
        const labelText = document.createElement('span');
        labelText.textContent = options.nameLabel || '新配置名称或 configs/ 路径';
        const input = document.createElement('input');
        input.type = 'text';
        input.value = '';
        input.placeholder = options.placeholder || '例如 rokkotsu_v2 或 configs/imported/rokkotsu_v2.toml';
        input.className = 'history-task-dialog-input';
        label.append(labelText, input);

        const groups = saveAsTargetGroups();
        const groupWrap = document.createElement('div');
        groupWrap.className = 'toml-save-as-group-list';
        const groupTitle = document.createElement('span');
        groupTitle.className = 'toml-save-as-group-title';
        groupTitle.textContent = '保存到分组';
        groupWrap.appendChild(groupTitle);

        const radios = [];
        for (const group of groups) {
            const option = document.createElement('label');
            option.className = 'toml-move-option';

            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'toml-save-as-target-group';
            radio.value = group.id;
            radio.checked = group.id === 'imported' || (!radios.length && !groups.some((item) => item.id === 'imported'));
            radios.push(radio);

            const text = document.createElement('span');
            const title = document.createElement('strong');
            title.textContent = group.label || group.id;
            const detail = document.createElement('small');
            detail.textContent = `${(group.files || []).length} 个配置`;
            text.append(title, detail);

            option.append(radio, text);
            groupWrap.appendChild(option);
        }

        const hint = document.createElement('p');
        hint.className = 'toml-save-as-hint';
        hint.textContent = options.hint || '只填写文件名时会创建到 configs/imported/，分组只影响右侧列表归类；必须使用新名称，不会覆盖当前选中配置。';

        const current = document.createElement('p');
        current.className = 'toml-save-as-current';
        current.textContent = options.currentText
            || (currentFile ? `当前选中配置: ${currentFile}` : '当前没有选中的配置文件，将使用编辑器内容创建新配置。');

        wrap.append(label, groupWrap, hint, current);

        return showHistoryTaskDialog({
            title: options.title || '另存新配置',
            description: options.description || '输入一个新名称，并选择它在右侧配置列表中的目标分组。',
            body: wrap,
            confirmText: options.confirmText || '创建配置文件',
            onOpen: () => input.focus(),
            getValue: () => {
                const checked = wrap.querySelector('input[name="toml-save-as-target-group"]:checked');
                return {
                    name: input.value,
                    group: checked?.value || 'imported',
                };
            },
        });
    }

    export function saveAsTargetGroups() {
        const trainingGroups = filterTrainingTomlGroups(tomlState.tomlFileGroups);
        const groups = reorderTomlFileGroups(trainingGroups)
            .filter((group) => group.trainable && group.movable && !group.locked && !group.user_group_locked);
        if (groups.some((group) => group.id === 'imported')) return groups;
        const imported = trainingGroups.find((group) => group.id === 'imported');
        if (imported && imported.trainable && !imported.locked && !imported.user_group_locked) {
            return [imported, ...groups];
        }
        return groups.length ? groups : [{
            id: 'imported',
            label: '导入配置',
            files: [],
        }];
    }

    export async function moveTomlFileToGroup(file, groupId) {
        if (!groupId || groupId === 'imported') return true;
        try {
            const res = await api('/api/config/file-groups/move-file', {
                method: 'POST',
                body: JSON.stringify({ file, group: groupId }),
            });
            if (!res.ok) {
                setTomlStatus('error', res.error || '另存成功，但移动到指定分组失败');
                return false;
            }
            return true;
        } catch (e) {
            setTomlStatus('error', '另存成功，但移动分组请求失败: ' + e.message);
            return false;
        }
    }

    export function normalizeTomlSaveAsPath(rawPath) {
        let file = String(rawPath || '').trim().replace(/\\/g, '/');
        file = file.replace(/^\/+/, '');
        if (!file) return '';
        if (!file.startsWith('configs/')) {
            file = `configs/imported/${file}`;
        }
        if (!file.toLowerCase().endsWith('.toml')) {
            file += '.toml';
        }
        return file;
    }

    export function exportTomlFilename(filePath) {
        const base = String(filePath || '').split('/').filter(Boolean).pop();
        if (!base) return 'anima-config.toml';
        return base.toLowerCase().endsWith('.toml') ? base : `${base}.toml`;
    }

    export function isFixedSystemTomlGroup(group) {
        return Boolean(
            group.id === 'web_config' ||
            group.id === 'presets' ||
            group.id === 'methods' ||
            group.id === 'gui_methods' ||
            group.system_locked
        );
    }

    export function isDatasetConfigGroup(group) {
        if (!group) return false;
        const id = String(group.id || '');
        const kind = String(group.kind || '').toLowerCase();
        if (kind === 'dataset' || kind === 'datasets' || id === 'datasets' || id === 'unfiled_datasets') return true;
        return (group.files || []).some((item) => String(item.path || '').replace(/\\/g, '/').startsWith('configs/datasets/'));
    }

    export function isTrainingTomlGroup(group) {
        return Boolean(group) && !isDatasetConfigGroup(group);
    }

    export function filterTrainingTomlGroups(groups) {
        return (Array.isArray(groups) ? groups : []).filter(isTrainingTomlGroup);
    }

    export function shouldShowTomlGroup(group) {
        return isTrainingTomlGroup(group) && !isFixedSystemTomlGroup(group);
    }

    export function reorderTomlFileGroups(groups) {
        return [...(groups || [])]
            .map((group, index) => ({ group, index }))
            .filter(({ group }) => isTrainingTomlGroup(group) && (group.user_managed || group.lockable || (group.files || []).length > 0))
            .sort((a, b) => {
                const aFixed = isFixedSystemTomlGroup(a.group);
                const bFixed = isFixedSystemTomlGroup(b.group);
                if (aFixed !== bFixed) return aFixed ? 1 : -1;
                return a.index - b.index;
            })
            .map((item) => item.group);
    }

    export function getSortableTomlGroups() {
        return [...(tomlState.tomlFileGroups || [])]
            .filter((group) => isTomlGroupDraggable(group));
    }

    export function isTomlGroupDraggable(group) {
        return Boolean(
            group?.id &&
            isTrainingTomlGroup(group) &&
            !isFixedSystemTomlGroup(group) &&
            !group.locked &&
            !group.user_group_locked &&
            (group.user_managed || group.lockable || (group.files || []).length > 0)
        );
    }

configureSamplePromptsBridge({
    currentSamplePromptText,
    normalizeSamplePromptsPath,
    isEditableSamplePromptsTextFilePath,
    isSamplePromptsFilePath,
    loadSamplePrompts,
    saveSamplePrompts,
});

configureTomlIoBridge({
    importTomlFile,
    handleTomlImport,
    exportTomlFile,
    downloadTomlContent,
    triggerDownload,
    downloadBlob,
    createTomlZipBlob,
    uniqueZipEntryName,
    saveTomlAs,
    createBlankPresetFromLoraTemplate,
    previewPatchedTomlContent,
    showTomlSaveAsDialog,
    saveAsTargetGroups,
    moveTomlFileToGroup,
    normalizeTomlSaveAsPath,
    exportTomlFilename,
    isFixedSystemTomlGroup,
    isDatasetConfigGroup,
    isTrainingTomlGroup,
    filterTrainingTomlGroups,
    shouldShowTomlGroup,
    reorderTomlFileGroups,
    getSortableTomlGroups,
    isTomlGroupDraggable,
});
