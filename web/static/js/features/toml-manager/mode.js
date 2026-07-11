/**
 * TOML manager mode / file list / output-run loading helpers.
 * Moved out of anima-app mechanical chunks.
 */
import { configureTomlManagerBridge } from '../anima-app/helpers/toml-manager-bridge.js?v=module-bootstrap-20260711-ir6';
import { handleDeletedTomlSelection } from '../anima-app/helpers/toml-actions-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    loadOutputRunConfig,
    loadTomlFile,
    preferredOutputRunKind,
    renderOutputRunManager,
    updateOutputRunActionState,
    updateOutputRunSelectionUI,
} from '../anima-app/helpers/output-run-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    confirmDiscardTomlChanges,
    hasPendingConfigChanges,
    setBadge,
    updateTomlDirtyState,
    updateTomlSelectionUI,
} from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir6';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { filterTrainingTomlGroups, reorderTomlFileGroups } from '../anima-app/helpers/toml-io-bridge.js?v=module-bootstrap-20260711-ir6';
import { populateTomlFileSelect } from '../anima-app/helpers/toml-drag-bridge.js?v=module-bootstrap-20260711-ir6';
import { api, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir6';
import { updateConfigStickyPlacement } from '../config-form/group-entry.js?v=module-bootstrap-20260711-ir6';
import {
    applyTomlLockState,
    setTomlStatus,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir6';

const datasetState = getDatasetState();
const tomlState = getTomlState();
const trainingState = getTrainingState();

function currentOutputRunState() {
    return datasetState.outputRunState || {};
}

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

    export function updateConfigPageSummary(mode = tomlState.tomlManagerMode) {
        const modeLabel = document.getElementById('config-sidebar-mode-label');
        const countLabel = document.getElementById('config-sidebar-file-count');
        const countName = document.getElementById('config-sidebar-count-label');
        const kicker = document.getElementById('config-workspace-kicker');
        const title = document.getElementById('config-workspace-title');
        const subtitle = document.getElementById('config-workspace-subtitle');
        if (modeLabel) modeLabel.textContent = mode === 'output' ? '快照' : '项目';
        if (countName) countName.textContent = mode === 'output' ? '运行目录' : '配置文件';
        if (kicker) kicker.textContent = mode === 'output' ? 'OUTPUT SNAPSHOT' : 'CONFIG PRESET';
        if (title) title.textContent = mode === 'output' ? '训练输出配置' : '训练配置';
        if (subtitle) {
            subtitle.textContent = mode === 'output'
                ? '查看全局输出文件夹里的训练快照，可复制为新的项目预设后继续编辑。'
                : '选择方法、变体、预设，编辑训练参数并引用数据集预设。';
        }
        if (!countLabel) return;
        if (mode === 'output') {
            countLabel.textContent = currentOutputRunState().loading ? '...' : String((currentOutputRunState().runs || []).length);
            return;
        }
        countLabel.textContent = String(tomlState.tomlFiles.length || 0);
    }

    export function setTomlManagerMode(mode) {
        const nextMode = mode === 'output' ? 'output' : 'project';
        tomlState.tomlManagerMode = nextMode;
        document.querySelectorAll('.toml-mode-btn').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.tomlMode === nextMode);
        });
        const projectManager = document.getElementById('toml-project-manager');
        const outputManager = document.getElementById('output-run-manager');
        const configWorkspace = document.getElementById('config-form-workspace');
        const stickyActions = document.getElementById('config-sticky-actions');
        const outputDetail = document.getElementById('output-run-detail-panel');
        const projectActions = document.querySelectorAll('.toml-primary-actions');
        const outputActions = document.getElementById('output-run-actions');
        if (projectManager) projectManager.hidden = nextMode !== 'project';
        if (outputManager) outputManager.hidden = nextMode !== 'output';
        if (configWorkspace) configWorkspace.hidden = nextMode !== 'project';
        if (stickyActions) stickyActions.hidden = nextMode !== 'project';
        if (outputDetail) outputDetail.hidden = nextMode !== 'output';
        projectActions.forEach((el) => {
            el.hidden = nextMode !== 'project';
        });
        if (outputActions) outputActions.hidden = nextMode !== 'output';
        updateConfigPageSummary(nextMode);
        if (nextMode === 'output') {
            const label = document.getElementById('toml-current-file');
            if (label) label.textContent = currentOutputRunState().file || currentOutputRunState().selectedRun || '训练输出配置';
            setBadge('toml-current-badge', false, '当前训练');
            setBadge('toml-trainable-badge', Boolean(currentOutputRunState().file), '只读快照');
            setBadge('toml-lock-badge', Boolean(currentOutputRunState().file), '只读');
            setBadge('toml-dirty-badge', false, '未保存');
            updateOutputRunActionState();
            if (!currentOutputRunState().runs.length && !currentOutputRunState().loading) {
                loadOutputRuns();
            } else {
                renderOutputRunManager();
            }
            return;
        }
        updateTomlSelectionUI(tomlState.currentTomlFile);
        updateTomlDirtyState();
        requestAnimationFrame(updateConfigStickyPlacement);
    }

    export async function switchTomlManagerMode(nextMode) {
        const normalizedMode = nextMode === 'output' ? 'output' : 'project';
        if (normalizedMode !== tomlState.tomlManagerMode && normalizedMode === 'output' && hasPendingConfigChanges(tomlState.currentTomlFile)) {
            if (!(await confirmDiscardTomlChanges('当前项目预设有未保存修改，切换到训练输出配置会暂时隐藏这些修改。是否继续？'))) {
                return false;
            }
        }
        setTomlManagerMode(normalizedMode);
        return true;
    }

    export async function loadTomlFileList(preferredFile = '', options = {}) {
        const groups = await api('/api/config/file-groups?kind=training');
        tomlState.tomlFileGroups = filterTrainingTomlGroups(groups);
        tomlState.tomlFileMeta = {};
        tomlState.tomlFiles = [];
        for (const group of tomlState.tomlFileGroups) {
            for (const item of group.files || []) {
                tomlState.tomlFiles.push(item.path);
                tomlState.tomlFileMeta[item.path] = item;
            }
        }
        populateTomlFileSelect(reorderTomlFileGroups(tomlState.tomlFileGroups));
        if (preferredFile && !tomlState.tomlFiles.includes(preferredFile) && tomlState.currentTomlFile === preferredFile) {
            await handleDeletedTomlSelection(preferredFile, '当前配置文件已不存在或已被删除');
            return;
        }
        if (preferredFile && tomlState.tomlFiles.includes(preferredFile)) {
            await loadTomlFile(preferredFile, { force: options.force === true });
            return;
        }
        if (options.skipDefaultLoad) {
            updateTomlSelectionUI('');
            applyTomlLockState('');
            updateTomlDirtyState();
            return;
        }
        if (!options.deferDefaultLoad) {
            await loadDefaultTomlFile({ force: options.force === true });
        }
    }

    export async function loadDefaultTomlFile(options = {}) {
        const variant = currentTrainingSourceState().method || val('variant-select');
        const methodsSubdir = currentTrainingSourceState().methods_subdir || 'gui-methods';
        const target = currentTrainingSourceState().file || `configs/${methodsSubdir}/${variant}.toml`;
        if (tomlState.tomlFiles.includes(target)) {
            await loadTomlFile(target, { force: options.force === true });
        } else if (tomlState.tomlFiles.length > 0) {
            await loadTomlFile(tomlState.tomlFiles[0], { force: options.force === true });
        }
    }

    export async function loadOutputRuns(options = {}) {
        if (location.protocol === 'file:') return;
        datasetState.outputRunState = {
            ...currentOutputRunState(),
            loading: true,
            error: '',
        };
        renderOutputRunManager();
        try {
            const data = await api('/api/config/output-runs');
            if (!data.ok) throw new Error(data.error || '读取训练输出配置失败');
            const runs = Array.isArray(data.runs) ? data.runs : [];
            let selectedRun = currentOutputRunState().selectedRun;
            if (selectedRun && !runs.some((item) => item.name === selectedRun)) selectedRun = '';
            if (!selectedRun && runs.length && options.keepSelection !== true) {
                selectedRun = runs[0].name || '';
            }
            datasetState.outputRunState = {
                ...currentOutputRunState(),
                loading: false,
                runs,
                outputRoot: data.output_root || '',
                selectedRun,
                error: '',
            };
            renderOutputRunManager();
            if (selectedRun) {
                await loadOutputRunConfig(selectedRun, preferredOutputRunKind(selectedRun));
            } else {
                updateOutputRunSelectionUI();
            }
        } catch (e) {
            datasetState.outputRunState = {
                ...currentOutputRunState(),
                loading: false,
                runs: [],
                content: '',
                file: '',
                error: e.message,
            };
            renderOutputRunManager();
            setTomlStatus('error', '读取训练输出配置失败: ' + e.message);
        }
    }


configureTomlManagerBridge({
    updateConfigPageSummary,
    setTomlManagerMode,
    switchTomlManagerMode,
    loadTomlFileList,
    loadDefaultTomlFile,
    loadOutputRuns,
});
