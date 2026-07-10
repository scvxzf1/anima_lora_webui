/**
 * Config quick-preset buttons co-located with the stage-schedule feature module.
 * Extracted so stage-resolution.js can focus on the curriculum dialog.
 */
import { NO_DATASET_REGULARIZATION_QUICK_PRESETS, RESOURCE_QUICK_PRESETS } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260707-93';
import { setFieldInputValue } from '../anima-app/chunks/13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260707-93';
import { handleFormFieldChange } from '../anima-app/chunks/14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260707-93';
import { fillGlobalModelPathsIntoConfigForm, resourceQuickCurrentValue, strongerSelectiveCheckpointValue } from '../anima-app/chunks/06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260707-93';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260707-93';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260707-93';

const configState = getConfigState();
const configFormState = configState.configFormState;

export function createFillGlobalModelPathsButton() {
    const btn = document.createElement('button');
    btn.id = 'btn-fill-global-model-paths';
    btn.type = 'button';
    btn.className = 'btn btn-small config-group-title-action';
    btn.textContent = '填写全局路径配置';
    btn.title = '用全局设置里的三项基础模型路径覆盖当前配置表单';
    btn.addEventListener('click', () => {
        fillGlobalModelPathsIntoConfigForm().catch((e) => {
            setTomlStatus('error', '填写全局路径配置失败: ' + e.message);
        });
    });
    return btn;
}

function createConfigQuickPresetsButton(options, content, collapseBtn) {
    const groupName = options.groupName || '';
    const btn = document.createElement('button');
    if (options.id) btn.id = options.id;
    btn.type = 'button';
    btn.className = ['btn btn-small config-group-title-action config-quick-preset-toggle', options.className || ''].filter(Boolean).join(' ');
    btn.textContent = options.text || '快速填写';
    btn.title = options.showTitle || `显示${groupName}预设，一键填写当前表单`;
    btn.setAttribute('aria-expanded', 'false');
    btn.addEventListener('click', () => {
        const panel = content.querySelector(`.${options.panelClass}`);
        if (!panel) return;
        const nextVisible = panel.hidden;
        panel.hidden = !nextVisible;
        btn.classList.toggle('active', nextVisible);
        btn.setAttribute('aria-expanded', String(nextVisible));
        btn.title = nextVisible
            ? (options.hideTitle || `收起${groupName}快速预设`)
            : (options.showTitle || `显示${groupName}预设，一键填写当前表单`);
        if (nextVisible && content.hidden) {
            content.hidden = false;
            collapseBtn.textContent = '收起';
            collapseBtn.setAttribute('aria-expanded', 'true');
            collapseBtn.title = '收起这个配置区';
            if (groupName) {
                configFormState.expandedGroups.add(groupName);
                configFormState.collapsedGroups.delete(groupName);
            }
        }
    });
    return btn;
}

function createConfigQuickPresetPanel(options) {
    const panel = document.createElement('div');
    panel.className = ['config-quick-presets', options.panelClass || ''].filter(Boolean).join(' ');
    panel.hidden = true;
    panel.setAttribute('aria-label', options.ariaLabel || `${options.groupName || '配置'}快速预设`);

    const label = document.createElement('span');
    label.className = 'config-quick-label';
    label.textContent = options.label || '快速预设';
    panel.appendChild(label);

    for (const preset of options.presets || []) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-small config-quick-preset-btn';
        if (options.datasetKey) btn.dataset[options.datasetKey] = preset.id;
        btn.textContent = preset.label;
        btn.title = preset.note;
        btn.addEventListener('click', () => options.applyPreset(preset));
        panel.appendChild(btn);
    }
    return panel;
}

export function createResourceQuickPresetsButton(content, collapseBtn) {
    return createConfigQuickPresetsButton({
        id: 'btn-resource-quick-presets',
        className: 'config-resource-quick-toggle',
        panelClass: 'config-resource-quick-presets',
        groupName: '显存与速度优化',
        showTitle: '显示显存与速度优化预设，一键填写当前表单',
        hideTitle: '收起显存与速度优化快速预设',
    }, content, collapseBtn);
}

export function createResourceQuickPresetPanel() {
    return createConfigQuickPresetPanel({
        panelClass: 'config-resource-quick-presets',
        groupName: '显存与速度优化',
        ariaLabel: '显存与速度优化快速预设',
        label: '快速预设',
        presets: RESOURCE_QUICK_PRESETS,
        datasetKey: 'resourcePreset',
        applyPreset: applyResourceQuickPreset,
    });
}

function applyResourceQuickPreset(preset) {
    for (const [key, value] of Object.entries(resourceQuickPresetPatch(preset))) {
        setFieldInputValue(key, value);
    }
    handleFormFieldChange();
    setTomlStatus('ok', `已填写显存与速度优化预设: ${preset.label}`);
}

function resourceQuickPresetPatch(preset) {
    const patch = {};
    for (const [key, value] of Object.entries(preset?.values || {})) {
        patch[key] = resourceQuickPresetValue(preset, key, value);
    }

    const selectiveCheckpoint = String(
        patch.selective_checkpoint ?? resourceQuickCurrentValue('selective_checkpoint') ?? 'off'
    ).trim();
    if (selectiveCheckpoint && selectiveCheckpoint !== 'off') {
        patch.gradient_checkpointing = false;
        patch.cpu_offload_checkpointing = false;
        patch.unsloth_offload_checkpointing = false;
    }

    const blocksToSwap = Number(patch.blocks_to_swap ?? resourceQuickCurrentValue('blocks_to_swap'));
    if (Number.isFinite(blocksToSwap) && blocksToSwap > 0) {
        patch.cpu_offload_checkpointing = false;
        patch.unsloth_offload_checkpointing = false;
    }
    return patch;
}

function resourceQuickPresetValue(preset, key, value) {
    const strategy = preset?.merge?.[key] || '';
    if (strategy === 'max') {
        const current = Number(resourceQuickCurrentValue(key));
        const next = Number(value);
        if (Number.isFinite(current) && Number.isFinite(next)) {
            return Math.max(current, next);
        }
    }
    if (strategy === 'checkpoint_strength_max') {
        return strongerSelectiveCheckpointValue(resourceQuickCurrentValue(key), value);
    }
    return value;
}

export function createNoDatasetRegularizationQuickPresetsButton(content, collapseBtn) {
    return createConfigQuickPresetsButton({
        id: 'btn-no-dataset-regularization-quick-presets',
        className: 'config-no-dataset-regularization-quick-toggle',
        panelClass: 'config-no-dataset-regularization-quick-presets',
        groupName: '无数据集正则化',
        showTitle: '显示无数据集正则化预设，一键填写当前表单',
        hideTitle: '收起无数据集正则化快速预设',
    }, content, collapseBtn);
}

export function createNoDatasetRegularizationQuickPresetPanel() {
    return createConfigQuickPresetPanel({
        panelClass: 'config-no-dataset-regularization-quick-presets',
        groupName: '无数据集正则化',
        ariaLabel: '无数据集正则化快速预设',
        label: '快速填写',
        presets: NO_DATASET_REGULARIZATION_QUICK_PRESETS,
        datasetKey: 'noDatasetRegularizationPreset',
        applyPreset: applyNoDatasetRegularizationQuickPreset,
    });
}

function applyNoDatasetRegularizationQuickPreset(preset) {
    for (const [key, value] of Object.entries(preset.values || {})) {
        setFieldInputValue(key, value);
    }
    handleFormFieldChange();
    const extra = preset.id === 'dop_roles'
        ? '，还需要填写泛化类别作为 DOP 类提示，例如 woman / character，并重新生成文本缓存'
        : '';
    setTomlStatus('ok', `已填写无数据集正则化预设: ${preset.label}${extra}`);
}
