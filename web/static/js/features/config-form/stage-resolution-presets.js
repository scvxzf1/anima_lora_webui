/**
 * Config quick-preset buttons co-located with the stage-schedule feature module.
 * Extracted so stage-resolution.js can focus on the curriculum dialog.
 */
import { NO_DATASET_REGULARIZATION_QUICK_PRESETS, RESOURCE_QUICK_PRESETS } from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260711-ir1';
import { setFieldInputValue } from './field-input.js?v=module-bootstrap-20260711-ir1';
import { handleFormFieldChange } from '../anima-app/chunks/14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260711-ir1';
import { fillGlobalModelPathsIntoConfigForm, resourceQuickCurrentValue, strongerSelectiveCheckpointValue } from './resource-values.js?v=module-bootstrap-20260711-ir1';
import { setTomlStatus } from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';

const configState = getConfigState();
const configFormState = configState.configFormState;

function normalizeQuickPresetMethodToken(value) {
    return String(value ?? '').trim().toLowerCase();
}

function sameQuickPresetValue(left, right) {
    if (left === right) return true;
    if (left == null && right == null) return true;
    if (typeof left === 'boolean' || typeof right === 'boolean') {
        return Boolean(left) === Boolean(right);
    }
    const leftNum = Number(left);
    const rightNum = Number(right);
    if (Number.isFinite(leftNum) && Number.isFinite(rightNum) && String(left).trim() !== '' && String(right).trim() !== '') {
        return leftNum === rightNum;
    }
    return String(left ?? '') === String(right ?? '');
}

function formatQuickPresetValue(value) {
    if (value === undefined) return '未设置';
    if (value === null) return 'null';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'string' && value.trim() === '') return '空';
    return String(value);
}

export function resolveQuickPresetMethodFamily(source = {}) {
    if (source == null) return '';
    if (typeof source === 'string' || typeof source === 'number' || typeof source === 'boolean') {
        return normalizeQuickPresetMethodToken(source);
    }

    const adapterKind = normalizeQuickPresetMethodToken(
        source.lora_adapter_kind
        ?? source.adapter_kind
        ?? source.methodFamily
        ?? source.method_family
    );
    if (adapterKind === 'lokr' || adapterKind === 'loha' || adapterKind === 'glora' || adapterKind === 'vera' || adapterKind === 'lora') {
        return adapterKind;
    }

    const flagMap = [
        ['use_lokr', 'lokr'],
        ['use_loha', 'loha'],
        ['use_glora', 'glora'],
        ['use_vera', 'vera'],
    ];
    for (const [flag, family] of flagMap) {
        const raw = source[flag];
        if (raw === true || raw === 1 || raw === '1' || String(raw).toLowerCase() === 'true') {
            return family;
        }
    }

    const method = normalizeQuickPresetMethodToken(
        source.method
        ?? source.network_module
        ?? source.variant
        ?? source.selectedMethod
    );
    if (!method) return '';
    if (method.includes('lokr')) return 'lokr';
    if (method.includes('loha')) return 'loha';
    if (method.includes('glora')) return 'glora';
    if (method.includes('vera')) return 'vera';
    return method;
}

export function isQuickPresetApplicable(preset, methodFamily) {
    const allowed = Array.isArray(preset?.applicableMethods)
        ? preset.applicableMethods.map(normalizeQuickPresetMethodToken).filter(Boolean)
        : [];
    if (!allowed.length) {
        return { ok: true };
    }

    const family = resolveQuickPresetMethodFamily(methodFamily);
    const tokens = new Set();
    if (family) {
        tokens.add(family);
        if (family === 'lokr') tokens.add('use_lokr');
        if (family === 'loha') tokens.add('use_loha');
        if (family === 'glora') tokens.add('use_glora');
        if (family === 'vera') tokens.add('use_vera');
    }
    if (methodFamily && typeof methodFamily === 'object') {
        for (const [key, value] of Object.entries(methodFamily)) {
            const token = normalizeQuickPresetMethodToken(key);
            if (!token) continue;
            if (value === true || value === 1 || value === '1' || String(value).toLowerCase() === 'true') {
                tokens.add(token);
            }
            if (token === 'lora_adapter_kind' || token === 'method' || token === 'methodfamily') {
                const normalized = resolveQuickPresetMethodFamily(value);
                if (normalized) tokens.add(normalized);
            }
        }
    }

    const matched = allowed.some((item) => tokens.has(item));
    if (matched) return { ok: true };
    const label = family || '当前方法';
    return {
        ok: false,
        reason: `${preset?.label || '该快捷资源'}仅适用于 ${allowed.join('/')}，当前是 ${label}`,
    };
}

export function previewQuickPresetDiff(preset, currentValues = {}) {
    const values = currentValues && typeof currentValues === 'object' ? currentValues : {};
    const patch = values.__patch && typeof values.__patch === 'object'
        ? values.__patch
        : (preset?.values || {});
    const diffs = [];
    for (const [key, toValue] of Object.entries(patch || {})) {
        if (key === '__patch') continue;
        const fromValue = Object.prototype.hasOwnProperty.call(values, key) ? values[key] : undefined;
        if (sameQuickPresetValue(fromValue, toValue)) continue;
        diffs.push({
            key,
            from: fromValue,
            to: toValue,
        });
    }
    return diffs;
}

function readCurrentQuickPresetMethodSource() {
    const currentConfig = configState.currentConfig || {};
    const draft = configFormState?.draftValues;
    const source = { ...currentConfig };

    const formKeys = ['lora_adapter_kind', 'use_lokr', 'use_loha', 'use_glora', 'use_vera'];
    for (const key of formKeys) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (input) {
            if (input.type === 'checkbox') {
                source[key] = input.checked;
            } else {
                source[key] = input.value;
            }
            continue;
        }
        if (draft?.has?.(key)) {
            source[key] = draft.get(key);
        }
    }

    const methodSelect = document.getElementById('method-select');
    if (methodSelect?.value) source.method = methodSelect.value;
    const variantSelect = document.getElementById('variant-select');
    if (variantSelect?.value) source.variant = variantSelect.value;
    return source;
}

function collectCurrentQuickPresetValues(keys = []) {
    const values = {};
    for (const key of keys) {
        values[key] = resourceQuickCurrentValue(key);
    }
    return values;
}

function formatQuickPresetDiffSummary(diffs, limit = 4) {
    if (!diffs.length) return '没有字段变化';
    const parts = diffs.slice(0, limit).map((item) => (
        `${item.key}: ${formatQuickPresetValue(item.from)} → ${formatQuickPresetValue(item.to)}`
    ));
    if (diffs.length > limit) {
        parts.push(`…另有 ${diffs.length - limit} 项`);
    }
    return parts.join('；');
}


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

    const methodGuard = Boolean(options.methodGuard);
    for (const preset of options.presets || []) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-small config-quick-preset-btn';
        if (options.datasetKey) btn.dataset[options.datasetKey] = preset.id;
        btn.textContent = preset.label;
        btn.title = preset.note || '';

        if (methodGuard) {
            const applicability = isQuickPresetApplicable(preset, readCurrentQuickPresetMethodSource());
            if (!applicability.ok) {
                btn.disabled = true;
                btn.setAttribute('aria-disabled', 'true');
                btn.title = `${preset.note || preset.label}（${applicability.reason}）`;
                btn.addEventListener('click', (event) => {
                    event.preventDefault();
                    setTomlStatus('error', applicability.reason);
                });
                panel.appendChild(btn);
                continue;
            }
        }

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
        methodGuard: true,
        applyPreset: applyResourceQuickPreset,
    });
}

function applyResourceQuickPreset(preset) {
    const methodSource = readCurrentQuickPresetMethodSource();
    const applicability = isQuickPresetApplicable(preset, methodSource);
    if (!applicability.ok) {
        setTomlStatus('error', applicability.reason || `${preset.label}不适用于当前方法`);
        return;
    }

    const patch = resourceQuickPresetPatch(preset);
    const currentValues = collectCurrentQuickPresetValues(Object.keys(patch));
    const diffs = previewQuickPresetDiff(preset, { ...currentValues, __patch: patch });
    if (!diffs.length) {
        setTomlStatus('ok', `${preset.label}：当前已是相同配置，无需修改`);
        return;
    }

    for (const [key, value] of Object.entries(resourceQuickPresetPatch(preset))) {
        setFieldInputValue(key, value);
    }
    handleFormFieldChange();
    setTomlStatus(
        'ok',
        `已填写 ${preset.label}；将修改 ${diffs.length} 个字段：${formatQuickPresetDiffSummary(diffs)}`
    );
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
