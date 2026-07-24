/**
 * Dataset experimental settings editors (nl-tag mix / scope / trigger clone).
 */
import {
    attachDatasetInlineHelp,
    createDatasetInlineHelp,
    createDatasetInlineHelpButton,
    datasetLocalHelpSpec,
} from './inline-help.js?v=module-bootstrap-20260714-stage-dataset5';
import { help } from '../../config/catalog.js?v=module-bootstrap-20260714-stage-dataset5';
import { createHelpContent } from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { datasetConfigLabel, datasetConfigValue } from '../anima-app/helpers/dataset-config-fields.js?v=module-bootstrap-20260714-stage-dataset5';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    nlTagMixSummary,
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
    normalizeNlTagMix,
    normalizeTriggerClone,
} from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260714-stage-dataset5';
import { datasetEditorStateForActivePanel, isDatasetTabActive, refreshDatasetEditorItem, renderDatasetEditor } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    createDatasetRowSettingInput,
    updateDatasetEditorRow,
    updateDatasetEditorRowSettingValue,
} from './row-fields.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    datasetExperimentalScopeIndices,
    setDatasetExperimentalScopeIndices,
    updateDatasetEditorRowNlTagMix,
    updateDatasetEditorRowTriggerClone,
    updateDatasetEditorRowsSettingValue,
} from './mutations.js?v=module-bootstrap-20260714-stage-dataset5';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

export function createDatasetNlTagMixEditor(row, index) {
        const mix = normalizeNlTagMix(row.nl_tag_mix);
        const panel = document.createElement('div');
        panel.className = ['dataset-nl-tag-mix', mix.enabled ? 'enabled' : ''].filter(Boolean).join(' ');
        panel.dataset.index = String(index);
        const helpDiv = createDatasetInlineHelp('dataset-inline-help dataset-nl-tag-help');

        const controls = document.createElement('div');
        controls.className = 'dataset-nl-tag-controls';

        const toggle = document.createElement('label');
        toggle.className = 'dataset-nl-tag-toggle';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = mix.enabled;
        checkbox.setAttribute('aria-label', 'captions格式nl/tag权重调整');
        checkbox.addEventListener('change', () => {
            updateDatasetEditorRowNlTagMix(index, {
                enabled: checkbox.checked,
                tag_ratio: mix.tag_ratio,
            }, { render: 'item' });
        });
        const toggleTitle = document.createElement('strong');
        toggleTitle.textContent = 'captions格式nl/tag权重调整';
        toggleTitle.title = 'captions格式nl/tag权重调整';
        const helpBtn = createDatasetInlineHelpButton(helpDiv, '查看 nl/tag 权重说明');
        toggle.append(checkbox, toggleTitle, helpBtn);

        const ratio = document.createElement('label');
        ratio.className = 'dataset-nl-tag-ratio';
        const ratioHead = document.createElement('span');
        ratioHead.textContent = 'tag';
        const ratioInput = document.createElement('input');
        ratioInput.type = 'range';
        ratioInput.min = '0';
        ratioInput.max = '100';
        ratioInput.step = '5';
        ratioInput.value = String(Math.round(mix.tag_ratio * 100));
        ratioInput.disabled = !mix.enabled;
        ratioInput.setAttribute('aria-label', 'tag 占比');
        ratioInput.addEventListener('input', () => {
            ratioNumber.value = ratioInput.value;
            const nextMix = {
                enabled: true,
                tag_ratio: Number(ratioInput.value) / 100,
            };
            summary.value = nlTagMixSummary(nextMix);
            summary.textContent = summary.value;
            updateDatasetEditorRowNlTagMix(index, {
                ...nextMix,
            }, { render: false });
        });
        ratio.append(ratioHead, ratioInput);

        const ratioNumber = document.createElement('input');
        ratioNumber.type = 'number';
        ratioNumber.min = '0';
        ratioNumber.max = '100';
        ratioNumber.step = '5';
        ratioNumber.value = String(Math.round(mix.tag_ratio * 100));
        ratioNumber.disabled = !mix.enabled;
        ratioNumber.className = 'dataset-nl-tag-number';
        ratioNumber.setAttribute('aria-label', 'tag 占比百分比');
        ratioNumber.addEventListener('input', () => {
            ratioInput.value = ratioNumber.value;
            const nextMix = {
                enabled: true,
                tag_ratio: Number(ratioNumber.value) / 100,
            };
            summary.value = nlTagMixSummary(nextMix);
            summary.textContent = summary.value;
            updateDatasetEditorRowNlTagMix(index, {
                ...nextMix,
            }, { render: false });
        });

        const summary = document.createElement('output');
        summary.className = 'dataset-nl-tag-summary';
        summary.value = nlTagMixSummary(mix);
        summary.textContent = nlTagMixSummary(mix);
        summary.title = summary.value;

        attachDatasetInlineHelp(helpBtn, helpDiv, datasetLocalHelpSpec('nlTagMix'), panel);

        controls.append(toggle, ratio, ratioNumber, summary);
        panel.append(controls, helpDiv);
        return panel;
    }

export function createDatasetIsRegToggleEditor(row, index) {
        const panel = document.createElement('div');
        panel.className = 'dataset-is-reg-toggle-panel';
        panel.dataset.index = String(index);
        const helpDiv = createDatasetInlineHelp('dataset-is-reg-help');
        const helpBtn = createDatasetInlineHelpButton(helpDiv, '查看正则化训练说明');

        const controls = document.createElement('div');
        controls.className = 'dataset-is-reg-controls';

        const toggleLabel = document.createElement('label');
        toggleLabel.className = 'dataset-is-reg-toggle';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = row.is_reg === true;
        checkbox.setAttribute('aria-label', '标记为正则化数据集');
        checkbox.addEventListener('change', () => {
            updateDatasetEditorRow(index, 'is_reg', checkbox.checked);
        });
        const toggleText = document.createElement('span');
        toggleText.textContent = '标记为正则化数据集';
        toggleText.title = '勾选后该组图片作为正则化样本。';
        toggleLabel.append(checkbox, toggleText);

        attachDatasetInlineHelp(
            helpBtn,
            helpDiv,
            () => createHelpContent('prior_loss_weight', row.settings?.prior_loss_weight ?? 1.0),
            panel,
        );

        controls.append(toggleLabel, helpBtn);
        panel.append(controls, helpDiv);
        return panel;
    }

export function createDatasetPriorLossWeightEditor(row, index) {
        const panel = document.createElement('div');
        panel.className = 'dataset-prior-loss-weight-panel';
        panel.dataset.index = String(index);

        const weightField = document.createElement('label');
        weightField.className = 'dataset-is-reg-weight-field';
        const weightLabel = document.createElement('span');
        weightLabel.className = 'dataset-is-reg-weight-label';
        weightLabel.textContent = '正则化损失权重';
        weightLabel.title = '正则化图像的损失值乘以此系数。';
        const weightInput = document.createElement('input');
        weightInput.type = 'number';
        weightInput.min = '0';
        weightInput.step = '0.1';
        const currentWeight = Number(row.settings?.prior_loss_weight ?? 1.0);
        weightInput.value = String(Number.isFinite(currentWeight) ? Math.max(0, currentWeight) : 1.0);
        weightInput.className = 'dataset-is-reg-weight-input';
        weightInput.title = '损失权重系数，配合“标记为正则化数据集”使用。';
        weightInput.setAttribute('aria-label', '正则化损失权重');
        weightInput.addEventListener('input', () => {
            const nextWeight = Number(weightInput.value);
            updateDatasetEditorRowSettingValue(
                index,
                'prior_loss_weight',
                Number.isFinite(nextWeight) ? Math.max(0, nextWeight) : 1.0,
            );
        });
        weightField.append(weightLabel, weightInput);
        panel.append(weightField);
        return panel;
    }

export function createDatasetMainPolicyRow(row, index) {
        const policyRow = document.createElement('div');
        policyRow.className = 'dataset-main-policy-row';
        policyRow.dataset.index = String(index);

        const controls = document.createElement('div');
        controls.className = 'dataset-main-policy-controls';
        controls.append(
            createDatasetNlTagMixEditor(row, index),
            createDatasetIsRegToggleEditor(row, index),
            createDatasetPriorLossWeightEditor(row, index),
        );
        policyRow.append(controls);
        return policyRow;
    }

export function createDatasetExperimentalScopePicker(index) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        const selected = new Set(datasetExperimentalScopeIndices(index, rows.length));
        const scope = document.createElement('div');
        scope.className = 'dataset-experimental-scope';

        const copy = document.createElement('div');
        copy.className = 'dataset-experimental-scope-copy';
        const titleRow = document.createElement('div');
        titleRow.className = 'dataset-inline-title-row';
        const title = document.createElement('strong');
        title.textContent = '生效范围 / 对多数据集负责';
        const helpDiv = createDatasetInlineHelp('dataset-inline-help dataset-experimental-scope-help');
        const helpBtn = createDatasetInlineHelpButton(helpDiv, '查看生效范围说明');
        titleRow.append(title, helpBtn);
        copy.appendChild(titleRow);

        const actions = document.createElement('div');
        actions.className = 'dataset-experimental-scope-actions';
        const selectAll = document.createElement('button');
        selectAll.type = 'button';
        selectAll.className = 'btn btn-small';
        selectAll.textContent = '全选数据集';
        selectAll.disabled = rows.length <= 1 || selected.size === rows.length;
        selectAll.title = rows.length <= 1
            ? '当前只有一组数据集'
            : '让这个实验框同时负责所有数据集组。';
        selectAll.addEventListener('click', () => {
            setDatasetExperimentalScopeIndices(index, rows.map((_row, rowIndex) => rowIndex));
            refreshDatasetEditorItem(index) || renderDatasetEditor();
        });
        actions.appendChild(selectAll);

        const chips = document.createElement('div');
        chips.className = 'dataset-experimental-scope-chips';
        rows.forEach((_row, rowIndex) => {
            const chip = document.createElement('label');
            chip.className = ['dataset-scope-chip', selected.has(rowIndex) ? 'selected' : ''].filter(Boolean).join(' ');
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = selected.has(rowIndex);
            input.setAttribute('aria-label', `第 ${rowIndex + 1} 组数据集生效`);
            input.addEventListener('change', () => {
                const next = new Set(datasetExperimentalScopeIndices(index, rows.length));
                if (input.checked) {
                    next.add(rowIndex);
                } else {
                    next.delete(rowIndex);
                }
                if (!next.size) {
                    next.add(index);
                }
                setDatasetExperimentalScopeIndices(index, [...next]);
                refreshDatasetEditorItem(index) || renderDatasetEditor();
            });
            const text = document.createElement('span');
            text.textContent = `第 ${rowIndex + 1} 组`;
            chip.append(input, text);
            chips.appendChild(chip);
        });

        attachDatasetInlineHelp(helpBtn, helpDiv, datasetLocalHelpSpec('scope'), scope);

        scope.append(copy, actions, chips, helpDiv);
        return scope;
    }

export function createDatasetTriggerCloneEditor(row, index) {
        const clone = normalizeTriggerClone(row.trigger_clone);
        const panel = document.createElement('div');
        panel.className = ['dataset-trigger-clone', clone.enabled ? 'enabled' : ''].filter(Boolean).join(' ');
        panel.dataset.index = String(index);
        const helpDiv = createDatasetInlineHelp('dataset-inline-help dataset-trigger-clone-help');

        const toggle = document.createElement('label');
        toggle.className = 'dataset-trigger-clone-toggle';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = clone.enabled;
        checkbox.setAttribute('aria-label', '触发提示词图像克隆');
        const toggleText = document.createElement('span');
        const toggleTitleRow = document.createElement('div');
        toggleTitleRow.className = 'dataset-inline-title-row';
        const toggleTitle = document.createElement('strong');
        toggleTitle.textContent = '触发提示词图像克隆';
        const helpBtn = createDatasetInlineHelpButton(helpDiv, '查看触发提示词图像克隆说明');
        toggleTitleRow.append(toggleTitle, helpBtn);
        toggleText.appendChild(toggleTitleRow);
        checkbox.addEventListener('change', () => {
            updateDatasetEditorRowTriggerClone(index, {
                enabled: checkbox.checked,
            }, { render: true });
        });
        toggle.append(checkbox, toggleText);

        const prompt = document.createElement('label');
        prompt.className = 'dataset-trigger-clone-prompt';
        const promptText = document.createElement('span');
        promptText.textContent = '触发提示词';
        const promptInput = document.createElement('input');
        promptInput.type = 'text';
        promptInput.className = 'field-input';
        promptInput.value = clone.prompt;
        promptInput.placeholder = '例如 my_character_token';
        promptInput.disabled = !clone.enabled;
        promptInput.addEventListener('input', () => {
            updateDatasetEditorRowTriggerClone(index, {
                enabled: true,
                prompt: promptInput.value,
            });
        });
        prompt.append(promptText, promptInput);

        const repeats = document.createElement('label');
        repeats.className = 'dataset-trigger-clone-repeats';
        const repeatsText = document.createElement('span');
        repeatsText.textContent = '克隆循环次数';
        const repeatsInput = document.createElement('input');
        repeatsInput.type = 'number';
        repeatsInput.min = '1';
        repeatsInput.step = '1';
        repeatsInput.value = String(clone.num_repeats);
        repeatsInput.disabled = !clone.enabled;
        repeatsInput.addEventListener('input', () => {
            updateDatasetEditorRowTriggerClone(index, {
                enabled: true,
                num_repeats: repeatsInput.value,
            });
        });
        repeats.append(repeatsText, repeatsInput);

        const summary = document.createElement('span');
        summary.className = 'dataset-trigger-clone-summary';
        summary.textContent = clone.enabled
            ? `额外训练权重 x${clone.num_repeats}`
            : '默认关闭';

        attachDatasetInlineHelp(helpBtn, helpDiv, datasetLocalHelpSpec('triggerClone'), panel);

        panel.append(toggle, prompt, repeats, summary, helpDiv);
        return panel;
    }
