/**
 * Dataset row main/advanced settings editors.
 */
import {
    attachDatasetInlineHelp,
    createDatasetInlineHelp,
    createDatasetInlineHelpButton,
    datasetLocalHelpSpec,
} from './inline-help.js?v=module-bootstrap-20260809-nf4-v2';
import { help } from '../../config/catalog.js?v=module-bootstrap-20260809-nf4-v2';
import { createHelpContent } from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { datasetConfigLabel, datasetConfigValue } from '../anima-app/helpers/dataset-config-fields.js?v=module-bootstrap-20260809-nf4-v2';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    nlTagMixSummary,
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
    normalizeNlTagMix,
    normalizeTriggerClone,
} from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260809-nf4-v2';
import { datasetEditorStateForActivePanel, isDatasetTabActive, refreshDatasetEditorItem, renderDatasetEditor } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { createDatasetRowSettingInput, updateDatasetEditorRow } from './row-fields.js?v=module-bootstrap-20260809-nf4-v2';
import {
    datasetExperimentalScopeIndices,
    setDatasetExperimentalScopeIndices,
    updateDatasetEditorRowNlTagMix,
    updateDatasetEditorRowTriggerClone,
    updateDatasetEditorRowsSettingValue,
} from './mutations.js?v=module-bootstrap-20260809-nf4-v2';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

export function createDatasetPathFilterEditor(row, index) {
        const panel = document.createElement('div');
        panel.className = 'dataset-path-filter-advanced';
        panel.dataset.index = String(index);
        const helpDiv = createDatasetInlineHelp('dataset-inline-help dataset-path-filter-help');

        const recursive = document.createElement('label');
        recursive.className = 'dataset-path-filter-recursive';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = row.recursive !== false;
        checkbox.setAttribute('aria-label', '递归扫描子目录');
        checkbox.addEventListener('change', () => {
            updateDatasetEditorRow(index, 'recursive', checkbox.checked);
        });
        const recursiveCopy = document.createElement('span');
        const recursiveTitleRow = document.createElement('div');
        recursiveTitleRow.className = 'dataset-inline-title-row';
        const recursiveTitle = document.createElement('strong');
        recursiveTitle.textContent = '递归扫描子目录 / recursive';
        const recursiveHelpBtn = createDatasetInlineHelpButton(helpDiv, '查看递归扫描说明');
        recursiveTitleRow.append(recursiveTitle, recursiveHelpBtn);
        recursiveCopy.appendChild(recursiveTitleRow);
        recursive.append(checkbox, recursiveCopy);

        const pattern = document.createElement('label');
        pattern.className = 'dataset-path-filter-pattern';
        const patternText = document.createElement('div');
        patternText.className = 'dataset-inline-title-row';
        const patternTitle = document.createElement('span');
        patternTitle.textContent = '路径筛选 / path_pattern';
        const patternHelpBtn = createDatasetInlineHelpButton(helpDiv, '查看路径筛选说明');
        patternText.append(patternTitle, patternHelpBtn);
        const patternInput = document.createElement('input');
        patternInput.type = 'text';
        patternInput.className = 'field-input';
        patternInput.value = String(row.path_pattern || '*').trim() || '*';
        patternInput.placeholder = '*';
        patternInput.title = '相对原始路径的 glob 筛选，例如 char_a/*；多个模式用 | 分隔。';
        patternInput.addEventListener('input', () => {
            updateDatasetEditorRow(index, 'path_pattern', patternInput.value);
        });
        pattern.append(patternText, patternInput);

        attachDatasetInlineHelp(recursiveHelpBtn, helpDiv, datasetLocalHelpSpec('recursive'), panel);
        attachDatasetInlineHelp(
            patternHelpBtn,
            helpDiv,
            () => createHelpContent('path_pattern', patternInput.value),
            panel,
        );

        panel.append(recursive, pattern, helpDiv);
        return panel;
    }

export function createDatasetRepeatSettingField(row, index) {
        const field = document.createElement('label');
        field.className = 'dataset-row-setting-field dataset-repeat-field dataset-repeat-setting-field';

        const labelRow = document.createElement('div');
        labelRow.className = 'dataset-row-setting-label';
        const label = document.createElement('span');
        label.className = 'field-name';
        label.textContent = '重复次数 / NUM_REPEATS';
        label.title = '这一组图片在每轮里重复使用几次。小数据集或重点角色可以适当提高，但过高会更容易过拟合。';
        labelRow.appendChild(label);

        const input = document.createElement('input');
        input.type = 'number';
        input.min = '1';
        input.step = '1';
        input.value = String(row.num_repeats || 1);
        input.className = 'field-input dataset-row-setting-input dataset-repeat-input';
        input.title = '每轮训练中这组数据的重复倍率。1 表示正常使用一次，2 表示等效看两遍。';
        input.addEventListener('input', () => updateDatasetEditorRow(index, 'num_repeats', input.value));

        field.append(labelRow, input);
        return field;
    }

export function createDatasetRowSettingsEditor(row, index) {
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const panel = document.createElement('div');
        panel.className = 'dataset-row-settings';
        const fields = [
            ['resolution', 'number'],
            ['enable_bucket', 'select'],
            ['validation_split', 'number'],
        ];
        const helpDiv = document.createElement('div');
        helpDiv.className = 'field-help dataset-row-settings-help';

        for (const [key, type] of fields) {
            const field = document.createElement('div');
            field.className = 'dataset-row-setting-field';
            const labelRow = document.createElement('div');
            labelRow.className = 'dataset-row-setting-label';
            const label = document.createElement('span');
            label.className = 'field-name';
            label.textContent = datasetConfigLabel(key);
            label.title = key;
            labelRow.appendChild(label);

            const btn = document.createElement('button');
            btn.className = 'info-toggle dataset-row-help-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '查看填写建议、好处、代价、风险和推荐';
            btn.addEventListener('click', () => {
                const wasActive = btn.classList.contains('active');
                panel.querySelectorAll('.dataset-row-help-toggle.active').forEach((activeBtn) => {
                    activeBtn.classList.remove('active');
                });
                helpDiv.classList.remove('visible');
                helpDiv.innerHTML = '';
                if (wasActive) return;
                btn.classList.add('active');
                helpDiv.appendChild(createHelpContent(key, datasetConfigValue(key, settings)));
                helpDiv.classList.add('visible');
            });
            labelRow.appendChild(btn);

            field.appendChild(labelRow);
            field.appendChild(createDatasetRowSettingInput(index, key, type, settings));
            panel.appendChild(field);
        }
        panel.appendChild(createDatasetRepeatSettingField(row, index));
        panel.appendChild(helpDiv);
        return panel;
    }

export function createDatasetAdvancedSettingsEditor(row, index) {
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const panel = document.createElement('div');
        panel.className = 'dataset-row-settings dataset-advanced-settings';
        const fields = [
            ['min_bucket_reso', 'number'],
            ['max_bucket_reso', 'number'],
            ['bucket_reso_steps', 'number'],
            ['bucket_no_upscale', 'select'],
            ['validation_split_num', 'number'],
            ['validation_seed', 'number'],
        ];
        const helpDiv = document.createElement('div');
        helpDiv.className = 'field-help dataset-row-settings-help';

        for (const [key, type] of fields) {
            const field = document.createElement('div');
            field.className = 'dataset-row-setting-field';
            const labelRow = document.createElement('div');
            labelRow.className = 'dataset-row-setting-label';
            const label = document.createElement('span');
            label.className = 'field-name';
            label.textContent = datasetConfigLabel(key);
            label.title = key;
            labelRow.appendChild(label);

            const btn = document.createElement('button');
            btn.className = 'info-toggle dataset-row-help-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '查看填写建议、好处、代价、风险和推荐';
            btn.addEventListener('click', () => {
                const wasActive = btn.classList.contains('active');
                panel.querySelectorAll('.dataset-row-help-toggle.active').forEach((activeBtn) => {
                    activeBtn.classList.remove('active');
                });
                helpDiv.classList.remove('visible');
                helpDiv.innerHTML = '';
                if (wasActive) return;
                btn.classList.add('active');
                helpDiv.appendChild(createHelpContent(key, datasetConfigValue(key, settings)));
                helpDiv.classList.add('visible');
            });
            labelRow.appendChild(btn);

            field.appendChild(labelRow);
            field.appendChild(createDatasetRowSettingInput(index, key, type, settings));
            panel.appendChild(field);
        }
        panel.appendChild(helpDiv);
        return panel;
    }

export function createDatasetCaptionExtensionEditor(row, index) {
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const panel = document.createElement('div');
        panel.className = 'dataset-caption-extension-advanced';
        panel.dataset.index = String(index);

        const copy = document.createElement('div');
        copy.className = 'dataset-caption-extension-copy';
        const titleRow = document.createElement('div');
        titleRow.className = 'dataset-caption-extension-title-row';
        const title = document.createElement('strong');
        title.textContent = '文本标注扩展名 / caption_extension';
        const helpDiv = createDatasetInlineHelp('dataset-caption-extension-help');
        const helpBtn = createDatasetInlineHelpButton(helpDiv, '查看文本标注扩展名说明');
        helpBtn.classList.add('dataset-caption-extension-help-toggle');
        titleRow.append(title, helpBtn);
        copy.appendChild(titleRow);

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input dataset-caption-extension-input';
        input.value = settings.caption_extension || '.txt';
        input.placeholder = '.txt';
        input.setAttribute('aria-label', '文本标注扩展名');
        input.addEventListener('input', () => {
            updateDatasetEditorRowsSettingValue(
                datasetExperimentalScopeIndices(index),
                'caption_extension',
                input.value,
            );
        });
        input.addEventListener('change', () => {
            updateDatasetEditorRowsSettingValue(
                datasetExperimentalScopeIndices(index),
                'caption_extension',
                input.value,
                { render: true },
            );
        });

        attachDatasetInlineHelp(
            helpBtn,
            helpDiv,
            () => createHelpContent('caption_extension', settings.caption_extension || '.txt'),
            panel,
        );

        panel.append(copy, input, helpDiv);
        return panel;
    }
