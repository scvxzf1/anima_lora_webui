/**
 * Dataset editor row builders and experimental subset controls.
 * Moved out of anima-app mechanical chunks.
 */
import {
    attachDatasetInlineHelp,
    bindDatasetExperimentalOpenState,
    createDatasetExperimentalAdvancedBody,
    createDatasetInlineHelp,
    createDatasetInlineHelpButton,
    datasetExperimentalOpenState,
    datasetLocalHelpSpec,
} from './inline-help.js?v=module-bootstrap-20260711-ir1';
import { createDatasetEditorDragHandle } from './config-input.js?v=module-bootstrap-20260711-ir1';
import { CAPTION_SOURCE_MODE_OPTIONS, help } from '../../config/catalog.js?v=module-bootstrap-20260711-ir1';
import { captionSourceModeLabel, normalizeCaptionSourceMode } from '../anima-app/helpers/caption-source.js?v=module-bootstrap-20260711-ir1';
import { createHelpContent } from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260711-ir1';
import { datasetConfigLabel, datasetConfigValue } from '../anima-app/helpers/dataset-config-fields.js?v=module-bootstrap-20260711-ir1';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    nlTagMixSummary,
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
    normalizeNlTagMix,
    normalizeTriggerClone,
} from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260711-ir1';
import { datasetPreviewValidationText } from '../anima-app/helpers/dataset-preview.js?v=module-bootstrap-20260711-ir1';
import { datasetEditorStateForActivePanel, isDatasetTabActive, refreshDatasetEditorItem, renderDatasetEditor } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir1';
import { createDatasetPathField, createDatasetRowCaptionSourceModeEditor, createDatasetRowSettingInput, openDatasetPreview, updateDatasetEditorRow } from './row-fields.js?v=module-bootstrap-20260711-ir1';
import { escapeHtml } from '../config-form/field-input.js?v=module-bootstrap-20260711-ir1';
import { datasetExperimentalScopeIndices, removeDatasetEditorRow, setDatasetExperimentalScopeIndices, updateDatasetEditorRowNlTagMix, updateDatasetEditorRowTriggerClone, updateDatasetEditorRowsSettingValue } from './mutations.js?v=module-bootstrap-20260711-ir1';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}


	    export function createDatasetEditorRow(row, index, item = null) {
	        const wrap = document.createElement('div');
	        wrap.className = 'dataset-editor-row';
	        wrap.dataset.index = String(index);
	        wrap.classList.toggle('is-selected', index === datasetState.selectedDatasetIndex);
	        wrap.addEventListener('click', (event) => {
	            if (event.target.closest('button, input, select, textarea, a, label, summary, .dataset-editor-drag-handle')) return;
	            if (datasetState.selectedDatasetIndex === index) return;
	            datasetState.selectedDatasetIndex = index;
	            renderDatasetEditor();
	        });
	        const head = document.createElement('div');
	        head.className = 'dataset-row-head';
        const titleBox = document.createElement('div');
        titleBox.className = 'dataset-row-title';
        const titleLine = document.createElement('div');
        titleLine.className = 'dataset-row-title-line';
        const mark = document.createElement('span');
        mark.className = 'dataset-row-mark';
        mark.textContent = '{}';
        const title = document.createElement('strong');
        title.textContent = `SUBSET ${index + 1} · 数据集组`;
        const subtitle = document.createElement('span');
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const mix = normalizeNlTagMix(row.nl_tag_mix);
        const triggerClone = normalizeTriggerClone(row.trigger_clone);
        const pathPattern = String(row.path_pattern || '*').trim() || '*';
        subtitle.textContent = [
            `${settings.resolution}px`,
            `桶 ${settings.min_bucket_reso}-${settings.max_bucket_reso}/${settings.bucket_reso_steps}`,
            `重复 ${row.num_repeats || 1}`,
            row.recursive === false ? '递归关闭' : '',
            pathPattern !== '*' ? `筛选 ${pathPattern}` : '',
            captionSourceModeLabel(settings.caption_source_mode),
            mix.enabled ? nlTagMixSummary(mix) : '',
            triggerClone.enabled ? `触发克隆 x${triggerClone.num_repeats}` : '',
	        ].filter(Boolean).join(' · ');
	        const dragHandle = createDatasetEditorDragHandle(index, item);
	        titleLine.append(mark, title);
	        titleBox.append(titleLine, subtitle);
        const headActions = document.createElement('div');
        headActions.className = 'dataset-row-head-actions';
        const badges = document.createElement('div');
        badges.className = 'dataset-row-badges';
        const bucketText = settings.enable_bucket
            ? `${settings.min_bucket_reso}-${settings.max_bucket_reso}/${settings.bucket_reso_steps}`
            : '关闭';
        const validationText = datasetPreviewValidationText(settings);
        const captionModeLabel = CAPTION_SOURCE_MODE_OPTIONS.find((option) => (
            option.value === normalizeCaptionSourceMode(settings.caption_source_mode)
        ))?.label || 'Auto';
        [
            ['分辨率', `${settings.resolution}px`],
            ['桶', bucketText],
            ['重复', row.num_repeats || 1],
            ['验证', validationText],
            ['标注', captionModeLabel],
        ].forEach(([label, value]) => {
            const badge = document.createElement('span');
            badge.className = 'dataset-row-badge';
            badge.innerHTML = `<small>${escapeHtml(String(label))}</small><strong>${escapeHtml(String(value))}</strong>`;
            badges.appendChild(badge);
        });
        headActions.appendChild(badges);
        if (isDatasetTabActive()) {
            const presetState = currentDatasetPresetState();
            const previewBtn = document.createElement('button');
            previewBtn.type = 'button';
            previewBtn.className = 'btn btn-small';
            previewBtn.textContent = '预览图片和标注';
            previewBtn.disabled = !presetState.selectedFile || presetState.dirty;
            previewBtn.title = previewBtn.disabled
                ? '请先保存当前数据集预设，再预览磁盘中的图片和同名标注。'
                : '打开这一组数据集的原始图预览，并读取同名 caption 标注。';
            previewBtn.addEventListener('click', () => openDatasetPreview(index));
            headActions.appendChild(previewBtn);
        }
	        head.append(dragHandle, titleBox, headActions);
        wrap.appendChild(head);

        const paths = document.createElement('div');
        paths.className = 'dataset-row-paths';
        paths.setAttribute('aria-label', `第 ${index + 1} 组数据集主路径`);
        paths.appendChild(createDatasetPathField(index, 'source_dir', '原始数据集路径', row.source_dir, 'image_dataset'));
        wrap.appendChild(paths);

        wrap.appendChild(createDatasetRowCaptionSourceModeEditor(settings, index));
        wrap.appendChild(createDatasetNlTagMixEditor(row, index));
        wrap.appendChild(createDatasetRowSettingsEditor(row, index));

        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'btn btn-small danger dataset-remove-btn';
        remove.textContent = '删除';
        remove.disabled = datasetEditorStateForActivePanel().datasets.length <= 1;
        remove.title = remove.disabled ? '至少保留一组数据集路径' : '从当前 dataset_config 中移除这一组路径，不会删除磁盘文件。';
        remove.addEventListener('click', () => removeDatasetEditorRow(index));
        const bottomActions = document.createElement('div');
        bottomActions.className = 'dataset-row-bottom-actions';
        bottomActions.append(remove);
        wrap.appendChild(bottomActions);
        return wrap;
    }

    export function createDatasetExperimentalFeaturesEditor(row, index) {
        const panel = document.createElement('details');
        panel.className = 'dataset-experimental-features';
        panel.dataset.index = String(index);
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const clone = normalizeTriggerClone(row.trigger_clone);
        const pathPattern = String(row.path_pattern || '*').trim() || '*';
        const defaultOpen = clone.enabled
            || row.recursive === false
            || pathPattern !== '*'
            || (settings.caption_extension && settings.caption_extension !== '.txt')
            || row.is_reg === true;
        panel.open = datasetExperimentalOpenState(index, defaultOpen);
        bindDatasetExperimentalOpenState(panel, index);

        const head = document.createElement('summary');
        head.className = 'dataset-experimental-head';
        const titleRow = document.createElement('div');
        titleRow.className = 'dataset-experimental-title-row';
        const title = document.createElement('strong');
        title.textContent = '实验性/高级/旧功能';
        const overviewHelp = createDatasetInlineHelp('dataset-inline-help dataset-experimental-overview-help');
        const overviewHelpBtn = createDatasetInlineHelpButton(overviewHelp, '查看高级功能说明');
        titleRow.append(title, overviewHelpBtn);
        const note = document.createElement('span');
        note.textContent = `对应第 ${index + 1} 组数据集`;
        head.append(titleRow, note);

        const { body, detailBtn } = createDatasetExperimentalAdvancedBody(row, index, overviewHelp, {
            createDatasetCaptionExtensionEditor,
            createDatasetExperimentalScopePicker,
            createDatasetPathFilterEditor,
            createDatasetTriggerCloneEditor,
        });
        attachDatasetInlineHelp(
            overviewHelpBtn,
            overviewHelp,
            datasetLocalHelpSpec('experimental'),
            panel,
            { openDetails: true },
        );
        attachDatasetInlineHelp(
            detailBtn,
            overviewHelp,
            datasetLocalHelpSpec('experimental'),
            panel,
            { openDetails: true },
        );

        const advancedSettings = createDatasetAdvancedSettingsEditor(row, index);
        const notice = body.querySelector('.dataset-experimental-notice');
        const overviewHelpNode = body.querySelector('.dataset-experimental-overview-help');
        if (overviewHelpNode && overviewHelpNode.nextSibling) {
            body.insertBefore(advancedSettings, overviewHelpNode.nextSibling);
        } else if (notice && notice.nextSibling) {
            body.insertBefore(advancedSettings, notice.nextSibling);
        } else {
            body.insertBefore(advancedSettings, body.firstChild);
        }
        panel.append(head, body);
        return panel;
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

    export function createDatasetNlTagMixEditor(row, index) {
        const mix = normalizeNlTagMix(row.nl_tag_mix);
        const panel = document.createElement('div');
        panel.className = ['dataset-nl-tag-mix', mix.enabled ? 'enabled' : ''].filter(Boolean).join(' ');
        panel.dataset.index = String(index);
        const helpDiv = createDatasetInlineHelp('dataset-inline-help dataset-nl-tag-help');

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
        const toggleText = document.createElement('span');
        const toggleTitleRow = document.createElement('div');
        toggleTitleRow.className = 'dataset-inline-title-row';
        const toggleTitle = document.createElement('strong');
        toggleTitle.textContent = 'captions格式nl/tag权重调整';
        const helpBtn = createDatasetInlineHelpButton(helpDiv, '查看 nl/tag 权重说明');
        toggleTitleRow.append(toggleTitle, helpBtn);
        toggleText.appendChild(toggleTitleRow);
        toggle.append(checkbox, toggleText);

        const ratio = document.createElement('label');
        ratio.className = 'dataset-nl-tag-ratio';
        const ratioHead = document.createElement('span');
        ratioHead.textContent = 'tag 占比';
        const ratioInput = document.createElement('input');
        ratioInput.type = 'range';
        ratioInput.min = '0';
        ratioInput.max = '100';
        ratioInput.step = '5';
        ratioInput.value = String(Math.round(mix.tag_ratio * 100));
        ratioInput.disabled = !mix.enabled;
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

        attachDatasetInlineHelp(helpBtn, helpDiv, datasetLocalHelpSpec('nlTagMix'), panel);

        panel.append(toggle, ratio, ratioNumber, summary, helpDiv);
        return panel;
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
