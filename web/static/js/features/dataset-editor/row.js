/**
 * Dataset editor main row shell and experimental features panel.
 */
import {
    attachDatasetInlineHelp,
    bindDatasetExperimentalOpenState,
    createDatasetExperimentalAdvancedBody,
    createDatasetInlineHelp,
    createDatasetInlineHelpButton,
    datasetExperimentalOpenState,
    datasetLocalHelpSpec,
} from './inline-help.js?v=module-bootstrap-20260809-nf4-v2';
import { createDatasetEditorDragHandle } from './item-drag.js?v=module-bootstrap-20260809-nf4-v2';
import { CAPTION_SOURCE_MODE_OPTIONS, help } from '../../config/catalog.js?v=module-bootstrap-20260809-nf4-v2';
import { normalizeCaptionSourceMode } from '../anima-app/helpers/caption-source.js?v=module-bootstrap-20260809-nf4-v2';
import { createHelpContent } from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { datasetConfigLabel, datasetConfigValue } from '../anima-app/helpers/dataset-config-fields.js?v=module-bootstrap-20260809-nf4-v2';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
    normalizeTriggerClone,
} from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260809-nf4-v2';
import { datasetPreviewValidationText } from '../anima-app/helpers/dataset-preview.js?v=module-bootstrap-20260809-nf4-v2';
import { datasetEditorStateForActivePanel, isDatasetTabActive, refreshDatasetEditorItem, renderDatasetEditor } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { createDatasetPathField, createDatasetRowCaptionSourceModeEditor, createDatasetRowSettingInput, updateDatasetEditorRow } from './row-fields.js?v=module-bootstrap-20260809-nf4-v2';
import { openDatasetPreview } from './preview.js?v=module-bootstrap-20260809-nf4-v2';
import { escapeHtml } from '../config-form/field-input.js?v=module-bootstrap-20260809-nf4-v2';
import {
    datasetExperimentalScopeIndices,
    removeDatasetEditorRow,
    setDatasetExperimentalScopeIndices,
    updateDatasetEditorRowNlTagMix,
    updateDatasetEditorRowTriggerClone,
    updateDatasetEditorRowsSettingValue,
} from './mutations.js?v=module-bootstrap-20260809-nf4-v2';
import {
    createDatasetAdvancedSettingsEditor,
    createDatasetCaptionExtensionEditor,
    createDatasetExperimentalScopePicker,
    createDatasetMainPolicyRow,
    createDatasetPathFilterEditor,
    createDatasetRepeatSettingField,
    createDatasetRowSettingsEditor,
    createDatasetTriggerCloneEditor,
} from './row-settings.js?v=module-bootstrap-20260809-nf4-v2';

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
        title.className = 'dataset-row-title-text';
        title.textContent = `SUBSET ${index + 1} · 数据集组`;
        const settings = normalizeDatasetDefaults(row.settings || datasetEditorStateForActivePanel().defaults || {});
        const dragHandle = createDatasetEditorDragHandle(index, item);
        // Left side keeps only mark + title; summary details live in right badges.
        titleLine.append(mark, title);
        titleBox.append(titleLine);
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
        wrap.appendChild(createDatasetMainPolicyRow(row, index));
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
            || (settings.caption_extension && settings.caption_extension !== '.txt');
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
