/**
 * Dataset editor row field builders and value updates.
 * Extracted from former chunk 12.
 */
import {
    CAPTION_SOURCE_MODE_OPTIONS,
    help,
} from '../../config/catalog.js?v=module-bootstrap-20260809-nf4-v2';
import {
    captionSourceModeLabel,
    normalizeCaptionSourceMode,
} from '../anima-app/helpers/caption-source.js?v=module-bootstrap-20260809-nf4-v2';
import { datasetConfigValue } from '../anima-app/helpers/dataset-config-fields.js?v=module-bootstrap-20260809-nf4-v2';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
} from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260809-nf4-v2';
import {
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
} from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { updateStepEstimatePanel } from '../config-form/step-estimate.js?v=module-bootstrap-20260809-nf4-v2';
import { setFieldInputValue } from '../config-form/field-input.js?v=module-bootstrap-20260809-nf4-v2';
import {
    markDatasetEditorDirty,
    updateDatasetEditorRowsSettingValue,
} from './mutations.js?v=module-bootstrap-20260809-nf4-v2';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

export function createDatasetRowCaptionSourceModeEditor(settings, index) {
        const current = normalizeCaptionSourceMode(settings.caption_source_mode, settings.prefer_json_caption);
        const panel = document.createElement('div');
        panel.className = 'dataset-caption-source';
        panel.dataset.mode = current;

        const head = document.createElement('div');
        head.className = 'dataset-caption-source-head';
        const copy = document.createElement('div');
        copy.className = 'dataset-caption-source-copy';
        const titleRow = document.createElement('div');
        titleRow.className = 'dataset-caption-source-title-row';
        const title = document.createElement('strong');
        title.textContent = '标注来源 / caption_source_mode';
        const helpId = `dataset-caption-source-notes-${++datasetState.datasetCaptionSourceHelpSeq}`;
        const helpBtn = document.createElement('button');
        helpBtn.className = 'info-toggle dataset-caption-source-help-toggle';
        helpBtn.type = 'button';
        helpBtn.textContent = '?';
        helpBtn.title = '展开标注来源注释';
        helpBtn.setAttribute('aria-label', '标注来源格式注释');
        helpBtn.setAttribute('aria-controls', helpId);
        helpBtn.setAttribute('aria-expanded', 'false');
        titleRow.append(title, helpBtn);
        const desc = document.createElement('span');
        desc.textContent = '默认 auto 自动识别；保存后预览和训练前预检测都会显示识别结果，也可以强制指定格式。';
        copy.append(titleRow, desc);
        const state = document.createElement('span');
        state.className = 'dataset-caption-source-state';
        state.textContent = captionSourceModeLabel(current);
        head.append(copy, state);

        const controls = document.createElement('div');
        controls.className = 'dataset-caption-source-options';
        CAPTION_SOURCE_MODE_OPTIONS.forEach((option) => {
            const label = document.createElement('label');
            label.className = ['dataset-caption-source-option', option.value === current ? 'selected' : ''].filter(Boolean).join(' ');
            const input = document.createElement('input');
            input.type = 'radio';
            input.name = `dataset-caption-source-${index}`;
            input.value = option.value;
            input.checked = option.value === current;
            input.setAttribute('aria-label', `${option.label} ${option.detail}`);
            input.addEventListener('change', () => {
                if (!input.checked) return;
                updateDatasetEditorRowsSettingValue(
                    [index],
                    'caption_source_mode',
                    option.value,
                    { render: 'item' },
                );
            });
            const labelText = document.createElement('span');
            labelText.textContent = option.label;
            const detail = document.createElement('small');
            detail.textContent = option.detail;
            label.append(input, labelText, detail);
            controls.appendChild(label);
        });

        const notes = document.createElement('ul');
        notes.className = 'dataset-caption-source-notes';
        notes.id = helpId;
        notes.hidden = true;
        [
            '"1.png+1.txt"*n = sd-scripts格式标注',
            '"1.png+1.json"*n = AnimaLoraToolkit格式标注',
            '"png*n"+captions.json = DiffPipeForge格式标注',
            'caption_extension 仅影响 txt 来源或 auto 回退到文本标注；json / captions.json 模式会忽略它。',
        ].forEach((text) => {
            const item = document.createElement('li');
            item.textContent = text;
            notes.appendChild(item);
        });
        helpBtn.addEventListener('click', () => {
            const nextVisible = notes.hidden;
            notes.hidden = !nextVisible;
            helpBtn.classList.toggle('active', nextVisible);
            helpBtn.setAttribute('aria-expanded', String(nextVisible));
            helpBtn.title = nextVisible ? '收起标注来源注释' : '展开标注来源注释';
        });

        panel.append(head, controls, notes);
        return panel;
    }

export function createDatasetRowSettingInput(index, key, type, settings) {
        let input;
        if (type === 'select') {
            input = document.createElement('select');
            const options = key === 'enable_bucket'
                ? [[true, '启用'], [false, '关闭']]
                : [[false, '允许放大'], [true, '不放大小图']];
            const current = Boolean(settings[key]);
            for (const [value, label] of options) {
                const opt = document.createElement('option');
                opt.value = value ? 'true' : 'false';
                opt.textContent = label;
                opt.selected = value === current;
                input.appendChild(opt);
            }
        } else {
            input = document.createElement('input');
            input.type = type;
            input.value = datasetConfigValue(key, settings);
            if (type === 'number') {
                input.min = '0';
                input.step = key === 'validation_split' ? '0.001' : (key === 'resolution' || key.endsWith('_reso') || key === 'bucket_reso_steps' ? '16' : '1');
            }
        }
        input.className = 'field-input dataset-row-setting-input';
        input.addEventListener('input', () => updateDatasetEditorRowSetting(index, key, input));
        input.addEventListener('change', () => updateDatasetEditorRowSetting(index, key, input));
        return input;
    }

export function createDatasetPathField(index, key, label, value, placeholder) {
        const field = document.createElement('label');
        field.className = 'dataset-path-field';
        field.dataset.key = key;
        const text = document.createElement('span');
        text.textContent = label;
        const titles = {
            source_dir: '原始图片和 caption 所在目录。预处理从这里读图；缩放图和 LoRA 缓存会写入本次训练运行目录。',
            image_dir: '缩放图目录。预处理会把图片按分辨率/分桶规则写到这里；训练从这里枚举训练图片。',
            cache_dir: 'LoRA 缓存目录。VAE latent、文本编码器缓存、PE 特征缓存会写到这里；训练用它加速。',
        };
        text.title = titles[key] || label;
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'field-input dataset-path-input';
        input.value = value || '';
        input.placeholder = placeholder;
        input.title = titles[key] || '';
        input.addEventListener('input', () => updateDatasetEditorRow(index, key, input.value));
        field.append(text, input);
        return field;
    }

export function updateDatasetDefault(key, input) {
        const state = datasetEditorStateForActivePanel();
        const defaults = normalizeDatasetDefaults(state.defaults || {});
        if (input.type === 'checkbox') {
            defaults[key] = input.checked;
        } else if (input.tagName === 'SELECT') {
            defaults[key] = input.value === 'true';
        } else if (input.type === 'number') {
            defaults[key] = key === 'validation_split' || key === 'prior_loss_weight'
                ? Math.max(0, Number(input.value) || 0)
                : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            defaults[key] = input.value;
        }
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.defaults = defaults;
        } else {
            datasetState.datasetEditorState.defaults = defaults;
        }
        markDatasetEditorDirty();
    }

export function updateDatasetEditorRow(index, key, value) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (!rows[index]) return;
        if (key === 'source_dir' && rows[index].source_dir !== value) {
            rows[index].image_dir = '';
            rows[index].cache_dir = '';
        }
        rows[index][key] = key === 'num_repeats'
            ? Math.max(1, Number.parseInt(value || '1', 10) || 1)
            : value;
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.datasets = rows;
        } else {
            datasetState.datasetEditorState.datasets = rows;
        }
        if (!isDatasetTabActive() && index === 0 && key === 'source_dir') {
            setFieldInputValue('source_image_dir', value);
        }
        markDatasetEditorDirty();
        if (key === 'num_repeats') {
            updateStepEstimatePanel();
        }
    }

    function updateDatasetEditorRowSetting(index, key, input) {
        let value;
        if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.tagName === 'SELECT') {
            value = input.value === 'true';
        } else if (input.type === 'number') {
            value = key === 'validation_split' || key === 'prior_loss_weight'
                ? Math.max(0, Number(input.value) || 0)
                : Math.max(0, Number.parseInt(input.value || '0', 10) || 0);
        } else {
            value = input.value;
        }
        updateDatasetEditorRowSettingValue(index, key, value);
    }

export function updateDatasetEditorRowSettingValue(index, key, value) {
        updateDatasetEditorRowsSettingValue([index], key, value);
    }
