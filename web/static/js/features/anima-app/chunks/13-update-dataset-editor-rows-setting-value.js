/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    DEFAULT_TRIGGER_CLONE,
    METHOD_GUIDE_ZH,
    PRESET_GUIDE_ZH,
    VARIANT_GUIDE_ZH,
    choiceHelp,
} from '../../../config/catalog.js?v=module-bootstrap-20260711-ir1';
import {
    normalizeDatasetDefaults,
    normalizeDatasetEditorRows,
    normalizeNlTagMix,
    normalizeTriggerClone,
} from '../helpers/dataset-values.js?v=module-bootstrap-20260711-ir1';
import {
    isTruthy,
    precisionPreferenceFromConfig,
} from '../helpers/config-values.js?v=module-bootstrap-20260711-ir1';
import {
    choiceLine,
    defaultMethodGuide,
    defaultPresetGuide,
    defaultVariantGuide,
} from '../helpers/choice-guide.js?v=module-bootstrap-20260711-ir1';
import {
    compactList,
    flagDetail,
    valueDetail,
} from '../helpers/config-field-display.js?v=module-bootstrap-20260711-ir1';
import {
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    renderDatasetPresetHeader,
} from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir1';
import { getAppShellState } from '../helpers/app-shell-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { escapeHtml, setFieldInputValue } from '../../config-form/field-input.js?v=module-bootstrap-20260711-ir1';
import {
    clearCurrentTrainingSource,
    setCurrentTrainingSourceFromVariant,
} from '../../training-source/source-state.js?v=module-bootstrap-20260711-ir1';
import { outputRunRuntimeFile } from '../../output-run/runtime-file.js?v=module-bootstrap-20260711-ir1';
import { activeMethodKey, inferMethodFromConfig } from '../../config-form/method-key.js?v=module-bootstrap-20260711-ir1';
import { handlePendingConfigSwitch, updateTomlDirtyState } from '../helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir1';
import { getConfigState } from '../helpers/config-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { val } from '../helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { updateStepEstimatePanel } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260711-ir1';

const appShellState = getAppShellState();
const configState = getConfigState();
const datasetState = getDatasetState();
const trainingState = getTrainingState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function datasetExperimentalScopeSelectionsState() {
    return datasetState.datasetExperimentalScopeSelections;
}

export {
    setFieldInputValue,
    escapeHtml,
    setCurrentTrainingSourceFromVariant,
    clearCurrentTrainingSource,
    outputRunRuntimeFile,
    activeMethodKey,
    inferMethodFromConfig,
};


    export function updateDatasetEditorRowsSettingValue(indices, key, value, options = {}) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        const targets = datasetValidTargetIndices(indices, rows.length);
        if (!targets.length) return;
        for (const targetIndex of targets) {
            const settings = normalizeDatasetDefaults(rows[targetIndex].settings || state.defaults || {});
            settings[key] = value;
            rows[targetIndex].settings = settings;
        }
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.datasets = rows;
        } else {
            datasetState.datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        if (options.render === 'item') {
            refreshDatasetEditorItems(targets) || renderDatasetEditor();
        } else if (options.render) {
            renderDatasetEditor();
        }
    }

    export function updateDatasetEditorRowNlTagMix(index, nextMix, options = {}) {
        updateDatasetEditorRowsNlTagMix([index], nextMix, options);
    }

    export function updateDatasetEditorRowsNlTagMix(indices, nextMix, options = {}) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        const targets = datasetValidTargetIndices(indices, rows.length);
        if (!targets.length) return;
        const mix = normalizeNlTagMix(nextMix);
        for (const targetIndex of targets) {
            rows[targetIndex].nl_tag_mix = mix;
        }
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.datasets = rows;
        } else {
            datasetState.datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        if (options.render === false) {
            return;
        }
        refreshDatasetEditorItems(targets) || renderDatasetEditor();
    }

    export function updateDatasetEditorRowTriggerClone(index, nextClone, options = {}) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (!rows[index]) return;
        rows[index].trigger_clone = normalizeTriggerClone({
            ...rows[index].trigger_clone,
            ...nextClone,
        });
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.datasets = rows;
        } else {
            datasetState.datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        if (options.render) {
            renderDatasetEditor();
        }
    }

    function datasetExperimentalScopeKey(index) {
        return `${isDatasetTabActive() ? 'dataset-preset' : 'config-dataset'}:${index}`;
    }

    export function datasetExperimentalScopeIndices(index, total = null) {
        const state = datasetEditorStateForActivePanel();
        const count = total ?? normalizeDatasetEditorRows(state.datasets).length;
        const key = datasetExperimentalScopeKey(index);
        const raw = datasetExperimentalScopeSelectionsState().get(key) || [index];
        const selected = datasetValidTargetIndices(raw, count);
        if (!selected.length && index >= 0 && index < count) {
            selected.push(index);
        }
        datasetExperimentalScopeSelectionsState().set(key, selected);
        return selected;
    }

    export function setDatasetExperimentalScopeIndices(index, indices) {
        const state = datasetEditorStateForActivePanel();
        const count = normalizeDatasetEditorRows(state.datasets).length;
        const selected = datasetValidTargetIndices(indices, count);
        if (!selected.length && index >= 0 && index < count) {
            selected.push(index);
        }
        datasetExperimentalScopeSelectionsState().set(datasetExperimentalScopeKey(index), selected);
    }

	    export function datasetValidTargetIndices(indices, count) {
	        return [...new Set((indices || [])
	            .map((value) => Number.parseInt(value, 10))
	            .filter((value) => Number.isInteger(value) && value >= 0 && value < count))]
	            .sort((left, right) => left - right);
	    }

	    function setDatasetEditorRowsAfterSort(rows) {
	        datasetExperimentalScopeSelectionsState().clear();
	        if (isDatasetTabActive()) {
	            datasetState.datasetPresetState.datasets = rows;
	        } else {
	            datasetState.datasetEditorState.datasets = rows;
	        }
	        markDatasetEditorDirty();
	        renderDatasetEditor();
	    }

	    export function moveDatasetEditorRow(sourceIndex, targetIndex, placeAfter = false) {
	        const rows = normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
	        if (rows.length <= 1) return false;
	        if (sourceIndex < 0 || sourceIndex >= rows.length || targetIndex < 0 || targetIndex >= rows.length) return false;
	        if (sourceIndex === targetIndex) return false;
	        let insertIndex = targetIndex + (placeAfter ? 1 : 0);
	        if (sourceIndex < insertIndex) insertIndex -= 1;
	        insertIndex = Math.max(0, Math.min(rows.length - 1, insertIndex));
	        if (insertIndex === sourceIndex) return false;
	        const [moved] = rows.splice(sourceIndex, 1);
	        rows.splice(insertIndex, 0, moved);
	        setDatasetEditorRowsAfterSort(rows);
	        return true;
	    }

	    export function moveDatasetEditorRowToIndex(sourceIndex, targetIndex) {
	        const rows = normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
	        const clamped = Math.max(0, Math.min(rows.length - 1, targetIndex));
	        if (clamped === sourceIndex) return false;
	        return moveDatasetEditorRow(sourceIndex, clamped, clamped > sourceIndex);
	    }

	    export function markDatasetEditorDirty() {
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.dirty = true;
            datasetState.datasetPresetState.status = '有未保存的数据集修改';
            renderDatasetPresetHeader();
        } else {
            datasetState.datasetEditorState.dirty = true;
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
        const dirty = document.querySelector('#dataset-editor .dataset-editor-dirty');
        if (dirty) {
            dirty.classList.add('active');
            dirty.textContent = '有未保存的数据集修改';
        }
    }

    export function addDatasetEditorRow() {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        rows.push({
            source_dir: '',
            image_dir: '',
            cache_dir: '',
            num_repeats: 1,
            trigger_clone: normalizeTriggerClone(DEFAULT_TRIGGER_CLONE),
            settings: normalizeDatasetDefaults(state.defaults || {}),
        });
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.datasets = rows;
            datasetState.datasetPresetState.dirty = true;
        } else {
            datasetState.datasetEditorState.datasets = rows;
            datasetState.datasetEditorState.dirty = true;
        }
        renderDatasetEditor();
        if (!isDatasetTabActive()) {
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
    }

    export function removeDatasetEditorRow(index) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (rows.length <= 1) return;
        rows.splice(index, 1);
        if (isDatasetTabActive()) {
            datasetState.datasetPresetState.datasets = rows;
            datasetState.datasetPresetState.dirty = true;
        } else {
            datasetState.datasetEditorState.datasets = rows;
            datasetState.datasetEditorState.dirty = true;
        }
        renderDatasetEditor();
        if (!isDatasetTabActive()) {
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
    }

    export function syncDatasetEditorToCompatFields() {
        const datasetEditorState = currentDatasetEditorState();
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        const first = rows[0];
        if (!first) return;
        setFieldInputValue('source_image_dir', first.source_dir);
        setFieldInputValue('resized_image_dir', first.image_dir);
        setFieldInputValue('lora_cache_dir', first.cache_dir);
        if (datasetEditorState.dataset_config) {
            setFieldInputValue('dataset_config', datasetEditorState.dataset_config);
        }
    }


    export function rememberSelectionSnapshot() {
        const selectionSnapshot = configState.selectionSnapshot;
        selectionSnapshot.method = val('method-select');
        selectionSnapshot.variant = val('variant-select');
        selectionSnapshot.preset = val('preset-select');
    }

    export function restoreSelectionSnapshot() {
        const selectionSnapshot = configState.selectionSnapshot;
        const methodSelect = document.getElementById('method-select');
        const variantSelect = document.getElementById('variant-select');
        const presetSelect = document.getElementById('preset-select');
        if (methodSelect && selectionSnapshot.method && [...methodSelect.options].some((opt) => opt.value === selectionSnapshot.method)) {
            methodSelect.value = selectionSnapshot.method;
        }
        if (variantSelect && selectionSnapshot.variant && [...variantSelect.options].some((opt) => opt.value === selectionSnapshot.variant)) {
            variantSelect.value = selectionSnapshot.variant;
        }
        if (presetSelect && selectionSnapshot.preset && [...presetSelect.options].some((opt) => opt.value === selectionSnapshot.preset)) {
            presetSelect.value = selectionSnapshot.preset;
        }
        setCurrentTrainingSourceFromVariant(val('variant-select'));
        updateChoiceGuide();
    }

    export async function confirmBeforeConfigSelectionChange(message) {
        const ok = await handlePendingConfigSwitch({
            targetLabel: '新的配置选择',
        });
        if (!ok) restoreSelectionSnapshot();
        return ok;
    }

    export function updateChoiceGuide(config = currentConfigState()) {
        const currentTrainingSource = currentTrainingSourceState();
        const container = document.getElementById('choice-guide');
        if (!container) return;
        container.innerHTML = '';
        const methodKey = activeMethodKey(config);
        container.appendChild(createChoiceCard('方法', methodKey, METHOD_GUIDE_ZH, defaultMethodGuide(), methodGuideFromConfig(methodKey, config)));
        const sourceKey = currentTrainingSource.method || val('variant-select');
        container.appendChild(createChoiceCard('配置', sourceKey, VARIANT_GUIDE_ZH, defaultVariantGuide(), configGuideFromCurrentSource(sourceKey, config)));
        const presetKey = val('preset-select');
        container.appendChild(createChoiceCard('预设', presetKey, PRESET_GUIDE_ZH, defaultPresetGuide(), presetGuideFromConfig(presetKey, config)));
    }

    export function createChoiceCard(kind, key, guideMap, fallback, overrideGuide = null) {
        const guide = overrideGuide || guideMap[key] || fallback;
        const helpId = `choice-guide-hint-${++configState.choiceGuideHintSeq}`;
        const card = document.createElement('article');
        card.className = 'choice-card';

        const heading = document.createElement('div');
        heading.className = 'choice-card-heading';
        const title = document.createElement('strong');
        title.textContent = `${kind}: ${key || '-'}`;
        const name = document.createElement('span');
        name.textContent = guide.title;
        heading.appendChild(title);
        heading.appendChild(name);
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'info-toggle choice-info-toggle';
        toggle.textContent = '?';
        toggle.title = `展开${kind}说明`;
        toggle.setAttribute('aria-label', `${kind}说明`);
        toggle.setAttribute('aria-expanded', 'false');
        toggle.setAttribute('aria-controls', helpId);
        heading.appendChild(toggle);
        card.appendChild(heading);

        const body = document.createElement('div');
        body.id = helpId;
        body.className = 'choice-card-body';
        body.hidden = true;
        body.appendChild(choiceLine('说明', guide.summary));
        body.appendChild(choiceLine('取舍', guide.tradeoff));
        body.appendChild(choiceLine('推荐', guide.recommend, 'choice-recommend'));
        if (Array.isArray(guide.details) && guide.details.length) {
            const details = document.createElement('ul');
            details.className = 'choice-details';
            for (const detail of guide.details) {
                const item = document.createElement('li');
                item.textContent = detail;
                details.appendChild(item);
            }
            body.appendChild(details);
        }
        toggle.addEventListener('click', () => {
            const nextOpen = body.hidden;
            body.hidden = !nextOpen;
            toggle.classList.toggle('active', nextOpen);
            toggle.setAttribute('aria-expanded', String(nextOpen));
            toggle.title = nextOpen ? `收起${kind}说明` : `展开${kind}说明`;
        });
        card.appendChild(body);
        return card;
    }


    export function methodGuideFromConfig(methodKey, config = currentConfigState()) {
        const base = METHOD_GUIDE_ZH[methodKey] || defaultMethodGuide();
        const details = compactList([
            flagDetail('use_glora', 'GLoRA', config.use_glora),
            flagDetail('use_vera', 'VeRA', config.use_vera),
            flagDetail('use_lokr', 'LoKr', config.use_lokr),
            flagDetail('use_loha', 'LoHa', config.use_loha),
            flagDetail('dora_wd', 'DoRA', config.dora_wd),
            isTruthy(config.use_lokr) ? valueDetail('lokr_factor', config.lokr_factor) : '',
            isTruthy(config.use_vera) ? valueDetail('vera_projection_prng_key', config.vera_projection_prng_key) : '',
            isTruthy(config.use_vera) ? valueDetail('vera_d_initial', config.vera_d_initial) : '',
            valueDetail('network_dim', config.network_dim),
            valueDetail('network_alpha', config.network_alpha),
            valueDetail('learning_rate', config.learning_rate),
            valueDetail('max_train_epochs', config.max_train_epochs),
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 当前表单已读取关键训练字段。`,
            details,
        };
    }

    export function configGuideFromCurrentSource(sourceKey, config = currentConfigState()) {
        const currentTrainingSource = currentTrainingSourceState();
        const globalSettings = appShellState.globalSettings;
        const isImported = currentTrainingSource.methods_subdir === 'imported';
        const base = isImported
            ? choiceHelp(
                '导入训练配置',
                `当前表单来自 ${currentTrainingSource.file || '导入配置'}。`,
                '它会按 base.toml → 当前预设 → 该 TOML 的顺序合并；不会强行加入变体下拉。',
                '适合把历史训练配置作为独立入口继续查看、预检测或训练。'
            )
            : (VARIANT_GUIDE_ZH[sourceKey] || defaultVariantGuide());
        const details = compactList([
            currentTrainingSource.file ? `文件: ${currentTrainingSource.file}` : '',
            config.dataset_config ? `数据集配置: ${config.dataset_config}` : '',
            config.output_name ? `输出名称: ${config.output_name}` : '',
            globalSettings?.output_root ? `Web 输出根目录: ${globalSettings.output_root}` : '',
            config.source_image_dir ? `原始数据集: ${config.source_image_dir}` : '',
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 已读取当前 TOML 的路径和输出信息。`,
            details,
        };
    }

    export function presetGuideFromConfig(presetKey, config = currentConfigState()) {
        const base = PRESET_GUIDE_ZH[presetKey] || defaultPresetGuide();
        const details = compactList([
            valueDetail('precision_preference', precisionPreferenceFromConfig(config)),
            valueDetail('optimizer_type', config.optimizer_type),
            valueDetail('lr_scheduler', config.lr_scheduler),
            valueDetail('train_batch_size', config.train_batch_size),
            valueDetail('gradient_accumulation_steps', config.gradient_accumulation_steps),
            valueDetail('sample_ratio', config.sample_ratio),
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 当前已合并后的预设/配置值如下。`,
            details,
        };
    }
