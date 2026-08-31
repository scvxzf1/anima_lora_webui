/**
 * No-dataset regularization intent layer for the config form.
 * Extracted from anima-app chunk 05a.
 */
import {
    NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY,
    NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY_OPEN,
    NO_DATASET_REGULARIZATION_CACHE_PATCH,
    NO_DATASET_REGULARIZATION_CONFLICT_MESSAGE,
    NO_DATASET_REGULARIZATION_CONFLICT_MODE,
    NO_DATASET_REGULARIZATION_DEFAULT_WEIGHT,
    NO_DATASET_REGULARIZATION_DOP_CLASS_REQUIRED,
    NO_DATASET_REGULARIZATION_MODE_SPECS,
} from '../anima-app/helpers/app-constants.js?v=module-bootstrap-20260831-release-v1';
import { originalConfigFieldValue, readFieldInputValue } from '../anima-app/helpers/config-form-bridge.js?v=module-bootstrap-20260831-release-v1';
import { setFieldInputValue } from './field-input.js?v=module-bootstrap-20260831-release-v1';
import {
    configureNoDatasetRegularizationModePanelUpdater,
    handleFormFieldChange,
} from './form-fields.js?v=module-bootstrap-20260831-release-v1';
import { appendFieldRows } from './field-rows.js?v=module-bootstrap-20260831-release-v1';
import {
    setTomlStatus,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260831-release-v1';

const configFormState = getConfigState().configFormState;

    function createNoDatasetRegularizationModePanel() {
        const panel = document.createElement('div');
        panel.className = 'no-dataset-regularization-panel';

        const head = document.createElement('div');
        head.className = 'no-dataset-regularization-head';
        const title = document.createElement('strong');
        title.textContent = '正则化方案';
        const note = document.createElement('span');
        note.textContent = '先选择意图，WebUI 会同步底层参数。';
        head.append(title, note);
        panel.appendChild(head);

        const modes = document.createElement('div');
        modes.className = 'no-dataset-regularization-modes';
        modes.setAttribute('role', 'radiogroup');
        modes.setAttribute('aria-label', '无数据集正则化方案');
        for (const spec of NO_DATASET_REGULARIZATION_MODE_SPECS) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'no-dataset-regularization-mode';
            btn.dataset.noDatasetRegularizationMode = spec.id;
            btn.setAttribute('role', 'radio');
            btn.setAttribute('aria-checked', 'false');
            const label = document.createElement('strong');
            label.textContent = spec.label;
            const desc = document.createElement('span');
            desc.textContent = spec.note;
            btn.append(label, desc);
            btn.addEventListener('click', () => applyNoDatasetRegularizationMode(spec.id));
            modes.appendChild(btn);
        }
        panel.appendChild(modes);

        const controls = document.createElement('div');
        controls.className = 'no-dataset-regularization-controls';
        controls.appendChild(createNoDatasetRegularizationNumberControl({
            key: 'prior_preservation_weight',
            modeClass: 'no-dataset-control-prior-weight',
            label: '先验保留权重',
            hint: '控制“别偏离底模太远”的辅助 loss 强度。0 关闭；0.05 轻约束；0.1 适合作为起步值；更大更保守，可能压慢角色/风格学习。开启后每步会多跑一次底模参考 forward，训练会变慢。',
        }));
        controls.appendChild(createNoDatasetRegularizationNumberControl({
            key: 'inverted_mask_prior_weight',
            modeClass: 'no-dataset-control-mask-weight',
            label: '遮罩外保护权重',
            hint: '默认 0.1；只在反转遮罩保护模式生效。',
        }));
        controls.appendChild(createNoDatasetRegularizationTextControl({
            key: 'diff_output_preservation_trigger',
            modeClass: 'no-dataset-control-dop',
            label: 'DOP 触发词',
            placeholder: '例如 sks / 角色名，可留空',
            hint: '可选。填 caption 中代表训练目标的词，例如 sks、角色名、产品名；DOP 会把它替换成下面的类提示。',
        }));
        controls.appendChild(createNoDatasetRegularizationTextControl({
            key: 'diff_output_preservation_class',
            modeClass: 'no-dataset-control-dop no-dataset-control-dop-class',
            label: 'DOP 类提示',
            placeholder: '角色: woman / character；风格: anime style',
            hint: '必填。填比触发词更泛化的类别：人物/角色用 woman、man、character；物体用 object、outfit、weapon；风格用 anime style、illustration style。',
        }));
        panel.appendChild(controls);

        const status = document.createElement('p');
        status.className = 'no-dataset-regularization-status';
        status.dataset.noDatasetRegularizationStatus = '1';
        panel.appendChild(status);

        requestAnimationFrame(updateNoDatasetRegularizationModePanel);
        return panel;
    }

    function createNoDatasetRegularizationAdvancedFields(fields, groupClass) {
        const details = document.createElement('details');
        details.className = 'no-dataset-regularization-advanced';
        const summary = document.createElement('summary');
        summary.textContent = NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY;
        details.appendChild(summary);
        const body = document.createElement('div');
        body.className = 'no-dataset-regularization-advanced-body';
        appendFieldRows(body, fields, groupClass);
        details.appendChild(body);
        details.addEventListener('toggle', () => {
            summary.textContent = details.open
                ? NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY_OPEN
                : NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY;
        });
        return details;
    }

    function createNoDatasetRegularizationNumberControl(options) {
        const label = document.createElement('label');
        label.className = ['no-dataset-regularization-control', options.modeClass || ''].filter(Boolean).join(' ');
        const title = document.createElement('span');
        title.textContent = options.label;
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '0';
        input.step = '0.01';
        input.dataset.noDatasetRegularizationMirror = options.key;
        input.addEventListener('input', () => updateNoDatasetRegularizationFieldFromMirror(input));
        input.addEventListener('change', () => updateNoDatasetRegularizationFieldFromMirror(input));
        const hint = document.createElement('em');
        hint.textContent = options.hint || '';
        label.append(title, input, hint);
        return label;
    }

    function createNoDatasetRegularizationTextControl(options) {
        const label = document.createElement('label');
        label.className = ['no-dataset-regularization-control', options.modeClass || ''].filter(Boolean).join(' ');
        const title = document.createElement('span');
        title.textContent = options.label;
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = options.placeholder || '';
        input.dataset.noDatasetRegularizationMirror = options.key;
        input.addEventListener('input', () => updateNoDatasetRegularizationFieldFromMirror(input));
        input.addEventListener('change', () => updateNoDatasetRegularizationFieldFromMirror(input));
        label.append(title, input);
        if (options.hint) {
            const hint = document.createElement('em');
            hint.textContent = options.hint;
            label.appendChild(hint);
        }
        return label;
    }

    function updateNoDatasetRegularizationFieldFromMirror(input) {
        const key = input?.dataset?.noDatasetRegularizationMirror;
        if (!key) return;
        const value = input.type === 'number' ? noDatasetRegularizationPositiveNumber(input.value) : input.value;
        setFieldInputValue(key, value);
        handleFormFieldChange();
    }

    function applyNoDatasetRegularizationMode(mode) {
        const current = readNoDatasetRegularizationValues();
        const priorWeight = noDatasetRegularizationPreferredWeight(current.prior_preservation_weight);
        const maskWeight = noDatasetRegularizationPreferredWeight(current.inverted_mask_prior_weight);
        const dopClass = String(current.diff_output_preservation_class || '');
        const dopTrigger = String(current.diff_output_preservation_trigger || '');
        const patch = noDatasetRegularizationPatchForMode(mode, { priorWeight, maskWeight, dopClass, dopTrigger });
        for (const [key, value] of Object.entries(patch)) {
            setFieldInputValue(key, value);
        }
        handleFormFieldChange();
        const spec = NO_DATASET_REGULARIZATION_MODE_SPECS.find((item) => item.id === mode);
        setTomlStatus('ok', `已切换无数据集正则化方案: ${spec?.label || mode}`);
    }

    function noDatasetRegularizationPatchForMode(mode, context = {}) {
        const priorWeight = context.priorWeight ?? NO_DATASET_REGULARIZATION_DEFAULT_WEIGHT;
        const maskWeight = context.maskWeight ?? NO_DATASET_REGULARIZATION_DEFAULT_WEIGHT;
        const dopClass = context.dopClass ?? '';
        const dopTrigger = context.dopTrigger ?? '';
        if (mode === 'blank') {
            return {
                prior_preservation_weight: priorWeight,
                blank_prompt_preservation: true,
                diff_output_preservation_trigger: '',
                diff_output_preservation_class: '',
                inverted_mask_prior_weight: 0.0,
                ...NO_DATASET_REGULARIZATION_CACHE_PATCH,
            };
        }
        if (mode === 'dop') {
            return {
                prior_preservation_weight: priorWeight,
                blank_prompt_preservation: false,
                diff_output_preservation_trigger: dopTrigger,
                diff_output_preservation_class: dopClass,
                inverted_mask_prior_weight: 0.0,
                ...NO_DATASET_REGULARIZATION_CACHE_PATCH,
            };
        }
        if (mode === 'mask') {
            return {
                prior_preservation_weight: 0.0,
                blank_prompt_preservation: false,
                diff_output_preservation_trigger: '',
                diff_output_preservation_class: '',
                inverted_mask_prior_weight: maskWeight,
                ...NO_DATASET_REGULARIZATION_CACHE_PATCH,
            };
        }
        return {
            prior_preservation_weight: 0.0,
            blank_prompt_preservation: false,
            diff_output_preservation_trigger: '',
            diff_output_preservation_class: '',
            inverted_mask_prior_weight: 0.0,
        };
    }

    function updateNoDatasetRegularizationModePanel() {
        const panel = document.querySelector('#config-form .no-dataset-regularization-panel');
        if (!panel) return;
        const values = readNoDatasetRegularizationValues();
        const modeState = inferNoDatasetRegularizationMode(values);
        panel.dataset.mode = modeState.mode;
        panel.classList.toggle('has-conflict', modeState.conflict);

        panel.querySelectorAll('[data-no-dataset-regularization-mode]').forEach((btn) => {
            const active = btn.dataset.noDatasetRegularizationMode === modeState.mode && !modeState.conflict;
            btn.classList.toggle('active', active);
            btn.setAttribute('aria-checked', String(active));
        });

        setNoDatasetRegularizationMirrorValue(panel, 'prior_preservation_weight', values.prior_preservation_weight);
        setNoDatasetRegularizationMirrorValue(panel, 'inverted_mask_prior_weight', values.inverted_mask_prior_weight);
        setNoDatasetRegularizationMirrorValue(panel, 'diff_output_preservation_trigger', values.diff_output_preservation_trigger);
        setNoDatasetRegularizationMirrorValue(panel, 'diff_output_preservation_class', values.diff_output_preservation_class);

        const status = panel.querySelector('[data-no-dataset-regularization-status]');
        if (status) {
            status.textContent = noDatasetRegularizationStatusMessage(modeState, values);
            status.classList.toggle('is-warning', modeState.conflict || (modeState.mode === 'dop' && !String(values.diff_output_preservation_class || '').trim()));
        }

        const advanced = document.querySelector('#config-form .no-dataset-regularization-advanced');
        if (advanced && modeState.conflict) {
            advanced.open = true;
            const summary = advanced.querySelector('summary');
            if (summary) summary.textContent = NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY_OPEN;
        }
    }

    function readNoDatasetRegularizationValues() {
        return {
            prior_preservation_weight: noDatasetRegularizationFieldValue('prior_preservation_weight'),
            blank_prompt_preservation: noDatasetRegularizationFieldValue('blank_prompt_preservation'),
            diff_output_preservation_trigger: noDatasetRegularizationFieldValue('diff_output_preservation_trigger'),
            diff_output_preservation_class: noDatasetRegularizationFieldValue('diff_output_preservation_class'),
            inverted_mask_prior_weight: noDatasetRegularizationFieldValue('inverted_mask_prior_weight'),
        };
    }

    function noDatasetRegularizationFieldValue(key) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (input) return readFieldInputValue(input, originalConfigFieldValue(key));
        if (configFormState.draftValues.has(key)) return configFormState.draftValues.get(key);
        return originalConfigFieldValue(key);
    }

    function inferNoDatasetRegularizationMode(values = readNoDatasetRegularizationValues()) {
        const priorEnabled = noDatasetRegularizationNumber(values.prior_preservation_weight) > 0;
        const maskEnabled = noDatasetRegularizationNumber(values.inverted_mask_prior_weight) > 0;
        const blankEnabled = Boolean(values.blank_prompt_preservation === true || values.blank_prompt_preservation === 'true');
        const dopEnabled = Boolean(String(values.diff_output_preservation_class || '').trim());
        const active = [
            blankEnabled && priorEnabled ? 'blank' : '',
            dopEnabled && priorEnabled ? 'dop' : '',
            maskEnabled ? 'mask' : '',
        ].filter(Boolean);
        const orphanPrior = priorEnabled && !blankEnabled && !dopEnabled;
        if (active.length > 1 || orphanPrior || (blankEnabled && dopEnabled)) {
            return { mode: NO_DATASET_REGULARIZATION_CONFLICT_MODE, conflict: true };
        }
        if (active[0]) return { mode: active[0], conflict: false };
        return { mode: 'off', conflict: false };
    }

    function noDatasetRegularizationStatusMessage(modeState, values) {
        if (modeState.conflict) {
            const priorEnabled = noDatasetRegularizationNumber(values.prior_preservation_weight) > 0;
            const blankEnabled = Boolean(values.blank_prompt_preservation === true || values.blank_prompt_preservation === 'true');
            const dopClass = String(values.diff_output_preservation_class || '').trim();
            if (priorEnabled && !blankEnabled && !dopClass) {
                return String(values.diff_output_preservation_trigger || '').trim()
                    ? NO_DATASET_REGULARIZATION_DOP_CLASS_REQUIRED
                    : '先验保留权重大于 0 时，请选择空提示先验，或填写 DOP 类提示。类提示应是泛化类别，例如 woman / character / anime style。';
            }
            return NO_DATASET_REGULARIZATION_CONFLICT_MESSAGE;
        }
        if (modeState.mode === 'blank') return '将使用空提示先验；训练前请确认文本缓存和 LLM 适配器输出缓存已生成。';
        if (modeState.mode === 'dop') {
            return String(values.diff_output_preservation_class || '').trim()
                ? '将使用 DOP/class prompt；修改类提示或触发词后需要重新生成文本缓存。'
                : NO_DATASET_REGULARIZATION_DOP_CLASS_REQUIRED;
        }
        if (modeState.mode === 'mask') return '将只保护遮罩外区域；需要 alpha mask、文本缓存和 LLM 适配器输出缓存。';
        return '当前关闭；不会写入额外无数据集正则化损失。';
    }

    function setNoDatasetRegularizationMirrorValue(panel, key, value) {
        const input = panel.querySelector(`[data-no-dataset-regularization-mirror="${CSS.escape(key)}"]`);
        if (!input) return;
        const next = input.type === 'number'
            ? String(noDatasetRegularizationNumber(value) || '')
            : String(value ?? '');
        if (input.value !== next) input.value = next;
    }

    function noDatasetRegularizationPreferredWeight(value) {
        const n = noDatasetRegularizationNumber(value);
        return n > 0 ? n : NO_DATASET_REGULARIZATION_DEFAULT_WEIGHT;
    }

    function noDatasetRegularizationPositiveNumber(value) {
        const n = noDatasetRegularizationNumber(value);
        return n > 0 ? n : 0.0;
    }

    function noDatasetRegularizationNumber(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n : 0;
    }

export {
    createNoDatasetRegularizationAdvancedFields,
    createNoDatasetRegularizationModePanel,
    updateNoDatasetRegularizationModePanel,
};

configureNoDatasetRegularizationModePanelUpdater(updateNoDatasetRegularizationModePanel);
