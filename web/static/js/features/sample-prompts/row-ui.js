/**
 * Sample-prompt row builders and config field UI helpers.
 * Moved out of anima-app mechanical chunks.
 */
import { formatCompactNumber } from '../history-detail/ui.js?v=module-bootstrap-20260831-release-v1';
import {
    EXTRA_FIELD_HELP_ZH,
    FIELD_HELP_ZH,
    FIELD_LABEL_ZH,
    FIELD_STRICT_SELECT_OPTIONS,
    FORM_SECTION_DEFS,
    NETWORK_ARG_FIELD_MAP,
    help,
} from '../../config/catalog.js?v=module-bootstrap-20260831-release-v1';
import { normalizeLoraAdapterKind, normalizePrecisionPreference } from '../anima-app/helpers/config-values.js?v=module-bootstrap-20260831-release-v1';
import { configureConfigFieldUiBridge } from '../anima-app/helpers/config-field-ui-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { handleFormFieldChange, markSamplePromptsEditorTouched } from '../config-form/form-fields.js?v=module-bootstrap-20260831-release-v1';

const configState = getConfigState();

    export function appendSamplePromptRow(rowsWrap, row) {
        const item = document.createElement('div');
        item.className = 'sample-prompt-row';

        const promptField = createSamplePromptTextField('提示词', 'prompt', row.prompt || '');
        const negativePromptField = createSamplePromptTextField('负面提示词 / n', 'negative_prompt', row.negative_prompt || '');
        const heightField = createSamplePromptInputField('长 / h', 'height', row.height || '', 'number', '1');
        const widthField = createSamplePromptInputField('宽 / w', 'width', row.width || '', 'number', '1');
        const cfgField = createSamplePromptInputField('CFG / g/l', 'cfg', row.cfg || '', 'number', '0.1');
        const stepsField = createSamplePromptInputField('步数 / s', 'steps', row.steps || '', 'number', '1');
        const seedField = createSamplePromptInputField('种子 / d', 'seed', row.seed || '', 'number', '1');
        const flowShiftField = createSamplePromptInputField('Flow Shift / fs', 'flow_shift', row.flow_shift || '', 'number', '0.1');
        const samplerField = createSamplePromptSelectField('采样器 / ss', 'sample_sampler', row.sample_sampler || '', ['euler', 'er_sde', 'lcm']);
        const extra = document.createElement('input');
        extra.type = 'hidden';
        extra.dataset.samplePromptField = 'extra';
        extra.value = row.extra || '';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn btn-small btn-subtle-danger sample-prompt-remove';
        removeBtn.textContent = '删除';
        removeBtn.addEventListener('click', () => {
            const editor = rowsWrap.closest('.sample-prompts-editor');
            const rowCount = rowsWrap.querySelectorAll('.sample-prompt-row').length;
            if (rowCount <= 1) {
                clearSamplePromptRow(item);
            } else {
                item.remove();
            }
            markSamplePromptsEditorTouched(editor);
            updateSamplePromptRemoveButtons(rowsWrap);
            handleFormFieldChange();
        });

        const rowActions = document.createElement('div');
        rowActions.className = 'sample-prompt-row-actions';
        rowActions.append(removeBtn);

        item.append(
            promptField,
            negativePromptField,
            heightField,
            widthField,
            cfgField,
            stepsField,
            seedField,
            flowShiftField,
            samplerField,
            extra,
            rowActions,
        );
        rowsWrap.appendChild(item);
        updateSamplePromptRemoveButtons(rowsWrap);
    }

    export function createSamplePromptTextField(labelText, field, value) {
        const label = document.createElement('label');
        label.className = 'sample-prompt-field sample-prompt-field-text';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = 'text';
        input.dataset.samplePromptField = field;
        input.value = value || '';
        label.append(span, input);
        return label;
    }

    export function createSamplePromptInputField(labelText, field, value, type = 'text', step = '') {
        const label = document.createElement('label');
        label.className = 'sample-prompt-field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = type;
        input.dataset.samplePromptField = field;
        input.value = value || '';
        if (type === 'number') {
            input.min = '0';
            input.step = step || '1';
        }
        label.append(span, input);
        return label;
    }

    export function createSamplePromptSelectField(labelText, field, value, options) {
        const label = document.createElement('label');
        label.className = 'sample-prompt-field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const select = document.createElement('select');
        select.dataset.samplePromptField = field;
        select.value = value || '';
        const values = options.map((option) => String(option));
        if (value && !values.includes(String(value))) {
            options = [value, ...options];
        }
        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = '默认';
        select.appendChild(empty);
        for (const option of options) {
            const opt = document.createElement('option');
            opt.value = String(option);
            opt.textContent = String(option);
            select.appendChild(opt);
        }
        select.value = value || '';
        label.append(span, select);
        return label;
    }

    export function clearSamplePromptRow(row) {
        row.querySelectorAll('[data-sample-prompt-field]').forEach((input) => {
            input.value = '';
        });
    }

    export function updateSamplePromptRemoveButtons(rowsWrap) {
        const rows = rowsWrap.querySelectorAll('.sample-prompt-row');
        rows.forEach((row) => {
            const button = row.querySelector('.sample-prompt-remove');
            if (!button) return;
            button.textContent = rows.length <= 1 ? '清空' : '删除';
        });
    }

    export function isNumericField(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) {
            return ['integer', 'number'].includes(networkArgSpec.valueType);
        }
        return typeof value === 'number' || [
            'max_train_epochs',
            'max_train_steps',
            'train_batch_size',
            'gradient_accumulation_steps',
            'sample_ratio',
            'sample_every_n_epochs',
            'sample_every_n_steps',
            'blocks_to_swap',
            'save_every_n_epochs',
            'save_last_n_epochs',
            'checkpointing_epochs',
            'network_dim',
            'network_alpha',
        ].includes(key);
    }

    export function isIntegerNumericField(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) return networkArgSpec.valueType === 'integer';
        if (key === 'network_alpha') return false;
        return [
            'max_train_epochs',
            'max_train_steps',
            'train_batch_size',
            'gradient_accumulation_steps',
            'sample_every_n_epochs',
            'sample_every_n_steps',
            'blocks_to_swap',
            'save_every_n_epochs',
            'save_last_n_epochs',
            'checkpointing_epochs',
            'network_dim',
        ].includes(key) || Number.isInteger(value);
    }

    export function allowsNegativeNumberField(key) {
        return ['b_cond_init', 'pe_lora_layer_from', 'save_last_n_epochs'].includes(key);
    }

    export function createSelectInput(key, value, options) {
        const select = document.createElement('select');
        select.className = 'field-input field-select';
        select.dataset.valueType = fieldValueTypeForKey(key, value);
        const strictOptions = selectUsesStrictOptions(key);
        if (strictOptions) select.dataset.strictOptions = '1';
        const normalizedValue = optionValue(value);
        const normalizedOptions = options.map(optionValue);
        const displayOptions = [...options];
        const hasCustomCurrentValue = !normalizedOptions.includes(normalizedValue);
        if (hasCustomCurrentValue) {
            displayOptions.unshift(value);
        }

        for (const option of displayOptions) {
            const opt = document.createElement('option');
            opt.value = optionValue(option);
            opt.textContent = optionLabel(key, option);
            if (opt.value === normalizedValue) opt.selected = true;
            if (strictOptions && hasCustomCurrentValue && opt.value === normalizedValue) {
                opt.disabled = true;
                opt.textContent = strictSelectCurrentValueLabel(key, option);
                opt.title = '旧配置里的自定义值；重新选择后会写回固定选项。';
            }
            select.appendChild(opt);
        }
        return select;
    }

    export function selectUsesStrictOptions(key) {
        return Boolean(FIELD_STRICT_SELECT_OPTIONS?.has?.(key));
    }

    export function strictSelectCurrentValueLabel(key, value) {
        if (key === 'block_swap_profile_jsonl') {
            return `自定义路径（旧值） / ${String(value ?? '')}`;
        }
        return `当前值 / ${String(value ?? '')}`;
    }

    Object.assign(FIELD_HELP_ZH, EXTRA_FIELD_HELP_ZH);

    export function fieldValueType(value) {
        if (Array.isArray(value)) return 'array';
        if (typeof value === 'boolean') return 'boolean';
        if (typeof value === 'number') return 'number';
        return 'string';
    }

    export function fieldValueTypeForKey(key, value) {
        const networkArgSpec = NETWORK_ARG_FIELD_MAP.get(key);
        if (networkArgSpec) {
            if (networkArgSpec.valueType === 'boolean' || networkArgSpec.valueType === 'booleanInt') return 'boolean';
            if (networkArgSpec.valueType === 'integer' || networkArgSpec.valueType === 'number') return 'number';
            return 'string';
        }
        if (key === 'lora_adapter_kind') return 'string';
        if (key === 'dora_wd' || key === 'use_lokr' || key === 'use_loha' || key === 'use_glora' || key === 'use_vera') return 'boolean';
        if (key === 'lokr_factor' || key === 'vera_projection_prng_key' || key === 'vera_d_initial') return 'number';
        if (key === 'vera_save_projection') return 'boolean';
        if (isNumericField(key, value)) return 'number';
        return fieldValueType(value);
    }

    export function optionValue(value) {
        if (value === null || value === undefined) return '';
        if (typeof value === 'boolean') return value ? 'true' : 'false';
        return String(value);
    }

    export function optionLabel(key, value) {
        if (key === 'lora_adapter_kind') {
            return {
                lora: '普通 LoRA',
                loha: 'LoHa',
                lokr: 'LoKr',
                glora: 'GLoRA',
                vera: 'VeRA',
            }[normalizeLoraAdapterKind(value)] || String(value);
        }
        if (key === 'use_lokr') {
            return value === true || value === 'true' ? '启用 LoKr' : '普通 LoRA';
        }
        if (key === 'use_glora') {
            return value === true || value === 'true' ? '启用 GLoRA' : '普通 LoRA';
        }
        if (key === 'use_loha') {
            return value === true || value === 'true' ? '启用 LoHa' : '普通 LoRA';
        }
        if (key === 'use_vera') {
            return value === true || value === 'true' ? '启用 VeRA' : '普通 LoRA';
        }
        if (key === 'use_moe_style' && (value === false || value === 'false')) {
            return '关闭专家路由 / false';
        }
        if (key === 'precision_preference') {
            return {
                bf16: '默认推荐 / bf16',
                fp16: '混合精度 / fp16/32',
                fp32: '全程 fp32 / full fp32',
            }[normalizePrecisionPreference(value)] || String(value);
        }
        if (key === 'splice_position') {
            return value === 'front_of_padding' ? 'Padding 前沿 / front_of_padding' : '序列末尾 / end_of_sequence';
        }
        if (key === 'lokr_project_chunk_bytes') {
            const bytes = Number(value);
            if (Number.isFinite(bytes) && bytes > 0) {
                return `${formatCompactNumber(bytes / (1024 * 1024))}MiB / ${Math.trunc(bytes)}`;
            }
            return String(value);
        }
        if (key === 'lokr_decompose_w2') {
            return value === true || value === 'true' ? '轻量分解 / true' : '完整 W2 / false';
        }
        if (key === 'lokr_use_einsum') {
            return value === false || value === 'false' ? '兼容旧路径 / false' : '结构化路径 / true';
        }
        if (key === 'peak_probe_level') {
            return {
                block: 'Block 边界 / block',
                ops: 'Block 内算子 / ops',
                lokr: 'LoKr delta / lokr',
                full: '全量事件 / full',
            }[value] || String(value);
        }
        if (key === 'block_swap_profile_jsonl') {
            return {
                off: '关闭 / off',
                auto: '自动写入任务目录 / auto',
            }[value] || String(value);
        }
        if (key === 'contrastive_negative_mode') {
            return {
                shuffled: '随机负样本 / shuffled',
                jaccard: 'Jaccard 降权 / jaccard',
                hard: '困难负样本 / hard',
            }[value] || String(value);
        }
        if (key === 'contrastive_objective') {
            return {
                infonce: 'InfoNCE / infonce',
                softrank: 'SoftRank / softrank',
            }[value] || String(value);
        }
        if (value === true) return '开启 / true';
        if (value === false) return '关闭 / false';
        return String(value);
    }

    export function generateDefaultHelp(key, value) {
        const typeStr = Array.isArray(value) ? '数组' :
            typeof value === 'boolean' ? '布尔值 (true/false)' :
            typeof value === 'number' ? '数值' : '字符串';
        const label = FIELD_LABEL_ZH[key] || key;
        const section = sectionTitleForField(key);
        const currentText = value === undefined ? '未设置' : JSON.stringify(value);
        return help(
            `${label} 是当前配置里的${section}字段，WebUI 暂时没有为它写专门教程。`,
            `按 ${typeStr} 填写。当前值: ${currentText}。如果你只是想正常训练，不需要为了“看懂它”而主动修改。`,
            ['保留这个字段可以完整复现当前 TOML 的训练行为。'],
            ['它通常属于低频或方法内部参数，改动后效果不一定能从字段名直观看出来。'],
            ['不了解来源时修改，可能导致训练启动失败、缓存失效，或让训练结果和预期不一致。'],
            '新手建议保持当前值；要改之前先看右侧 TOML 所属变体，或复制一份新配置做实验。'
        );
    }

    export function sectionTitleForField(key) {
        for (const section of FORM_SECTION_DEFS) {
            if ((section.keys || []).includes(key)) return section.title;
        }
        if (String(key).includes('cache')) return '缓存/预处理';
        if (String(key).includes('sample')) return '训练中预览图';
        if (String(key).includes('router') || String(key).includes('repa') || String(key).includes('reft')) return '方法内部';
        return '高级配置';
    }

    export function createHelpContent(key, value) {
        const spec = getHelpSpec(key, value);
        const content = document.createElement('div');
        content.className = 'help-content';
        addHelpSection(content, '作用', spec.summary, 'summary');
        addHelpSection(content, '怎么填', spec.fill, 'fill');
        addHelpSection(content, '好处', spec.benefit, 'benefit');
        addHelpSection(content, '代价', spec.cost, 'cost');
        addHelpSection(content, '风险', spec.risk, 'risk');
        addHelpSection(content, '推荐', spec.recommend, 'recommend');
        addHelpSection(content, 'PS', spec.ps, 'ps');
        return content;
    }

    export function addHelpSection(parent, title, body, kind) {
        if (body === undefined || body === null || body === '') return;
        if (Array.isArray(body) && body.length === 0) return;

        const section = document.createElement('section');
        section.className = `help-section help-${kind}`;

        const heading = document.createElement('div');
        heading.className = 'help-heading';
        heading.textContent = title;
        section.appendChild(heading);

        if (Array.isArray(body)) {
            const list = document.createElement('ul');
            for (const item of body) {
                if (!item) continue;
                const li = document.createElement('li');
                li.textContent = item;
                list.appendChild(li);
            }
            section.appendChild(list);
        } else {
            const text = document.createElement('p');
            text.textContent = body;
            section.appendChild(text);
        }
        parent.appendChild(section);
    }

    export function getHelpSpec(key, value) {
        // 优先使用内置中文说明
        if (FIELD_HELP_ZH[key]) return FIELD_HELP_ZH[key];
        // 其次从服务端获取的 field help 中取英文（作为兜底）
        const remote = configState.fieldHelp[key];
        if (remote) {
            const remoteText = remote.en || remote.ko || '';
            if (remoteText) {
                const label = FIELD_LABEL_ZH[key] || key;
                return help(
                    `${label} 来自项目配置 schema 或方法配置，属于当前训练链路的一部分。`,
                    `${remoteText} 新手只需要确认当前值来自可信变体；不要为了试错随手改。`,
                    ['能保留上游配置说明，帮助你追踪字段来源。'],
                    ['英文说明通常偏开发者视角，仍需要结合当前方法和 TOML 判断。'],
                    ['如果字段和当前方法不匹配，可能训练启动后才暴露错误。'],
                    '不确定时保持当前变体默认值；需要实验时先另存为新配置。'
                );
            }
        }
        return generateDefaultHelp(key, value);
    }

    // ── TOML 编辑器 ──

configureConfigFieldUiBridge({
    appendSamplePromptRow,
    createSamplePromptTextField,
    createSamplePromptInputField,
    clearSamplePromptRow,
    updateSamplePromptRemoveButtons,
    isNumericField,
    isIntegerNumericField,
    allowsNegativeNumberField,
    createSelectInput,
    selectUsesStrictOptions,
    strictSelectCurrentValueLabel,
    fieldValueType,
    fieldValueTypeForKey,
    optionValue,
    optionLabel,
    generateDefaultHelp,
    sectionTitleForField,
    createHelpContent,
    addHelpSection,
    getHelpSpec,
});
