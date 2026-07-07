/**
 * Dataset editor inline help helpers and small advanced-field panels.
 */
import { help } from '../../../config/catalog.js?v=module-bootstrap-20260707-93';
import { createHelpContent } from '../helpers/config-field-ui-bridge.js?v=module-bootstrap-20260707-93';
import { getDatasetState } from '../helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import { isDatasetTabActive } from '../helpers/dataset-render-bridge.js?v=module-bootstrap-20260707-93';
import { currentTrainingConfigFile } from '../helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260707-93';
import {
    updateDatasetEditorRow,
    updateDatasetEditorRowSettingValue,
} from './12-create-dataset-row-caption-source-mode-editor.js?v=module-bootstrap-20260707-93';

const datasetState = getDatasetState();

function currentDatasetPresetState() {
    return datasetState.datasetPresetState || {};
}

function currentDatasetEditorState() {
    return datasetState.datasetEditorState || {};
}

let datasetInlineHelpSeq = 0;

function createDatasetInlineHelp(className = '') {
    const helpDiv = document.createElement('div');
    helpDiv.id = `dataset-inline-help-${++datasetInlineHelpSeq}`;
    const classNames = new Set(['field-help', 'dataset-inline-help']);
    String(className || '').split(/\s+/).filter(Boolean).forEach((name) => classNames.add(name));
    helpDiv.className = [...classNames].join(' ');
    return helpDiv;
}

function createDatasetInlineHelpButton(helpDiv, label = '查看说明') {
    const btn = document.createElement('button');
    btn.className = 'info-toggle dataset-inline-help-toggle dataset-inline-help-control';
    btn.textContent = '?';
    btn.type = 'button';
    btn.title = label;
    btn.setAttribute('aria-label', label);
    btn.setAttribute('aria-controls', helpDiv.id);
    btn.setAttribute('aria-expanded', 'false');
    return btn;
}

function createDatasetInlineHelpAction(helpDiv, label = '查看详细说明') {
    const btn = document.createElement('button');
    btn.className = 'btn btn-small dataset-inline-help-action dataset-inline-help-control';
    btn.textContent = label;
    btn.type = 'button';
    btn.title = label;
    btn.setAttribute('aria-label', label);
    btn.setAttribute('aria-controls', helpDiv.id);
    btn.setAttribute('aria-expanded', 'false');
    return btn;
}

function createDatasetLocalHelpContent(spec) {
    const content = document.createElement('div');
    content.className = 'help-content';
    addDatasetHelpSection(content, '作用', spec.summary, 'summary');
    addDatasetHelpSection(content, '怎么填', spec.fill, 'fill');
    addDatasetHelpSection(content, '好处', spec.benefit, 'benefit');
    addDatasetHelpSection(content, '代价', spec.cost, 'cost');
    addDatasetHelpSection(content, '风险', spec.risk, 'risk');
    addDatasetHelpSection(content, '推荐', spec.recommend, 'recommend');
    return content;
}

function addDatasetHelpSection(parent, title, body, kind) {
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
        body.forEach((item) => {
            if (!item) return;
            const li = document.createElement('li');
            li.textContent = item;
            list.appendChild(li);
        });
        section.appendChild(list);
    } else {
        const text = document.createElement('p');
        text.textContent = body;
        section.appendChild(text);
    }
    parent.appendChild(section);
}

function datasetLocalHelpSpec(kind) {
    const specs = {
        experimental: {
            summary: '收纳按单组数据集保存的高级兼容项、旧格式入口和需要先小范围验证的功能。',
            fill: '只在确实需要对应功能时展开并修改；不确定时保持默认状态。',
            benefit: ['让低频选项不干扰主路径、重复次数和分桶等常用配置。'],
            cost: ['启用后可能影响数据枚举、caption 处理或运行时生成的数据集副本。'],
            risk: ['多数据集场景下要确认生效范围，避免把实验选项同步到不该改的组。'],
            recommend: '先在一组数据集上验证，再用“生效范围”同步到其他组。'
        },
        recursive: {
            summary: '控制是否扫描原始数据集目录下的子目录。',
            fill: '默认开启。关闭后只读取原始路径第一层图片，不进入更深层文件夹。',
            benefit: ['目录按角色、姿态或来源分层时，开启后无需逐个添加路径。'],
            cost: ['子目录里混入无关图片时也会被纳入训练。'],
            risk: ['关闭后如果图片都在子目录中，预处理可能显示没有可训练图片。'],
            recommend: '大多数数据集保持开启；只想严格使用第一层图片时再关闭。'
        },
        triggerClone: {
            summary: '训练启动前在本次运行目录生成额外训练子集，用触发提示词强化指定概念。',
            fill: '先开启，再填写触发提示词和循环次数；原始数据集不会被修改。',
            benefit: ['可以在不改原图和原 caption 的情况下增加触发词相关样本权重。'],
            cost: ['会增加本次运行目录的数据量、预处理时间和训练步数权重。'],
            risk: ['循环次数过高会让触发词过强，可能更容易过拟合。'],
            recommend: '先用 1-2 次小范围验证；确认触发词有效后再提高。'
        },
        nlTagMix: {
            summary: '面向 DiffPipeForge captions.json 的多标注数据集，按 tag/nl 比例重建运行时 captions.json。',
            fill: '只在标注来源使用 captions.json 且确实存在短标签串和自然语言句子混合时开启。',
            benefit: ['能控制短标签和自然语言描述在训练中的抽样比例。'],
            cost: ['会在运行目录生成重建后的 captions.json 和 results.json，排查时需要看运行快照。'],
            risk: ['标注分类不符合预期时，比例调整可能改变训练语义。'],
            recommend: '普通 txt/json sidecar 数据集保持关闭。'
        },
        scope: {
            summary: '决定当前高级功能面板会同步写入哪些数据集组。',
            fill: '单组数据集保持当前组即可；多组需要共用同一个高级设置时再选择更多组。',
            benefit: ['能减少多组数据集重复填写相同高级项。'],
            cost: ['同步范围越大，误改影响面越大。'],
            risk: ['保存时仍会按每组独立落盘，选择错误会让某些组带上不需要的实验设置。'],
            recommend: '先只改当前组；确认无误后再全选。'
        },
    };
    return specs[kind] || specs.experimental;
}

function createDatasetHelpNode(factory) {
    if (typeof factory === 'function') return factory();
    return createDatasetLocalHelpContent(factory);
}

function attachDatasetInlineHelp(btn, helpDiv, factory, scope, options = {}) {
    btn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        const wasActive = btn.classList.contains('active');
        scope.querySelectorAll('.dataset-inline-help-control.active').forEach((activeBtn) => {
            activeBtn.classList.remove('active');
            activeBtn.setAttribute('aria-expanded', 'false');
        });
        scope.querySelectorAll('.dataset-inline-help.visible').forEach((activeHelp) => {
            activeHelp.classList.remove('visible');
            activeHelp.innerHTML = '';
        });
        if (wasActive) return;
        if (options.openDetails && scope?.tagName === 'DETAILS') {
            scope.open = true;
        }
        btn.classList.add('active');
        btn.setAttribute('aria-expanded', 'true');
        helpDiv.innerHTML = '';
        helpDiv.appendChild(createDatasetHelpNode(factory));
        helpDiv.classList.add('visible');
    });
}

function createDatasetAdvancedSection(title, description, children, modifierClass = '') {
    const section = document.createElement('section');
    section.className = ['dataset-advanced-section', modifierClass].filter(Boolean).join(' ');
    const head = document.createElement('div');
    head.className = 'dataset-advanced-section-head';
    const heading = document.createElement('strong');
    heading.textContent = title;
    const copy = document.createElement('span');
    copy.textContent = description;
    head.append(heading, copy);
    const grid = document.createElement('div');
    grid.className = 'dataset-advanced-grid';
    children.filter(Boolean).forEach((child) => grid.appendChild(child));
    section.append(head, grid);
    return section;
}

function createDatasetExperimentalNotice(helpDiv) {
    const notice = document.createElement('div');
    notice.className = 'dataset-experimental-notice';
    const text = document.createElement('span');
    text.className = 'dataset-experimental-notice-text';
    text.textContent = '包含高级兼容项及非常规功能，建议先在小范围验证。';
    const detailBtn = createDatasetInlineHelpAction(helpDiv, '查看详细说明');
    detailBtn.classList.add('dataset-experimental-notice-button');
    notice.append(text, detailBtn);
    return { notice, detailBtn };
}

function createDatasetExperimentalAdvancedBody(row, index, overviewHelp, deps) {
    const {
        createDatasetCaptionExtensionEditor,
        createDatasetExperimentalScopePicker,
        createDatasetPathFilterEditor,
        createDatasetTriggerCloneEditor,
    } = deps;
    const body = document.createElement('div');
    body.className = 'dataset-experimental-body';
    const overview = createDatasetExperimentalNotice(overviewHelp);
    body.append(
        overview.notice,
        overviewHelp,
        createDatasetAdvancedSection(
            '数据与路径规则',
            '决定程序如何枚举本地图片、筛选路径和读取文本标注。',
            [
                createDatasetPathFilterEditor(row, index),
                createDatasetCaptionExtensionEditor(row, index),
            ],
            'dataset-advanced-data-rules',
        ),
        createDatasetAdvancedSection(
            '训练行为与策略',
            '影响数据集权重、生效范围和运行时生成的训练副本。',
            [
                createDatasetExperimentalScopePicker(index),
                createDatasetIsRegEditor(row, index),
                createDatasetTriggerCloneEditor(row, index),
            ],
            'dataset-advanced-training-rules',
        ),
    );
    return { body, detailBtn: overview.detailBtn };
}

function datasetExperimentalOpenKey(index) {
    const context = isDatasetTabActive()
        ? `preset:${currentDatasetPresetState().selectedFile || 'new'}`
        : `config:${currentDatasetEditorState().dataset_config || (typeof currentTrainingConfigFile === 'function' ? currentTrainingConfigFile() : '') || 'current'}`;
    return `${context}:${index}`;
}

function setDatasetExperimentalOpenState(index, open) {
    datasetState.datasetExperimentalOpenStates.set(datasetExperimentalOpenKey(index), Boolean(open));
}

function datasetExperimentalOpenState(index, defaultOpen) {
    const key = datasetExperimentalOpenKey(index);
    return datasetState.datasetExperimentalOpenStates.has(key)
        ? datasetState.datasetExperimentalOpenStates.get(key)
        : Boolean(defaultOpen);
}

function captureDatasetExperimentalOpenStates(root = document) {
    root.querySelectorAll('.dataset-experimental-features[data-index]').forEach((panel) => {
        const index = Number.parseInt(panel.dataset.index || '-1', 10);
        if (Number.isInteger(index) && index >= 0) {
            setDatasetExperimentalOpenState(index, panel.open);
        }
    });
}

function bindDatasetExperimentalOpenState(panel, index) {
    panel.addEventListener('toggle', () => {
        setDatasetExperimentalOpenState(index, panel.open);
    });
}

function createDatasetIsRegEditor(row, index) {
    const panel = document.createElement('div');
    panel.className = 'dataset-is-reg-advanced';
    panel.dataset.index = String(index);

    const copy = document.createElement('div');
    copy.className = 'dataset-is-reg-copy';
    const titleRow = document.createElement('div');
    titleRow.className = 'dataset-inline-title-row';
    const title = document.createElement('strong');
    title.textContent = '正则化训练 / Regularization';
    const helpDiv = createDatasetInlineHelp('dataset-is-reg-help');
    const helpBtn = createDatasetInlineHelpButton(helpDiv, '查看正则化训练说明');
    titleRow.append(title, helpBtn);
    copy.appendChild(titleRow);

    const actions = document.createElement('div');
    actions.className = 'dataset-is-reg-actions';

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
    weightInput.addEventListener('input', () => {
        const nextWeight = Number(weightInput.value);
        updateDatasetEditorRowSettingValue(
            index,
            'prior_loss_weight',
            Number.isFinite(nextWeight) ? Math.max(0, nextWeight) : 1.0,
        );
    });
    weightField.append(weightLabel, weightInput);

    actions.append(toggleLabel, weightField);

    attachDatasetInlineHelp(
        helpBtn,
        helpDiv,
        () => createHelpContent('prior_loss_weight', weightInput.value),
        panel,
    );

    panel.append(copy, actions, helpDiv);
    return panel;
}

export {
    attachDatasetInlineHelp,
    bindDatasetExperimentalOpenState,
    captureDatasetExperimentalOpenStates,
    createDatasetExperimentalAdvancedBody,
    createDatasetInlineHelp,
    createDatasetInlineHelpButton,
    createDatasetIsRegEditor,
    datasetExperimentalOpenState,
    datasetLocalHelpSpec,
};
