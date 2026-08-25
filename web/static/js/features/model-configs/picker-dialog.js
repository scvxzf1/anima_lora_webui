import { fetchModelConfigLibrary } from './api.js?v=module-bootstrap-20260809-nf4-v2';
import { MODEL_CONFIG_PATH_FIELDS, modelFamilyLabel } from './model-config-data.js?v=module-bootstrap-20260824-zimage-v1';

function isConfigComplete(item) {
    return Boolean(item) && MODEL_CONFIG_PATH_FIELDS.every(({ key }) => Boolean(item[key]));
}

function pickerDetail(item, defaultId) {
    const detail = document.createElement('section');
    detail.className = 'model-config-picker-detail';
    if (!item) {
        detail.classList.add('is-empty');
        const title = document.createElement('strong');
        title.textContent = '未选择模型配置';
        const message = document.createElement('span');
        message.textContent = '从左侧列表选择一项后查看完整路径。';
        detail.append(title, message);
        return detail;
    }
    const heading = document.createElement('div');
    heading.className = 'model-config-picker-detail-head';
    const headingText = document.createElement('div');
    headingText.className = 'model-config-picker-detail-title';
    const eyebrow = document.createElement('span');
    eyebrow.textContent = '已选配置';
    const title = document.createElement('strong');
    title.textContent = item.name;
    headingText.append(eyebrow, title);
    const badges = document.createElement('div');
    badges.className = 'model-config-picker-detail-badges';
    const family = document.createElement('span');
    family.className = 'model-config-picker-badge';
    family.textContent = modelFamilyLabel(item.model_family);
    badges.appendChild(family);
    if (item.id === defaultId) {
        const defaultBadge = document.createElement('span');
        defaultBadge.className = 'model-config-picker-badge is-default';
        defaultBadge.textContent = '默认';
        badges.appendChild(defaultBadge);
    }
    heading.append(headingText, badges);
    detail.appendChild(heading);
    for (const field of MODEL_CONFIG_PATH_FIELDS) {
        const row = document.createElement('div');
        row.className = 'model-config-picker-detail-row';
        const label = document.createElement('span');
        label.className = 'model-config-picker-detail-label';
        label.textContent = field.label;
        const path = document.createElement('code');
        path.textContent = item[field.key] || '未配置';
        row.append(label, path);
        detail.appendChild(row);
    }
    return detail;
}

function pickerOption(item, library, pickerState, onSelect) {
    const button = document.createElement('button');
    const complete = isConfigComplete(item);
    button.type = 'button';
    button.className = `model-config-picker-option${item.id === pickerState.selectedId ? ' active' : ''}`;
    button.disabled = !complete;

    const head = document.createElement('span');
    head.className = 'model-config-picker-option-head';
    const title = document.createElement('strong');
    title.textContent = item.name;
    const family = document.createElement('span');
    family.className = 'model-config-picker-option-family';
    family.textContent = modelFamilyLabel(item.model_family);
    head.append(title, family);

    const meta = document.createElement('span');
    meta.className = `model-config-picker-option-status${complete ? '' : ' is-incomplete'}`;
    meta.textContent = complete
        ? (item.id === library.defaultId ? '默认配置' : '可用')
        : '路径不完整';
    const path = document.createElement('small');
    path.textContent = item.pretrained_model_name_or_path;
    button.append(head, meta, path);
    button.addEventListener('click', onSelect);
    return button;
}

function renderPickerBody(dialog, library, pickerState) {
    const body = document.getElementById('global-model-config-picker-body');
    if (!body) return;
    body.innerHTML = '';
    const toolbar = document.createElement('div');
    toolbar.className = 'model-config-picker-toolbar';
    const toolbarSummary = document.createElement('div');
    toolbarSummary.className = 'model-config-picker-toolbar-summary';
    const toolbarTitle = document.createElement('strong');
    toolbarTitle.textContent = '可用配置';
    const toolbarCount = document.createElement('span');
    const search = document.createElement('input');
    search.type = 'search';
    search.className = 'model-config-picker-search';
    search.placeholder = '搜索名称、格式或路径';
    search.value = pickerState.search;
    toolbarSummary.append(toolbarTitle, toolbarCount);
    toolbar.append(toolbarSummary, search);

    const workspace = document.createElement('div');
    workspace.className = 'model-config-picker-workspace';
    const list = document.createElement('div');
    list.className = 'model-config-picker-list';
    const detail = document.createElement('div');
    detail.className = 'model-config-picker-preview';
    const actions = document.createElement('div');
    actions.className = 'model-config-picker-footer';
    const apply = document.createElement('button');
    apply.type = 'button';
    apply.className = 'btn btn-primary';
    apply.textContent = '应用此模型配置';
    const selected = library.items.find((item) => item.id === pickerState.selectedId);
    const selectedComplete = isConfigComplete(selected);
    apply.disabled = !selectedComplete;
    apply.addEventListener('click', () => dialog.close('apply'));
    const selectionStatus = document.createElement('div');
    selectionStatus.className = 'model-config-picker-selection-status';
    const selectionLabel = document.createElement('span');
    selectionLabel.textContent = selectedComplete ? '当前选择' : '请选择完整配置';
    const selectionName = document.createElement('strong');
    selectionName.textContent = selected?.name || '未选择';
    selectionStatus.append(selectionLabel, selectionName);
    actions.append(selectionStatus, apply);

    const query = pickerState.search.trim().toLocaleLowerCase();
    const visible = library.items.filter((item) => (
        !query
        || item.name.toLocaleLowerCase().includes(query)
        || modelFamilyLabel(item.model_family).toLocaleLowerCase().includes(query)
        || MODEL_CONFIG_PATH_FIELDS.some(({ key }) => item[key].toLocaleLowerCase().includes(query))
    ));
    toolbarCount.textContent = query
        ? `${visible.length} / ${library.items.length} 项`
        : `${visible.length} 项`;
    for (const item of visible) {
        list.appendChild(pickerOption(item, library, pickerState, () => {
            pickerState.selectedId = item.id;
            renderPickerBody(dialog, library, pickerState);
        }));
    }
    if (!visible.length) {
        const empty = document.createElement('p');
        empty.className = 'model-config-list-empty';
        empty.textContent = '没有匹配的模型配置';
        list.appendChild(empty);
    }
    detail.appendChild(pickerDetail(selected, library.defaultId));
    workspace.append(list, detail);
    body.append(toolbar, workspace, actions);
    search.addEventListener('input', (event) => {
        pickerState.search = event.target.value || '';
        renderPickerBody(dialog, library, pickerState);
        document.querySelector('.model-config-picker-search')?.focus();
    });
}

function showDialog(dialog) {
    if (dialog.showModal && !dialog.open) dialog.showModal();
    else if (!dialog.open) dialog.setAttribute('open', 'open');
}

export async function openModelConfigPickerDialog() {
    const dialog = document.getElementById('global-model-config-picker-dialog');
    if (!dialog) return null;
    let library = await fetchModelConfigLibrary();
    const pickerState = { selectedId: library.defaultId || library.items[0]?.id || '', search: '' };
    renderPickerBody(dialog, library, pickerState);
    const refresh = document.getElementById('btn-model-config-picker-refresh');
    if (refresh) {
        refresh.onclick = async () => {
            refresh.disabled = true;
            try {
                library = await fetchModelConfigLibrary();
                pickerState.selectedId = library.items.some((item) => item.id === pickerState.selectedId)
                    ? pickerState.selectedId
                    : (library.defaultId || library.items[0]?.id || '');
                renderPickerBody(dialog, library, pickerState);
                const meta = document.getElementById('global-model-config-picker-dialog-meta');
                if (meta) meta.textContent = '选择后会填写模型格式、DiT、Qwen3 和 VAE。';
            } catch (error) {
                const meta = document.getElementById('global-model-config-picker-dialog-meta');
                if (meta) meta.textContent = error.message;
            } finally {
                refresh.disabled = false;
            }
        };
    }
    showDialog(dialog);
    document.querySelector('.model-config-picker-search')?.focus({ preventScroll: true });

    return new Promise((resolve) => {
        dialog.addEventListener('close', () => {
            const selected = library.items.find((item) => item.id === pickerState.selectedId) || null;
            resolve(dialog.returnValue === 'apply' ? selected : null);
            dialog.returnValue = '';
        }, { once: true });
    });
}

export function bindModelConfigPickerEvents() {
    const manage = document.getElementById('btn-model-config-picker-manage');
    if (manage && manage.dataset.bound !== '1') {
        manage.dataset.bound = '1';
        manage.addEventListener('click', () => {
            document.getElementById('global-model-config-picker-dialog')?.close();
            document.querySelector('[data-tab="model-config"]')?.click();
        });
    }
}
