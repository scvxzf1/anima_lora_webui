import { fetchModelConfigLibrary } from './api.js?v=model-configs-20260809-1';
import { MODEL_CONFIG_PATH_FIELDS, modelFamilyLabel } from './model-config-data.js?v=model-configs-20260809-1';

function pickerDetail(item) {
    const detail = document.createElement('section');
    detail.className = 'model-config-picker-detail';
    if (!item) {
        detail.textContent = '请选择一项模型配置';
        return detail;
    }
    const heading = document.createElement('div');
    heading.className = 'model-config-picker-detail-head';
    const title = document.createElement('strong');
    title.textContent = item.name;
    const family = document.createElement('span');
    family.textContent = modelFamilyLabel(item.model_family);
    heading.append(title, family);
    detail.appendChild(heading);
    for (const field of MODEL_CONFIG_PATH_FIELDS) {
        const row = document.createElement('div');
        const label = document.createElement('span');
        label.textContent = field.label;
        const path = document.createElement('code');
        path.textContent = item[field.key];
        row.append(label, path);
        detail.appendChild(row);
    }
    return detail;
}

function renderPickerBody(dialog, library, pickerState) {
    const body = document.getElementById('global-model-config-picker-body');
    if (!body) return;
    body.innerHTML = '';
    const toolbar = document.createElement('div');
    toolbar.className = 'model-config-picker-toolbar';
    const search = document.createElement('input');
    search.type = 'search';
    search.className = 'model-config-picker-search';
    search.placeholder = '搜索名称、格式或路径';
    search.value = pickerState.search;
    toolbar.appendChild(search);

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
    const selectedComplete = selected && MODEL_CONFIG_PATH_FIELDS.every(({ key }) => Boolean(selected[key]));
    apply.disabled = !selectedComplete;
    apply.addEventListener('click', () => dialog.close('apply'));
    actions.appendChild(apply);

    const query = pickerState.search.trim().toLocaleLowerCase();
    const visible = library.items.filter((item) => (
        !query
        || item.name.toLocaleLowerCase().includes(query)
        || modelFamilyLabel(item.model_family).toLocaleLowerCase().includes(query)
        || MODEL_CONFIG_PATH_FIELDS.some(({ key }) => item[key].toLocaleLowerCase().includes(query))
    ));
    for (const item of visible) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `model-config-picker-option${item.id === pickerState.selectedId ? ' active' : ''}`;
        button.disabled = !MODEL_CONFIG_PATH_FIELDS.every(({ key }) => Boolean(item[key]));
        const title = document.createElement('strong');
        title.textContent = item.name;
        const meta = document.createElement('span');
        const isComplete = MODEL_CONFIG_PATH_FIELDS.every(({ key }) => Boolean(item[key]));
        meta.textContent = `${modelFamilyLabel(item.model_family)}${item.id === library.defaultId ? ' · 默认' : ''}${isComplete ? '' : ' · 路径不完整'}`;
        const path = document.createElement('small');
        path.textContent = item.pretrained_model_name_or_path;
        button.append(title, meta, path);
        button.addEventListener('click', () => {
            pickerState.selectedId = item.id;
            renderPickerBody(dialog, library, pickerState);
        });
        list.appendChild(button);
    }
    if (!visible.length) {
        const empty = document.createElement('p');
        empty.className = 'model-config-list-empty';
        empty.textContent = '没有匹配的模型配置';
        list.appendChild(empty);
    }
    detail.appendChild(pickerDetail(library.items.find((item) => item.id === pickerState.selectedId)));
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
