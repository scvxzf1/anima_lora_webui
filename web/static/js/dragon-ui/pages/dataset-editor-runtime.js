/* Stable bindings and incremental UI updates for the dataset editor. */

const SUMMARY_FIELDS = new Set(['is_reg', 'num_repeats', 'resolution']);

export function createDatasetEditorBindings(root) {
    return {
        dirty: root.querySelector('[data-dataset-dirty]'),
        dirtyText: root.querySelector('[data-dataset-dirty-text]'),
        status: root.querySelector('[data-savebar-status]'),
        syncCard: root.querySelector('[data-dataset-sync-card]'),
    };
}

export function renderDatasetDirtyState(bindings, state) {
    const dirtyText = state.dirty
        ? '有未保存更改'
        : (state.selectedFile === state.datasetConfig ? '已同步至配置' : '已同步至预设');
    const statusText = state.readonly
        ? '系统预设只读，请复制后编辑。'
        : (state.dirty ? '当前修改尚未写入 TOML。' : '数据集预设已同步。');
    setText(bindings.dirtyText, dirtyText);
    setDataset(bindings.dirty, 'state', state.dirty ? 'dirty' : 'synced');
    setText(bindings.status, statusText);
    setAttribute(bindings.syncCard, 'data-dirty', String(state.dirty));
}

export function disableDatasetPreviews(root) {
    root.querySelectorAll('[data-dataset-preview]').forEach((button) => {
        if (!button.disabled) button.disabled = true;
    });
}

export function updateDatasetRowSummaryForControl(control) {
    if (!SUMMARY_FIELDS.has(control?.dataset?.field)) return false;
    const row = control.closest('[data-dataset-row]');
    return updateDatasetRowSummary(row);
}

export function updateDatasetRowSummaries(root) {
    root.querySelectorAll('[data-dataset-row]').forEach(updateDatasetRowSummary);
}

function updateDatasetRowSummary(row) {
    if (!row) return false;
    const isReg = row.querySelector('[data-field="is_reg"]')?.value === 'true';
    const repeat = row.querySelector('[data-field="num_repeats"]')?.value || '1';
    const resolution = row.querySelector('[data-field="resolution"]')?.value || '1024';
    const summary = row.querySelector('[data-row-summary]');
    const text = `${isReg ? '正则数据' : '训练数据'} · 重复 ${repeat} · ${resolution}px`;
    if (summary && summary.textContent !== text) summary.textContent = text;
    return Boolean(summary);
}

function setText(node, value) {
    if (node && node.textContent !== value) node.textContent = value;
}

function setDataset(node, key, value) {
    if (node && node.dataset[key] !== value) node.dataset[key] = value;
}

function setAttribute(node, key, value) {
    if (node && node.getAttribute(key) !== value) node.setAttribute(key, value);
}
