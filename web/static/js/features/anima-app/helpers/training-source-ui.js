export function textNode(tag, text) {
    const node = document.createElement(tag);
    node.textContent = text || '';
    return node;
}

export function optionNodeLocal(value, label) {
    const option = document.createElement('option');
    option.value = value || '';
    option.textContent = label || value || '';
    return option;
}

export function summaryLine(label, value) {
    const row = document.createElement('div');
    const key = document.createElement('span');
    key.textContent = label;
    const val = document.createElement('code');
    val.textContent = value || '-';
    row.append(key, val);
    return row;
}
