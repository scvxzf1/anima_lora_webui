/* Path actions and validation for Dragon dataset rows. */

const validationTimers = new WeakMap();
const validationVersions = new WeakMap();

export function bindDatasetPathTools(api, root, { onFeedback } = {}) {
    const cleanups = [];
    root.querySelectorAll('[data-dataset-row]').forEach((row) => {
        const input = row.querySelector('[data-field="source_dir"]');
        if (!input) return;
        const copyButton = row.querySelector('[data-dataset-copy]');
        const onInput = () => {
            if (copyButton) copyButton.disabled = !String(input.value || '').trim();
            schedulePathValidation(api, row, input);
        };
        const copy = row.querySelector('[data-dataset-copy]');
        const browse = row.querySelector('[data-dataset-browse]');
        const onCopy = () => copyPath(input, onFeedback);
        const onBrowse = () => chooseDirectory(input, onFeedback);
        input.addEventListener('input', onInput);
        copy?.addEventListener('click', onCopy);
        browse?.addEventListener('click', onBrowse);
        cleanups.push(() => {
            input.removeEventListener('input', onInput);
            copy?.removeEventListener('click', onCopy);
            browse?.removeEventListener('click', onBrowse);
            const timer = validationTimers.get(input);
            if (timer) window.clearTimeout(timer);
            validationTimers.delete(input);
            validationVersions.set(input, Number(validationVersions.get(input) || 0) + 1);
        });
        schedulePathValidation(api, row, input, 80);
    });
    return () => cleanups.forEach((cleanup) => cleanup());
}

export function refreshDatasetPathStatus(api, row) {
    const input = row?.querySelector('[data-field="source_dir"]');
    if (input) schedulePathValidation(api, row, input, 0);
}

function schedulePathValidation(api, row, input, delay = 420) {
    const previous = validationTimers.get(input);
    if (previous) window.clearTimeout(previous);
    setPathStatus(row, 'checking', '正在检测…');
    validationTimers.set(input, window.setTimeout(() => validatePath(api, row, input), delay));
}

async function validatePath(api, row, input) {
    const path = String(input.value || '').trim();
    if (!path) {
        setPathStatus(row, 'idle', '请输入路径');
        return;
    }
    const version = Number(validationVersions.get(input) || 0) + 1;
    validationVersions.set(input, version);
    try {
        const params = new URLSearchParams({ source_image_dir: path, inspect: '1' });
        const result = await api(`/api/config/data-dirs/suggest?${params.toString()}`);
        if (!input.isConnected || validationVersions.get(input) !== version) return;
        if (result.ok === false) throw new Error(result.error || '路径检测失败');
        if (typeof result.source_exists !== 'boolean') {
            setPathStatus(row, 'idle', '重启服务后检测');
            return;
        }
        if (result.source_inspection_error) {
            setPathStatus(row, 'invalid', '目录无法完整读取');
            return;
        }
        if (!result.source_exists) {
            setPathStatus(row, 'invalid', '路径不存在');
        } else if (!result.source_is_dir) {
            setPathStatus(row, 'invalid', '路径不是目录');
        } else {
            const count = Number(result.source_image_count || 0);
            setPathStatus(row, count > 0 ? 'valid' : 'empty', count > 0 ? `检测到 ${count} 张图片` : '目录存在，未检测到图片');
        }
    } catch (error) {
        if (!input.isConnected || validationVersions.get(input) !== version) return;
        setPathStatus(row, 'invalid', error?.message || '路径检测失败');
    }
}

function setPathStatus(row, state, message) {
    const status = row.querySelector('[data-dataset-path-status]');
    if (!status) return;
    status.dataset.state = state;
    const text = status.lastElementChild;
    if (text) text.textContent = message;
}

async function copyPath(input, onFeedback) {
    const path = String(input.value || '').trim();
    if (!path) return;
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(path);
        } else {
            input.select();
            document.execCommand('copy');
            input.setSelectionRange(path.length, path.length);
        }
        onFeedback?.('目录路径已复制', 'success');
    } catch {
        onFeedback?.('无法访问剪贴板，请手动复制路径', 'error');
    }
}

async function chooseDirectory(input, onFeedback) {
    try {
        const name = window.showDirectoryPicker
            ? (await window.showDirectoryPicker({ mode: 'read' }))?.name
            : await chooseDirectoryFallback();
        if (!name) return;
        input.value = name;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        onFeedback?.('已填入所选目录名；绝对路径可直接粘贴', 'success');
    } catch (error) {
        if (error?.name !== 'AbortError') onFeedback?.('无法读取所选目录', 'error');
    }
}

function chooseDirectoryFallback() {
    return new Promise((resolve) => {
        const picker = document.createElement('input');
        picker.type = 'file';
        picker.multiple = true;
        picker.setAttribute('webkitdirectory', '');
        picker.hidden = true;
        picker.addEventListener('change', () => {
            const relative = picker.files?.[0]?.webkitRelativePath || '';
            picker.remove();
            resolve(relative.split('/').filter(Boolean)[0] || '');
        }, { once: true });
        document.body.appendChild(picker);
        picker.click();
    });
}
