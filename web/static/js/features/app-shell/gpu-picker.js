export function createGpuPicker({
    storageKey,
    api,
    storage = window.localStorage,
} = {}) {
    let availableGpus = [];
    let selectedGpuWhitelist = [];

    function loadStoredGpuWhitelist() {
        try {
            const parsed = JSON.parse(storage.getItem(storageKey) || '[]');
            if (!Array.isArray(parsed)) return [];
            return parsed
                .map((item) => Number(item))
                .filter((item, index, list) => Number.isInteger(item) && item >= 0 && list.indexOf(item) === index);
        } catch (_) {
            return [];
        }
    }

    function saveGpuWhitelist() {
        try {
            storage.setItem(storageKey, JSON.stringify(selectedGpuWhitelist));
        } catch (_) {
            // 浏览器禁用 localStorage 时，本次页面内选择仍然有效。
        }
    }

    async function loadGpuOptions() {
        selectedGpuWhitelist = loadStoredGpuWhitelist();
        renderGpuPicker();
        if (location.protocol === 'file:') {
            updateGpuPickerNote('静态打开无法读取本机 GPU；选择会在服务模式下生效。');
            return;
        }
        try {
            const payload = await api('/api/training/gpus');
            availableGpus = Array.isArray(payload.gpus) ? payload.gpus : [];
            selectedGpuWhitelist = sanitizeGpuWhitelist(selectedGpuWhitelist);
            saveGpuWhitelist();
            renderGpuPicker();
        } catch (e) {
            availableGpus = [];
            renderGpuPicker();
            updateGpuPickerNote('读取 GPU 列表失败，训练会使用默认可见 GPU。');
        }
    }

    function sanitizeGpuWhitelist(list) {
        const selected = Array.isArray(list) ? list : [];
        if (!availableGpus.length) return selected.filter((item) => Number.isInteger(item) && item >= 0);
        const known = new Set(availableGpus.map((gpu) => Number(gpu.index)));
        return selected.filter((item) => known.has(item));
    }

    function renderGpuPicker() {
        const toggle = document.getElementById('gpu-picker-toggle');
        const list = document.getElementById('gpu-option-list');
        const allCheckbox = document.getElementById('gpu-all-checkbox');
        if (!toggle || !list || !allCheckbox) return;

        const selected = new Set(selectedGpuWhitelist);
        const allSelected = selectedGpuWhitelist.length === 0;
        allCheckbox.checked = allSelected;
        allCheckbox.indeterminate = false;
        allCheckbox.disabled = allSelected;
        toggle.textContent = gpuPickerSummary();
        toggle.title = gpuPickerTitle();
        list.innerHTML = '';

        if (!availableGpus.length) {
            const empty = document.createElement('div');
            empty.className = 'gpu-picker-note';
            empty.textContent = '未读取到 NVIDIA GPU；保持“全部 GPU”时会沿用系统默认可见设备。';
            list.appendChild(empty);
            updateGpuPickerNote('选择为空表示不限制 GPU，训练使用系统默认可见设备。');
            return;
        }

        for (const gpu of availableGpus) {
            const index = Number(gpu.index);
            const option = document.createElement('label');
            option.className = 'gpu-option';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = String(index);
            checkbox.checked = selected.has(index);
            checkbox.addEventListener('change', () => toggleGpuSelection(index, checkbox.checked));

            const body = document.createElement('span');
            const name = document.createElement('span');
            name.className = 'gpu-option-name';
            name.textContent = gpu.label || `GPU ${index} · ${gpu.name || '未命名显卡'}`;
            body.appendChild(name);
            const meta = document.createElement('span');
            meta.className = 'gpu-option-meta';
            meta.textContent = gpu.memory_total_gb
                ? `显存 ${gpu.memory_total_gb} GB · 训练时写入 CUDA_VISIBLE_DEVICES=${index}`
                : `训练时写入 CUDA_VISIBLE_DEVICES=${index}`;
            body.appendChild(meta);
            option.append(checkbox, body);
            list.appendChild(option);
        }

        updateGpuPickerNote(allSelected
            ? '当前不限制 GPU，训练会使用系统默认可见设备。'
            : `当前训练白名单: ${selectedGpuWhitelist.join(', ')}`);
    }

    function gpuPickerSummary() {
        if (!selectedGpuWhitelist.length) return '全部 GPU';
        const names = selectedGpuWhitelist.map((index) => {
            const gpu = availableGpus.find((item) => Number(item.index) === Number(index));
            return gpu?.name ? `GPU ${index} · ${gpu.name}` : `GPU ${index}`;
        });
        if (names.length <= 2) return names.join(' / ');
        return `${names.slice(0, 2).join(' / ')} 等 ${names.length} 张`;
    }

    function gpuPickerTitle() {
        return [
            '选择训练时允许使用的 GPU 白名单。',
            '留空/全部 GPU 表示不覆盖系统默认可见设备。',
            '选择会保存在本机浏览器，并在开始训练或自动预处理后训练时生效。',
        ].join('\n');
    }

    function updateGpuPickerNote(text) {
        const note = document.getElementById('gpu-picker-note');
        if (note) note.textContent = text;
    }

    function setGpuWhitelist(next) {
        selectedGpuWhitelist = sanitizeGpuWhitelist(next);
        saveGpuWhitelist();
        renderGpuPicker();
    }

    function toggleGpuSelection(index, checked) {
        const selected = new Set(selectedGpuWhitelist);
        if (checked) selected.add(index);
        else selected.delete(index);
        setGpuWhitelist([...selected].sort((a, b) => a - b));
    }

    function selectedGpuPayload() {
        return selectedGpuWhitelist.slice().sort((a, b) => a - b);
    }

    function closeGpuPickerPanel() {
        const panel = document.getElementById('gpu-picker-panel');
        const toggle = document.getElementById('gpu-picker-toggle');
        if (!panel || !toggle) return;
        panel.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
    }

    function initGpuPickerEvents() {
        const picker = document.getElementById('gpu-picker');
        const toggle = document.getElementById('gpu-picker-toggle');
        const panel = document.getElementById('gpu-picker-panel');
        const allCheckbox = document.getElementById('gpu-all-checkbox');
        if (!picker || !toggle || !panel || !allCheckbox) return;
        toggle.addEventListener('click', () => {
            const nextOpen = panel.hidden;
            panel.hidden = !nextOpen;
            toggle.setAttribute('aria-expanded', String(nextOpen));
        });
        allCheckbox.addEventListener('change', () => setGpuWhitelist([]));
        document.addEventListener('click', (event) => {
            if (!picker.contains(event.target)) closeGpuPickerPanel();
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') closeGpuPickerPanel();
        });
    }

    return {
        initGpuPickerEvents,
        loadGpuOptions,
        selectedGpuPayload,
    };
}
