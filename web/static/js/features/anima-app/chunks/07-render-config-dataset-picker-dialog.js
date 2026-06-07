/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.renderConfigDatasetPickerDialog = function renderConfigDatasetPickerDialog() {
        const dialog = document.getElementById('config-dataset-picker-dialog');
        const body = document.getElementById('config-dataset-picker-dialog-body');
        if (!dialog || !body) return;
        body.innerHTML = '';

        const toolbar = document.createElement('div');
        toolbar.className = 'config-dataset-dialog-toolbar';
        const search = document.createElement('input');
        search.type = 'search';
        search.className = 'field-input config-dataset-search';
        search.placeholder = '搜索数据集预设、路径或原始目录';
        search.value = configDatasetPickerSearch;
        search.addEventListener('input', () => {
            const cursor = search.selectionStart ?? search.value.length;
            configDatasetPickerSearch = search.value;
            renderConfigDatasetPickerDialog();
            const nextSearch = document.querySelector('#config-dataset-picker-dialog .config-dataset-search');
            if (nextSearch) {
                nextSearch.focus();
                nextSearch.setSelectionRange(cursor, cursor);
            }
        });
        toolbar.appendChild(search);
        body.appendChild(toolbar);

        const workspace = document.createElement('div');
        workspace.className = 'config-dataset-workspace config-dataset-dialog-workspace';
        workspace.appendChild(createConfigDatasetPresetList());
        workspace.appendChild(createConfigDatasetPresetPreview());
        body.appendChild(workspace);
    }

    globalThis.datasetPresetOptionLabel = function datasetPresetOptionLabel(preset) {
        const summary = preset?.summary || {};
        const name = preset?.label || preset?.filename || preset?.path || '未命名预设';
        const count = Number(summary.dataset_count || 0);
        const repeats = Number(summary.repeat_total || 0);
        const lock = preset?.readonly ? '只读 · ' : '';
        return `${lock}${name} · ${count || 0} 组 · 重复 ${repeats || 0}`;
    }

    globalThis.createConfigDatasetPresetList = function createConfigDatasetPresetList() {
        const list = document.createElement('div');
        list.className = 'config-dataset-preset-list';
        const noneBtn = createConfigDatasetPresetButton(null);
        list.appendChild(noneBtn);

        const presets = filteredConfigDatasetPresets();
        if (!presets.length && configDatasetPickerSearch.trim()) {
            const empty = document.createElement('p');
            empty.className = 'config-dataset-picker-empty';
            empty.textContent = '没有匹配的数据集预设。';
            list.appendChild(empty);
            return list;
        }

        for (const preset of presets) {
            list.appendChild(createConfigDatasetPresetButton(preset));
        }
        return list;
    }

    globalThis.createConfigDatasetPresetButton = function createConfigDatasetPresetButton(preset) {
        const isNone = !preset;
        const file = isNone ? '' : preset.path;
        const summary = preset?.summary || {};
        const active = file === selectedConfigDatasetFile;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = [
            'config-dataset-preset-option',
            active ? 'active' : '',
            preset?.readonly ? 'readonly' : '',
        ].filter(Boolean).join(' ');
        btn.dataset.file = file;
        const title = document.createElement('strong');
        title.textContent = isNone
            ? '不使用独立数据集预设'
            : (preset.label || preset.filename || preset.path || '未命名预设');
        const path = document.createElement('span');
        path.textContent = isNone ? '沿用当前训练配置文件中的数据集字段' : preset.path;
        const meta = document.createElement('small');
        meta.textContent = isNone
            ? '当前配置'
            : `${Number(summary.dataset_count || 0)} 组 · 重复 ${Number(summary.repeat_total || 0)}${preset.readonly ? ' · 只读' : ''}`;
        btn.append(title, path, meta);
        btn.addEventListener('click', () => selectConfigDatasetPreset(file));
        return btn;
    }

    globalThis.filteredConfigDatasetPresets = function filteredConfigDatasetPresets() {
        const keyword = configDatasetPickerSearch.trim().toLowerCase();
        const presets = datasetPresetState.presets || [];
        if (!keyword) return presets;
        return presets.filter((preset) => {
            const summary = preset.summary || {};
            return [
                preset.label,
                preset.filename,
                preset.path,
                summary.source_dir,
                            ].some((value) => String(value || '').toLowerCase().includes(keyword));
        });
    }

    globalThis.createConfigDatasetPresetPreview = function createConfigDatasetPresetPreview() {
        const preview = document.createElement('div');
        preview.className = 'config-dataset-preview';
        const summary = document.createElement('div');
        summary.className = 'config-dataset-summary';
        summary.appendChild(createConfigDatasetSummary());
        preview.appendChild(summary);
        preview.appendChild(createConfigDatasetPreviewImage());
        return preview;
    }

    globalThis.createConfigDatasetPreviewImage = function createConfigDatasetPreviewImage() {
        const box = document.createElement('div');
        box.className = 'config-dataset-preview-image';
        const state = configDatasetPreviewState;
        if (!selectedConfigDatasetFile) {
            box.classList.add('empty');
            box.textContent = '选择一个数据集预设后，这里会显示第一张原始图。';
            return box;
        }
        if (state.file !== selectedConfigDatasetFile || state.loading) {
            box.classList.add('empty');
            box.textContent = '正在读取第一张原始图...';
            return box;
        }
        if (state.error) {
            box.classList.add('empty');
            box.textContent = state.error;
            return box;
        }
        const image = Array.isArray(state.payload?.images) ? state.payload.images[0] : null;
        if (!image) {
            box.classList.add('empty');
            box.textContent = state.payload?.message || '没有找到可预览的原始图。';
            return box;
        }
        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name || '数据集预览图';
        img.loading = 'lazy';
        img.addEventListener('error', () => {
            box.classList.add('empty');
            box.textContent = '预览图加载失败。';
        });
        const caption = document.createElement('div');
        caption.className = 'config-dataset-preview-caption';
        const name = document.createElement('strong');
        name.textContent = image.name || '-';
        const path = document.createElement('span');
        path.textContent = state.payload?.directory || image.file || '';
        caption.append(name, path);
        box.append(img, caption);
        return box;
    }

    globalThis.createConfigDatasetSummary = function createConfigDatasetSummary() {
        const wrap = document.createElement('div');
        const preset = datasetPresetByFile(selectedConfigDatasetFile);
        const summary = selectedConfigDatasetSummary || preset?.summary || {};
        if (!selectedConfigDatasetFile) {
            wrap.className = 'config-dataset-summary-empty';
            wrap.textContent = '未选择独立数据集预设；训练会沿用当前配置文件里的数据集字段。';
            return wrap;
        }
        const items = [
            ['预设文件', selectedConfigDatasetFile],
            ['数据组数', String(summary.dataset_count || 0)],
            ['重复合计', String(summary.repeat_total || 0)],
            ['第 1 组原始路径', summary.source_dir || '-'],
        ];
        if (selectedConfigDatasetFile !== (currentConfig.dataset_config || '')) {
            items.unshift(['状态', '未保存，保存当前配置后生效']);
        }
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            row.append(key, valEl);
            wrap.appendChild(row);
        }
        return wrap;
    }

    globalThis.selectConfigDatasetPreset = async function selectConfigDatasetPreset(file) {
        selectedConfigDatasetFile = file || '';
        selectedConfigDatasetSummary = datasetPresetSummaryByFile(selectedConfigDatasetFile);
        configDatasetPreviewState = {
            file: '',
            loading: false,
            payload: null,
            error: '',
        };
        renderConfigDatasetPicker();
        renderConfigDatasetPickerDialog();
        updateTomlDirtyState();
        await loadStepEstimate();
    }

    globalThis.datasetPresetByFile = function datasetPresetByFile(file) {
        return (datasetPresetState.presets || []).find((item) => item.path === file) || null;
    }

    globalThis.datasetPresetSummaryByFile = function datasetPresetSummaryByFile(file) {
        return datasetPresetByFile(file)?.summary || null;
    }

    globalThis.ensureConfigDatasetPreview = function ensureConfigDatasetPreview() {
        if (!selectedConfigDatasetFile) return;
        if (configDatasetPreviewState.file === selectedConfigDatasetFile && (configDatasetPreviewState.loading || configDatasetPreviewState.payload || configDatasetPreviewState.error)) {
            return;
        }
        loadConfigDatasetPresetPreview(selectedConfigDatasetFile);
    }

    globalThis.loadConfigDatasetPresetPreview = async function loadConfigDatasetPresetPreview(file) {
        if (!file || location.protocol === 'file:') return;
        const requestSeq = ++configDatasetPreviewRequestSeq;
        configDatasetPreviewState = {
            file,
            loading: true,
            payload: null,
            error: '',
        };
        renderConfigDatasetPreviewArea();
        try {
            const params = new URLSearchParams({
                file,
                dataset_index: '0',
                source: 'source',
                limit: '1',
            });
            const payload = await datasetPresetApi(`/api/config/dataset-presets/images?${params.toString()}`);
            if (requestSeq !== configDatasetPreviewRequestSeq || file !== selectedConfigDatasetFile) return;
            if (!payload.ok) throw new Error(payload.error || '读取数据集预览失败');
            configDatasetPreviewState = {
                file,
                loading: false,
                payload,
                error: '',
            };
        } catch (e) {
            if (requestSeq !== configDatasetPreviewRequestSeq || file !== selectedConfigDatasetFile) return;
            configDatasetPreviewState = {
                file,
                loading: false,
                payload: null,
                error: e.message || '读取数据集预览失败',
            };
        }
        renderConfigDatasetPreviewArea();
    }

    globalThis.renderConfigDatasetPreviewArea = function renderConfigDatasetPreviewArea() {
        const previews = document.querySelectorAll('.config-dataset-preview');
        if (!previews.length) return;
        previews.forEach((preview) => {
            preview.innerHTML = '';
            const summary = document.createElement('div');
            summary.className = 'config-dataset-summary';
            summary.appendChild(createConfigDatasetSummary());
            preview.appendChild(summary);
            preview.appendChild(createConfigDatasetPreviewImage());
        });
    }

    globalThis.createDatasetEditor = function createDatasetEditor() {
        const panel = document.createElement('div');
        panel.id = 'dataset-editor';
        panel.className = 'dataset-editor';
        renderDatasetEditor(panel);
        return panel;
    }

    globalThis.renderDatasetPresetList = function renderDatasetPresetList() {
        const list = document.getElementById('dataset-preset-list');
        if (!list) return;
        list.innerHTML = '';
        const presets = datasetPresetState.presets || [];
        const groups = datasetPresetGroupsForDisplay();
        updateDatasetPresetPageSummary();
        const showErrorAsEmptyState = datasetPresetState.error && !presets.length;
        if (datasetPresetState.loading && !presets.length) {
            const loading = document.createElement('p');
            loading.className = 'dataset-preset-empty';
            loading.textContent = '正在读取数据集预设...';
            list.appendChild(loading);
            return;
        }
        if (showErrorAsEmptyState) {
            const error = document.createElement('p');
            error.className = 'dataset-preset-empty error';
            error.textContent = datasetPresetState.error;
            list.appendChild(error);
        }
        if (!presets.length && !groups.length) {
            const empty = document.createElement('p');
            empty.className = 'dataset-preset-empty';
            empty.textContent = datasetPresetState.error ? '读取数据集预设失败。' : '还没有数据集预设。';
            list.appendChild(empty);
            return;
        }
        if (!groups.length) {
            const empty = document.createElement('p');
            empty.className = 'dataset-preset-empty';
            empty.textContent = '没有匹配的数据集预设。';
            list.appendChild(empty);
            return;
        }
        const stored = readDatasetPresetGroupState();
        for (const group of groups) {
            list.appendChild(createDatasetPresetGroupNode(group, stored));
        }
    }

    globalThis.updateDatasetPresetPageSummary = function updateDatasetPresetPageSummary() {
        const summary = document.getElementById('dataset-page-summary');
        if (!summary) return;
        const presets = datasetPresetState.presets || [];
        const groups = datasetPresetState.groups || [];
        const totalDatasets = presets.reduce((sum, preset) => sum + Number(preset.summary?.dataset_count || 0), 0);
        const totalRepeats = presets.reduce((sum, preset) => sum + Number(preset.summary?.repeat_total || 0), 0);
        summary.innerHTML = '';
        [
            ['预设', presets.length],
            ['分组', groups.length || 1],
            ['子集', totalDatasets],
            ['重复', totalRepeats],
        ].forEach(([label, value]) => {
            const item = document.createElement('span');
            item.className = 'dataset-page-summary-stat';
            item.innerHTML = `<strong>${escapeHtml(String(value))}</strong><small>${escapeHtml(label)}</small>`;
            summary.appendChild(item);
        });
        if (datasetPresetState.dirty) {
            const dirty = document.createElement('span');
            dirty.className = 'dataset-page-summary-dirty';
            dirty.textContent = '当前预设未保存';
            summary.appendChild(dirty);
        }
    }

    globalThis.datasetPresetGroupsForDisplay = function datasetPresetGroupsForDisplay() {
        const keyword = datasetPresetState.search.trim().toLowerCase();
        const presetMap = new Map((datasetPresetState.presets || []).map((preset) => [preset.path, preset]));
        const sourceGroups = (datasetPresetState.groups || []).length
            ? datasetPresetState.groups
            : [{
                id: 'datasets',
                label: '数据集配置',
                open: false,
                kind: 'dataset',
                files: datasetPresetState.presets || [],
                movable: true,
            }];
        const covered = new Set();
        const groups = [];

        for (const rawGroup of sourceGroups) {
            const files = (rawGroup.files || [])
                .map((item) => presetMap.get(item.path) || item)
                .filter((item) => item?.path && presetMap.has(item.path))
                .filter((item) => datasetPresetMatchesSearch(item, keyword));
            (rawGroup.files || []).forEach((item) => {
                if (item?.path && presetMap.has(item.path)) covered.add(item.path);
            });
            if (keyword && !files.length) continue;
            if (!files.length && rawGroup.kind !== 'dataset' && rawGroup.id !== 'datasets' && rawGroup.id !== 'unfiled_datasets') continue;
            groups.push({ ...rawGroup, files });
        }

        const ungrouped = (datasetPresetState.presets || [])
            .filter((preset) => !covered.has(preset.path))
            .filter((preset) => datasetPresetMatchesSearch(preset, keyword));
        if (ungrouped.length) {
            groups.push({
                id: 'unfiled_datasets',
                label: '未分组数据集配置',
                open: true,
                kind: 'dataset',
                movable: true,
                files: ungrouped,
            });
        }
        return sortDatasetPresetGroups(groups);
    }

    globalThis.isUnfiledDatasetGroup = function isUnfiledDatasetGroup(group) {
        return group?.id === 'unfiled_datasets';
    }

    globalThis.sortDatasetPresetGroups = function sortDatasetPresetGroups(groups) {
        return [...groups].sort((a, b) => {
            if (isUnfiledDatasetGroup(a)) return -1;
            if (isUnfiledDatasetGroup(b)) return 1;
            return 0;
        });
    }

    globalThis.orderDatasetPresetsForGroups = function orderDatasetPresetsForGroups(presets, groups) {
        const presetMap = new Map((presets || []).map((preset) => [preset.path, preset]));
        const ordered = [];
        const seen = new Set();
        for (const group of sortDatasetPresetGroups(groups || [])) {
            for (const item of group.files || []) {
                if (!item?.path || seen.has(item.path) || !presetMap.has(item.path)) continue;
                ordered.push(presetMap.get(item.path));
                seen.add(item.path);
            }
        }
        for (const preset of presets || []) {
            if (!preset?.path || seen.has(preset.path)) continue;
            ordered.push(preset);
        }
        return ordered;
    }

    globalThis.datasetPresetMatchesSearch = function datasetPresetMatchesSearch(preset, keyword) {
        if (!keyword) return true;
        const summary = preset?.summary || {};
        return [
            preset?.label,
            preset?.filename,
            preset?.path,
            summary.source_dir,
            summary.image_dir,
            summary.cache_dir,
        ].some((value) => String(value || '').toLowerCase().includes(keyword));
    }

    globalThis.eventTargetClosest = function eventTargetClosest(event, selector) {
        const target = event?.target;
        return target instanceof Element ? target.closest(selector) : null;
    }

    globalThis.createFileGroupDragImage = function createFileGroupDragImage(payload) {
        const image = document.createElement('div');
        image.className = 'file-group-drag-image';
        image.textContent = payload.file || payload.groupId || '移动项目';
        document.body.appendChild(image);
        return image;
    }

    globalThis.removeFileGroupDragImage = function removeFileGroupDragImage(image) {
        if (image?.parentNode) image.parentNode.removeChild(image);
    }

    globalThis.setFileGroupDragData = function setFileGroupDragData(event, payload) {
        const data = payload.file || payload.groupId || payload.target || 'move';
        const transfer = event?.dataTransfer;
        if (!transfer) return;
        let image = null;
        try {
            transfer.setData('text/plain', data);
            transfer.setData('application/x-anima-file-group', JSON.stringify({
                target: payload.target || '',
                scope: payload.scope || '',
                file: payload.file || '',
                groupId: payload.groupId || '',
            }));
            transfer.effectAllowed = 'move';
            image = createFileGroupDragImage(payload);
            transfer.setDragImage(image, 12, 12);
        } catch (e) {
            /* 部分浏览器会限制 DataTransfer 写入；内存态拖拽仍可继续。 */
        } finally {
            if (image) window.setTimeout(() => removeFileGroupDragImage(image), 0);
        }
    }

    globalThis.canBeginFileGroupDrag = function canBeginFileGroupDrag(payload, disabled) {
        if (disabled || (payload.canDrag && !payload.canDrag())) {
            if (payload.blockedMessage) payload.blockedMessage();
            return false;
        }
        return true;
    }

    globalThis.beginFileGroupDrag = function beginFileGroupDrag(payload, handle) {
        fileGroupDragState = payload;
        payload.sourceElement?.classList.add('file-group-dragging');
        handle?.classList.add('dragging');
    }

    globalThis.createFileGroupPointerDragImage = function createFileGroupPointerDragImage(payload) {
        const image = createFileGroupDragImage(payload);
        image.classList.add('file-group-drag-image-pointer');
        return image;
    }

    globalThis.moveFileGroupPointerDragImage = function moveFileGroupPointerDragImage(image, x, y) {
        if (!image) return;
        image.style.left = `${x + 14}px`;
        image.style.top = `${y + 14}px`;
    }

    globalThis.registerFileGroupDropTarget = function registerFileGroupDropTarget(node, resolve) {
        node.setAttribute(FILE_GROUP_DROP_TARGET_ATTR, '1');
        fileGroupDropTargets.set(node, resolve);
        fileGroupDropTargetNodes.add(node);
    }
