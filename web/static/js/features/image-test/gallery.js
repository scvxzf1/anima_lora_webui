import {
    IMAGE_TEST_HISTORY_RANGE_OPTIONS,
    daysForImageTestHistoryRange,
    normalizeImageTestHistoryRange,
} from './state.js?v=module-bootstrap-20260704-1';
import { createZipDataBlob, downloadBlob as triggerBlobDownload } from '../../shared/download.js?v=module-bootstrap-20260704-1';

const DEFAULT_FILTER_VALUE = '7';
const DEFAULT_SELECTION_NOTE = 'Shift 连选仅覆盖当前已展开且当前可见的图片；Ctrl/⌘ 可增量点选。';
const DEFAULT_EXPANDED_DAYS = 7;
const GROUP_INITIAL_RENDER_COUNT = 24;
const GROUP_RENDER_INCREMENT = 24;
const GROUP_LOAD_MORE_ROOT_MARGIN = '320px 0px';
const GROUP_VIRTUALIZE_THRESHOLD = 48;
const GROUP_VIRTUAL_OVERSCAN_ROWS = 2;
const EXPORT_BACKGROUND = '#ffffff';
const EXPORT_CELL_BACKGROUND = '#f8fafc';
const EXPORT_TEXT = '#0f172a';
const EXPORT_META = '#475569';
const EXPORT_MAX_EDGE = 8192;
const EXPORT_CELL_MIN = 220;
const EXPORT_CELL_MAX = 1024;

export function createImageTestGallery({
    formatBytes,
    openPreviewDialog,
    requestHistoryReload,
    initialFilterValue = DEFAULT_FILTER_VALUE,
}) {
    const state = {
        initialized: false,
        currentPayload: {},
        filterValue: normalizeImageTestHistoryRange(initialFilterValue, DEFAULT_FILTER_VALUE),
        expansionInitialized: false,
        expandedGroupKeys: new Set(),
        selectedKeys: new Set(),
        lastSelectedKey: '',
        knownImageKeys: new Set(),
        freshKeys: new Set(),
        freshCountsByGroup: new Map(),
        imageMap: new Map(),
        groupsByKey: new Map(),
        renderWindowByGroup: new Map(),
        virtualWindowByGroup: new Map(),
        filteredOrderedKeys: [],
        visibleOrderedKeys: [],
        exportPending: false,
        rawExportPending: false,
        loadMoreObserver: null,
        virtualRefreshFrame: 0,
        selectionMessage: '',
        selectionTone: '',
    };

    function init() {
        if (state.initialized) return;
        state.initialized = true;
        document.getElementById('image-test-history-filter')?.addEventListener('click', handleFilterClick);
        document.getElementById('btn-image-test-clear-selection')?.addEventListener('click', clearSelection);
        document.getElementById('btn-image-test-export-merged')?.addEventListener('click', () => {
            void exportMergedSelection();
        });
        document.getElementById('btn-image-test-export-originals')?.addEventListener('click', () => {
            void exportOriginalZipSelection();
        });
        window.addEventListener('scroll', scheduleVirtualWindowRefresh, { passive: true });
        window.addEventListener('resize', scheduleVirtualWindowRefresh);
        syncFilterButtons();
        syncSelectionToolbar();
    }

    function render(payload = {}) {
        const normalizedPayload = payload && typeof payload === 'object' ? payload : {};
        const images = Array.isArray(normalizedPayload.images) ? normalizedPayload.images : [];
        state.currentPayload = normalizedPayload;
        syncFreshKeys(images);
        state.imageMap = new Map(images.map((image) => [imageKey(image), image]));

        const groups = buildHistoryGroups(images, state.filterValue);
        state.groupsByKey = new Map(groups.map((group) => [group.key, group]));
        syncFreshGroupCounts(groups);
        syncExpandedGroupKeys(groups);
        syncGroupRenderWindows(groups);
        state.filteredOrderedKeys = groups.flatMap((group) => group.items.map((image) => imageKey(image)));
        refreshVisibleOrderedKeys();
        pruneSelectionToFilteredImages();
        renderHistoryHeader(normalizedPayload, groups);
        renderHistoryGroups(normalizedPayload, groups);
        syncFilterButtons();
        syncSelectionToolbar();
    }

    function setLoading() {
        disconnectLoadMoreObserver();
        cancelVirtualWindowRefresh();
        state.groupsByKey = new Map();
        state.virtualWindowByGroup = new Map();
        clearGrid();
        clearSelection();
        const count = document.getElementById('image-test-count');
        if (count) count.textContent = '读取中';
        const empty = document.getElementById('image-test-empty');
        if (empty) {
            empty.hidden = false;
            empty.textContent = '正在读取 output/tests 中的结果图...';
        }
    }

    function setEmpty(message) {
        state.currentPayload = {};
        state.imageMap = new Map();
        state.freshCountsByGroup = new Map();
        state.groupsByKey = new Map();
        state.renderWindowByGroup = new Map();
        state.virtualWindowByGroup = new Map();
        state.filteredOrderedKeys = [];
        state.visibleOrderedKeys = [];
        disconnectLoadMoreObserver();
        cancelVirtualWindowRefresh();
        clearSelection();
        clearGrid();
        const empty = document.getElementById('image-test-empty');
        if (empty) {
            empty.hidden = false;
            empty.textContent = message;
        }
        const count = document.getElementById('image-test-count');
        if (count) count.textContent = '0 张';
    }

    function handleFilterClick(event) {
        const button = event.target.closest('[data-range]');
        if (!(button instanceof HTMLButtonElement)) return;
        const nextValue = normalizeImageTestHistoryRange(button.dataset.range, DEFAULT_FILTER_VALUE);
        if (nextValue === state.filterValue) return;
        state.filterValue = nextValue;
        state.expansionInitialized = false;
        clearSelection();
        render(state.currentPayload);
        requestHistoryReload?.(nextValue);
    }

    function syncFilterButtons() {
        document.querySelectorAll('#image-test-history-filter [data-range]').forEach((button) => {
            const active = button.getAttribute('data-range') === state.filterValue;
            button.classList.toggle('is-active', active);
            button.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
    }

    function renderHistoryHeader(payload, groups) {
        const title = document.getElementById('image-test-title');
        const subtitle = document.getElementById('image-test-subtitle');
        const count = document.getElementById('image-test-count');
        const filteredCount = state.filteredOrderedKeys.length;
        const total = Number(payload.total ?? payload.count ?? filteredCount) || 0;
        const filterLabel = historyRangeLabel(state.filterValue);
        if (title) {
            title.textContent = payload.label || '推理预览';
        }
        if (subtitle) {
            const directoryText = payload.directory
                ? `目录: ${payload.directory}`
                : '尚未找到 output/tests 结果目录。';
            const groupingText = groups.length ? `已按日期分组，默认展示 ${filterLabel}` : `当前筛选: ${filterLabel}`;
            subtitle.textContent = `${directoryText} · ${groupingText}`;
        }
        if (count) {
            count.textContent = total > filteredCount ? `${filteredCount} / ${total} 张` : `${filteredCount} 张`;
        }
    }

    function renderHistoryGroups(payload, groups) {
        const grid = document.getElementById('image-test-grid');
        const empty = document.getElementById('image-test-empty');
        if (!grid || !empty) return;
        disconnectLoadMoreObserver();
        grid.innerHTML = '';

        if (!groups.length) {
            empty.hidden = false;
            empty.textContent = emptyMessageForCurrentFilter(payload);
            return;
        }

        empty.hidden = true;
        groups.forEach((group) => {
            grid.appendChild(createHistoryGroup(group));
        });
        installLoadMoreObserver();
        scheduleVirtualWindowRefresh();
    }

    function createHistoryGroup(group) {
        const expanded = state.expandedGroupKeys.has(group.key);
        const visibleCount = visibleCountForGroup(group.key, group.items.length);
        const freshCount = freshCountForGroup(group.key);
        const section = document.createElement('section');
        section.className = 'image-test-history-group';
        section.dataset.groupKey = group.key;
        section.classList.toggle('is-fresh-group', freshCount > 0);

        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'image-test-history-group-toggle';
        toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        toggle.addEventListener('click', () => {
            toggleHistoryGroup(group.key);
        });

        const arrow = document.createElement('span');
        arrow.className = 'image-test-history-group-arrow';
        arrow.setAttribute('aria-hidden', 'true');

        const copy = document.createElement('span');
        copy.className = 'image-test-history-group-copy';
        const strong = document.createElement('strong');
        strong.textContent = group.key;
        const meta = document.createElement('span');
        meta.textContent = `${group.metaLabel} · ${group.items.length} 张`;
        copy.append(strong, meta);
        if (freshCount > 0) {
            copy.appendChild(createFreshBadge(freshCount));
        }

        const count = document.createElement('span');
        count.className = 'image-test-history-group-count';
        count.textContent = `${group.items.length}`;

        toggle.append(arrow, copy, count);
        section.appendChild(toggle);

        const body = document.createElement('div');
        body.className = 'preview-grid image-test-history-group-grid';
        body.hidden = !expanded;
        if (expanded) {
            renderGroupBody(body, group, visibleCount);
        }
        section.appendChild(body);
        if (expanded && visibleCount < group.items.length) {
            section.appendChild(createLoadMoreFooter(group, visibleCount));
        }
        return section;
    }

    function createFreshBadge(count) {
        const badge = document.createElement('span');
        badge.className = 'image-test-history-group-fresh-badge';
        badge.textContent = `新增 ${count}`;
        return badge;
    }

    function appendGroupCards(body, images, offset = 0) {
        images.forEach((image, index) => {
            body.appendChild(createImageCard(image, offset + index));
        });
    }

    function renderGroupBody(body, group, visibleCount) {
        body.innerHTML = '';
        const windowState = resolveVirtualWindow(group.key, visibleCount);
        if (windowState.topSpacerPx > 0) {
            body.appendChild(createVirtualSpacer(windowState.topSpacerPx));
        }
        appendGroupCards(
            body,
            group.items.slice(windowState.startIndex, windowState.endIndex),
            windowState.startIndex,
        );
        if (windowState.bottomSpacerPx > 0) {
            body.appendChild(createVirtualSpacer(windowState.bottomSpacerPx));
        }
        body.dataset.virtualized = windowState.virtualized ? 'true' : 'false';
    }

    function createVirtualSpacer(heightPx) {
        const spacer = document.createElement('div');
        spacer.className = 'image-test-history-virtual-spacer';
        spacer.setAttribute('aria-hidden', 'true');
        spacer.style.height = `${Math.max(0, Math.round(heightPx))}px`;
        return spacer;
    }

    function createImageCard(image, index = 0) {
        const key = imageKey(image);
        const selected = state.selectedKeys.has(key);
        const fresh = state.freshKeys.has(key);
        const card = document.createElement('article');
        card.className = 'preview-card image-test-card image-test-history-card';
        card.dataset.imageKey = key;
        card.tabIndex = 0;
        card.setAttribute('role', 'checkbox');
        card.setAttribute('aria-checked', selected ? 'true' : 'false');
        card.classList.toggle('is-selected', selected);
        card.classList.toggle('is-new', fresh);
        card.title = '单击单选；Ctrl/⌘ 可增量点选；Shift 连选当前可见图片；点“查看大图”打开预览。';
        card.addEventListener('click', (event) => {
            if (event.target.closest('[data-image-action="preview"]')) return;
            toggleImageSelection(key, {
                shiftKey: event.shiftKey,
                additive: event.ctrlKey || event.metaKey,
            });
        });
        card.addEventListener('keydown', (event) => {
            if (event.key === ' ' || event.key === 'Enter') {
                event.preventDefault();
                toggleImageSelection(key, {
                    shiftKey: event.shiftKey,
                    additive: event.ctrlKey || event.metaKey,
                });
            }
        });

        const imageWrap = document.createElement('div');
        imageWrap.className = 'preview-card-image image-test-history-card-image';

        const img = document.createElement('img');
        img.src = image.url;
        img.alt = image.name || '生图结果';
        img.loading = index < 6 ? 'eager' : 'lazy';

        const errorMessage = document.createElement('span');
        errorMessage.className = 'preview-card-error-message';
        errorMessage.textContent = '图片加载失败';
        errorMessage.hidden = true;

        img.addEventListener('load', () => {
            card.classList.remove('preview-card-error');
            errorMessage.hidden = true;
        });
        img.addEventListener('error', () => {
            card.classList.add('preview-card-error');
            errorMessage.hidden = false;
        });

        const selectionMark = document.createElement('span');
        selectionMark.className = 'image-test-history-card-selection';
        selectionMark.setAttribute('aria-hidden', 'true');
        selectionMark.textContent = '✓';

        const freshBadge = document.createElement('span');
        freshBadge.className = 'image-test-history-card-fresh';
        freshBadge.hidden = !fresh;
        freshBadge.textContent = '新';

        imageWrap.append(img, selectionMark, freshBadge, errorMessage);

        const meta = document.createElement('div');
        meta.className = 'preview-card-meta image-test-history-card-meta';
        const head = document.createElement('strong');
        head.textContent = image.name || '未命名结果';
        head.title = image.file || image.name || '';

        const file = document.createElement('span');
        file.className = 'preview-card-filename';
        file.textContent = image.file || image.name || '';
        file.title = image.file || image.name || '';

        const details = document.createElement('span');
        details.textContent = imageCardMetaText(image, formatBytes);

        const time = document.createElement('span');
        time.className = 'image-test-history-card-time';
        time.textContent = imageTimestampText(image);

        const actions = document.createElement('div');
        actions.className = 'image-test-history-card-actions';
        const previewButton = document.createElement('button');
        previewButton.type = 'button';
        previewButton.className = 'btn btn-small image-test-history-card-preview';
        previewButton.dataset.imageAction = 'preview';
        previewButton.textContent = '查看大图';
        previewButton.addEventListener('click', (event) => {
            event.stopPropagation();
            openPreviewDialog(image);
        });
        actions.appendChild(previewButton);

        meta.append(head, file, details, time, actions);
        card.append(imageWrap, meta);
        return card;
    }

    function toggleHistoryGroup(groupKey) {
        if (state.expandedGroupKeys.has(groupKey)) {
            state.expandedGroupKeys.delete(groupKey);
        } else {
            state.expandedGroupKeys.add(groupKey);
        }
        refreshVisibleOrderedKeys();
        render(state.currentPayload);
    }

    function toggleImageSelection(imageKeyValue, options = {}) {
        const next = options.additive ? new Set(state.selectedKeys) : new Set();
        if (options.shiftKey && state.lastSelectedKey) {
            const range = visibleSelectionRange(state.lastSelectedKey, imageKeyValue);
            if (range.length) {
                range.forEach((key) => next.add(key));
            } else {
                toggleSingleSelection(next, imageKeyValue, { additive: options.additive });
            }
        } else {
            toggleSingleSelection(next, imageKeyValue, { additive: options.additive });
        }
        state.selectedKeys = next;
        state.lastSelectedKey = imageKeyValue;
        setSelectionStatus('', '');
        syncCardSelectionStates();
        syncSelectionToolbar();
    }

    function toggleSingleSelection(selectedKeys, imageKeyValue, options = {}) {
        if (options.additive) {
            if (selectedKeys.has(imageKeyValue)) {
                selectedKeys.delete(imageKeyValue);
            } else {
                selectedKeys.add(imageKeyValue);
            }
            return;
        }
        if (state.selectedKeys.size === 1 && state.selectedKeys.has(imageKeyValue)) {
            selectedKeys.delete(imageKeyValue);
        } else {
            selectedKeys.clear();
            selectedKeys.add(imageKeyValue);
        }
    }

    function visibleSelectionRange(fromKey, toKey) {
        const start = state.visibleOrderedKeys.indexOf(fromKey);
        const end = state.visibleOrderedKeys.indexOf(toKey);
        if (start === -1 || end === -1) return [];
        const [from, to] = start < end ? [start, end] : [end, start];
        return state.visibleOrderedKeys.slice(from, to + 1);
    }

    function pruneSelectionToFilteredImages() {
        const allowed = new Set(state.filteredOrderedKeys);
        state.selectedKeys = new Set(
            [...state.selectedKeys].filter((key) => allowed.has(key)),
        );
        if (!allowed.has(state.lastSelectedKey)) {
            state.lastSelectedKey = '';
        }
    }

    function syncCardSelectionStates() {
        document.querySelectorAll('#image-test-grid [data-image-key]').forEach((card) => {
            const selected = state.selectedKeys.has(card.getAttribute('data-image-key') || '');
            card.classList.toggle('is-selected', selected);
            card.setAttribute('aria-checked', selected ? 'true' : 'false');
        });
    }

    function clearSelection() {
        state.selectedKeys = new Set();
        state.lastSelectedKey = '';
        setSelectionStatus('', '');
        syncCardSelectionStates();
        syncSelectionToolbar();
    }

    function syncSelectionToolbar() {
        const toolbar = document.getElementById('image-test-selection-toolbar');
        const summary = document.getElementById('image-test-selection-summary');
        const note = document.getElementById('image-test-selection-note');
        const exportButton = document.getElementById('btn-image-test-export-merged');
        const exportOriginalsButton = document.getElementById('btn-image-test-export-originals');
        if (!toolbar || !summary || !note || !exportButton || !exportOriginalsButton) return;

        const selectedCount = state.selectedKeys.size;
        toolbar.hidden = selectedCount <= 0;
        summary.textContent = `已选 ${selectedCount} 张`;
        note.textContent = state.selectionMessage || DEFAULT_SELECTION_NOTE;
        note.classList.remove('is-success', 'is-warning', 'is-error');
        if (state.selectionTone) {
            note.classList.add(`is-${state.selectionTone}`);
        }
        exportButton.disabled = selectedCount <= 0 || state.exportPending || state.rawExportPending;
        exportOriginalsButton.disabled = selectedCount <= 0 || state.exportPending || state.rawExportPending;
        exportButton.textContent = state.exportPending ? '正在导出...' : '导出合并图';
        exportOriginalsButton.textContent = state.rawExportPending ? '打包中...' : '打包原图';
    }

    function setSelectionStatus(message, tone = '') {
        state.selectionMessage = message;
        state.selectionTone = tone;
    }

    async function exportMergedSelection() {
        const selectedImages = selectedImagesInDisplayOrder();
        if (!selectedImages.length || state.exportPending || state.rawExportPending) {
            return;
        }

        state.exportPending = true;
        setSelectionStatus(`正在合并 ${selectedImages.length} 张图片...`, 'warning');
        syncSelectionToolbar();
        try {
            const blob = await createMergedImageBlob(selectedImages);
            triggerBlobDownload(blob, mergedFileName());
            setSelectionStatus(`已导出 ${selectedImages.length} 张图片的合并图。`, 'success');
        } catch (error) {
            setSelectionStatus(`导出失败：${error?.message || '无法生成合并图。'}`, 'error');
        } finally {
            state.exportPending = false;
            syncSelectionToolbar();
        }
    }

    async function exportOriginalZipSelection() {
        const selectedImages = selectedImagesInDisplayOrder();
        if (!selectedImages.length || state.exportPending || state.rawExportPending) {
            return;
        }

        state.rawExportPending = true;
        setSelectionStatus(`正在打包 ${selectedImages.length} 张原图...`, 'warning');
        syncSelectionToolbar();
        try {
            const entries = await Promise.all(selectedImages.map(async (image) => ({
                name: imageDownloadName(image),
                data: await fetchImageBytes(image),
            })));
            const blob = createZipDataBlob(entries, normalizeZipEntryName);
            triggerBlobDownload(blob, originalsZipFileName());
            setSelectionStatus(`已导出 ${selectedImages.length} 张原图 zip。`, 'success');
        } catch (error) {
            setSelectionStatus(`原图打包失败：${error?.message || '无法读取图片。'}`, 'error');
        } finally {
            state.rawExportPending = false;
            syncSelectionToolbar();
        }
    }

    function selectedImagesInDisplayOrder() {
        return state.filteredOrderedKeys
            .filter((key) => state.selectedKeys.has(key))
            .map((key) => state.imageMap.get(key))
            .filter(Boolean);
    }

    async function createMergedImageBlob(images) {
        const loaded = await Promise.all(images.map(async (image) => ({
            image,
            bitmap: await loadImageElement(image.url),
        })));
        const count = loaded.length;
        const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
        const rows = Math.max(1, Math.ceil(count / columns));
        const naturalCellWidth = Math.max(...loaded.map((item) => item.bitmap.naturalWidth || item.bitmap.width || EXPORT_CELL_MIN));
        const naturalCellHeight = Math.max(...loaded.map((item) => item.bitmap.naturalHeight || item.bitmap.height || EXPORT_CELL_MIN));

        let cellWidth = Math.min(EXPORT_CELL_MAX, Math.max(EXPORT_CELL_MIN, naturalCellWidth));
        let cellHeight = Math.min(EXPORT_CELL_MAX, Math.max(EXPORT_CELL_MIN, naturalCellHeight));
        const gap = 16;
        const padding = 20;
        const labelHeight = 34;
        const labelGap = 10;

        let width = padding * 2 + columns * cellWidth + (columns - 1) * gap;
        let height = padding * 2 + rows * (cellHeight + labelHeight) + (rows - 1) * gap;
        const maxEdge = Math.max(width, height);
        if (maxEdge > EXPORT_MAX_EDGE) {
            const scale = EXPORT_MAX_EDGE / maxEdge;
            cellWidth = Math.max(160, Math.floor(cellWidth * scale));
            cellHeight = Math.max(160, Math.floor(cellHeight * scale));
            width = padding * 2 + columns * cellWidth + (columns - 1) * gap;
            height = padding * 2 + rows * (cellHeight + labelHeight) + (rows - 1) * gap;
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('当前浏览器不支持 Canvas。');
        }

        ctx.fillStyle = EXPORT_BACKGROUND;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.textBaseline = 'top';
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        loaded.forEach(({ image, bitmap }, index) => {
            const column = index % columns;
            const row = Math.floor(index / columns);
            const originX = padding + column * (cellWidth + gap);
            const originY = padding + row * (cellHeight + labelHeight + gap);

            ctx.fillStyle = EXPORT_CELL_BACKGROUND;
            ctx.fillRect(originX, originY, cellWidth, cellHeight);

            const drawWidth = bitmap.naturalWidth || bitmap.width || cellWidth;
            const drawHeight = bitmap.naturalHeight || bitmap.height || cellHeight;
            const scale = Math.min(cellWidth / drawWidth, cellHeight / drawHeight, 1);
            const fittedWidth = Math.max(1, Math.round(drawWidth * scale));
            const fittedHeight = Math.max(1, Math.round(drawHeight * scale));
            const imageX = originX + Math.round((cellWidth - fittedWidth) / 2);
            const imageY = originY + Math.round((cellHeight - fittedHeight) / 2);
            ctx.drawImage(bitmap, imageX, imageY, fittedWidth, fittedHeight);

            ctx.fillStyle = EXPORT_TEXT;
            ctx.font = '600 16px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif';
            ctx.fillText(trimExportLabel(image.name || image.file || `image-${index + 1}`), originX, originY + cellHeight + labelGap);

            ctx.fillStyle = EXPORT_META;
            ctx.font = '12px "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif';
            ctx.fillText(
                trimExportLabel(imageTimestampText(image), 48),
                originX,
                originY + cellHeight + labelGap + 18,
            );
        });

        return await new Promise((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                    return;
                }
                reject(new Error('浏览器未返回导出数据。'));
            }, 'image/png');
        });
    }

    function buildHistoryGroups(images, filterValue) {
        const cutoffMs = cutoffTimestampMs(filterValue);
        const grouped = new Map();
        images.forEach((image) => {
            const timestampMs = imageTimestampMs(image);
            if (cutoffMs != null && timestampMs < cutoffMs) {
                return;
            }
            const key = dateKeyFromTimestamp(timestampMs);
            const list = grouped.get(key) || [];
            list.push(image);
            grouped.set(key, list);
        });

        return [...grouped.entries()]
            .map(([key, items]) => ({
                key,
                dateMs: dateStartMs(key),
                items: [...items].sort((a, b) => imageTimestampMs(b) - imageTimestampMs(a)),
                metaLabel: historyGroupMetaLabel(key),
            }))
            .sort((a, b) => b.dateMs - a.dateMs);
    }

    function syncExpandedGroupKeys(groups) {
        const previousExpanded = new Set(state.expandedGroupKeys);
        const keys = new Set(groups.map((group) => group.key));
        const hadGroupsBefore = state.filteredOrderedKeys.length > 0;
        if (!state.expansionInitialized) {
            state.expandedGroupKeys = new Set(defaultExpandedGroupKeys(groups));
            state.expansionInitialized = true;
            return;
        }
        const nextExpanded = new Set(
            [...state.expandedGroupKeys].filter((key) => keys.has(key)),
        );
        if (hadGroupsBefore) {
            groups.forEach((group) => {
                if (!previousExpanded.has(group.key) && shouldAutoExpandGroup(group)) {
                    nextExpanded.add(group.key);
                }
            });
        }
        groups.forEach((group) => {
            if (freshCountForGroup(group.key) > 0 && shouldAutoExpandFreshGroup(group)) {
                nextExpanded.add(group.key);
            }
        });
        state.expandedGroupKeys = (!nextExpanded.size && groups.length && !hadGroupsBefore)
            ? new Set(defaultExpandedGroupKeys(groups))
            : nextExpanded;
    }

    function syncGroupRenderWindows(groups) {
        const keys = new Set(groups.map((group) => group.key));
        const nextWindows = new Map();
        groups.forEach((group) => {
            const existing = Number(state.renderWindowByGroup.get(group.key) || 0);
            const visibleCount = Math.min(
                group.items.length,
                existing > 0 ? existing : GROUP_INITIAL_RENDER_COUNT,
            );
            nextWindows.set(group.key, visibleCount);
        });
        state.renderWindowByGroup = nextWindows;
        [...state.renderWindowByGroup.keys()].forEach((key) => {
            if (!keys.has(key)) {
                state.renderWindowByGroup.delete(key);
            }
        });
    }

    function defaultExpandedGroupKeys(groups) {
        const cutoffMs = Date.now() - DEFAULT_EXPANDED_DAYS * 24 * 60 * 60 * 1000;
        return groups
            .filter((group) => group.dateMs >= cutoffMs)
            .map((group) => group.key);
    }

    function shouldAutoExpandGroup(group) {
        return group.dateMs >= startOfTodayMs();
    }

    function shouldAutoExpandFreshGroup(group) {
        return group.dateMs >= startOfTodayMs();
    }

    function visibleCountForGroup(groupKey, totalCount) {
        const current = Number(state.renderWindowByGroup.get(groupKey) || GROUP_INITIAL_RENDER_COUNT);
        return Math.max(0, Math.min(totalCount, current));
    }

    function refreshVisibleOrderedKeys() {
        state.visibleOrderedKeys = [...state.groupsByKey.values()].flatMap((group) => {
            if (!state.expandedGroupKeys.has(group.key)) {
                return [];
            }
            const visibleCount = visibleCountForGroup(group.key, group.items.length);
            const windowState = resolveVirtualWindow(group.key, visibleCount);
            const startIndex = windowState.virtualized ? windowState.startIndex : 0;
            const endIndex = windowState.virtualized ? windowState.endIndex : visibleCount;
            return group.items.slice(startIndex, endIndex).map((image) => imageKey(image));
        });
    }

    function createLoadMoreFooter(group, visibleCount) {
        const footer = document.createElement('div');
        footer.className = 'image-test-history-load-more';
        footer.dataset.imageTestLoadMoreFooter = group.key;
        footer.dataset.groupKey = group.key;

        const meta = document.createElement('span');
        meta.className = 'image-test-history-load-more-meta';
        meta.textContent = `已渲染 ${visibleCount} / ${group.items.length} 张`;

        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-small image-test-history-load-more-btn';
        button.textContent = `继续加载 ${Math.min(GROUP_RENDER_INCREMENT, group.items.length - visibleCount)} 张`;
        button.addEventListener('click', () => {
            expandGroupRenderWindow(group.key);
        });

        footer.append(meta, button);
        return footer;
    }

    function installLoadMoreObserver() {
        if (!('IntersectionObserver' in window)) return;
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                const groupKey = String(entry.target?.getAttribute?.('data-group-key') || '').trim();
                if (!groupKey) return;
                observer.unobserve(entry.target);
                expandGroupRenderWindow(groupKey);
            });
        }, {
            root: null,
            rootMargin: GROUP_LOAD_MORE_ROOT_MARGIN,
            threshold: 0,
        });
        document.querySelectorAll('[data-image-test-load-more-footer]').forEach((footer) => {
            observer.observe(footer);
        });
        state.loadMoreObserver = observer;
    }

    function disconnectLoadMoreObserver() {
        state.loadMoreObserver?.disconnect?.();
        state.loadMoreObserver = null;
    }

    function expandGroupRenderWindow(groupKey) {
        const group = state.groupsByKey.get(groupKey);
        if (!group) return;
        const previousCount = visibleCountForGroup(groupKey, group.items.length);
        const nextCount = Math.min(group.items.length, previousCount + GROUP_RENDER_INCREMENT);
        if (nextCount <= previousCount) return;
        state.renderWindowByGroup.set(groupKey, nextCount);
        refreshVisibleOrderedKeys();
        const section = findGroupSection(groupKey);
        const body = section?.querySelector('.image-test-history-group-grid');
        if (body && !body.hidden) {
            renderGroupBody(body, group, nextCount);
        }
        updateLoadMoreFooter(section, group, nextCount);
        scheduleVirtualWindowRefresh();
    }

    function updateLoadMoreFooter(section, group, visibleCount) {
        if (!section) return;
        const existing = section.querySelector('[data-image-test-load-more-footer]');
        if (visibleCount >= group.items.length) {
            existing?.remove();
            return;
        }
        const nextFooter = createLoadMoreFooter(group, visibleCount);
        if (existing) {
            existing.replaceWith(nextFooter);
        } else {
            section.appendChild(nextFooter);
        }
        if (state.loadMoreObserver) {
            state.loadMoreObserver.observe(nextFooter);
        }
    }

    function findGroupSection(groupKey) {
        return [...document.querySelectorAll('#image-test-grid .image-test-history-group[data-group-key]')]
            .find((element) => element.getAttribute('data-group-key') === groupKey);
    }

    function scheduleVirtualWindowRefresh() {
        if (state.virtualRefreshFrame) return;
        state.virtualRefreshFrame = window.requestAnimationFrame(() => {
            state.virtualRefreshFrame = 0;
            refreshVirtualWindows();
        });
    }

    function cancelVirtualWindowRefresh() {
        if (!state.virtualRefreshFrame) return;
        window.cancelAnimationFrame(state.virtualRefreshFrame);
        state.virtualRefreshFrame = 0;
    }

    function refreshVirtualWindows() {
        let changed = false;
        state.groupsByKey.forEach((group, groupKey) => {
            if (!state.expandedGroupKeys.has(groupKey)) {
                state.virtualWindowByGroup.delete(groupKey);
                changed = true;
                return;
            }
            const visibleCount = visibleCountForGroup(groupKey, group.items.length);
            const nextWindow = nextVirtualWindow(groupKey, visibleCount);
            const previous = state.virtualWindowByGroup.get(groupKey);
            if (virtualWindowEquals(previous, nextWindow)) {
                return;
            }
            changed = true;
            state.virtualWindowByGroup.set(groupKey, nextWindow);
            const section = findGroupSection(groupKey);
            const body = section?.querySelector('.image-test-history-group-grid');
            if (body && !body.hidden) {
                renderGroupBody(body, group, visibleCount);
            }
        });
        if (changed) {
            refreshVisibleOrderedKeys();
        }
    }

    function resolveVirtualWindow(groupKey, visibleCount) {
        const existing = state.virtualWindowByGroup.get(groupKey);
        if (existing) {
            return clampVirtualWindow(existing, visibleCount);
        }
        const fallback = defaultVirtualWindow(visibleCount);
        state.virtualWindowByGroup.set(groupKey, fallback);
        return fallback;
    }

    function nextVirtualWindow(groupKey, visibleCount) {
        if (!shouldVirtualizeGroup(visibleCount)) {
            return defaultVirtualWindow(visibleCount);
        }
        const section = findGroupSection(groupKey);
        const body = section?.querySelector('.image-test-history-group-grid');
        const metrics = measureVirtualMetrics(body);
        if (!metrics) {
            return defaultVirtualWindow(visibleCount);
        }
        const { columnCount, rowHeight } = metrics;
        const totalRows = Math.max(1, Math.ceil(visibleCount / columnCount));
        const scrollTop = window.scrollY || window.pageYOffset || 0;
        const viewportHeight = window.innerHeight || document.documentElement.clientHeight || rowHeight;
        const bodyRect = body.getBoundingClientRect();
        const bodyTop = scrollTop + bodyRect.top;
        const viewportTop = Math.max(0, scrollTop - rowHeight * GROUP_VIRTUAL_OVERSCAN_ROWS);
        const viewportBottom = scrollTop + viewportHeight + rowHeight * GROUP_VIRTUAL_OVERSCAN_ROWS;
        const startRow = clampInt(Math.floor((viewportTop - bodyTop) / rowHeight), 0, totalRows - 1);
        const endRow = clampInt(Math.floor((viewportBottom - bodyTop) / rowHeight), 0, totalRows - 1);
        const startIndex = Math.min(visibleCount, startRow * columnCount);
        const endIndex = Math.max(
            startIndex + 1,
            Math.min(visibleCount, (endRow + 1) * columnCount),
        );
        return {
            virtualized: true,
            startIndex,
            endIndex,
            topSpacerPx: startRow * rowHeight,
            bottomSpacerPx: Math.max(0, (totalRows - endRow - 1) * rowHeight),
        };
    }

    function defaultVirtualWindow(visibleCount) {
        const virtualized = shouldVirtualizeGroup(visibleCount);
        return {
            virtualized,
            startIndex: 0,
            endIndex: virtualized
                ? Math.max(0, Math.min(visibleCount, GROUP_INITIAL_RENDER_COUNT))
                : Math.max(0, visibleCount),
            topSpacerPx: 0,
            bottomSpacerPx: 0,
        };
    }

    function clampVirtualWindow(windowState, visibleCount) {
        const startIndex = clampInt(Number(windowState?.startIndex || 0), 0, Math.max(visibleCount - 1, 0));
        const endIndex = Math.max(
            startIndex + (visibleCount > 0 ? 1 : 0),
            clampInt(Number(windowState?.endIndex || visibleCount), 0, visibleCount),
        );
        return {
            virtualized: shouldVirtualizeGroup(visibleCount) && windowState?.virtualized !== false,
            startIndex,
            endIndex: Math.min(endIndex, visibleCount),
            topSpacerPx: Math.max(0, Number(windowState?.topSpacerPx || 0)),
            bottomSpacerPx: Math.max(0, Number(windowState?.bottomSpacerPx || 0)),
        };
    }

    function virtualWindowEquals(previous, next) {
        return Boolean(
            previous
            && next
            && previous.virtualized === next.virtualized
            && previous.startIndex === next.startIndex
            && previous.endIndex === next.endIndex
            && Math.round(previous.topSpacerPx || 0) === Math.round(next.topSpacerPx || 0)
            && Math.round(previous.bottomSpacerPx || 0) === Math.round(next.bottomSpacerPx || 0)
        );
    }

    function shouldVirtualizeGroup(visibleCount) {
        return visibleCount > GROUP_VIRTUALIZE_THRESHOLD;
    }

    function measureVirtualMetrics(body) {
        if (!(body instanceof HTMLElement)) return null;
        const card = body.querySelector('.image-test-history-card');
        if (!(card instanceof HTMLElement)) return null;
        const cardRect = card.getBoundingClientRect();
        if (cardRect.width <= 0 || cardRect.height <= 0 || body.clientWidth <= 0) {
            return null;
        }
        const styles = window.getComputedStyle(body);
        const gap = Number.parseFloat(styles.rowGap || styles.gap || '0') || 0;
        const columnCount = Math.max(1, Math.round((body.clientWidth + gap) / (cardRect.width + gap)));
        return {
            columnCount,
            rowHeight: Math.max(1, cardRect.height + gap),
        };
    }

    function clampInt(value, min, max) {
        return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
    }

    function emptyMessageForCurrentFilter(payload) {
        if (Array.isArray(payload.images) && payload.images.length) {
            return `当前筛选范围“${historyRangeLabel(state.filterValue)}”内没有图片，试试放宽时间范围。`;
        }
        return payload.message || '还没有生图结果。';
    }

    function historyRangeLabel(value) {
        return IMAGE_TEST_HISTORY_RANGE_OPTIONS.find((item) => item.value === value)?.label || '近 7 天';
    }

    function cutoffTimestampMs(filterValue) {
        const days = daysForImageTestHistoryRange(filterValue);
        return typeof days === 'number'
            ? Date.now() - days * 24 * 60 * 60 * 1000
            : null;
    }

    function syncFreshKeys(images) {
        const nextKeys = images.map((image) => imageKey(image));
        const nextKnown = new Set(nextKeys);
        state.freshKeys = state.knownImageKeys.size
            ? new Set(nextKeys.filter((key) => !state.knownImageKeys.has(key)))
            : new Set();
        state.knownImageKeys = nextKnown;
    }

    function syncFreshGroupCounts(groups) {
        state.freshCountsByGroup = new Map(
            groups.map((group) => [
                group.key,
                group.items.reduce((count, image) => (
                    state.freshKeys.has(imageKey(image)) ? count + 1 : count
                ), 0),
            ]),
        );
    }

    function freshCountForGroup(groupKey) {
        return Number(state.freshCountsByGroup.get(groupKey) || 0);
    }

    function clearGrid() {
        const grid = document.getElementById('image-test-grid');
        if (grid) grid.innerHTML = '';
    }

    async function fetchImageBytes(image) {
        const response = await fetch(image.url, { credentials: 'same-origin' });
        if (!response.ok) {
            throw new Error(`读取 ${image.name || image.file || '图片'} 失败`);
        }
        return new Uint8Array(await response.arrayBuffer());
    }

    function mergedFileName() {
        const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+$/, '').replace('T', '-');
        return `anima-image-test-merged-${stamp}.png`;
    }

    function originalsZipFileName() {
        const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+$/, '').replace('T', '-');
        return `anima-image-test-originals-${stamp}.zip`;
    }

    function imageDownloadName(image) {
        return String(image?.name || image?.file || 'image-test-result.png')
            .split('/')
            .filter(Boolean)
            .pop() || 'image-test-result.png';
    }

    function normalizeZipEntryName(name, usedNames = new Set()) {
        const base = String(name || 'image-test-result.png')
            .replace(/[\\/:*?"<>|\r\n\t]+/g, '_')
            .replace(/\s+/g, '_')
            .replace(/^[._]+|[._]+$/g, '') || 'image-test-result.png';
        if (!usedNames.has(base)) {
            return base;
        }
        const dotIndex = base.lastIndexOf('.');
        const stem = dotIndex > 0 ? base.slice(0, dotIndex) : base;
        const ext = dotIndex > 0 ? base.slice(dotIndex) : '';
        let index = 2;
        let candidate = `${stem}_${index}${ext}`;
        while (usedNames.has(candidate)) {
            index += 1;
            candidate = `${stem}_${index}${ext}`;
        }
        return candidate;
    }

    function loadImageElement(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.decoding = 'async';
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('有图片无法读取，无法完成合并导出。'));
            img.src = url;
        });
    }

    function imageKey(image) {
        return String(image?.file || image?.url || image?.name || '').trim();
    }

    function imageTimestampMs(image) {
        const generated = Number((image?.sample || {}).generated_at);
        const modified = Number(image?.mtime);
        if (Number.isFinite(generated) && generated > 0) return generated * 1000;
        if (Number.isFinite(modified) && modified > 0) return modified * 1000;
        return 0;
    }

    function imageTimestampText(image) {
        const sampleText = String((image?.sample || {}).generated_at_text || '').trim();
        if (sampleText) return sampleText;
        const modifiedText = String(image?.mtime_text || '').trim();
        return modifiedText || '时间未知';
    }

    function imageCardMetaText(image, formatBytesFn) {
        const dims = image?.width && image?.height ? `${image.width}x${image.height}` : '尺寸未知';
        const parts = [
            dims,
            image?.sample?.parameters?.sample_steps ? `${image.sample.parameters.sample_steps} steps` : '',
            image?.sample?.sampler || image?.sample?.parameters?.sample_sampler || '',
            formatBytesFn(Number(image?.size_bytes || 0)),
        ].filter(Boolean);
        return parts.join(' · ');
    }

    function dateKeyFromTimestamp(timestampMs) {
        const date = timestampMs > 0 ? new Date(timestampMs) : new Date();
        return [
            date.getFullYear(),
            pad2(date.getMonth() + 1),
            pad2(date.getDate()),
        ].join('-');
    }

    function dateStartMs(dateKey) {
        return new Date(`${dateKey}T00:00:00`).getTime();
    }

    function historyGroupMetaLabel(dateKey) {
        const diffDays = Math.max(0, Math.floor((startOfTodayMs() - dateStartMs(dateKey)) / (24 * 60 * 60 * 1000)));
        if (diffDays === 0) return '今天';
        if (diffDays === 1) return '昨天';
        if (diffDays < 7) return `${diffDays} 天前`;
        if (diffDays < 30) return '近 30 天';
        return '更早记录';
    }

    function startOfTodayMs() {
        const now = new Date();
        return new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    }

    function trimExportLabel(value, maxLength = 34) {
        const text = String(value || '').trim();
        if (text.length <= maxLength) return text;
        return `${text.slice(0, maxLength - 1)}…`;
    }

    function pad2(value) {
        return String(value).padStart(2, '0');
    }

    return {
        currentFilter: () => state.filterValue,
        init,
        render,
        setLoading,
        setEmpty,
    };
}
