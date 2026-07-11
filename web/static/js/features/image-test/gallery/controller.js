/** Gallery controller wiring for image-test feature. */
import { createGalleryExport } from './export.js?v=module-bootstrap-20260711-ir2';
import { createLoadMoreApi, findGroupSection } from './load-more.js?v=module-bootstrap-20260711-ir2';
import { createVirtualWindowApi } from './virtual-window.js?v=module-bootstrap-20260711-ir2';
import { createSelectionApi } from './selection.js?v=module-bootstrap-20260711-ir2';
import { createCardsApi } from './cards.js?v=module-bootstrap-20260711-ir2';
import { createHistoryGroupsApi } from './history-groups.js?v=module-bootstrap-20260711-ir2';
import { imageKey } from './image-meta.js?v=module-bootstrap-20260711-ir2';
import { normalizeImageTestHistoryRange } from '../state.js?v=module-bootstrap-20260711-ir2';
import { DEFAULT_FILTER_VALUE } from './constants.js?v=module-bootstrap-20260711-ir2';

export function createInitialGalleryState(initialFilterValue = DEFAULT_FILTER_VALUE) {
    return {
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
        deletePending: false,
        loadMoreObserver: null,
        virtualRefreshFrame: 0,
        selectionMessage: '',
        selectionTone: '',
    };
}

export function createGalleryController({
    state,
    formatBytes,
    openPreviewDialog,
    requestImageDelete,
    requestHistoryReload,
    render,
}) {
    let exportApi;
    let virtualApi;
    let loadMoreApi;
    let selectionApi;
    let cardsApi;
    let historyGroupsApi;

    function ensureGalleryApis() {
        if (exportApi) return;

        exportApi = createGalleryExport({
            state,
            setSelectionStatus: (...args) => selectionApi.setSelectionStatus(...args),
            syncSelectionToolbar: (...args) => selectionApi.syncSelectionToolbar(...args),
            formatBytes,
        });

        selectionApi = createSelectionApi({
            state,
            requestImageDelete,
            selectedImagesInDisplayOrder: () => exportApi.selectedImagesInDisplayOrder(),
        });

        historyGroupsApi = createHistoryGroupsApi({
            state,
            resolveVirtualWindow: (...args) => virtualApi.resolveVirtualWindow(...args),
        });

        cardsApi = createCardsApi({
            state,
            formatBytes,
            openPreviewDialog,
            toggleImageSelection: (...args) => selectionApi.toggleImageSelection(...args),
            resolveVirtualWindow: (...args) => virtualApi.resolveVirtualWindow(...args),
            createLoadMoreFooter: (...args) => loadMoreApi.createLoadMoreFooter(...args),
            freshCountForGroup: (...args) => historyGroupsApi.freshCountForGroup(...args),
            visibleCountForGroup: (...args) => historyGroupsApi.visibleCountForGroup(...args),
            refreshVisibleOrderedKeys: (...args) => historyGroupsApi.refreshVisibleOrderedKeys(...args),
            requestRerender: () => render(state.currentPayload),
        });

        virtualApi = createVirtualWindowApi({
            state,
            findGroupSection,
            visibleCountForGroup: (...args) => historyGroupsApi.visibleCountForGroup(...args),
            renderGroupBody: (...args) => cardsApi.renderGroupBody(...args),
            refreshVisibleOrderedKeys: (...args) => historyGroupsApi.refreshVisibleOrderedKeys(...args),
        });

        loadMoreApi = createLoadMoreApi({
            state,
            renderGroupBody: (...args) => cardsApi.renderGroupBody(...args),
            refreshVisibleOrderedKeys: (...args) => historyGroupsApi.refreshVisibleOrderedKeys(...args),
            scheduleVirtualWindowRefresh: (...args) => virtualApi.scheduleVirtualWindowRefresh(...args),
            findGroupSectionFn: findGroupSection,
        });
    }

    function apis() {
        ensureGalleryApis();
        return {
            exportApi,
            virtualApi,
            loadMoreApi,
            selectionApi,
            cardsApi,
            historyGroupsApi,
        };
    }

    function scheduleVirtualWindowRefresh() {
        return apis().virtualApi.scheduleVirtualWindowRefresh();
    }

    function cancelVirtualWindowRefresh() {
        return apis().virtualApi.cancelVirtualWindowRefresh();
    }

    function installLoadMoreObserver() {
        return apis().loadMoreApi.installLoadMoreObserver();
    }

    function disconnectLoadMoreObserver() {
        return apis().loadMoreApi.disconnectLoadMoreObserver();
    }

    async function exportMergedSelection() {
        return apis().exportApi.exportMergedSelection();
    }

    async function exportOriginalZipSelection() {
        return apis().exportApi.exportOriginalZipSelection();
    }

    function clearGrid() {
        const grid = document.getElementById('image-test-grid');
        if (grid) grid.innerHTML = '';
    }

    function syncFilterButtons() {
        document.querySelectorAll('#image-test-history-filter [data-range]').forEach((button) => {
            const active = button.getAttribute('data-range') === state.filterValue;
            button.classList.toggle('is-active', active);
            button.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
    }

    function renderHistoryGroups(payload, groups) {
        const { cardsApi, historyGroupsApi } = apis();
        const grid = document.getElementById('image-test-grid');
        const empty = document.getElementById('image-test-empty');
        if (!grid || !empty) return;
        disconnectLoadMoreObserver();
        grid.innerHTML = '';

        if (!groups.length) {
            empty.hidden = false;
            empty.textContent = historyGroupsApi.emptyMessageForCurrentFilter(payload);
            return;
        }

        empty.hidden = true;
        groups.forEach((group) => {
            grid.appendChild(cardsApi.createHistoryGroup(group));
        });
        installLoadMoreObserver();
        scheduleVirtualWindowRefresh();
    }

    function handleFilterClick(event) {
        const { selectionApi } = apis();
        const button = event.target.closest('[data-range]');
        if (!(button instanceof HTMLButtonElement)) return;
        const nextValue = normalizeImageTestHistoryRange(button.dataset.range, DEFAULT_FILTER_VALUE);
        if (nextValue === state.filterValue) return;
        state.filterValue = nextValue;
        state.expansionInitialized = false;
        selectionApi.clearSelection();
        render(state.currentPayload);
        requestHistoryReload?.(nextValue);
    }

    function bindToolbarEvents() {
        const { selectionApi } = apis();
        document.getElementById('image-test-history-filter')?.addEventListener('click', handleFilterClick);
        document.getElementById('btn-image-test-clear-selection')?.addEventListener('click', () => {
            selectionApi.clearSelection();
        });
        document.getElementById('btn-image-test-delete-selected')?.addEventListener('click', () => {
            void selectionApi.deleteSelectedImages();
        });
        document.getElementById('btn-image-test-export-merged')?.addEventListener('click', () => {
            void exportMergedSelection();
        });
        document.getElementById('btn-image-test-export-originals')?.addEventListener('click', () => {
            void exportOriginalZipSelection();
        });
        window.addEventListener('scroll', scheduleVirtualWindowRefresh, { passive: true });
        window.addEventListener('resize', scheduleVirtualWindowRefresh);
    }

    function renderPayload(payload = {}) {
        const { historyGroupsApi, selectionApi } = apis();
        const normalizedPayload = payload && typeof payload === 'object' ? payload : {};
        const images = Array.isArray(normalizedPayload.images) ? normalizedPayload.images : [];
        state.currentPayload = normalizedPayload;
        historyGroupsApi.syncFreshKeys(images);
        state.imageMap = new Map(images.map((image) => [imageKey(image), image]));

        const groups = historyGroupsApi.buildHistoryGroups(images, state.filterValue);
        state.groupsByKey = new Map(groups.map((group) => [group.key, group]));
        historyGroupsApi.syncFreshGroupCounts(groups);
        historyGroupsApi.syncExpandedGroupKeys(groups);
        historyGroupsApi.syncGroupRenderWindows(groups);
        state.filteredOrderedKeys = groups.flatMap((group) => group.items.map((image) => imageKey(image)));
        historyGroupsApi.refreshVisibleOrderedKeys();
        selectionApi.pruneSelectionToFilteredImages();
        historyGroupsApi.renderHistoryHeader(normalizedPayload, groups);
        renderHistoryGroups(normalizedPayload, groups);
        syncFilterButtons();
        selectionApi.syncSelectionToolbar();
    }

    function setLoading() {
        const { selectionApi } = apis();
        disconnectLoadMoreObserver();
        cancelVirtualWindowRefresh();
        state.groupsByKey = new Map();
        state.virtualWindowByGroup = new Map();
        clearGrid();
        selectionApi.clearSelection();
        const count = document.getElementById('image-test-count');
        if (count) count.textContent = '读取中';
        const empty = document.getElementById('image-test-empty');
        if (empty) {
            empty.hidden = false;
            empty.textContent = '正在读取 output/tests 中的结果图...';
        }
    }

    function setEmpty(message) {
        const { selectionApi } = apis();
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
        selectionApi.clearSelection();
        clearGrid();
        const empty = document.getElementById('image-test-empty');
        if (empty) {
            empty.hidden = false;
            empty.textContent = message;
        }
        const count = document.getElementById('image-test-count');
        if (count) count.textContent = '0 张';
    }

    function init() {
        if (state.initialized) return;
        state.initialized = true;
        bindToolbarEvents();
        syncFilterButtons();
        apis().selectionApi.syncSelectionToolbar();
    }

    return {
        ensureGalleryApis,
        apis,
        init,
        renderPayload,
        setLoading,
        setEmpty,
        // Keep names referenced by source tests / comments in gallery.js.
        scheduleVirtualWindowRefresh,
        exportMergedSelection,
        exportOriginalZipSelection,
    };
}
