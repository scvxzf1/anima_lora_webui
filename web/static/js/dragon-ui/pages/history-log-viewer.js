/* Windowed history-log rendering with measured-height, scroll-stable segments. */

import { appendHighlightedLog } from './log-highlighter.js?v=dragon-ui-20260825v1';

const DEFAULT_LOG_ROW_HEIGHT = 22;
const LOG_SEGMENT_SIZE = 120;
const LOG_OVERSCAN_SEGMENTS = 1;
const LOG_FETCH_PAGE_SIZE = 360;
const MAX_CACHED_LOG_PAGES = 12;

export function bindHistoryLogViewer(root, records = [], options = {}) {
    const viewer = root?.querySelector('[data-history-log-viewer]');
    const status = root?.querySelector('[data-history-log-window-status]');
    if (!viewer) return () => {};

    const rows = normalizeLogRows(records);
    const reportedTotal = Math.max(rows.length, numeric(options.total) ?? rows.length);
    const sourceOffset = Math.max(0, reportedTotal - rows.length);
    if (typeof options.loadRange === 'function' && reportedTotal > rows.length) {
        return bindPagedHistoryLogViewer(root, viewer, status, rows, reportedTotal, {
            ...options,
            offset: numeric(options.offset) ?? sourceOffset,
        });
    }
    if (!rows.length) {
        if (status) status.textContent = '0 行';
        disableLogSearch(root);
        return () => {};
    }

    const searchState = createLogSearchState(rows);
    const controller = createLogWindowController(viewer, status, rows, reportedTotal, sourceOffset, searchState);
    const unbindSearch = bindLogSearch(root, viewer, searchState, controller);
    const panel = viewer.closest('[data-history-detail-panel]');
    const visibilityObserver = typeof MutationObserver === 'function' && panel
        ? new MutationObserver(controller.initializeAtLatest)
        : null;
    visibilityObserver?.observe(panel, { attributes: true, attributeFilter: ['hidden'] });
    const resizeObserver = typeof ResizeObserver === 'function'
        ? new ResizeObserver(controller.handleResize)
        : null;
    resizeObserver?.observe(viewer);

    viewer.addEventListener('scroll', controller.scheduleRender, { passive: true });
    controller.renderWindow(true);
    scheduleFrame(controller.initializeAtLatest);

    return () => {
        viewer.removeEventListener('scroll', controller.scheduleRender);
        visibilityObserver?.disconnect();
        resizeObserver?.disconnect();
        unbindSearch();
        controller.destroy();
    };
}

function bindPagedHistoryLogViewer(root, viewer, status, initialRows, total, options) {
    const store = createPagedLogStore(total, initialRows, options.offset);
    const searchState = { query: '', activeRow: null };
    let refreshSearch = () => {};
    const controller = createPagedLogWindowController(
        viewer,
        status,
        store,
        options.loadRange,
        searchState,
        () => refreshSearch(),
    );
    const searchBinding = typeof options.searchMatch === 'function'
        ? bindPagedServerLogSearch(root, controller, searchState, options.searchMatch)
        : bindPagedLogSearch(root, viewer, store, controller, searchState);
    refreshSearch = searchBinding.refresh;
    const panel = viewer.closest('[data-history-detail-panel]');
    const visibilityObserver = typeof MutationObserver === 'function' && panel
        ? new MutationObserver(controller.initializeAtLatest)
        : null;
    visibilityObserver?.observe(panel, { attributes: true, attributeFilter: ['hidden'] });
    const resizeObserver = typeof ResizeObserver === 'function'
        ? new ResizeObserver(controller.handleResize)
        : null;
    resizeObserver?.observe(viewer);

    viewer.addEventListener('scroll', controller.scheduleRender, { passive: true });
    scheduleFrame(controller.initializeAtLatest);

    return () => {
        viewer.removeEventListener('scroll', controller.scheduleRender);
        visibilityObserver?.disconnect();
        resizeObserver?.disconnect();
        searchBinding.cleanup();
        controller.destroy();
    };
}

function createPagedLogStore(total, initialRows, initialOffset) {
    const rows = new Map();
    const pages = new Map();

    const storePage = (key, offset, incomingRows) => {
        const indices = [];
        incomingRows.forEach((row, localIndex) => {
            const index = offset + localIndex;
            if (index < 0 || index >= total) return;
            rows.set(index, { row, key });
            indices.push(index);
        });
        pages.delete(key);
        pages.set(key, indices);
        while (pages.size > MAX_CACHED_LOG_PAGES) {
            const oldestKey = pages.keys().next().value;
            const oldestIndices = pages.get(oldestKey) || [];
            pages.delete(oldestKey);
            oldestIndices.forEach((index) => {
                if (rows.get(index)?.key === oldestKey) rows.delete(index);
            });
        }
    };

    storePage(`initial:${initialOffset}`, initialOffset, initialRows);
    return {
        total,
        get: (index) => rows.get(index)?.row || null,
        hasRange: (start, end) => {
            for (let index = start; index < end; index += 1) {
                if (!rows.has(index)) return false;
            }
            return true;
        },
        storePage,
        loadedEntries: () => [...rows.entries()].map(([index, value]) => [index, value.row]),
        loadedCount: () => rows.size,
    };
}

function createPagedLogWindowController(viewer, status, store, loadRange, searchState, onRowsLoaded) {
    let frame = null;
    let initialized = false;
    let destroyed = false;
    let renderedStart = -1;
    let renderedEnd = -1;
    let rowHeight = resolveLogRowHeight(viewer);
    const pendingPages = new Set();

    const ensureRange = (start, end) => {
        const firstPage = Math.floor(start / LOG_FETCH_PAGE_SIZE) * LOG_FETCH_PAGE_SIZE;
        for (let pageStart = firstPage; pageStart < end; pageStart += LOG_FETCH_PAGE_SIZE) {
            const pageEnd = Math.min(store.total, pageStart + LOG_FETCH_PAGE_SIZE);
            if (store.hasRange(pageStart, pageEnd) || pendingPages.has(pageStart)) continue;
            pendingPages.add(pageStart);
            Promise.resolve(loadRange(pageStart, pageEnd - pageStart)).then((payload) => {
                if (destroyed) return;
                const offset = numeric(payload?.offset) ?? pageStart;
                store.storePage(pageStart, offset, normalizeLogRows(payload?.logs));
                renderedStart = -1;
                renderedEnd = -1;
                onRowsLoaded();
                renderWindow(true);
            }).catch(() => {
                viewer.dataset.historyLogLoadError = 'true';
            }).finally(() => {
                pendingPages.delete(pageStart);
            });
        }
    };

    const renderWindow = (force = false) => {
        if (destroyed) return;
        const range = logWindowRange(viewer, store.total, rowHeight);
        updateLogStatus(status, range.firstVisible + 1, range.lastVisible, store.total);
        ensureRange(range.start, range.end);
        const { start, end } = range;
        if (!force && start === renderedStart && end === renderedEnd) return;
        renderedStart = start;
        renderedEnd = end;
        replacePagedLogWindow(viewer, store, range, rowHeight, searchState);
    };

    const scheduleRender = () => {
        if (frame != null || destroyed) return;
        frame = -1;
        const scheduledFrame = scheduleFrame(() => {
            frame = null;
            renderWindow();
        });
        if (frame != null) frame = scheduledFrame;
    };

    const initializeAtLatest = () => {
        if (destroyed || initialized || viewer.clientHeight <= 0) return;
        viewer.replaceChildren(logSpacer(store.total * rowHeight, 'bottom'));
        viewer.scrollTop = Math.max(0, store.total * rowHeight - viewer.clientHeight);
        initialized = true;
        renderWindow(true);
    };

    const jumpToRow = (rowIndex) => {
        if (destroyed || rowIndex == null || viewer.clientHeight <= 0) return;
        const centeredTop = rowIndex * rowHeight - (viewer.clientHeight - rowHeight) / 2;
        viewer.scrollTop = clamp(centeredTop, 0, Math.max(0, store.total * rowHeight - viewer.clientHeight));
        initialized = true;
        renderWindow(true);
    };

    return {
        renderWindow,
        jumpToRow,
        scheduleRender,
        initializeAtLatest,
        rowIndexAtScroll: () => Math.floor(viewer.scrollTop / rowHeight),
        handleResize: () => {
            const nextRowHeight = resolveLogRowHeight(viewer);
            if (initialized && nextRowHeight !== rowHeight) {
                const anchorRow = viewer.scrollTop / rowHeight;
                rowHeight = nextRowHeight;
                viewer.scrollTop = anchorRow * rowHeight;
            } else {
                rowHeight = nextRowHeight;
            }
            if (!initialized) initializeAtLatest();
            else renderWindow(true);
        },
        destroy: () => {
            destroyed = true;
            cancelFrame(frame);
        },
    };
}

function replacePagedLogWindow(viewer, store, range, rowHeight, searchState) {
    const fragment = document.createDocumentFragment();
    fragment.appendChild(logSpacer(range.start * rowHeight, 'top'));
    for (let index = range.start; index < range.end; index += 1) {
        const row = store.get(index);
        fragment.appendChild(row
            ? renderLogRow(row, index, index, store.total, searchState)
            : renderPendingLogRow(index, store.total));
    }
    fragment.appendChild(logSpacer((store.total - range.end) * rowHeight, 'bottom'));
    viewer.replaceChildren(fragment);
    viewer.dataset.historyLogRenderStart = String(range.start);
    viewer.dataset.historyLogRenderEnd = String(range.end);
}

function renderPendingLogRow(index, total) {
    const line = document.createElement('div');
    line.className = 'dragon-log-line dragon-history-log-line dragon-history-log-line-pending';
    line.dataset.logIndex = String(index);
    line.setAttribute('role', 'listitem');
    line.setAttribute('aria-posinset', String(index + 1));
    line.setAttribute('aria-setsize', String(total));
    line.setAttribute('aria-busy', 'true');
    line.setAttribute('aria-label', `正在加载第 ${index + 1} 行`);
    return line;
}

function bindPagedLogSearch(root, viewer, store, controller, searchState) {
    const input = root.querySelector('[data-history-log-search]');
    const previous = root.querySelector('[data-history-log-search-previous]');
    const next = root.querySelector('[data-history-log-search-next]');
    const status = root.querySelector('[data-history-log-search-status]');
    let matches = [];
    let cursor = -1;

    const refresh = () => {
        const query = String(input?.value || '').trim().toLocaleLowerCase();
        searchState.query = query;
        matches = query
            ? store.loadedEntries()
                .filter(([_index, row]) => row.searchText.includes(query))
                .map(([index]) => index)
                .sort((a, b) => a - b)
            : [];
        if (!matches.length) cursor = -1;
        else cursor = clamp(cursor, 0, matches.length - 1);
        searchState.activeRow = matches[cursor] ?? null;
        const loadedLabel = store.loadedCount() < store.total ? ` · 已加载 ${store.loadedCount()} / ${store.total} 行` : '';
        if (status) status.textContent = matches.length ? `${cursor + 1} / ${matches.length}${loadedLabel}` : `0 个匹配${loadedLabel}`;
        if (previous) previous.disabled = !matches.length;
        if (next) next.disabled = !matches.length;
    };
    const update = () => {
        cursor = 0;
        refresh();
        if (matches.length) controller.jumpToRow(matches[cursor]);
        else controller.renderWindow(true);
    };
    const move = (delta) => {
        if (!matches.length) return;
        cursor = (cursor + delta + matches.length) % matches.length;
        refresh();
        controller.jumpToRow(matches[cursor]);
    };
    const handleKeydown = (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            move(event.shiftKey ? -1 : 1);
        } else if (event.key === 'Escape' && input.value) {
            input.value = '';
            update();
        }
    };
    const showPrevious = () => move(-1);
    const showNext = () => move(1);
    input?.addEventListener('input', update);
    input?.addEventListener('keydown', handleKeydown);
    previous?.addEventListener('click', showPrevious);
    next?.addEventListener('click', showNext);
    refresh();
    return {
        refresh,
        cleanup: () => {
            input?.removeEventListener('input', update);
            input?.removeEventListener('keydown', handleKeydown);
            previous?.removeEventListener('click', showPrevious);
            next?.removeEventListener('click', showNext);
        },
    };
}

function bindPagedServerLogSearch(root, controller, searchState, searchMatch) {
    const input = root.querySelector('[data-history-log-search]');
    const previous = root.querySelector('[data-history-log-search-previous]');
    const next = root.querySelector('[data-history-log-search-next]');
    const status = root.querySelector('[data-history-log-search-status]');
    let activeIndex = null;
    let ordinal = 0;
    let matchesTotal = 0;
    let timer = null;
    let requestSequence = 0;

    const updateControls = (text = '') => {
        if (status) status.textContent = text || (matchesTotal ? `${ordinal} / ${matchesTotal}` : '0 个匹配');
        if (previous) previous.disabled = !matchesTotal;
        if (next) next.disabled = !matchesTotal;
    };
    const run = async (direction, cursor) => {
        const query = String(input?.value || '').trim();
        if (!query) return;
        const sequence = ++requestSequence;
        updateControls('搜索中…');
        try {
            const payload = await searchMatch(query, cursor, direction);
            if (sequence !== requestSequence) return;
            matchesTotal = Number(payload?.matches_total) || 0;
            ordinal = Number(payload?.match_ordinal) || 0;
            activeIndex = payload?.match_index == null ? null : numeric(payload.match_index);
            searchState.query = query.toLocaleLowerCase();
            searchState.activeRow = activeIndex;
            updateControls();
            if (activeIndex != null) controller.jumpToRow(activeIndex);
            else controller.renderWindow(true);
        } catch (_error) {
            if (sequence === requestSequence) updateControls('搜索失败');
        }
    };
    const update = () => {
        clearTimeout(timer);
        requestSequence += 1;
        activeIndex = null;
        ordinal = 0;
        matchesTotal = 0;
        searchState.query = String(input?.value || '').trim().toLocaleLowerCase();
        searchState.activeRow = null;
        updateControls();
        controller.renderWindow(true);
        if (searchState.query) timer = setTimeout(() => run('forward', controller.rowIndexAtScroll()), 180);
    };
    const move = (delta) => {
        clearTimeout(timer);
        const direction = delta < 0 ? 'backward' : 'forward';
        const cursor = activeIndex == null ? controller.rowIndexAtScroll() : activeIndex + delta;
        run(direction, cursor);
    };
    const handleKeydown = (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            move(event.shiftKey ? -1 : 1);
        } else if (event.key === 'Escape' && input.value) {
            input.value = '';
            update();
        }
    };
    const showPrevious = () => move(-1);
    const showNext = () => move(1);
    input?.addEventListener('input', update);
    input?.addEventListener('keydown', handleKeydown);
    previous?.addEventListener('click', showPrevious);
    next?.addEventListener('click', showNext);
    updateControls();
    return {
        refresh: updateControls,
        cleanup: () => {
            requestSequence += 1;
            clearTimeout(timer);
            input?.removeEventListener('input', update);
            input?.removeEventListener('keydown', handleKeydown);
            previous?.removeEventListener('click', showPrevious);
            next?.removeEventListener('click', showNext);
        },
    };
}

function createLogWindowController(viewer, status, rows, reportedTotal, sourceOffset, searchState) {
    let frame = null;
    let initialized = false;
    let destroyed = false;
    let renderedStart = -1;
    let renderedEnd = -1;
    let rowHeight = resolveLogRowHeight(viewer);

    const renderWindow = (force = false) => {
        if (destroyed) return;
        const range = logWindowRange(viewer, rows.length, rowHeight);
        updateLogStatus(status, sourceOffset + range.firstVisible + 1, sourceOffset + range.lastVisible, reportedTotal);
        const { start, end } = range;
        if (!force && start === renderedStart && end === renderedEnd) return;
        renderedStart = start;
        renderedEnd = end;
        replaceLogWindow(viewer, rows, range, sourceOffset, reportedTotal, searchState.snapshot(), rowHeight);
    };

    const scheduleRender = () => {
        if (frame != null || destroyed) return;
        frame = -1;
        const scheduledFrame = scheduleFrame(() => {
            frame = null;
            renderWindow();
        });
        if (frame != null) frame = scheduledFrame;
    };

    const initializeAtLatest = () => {
        if (destroyed || initialized || viewer.clientHeight <= 0) return;
        renderWindow(true);
        viewer.scrollTop = Math.max(0, rows.length * rowHeight - viewer.clientHeight);
        initialized = true;
        renderWindow(true);
    };

    const jumpToRow = (rowIndex) => {
        if (destroyed || rowIndex == null || viewer.clientHeight <= 0) return;
        const centeredTop = rowIndex * rowHeight - (viewer.clientHeight - rowHeight) / 2;
        const maximumTop = Math.max(0, rows.length * rowHeight - viewer.clientHeight);
        viewer.scrollTop = clamp(centeredTop, 0, maximumTop);
        initialized = true;
        renderWindow(true);
    };

    return {
        renderWindow,
        jumpToRow,
        scheduleRender,
        initializeAtLatest,
        rowIndexAtScroll: () => Math.floor(viewer.scrollTop / rowHeight),
        handleResize: () => {
            const nextRowHeight = resolveLogRowHeight(viewer);
            if (initialized && nextRowHeight !== rowHeight) {
                const anchorRow = viewer.scrollTop / rowHeight;
                rowHeight = nextRowHeight;
                viewer.scrollTop = anchorRow * rowHeight;
            } else {
                rowHeight = nextRowHeight;
            }
            if (!initialized) initializeAtLatest();
            else renderWindow(true);
        },
        destroy: () => {
            destroyed = true;
            cancelFrame(frame);
        },
    };
}

function logWindowRange(viewer, rowCount, rowHeight) {
    const viewportHeight = Math.max(viewer.clientHeight || 0, rowHeight * 10);
    const firstVisible = clamp(Math.floor(viewer.scrollTop / rowHeight), 0, rowCount - 1);
    const lastVisible = clamp(
        Math.ceil((viewer.scrollTop + viewportHeight) / rowHeight),
        firstVisible + 1,
        rowCount,
    );
    const firstSegment = Math.floor(firstVisible / LOG_SEGMENT_SIZE);
    const totalSegments = Math.ceil(rowCount / LOG_SEGMENT_SIZE);
    const startSegment = Math.max(0, firstSegment - LOG_OVERSCAN_SEGMENTS);
    const endSegment = Math.min(totalSegments, startSegment + 3);
    return {
        firstVisible,
        lastVisible,
        start: startSegment * LOG_SEGMENT_SIZE,
        end: Math.min(rowCount, endSegment * LOG_SEGMENT_SIZE),
    };
}

function replaceLogWindow(viewer, rows, range, sourceOffset, reportedTotal, search, rowHeight) {
    const fragment = document.createDocumentFragment();
    fragment.appendChild(logSpacer(range.start * rowHeight, 'top'));
    for (let index = range.start; index < range.end; index += 1) {
        fragment.appendChild(renderLogRow(rows[index], index, sourceOffset + index, reportedTotal, search));
    }
    fragment.appendChild(logSpacer((rows.length - range.end) * rowHeight, 'bottom'));
    viewer.replaceChildren(fragment);
    viewer.dataset.historyLogRenderStart = String(range.start);
    viewer.dataset.historyLogRenderEnd = String(range.end);
}

function normalizeLogRows(records) {
    return (Array.isArray(records) ? records : []).map((record, index) => {
        const raw = typeof record === 'string'
            ? record
            : record?.line ?? record?.message ?? record?.text ?? JSON.stringify(record ?? '');
        const text = stripAnsi(raw).replace(/\r?\n/g, ' ↩ ');
        return { text, searchText: text.toLocaleLowerCase(), level: logLevel(record?.level, text), index };
    });
}

function renderLogRow(row, rowIndex, index, total, search) {
    const line = document.createElement('div');
    line.className = 'dragon-log-line dragon-history-log-line';
    line.dataset.logIndex = String(index);
    if (row.level) line.dataset.level = row.level;
    line.setAttribute('role', 'listitem');
    line.setAttribute('aria-posinset', String(index + 1));
    line.setAttribute('aria-setsize', String(total));
    const matches = search.query && row.searchText.includes(search.query);
    if (matches) appendHighlightedText(line, row.text, row.searchText, search.query);
    else appendHighlightedLog(line, row.text);
    if (rowIndex === search.activeRow) {
        line.dataset.searchActive = 'true';
        line.setAttribute('aria-current', 'true');
    }
    return line;
}

function appendHighlightedText(line, text, searchText, query) {
    let cursor = 0;
    let match = searchText.indexOf(query);
    while (match >= 0) {
        if (match > cursor) line.appendChild(document.createTextNode(text.slice(cursor, match)));
        const mark = document.createElement('mark');
        mark.textContent = text.slice(match, match + query.length);
        line.appendChild(mark);
        cursor = match + query.length;
        match = searchText.indexOf(query, cursor);
    }
    if (cursor < text.length) line.appendChild(document.createTextNode(text.slice(cursor)));
}

function createLogSearchState(rows) {
    let query = '';
    let matches = [];
    let cursor = -1;
    const snapshot = () => ({ query, matches, cursor, activeRow: matches[cursor] ?? null });
    return {
        snapshot,
        update(value, startIndex) {
            query = String(value || '').trim().toLocaleLowerCase();
            matches = query ? rows.flatMap((row, index) => row.searchText.includes(query) ? [index] : []) : [];
            cursor = matches.findIndex((index) => index >= startIndex);
            if (matches.length && cursor < 0) cursor = 0;
            return snapshot();
        },
        move(delta) {
            if (matches.length) cursor = (cursor + delta + matches.length) % matches.length;
            return snapshot();
        },
    };
}

function bindLogSearch(root, viewer, state, controller) {
    const input = root.querySelector('[data-history-log-search]');
    const previous = root.querySelector('[data-history-log-search-previous]');
    const next = root.querySelector('[data-history-log-search-next]');
    const status = root.querySelector('[data-history-log-search-status]');
    if (!input) return () => {};

    const show = (snapshot, jump = true) => {
        updateLogSearchControls(status, previous, next, snapshot);
        if (jump && snapshot.activeRow != null) controller.jumpToRow(snapshot.activeRow);
        else controller.renderWindow(true);
    };
    const update = () => show(state.update(input.value, controller.rowIndexAtScroll()));
    const move = (delta) => show(state.move(delta));
    const handleKeydown = (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            move(event.shiftKey ? -1 : 1);
        } else if (event.key === 'Escape' && input.value) {
            input.value = '';
            show(state.update('', 0), false);
        }
    };
    const showPrevious = () => move(-1);
    const showNext = () => move(1);
    input.addEventListener('input', update);
    input.addEventListener('keydown', handleKeydown);
    previous?.addEventListener('click', showPrevious);
    next?.addEventListener('click', showNext);
    updateLogSearchControls(status, previous, next, state.snapshot());
    return () => {
        input.removeEventListener('input', update);
        input.removeEventListener('keydown', handleKeydown);
        previous?.removeEventListener('click', showPrevious);
        next?.removeEventListener('click', showNext);
    };
}

function updateLogSearchControls(status, previous, next, state) {
    const hasMatches = state.matches.length > 0;
    if (status) status.textContent = hasMatches ? `${state.cursor + 1} / ${state.matches.length}` : '0 个匹配';
    if (previous) previous.disabled = !hasMatches;
    if (next) next.disabled = !hasMatches;
}

function disableLogSearch(root) {
    const input = root.querySelector('[data-history-log-search]');
    if (input) input.disabled = true;
    root.querySelectorAll('[data-history-log-search-previous], [data-history-log-search-next]')
        .forEach((button) => { button.disabled = true; });
}

function logSpacer(height, position) {
    const spacer = document.createElement('div');
    spacer.className = 'dragon-history-log-spacer';
    spacer.dataset.logSpacer = position;
    spacer.style.height = `${Math.max(0, height)}px`;
    spacer.setAttribute('aria-hidden', 'true');
    return spacer;
}

function updateLogStatus(element, start, end, total) {
    if (element) element.textContent = `第 ${start}–${end} / ${total} 行`;
}

function logLevel(value, text) {
    const explicit = String(value || '').trim().toLowerCase();
    if (['error', 'warning', 'info'].includes(explicit)) return explicit;
    if (/\b(error|exception|traceback|fatal)\b/i.test(text)) return 'error';
    if (/\b(warn|warning)\b/i.test(text)) return 'warning';
    return '';
}

function stripAnsi(value) {
    return String(value ?? '').replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
}

function resolveLogRowHeight(viewer) {
    const computed = typeof window.getComputedStyle === 'function' ? window.getComputedStyle(viewer) : null;
    const lineHeight = Number.parseFloat(computed?.lineHeight);
    const configured = Number.parseFloat(computed?.getPropertyValue('--dragon-history-log-row-height'));
    if (Number.isFinite(lineHeight) && lineHeight > 0) return lineHeight;
    if (Number.isFinite(configured) && configured > 0) return configured;
    return DEFAULT_LOG_ROW_HEIGHT;
}

function scheduleFrame(callback) {
    if (typeof requestAnimationFrame === 'function') return requestAnimationFrame(callback);
    return setTimeout(callback, 0);
}

function cancelFrame(frame) {
    if (frame == null) return;
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(frame);
    else clearTimeout(frame);
}

function numeric(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}
