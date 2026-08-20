/* Windowed history-log rendering with fixed-height, scroll-stable segments. */

const LOG_ROW_HEIGHT = 22;
const LOG_SEGMENT_SIZE = 120;
const LOG_OVERSCAN_SEGMENTS = 1;

export function bindHistoryLogViewer(root, records = [], options = {}) {
    const viewer = root?.querySelector('[data-history-log-viewer]');
    const status = root?.querySelector('[data-history-log-window-status]');
    if (!viewer) return () => {};

    const rows = normalizeLogRows(records);
    const reportedTotal = Math.max(rows.length, numeric(options.total) ?? rows.length);
    const sourceOffset = Math.max(0, reportedTotal - rows.length);
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

function createLogWindowController(viewer, status, rows, reportedTotal, sourceOffset, searchState) {
    let frame = null;
    let initialized = false;
    let destroyed = false;
    let renderedStart = -1;
    let renderedEnd = -1;

    const renderWindow = (force = false) => {
        if (destroyed) return;
        const range = logWindowRange(viewer, rows.length);
        updateLogStatus(status, sourceOffset + range.firstVisible + 1, sourceOffset + range.lastVisible, reportedTotal);
        const { start, end } = range;
        if (!force && start === renderedStart && end === renderedEnd) return;
        renderedStart = start;
        renderedEnd = end;
        replaceLogWindow(viewer, rows, range, sourceOffset, reportedTotal, searchState.snapshot());
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
        viewer.scrollTop = Math.max(0, rows.length * LOG_ROW_HEIGHT - viewer.clientHeight);
        initialized = true;
        renderWindow(true);
    };

    const jumpToRow = (rowIndex) => {
        if (destroyed || rowIndex == null || viewer.clientHeight <= 0) return;
        const centeredTop = rowIndex * LOG_ROW_HEIGHT - (viewer.clientHeight - LOG_ROW_HEIGHT) / 2;
        const maximumTop = Math.max(0, rows.length * LOG_ROW_HEIGHT - viewer.clientHeight);
        viewer.scrollTop = clamp(centeredTop, 0, maximumTop);
        initialized = true;
        renderWindow(true);
    };

    return {
        renderWindow,
        jumpToRow,
        scheduleRender,
        initializeAtLatest,
        handleResize: () => {
            if (!initialized) initializeAtLatest();
            else renderWindow(true);
        },
        destroy: () => {
            destroyed = true;
            cancelFrame(frame);
        },
    };
}

function logWindowRange(viewer, rowCount) {
    const viewportHeight = Math.max(viewer.clientHeight || 0, LOG_ROW_HEIGHT * 10);
    const firstVisible = clamp(Math.floor(viewer.scrollTop / LOG_ROW_HEIGHT), 0, rowCount - 1);
    const lastVisible = clamp(
        Math.ceil((viewer.scrollTop + viewportHeight) / LOG_ROW_HEIGHT),
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

function replaceLogWindow(viewer, rows, range, sourceOffset, reportedTotal, search) {
    const fragment = document.createDocumentFragment();
    fragment.appendChild(logSpacer(range.start * LOG_ROW_HEIGHT, 'top'));
    for (let index = range.start; index < range.end; index += 1) {
        fragment.appendChild(renderLogRow(rows[index], index, sourceOffset + index, reportedTotal, search));
    }
    fragment.appendChild(logSpacer((rows.length - range.end) * LOG_ROW_HEIGHT, 'bottom'));
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
    else line.textContent = row.text;
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
    const update = () => show(state.update(input.value, Math.floor(viewer.scrollTop / LOG_ROW_HEIGHT)));
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
