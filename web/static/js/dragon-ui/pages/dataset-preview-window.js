/* Bounded, bidirectional page window for the dataset preview dialog. */

export const DATASET_PREVIEW_MAX_RESIDENT_PAGES = 3;
export const DATASET_PREVIEW_MAX_CACHED_PAGES = 8;

export function createDatasetPreviewWindow(payload, imageKey) {
    const page = previewPage(payload, imageKey);
    const session = {
        total: payload.total,
        limit: payload.limit,
        activeOffset: page.offset,
        startOffset: page.offset,
        loadedEnd: page.endOffset,
        hasMoreBefore: payload.has_more_before,
        hasMoreAfter: payload.has_more_after,
        legacyPagination: payload.legacy_pagination,
        pages: new Map([[page.offset, page]]),
        pageCache: new Map([[page.offset, payload]]),
        imagesByKey: new Map(page.images.map((image) => [imageKey(image), image])),
    };
    syncDatasetPreviewWindow(session);
    return session;
}

export function datasetPreviewRequestOffset(session, direction) {
    if (direction === 'before') return Math.max(0, session.startOffset - session.limit);
    if (direction === 'after') return Math.min(session.total, session.loadedEnd);
    throw new Error(`未知的数据集预览加载方向：${direction}`);
}

export function mergeDatasetPreviewPage(session, payload, direction, imageKey) {
    const page = previewPage(payload, imageKey);
    validatePageBoundary(session, page, payload, direction);
    const images = page.images.filter((image) => !session.imagesByKey.has(imageKey(image)));
    if (page.images.length && !images.length) throw new Error('分页接口返回了重复图片，已暂停自动加载。');

    page.images = images;
    session.pages.set(page.offset, page);
    for (const image of images) session.imagesByKey.set(imageKey(image), image);
    cacheDatasetPreviewPage(session, payload);
    session.legacyPagination = session.legacyPagination || payload.legacy_pagination;
    syncDatasetPreviewWindow(session);
    return page;
}

export function getCachedDatasetPreviewPage(session, offset) {
    const payload = session.pageCache.get(offset);
    if (!payload) return null;
    session.pageCache.delete(offset);
    session.pageCache.set(offset, payload);
    return payload;
}

export function trimDatasetPreviewWindow(
    session,
    direction,
    imageKey,
    maxPages = DATASET_PREVIEW_MAX_RESIDENT_PAGES,
) {
    const removedOffsets = [];
    while (session.pages.size > maxPages) {
        const offsets = sortedPageOffsets(session);
        const offset = direction === 'before' ? offsets.at(-1) : offsets[0];
        const page = session.pages.get(offset);
        session.pages.delete(offset);
        for (const image of page?.images || []) session.imagesByKey.delete(imageKey(image));
        removedOffsets.push(offset);
    }
    syncDatasetPreviewWindow(session);
    return removedOffsets;
}

export function preserveDatasetPreviewAnchor(scroller, anchor, mutate) {
    if (!scroller || !anchor) {
        mutate();
        return;
    }
    const before = anchor.getBoundingClientRect().top;
    mutate();
    if (anchor.isConnected === false) return;
    const delta = anchor.getBoundingClientRect().top - before;
    if (delta) scroller.scrollTop += delta;
}

export function findDatasetPreviewAnchor(scroller, grid) {
    if (!scroller || !grid) return null;
    const threshold = datasetPreviewContentTop(scroller, grid);
    const cards = grid.querySelectorAll('[data-dataset-preview-page-offset]');
    for (const card of cards) {
        if (card.getBoundingClientRect().bottom > threshold) return card;
    }
    return cards.length ? cards[cards.length - 1] : null;
}

export function visibleDatasetPreviewOffset(scroller, grid, fallback = 0) {
    const anchor = findDatasetPreviewAnchor(scroller, grid);
    const offset = Number(anchor?.dataset?.datasetPreviewPageOffset);
    return Number.isFinite(offset) && offset >= 0 ? Math.floor(offset) : fallback;
}

export function alignDatasetPreviewOffset(scroller, grid, offset) {
    if (!scroller || !grid) return;
    const card = grid.querySelector(`[data-dataset-preview-page-offset="${offset}"]`);
    if (!card) return;
    const delta = card.getBoundingClientRect().top - datasetPreviewContentTop(scroller, grid);
    if (delta) scroller.scrollTop += delta;
}

function previewPage(payload, imageKey) {
    const images = [];
    const keys = new Set();
    for (const image of payload.images || []) {
        const key = imageKey(image);
        if (keys.has(key)) continue;
        keys.add(key);
        images.push(image);
    }
    return {
        offset: payload.offset,
        endOffset: Math.max(payload.offset + images.length, payload.next_offset),
        images,
    };
}

function validatePageBoundary(session, page, payload, direction) {
    if (direction === 'before') {
        if (page.offset >= session.startOffset || page.endOffset > session.startOffset) {
            throw new Error('上一批分页边界重叠，已暂停自动加载。');
        }
        if (!page.images.length && payload.has_more_before) {
            throw new Error('分页接口没有返回上一批图片，已暂停自动加载。');
        }
        return;
    }
    if (direction !== 'after') throw new Error(`未知的数据集预览加载方向：${direction}`);
    if (page.offset < session.loadedEnd) throw new Error('下一批分页边界重叠，已暂停自动加载。');
    if (!page.images.length && payload.has_more_after) {
        throw new Error('分页接口没有返回下一批图片，已暂停自动加载。');
    }
}

function cacheDatasetPreviewPage(session, payload) {
    session.pageCache.delete(payload.offset);
    session.pageCache.set(payload.offset, payload);
    while (session.pageCache.size > DATASET_PREVIEW_MAX_CACHED_PAGES) {
        session.pageCache.delete(session.pageCache.keys().next().value);
    }
}

function syncDatasetPreviewWindow(session) {
    const offsets = sortedPageOffsets(session);
    session.startOffset = offsets[0] ?? 0;
    session.loadedEnd = offsets.reduce(
        (end, offset) => Math.max(end, session.pages.get(offset)?.endOffset || offset),
        session.startOffset,
    );
    session.hasMoreBefore = session.startOffset > 0;
    session.hasMoreAfter = session.loadedEnd < session.total;
    session.loadedPageCount = session.pages.size;
}

function sortedPageOffsets(session) {
    return [...session.pages.keys()].sort((left, right) => left - right);
}

function datasetPreviewContentTop(scroller, grid) {
    const scrollerTop = scroller.getBoundingClientRect().top;
    const pagination = grid.closest?.('.dataset-preview-results')
        ?.querySelector?.('[data-dataset-preview-pagination]');
    if (!pagination || pagination.hidden) return scrollerTop + 8;
    return Math.max(scrollerTop, pagination.getBoundingClientRect().bottom) + 8;
}
