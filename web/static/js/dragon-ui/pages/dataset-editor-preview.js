/* Image and caption preview for the Dragon dataset workspace. */

import { escapeAttribute, escapeHtml } from './dataset-editor-fields.js?v=dragon-ui-20260828v54';
import { createDatasetPreviewDetailController } from './dataset-preview-detail.js?v=dragon-ui-20260902v4';
import {
    alignDatasetPreviewOffset,
    createDatasetPreviewWindow,
    datasetPreviewRequestOffset,
    findDatasetPreviewAnchor,
    getCachedDatasetPreviewPage,
    mergeDatasetPreviewPage,
    preserveDatasetPreviewAnchor,
    trimDatasetPreviewWindow,
    visibleDatasetPreviewOffset,
} from './dataset-preview-window.js?v=dragon-ui-20260831v3';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { copyText } from '../../shared/dom.js?v=dragon-ui-20260812v35';

export const DATASET_PREVIEW_PAGE_SIZE = 24;
export const DATASET_PREVIEW_RESTART_MESSAGE = '分页接口尚未生效，请重启 WebUI 服务后再翻页或继续加载。';

const DATASET_PREVIEW_BEFORE_ROOT_MARGIN = '640px 0px 0px 0px';
const DATASET_PREVIEW_AFTER_ROOT_MARGIN = '0px 0px 640px 0px';

export function normalizeDatasetPreviewPayload(payload, requestedOffset = 0, requestedLimit = DATASET_PREVIEW_PAGE_SIZE) {
    if (!payload || typeof payload !== 'object') throw new Error('数据集预览接口返回了无效响应');
    if (payload.ok === false) throw new Error(payload.error || '读取数据集预览失败');
    const images = Array.isArray(payload.images) ? payload.images : [];
    const cleanRequestedOffset = nonnegativeInteger(requestedOffset, 0);
    const cleanRequestedLimit = positiveInteger(requestedLimit, DATASET_PREVIEW_PAGE_SIZE);
    const hasOffset = hasFiniteNumber(payload, 'offset');
    if (cleanRequestedOffset > 0 && !hasOffset) throw new Error(DATASET_PREVIEW_RESTART_MESSAGE);

    const offset = hasOffset ? nonnegativeInteger(payload.offset, 0) : 0;
    const limit = positiveInteger(payload.limit, cleanRequestedLimit);
    const total = Math.max(images.length, nonnegativeInteger(payload.total, images.length));
    const returned = images.length;
    const fallbackNextOffset = Math.min(total, offset + returned);
    const nextOffset = hasFiniteNumber(payload, 'next_offset')
        ? Math.min(total, Math.max(offset, nonnegativeInteger(payload.next_offset, fallbackNextOffset)))
        : fallbackNextOffset;
    const hasMoreBefore = typeof payload.has_more_before === 'boolean'
        ? payload.has_more_before
        : offset > 0;
    const hasMoreAfter = typeof payload.has_more_after === 'boolean'
        ? payload.has_more_after
        : nextOffset < total;
    const legacyPagination = !hasOffset
        || !hasFiniteNumber(payload, 'returned')
        || !hasFiniteNumber(payload, 'next_offset')
        || typeof payload.has_more_before !== 'boolean'
        || typeof payload.has_more_after !== 'boolean';
    return {
        ...payload,
        images,
        total,
        offset,
        limit,
        returned,
        next_offset: nextOffset,
        has_more_before: hasMoreBefore,
        has_more_after: hasMoreAfter,
        legacy_pagination: legacyPagination,
    };
}

export async function openDatasetPreview(api, state, datasetIndex, offset = 0) {
    const file = state.selectedFile || state.datasetConfig;
    if (!file) throw new Error('请先保存或选择一个数据集预设，再打开图片预览');
    if (state.dirty) throw new Error('当前数据集有未保存修改，请先保存后再预览');
    const dialog = document.getElementById('dataset-preview-dialog');
    if (!dialog) throw new Error('找不到数据集预览窗口，请刷新页面');
    const cleanOffset = nonnegativeInteger(offset, 0);
    const requestSequence = nextRequestSequence(state);
    state.datasetPreviewDetailController?.restore({ focus: false, defer: false });
    state.datasetPreviewSession = null;
    renderPreviewState({ loading: true, datasetIndex });
    openDatasetPreviewDialog(dialog);
    try {
        const payload = await requestDatasetPreviewPage(api, file, datasetIndex, cleanOffset);
        if (requestSequence !== state.datasetPreviewRequestSequence) return;
        const session = createPreviewSession(payload, file, datasetIndex);
        state.datasetPreviewSession = session;
        state.previewIndex = datasetIndex;
        state.previewOffset = session.activeOffset;
        renderPreviewState({ payload, session, datasetIndex });
        resetPreviewScrollPosition();
    } catch (error) {
        if (requestSequence !== state.datasetPreviewRequestSequence) return;
        renderPreviewState({ error: error.message, datasetIndex });
        throw error;
    }
}

export function bindDatasetPreviewRefresh(api, state) {
    const refreshButton = document.getElementById('btn-refresh-dataset-preview');
    const grid = document.getElementById('dataset-preview-grid');
    const dialog = document.getElementById('dataset-preview-dialog');
    const chrome = ensurePreviewChrome();
    if (!refreshButton || !grid || !chrome) return () => {};
    const detail = createDatasetPreviewDetailController(dialog);
    state.datasetPreviewDetailController = detail;

    const refreshHandler = async () => {
        refreshButton.disabled = true;
        try {
            await openDatasetPreview(api, state, state.previewIndex || 0, state.previewOffset || 0);
        } catch (_error) {
            // The dialog already owns the detailed error state.
        } finally {
            refreshButton.disabled = false;
        }
    };
    const pageClickHandler = async (event) => {
        const pageButton = event.target?.closest?.('[data-dataset-preview-page]');
        if (!pageButton || pageButton.disabled) return;
        await navigateDatasetPreview(api, state, nonnegativeInteger(pageButton.dataset.offset, 0));
    };
    const jumpHandler = async (event) => {
        if (!event.target?.matches?.('[data-dataset-preview-jump]')) return;
        event.preventDefault();
        const session = state.datasetPreviewSession;
        const input = event.target.querySelector('[data-dataset-preview-page-input]');
        if (!session || !input || !input.reportValidity()) return;
        const totalPages = Math.max(1, Math.ceil(session.total / session.limit));
        const page = Math.min(totalPages, Math.max(1, positiveInteger(input.value, 1)));
        await navigateDatasetPreview(api, state, (page - 1) * session.limit);
    };
    const directionalLoadHandler = (event) => {
        const button = event.target?.closest?.('[data-dataset-preview-load-direction]');
        if (!button || button.disabled) return;
        const direction = button.dataset.datasetPreviewLoadDirection;
        void loadDatasetPreviewDirection(api, state, direction, { automatic: false });
    };
    const copyHandler = (event) => {
        const button = event.target?.closest?.('[data-caption-copy]');
        if (button) void copyPreviewCaption(button);
    };
    const imageHandler = (event) => {
        const button = event.target?.closest?.('[data-dataset-preview-image]');
        if (!button || !grid.contains(button)) return;
        const key = String(button.dataset.datasetPreviewImage || '');
        const image = state.datasetPreviewSession?.imagesByKey?.get(key);
        if (image) detail.open(image, button);
    };
    const dialogCloseHandler = () => {
        detail.restore({ focus: false, defer: false });
        nextRequestSequence(state);
        state.datasetPreviewSession = null;
    };
    const dialogCancelHandler = (event) => {
        detail.handleCancel(event);
    };

    refreshButton.addEventListener('click', refreshHandler);
    chrome.pagination.addEventListener('click', pageClickHandler);
    chrome.pagination.addEventListener('submit', jumpHandler);
    chrome.loadBefore.addEventListener('click', directionalLoadHandler);
    chrome.loadAfter.addEventListener('click', directionalLoadHandler);
    grid.addEventListener('click', copyHandler);
    grid.addEventListener('click', imageHandler);
    dialog?.addEventListener('close', dialogCloseHandler);
    dialog?.addEventListener('cancel', dialogCancelHandler);
    const observers = [
        installDirectionalLoadObserver(api, state, chrome.beforeSentinel, 'before'),
        installDirectionalLoadObserver(api, state, chrome.afterSentinel, 'after'),
    ];
    const removeScrollSpy = installDatasetPreviewScrollSpy(state, grid);
    return () => {
        nextRequestSequence(state);
        state.datasetPreviewSession = null;
        refreshButton.removeEventListener('click', refreshHandler);
        chrome.pagination.removeEventListener('click', pageClickHandler);
        chrome.pagination.removeEventListener('submit', jumpHandler);
        chrome.loadBefore.removeEventListener('click', directionalLoadHandler);
        chrome.loadAfter.removeEventListener('click', directionalLoadHandler);
        grid.removeEventListener('click', copyHandler);
        grid.removeEventListener('click', imageHandler);
        dialog?.removeEventListener('close', dialogCloseHandler);
        dialog?.removeEventListener('cancel', dialogCancelHandler);
        observers.forEach((observer) => observer?.disconnect());
        removeScrollSpy();
        detail.dispose();
        if (state.datasetPreviewDetailController === detail) state.datasetPreviewDetailController = null;
    };
}

async function navigateDatasetPreview(api, state, offset) {
    try {
        await openDatasetPreview(api, state, state.previewIndex || 0, offset);
    } catch (_error) {
        // openDatasetPreview renders the actionable failure in the dialog.
    }
}

async function loadDatasetPreviewDirection(api, state, direction, { automatic }) {
    const session = state.datasetPreviewSession;
    const boundaryFlag = direction === 'before' ? 'hasMoreBefore' : 'hasMoreAfter';
    if (!session || session.loadingDirection || !session[boundaryFlag]) return;
    if (automatic && (session.legacyPagination || session.autoLoadStopped[direction])) return;
    session.loadingDirection = direction;
    session.loadErrors[direction] = '';
    if (!automatic) session.autoLoadStopped[direction] = false;
    renderPreviewControls(session);
    const requestOffset = datasetPreviewRequestOffset(session, direction);
    const requestSequence = nextRequestSequence(state);
    try {
        const payload = getCachedDatasetPreviewPage(session, requestOffset)
            || await requestDatasetPreviewPage(api, session.file, session.datasetIndex, requestOffset);
        if (requestSequence !== state.datasetPreviewRequestSequence || state.datasetPreviewSession !== session) {
            clearStaleDatasetPreviewLoad(state, session, direction);
            return;
        }
        const page = mergeDatasetPreviewPage(session, payload, direction, previewImageKey);
        insertPreviewPage(page, direction);
        trimPreviewPages(session, direction);
        session.loadingDirection = '';
        session.autoLoadStopped[direction] = false;
        updatePreviewSummary(session);
        renderPreviewControls(session);
        if (direction === 'before' && session.alignInitialOffsetAfterPrepend) {
            alignInitialPreviewPage(session);
        }
        syncActivePreviewPage(state);
    } catch (error) {
        if (requestSequence !== state.datasetPreviewRequestSequence || state.datasetPreviewSession !== session) {
            clearStaleDatasetPreviewLoad(state, session, direction);
            return;
        }
        session.loadingDirection = '';
        session.autoLoadStopped[direction] = true;
        session.loadErrors[direction] = error.message;
        renderPreviewControls(session);
    }
}

function clearStaleDatasetPreviewLoad(state, session, direction) {
    if (state.datasetPreviewSession !== session || session.loadingDirection !== direction) return;
    session.loadingDirection = '';
    renderPreviewControls(session);
}

async function requestDatasetPreviewPage(api, file, datasetIndex, offset) {
    const params = new URLSearchParams({
        file,
        dataset_index: String(datasetIndex),
        source: 'source',
        limit: String(DATASET_PREVIEW_PAGE_SIZE),
        offset: String(Math.max(0, Number(offset) || 0)),
    });
    const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
    return normalizeDatasetPreviewPayload(payload, offset, DATASET_PREVIEW_PAGE_SIZE);
}

function createPreviewSession(payload, file, datasetIndex) {
    return {
        ...createDatasetPreviewWindow(payload, previewImageKey),
        file,
        datasetIndex,
        loadingDirection: '',
        autoLoadStopped: { before: false, after: false },
        loadErrors: { before: '', after: '' },
        alignInitialOffsetAfterPrepend: payload.offset > 0,
        summaryPayload: payload,
    };
}

function renderPreviewState({ payload = null, session = null, loading = false, error = '', datasetIndex = 0 }) {
    const title = document.getElementById('dataset-preview-dialog-title');
    const meta = document.getElementById('dataset-preview-dialog-meta');
    const details = document.getElementById('dataset-preview-details');
    const grid = document.getElementById('dataset-preview-grid');
    const empty = document.getElementById('dataset-preview-empty');
    const chrome = ensurePreviewChrome();
    if (!title || !meta || !details || !grid || !empty || !chrome) return;
    title.textContent = `第 ${datasetIndex + 1} 组数据集预览`;
    grid.innerHTML = '';
    details.innerHTML = '';
    chrome.pagination.hidden = true;
    chrome.loadBefore.hidden = true;
    chrome.loadAfter.hidden = true;
    if (loading || error) {
        meta.textContent = error || '正在读取图片与标注…';
        empty.hidden = false;
        empty.textContent = error || '正在扫描数据集目录…';
        return;
    }

    renderPreviewDetails(details, payload);
    if (!payload.images.length) {
        meta.textContent = `${payload.source_label || '原始图目录'} · ${payload.directory || '-'} · 共 0 张`;
        empty.hidden = false;
        empty.textContent = payload.message || '当前目录没有可预览图片。';
        return;
    }
    empty.hidden = true;
    grid.innerHTML = payload.images.map((image) => renderPreviewCard(image, payload.offset)).join('');
    updatePreviewSummary(session);
    renderPreviewControls(session);
}

function renderPreviewDetails(details, payload) {
    const row = payload.row || {};
    const settings = payload.settings || row.settings || {};
    [
        ['数据集文件', payload.file || '-'],
        ['当前目录', payload.directory || '-'],
        ['原始路径', row.source_dir || '-'],
        ['重复次数', row.num_repeats ?? '-'],
        ['分辨率', settings.resolution ? `${settings.resolution}px` : '-'],
        ['标注来源', payload.caption_source_label || '-'],
    ].forEach(([label, value]) => {
        details.insertAdjacentHTML('beforeend', `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`);
    });
}

function updatePreviewSummary(session) {
    const meta = document.getElementById('dataset-preview-dialog-meta');
    if (!meta || !session) return;
    const payload = session.summaryPayload;
    const rangeStart = session.total ? session.startOffset + 1 : 0;
    const captionSummary = session.loadedPageCount === 1
        ? ` · ${payload.caption_summary || '未识别到标注'}`
        : '';
    meta.textContent = `${payload.source_label || '原始图目录'} · ${payload.directory || '-'} · 已加载第 ${rangeStart}-${session.loadedEnd} 张 / 共 ${session.total} 张${captionSummary}`;
}

function ensurePreviewChrome() {
    const grid = document.getElementById('dataset-preview-grid');
    const results = grid?.closest('.dataset-preview-results');
    if (!grid || !results) return null;
    let pagination = results.querySelector('[data-dataset-preview-pagination]');
    if (!pagination) {
        pagination = document.createElement('nav');
        pagination.className = 'dataset-preview-pagination';
        pagination.dataset.datasetPreviewPagination = '';
        pagination.setAttribute('aria-label', '数据集图片分页');
        pagination.hidden = true;
        pagination.innerHTML = `
            <button class="dragon-icon-button dataset-preview-page-button" type="button" data-dataset-preview-page="previous" aria-label="上一页" title="上一页">
                ${renderIcon('chevronUp', 'dragon-btn-icon dataset-preview-page-icon previous')}
            </button>
            <form class="dataset-preview-page-jump" data-dataset-preview-jump>
                <span>第</span>
                <input class="dataset-preview-page-input" type="number" min="1" step="1" inputmode="numeric" data-dataset-preview-page-input aria-label="页码">
                <span>/ <span data-dataset-preview-page-total>1</span> 页</span>
                <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="submit">跳转</button>
            </form>
            <button class="dragon-icon-button dataset-preview-page-button" type="button" data-dataset-preview-page="next" aria-label="下一页" title="下一页">
                ${renderIcon('chevronUp', 'dragon-btn-icon dataset-preview-page-icon next')}
            </button>
            <span class="dataset-preview-page-status" data-dataset-preview-page-status aria-live="polite"></span>
        `;
        results.insertBefore(pagination, results.firstChild);
    }
    const loadBefore = ensureDirectionalLoadControl(results, grid, 'before');
    const loadAfter = ensureDirectionalLoadControl(results, grid, 'after');
    return {
        pagination,
        loadBefore,
        loadAfter,
        beforeSentinel: loadBefore.querySelector('[data-dataset-preview-sentinel]'),
        afterSentinel: loadAfter.querySelector('[data-dataset-preview-sentinel]'),
    };
}

function ensureDirectionalLoadControl(results, grid, direction) {
    let control = results.querySelector(`[data-dataset-preview-load-wrap="${direction}"]`);
    if (control) return control;
    const before = direction === 'before';
    control = document.createElement('div');
    control.className = `dataset-preview-load-more dataset-preview-load-${direction}`;
    control.dataset.datasetPreviewLoadWrap = direction;
    control.hidden = true;
    control.innerHTML = `
        <span class="dataset-preview-sentinel" data-dataset-preview-sentinel aria-hidden="true"></span>
        <span class="dataset-preview-load-more-status" data-dataset-preview-load-more-status aria-live="polite"></span>
        <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-dataset-preview-load-direction="${direction}">${before ? '加载上一批' : '继续加载'}</button>
    `;
    if (before) results.insertBefore(control, grid);
    else results.append(control);
    return control;
}

function renderPreviewControls(session) {
    const chrome = ensurePreviewChrome();
    if (!chrome || !session) return;
    const pageCount = Math.max(1, Math.ceil(session.total / session.limit));
    const activePage = Math.min(pageCount, Math.floor(session.activeOffset / session.limit) + 1);
    const loading = Boolean(session.loadingDirection);
    const previous = chrome.pagination.querySelector('[data-dataset-preview-page="previous"]');
    const next = chrome.pagination.querySelector('[data-dataset-preview-page="next"]');
    const input = chrome.pagination.querySelector('[data-dataset-preview-page-input]');
    const total = chrome.pagination.querySelector('[data-dataset-preview-page-total]');
    const status = chrome.pagination.querySelector('[data-dataset-preview-page-status]');
    const jump = chrome.pagination.querySelector('[data-dataset-preview-jump] button[type="submit"]');
    previous.disabled = loading || activePage <= 1;
    previous.dataset.offset = String(Math.max(0, session.activeOffset - session.limit));
    next.disabled = loading || activePage >= pageCount;
    next.dataset.offset = String(Math.min((pageCount - 1) * session.limit, session.activeOffset + session.limit));
    input.value = String(activePage);
    input.max = String(pageCount);
    input.disabled = loading || pageCount <= 1;
    jump.disabled = loading || pageCount <= 1;
    total.textContent = String(pageCount);
    status.textContent = `已加载 ${session.loadedEnd - session.startOffset} 张（${session.startOffset + 1}-${session.loadedEnd}）/ 共 ${session.total} 张`;
    chrome.pagination.hidden = pageCount <= 1;

    renderDirectionalLoadControl(chrome.loadBefore, session, 'before');
    renderDirectionalLoadControl(chrome.loadAfter, session, 'after');
}

function renderDirectionalLoadControl(control, session, direction) {
    const before = direction === 'before';
    const hasMore = before ? session.hasMoreBefore : session.hasMoreAfter;
    const loading = session.loadingDirection === direction;
    const loadError = session.loadErrors[direction];
    const status = control.querySelector('[data-dataset-preview-load-more-status]');
    const button = control.querySelector('[data-dataset-preview-load-direction]');
    control.hidden = !hasMore && !loading && !loadError;
    button.disabled = Boolean(session.loadingDirection);
    button.textContent = loadError ? '重试' : (before ? '加载上一批' : '继续加载');
    control.classList.toggle('has-error', Boolean(loadError || session.legacyPagination));
    if (loading) {
        status.textContent = before ? '正在加载上一批图片…' : '正在加载下一批图片…';
    } else if (loadError) {
        status.textContent = loadError;
    } else if (session.legacyPagination) {
        status.textContent = '当前 WebUI 服务需要重启后才能继续加载';
    } else {
        const remaining = before ? session.startOffset : session.total - session.loadedEnd;
        status.textContent = before
            ? `前面还有 ${Math.max(0, remaining)} 张未加载`
            : `后面还有 ${Math.max(0, remaining)} 张未加载`;
    }
}

function installDirectionalLoadObserver(api, state, sentinel, direction) {
    const Observer = globalThis.IntersectionObserver;
    const root = document.querySelector('#dataset-preview-dialog .dataset-preview-dialog-body');
    if (!Observer || !root || !sentinel) return null;
    const observer = new Observer((entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        void loadDatasetPreviewDirection(api, state, direction, { automatic: true });
    }, {
        root,
        rootMargin: direction === 'before'
            ? DATASET_PREVIEW_BEFORE_ROOT_MARGIN
            : DATASET_PREVIEW_AFTER_ROOT_MARGIN,
        threshold: 0.01,
    });
    observer.observe(sentinel);
    return observer;
}

function insertPreviewPage(page, direction) {
    if (!page.images.length) return;
    const grid = document.getElementById('dataset-preview-grid');
    const scroller = document.querySelector('#dataset-preview-dialog .dataset-preview-dialog-body');
    if (!grid) return;
    const markup = page.images.map((image) => renderPreviewCard(image, page.offset)).join('');
    if (direction === 'before') {
        preserveDatasetPreviewAnchor(scroller, grid.firstElementChild, () => {
            grid.insertAdjacentHTML('afterbegin', markup);
        });
        return;
    }
    grid.insertAdjacentHTML('beforeend', markup);
}

function trimPreviewPages(session, direction) {
    const grid = document.getElementById('dataset-preview-grid');
    const scroller = document.querySelector('#dataset-preview-dialog .dataset-preview-dialog-body');
    if (!grid) return;
    const anchor = findDatasetPreviewAnchor(scroller, grid);
    const removedOffsets = trimDatasetPreviewWindow(session, direction, previewImageKey);
    if (!removedOffsets.length) return;
    const anchorOffset = Number(anchor?.dataset?.datasetPreviewPageOffset);
    const stableAnchor = removedOffsets.includes(anchorOffset)
        ? findSurvivingPreviewAnchor(grid, removedOffsets, direction)
        : anchor;
    preserveDatasetPreviewAnchor(scroller, stableAnchor, () => {
        for (const offset of removedOffsets) {
            grid.querySelectorAll(`[data-dataset-preview-page-offset="${offset}"]`).forEach((card) => card.remove());
        }
    });
}

function findSurvivingPreviewAnchor(grid, removedOffsets, direction) {
    const cards = [...grid.querySelectorAll('[data-dataset-preview-page-offset]')]
        .filter((card) => !removedOffsets.includes(Number(card.dataset.datasetPreviewPageOffset)));
    if (!cards.length) return null;
    return direction === 'after' ? cards[0] : cards[cards.length - 1];
}

function installDatasetPreviewScrollSpy(state, grid) {
    const scroller = document.querySelector('#dataset-preview-dialog .dataset-preview-dialog-body');
    if (!scroller) return () => {};
    let frame = 0;
    const schedule = () => {
        if (frame) return;
        frame = window.requestAnimationFrame(() => {
            frame = 0;
            syncActivePreviewPage(state, scroller, grid);
        });
    };
    scroller.addEventListener('scroll', schedule, { passive: true });
    return () => {
        scroller.removeEventListener('scroll', schedule);
        if (frame) window.cancelAnimationFrame(frame);
    };
}

function syncActivePreviewPage(state, scroller = null, grid = null) {
    const session = state.datasetPreviewSession;
    if (!session) return;
    const root = scroller || document.querySelector('#dataset-preview-dialog .dataset-preview-dialog-body');
    const previewGrid = grid || document.getElementById('dataset-preview-grid');
    const offset = visibleDatasetPreviewOffset(root, previewGrid, session.activeOffset);
    if (offset === session.activeOffset) return;
    session.activeOffset = offset;
    state.previewOffset = offset;
    renderPreviewControls(session);
}

function resetPreviewScrollPosition() {
    const scroller = document.querySelector('#dataset-preview-dialog .dataset-preview-dialog-body');
    if (scroller) scroller.scrollTop = 0;
}

function alignInitialPreviewPage(session) {
    const scroller = document.querySelector('#dataset-preview-dialog .dataset-preview-dialog-body');
    const grid = document.getElementById('dataset-preview-grid');
    alignDatasetPreviewOffset(scroller, grid, session.activeOffset);
    session.alignInitialOffsetAfterPrepend = false;
}

async function copyPreviewCaption(button) {
    const text = button.closest('.dataset-preview-caption')?.querySelector('pre')?.textContent || '';
    const label = button.querySelector('span');
    const original = label?.textContent || '复制标注';
    button.disabled = true;
    try {
        await copyText(text);
        if (label) label.textContent = '已复制';
    } catch (_error) {
        if (label) label.textContent = '复制失败';
    } finally {
        window.setTimeout(() => {
            if (!button.isConnected) return;
            if (label) label.textContent = original;
            button.disabled = false;
        }, 1000);
    }
}

function openDatasetPreviewDialog(dialog) {
    if (!dialog || dialog.open) return;
    if (typeof dialog.showModal === 'function') {
        dialog.showModal();
    } else {
        dialog.setAttribute('open', 'open');
    }
}

function renderPreviewCard(image, pageOffset) {
    const caption = image.caption || {};
    const key = previewImageKey(image);
    const label = image.name || '数据集图片';
    const previewUrl = image.thumbnail_url || image.url;
    const copyButton = caption.ok
        ? `<button class="btn btn-small" type="button" data-caption-copy>${renderIcon('copy', 'dragon-btn-icon')}<span>复制标注</span></button>`
        : '';
    return `
        <article class="dataset-preview-card" data-dataset-preview-page-offset="${nonnegativeInteger(pageOffset, 0)}">
            <button class="dataset-preview-image-btn" type="button" data-dataset-preview-image="${escapeAttribute(key)}" aria-label="打开 ${escapeAttribute(label)} 详情"><img src="${escapeAttribute(previewUrl)}" alt="${escapeAttribute(label)}" width="320" height="240" loading="lazy" decoding="async" fetchpriority="low"></button>
            <div class="dataset-preview-card-body">
                <strong>${escapeHtml(image.name || '-')}</strong><span>${escapeHtml(image.file || '')}</span>
                <div class="dataset-preview-caption${caption.ok ? '' : ' missing'}">
                    <div><span>${caption.ok ? `标注 ${escapeHtml(caption.format_label || caption.extension || '')}` : `缺少标注 · ${escapeHtml(caption.source_label || '自动识别')}`}</span>${copyButton}</div>
                    <pre>${escapeHtml(caption.ok ? (caption.text || '(空标注)') : '未按当前标注来源找到 caption 文件')}</pre>
                </div>
            </div>
        </article>
    `;
}

function previewImageKey(image) {
    return String(image?.file || image?.url || image?.name || '');
}

function nextRequestSequence(state) {
    const sequence = Number(state.datasetPreviewRequestSequence || 0) + 1;
    state.datasetPreviewRequestSequence = sequence;
    return sequence;
}

function hasFiniteNumber(value, key) {
    return value[key] !== '' && value[key] !== null && value[key] !== undefined && Number.isFinite(Number(value[key]));
}

function nonnegativeInteger(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : fallback;
}

function positiveInteger(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}
