import {
    imageCardMetaText as formatImageCardMetaText,
    imageKey,
    imageTimestampText,
} from './image-meta.js?v=module-bootstrap-20260711-ir1';

/**
 * 历史分组与图片卡片 DOM。
 *
 * @param {object} deps
 * @param {object} deps.state
 * @param {Function} deps.formatBytes
 * @param {Function} deps.openPreviewDialog
 * @param {(key: string, options?: object) => void} deps.toggleImageSelection
 * @param {(groupKey: string, visibleCount: number) => object} deps.resolveVirtualWindow
 * @param {(group: object, visibleCount: number) => Element} deps.createLoadMoreFooter
 * @param {(groupKey: string) => number} deps.freshCountForGroup
 * @param {(groupKey: string, totalCount: number) => number} deps.visibleCountForGroup
 * @param {() => void} deps.refreshVisibleOrderedKeys
 * @param {() => void} deps.requestRerender
 */
export function createCardsApi({
    state,
    formatBytes,
    openPreviewDialog,
    toggleImageSelection,
    resolveVirtualWindow,
    createLoadMoreFooter,
    freshCountForGroup,
    visibleCountForGroup,
    refreshVisibleOrderedKeys,
    requestRerender,
}) {
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
        details.textContent = formatImageCardMetaText(image, formatBytes);

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
        requestRerender();
    }

    return {
        createHistoryGroup,
        createFreshBadge,
        appendGroupCards,
        renderGroupBody,
        createVirtualSpacer,
        createImageCard,
        toggleHistoryGroup,
    };
}
