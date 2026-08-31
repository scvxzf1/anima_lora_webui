import {
    GROUP_INITIAL_RENDER_COUNT,
    GROUP_VIRTUALIZE_THRESHOLD,
    GROUP_VIRTUAL_OVERSCAN_ROWS,
} from './constants.js?v=module-bootstrap-20260831-release-v1';

/**
 * 纯函数：限制整数到 [min, max]。
 */
export function clampInt(value, min, max) {
    return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}

/**
 * 纯函数：是否启用虚拟窗口。
 */
export function shouldVirtualizeGroup(visibleCount) {
    return visibleCount > GROUP_VIRTUALIZE_THRESHOLD;
}

/**
 * 纯函数：测量网格列数与行高。
 */
export function measureVirtualMetrics(body) {
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

/**
 * 纯函数：比较两个虚拟窗口状态是否等价。
 */
export function virtualWindowEquals(previous, next) {
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

/**
 * 纯函数：未虚拟化或回退时的默认窗口。
 */
export function defaultVirtualWindow(visibleCount) {
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

/**
 * 纯函数：把窗口状态裁到当前可见数量范围。
 */
export function clampVirtualWindow(windowState, visibleCount) {
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

/**
 * 工厂：注入 state / DOM / 渲染回调，避免与 gallery 主模块循环依赖。
 *
 * @param {object} deps
 * @param {object} deps.state gallery 内部状态（需含 virtualWindowByGroup / virtualRefreshFrame / groupsByKey / expandedGroupKeys）
 * @param {(groupKey: string) => Element|undefined|null} deps.findGroupSection
 * @param {(groupKey: string, totalCount: number) => number} deps.visibleCountForGroup
 * @param {(body: Element, group: object, visibleCount: number) => void} deps.renderGroupBody
 * @param {() => void} deps.refreshVisibleOrderedKeys
 */
export function createVirtualWindowApi({
    state,
    findGroupSection,
    visibleCountForGroup,
    renderGroupBody,
    refreshVisibleOrderedKeys,
}) {
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

    return {
        scheduleVirtualWindowRefresh,
        cancelVirtualWindowRefresh,
        refreshVirtualWindows,
        resolveVirtualWindow,
        nextVirtualWindow,
        defaultVirtualWindow,
        clampVirtualWindow,
        virtualWindowEquals,
        shouldVirtualizeGroup,
        measureVirtualMetrics,
        clampInt,
    };
}
