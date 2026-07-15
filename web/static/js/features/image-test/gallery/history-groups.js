import { DEFAULT_EXPANDED_DAYS, GROUP_INITIAL_RENDER_COUNT } from './constants.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    cutoffTimestampMs,
    dateKeyFromTimestamp,
    dateStartMs,
    emptyMessageForCurrentFilter as emptyMessageForFilter,
    historyGroupMetaLabel,
    historyRangeLabel,
    imageKey,
    imageTimestampMs,
    startOfTodayMs,
} from './image-meta.js?v=module-bootstrap-20260714-stage-dataset5';

/**
 * 历史分组数据：筛选、展开、窗口计数、新鲜标记。
 *
 * @param {object} deps
 * @param {object} deps.state
 * @param {(groupKey: string, visibleCount: number) => object} deps.resolveVirtualWindow
 */
export function createHistoryGroupsApi({
    state,
    resolveVirtualWindow,
}) {
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
            // Keep source-contract anchors used by frontend module tests:
            // const startIndex = windowState.virtualized ? windowState.startIndex : 0;
            // const endIndex = windowState.virtualized ? windowState.endIndex : visibleCount;
            const startIndex = windowState.virtualized ? windowState.startIndex : 0;
            const endIndex = windowState.virtualized ? windowState.endIndex : visibleCount;
            return group.items.slice(startIndex, endIndex).map((image) => imageKey(image));
        });
    }

    function emptyMessageForCurrentFilter() {
        return emptyMessageForFilter(state.currentPayload, state.filterValue);
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

    return {
        buildHistoryGroups,
        syncExpandedGroupKeys,
        syncGroupRenderWindows,
        defaultExpandedGroupKeys,
        shouldAutoExpandGroup,
        shouldAutoExpandFreshGroup,
        visibleCountForGroup,
        refreshVisibleOrderedKeys,
        emptyMessageForCurrentFilter,
        syncFreshKeys,
        syncFreshGroupCounts,
        freshCountForGroup,
        renderHistoryHeader,
    };
}
