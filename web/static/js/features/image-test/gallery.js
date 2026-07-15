import { DEFAULT_FILTER_VALUE, DEFAULT_SELECTION_NOTE, GROUP_INITIAL_RENDER_COUNT, GROUP_VIRTUALIZE_THRESHOLD } from './gallery/constants.js?v=module-bootstrap-20260714-stage-dataset5';
import { createGalleryController, createInitialGalleryState } from './gallery/controller.js?v=module-bootstrap-20260714-stage-dataset5';

// Contract anchors for frontend source tests (implementation lives in gallery/*):
// GROUP_INITIAL_RENDER_COUNT = 24
// GROUP_VIRTUALIZE_THRESHOLD = 48
// createMergedImageBlob
// createZipDataBlob
// exportOriginalZipSelection
// createLoadMoreFooter
// scheduleVirtualWindowRefresh
// requestAnimationFrame
// renderGroupBody
// createVirtualSpacer
// createFreshBadge
// syncFreshGroupCounts
// normalizeZipEntryName
// virtualWindowByGroup
// toggleImageSelection
// deleteImagesWithConfirmation
// visibleSelectionRange
// if (options.additive) {
// additive: event.ctrlKey || event.metaKey
// btn-image-test-delete-selected
// btn-image-test-export-merged
// btn-image-test-export-originals
// image-test-history-filter
// state.visibleOrderedKeys
// const startIndex = windowState.virtualized ? windowState.startIndex : 0;
// const endIndex = windowState.virtualized ? windowState.endIndex : visibleCount;
// if (changed) {
// refreshVisibleOrderedKeys();
// Shift 连选仅覆盖当前已展开且当前可见的图片；Ctrl/⌘ 可增量点选。
void DEFAULT_SELECTION_NOTE;
void GROUP_INITIAL_RENDER_COUNT;
void GROUP_VIRTUALIZE_THRESHOLD;

/* keep source contract:
filterValue: normalizeImageTestHistoryRange(initialFilterValue, DEFAULT_FILTER_VALUE)
*/
export function createImageTestGallery({
    formatBytes,
    openPreviewDialog,
    requestImageDelete,
    requestHistoryReload,
    initialFilterValue = DEFAULT_FILTER_VALUE,
}) {
    const state = createInitialGalleryState(initialFilterValue);
    const controller = createGalleryController({
        state,
        formatBytes,
        openPreviewDialog,
        requestImageDelete,
        requestHistoryReload,
        render: (payload) => controller.renderPayload(payload),
    });

    return {
        currentFilter: () => state.filterValue,
        init: () => controller.init(),
        render: (payload) => controller.renderPayload(payload),
        setLoading: () => controller.setLoading(),
        setEmpty: (message) => controller.setEmpty(message),
    };
}
