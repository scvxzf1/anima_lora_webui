/**
 * Compatibility barrel for former chunk 07.
 * Side-effect imports register dataset picker dialog / preset list renderers.
 * Drag primitives live in toml-manager/file-group-drag.js.
 */
import '../../config-form/dataset-picker-dialog.js?v=module-bootstrap-20260809-nf4-v2';
import '../../dataset-editor/preset-page.js?v=module-bootstrap-20260809-nf4-v2';
export {
    createDatasetEditor,
    updateDatasetPresetPageSummary,
} from '../../dataset-editor/preset-page.js?v=module-bootstrap-20260809-nf4-v2';
export {
    beginFileGroupDrag,
    canBeginFileGroupDrag,
    createFileGroupPointerDragImage,
    eventTargetClosest,
    moveFileGroupPointerDragImage,
    registerFileGroupDropTarget,
    removeFileGroupDragImage,
    setFileGroupDragData,
} from '../../toml-manager/file-group-drag.js?v=module-bootstrap-20260809-nf4-v2';
