/**
 * File-group drag/drop public surface.
 * Split into file-group-drag-core.js and file-group-drag-targets.js.
 */
export {
    eventTargetClosest,
    originClosest,
    fileGroupDropTargetPriority,
    removeFileGroupDragImage,
    setFileGroupDragData,
    canBeginFileGroupDrag,
    beginFileGroupDrag,
    createFileGroupPointerDragImage,
    moveFileGroupPointerDragImage,
    registerFileGroupDropTarget,
    autoScrollFileGroupPointerDrag,
    markFileGroupDropTarget,
    clearFileGroupDropTarget,
    createFileGroupDragHandle,
    finishFileGroupDrag,
    configFileDropIndex,
    configGroupDropIndex,
    fileGroupContainsRelatedTarget,
} from './file-group-drag-core.js?v=module-bootstrap-20260711-ir6';

export {
    setupFileGroupRowDropTarget,
    setupFileGroupListDropTarget,
    setupFileGroupHeaderDropTarget,
} from './file-group-drag-targets.js?v=module-bootstrap-20260711-ir6';

// file-group-drag module end
