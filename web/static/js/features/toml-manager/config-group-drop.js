/**
 * Config group drop-target public surface.
 * Dataset editor rendering moved to dataset-editor/dataset-render.js.
 */
export { setupConfigGroupDropTarget } from './config-group-drop-target.js?v=module-bootstrap-20260711-ir1';

// Keep dataset render bridge side effects loaded for existing chunk/feature imports.
import '../dataset-editor/dataset-render.js?v=module-bootstrap-20260711-ir1';
