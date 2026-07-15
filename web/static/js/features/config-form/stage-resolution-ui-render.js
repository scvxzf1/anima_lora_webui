/**
 * Stage schedule dialog re-render host.
 * Keeps widgets <-> dialog free of circular imports while still allowing
 * mutation handlers to request a full dialog refresh.
 */
let stageResolutionRenderer = null;

export function registerStageResolutionRenderer(renderer) {
    stageResolutionRenderer = typeof renderer === 'function' ? renderer : null;
}

export function requestStageResolutionRender() {
    if (typeof stageResolutionRenderer === 'function') {
        stageResolutionRenderer();
    }
}
