/* Shared viewport contracts for JavaScript behavior. Keep the matching CSS
 * media queries beside their components; JavaScript imports these names so
 * behavior does not scatter raw breakpoint values across feature modules.
 */

export const DRAGON_VIEWPORT_QUERIES = Object.freeze({
    mobileNavigation: '(max-width: 833px)',
    trainingPresetSidebar: '(min-width: 1001px)',
});

export function matchesDragonViewport(query, fallback = false) {
    if (typeof window.matchMedia !== 'function') return fallback;
    return window.matchMedia(query).matches;
}
