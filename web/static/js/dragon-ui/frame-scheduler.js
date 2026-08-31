/* Coalesce repeated UI work to one callback per animation frame. */

export function createFrameScheduler(callback, view = globalThis) {
    let frame = null;
    let pendingArgs = [];
    let animationFrame = false;
    const flush = () => {
        frame = null;
        const args = pendingArgs;
        pendingArgs = [];
        callback(...args);
    };
    const schedule = (...args) => {
        pendingArgs = args;
        if (frame != null) return;
        animationFrame = typeof view?.requestAnimationFrame === 'function';
        frame = animationFrame ? view.requestAnimationFrame(flush) : view.setTimeout(flush, 0);
    };
    const cancel = () => {
        if (frame == null) return;
        if (animationFrame) view.cancelAnimationFrame?.(frame);
        else view.clearTimeout?.(frame);
        frame = null;
        pendingArgs = [];
    };
    return { schedule, cancel };
}
