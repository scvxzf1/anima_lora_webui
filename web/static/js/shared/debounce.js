/**
 * Shared debounce / throttle helpers for WebUI input and high-frequency UI paths.
 */

/**
 * Debounce a function: invoke `fn` once after `waitMs` of quiet.
 * Leading-edge is not used; trailing-edge only (good for search boxes).
 *
 * @template {(...args: any[]) => any} T
 * @param {T} fn
 * @param {number} waitMs
 * @returns {T & { cancel: () => void; flush: () => void }}
 */
export function debounce(fn, waitMs = 150) {
    let timer = null;
    let lastArgs = null;
    let lastThis = null;

    function invoke() {
        timer = null;
        if (!lastArgs) return;
        const args = lastArgs;
        const ctx = lastThis;
        lastArgs = null;
        lastThis = null;
        fn.apply(ctx, args);
    }

    function debounced(...args) {
        lastArgs = args;
        lastThis = this;
        if (timer != null) window.clearTimeout(timer);
        timer = window.setTimeout(invoke, Math.max(0, Number(waitMs) || 0));
    }

    debounced.cancel = () => {
        if (timer != null) window.clearTimeout(timer);
        timer = null;
        lastArgs = null;
        lastThis = null;
    };

    debounced.flush = () => {
        if (timer == null) return;
        window.clearTimeout(timer);
        invoke();
    };

    return debounced;
}

/**
 * Throttle a function: at most one call per `waitMs`.
 * Uses trailing call with latest args so the final value is not lost.
 *
 * @template {(...args: any[]) => any} T
 * @param {T} fn
 * @param {number} waitMs
 * @returns {T & { cancel: () => void }}
 */
export function throttle(fn, waitMs = 100) {
    let timer = null;
    let lastInvoke = 0;
    let pendingArgs = null;
    let pendingThis = null;

    function run(args, ctx) {
        lastInvoke = Date.now();
        pendingArgs = null;
        pendingThis = null;
        fn.apply(ctx, args);
    }

    function throttled(...args) {
        const now = Date.now();
        const remaining = Math.max(0, Number(waitMs) || 0) - (now - lastInvoke);
        pendingArgs = args;
        pendingThis = this;
        if (remaining <= 0) {
            if (timer != null) {
                window.clearTimeout(timer);
                timer = null;
            }
            run(args, this);
            return;
        }
        if (timer == null) {
            timer = window.setTimeout(() => {
                timer = null;
                if (pendingArgs) run(pendingArgs, pendingThis);
            }, remaining);
        }
    }

    throttled.cancel = () => {
        if (timer != null) window.clearTimeout(timer);
        timer = null;
        pendingArgs = null;
        pendingThis = null;
    };

    return throttled;
}
