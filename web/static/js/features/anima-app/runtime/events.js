export function createRuntimeEvents() {
    const cleanups = new Set();

    function addCleanup(cleanup) {
        if (typeof cleanup === 'function') {
            cleanups.add(cleanup);
        }
        return cleanup;
    }

    return {
        addCleanup,
        listen(target, eventName, handler, options) {
            if (!target?.addEventListener) return null;
            target.addEventListener(eventName, handler, options);
            return addCleanup(() => target.removeEventListener(eventName, handler, options));
        },
        clear() {
            for (const cleanup of [...cleanups]) {
                cleanup();
                cleanups.delete(cleanup);
            }
        },
    };
}
