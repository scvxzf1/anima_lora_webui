export function createVisibilityPoller({ poll, delay, onError = null }) {
    let timer = null;
    let running = null;
    let stopped = true;

    const clearTimer = () => {
        if (timer != null) window.clearTimeout(timer);
        timer = null;
    };

    const nextDelay = () => {
        const value = typeof delay === 'function' ? delay() : delay;
        return Math.max(0, Number(value) || 0);
    };

    const schedule = () => {
        clearTimer();
        if (stopped || document.hidden) return;
        timer = window.setTimeout(run, nextDelay());
    };

    const run = async () => {
        clearTimer();
        if (stopped || document.hidden) return;
        if (running) return running;
        running = Promise.resolve().then(poll);
        try {
            await running;
        } catch (error) {
            onError?.(error);
        } finally {
            running = null;
            schedule();
        }
    };

    const handleVisibility = () => {
        if (document.hidden) clearTimer();
        else void run();
    };

    return {
        start() {
            if (!stopped) return;
            stopped = false;
            document.addEventListener('visibilitychange', handleVisibility);
            schedule();
        },
        reschedule: schedule,
        stop() {
            if (stopped) return;
            stopped = true;
            clearTimer();
            document.removeEventListener('visibilitychange', handleVisibility);
        },
    };
}
