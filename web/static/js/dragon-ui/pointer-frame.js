/* Frame-batched pointer movement for chart hover interactions. */

import { createFrameScheduler } from './frame-scheduler.js?v=dragon-ui-20260826v1';

export function bindLatestPointerMove(target, callback) {
    if (!target) return () => {};
    const view = target.ownerDocument?.defaultView;
    const eventName = pointerEventName(target, 'pointermove', 'mousemove');
    const scheduler = createFrameScheduler(callback, view);
    const onMove = (event) => {
        scheduler.schedule({ clientX: event.clientX, clientY: event.clientY, target: event.target });
    };
    target.addEventListener(eventName, onMove, { passive: true });
    return () => {
        target.removeEventListener(eventName, onMove);
        scheduler.cancel();
    };
}

export function pointerEventName(target, pointerName, mouseName) {
    return typeof target?.ownerDocument?.defaultView?.PointerEvent === 'function' ? pointerName : mouseName;
}
