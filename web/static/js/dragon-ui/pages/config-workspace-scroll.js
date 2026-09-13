import { dragonScrollBehavior } from '../motion.js?v=dragon-ui-20260824v1';

export function configScrollContainer(canvas) {
    return getComputedStyle(canvas).overflowY === 'visible' ? window : canvas;
}

export function configScrollPosition(canvas) {
    const container = configScrollContainer(canvas);
    return container === window ? window.scrollY : canvas.scrollTop;
}

export function restoreConfigScroll(canvas, top) {
    configScrollContainer(canvas).scrollTo({ top, behavior: 'instant' });
}

function viewportBounds(canvas) {
    if (configScrollContainer(canvas) !== window) return canvas.getBoundingClientRect();
    const detail = canvas.closest('.dragon-config-all-detail');
    const toolbar = detail?.querySelector('.dragon-config-all-toolbar');
    const footer = detail?.querySelector('.dragon-config-all-footer');
    const toolbarTop = toolbar ? Number.parseFloat(getComputedStyle(toolbar).top) || 0 : 0;
    return {
        top: toolbarTop + (toolbar?.getBoundingClientRect().height || 0),
        bottom: Math.min(window.innerHeight, footer?.getBoundingClientRect().top ?? window.innerHeight),
    };
}

function configCanvasPadding(canvas) {
    return Number.parseFloat(getComputedStyle(canvas).scrollPaddingTop) || 12;
}

export function scrollConfigCanvasTo(canvas, target, behavior = dragonScrollBehavior()) {
    if (!canvas || !target) return;
    const container = configScrollContainer(canvas);
    const top = configScrollPosition(canvas) + target.getBoundingClientRect().top
        - viewportBounds(canvas).top - configCanvasPadding(canvas);
    container.scrollTo({ top: Math.max(0, top), behavior });
}

export function bindConfigWorkspaceScroll(root, onRadarChange) {
    const canvas = root.querySelector('#dragon-config-fields');
    if (!canvas) return () => {};
    const dividers = [...canvas.querySelectorAll('[data-config-section-divider]')];
    let frame = 0;
    let activeRadar = '';
    const syncRadar = () => {
        frame = 0;
        const threshold = Math.ceil(viewportBounds(canvas).top) + configCanvasPadding(canvas);
        const visible = dividers.filter((divider) => !divider.closest('[data-config-section]')?.hidden);
        let active = visible[0]?.dataset.configSectionDivider || 'input';
        for (const divider of visible) {
            if (divider.getBoundingClientRect().top > threshold + 1) break;
            active = divider.dataset.configSectionDivider;
        }
        if (active === activeRadar) return;
        activeRadar = active;
        onRadarChange?.(active);
    };
    const scheduleRadar = () => {
        if (!frame) frame = window.requestAnimationFrame(syncRadar);
    };
    const keepFocusVisible = (event) => {
        const block = event.target?.closest?.('.dragon-config-block');
        if (!block) return;
        const bounds = viewportBounds(canvas);
        const rect = block.getBoundingClientRect();
        const padding = configCanvasPadding(canvas);
        if (rect.top < bounds.top + padding || rect.bottom > bounds.bottom - padding) {
            scrollConfigCanvasTo(canvas, block);
        }
    };
    canvas.addEventListener('scroll', scheduleRadar, { passive: true });
    window.addEventListener('scroll', scheduleRadar, { passive: true });
    window.addEventListener('resize', scheduleRadar);
    canvas.addEventListener('focusin', keepFocusVisible);
    syncRadar();
    return () => {
        canvas.removeEventListener('scroll', scheduleRadar);
        window.removeEventListener('scroll', scheduleRadar);
        window.removeEventListener('resize', scheduleRadar);
        canvas.removeEventListener('focusin', keepFocusVisible);
        if (frame) window.cancelAnimationFrame(frame);
    };
}
