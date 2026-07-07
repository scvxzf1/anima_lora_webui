function domMethod(ctx, name, fallback) {
    return ctx.dom?.[name]?.bind(ctx.dom) || fallback;
}

export function createRuntimeDom(ctx) {
    return Object.freeze({
        byId: domMethod(ctx, 'byId', (id) => document.getElementById(id)),
        optionalById: domMethod(ctx, 'optionalById', (id) => document.getElementById(id)),
        requireById: domMethod(ctx, 'requireById', (id) => {
            const el = document.getElementById(id);
            if (el) return el;
            throw new Error(`[webui-dom-contract] missing required DOM node: #${id}`);
        }),
        bindEvent: domMethod(ctx, 'bindEvent', (id, eventName, handler, options = {}) => {
            const el = document.getElementById(id);
            if (!el) return false;
            el.addEventListener(eventName, handler, options.listenerOptions);
            return true;
        }),
        val: domMethod(ctx, 'val', (id) => document.getElementById(id)?.value || ''),
        setText: domMethod(ctx, 'setText', (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        }),
        setButtonDisabled: domMethod(ctx, 'setButtonDisabled', (id, disabled) => {
            const btn = document.getElementById(id);
            if (btn) btn.disabled = Boolean(disabled);
        }),
        populateSelect: domMethod(ctx, 'populateSelect', () => {}),
        copyText: domMethod(ctx, 'copyText', async () => {}),
    });
}
