/**
 * Shared dialog helpers for native <dialog> UX consistency.
 */

import { renderIcon } from '../dragon-ui/icons.js?v=dragon-ui-20260812v35';

const DRAGON_DIALOG_SELECTOR = '[data-dragon-dialog-host]';
const dialogQueue = [];
let activeDialogRequest = null;
let dialogHost = null;
let dialogRequestId = 0;

/**
 * Bind "click the backdrop (dialog itself) to close" for browse/preview dialogs.
 * Confirm dialogs should NOT use this — keep them button/Esc only.
 *
 * Safe to call multiple times: uses a data flag to avoid duplicate listeners.
 *
 * @param {HTMLDialogElement | string | null} dialogOrId
 * @param {{ onClose?: () => void }} [options]
 * @returns {HTMLDialogElement | null}
 */
export function bindDialogBackdropClose(dialogOrId, options = {}) {
    const dialog = typeof dialogOrId === 'string'
        ? document.getElementById(dialogOrId)
        : dialogOrId;
    if (!dialog || dialog.dataset.backdropCloseBound === '1') return dialog || null;
    dialog.dataset.backdropCloseBound = '1';
    dialog.addEventListener('click', (event) => {
        if (event.target !== dialog) return;
        if (typeof dialog.close === 'function') {
            dialog.close();
        } else {
            dialog.hidden = true;
            dialog.removeAttribute('open');
            dialog.setAttribute('aria-hidden', 'true');
        }
        if (typeof options.onClose === 'function') options.onClose();
    });
    return dialog;
}

/**
 * Bind backdrop close for multiple browse dialogs by id.
 * Missing ids are skipped.
 *
 * @param {string[]} ids
 * @returns {number} number of dialogs bound
 */
export function bindBrowseDialogBackdropClose(ids = []) {
    let bound = 0;
    for (const id of ids) {
        if (bindDialogBackdropClose(id)) bound += 1;
    }
    return bound;
}

/**
 * Open a Dragon-styled modal and resolve with the selected action.
 *
 * This is deliberately a queued, singleton surface.  A host mounted on
 * document.body survives page transitions, so an async confirmation cannot be
 * orphaned when the router replaces the current page.
 *
 * @param {{mode?: 'confirm'|'prompt'|'alert', title?: string, eyebrow?: string,
 *   message?: string, description?: string, value?: string, placeholder?: string,
 *   label?: string, confirmText?: string, cancelText?: string, tone?: string,
 *   icon?: string}|string} options
 * @returns {Promise<boolean|string|null|undefined>}
 */
export function openDragonDialog(options = {}) {
    return enqueueDragonDialog(options, null);
}

/** @returns {Promise<boolean>} */
export function confirmDragonDialog(options = {}) {
    return enqueueDragonDialog(options, 'confirm');
}

/** @returns {Promise<string|null>} */
export function promptDragonDialog(options = {}) {
    return enqueueDragonDialog(options, 'prompt');
}

/** @returns {Promise<void>} */
export function alertDragonDialog(options = {}) {
    return enqueueDragonDialog(options, 'alert').then(() => undefined);
}

// Readable aliases for feature modules that prefer the verb-first naming style.
export const showDragonConfirm = confirmDragonDialog;
export const showDragonPrompt = promptDragonDialog;
export const showDragonAlert = alertDragonDialog;

function enqueueDragonDialog(options, mode) {
    const request = normalizeDialogRequest(options, mode);
    return new Promise((resolve) => {
        dialogQueue.push({ ...request, resolve, id: ++dialogRequestId });
        drainDialogQueue();
    });
}

function normalizeDialogRequest(options, mode) {
    const source = typeof options === 'string' ? { message: options } : (options || {});
    const selectedMode = mode || source.mode || 'confirm';
    const defaultTitle = selectedMode === 'prompt'
        ? '输入内容'
        : (selectedMode === 'alert' ? '提示' : '确认操作');
    const defaultConfirm = selectedMode === 'prompt'
        ? '确定'
        : (selectedMode === 'alert' ? '关闭' : '确认');
    const tone = ['neutral', 'info', 'success', 'warning', 'danger'].includes(source.tone)
        ? source.tone
        : 'neutral';
    return {
        mode: ['confirm', 'prompt', 'alert'].includes(selectedMode) ? selectedMode : 'confirm',
        title: String(source.title ?? defaultTitle),
        eyebrow: String(source.eyebrow ?? 'Dragon trainer'),
        message: String(source.message ?? ''),
        description: String(source.description ?? ''),
        value: String(source.value ?? ''),
        placeholder: String(source.placeholder ?? ''),
        label: String(source.label ?? '输入内容'),
        confirmText: String(source.confirmText ?? defaultConfirm),
        cancelText: String(source.cancelText ?? '取消'),
        tone,
        icon: String(source.icon ?? (selectedMode === 'prompt' ? 'edit' : 'circleHelp')),
    };
}

function drainDialogQueue() {
    if (activeDialogRequest || !dialogQueue.length) return;
    const request = dialogQueue.shift();
    activeDialogRequest = request;
    const dialog = ensureDialogHost();
    if (!dialog) {
        activeDialogRequest = null;
        request.resolve(cancelResult(request));
        drainDialogQueue();
        return;
    }
    renderDialog(dialog, request);
    startDialogRequest(dialog, request);
}

function ensureDialogHost() {
    if (typeof document === 'undefined' || !document.body) return null;
    if (dialogHost && document.body.contains(dialogHost)) return dialogHost;
    dialogHost = document.querySelector(DRAGON_DIALOG_SELECTOR);
    if (!dialogHost) {
        dialogHost = document.createElement('dialog');
        dialogHost.className = 'dragon-dialog-host';
        dialogHost.dataset.dragonDialogHost = 'true';
        document.body.appendChild(dialogHost);
    }
    dialogHost.classList.add('dragon-dialog-host');
    dialogHost.setAttribute('role', 'dialog');
    dialogHost.setAttribute('aria-modal', 'true');
    dialogHost.tabIndex = -1;
    return dialogHost;
}

function renderDialog(dialog, request) {
    dialog.dataset.mode = request.mode;
    dialog.dataset.tone = request.tone;
    dialog.returnValue = '';
    dialog.innerHTML = `
        <div class="dragon-dialog-shell">
            <header class="dragon-dialog-header">
                <div class="dragon-dialog-heading">
                    <span class="dragon-dialog-symbol" data-dragon-dialog-symbol></span>
                    <div>
                        <span class="dragon-eyebrow" data-dragon-dialog-eyebrow></span>
                        <h2 id="dragon-dialog-title" data-dragon-dialog-title></h2>
                    </div>
                </div>
                <button class="dragon-dialog-close" type="button" data-dragon-dialog-close aria-label="关闭弹窗" title="关闭">${renderIcon('x', 'dragon-dialog-close-icon')}</button>
            </header>
            <div class="dragon-dialog-body" id="dragon-dialog-body">
                <p class="dragon-dialog-message" data-dragon-dialog-message></p>
                <p class="dragon-dialog-description" data-dragon-dialog-description></p>
                <div class="dragon-dialog-field" data-dragon-dialog-field>
                    <label for="dragon-dialog-input" data-dragon-dialog-label></label>
                    <input id="dragon-dialog-input" class="dragon-input" type="text" autocomplete="off" />
                </div>
            </div>
            <footer class="dragon-dialog-actions">
                <button class="dragon-btn dragon-btn-secondary" type="button" data-dragon-dialog-cancel></button>
                <button class="dragon-btn dragon-btn-primary" type="button" data-dragon-dialog-confirm>${renderIcon('check', 'dragon-btn-icon')}<span data-dragon-dialog-confirm-text></span></button>
            </footer>
        </div>`;

    dialog.querySelector('[data-dragon-dialog-symbol]').innerHTML = renderIcon(request.icon, 'dragon-dialog-symbol-icon');
    dialog.querySelector('[data-dragon-dialog-eyebrow]').textContent = request.eyebrow;
    dialog.querySelector('[data-dragon-dialog-title]').textContent = request.title;
    const message = dialog.querySelector('[data-dragon-dialog-message]');
    message.textContent = request.message;
    message.hidden = !request.message;
    const description = dialog.querySelector('[data-dragon-dialog-description]');
    description.textContent = request.description;
    description.hidden = !request.description;
    const field = dialog.querySelector('[data-dragon-dialog-field]');
    const input = dialog.querySelector('#dragon-dialog-input');
    field.hidden = request.mode !== 'prompt';
    input.value = request.value;
    input.placeholder = request.placeholder;
    dialog.querySelector('[data-dragon-dialog-label]').textContent = request.label;
    const cancel = dialog.querySelector('[data-dragon-dialog-cancel]');
    cancel.textContent = request.cancelText;
    cancel.hidden = request.mode === 'alert';
    const confirm = dialog.querySelector('[data-dragon-dialog-confirm]');
    confirm.dataset.tone = request.tone;
    confirm.querySelector('[data-dragon-dialog-confirm-text]').textContent = request.confirmText;
    dialog.setAttribute('aria-labelledby', 'dragon-dialog-title');
    dialog.setAttribute('aria-describedby', 'dragon-dialog-body');
}

function startDialogRequest(dialog, request) {
    const previousFocus = readActiveElement();
    const input = dialog.querySelector('#dragon-dialog-input');
    const cancel = dialog.querySelector('[data-dragon-dialog-cancel]');
    const confirm = dialog.querySelector('[data-dragon-dialog-confirm]');
    let nativeDialog = supportsNativeDialog(dialog);
    let settled = false;
    const cleanups = [];
    const cancelCurrent = () => finish(cancelResult(request));
    const confirmCurrent = () => finish(request.mode === 'prompt' ? input.value : true);
    const finish = (result) => {
        if (settled) return;
        settled = true;
        cleanups.splice(0).forEach((cleanup) => cleanup());
        if (nativeDialog && dialog.open) {
            dialog.returnValue = result === true ? 'confirm' : 'cancel';
            try { dialog.close(); } catch { /* already closed by the browser */ }
        }
        dialog.removeAttribute('data-fallback-open');
        dialog.removeAttribute('open');
        dialog.hidden = true;
        dialog.setAttribute('aria-hidden', 'true');
        if (activeDialogRequest === request) activeDialogRequest = null;
        request.resolve(result);
        restoreFocus(previousFocus);
        drainDialogQueue();
    };
    const onClose = () => cancelCurrent();
    const onCancel = (event) => { event.preventDefault(); cancelCurrent(); };
    const onBackdropClick = (event) => { if (event.target === dialog) cancelCurrent(); };
    const onEscape = (event) => {
        if (!nativeDialog && event.key === 'Escape') {
            event.preventDefault();
            cancelCurrent();
        }
    };
    const onInputKeydown = (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            confirmCurrent();
        } else if (event.key === 'Escape') {
            event.preventDefault();
            cancelCurrent();
        }
    };
    addListener(dialog, 'close', onClose, cleanups);
    addListener(dialog, 'cancel', onCancel, cleanups);
    addListener(dialog, 'click', onBackdropClick, cleanups);
    addListener(document, 'keydown', onEscape, cleanups, true);
    addListener(input, 'keydown', onInputKeydown, cleanups);
    addListener(dialog.querySelector('[data-dragon-dialog-close]'), 'click', cancelCurrent, cleanups);
    addListener(cancel, 'click', cancelCurrent, cleanups);
    addListener(confirm, 'click', confirmCurrent, cleanups);

    dialog.hidden = false;
    dialog.removeAttribute('aria-hidden');
    if (nativeDialog) {
        try {
            dialog.showModal();
        } catch {
            nativeDialog = false;
        }
    }
    if (!nativeDialog) {
        dialog.dataset.fallbackOpen = 'true';
        dialog.setAttribute('open', '');
        dialog.setAttribute('aria-hidden', 'false');
    }
    focusDialogControl(request, input, cancel, confirm);
}

function addListener(target, eventName, handler, cleanups, capture = false) {
    if (!target?.addEventListener) return;
    target.addEventListener(eventName, handler, capture);
    cleanups.push(() => target.removeEventListener(eventName, handler, capture));
}

function supportsNativeDialog(dialog) {
    return typeof dialog?.showModal === 'function' && typeof dialog?.close === 'function';
}

function focusDialogControl(request, input, cancel, confirm) {
    const target = request.mode === 'prompt' ? input : (request.mode === 'alert' ? confirm : cancel);
    Promise.resolve().then(() => {
        if (!target || target.hidden || typeof target.focus !== 'function') return;
        target.focus();
        if (request.mode === 'prompt' && typeof target.select === 'function') target.select();
    });
}

function readActiveElement() {
    if (typeof document === 'undefined') return null;
    const element = document.activeElement;
    return element && typeof element.focus === 'function' ? element : null;
}

function restoreFocus(element) {
    if (!element || element === dialogHost || !element.isConnected || element.disabled) return;
    try { element.focus(); } catch { /* focus target disappeared */ }
}

function cancelResult(request) {
    return request.mode === 'prompt' ? null : (request.mode === 'alert' ? undefined : false);
}
