/**
 * Shared dialog helpers for native <dialog> UX consistency.
 */

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
