/**
 * Chunked DOM append helpers for large history workbench lists.
 * Prefer incremental append over one-shot virtualization for this round.
 */

export const HISTORY_RENDER_CHUNK_SIZE = 24;

/**
 * @param {ParentNode} parent
 * @param {Node[]} nodes
 * @param {{ chunkSize?: number, signal?: { cancelled: boolean }, onDone?: () => void }} [options]
 * @returns {{ cancel: () => void, done: Promise<void> }}
 */
export function appendNodesInChunks(parent, nodes, options = {}) {
    const chunkSize = Math.max(1, Number(options.chunkSize) || HISTORY_RENDER_CHUNK_SIZE);
    const list = Array.isArray(nodes) ? nodes.filter(Boolean) : [];
    const signal = options.signal || { cancelled: false };
    let index = 0;
    let resolveDone;
    const done = new Promise((resolve) => {
        resolveDone = resolve;
    });

    const finish = () => {
        if (typeof options.onDone === 'function') {
            try {
                options.onDone();
            } catch {
                // ignore callback errors
            }
        }
        resolveDone();
    };

    if (!parent || !list.length) {
        finish();
        return {
            cancel: () => {
                signal.cancelled = true;
            },
            done,
        };
    }

    const pump = () => {
        if (signal.cancelled) {
            finish();
            return;
        }
        const fragment = document.createDocumentFragment();
        const end = Math.min(index + chunkSize, list.length);
        for (; index < end; index += 1) {
            fragment.appendChild(list[index]);
        }
        parent.appendChild(fragment);
        if (index >= list.length) {
            finish();
            return;
        }
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(pump);
        } else {
            setTimeout(pump, 0);
        }
    };

    pump();
    return {
        cancel: () => {
            signal.cancelled = true;
        },
        done,
    };
}

/**
 * Build nodes eagerly, then append in chunks to keep first paint lighter.
 * @param {ParentNode} parent
 * @param {unknown[]} items
 * @param {(item: unknown, index: number) => Node | null | undefined} createNode
 * @param {{ chunkSize?: number, signal?: { cancelled: boolean }, onDone?: () => void }} [options]
 */
export function renderItemsInChunks(parent, items, createNode, options = {}) {
    const source = Array.isArray(items) ? items : [];
    const chunkSize = Math.max(1, Number(options.chunkSize) || HISTORY_RENDER_CHUNK_SIZE);
    const signal = options.signal || { cancelled: false };
    let index = 0;
    let resolveDone;
    const done = new Promise((resolve) => {
        resolveDone = resolve;
    });

    const finish = () => {
        if (typeof options.onDone === 'function') {
            try {
                options.onDone();
            } catch {
                // ignore callback errors
            }
        }
        resolveDone();
    };

    if (!parent || !source.length) {
        finish();
        return {
            cancel: () => {
                signal.cancelled = true;
            },
            done,
        };
    }

    const pump = () => {
        if (signal.cancelled) {
            finish();
            return;
        }
        const fragment = document.createDocumentFragment();
        const end = Math.min(index + chunkSize, source.length);
        for (; index < end; index += 1) {
            const node = createNode(source[index], index);
            if (node) fragment.appendChild(node);
        }
        parent.appendChild(fragment);
        if (index >= source.length) {
            finish();
            return;
        }
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(pump);
        } else {
            setTimeout(pump, 0);
        }
    };

    pump();
    return {
        cancel: () => {
            signal.cancelled = true;
        },
        done,
    };
}
