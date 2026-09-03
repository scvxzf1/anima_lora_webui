/* Cross-route state for the tagging workbench and its dedicated tools. */

const WORKSPACE_KEY = 'dragon-tagging-workspace-v2';
const RETURN_KEY = 'dragon-tagging-return-v1';
const MAX_SELECTED_FILES = 500;

let memorySnapshot = null;

export function readTaggingWorkspaceState() {
    if (memorySnapshot) return cloneSnapshot(memorySnapshot, true);
    const stored = readSession(WORKSPACE_KEY);
    return stored ? normalizeSnapshot(stored, false) : {};
}

export function saveTaggingWorkspaceState(state, { capturePosition = false } = {}) {
    if (!state || typeof state !== 'object') return;
    const root = state.root || null;
    const grid = root?.querySelector?.('[data-tagging-image-grid]');
    const snapshot = normalizeSnapshot({
        datasetFile: state.datasetFile,
        datasetIndex: state.datasetIndex,
        source: state.source,
        rows: state.rows,
        images: state.images,
        imagesLoaded: state.imagesLoaded,
        directory: state.directory,
        total: state.total,
        nextOffset: state.nextOffset,
        hasMore: state.hasMore,
        selectedFiles: Array.from(state.selectedFiles || []),
        systemPrompt: state.systemPrompt,
        userPrompt: state.userPrompt,
        currentPresetId: state.currentPresetId,
        providerProfileId: state.activeProfileId || state.providerProfileId,
        sourceExpanded: state.sourceExpanded,
        job: state.job,
        jobId: state.job?.id || state.jobId,
        scrollY: capturePosition ? Number(globalThis.scrollY || 0) : state.savedScrollY,
        gridScrollTop: capturePosition ? Number(grid?.scrollTop || 0) : state.gridScrollTop,
        focusTarget: readFocusTarget(root),
    }, true);
    memorySnapshot = snapshot;
    writeSession(WORKSPACE_KEY, serializeSnapshot(snapshot));
}

export function openTaggingTool(state, page) {
    saveTaggingWorkspaceState(state, { capturePosition: true });
    writeSession(RETURN_KEY, {
        hash: globalThis.location?.hash || '#page/captioning',
        scrollY: Number(globalThis.scrollY || 0),
        gridScrollTop: Number(state.root?.querySelector?.('[data-tagging-image-grid]')?.scrollTop || 0),
        focusTarget: readFocusTarget(state.root),
    });
    if (globalThis.location) globalThis.location.hash = `#page/${page}`;
}

export function returnToTaggingWorkspace() {
    const target = readSession(RETURN_KEY)?.hash || '#page/captioning';
    if (globalThis.location) globalThis.location.hash = target;
}

export function restoreTaggingWorkspacePosition(root) {
    const saved = readSession(RETURN_KEY);
    const snapshot = readTaggingWorkspaceState();
    if (!saved && !snapshot.scrollY && !snapshot.gridScrollTop && !snapshot.focusTarget) return;
    const restore = () => {
        const grid = root?.querySelector?.('[data-tagging-image-grid]');
        if (grid) grid.scrollTop = Number(saved?.gridScrollTop ?? snapshot.gridScrollTop ?? 0);
        globalThis.scrollTo?.({ top: Number(saved?.scrollY ?? snapshot.scrollY ?? 0), behavior: 'auto' });
        restoreFocus(root, saved?.focusTarget || snapshot.focusTarget || '');
    };
    restore();
    globalThis.requestAnimationFrame?.(restore);
    try {
        globalThis.sessionStorage?.removeItem(RETURN_KEY);
    } catch {
        // Session storage is optional in embedded WebViews.
    }
}

export function updateTaggingPromptDraft(values = {}) {
    const current = readTaggingWorkspaceState();
    memorySnapshot = normalizeSnapshot({ ...current, ...values }, true);
    writeSession(WORKSPACE_KEY, serializeSnapshot(memorySnapshot));
}

export function updateTaggingProviderProfile(profileId) {
    const current = readTaggingWorkspaceState();
    memorySnapshot = normalizeSnapshot({ ...current, providerProfileId: profileId }, true);
    writeSession(WORKSPACE_KEY, serializeSnapshot(memorySnapshot));
}

function normalizeSnapshot(value, includeMemory) {
    const source = value && typeof value === 'object' ? value : {};
    const selectedFiles = Array.isArray(source.selectedFiles)
        ? source.selectedFiles.map(cleanFile).filter(Boolean).slice(0, MAX_SELECTED_FILES)
        : [];
    const snapshot = {
        datasetFile: clean(source.datasetFile, 2048),
        datasetIndex: normalizeIndex(source.datasetIndex),
        source: source.source === 'training' ? 'training' : 'source',
        directory: clean(source.directory, 4096),
        total: nonNegative(source.total),
        nextOffset: nonNegative(source.nextOffset),
        hasMore: source.hasMore === true,
        selectedFiles,
        systemPrompt: clean(source.systemPrompt, 10000),
        userPrompt: clean(source.userPrompt, 10000),
        currentPresetId: clean(source.currentPresetId, 64),
        providerProfileId: clean(source.providerProfileId, 64),
        sourceExpanded: source.sourceExpanded === true,
        jobId: clean(source.jobId || source.job?.id, 64),
        scrollY: nonNegative(source.scrollY),
        gridScrollTop: nonNegative(source.gridScrollTop),
        focusTarget: clean(source.focusTarget || (source.focusFile ? `image:${source.focusFile}` : ''), 8192),
    };
    if (includeMemory) {
        snapshot.rows = Array.isArray(source.rows) ? source.rows : [];
        snapshot.images = Array.isArray(source.images) ? source.images : [];
        snapshot.imagesLoaded = source.imagesLoaded === true;
        snapshot.job = source.job && typeof source.job === 'object' ? source.job : null;
    }
    return snapshot;
}

function serializeSnapshot(snapshot) {
    const { rows, images, imagesLoaded, job, ...serializable } = snapshot;
    return serializable;
}

function cloneSnapshot(snapshot, includeMemory) {
    return normalizeSnapshot({
        ...snapshot,
        selectedFiles: [...(snapshot.selectedFiles || [])],
        rows: includeMemory ? [...(snapshot.rows || [])] : undefined,
        images: includeMemory ? [...(snapshot.images || [])] : undefined,
    }, includeMemory);
}

function readSession(key) {
    try {
        const raw = globalThis.sessionStorage?.getItem(key) || '';
        const value = raw ? JSON.parse(raw) : null;
        return value && typeof value === 'object' ? value : null;
    } catch {
        return null;
    }
}

function writeSession(key, value) {
    try {
        globalThis.sessionStorage?.setItem(key, JSON.stringify(value));
    } catch {
        // In-memory state still preserves navigation within the current page.
    }
}

function clean(value, maxLength) {
    return String(value ?? '').trim().slice(0, maxLength);
}

function cleanFile(value) {
    return clean(value, 4096);
}

function normalizeIndex(value) {
    const number = Number(value);
    return Number.isInteger(number) && number >= 0 ? number : 0;
}

function nonNegative(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(0, number) : 0;
}

function readFocusTarget(root) {
    const active = root && globalThis.document?.activeElement;
    if (!active || !root.contains?.(active)) return '';
    const tool = active.dataset?.taggingOpenTool;
    if (tool) return `tool:${tool}`;
    const file = active.dataset?.file;
    if (file) return `image:${file}`;
    if (active.matches?.('[data-tagging-source-details] > summary')) return 'source';
    return '';
}

function restoreFocus(root, target) {
    if (!root || !target) return;
    let element = null;
    if (target.startsWith('tool:')) {
        const tool = target.slice(5);
        element = [...(root.querySelectorAll?.('[data-tagging-open-tool]') || [])]
            .find((node) => node.dataset?.taggingOpenTool === tool) || null;
    } else if (target.startsWith('image:')) {
        const file = target.slice(6);
        element = [...(root.querySelectorAll?.('[data-tagging-image]') || [])]
            .find((node) => node.dataset?.file === file) || null;
    } else if (target === 'source') {
        element = root.querySelector?.('[data-tagging-source-details] > summary') || null;
    }
    if (!element) element = root.querySelector?.('[data-tagging-source-details] > summary') || root.querySelector?.('h1');
    element?.focus?.({ preventScroll: true });
}
