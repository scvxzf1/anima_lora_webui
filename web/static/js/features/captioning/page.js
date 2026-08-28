import { captioningApi } from './api.js?v=dragon-ui-20260829v12';
import { normalizePresets } from './presets.js?v=dragon-ui-20260829v12';
import { bindProviderPanel, renderProviderPanel } from './settings-panel.js?v=dragon-ui-20260829v12';
import { mountCaptioningSuite } from './suite.js?v=dragon-ui-20260829v12';

export async function loadCaptioning() {
    const [settings, jobs, engines, presetPayload, routing, workspace] = await Promise.all([captioningApi('/settings'), captioningApi('/jobs'), captioningApi('/engines'), captioningApi('/presets'), captioningApi('/routing'), captioningApi('/workspace')]);
    const prefill = readPrefill();
    const state = { active: true, settings, routing, workspace, engines, presets: normalizePresets(presetPayload), jobs: jobs.jobs || [], selectedJobId: '', selectedJob: null, selectedItemId: '', editorMode: 'pills', zoom: 1, governanceOpen: false, jobRequestId: 0, pollTimer: null, workspaceData: {} };
    return {
        html: `<div class="dragon-page dragon-page-wide dragon-caption-page" data-caption-page>${renderProviderPanel(settings, routing)}<div data-caption-suite-host></div></div>`,
        onMount(root) { state.root = root; mountCaptioningSuite(root, state, prefill); bindProviderPanel(root, state); },
        onUnmount() {
            state.active = false;
            state.jobRequestId += 1;
            if (state.pollTimer) window.clearTimeout(state.pollTimer);
            if (state.root) state.root.onkeydown = null;
        },
    };
}

function readPrefill() {
    try {
        const raw = sessionStorage.getItem('dragon-captioning-prefill');
        sessionStorage.removeItem('dragon-captioning-prefill');
        return raw ? JSON.parse(raw) : {};
    } catch { return {}; }
}
