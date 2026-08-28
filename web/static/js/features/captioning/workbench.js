import { captioningApi, jsonOptions } from './api.js?v=dragon-ui-20260829v12';
import { bindControlBar, renderControlBar } from './control-bar.js?v=dragon-ui-20260829v12';
import { bindGallery, renderGallery } from './gallery.js?v=dragon-ui-20260829v12';
import { bindGovernance, renderGovernance } from './governance.js?v=dragon-ui-20260829v12';
import { bindInspector, renderInspector } from './inspector.js?v=dragon-ui-20260829v12';
import { escapeAttribute, escapeHtml, showFeedback, withBusy } from './utils.js?v=dragon-ui-20260829v12';

export function renderWorkbench(state, prefill = {}) {
    const item = selectedItem(state);
    const selectedCount = (state.workspaceData.gallerySelected || []).length;
    const job = state.selectedJob;
    return `<div class="dragon-caption-workbench" data-caption-workbench>
        ${renderControlBar(state, prefill)}
        <div class="dragon-caption-main">${renderGallery(state)}${renderInspector(state.selectedJob, item, state.editorMode, state.zoom)}</div>
        <footer class="dragon-caption-footer"><div class="dragon-caption-job-context"><label>当前任务 <select class="dragon-select" data-caption-job-select>${jobOptions(state.jobs, state.selectedJobId)}</select></label><span>${job ? `${escapeHtml(job.directory)} · ${job.completed || 0}/${job.total || job.results?.length || 0}` : '未选择任务'}</span></div><div class="dragon-caption-bulk-actions"><span class="dragon-caption-selection-count" aria-live="polite">已选 ${selectedCount} 项</span><button class="dragon-btn dragon-btn-secondary" type="button" data-caption-governance-open ${job ? '' : 'disabled'}>词频统计 / 查找替换</button><button class="dragon-btn dragon-btn-secondary" type="button" data-caption-commit-selected ${selectedCount ? '' : 'disabled'}>写回已选 ${selectedCount ? `(${selectedCount})` : ''}</button><button class="dragon-btn dragon-btn-primary" type="button" data-caption-commit-all ${job ? '' : 'disabled'}>覆盖写回全部</button></div></footer>
        ${state.governanceOpen ? renderGovernance(state.selectedJob) : ''}
    </div>`;
}

export function mountWorkbench(root, state, prefill = {}) {
    const host = root.querySelector('[data-caption-workbench-host]');
    const actions = {
        refresh: (selected) => refreshJobs(root, state, actions, selected, prefill),
        selectItem: (itemId) => { state.selectedItemId = itemId; state.zoom = 1; renderInto(root, state, actions, prefill); },
        renderInspector: () => renderInto(root, state, actions, prefill),
        renderWorkspace: () => renderInto(root, state, actions, prefill),
    };
    state.actions = actions;
    host.innerHTML = renderWorkbench(state, prefill);
    bindAll(root, state, actions, prefill);
    if (state.jobs.length && state.selectedJob?.id !== state.selectedJobId) selectJob(root, state, actions, state.selectedJobId || state.jobs[0].id, prefill).catch((error) => showFeedback(root, error.message, 'error'));
}

async function refreshJobs(root, state, actions, loadSelected, prefill) {
    const payload = await captioningApi('/jobs');
    state.jobs = payload.jobs || [];
    if (loadSelected && state.selectedJobId) await selectJob(root, state, actions, state.selectedJobId, prefill);
    else renderInto(root, state, actions, prefill);
}

async function selectJob(root, state, actions, jobId, prefill) {
    state.selectedJobId = jobId;
    const requestId = ++state.jobRequestId;
    const payload = await captioningApi(`/jobs/${encodeURIComponent(jobId)}`);
    if (!state.active || requestId !== state.jobRequestId || state.selectedJobId !== jobId) return;
    state.selectedJob = payload.job;
    if (!payload.job.results.some((item) => item.id === state.selectedItemId)) state.selectedItemId = payload.job.results[0]?.id || '';
    renderInto(root, state, actions, prefill);
    schedulePoll(root, state, actions, prefill);
}

function renderInto(root, state, actions, prefill) {
    const host = root.querySelector('[data-caption-workbench-host]');
    host.innerHTML = renderWorkbench(state, prefill);
    bindAll(root, state, actions, prefill);
}

function bindAll(root, state, actions, prefill) {
    bindControlBar(root, state, actions);
    bindGallery(root, state, actions.selectItem);
    bindInspector(root, state, actions);
    bindGovernance(root, state, actions);
    root.querySelector('[data-caption-job-select]')?.addEventListener('change', (event) => selectJob(root, state, actions, event.target.value, prefill).catch((error) => showFeedback(root, error.message, 'error')));
    root.querySelector('[data-caption-governance-open]')?.addEventListener('click', () => { state.governanceOpen = true; renderInto(root, state, actions, prefill); });
    root.querySelector('[data-caption-commit-all]')?.addEventListener('click', async (event) => {
        const count = state.selectedJob?.results?.filter((item) => ['ready', 'committed'].includes(item.state)).length || 0;
        if (!window.confirm(`将覆盖写回 ${count} 个 .txt 标注文件。是否继续？`)) return;
        await withBusy(event.currentTarget, async () => {
            try {
                const response = await captioningApi(`/jobs/${encodeURIComponent(state.selectedJobId)}/commit`, jsonOptions('POST', {all: true, item_ids: [], write_mode: 'replace'}));
                await actions.refresh(true);
                const summary = `写入 ${response.written}，冲突 ${response.conflicts}，跳过 ${response.skipped}`;
                showFeedback(root, summary, response.conflicts ? 'error' : 'success');
            } catch (error) { showFeedback(root, error.message, 'error'); }
        });
    });
    root.querySelector('[data-caption-commit-selected]')?.addEventListener('click', async (event) => {
        await withBusy(event.currentTarget, async () => {
            try {
                const itemIds = state.workspaceData.gallerySelected || [];
                const response = await captioningApi(`/jobs/${encodeURIComponent(state.selectedJobId)}/commit`, jsonOptions('POST', {all: false, item_ids: itemIds, write_mode: 'replace'}));
                await actions.refresh(true); showFeedback(root, `已写入 ${response.written} 项，冲突 ${response.conflicts} 项`, response.conflicts ? 'error' : 'success');
            } catch (error) { showFeedback(root, error.message, 'error'); }
        });
    });
    root.onkeydown = (event) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key) || ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
        const results = state.selectedJob?.results || [];
        if (!results.length) return;
        const index = Math.max(0, results.findIndex((item) => item.id === state.selectedItemId));
        actions.selectItem(results[(index + (event.key === 'ArrowRight' ? 1 : -1) + results.length) % results.length].id);
    };
}

function schedulePoll(root, state, actions, prefill) {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    if (!['queued', 'running', 'paused'].includes(state.selectedJob?.state)) return;
    state.pollTimer = window.setTimeout(() => selectJob(root, state, actions, state.selectedJobId, prefill).catch(() => {}), 1500);
}

function selectedItem(state) { return state.selectedJob?.results?.find((item) => item.id === state.selectedItemId) || null; }
function jobOptions(jobs, selected) { return jobs.map((job) => `<option value="${escapeAttribute(job.id)}" ${job.id === selected ? 'selected' : ''}>${escapeHtml(job.directory.split(/[\\/]/).pop() || job.directory)} · ${escapeHtml(job.model)}</option>`).join(''); }
