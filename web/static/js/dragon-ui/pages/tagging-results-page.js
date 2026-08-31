/* Dedicated image-to-caption review and write-back page. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import {
    commitTaggingJob,
    loadTaggingJob,
    loadTaggingJobs,
    updateTaggingItem,
} from './tagging-api.js?v=dragon-ui-20260831v3';
import {
    readTaggingWorkspaceState,
    returnToTaggingWorkspace,
    updateTaggingPromptDraft,
} from './tagging-workspace-state.js?v=dragon-ui-20260831v3';

const api = createApiClient();
const RESULT_BATCH_SIZE = 24;
const POLL_INTERVAL_MS = 1500;

export async function loadTaggingResultsPage() {
    const jobsPayload = await loadTaggingJobs(api);
    const jobs = Array.isArray(jobsPayload.jobs) ? jobsPayload.jobs : [];
    const workspace = readTaggingWorkspaceState();
    const selectedJob = jobs.find((item) => item.id === workspace.jobId) || jobs[0] || null;
    const job = selectedJob?.id ? (await loadTaggingJob(api, selectedJob.id)).job : null;
    const state = {
        active: true,
        jobs,
        job,
        selectedItemIds: new Set(),
        expandedItemIds: new Set(),
        dirtyItemIds: new Set(),
        savingItemIds: new Set(),
        visibleCount: Math.min(RESULT_BATCH_SIZE, job?.items?.length || 0),
        committing: false,
        error: '',
        notice: '',
        root: null,
        cleanup: null,
        stopObserver: null,
        pollTimer: null,
        requestId: 0,
        jobEpoch: 0,
    };
    return {
        html: renderPage(state),
        onMount: (root) => mountPage(root, state),
        beforeLeave: () => !state.dirtyItemIds.size || window.confirm('有尚未保存的打标结果，仍要离开吗？'),
        onUnmount: () => disposePage(state),
    };
}

function mountPage(root, state) {
    state.root = root;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    root.addEventListener('click', (event) => handleClick(state, event), options);
    root.addEventListener('change', (event) => handleChange(state, event), options);
    root.addEventListener('input', (event) => handleInput(state, event), options);
    root.addEventListener('toggle', (event) => handleToggle(state, event), { ...options, capture: true });
    state.cleanup = () => controller.abort();
    reconnectObserver(state);
    if (isBusy(state.job)) schedulePoll(state);
}

function disposePage(state) {
    state.active = false;
    state.jobEpoch += 1;
    state.requestId += 1;
    clearPoll(state);
    state.stopObserver?.();
    state.cleanup?.();
}

function renderPage(state) {
    const job = state.job;
    const items = job?.items || [];
    return `<div class="dragon-page dragon-page-wide dragon-caption-page dragon-tagging-tool-page dragon-tagging-results-page" data-tagging-results-page>
        <header class="dragon-tagging-tool-header">
            <div><button class="dragon-icon-button" type="button" data-results-back aria-label="返回打标工作台" title="返回">${renderIcon('chevronDown')}</button><span><span class="dragon-eyebrow">FINAL CAPTIONS</span><h1>最终打标结果</h1></span></div>
            <label class="dragon-field dragon-tagging-job-select"><span>任务</span><select class="dragon-select" data-results-job>${jobOptions(state)}</select></label>
        </header>
        ${feedback(state)}
        ${job ? renderResultsWorkspace(state, items) : renderEmptyResults()}
    </div>`;
}

function renderResultsWorkspace(state, items) {
    const job = state.job;
    const busy = isBusy(job);
    const candidates = committableItems(items);
    return `<section class="dragon-tagging-results-shell">
        <header class="dragon-tagging-results-toolbar">
            <div><span class="dragon-status-badge" data-state="${busy ? 'running' : job.state === 'failed' ? 'error' : 'active'}"><i aria-hidden="true"></i><b>${escapeHtml(jobStateLabel(job.state))}</b></span><span>${Number(job.completed || 0)}/${Number(job.total || items.length)} 完成</span></div>
            <div><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-results-select-all ${candidates.length && !busy ? '' : 'disabled'}>${renderIcon('check', 'dragon-btn-icon')}<span>全选候选</span></button><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-results-clear ${state.selectedItemIds.size ? '' : 'disabled'}><span>清空</span></button><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-results-commit-selected ${state.selectedItemIds.size && !busy && !state.committing ? '' : 'disabled'}>${renderIcon('save', 'dragon-btn-icon')}<span>写回已选 <b data-results-selected-count>${state.selectedItemIds.size}</b></span></button><button class="dragon-btn dragon-btn-primary dragon-btn-sm" type="button" data-results-commit-all ${candidates.length && !busy && !state.committing ? '' : 'disabled'}><span>全部写回 TXT</span></button></div>
        </header>
        <div class="dragon-tagging-review-list" data-results-list>${renderResultRows(state, 0, state.visibleCount)}${renderResultSentinel(state)}</div>
    </section>`;
}

function renderResultRows(state, start, end) {
    const busy = isBusy(state.job);
    return (state.job?.items || []).slice(start, end).map((item) => renderResultRow(state, item, busy)).join('');
}

function renderResultRow(state, item, busy) {
    const selected = state.selectedItemIds.has(item.id);
    const expanded = state.expandedItemIds.has(item.id);
    const text = String(item.proposed_caption || '');
    const status = itemStateLabel(item.state);
    return `<details class="dragon-tagging-review-item" data-result-item="${escapeAttribute(item.id)}" data-state="${escapeAttribute(item.state || '')}" ${expanded ? 'open' : ''}>
        <summary>
            <input type="checkbox" data-result-select data-item-id="${escapeAttribute(item.id)}" ${selected ? 'checked' : ''} ${text.trim() && !busy ? '' : 'disabled'} aria-label="选择 ${escapeAttribute(item.name || item.file || '结果')}">
            <span class="dragon-tagging-review-thumb">${item.url ? `<img src="${escapeAttribute(item.thumbnail_url || item.url)}" alt="${escapeAttribute(item.name || '图片')}" width="240" height="135" loading="lazy" decoding="async">` : renderIcon('panels')}</span>
            <span class="dragon-tagging-review-summary"><strong title="${escapeAttribute(item.file || '')}">${escapeHtml(item.name || item.file || '-')}</strong><small>${escapeHtml(excerpt(text) || item.error || '暂无候选标注')}</small></span>
            <span class="dragon-status-badge" data-state="${statusTone(item.state)}"><i aria-hidden="true"></i><b>${escapeHtml(status)}</b></span>
            ${renderIcon('chevronDown')}
        </summary>
        ${expanded ? renderResultDetail(state, item, busy) : ''}
    </details>`;
}

function renderResultDetail(state, item, busy) {
    const text = String(item.proposed_caption || '');
    return `<div class="dragon-tagging-review-detail">
            <figure>${item.url ? `<img data-result-detail-image src="${escapeAttribute(item.url)}" alt="${escapeAttribute(item.name || '图片')}" width="960" height="540" loading="lazy" decoding="async">` : `<span>${renderIcon('panels')}</span>`}<figcaption title="${escapeAttribute(item.file || '')}">${escapeHtml(item.file || '')}</figcaption></figure>
            <div class="dragon-tagging-review-editor">
                <label class="dragon-field"><span>最终标注</span><textarea class="dragon-textarea" rows="9" data-result-caption data-item-id="${escapeAttribute(item.id)}" ${busy ? 'disabled' : ''}>${escapeHtml(text)}</textarea></label>
                ${item.caption ? `<details class="dragon-tagging-original-caption"><summary>原始标注</summary><p>${escapeHtml(item.caption)}</p></details>` : ''}
                ${item.error ? `<p class="dragon-tagging-result-error" role="alert">${escapeHtml(item.error)}</p>` : ''}
                ${item.commit_error ? `<p class="dragon-tagging-result-error" role="alert">写回失败：${escapeHtml(item.commit_error)}</p>` : ''}
                <footer><span>${item.caption_file ? `已写入 ${escapeHtml(item.caption_file)}` : '目标：图片同名 .txt'}</span><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-result-save data-item-id="${escapeAttribute(item.id)}" ${busy || state.savingItemIds.has(item.id) ? 'disabled' : ''}>${renderIcon('save', 'dragon-btn-icon')}<span>${state.savingItemIds.has(item.id) ? '保存中…' : '保存修改'}</span></button></footer>
            </div>
        </div>`;
}

function renderResultSentinel(state) {
    const total = state.job?.items?.length || 0;
    return `<div class="dragon-tagging-results-sentinel" data-results-sentinel ${state.visibleCount < total ? '' : 'hidden'}><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-results-more>${renderIcon('chevronDown', 'dragon-btn-icon')}<span>加载更多</span></button></div>`;
}

function renderEmptyResults() {
    return `<div class="dragon-empty-state dragon-tagging-tool-empty"><span class="dragon-empty-state-icon">${renderIcon('list')}</span><strong>暂无打标结果</strong><button class="dragon-btn dragon-btn-secondary" type="button" data-results-back>返回打标工作台</button></div>`;
}

function handleClick(state, event) {
    const checkbox = event.target.closest?.('[data-result-select]');
    if (checkbox) event.stopPropagation();
    const target = event.target.closest?.('[data-results-back], [data-results-more], [data-results-select-all], [data-results-clear], [data-results-commit-selected], [data-results-commit-all], [data-result-save]');
    if (!target) return;
    if (target.matches('[data-results-back]')) return returnToTaggingWorkspace();
    if (target.matches('[data-results-more]')) return revealMore(state);
    if (target.matches('[data-results-select-all]')) return selectAllCandidates(state);
    if (target.matches('[data-results-clear]')) return clearSelection(state);
    if (target.matches('[data-results-commit-selected]')) return run(() => commitResults(state, false));
    if (target.matches('[data-results-commit-all]')) return run(() => commitResults(state, true));
    if (target.matches('[data-result-save]')) return run(() => saveItem(state, target.dataset.itemId));
}

function handleChange(state, event) {
    const target = event.target;
    if (target.matches('[data-results-job]')) return run(() => selectJob(state, target.value));
    if (target.matches('[data-result-select]')) {
        if (target.checked) state.selectedItemIds.add(target.dataset.itemId);
        else state.selectedItemIds.delete(target.dataset.itemId);
        return syncSelection(state);
    }
    if (target.matches('[data-result-caption]')) return run(() => saveItem(state, target.dataset.itemId));
}

function handleInput(state, event) {
    if (!event.target.matches('[data-result-caption]')) return;
    state.dirtyItemIds.add(event.target.dataset.itemId);
}

function handleToggle(state, event) {
    const details = event.target;
    if (!details.matches?.('[data-result-item]')) return;
    if (details.open) {
        state.expandedItemIds.add(details.dataset.resultItem);
        const item = state.job?.items?.find((entry) => entry.id === details.dataset.resultItem);
        if (item && !details.querySelector('.dragon-tagging-review-detail')) {
            details.insertAdjacentHTML('beforeend', renderResultDetail(state, item, isBusy(state.job)));
        }
    } else state.expandedItemIds.delete(details.dataset.resultItem);
}

async function selectJob(state, jobId) {
    if (state.dirtyItemIds.size && !window.confirm('放弃尚未保存的修改并切换任务吗？')) return;
    const epoch = ++state.jobEpoch;
    state.requestId += 1;
    clearPoll(state);
    state.savingItemIds.clear();
    state.committing = false;
    state.selectedItemIds.clear();
    state.expandedItemIds.clear();
    state.dirtyItemIds.clear();
    state.visibleCount = RESULT_BATCH_SIZE;
    await hydrateJob(state, jobId, epoch);
    if (state.active && state.jobEpoch === epoch) updateTaggingPromptDraft({ jobId });
}

async function saveItem(state, itemId) {
    if (!state.job?.id || state.savingItemIds.has(itemId)) return;
    const jobId = state.job.id;
    const epoch = state.jobEpoch;
    const requestId = state.requestId;
    const textarea = state.root?.querySelector(`[data-result-caption][data-item-id="${cssEscape(itemId)}"]`);
    const text = textarea?.value ?? state.job.items.find((item) => item.id === itemId)?.proposed_caption ?? '';
    state.savingItemIds.add(itemId);
    try {
        const payload = await updateTaggingItem(api, jobId, itemId, text);
        if (!isCurrentJob(state, jobId, epoch, requestId)) return;
        state.job = payload.job || state.job;
        state.dirtyItemIds.delete(itemId);
        state.notice = '结果修改已保存。';
        state.error = '';
    } catch (error) {
        if (isCurrentJob(state, jobId, epoch, requestId)) state.error = error.message || '保存结果失败';
    } finally {
        if (isCurrentJob(state, jobId, epoch, requestId)) {
            state.savingItemIds.delete(itemId);
            rerender(state);
        }
    }
}

async function commitResults(state, all) {
    if (!state.job?.id || state.committing) return;
    const jobId = state.job.id;
    const epoch = state.jobEpoch;
    const requestId = state.requestId;
    if (state.dirtyItemIds.size && !await saveDirtyItems(state, jobId, epoch, requestId)) return;
    if (!isCurrentJob(state, jobId, epoch, requestId)) return;
    if (all && !window.confirm('将全部候选写入图片同名 TXT，是否继续？')) return;
    const itemIds = all ? [] : [...state.selectedItemIds];
    if (!all && !itemIds.length) return;
    state.committing = true;
    rerender(state);
    try {
        const payload = await commitTaggingJob(api, jobId, { all, itemIds });
        if (!isCurrentJob(state, jobId, epoch, requestId)) return;
        state.job = payload.job || state.job;
        state.notice = `已写回 ${Number(payload.written || 0)} 个 TXT。`;
        state.error = payload.errors?.length ? `${payload.errors.length} 项写回失败。` : '';
    } catch (error) {
        if (isCurrentJob(state, jobId, epoch, requestId)) state.error = error.message || '写回 TXT 失败';
    } finally {
        if (isCurrentJob(state, jobId, epoch, requestId)) {
            state.committing = false;
            rerender(state);
        }
    }
}

async function saveDirtyItems(state, jobId, epoch, requestId) {
    const drafts = [...state.dirtyItemIds].map((itemId) => ({
        itemId,
        text: state.root?.querySelector(`[data-result-caption][data-item-id="${cssEscape(itemId)}"]`)?.value ?? '',
    }));
    try {
        for (const draft of drafts) {
            const payload = await updateTaggingItem(api, jobId, draft.itemId, draft.text);
            if (!isCurrentJob(state, jobId, epoch, requestId)) return false;
            state.job = payload.job || state.job;
            state.dirtyItemIds.delete(draft.itemId);
        }
        return true;
    } catch (error) {
        if (isCurrentJob(state, jobId, epoch, requestId)) {
            state.error = error.message || '保存结果失败';
            rerender(state);
        }
        return false;
    }
}

async function hydrateJob(state, jobId, epoch = state.jobEpoch) {
    const requestId = ++state.requestId;
    try {
        const payload = await loadTaggingJob(api, jobId);
        if (!isCurrentRequest(state, epoch, requestId)) return;
        state.job = payload.job || null;
        state.visibleCount = Math.min(Math.max(state.visibleCount, RESULT_BATCH_SIZE), state.job?.items?.length || 0);
        rerender(state);
        if (isBusy(state.job)) schedulePoll(state);
        else clearPoll(state);
    } catch (error) {
        if (isCurrentRequest(state, epoch, requestId)) {
            state.error = error.message || '读取打标结果失败';
            rerender(state);
        }
    }
}

function revealMore(state) {
    const total = state.job?.items?.length || 0;
    const previousCount = state.visibleCount;
    const nextCount = Math.min(total, previousCount + RESULT_BATCH_SIZE);
    if (nextCount <= previousCount) return;
    state.visibleCount = nextCount;
    const list = state.root?.querySelector('[data-results-list]');
    const sentinel = list?.querySelector('[data-results-sentinel]');
    if (!list || !sentinel) return rerender(state);
    sentinel.insertAdjacentHTML('beforebegin', renderResultRows(state, previousCount, nextCount));
    updateResultSentinel(state);
}

function updateResultSentinel(state) {
    const sentinel = state.root?.querySelector('[data-results-sentinel]');
    if (sentinel) sentinel.hidden = state.visibleCount >= (state.job?.items?.length || 0);
}

function selectAllCandidates(state) {
    state.selectedItemIds = new Set(committableItems(state.job?.items || []).map((item) => item.id));
    syncSelection(state);
}

function clearSelection(state) {
    state.selectedItemIds.clear();
    syncSelection(state);
}

function syncSelection(state) {
    state.root?.querySelectorAll('[data-results-selected-count]').forEach((node) => { node.textContent = String(state.selectedItemIds.size); });
    const commit = state.root?.querySelector('[data-results-commit-selected]');
    if (commit) commit.disabled = !state.selectedItemIds.size || isBusy(state.job) || state.committing;
    const clear = state.root?.querySelector('[data-results-clear]');
    if (clear) clear.disabled = !state.selectedItemIds.size;
}

function reconnectObserver(state) {
    state.stopObserver?.();
    const sentinel = state.root?.querySelector('[data-results-sentinel]');
    if (!sentinel || typeof IntersectionObserver !== 'function') return;
    const observer = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) revealMore(state);
    }, { rootMargin: '300px 0px', threshold: 0.01 });
    observer.observe(sentinel);
    state.stopObserver = () => observer.disconnect();
}

function rerender(state) {
    if (!state.root) return;
    const scrollY = globalThis.scrollY || 0;
    state.root.innerHTML = renderPage(state);
    reconnectObserver(state);
    globalThis.requestAnimationFrame?.(() => globalThis.scrollTo?.({ top: scrollY, behavior: 'auto' }));
}

function schedulePoll(state) {
    clearPoll(state);
    if (!state.active || !state.job?.id || !isBusy(state.job)) return;
    state.pollTimer = globalThis.setTimeout(async () => {
        state.pollTimer = null;
        await hydrateJob(state, state.job.id, state.jobEpoch);
    }, POLL_INTERVAL_MS);
}

function clearPoll(state) {
    if (state.pollTimer != null) globalThis.clearTimeout(state.pollTimer);
    state.pollTimer = null;
}

function jobOptions(state) {
    if (!state.jobs.length) return '<option value="">暂无任务</option>';
    return state.jobs.map((job) => `<option value="${escapeAttribute(job.id)}" ${job.id === state.job?.id ? 'selected' : ''}>${escapeHtml(job.created_at_text || job.id)} · ${escapeHtml(jobStateLabel(job.state))} · ${Number(job.total || 0)} 张</option>`).join('');
}

function feedback(state) {
    return `${state.error ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="error" role="alert">${escapeHtml(state.error)}</div>` : ''}${state.notice ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="success" role="status">${escapeHtml(state.notice)}</div>` : ''}`;
}

function committableItems(items) {
    return items.filter((item) => String(item.proposed_caption || '').trim() && (
        ['ready', 'failed'].includes(item.state)
        || (item.state === 'committed' && String(item.caption || '') !== String(item.proposed_caption || ''))
    ));
}

function isBusy(job) {
    return ['queued', 'running'].includes(job?.state);
}

function isCurrentJob(state, jobId, epoch, requestId) {
    return isCurrentRequest(state, epoch, requestId) && state.job?.id === jobId;
}

function isCurrentRequest(state, epoch, requestId) {
    return Boolean(state.active) && state.jobEpoch === epoch && state.requestId === requestId;
}

function statusTone(value) {
    if (['ready', 'committed'].includes(value)) return 'active';
    if (value === 'failed') return 'error';
    if (value === 'running') return 'running';
    return 'idle';
}

function itemStateLabel(value) {
    return { queued: '排队中', running: '调用中', ready: '待审阅', committed: '已写回', failed: '失败', canceled: '已停止', empty: '内容为空' }[value] || value || '未知';
}

function jobStateLabel(value) {
    return { queued: '排队中', running: '处理中', completed: '全部完成', partial: '部分完成', failed: '失败', canceled: '已停止' }[value] || value || '未知';
}

function excerpt(value) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    return text.length > 120 ? `${text.slice(0, 120)}…` : text;
}

function cssEscape(value) {
    if (globalThis.CSS?.escape) return globalThis.CSS.escape(String(value || ''));
    return String(value || '').replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

function run(fn) {
    Promise.resolve().then(fn).catch((error) => console.error('[dragon-tagging-results]', error));
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
