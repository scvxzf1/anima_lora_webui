import { renderIcon } from '../icons.js?v=dragon-ui-20260902v36';
import { renderCaptionEditor } from './tagging-results-editor.js?v=dragon-ui-20260902v1';
import { renderTaggingResultImageDialog } from './tagging-results-image-preview.js?v=dragon-ui-20260902v5';

export function renderResultsPage(state, datasetPickerHtml = '') {
    const job = state.job;
    const items = job?.items || [];
    return `<div class="dragon-page dragon-page-wide dragon-caption-page dragon-tagging-tool-page dragon-tagging-results-page" data-tagging-results-page>
        <header class="dragon-tagging-tool-header dragon-tagging-results-header">
            <div><button class="dragon-icon-button" type="button" data-results-back aria-label="返回打标工作台" title="返回">${renderIcon('chevronDown')}</button><span><span class="dragon-eyebrow">FINAL CAPTIONS</span><h1>最终打标结果</h1></span></div>
            <div class="dragon-tagging-results-header-controls">${renderDatasetControl(state)}<label class="dragon-field dragon-tagging-job-select"><span>任务 <b>${filteredJobs(state).length}</b></span><select class="dragon-select" data-results-job>${jobOptions(state)}</select></label></div>
        </header>
        <div data-results-feedback>${renderResultsFeedback(state)}</div>
        ${job ? renderResultsWorkspace(state, items) : renderEmptyResults(state)}
        ${datasetPickerHtml}
        ${renderTaggingResultImageDialog()}
    </div>`;
}

export function renderResultsWorkspace(state, items) {
    const job = state.job;
    const busy = isBusy(job);
    const candidates = committableItems(items);
    const selectedWritableCount = candidates.filter((item) => state.selectedItemIds.has(item.id)).length;
    return `<section class="dragon-tagging-results-shell">
        <header class="dragon-tagging-results-toolbar">
            <div class="dragon-tagging-results-toolbar-row dragon-tagging-results-toolbar-primary">
                <div class="dragon-tagging-results-state"><span class="dragon-status-badge" data-results-job-status data-state="${jobStatusTone(job.state)}"><i aria-hidden="true"></i><b>${escapeHtml(jobStateLabel(job.state))}</b></span><strong data-results-progress>${Number(job.completed || 0)}/${Number(job.total || items.length)} 完成</strong><small>${escapeHtml(job.profile_name || job.settings?.profile_name || '未记录接入方式')}</small></div>
                <div class="dragon-tagging-results-primary-actions"><button class="dragon-icon-button" type="button" data-results-refresh aria-label="刷新任务结果" title="刷新">${renderIcon('refresh')}</button>${renderRerunControls(state, busy)}</div>
            </div>
            <div class="dragon-tagging-results-toolbar-row dragon-tagging-results-toolbar-secondary">
                <div class="dragon-segmented" role="group" aria-label="标注查看模式"><button type="button" data-results-mode="tags" data-active="${state.viewMode === 'tags'}">${renderIcon('tags')}<span>Tag</span></button><button type="button" data-results-mode="raw" data-active="${state.viewMode === 'raw'}">${renderIcon('list')}<span>原文</span></button></div>
                <div class="dragon-tagging-results-batch-actions"><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-results-select-all ${candidates.length && !busy ? '' : 'disabled'}>${renderIcon('check', 'dragon-btn-icon')}<span>全选候选</span></button><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-results-clear ${state.selectedItemIds.size ? '' : 'disabled'}><span>清空</span></button><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-results-commit-selected ${selectedWritableCount && !busy && !state.committing ? '' : 'disabled'}>${renderIcon('save', 'dragon-btn-icon')}<span>写回已选 <b data-results-selected-count>${selectedWritableCount}</b></span></button><button class="dragon-btn dragon-btn-primary dragon-btn-sm" type="button" data-results-commit-all ${candidates.length && !busy && !state.committing ? '' : 'disabled'}><span>全部写回 TXT</span></button></div>
            </div>
        </header>
        <div class="dragon-tagging-review-list" data-results-list>${renderResultRows(state, 0, state.visibleCount)}${renderResultSentinel(state)}</div>
    </section>`;
}

export function renderResultRows(state, start, end) {
    const busy = isBusy(state.job);
    return (state.job?.items || []).slice(start, end).map((item) => renderResultCard(state, item, busy)).join('');
}

export function renderResultSentinel(state) {
    const total = state.job?.items?.length || 0;
    return `<div class="dragon-tagging-results-sentinel" data-results-sentinel ${state.visibleCount < total ? '' : 'hidden'}><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-results-more>${renderIcon('chevronDown', 'dragon-btn-icon')}<span>加载更多</span></button></div>`;
}

export function renderResultsFeedback(state) {
    return `${state.error ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="error" role="alert">${escapeHtml(state.error)}</div>` : ''}${state.notice ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="success" role="status">${escapeHtml(state.notice)}</div>` : ''}`;
}

export function filteredJobs(state) {
    const selected = String(state.datasetFile || '');
    return selected ? state.jobs.filter((job) => String(job.dataset_file || '') === selected) : state.jobs;
}

export function committableItems(items) {
    return items.filter((item) => String(item.proposed_caption || '').trim() && (
        ['ready', 'failed'].includes(item.state)
        || (item.state === 'committed' && String(item.caption || '') !== String(item.proposed_caption || ''))
    ));
}

export function isBusy(job) {
    return ['queued', 'running'].includes(job?.state);
}

export function jobStateLabel(value) {
    return { queued: '待处理', running: '处理中', completed: '全部完成', partial: '部分完成', failed: '失败', canceled: '已停止' }[value] || value || '未知';
}

export function itemStateLabel(value) {
    return { queued: '', running: '处理中', ready: '待审阅', committed: '已写回', failed: '失败', canceled: '已停止', empty: '内容为空' }[value] || value || '未知';
}

export function statusTone(value) {
    if (value === 'queued') return 'queued';
    if (['ready', 'committed'].includes(value)) return 'active';
    if (value === 'failed') return 'error';
    if (value === 'running') return 'running';
    return 'idle';
}

export function jobStatusTone(value) {
    if (value === 'queued') return 'queued';
    if (value === 'running') return 'running';
    if (value === 'failed') return 'error';
    return 'active';
}

function renderDatasetControl(state) {
    const file = String(state.datasetFile || '');
    return `<div class="dragon-tagging-results-dataset" data-results-dataset-card><span class="dragon-eyebrow">数据集</span><strong title="${escapeAttribute(file)}">${escapeHtml(shortName(file) || '全部数据集')}</strong><code title="${escapeAttribute(file)}">${escapeHtml(file || '显示所有保留任务')}</code><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-results-dataset-open>${renderIcon('folder', 'dragon-btn-icon')}<span>${file ? '更换预设' : '选择预设'}</span></button></div>`;
}

function renderRerunControls(state, busy) {
    const profiles = (state.providerProfiles || []).filter((profile) => profile.available);
    const options = profiles.length ? profiles.map((profile) => `<option value="${escapeAttribute(profile.id)}" ${profile.id === state.rerunProfileId ? 'selected' : ''}>${escapeHtml(profile.name)}</option>`).join('') : '<option value="">暂无可用接入</option>';
    const rerunLabel = state.rerunning
        ? '正在提交…'
        : state.selectedItemIds.size ? `重新打标已选 ${state.selectedItemIds.size}` : '重新打标';
    return `<div class="dragon-tagging-rerun-controls"><label><span class="visually-hidden">重新打标方式</span><select class="dragon-select" data-results-rerun-profile ${busy || state.rerunning ? 'disabled' : ''}>${options}</select></label><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-results-rerun ${!busy && !state.rerunning && state.rerunProfileId ? '' : 'disabled'}>${renderIcon('refresh', 'dragon-btn-icon')}<span data-results-rerun-label>${rerunLabel}</span></button></div>`;
}

function renderResultCard(state, item, busy) {
    const selected = state.selectedItemIds.has(item.id);
    const text = draftText(state, item);
    const dirty = state.dirtyItemIds.has(item.id) && text !== String(item.proposed_caption || '');
    const status = itemStateLabel(item.state);
    const hideStatus = ['ready', 'queued'].includes(item.state);
    const language = state.itemLanguages.get(item.id) || 'en';
    const translating = state.translatingItemIds.has(item.id);
    return `<article class="dragon-tagging-review-item" data-result-item="${escapeAttribute(item.id)}" data-state="${escapeAttribute(item.state || '')}">
        <div class="dragon-tagging-review-media">${item.url ? `<button class="dragon-tagging-review-image" type="button" data-result-image-open data-item-id="${escapeAttribute(item.id)}" aria-label="预览 ${escapeAttribute(item.name || item.file || '图片')}"><img src="${escapeAttribute(item.thumbnail_url || item.url)}" alt="${escapeAttribute(item.name || '图片')}" width="640" height="480" loading="lazy" decoding="async" fetchpriority="low"></button>` : `<span>${renderIcon('panels')}</span>`}<label class="dragon-tagging-review-select"><input type="checkbox" data-result-select data-item-id="${escapeAttribute(item.id)}" ${selected ? 'checked' : ''} ${!item.id || busy ? 'disabled' : ''}><span class="visually-hidden">选择 ${escapeAttribute(item.name || item.file || '结果')}</span></label><span class="dragon-status-badge" data-result-item-status data-state="${statusTone(item.state)}" ${hideStatus ? 'hidden' : ''}><i aria-hidden="true"></i><b>${hideStatus ? '' : escapeHtml(status)}</b></span></div>
        <div class="dragon-tagging-review-body"><header><strong title="${escapeAttribute(item.file || '')}">${escapeHtml(item.name || item.file || '-')}</strong><small title="${escapeAttribute(item.file || '')}">${escapeHtml(item.file || '')}</small></header>
            <div data-result-editor-host>${renderCaptionEditor({ itemId: item.id, text, mode: state.viewMode, busy, saving: state.savingItemIds.has(item.id) })}</div>
            ${item.caption ? `<details class="dragon-tagging-original-caption"><summary>写回前标注</summary><p>${escapeHtml(item.caption)}</p></details>` : ''}
            <div data-result-item-feedback>${renderItemFeedback(item)}</div>
            <footer><span data-result-target>${item.caption_file ? `已写入 ${escapeHtml(item.caption_file)}` : '目标：图片同名 .txt'}</span><div><button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-result-translate data-item-id="${escapeAttribute(item.id)}" ${busy || translating ? 'disabled' : ''}>${renderIcon('languages', 'dragon-btn-icon')}<span>${translating ? '翻译中…' : language === 'en' ? '中文' : 'EN'}</span></button><button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button" data-result-save data-item-id="${escapeAttribute(item.id)}" ${busy || state.savingItemIds.has(item.id) || !dirty ? 'disabled' : ''}>${renderIcon('save', 'dragon-btn-icon')}<span data-result-save-label>${state.savingItemIds.has(item.id) ? '保存中…' : '保存修改'}</span></button></div></footer>
        </div>
    </article>`;
}

function renderEmptyResults(state) {
    const message = state.datasetFile ? '当前数据集暂无保留任务' : '暂无打标结果';
    return `<div class="dragon-empty-state dragon-tagging-tool-empty"><span class="dragon-empty-state-icon">${renderIcon('list')}</span><strong>${message}</strong><button class="dragon-btn dragon-btn-secondary" type="button" data-results-back>返回打标工作台</button></div>`;
}

function jobOptions(state) {
    const jobs = filteredJobs(state);
    if (!jobs.length) return '<option value="">暂无任务</option>';
    return jobs.map((job) => `<option value="${escapeAttribute(job.id)}" ${job.id === state.job?.id ? 'selected' : ''}>${escapeHtml(job.created_at_text || job.id)} · ${escapeHtml(jobStateLabel(job.state))} · ${Number(job.total || 0)} 张</option>`).join('');
}

function draftText(state, item) {
    return state.drafts.has(item.id) ? String(state.drafts.get(item.id) || '') : String(item.proposed_caption || '');
}

function renderItemFeedback(item) {
    return `${item.error ? `<p class="dragon-tagging-result-error" role="alert">${escapeHtml(item.error)}</p>` : ''}${item.commit_error ? `<p class="dragon-tagging-result-error" role="alert">写回失败：${escapeHtml(item.commit_error)}</p>` : ''}`;
}

function shortName(value) {
    return String(value || '').replace(/\\/g, '/').split('/').filter(Boolean).pop() || '';
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
