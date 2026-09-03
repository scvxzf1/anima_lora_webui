/* Image-to-caption review controller. Rendering and tag editing live in sibling modules. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { confirmDragonDialog } from '../../shared/dialog.js?v=dragon-ui-20260901v2';
import {
    mountDatasetPresetPicker,
    renderDatasetPresetPickerDialog,
} from '../dataset-preset-picker.js?v=dragon-ui-20260902v2';
import { createVisibilityPoller } from '../visibility-poller.js?v=dragon-ui-20260826v2';
import {
    commitTaggingJob,
    loadProviderProfiles,
    loadTagDictionaryDownload,
    loadTagDictionaryStatus,
    loadTaggingJob,
    loadTaggingJobs,
    rerunTaggingJob,
    startTagDictionaryDownload,
    translateCaptionTags,
    updateTaggingItem,
} from './tagging-api.js?v=dragon-ui-20260902v8';
import {
    appendCaptionTag,
    joinCaptionTags,
    moveCaptionTag,
    removeCaptionTag,
    renderCaptionEditor,
    replaceCaptionTag,
    splitCaptionTags,
} from './tagging-results-editor.js?v=dragon-ui-20260902v1';
import { mountTaggingResultImagePreview } from './tagging-results-image-preview.js?v=dragon-ui-20260902v5';
import {
    committableItems,
    filteredJobs,
    isBusy,
    itemStateLabel,
    jobStatusTone,
    jobStateLabel,
    renderResultRows,
    renderResultsFeedback,
    renderResultsPage,
    statusTone,
} from './tagging-results-view.js?v=dragon-ui-20260902v8';
import {
    readTaggingWorkspaceState,
    returnToTaggingWorkspace,
    updateTaggingPromptDraft,
} from './tagging-workspace-state.js?v=dragon-ui-20260831v4';

const api = createApiClient();
const RESULT_BATCH_SIZE = 24;
const POLL_INTERVAL_MS = 2000;
const DOWNLOAD_POLL_MS = 500;
const DATASET_PICKER_HTML = renderDatasetPresetPickerDialog({
    title: '选择结果数据集',
    description: '结果页仅显示所选数据集对应的保留任务。',
    applyLabel: '查看此数据集',
});

export async function loadTaggingResultsPage() {
    const [jobsPayload, profilesPayload] = await Promise.all([
        loadTaggingJobs(api),
        loadProviderProfiles(api).catch(() => ({ profiles: [] })),
    ]);
    const jobs = Array.isArray(jobsPayload.jobs) ? jobsPayload.jobs : [];
    const workspace = readTaggingWorkspaceState();
    const selectedJob = jobs.find((item) => item.id === workspace.jobId)
        || jobs.find((item) => item.dataset_file === workspace.datasetFile)
        || jobs[0]
        || null;
    const job = selectedJob?.id ? (await loadTaggingJob(api, selectedJob.id)).job : null;
    const profiles = Array.isArray(profilesPayload.profiles) ? profilesPayload.profiles : [];
    const state = createState({ jobs, job, profiles, workspace });
    return {
        html: renderResultsPage(state, DATASET_PICKER_HTML),
        onMount: (root) => mountPage(root, state),
        beforeLeave: () => confirmLeave(state),
        onUnmount: () => disposePage(state),
    };
}

function createState({ jobs, job, profiles, workspace }) {
    return {
        active: true,
        jobs,
        job,
        datasetFile: job?.dataset_file || workspace.datasetFile || '',
        providerProfiles: profiles,
        rerunProfileId: chooseRerunProfile(job, profiles),
        selectedItemIds: new Set(),
        dirtyItemIds: new Set(),
        savingItemIds: new Set(),
        translatingItemIds: new Set(),
        drafts: new Map(),
        languageDrafts: new Map(),
        itemLanguages: new Map(),
        viewMode: readViewMode(),
        visibleCount: Math.min(RESULT_BATCH_SIZE, job?.items?.length || 0),
        committing: false,
        rerunning: false,
        error: '',
        notice: '',
        root: null,
        cleanup: null,
        stopObserver: null,
        jobPoller: null,
        datasetPicker: null,
        imagePreview: null,
        draggedTag: null,
        jobSignature: jobSignature(job),
        forceJobRender: false,
        requestId: 0,
        jobEpoch: 0,
    };
}

function mountPage(root, state) {
    state.root = root;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    root.addEventListener('click', (event) => handleClick(state, event), options);
    root.addEventListener('change', (event) => handleChange(state, event), options);
    root.addEventListener('input', (event) => handleInput(state, event), options);
    root.addEventListener('keydown', (event) => handleKeydown(state, event), options);
    root.addEventListener('dragstart', (event) => handleDragStart(state, event), options);
    root.addEventListener('dragover', (event) => handleDragOver(state, event), options);
    root.addEventListener('drop', (event) => handleDrop(state, event), options);
    root.addEventListener('dragend', () => clearDragState(state), options);
    state.cleanup = () => controller.abort();
    mountDatasetPicker(state);
    mountImagePreview(state);
    reconnectObserver(state);
    state.jobPoller = createVisibilityPoller({
        delay: POLL_INTERVAL_MS,
        poll: async () => {
            const jobId = state.job?.id;
            if (!state.active || !jobId || !isBusy(state.job)) return state.jobPoller?.stop();
            await hydrateJob(state, jobId, state.jobEpoch);
            if (!isBusy(state.job)) state.jobPoller?.stop();
        },
    });
    if (isBusy(state.job)) schedulePoll(state);
}

function mountDatasetPicker(state) {
    state.datasetPicker?.dispose();
    state.datasetPicker = mountDatasetPresetPicker(state.root, {
        api,
        getCurrentFile: () => state.datasetFile,
        onApply: (file) => selectDataset(state, file),
    });
}

function mountImagePreview(state) {
    state.imagePreview?.dispose();
    state.imagePreview = mountTaggingResultImagePreview(state.root, {
        getImage: (itemId) => resultPreviewImage(state, itemId),
        getEditor: (image) => {
            const itemId = String(image?.id || '');
            const text = itemId ? draftFor(state, itemId) : '';
            return {
                itemId,
                text,
                mode: state.viewMode,
                busy: isBusy(state.job),
                saving: state.savingItemIds.has(itemId),
                translating: state.translatingItemIds.has(itemId),
                dirty: state.dirtyItemIds.has(itemId) && text !== originalCaptionFor(state, itemId),
                language: state.itemLanguages.get(itemId) || 'en',
            };
        },
    });
}

function disposePage(state) {
    state.active = false;
    state.jobEpoch += 1;
    state.requestId += 1;
    clearPoll(state);
    state.jobPoller = null;
    state.datasetPicker?.dispose();
    state.datasetPicker = null;
    state.imagePreview?.dispose();
    state.imagePreview = null;
    state.stopObserver?.();
    state.cleanup?.();
    state.root = null;
}

async function confirmLeave(state) {
    if (!state.dirtyItemIds.size) return true;
    return confirmDragonDialog({
        title: '离开最终结果',
        message: '仍有尚未保存的标注修改。',
        description: '离开后这些本地草稿会丢失。',
        confirmText: '仍然离开',
        tone: 'warning',
    });
}

function handleClick(state, event) {
    if (event.target.closest?.('[data-result-select]')) event.stopPropagation();
    const target = event.target.closest?.('[data-results-back], [data-results-dataset-open], [data-results-refresh], [data-results-rerun], [data-results-mode], [data-results-more], [data-results-select-all], [data-results-clear], [data-results-commit-selected], [data-results-commit-all], [data-result-image-open], [data-result-save], [data-result-translate], [data-result-tag-remove], [data-result-tag-add-button]');
    if (!target) return;
    if (target.matches('[data-results-back]')) return returnToTaggingWorkspace();
    if (target.matches('[data-results-dataset-open]')) return run(() => state.datasetPicker?.open());
    if (target.matches('[data-results-refresh]')) return run(() => refreshCurrentJob(state));
    if (target.matches('[data-results-rerun]')) return run(() => rerunJob(state));
    if (target.matches('[data-results-mode]')) return setViewMode(state, target.dataset.resultsMode);
    if (target.matches('[data-results-more]')) return revealMore(state);
    if (target.matches('[data-results-select-all]')) return selectAllCandidates(state);
    if (target.matches('[data-results-clear]')) return clearSelection(state);
    if (target.matches('[data-results-commit-selected]')) return run(() => commitResults(state, false));
    if (target.matches('[data-results-commit-all]')) return run(() => commitResults(state, true));
    if (target.matches('[data-result-image-open]')) return state.imagePreview?.open(target.dataset.itemId, target);
    if (target.matches('[data-result-save]')) return run(() => saveItem(state, target.dataset.itemId));
    if (target.matches('[data-result-translate]')) return run(() => translateItem(state, target.dataset.itemId));
    if (target.matches('[data-result-tag-remove]')) return removeTag(state, target.dataset.itemId, Number(target.dataset.tagIndex));
    if (target.matches('[data-result-tag-add-button]')) return addTagFromInput(state, target.dataset.itemId, target);
}

function handleChange(state, event) {
    const target = event.target;
    if (target.matches('[data-results-job]') && target.value) return run(() => selectJob(state, target.value));
    if (target.matches('[data-results-rerun-profile]')) {
        state.rerunProfileId = target.value;
        return;
    }
    if (!target.matches('[data-result-select]')) return;
    if (target.checked) state.selectedItemIds.add(target.dataset.itemId);
    else state.selectedItemIds.delete(target.dataset.itemId);
    syncSelection(state);
}

function handleInput(state, event) {
    const target = event.target;
    if (target.matches('[data-result-caption]')) return setDraft(state, target.dataset.itemId, target.value);
    if (!target.matches('[data-result-tag-text]')) return;
    const chip = target.closest('[data-result-tag]');
    if (!chip) return;
    setDraft(
        state,
        chip.dataset.itemId,
        replaceCaptionTag(draftFor(state, chip.dataset.itemId), Number(chip.dataset.tagIndex), target.textContent),
    );
}

function handleKeydown(state, event) {
    const target = event.target;
    if (target.matches('[data-result-tag-add]') && event.key === 'Enter') {
        event.preventDefault();
        addTagFromInput(state, target.dataset.itemId, target);
    }
    if (target.matches('[data-result-tag-text]') && event.key === 'Enter') {
        event.preventDefault();
        target.blur();
    }
}

function handleDragStart(state, event) {
    const chip = event.target.closest?.('[data-result-tag]');
    if (!chip) return;
    state.draggedTag = { itemId: chip.dataset.itemId, index: Number(chip.dataset.tagIndex) };
    event.dataTransfer?.setData('text/plain', `${chip.dataset.itemId}:${chip.dataset.tagIndex}`);
    if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
    chip.dataset.dragging = 'true';
}

function handleDragOver(state, event) {
    const chip = event.target.closest?.('[data-result-tag]');
    if (!chip || chip.dataset.itemId !== state.draggedTag?.itemId) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
    state.root?.querySelectorAll('[data-result-tag][data-drop-target]').forEach((node) => node.removeAttribute('data-drop-target'));
    chip.dataset.dropTarget = 'true';
}

function handleDrop(state, event) {
    const chip = event.target.closest?.('[data-result-tag]');
    const dragged = state.draggedTag;
    if (!chip || !dragged || chip.dataset.itemId !== dragged.itemId) return;
    event.preventDefault();
    const next = moveCaptionTag(draftFor(state, dragged.itemId), dragged.index, Number(chip.dataset.tagIndex));
    setDraft(state, dragged.itemId, next);
    clearDragState(state);
    rerender(state);
}

function clearDragState(state) {
    state.draggedTag = null;
    state.root?.querySelectorAll('[data-result-tag][data-dragging], [data-result-tag][data-drop-target]').forEach((node) => {
        node.removeAttribute('data-dragging');
        node.removeAttribute('data-drop-target');
    });
}

function addTagFromInput(state, itemId, source = null) {
    const scope = source?.closest?.('[data-result-editor-host], [data-result-preview-editor]') || state.root;
    const input = source?.matches?.('[data-result-tag-add]')
        ? source
        : scope?.querySelector(`[data-result-tag-add][data-item-id="${cssEscape(itemId)}"]`);
    if (!input?.value.trim()) return;
    setDraft(state, itemId, appendCaptionTag(draftFor(state, itemId), input.value));
    input.value = '';
    rerender(state);
}

function removeTag(state, itemId, index) {
    setDraft(state, itemId, removeCaptionTag(draftFor(state, itemId), index));
    rerender(state);
}

function setDraft(state, itemId, text) {
    const value = String(text ?? '');
    if (value === originalCaptionFor(state, itemId)) {
        state.drafts.delete(itemId);
        state.dirtyItemIds.delete(itemId);
    } else {
        state.drafts.set(itemId, value);
        state.dirtyItemIds.add(itemId);
    }
    const language = state.itemLanguages.get(itemId) || 'en';
    const translations = state.languageDrafts.get(itemId) || {};
    translations[language] = value;
    state.languageDrafts.set(itemId, translations);
    syncDraftCardEditor(state, itemId, value);
    syncItemSaveControl(state, itemId);
    syncItemTranslationControl(state, itemId);
    state.imagePreview?.sync?.();
}

function syncDraftCardEditor(state, itemId, text) {
    const card = state.root?.querySelector(`[data-result-item="${cssEscape(itemId)}"]`);
    const host = card?.querySelector('[data-result-editor-host]');
    if (!host || host.contains(globalThis.document?.activeElement)) return;
    host.innerHTML = renderCaptionEditor({
        itemId,
        text,
        mode: state.viewMode,
        busy: isBusy(state.job),
        saving: state.savingItemIds.has(itemId),
    });
}

function draftFor(state, itemId) {
    if (state.drafts.has(itemId)) return String(state.drafts.get(itemId) || '');
    return String(state.job?.items?.find((item) => item.id === itemId)?.proposed_caption || '');
}

function originalCaptionFor(state, itemId) {
    return String(state.job?.items?.find((item) => item.id === itemId)?.proposed_caption || '');
}

function resultPreviewImage(state, itemId) {
    const item = state.job?.items?.find((entry) => entry.id === itemId);
    if (!item) return null;
    const caption = draftFor(state, itemId);
    return {
        ...item,
        url: item.url || item.thumbnail_url || '',
        name: item.name || item.file || '打标图片',
        caption: {
            ok: Boolean(caption.trim()),
            text: caption,
            format_label: '当前候选标注',
            file: item.caption_file || '',
            source_label: state.job?.profile_name || state.job?.settings?.profile_name || '打标任务',
        },
    };
}

function setViewMode(state, mode) {
    if (!['tags', 'raw'].includes(mode) || state.viewMode === mode) return;
    state.viewMode = mode;
    globalThis.localStorage?.setItem('dragon.tagging.results.mode', mode);
    rerender(state);
}

async function selectDataset(state, file) {
    if (!await confirmDiscardDrafts(state, '切换数据集')) return false;
    state.datasetFile = String(file || '');
    const next = filteredJobs(state)[0] || null;
    if (!next) {
        resetJobState(state, null);
        rerender(state);
        return true;
    }
    await selectJob(state, next.id, { skipConfirm: true });
    return true;
}

async function selectJob(state, jobId, { skipConfirm = false } = {}) {
    if (!skipConfirm && !await confirmDiscardDrafts(state, '切换任务')) return;
    const epoch = ++state.jobEpoch;
    state.requestId += 1;
    clearPoll(state);
    resetJobState(state, state.job);
    state.visibleCount = RESULT_BATCH_SIZE;
    state.forceJobRender = true;
    await hydrateJob(state, jobId, epoch);
    if (state.active && state.jobEpoch === epoch) updateTaggingPromptDraft({ jobId });
}

async function confirmDiscardDrafts(state, title) {
    if (!state.dirtyItemIds.size) return true;
    return confirmDragonDialog({
        title,
        message: '放弃尚未保存的修改吗？',
        description: '当前卡片中的本地草稿不会写入任务。',
        confirmText: '放弃并切换',
        tone: 'warning',
    });
}

function resetJobState(state, nextJob) {
    state.job = nextJob;
    state.savingItemIds.clear();
    state.translatingItemIds.clear();
    state.selectedItemIds.clear();
    state.dirtyItemIds.clear();
    state.drafts.clear();
    state.languageDrafts.clear();
    state.itemLanguages.clear();
    state.committing = false;
    state.rerunning = false;
}

async function saveItem(state, itemId) {
    if (!state.job?.id || !state.dirtyItemIds.has(itemId) || state.savingItemIds.has(itemId)) return;
    const jobId = state.job.id;
    const epoch = state.jobEpoch;
    const requestId = state.requestId;
    state.savingItemIds.add(itemId);
    rerender(state);
    try {
        const payload = await updateTaggingItem(api, jobId, itemId, draftFor(state, itemId));
        if (!isCurrentJob(state, jobId, epoch, requestId)) return;
        state.job = payload.job || state.job;
        resetTranslationCacheAfterSave(state, itemId);
        state.drafts.delete(itemId);
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
    if (all && !await confirmDragonDialog({ title: '写回全部结果', message: '将全部候选写入图片同名 TXT。', description: '已有 TXT 会按当前候选内容更新。', confirmText: '全部写回', tone: 'warning' })) return;
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
    try {
        for (const itemId of [...state.dirtyItemIds]) {
            const payload = await updateTaggingItem(api, jobId, itemId, draftFor(state, itemId));
            if (!isCurrentJob(state, jobId, epoch, requestId)) return false;
            state.job = payload.job || state.job;
            resetTranslationCacheAfterSave(state, itemId);
            state.drafts.delete(itemId);
            state.dirtyItemIds.delete(itemId);
        }
        return true;
    } catch (error) {
        if (isCurrentJob(state, jobId, epoch, requestId)) state.error = error.message || '保存结果失败';
        return false;
    }
}

async function rerunJob(state) {
    if (!state.job?.id || !state.rerunProfileId || state.rerunning) return;
    const sourceJobId = state.job.id;
    const profileId = state.rerunProfileId;
    const profile = state.providerProfiles.find((item) => item.id === profileId);
    const selectedItemIds = (state.job.items || [])
        .filter((item) => state.selectedItemIds.has(item.id))
        .map((item) => item.id);
    const rerunCount = selectedItemIds.length || Number(state.job.total || state.job.items?.length || 0);
    const rerunScope = selectedItemIds.length ? `选中的 ${selectedItemIds.length} 张图片` : `全部 ${rerunCount} 张图片`;
    const discardsDrafts = state.dirtyItemIds.size > 0;
    const confirmed = await confirmDragonDialog({
        title: selectedItemIds.length ? '重新打标选中图片' : '重新打标当前任务',
        message: `使用“${profile?.name || '所选接入'}”重新处理${rerunScope}。`,
        description: discardsDrafts
            ? '当前未保存的修改会丢失。结果会在当前任务中原位更新，审阅后再手动写回 TXT。'
            : '结果会在当前任务中原位更新，不会立即覆盖现有 TXT；审阅后再手动写回。',
        confirmText: '开始重新打标',
        tone: discardsDrafts ? 'warning' : 'info',
    });
    if (!confirmed || !state.active || state.job?.id !== sourceJobId) return;
    const epoch = ++state.jobEpoch;
    const requestId = ++state.requestId;
    clearPoll(state);
    state.rerunning = true;
    rerender(state);
    try {
        const payload = await rerunTaggingJob(api, sourceJobId, profileId, selectedItemIds);
        if (!isCurrentJob(state, sourceJobId, epoch, requestId)) return;
        const nextJob = payload.job;
        if (!nextJob || nextJob.id !== sourceJobId) throw new Error('重新打标未复用当前任务');
        updateJobSummary(state, nextJob);
        resetJobState(state, nextJob);
        state.datasetFile = nextJob?.dataset_file || state.datasetFile;
        state.rerunProfileId = chooseRerunProfile(nextJob, state.providerProfiles);
        state.visibleCount = Math.min(RESULT_BATCH_SIZE, nextJob?.items?.length || 0);
        state.jobSignature = jobSignature(nextJob);
        state.notice = '已在当前任务中重新开始处理。';
        updateTaggingPromptDraft({ jobId: sourceJobId });
        if (isBusy(nextJob)) schedulePoll(state);
    } catch (error) {
        if (isCurrentJob(state, sourceJobId, epoch, requestId)) state.error = error.message || '重新打标失败';
    } finally {
        if (state.active && state.jobEpoch === epoch && state.requestId === requestId) {
            state.rerunning = false;
            rerender(state);
        }
    }
}

async function translateItem(state, itemId) {
    if (!state.job?.id || state.translatingItemIds.has(itemId)) return;
    const jobId = state.job.id;
    const epoch = state.jobEpoch;
    const requestId = state.requestId;
    const isCurrent = () => isCurrentJob(state, jobId, epoch, requestId);
    const currentLanguage = state.itemLanguages.get(itemId) || 'en';
    const targetLanguage = currentLanguage === 'en' ? 'zh' : 'en';
    const translations = state.languageDrafts.get(itemId) || { [currentLanguage]: draftFor(state, itemId) };
    if (translations[targetLanguage] != null) {
        state.itemLanguages.set(itemId, targetLanguage);
        setDraft(state, itemId, translations[targetLanguage]);
        return;
    }
    state.translatingItemIds.add(itemId);
    syncItemTranslationControl(state, itemId);
    try {
        if (!await ensureTagDictionary(state, isCurrent) || !isCurrent()) return;
        const source = translations[currentLanguage] ?? draftFor(state, itemId);
        const payload = await translateCaptionTags(api, splitCaptionTags(source), targetLanguage);
        if (!isCurrent()) return;
        const translated = joinCaptionTags(payload.translations || []);
        translations[currentLanguage] = source;
        translations[targetLanguage] = translated;
        state.languageDrafts.set(itemId, translations);
        state.itemLanguages.set(itemId, targetLanguage);
        setDraft(state, itemId, translated);
        state.notice = `已匹配翻译 ${Number(payload.matched || 0)}/${Number(payload.total || 0)} 个 tag。`;
    } catch (error) {
        if (isCurrent()) state.error = error.message || '标签翻译失败';
    } finally {
        if (isCurrent()) {
            state.translatingItemIds.delete(itemId);
            syncItemTranslationControl(state, itemId);
            syncFeedback(state);
        }
    }
}

async function ensureTagDictionary(state, isCurrent = () => state.active) {
    const status = await loadTagDictionaryStatus(api);
    if (!isCurrent()) return false;
    if (status.installed) return true;
    const confirmed = await confirmDragonDialog({
        title: '安装本地中英标签词典',
        message: '需要下载约 23 MB 的 Danbooru 中英标签词典。',
        description: '词典来自固定版本的 ffdkj 数据集，保存在本机用户数据目录；标注不会发送给 Google 或其他翻译服务。',
        confirmText: '下载并继续',
        tone: 'info',
    });
    if (!confirmed || !isCurrent()) return false;
    const started = await startTagDictionaryDownload(api);
    if (!isCurrent()) return false;
    const downloadId = started.download?.id;
    if (!downloadId) throw new Error('未能创建标签词典下载任务');
    state.notice = '正在下载并构建本地标签词典…';
    syncFeedback(state);
    for (;;) {
        if (!isCurrent()) return false;
        const payload = await loadTagDictionaryDownload(api, downloadId);
        if (!isCurrent()) return false;
        const download = payload.download || {};
        if (download.state === 'completed') return true;
        if (['error', 'canceled'].includes(download.state)) throw new Error(download.error || '标签词典下载未完成');
        await wait(DOWNLOAD_POLL_MS);
    }
}

async function refreshCurrentJob(state) {
    if (!state.job?.id) return;
    const jobId = state.job.id;
    const epoch = state.jobEpoch;
    const jobsPayload = await loadTaggingJobs(api);
    if (!state.active || state.jobEpoch !== epoch || state.job?.id !== jobId) return;
    state.jobs = Array.isArray(jobsPayload.jobs) ? jobsPayload.jobs : state.jobs;
    state.forceJobRender = true;
    await hydrateJob(state, jobId, epoch);
    if (!state.active || state.jobEpoch !== epoch || state.job?.id !== jobId) return;
    state.notice = '任务结果已刷新。';
    syncFeedback(state);
}

async function hydrateJob(state, jobId, epoch = state.jobEpoch) {
    const requestId = ++state.requestId;
    try {
        const payload = await loadTaggingJob(api, jobId);
        if (!isCurrentRequest(state, epoch, requestId)) return;
        const previousJobId = state.job?.id || '';
        const nextJob = payload.job || null;
        const changed = state.jobSignature !== jobSignature(nextJob);
        const forceRender = state.forceJobRender;
        state.forceJobRender = false;
        state.job = nextJob;
        const currentIds = new Set((nextJob?.items || []).map((item) => item.id).filter(Boolean));
        state.selectedItemIds = new Set([...state.selectedItemIds].filter((itemId) => currentIds.has(itemId)));
        state.datasetFile = nextJob?.dataset_file || state.datasetFile;
        state.rerunProfileId ||= chooseRerunProfile(nextJob, state.providerProfiles);
        state.visibleCount = Math.min(Math.max(state.visibleCount, RESULT_BATCH_SIZE), nextJob?.items?.length || 0);
        state.jobSignature = jobSignature(nextJob);
        if (changed || forceRender) {
            const sameJob = Boolean(previousJobId && previousJobId === nextJob?.id);
            if (forceRender || !sameJob || !syncResultsJobView(state)) rerender(state);
        }
        if (isBusy(nextJob)) schedulePoll(state);
        else clearPoll(state);
    } catch (error) {
        if (isCurrentRequest(state, epoch, requestId)) {
            state.error = error.message || '读取打标结果失败';
            syncFeedback(state);
        }
    }
}

function revealMore(state) {
    const total = state.job?.items?.length || 0;
    const previousCount = state.visibleCount;
    state.visibleCount = Math.min(total, previousCount + RESULT_BATCH_SIZE);
    if (state.visibleCount <= previousCount) return;
    const list = state.root?.querySelector('[data-results-list]');
    const sentinel = list?.querySelector('[data-results-sentinel]');
    if (!list || !sentinel) return rerender(state);
    sentinel.insertAdjacentHTML('beforebegin', renderResultRows(state, previousCount, state.visibleCount));
    updateResultSentinel(state);
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
    const busy = isBusy(state.job);
    const candidates = committableItems(state.job?.items || []);
    const selectedWritableCount = candidates.filter((item) => state.selectedItemIds.has(item.id)).length;
    state.root?.querySelectorAll('[data-results-selected-count]').forEach((node) => { node.textContent = String(selectedWritableCount); });
    state.root?.querySelectorAll('[data-results-rerun-label]').forEach((node) => {
        node.textContent = state.selectedItemIds.size ? `重新打标已选 ${state.selectedItemIds.size}` : '重新打标';
    });
    setDisabled(state.root, '[data-results-commit-selected]', !selectedWritableCount || busy || state.committing);
    setDisabled(state.root, '[data-results-clear]', !state.selectedItemIds.size);
    setDisabled(state.root, '[data-results-select-all]', !candidates.length || busy);
    setDisabled(state.root, '[data-results-commit-all]', !candidates.length || busy || state.committing);
}

function syncResultsJobView(state) {
    const root = state.root;
    const job = state.job;
    if (!root || !job?.id || !root.querySelector('[data-results-job-status]')) return false;
    const busy = isBusy(job);
    const status = root.querySelector('[data-results-job-status]');
    status.dataset.state = jobStatusTone(job.state);
    const statusLabel = status.querySelector('b');
    if (statusLabel) statusLabel.textContent = jobStateLabel(job.state);
    setText(root, '[data-results-progress]', `${Number(job.completed || 0)}/${Number(job.total || job.items?.length || 0)} 完成`);
    const itemsById = new Map((job.items || []).map((item) => [item.id, item]));
    root.querySelectorAll('[data-result-item]').forEach((card) => syncResultCard(state, card, itemsById.get(card.dataset.resultItem), busy));
    syncSelection(state);
    updateResultSentinel(state);
    syncFeedback(state);
    return true;
}

function syncResultCard(state, card, item, busy) {
    if (!item) return;
    card.dataset.state = item.state || '';
    const status = card.querySelector('[data-result-item-status]');
    if (status) {
        const hideStatus = ['ready', 'queued'].includes(item.state);
        status.hidden = hideStatus;
        status.dataset.state = statusTone(item.state);
        const label = status.querySelector('b');
        if (label) label.textContent = hideStatus ? '' : itemStateLabel(item.state);
    }
    const checkbox = card.querySelector('[data-result-select]');
    if (checkbox) {
        checkbox.checked = state.selectedItemIds.has(item.id);
        checkbox.disabled = !item.id || busy;
    }
    if (state.dirtyItemIds.has(item.id) && draftFor(state, item.id) === String(item.proposed_caption || '')) {
        state.drafts.delete(item.id);
        state.dirtyItemIds.delete(item.id);
    }
    if (!state.dirtyItemIds.has(item.id)) {
        const editor = card.querySelector('[data-result-editor-host]');
        if (editor && !editor.contains(globalThis.document?.activeElement)) {
            editor.innerHTML = renderCaptionEditor({ itemId: item.id, text: item.proposed_caption || '', mode: state.viewMode, busy, saving: state.savingItemIds.has(item.id) });
        }
    }
    syncItemSaveControl(state, item.id, card);
}

function syncItemSaveControl(state, itemId, card = null) {
    const itemCard = card || state.root?.querySelector(`[data-result-item="${cssEscape(itemId)}"]`);
    const disabled = isBusy(state.job) || state.savingItemIds.has(itemId) || !state.dirtyItemIds.has(itemId);
    itemCard?.querySelectorAll('[data-result-save]')?.forEach((button) => { button.disabled = disabled; });
    state.root?.querySelectorAll('[data-result-save]')?.forEach((button) => {
        if (button.dataset.itemId === itemId) button.disabled = disabled;
    });
}

function syncItemTranslationControl(state, itemId) {
    const translating = state.translatingItemIds.has(itemId);
    const labelText = translating
        ? '翻译中…'
        : state.itemLanguages.get(itemId) === 'zh' ? 'EN' : '中文';
    const selector = `[data-result-translate][data-item-id="${cssEscape(itemId)}"]`;
    state.root?.querySelectorAll(selector).forEach((button) => {
        button.disabled = isBusy(state.job) || translating;
        const label = button.querySelector('[data-result-translate-label]') || button.querySelector('span:last-child');
        if (label) label.textContent = labelText;
    });
}

function resetTranslationCacheAfterSave(state, itemId) {
    const language = state.itemLanguages.get(itemId) || 'en';
    state.languageDrafts.set(itemId, { [language]: originalCaptionFor(state, itemId) });
}

function updateResultSentinel(state) {
    const sentinel = state.root?.querySelector('[data-results-sentinel]');
    if (sentinel) sentinel.hidden = state.visibleCount >= (state.job?.items?.length || 0);
}

function syncFeedback(state) {
    const host = state.root?.querySelector('[data-results-feedback]');
    if (host) host.innerHTML = renderResultsFeedback(state);
}

function reconnectObserver(state) {
    state.stopObserver?.();
    const sentinel = state.root?.querySelector('[data-results-sentinel]');
    if (!sentinel || typeof IntersectionObserver !== 'function') return;
    const observer = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) revealMore(state);
    }, { rootMargin: '420px 0px', threshold: 0.01 });
    observer.observe(sentinel);
    state.stopObserver = () => observer.disconnect();
}

function rerender(state) {
    if (!state.active || !state.root) return;
    const scrollY = globalThis.scrollY || 0;
    const previewItemId = state.imagePreview?.isOpen?.() ? state.imagePreview.getActiveItemId?.() : '';
    state.imagePreview?.dispose();
    state.imagePreview = null;
    state.root.innerHTML = renderResultsPage(state, DATASET_PICKER_HTML);
    mountDatasetPicker(state);
    mountImagePreview(state);
    reconnectObserver(state);
    if (previewItemId) {
        const trigger = [...state.root.querySelectorAll('[data-result-image-open]')]
            .find((button) => button.dataset.itemId === previewItemId);
        if (trigger) state.imagePreview?.open(previewItemId, trigger);
    }
    globalThis.requestAnimationFrame?.(() => globalThis.scrollTo?.({ top: scrollY, behavior: 'auto' }));
}

function schedulePoll(state) {
    if (!state.active || !state.job?.id || !isBusy(state.job)) return clearPoll(state);
    state.jobPoller?.start();
}

function clearPoll(state) {
    state.jobPoller?.stop();
}

function chooseRerunProfile(job, profiles) {
    const available = profiles.filter((profile) => profile.available);
    return available.find((profile) => profile.id === job?.profile_id)?.id || available[0]?.id || '';
}

function updateJobSummary(state, job) {
    const summary = state.jobs.find((entry) => entry.id === job.id);
    if (!summary) return;
    Object.assign(summary, {
        state: job.state,
        created_at: job.created_at,
        created_at_text: job.created_at_text,
        dataset_file: job.dataset_file,
        dataset_index: job.dataset_index,
        source: job.source,
        profile_id: job.profile_id,
        profile_name: job.profile_name,
        total: job.total,
        completed: job.completed,
        failed: job.failed,
        canceled: job.canceled,
        model: job.settings?.model || summary.model || '',
    });
}

function readViewMode() {
    const value = globalThis.localStorage?.getItem('dragon.tagging.results.mode');
    return value === 'raw' ? 'raw' : 'tags';
}

function jobSignature(job) {
    if (!job) return '';
    const items = (job.items || []).map((item) => [item.id || item.file || '', item.state || '', String(item.proposed_caption || '').length, String(item.caption || '').length, String(item.error || '').length, String(item.commit_error || '').length].join(':')).join('|');
    return [job.id || '', job.state || '', job.total || 0, job.completed || 0, job.failed || 0, job.canceled || 0, job.error || '', items].join('|');
}

function isCurrentJob(state, jobId, epoch, requestId) {
    return isCurrentRequest(state, epoch, requestId) && state.job?.id === jobId;
}

function isCurrentRequest(state, epoch, requestId) {
    return Boolean(state.active) && state.jobEpoch === epoch && state.requestId === requestId;
}

function setText(root, selector, value) {
    const node = root?.querySelector?.(selector);
    if (node) node.textContent = value;
}

function setDisabled(root, selector, disabled) {
    const node = root?.querySelector?.(selector);
    if (node) node.disabled = Boolean(disabled);
}

function cssEscape(value) {
    if (globalThis.CSS?.escape) return globalThis.CSS.escape(String(value || ''));
    return String(value || '').replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

function wait(milliseconds) {
    return new Promise((resolve) => globalThis.setTimeout(resolve, milliseconds));
}

function run(fn) {
    Promise.resolve().then(fn).catch((error) => console.error('[dragon-tagging-results]', error));
}
