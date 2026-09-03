/* Dragon tagging workbench controller. Provider calls stay server-side. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { loadTrainingContext } from './training-controls.js?v=dragon-ui-20260901v115';
import {
    mountTaggingView,
    renderTaggingView,
    syncTaggingSelectionView,
    syncTaggingJobView,
} from './tagging-view.js?v=dragon-ui-20260902v15';
import {
    appendTaggingImageCards,
    syncTaggingSource,
    updateLoadSentinel,
} from './tagging-source-view.js?v=dragon-ui-20260901v9';
import {
    TAGGING_IMAGE_PAGE_SIZE,
    TAGGING_JOB_ITEM_LIMIT,
    cancelTaggingJob,
    createTaggingJob,
    activateProviderProfile,
    loadImages,
    loadPreset,
    loadPromptPresets,
    loadProviderProfiles,
    loadTaggingJob,
    loadTaggingJobs,
    loadTaggingSettings,
    saveTaggingSettings,
    testTaggingProvider,
} from './tagging-api.js?v=dragon-ui-20260901v6';
import { consumeTaggingPrefill } from './tagging-context.js?v=dragon-ui-20260831v2';
import { createVisibilityPoller } from '../visibility-poller.js?v=dragon-ui-20260826v2';
import {
    openTaggingTool,
    readTaggingWorkspaceState,
    restoreTaggingWorkspacePosition,
    saveTaggingWorkspaceState,
} from './tagging-workspace-state.js?v=dragon-ui-20260831v4';

const api = createApiClient();
const DEFAULT_USER_PROMPT = '用训练 caption 描述主体、服装、姿态、镜头和背景，不要添加无法确认的内容。';
const POLL_INTERVAL_MS = 2000;

export async function loadTagging(routeContext = {}) {
    const saved = readTaggingWorkspaceState();
    const state = createState(routeContext, saved);
    const prefill = consumeTaggingPrefill();
    await loadInitialData(state, prefill, saved);
    return {
        html: renderTaggingViewHtml(state),
        onMount: (root) => mountPage(root, state),
        onUnmount: () => disposePage(state),
    };
}

export const loadCaptioning = loadTagging;

function createState(routeContext, saved) {
    const restoreImages = saved.sourceExpanded === true && saved.imagesLoaded === true;
    return {
        routeContext,
        active: true,
        settings: null,
        presets: [],
        promptPresets: [],
        providerProfiles: [],
        activeProfileId: saved.providerProfileId || '',
        currentPresetId: saved.currentPresetId || '',
        rows: Array.isArray(saved.rows) ? saved.rows : [],
        datasetFile: saved.datasetFile || '',
        datasetIndex: Number(saved.datasetIndex || 0),
        source: saved.source || 'source',
        images: restoreImages && Array.isArray(saved.images) ? saved.images : [],
        imagesLoaded: restoreImages,
        directory: saved.directory || '',
        total: Number(saved.total || 0),
        nextOffset: Number(saved.nextOffset || 0),
        hasMore: saved.hasMore === true,
        selectedFiles: new Set(saved.selectedFiles || []),
        systemPrompt: saved.systemPrompt || '',
        userPrompt: saved.userPrompt || DEFAULT_USER_PROMPT,
        sourceExpanded: saved.sourceExpanded === true,
        job: saved.job || null,
        jobId: saved.jobId || saved.job?.id || '',
        jobs: [],
        loadingImages: false,
        loadingMore: false,
        loadingPreset: false,
        selectingAll: false,
        error: '',
        notice: '',
        submitting: false,
        testingProvider: false,
        switchingProfile: false,
        requestSequence: 0,
        loadMoreRequestId: 0,
        jobRequestId: 0,
        pollTimer: null,
        jobPoller: null,
        jobSignature: jobSignature(saved.job),
        root: null,
        cleanupView: null,
        reconnectImageObserver: null,
        gridScrollTop: Number(saved.gridScrollTop || 0),
        savedScrollY: Number(saved.scrollY || 0),
    };
}

async function loadInitialData(state, prefill, saved) {
    const results = await Promise.allSettled([
        loadTrainingContext({ includeGpus: false }),
        loadTaggingSettings(api),
        api('/api/config/dataset-presets'),
        loadTaggingJobs(api),
        loadPromptPresets(api),
        loadProviderProfiles(api),
    ]);
    const trainingContext = valueFrom(results[0], {});
    state.settings = publicSettings(valueFrom(results[1], {}));
    state.systemPrompt ||= state.settings.system_prompt || '';
    const library = valueFrom(results[2], {});
    state.presets = Array.isArray(library.presets) ? library.presets : [];
    state.jobs = valueFrom(results[3], {}).jobs || [];
    state.promptPresets = valueFrom(results[4], {}).presets || [];
    const profileLibrary = valueFrom(results[5], {});
    state.providerProfiles = Array.isArray(profileLibrary.profiles) ? profileLibrary.profiles : [];
    state.activeProfileId = profileLibrary.active_profile_id || state.settings.profile_id || state.activeProfileId;
    const failed = results.find((result) => result.status === 'rejected');
    if (failed) state.error = failed.reason?.message || '读取打标工作台数据失败';

    const linked = await readLinkedDataset(trainingContext, state);
    const prefillFile = prefill.dataset_file || '';
    state.datasetFile = chooseDatasetFile(prefillFile || saved.datasetFile, linked.dataset_config, state.presets);
    state.datasetIndex = prefillFile ? Number(prefill.dataset_index || 0) : Number(saved.datasetIndex || 0);
    state.source = prefillFile ? (prefill.source || 'source') : (saved.source || 'source');
    state.userPrompt = prefill.prompt || saved.userPrompt || DEFAULT_USER_PROMPT;
    if (prefillFile) state.selectedFiles = new Set(prefill.selected_files || []);
    if (prefill.image_file) state.selectedFiles.add(prefill.image_file);

    const cacheMatches = !prefillFile && state.sourceExpanded && cachedDatasetMatches(saved, state);
    if (state.datasetFile) {
        await loadDatasetAndImages(state, {
            reuseImages: cacheMatches,
            selectedFiles: [...state.selectedFiles],
            loadImages: state.sourceExpanded,
        });
    }
    const jobId = state.jobId || latestMatchingJob(state)?.id || '';
    if (jobId) await hydrateJob(state, jobId);
    saveTaggingWorkspaceState(state);
}

async function readLinkedDataset(context, state) {
    if (!context?.configFile) return {};
    const params = new URLSearchParams({
        variant: context.variant || 'lora',
        preset: context.preset || 'default',
        methods_subdir: context.methodsSubdir || 'gui-methods',
        config_file: context.configFile,
    });
    try {
        const payload = await api(`/api/config/datasets?${params.toString()}`);
        return payload?.ok === false ? {} : payload;
    } catch (error) {
        state.error ||= error.message || '读取当前数据集失败';
        return {};
    }
}

async function loadDatasetAndImages(state, options = {}) {
    const sequence = ++state.requestSequence;
    state.loadingPreset = true;
    state.error = '';
    try {
        const preset = await loadPreset(api, state.datasetFile);
        if (!isCurrent(state, sequence)) return;
        state.rows = Array.isArray(preset.datasets) ? preset.datasets : [];
        state.datasetIndex = clampIndex(state.datasetIndex, state.rows.length);
        if (options.reuseImages && state.imagesLoaded) {
            state.loadingImages = false;
            state.loadingPreset = false;
            return;
        }
        if (options.loadImages === false) {
            state.selectedFiles = new Set(options.selectedFiles || state.selectedFiles);
            return;
        }
        await refreshImages(state, { sequence, selectedFiles: options.selectedFiles });
    } catch (error) {
        if (isCurrent(state, sequence)) resetImagesWithError(state, error.message || '读取数据集预设失败');
    } finally {
        if (isCurrent(state, sequence)) state.loadingPreset = false;
    }
}

async function refreshImages(state, options = {}) {
    const sequence = options.sequence || ++state.requestSequence;
    const preserveDom = options.preserveDom === true;
    state.loadMoreRequestId += 1;
    state.loadingImages = true;
    syncSourceOrRerender(state, preserveDom);
    try {
        const payload = await loadImages(api, state.datasetFile, state.datasetIndex, state.source, {
            limit: TAGGING_IMAGE_PAGE_SIZE,
            offset: 0,
        });
        if (!isCurrent(state, sequence)) return;
        state.images = uniqueImages(payload.images);
        state.directory = payload.directory || '';
        state.total = Number(payload.total ?? state.images.length);
        state.nextOffset = Number(payload.next_offset ?? state.images.length);
        state.hasMore = payload.has_more_after ?? state.nextOffset < state.total;
        state.imagesLoaded = true;
        state.selectedFiles = new Set(options.selectedFiles || state.selectedFiles);
        state.error = '';
    } catch (error) {
        if (isCurrent(state, sequence)) resetImagesWithError(state, error.message || '扫描数据集图片失败');
    } finally {
        if (isCurrent(state, sequence)) {
            state.loadingImages = false;
            syncSourceOrRerender(state, preserveDom);
            saveTaggingWorkspaceState(state);
        }
    }
}

async function loadMoreImages(state) {
    if (!state.active || state.loadingMore || !state.hasMore || !state.datasetFile) return false;
    const requestId = ++state.loadMoreRequestId;
    const sequence = state.requestSequence;
    const offset = state.nextOffset;
    state.loadingMore = true;
    updateLoadSentinel(state.root, state);
    try {
        const payload = await loadImages(api, state.datasetFile, state.datasetIndex, state.source, {
            limit: TAGGING_IMAGE_PAGE_SIZE,
            offset,
        });
        if (!state.active || sequence !== state.requestSequence || requestId !== state.loadMoreRequestId) return false;
        const existing = new Set(state.images.map((image) => image.file));
        const added = uniqueImages(payload.images).filter((image) => image.file && !existing.has(image.file));
        state.images.push(...added);
        state.total = Number(payload.total ?? state.total ?? state.images.length);
        const reportedNextOffset = Number(payload.next_offset);
        const nextOffset = Number.isFinite(reportedNextOffset) ? reportedNextOffset : offset + uniqueImages(payload.images).length;
        if (nextOffset <= offset && (payload.has_more_after ?? true)) {
            throw new Error('图片分页没有继续前进，请刷新后重试');
        }
        state.nextOffset = Math.max(offset, nextOffset);
        state.hasMore = payload.has_more_after ?? state.nextOffset < state.total;
        appendTaggingImageCards(state.root, state, added);
        saveTaggingWorkspaceState(state);
        return added.length > 0;
    } catch (error) {
        if (state.active && sequence === state.requestSequence) {
            state.error = error.message || '加载更多图片失败';
            syncTaggingSelectionView(state.root, state);
        }
        return false;
    } finally {
        if (requestId === state.loadMoreRequestId) {
            state.loadingMore = false;
            updateLoadSentinel(state.root, state);
        }
    }
}

function mountPage(root, state) {
    state.root = root;
    state.cleanupView = mountTaggingView(root, state, {
        selectDataset: (file) => selectDataset(state, file),
        selectIndex: (index) => selectIndex(state, index),
        selectSource: (source) => selectSource(state, source),
        refreshImages: () => refreshImages(state, { preserveDom: true }),
        loadMoreImages: () => loadMoreImages(state),
        selectAll: () => selectAll(state),
        clearSelection: () => clearSelection(state),
        toggleImage: (file, checked) => toggleImage(state, file, checked),
        submitJob: (prompts) => submitJob(state, prompts),
        cancelJob: () => cancelJob(state),
        refreshJob: () => refreshJob(state),
        applyPromptPreset: (presetId) => applyPromptPreset(state, presetId),
        selectProviderProfile: (profileId) => selectProviderProfile(state, profileId),
        updatePromptDraft: (key, value) => updatePromptDraft(state, key, value),
        setSourceExpanded: (open) => setSourceExpanded(state, open),
        openTool: (page) => openTaggingTool(state, page),
        saveSettings: (data) => saveSettings(state, data),
        testProvider: (mode) => testProvider(state, mode),
    });
    restoreTaggingWorkspacePosition(root);
    state.jobPoller = createVisibilityPoller({
        delay: POLL_INTERVAL_MS,
        poll: async () => {
            const jobId = state.job?.id;
            if (!state.active || !jobId || !isJobBusy(state)) {
                state.jobPoller?.stop();
                return;
            }
            await hydrateJob(state, jobId);
            if (!isJobBusy(state)) state.jobPoller?.stop();
        },
    });
    if (state.job && ['queued', 'running'].includes(state.job.state)) schedulePoll(state);
}

function disposePage(state) {
    saveTaggingWorkspaceState(state, { capturePosition: true });
    state.active = false;
    state.jobRequestId += 1;
    state.requestSequence += 1;
    state.loadMoreRequestId += 1;
    clearPoll(state);
    state.jobPoller = null;
    state.cleanupView?.();
    state.cleanupView = null;
    state.root = null;
}

async function selectDataset(state, file) {
    if (isJobBusy(state) || !file || file === state.datasetFile) return;
    clearCurrentJob(state);
    state.datasetFile = file;
    state.datasetIndex = 0;
    state.selectedFiles.clear();
    resetImages(state);
    await loadDatasetAndImages(state);
}

async function selectIndex(state, index) {
    const next = clampIndex(index, state.rows.length);
    if (isJobBusy(state) || next === state.datasetIndex) return;
    clearCurrentJob(state);
    state.datasetIndex = next;
    state.selectedFiles.clear();
    await refreshImages(state, { preserveDom: true });
}

async function selectSource(state, source) {
    const next = source === 'training' ? 'training' : 'source';
    if (isJobBusy(state) || next === state.source) return;
    clearCurrentJob(state);
    state.source = next;
    state.selectedFiles.clear();
    await refreshImages(state, { preserveDom: true });
}

async function selectAll(state) {
    if (state.selectingAll || isJobBusy(state)) return;
    const sequence = state.requestSequence;
    state.selectingAll = true;
    syncTaggingSelectionView(state.root, state);
    try {
        await ensureImagesLoaded(state);
        const target = Math.min(Number(state.total || 0), TAGGING_JOB_ITEM_LIMIT);
        const selectedFiles = await collectSelectionFiles(state, target);
        if (!isCurrent(state, sequence)) return;
        state.selectedFiles = selectedFiles;
        state.notice = state.total > TAGGING_JOB_ITEM_LIMIT ? `已选择前 ${TAGGING_JOB_ITEM_LIMIT} 张，达到单次任务上限。` : '';
    } catch (error) {
        if (isCurrent(state, sequence)) {
            state.error = error.message || '全选图片失败';
            syncTaggingSelectionView(state.root, state);
        }
    } finally {
        if (isCurrent(state, sequence)) {
            state.selectingAll = false;
            syncTaggingSelectionView(state.root, state);
            saveTaggingWorkspaceState(state);
        }
    }
}

async function collectSelectionFiles(state, target) {
    const selected = new Set(state.images.slice(0, target).map((image) => image.file).filter(Boolean));
    let offset = state.nextOffset;
    const sequence = state.requestSequence;
    while (selected.size < target && offset < state.total && state.active) {
        const payload = await loadImages(api, state.datasetFile, state.datasetIndex, state.source, {
            limit: Math.min(TAGGING_IMAGE_PAGE_SIZE, target - selected.size),
            offset,
        });
        if (!isCurrent(state, sequence)) throw new Error('数据集已切换，请重新全选');
        uniqueImages(payload.images).forEach((image) => {
            if (selected.size < target && image.file) selected.add(image.file);
        });
        const nextOffset = Number(payload.next_offset ?? offset + TAGGING_IMAGE_PAGE_SIZE);
        if (nextOffset <= offset) break;
        offset = nextOffset;
    }
    return selected;
}

function clearSelection(state) {
    state.selectedFiles.clear();
    state.notice = '';
    syncTaggingSelectionView(state.root, state);
    saveTaggingWorkspaceState(state);
}

function toggleImage(state, file, checked) {
    if (!file) return;
    if (checked && state.selectedFiles.size >= TAGGING_JOB_ITEM_LIMIT && !state.selectedFiles.has(file)) {
        state.notice = `单次任务最多选择 ${TAGGING_JOB_ITEM_LIMIT} 张图片。`;
    } else if (checked) {
        state.selectedFiles.add(file);
        state.notice = '';
    } else {
        state.selectedFiles.delete(file);
    }
    syncTaggingSelectionView(state.root, state);
    saveTaggingWorkspaceState(state);
}

async function submitJob(state, prompts = {}) {
    state.systemPrompt = String(prompts.systemPrompt ?? state.systemPrompt).trim();
    state.userPrompt = String(prompts.userPrompt ?? state.userPrompt).trim();
    const profile = (state.providerProfiles || []).find((item) => item.id === state.activeProfileId);
    const promptsRequired = profile?.kind !== 'local';
    if (isJobBusy(state) || !state.datasetFile || !state.selectedFiles.size || (promptsRequired && (!state.systemPrompt || !state.userPrompt))) return;
    const imageByFile = new Map(state.images.map((image) => [image.file, image]));
    const items = [...state.selectedFiles].map((file) => {
        const image = imageByFile.get(file);
        return image
            ? { file, url: image.url || '', thumbnail_url: image.thumbnail_url || '' }
            : file;
    });
    state.submitting = true;
    rerender(state);
    try {
        const payload = await createTaggingJob(api, {
            dataset_file: state.datasetFile,
            dataset_index: state.datasetIndex,
            source: state.source,
            system_prompt: state.systemPrompt,
            user_prompt: state.userPrompt,
            profile_id: state.activeProfileId,
            items,
        });
        if (!state.active) return;
        state.job = payload.job || null;
        state.jobId = state.job?.id || '';
        state.notice = '任务已创建。';
        state.error = '';
        if (state.job?.id) schedulePoll(state);
    } catch (error) {
        if (state.active) state.error = error.message || '创建打标任务失败';
    } finally {
        state.submitting = false;
        rerender(state);
        saveTaggingWorkspaceState(state);
    }
}

async function refreshJob(state) {
    if (state.job?.id) await hydrateJob(state, state.job.id);
}

async function hydrateJob(state, jobId) {
    const requestId = ++state.jobRequestId;
    try {
        const payload = await loadTaggingJob(api, jobId);
        if (!state.active || requestId !== state.jobRequestId) return;
        const nextJob = payload.job || null;
        const changed = state.jobSignature !== jobSignature(nextJob);
        state.job = nextJob;
        state.jobId = state.job?.id || '';
        state.jobSignature = jobSignature(state.job);
        if (changed) {
            syncTaggingJobView(state.root, state);
            saveTaggingWorkspaceState(state);
        }
        if (state.job && ['queued', 'running'].includes(state.job.state)) schedulePoll(state);
        else clearPoll(state);
    } catch (error) {
        if (state.active && requestId === state.jobRequestId) {
            const message = error.message || '读取打标任务失败';
            if (state.error !== message) {
                state.error = message;
                syncTaggingJobView(state.root, state);
            }
        }
    }
}

function schedulePoll(state) {
    if (!state.active || !state.job?.id || !['queued', 'running'].includes(state.job.state)) return clearPoll(state);
    if (!state.jobPoller) return;
    state.jobPoller.start();
}

function clearPoll(state) {
    state.jobPoller?.stop();
    state.pollTimer = null;
}

async function cancelJob(state) {
    if (!state.job?.id) return;
    try {
        const payload = await cancelTaggingJob(api, state.job.id);
        state.job = payload.job || state.job;
        state.jobSignature = jobSignature(state.job);
        if (['queued', 'running'].includes(state.job?.state)) schedulePoll(state);
        else clearPoll(state);
        rerender(state);
        saveTaggingWorkspaceState(state);
    } catch (error) {
        state.error = error.message || '停止任务失败';
        rerender(state);
    }
}

function applyPromptPreset(state, presetId) {
    const preset = state.promptPresets.find((item) => item.id === presetId);
    state.currentPresetId = preset?.id || '';
    if (preset) {
        state.systemPrompt = preset.system_prompt || state.systemPrompt;
        state.userPrompt = preset.user_prompt || state.userPrompt;
    }
    rerender(state);
    saveTaggingWorkspaceState(state);
}

async function selectProviderProfile(state, profileId) {
    const profile = state.providerProfiles.find((item) => item.id === profileId);
    if (!profile || profile.id === state.activeProfileId || isJobBusy(state)) return;
    if (!profile.available) {
        state.error = profile.kind === 'local' ? '这个本地模型尚未安装或启用。' : '这个接入预设尚未配置完成。';
        rerender(state);
        return;
    }
    state.switchingProfile = true;
    syncTaggingSelectionView(state.root, state);
    try {
        const payload = await activateProviderProfile(api, profile.id);
        state.providerProfiles = payload.profiles || state.providerProfiles;
        state.activeProfileId = payload.active_profile_id || profile.id;
        state.settings = publicSettings(await loadTaggingSettings(api));
        state.notice = `已切换到 ${profile.name}。`;
        state.error = '';
    } catch (error) {
        state.error = error.message || '切换接入预设失败';
    } finally {
        state.switchingProfile = false;
    }
    rerender(state);
    saveTaggingWorkspaceState(state);
}

function updatePromptDraft(state, key, value) {
    if (!['systemPrompt', 'userPrompt'].includes(key)) return;
    state[key] = String(value || '').slice(0, 10_000);
    saveTaggingWorkspaceState(state);
    syncTaggingSelectionView(state.root, state);
}

async function setSourceExpanded(state, open) {
    const expanded = Boolean(open);
    if (state.sourceExpanded === expanded) {
        if (expanded) await ensureImagesLoaded(state);
        return;
    }
    const grid = state.root?.querySelector?.('[data-tagging-image-grid]');
    if (!expanded && grid) state.gridScrollTop = grid.scrollTop;
    state.sourceExpanded = expanded;
    // Native details already changed visibility; keep the image grid mounted.
    saveTaggingWorkspaceState(state);
    if (expanded) await ensureImagesLoaded(state, { preserveDom: true });
}

async function ensureImagesLoaded(state, options = {}) {
    if (state.imagesLoaded || state.loadingImages || !state.datasetFile) return;
    await refreshImages(state, options);
}

async function saveSettings(state, data) {
    try {
        state.settings = publicSettings(await saveTaggingSettings(api, data));
        state.notice = '外部 API 设置已保存。';
        state.error = '';
    } catch (error) {
        state.error = error.message || '保存外部 API 设置失败';
    }
    rerender(state);
}

async function testProvider(state, mode) {
    if (state.testingProvider) return;
    state.testingProvider = true;
    const feedback = state.root?.querySelector('[data-tagging-settings-feedback]');
    const dialog = state.root?.querySelector('[data-tagging-settings-dialog]');
    dialog?.setAttribute('aria-busy', 'true');
    state.root?.querySelectorAll('[data-tagging-test]').forEach((button) => { button.disabled = true; });
    if (feedback) feedback.textContent = mode === 'actual' ? '正在实际调用…' : '正在测试连通…';
    try {
        const result = await testTaggingProvider(api, mode);
        if (feedback) feedback.textContent = mode === 'ping' ? `连通成功（${Number(result.elapsed_ms || 0)} ms）` : `调用成功：${result.response || '已返回'}`;
    } catch (error) {
        if (feedback) feedback.textContent = error.message || '外部 API 测试失败';
    } finally {
        state.testingProvider = false;
        dialog?.removeAttribute('aria-busy');
        state.root?.querySelectorAll('[data-tagging-test]').forEach((button) => { button.disabled = false; });
    }
}

function clearCurrentJob(state) {
    clearPoll(state);
    state.jobRequestId += 1;
    state.job = null;
    state.jobId = '';
    state.jobSignature = '';
    state.notice = '';
}

function resetImages(state) {
    state.images = [];
    state.imagesLoaded = false;
    state.directory = '';
    state.total = 0;
    state.nextOffset = 0;
    state.hasMore = false;
}

function resetImagesWithError(state, message) {
    resetImages(state);
    state.error = message;
}

function rerender(state) {
    if (!state.active || !state.root) return;
    state.jobSignature = jobSignature(state.job);
    renderTaggingView(state.root, state);
}

function syncSourceOrRerender(state, preserveDom) {
    if (!preserveDom || !state.active || !state.root || !syncTaggingSource(state.root, state, { jobBusy: isJobBusy(state) })) {
        rerender(state);
        return false;
    }
    syncTaggingSelectionView(state.root, state);
    state.reconnectImageObserver?.();
    return true;
}

function renderTaggingViewHtml(state) {
    const holder = { innerHTML: '', querySelector: () => null };
    renderTaggingView(holder, { ...state, active: false });
    return holder.innerHTML;
}

function latestMatchingJob(state) {
    return state.jobs.find((job) => job.dataset_file === state.datasetFile
        && Number(job.dataset_index || 0) === state.datasetIndex
        && (job.source || 'source') === state.source
        && (!job.profile_id || job.profile_id === state.activeProfileId));
}

function cachedDatasetMatches(saved, state) {
    return Array.isArray(saved.images) && saved.images.length > 0
        && saved.datasetFile === state.datasetFile
        && Number(saved.datasetIndex || 0) === state.datasetIndex
        && (saved.source || 'source') === state.source;
}

function chooseDatasetFile(preferred, linkedFile, presets) {
    const candidates = [preferred, linkedFile].filter(Boolean);
    const known = new Set(presets.map((item) => item.path || item.file).filter(Boolean));
    return candidates.find((file) => known.has(file)) || candidates[0] || known.values().next().value || '';
}

function uniqueImages(values) {
    const images = Array.isArray(values) ? values : [];
    const seen = new Set();
    return images.filter((image) => {
        const file = String(image?.file || '');
        if (!file || seen.has(file)) return false;
        seen.add(file);
        return true;
    });
}

function valueFrom(result, fallback) {
    return result?.status === 'fulfilled' && result.value ? result.value : fallback;
}

function publicSettings(payload) {
    return payload?.settings && typeof payload.settings === 'object' ? payload.settings : payload || {};
}

function isCurrent(state, sequence) {
    return state.active && sequence === state.requestSequence;
}

function isJobBusy(state) {
    return state.submitting || ['queued', 'running'].includes(state.job?.state);
}

function jobSignature(job) {
    if (!job) return '';
    return [job.id || '', job.state || '', job.total || 0, job.completed || 0, job.failed || 0, job.canceled || 0, job.error || ''].join('|');
}

function clampIndex(value, length) {
    if (!length) return 0;
    const number = Number(value);
    return Math.max(0, Math.min(length - 1, Number.isInteger(number) ? number : 0));
}
