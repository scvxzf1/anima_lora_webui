/* API helpers for the Dragon external-API tagging page. */

export const TAGGING_API_PREFIX = '/api/captioning';
export const TAGGING_IMAGE_PAGE_SIZE = 48;
export const TAGGING_JOB_ITEM_LIMIT = 500;

export async function taggingApi(api, path, options = {}) {
    const payload = await api(`${TAGGING_API_PREFIX}${path}`, options);
    if (payload?.ok === false) throw new Error(payload.error || '打标接口请求失败');
    return payload;
}

export function jsonOptions(method, body) {
    return {
        method,
        body: JSON.stringify(body ?? {}),
    };
}

export async function loadPreset(api, file) {
    const params = new URLSearchParams({ file });
    const payload = await api(`/api/config/dataset-presets/read?${params.toString()}`);
    if (payload?.ok === false) throw new Error(payload.error || '读取数据集预设失败');
    return payload;
}

export async function loadImages(api, file, datasetIndex, source = 'source', { limit = TAGGING_IMAGE_PAGE_SIZE, offset = 0 } = {}) {
    const params = new URLSearchParams({
        file,
        dataset_index: String(datasetIndex),
        source,
        limit: String(limit),
        offset: String(offset),
    });
    const payload = await api(`/api/config/dataset-presets/images?${params.toString()}`);
    if (payload?.ok === false) throw new Error(payload.error || '读取数据集图片失败');
    return payload;
}

export async function loadTaggingSettings(api) {
    return taggingApi(api, '/settings');
}

export async function saveTaggingSettings(api, payload) {
    return taggingApi(api, '/settings', jsonOptions('PUT', payload));
}

export async function testTaggingProvider(api, mode = 'ping') {
    return taggingApi(api, '/test', jsonOptions('POST', { mode }));
}

export async function loadProviderProfiles(api) {
    return taggingApi(api, '/profiles');
}

export async function loadTrainingGpus(api) {
    const payload = await api('/api/training/gpus');
    if (payload?.ok === false) throw new Error(payload.error || '读取 GPU 列表失败');
    return payload;
}

export async function createProviderProfile(api, payload) {
    return taggingApi(api, '/profiles', jsonOptions('POST', payload));
}

export async function updateProviderProfile(api, profileId, payload) {
    return taggingApi(api, `/profiles/${encodeURIComponent(profileId)}`, jsonOptions('PUT', payload));
}

export async function deleteProviderProfile(api, profileId) {
    return taggingApi(api, `/profiles/${encodeURIComponent(profileId)}`, { method: 'DELETE' });
}

export async function activateProviderProfile(api, profileId) {
    return taggingApi(api, `/profiles/${encodeURIComponent(profileId)}/activate`, jsonOptions('POST'));
}

export async function loadTaggingModelAssets(api) {
    return taggingApi(api, '/model-assets');
}

export async function loadTaggingModelAsset(api, assetId) {
    return taggingApi(api, `/model-assets/${encodeURIComponent(assetId)}`);
}

export async function startTaggingModelDownload(api, assetId) {
    return taggingApi(api, `/model-assets/${encodeURIComponent(assetId)}/download`, jsonOptions('POST'));
}

export async function loadTaggingDownloads(api) {
    return taggingApi(api, '/downloads');
}

export async function loadTaggingDownload(api, downloadId) {
    return taggingApi(api, `/downloads/${encodeURIComponent(downloadId)}`);
}

export async function cancelTaggingDownload(api, downloadId) {
    return taggingApi(api, `/downloads/${encodeURIComponent(downloadId)}/cancel`, jsonOptions('POST'));
}

export async function loadTagDictionaryStatus(api) {
    return taggingApi(api, '/tag-dictionary');
}

export async function startTagDictionaryDownload(api) {
    return taggingApi(api, '/tag-dictionary/download', jsonOptions('POST'));
}

export async function loadTagDictionaryDownload(api, downloadId) {
    return taggingApi(api, `/tag-dictionary/downloads/${encodeURIComponent(downloadId)}`);
}

export async function cancelTagDictionaryDownload(api, downloadId) {
    return taggingApi(api, `/tag-dictionary/downloads/${encodeURIComponent(downloadId)}/cancel`, jsonOptions('POST'));
}

export async function translateCaptionTags(api, tags, targetLanguage) {
    return taggingApi(api, '/translate-tags', jsonOptions('POST', { tags, target_language: targetLanguage }));
}

export async function testProviderProfile(api, profileId, mode = 'ping') {
    return taggingApi(api, '/test', jsonOptions('POST', { mode, profile_id: profileId }));
}

export async function loadPromptPresets(api) {
    return taggingApi(api, '/prompt-presets');
}

export async function createPromptPreset(api, payload) {
    return taggingApi(api, '/prompt-presets', jsonOptions('POST', payload));
}

export async function updatePromptPreset(api, presetId, payload) {
    return taggingApi(api, `/prompt-presets/${encodeURIComponent(presetId)}`, jsonOptions('PUT', payload));
}

export async function deletePromptPreset(api, presetId) {
    return taggingApi(api, `/prompt-presets/${encodeURIComponent(presetId)}`, { method: 'DELETE' });
}

export async function loadTaggingLogs(api, { after = 0, limit, jobId = '' } = {}) {
    const params = new URLSearchParams({ after: String(after || 0) });
    if (limit != null) params.set('limit', String(limit));
    if (jobId) params.set('job_id', jobId);
    return taggingApi(api, `/logs?${params.toString()}`);
}

export async function clearTaggingLogs(api) {
    return taggingApi(api, '/logs', { method: 'DELETE' });
}

export async function loadTaggingJobs(api) {
    return taggingApi(api, '/jobs');
}

export async function createTaggingJob(api, payload) {
    return taggingApi(api, '/jobs', jsonOptions('POST', payload));
}

export async function loadTaggingJob(api, jobId) {
    return taggingApi(api, `/jobs/${encodeURIComponent(jobId)}`);
}

export async function rerunTaggingJob(api, jobId, profileId, itemIds = []) {
    return taggingApi(
        api,
        `/jobs/${encodeURIComponent(jobId)}/rerun`,
        jsonOptions('POST', { profile_id: profileId, item_ids: Array.isArray(itemIds) ? itemIds : [] }),
    );
}

export async function cancelTaggingJob(api, jobId) {
    return taggingApi(api, `/jobs/${encodeURIComponent(jobId)}/cancel`, jsonOptions('POST'));
}

export async function updateTaggingItem(api, jobId, itemId, proposedCaption) {
    return taggingApi(
        api,
        `/jobs/${encodeURIComponent(jobId)}/items/${encodeURIComponent(itemId)}`,
        jsonOptions('PATCH', { proposed_caption: proposedCaption }),
    );
}

export async function commitTaggingJob(api, jobId, { all = false, itemIds = [] } = {}) {
    return taggingApi(
        api,
        `/jobs/${encodeURIComponent(jobId)}/commit`,
        jsonOptions('POST', { all, item_ids: itemIds }),
    );
}
