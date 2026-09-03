/* Connection profile manager for external and future local taggers. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import {
    activateProviderProfile,
    createProviderProfile,
    deleteProviderProfile,
    cancelTaggingDownload,
    loadTaggingDownload,
    loadTaggingModelAssets,
    loadProviderProfiles,
    loadTrainingGpus,
    startTaggingModelDownload,
    testProviderProfile,
    updateProviderProfile,
} from './tagging-api.js?v=dragon-ui-20260902v9';
import { gpuIndexPayload, normalizeGpuIndex, renderTaggingGpuOptions } from './tagging-gpu-picker.js?v=dragon-ui-20260902v1';
import { returnToTaggingWorkspace } from './tagging-workspace-state.js?v=dragon-ui-20260831v4';

const api = createApiClient();
const PROVIDER_PROFILE_ROUTE = 'captioning-providers';

export async function loadTaggingProviderProfilesPage() {
    const [payload, assetPayload, gpuPayload] = await Promise.all([
        loadProviderProfiles(api),
        loadTaggingModelAssets(api).catch(() => ({ assets: [] })),
        loadTrainingGpus(api).catch(() => ({ gpus: [] })),
    ]);
    const profiles = Array.isArray(payload.profiles) ? payload.profiles : [];
    const assets = Array.isArray(assetPayload.assets) ? assetPayload.assets : [];
    const gpus = Array.isArray(gpuPayload.gpus) ? gpuPayload.gpus : [];
    const selected = profiles.find((profile) => profile.id === payload.active_profile_id) || profiles[0] || null;
    const activeAsset = assets.find((asset) => asset.state === 'downloading' && asset.download?.id) || null;
    const downloads = Array.isArray(assetPayload.downloads) ? assetPayload.downloads : [];
    const activeDownload = activeAsset?.download
        || downloads.find((download) => ['queued', 'downloading', 'publishing', 'cancel_requested'].includes(download?.state))
        || null;
    const state = {
        profiles,
        assets,
        gpus,
        providerTypes: Array.isArray(payload.provider_types) ? payload.provider_types : [],
        activeId: payload.active_profile_id || selected?.id || '',
        selectedId: selected?.id || '',
        draft: selected ? profileDraft(selected) : emptyDraft('openai_compatible'),
        dirty: false,
        busy: false,
        testing: false,
        allowLeave: false,
        error: '',
        notice: '',
        root: null,
        cleanup: null,
        active: true,
        operationId: 0,
        draftVersion: 0,
        downloadId: activeDownload?.id || '',
        downloadAssetId: activeAsset?.id || activeDownload?.asset_id || '',
        download: activeDownload,
        downloadTimer: null,
        downloadEpoch: 0,
    };
    return {
        html: renderPage(state),
        onMount: (root) => mountPage(root, state),
        beforeLeave: () => state.allowLeave || !state.dirty || confirmAction('接入预设有未保存修改，仍要离开吗？'),
        onUnmount: () => {
            state.active = false;
            state.operationId += 1;
            state.downloadEpoch += 1;
            clearDownloadPoll(state);
            state.cleanup?.();
        },
    };
}

function mountPage(root, state) {
    state.root = root;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    root.addEventListener('click', (event) => handleClick(state, event), options);
    root.addEventListener('input', (event) => handleInput(state, event), options);
    root.addEventListener('change', (event) => handleChange(state, event), options);
    root.addEventListener('submit', (event) => handleSubmit(state, event), options);
    state.cleanup = () => controller.abort();
    if (state.downloadId) pollDownload(state, state.downloadId);
}

function renderPage(state) {
    return `<div class="dragon-page dragon-page-wide dragon-caption-page dragon-tagging-tool-page" data-tagging-provider-page data-route="${PROVIDER_PROFILE_ROUTE}">
        <header class="dragon-tagging-tool-header">
            <div><button class="dragon-icon-button" type="button" data-provider-back aria-label="返回打标工作台" title="返回">${renderIcon('chevronDown')}</button><span><span class="dragon-eyebrow">CONNECTIONS</span><h1>接入预设</h1></span></div>
            <button class="dragon-btn dragon-btn-primary" type="button" data-provider-new>${renderIcon('filePlus', 'dragon-btn-icon')}<span>新建接入</span></button>
        </header>
        ${feedback(state)}
        <div class="dragon-tagging-library-layout dragon-tagging-provider-layout">
            <aside class="dragon-tagging-preset-library" aria-label="接入预设列表">
                <header><strong>接入</strong><span>${state.profiles.length}</span></header>
                <div data-provider-list>${state.profiles.length ? state.profiles.map((profile) => renderProfileItem(profile, state.selectedId, state.activeId)).join('') : '<div class="dragon-tagging-library-empty">暂无接入预设</div>'}</div>
            </aside>
            <form class="dragon-tagging-preset-editor dragon-tagging-provider-editor" data-provider-form>
                ${renderEditor(state)}
            </form>
        </div>
    </div>`;
}

function renderEditor(state) {
    const draft = state.draft;
    const existing = Boolean(state.selectedId);
    const local = draft.provider !== 'openai_compatible';
    const type = providerType(state, draft.provider);
    return `<header><div><span class="dragon-eyebrow">${existing ? 'EDIT' : 'NEW'}</span><h2>${existing ? '编辑接入' : '新建接入'}</h2><p>${escapeHtml(type?.label || '选择一种接入方式')}</p></div>${state.dirty ? '<span class="dragon-tagging-dirty-state">未保存</span>' : ''}</header>
        <label class="dragon-field"><span>预设名称</span><input class="dragon-input" type="text" name="name" maxlength="80" value="${escapeAttribute(draft.name)}" required></label>
        <label class="dragon-field"><span>接入方式</span><select class="dragon-select" name="provider" ${existing ? 'disabled' : ''}>${providerOptions(state, draft.provider)}</select></label>
        ${local ? renderLocalEditor(state, type) : renderExternalEditor(state)}
        <footer class="dragon-tagging-provider-footer"><button class="dragon-btn dragon-btn-danger" type="button" data-provider-delete ${existing && !state.busy ? '' : 'disabled'}>${renderIcon('trash', 'dragon-btn-icon')}<span>删除</span></button><span></span><button class="dragon-btn dragon-btn-secondary" type="button" data-provider-test ${existing && !state.busy && !state.testing ? '' : 'disabled'}>${renderIcon('activity', 'dragon-btn-icon')}<span>${state.testing ? '测试中…' : '测试接入'}</span></button><button class="dragon-btn dragon-btn-secondary" type="button" data-provider-activate ${existing && state.selectedId !== state.activeId && profileActivatable(state) && !state.busy ? '' : 'disabled'}>${renderIcon('check', 'dragon-btn-icon')}<span>设为当前</span></button><button class="dragon-btn dragon-btn-primary" type="submit" ${state.busy || !validDraft(draft) ? 'disabled' : ''}>${renderIcon('save', 'dragon-btn-icon')}<span>${state.busy ? '保存中…' : '保存接入'}</span></button></footer>`;
}

function renderExternalEditor(state) {
    const draft = state.draft;
    return `<section class="dragon-tagging-provider-section"><div class="dragon-tagging-provider-section-head"><span class="dragon-eyebrow">EXTERNAL API</span><p>API Key 仅保存在服务端的独立 secrets 文件中。</p></div>
        <label class="dragon-field dragon-tagging-field-wide"><span>兼容 API 地址</span><input class="dragon-input" type="url" name="base_url" value="${escapeAttribute(draft.base_url)}" required></label>
        <div class="dragon-tagging-settings-grid"><label class="dragon-field"><span>模型</span><input class="dragon-input" type="text" name="model" value="${escapeAttribute(draft.model)}" required></label><label class="dragon-field"><span>API Key ${draft.api_key_configured ? '<small>（留空保持）</small>' : ''}</span><input class="dragon-input" type="password" name="api_key" value="" autocomplete="new-password"></label><label class="dragon-field"><span>请求超时（秒）</span><input class="dragon-input" type="number" name="timeout_seconds" min="5" max="900" step="1" value="${Number(draft.timeout_seconds)}"></label><label class="dragon-field"><span>失败重试次数</span><input class="dragon-input" type="number" name="retry_count" min="0" max="6" step="1" value="${Number(draft.retry_count)}"></label><label class="dragon-field"><span>重试间隔（秒）</span><input class="dragon-input" type="number" name="retry_interval_seconds" min="0" max="60" step="0.1" value="${Number(draft.retry_interval_seconds)}"></label><label class="dragon-field"><span>并发上限</span><input class="dragon-input" type="number" name="concurrency" min="1" max="8" step="1" value="${Number(draft.concurrency)}"></label></div>
        <div class="dragon-tagging-provider-checks"><label class="dragon-tagging-check"><input type="checkbox" name="allow_private_network" ${draft.allow_private_network ? 'checked' : ''}><span>允许私有网络 API</span></label><label class="dragon-tagging-check"><input type="checkbox" name="clear_api_key"><span>清除已保存 Key</span></label></div></section>`;
}

function renderLocalEditor(state, type) {
    const draft = state.draft;
    const asset = assetForDraft(state);
    const assetLabel = draft.asset_id || (draft.provider === 'wd14' ? 'wd14-eva02-large-v3' : 'cltagger-v1-02');
    const rawAssetState = asset?.state || (draft.asset_id ? 'unknown' : 'missing');
    const assetState = draft.status === 'needs_runtime' && rawAssetState === 'installed'
        ? 'needs_runtime'
        : rawAssetState;
    const download = asset?.download || (state.downloadId && state.downloadAssetId === draft.asset_id ? state.download : null);
    const busyDownload = assetState === 'downloading' || Boolean(download && !isTerminalDownload(download));
    const authRequired = Boolean(asset?.requires_auth);
    const authConfigured = asset?.auth_configured !== false;
    const canDownload = Boolean(asset && ['missing', 'partial', 'corrupt'].includes(assetState) && !busyDownload && !state.busy && (!authRequired || authConfigured));
    const stateText = assetStateLabel(assetState);
    const authNotice = authRequired
        ? (authConfigured
            ? '<small>此资产来自 gated Hugging Face 仓库，下载请求会使用本机登录凭据。</small>'
            : `<small>${escapeHtml(asset.auth_hint || '请先在 Hugging Face 登录并接受模型条款。')}</small>`)
        : '';
    return `<section class="dragon-tagging-provider-section dragon-tagging-local-section"><div class="dragon-tagging-provider-section-head"><span class="dragon-eyebrow">LOCAL MODEL</span><p>本地权重不随仓库发布，仅在用户明确操作后下载到受控模型目录。</p></div>
        <div class="dragon-tagging-provider-status" data-state="${escapeAttribute(assetState)}"><i aria-hidden="true"></i><strong>${escapeHtml(stateText.title)}</strong><span>${escapeHtml(stateText.detail)}${authNotice}</span></div>
        <label class="dragon-field"><span>模型资产</span><select class="dragon-select" name="asset_id" required>${assetOptions(state, draft.provider, assetLabel)}</select></label>
        ${asset ? `<div class="dragon-tagging-asset-meta"><span>${escapeHtml(asset.label || asset.id)}</span><span>${formatBytes(asset.total_size)}</span><span>${escapeHtml(asset.repo || '')}</span></div>` : '<div class="dragon-tagging-asset-meta" data-state="unknown"><span>当前资产不在固定 manifest 中</span></div>'}
        ${busyDownload ? renderDownloadProgress(download) : ''}
        <div class="dragon-tagging-settings-grid"><label class="dragon-field"><span>执行设备</span><select class="dragon-select" name="device"><option value="auto" ${draft.device === 'auto' ? 'selected' : ''}>自动</option><option value="cpu" ${draft.device === 'cpu' ? 'selected' : ''}>CPU</option><option value="cuda" ${draft.device === 'cuda' ? 'selected' : ''}>CUDA</option></select></label>${draft.device === 'cuda' ? `<label class="dragon-field"><span>CUDA 设备</span><select class="dragon-select" name="gpu_index">${renderTaggingGpuOptions(state.gpus, draft.gpu_index)}</select></label>` : ''}<label class="dragon-field"><span>Batch size</span><input class="dragon-input" type="number" name="batch_size" min="1" max="64" step="1" value="${Number(draft.batch_size)}"></label><label class="dragon-field"><span>通用阈值</span><input class="dragon-input" type="number" name="general_threshold" min="0" max="1" step="0.01" value="${Number(draft.general_threshold)}"></label><label class="dragon-field"><span>角色阈值</span><input class="dragon-input" type="number" name="character_threshold" min="0" max="1" step="0.01" value="${Number(draft.character_threshold)}"></label></div>
        ${draft.provider === 'cltagger' ? `<div class="dragon-tagging-provider-checks"><span class="dragon-field-label">附加标签类别</span>${categoryGate('add_copyright_tag', 'Copyright（版权）', draft.add_copyright_tag)}${categoryGate('add_artist_tag', 'Artist（画师）', draft.add_artist_tag)}${categoryGate('add_meta_tag', 'Meta（元信息）', draft.add_meta_tag)}${categoryGate('add_model_tag', 'Model（模型）', draft.add_model_tag)}${categoryGate('add_rating_tag', 'Rating（评级）', draft.add_rating_tag)}${categoryGate('add_quality_tag', 'Quality（质量）', draft.add_quality_tag)}</div>` : ''}
        <label class="dragon-field"><span>Blacklist（每行一个标签）</span><textarea class="dragon-textarea" name="blacklist" rows="4" placeholder="可选">${escapeHtml((draft.blacklist || []).join('\n'))}</textarea></label>
        <div class="dragon-tagging-local-actions"><button class="dragon-btn dragon-btn-secondary" type="button" data-provider-download ${canDownload ? '' : 'disabled'}>${renderIcon('download', 'dragon-btn-icon')}<span>下载模型</span></button><button class="dragon-btn dragon-btn-secondary" type="button" data-provider-download-cancel ${busyDownload ? '' : 'disabled'}>${renderIcon('x', 'dragon-btn-icon')}<span>取消下载</span></button><span>仅在点击下载后获取，文件会安装到受控模型目录。</span></div></section>`;
}

function assetForDraft(state) {
    return state.assets.find((asset) => asset.id === state.draft.asset_id && asset.provider === state.draft.provider) || null;
}

function assetOptions(state, provider, selected) {
    const assets = state.assets.filter((asset) => asset.provider === provider);
    if (!assets.some((asset) => asset.id === selected) && selected) {
        assets.unshift({ id: selected, label: `${selected}（未收录）`, state: 'unknown' });
    }
    if (!assets.length) return `<option value="${escapeAttribute(selected)}" selected>${escapeHtml(selected || '暂无可用资产')}</option>`;
    return assets.map((asset) => {
        const authLabel = asset.requires_auth ? ' · 需 HF 授权' : '';
        return `<option value="${escapeAttribute(asset.id)}" ${asset.id === selected ? 'selected' : ''}>${escapeHtml(asset.label || asset.id)} · ${escapeHtml(assetStateLabel(asset.state).title)}${authLabel}</option>`;
    }).join('');
}

function assetStateLabel(state) {
    const labels = {
        installed: { title: '已安装', detail: '完整性校验通过，可用于本地打标。' },
        missing: { title: '未安装', detail: '尚未下载模型文件。' },
        partial: { title: '未完成', detail: '检测到部分文件，可重新下载缺失内容。' },
        corrupt: { title: '需重新下载', detail: '文件大小或校验值不匹配。' },
        downloading: { title: '下载中', detail: '正在获取模型文件。' },
        needs_runtime: { title: '需安装 ONNX Runtime', detail: '模型文件已安装，但当前环境缺少可用的 ONNX Runtime。' },
        needs_install: { title: '需重新下载', detail: '模型文件尚未完整安装。' },
        unknown_asset: { title: '未知资产', detail: '该资产不在当前固定 manifest 中。' },
        unknown: { title: '未知资产', detail: '该资产不在当前固定 manifest 中。' },
    };
    return labels[state] || labels.unknown;
}

function renderDownloadProgress(download) {
    const bytes = Number(download?.bytes_downloaded || 0);
    const total = Number(download?.total_bytes || 0);
    const percent = total > 0 ? Math.min(100, Math.round((bytes / total) * 100)) : 0;
    return `<div class="dragon-tagging-download-progress" data-state="${escapeAttribute(download?.state || 'downloading')}" role="status"><div><strong>${escapeHtml(download?.current_file || '准备下载')}</strong><span>${percent}% · ${formatBytes(bytes)} / ${formatBytes(total)}</span></div><progress max="100" value="${percent}"></progress></div>`;
}

function formatBytes(value) {
    const bytes = Math.max(0, Number(value) || 0);
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function renderProfileItem(profile, selectedId, activeId) {
    const state = profile.id === activeId ? '当前' : profileStatusLabel(profile);
    return `<button type="button" data-provider-select="${escapeAttribute(profile.id)}" data-active="${profile.id === selectedId}"><span><strong>${escapeHtml(profile.name)}</strong><small>${escapeHtml(profile.provider_label || profile.provider)} · ${escapeHtml(state)}</small></span>${profile.id === activeId ? '<span class="dragon-tagging-profile-current">当前</span>' : renderIcon('chevronDown')}</button>`;
}

function profileStatusLabel(profile) {
    if (profile.status === 'not_installed') return '未安装';
    if (profile.status === 'needs_install') return '需重新下载';
    if (profile.status === 'needs_runtime') return '需安装 ONNX Runtime';
    if (profile.status === 'unknown_asset') return '未知资产';
    if (profile.status === 'needs_config') return '待配置';
    return '可用';
}

function providerOptions(state, selected) {
    const types = state.providerTypes.length ? state.providerTypes : [{ id: 'openai_compatible', label: 'OpenAI-compatible API', kind: 'external', implemented: true }];
    return types.map((type) => `<option value="${escapeAttribute(type.id)}" ${type.id === selected ? 'selected' : ''}>${escapeHtml(type.label)}${type.implemented ? '' : '（预留）'}</option>`).join('');
}

function providerType(state, provider) {
    return state.providerTypes.find((type) => type.id === provider) || null;
}

function profileActivatable(state) {
    const type = providerType(state, state.draft.provider);
    if (!type?.implemented) return false;
    if (state.draft.provider !== 'openai_compatible') {
        if (['needs_runtime', 'unknown_asset'].includes(state.draft.status)) return false;
        if (!selectedGpuAvailable(state)) return false;
        const asset = assetForDraft(state);
        return Boolean((state.draft.status === 'ready' || asset?.state === 'installed') && state.draft.runtime_available !== false);
    }
    if (state.draft.available === false || state.draft.status === 'needs_config') return false;
    return state.draft.status !== 'not_installed' && state.draft.status !== 'needs_install';
}

function selectedGpuAvailable(state) {
    if (state.draft.device !== 'cuda' || !state.draft.gpu_index) return true;
    return state.gpus.some((gpu) => normalizeGpuIndex(gpu?.index) === state.draft.gpu_index);
}

function profileDraft(profile) {
    const config = profile.config || {};
    return {
        id: profile.id,
        name: profile.name || '',
        provider: profile.provider || 'openai_compatible',
        status: profile.status || '',
        asset_state: profile.asset_state || '',
        available: profile.available !== false,
        runtime_available: profile.runtime_available !== false,
        runtime_message: profile.runtime_message || '',
        api_key_configured: profile.api_key_configured === true,
        base_url: config.base_url || '',
        model: config.model || '',
        timeout_seconds: Number(config.timeout_seconds ?? 120),
        retry_count: Number(config.retry_count ?? 2),
        retry_interval_seconds: Number(config.retry_interval_seconds ?? 1.5),
        concurrency: Number(config.concurrency ?? 2),
        allow_private_network: config.allow_private_network === true,
        asset_id: config.asset_id || profile.asset_id || '',
        device: config.device || 'auto',
        gpu_index: normalizeGpuIndex(config.gpu_index),
        batch_size: Number(config.batch_size ?? 8),
        general_threshold: Number(config.general_threshold ?? 0.35),
        character_threshold: Number(config.character_threshold ?? (profile.provider === 'cltagger' ? 0.6 : 0.85)),
        blacklist: Array.isArray(config.blacklist) ? config.blacklist : [],
        add_copyright_tag: config.add_copyright_tag !== false,
        add_artist_tag: config.add_artist_tag === true,
        add_meta_tag: config.add_meta_tag === true,
        add_model_tag: config.add_model_tag === true,
        add_rating_tag: config.add_rating_tag === true,
        add_quality_tag: config.add_quality_tag === true,
    };
}

function emptyDraft(provider) {
    const draft = profileDraft({ provider, name: '', config: {} });
    if (provider !== 'openai_compatible' && !draft.asset_id) {
        draft.asset_id = provider === 'wd14' ? 'wd14-eva02-large-v3' : 'cltagger-v1-02';
    }
    return draft;
}

function validDraft(draft) {
    if (!draft?.name?.trim()) return false;
    if (draft.provider === 'openai_compatible') return Boolean(draft.base_url?.trim() && draft.model?.trim());
    return Boolean(draft.asset_id?.trim());
}

function feedback(state) {
    return `${state.error ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="error" role="alert">${escapeHtml(state.error)}</div>` : ''}${state.notice ? `<div class="dragon-config-feedback dragon-config-feedback-visible" data-tone="success" role="status">${escapeHtml(state.notice)}</div>` : ''}`;
}

function handleClick(state, event) {
    const target = event.target.closest?.('[data-provider-back], [data-provider-new], [data-provider-select], [data-provider-delete], [data-provider-test], [data-provider-activate], [data-provider-download], [data-provider-download-cancel]');
    if (!target) return;
    if (target.matches('[data-provider-back]')) return leavePage(state);
    if (target.matches('[data-provider-new]')) return startNew(state);
    if (target.matches('[data-provider-select]')) return selectProfile(state, target.dataset.providerSelect);
    if (target.matches('[data-provider-delete]')) return run(() => removeProfile(state));
    if (target.matches('[data-provider-test]')) return run(() => testProfile(state));
    if (target.matches('[data-provider-activate]')) return run(() => activateProfile(state));
    if (target.matches('[data-provider-download]')) return run(() => downloadAsset(state));
    if (target.matches('[data-provider-download-cancel]')) return run(() => cancelAssetDownload(state));
}

function handleInput(state, event) {
    const input = event.target;
    if (!input.form?.matches('[data-provider-form]') || !input.name) return;
    updateDraftField(state, input.name, input.value);
}

function handleChange(state, event) {
    const input = event.target;
    if (!input.form?.matches('[data-provider-form]') || !input.name) return;
    if (input.type === 'checkbox') updateDraftField(state, input.name, input.checked);
    else if (input.name === 'provider') {
        state.draft = { ...emptyDraft(input.value), name: state.draft.name, provider: input.value };
        state.dirty = true;
        rerender(state);
    } else if (input.name === 'asset_id') {
        updateDraftField(state, input.name, input.value);
        rerender(state, { preserveListScroll: true });
    } else if (input.name === 'device') {
        updateDraftField(state, input.name, input.value);
        rerender(state, { preserveListScroll: true });
    } else updateDraftField(state, input.name, input.value);
}

function handleSubmit(state, event) {
    if (!event.target.matches('[data-provider-form]')) return;
    event.preventDefault();
    run(() => saveProfile(state));
}

function updateDraftField(state, key, value) {
    const numeric = ['timeout_seconds', 'retry_count', 'retry_interval_seconds', 'concurrency', 'batch_size', 'general_threshold', 'character_threshold'];
    let next = value;
    if (numeric.includes(key)) next = Number(value);
    if (key === 'gpu_index') next = normalizeGpuIndex(value);
    if (key === 'blacklist') next = String(value || '').split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
    state.draft[key] = next;
    state.dirty = true;
    state.draftVersion += 1;
    syncDirtyUi(state);
}

function selectProfile(state, profileId) {
    if (state.dirty && !confirmAction('放弃当前未保存修改并切换接入预设吗？')) return;
    const profile = state.profiles.find((item) => item.id === profileId);
    if (!profile) return;
    state.operationId += 1;
    state.busy = false;
    state.testing = false;
    state.selectedId = profile.id;
    state.draft = profileDraft(profile);
    state.dirty = false;
    state.error = '';
    rerender(state, { preserveListScroll: true });
}

function startNew(state) {
    if (state.dirty && !confirmAction('放弃当前未保存修改并新建接入预设吗？')) return;
    state.operationId += 1;
    state.busy = false;
    state.testing = false;
    state.selectedId = '';
    state.draft = emptyDraft('openai_compatible');
    state.dirty = false;
    state.error = '';
    rerender(state, { preserveListScroll: true, focusName: true });
}

async function saveProfile(state) {
    if (state.busy || !validDraft(state.draft)) return;
    const operationId = ++state.operationId;
    const draftVersion = state.draftVersion;
    const selectedId = state.selectedId;
    state.busy = true;
    rerender(state, { preserveListScroll: true });
    try {
        const payload = profilePayload(state.draft);
        const response = selectedId ? await updateProviderProfile(api, selectedId, payload) : await createProviderProfile(api, payload);
        if (!state.active || state.operationId !== operationId || state.draftVersion !== draftVersion) return;
        applyResponse(state, response, response.profile?.id || selectedId);
        state.dirty = false;
        state.notice = '接入预设已保存。';
        state.error = '';
    } catch (error) {
        if (state.active && state.operationId === operationId) state.error = error.message || '保存接入预设失败';
    } finally {
        if (state.active && state.operationId === operationId) {
            state.busy = false;
            rerender(state, { preserveListScroll: true });
        }
    }
}

async function removeProfile(state) {
    if (!state.selectedId || !confirmAction('删除这个接入预设吗？')) return;
    const operationId = ++state.operationId;
    state.busy = true;
    try {
        const response = await deleteProviderProfile(api, state.selectedId);
        if (!state.active || state.operationId !== operationId) return;
        applyResponse(state, response, response.active_profile_id || response.profiles?.[0]?.id || '');
        state.notice = '接入预设已删除。';
        state.dirty = false;
    } catch (error) {
        if (state.active && state.operationId === operationId) state.error = error.message || '删除接入预设失败';
    } finally {
        if (state.active && state.operationId === operationId) {
            state.busy = false;
            rerender(state);
        }
    }
}

async function activateProfile(state) {
    if (!state.selectedId || state.selectedId === state.activeId || !profileActivatable(state)) return;
    const operationId = ++state.operationId;
    const selectedId = state.selectedId;
    state.busy = true;
    rerender(state);
    try {
        const response = await activateProviderProfile(api, selectedId);
        if (!state.active || state.operationId !== operationId || state.selectedId !== selectedId) return;
        applyResponse(state, response, selectedId);
        state.notice = '已切换当前接入。';
        state.error = '';
    } catch (error) {
        if (state.active && state.operationId === operationId) state.error = error.message || '切换当前接入失败';
    } finally {
        if (state.active && state.operationId === operationId) {
            state.busy = false;
            rerender(state);
        }
    }
}

async function testProfile(state) {
    if (!state.selectedId || state.testing) return;
    state.testing = true;
    state.error = '';
    rerender(state);
    try {
        const result = await testProviderProfile(api, state.selectedId, 'ping');
        state.notice = result.available === false ? (result.error || '本地模型尚未启用。') : `测试成功（${Number(result.elapsed_ms || 0)} ms）`;
    } catch (error) {
        state.error = error.message || '测试接入失败';
    } finally {
        state.testing = false;
        rerender(state);
    }
}

async function downloadAsset(state) {
    const asset = assetForDraft(state);
    if (!asset || state.downloadId) return;
    const epoch = ++state.downloadEpoch;
    state.error = '';
    state.notice = '';
    try {
        const response = await startTaggingModelDownload(api, asset.id);
        if (!state.active || epoch !== state.downloadEpoch) return;
        const download = response.download || null;
        state.downloadId = download?.id || '';
        state.downloadAssetId = asset.id;
        state.download = download;
        patchAssetDownload(state, asset.id, download);
        rerender(state, { preserveListScroll: true });
        if (download && !isTerminalDownload(download)) pollDownload(state, download.id);
        else await finishDownload(state, download);
    } catch (error) {
        if (state.active && epoch === state.downloadEpoch) {
            state.error = error.message || '启动模型下载失败';
            rerender(state, { preserveListScroll: true });
        }
    }
}

async function cancelAssetDownload(state) {
    if (!state.downloadId) return;
    const id = state.downloadId;
    try {
        const response = await cancelTaggingDownload(api, id);
        if (!state.active || state.downloadId !== id) return;
        state.download = response.download || state.download;
        rerender(state, { preserveListScroll: true });
        await finishDownload(state, state.download);
    } catch (error) {
        if (state.active) state.error = error.message || '取消模型下载失败';
    }
}

function pollDownload(state, downloadId) {
    clearDownloadPoll(state);
    const epoch = state.downloadEpoch;
    const tick = async () => {
        if (!state.active || epoch !== state.downloadEpoch || state.downloadId !== downloadId) return;
        try {
            const response = await loadTaggingDownload(api, downloadId);
            if (!state.active || epoch !== state.downloadEpoch || state.downloadId !== downloadId) return;
            state.download = response.download || state.download;
            patchAssetDownload(state, state.downloadAssetId, state.download);
            rerender(state, { preserveListScroll: true });
            if (isTerminalDownload(state.download)) {
                await finishDownload(state, state.download);
                return;
            }
        } catch (error) {
            if (state.active && epoch === state.downloadEpoch) {
                state.error = error.message || '读取模型下载状态失败';
                rerender(state, { preserveListScroll: true });
            }
            return;
        }
        if (state.active && epoch === state.downloadEpoch) state.downloadTimer = setTimeout(tick, 1200);
    };
    state.downloadTimer = setTimeout(tick, 0);
}

async function finishDownload(state, download) {
    clearDownloadPoll(state);
    const terminal = download?.state || '';
    if (terminal === 'completed') state.notice = '模型已下载并完成完整性校验。';
    else if (terminal === 'error') state.error = download.error || '模型下载失败';
    else if (terminal === 'canceled' || terminal === 'cancel_requested') state.notice = '模型下载已取消。';
    try {
        const payload = await loadTaggingModelAssets(api);
        if (state.active && Array.isArray(payload.assets)) state.assets = payload.assets;
    } catch {
        // The terminal job remains visible even if a status refresh races with
        // a server restart.
    }
    if (state.active) {
        state.downloadId = '';
        state.download = null;
        state.downloadAssetId = '';
        rerender(state, { preserveListScroll: true });
    }
}

function patchAssetDownload(state, assetId, download) {
    if (!assetId || !download) return;
    state.assets = state.assets.map((asset) => asset.id === assetId
        ? { ...asset, state: isTerminalDownload(download) ? asset.state : 'downloading', download }
        : asset);
}

function isTerminalDownload(download) {
    return ['completed', 'error', 'canceled'].includes(download?.state);
}

function clearDownloadPoll(state) {
    if (state.downloadTimer != null) {
        clearTimeout(state.downloadTimer);
        state.downloadTimer = null;
    }
}

function applyResponse(state, response, selectedId) {
    state.profiles = Array.isArray(response.profiles) ? response.profiles : state.profiles;
    state.activeId = response.active_profile_id || state.activeId;
    state.selectedId = selectedId || state.activeId;
    const selected = state.profiles.find((profile) => profile.id === state.selectedId) || state.profiles[0];
    state.selectedId = selected?.id || '';
    state.draft = selected ? profileDraft(selected) : emptyDraft('openai_compatible');
}

function profilePayload(draft) {
    const config = draft.provider === 'openai_compatible'
        ? {
            base_url: draft.base_url,
            model: draft.model,
            timeout_seconds: draft.timeout_seconds,
            retry_count: draft.retry_count,
            retry_interval_seconds: draft.retry_interval_seconds,
            concurrency: draft.concurrency,
            allow_private_network: draft.allow_private_network,
        }
        : {
            asset_id: draft.asset_id,
            device: draft.device,
            gpu_index: gpuIndexPayload(draft.device, draft.gpu_index),
            batch_size: draft.batch_size,
            general_threshold: draft.general_threshold,
            character_threshold: draft.character_threshold,
            blacklist: draft.blacklist,
            ...(draft.provider === 'cltagger' ? {
                add_copyright_tag: draft.add_copyright_tag,
                add_artist_tag: draft.add_artist_tag,
                add_meta_tag: draft.add_meta_tag,
                add_model_tag: draft.add_model_tag,
                add_rating_tag: draft.add_rating_tag,
                add_quality_tag: draft.add_quality_tag,
            } : {}),
        };
    const payload = { name: draft.name, provider: draft.provider, config };
    const form = draft;
    const key = String(form.api_key || '').trim();
    if (key) payload.api_key = key;
    if (form.clear_api_key) payload.clear_api_key = true;
    return payload;
}

function categoryGate(name, label, checked) {
    return `<label class="dragon-tagging-check"><input type="checkbox" name="${name}" ${checked ? 'checked' : ''}><span>${label}</span></label>`;
}

function rerender(state, options = {}) {
    if (!state.root) return;
    const list = state.root.querySelector('[data-provider-list]');
    const listScrollTop = options.preserveListScroll ? list?.scrollTop || 0 : 0;
    state.root.innerHTML = renderPage(state);
    const nextList = state.root.querySelector('[data-provider-list]');
    if (nextList) nextList.scrollTop = listScrollTop;
    if (options.focusName) state.root.querySelector('[name="name"]')?.focus();
}

function syncDirtyUi(state) {
    const save = state.root?.querySelector('[data-provider-form] > footer button[type="submit"]');
    if (save) save.disabled = state.busy || !validDraft(state.draft);
}

function leavePage(state) {
    if (state.dirty && !confirmAction('接入预设有未保存修改，仍要返回吗？')) return;
    state.allowLeave = true;
    returnToTaggingWorkspace();
}

function run(fn) {
    Promise.resolve().then(fn).catch((error) => console.error('[dragon-tagging-profiles]', error));
}

function confirmAction(message) {
    const confirm = globalThis.confirm;
    return typeof confirm === 'function' ? confirm(message) : true;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
