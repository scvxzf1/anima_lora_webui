from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_tagging_routes_are_lazy_and_keep_dedicated_tools_under_the_active_nav() -> None:
    nav = _read("js/dragon-ui/nav.js")
    loaders = _read("js/dragon-ui/page-loaders.js")
    routes = _read("js/dragon-ui/index.js")
    router = _read("js/dragon-ui/router.js")
    dataset = _read("js/dragon-ui/pages/dataset-editor.js")

    assert "{ id: 'captioning', label: '打标', hash: '#page/captioning' }" in nav
    assert "hash.startsWith('#page/captioning')" in nav
    assert "captioning: styledPage('captioning', loadCaptioning)" in loaders
    assert "'captioning-prompts': styledPage('captioning'" in loaders
    assert "'captioning-providers': styledPage('captioning'" in loaders
    assert "tagging-provider-profiles-page.js" in loaders
    assert "'captioning-results': styledPage('captioning'" in loaders
    assert "'captioning-logs': styledPage('captioning'" in loaders
    assert "tagging-prompt-presets-page.js" in loaders
    assert "tagging-results-controller.js" in loaders
    assert "tagging-logs-page.js" in loaders
    assert "return page === 'tagging' ? 'captioning' : page" in routes
    assert "'captioning-prompts': '提示词预设'" in router
    assert "'captioning-providers': '接入预设'" in router
    assert 'data-workspace-action="tagging"' in dataset
    assert "writeTaggingPrefill" in dataset


def test_main_workspace_moves_prompts_out_of_provider_settings_and_links_tools() -> None:
    page = _read("js/dragon-ui/pages/tagging.js")
    view = _read("js/dragon-ui/pages/tagging-view.js")
    api = _read("js/dragon-ui/pages/tagging-api.js")
    settings_dialog = view.split("function renderSettingsDialog", 1)[1].split("function handleClick", 1)[0]

    assert "data-tagging-system-prompt" in view
    assert "data-tagging-user-prompt" in view
    assert 'name="system_prompt"' in view
    assert 'name="user_prompt"' in view
    assert 'data-tagging-open-tool="captioning-prompts"' in view
    assert 'data-tagging-open-tool="captioning-results"' in view
    assert 'data-tagging-open-tool="captioning-logs"' in view
    assert 'type="password" name="api_key" value=""' in settings_dialog
    assert 'name="system_prompt"' not in settings_dialog
    assert "system_prompt: state.systemPrompt" in page
    assert "user_prompt: state.userPrompt" in page
    assert "loadPromptPresets" in page
    assert "loadProviderProfiles" in page
    assert "profile_id: state.activeProfileId" in page
    assert "jobStateLabel(job.state, job)" in view
    assert "正在本地打标" in view
    assert "正在调用外部 API" in view
    assert "function isLocalJob(job)" in view
    assert "state.switchingProfile" in view
    assert "switchingProfile: false" in page
    assert "state.switchingProfile = true" in page
    assert "profileSelect.disabled = Boolean(state.switchingProfile" in view
    assert "data-tagging-provider-profile" in view
    assert "captioning-providers" in view
    assert "'/prompt-presets'" in api
    assert "'/logs'" in api


def test_image_picker_uses_48_item_batches_infinite_scroll_and_500_item_selection_limit() -> None:
    api = _read("js/dragon-ui/pages/tagging-api.js")
    controller = _read("js/dragon-ui/pages/tagging.js")
    source = _read("js/dragon-ui/pages/tagging-source-view.js")
    context = _read("js/dragon-ui/pages/tagging-context.js")

    assert "TAGGING_IMAGE_PAGE_SIZE = 48" in api
    assert "offset: String(offset)" in api
    assert "limit: String(limit)" in api
    assert "async function loadMoreImages" in controller
    assert "appendTaggingImageCards" in controller
    assert "async function collectSelectionFiles" in controller
    assert "const selectedFiles = await collectSelectionFiles(state, target)" in controller
    assert "state.selectedFiles = selectedFiles" in controller
    assert "const items = [...state.selectedFiles]" in controller
    assert "TAGGING_JOB_ITEM_LIMIT = 500" in api
    assert "const MAX_SELECTED_FILES = 500" in context
    assert "IntersectionObserver" in source
    assert "rootMargin: '280px 0px'" in source
    assert "data-tagging-load-sentinel" in source
    assert "data-tagging-source-status" in source
    assert "export function renderTaggingSourceBody" in source
    assert "export function syncTaggingSource" in source
    assert "currentGrid.innerHTML = renderImageGrid(state)" in source
    assert "body.innerHTML = renderTaggingSourceBody" not in source
    assert "currentGrid.scrollTop = scrollTop" in source
    assert "data-tagging-image-load-status" in source
    assert "sourceSummary.textContent = sourceStatus(state)" in source
    assert "state.sourceExpanded ? 'open' : ''" in source
    assert '<div class="dragon-tagging-source-body">' in source
    assert "state.sourceExpanded ? `<div class=\"dragon-tagging-source-body\">" not in source
    assert "loadImages: state.sourceExpanded" in controller
    assert "if (options.loadImages === false)" in controller
    assert "if (state.sourceExpanded === expanded)" in controller
    assert "ensureImagesLoaded(state, { preserveDom: true })" in controller
    assert "const preserveDom = options.preserveDom === true" in controller
    assert "syncSourceOrRerender(state, preserveDom)" in controller
    toggle_handler = controller.split("async function setSourceExpanded", 1)[1].split(
        "async function ensureImagesLoaded", 1
    )[0]
    assert "rerender(state)" not in toggle_handler
    assert "saveTaggingWorkspaceState(state)" in toggle_handler


def test_tagging_source_toggle_release_chain_is_current() -> None:
    view = _read("js/dragon-ui/pages/tagging-view.js")
    controller = _read("js/dragon-ui/pages/tagging.js")
    facade = _read("js/dragon-ui/pages/captioning.js")
    loaders = _read("js/dragon-ui/page-loaders.js")

    assert "tagging-source-view.js?v=dragon-ui-20260901v9" in view
    assert "tagging-source-view.js?v=dragon-ui-20260901v9" in controller
    assert "tagging-view.js?v=dragon-ui-20260902v15" in controller
    assert "tagging.js?v=dragon-ui-20260902v16" in facade
    assert "captioning.js?v=dragon-ui-20260902v16" in loaders


def test_prompt_results_and_logs_pages_expose_complete_management_contracts() -> None:
    prompts = _read("js/dragon-ui/pages/tagging-prompt-presets-page.js")
    results = _read("js/dragon-ui/pages/tagging-results-controller.js")
    results_view = _read("js/dragon-ui/pages/tagging-results-view.js")
    results_css = _read("css/dragon/06c-dragon-captioning.css")
    results_editor = _read("js/dragon-ui/pages/tagging-results-editor.js")
    dataset_picker = _read("js/dragon-ui/dataset-preset-picker.js")
    logs = _read("js/dragon-ui/pages/tagging-logs-page.js")
    workspace = _read("js/dragon-ui/pages/tagging-workspace-state.js")

    for operation in ("createPromptPreset", "updatePromptPreset", "deletePromptPreset"):
        assert operation in prompts
    assert "beforeLeave" in prompts
    assert "应用并返回" in prompts
    assert "RESULT_BATCH_SIZE = 24" in results
    assert "data-result-item" in results_view
    assert "<details data-result-item" not in results_view
    assert "item.url" in results_view
    assert "data-result-image-open" in results_view
    assert "mountTaggingResultImagePreview" in results
    result_preview = _read("js/dragon-ui/pages/tagging-results-image-preview.js")
    assert "createDatasetPreviewDetailController" in result_preview
    assert "关闭预览" in result_preview
    assert "insertAdjacentHTML('beforebegin', renderResultRows" in results
    assert "function isCurrentJob" in results
    assert "updateTaggingItem" in results
    assert "commitTaggingJob" in results
    assert "rerunTaggingJob" in results
    assert "translateCaptionTags" in results
    assert "data-results-rerun" in results_view
    assert "data-result-translate" in results_view
    assert "data-results-mode=\"tags\"" in results_view
    assert "data-results-mode=\"raw\"" in results_view
    assert "const hideStatus = ['ready', 'queued'].includes(item.state)" in results_view
    assert "data-state=\"${jobStatusTone(job.state)}\"" in results_view
    assert "jobStatusTone" in results
    assert "当前任务中原位更新" in results
    assert "创建新任务" not in results
    assert "排队中" not in results_view
    assert "排队中" not in results
    assert '.dragon-tagging-review-item[data-state="queued"]' in results_css
    assert 'border: 3px solid transparent;' in results_css
    assert 'border-color: var(--dragon-accent);' in results_css
    assert '.dragon-tagging-review-item[data-state="running"]' in results_css
    assert 'border-color: var(--dragon-error);' in results_css
    assert '.dragon-tagging-results-page .dragon-status-badge[data-result-item-status]' in results_css
    assert 'display: none;' in results_css.split('.dragon-tagging-results-page .dragon-status-badge[data-result-item-status]', 1)[1].split('}', 1)[0]
    assert "checkbox.disabled = !item.id || busy" in results
    assert "selectedItemIds" in results_view
    assert "item_ids: Array.isArray(itemIds) ? itemIds : []" in _read("js/dragon-ui/pages/tagging-api.js")
    assert "重新打标选中图片" in results
    assert "renderResultPreviewEditor" in result_preview
    assert "status.hidden = hideStatus" in results
    assert "draggable=\"${busy ? 'false' : 'true'}\"" in results_editor
    assert "data-result-tag-add" in results_editor
    assert "图片同名 .txt" in results_view
    assert "mountDatasetPresetPicker" in results
    assert "data-dataset-picker-search" in dataset_picker
    assert "data-dataset-picker-preview" in dataset_picker
    assert "openSequence" in dataset_picker
    assert "state.applying" in dataset_picker
    assert "LOG_DOM_WINDOW = 400" in logs
    assert "log_retention_lines" in logs
    assert "data-logs-clear" in logs
    assert "POLL_INTERVAL_MS = 1500" in logs
    assert "openTaggingTool" in workspace
    assert "restoreTaggingWorkspacePosition" in workspace
    assert "returnToTaggingWorkspace" in workspace


def test_tagging_polling_keeps_visible_dom_stable_and_pauses_offscreen() -> None:
    controller = _read("js/dragon-ui/pages/tagging.js")
    view = _read("js/dragon-ui/pages/tagging-view.js")
    results = _read("js/dragon-ui/pages/tagging-results-controller.js")
    results_view = _read("js/dragon-ui/pages/tagging-results-view.js")
    logs = _read("js/dragon-ui/pages/tagging-logs-page.js")
    poller = _read("js/dragon-ui/visibility-poller.js")

    assert "createVisibilityPoller" in controller
    assert "syncTaggingJobView" in controller
    assert "if (changed)" in controller
    assert "createVisibilityPoller" in results
    assert "syncResultsJobView" in results
    assert "state.root.innerHTML = renderResultsPage(state, DATASET_PICKER_HTML)" in results
    assert "if (!state.active || !state.root) return" in results
    assert "isCurrentJob(state, sourceJobId, epoch, requestId)" in results
    assert "if (!isCurrent()) return" in results
    assert "data-result-save-label" in results_view
    assert "state.dirtyItemIds.has(item.id) && text !== String(item.proposed_caption || '')" in results_view
    assert "value === originalCaptionFor(state, itemId)" in results
    assert "state.dirtyItemIds.delete(itemId)" in results
    assert "syncItemSaveControl(state, itemId)" in results
    translate_item = results.split("async function translateItem", 1)[1].split(
        "async function ensureTagDictionary", 1
    )[0]
    assert "rerender(state)" not in translate_item
    assert "syncItemTranslationControl(state, itemId)" in translate_item
    assert "syncFeedback(state)" in translate_item
    assert results.count("resetTranslationCacheAfterSave(state, itemId)") >= 2
    assert "state.imagePreview?.dispose();\n    state.imagePreview = null;\n    state.root.innerHTML" in results
    assert "data-result-item-feedback" in results_view
    assert "appendLogLines" in logs
    assert "renderLogWindow(state, { keepBottom: true })" in logs
    assert "syncLogControls" in logs
    assert "refreshImages: () => refreshImages(state, { preserveDom: true })" in controller
    assert "await refreshImages(state, { preserveDom: true })" in controller
    assert "state.selectingAll = true;\n    syncTaggingSelectionView" in controller
    assert "data-tagging-directory" in _read("js/dragon-ui/pages/tagging-source-view.js")
    assert "document.hidden" in poller
    assert "if (running) return running" in poller
    assert "state.jobPoller.reschedule()" not in controller
    assert "state.jobPoller.reschedule()" not in results
    assert "state.logPoller.reschedule()" not in logs


def test_tagging_result_preview_and_save_button_interactions() -> None:
    jsdom_api = ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for tagging result interaction checks")
    script = """
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
const dom = new JSDOM('<!doctype html><html><body><main id="root"></main><button id="trigger">open</button></body></html>', {
    url: 'http://localhost/',
    pretendToBeVisual: true,
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.CSS = dom.window.CSS;
dom.window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
dom.window.HTMLDialogElement.prototype.close = function () {
    this.removeAttribute('open');
    this.dispatchEvent(new dom.window.Event('close'));
};

const previewModule = await import('./web/static/js/dragon-ui/pages/tagging-results-image-preview.js?test=tagging-preview-v1');
const viewModule = await import('./web/static/js/dragon-ui/pages/tagging-results-view.js?test=tagging-save-v1');
const root = document.getElementById('root');
root.innerHTML = previewModule.renderTaggingResultImageDialog();
const editorState = {
    itemId: 'item-1',
    text: '1girl, solo',
    mode: 'raw',
    dirty: false,
    busy: false,
    saving: false,
    translating: false,
    language: 'en',
};
const preview = previewModule.mountTaggingResultImagePreview(root, {
    getImage: () => ({
        id: 'item-1',
        url: '/api/captioning/image/test',
        name: 'sample.png',
        file: '/data/sample.png',
        caption: { ok: true, text: '1girl, solo', format_label: '当前候选标注' },
    }),
    getEditor: () => editorState,
});
if (!preview.open('item-1', document.getElementById('trigger'))) throw new Error('preview did not open');
const dialog = root.querySelector('[data-results-image-dialog]');
if (!dialog.open) throw new Error('preview dialog is not open');
if (dialog.querySelector('[data-dataset-preview-detail-image]').getAttribute('src') !== '/api/captioning/image/test') throw new Error('full image missing');
if (dialog.querySelector('[data-dataset-preview-detail-back] span').textContent !== '关闭预览') throw new Error('result close label mismatch');
if (!dialog.querySelector('[data-result-caption]')) throw new Error('preview editor missing');
if (!dialog.querySelector('[data-result-save]').disabled) throw new Error('preview save should start disabled');
if (dialog.textContent.includes('修改只保存到当前任务结果')) throw new Error('redundant preview helper text remains');
const detailView = dialog.querySelector('[data-dataset-preview-detail]');
const imageNode = dialog.querySelector('[data-dataset-preview-detail-image]');
const editorHost = dialog.querySelector('[data-result-preview-caption-editor]');
const focusedEditor = dialog.querySelector('[data-result-caption]');
focusedEditor.focus();
focusedEditor.value = 'typing must survive';
editorState.text = '单人, 独自';
editorState.language = 'zh';
editorState.dirty = true;
preview.sync();
if (dialog.querySelector('[data-result-caption]') !== focusedEditor) throw new Error('focused preview editor was replaced');
if (focusedEditor.value !== 'typing must survive') throw new Error('focused preview draft was overwritten');
dialog.querySelector('[data-result-translate]').focus();
preview.sync();
if (root.querySelector('[data-results-image-dialog]') !== dialog) throw new Error('preview dialog was replaced');
if (dialog.querySelector('[data-dataset-preview-detail]') !== detailView) throw new Error('preview detail was replaced');
if (dialog.querySelector('[data-dataset-preview-detail-image]') !== imageNode) throw new Error('preview image was replaced');
if (dialog.querySelector('[data-result-preview-caption-editor]') !== editorHost) throw new Error('preview editor host was replaced');
if (dialog.querySelector('[data-result-caption]').value !== '单人, 独自') throw new Error('preview translation did not sync');
if (dialog.querySelector('[data-result-translate-label]').textContent !== 'EN') throw new Error('preview language label did not sync');
if (dialog.querySelector('[data-result-save]').disabled) throw new Error('translated preview should be saveable');
dialog.querySelector('[data-dataset-preview-detail-back]').dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }));
if (dialog.open) throw new Error('preview did not close');
if (document.activeElement?.id !== 'trigger') throw new Error('preview close did not restore focus');
if (!preview.open('item-1', document.getElementById('trigger'))) throw new Error('preview did not reopen');
preview.dispose();
if (document.activeElement?.id !== 'trigger') throw new Error('preview dispose did not restore focus');

const item = { id: 'item-1', state: 'ready', name: 'sample.png', file: '/data/sample.png', url: '/image', proposed_caption: '1girl, solo' };
const state = {
    job: { id: 'job-1', state: 'completed', items: [item] },
    selectedItemIds: new Set(),
    dirtyItemIds: new Set(),
    drafts: new Map(),
    itemLanguages: new Map(),
    translatingItemIds: new Set(),
    savingItemIds: new Set(),
    viewMode: 'raw',
};
const saveDisabled = () => {
    root.innerHTML = viewModule.renderResultRows(state, 0, 1);
    return root.querySelector('[data-result-save]').disabled;
};
if (!saveDisabled()) throw new Error('unchanged result save button must be disabled');
state.dirtyItemIds.add(item.id);
state.drafts.set(item.id, '1girl, solo, smile');
if (saveDisabled()) throw new Error('changed result save button must be enabled');
state.drafts.set(item.id, item.proposed_caption);
if (!saveDisabled()) throw new Error('reverted result save button must be disabled');

const emptyItem = { id: 'item-empty', state: 'failed', name: 'empty.png', file: '/data/empty.png', url: '/empty', proposed_caption: '' };
state.job = { id: 'job-1', state: 'completed', items: [emptyItem] };
root.innerHTML = viewModule.renderResultRows(state, 0, 1);
if (root.querySelector('[data-result-select]').disabled) throw new Error('failed or empty result must remain selectable for rerun');
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tagging_job_state_label_distinguishes_local_and_external() -> None:
    view_uri = (STATIC / "js" / "dragon-ui" / "pages" / "tagging-view.js").as_uri()
    script = f"""
        const {{ jobStateLabel }} = await import('{view_uri}?test=provider-label-v1');
        const local = {{ settings: {{ provider: 'cltagger' }} }};
        const external = {{ settings: {{ provider: 'openai_compatible' }} }};
        const unknown = {{ settings: {{ provider: 'future_provider' }} }};
        if (jobStateLabel('running', local) !== '正在本地打标') throw new Error('local label mismatch');
        if (jobStateLabel('running', external) !== '正在调用外部 API') throw new Error('external label mismatch');
        if (jobStateLabel('running', unknown) !== '正在处理') throw new Error('unknown label mismatch');
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_provider_profile_page_reserves_local_models_and_keeps_secret_safe_contract() -> None:
    page = _read("js/dragon-ui/pages/tagging-provider-profiles-page.js")
    api = _read("js/dragon-ui/pages/tagging-api.js")
    assert "loadProviderProfiles" in api
    assert "loadTrainingGpus" in api
    assert "createProviderProfile" in api
    assert "updateProviderProfile" in api
    assert "deleteProviderProfile" in api
    assert "activateProviderProfile" in api
    assert "testProviderProfile" in api
    assert "loadTaggingModelAssets" in api
    assert "startTaggingModelDownload" in api
    assert "loadTaggingDownload" in api
    assert "cancelTaggingDownload" in api
    assert "captioning-providers" in page
    assert "wd14" in page and "cltagger" in page
    assert "data-provider-download" in page
    assert "data-provider-download-cancel" in page
    assert "仅在点击下载后获取" in page
    assert "api_key" in page
    assert "profilePayload" in page
    assert "returnToTaggingWorkspace" in page
    assert "gpu_index" in page
    assert "selectedGpuAvailable" in page
    assert "state.operationId !== operationId" in page


def test_local_tagging_gpu_picker_uses_single_training_gpu_index() -> None:
    helper_uri = (STATIC / "js" / "dragon-ui" / "pages" / "tagging-gpu-picker.js").as_uri()
    script = f"""
        const picker = await import('{helper_uri}?test=tagging-gpu-picker-v1');
        const html = picker.renderTaggingGpuOptions([
            {{ index: 0, label: 'GPU 0 · 10 GB' }},
            {{ index: 2, label: 'GPU 2 · 24 GB' }},
        ], 2);
        if (!html.includes('GPU 2 · 24 GB') || !html.includes('value=\"2\" selected')) throw new Error('selected GPU missing');
        if (picker.gpuIndexPayload('cuda', '2') !== 2) throw new Error('CUDA payload mismatch');
        if (picker.gpuIndexPayload('cpu', '2') !== null) throw new Error('CPU must clear GPU index');
        const missing = picker.renderTaggingGpuOptions([], 4);
        if (!missing.includes('当前不可用')) throw new Error('missing GPU state was lost');
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_prompt_library_includes_builtin_templates_and_local_category_labels() -> None:
    prompts = _read("js/dragon-ui/pages/tagging-prompt-presets-page.js")
    providers = _read("js/dragon-ui/pages/tagging-provider-profiles-page.js")
    assert "复制为新预设" in prompts
    assert "内置模板" in prompts
    assert "builtin-detailed" not in prompts  # IDs are owned by the server manifest
    for label in ("版权", "画师", "元信息", "模型", "评级", "质量"):
        assert label in providers


def test_tagging_prefill_is_one_shot_secret_safe_and_bounded_to_job_limit() -> None:
    context_uri = (STATIC / "js" / "dragon-ui" / "pages" / "tagging-context.js").as_uri()
    script = f"""
        const values = new Map();
        globalThis.sessionStorage = {{
            setItem(key, value) {{ values.set(key, value); }},
            getItem(key) {{ return values.get(key) || null; }},
            removeItem(key) {{ values.delete(key); }},
        }};
        const context = await import('{context_uri}?test=one-shot-v2');
        const selected = Array.from({{ length: 550 }}, (_, index) => `images/${{index}}.png`);
        if (!context.writeTaggingPrefill({{
            dataset_file: 'configs/datasets/example.toml',
            dataset_index: 2,
            source: 'training',
            image_file: 'images/sample.png',
            selected_files: selected,
            api_key: 'must-not-be-stored',
        }})) throw new Error('prefill was not written');
        const serialized = values.get(context.TAGGING_PREFILL_STORAGE_KEY) || '';
        if (serialized.includes('must-not-be-stored') || serialized.includes('api_key')) throw new Error('secret leaked');
        const first = context.consumeTaggingPrefill();
        if (first.selected_files.length !== 500) throw new Error(`selection cap mismatch: ${{first.selected_files.length}}`);
        if (first.dataset_index !== 2 || first.source !== 'training') throw new Error('prefill mismatch');
        if (Object.keys(context.consumeTaggingPrefill()).length !== 0) throw new Error('prefill was not consumed');
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tagging_workspace_state_restores_selection_and_original_return_hash() -> None:
    state_uri = (STATIC / "js" / "dragon-ui" / "pages" / "tagging-workspace-state.js").as_uri()
    script = f"""
        const values = new Map();
        globalThis.sessionStorage = {{
            setItem(key, value) {{ values.set(key, value); }},
            getItem(key) {{ return values.get(key) || null; }},
            removeItem(key) {{ values.delete(key); }},
        }};
        globalThis.location = {{ hash: '#page/captioning' }};
        globalThis.scrollY = 320;
        globalThis.scrollTo = () => {{}};
        globalThis.requestAnimationFrame = (callback) => callback();
        const store = await import('{state_uri}?test=return-v1');
        const root = {{ querySelector() {{ return {{ scrollTop: 91 }}; }} }};
        const state = {{
            datasetFile: 'configs/datasets/example.toml', datasetIndex: 1, source: 'source',
            selectedFiles: new Set(['a.png', 'b.png']), images: [{{ file: 'a.png' }}], rows: [],
            systemPrompt: 'system', userPrompt: 'user', root,
        }};
        store.openTaggingTool(state, 'captioning-results');
        if (globalThis.location.hash !== '#page/captioning-results') throw new Error('tool route mismatch');
        const restored = store.readTaggingWorkspaceState();
        if (restored.selectedFiles.length !== 2 || restored.systemPrompt !== 'system') throw new Error('workspace state mismatch');
        store.returnToTaggingWorkspace();
        if (globalThis.location.hash !== '#page/captioning') throw new Error('return route mismatch');
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tagging_layout_is_wide_16_by_9_lazy_and_responsive() -> None:
    css = _read("css/dragon/06c-dragon-captioning.css")

    assert "width: min(1920px, 80vw)" in css
    assert "box-sizing: border-box" in css
    assert "min-height: min(80dvh" in css
    assert "body[data-dragon-ui] #dragon-root .dragon-caption-page :is(" in css
    assert ".dragon-tagging-source {" in css
    assert "padding: 0" in css
    assert "grid-template-columns: repeat(auto-fill, minmax(180px, 1fr))" in css
    assert "grid-template-columns: repeat(auto-fill, minmax(min(360px, 100%), 1fr))" in css
    assert "aspect-ratio: 16 / 9" in css
    assert "object-fit: contain" in css
    assert "content-visibility: auto" in css
    assert "contain-intrinsic-size" in css
    assert "scrollbar-gutter: stable" in css
    assert ".dragon-tagging-load-sentinel[hidden]," in css
    assert ".dragon-tagging-results-sentinel[hidden]" in css
    assert "@media (max-width: 760px)" in css
    assert ".dragon-tagging-prompt-grid," in css
    assert ".dragon-tagging-review-detail," in css
    assert "grid-template-columns: 1fr" in css
    assert ".dragon-tagging-results-toolbar-row" in css
    assert ".dragon-tagging-chip-list" in css
    assert "flex-wrap: wrap" in css


def test_prompt_preset_library_header_keeps_inner_spacing() -> None:
    css = _read("css/dragon/06c-dragon-captioning.css")
    route_styles = _read("js/dragon-ui/route-styles.js")
    selector = (
        "body[data-dragon-ui] #dragon-root .dragon-caption-page "
        ".dragon-tagging-preset-library > header {"
    )
    rule = css.split(selector, 1)[1].split("}", 1)[0]

    assert "padding: var(--dragon-sp-3);" in rule
    assert "border-bottom: 1px solid var(--dragon-border-light);" in rule
    assert "06c-dragon-captioning.css?v=dragon-ui-20260902v18" in route_styles


def test_prompt_preset_library_items_keep_horizontal_inset() -> None:
    css = _read("css/dragon/06c-dragon-captioning.css")
    selector = ".dragon-tagging-preset-library > div {"
    rule = css.split(selector, 1)[1].split("}", 1)[0]

    assert "padding: var(--dragon-sp-3) var(--dragon-sp-4);" in rule


def test_tagging_javascript_modules_parse() -> None:
    files = [
        "js/dragon-ui/pages/tagging-context.js",
        "js/dragon-ui/pages/tagging-api.js",
        "js/dragon-ui/pages/tagging-workspace-state.js",
        "js/dragon-ui/pages/tagging-source-view.js",
        "js/dragon-ui/pages/tagging-view.js",
        "js/dragon-ui/pages/tagging.js",
        "js/dragon-ui/pages/tagging-prompt-presets-page.js",
        "js/dragon-ui/pages/tagging-provider-profiles-page.js",
        "js/dragon-ui/pages/tagging-results-page.js",
        "js/dragon-ui/pages/tagging-results-controller.js",
        "js/dragon-ui/pages/tagging-results-view.js",
        "js/dragon-ui/pages/tagging-results-editor.js",
        "js/dragon-ui/pages/tagging-logs-page.js",
        "js/dragon-ui/dataset-preset-picker.js",
        "js/dragon-ui/pages/captioning.js",
        "js/dragon-ui/page-loaders.js",
        "js/dragon-ui/index.js",
        "js/dragon-ui/nav.js",
    ]
    for relative in files:
        result = subprocess.run(
            ["node", "--check", str(STATIC / relative)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, f"{relative}: {result.stderr}"


def test_result_tag_editor_keeps_raw_text_and_drag_order_in_sync() -> None:
    editor_uri = (STATIC / "js" / "dragon-ui" / "pages" / "tagging-results-editor.js").as_uri()
    script = f"""
        const editor = await import('{editor_uri}?test=result-tag-editor-v1');
        const source = '1girl, blue_hair\\nsolo';
        const tags = editor.splitCaptionTags(source);
        if (JSON.stringify(tags) !== JSON.stringify(['1girl', 'blue_hair', 'solo'])) throw new Error('split mismatch');
        if (editor.moveCaptionTag(source, 2, 0) !== 'solo, 1girl, blue_hair') throw new Error('move mismatch');
        if (editor.replaceCaptionTag(source, 1, 'white hair') !== '1girl, white hair, solo') throw new Error('replace mismatch');
        if (editor.removeCaptionTag(source, 0) !== 'blue_hair, solo') throw new Error('remove mismatch');
        if (editor.appendCaptionTag(source, 'smile,') !== '1girl, blue_hair, solo, smile') throw new Error('append mismatch');
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
