from __future__ import annotations

import subprocess
from pathlib import Path


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
    assert "'captioning-results': styledPage('captioning'" in loaders
    assert "'captioning-logs': styledPage('captioning'" in loaders
    assert "tagging-prompt-presets-page.js" in loaders
    assert "tagging-results-page.js" in loaders
    assert "tagging-logs-page.js" in loaders
    assert "return page === 'tagging' ? 'captioning' : page" in routes
    assert "'captioning-prompts': '提示词预设'" in router
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
    assert "body.innerHTML = renderTaggingSourceBody" in source
    assert "nextGrid.scrollTop = scrollTop" in source
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

    assert "tagging-source-view.js?v=dragon-ui-20260831v6" in view
    assert "tagging-source-view.js?v=dragon-ui-20260831v6" in controller
    assert "tagging-view.js?v=dragon-ui-20260831v7" in controller
    assert "tagging.js?v=dragon-ui-20260831v8" in facade
    assert "captioning.js?v=dragon-ui-20260831v8" in loaders


def test_prompt_results_and_logs_pages_expose_complete_management_contracts() -> None:
    prompts = _read("js/dragon-ui/pages/tagging-prompt-presets-page.js")
    results = _read("js/dragon-ui/pages/tagging-results-page.js")
    logs = _read("js/dragon-ui/pages/tagging-logs-page.js")
    workspace = _read("js/dragon-ui/pages/tagging-workspace-state.js")

    for operation in ("createPromptPreset", "updatePromptPreset", "deletePromptPreset"):
        assert operation in prompts
    assert "beforeLeave" in prompts
    assert "应用并返回" in prompts
    assert "RESULT_BATCH_SIZE = 24" in results
    assert "data-result-item" in results
    assert "state.expandedItemIds.has(item.id)" in results
    assert "state.thumbnail_url || item.url" not in results
    assert "item.thumbnail_url || item.url" in results
    assert "insertAdjacentHTML('beforebegin', renderResultRows" in results
    assert "expanded ? renderResultDetail(state, item, busy) : ''" in results
    assert "function isCurrentJob" in results
    assert "updateTaggingItem" in results
    assert "commitTaggingJob" in results
    assert "图片同名 .txt" in results
    assert "LOG_DOM_WINDOW = 400" in logs
    assert "log_retention_lines" in logs
    assert "data-logs-clear" in logs
    assert "POLL_INTERVAL_MS = 1500" in logs
    assert "openTaggingTool" in workspace
    assert "restoreTaggingWorkspacePosition" in workspace
    assert "returnToTaggingWorkspace" in workspace


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
    assert "06c-dragon-captioning.css?v=dragon-ui-20260831v8" in route_styles


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
        "js/dragon-ui/pages/tagging-results-page.js",
        "js/dragon-ui/pages/tagging-logs-page.js",
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
