from __future__ import annotations

import re
from collections import defaultdict

from tests.frontend_test_support import STATIC_DIR


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_router_shows_accessible_loading_and_failure_states() -> None:
    router = _read("js/dragon-ui/router.js")
    base_css = _read("css/dragon/01-dragon-base.css")

    assert "renderLoadingState(pageType)" in router
    assert 'role="status" aria-live="polite"' in router
    assert "mountElement.setAttribute('aria-busy', 'true')" in router
    assert "mountElement.removeAttribute('aria-busy')" in router
    assert 'role="alert"' in router
    assert "正在读取最新内容" in router
    assert "wrapper.classList.remove('dragon-page-enter')" in router
    assert ".dragon-route-loading" in base_css
    assert ".dragon-main {" in base_css
    assert "box-sizing: border-box;" in base_css


def test_dataset_initial_load_skips_gpu_and_defers_merged_config() -> None:
    controls = _read("js/dragon-ui/pages/training-controls.js")
    dataset = _read("js/dragon-ui/pages/dataset-editor.js")

    assert "includeGpus = true" in controls
    assert "includeGpus ? api('/api/training/gpus')" in controls
    assert "preserveGpuSelection" in controls
    assert "loadTrainingContext({ includeGpus: false })" in dataset

    initial_load = dataset[dataset.index("export async function loadDatasetEditor"):dataset.index("async function loadLinkedDataset")]
    assert "mergedConfigUrl" not in initial_load
    assert "trainingConfig: null" in initial_load
    assert "await ensureTrainingConfig(state)" in dataset
    assert "api(mergedConfigUrl(state.context))" in dataset


def test_dashboard_uses_one_semantic_progress_visual() -> None:
    dashboard = _read("js/dragon-ui/pages/dashboard.js")

    assert dashboard.count('role="progressbar"') == 1
    assert 'class="dragon-dashboard-progress-visual"' in dashboard
    assert 'class="dragon-dashboard-progress-track"' not in dashboard
    assert "ring.setAttribute('aria-valuenow'" in dashboard
    assert "function isActiveState" in dashboard
    for state in ("compiling", "caching", "saving", "preprocessing", "starting", "stopping"):
        assert f"'{state}'" in dashboard


def test_ui_bootstrap_falls_back_without_preloading_classic_runtime() -> None:
    html = _read("index.html")
    bootstrap = _read("js/ui-bootstrap.js")
    classic = _read("app.js")

    assert 'type="module" src="/static/js/ui-bootstrap.js?v=' in html
    assert 'type="module" src="/static/app.js?v=' not in html
    assert "const CLASSIC_ENTRY = '/static/app.js?v=" in bootstrap
    assert "await bootDragonUI({ dragonLoader, stylesheetLoader })" in bootstrap
    assert "await bootClassicUI({ classicLoader, modeLoader, stylesheetLoader })" in bootstrap
    assert "falling back to classic UI" in bootstrap
    assert "export function startClassicUI()" in classic
    assert "if (!document.body.hasAttribute('data-dragon-ui'))" not in classic


def test_dragon_stylesheet_is_only_inserted_for_dragon_mode() -> None:
    html = _read("index.html")
    bootstrap = _read("js/ui-bootstrap.js")
    dragon_styles = _read("css/dragon-style.css")
    route_styles = _read("js/dragon-ui/route-styles.js")

    assert 'id="app-ui-stylesheet" rel="stylesheet"' in html
    assert "const mode = explicitMode || (storedMode === 'classic' ? 'classic' : 'dragon')" in html
    assert "stylesheet.href = mode === 'dragon'" in html
    assert "? '/static/css/dragon-style.css?v=" in html
    assert ": '/static/style.css?v=" in html
    assert '<link rel="stylesheet" href="/static/css/dragon-style.css' not in html
    assert '<link rel="stylesheet" href="/static/style.css' not in html
    assert "await stylesheetLoader('classic')" in bootstrap
    assert "await stylesheetLoader('dragon')" in bootstrap
    assert "00-dragon-tokens.css" in dragon_styles
    assert "01-dragon-base.css" in dragon_styles
    assert "02-dragon-nav.css" in dragon_styles
    assert "02a-dragon-controls.css" in dragon_styles
    assert "06a-dragon-shared-dialogs.css" not in dragon_styles
    assert "06a-dragon-shared-dialogs.css" in route_styles


def test_dragon_route_styles_replace_only_after_all_candidates_load() -> None:
    route_styles = _read("js/dragon-ui/route-styles.js")

    for route_key in ("dashboard", "config", "dataset", "live", "history-list", "history-detail", "pages"):
        assert f"{route_key}: [" in route_styles or f"'{route_key}': [" in route_styles
    assert "Promise.all(files.map" in route_styles
    assert route_styles.index("activeLinks.forEach((link) => link.remove())") > route_styles.index(".then(() =>")
    assert "nextLinks.forEach((link) => link.remove())" in route_styles
    assert "document.querySelectorAll('link[data-dragon-route-style]')" in route_styles
    assert "link.dataset.dragonRouteStyle = 'true'" in route_styles


def test_dragon_pages_are_loaded_on_demand_by_route() -> None:
    entry = _read("js/dragon-ui/index.js")
    loaders = _read("js/dragon-ui/page-loaders.js")

    assert "import { loadDashboard } from './pages/" not in entry
    assert "createDragonPageLoaders()" in entry
    for module in (
        "config-page",
        "dataset-editor",
        "live-training",
        "history",
        "queue",
        "model-config",
    ):
        assert f"import('./pages/{module}.js?v=" in loaders


def test_navigation_uses_small_category_catalog_instead_of_full_form_layout() -> None:
    category_map = _read("js/dragon-ui/category-map.js")
    category_defs = _read("js/config/catalog/form-category-defs.js")
    form_layout = _read("js/config/catalog/form-layout.js")

    assert "form-category-defs.js" in category_map
    assert "form-layout.js" not in category_map
    assert "export const FORM_CATEGORY_DEFS" in category_defs
    assert len(category_defs.encode("utf-8")) < 3_000
    assert "import { FORM_CATEGORY_DEFS } from './form-category-defs.js" in form_layout
    assert "export { FORM_CATEGORY_DEFS };" in form_layout


def test_live_pages_pause_fallback_polling_while_hidden() -> None:
    poller = _read("js/dragon-ui/visibility-poller.js")
    dashboard = _read("js/dragon-ui/pages/dashboard.js")
    live = _read("js/dragon-ui/pages/live-training.js")

    assert "if (stopped || document.hidden) return" in poller
    assert "if (running) return running" in poller
    assert "document.addEventListener('visibilitychange', handleVisibility)" in poller
    assert "document.removeEventListener('visibilitychange', handleVisibility)" in poller
    assert "createVisibilityPoller" in dashboard
    assert "createVisibilityPoller" in live


def test_live_updates_use_mount_time_dom_bindings() -> None:
    live = _read("js/dragon-ui/pages/live-training.js")
    dom = _read("js/dragon-ui/pages/live-training-dom.js")
    render_body = live[live.index("function renderLiveState"):live.index("function createLiveRenderScheduler")]

    assert "createLiveDomBindings(root)" in live
    assert "querySelector(" not in render_body
    assert "querySelectorAll(" not in render_body
    assert "if (node && node.textContent !== next)" in dom
    assert "if (node.style.width !== next)" in dom
    assert "if (node.dataset[key] !== next)" in dom
    assert "if (node && node[key] !== value)" in dom
    assert "if (node.getAttribute(key) !== next)" in dom


def test_live_log_search_batches_render_and_releases_listeners() -> None:
    live = _read("js/dragon-ui/pages/live-training.js")
    tools = _read("js/dragon-ui/pages/live-training-log-tools.js")

    assert "createLiveLogBindings(root)" in live
    assert "unbindLogTools?.()" in live
    assert "window.setTimeout(() =>" in tools
    assert "}, 100);" in tools
    assert "cleanups.push(() => node.removeEventListener" in tools
    assert "if (searchTimer) window.clearTimeout(searchTimer)" in tools


def test_history_filtering_reuses_one_derived_task_list() -> None:
    history = _read("js/dragon-ui/pages/history.js")
    collections = _read("js/dragon-ui/pages/history-collections.js")
    update_body = history[history.index("function updateHistoryResults"):history.index("async function refreshHistory")]

    assert "renderHistoryResults" not in history
    assert update_body.count("filterHistoryTasks(") == 1
    assert "renderHistoryCollectionWorkbench(state.tasks, state.filters, state.workspace, filtered)" in update_body
    assert "renderHistorySummary(state.tasks, state.filters, filtered.length)" in update_body
    assert "stats?.addEventListener('click', handleStatClick)" in history
    assert "button.onclick" not in history
    assert "Array.isArray(filteredTasks) ? filteredTasks" in collections


def test_history_search_reuses_a_task_text_index() -> None:
    history = _read("js/dragon-ui/pages/history.js")
    model = _read("js/dragon-ui/pages/history-model.js")

    assert "createHistorySearchIndex(state.tasks)" in history
    assert "filterHistoryTasks(state.tasks, state.filters, state.searchIndex)" in history
    assert "resolveArchiveScopeForMatches(state, key, filtered)" in history
    assert "new WeakMap()" in model
    assert "searchIndex?.get(task)" in model


def test_history_chart_hover_updates_are_frame_batched() -> None:
    loss_chart = _read("js/dragon-ui/pages/history-chart.js")
    system_chart = _read("js/dragon-ui/pages/history-system-charts.js")
    scheduler = _read("js/dragon-ui/pointer-frame.js")
    frame_scheduler = _read("js/dragon-ui/frame-scheduler.js")

    assert "bindLatestPointerMove(hitarea, onMove)" in loss_chart
    assert "bindLatestPointerMove(hitarea, onMove)" in system_chart
    assert "document.addEventListener('pointermove'" not in loss_chart
    assert "if (index === lastIndex) return" in loss_chart
    assert "row === elements.lastRow" in system_chart
    assert "const points = new Map(SYSTEM_SERIES.map" in system_chart
    assert "elements.points.get(spec.id)" in system_chart
    assert "requestAnimationFrame(flush)" in frame_scheduler
    assert "scheduler.schedule({ clientX:" in scheduler


def test_config_search_batches_dom_filter_updates() -> None:
    config = _read("js/dragon-ui/pages/config-page.js")

    assert "const fieldRecords = fields.map" in config
    assert "const sectionRecords =" in config
    assert "const groupRecords =" in config
    assert "window.setTimeout(() =>" in config
    assert "}, 100);" in config
    assert "input.removeEventListener('input', scheduleSearchUpdate)" in config
    assert "window.clearTimeout(searchTimer)" in config


def test_config_dirty_updates_only_touch_the_changed_field() -> None:
    config = _read("js/dragon-ui/pages/config-page.js")
    dirty_state = _read("js/dragon-ui/pages/config-dirty-state.js")

    assert "syncDirty(captureDraftValue(field, state))" in config
    assert "updateConfigDirtyKey(state, changedKey" in config
    assert "if (!changedKey || state.showChangedOnly" in config
    assert "createConfigDirtyBindings(wrapper)" in config
    assert "const entries = changedKey" in dirty_state
    assert "root.querySelectorAll('[data-config-field-key]')" in dirty_state


def test_config_defers_full_help_catalog_until_dialog_use() -> None:
    config = _read("js/dragon-ui/pages/config-page.js")

    assert "field-help-summary.js" in config
    assert "import { FIELD_HELP_ZH }" not in config
    assert "fieldHelpCatalogPromise = import('../../config/catalog/field-help.js?v=" in config
    assert "bindConfigFieldHelpDialog(wrapper, loadFieldHelpCatalog)" in config


def test_config_defers_dataset_preset_library_until_picker_use() -> None:
    feature = _read("js/dragon-ui/pages/config-training-data.js")
    initial_bind = feature[feature.index("export function bindTrainingDataTools"):feature.index("function createTrainingDataState")]

    assert "from './dataset-editor-presets.js" not in feature
    assert "await import('./dataset-editor-presets.js?v=" in feature
    assert "loadPresetLibrary(root, dialog, state)" not in initial_bind
    assert "if (!state.library.presets.length) await loadPresetLibrary" in feature


def test_dataset_editor_dirty_and_summary_updates_are_incremental() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    runtime = _read("js/dragon-ui/pages/dataset-editor-runtime.js")

    assert "if (state.dirty) return" in page
    assert "state.ui = createDatasetEditorBindings(root)" in page
    assert page.count("updateDatasetRowSummaryForControl(event.target)") == 2
    assert "const SUMMARY_FIELDS = new Set" in runtime
    assert "root.querySelectorAll('[data-dataset-preview]')" in runtime
    assert "root.querySelectorAll('[data-dataset-row]').forEach(updateDatasetRowSummary)" in runtime


def test_dataset_editor_releases_preview_and_path_resources() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    paths = _read("js/dragon-ui/pages/dataset-editor-paths.js")
    preview = _read("js/dragon-ui/pages/dataset-editor-preview.js")
    controller = _read("js/dragon-ui/pages/dataset-preview-controller.js")

    assert "state.rowPathCleanup?.()" in page
    assert "state.previewController?.dispose()" in page
    assert "from './dataset-editor-preview.js" not in page
    assert "import('./dataset-editor-preview.js?v=" in controller
    assert "validationTimers.delete(input)" in paths
    assert "input.removeEventListener('input', onInput)" in paths
    assert "refreshButton.removeEventListener('click', refreshHandler)" in preview
    assert "chrome.pagination.removeEventListener('submit', jumpHandler)" in preview
    assert "observer?.disconnect()" in preview


def test_config_defers_model_picker_controller_until_dialog_use() -> None:
    config = _read("js/dragon-ui/pages/config-page.js")
    shell = _read("js/dragon-ui/pages/model-quick-picker-shell.js")

    assert "from './model-quick-picker-shell.js?v=" in config
    assert "from './model-quick-picker.js?v=" not in config
    assert "await import('./model-quick-picker.js?v=" in shell
    assert "trigger.removeEventListener('click', open)" in shell


def test_dragon_module_cache_tokens_do_not_duplicate_module_instances() -> None:
    versions: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(r"(?:from\s*|import\s*\()['\"]([^'\"]+\.js)\?v=([^'\"]+)['\"]")
    dragon_root = STATIC_DIR / "js" / "dragon-ui"

    for source_file in dragon_root.rglob("*.js"):
        for relative, token in pattern.findall(source_file.read_text(encoding="utf-8")):
            target = (source_file.parent / relative).resolve()
            versions[str(target)].add(token)

    duplicates = {target: sorted(tokens) for target, tokens in versions.items() if len(tokens) > 1}
    assert duplicates == {}


def test_history_list_does_not_load_detail_view_dependencies() -> None:
    controller = _read("js/dragon-ui/pages/history.js")
    collections = _read("js/dragon-ui/pages/history-collections.js")
    loaders = _read("js/dragon-ui/page-loaders.js")

    assert "history-list-view.js" in controller
    assert "history-view.js" not in controller
    assert "history-detail.js" not in controller
    assert "history-model.js" in collections
    assert "history-view.js" not in collections
    assert "context.taskId ? loadHistoryDetail(context) : loadHistoryList(context)" in loaders


def test_history_detail_view_does_not_duplicate_list_presentation() -> None:
    detail_view = _read("js/dragon-ui/pages/history-view.js")

    assert "history-model.js" in detail_view
    assert "export function renderHistoryDetailPage" in detail_view
    assert "export function renderHistoryDetailError" in detail_view
    for list_symbol in (
        "renderHistoryPage",
        "renderHistoryResults",
        "renderHistoryStats",
        "renderHistorySummary",
        "filterHistoryTasks",
        "historyFilterControls",
    ):
        assert list_symbol not in detail_view


def test_history_detail_defers_log_network_and_viewer_until_the_log_tab() -> None:
    detail = _read("js/dragon-ui/pages/history-detail.js")
    controller = _read("js/dragon-ui/pages/history-log-controller.js")

    assert "from './history-log-viewer.js" not in detail
    assert "activeTab === 'logs' ? loadHistoryLogPage(taskId)" in detail
    assert "logsLoaded: Boolean(logPage)" in detail
    assert "import('./history-log-viewer.js?v=" in controller
    assert "model.ensureLogs()" in controller
    assert "activateTab(tab) { if (tab === 'logs')" in controller


def test_history_detail_defers_hidden_chart_modules_and_svg_rendering() -> None:
    detail = _read("js/dragon-ui/pages/history-detail.js")
    controller = _read("js/dragon-ui/pages/history-metrics-controller.js")
    view = _read("js/dragon-ui/pages/history-view.js")

    assert "from './history-chart.js" not in detail
    assert "from './history-system-charts.js" not in detail
    assert "import('./history-chart.js?v=" in controller
    assert "import('./history-system-charts.js?v=" in controller
    assert "activateTab(tab) { if (tab === 'metrics')" in controller
    assert "data-history-system-host" in view
