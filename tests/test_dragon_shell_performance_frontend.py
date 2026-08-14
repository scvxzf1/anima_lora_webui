from __future__ import annotations

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

    assert 'id="app-ui-stylesheet" rel="stylesheet"' in html
    assert "const mode = explicitMode || (storedMode === 'classic' ? 'classic' : 'dragon')" in html
    assert "stylesheet.href = mode === 'dragon'" in html
    assert "? '/static/css/dragon-style.css?v=" in html
    assert ": '/static/style.css?v=" in html
    assert '<link rel="stylesheet" href="/static/css/dragon-style.css' not in html
    assert '<link rel="stylesheet" href="/static/style.css' not in html
    assert "await stylesheetLoader('classic')" in bootstrap
    assert "await stylesheetLoader('dragon')" in bootstrap
    assert "06a-dragon-shared-dialogs.css" in dragon_styles
