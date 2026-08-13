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
