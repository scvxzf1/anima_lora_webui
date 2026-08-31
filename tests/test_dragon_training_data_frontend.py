from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import STATIC_DIR


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_training_config_mounts_dataset_workflow_and_step_estimate() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    all_view = _read("js/dragon-ui/pages/config-all-view.js")
    feature = _read("js/dragon-ui/pages/config-training-data.js")

    assert "renderDatasetConfigField(value, block)" in page
    assert "renderChapterLead: (chapter) => chapter.id === 'foundation'" in page
    assert "chapter.id === 'training' ? renderStepEstimatePanel() : ''" in page
    assert "sub.id === 'common' ? renderStepEstimatePanel() : ''" in page
    assert "sub.id === 'required' ? renderDatasetPickerDialog() : ''" in page
    assert "bindTrainingDataTools(wrapper" in page
    assert "if (!state.scopeKeys.includes('dataset_config'))" in page
    assert "wrapper.dispatchEvent(new CustomEvent('dragon-config-saved'))" in page
    assert "state.trainingDataCleanup?.()" in page

    assert "renderChapterFooter = () => ''" in all_view
    assert "renderChapterLead = () => ''" in all_view
    assert "renderDatasetDialog = () => ''" in all_view
    assert "${renderChapterFooter(chapter)}" in all_view
    assert "${renderDatasetDialog()}" in all_view

    for marker in (
        'data-config-dataset-action="open"',
        'data-config-dataset-action="preview"',
        'data-config-dataset-dialog',
        'data-step-estimate',
        '/api/config/steps?',
        '/api/config/dataset-presets/images?',
        "loadDatasetPresetLibrary(api)",
        "input.dispatchEvent(new Event('input', { bubbles: true }))",
        'data-baseline-value=',
        "input.dataset.baselineValue = input.value",
    ):
        assert marker in feature


def test_step_estimate_recomputes_from_live_training_values() -> None:
    module_uri = (STATIC_DIR / "js/dragon-ui/pages/config-training-data.js").as_uri()
    script = f"""
globalThis.window = {{ fetch() {{}} }};
const mod = await import({json.dumps(module_uri)});
const result = mod.calculateStepEstimate({{
  train_image_count: 60,
  weighted_image_count: 120,
  train_batch_size: 1,
  gradient_accumulation_steps: 1,
  sample_ratio: 1,
  max_train_epochs: 4,
  max_train_steps: 0,
}}, {{
  train_batch_size: '2',
  gradient_accumulation_steps: '2',
  sample_ratio: '0.5',
  max_train_epochs: '4',
  max_train_steps: '999',
}});
console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["repeatedImages"] == 60
    assert payload["effectiveBatch"] == 4
    assert payload["stepsPerEpoch"] == 15
    assert payload["durationMode"] == "epochs"
    assert payload["totalSteps"] == 60


def test_dataset_picker_preserves_group_order_and_filters_summary_paths() -> None:
    module_uri = (STATIC_DIR / "js/dragon-ui/pages/config-training-data.js").as_uri()
    script = f"""
globalThis.window = {{ fetch() {{}} }};
const mod = await import({json.dumps(module_uri)});
const payload = {{
  presets: [
    {{path: 'configs/datasets/a.toml', label: 'Alpha', summary: {{source_dir: 'images/alpha'}}}},
    {{path: 'configs/datasets/b.toml', label: 'Beta', summary: {{source_dir: 'images/beta'}}}},
  ],
  groups: [
    {{id: 'second', label: '第二组', files: [{{path: 'configs/datasets/b.toml'}}]}},
    {{id: 'first', label: '第一组', files: [{{path: 'configs/datasets/a.toml'}}]}},
  ],
}};
const grouped = mod.groupDatasetPresets(payload, 'alpha');
console.log(JSON.stringify(grouped.map((group) => [group.id, group.files.map((item) => item.path)])));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [["first", ["configs/datasets/a.toml"]]]


def test_training_data_styles_are_responsive_and_cache_reachable() -> None:
    css = _read("css/dragon/04d-dragon-training-data.css")
    route_styles = _read("js/dragon-ui/route-styles.js")
    index = _read("index.html")
    page_loaders = _read("js/dragon-ui/page-loaders.js")

    assert "04d-dragon-training-data.css?v=dragon-ui-20260825v5" in route_styles
    assert ".dragon-config-all-detail .dragon-config-dataset-card" in css
    assert "grid-column: 1 / -1;" in css
    assert "body[data-dragon-ui] .dragon-dataset-picker-dialog" in css
    assert ".dragon-step-estimate-grid" in css
    assert "@container config-all-detail (max-width: 900px)" in css
    assert "@media (max-width: 734px)" in css
    assert "@media (max-width: 430px)" in css
    assert "dragon-style.css?v=dragon-ui-20260828v155" in index
    assert "config-page.js?v=dragon-ui-20260831v153" in page_loaders
