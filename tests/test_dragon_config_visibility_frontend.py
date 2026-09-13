import json
import shutil
import subprocess
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / "web" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def _node_json(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=STATIC.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return json.loads(result.stdout)


def test_config_visibility_levels_classify_known_and_unknown_fields() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for config visibility checks")
    module_uri = (STATIC / "js/dragon-ui/pages/config-field-tiers.js").resolve().as_uri()
    script = f"""
const mod = await import({json.dumps(module_uri + '?visibility-level-test')});
const level = (key) => mod.configFieldVisibilityLevel(key);
console.log(JSON.stringify({{
  options: mod.CONFIG_VISIBILITY_OPTIONS.map((item) => item.id),
  newcomer: level('learning_rate'),
  promotedNewcomer: level('sample_prompts'),
  beginnerAdvanced: level('torch_compile'),
  defaultPipeline: level('pipeline_parallel'),
  defaultProbe: level('memory_probe_jsonl'),
  defaultCache: level('reuse_vae_latents'),
  beginnerExperimental: level('weighting_scheme'),
  defaultIsNotAdvanced: mod.configFieldIsAdvanced('torch_compile'),
  newcomerIsNotAdvanced: mod.configFieldIsAdvanced('sample_prompts'),
  advanced: level('activation_memory_budget'),
  unknown: level('future_config_key'),
  newcomerVisible: mod.configFieldVisibleAtLevel(level('learning_rate'), 'newcomer'),
  beginnerHidden: mod.configFieldVisibleAtLevel(level('torch_compile'), 'newcomer'),
  beginnerVisible: mod.configFieldVisibleAtLevel(level('weighting_scheme'), 'beginner'),
  advancedVisible: mod.configFieldVisibleAtLevel(level('activation_memory_budget'), 'all'),
}}));
"""
    payload = _node_json(script)

    assert payload["options"] == ["newcomer", "beginner", "all"]
    assert payload["newcomer"] == "newcomer"
    assert payload["promotedNewcomer"] == "newcomer"
    assert payload["beginnerAdvanced"] == "beginner"
    assert payload["defaultPipeline"] == "beginner"
    assert payload["defaultProbe"] == "beginner"
    assert payload["defaultCache"] == "beginner"
    assert payload["beginnerExperimental"] == "beginner"
    assert payload["defaultIsNotAdvanced"] is False
    assert payload["newcomerIsNotAdvanced"] is False
    assert payload["advanced"] == "advanced"
    assert payload["unknown"] == "advanced"
    assert payload["newcomerVisible"] is True
    assert payload["beginnerHidden"] is False
    assert payload["beginnerVisible"] is True
    assert payload["advancedVisible"] is True


def test_config_visibility_preferences_round_trip_and_normalize_invalid_values() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for config visibility checks")
    module_uri = (STATIC / "js/dragon-ui/pages/config-ui-preferences.js").resolve().as_uri()
    script = f"""
const store = new Map();
globalThis.localStorage = {{
  getItem: (key) => store.get(key) ?? null,
  setItem: (key, value) => store.set(key, String(value)),
}};
const mod = await import({json.dumps(module_uri + '?visibility-preference-test')});
const initial = {{ level: mod.preferredConfigVisibilityLevel(), hide: mod.preferredConfigHideUnavailable() }};
mod.persistConfigVisibilityLevel('beginner');
mod.persistConfigHideUnavailable(true);
const saved = {{ level: mod.preferredConfigVisibilityLevel(), hide: mod.preferredConfigHideUnavailable() }};
mod.persistConfigVisibilityLevel('invalid');
const normalized = mod.preferredConfigVisibilityLevel();
console.log(JSON.stringify({{ initial, saved, normalized }}));
"""
    payload = _node_json(script)

    assert payload["initial"] == {"level": "all", "hide": False}
    assert payload["saved"] == {"level": "beginner", "hide": True}
    assert payload["normalized"] == "beginner"


def test_all_config_visibility_controls_and_filter_contract_are_wired() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    view = _read("js/dragon-ui/pages/config-all-view.js")
    tiers = _read("js/dragon-ui/pages/config-field-tiers.js")
    controls = _read("js/dragon-ui/pages/config-visibility-controls.js")
    field_filter = _read("js/dragon-ui/pages/config-field-filter.js")
    data = _read("js/dragon-ui/pages/config-training-data.js")
    css = _read("css/dragon/04c-dragon-config-all.css")

    for label in ("精简", "默认", "全部适用"):
        assert label in tiers
    assert "新人" not in tiers
    assert "初学者" not in tiers
    assert "<span>显示范围</span>" in view
    assert "隐藏不可用配置项" in view
    assert 'data-config-visibility-level="${option.id}"' in view
    assert 'class="dragon-config-visibility-label"' in view
    assert "keys: ['dataset_config']" in page
    assert "blockEntries" in page
    assert "chapterLeadCounts" not in view
    assert 'data-config-hide-unavailable' in view
    assert 'data-config-show-all-candidates' in view
    assert 'role="menuitemradio"' in view
    assert "bindConfigVisibilityControls" in page
    assert "configFieldVisibleAtLevel" in field_filter
    assert "matchesVisibility" in field_filter
    assert "matchesAvailability" in field_filter
    assert "matchesPresentation" in field_filter
    assert "显示 ${visible} / 适用 ${applicable} · 候选 ${fields.length}" in field_filter
    assert "state?.filterUpdate?.()" in page
    assert "data-config-visibility-level" in data
    assert 'data-config-advanced="${Boolean(block.advanced)}"' in page
    assert "--dragon-config-advanced-inset" not in css
    assert "document.addEventListener('keydown', escapeHandler)" in controls
