from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.frontend_test_support import REPO_ROOT, STATIC_DIR


def test_dragon_config_boolean_fields_keep_switch_semantics() -> None:
    """Sparse merged configs must not turn method switches into text inputs."""
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon config field checks")

    types_uri = (STATIC_DIR / "js/dragon-ui/pages/config-field-types.js").resolve().as_uri()
    values_uri = (STATIC_DIR / "js/dragon-ui/pages/config-values.js").resolve().as_uri()
    options_uri = (STATIC_DIR / "js/config/catalog/labels-options.js").resolve().as_uri()
    script = f"""
const types = await import({json.dumps(types_uri + '?boolean-test')});
const values = await import({json.dumps(values_uri + '?boolean-test')});
const catalog = await import({json.dumps(options_uri + '?boolean-test')});

const fakeInput = ({{ value = '', type = 'text', checked = false, toggle = false }} = {{}}) => ({{
  value,
  type,
  checked,
  dataset: {{ key: '' }},
  classList: {{ contains: (name) => toggle && name === 'dragon-toggle' }},
}});

const booleanOptionKeys = Object.entries(catalog.FIELD_OPTIONS)
  .filter(([, options]) => types.isBooleanConfigField('', undefined, options))
  .map(([key]) => key);
const toggle = fakeInput({{ value: 'ignored', toggle: true }});
toggle.dataset.key = 'use_ortho';
toggle.dataset.checked = 'true';

console.log(JSON.stringify({{
  missingOrtho: values.displayConfigValue('use_ortho', {{}}),
  missingTLoRA: values.displayConfigValue('use_timestep_mask', {{}}),
  missingRoute: values.displayConfigValue('route_per_layer', {{}}),
  yes: types.normalizeBooleanConfigValue('use_ortho', 'yes'),
  no: types.normalizeBooleanConfigValue('use_ortho', 'no', true),
  booleanOptionKeys,
  cmmdIsToggle: types.isBooleanConfigField('use_cmmd', undefined, catalog.FIELD_OPTIONS.use_cmmd),
  moeIsToggle: types.isBooleanConfigField('use_moe_style', 'false', catalog.FIELD_OPTIONS.use_moe_style),
  toggleSerialized: values.serializeConfigValue(toggle, false),
  textBooleanSerialized: values.serializeConfigValue(fakeInput({{ value: 'false' }}), true),
  selectPreservesString: values.serializeConfigValue(fakeInput({{ value: 'shared_A' }}), 'false'),
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["missingOrtho"] is False
    assert payload["missingTLoRA"] is False
    assert payload["missingRoute"] is False
    assert payload["yes"] is True
    assert payload["no"] is False
    assert {"use_cmmd", "route_per_layer", "reuse_vae_latents"}.issubset(
        payload["booleanOptionKeys"]
    )
    assert payload["cmmdIsToggle"] is True
    assert payload["moeIsToggle"] is False
    assert payload["toggleSerialized"] is True
    assert payload["textBooleanSerialized"] is False
    assert payload["selectPreservesString"] == "shared_A"


def test_dragon_config_boolean_rendering_contract_is_explicit() -> None:
    page = (STATIC_DIR / "js/dragon-ui/pages/config-page.js").read_text(encoding="utf-8")
    metadata = (STATIC_DIR / "js/dragon-ui/pages/config-block-metadata.js").read_text(encoding="utf-8")
    values = (STATIC_DIR / "js/dragon-ui/pages/config-values.js").read_text(encoding="utf-8")

    assert "isBooleanConfigField(key, value, options)" in page
    assert "if (options && !booleanField)" in page
    assert 'role="switch"' in page
    assert "if (isBooleanConfigField(key, value, options)) return 'toggle';" in metadata
    assert "input.classList?.contains?.('dragon-toggle')" in values
    assert "input.dataset.checked === 'true'" in values
