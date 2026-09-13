import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web/static/js"


def test_auto_fields_types_groups_and_manual_preservation():
    if not shutil.which("node"):
        pytest.skip("node required")
    modules = {
        "types": "dragon-ui/pages/config-field-types.js",
        "availability": "dragon-ui/pages/config-field-availability.js",
        "catalog": "dragon-ui/pages/config-field-catalog.js",
        "map": "dragon-ui/pages/config-field-map.js",
        "controls": "config/catalog/resource-controls.js",
        "families": "features/config-form/model-family.js",
    }
    script = "\n".join(
        f"const {key} = await import({json.dumps((STATIC / path).as_uri())});"
        for key, path in modules.items()
    )
    script += """
const resolve = (key, auto) => availability.configFieldAvailability(key, {
  values: { auto_block_swap: auto, blocks_to_swap: 0 }, modelFamily: 'anima',
});
console.log(JSON.stringify({
  boolean: types.isBooleanConfigField('auto_block_swap', undefined),
  defaultValue: types.booleanDefaultForKey('auto_block_swap'),
  manual: resolve('blocks_to_swap', true),
  manualOff: resolve('blocks_to_swap', false),
  childOn: resolve('block_swap_restore_mode', true),
  childOff: resolve('auto_block_swap_timeout', false),
  reserveOff: resolve('auto_block_swap_vram_reserve_percent', false),
  preferenceOn: resolve('auto_block_swap_preference', true),
  bounds: controls.RESOURCE_NUMBER_CONSTRAINTS.auto_block_swap_vram_reserve_percent,
  labels: ['balanced', 'vram', 'ram'].map(value => controls.resourceOptionLabel('auto_block_swap_preference', value)),
  ramSupported: ['anima', 'krea2_raw', 'z_image'].map(family => families.modelFamilyOptionSupported('auto_block_swap_preference', family, 'ram')),
  dynamicWindow: availability.configFieldAvailability('auto_block_swap_interval', {
    values: { auto_block_swap: true, auto_block_swap_mode: 'dynamic' }, modelFamily: 'krea2_raw',
  }),
  dynamicTrials: availability.configFieldAvailability('auto_block_swap_max_trials', {
    values: { auto_block_swap: true, auto_block_swap_mode: 'dynamic' }, modelFamily: 'krea2_raw',
  }),
  keys: map.keysForConfigSubItem({id:'block-swap'}),
  entry: catalog.configFieldCatalogEntry('auto_block_swap'),
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    values = json.loads(result.stdout)
    assert values["boolean"] and not values["defaultValue"]
    assert not values["manual"]["enabled"]
    assert values["manualOff"]["enabled"]
    assert values["childOn"]["enabled"]
    assert not values["childOff"]["enabled"]
    assert not values["reserveOff"]["enabled"]
    assert values["preferenceOn"]["enabled"]
    assert values["bounds"] == {"min": "0", "max": "90", "step": "0.1"}
    assert values["labels"] == ["均衡", "优先节省显存", "优先节省内存"]
    assert values["ramSupported"] == [False, True, False]
    assert values["dynamicWindow"]["enabled"]
    assert not values["dynamicTrials"]["enabled"]
    assert all(
        key in values["keys"]
        for key in (
            "auto_block_swap",
            "auto_block_swap_mode",
            "auto_block_swap_interval",
            "auto_block_swap_vram_reserve_percent",
            "auto_block_swap_preference",
            "auto_block_swap_max_trials",
            "auto_block_swap_timeout",
        )
    )
    assert values["entry"]["stage"] == "resources"


def test_parser_schema_preserves_manual_integer_contract():
    import train
    from library.config import schema

    parser = train.setup_parser()
    schema.populate_schema(parser)
    args = parser.parse_args(["--auto_block_swap", "--blocks_to_swap", "12"])
    assert args.auto_block_swap and args.blocks_to_swap == 12
    assert schema.CONFIG_SCHEMA["auto_block_swap"].type == "bool"
    assert schema.CONFIG_SCHEMA["blocks_to_swap"].type == "int"
    assert schema.CONFIG_SCHEMA["auto_block_swap_max_trials"].type == "int"
    assert schema.CONFIG_SCHEMA["auto_block_swap_interval"].type == "int"
    assert schema.CONFIG_SCHEMA["auto_block_swap_mode"].type == "str"
    assert schema.CONFIG_SCHEMA["auto_block_swap_vram_reserve_percent"].type == "float"
    assert schema.CONFIG_SCHEMA["auto_block_swap_preference"].type == "str"


def test_auto_module_imports_have_one_cache_identity():
    token = "auto-block-swap-20260908-v3"
    versions = {}
    pattern = re.compile(r"(?:from\s*|import\s*\()['\"]([^'\"]+\.js)\?v=([^'\"]+)['\"]")
    for source in (ROOT / "web/static").rglob("*.js"):
        for relative, version in pattern.findall(source.read_text(encoding="utf-8")):
            target = (source.parent / relative).resolve()
            versions.setdefault(target, set()).add(version)
    auto_targets = {
        target: values for target, values in versions.items() if token in values
    }
    assert auto_targets
    assert all(values == {token} for values in auto_targets.values()), auto_targets
