import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NEXT_DOMAIN = ROOT / "web" / "frontend-next" / "src" / "features" / "training-config" / "domain"
SNAPSHOT = ROOT / "docs" / "configuration" / "dragon-training-config-215.md"
GRADIENT_FLOW_KEYS = [
    "gradient_flow_probe_jsonl",
    "gradient_flow_probe_every_n_steps",
    "gradient_flow_probe_dense_steps",
]
AUTO_RESOURCE_KEYS = ["auto_block_swap_vram_reserve_percent", "auto_block_swap_preference"]


def _frozen_keys() -> list[str]:
    text = SNAPSHOT.read_text(encoding="utf-8")
    return re.findall(r"^\|\s*\d+\s*\|.*?`([^`]+)`", text, flags=re.MULTILINE)


def _node_json(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon config catalog checks")
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return json.loads(result.stdout)


def test_stage_catalog_covers_frozen_and_current_runtime_fields_once() -> None:
    frozen = _frozen_keys()
    assert len(frozen) == 215
    assert len(set(frozen)) == 215
    current = [*frozen, *GRADIENT_FLOW_KEYS, *AUTO_RESOURCE_KEYS]
    module_uri = (NEXT_DOMAIN / "config-field-catalog.js").resolve().as_uri()
    payload = _node_json(f"""
const mod = await import({json.dumps(module_uri + '?coverage')});
const keys = {json.dumps(current)};
const entries = keys.map((key) => mod.configFieldCatalogEntry(key));
const counts = Object.fromEntries(mod.CONFIG_STAGE_META.map((stage) => [
  stage.id,
  entries.filter((entry) => entry.stage === stage.id).length,
]));
console.log(JSON.stringify({{
  missing: keys.filter((key) => !mod.CONFIG_FIELD_CATALOG[key]),
  uniqueOwners: new Set(entries.map((entry) => entry.key)).size,
  counts,
  errors: mod.validateConfigFieldCatalog(),
  gradient: {json.dumps(GRADIENT_FLOW_KEYS)}.map((key) => mod.CONFIG_FIELD_CATALOG[key]),
  unknown: mod.configFieldCatalogEntry('__new_runtime_field__'),
}}));
""")

    assert payload["missing"] == []
    assert payload["uniqueOwners"] == 220
    assert payload["counts"] == {
        "input": 21,
        "method": 105,
        "training": 44,
        "resources": 50,
    }
    assert payload["errors"] == []
    assert {entry["stage"] for entry in payload["gradient"]} == {"resources"}
    assert {entry["cluster"] for entry in payload["gradient"]} == {"diagnostics"}
    assert payload["unknown"]["location"] == "audit_only"
    assert payload["unknown"]["cluster"] == "unclassified"


def test_stage_catalog_rejects_invalid_ownership_and_dependencies() -> None:
    module_uri = (NEXT_DOMAIN / "config-field-catalog.js").resolve().as_uri()
    payload = _node_json(f"""
const mod = await import({json.dumps(module_uri + '?negative-invariants')});
const entry = (key, stage, cluster, siblingOrder, dependsOn = []) => ({{
  key, stage, cluster, siblingOrder, dependsOn, controlledBy: [],
  location: 'body', policy: {{ kind: 'always' }},
}});
const duplicate = entry('same', 'input', 'models', 10);
const cases = {{
  duplicate: mod.validateConfigFieldCatalog([duplicate, {{ ...duplicate }}]),
  missing: mod.validateConfigFieldCatalog([entry('child', 'input', 'models', 20, ['absent'])]),
  unknown: mod.validateConfigFieldCatalog([entry('odd', 'unknown', 'none', 10)]),
  reverse: mod.validateConfigFieldCatalog([
    entry('early', 'input', 'models', 10, ['late']),
    entry('late', 'method', 'contract', 10),
  ]),
  cycle: mod.validateConfigFieldCatalog([
    entry('a', 'training', 'volume', 10, ['b']),
    entry('b', 'training', 'volume', 20, ['a']),
  ]),
}};
console.log(JSON.stringify(Object.fromEntries(Object.entries(cases).map(([name, errors]) => [
  name,
  errors.map((error) => error.code),
]))));
""")

    assert "duplicate-owner" in payload["duplicate"]
    assert "missing-reference" in payload["missing"]
    assert "unknown-stage" in payload["unknown"]
    assert "reverse-dependency" in payload["reverse"]
    assert "dependency-cycle" in payload["cycle"]


def test_stage_catalog_drives_deterministic_dependency_order() -> None:
    current = [*_frozen_keys(), *GRADIENT_FLOW_KEYS, *AUTO_RESOURCE_KEYS]
    module_uri = (NEXT_DOMAIN / "config-field-catalog.js").resolve().as_uri()
    payload = _node_json(f"""
const mod = await import({json.dumps(module_uri + '?ordering')});
const keys = {json.dumps(list(reversed(current)))};
const ordered = mod.sortConfigCatalogItems(keys.map((key) => mod.configFieldCatalogEntry(key)))
  .map((entry) => entry.key);
const lokr = ordered.filter((key) => key === 'lokr_factor' || key.startsWith('lokr_'));
console.log(JSON.stringify({{ ordered, lokr }}));
""")
    ordered = payload["ordered"]
    index = {key: position for position, key in enumerate(ordered)}

    assert index["model_family"] < index["pretrained_model_name_or_path"] < index["dataset_config"]
    assert index["lora_adapter_kind"] < index["network_weights"] < index["network_dim"]
    assert index["use_vae_cache"] < index["reuse_vae_latents"]
    assert index["timestep_sampling"] < index["weighting_scheme"] < index["min_snr_gamma"]
    assert index["torch_compile"] < index["compile_dynamic_seq"] < index["compile_seq_bands"]
    assert index["gradient_flow_probe_jsonl"] < index["gradient_flow_probe_every_n_steps"]

    lokr_positions = [index[key] for key in payload["lokr"]]
    assert len(lokr_positions) == 9
    assert lokr_positions == list(range(min(lokr_positions), max(lokr_positions) + 1))
