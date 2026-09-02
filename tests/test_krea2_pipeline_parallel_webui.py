from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.web_config_test_support import _write_selected_checkpoint_preflight_config
from web.services import config_service
from web.services.config.form_metadata import FIELD_HELP, FORM_GROUPS


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
PP_FIELDS = {
    "pipeline_parallel",
    "pipeline_parallel_stages",
    "pipeline_parallel_microbatches",
    "pipeline_parallel_schedule",
    "pipeline_parallel_split",
}


def _read(path: str) -> str:
    return (STATIC / path).read_text(encoding="utf-8")


def _preflight(*, world_size: int | None = None) -> dict:
    return config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
        world_size=world_size,
    )


def _messages(result: dict, key: str) -> list[str]:
    return [item["message"] for item in result["errors"] if item["key"] == key]


def test_pipeline_fields_are_exposed_with_help_and_safe_defaults() -> None:
    defaults = _read("js/config/catalog/defaults.js")
    labels = _read("js/config/catalog/labels-options.js")
    layout = _read("js/config/catalog/form-layout.js")
    summary = _read("js/config/catalog/field-help-summary.js")
    training_help = _read("js/config/catalog/field-help-training.js")

    assert PP_FIELDS <= FORM_GROUPS["Performance"]
    assert PP_FIELDS <= FIELD_HELP.keys()
    assert all({"en", "ko"} <= FIELD_HELP[key].keys() for key in PP_FIELDS)
    assert "pipeline_parallel: false" in defaults
    assert "pipeline_parallel_stages: 2" in defaults
    assert "pipeline_parallel_microbatches: 4" in defaults
    assert "pipeline_parallel_schedule: '1f1b'" in defaults
    assert "pipeline_parallel_split: 'balanced'" in defaults
    for key in PP_FIELDS:
        assert f"{key}:" in labels
        assert f"'{key}'" in layout
        assert f'"{key}"' in summary
        assert f"{key}: help(" in training_help


def test_pipeline_field_availability_tracks_family_and_master_switch() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for config availability checks")
    module_uri = (
        (STATIC / "js/dragon-ui/pages/config-field-availability.js").resolve().as_uri()
    )
    script = f"""
const mod = await import({json.dumps(module_uri + "?krea2-pp-test")});
const check = (key, context) => mod.configFieldAvailability(key, context);
console.log(JSON.stringify({{
  animaToggle: check('pipeline_parallel', {{ modelFamily: 'anima', pipelineParallel: false }}),
  kreaAliasToggle: check('pipeline_parallel', {{ modelFamily: 'krea2', pipelineParallel: false }}),
  disabledChild: check('pipeline_parallel_microbatches', {{ modelFamily: 'krea2_raw', pipelineParallel: false }}),
  enabledChild: check('pipeline_parallel_microbatches', {{ modelFamily: 'krea2_raw', pipelineParallel: true }}),
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(result.stdout)

    assert payload["animaToggle"]["enabled"] is False
    assert payload["animaToggle"]["code"] == "pipeline-parallel-model-family"
    assert payload["kreaAliasToggle"] == {"enabled": True, "reason": "", "code": None}
    assert payload["disabledChild"]["enabled"] is False
    assert payload["disabledChild"]["code"] == "pipeline-parallel-disabled"
    assert payload["enabledChild"] == {"enabled": True, "reason": "", "code": None}


def test_classic_form_rerenders_when_pipeline_switch_changes() -> None:
    form_fields = _read("js/features/config-form/form-fields-ui.js")

    assert "const isKrea2 = isKrea2ModelFamily(family)" in form_fields
    assert "|| event?.target?.dataset?.key === 'pipeline_parallel'" in form_fields
    assert "!currentPipelineParallelEnabled()" in form_fields


def test_classic_krea_aliases_share_family_and_live_compat_behavior() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for model-family checks")
    family_uri = (STATIC / "js/features/config-form/model-family.js").resolve().as_uri()
    compat_uri = (STATIC / "js/features/config-form/live-compat.js").resolve().as_uri()
    script = f"""
const family = await import({json.dumps(family_uri + "?krea2-alias-test")});
const compat = await import({json.dumps(compat_uri + "?krea2-alias-test")});
const config = {{ compile_inductor_mode: 'reduce-overhead' }};
console.log(JSON.stringify({{
  alias: family.isKrea2ModelFamily(' KREA2 '),
  canonical: family.isKrea2ModelFamily('krea2_raw'),
  anima: family.isKrea2ModelFamily('anima'),
  aliasCodes: compat.collectLiveCompatIssues({{ ...config, model_family: 'krea2' }}).map((item) => item.code),
  canonicalCodes: compat.collectLiveCompatIssues({{ ...config, model_family: 'krea2_raw' }}).map((item) => item.code),
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(result.stdout)

    assert payload["alias"] is True
    assert payload["canonical"] is True
    assert payload["anima"] is False
    assert payload["aliasCodes"] == payload["canonicalCodes"]
    assert "krea2_compile_inductor_mode" in payload["aliasCodes"]


def test_pipeline_frontend_cache_chain_reaches_both_ui_modes() -> None:
    module_token = "module-bootstrap-20260903-pp-audit-v2"
    catalog_token = "module-bootstrap-20260902-krea2-pp-v1"
    dragon_token = "dragon-ui-20260902-krea2-pp-v1"

    assert "ui-bootstrap.js?v=" in _read("index.html")
    bootstrap = _read("js/ui-bootstrap.js")
    assert f"app.js?v={module_token}" in bootstrap
    assert "dragon-ui/index.js?v=" in bootstrap
    assert f"config/catalog.js?v={catalog_token}" in _read("app.js")
    assert f"anima-app/index.js?v={module_token}" in _read("app.js")
    assert f"02-ensure-history-detail-feature.js?v={module_token}" in _read(
        "js/features/anima-app/index.js"
    )
    assert f"config-form/index.js?v={module_token}" in _read(
        "js/features/anima-app/chunks/02-ensure-history-detail-feature.js"
    )
    assert f"form-fields.js?v={module_token}" in _read(
        "js/features/config-form/index.js"
    )
    assert f"form-fields-ui.js?v={module_token}" in _read(
        "js/features/config-form/form-fields.js"
    )
    assert f"model-family.js?v={module_token}" in _read(
        "js/features/config-form/form-fields-ui.js"
    )
    assert f"live-compat.js?v={module_token}" in _read(
        "js/features/config-form/form-fields-ui.js"
    )
    assert f"page-loaders.js?v={dragon_token}" in _read("js/dragon-ui/index.js")
    assert f"config-page.js?v={dragon_token}" in _read("js/dragon-ui/page-loaders.js")
    assert f"config-field-availability.js?v={dragon_token}" in _read(
        "js/dragon-ui/pages/config-page.js"
    )


def test_preflight_rejects_pipeline_for_non_krea_family(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        ["pipeline_parallel = true"],
    )

    result = _preflight()

    assert result["ok"] is False
    assert any(
        "仅支持 Krea-2 Raw" in message
        for message in _messages(result, "pipeline_parallel")
    )


def test_preflight_keeps_valid_krea_pipeline_launch_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'model_family = "krea2_raw"',
            "pipeline_parallel = true",
            "pipeline_parallel_stages = 2",
            "pipeline_parallel_microbatches = 4",
            'pipeline_parallel_schedule = "1f1b"',
            'pipeline_parallel_split = "balanced"',
            "torch_compile = false",
            "blocks_to_swap = 0",
            'selective_checkpoint = "off"',
        ],
    )

    result = _preflight()
    messages = _messages(result, "pipeline_parallel")

    assert result["ok"] is False
    assert any("主训练 loop 的 1F1B 调度尚未接入" in message for message in messages)
    assert not any("流水线配置无效" in message for message in messages)


@pytest.mark.parametrize("world_size", [1, 3])
def test_preflight_rejects_pipeline_when_selected_gpu_count_is_not_two(
    tmp_path: Path, monkeypatch, world_size: int
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'model_family = "krea2_raw"',
            "pipeline_parallel = true",
            "pipeline_parallel_stages = 2",
            "torch_compile = false",
            "blocks_to_swap = 0",
            'selective_checkpoint = "off"',
        ],
    )

    result = _preflight(world_size=world_size)
    messages = _messages(result, "pipeline_parallel")

    assert result["ok"] is False
    assert any("流水线配置无效" in message for message in messages)
    assert not any(
        "主训练 loop 的 1F1B 调度尚未接入" in message for message in messages
    )


def test_preflight_reports_invalid_pipeline_config_before_runtime_gate(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'model_family = "krea2_raw"',
            "pipeline_parallel = true",
            "torch_compile = true",
            "blocks_to_swap = 0",
            'selective_checkpoint = "off"',
        ],
    )

    result = _preflight()
    messages = _messages(result, "pipeline_parallel")

    assert result["ok"] is False
    assert any("流水线配置无效" in message for message in messages)
    assert not any(
        "主训练 loop 的 1F1B 调度尚未接入" in message for message in messages
    )
