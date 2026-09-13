from pathlib import Path
import tomllib

import pytest


CATALOG_DIR = (
    Path(__file__).resolve().parents[1] / "web" / "static" / "js" / "config" / "catalog"
)


def _read_catalog(name: str) -> str:
    return (CATALOG_DIR / name).read_text(encoding="utf-8")


def test_v100_flash_fields_are_exposed_in_frontend_catalog() -> None:
    catalog = (CATALOG_DIR.parent / "catalog.js").read_text(encoding="utf-8")
    field_help = _read_catalog("field-help.js")
    labels_options = _read_catalog("labels-options.js")
    form_layout = _read_catalog("form-layout.js")
    help_training = _read_catalog("field-help-training.js")

    assert "v100_flash_stability: 'V100 Flash 诊断模式'" in labels_options
    assert "debug_finite_checks: '有限值快速失败'" in labels_options
    assert "v100_flash_stability: ['off', 'hybrid', 'safe']" in labels_options
    assert "'attn_mode', 'v100_flash_stability', 'torch_compile'" in form_layout
    assert "'compile_dynamic_seq', 'compile_seq_bands', 'debug_finite_checks'" in form_layout
    assert "v100_flash_stability: help(" in help_training
    assert "debug_finite_checks: help(" in help_training
    assert "./catalog/field-help.js?v=auto-block-swap-20260908-v3" in catalog
    assert "./field-help-training.js?v=auto-block-swap-20260908-v3" in field_help


@pytest.mark.parametrize("relative", [
    "configs/base.toml",
    "configs/methods/krea2_lora.toml",
    "configs/methods/z_image_lora.toml",
])
def test_production_attention_defaults_match_frontend_help(relative: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / relative).read_text(encoding="utf-8"))
    assert config["attn_mode"] == "flash"
    help_training = _read_catalog("field-help-training.js")
    attention_help = help_training.split("    attn_mode: help(", 1)[1].split(
        "    v100_flash_stability: help(", 1
    )[0]
    assert "Anima、Krea-2 和 Z-Image 的生产默认配置均使用 flash" in attention_help
    assert "默认 torch 保持历史行为" not in attention_help
    assert "V100 稳定性专用配置仍以 torch 为准" in attention_help
