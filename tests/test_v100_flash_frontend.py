from pathlib import Path


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
    assert "'compile_dynamic_seq', 'debug_finite_checks'" in form_layout
    assert "v100_flash_stability: help(" in help_training
    assert "debug_finite_checks: help(" in help_training
    assert "./catalog/field-help.js?v=module-bootstrap-20260809-nf4-v2" in catalog
    assert "./field-help-training.js?v=dragon-ui-20260825v1" in field_help
