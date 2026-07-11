"""C-R2/C-R11 residual: configs_root hot-swap matrix across domain modules."""

from __future__ import annotations

from pathlib import Path

from web.services import config_service
from web.services.config import (
    dataset_editor,
    dataset_media,
    dataset_preset_paths,
    dataset_presets_api,
    dataset_rows,
    datasets,
    file_group_runtime,
    file_groups,
    merge,
    output_runs,
    preflight_runtime,
    raw_files,
    sample_prompts,
)


DOMAIN_MODULES = (
    merge,
    preflight_runtime,
    raw_files,
    sample_prompts,
    file_groups,
    file_group_runtime,
    datasets,
    dataset_media,
    dataset_editor,
    dataset_rows,
    dataset_preset_paths,
    dataset_presets_api,
    output_runs,
)


def test_set_configs_root_broadcasts_to_all_domain_modules(tmp_path, monkeypatch):
    target = (tmp_path / "broadcast-configs").resolve()
    target.mkdir(parents=True)

    # Snapshot originals via monkeypatch so we do not leak across tests.
    for mod in (config_service, *DOMAIN_MODULES):
        if hasattr(mod, "CONFIGS_DIR"):
            monkeypatch.setattr(mod, "CONFIGS_DIR", getattr(mod, "CONFIGS_DIR"), raising=False)

    resolved = config_service.set_configs_root(target)
    assert resolved == target
    assert config_service.CONFIGS_DIR == target

    for mod in DOMAIN_MODULES:
        assert getattr(mod, "CONFIGS_DIR") == target, f"{mod.__name__}.CONFIGS_DIR not hot-swapped"
        if hasattr(mod, "DATASET_PRESETS_DIR"):
            assert Path(getattr(mod, "DATASET_PRESETS_DIR")) == target / "datasets"
        if hasattr(mod, "IMPORTED_CONFIGS_DIR"):
            assert Path(getattr(mod, "IMPORTED_CONFIGS_DIR")) == target / "imported"
        if hasattr(mod, "GUI_METHODS_DIR"):
            assert Path(getattr(mod, "GUI_METHODS_DIR")) == target / "gui-methods"
