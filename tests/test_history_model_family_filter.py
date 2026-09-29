from __future__ import annotations

from web.services.training.history_config_chips import (
    history_config_chips_from_snapshot_text,
)


def test_history_config_chips_resolve_model_family() -> None:
    assert history_config_chips_from_snapshot_text(
        'model_family = "krea2_raw"\n'
    )["model_family"] == "krea2_raw"
    assert history_config_chips_from_snapshot_text(
        'model_family = "z_image"\n'
    )["model_family"] == "z_image"
    assert history_config_chips_from_snapshot_text(
        'network_module = "networks.lora_anima"\n'
    )["model_family"] == "anima"
    assert history_config_chips_from_snapshot_text(
        'model_family = "unknown"\n'
    )["model_family"] == ""
