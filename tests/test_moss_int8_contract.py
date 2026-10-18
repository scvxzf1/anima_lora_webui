from __future__ import annotations

from library.runtime.int8_linear import classify_frozen_linear_module, selected_int8_linear_modules
from library.training.compat_matrix import check_training_compat


def test_moss_int8_transfer_is_allowed_but_warns(caplog) -> None:
    import logging

    caplog.set_level(logging.WARNING, logger="library.training.moss_compat")
    result = check_training_compat(
        {
            "model_family": "moss_image",
            "block_swap_transfer_dtype": "int8",
            "blocks_to_swap": 20,
            "train_batch_size": 1,
        }
    )

    assert result.ok
    warnings = {issue.code: issue.message for issue in result.warnings}
    assert "moss_image_int8_experimental" in warnings
    assert "0.3%-2.0%" in warnings["moss_image_int8_experimental"]
    assert any("MOSS-Image int8 block-swap transfer is experimental" in record.message for record in caplog.records)


def test_moss_projection_names_are_int8_candidates() -> None:
    assert classify_frozen_linear_module("blocks.0.proj.q", scope="all") == (0, "attention")
    assert classify_frozen_linear_module("blocks.0.proj.k", scope="all") == (0, "attention")
    assert classify_frozen_linear_module("blocks.0.proj.v", scope="all") == (0, "attention")
    assert classify_frozen_linear_module("blocks.0.proj.o", scope="all") == (0, "attention")
    assert classify_frozen_linear_module("blocks.0.proj.ffn_up_gate", scope="all") == (0, "mlp")
    assert classify_frozen_linear_module("blocks.0.proj.ffn_down", scope="all") == (0, "mlp")
    selected = selected_int8_linear_modules("all")
    assert {
        "proj.q", "proj.k", "proj.v", "proj.o",
        "proj.ffn_up_gate", "proj.ffn_down",
    }.issubset(selected)
