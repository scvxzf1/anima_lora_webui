from __future__ import annotations

from pathlib import Path

from tests.web_config_test_support import _write_selected_checkpoint_preflight_config
from web.services import config_service


def _preflight() -> dict:
    return config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )


def _messages(result: dict, level: str, key: str) -> list[str]:
    return [item["message"] for item in result[level] if item["key"] == key]


def test_preflight_uses_shared_matrix_for_unsloth_cpu_conflict(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "gradient_checkpointing = true",
            "cpu_offload_checkpointing = true",
            "unsloth_offload_checkpointing = true",
        ],
    )

    result = _preflight()

    assert result["ok"] is False
    assert any(
        "unsloth_offload_checkpointing" in msg
        for msg in _messages(result, "errors", "cpu_offload_checkpointing")
    )


def test_preflight_uses_shared_matrix_for_block_swap_soft_tokens_and_functional_loss(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "blocks_to_swap = 8",
            'network_module = "networks.methods.soft_tokens"',
            "functional_loss_weight = 0.1",
        ],
    )

    result = _preflight()
    block_messages = _messages(result, "errors", "blocks_to_swap")

    assert result["ok"] is False
    assert any("Soft Tokens" in msg for msg in block_messages)
    assert any("functional_loss_weight" in msg for msg in block_messages)


def test_preflight_warns_cpu_offload_without_gradient_checkpointing(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "gradient_checkpointing = false",
            "cpu_offload_checkpointing = true",
            "unsloth_offload_checkpointing = false",
        ],
    )

    result = _preflight()

    assert result["ok"] is True
    assert any(
        "gradient_checkpointing" in msg
        for msg in _messages(result, "warnings", "cpu_offload_checkpointing")
    )


def test_preflight_warns_block_swap_cudagraph_compile_downgrades(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "blocks_to_swap = 8",
            "torch_compile = true",
            'dynamo_backend = "cudagraphs"',
        ],
    )

    result = _preflight()

    assert result["ok"] is True
    assert any("关闭 torch_compile" in msg for msg in _messages(result, "warnings", "torch_compile"))


def test_preflight_warns_block_swap_inductor_mode_downgrade(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "blocks_to_swap = 8",
            "torch_compile = true",
            'dynamo_backend = "inductor"',
            'compile_inductor_mode = "max-autotune"',
        ],
    )

    result = _preflight()

    assert result["ok"] is True
    assert any(
        "max-autotune-no-cudagraphs" in msg
        for msg in _messages(result, "warnings", "compile_inductor_mode")
    )
