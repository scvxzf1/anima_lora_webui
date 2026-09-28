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


def test_preflight_rejects_qwen_network_args_outside_plain_lora(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'model_family = "qwen_image_2_1"',
            'network_module = "networks.lora_anima"',
            'network_args = ["dora_wd=true"]',
        ],
    )

    result = _preflight()

    assert result["ok"] is False
    assert any(
        "plain LoRA" in message
        for message in _messages(result, "errors", "network_module")
    )


def test_preflight_rejects_qwen_register_tokens_before_network_creation(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'model_family = "qwen_image_2_1"',
            'network_module = "networks.lora_anima"',
            'network_args = ["num_registers=4"]',
        ],
    )

    result = _preflight()

    assert result["ok"] is False
    assert any(
        "num_registers" in message
        for message in _messages(result, "errors", "network_module")
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


def test_preflight_defers_unresolved_auto_precision_contract(tmp_path: Path, monkeypatch) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'adaptive_precision = "auto"',
            'mixed_precision = "bf16"',
            "adaptive_oom_retry = true",
        ],
    )

    result = _preflight()

    assert result["ok"] is True
    assert not _messages(result, "errors", "adaptive_precision")
    warnings = _messages(result, "warnings", "adaptive_precision")
    assert len(warnings) == 1
    assert "compute capability" in warnings[0]


def test_preflight_does_not_use_raw_mixed_precision_for_auto_krea2_attention(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'model_family = "krea2_raw"',
            'adaptive_precision = "auto"',
            'mixed_precision = "fp16"',
            'attn_mode = "flash"',
        ],
    )

    result = _preflight()

    assert result["ok"] is True
    assert not _messages(result, "errors", "adaptive_precision")
    assert len(_messages(result, "warnings", "adaptive_precision")) == 1


def test_preflight_keeps_explicit_fp16_precision_contract_strict(
    tmp_path: Path, monkeypatch
) -> None:
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'model_family = "krea2_raw"',
            'adaptive_precision = "fp16_fp32"',
            'mixed_precision = "fp16"',
            'attn_mode = "flash"',
        ],
    )

    result = _preflight()

    assert result["ok"] is False
    assert any(
        "attn_mode='torch'" in msg
        for msg in _messages(result, "errors", "adaptive_precision")
    )
