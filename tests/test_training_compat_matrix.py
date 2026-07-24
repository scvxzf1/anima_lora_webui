from __future__ import annotations

import argparse

import pytest

from library.training.compat_matrix import (
    apply_training_compat_mutations,
    check_training_compat,
)


def _codes(items) -> set[str]:
    return {item.code for item in items}


def test_matrix_rejects_unsloth_with_cpu_offload_and_suggests_grad_checkpoint() -> None:
    result = check_training_compat(
        {
            "unsloth_offload_checkpointing": True,
            "cpu_offload_checkpointing": True,
            "gradient_checkpointing": False,
        }
    )

    assert "unsloth_cpu_offload" in _codes(result.errors)
    assert "unsloth_enables_gradient_checkpointing" in _codes(result.warnings)
    assert [(m.key, m.value) for m in result.mutations] == [("gradient_checkpointing", True)]


def test_matrix_rejects_block_swap_soft_tokens_and_functional_loss() -> None:
    result = check_training_compat(
        {
            "blocks_to_swap": 8,
            "network_module": "networks.methods.soft_tokens",
            "functional_loss_weight": 0.1,
        }
    )

    assert {"block_swap_soft_tokens", "block_swap_functional_loss"} <= _codes(result.errors)


def test_matrix_warns_cpu_offload_without_full_checkpointing() -> None:
    result = check_training_compat(
        {
            "cpu_offload_checkpointing": True,
            "gradient_checkpointing": False,
        }
    )

    assert result.ok
    assert "cpu_offload_without_gradient_checkpointing" in _codes(result.warnings)


def test_matrix_downgrades_block_swap_cudagraph_compile_modes() -> None:
    cudagraphs = check_training_compat(
        {
            "blocks_to_swap": 8,
            "torch_compile": True,
            "dynamo_backend": "cudagraphs",
        }
    )
    max_autotune = check_training_compat(
        {
            "blocks_to_swap": 8,
            "torch_compile": True,
            "dynamo_backend": "inductor",
            "compile_inductor_mode": "max-autotune",
        }
    )

    assert [(m.key, m.value) for m in cudagraphs.mutations] == [("torch_compile", False)]
    assert [(m.key, m.value) for m in max_autotune.mutations] == [
        ("compile_inductor_mode", "max-autotune-no-cudagraphs")
    ]


def test_apply_training_compat_mutations_supports_namespace_and_dict() -> None:
    ns = argparse.Namespace(gradient_checkpointing=False)
    mapping = {"gradient_checkpointing": False}
    result = check_training_compat({"unsloth_offload_checkpointing": True})

    apply_training_compat_mutations(ns, result)
    apply_training_compat_mutations(mapping, result)

    assert ns.gradient_checkpointing is True
    assert mapping["gradient_checkpointing"] is True


def test_trainer_uses_shared_matrix_for_compile_mutations() -> None:
    import train

    args = argparse.Namespace(
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs=False,
        cache_llm_adapter_outputs=False,
        network_train_unet_only=True,
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
        gradient_checkpointing=False,
        blocks_to_swap=8,
        torch_compile=True,
        dynamo_backend="inductor",
        compile_inductor_mode="reduce-overhead",
        network_module="networks.lora_anima",
        functional_loss_weight=0.0,
        selective_checkpoint="off",
        selective_checkpoint_blocks="",
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="foreach",
    )

    train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)

    assert args.torch_compile is True
    assert args.compile_inductor_mode is None


def test_trainer_raises_shared_matrix_errors() -> None:
    import train

    args = argparse.Namespace(
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs=False,
        cache_llm_adapter_outputs=False,
        network_train_unet_only=True,
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
        gradient_checkpointing=False,
        blocks_to_swap=8,
        torch_compile=False,
        dynamo_backend="inductor",
        compile_inductor_mode=None,
        network_module="networks.methods.soft_tokens",
        functional_loss_weight=0.0,
        selective_checkpoint="off",
        selective_checkpoint_blocks="",
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="foreach",
    )

    with pytest.raises(ValueError, match="soft_tokens"):
        train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)


def test_matrix_rejects_convrot_with_block_swap_int8() -> None:
    result = check_training_compat(
        {
            "base_compute": "w8a16_convrot",
            "block_swap_transfer_dtype": "int8",
        }
    )
    assert "convrot_block_swap_int8_mutex" in _codes(result.errors)


def test_matrix_accepts_convrot_with_bf16_transfer() -> None:
    result = check_training_compat(
        {
            "base_compute": "w8a16_convrot",
            "block_swap_transfer_dtype": "bf16",
        }
    )
    assert "convrot_block_swap_int8_mutex" not in _codes(result.errors)
    assert "invalid_base_compute" not in _codes(result.errors)


class _CacheableDataset:
    datasets: list[object] = []

    def is_text_encoder_output_cacheable(self, *, cache_supports_dropout: bool) -> bool:
        return True

    def verify_bucket_reso_steps(self, steps: int) -> None:
        assert steps == 16
