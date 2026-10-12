"""CPU-only contract tests for native FP16 AMP product entry points."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from library.runtime import accelerator as runtime
from library.training.adaptive_runtime.native_precision import accelerator_handlers, native_enabled
from library.training.adaptive_runtime.precision import resolve_adaptive_precision
from library.training.adaptive_runtime.training_config import (
    add_arguments, configuration_errors, islands_enabled, require_training_contract,
)


def args(**changes):
    return SimpleNamespace(**{
        "adaptive_precision": "native_fp16", "mixed_precision": "fp16",
        "model_family": "anima", "adaptive_loss_scale": 1024.0,
        "adaptive_fp32_modules": [], "attn_mode": "torch",
        "gradient_checkpointing": True, "selective_checkpoint": "off",
        "network_module": "networks.lora_anima", "network_train_unet_only": True,
        "logging_dir": None, "log_with": None, "gradient_accumulation_steps": 1,
        "save_precision": "bf16", **changes,
    })


def test_native_cli_dtype_and_handlers():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert parser.parse_args(["--adaptive_precision", "native_fp16"]).adaptive_precision == "native_fp16"
    assert parser.parse_args([]).adaptive_precision == "off"
    assert runtime.prepare_dtype(args()) == (torch.float16, torch.bfloat16)
    assert native_enabled(args()) and not islands_enabled(args())
    handlers = accelerator_handlers(args())
    assert len(handlers) == 1 and handlers[0].init_scale == 1024.0
    assert handlers[0].growth_interval == 2000


@pytest.mark.parametrize("changes", [
    {"mixed_precision": "bf16"}, {"model_family": "zimage"},
    {"model_family": "unknown"}, {"base_compute": "nf4"}, {"nf4_prequantized_path": "nf4"},
    {"full_fp16": True}, {"full_bf16": True}, {"torch_compile": True},
    {"dim_from_weights": True, "network_weights": "non-lora.safetensors"},
    {"gradient_checkpointing": False}, {"selective_checkpoint": "every_other"},
    {"attn_mode": "flash"}, {"attn_mode": "sdpa"}, {"adaptive_oom_retry": True},
    {"adaptive_fp32_modules": ["blocks.0.*"]}, {"adaptive_fp32_modules": "wrong"},
    {"adaptive_loss_scale": float("nan")}, {"adaptive_loss_scale": float("inf")},
    {"adaptive_loss_scale": True}, {"adaptive_loss_scale": 1},
    {"adaptive_loss_scale": 2**24 + 1}, {"network_module": "networks.methods.soft_tokens"},
    {"dora_wd": True}, {"use_lokr": True}, {"use_loha": True},
    {"use_moe_style": "shared_A"}, {"use_timestep_mask": True}, {"add_reft": True},
    {"network_args": ["use_lokr=True"]}, {"network_args": ["dora_wd=True"]},
    {"train_llm_adapter": True}, {"network_train_text_encoder_only": True},
    {"network_train_unet_only": False}, {"router_source": "fei"}, {"num_registers": 2},
])
def test_native_contract_rejects_unsupported_combinations(changes):
    with pytest.raises(ValueError):
        require_training_contract(args(**changes))


def test_native_sampling_save_and_resume_remain_available():
    require_training_contract(args(sample_at_first=True, sample_every_n_steps=1,
                                   sample_every_n_epochs=1, save_state=True,
                                   save_state_on_train_end=True, resume="saved-state",
                                   checkpointing_epochs=1, save_every_n_epochs=1))
    require_training_contract(args(adaptive_loss_scale=2**24))
    require_training_contract(args(network_args=["rank_dropout=0.1", "dora_wd=False"]))


def test_native_contract_uses_effective_adapter_options():
    require_training_contract(args(use_lokr=True, network_args=["use_lokr=False"]))
    with pytest.raises(ValueError, match="one process"):
        require_training_contract(args(), world_size=2)


@pytest.mark.parametrize("capability,mode,mixed", [
    ((7, 5), "native_fp16", "fp16"), ((8, 6), "bf16", "bf16"),
    ((6, 1), "fp32", "no"),
])
def test_auto_anima_hardware_route(capability, mode, mixed):
    configured = args(adaptive_precision="auto", mixed_precision="bf16", attn_mode="flash")
    assert resolve_adaptive_precision(configured, get_capability=lambda: capability) == mode
    assert configured.mixed_precision == mixed
    if mode == "native_fp16":
        assert configured.attn_mode == "torch"
        assert "not numerical" in configured.adaptive_resolution_reason
    require_training_contract(configured)


@pytest.mark.parametrize("family", ["krea2", "krea2_raw", "zimage", "z_image"])
def test_other_family_fp16_routes_stay_islands(family):
    configured = args(model_family=family, adaptive_precision="auto", attn_mode="flash")
    assert resolve_adaptive_precision(configured, get_capability=lambda: (7, 5)) == "fp16_fp32"
    assert configured.attn_mode == "torch"


@pytest.mark.parametrize("cached", ["fp16_fp32", "native_fp16"])
def test_cached_mode_identity_is_preserved(cached):
    configured = args(adaptive_precision="auto", adaptive_resolved_mode=cached,
                      adaptive_candidate="fp16")
    assert resolve_adaptive_precision(configured, get_capability=lambda: pytest.fail("reprobed")) == cached
    assert islands_enabled(configured) is (cached == "fp16_fp32")


def test_auto_native_oom_requires_seed_and_unknown_family_fail_closed():
    configured = args(adaptive_precision="auto", adaptive_oom_retry=True)
    resolve_adaptive_precision(configured, get_capability=lambda: (7, 5))
    assert any("explicit integer seed" in error for error in configuration_errors(configured))
    with pytest.raises(ValueError, match="model_family must be one of"):
        resolve_adaptive_precision(args(model_family="unknown", adaptive_precision="auto"),
                                   get_capability=lambda: (7, 5))


def test_accelerator_native_autocast_kept_and_off_unchanged(monkeypatch):
    seen = []
    monkeypatch.setattr(runtime, "Accelerator", lambda **kwargs: seen.append(kwargs) or
                        SimpleNamespace(num_processes=1, device=torch.device("cuda"), scaler=object()))
    runtime.prepare_accelerator(args())
    assert len(seen[-1]["kwargs_handlers"]) == 1
    assert seen[-1]["kwargs_handlers"][0].init_scale == 1024.0
    runtime.prepare_accelerator(args(adaptive_precision="off"))
    assert "kwargs_handlers" not in seen[-1]
    assert runtime.prepare_dtype(args(adaptive_precision="off"))[0] == torch.float16


def test_native_does_not_install_islands_or_cast_fp32_adapters(monkeypatch):
    from library.training import model_loading
    from library.training.adaptive_runtime import training_precision
    from networks.lora_modules.lora import LoRAModule

    monkeypatch.setattr(training_precision, "install_training_precision",
                        lambda *a, **k: pytest.fail("installed islands"))
    model = torch.nn.Sequential(torch.nn.Linear(4, 4)).half()
    model_loading._install_adaptive_precision(model, args(), SimpleNamespace(num_processes=1), "anima")
    assert not hasattr(model, "_adaptive_training_precision")
    module = LoRAModule("native", model[0], lora_dim=2, alpha=2)
    assert all(p.dtype == torch.float32 for p in module.lora_down.parameters())
    assert all(p.dtype == torch.float32 for p in module.lora_up.parameters())


def test_schema_and_web_options_accept_native_fp16():
    from pathlib import Path
    from library.config.schema import populate_schema, validate_entry
    from web.services.config.preflight_compat import _adaptive_precision_requires_runtime_resolution

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    populate_schema(parser)
    assert validate_entry("adaptive_precision", "native_fp16")[1] == "native_fp16"
    assert not _adaptive_precision_requires_runtime_resolution({
        "adaptive_precision": "auto", "adaptive_resolved_mode": "native_fp16"})
    assert _adaptive_precision_requires_runtime_resolution({"adaptive_precision": "auto"})
    options = Path("web/frontend-next/src/features/training-config/domain/labels-options.js").read_text()
    assert "adaptive_precision: ['off', 'auto', 'native_fp16', 'fp16_fp32']" in options


def test_native_resume_preserves_loaded_scaler_state(tmp_path):
    scaler = SimpleNamespace(scale=1024.0)
    accelerator = SimpleNamespace(
        scaler=scaler, load_state=lambda path: setattr(scaler, "scale", 64.0))
    runtime.resume_from_local_or_hf_if_specified(
        accelerator, args(resume=str(tmp_path), resume_from_huggingface=False))
    assert scaler.scale == 64.0


def test_no_cuda_auto_preflight_defers_hardware_contract(monkeypatch):
    from web.services.config.preflight_compat import _adaptive_precision_requires_runtime_resolution

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    configured = args(adaptive_precision="auto", mixed_precision="fp16")
    assert _adaptive_precision_requires_runtime_resolution(vars(configured))
    assert resolve_adaptive_precision(configured) == "native_fp16"
    assert configured.attn_mode == "torch"
