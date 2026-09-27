import argparse
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from library.runtime.accelerator import prepare_dtype, resume_from_local_or_hf_if_specified
from library.training.adaptive_runtime.training_config import (
    add_arguments, configuration_errors, require_training_contract,
)
from library.training.adaptive_runtime.precision import resolve_adaptive_precision
from library.training.adaptive_runtime.training_precision import (
    PRECISION_STATE, accelerator_handlers, install_training_precision,
    preserve_precision_cast, realized_precision_manifest_id, register_precision_checkpoint,
    validate_training_network,
)
from library.training.adaptive_runtime.training_runtime import TrainingRuntime


def args(**changes):
    return SimpleNamespace(**{**dict(adaptive_precision="fp16_fp32", mixed_precision="fp16",
                                    adaptive_fp32_modules=["2"], adaptive_loss_scale=1024.0,
                                    model_family="krea2_raw", save_precision="bf16"), **changes})


def base():
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4),
                               torch.nn.Linear(4, 2)).to(torch.bfloat16)


def test_cli_opt_in_and_dtype_defaults():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert parser.parse_args([]).adaptive_precision == "off"
    configured = parser.parse_args(["--adaptive_precision", "fp16_fp32",
                                    "--adaptive_fp32_modules", "blocks.0.*", "output"])
    assert configured.adaptive_fp32_modules == ["blocks.0.*", "output"]
    assert parser.parse_args(["--adaptive_precision", "auto"]).adaptive_precision == "auto"
    assert prepare_dtype(args()) == (torch.float32, torch.bfloat16)
    assert prepare_dtype(args(adaptive_precision="off")) == (torch.float16, torch.bfloat16)
    assert configuration_errors({}) == []


@pytest.mark.parametrize("change", [
    {"model_family": "unknown"}, {"mixed_precision": "bf16"},
    {"base_compute": "nf4"}, {"torch_compile": True}, {"full_fp16": True},
    {"auto_block_swap": True}, {"use_moe_style": "shared_A"}, {"network_args": ["x=y"]},
    {"attn_mode": "flash"}, {"sample_every_n_steps": 20}, {"network_train_unet_only": False},
    {"adaptive_loss_scale": float("nan")}, {"adaptive_loss_scale": 1},
    {"adaptive_fp32_modules": "wrong"}, {"block_swap_restore_mode": "slab"},
])
def test_unsupported_training_contract_fails_closed(change):
    with pytest.raises(ValueError):
        require_training_contract(args(**change))


def test_single_process_contract():
    with pytest.raises(ValueError, match="one process"):
        require_training_contract(args(), world_size=2)


@pytest.mark.parametrize("capability,mode,mixed,candidate", [
    ((8, 0), "bf16", "bf16", "bf16"),
    ((7, 5), "fp16_fp32", "fp16", "fp16"),
    ((6, 0), "fp16_fp32", "fp16", "fp16"),
    ((6, 1), "fp32", "no", "fp32"),
])
def test_auto_precision_uses_shared_compute_capability_policy(capability, mode, mixed, candidate):
    configured = SimpleNamespace(adaptive_precision="auto", mixed_precision="bf16")
    assert resolve_adaptive_precision(configured, get_capability=lambda: capability) == mode
    assert configured.adaptive_candidate == candidate
    assert configured.mixed_precision == mixed


def test_resolved_auto_does_not_probe_hardware_again(monkeypatch):
    configured = SimpleNamespace(
        adaptive_precision="auto", adaptive_resolved_mode="fp16_fp32",
        adaptive_candidate="fp16", mixed_precision="fp16", model_family="krea2_raw",
        attn_mode="torch",
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda: (_ for _ in ()).throw(AssertionError("re-probed")))
    assert resolve_adaptive_precision(configured) == "fp16_fp32"


def test_auto_pre_ampere_selects_native_attention_for_krea_and_z_image():
    for family in ("krea2_raw", "z_image"):
        configured = SimpleNamespace(
            adaptive_precision="auto", mixed_precision="bf16",
            model_family=family, attn_mode="flash",
        )
        assert resolve_adaptive_precision(configured, get_capability=lambda: (7, 5)) == "fp16_fp32"
        assert configured.attn_mode == "torch"


def test_auto_fp32_path_is_valid_for_z_image_on_older_architecture():
    from library.training.compat_matrix import check_training_compat

    configured = SimpleNamespace(
        adaptive_precision="auto", mixed_precision="bf16",
        model_family="z_image", attn_mode="flash",
    )
    assert resolve_adaptive_precision(configured, get_capability=lambda: (6, 1)) == "fp32"
    assert configured.mixed_precision == "no" and configured.attn_mode == "torch"
    result = check_training_compat(vars(configured))
    assert not any(issue.code == "z_image_bf16_only" for issue in result.errors)


@pytest.mark.parametrize(
    "family,capability,expected_mode,expected_mixed,expected_attention",
    [
        ("anima", (8, 0), "bf16", "bf16", "flash"),
        ("anima", (7, 5), "fp16_fp32", "fp16", "flash"),
        ("z_image", (8, 0), "bf16", "bf16", "flash"),
        ("z_image", (7, 5), "fp16_fp32", "fp16", "torch"),
        ("z_image", (6, 1), "fp32", "no", "torch"),
    ],
)
def test_auto_precision_is_validated_against_each_dit_family(
    family, capability, expected_mode, expected_mixed, expected_attention
):
    configured = args(
        model_family=family,
        adaptive_precision="auto",
        mixed_precision="bf16",
        attn_mode="flash",
    )
    assert resolve_adaptive_precision(configured, get_capability=lambda: capability) == expected_mode
    assert configured.mixed_precision == expected_mixed
    assert configured.attn_mode == expected_attention
    require_training_contract(configured)


@pytest.mark.parametrize("family,attn_mode", [
    ("anima", "flash"), ("krea2_raw", "torch"), ("z_image", "torch"),
])
def test_fp16_fp32_contract_accepts_all_registered_dit_families(family, attn_mode):
    require_training_contract(args(model_family=family, attn_mode=attn_mode))


def test_z_image_fp16_fp32_contract_rejects_flash():
    with pytest.raises(ValueError, match="Z-Image.*attn_mode"):
        require_training_contract(args(model_family="z_image", attn_mode="flash"))


@pytest.mark.parametrize("family", ["anima", "krea2_raw", "z_image"])
def test_shared_loader_precision_helper_stamps_family_manifest(monkeypatch, family):
    from library.training import model_loading

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    hooks = {}
    accelerator = SimpleNamespace(
        num_processes=1,
        register_save_state_pre_hook=lambda fn: hooks.update(save=fn),
        register_load_state_pre_hook=lambda fn: hooks.update(load=fn),
    )
    model_loading._install_adaptive_precision(
        model, args(model_family=family, adaptive_fp32_modules=[]), accelerator, family
    )
    assert model._adaptive_training_precision["model_family"] == family
    assert model[0].weight.dtype == torch.float16
    assert set(hooks) == {"save", "load"}


def test_z_image_training_loader_installs_islands_before_attention(monkeypatch):
    from library.training import model_loading

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def enable_gradient_checkpointing(self):
            self.gradient_checkpointing = True

    model = FakeModel()
    seen = {}
    monkeypatch.setattr(
        "library.models.z_image.weights.load_z_image_transformer",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "library.models.z_image.attention_backend.prepare_z_image_attention",
        lambda target, mode, *, dtype: seen.update(
            attention=(target, mode, dtype)
        ) or "torch",
    )
    monkeypatch.setattr(model_loading, "_maybe_probe_components", lambda *a, **k: None)
    hooks = {}
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=1,
        register_save_state_pre_hook=lambda fn: hooks.update(save=fn),
        register_load_state_pre_hook=lambda fn: hooks.update(load=fn),
    )
    config = args(
        model_family="z_image",
        pretrained_model_name_or_path="z-image",
        gradient_checkpointing=True,
        blocks_to_swap=0,
        attn_mode="torch",
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="foreach",
        adaptive_fp32_modules=[],
    )
    trainer = SimpleNamespace(is_swapping_blocks=False, peak_probe=None)
    loaded, _ = model_loading._load_z_image_dit(
        trainer, config, torch.float32, accelerator, [None]
    )
    assert loaded is model and loaded.proj.weight.dtype == torch.float16
    assert seen["attention"] == (model, "torch", torch.float32)
    assert set(hooks) == {"save", "load"}


def test_anima_training_loader_installs_islands_before_block_swap(monkeypatch):
    from library.anima import checkpoint as anima_checkpoint
    from library.training import model_loading

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(4, 4)
            self.output = torch.nn.Linear(4, 2)

    model = FakeModel().to(torch.bfloat16)
    events = []

    def fake_load(device, path, attn_mode, loading_device, dit_weight_dtype, **kwargs):
        assert attn_mode == "torch"
        assert dit_weight_dtype == torch.float32
        events.append("load")
        return model

    def fake_swap(*args, **kwargs):
        assert hasattr(model, "_adaptive_training_precision")
        assert model.input.weight.dtype == torch.float16
        assert model.output.weight.dtype == torch.float32
        events.append("swap")

    model.enable_block_swap = fake_swap
    monkeypatch.setattr(model_loading.anima_utils, "load_anima_model", fake_load)
    monkeypatch.setattr(
        anima_checkpoint,
        "inspect_anima_checkpoint",
        lambda path: SimpleNamespace(num_blocks=2, variant="test"),
    )
    monkeypatch.setattr(anima_checkpoint, "apply_layout_to_args", lambda *args: None)
    monkeypatch.setattr(
        "library.anima.compat.require_training_compatibility",
        lambda *args: "test",
    )
    monkeypatch.setattr(model_loading, "resolve_v100_flash_stability", lambda args: "off")
    monkeypatch.setattr(model_loading, "resolve_debug_finite_checks", lambda args, stability: False)
    monkeypatch.setattr(model_loading, "resolve_block_swap_profile_jsonl", lambda args: None)
    monkeypatch.setattr(model_loading, "_maybe_probe_components", lambda *args, **kwargs: None)

    hooks = {}
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=1,
        register_save_state_pre_hook=lambda fn: hooks.update(save=fn),
        register_load_state_pre_hook=lambda fn: hooks.update(load=fn),
    )
    config = args(
        model_family="anima",
        pretrained_model_name_or_path="anima",
        adaptive_fp32_modules=["output"],
        xformers=False,
        attn_mode="torch",
        blocks_to_swap=1,
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="foreach",
        unsloth_offload_checkpointing=False,
    )
    trainer = SimpleNamespace(is_swapping_blocks=False, peak_probe=None)

    loaded, _ = model_loading._load_anima_dit(
        trainer, config, torch.float32, accelerator, [None]
    )

    assert loaded is model
    assert trainer.is_swapping_blocks is True
    assert events == ["load", "swap"]
    assert set(hooks) == {"save", "load"}


def test_fp32_residual_and_explicit_sensitive_linear():
    model = base()
    manifest = install_training_precision(model, args())
    assert manifest["assignments"] == {"0": "fp16", "2": "fp32"}
    assert model[0].weight.dtype == torch.float16
    assert all(p.dtype == torch.float32 for p in model[1:].parameters())
    value = torch.randn(2, 4, requires_grad=True)
    result = model(value)
    result.square().mean().backward()
    assert result.dtype == value.grad.dtype == torch.float32
    assert preserve_precision_cast(model, torch.float32) is None
    model[0].float()
    with pytest.raises(ValueError, match="precision drift"):
        preserve_precision_cast(model, torch.float32)
    assert preserve_precision_cast(base(), torch.bfloat16) == torch.bfloat16


def test_realized_dtype_manifest_is_stable_and_detects_drift():
    model = base()
    install_training_precision(model, args())
    names = model._adaptive_precision_manifest_names
    expected = model._adaptive_precision_manifest_id
    assert realized_precision_manifest_id(model, names) == expected
    for _ in range(3):
        assert realized_precision_manifest_id(model, names) == expected
    model[0].float()
    assert realized_precision_manifest_id(model, names) != expected


def test_first_runtime_check_compares_installed_manifest(tmp_path):
    model = base()
    install_training_precision(model, args())
    model[1].half()
    runtime = TrainingRuntime(tmp_path)
    with pytest.raises(ValueError, match="installed dtype manifest drifted"):
        runtime.validate_precision(model)
    assert runtime.precision_manifest_id is None


def test_bad_pattern_or_alias_rejected_before_install():
    model = base()
    with pytest.raises(ValueError, match="Unknown FP32"):
        install_training_precision(model, args(adaptive_fp32_modules=["missing"]))
    assert model[0].weight.dtype == torch.bfloat16 and model[0].weight.requires_grad
    model[2] = model[0]
    with pytest.raises(ValueError, match="Aliased"):
        install_training_precision(model, args())


def test_accelerate_handlers_preserve_scaler_while_disabling_global_autocast():
    autocast, scaler = accelerator_handlers(args())
    assert autocast.enabled is False
    assert scaler.init_scale == 1024.0
    # Standard dynamic loss scaling remains owned by Accelerate, not the probe scaler.
    assert scaler.growth_interval == 2000


def test_actual_adapter_inventory_must_be_plain_fp32_lora():
    from networks.lora_modules.lora import LoRAModule

    model = base()
    install_training_precision(model, args())
    network = torch.nn.Module()
    network.unet_loras = torch.nn.ModuleList([LoRAModule("test", model[0], lora_dim=2, alpha=2)])
    validate_training_network(args(), network)
    network.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="only FP32 LoRA"):
        validate_training_network(args(), network)
    del network.extra
    network.half()
    with pytest.raises(ValueError, match="only FP32 LoRA"):
        validate_training_network(args(), network)
    network.unet_loras = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    with pytest.raises(ValueError, match="plain DiT LoRA"):
        validate_training_network(args(), network)


def test_shared_compat_surface_reports_experimental_contract():
    from library.training.compat_matrix import check_training_compat

    result = check_training_compat(vars(args(torch_compile=True)))
    assert any(issue.code == "adaptive_precision_contract" for issue in result.errors)


def test_checkpoint_contract_rejects_changes_before_load(tmp_path):
    hooks = {}
    accelerator = SimpleNamespace(
        is_main_process=True,
        register_save_state_pre_hook=lambda fn: hooks.update(save=fn),
        register_load_state_pre_hook=lambda fn: hooks.update(load=fn))
    manifest = install_training_precision(base(), args())
    register_precision_checkpoint(accelerator, manifest)
    with pytest.raises(ValueError, match="missing or changed"):
        hooks["load"]([], tmp_path)
    hooks["save"]([], [], tmp_path)
    hooks["load"]([], tmp_path)
    original = deepcopy(manifest)
    manifest["assignments"]["2"] = "fp16"
    hooks["save"]([], [], tmp_path)
    assert json.loads((tmp_path / PRECISION_STATE).read_text()) == original
    (tmp_path / PRECISION_STATE).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing or changed"):
        hooks["load"]([], tmp_path)


def test_precision_checkpoint_cannot_silently_resume_in_old_mode(tmp_path):
    (tmp_path / PRECISION_STATE).write_text("{}")
    accelerator = SimpleNamespace(load_state=lambda path: pytest.fail("must refuse before loading"))
    with pytest.raises(ValueError, match="explicit FP16/FP32"):
        resume_from_local_or_hf_if_specified(
            accelerator, args(adaptive_precision="off", resume=str(tmp_path), resume_from_huggingface=False))


def test_krea_nf4_refused_before_weight_load(monkeypatch):
    from library.models.krea2_raw import quantize, weights
    from library.training.model_loading import _load_krea2_dit

    monkeypatch.setattr(quantize, "inspect_nf4_checkpoint", lambda path: SimpleNamespace(is_nf4=True))
    monkeypatch.setattr(weights, "load_krea2_dit", lambda *a, **k: pytest.fail("must not load NF4"))
    with pytest.raises(ValueError, match="not NF4"):
        _load_krea2_dit(None, args(pretrained_model_name_or_path="nf4"), torch.float32,
                        SimpleNamespace(num_processes=1), [])


def test_training_loader_installs_islands_before_offloader(monkeypatch):
    from library.models.krea2_raw import attention_backend, quantize, weights
    from library.training import model_loading

    model = base()
    events = []

    def load(*a, **kwargs):
        assert kwargs["dtype"] == torch.bfloat16
        events.append("load")
        return model

    def prepare(target, mode, *, dtype, compile_enabled):
        assert target is model and dtype == torch.float32
        assert model[0].weight.dtype == torch.float16 and model[2].weight.dtype == torch.float32
        events.append("attention")
        return mode

    def swap(*a, **kwargs):
        assert hasattr(model, "_adaptive_training_precision")
        assert model[0]._adaptive_dtype == torch.float16
        events.append("swap")

    model.enable_block_swap = swap
    monkeypatch.setattr(quantize, "inspect_nf4_checkpoint", lambda path: SimpleNamespace(is_nf4=False))
    monkeypatch.setattr(weights, "load_krea2_dit", load)
    monkeypatch.setattr(attention_backend, "prepare_krea2_attention", prepare)
    monkeypatch.setattr(model_loading, "_maybe_probe_components", lambda *a, **k: None)
    monkeypatch.setattr(model_loading, "resolve_block_swap_profile_jsonl", lambda a: None)
    accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"), is_main_process=True,
                                  register_save_state_pre_hook=lambda fn: None,
                                  register_load_state_pre_hook=lambda fn: None)
    configured = args(pretrained_model_name_or_path="base", blocks_to_swap=2,
                      attn_mode="torch", block_swap_restore_mode="foreach",
                      block_swap_transfer_dtype="bf16", unsloth_offload_checkpointing=False)
    result, _ = model_loading._load_krea2_dit(
        SimpleNamespace(is_swapping_blocks=True, peak_probe=None), configured,
        torch.float32, accelerator, [])
    assert result is model and events == ["load", "attention", "swap"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Accelerate precision bridge")
def test_real_accelerate_scaler_clip_and_resume(tmp_path):
    from library.runtime.accelerator import prepare_accelerator
    from networks.lora_modules.lora import LoRAModule

    torch.manual_seed(20260921)
    configured = args(logging_dir=None, log_with=None, gradient_accumulation_steps=1)
    accelerator = prepare_accelerator(configured)
    assert accelerator.native_amp is True and accelerator.scaler.is_enabled()
    model = base()
    manifest = install_training_precision(model, configured)
    network = LoRAModule("training_precision", model[0], lora_dim=2, alpha=2)
    network.apply_to()
    model.cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    network, optimizer, scheduler = accelerator.prepare(network, optimizer, scheduler)
    register_precision_checkpoint(accelerator, manifest)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with accelerator.accumulate(network), accelerator.autocast():
            assert not torch.is_autocast_enabled("cuda")
            loss = model(torch.randn(2, 4, device="cuda")).square().mean()
            assert loss.dtype == torch.float32
            accelerator.backward(loss)
            parameters = list(network.parameters())
            scale = accelerator.scaler.get_scale()
            expected_norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad.detach().float() / scale)
                for p in parameters if p.grad is not None]))
            norm = accelerator.clip_grad_norm_(parameters, 1.0)
            torch.testing.assert_close(norm, expected_norm)
            optimizer.step()
            scheduler.step()
        assert not accelerator.optimizer_step_was_skipped
        assert model[0].weight.dtype == torch.float16 and model[2].weight.dtype == torch.float32

    step()
    state_dir = tmp_path / "state"
    accelerator.save_state(str(state_dir))
    assert (state_dir / PRECISION_STATE).is_file()
    assert accelerator.scaler.state_dict()["_growth_tracker"] == 1
    step()
    expected = deepcopy(accelerator.unwrap_model(network).state_dict())
    expected_optimizer = deepcopy(optimizer.state_dict())
    expected_scheduler = deepcopy(scheduler.state_dict())
    expected_scaler = deepcopy(accelerator.scaler.state_dict())
    accelerator.load_state(str(state_dir))
    assert accelerator.scaler.state_dict()["_growth_tracker"] == 1
    step()
    torch.testing.assert_close(accelerator.unwrap_model(network).state_dict(), expected, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)
    assert scheduler.state_dict() == expected_scheduler
    assert accelerator.scaler.state_dict() == expected_scaler
    accelerator.end_training()
