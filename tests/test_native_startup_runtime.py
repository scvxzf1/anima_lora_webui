from types import SimpleNamespace as NS

import pytest
import torch

from library.training.adaptive_runtime.contract import (freeze_native_sources,
    precision_contract_id, precision_request, validate_native_sources)
from library.training.adaptive_runtime.native_runtime import NativeStartupRuntime
from library.training.adaptive_runtime.training_precision import validate_local_precision_contract
from library.training.adaptive_runtime.training_runtime import validate_entry
from networks.lora_modules.lora import LoRAModule


def args(**changes):
    values = dict(adaptive_precision="native_fp16", mixed_precision="fp16", model_family="anima",
                  network_dim=2, network_alpha=2, max_train_steps=2, blocks_to_swap=0)
    values.update(changes)
    return NS(**values)


def fixture(tmp_path):
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2).half() for _ in range(28)])
    for block in model.blocks:
        block.fp32_residual = True
        block.gradient_checkpointing = True
    model.final_layer = torch.nn.Linear(2, 2).half()
    model.final_layer.fp32_residual = True
    model.register_buffer("_mod_guidance_delta", torch.zeros(1, 2))
    model.requires_grad_(False)
    network = torch.nn.Module()
    network.unet_loras = torch.nn.ModuleList([LoRAModule("tiny", model.blocks[0], lora_dim=2)])
    network.text_encoder_loras = []
    scaler = NS(is_enabled=lambda: True, get_growth_factor=lambda: 2.,
                get_backoff_factor=lambda: .5, get_growth_interval=lambda: 2000,
                state_dict=lambda: dict(scale=1024., _growth_tracker=0, growth_factor=2.,
                                        backoff_factor=.5, growth_interval=2000))
    accelerator = NS(native_amp=True, mixed_precision="fp16", scaler=scaler, optimizer_step_was_skipped=False,
                     unwrap_model=lambda value: value)
    runtime = NativeStartupRuntime(tmp_path, args=args())
    runtime.capture_model(model)
    return runtime, model, network, accelerator


def prepared(tmp_path):
    runtime, model, network, accelerator = fixture(tmp_path)
    runtime.prepared(model, network, accelerator)
    return runtime, model, network, accelerator


def boundary(runtime, accelerator, step, skipped=False):
    state = NS(accelerator=accelerator, global_step=step, args=runtime.args)
    runtime.phase("optimizer")
    accelerator.optimizer_step_was_skipped = skipped
    runtime.after_optimizer(state)
    runtime.commit(state)
    return state


def test_scalar_boundaries_skip_sticky_finish(tmp_path, monkeypatch):
    runtime, model, network, accelerator = prepared(tmp_path)
    monkeypatch.setattr(torch.Tensor, "clone", lambda *_: pytest.fail("must not clone tensors"))
    state = boundary(runtime, accelerator, 1, skipped=True)
    assert runtime.optimizer_started and runtime.skipped_steps == 1
    with pytest.raises(RuntimeError, match="completing"):
        runtime.finish(state)
    state = boundary(runtime, accelerator, 2)
    runtime.finish(state)
    result = runtime.result("ok")
    assert result["finished"] and result["effective_steps"] == 1
    assert result["schema"] == "native_training_precision_v1"
    assert result["committed_checkpoint"] is None
    assert not result["data_cursor_resume_supported"]
    assert [p.name for p in tmp_path.iterdir()] == ["latest-state.json"]
    assert not hasattr(model, "_adaptive_training_precision")


def test_unprepared_cannot_train_or_succeed(tmp_path):
    runtime, *_ = fixture(tmp_path)
    for stage in ("forward", "backward", "optimizer"):
        with pytest.raises(RuntimeError, match="prepared"):
            runtime.phase(stage)
    with pytest.raises(RuntimeError, match="completion"):
        runtime.result("ok")


@pytest.mark.parametrize("change", ["dtype", "shape", "buffer", "guard", "flag", "amp", "scaler", "autocast"])
def test_execution_drift_rejected(tmp_path, change):
    runtime, model, network, accelerator = prepared(tmp_path)
    if change == "dtype":
        model.blocks[0].weight.data = model.blocks[0].weight.float()
    elif change == "shape":
        model.blocks[0].weight.data = model.blocks[0].weight[:1]
    elif change == "buffer":
        model._mod_guidance_delta = model._mod_guidance_delta.half()
    elif change == "guard":
        model.final_layer.fp32_residual = False
    elif change == "flag":
        network.unet_loras[0].fp32_compute = True
    elif change == "amp":
        accelerator.native_amp = False
    elif change == "scaler":
        accelerator.scaler.get_growth_factor = lambda: 3.
    else:
        accelerator.autocast_handler = NS(enabled=False)
    with pytest.raises(ValueError, match="precision_contract_mismatch"):
        runtime.validate_precision()


def test_dynamic_scaler_scale_is_allowed(tmp_path):
    runtime, _, _, accelerator = prepared(tmp_path)
    previous = accelerator.scaler.state_dict()
    previous.update(scale=512., _growth_tracker=12)
    accelerator.scaler.state_dict = lambda: previous
    runtime.validate_precision()


def test_guard_drift_is_rejected_before_optimizer_starts(tmp_path):
    runtime, model, _, _ = prepared(tmp_path)
    model.final_layer.fp32_residual = False
    with pytest.raises(ValueError, match="precision_contract_mismatch"):
        runtime.phase("optimizer")
    assert runtime.optimizer_started is False
    assert runtime.attempted_steps == 0


def test_policy_observation_never_reads_dynamic_scaler_state(tmp_path):
    runtime, _, _, accelerator = prepared(tmp_path)
    accelerator.scaler.state_dict = lambda: pytest.fail("state_dict synchronizes dynamic CUDA tensors")
    state = boundary(runtime, accelerator, 1)
    runtime.validate_precision()
    assert state.global_step == 1


@pytest.mark.parametrize("change", ["mixed_precision", "checkpoint", "accumulation"])
def test_actual_precision_checkpoint_and_accumulation_checked(tmp_path, change):
    runtime, model, network, accelerator = fixture(tmp_path)
    if change == "mixed_precision":
        accelerator.mixed_precision = "bf16"
    elif change == "checkpoint":
        model.blocks[0].gradient_checkpointing = False
    else:
        runtime.args.gradient_accumulation_steps = 2
    with pytest.raises(ValueError):
        runtime.prepared(model, network, accelerator)


def test_default_base_compute_survives_normal_bootstrap_normalization(tmp_path):
    from library.training.bootstrap import TrainingBootstrap
    config = args()
    identity = precision_contract_id(config)
    assert not TrainingBootstrap.maybe_apply_convrot_base(config, object(), unet=object())
    assert config.base_compute == "bf16"
    assert precision_contract_id(config) == identity


def test_source_change_during_loading_is_detected_before_execution(tmp_path):
    runtime, model, network, accelerator = fixture(tmp_path)
    source = tmp_path / "checkpoint.bin"
    source.write_bytes(b"original")
    runtime.args.pretrained_model_name_or_path = str(source)
    freeze_native_sources(runtime.args)
    source.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="source (changed during loading|contents changed)"):
        runtime.prepared(model, network, accelerator)


def test_extra_trainable_rejected(tmp_path):
    runtime, model, network, accelerator = fixture(tmp_path)
    network.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="FP32 LoRA"):
        runtime.prepared(model, network, accelerator)


def test_shape_count_guard_rejected(tmp_path):
    runtime, model, network, accelerator = fixture(tmp_path)
    model.blocks.append(torch.nn.Linear(2, 2).half())
    with pytest.raises(ValueError, match="precision_contract_mismatch"):
        runtime.prepared(model, network, accelerator)


def test_native_identity_freezes_numeric_not_swap(tmp_path):
    first = args(blocks_to_swap=0)
    baseline = precision_contract_id(first)
    assert precision_request(first)["schema"] == "native_precision_request_v1"
    assert precision_contract_id(args(blocks_to_swap=10)) == baseline
    for changes in ({"base_compute": "fp32"}, {"network_alpha": 3}, {"network_dim": 4}, {"network_dropout": .1},
                    {"block_swap_transfer_dtype": "fp32"}, {"network_args": ["lora_fp32_compute=true"]}):
        assert precision_contract_id(args(**changes)) != baseline
    runtime, model, network, accelerator = prepared(tmp_path)
    identity = runtime.precision_manifest_id
    runtime.args.blocks_to_swap = 12
    runtime.validate_precision()
    assert runtime.precision_manifest_id == identity


def test_native_source_content_and_path_identity(tmp_path):
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(b"one")
    config = args(pretrained_model_name_or_path=str(path))
    freeze_native_sources(config)
    baseline = precision_contract_id(config)
    validate_native_sources(config)
    path.write_bytes(b"two")
    with pytest.raises(ValueError, match="source contents"):
        validate_native_sources(config)
    # Frozen H stays stable; fresh supervisor freezing produces a new H.
    assert precision_contract_id(config) == baseline
    freeze_native_sources(config)
    assert precision_contract_id(config) != baseline


def test_cached_auto_native_contract_and_entry(tmp_path):
    config = args(adaptive_precision="auto", adaptive_resolved_mode="native_fp16",
                  _adaptive_training_worker={"directory": str(tmp_path)})
    validate_local_precision_contract(config, precision_contract_id(config))
    trainer = NS(_adaptive_training_runtime=NativeStartupRuntime(tmp_path, args=config))
    validate_entry(trainer, config)
    config.adaptive_resolved_mode = "fp16_fp32"
    with pytest.raises(ValueError, match="isolated"):
        validate_entry(trainer, config)


def test_load_hook_measures_before_offloader_and_no_islands(tmp_path, monkeypatch):
    from library.training.model_loading import _install_adaptive_precision
    runtime, model, _, accelerator = fixture(tmp_path)
    runtime._base = None
    measured = []
    runtime.validate_swap_capacity = lambda loaded, family: measured.append((loaded, family))
    trainer = NS(_adaptive_training_runtime=runtime)
    _install_adaptive_precision(model, runtime.args, accelerator, "anima", trainer=trainer)
    assert measured == [(model, "anima")]
    assert trainer._native_precision_model is model
    assert not hasattr(model, "_adaptive_training_precision")
    ordinary = torch.nn.Linear(2, 2)
    _install_adaptive_precision(ordinary, args(), accelerator, "anima", trainer=NS())
    assert ordinary.weight.dtype == torch.float32
