from __future__ import annotations

import argparse
from types import ModuleType
from types import SimpleNamespace

import library.training.bootstrap as bootstrap_mod
import train
from library.training.bootstrap import TrainingBootstrap


def test_trainer_uses_default_bootstrap():
    trainer = train.AnimaTrainer()

    assert isinstance(trainer.bootstrap, TrainingBootstrap)


def test_trainer_accepts_injected_bootstrap():
    class FakeBootstrap:
        def __init__(self) -> None:
            self.called = False

        def prepare_dataset(self, trainer, args):
            self.called = True
            return SimpleNamespace(
                train_dataset_group="train",
                val_dataset_group="val",
                current_epoch="epoch",
                current_step="step",
                collator="collator",
                use_user_config=True,
                use_dreambooth_method=False,
            )

    bootstrap = FakeBootstrap()
    trainer = train.AnimaTrainer(bootstrap=bootstrap)

    result = trainer._prepare_dataset(SimpleNamespace())

    assert bootstrap.called is True
    assert result == ("train", "val", "epoch", "step", "collator", True, False)


def test_bootstrap_batch_size_override_matches_trainer_wrapper():
    user_config = {
        "datasets": [
            {"batch_size": 1, "subsets": []},
            {"subsets": []},
        ]
    }

    train.AnimaTrainer._apply_train_batch_size_to_user_config(
        user_config,
        argparse.Namespace(train_batch_size=3),
    )

    assert [dataset["batch_size"] for dataset in user_config["datasets"]] == [3, 3]


def test_bootstrap_forwards_top_level_network_kwargs_with_cli_precedence():
    args = SimpleNamespace(
        network_args=["router_targets=cli_value"],
        router_targets="toml_value",
        dora_wd=True,
        use_lokr=True,
        gradient_accumulation_steps=4,
    )

    net_kwargs = TrainingBootstrap.build_net_kwargs(args)

    assert net_kwargs["router_targets"] == "cli_value"
    assert net_kwargs["dora_wd"] == "True"
    assert net_kwargs["use_lokr"] == "True"
    assert net_kwargs["gradient_accumulation_steps"] == "4"


def test_bootstrap_forwards_register_token_kwargs():
    args = SimpleNamespace(
        network_args=["register_init_std=0.11"],
        num_registers=8,
        register_insert_block=6,
        register_lr_scale=42,
        register_init_std=0.2,
    )

    net_kwargs = TrainingBootstrap.build_net_kwargs(args)

    assert net_kwargs["num_registers"] == "8"
    assert net_kwargs["register_insert_block"] == "6"
    assert net_kwargs["register_lr_scale"] == "42"
    assert net_kwargs["register_init_std"] == "0.11"


def test_bootstrap_auto_enables_lora_fp32_compute_on_v100_fp16(monkeypatch):
    seen = {}
    monkeypatch.setattr(bootstrap_mod.torch.cuda, "is_available", lambda: True)

    def fake_get_device_capability(device=None):
        seen["device"] = device
        return (7, 0)

    monkeypatch.setattr(
        bootstrap_mod.torch.cuda,
        "get_device_capability",
        fake_get_device_capability,
    )
    accelerator = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    args = SimpleNamespace(mixed_precision="fp16")

    enabled = TrainingBootstrap.should_auto_enable_lora_fp32_compute(
        args,
        accelerator,
        {},
    )

    assert enabled is True
    assert seen["device"] is accelerator.device


def test_bootstrap_does_not_auto_enable_lora_fp32_compute_when_user_set(monkeypatch):
    def fail_if_cuda_checked():
        raise AssertionError("explicit lora_fp32_compute should skip CUDA probing")

    monkeypatch.setattr(bootstrap_mod.torch.cuda, "is_available", fail_if_cuda_checked)
    args = SimpleNamespace(mixed_precision="fp16")

    enabled = TrainingBootstrap.should_auto_enable_lora_fp32_compute(
        args,
        SimpleNamespace(device=SimpleNamespace(type="cuda")),
        {"lora_fp32_compute": "false"},
    )

    assert enabled is False


def test_bootstrap_auto_lora_fp32_compute_fails_closed_on_capability_error(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(bootstrap_mod.torch.cuda, "is_available", lambda: True)

    def raise_capability_error(device=None):
        del device
        raise RuntimeError("cuda probe failed")

    monkeypatch.setattr(
        bootstrap_mod.torch.cuda,
        "get_device_capability",
        raise_capability_error,
    )
    args = SimpleNamespace(mixed_precision="fp16")

    enabled = TrainingBootstrap.should_auto_enable_lora_fp32_compute(
        args,
        SimpleNamespace(device=SimpleNamespace(type="cuda")),
        {},
    )

    assert enabled is False
    assert "could not read GPU compute capability" in caplog.text


def test_bootstrap_warns_register_tokens_with_block_swap(monkeypatch, caplog):
    class FakeNetwork:
        extra_seq_tokens = 4

        def apply_to(self, *args, **kwargs):
            return None

        def enable_gradient_checkpointing(self):
            return None

    fake_module = ModuleType("fake_register_network")
    fake_module.create_network = lambda *args, **kwargs: FakeNetwork()

    class FakeTrainer:
        def post_process_network(self, *args, **kwargs):
            return None

        def is_train_text_encoder(self, args):
            return False

        def get_text_encoders_train_flags(self, args, text_encoders):
            return []

    class FakeAccelerator:
        device = "cpu"

        def print(self, *args, **kwargs):
            return None

    class FakeUnet:
        def enable_gradient_checkpointing(self, cpu_offload=False):
            return None

    args = SimpleNamespace(
        network_module="fake_register_network",
        base_weights=None,
        base_weights_multiplier=None,
        dim_from_weights=False,
        network_weights=None,
        network_dropout=None,
        network_dim=4,
        network_alpha=4.0,
        scale_weight_norms=False,
        network_train_text_encoder_only=False,
        gradient_checkpointing=False,
        cpu_offload_checkpointing=False,
        torch_compile=False,
        blocks_to_swap=12,
        network_args=None,
    )

    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)
    bootstrap = TrainingBootstrap()
    result = bootstrap.create_and_apply_network(
        FakeTrainer(),
        args,
        FakeAccelerator(),
        vae=None,
        text_encoder=[],
        unet=FakeUnet(),
        text_encoders=[],
        weight_dtype=None,
    )

    assert result is not None
    assert "Register tokens + blocks_to_swap>0 is unaudited" in caplog.text


def test_bootstrap_widens_compile_seq_range_for_register_tokens(monkeypatch):
    captured = {}

    class FakeNetwork:
        extra_seq_tokens = 4

        def apply_to(self, *args, **kwargs):
            return None

        def enable_gradient_checkpointing(self):
            return None

    fake_module = ModuleType("fake_register_network")
    fake_module.create_network = lambda *args, **kwargs: FakeNetwork()

    class FakeTrainer:
        def post_process_network(self, *args, **kwargs):
            return None

        def is_train_text_encoder(self, args):
            return False

        def get_text_encoders_train_flags(self, args, text_encoders):
            return []

    class FakeAccelerator:
        device = "cpu"

        def print(self, *args, **kwargs):
            return None

    class FakeUnet:
        patch_spatial = 2
        vae_spatial_compression = 8

        def enable_gradient_checkpointing(self, cpu_offload=False):
            return None

    def fake_compile_blocks_for_training(unet, network, **kwargs):
        del unet, network
        captured.update(kwargs)

    import library.runtime.harness as harness

    args = SimpleNamespace(
        network_module="fake_register_network",
        base_weights=None,
        base_weights_multiplier=None,
        dim_from_weights=False,
        network_weights=None,
        network_dropout=None,
        network_dim=4,
        network_alpha=4.0,
        scale_weight_norms=False,
        network_train_text_encoder_only=False,
        gradient_checkpointing=False,
        cpu_offload_checkpointing=False,
        torch_compile=True,
        blocks_to_swap=0,
        network_args=None,
        dynamo_backend="eager",
        compile_inductor_mode=None,
        bucket_resolutions=[(896, 1152), (960, 1120)],
        compile_dynamic_seq=True,
        activation_memory_budget=1.0,
        partitioner_recompute_views=False,
        partitioner_aggressive_recomputation=False,
    )

    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)
    monkeypatch.setattr(
        harness, "compile_blocks_for_training", fake_compile_blocks_for_training
    )
    bootstrap = TrainingBootstrap()
    result = bootstrap.create_and_apply_network(
        FakeTrainer(),
        args,
        FakeAccelerator(),
        vae=None,
        text_encoder=[],
        unet=FakeUnet(),
        text_encoders=[],
        weight_dtype=None,
    )

    assert result is not None
    assert captured["n_token_families"] == 2
    assert captured["seq_range"] == (4032, 4204)


def test_bootstrap_compiles_after_apply_load_and_gradient_checkpointing(monkeypatch):
    events: list[str] = []

    class FakeNetwork:
        extra_seq_tokens = 0

        def apply_to(self, *args, **kwargs):
            del args, kwargs
            events.append("apply_to")

        def load_weights(self, path):
            events.append("load_weights")
            return {"path": path}

        def enable_gradient_checkpointing(self):
            events.append("network_grad_ckpt")

    fake_module = ModuleType("fake_compile_order_network")
    fake_module.create_network = lambda *args, **kwargs: FakeNetwork()

    class FakeTrainer:
        def post_process_network(self, *args, **kwargs):
            return None

        def is_train_text_encoder(self, args):
            return False

        def get_text_encoders_train_flags(self, args, text_encoders):
            return []

    class FakeAccelerator:
        device = "cpu"

        def print(self, *args, **kwargs):
            return None

    class FakeUnet:
        def enable_gradient_checkpointing(self, cpu_offload=False):
            del cpu_offload
            events.append("unet_grad_ckpt")

    def fake_compile_blocks_for_training(unet, network, **kwargs):
        del unet, network, kwargs
        events.append("compile")

    import library.runtime.harness as harness

    args = SimpleNamespace(
        network_module="fake_compile_order_network",
        base_weights=None,
        base_weights_multiplier=None,
        dim_from_weights=False,
        network_weights="fake-network.safetensors",
        network_dropout=None,
        network_dim=4,
        network_alpha=4.0,
        scale_weight_norms=False,
        network_train_text_encoder_only=False,
        gradient_checkpointing=True,
        cpu_offload_checkpointing=False,
        torch_compile=True,
        blocks_to_swap=0,
        network_args=None,
        dynamo_backend="eager",
        compile_inductor_mode=None,
        bucket_resolutions=None,
        compile_dynamic_seq=False,
        activation_memory_budget=1.0,
        partitioner_recompute_views=False,
        partitioner_aggressive_recomputation=False,
    )

    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)
    monkeypatch.setattr(
        harness, "compile_blocks_for_training", fake_compile_blocks_for_training
    )
    result = TrainingBootstrap().create_and_apply_network(
        FakeTrainer(),
        args,
        FakeAccelerator(),
        vae=None,
        text_encoder=[],
        unet=FakeUnet(),
        text_encoders=[],
        weight_dtype=None,
    )

    assert result is not None
    compile_idx = events.index("compile")
    assert compile_idx > events.index("apply_to")
    assert compile_idx > events.index("load_weights")
    assert compile_idx > events.index("unet_grad_ckpt")
    assert compile_idx > events.index("network_grad_ckpt")


def test_maybe_enable_fp32_residual_only_for_fp16_anima():
    calls: list[str] = []

    class DummyAnima:
        def enable_fp32_residual(self):
            calls.append("enable")

    class OtherUNet:
        def enable_fp32_residual(self):
            calls.append("other")

    assert (
        TrainingBootstrap.maybe_enable_fp32_residual(
            SimpleNamespace(mixed_precision="fp16"),
            DummyAnima(),
            anima_cls=DummyAnima,
        )
        is True
    )
    assert calls == ["enable"]

    assert (
        TrainingBootstrap.maybe_enable_fp32_residual(
            SimpleNamespace(mixed_precision="bf16"),
            DummyAnima(),
            anima_cls=DummyAnima,
        )
        is False
    )
    assert (
        TrainingBootstrap.maybe_enable_fp32_residual(
            SimpleNamespace(mixed_precision="fp16"),
            OtherUNet(),
            anima_cls=DummyAnima,
        )
        is False
    )
    assert calls == ["enable"]


def test_bootstrap_enables_fp32_residual_before_compile(monkeypatch):
    events: list[str] = []

    class FakeNetwork:
        extra_seq_tokens = 0

        def apply_to(self, *args, **kwargs):
            del args, kwargs
            events.append("apply_to")

        def load_weights(self, path):
            events.append("load_weights")
            return {"path": path}

        def enable_gradient_checkpointing(self):
            events.append("network_grad_ckpt")

    fake_module = ModuleType("fake_fp32_residual_network")
    fake_module.create_network = lambda *args, **kwargs: FakeNetwork()

    class FakeTrainer:
        def post_process_network(self, *args, **kwargs):
            return None

        def is_train_text_encoder(self, args):
            return False

        def get_text_encoders_train_flags(self, args, text_encoders):
            return []

    class FakeAccelerator:
        device = "cpu"

        def print(self, *args, **kwargs):
            return None

    class FakeUnet:
        def enable_gradient_checkpointing(self, cpu_offload=False):
            del cpu_offload
            events.append("unet_grad_ckpt")

        def enable_fp32_residual(self):
            events.append("enable_fp32_residual")

    def fake_compile_blocks_for_training(unet, network, **kwargs):
        del unet, network, kwargs
        events.append("compile")

    import library.runtime.harness as harness

    # Treat FakeUnet as Anima for the bootstrap type gate.
    monkeypatch.setattr(
        "library.anima.models.Anima",
        FakeUnet,
        raising=False,
    )
    # Also cover the helper's anima_cls path if bootstrap imports models module.
    import library.anima.models as anima_models

    monkeypatch.setattr(anima_models, "Anima", FakeUnet)

    args = SimpleNamespace(
        network_module="fake_fp32_residual_network",
        base_weights=None,
        base_weights_multiplier=None,
        dim_from_weights=False,
        network_weights="fake-network.safetensors",
        network_dropout=None,
        network_dim=4,
        network_alpha=4.0,
        scale_weight_norms=False,
        network_train_text_encoder_only=False,
        gradient_checkpointing=True,
        cpu_offload_checkpointing=False,
        torch_compile=True,
        mixed_precision="fp16",
        blocks_to_swap=0,
        network_args=None,
        dynamo_backend="eager",
        compile_inductor_mode=None,
        bucket_resolutions=None,
        compile_dynamic_seq=False,
        activation_memory_budget=1.0,
        partitioner_recompute_views=False,
        partitioner_aggressive_recomputation=False,
    )

    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)
    monkeypatch.setattr(
        harness, "compile_blocks_for_training", fake_compile_blocks_for_training
    )
    result = TrainingBootstrap().create_and_apply_network(
        FakeTrainer(),
        args,
        FakeAccelerator(),
        vae=None,
        text_encoder=[],
        unet=FakeUnet(),
        text_encoders=[],
        weight_dtype=None,
    )

    assert result is not None
    assert "enable_fp32_residual" in events
    enable_idx = events.index("enable_fp32_residual")
    compile_idx = events.index("compile")
    assert enable_idx > events.index("apply_to")
    assert enable_idx > events.index("load_weights")
    assert enable_idx > events.index("unet_grad_ckpt")
    assert enable_idx > events.index("network_grad_ckpt")
    assert enable_idx < compile_idx


def test_bootstrap_enables_fp32_residual_even_without_compile(monkeypatch):
    events: list[str] = []

    class FakeNetwork:
        extra_seq_tokens = 0

        def apply_to(self, *args, **kwargs):
            del args, kwargs
            events.append("apply_to")

        def enable_gradient_checkpointing(self):
            return None

    fake_module = ModuleType("fake_fp32_residual_no_compile_network")
    fake_module.create_network = lambda *args, **kwargs: FakeNetwork()

    class FakeTrainer:
        def post_process_network(self, *args, **kwargs):
            return None

        def is_train_text_encoder(self, args):
            return False

        def get_text_encoders_train_flags(self, args, text_encoders):
            return []

    class FakeAccelerator:
        device = "cpu"

        def print(self, *args, **kwargs):
            return None

    class FakeUnet:
        def enable_fp32_residual(self):
            events.append("enable_fp32_residual")

    import library.anima.models as anima_models

    monkeypatch.setattr(anima_models, "Anima", FakeUnet)
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)

    args = SimpleNamespace(
        network_module="fake_fp32_residual_no_compile_network",
        base_weights=None,
        base_weights_multiplier=None,
        dim_from_weights=False,
        network_weights=None,
        network_dropout=None,
        network_dim=4,
        network_alpha=4.0,
        scale_weight_norms=False,
        network_train_text_encoder_only=False,
        gradient_checkpointing=False,
        cpu_offload_checkpointing=False,
        torch_compile=False,
        mixed_precision="fp16",
        blocks_to_swap=0,
        network_args=None,
    )

    result = TrainingBootstrap().create_and_apply_network(
        FakeTrainer(),
        args,
        FakeAccelerator(),
        vae=None,
        text_encoder=[],
        unet=FakeUnet(),
        text_encoders=[],
        weight_dtype=None,
    )

    assert result is not None
    assert events == ["apply_to", "enable_fp32_residual"]
