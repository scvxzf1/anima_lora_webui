from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from library.inference import models as inference_models


class _TinyModel(torch.nn.Module):
    pass


def _base_args(adapter_path):
    return SimpleNamespace(
        dit="base.safetensors",
        attn_mode="torch",
        lora_weight=[str(adapter_path)] if not isinstance(adapter_path, list) else [str(path) for path in adapter_path],
        lora_multiplier=None,
        pgraft=False,
        pooled_text_proj=None,
        compile=False,
        compile_blocks=False,
    )


def _write_adapter(path, tensors, metadata):
    save_file(tensors, path, metadata=metadata)
    return path


def test_adapter_capability_classifies_static_merge_plugins(tmp_path):
    cases = [
        (
            "dora",
            {"lora_unet_blocks_0_q_proj.dora_scale": torch.ones(6)},
            {"ss_network_spec": "dora"},
            "DoRA",
        ),
        (
            "glora",
            {"lora_unet_blocks_0_q_proj.glora_A": torch.ones(2, 4)},
            {"ss_network_spec": "glora", "ss_network_dim": "2"},
            "GLoRA",
        ),
        (
            "loha",
            {"lora_unet_blocks_0_q_proj.hada_w1_a": torch.ones(4, 2)},
            {"ss_network_spec": "loha", "ss_network_dim": "2"},
            "LoHa",
        ),
        (
            "lokr",
            {"lora_unet_blocks_0_q_proj.lokr_w1": torch.ones(2, 2)},
            {"ss_network_spec": "lokr", "ss_network_dim": "2"},
            "LoKr",
        ),
        (
            "vera",
            {"lora_unet_blocks_0_q_proj.vera_lambda_b": torch.ones(4)},
            {"ss_network_spec": "vera", "ss_network_dim": "2"},
            "VeRA",
        ),
    ]

    for stem, tensors, metadata, expected_kind in cases:
        path = _write_adapter(tmp_path / f"{stem}.safetensors", tensors, metadata)

        capability = inference_models._classify_adapter_capability(str(path))

        assert capability.kind == expected_kind
        assert capability.supports_static_merge is True
        assert capability.requires_dynamic_hook is False


def test_load_dit_model_routes_lokr_to_network_merge(tmp_path, monkeypatch):
    adapter_path = tmp_path / "adapter_lokr.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_q_proj.lokr_w1": torch.ones(2, 2),
            "lora_unet_blocks_0_q_proj.lokr_w2": torch.ones(3, 2),
            "lora_unet_blocks_0_q_proj.alpha": torch.tensor(32.0),
        },
        adapter_path,
        metadata={"ss_network_spec": "lokr", "ss_network_dim": "32"},
    )

    captured: dict[str, object] = {}

    def fake_load_anima_model(*args, **kwargs):
        captured["lora_weights_list"] = kwargs["lora_weights_list"]
        captured["lora_multipliers"] = kwargs["lora_multipliers"]
        return _TinyModel()

    class FakeNetwork:
        def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
            captured["merged_weights"] = weights_sd
            captured["merge_dtype"] = dtype
            captured["merge_device"] = device

    def fake_create_network_from_weights(**kwargs):
        captured["create_kwargs"] = kwargs
        return FakeNetwork(), kwargs["weights_sd"]

    import networks.lora_anima as lora_anima

    monkeypatch.setattr(inference_models.anima_utils, "load_anima_model", fake_load_anima_model)
    monkeypatch.setattr(lora_anima, "create_network_from_weights", fake_create_network_from_weights)
    monkeypatch.setattr(inference_models, "clean_memory_on_device", lambda device: None)

    model = inference_models.load_dit_model(
        _base_args(adapter_path),
        torch.device("cpu"),
        dit_weight_dtype=torch.float32,
    )

    assert isinstance(model, _TinyModel)
    assert captured["lora_weights_list"] is None
    assert captured["lora_multipliers"] is None
    assert "lora_unet_blocks_0_q_proj.lokr_w1" in captured["merged_weights"]
    assert captured["create_kwargs"]["file"] == str(adapter_path)
    assert captured["create_kwargs"]["metadata"]["ss_network_spec"] == "lokr"


def test_load_dit_model_pgraft_lokr_uses_dynamic_hooks_without_network_merge(
    tmp_path, monkeypatch
):
    adapter_path = tmp_path / "adapter_lokr.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_q_proj.lokr_w1": torch.ones(2, 2),
            "lora_unet_blocks_0_q_proj.lokr_w2": torch.ones(3, 2),
            "lora_unet_blocks_0_q_proj.alpha": torch.tensor(32.0),
        },
        adapter_path,
        metadata={"ss_network_spec": "lokr", "ss_network_dim": "32"},
    )

    captured: dict[str, object] = {}

    def fake_load_anima_model(*args, **kwargs):
        captured["lora_weights_list"] = kwargs["lora_weights_list"]
        captured["lora_multipliers"] = kwargs["lora_multipliers"]
        return _TinyModel()

    class FakeNetwork(torch.nn.Module):
        def apply_to(self, text_encoders, unet, *, apply_text_encoder, apply_unet):
            captured["applied"] = (apply_text_encoder, apply_unet)

        def load_state_dict(self, weights_sd, strict=False):
            captured["loaded_weights"] = weights_sd
            captured["load_strict"] = strict
            return SimpleNamespace(unexpected_keys=[])

        def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
            raise AssertionError("P-GRAFT must not statically merge adapter weights")

    def fake_create_network_from_weights(**kwargs):
        captured["create_kwargs"] = kwargs
        return FakeNetwork(), kwargs["weights_sd"]

    import networks.lora_anima as lora_anima

    monkeypatch.setattr(inference_models.anima_utils, "load_anima_model", fake_load_anima_model)
    monkeypatch.setattr(lora_anima, "create_network_from_weights", fake_create_network_from_weights)
    monkeypatch.setattr(inference_models, "clean_memory_on_device", lambda device: None)

    args = _base_args(adapter_path)
    args.pgraft = True
    model = inference_models.load_dit_model(
        args,
        torch.device("cpu"),
        dit_weight_dtype=torch.float32,
    )

    assert isinstance(model, _TinyModel)
    assert captured["lora_weights_list"] is None
    assert captured["lora_multipliers"] is None
    assert captured["create_kwargs"]["file"] == str(adapter_path)
    assert captured["create_kwargs"]["metadata"]["ss_network_spec"] == "lokr"
    assert "lora_unet_blocks_0_q_proj.lokr_w1" in captured["loaded_weights"]
    assert captured["applied"] == (False, True)
    assert model._pgraft_network is not None


def test_load_dit_model_preserves_multiplier_index_for_mixed_static_merges(
    tmp_path, monkeypatch
):
    lora_path = _write_adapter(
        tmp_path / "adapter_lora.safetensors",
        {
            "lora_unet_blocks_0_q_proj.lora_down.weight": torch.ones(2, 4),
            "lora_unet_blocks_0_q_proj.lora_up.weight": torch.ones(6, 2),
            "lora_unet_blocks_0_q_proj.alpha": torch.tensor(2.0),
        },
        {"ss_network_spec": "lora"},
    )
    lokr_path = _write_adapter(
        tmp_path / "adapter_lokr.safetensors",
        {
            "lora_unet_blocks_0_q_proj.lokr_w1": torch.ones(2, 2),
            "lora_unet_blocks_0_q_proj.lokr_w2": torch.ones(3, 2),
            "lora_unet_blocks_0_q_proj.alpha": torch.tensor(32.0),
        },
        {"ss_network_spec": "lokr", "ss_network_dim": "32"},
    )

    captured: dict[str, object] = {}

    def fake_load_anima_model(*args, **kwargs):
        captured["lora_weights_list"] = kwargs["lora_weights_list"]
        captured["lora_multipliers"] = kwargs["lora_multipliers"]
        return _TinyModel()

    class FakeNetwork:
        def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
            captured["merged_weights"] = weights_sd

    def fake_create_network_from_weights(**kwargs):
        captured["create_kwargs"] = kwargs
        return FakeNetwork(), kwargs["weights_sd"]

    import networks.lora_anima as lora_anima

    monkeypatch.setattr(inference_models.anima_utils, "load_anima_model", fake_load_anima_model)
    monkeypatch.setattr(lora_anima, "create_network_from_weights", fake_create_network_from_weights)
    monkeypatch.setattr(inference_models, "clean_memory_on_device", lambda device: None)

    args = _base_args([lora_path, lokr_path])
    args.lora_multiplier = [0.3, 0.7]
    model = inference_models.load_dit_model(
        args,
        torch.device("cpu"),
        dit_weight_dtype=torch.float32,
    )

    assert isinstance(model, _TinyModel)
    assert len(captured["lora_weights_list"]) == 1
    assert captured["lora_multipliers"] == [0.3]
    assert captured["create_kwargs"]["multiplier"] == 0.7
    assert "lora_unet_blocks_0_q_proj.lokr_w1" in captured["merged_weights"]


def test_load_dit_model_keeps_plain_lora_on_legacy_static_merge(tmp_path, monkeypatch):
    adapter_path = tmp_path / "adapter_lora.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_q_proj.lora_down.weight": torch.ones(2, 4),
            "lora_unet_blocks_0_q_proj.lora_up.weight": torch.ones(6, 2),
            "lora_unet_blocks_0_q_proj.alpha": torch.tensor(2.0),
        },
        adapter_path,
        metadata={"ss_network_spec": "lora"},
    )

    captured: dict[str, object] = {}

    def fake_load_anima_model(*args, **kwargs):
        captured["lora_weights_list"] = kwargs["lora_weights_list"]
        captured["lora_multipliers"] = kwargs["lora_multipliers"]
        return _TinyModel()

    def fail_create_network_from_weights(**kwargs):
        raise AssertionError("plain LoRA should not use network merge")

    import networks.lora_anima as lora_anima

    monkeypatch.setattr(inference_models.anima_utils, "load_anima_model", fake_load_anima_model)
    monkeypatch.setattr(lora_anima, "create_network_from_weights", fail_create_network_from_weights)
    monkeypatch.setattr(inference_models, "clean_memory_on_device", lambda device: None)

    model = inference_models.load_dit_model(
        _base_args(adapter_path),
        torch.device("cpu"),
        dit_weight_dtype=torch.float32,
    )

    assert isinstance(model, _TinyModel)
    assert len(captured["lora_weights_list"]) == 1
    assert captured["lora_multipliers"] == [1.0]
