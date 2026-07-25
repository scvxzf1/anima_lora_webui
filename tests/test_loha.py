from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from library.inference import models as inference_models
from networks.lora_anima.factory import create_network_from_weights
from networks.plugins.loha.autograd import loha_linear, make_hada_weight
from networks.plugins.loha.module import LoHaModule
from networks.plugins.loha.save import defuse_loha_qkv, save_loha_weights


def test_loha_initial_forward_is_zero_delta():
    base = torch.nn.Linear(4, 3, bias=False)
    loha = LoHaModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    loha.apply_to()

    x = torch.randn(2, 5, 4)
    base_out = F.linear(x, loha.org_module_ref[0].weight)

    torch.testing.assert_close(loha.org_module_ref[0](x), base_out)


def test_loha_forward_matches_hadamard_weight():
    base = torch.nn.Linear(4, 3, bias=False)
    loha = LoHaModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    loha.apply_to()

    with torch.no_grad():
        loha.org_module_ref[0].weight.zero_()
        loha.hada_w1_a.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        loha.hada_w1_b.copy_(torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]]))
        loha.hada_w2_a.copy_(torch.tensor([[1.5, 0.5], [0.25, 1.25], [2.0, 0.75]]))
        loha.hada_w2_b.copy_(torch.tensor([[1.0, 0.5, 0.25, 0.75], [1.5, 2.0, 2.5, 3.0]]))

    x = torch.randn(2, 5, 4)
    weight = (loha.hada_w1_a @ loha.hada_w1_b) * (loha.hada_w2_a @ loha.hada_w2_b)

    torch.testing.assert_close(loha.org_module_ref[0](x), F.linear(x, weight))


def test_loha_eval_forward_ignores_stale_timestep_mask():
    base = torch.nn.Linear(4, 3, bias=False)
    loha = LoHaModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    loha.apply_to()

    with torch.no_grad():
        loha.org_module_ref[0].weight.zero_()
        loha.hada_w1_a.fill_(1.0)
        loha.hada_w1_b.fill_(1.0)
        loha.hada_w2_a.fill_(1.0)
        loha.hada_w2_b.fill_(1.0)
        loha._timestep_mask.zero_()

    x = torch.ones(1, 4)

    loha.train()
    train_out = loha.org_module_ref[0](x)

    loha.eval()
    eval_out = loha.org_module_ref[0](x)

    torch.testing.assert_close(train_out, torch.zeros_like(train_out))
    assert torch.count_nonzero(eval_out).item() == eval_out.numel()


def test_loha_autograd_matches_materialized_weight_grad():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 4, requires_grad=True)
    w1_a = torch.randn(3, 2, requires_grad=True)
    w1_b = torch.randn(2, 4, requires_grad=True)
    w2_a = torch.randn(3, 2, requires_grad=True)
    w2_b = torch.randn(2, 4, requires_grad=True)
    scale = 0.5

    y_ref = F.linear(x.float(), make_hada_weight(w1_a, w1_b, w2_a, w2_b)) * scale
    y_ref.sum().backward()
    grads_ref = (
        x.grad.detach().clone(),
        w1_a.grad.detach().clone(),
        w1_b.grad.detach().clone(),
        w2_a.grad.detach().clone(),
        w2_b.grad.detach().clone(),
    )

    for tensor in (x, w1_a, w1_b, w2_a, w2_b):
        tensor.grad = None

    y = loha_linear(x, w1_a, w1_b, w2_a, w2_b, scale=scale)
    y.sum().backward()
    grads = (x.grad, w1_a.grad, w1_b.grad, w2_a.grad, w2_b.grad)

    torch.testing.assert_close(y, y_ref.to(dtype=y.dtype), rtol=1e-5, atol=1e-5)
    for got, expected in zip(grads, grads_ref):
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_loha_merge_to_matches_forward_delta():
    base = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) * 0.01)

    loha = LoHaModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    with torch.no_grad():
        loha.hada_w1_a.copy_(torch.tensor([[1.0, 0.5], [0.25, 1.5], [2.0, 0.75]]))
        loha.hada_w1_b.copy_(torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]]))
        loha.hada_w2_a.copy_(torch.tensor([[1.5, 0.5], [0.25, 1.25], [2.0, 0.75]]))
        loha.hada_w2_b.copy_(torch.tensor([[1.0, 0.5, 0.25, 0.75], [1.5, 2.0, 2.5, 3.0]]))

    x = torch.randn(2, 4)
    expected = base(x) + F.linear(x, loha.get_weight())

    sd = {
        "hada_w1_a": loha.hada_w1_a.detach().clone(),
        "hada_w1_b": loha.hada_w1_b.detach().clone(),
        "hada_w2_a": loha.hada_w2_a.detach().clone(),
        "hada_w2_b": loha.hada_w2_b.detach().clone(),
    }
    base2 = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        base2.weight.copy_(base.weight)
    loha2 = LoHaModule(
        "lora_unet_test",
        base2,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    loha2.merge_to(sd, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(base2(x), expected)


def test_loha_fuse_unfuse_roundtrip():
    base = torch.nn.Linear(4, 3, bias=False)
    loha = LoHaModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    loha.apply_to()
    with torch.no_grad():
        loha.hada_w1_a.fill_(0.5)
        loha.hada_w1_b.fill_(0.25)
        loha.hada_w2_a.fill_(0.75)
        loha.hada_w2_b.fill_(0.125)

    x = torch.randn(2, 4)
    before = base(x).detach().clone()
    weight_before = base.weight.detach().clone()
    loha.fuse_weight()
    fused = base(x).detach().clone()
    # Fuse bakes ΔW into the base Linear; forward then skips the adapter path
    # but the numerical output must match the unfused adapter forward.
    torch.testing.assert_close(before, fused)
    assert not torch.allclose(weight_before, base.weight)
    loha.unfuse_weight()
    after = base(x).detach().clone()
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(weight_before, base.weight)


def test_defuse_loha_qkv_splits_fused_prefix():
    r, in_dim, out_dim = 2, 4, 3
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    state = {
        f"{prefix}.hada_w1_a": torch.randn(3 * out_dim, r),
        f"{prefix}.hada_w1_b": torch.randn(r, in_dim),
        f"{prefix}.hada_w2_a": torch.randn(3 * out_dim, r),
        f"{prefix}.hada_w2_b": torch.randn(r, in_dim),
        f"{prefix}.alpha": torch.tensor(float(r)),
    }

    defuse_loha_qkv(state)

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert state[f"{base}_{suffix}.hada_w1_a"].shape == (out_dim, r)
        assert state[f"{base}_{suffix}.hada_w1_b"].shape == (r, in_dim)
        assert state[f"{base}_{suffix}.hada_w2_a"].shape == (out_dim, r)
        assert state[f"{base}_{suffix}.hada_w2_b"].shape == (r, in_dim)
        assert f"{base}_{suffix}.alpha" in state
    assert f"{prefix}.hada_w1_a" not in state


def test_create_network_from_loha_weights_loads_and_runs():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 6, bias=False)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    weights_sd = {
        "lora_unet_blocks_0_q_proj.hada_w1_a": torch.randn(6, 2),
        "lora_unet_blocks_0_q_proj.hada_w1_b": torch.randn(2, 4),
        "lora_unet_blocks_0_q_proj.hada_w2_a": torch.randn(6, 2),
        "lora_unet_blocks_0_q_proj.hada_w2_b": torch.randn(2, 4),
        "lora_unet_blocks_0_q_proj.alpha": torch.tensor(2.0),
    }

    unet = TinyUnet()
    network, weights = create_network_from_weights(
        multiplier=1.0,
        file="",
        ae=None,
        text_encoders=[],
        unet=unet,
        weights_sd=weights_sd,
        metadata={"ss_network_spec": "loha", "ss_network_dim": "2"},
    )

    assert len(network.unet_loras) == 1
    loha = network.unet_loras[0]
    assert isinstance(loha, LoHaModule)
    assert loha.lora_dim == 2
    network.apply_to([], unet, False, True)
    info = network.load_state_dict(weights, strict=False)
    assert info.missing_keys == []
    assert info.unexpected_keys == []

    x = torch.randn(2, 4)
    y = unet.blocks[0].q_proj(x)
    assert y.shape == (2, 6)


def test_save_loha_weights_writes_metadata_and_keys(tmp_path: Path):
    r, in_dim, out_dim = 2, 4, 3
    prefix = "lora_unet_blocks_0_self_attn_q_proj"
    state = {
        f"{prefix}.hada_w1_a": torch.randn(out_dim, r),
        f"{prefix}.hada_w1_b": torch.randn(r, in_dim),
        f"{prefix}.hada_w2_a": torch.randn(out_dim, r),
        f"{prefix}.hada_w2_b": torch.randn(r, in_dim),
        f"{prefix}.alpha": torch.tensor(float(r)),
    }
    out = tmp_path / "loha.safetensors"
    metadata = {"ss_network_spec": "loha", "ss_network_dim": "2"}
    assert save_loha_weights(state, str(out), dtype=torch.float32, metadata=metadata)
    loaded = load_file(str(out))
    assert f"{prefix}.hada_w1_a" in loaded
    assert f"{prefix}.hada_w2_b" in loaded


def test_load_dit_model_routes_loha_to_network_merge(tmp_path, monkeypatch):
    adapter_path = tmp_path / "adapter_loha.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_q_proj.hada_w1_a": torch.ones(6, 2),
            "lora_unet_blocks_0_q_proj.hada_w1_b": torch.ones(2, 4),
            "lora_unet_blocks_0_q_proj.hada_w2_a": torch.ones(6, 2),
            "lora_unet_blocks_0_q_proj.hada_w2_b": torch.ones(2, 4),
            "lora_unet_blocks_0_q_proj.alpha": torch.tensor(2.0),
        },
        adapter_path,
        metadata={"ss_network_spec": "loha", "ss_network_dim": "2"},
    )

    captured: dict[str, object] = {}

    class _TinyModel(torch.nn.Module):
        pass

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

    args = SimpleNamespace(
        dit="base.safetensors",
        attn_mode="torch",
        lora_weight=[str(adapter_path)],
        lora_multiplier=None,
        pgraft=False,
        pooled_text_proj=None,
        compile=False,
        compile_blocks=False,
    )
    model = inference_models.load_dit_model(
        args,
        torch.device("cpu"),
        dit_weight_dtype=torch.float32,
    )

    assert isinstance(model, _TinyModel)
    assert captured["lora_weights_list"] is None
    assert captured["lora_multipliers"] is None
    assert "lora_unet_blocks_0_q_proj.hada_w1_a" in captured["merged_weights"]
    assert captured["create_kwargs"]["metadata"]["ss_network_spec"] == "loha"
