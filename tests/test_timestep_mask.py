"""T-LoRA Phase-1 invariants: schedule, shared buffer, train-only, metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training.forward.router_conditioning import apply_router_conditioning
from library.training.metadata import finalize_metadata
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.persistence import stamp_lora_save_metadata
from networks.lora_anima.routing_state import clear_timestep_mask, set_timestep_mask
from networks.lora_modules.lora import LoRAModule
from networks.lora_modules.ortho import OrthoLoRAModule
from networks.registry import NETWORK_REGISTRY


def _cfg(
    *,
    use_timestep_mask: bool = True,
    min_rank: int = 1,
    alpha_rank_scale: float = 1.0,
    network_dim: int = 4,
) -> LoRANetworkCfg:
    return LoRANetworkCfg.from_kwargs(
        {
            "use_timestep_mask": use_timestep_mask,
            "min_rank": min_rank,
            "alpha_rank_scale": alpha_rank_scale,
        },
        network_dim=network_dim,
        network_alpha=network_dim,
        neuron_dropout=None,
        module_class=LoRAModule,
    )


def _fake_network(cfg: LoRANetworkCfg, *, n_modules: int = 2):
    net = SimpleNamespace()
    net.cfg = cfg
    net.text_encoder_loras = []
    net.unet_loras = []
    net.text_encoder_refts = []
    net.unet_refts = []
    for i in range(n_modules):
        base = torch.nn.Linear(8, 8, bias=False)
        mod = LoRAModule(f"m{i}", base, lora_dim=cfg.lora_dim, alpha=cfg.lora_dim)
        net.unet_loras.append(mod)
    return net


def test_set_timestep_mask_schedule_noise_to_clean():
    cfg = _cfg(min_rank=2, network_dim=4, alpha_rank_scale=1.0)
    net = _fake_network(cfg)

    set_timestep_mask(net, torch.tensor([0.0]))  # pure noise → full rank
    assert torch.equal(net._shared_timestep_mask, torch.ones(1, 4))

    set_timestep_mask(net, torch.tensor([1.0]))  # clean → min_rank
    assert torch.equal(net._shared_timestep_mask, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))

    set_timestep_mask(net, torch.tensor([0.5]))
    assert torch.equal(net._shared_timestep_mask, torch.tensor([[1.0, 1.0, 1.0, 0.0]]))


def test_set_timestep_mask_shared_buffer_identity():
    cfg = _cfg(min_rank=1, network_dim=4)
    net = _fake_network(cfg, n_modules=3)
    set_timestep_mask(net, torch.tensor([0.25, 0.75]))
    shared = net._shared_timestep_mask
    assert shared is not None
    for mod in net.unet_loras:
        assert mod._timestep_mask is shared


def test_clear_timestep_mask_fills_ones():
    cfg = _cfg(min_rank=1, network_dim=4)
    net = _fake_network(cfg)
    set_timestep_mask(net, torch.tensor([1.0]))
    assert float(net._shared_timestep_mask.sum()) < 4.0
    clear_timestep_mask(net)
    assert torch.equal(net._shared_timestep_mask, torch.ones(1, 4))


def test_set_timestep_mask_noop_when_disabled():
    cfg = _cfg(use_timestep_mask=False, network_dim=4)
    net = _fake_network(cfg)
    set_timestep_mask(net, torch.tensor([0.0]))
    assert getattr(net, "_shared_timestep_mask", None) is None
    # modules keep default ones
    assert torch.equal(net.unet_loras[0]._timestep_mask, torch.ones(1, 4))


def test_stamp_tlora_metadata_when_enabled():
    cfg = _cfg(min_rank=8, alpha_rank_scale=1.25, network_dim=32)
    metadata: dict[str, str] = {}
    stamp_lora_save_metadata(metadata, cfg, NETWORK_REGISTRY["lora"])
    assert metadata["ss_use_timestep_mask"] == "true"
    assert metadata["ss_min_rank"] == "8"
    assert metadata["ss_alpha_rank_scale"] == "1.25"


def test_stamp_tlora_metadata_absent_when_disabled():
    cfg = _cfg(use_timestep_mask=False, network_dim=16)
    metadata: dict[str, str] = {}
    stamp_lora_save_metadata(metadata, cfg, NETWORK_REGISTRY["lora"])
    assert "ss_use_timestep_mask" not in metadata
    assert "ss_min_rank" not in metadata
    assert "ss_alpha_rank_scale" not in metadata


def test_finalize_metadata_always_writes_network_args():
    metadata, minimum = finalize_metadata(
        {"ss_network_module": "networks.lora_anima"},
        net_kwargs={"use_timestep_mask": "True", "min_rank": "8"},
    )
    assert "ss_network_args" in metadata
    assert "use_timestep_mask" in metadata["ss_network_args"]
    assert minimum["ss_network_args"] == metadata["ss_network_args"]


def test_router_conditioning_sets_mask_only_when_training():
    calls: list[str] = []

    class _Net:
        def set_timestep_mask(self, timesteps, max_timestep=1.0):
            calls.append(("set", float(timesteps.mean().item()), float(max_timestep)))

        def set_reft_timestep_mask(self, timesteps, max_timestep=1.0):
            calls.append(("set_reft", float(max_timestep)))

        def clear_timestep_mask(self):
            calls.append(("clear",))

    net = _Net()
    apply_router_conditioning(
        network=net,
        noisy_model_input=torch.zeros(1, 4, 4, 4),
        timesteps=torch.tensor([0.3]),
        is_train=True,
        warmup_step=0,
        max_train_steps=10,
    )
    assert calls[0][0] == "set"
    assert calls[0][1] == pytest.approx(0.3)
    assert calls[0][2] == 1.0
    assert calls[1] == ("set_reft", 1.0)

    calls.clear()
    apply_router_conditioning(
        network=net,
        noisy_model_input=torch.zeros(1, 4, 4, 4),
        timesteps=torch.tensor([0.3]),
        is_train=False,
        warmup_step=0,
        max_train_steps=10,
    )
    assert calls == [("clear",)]


def test_ortho_eval_ignores_stale_timestep_mask():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 8, bias=False)
    ortho = OrthoLoRAModule("lora_unet_test", base, multiplier=1.0, lora_dim=4, alpha=4)
    ortho.apply_to()
    with torch.no_grad():
        ortho.lambda_layer.fill_(1.0)
        ortho.S_p.normal_(0.0, 0.2)
        ortho.S_q.normal_(0.0, 0.2)
        ortho._timestep_mask.zero_()

    x = torch.ones(2, 8)
    ortho.train()
    train_out = base(x)
    ortho.eval()
    eval_out = base(x)

    # Zero mask kills the adapter contribution while training.
    with torch.no_grad():
        # Recompute base-only path: org_forward is stored on the module.
        base_only = ortho.org_forward(x)
    torch.testing.assert_close(train_out, base_only, atol=1e-5, rtol=1e-5)
    # Eval must ignore the stale zero mask and still apply the adapter.
    assert (eval_out - base_only).abs().sum().item() > 1e-3


def test_plain_lora_eval_ignores_stale_timestep_mask():
    torch.manual_seed(1)
    base = torch.nn.Linear(8, 4, bias=False)
    lora = LoRAModule("lora_unet_test", base, multiplier=1.0, lora_dim=4, alpha=4)
    lora.apply_to()
    with torch.no_grad():
        lora.lora_down.weight.normal_(0.0, 0.2)
        lora.lora_up.weight.normal_(0.0, 0.2)
        lora._timestep_mask.zero_()

    x = torch.ones(2, 8)
    lora.train()
    train_out = base(x)
    lora.eval()
    eval_out = base(x)

    base_only = lora.org_forward(x)
    torch.testing.assert_close(train_out, base_only, atol=1e-5, rtol=1e-5)
    assert (eval_out - base_only).abs().sum().item() > 1e-3


def test_sample_and_validation_paths_clear_timestep_mask():
    """Source contract: sample/validation must clear stale T-LoRA buffers."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sample_src = (root / "library/anima/training.py").read_text(encoding="utf-8")
    val_src = (root / "library/training/validation.py").read_text(encoding="utf-8")

    sample_fn = sample_src[
        sample_src.index("def sample_images(") : sample_src.index(
            "def _sample_image_inference("
        )
    ]
    assert "net.eval()" in sample_fn
    assert "clear_timestep_mask" in sample_fn

    val_fn = val_src[
        val_src.index("def run_validation(") : val_src.index("def _try_cmmd_validation(")
    ]
    assert ".eval()" in val_fn
    assert "clear_timestep_mask" in val_fn
