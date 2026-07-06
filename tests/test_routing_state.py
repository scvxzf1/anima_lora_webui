"""Routing-state extraction invariants for LoRANetwork."""

from __future__ import annotations

import torch

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import ContentRouter, GlobalRouter, LoRANetwork
from networks.lora_modules import HydraLoRAModule
from tests.test_chimera_router_stats import _make_minimal_chimera_network


def _make_minimal_per_layer_fei_network() -> LoRANetwork:
    cfg = LoRANetworkCfg(
        num_experts=3,
        use_moe_style="shared_A",
        route_per_layer=True,
        router_source="fei",
        fei_feature_dim=2,
        lora_dim=4,
        alpha=4.0,
    )
    net = LoRANetwork.__new__(LoRANetwork)
    torch.nn.Module.__init__(net)
    net.cfg = cfg
    net.unet_loras = []
    net.text_encoder_loras = []
    net.text_encoder_refts = []
    net.unet_refts = []
    net._last_sigma = None
    net.use_fei_router = True
    net.global_router = None
    net.freq_router = None
    net.content_router = None

    for i in range(2):
        org = torch.nn.Linear(8, 8, bias=False)
        mod = HydraLoRAModule(
            lora_name=f"m{i}",
            org_module=org,
            lora_dim=cfg.lora_dim,
            alpha=cfg.alpha,
            num_experts=cfg.num_experts,
            fei_feature_dim=cfg.fei_feature_dim,
        )
        net.add_module(f"lora_m{i}", mod)
        net.unet_loras.append(mod)

    net._wire_shared_sigma_buffers()
    net._wire_shared_fei_buffers()
    net._wire_shared_routing_buffers()
    return net


def test_set_fei_recovers_aliasing_and_clear_fei_zeroes_live_buffer() -> None:
    net = _make_minimal_per_layer_fei_network()
    for lora in net._fei_aware_loras:
        lora._buffers["_fei"] = lora._buffers["_fei"].clone()
    pre_canonical = net._fei_aware_loras[0]._buffers["_fei"]
    assert net._shared_fei[2] is not pre_canonical

    fei = torch.tensor([[0.25, 0.75]])
    net.set_fei(fei)

    canonical = net._fei_aware_loras[0]._buffers["_fei"]
    assert net._shared_fei[2] is canonical
    for lora in net._fei_aware_loras:
        assert lora._buffers["_fei"] is canonical
        assert torch.allclose(lora._fei, fei)

    net.clear_fei()
    assert net._shared_fei[2] is canonical
    assert torch.allclose(canonical, torch.zeros_like(canonical))


def test_chimera_routing_weight_broadcasts_keep_live_gate_tensors() -> None:
    net = _make_minimal_chimera_network(K_c=3, K_f=3)
    net._wire_shared_content_routing_buffers()

    freq_gates = torch.softmax(torch.randn(2, 3, requires_grad=True), dim=-1)
    content_gates = torch.softmax(torch.randn(2, 3, requires_grad=True), dim=-1)
    net.set_freq_routing_weights(freq_gates)
    net.set_content_routing_weights(content_gates)

    freq_canonical = net._chimera_aware_loras[0]._buffers["_freq_routing_weights"]
    content_canonical = net._content_aware_loras[0]._buffers[
        "_content_routing_weights"
    ]
    assert freq_canonical.requires_grad
    assert freq_canonical.grad_fn is not None
    assert content_canonical.requires_grad
    assert content_canonical.grad_fn is not None
    for lora in net._chimera_aware_loras:
        assert lora._buffers["_freq_routing_weights"] is freq_canonical
    for lora in net._content_aware_loras:
        assert lora._buffers["_content_routing_weights"] is content_canonical


def test_chimera_freq_router_receives_gradient_from_forward() -> None:
    torch.manual_seed(0)
    net = _make_minimal_chimera_network(K_c=3, K_f=3)
    with torch.no_grad():
        net.freq_router.net[-1].weight.normal_(std=0.1)
        net.freq_router.net[-1].bias.normal_(std=0.1)
        for lora in net.unet_loras:
            lora.lambda_f.fill_(1.0)

    net.set_sigma(torch.tensor([0.2, 0.8]))
    net.set_fei(torch.randn(2, 2))
    loss = net.unet_loras[0](torch.randn(2, 8)).pow(2).sum()
    loss.backward()

    grad = net.freq_router.net[-1].weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_chimera_content_router_receives_gradient_from_forward() -> None:
    torch.manual_seed(0)
    net = _make_minimal_chimera_network(K_c=3, K_f=3)
    for lora in net.unet_loras:
        lora.use_global_content_router = True
    net._wire_shared_content_routing_buffers()
    net.content_router = ContentRouter(
        input_dim=8,
        num_content_experts=3,
        hidden_dim=8,
        tau=1.0,
        init_std=0.1,
        apply_layer_norm=False,
    )
    net.add_module("content_router", net.content_router)
    net.use_content_router = True
    with torch.no_grad():
        net.content_router.net[-1].weight.normal_(std=0.1)
        net.content_router.net[-1].bias.normal_(std=0.1)
        for lora in net.unet_loras:
            lora.lambda_c.fill_(1.0)

    net.set_content(torch.randn(2, 8))
    loss = net.unet_loras[0](torch.randn(2, 8)).pow(2).sum()
    loss.backward()

    grad = net.content_router.net[-1].weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_clear_step_caches_clears_all_router_transients() -> None:
    net = _make_minimal_chimera_network(K_c=3, K_f=3)
    net.global_router = GlobalRouter(input_dim=2, num_experts=3, hidden_dim=8)
    net.content_router = ContentRouter(
        input_dim=8,
        num_content_experts=3,
        hidden_dim=8,
        apply_layer_norm=False,
    )
    net._last_sigma = torch.ones(1)
    net._router_stats_cache = {"stale": 1.0}
    net._chimera_router_stats_cache = {"stale": 1.0}
    for lora in net.unet_loras:
        lora._last_gate = torch.ones(1, 6)
    for router in (net.global_router, net.freq_router, net.content_router):
        router._last_gates = torch.ones(1, 3)
        router._last_input = torch.ones(1, 2)
    net.global_router._last_fei = torch.ones(1, 2)

    net.clear_step_caches()

    assert net._last_sigma is None
    assert net._router_stats_cache is None
    assert net._chimera_router_stats_cache is None
    assert all(lora._last_gate is None for lora in net.unet_loras)
    assert net.global_router._last_gates is None
    assert net.global_router._last_input is None
    assert net.global_router._last_fei is None
    assert net.freq_router._last_gates is None
    assert net.freq_router._last_input is None
    assert net.content_router._last_gates is None
    assert net.content_router._last_input is None
