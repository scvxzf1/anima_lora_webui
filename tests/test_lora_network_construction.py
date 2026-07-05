from __future__ import annotations

import torch

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.factory import create_network
from networks.lora_anima.network import GlobalRouter, LoRANetwork
from networks.lora_anima.targeting import (
    collect_lora_target_candidates,
    compile_lora_target_patterns,
)
from networks.lora_modules import HydraLoRAModule, LoRAModule, StackedExpertsLoRAModule


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 8, bias=False)


class TinyDiT(torch.nn.Module):
    def __init__(self, block_count: int = 1) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(block_count)])


def _plain_cfg() -> LoRANetworkCfg:
    return LoRANetworkCfg(
        module_class=LoRAModule,
        lora_dim=2,
        alpha=2.0,
    )


def test_lora_network_builds_plain_modules_with_stable_names() -> None:
    net = LoRANetwork([], TinyDiT(), _plain_cfg())

    assert [lora.lora_name for lora in net.unet_loras] == [
        "lora_unet_blocks_0_q_proj",
        "lora_unet_blocks_0_k_proj",
    ]
    assert [lora.original_name for lora in net.unet_loras] == [
        "blocks.0.q_proj",
        "blocks.0.k_proj",
    ]
    assert all(type(lora) is LoRAModule for lora in net.unet_loras)
    assert net.text_encoder_loras == []
    assert net.global_router is None


def test_include_patterns_can_override_exclude_patterns() -> None:
    excluded = LoRANetworkCfg(
        module_class=LoRAModule,
        lora_dim=2,
        alpha=2.0,
        exclude_patterns=[r"blocks\.0\.q_proj"],
    )
    restored = LoRANetworkCfg(
        module_class=LoRAModule,
        lora_dim=2,
        alpha=2.0,
        exclude_patterns=[r"blocks\.0\.q_proj"],
        include_patterns=[r"blocks\.0\.q_proj"],
    )

    excluded_net = LoRANetwork([], TinyDiT(), excluded)
    restored_net = LoRANetwork([], TinyDiT(), restored)

    assert [lora.original_name for lora in excluded_net.unet_loras] == [
        "blocks.0.k_proj",
    ]
    assert [lora.original_name for lora in restored_net.unet_loras] == [
        "blocks.0.q_proj",
        "blocks.0.k_proj",
    ]


def test_layer_range_filters_only_matching_block_paths() -> None:
    cfg = LoRANetworkCfg(
        module_class=LoRAModule,
        lora_dim=2,
        alpha=2.0,
        layer_start=1,
        layer_end=2,
    )

    net = LoRANetwork([], TinyDiT(block_count=3), cfg)

    assert [lora.original_name for lora in net.unet_loras] == [
        "blocks.1.q_proj",
        "blocks.1.k_proj",
    ]


def test_collect_lora_target_candidates_keeps_warm_start_order_and_skips() -> None:
    candidates = collect_lora_target_candidates(
        root_module=TinyDiT(block_count=2),
        prefix="lora_unet",
        target_replace_modules=["Block"],
        exclude_patterns=compile_lora_target_patterns(
            [r"blocks\.0\.q_proj", r"blocks\.1\.k_proj"]
        ),
        include_patterns=compile_lora_target_patterns([r"blocks\.0\.q_proj"]),
        is_unet=True,
        layer_start=None,
        layer_end=None,
        modules_dim={
            "lora_unet_blocks_0_q_proj": 4,
            "lora_unet_blocks_0_k_proj": 0,
            "lora_unet_blocks_1_q_proj": 8,
        },
        modules_alpha={
            "lora_unet_blocks_0_q_proj": 2.0,
            "lora_unet_blocks_0_k_proj": 0.0,
            "lora_unet_blocks_1_q_proj": 4.0,
        },
        reg_dims=None,
        default_dim=None,
        lora_dim=2,
        alpha=2.0,
    )

    assert [item.original_name for item in candidates] == [
        "blocks.0.q_proj",
        "blocks.0.k_proj",
        "blocks.1.q_proj",
    ]
    assert [(item.dim, item.alpha, item.skipped) for item in candidates] == [
        (4, 2.0, False),
        (None, None, True),
        (8, 4.0, False),
    ]


def test_router_targets_mix_hydra_and_plain_lora_modules() -> None:
    cfg = LoRANetworkCfg(
        module_class=HydraLoRAModule,
        lora_dim=2,
        alpha=2.0,
        num_experts=3,
        use_moe_style="shared_A",
        route_per_layer=True,
        router_source="sigma",
        router_targets="q_proj",
        sigma_feature_dim=4,
    )

    net = LoRANetwork([], TinyDiT(), cfg)
    modules = {lora.original_name: lora for lora in net.unet_loras}

    assert type(modules["blocks.0.q_proj"]) is HydraLoRAModule
    assert type(modules["blocks.0.k_proj"]) is LoRAModule
    assert net._hydra_router_hits == 1
    assert net._hydra_router_misses == 1
    assert net._sigma_router_hits == 1
    assert net.global_router is None


def test_create_network_global_fei_shared_a_cell_uses_real_builder_path() -> None:
    net = create_network(
        multiplier=1.0,
        network_dim=2,
        network_alpha=2.0,
        vae=None,
        text_encoders=[],
        unet=TinyDiT(),
        use_moe_style="shared_A",
        route_per_layer=False,
        router_source="fei",
        router_targets="q_proj",
        num_experts=3,
        fei_feature_dim=2,
        router_hidden_dim=8,
        router_tau=0.7,
        balance_loss_weight=0.2,
        balance_loss_warmup_ratio=0.25,
    )
    modules = {lora.original_name: lora for lora in net.unet_loras}

    assert type(modules["blocks.0.q_proj"]) is HydraLoRAModule
    assert modules["blocks.0.q_proj"].use_global_router is True
    assert type(modules["blocks.0.k_proj"]) is LoRAModule
    assert isinstance(net.global_router, GlobalRouter)
    assert len(net._routing_aware_loras) == 1
    assert net._hydra_router_hits == 1
    assert net._hydra_router_misses == 1
    assert net._global_router_hits == 1
    assert net._fei_router_hits == 0
    assert net.use_fei_router is True
    assert net.use_sigma_router is False
    assert net._network_spec.name == "hydra"
    assert net._use_hydra is True
    assert net._balance_loss_target_weight == 0.2
    assert net._balance_loss_warmup_ratio == 0.25
    assert net._balance_loss_weight == 0.0


def test_global_fei_cell_builds_network_router_from_real_init() -> None:
    cfg = LoRANetworkCfg(
        module_class=StackedExpertsLoRAModule,
        lora_dim=2,
        alpha=2.0,
        num_experts=3,
        use_moe_style="independent_A",
        route_per_layer=False,
        router_source="fei",
        fei_feature_dim=2,
        router_hidden_dim=8,
        router_tau=0.7,
    )

    net = LoRANetwork([], TinyDiT(), cfg)

    assert all(type(lora) is StackedExpertsLoRAModule for lora in net.unet_loras)
    assert isinstance(net.global_router, GlobalRouter)
    assert len(net._routing_aware_loras) == 2
    assert net._shared_routing_weights is net.unet_loras[0]._buffers["_routing_weights"]
    assert net.use_fei_router is True
    assert net.use_sigma_router is False
