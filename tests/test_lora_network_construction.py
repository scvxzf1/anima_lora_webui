from __future__ import annotations

from inspect import signature

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.factory import create_network, create_network_from_weights
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


def test_lora_network_public_api_surface_stays_compatible() -> None:
    expected_methods = {
        "apply_to",
        "load_weights",
        "save_weights",
        "merge_to",
        "is_mergeable",
        "prepare_optimizer_params_with_multiple_te_lrs",
        "set_multiplier",
        "set_timestep_mask",
        "clear_timestep_mask",
        "set_sigma",
        "clear_sigma",
        "set_fei",
        "clear_fei",
        "set_routing_weights",
        "clear_routing_weights",
        "set_crossattn_routing",
        "set_content",
        "clear_step_caches",
        "get_balance_loss",
        "get_router_stats",
        "get_chimera_router_stats",
        "metrics",
    }

    missing = [
        name for name in sorted(expected_methods)
        if not callable(getattr(LoRANetwork, name, None))
    ]

    assert missing == []
    assert list(signature(LoRANetwork.apply_to).parameters) == [
        "self",
        "text_encoders",
        "unet",
        "apply_text_encoder",
        "apply_unet",
    ]
    assert list(signature(LoRANetwork.load_weights).parameters) == ["self", "file"]
    assert list(signature(LoRANetwork.save_weights).parameters) == [
        "self",
        "file",
        "dtype",
        "metadata",
    ]
    optimizer_signature = signature(
        LoRANetwork.prepare_optimizer_params_with_multiple_te_lrs
    )
    assert list(optimizer_signature.parameters) == [
        "self",
        "text_encoder_lr",
        "unet_lr",
        "default_lr",
    ]
    assert list(signature(LoRANetwork.set_sigma).parameters) == ["self", "sigmas"]
    assert list(signature(LoRANetwork.set_fei).parameters) == ["self", "fei"]


def test_load_weights_facade_strips_orig_mod_keys(tmp_path) -> None:
    unet = TinyDiT()
    net = LoRANetwork([], unet, _plain_cfg())
    net.apply_to([], unet)
    target_key = "lora_unet_blocks_0_q_proj.lora_down.weight"
    compiled_key = target_key.replace(
        "lora_unet_blocks",
        "lora_unet__orig_mod_blocks",
        1,
    )
    expected = torch.full_like(net.state_dict()[target_key], 0.25)
    out = tmp_path / "compiled_keys.safetensors"
    save_file({compiled_key: expected}, str(out))

    info = net.load_weights(str(out))

    assert info.unexpected_keys == []
    assert torch.equal(net.state_dict()[target_key], expected)


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


def test_create_network_from_weights_restores_mixed_hydra_plain_router_names() -> None:
    hydra_name = "lora_unet_blocks_0_q_proj"
    plain_name = "lora_unet_blocks_0_k_proj"
    weights_sd = {
        f"{hydra_name}.lora_down.weight": torch.randn(2, 8),
        f"{hydra_name}.lora_up_weight": torch.randn(3, 8, 2),
        f"{hydra_name}.router.weight": torch.randn(3, 2),
        f"{hydra_name}.alpha": torch.tensor(2.0),
        f"{plain_name}.lora_down.weight": torch.randn(2, 8),
        f"{plain_name}.lora_up.weight": torch.randn(8, 2),
        f"{plain_name}.alpha": torch.tensor(2.0),
    }

    net, _ = create_network_from_weights(
        multiplier=1.0,
        file=None,
        ae=None,
        text_encoders=[],
        unet=TinyDiT(),
        weights_sd=weights_sd,
        for_inference=True,
        metadata={
            "ss_use_moe_style": "shared_A",
            "ss_route_per_layer": "True",
            "ss_router_source": "input",
        },
    )
    modules = {lora.original_name: lora for lora in net.unet_loras}

    assert net.cfg.use_moe_style == "shared_A"
    assert net.cfg.route_per_layer is True
    assert net.cfg.router_source == "input"
    assert list(net.cfg.hydra_router_names or []) == [hydra_name]
    assert net._hydra_router_names == {hydra_name}
    assert type(modules["blocks.0.q_proj"]) is HydraLoRAModule
    assert type(modules["blocks.0.k_proj"]) is LoRAModule
    assert net._hydra_router_hits == 1
    assert net._hydra_router_misses == 1
    assert net._network_spec.name == "hydra"
    assert net._use_hydra is True


def test_create_network_from_weights_recovers_fei_router_names_from_metadata_widths() -> None:
    hydra_name = "lora_unet_blocks_0_q_proj"
    rank = 2
    fei_dim = 3
    weights_sd = {
        f"{hydra_name}.lora_down.weight": torch.randn(rank, 8),
        f"{hydra_name}.lora_up_weight": torch.randn(3, 8, rank),
        f"{hydra_name}.router.weight": torch.randn(3, rank + fei_dim),
        f"{hydra_name}.alpha": torch.tensor(float(rank)),
    }

    net, _ = create_network_from_weights(
        multiplier=1.0,
        file=None,
        ae=None,
        text_encoders=[],
        unet=TinyDiT(),
        weights_sd=weights_sd,
        for_inference=True,
        metadata={
            "ss_use_moe_style": "shared_A",
            "ss_route_per_layer": "True",
            "ss_router_source": "fei",
            "ss_fei_feature_dim": str(fei_dim),
            "ss_fei_sigma_low_div": "5.0",
        },
    )

    assert net.cfg.use_moe_style == "shared_A"
    assert net.cfg.route_per_layer is True
    assert net.cfg.router_source == "fei"
    assert net.cfg.fei_feature_dim == fei_dim
    assert net.cfg.fei_sigma_low_div == 5.0
    assert net.cfg.fei_router_names == [hydra_name]
    assert net.cfg.sigma_router_names is None
    assert net.cfg.hydra_router_names is None
    assert net._network_spec.name == "hydra"


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


def test_save_weights_stamps_three_axis_metadata_for_shared_a_global_fei(
    tmp_path,
) -> None:
    unet = TinyDiT()
    net = create_network(
        multiplier=1.0,
        network_dim=2,
        network_alpha=2.0,
        vae=None,
        text_encoders=[],
        unet=unet,
        use_moe_style="shared_A",
        route_per_layer=False,
        router_source="fei",
        router_targets="q_proj",
        num_experts=3,
        fei_feature_dim=2,
        fei_sigma_low_div=4.0,
        router_hidden_dim=8,
    )
    net.apply_to([], unet)
    out = tmp_path / "shared_a.safetensors"

    net.save_weights(str(out), dtype=torch.float32, metadata={})

    moe_out = tmp_path / "shared_a_moe.safetensors"
    assert moe_out.exists()
    with safe_open(str(moe_out), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        keys = set(handle.keys())

    assert metadata["ss_use_moe_style"] == "shared_A"
    assert metadata["ss_route_per_layer"] == "false"
    assert metadata["ss_router_source"] == "fei"
    assert metadata["ss_fei_feature_dim"] == "2"
    assert metadata["ss_fei_sigma_low_div"] == "4.0"
    assert "ss_network_spec" not in metadata
    assert "global_router.net.0.weight" in keys
    assert "global_router.net.2.bias" in keys
    assert "lora_unet_blocks_0_q_proj.lora_ups.0.weight" in keys
    assert "lora_unet_blocks_0_k_proj.lora_up.weight" in keys
    assert not any("_routing_weights" in key for key in keys)
    assert not any(
        ".router." in key
        for key in keys
        if key.startswith("lora_unet_blocks_0_q_proj")
    )


def test_save_weights_preserves_non_empty_metadata_and_stamps_network_spec(
    tmp_path,
) -> None:
    unet = TinyDiT()
    net = create_network(
        multiplier=1.0,
        network_dim=2,
        network_alpha=2.0,
        vae=None,
        text_encoders=[],
        unet=unet,
        use_moe_style="shared_A",
        route_per_layer=False,
        router_source="fei",
        router_targets="q_proj",
        num_experts=3,
        fei_feature_dim=2,
        router_hidden_dim=8,
    )
    net.apply_to([], unet)
    out = tmp_path / "shared_a_marker.safetensors"

    net.save_weights(str(out), dtype=torch.float32, metadata={"marker": "kept"})

    moe_out = tmp_path / "shared_a_marker_moe.safetensors"
    assert moe_out.exists()
    with safe_open(str(moe_out), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()

    assert metadata["marker"] == "kept"
    assert metadata["ss_network_spec"] == "hydra"
    assert metadata["ss_use_moe_style"] == "shared_A"
    assert metadata["ss_route_per_layer"] == "false"
    assert metadata["ss_router_source"] == "fei"


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
