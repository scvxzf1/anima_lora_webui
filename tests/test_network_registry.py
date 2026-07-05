"""Tests for the M2 network registry and save pipeline.

Covers:

* ``resolve_network_spec`` precedence and mutual-exclusion rules.
* The ``networks.lora_save`` pipeline round-trips a synthetic state_dict
  for each save_variant, emitting the expected file(s) and preserving
  tensor shapes through the per-variant conversion.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from networks import (
    NETWORK_REGISTRY,
    SHARED_KWARG_FLAGS,
    ModuleCreationContext,
    NetworkSpec,
    all_network_kwargs,
    resolve_network_spec,
)
from networks import lora_save
from networks.lora_anima.factory import create_network_from_weights
from networks.lora_modules import DoRALoRAModule
from networks.plugins.glora.module import GLoRAModule
from networks.plugins.loha.module import LoHaModule
from networks.plugins.lokr.module import LoKrModule
from networks.plugins.vera.module import VeRAModule


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


EXPECTED_VARIANTS = {
    "dora",
    "glora",
    "lora",
    "loha",
    "lokr",
    "ortho",
    "hydra",
    "ortho_hydra",
    "vera",
}


def test_registry_has_expected_variants():
    assert EXPECTED_VARIANTS.issubset(NETWORK_REGISTRY.keys())
    for name, spec in NETWORK_REGISTRY.items():
        assert isinstance(spec, NetworkSpec)
        assert spec.name == name


def test_all_network_kwargs_is_union_of_shared_and_specs():
    """`all_network_kwargs()` must cover every kwarg any variant declares.

    Guards against the drift mode that previously silently dropped
    `router_targets` (originally a trio of `hydra_router_layers` /
    `sigma_router_layers` / `fei_router_layers` before they were
    consolidated): a kwarg declared on a NetworkSpec but missing from the
    forwarding list.
    """
    all_kw = set(all_network_kwargs())
    assert set(SHARED_KWARG_FLAGS).issubset(all_kw)
    for spec in NETWORK_REGISTRY.values():
        assert set(spec.kwarg_flags).issubset(all_kw), (
            f"{spec.name}.kwarg_flags has keys missing from all_network_kwargs(): "
            f"{set(spec.kwarg_flags) - all_kw}"
        )


def test_hydra_router_kwargs_registered():
    """Regression pin: the bug that motivated the M2 finish.

    `router_targets` + σ-conditional router kwargs must be registered on
    the hydra / ortho_hydra specs so they flow through argparse schema
    and into `create_network`. If these drop off the spec, the router
    silently defaults to uniform MoE over every target module.
    """
    must_have = {
        "router_targets",
        "sigma_feature_dim",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
    }
    for variant in ("hydra", "ortho_hydra"):
        flags = set(NETWORK_REGISTRY[variant].kwarg_flags)
        missing = must_have - flags
        assert not missing, f"{variant} spec missing kwarg_flags: {missing}"
    # and the union exposes them
    assert must_have.issubset(set(all_network_kwargs()))


def test_register_token_kwargs_registered():
    must_have = {
        "num_registers",
        "register_insert_block",
        "register_lr_scale",
        "register_init_std",
    }
    assert must_have.issubset(set(SHARED_KWARG_FLAGS))
    assert must_have.issubset(set(all_network_kwargs()))


def test_lora_dtype_policy_kwargs_registered():
    must_have = {"lora_fp32_compute", "down_init"}
    assert must_have.issubset(set(SHARED_KWARG_FLAGS))
    assert must_have.issubset(set(all_network_kwargs()))


def test_lokr_kwargs_registered():
    must_have = {
        "use_lokr",
        "lokr_factor",
        "lokr_factor_group_size",
        "lokr_project_chunk_bytes",
        "lokr_grouped_delta_backend",
        "lokr_use_einsum",
        "lokr_decompose_w2",
    }
    assert must_have.issubset(set(all_network_kwargs()))
    assert "lokr_factor" in set(NETWORK_REGISTRY["lokr"].kwarg_flags)
    assert "lokr_factor_group_size" in set(NETWORK_REGISTRY["lokr"].kwarg_flags)
    assert "lokr_project_chunk_bytes" in set(NETWORK_REGISTRY["lokr"].kwarg_flags)
    assert "lokr_grouped_delta_backend" in set(NETWORK_REGISTRY["lokr"].kwarg_flags)
    assert "lokr_use_einsum" in set(NETWORK_REGISTRY["lokr"].kwarg_flags)
    assert "lokr_decompose_w2" in set(NETWORK_REGISTRY["lokr"].kwarg_flags)


def test_lokr_module_kwargs_forward_grouped_delta_backend():
    ctx = ModuleCreationContext(
        cfg=SimpleNamespace(
            plugin_args={
                "lokr_factor": 4,
                "lokr_factor_group_size": 2,
                "lokr_project_chunk_bytes": 2048,
                "lokr_grouped_delta_backend": "triton",
                "lokr_use_einsum": "false",
                "lokr_decompose_w2": "true",
            }
        ),
        is_unet=True,
        lora_name="lora_unet_test",
        original_name="blocks.0.mlp.layer1",
        child_module=torch.nn.Linear(8, 8, bias=False),
        module_class=LoKrModule,
    )
    kwargs = NETWORK_REGISTRY["lokr"].module_kwargs(ctx)
    assert kwargs["factor"] == 4
    assert kwargs["lokr_factor_group_size"] == 2
    assert kwargs["lokr_project_chunk_bytes"] == 2048
    assert kwargs["lokr_grouped_delta_backend"] == "triton"
    assert kwargs["lokr_use_einsum"] is False
    assert kwargs["lokr_decompose_w2"] is True


def test_lokr_module_kwargs_default_keeps_full_w2():
    ctx = ModuleCreationContext(
        cfg=SimpleNamespace(plugin_args={}),
        is_unet=True,
        lora_name="lora_unet_test",
        original_name="blocks.0.mlp.layer1",
        child_module=torch.nn.Linear(8, 8, bias=False),
        module_class=LoKrModule,
    )
    kwargs = NETWORK_REGISTRY["lokr"].module_kwargs(ctx)
    assert kwargs["lokr_use_einsum"] is True
    assert kwargs["lokr_decompose_w2"] is False


def test_dora_kwargs_registered():
    assert "dora_wd" in set(all_network_kwargs())
    assert "dora_wd" in set(SHARED_KWARG_FLAGS)
    assert NETWORK_REGISTRY["dora"].module_class is DoRALoRAModule


def test_loha_kwargs_registered():
    must_have = {"use_loha"}
    assert must_have.issubset(set(all_network_kwargs()))
    assert "use_loha" in set(NETWORK_REGISTRY["loha"].kwarg_flags)


def test_glora_kwargs_registered():
    must_have = {"use_glora"}
    assert must_have.issubset(set(all_network_kwargs()))
    assert "use_glora" in set(NETWORK_REGISTRY["glora"].kwarg_flags)
    assert NETWORK_REGISTRY["glora"].module_class is GLoRAModule


def test_vera_kwargs_registered():
    must_have = {
        "use_vera",
        "vera_projection_prng_key",
        "vera_d_initial",
        "vera_save_projection",
    }
    assert must_have.issubset(set(all_network_kwargs()))
    assert must_have.issubset(set(NETWORK_REGISTRY["vera"].kwarg_flags))


def test_lokr_lives_in_plugin_not_core_imports():
    core_files = [
        Path("networks/__init__.py"),
        Path("networks/lora_anima/network.py"),
        Path("networks/lora_anima/factory.py"),
        Path("networks/lora_save.py"),
        Path("networks/lora_modules/__init__.py"),
        Path("networks/lora_modules/custom_autograd.py"),
    ]
    forbidden = ("LoKrModule", "lokr_project", "networks.lora_modules.lokr")
    for path in core_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} still references {token}"


# ---------------------------------------------------------------------------
# resolve_network_spec precedence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "lora"),
        ({"use_glora": "true"}, "glora"),
        ({"use_loha": "true"}, "loha"),
        ({"use_lokr": "true"}, "lokr"),
        ({"use_vera": "true"}, "vera"),
        ({"dora_wd": "true"}, "dora"),
        ({"use_ortho": "true"}, "ortho"),
        ({"use_moe_style": "shared_A"}, "hydra"),
        ({"use_moe_style": "shared_A", "use_ortho": "true"}, "ortho_hydra"),
        ({"use_moe_style": "independent_A"}, "stacked_experts_global_fei"),
        # Falsey forms of use_moe_style resolve to plain LoRA.
        ({"use_moe_style": False}, "lora"),
        ({"use_moe_style": "false"}, "lora"),
        ({"use_moe_style": ""}, "lora"),
    ],
)
def test_resolve_precedence(kwargs, expected):
    spec = resolve_network_spec(kwargs)
    assert spec.name == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_lokr": "true", "use_ortho": "true"},
        {"use_lokr": "true", "use_moe_style": "shared_A"},
        {"use_lokr": "true", "use_chimera_hydra": "true"},
    ],
)
def test_lokr_mutual_exclusion(kwargs):
    with pytest.raises(ValueError, match="use_lokr is mutually exclusive"):
        resolve_network_spec(kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dora_wd": "true", "use_ortho": "true"}, "incompatible"),
        ({"dora_wd": "true", "use_moe_style": "shared_A"}, "plain LoRA"),
        ({"dora_wd": "true", "use_chimera_hydra": "true"}, "ChimeraHydra"),
        ({"dora_wd": "true", "use_loha": "true"}, "mutually exclusive"),
        ({"dora_wd": "true", "use_lokr": "true"}, "mutually exclusive"),
        ({"dora_wd": "true", "use_vera": "true"}, "mutually exclusive"),
    ],
)
def test_dora_mutual_exclusion(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_network_spec(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_loha": "true", "use_lokr": "true"},
        {"use_loha": "true", "use_ortho": "true"},
        {"use_loha": "true", "use_moe_style": "shared_A"},
        {"use_loha": "true", "use_chimera_hydra": "true"},
    ],
)
def test_loha_mutual_exclusion(kwargs):
    with pytest.raises(ValueError, match="use_loha is mutually exclusive"):
        resolve_network_spec(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_glora": "true", "use_loha": "true"},
        {"use_glora": "true", "use_lokr": "true"},
        {"use_glora": "true", "use_vera": "true"},
        {"use_glora": "true", "dora_wd": "true"},
        {"use_glora": "true", "use_ortho": "true"},
        {"use_glora": "true", "use_moe_style": "shared_A"},
        {"use_glora": "true", "use_chimera_hydra": "true"},
    ],
)
def test_glora_mutual_exclusion(kwargs):
    with pytest.raises(ValueError, match="use_glora is mutually exclusive"):
        resolve_network_spec(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_vera": "true", "use_lokr": "true"},
        {"use_vera": "true", "use_loha": "true"},
        {"use_vera": "true", "use_ortho": "true"},
        {"use_vera": "true", "use_moe_style": "shared_A"},
        {"use_vera": "true", "use_chimera_hydra": "true"},
    ],
)
def test_vera_mutual_exclusion(kwargs):
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_network_spec(kwargs)


# ---------------------------------------------------------------------------
# save_network_weights round-trips — synthetic state_dicts, one per variant
# ---------------------------------------------------------------------------


def _alpha(value: float) -> torch.Tensor:
    return torch.tensor(float(value))


def _make_std_lora_sd(prefix: str, r: int, in_dim: int, out_dim: int) -> dict:
    """Fake fused-qkv LoRA state_dict entry (runtime form).

    The runtime uses fused self_attn.qkv_proj; save defuses it into q/k/v.
    """
    return {
        f"{prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{prefix}.lora_up.weight": torch.randn(3 * out_dim, r),
        f"{prefix}.alpha": _alpha(r),
    }


def _save_and_reload(
    state_dict: dict,
    tmp_path: Path,
    save_variant: str,
    filename: str = "out.safetensors",
) -> dict[str, torch.Tensor]:
    out = tmp_path / filename
    lora_save.save_network_weights(
        dict(state_dict),  # copy — save mutates
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_network_spec": save_variant},
        save_variant=save_variant,
    )
    # hydra writes *_moe.safetensors alongside (not the main file)
    if save_variant in ("hydra_moe", "ortho_hydra_to_hydra"):
        moe_path = tmp_path / (out.stem + "_moe.safetensors")
        assert moe_path.exists(), f"expected _moe file at {moe_path}"
        return load_file(str(moe_path))
    assert out.exists()
    return load_file(str(out))


def test_save_standard_lora_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = _make_std_lora_sd(prefix, r, in_dim, out_dim)

    loaded = _save_and_reload(sd, tmp_path, save_variant="standard")

    # qkv_proj should be defused into q/k/v with matching shapes
    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert f"{base}_{suffix}.lora_down.weight" in loaded
        assert f"{base}_{suffix}.lora_up.weight" in loaded
        assert f"{base}_{suffix}.alpha" in loaded
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.lora_up.weight"].shape == (out_dim, r)
    # fused key must be gone
    assert f"{prefix}.lora_down.weight" not in loaded


def test_save_standard_dora_roundtrip_exports_dora_scale(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = _make_std_lora_sd(prefix, r, in_dim, out_dim)
    sd[f"{prefix}.magnitude"] = torch.arange(
        1,
        3 * out_dim + 1,
        dtype=torch.float32,
    )

    loaded = _save_and_reload(sd, tmp_path, save_variant="standard")

    base = "lora_unet_blocks_0_self_attn"
    for idx, suffix in enumerate(("q_proj", "k_proj", "v_proj")):
        key = f"{base}_{suffix}.dora_scale"
        assert key in loaded
        assert loaded[key].shape == (out_dim,)
        expected = sd[f"{prefix}.magnitude"][idx * out_dim : (idx + 1) * out_dim]
        assert torch.equal(loaded[key], expected)
    assert f"{prefix}.magnitude" not in loaded


def test_save_lokr_roundtrip(tmp_path: Path):
    factor, in_dim, out_dim = 2, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.lokr_w1": torch.randn(factor, factor),
        f"{prefix}.lokr_w2": torch.randn((3 * out_dim) // factor, in_dim // factor),
        f"{prefix}.alpha": _alpha(32),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="lokr")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lokr_w1"].shape == (factor, factor)
        assert loaded[f"{base}_{suffix}.lokr_w2"].shape == (
            out_dim // factor,
            in_dim // factor,
        )
        assert f"{base}_{suffix}.alpha" in loaded
    assert f"{prefix}.lokr_w1" not in loaded
    assert f"{prefix}.lokr_w2" not in loaded


def test_save_lokr_decomposed_w2_roundtrip(tmp_path: Path):
    factor, rank, in_dim, out_dim = 2, 3, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.lokr_w1": torch.randn(factor, factor),
        f"{prefix}.lokr_w2_a": torch.randn((3 * out_dim) // factor, rank),
        f"{prefix}.lokr_w2_b": torch.randn(rank, in_dim // factor),
        f"{prefix}.alpha": _alpha(32),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="lokr")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lokr_w1"].shape == (factor, factor)
        assert loaded[f"{base}_{suffix}.lokr_w2_a"].shape == (
            out_dim // factor,
            rank,
        )
        assert loaded[f"{base}_{suffix}.lokr_w2_b"].shape == (rank, in_dim // factor)
        assert f"{base}_{suffix}.alpha" in loaded
    assert f"{prefix}.lokr_w1" not in loaded
    assert f"{prefix}.lokr_w2_a" not in loaded
    assert f"{prefix}.lokr_w2_b" not in loaded


def test_save_loha_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.hada_w1_a": torch.randn(3 * out_dim, r),
        f"{prefix}.hada_w1_b": torch.randn(r, in_dim),
        f"{prefix}.hada_w2_a": torch.randn(3 * out_dim, r),
        f"{prefix}.hada_w2_b": torch.randn(r, in_dim),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="loha")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.hada_w1_a"].shape == (out_dim, r)
        assert loaded[f"{base}_{suffix}.hada_w1_b"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.hada_w2_a"].shape == (out_dim, r)
        assert loaded[f"{base}_{suffix}.hada_w2_b"].shape == (r, in_dim)
        assert f"{base}_{suffix}.alpha" in loaded
    assert f"{prefix}.hada_w1_a" not in loaded
    assert f"{prefix}.hada_w2_a" not in loaded


def test_save_glora_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.a1.weight": torch.randn(in_dim, r),
        f"{prefix}.a2.weight": torch.randn(r, in_dim),
        f"{prefix}.b1.weight": torch.randn(3 * out_dim, r),
        f"{prefix}.b2.weight": torch.randn(r, in_dim),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="glora")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.a1.weight"].shape == (in_dim, r)
        assert loaded[f"{base}_{suffix}.a2.weight"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.b1.weight"].shape == (out_dim, r)
        assert loaded[f"{base}_{suffix}.b2.weight"].shape == (r, in_dim)
        assert f"{base}_{suffix}.alpha" in loaded
    assert f"{prefix}.a1.weight" not in loaded
    assert f"{prefix}.b1.weight" not in loaded


def test_save_vera_roundtrip(tmp_path: Path):
    r, out_dim = 4, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.vera_lambda_b": torch.randn(3 * out_dim),
        f"{prefix}.vera_lambda_d": torch.randn(r),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="vera")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.vera_lambda_b"].shape == (out_dim,)
        assert loaded[f"{base}_{suffix}.vera_lambda_d"].shape == (r,)
        assert f"{base}_{suffix}.alpha" in loaded
    assert f"{prefix}.vera_lambda_b" not in loaded
    assert f"{prefix}.vera_lambda_d" not in loaded


def test_create_network_from_lokr_weights_uses_lokr_module():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 6, bias=False)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    weights_sd = {
        "lora_unet_blocks_0_q_proj.lokr_w1": torch.randn(2, 2),
        "lora_unet_blocks_0_q_proj.lokr_w2": torch.randn(3, 2),
        "lora_unet_blocks_0_q_proj.alpha": _alpha(32),
    }

    unet = TinyUnet()
    network, weights = create_network_from_weights(
        multiplier=1.0,
        file="",
        ae=None,
        text_encoders=[],
        unet=unet,
        weights_sd=weights_sd,
        metadata={"ss_network_spec": "lokr", "ss_network_dim": "32"},
    )

    assert len(network.unet_loras) == 1
    lokr = network.unet_loras[0]
    assert isinstance(lokr, LoKrModule)
    assert lokr.lora_dim == 32
    assert lokr.factor == 2
    network.apply_to([], unet, False, True)
    info = network.load_state_dict(weights, strict=False)
    assert info.missing_keys == []
    assert info.unexpected_keys == []


def test_create_network_from_decomposed_lokr_weights_uses_default_einsum():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 6, bias=False)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    weights_sd = {
        "lora_unet_blocks_0_q_proj.lokr_w1": torch.randn(2, 2),
        "lora_unet_blocks_0_q_proj.lokr_w2_a": torch.randn(3, 2),
        "lora_unet_blocks_0_q_proj.lokr_w2_b": torch.randn(2, 2),
        "lora_unet_blocks_0_q_proj.alpha": _alpha(2),
    }

    unet = TinyUnet()
    network, weights = create_network_from_weights(
        multiplier=1.0,
        file="",
        ae=None,
        text_encoders=[],
        unet=unet,
        weights_sd=weights_sd,
        metadata={"ss_network_spec": "lokr", "ss_network_dim": "2"},
    )

    assert len(network.unet_loras) == 1
    lokr = network.unet_loras[0]
    assert isinstance(lokr, LoKrModule)
    assert lokr.lokr_use_einsum is True
    assert hasattr(lokr, "lokr_w2_a")
    network.apply_to([], unet, False, True)
    info = network.load_state_dict(weights, strict=False)
    assert info.missing_keys == []
    assert info.unexpected_keys == []


def test_create_network_from_dora_split_dora_magnitude_uses_dora_module():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.qkv_proj = torch.nn.Linear(4, 18, bias=False)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    rank = 2
    out_per = 6
    weights_sd = {}
    down = torch.randn(rank, 4)
    for letter in ("q", "k", "v"):
        prefix = f"lora_unet_blocks_0_self_attn_{letter}_proj"
        weights_sd[f"{prefix}.lora_down.weight"] = down.clone()
        weights_sd[f"{prefix}.lora_up.weight"] = torch.randn(out_per, rank)
        weights_sd[f"{prefix}.alpha"] = _alpha(rank)
        weights_sd[f"{prefix}.dora_magnitude"] = torch.rand(out_per) + 0.5

    unet = TinyUnet()
    network, weights = create_network_from_weights(
        multiplier=1.0,
        file="",
        ae=None,
        text_encoders=[],
        unet=unet,
        weights_sd=weights_sd,
        metadata={"ss_network_spec": "dora"},
    )

    assert len(network.unet_loras) == 1
    dora = network.unet_loras[0]
    assert isinstance(dora, DoRALoRAModule)
    assert network.cfg.use_dora is True
    fused_prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    assert f"{fused_prefix}.magnitude" in weights
    assert weights[f"{fused_prefix}.magnitude"].shape == (3 * out_per,)
    assert not any(key.endswith(".dora_magnitude") for key in weights)

    network.apply_to([], unet, False, True)
    info = network.load_state_dict(weights, strict=False)
    assert info.missing_keys == []
    assert info.unexpected_keys == []


def test_create_network_from_loha_weights_uses_loha_module():
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
        "lora_unet_blocks_0_q_proj.alpha": _alpha(2),
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


def test_create_network_from_glora_weights_uses_glora_module():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 6, bias=False)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    weights_sd = {
        "lora_unet_blocks_0_q_proj.a1.weight": torch.randn(4, 2),
        "lora_unet_blocks_0_q_proj.a2.weight": torch.randn(2, 4),
        "lora_unet_blocks_0_q_proj.b1.weight": torch.randn(6, 2),
        "lora_unet_blocks_0_q_proj.b2.weight": torch.randn(2, 4),
        "lora_unet_blocks_0_q_proj.alpha": _alpha(2),
    }

    unet = TinyUnet()
    network, weights = create_network_from_weights(
        multiplier=1.0,
        file="",
        ae=None,
        text_encoders=[],
        unet=unet,
        weights_sd=weights_sd,
        metadata={"ss_network_spec": "glora", "ss_network_dim": "2"},
    )

    assert len(network.unet_loras) == 1
    glora = network.unet_loras[0]
    assert isinstance(glora, GLoRAModule)
    assert glora.lora_dim == 2
    network.apply_to([], unet, False, True)
    info = network.load_state_dict(weights, strict=False)
    assert info.missing_keys == []
    assert info.unexpected_keys == []


def test_create_network_from_split_glora_qkv_fuses_runtime_keys():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.qkv_proj = torch.nn.Linear(4, 18, bias=False)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    rank = 2
    out_per = 6
    shared_a1 = torch.randn(4, rank)
    shared_a2 = torch.randn(rank, 4)
    shared_b2 = torch.randn(rank, 4)
    weights_sd = {}
    for letter in ("q", "k", "v"):
        prefix = f"lora_unet_blocks_0_self_attn_{letter}_proj"
        weights_sd[f"{prefix}.a1.weight"] = shared_a1.clone()
        weights_sd[f"{prefix}.a2.weight"] = shared_a2.clone()
        weights_sd[f"{prefix}.b1.weight"] = torch.randn(out_per, rank)
        weights_sd[f"{prefix}.b2.weight"] = shared_b2.clone()
        weights_sd[f"{prefix}.alpha"] = _alpha(rank)

    unet = TinyUnet()
    network, weights = create_network_from_weights(
        multiplier=1.0,
        file="",
        ae=None,
        text_encoders=[],
        unet=unet,
        weights_sd=weights_sd,
        metadata={"ss_network_spec": "glora", "ss_network_dim": str(rank)},
    )

    assert len(network.unet_loras) == 1
    glora = network.unet_loras[0]
    assert isinstance(glora, GLoRAModule)
    fused_prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    assert weights[f"{fused_prefix}.a1.weight"].shape == (4, rank)
    assert weights[f"{fused_prefix}.a2.weight"].shape == (rank, 4)
    assert weights[f"{fused_prefix}.b1.weight"].shape == (3 * out_per, rank)
    assert weights[f"{fused_prefix}.b2.weight"].shape == (rank, 4)
    network.apply_to([], unet, False, True)
    info = network.load_state_dict(weights, strict=False)
    assert info.missing_keys == []
    assert info.unexpected_keys == []


def test_create_network_from_split_glora_qkv_rejects_unshared_input_factors():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.qkv_proj = torch.nn.Linear(4, 18, bias=False)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    rank = 2
    out_per = 6
    shared_a1 = torch.randn(4, rank)
    shared_b2 = torch.randn(rank, 4)
    weights_sd = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"lora_unet_blocks_0_self_attn_{letter}_proj"
        weights_sd[f"{prefix}.a1.weight"] = shared_a1.clone()
        weights_sd[f"{prefix}.a2.weight"] = torch.randn(rank, 4) + idx
        weights_sd[f"{prefix}.b1.weight"] = torch.randn(out_per, rank)
        weights_sd[f"{prefix}.b2.weight"] = shared_b2.clone()
        weights_sd[f"{prefix}.alpha"] = _alpha(rank)

    with pytest.raises(RuntimeError, match="Split GLoRA checkpoint"):
        create_network_from_weights(
            multiplier=1.0,
            file="",
            ae=None,
            text_encoders=[],
            unet=TinyUnet(),
            weights_sd=weights_sd,
            metadata={"ss_network_spec": "glora", "ss_network_dim": str(rank)},
        )


def test_create_network_from_vera_weights_uses_vera_module():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.qkv_proj = torch.nn.Linear(4, 18, bias=False)

    class TinyUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    shared_lambda_d = torch.randn(3)
    weights_sd = {
        "lora_unet_blocks_0_self_attn_q_proj.vera_lambda_b": torch.randn(6),
        "lora_unet_blocks_0_self_attn_k_proj.vera_lambda_b": torch.randn(6),
        "lora_unet_blocks_0_self_attn_v_proj.vera_lambda_b": torch.randn(6),
        "lora_unet_blocks_0_self_attn_q_proj.vera_lambda_d": shared_lambda_d,
        "lora_unet_blocks_0_self_attn_k_proj.vera_lambda_d": shared_lambda_d.clone(),
        "lora_unet_blocks_0_self_attn_v_proj.vera_lambda_d": shared_lambda_d.clone(),
        "lora_unet_blocks_0_self_attn_q_proj.alpha": _alpha(3),
    }

    unet = TinyUnet()
    network, weights = create_network_from_weights(
        multiplier=1.0,
        file="",
        ae=None,
        text_encoders=[],
        unet=unet,
        weights_sd=weights_sd,
        metadata={
            "ss_network_spec": "vera",
            "ss_vera_projection_prng_key": "7",
            "ss_vera_d_initial": "0.1",
        },
    )

    assert len(network.unet_loras) == 1
    vera = network.unet_loras[0]
    assert isinstance(vera, VeRAModule)
    assert vera.lora_dim == 3
    assert vera.projection_bank is not None
    network.apply_to([], unet, False, True)
    info = network.load_state_dict(weights, strict=False)
    assert info.missing_keys == []
    assert info.unexpected_keys == []


def test_save_ortho_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    # OrthoLoRA (PSOFT) runtime keys: Cayley params + frozen SVD bases
    sd = {
        f"{prefix}.S_p": torch.randn(r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_basis": torch.randn(3 * out_dim, r),
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_to_lora")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.lora_up.weight"].shape == (out_dim, r)
    for k in loaded:
        assert not k.endswith(".S_p") and not k.endswith(".S_q")
        assert not k.endswith(".P_basis") and not k.endswith(".Q_basis")


def test_save_hydra_moe_roundtrip(tmp_path: Path):
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{prefix}.lora_up_weight": torch.randn(E, 3 * out_dim, r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="hydra_moe")

    base = "lora_unet_blocks_0_self_attn"
    # per-expert ups expanded, qkv defused per-expert
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)
        assert loaded[f"{base}_{suffix}.router.weight"].shape == (E, in_dim)
        assert loaded[f"{base}_{suffix}.router.bias"].shape == (E,)
    # fused lora_up_weight must be gone (expanded into per-expert keys)
    for k in loaded:
        assert not k.endswith(".lora_up_weight")


def test_save_hydra_moe_mixed_with_plain_lora_qkv_defuses_up(tmp_path: Path):
    """Regression: when ``router_targets`` filters some fused-qkv modules
    out of MoE, the resulting plain-LoRA leg for those modules must also be
    q/k/v-defused by the hydra save pipeline. Previously only ``lora_down`` /
    ``alpha`` were split; ``lora_up.weight`` stayed fused, producing a
    mismatched checkpoint.
    """
    E, r, in_dim, out_dim = 4, 4, 8, 12

    # Hydra-routed module (cross_attn.kv — regex-matched target)
    hydra_prefix = "lora_unet_blocks_0_cross_attn_kv_proj"
    # Plain-LoRA module (self_attn.qkv — regex-excluded by router_targets)
    plain_prefix = "lora_unet_blocks_0_self_attn_qkv_proj"

    sd = {
        # hydra leg — stacked lora_up_weight
        f"{hydra_prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{hydra_prefix}.lora_up_weight": torch.randn(E, 2 * out_dim, r),
        f"{hydra_prefix}.router.weight": torch.randn(E, r),
        f"{hydra_prefix}.router.bias": torch.randn(E),
        f"{hydra_prefix}.alpha": _alpha(r),
        # plain LoRA leg — standard single lora_up.weight, no router
        f"{plain_prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{plain_prefix}.lora_up.weight": torch.randn(3 * out_dim, r),
        f"{plain_prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="hydra_moe")

    # Hydra leg: split into k/v with per-expert ups
    hydra_base = "lora_unet_blocks_0_cross_attn"
    for suffix in ("k_proj", "v_proj"):
        assert loaded[f"{hydra_base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{hydra_base}_{suffix}.lora_ups.{e}.weight"].shape == (
                out_dim,
                r,
            )

    # Plain leg: must also be defused — lora_up.weight split per q/k/v,
    # fused prefix fully gone.
    plain_base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{plain_base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{plain_base}_{suffix}.lora_up.weight"].shape == (out_dim, r), (
            f"plain-LoRA self_attn_{suffix} lora_up.weight missing or still fused — "
            "hydra save pipeline didn't defuse the plain leg"
        )
        assert f"{plain_base}_{suffix}.alpha" in loaded
        # plain leg must NOT have hydra-only keys
        assert f"{plain_base}_{suffix}.lora_ups.0.weight" not in loaded
        assert f"{plain_base}_{suffix}.router.weight" not in loaded
    # fused prefix must be entirely purged
    for k in loaded:
        assert not k.startswith(plain_prefix), (
            f"fused plain-LoRA key survived: {k}"
        )


def test_save_ortho_hydra_roundtrip(tmp_path: Path):
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    # OrthoHydraLoRA runtime keys: S_p is 3-D (E, r, r); P_bases is (E, out, r)
    sd = {
        f"{prefix}.S_p": torch.randn(E, r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_bases": torch.randn(E, 3 * out_dim, r),
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_hydra_to_hydra")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)
    for k in loaded:
        assert not k.endswith(".S_p") and not k.endswith(".S_q")
        assert not k.endswith(".P_bases") and not k.endswith(".P_basis")


def test_save_ortho_hydra_legacy_P_basis_still_bakes(tmp_path: Path):
    """Legacy OrthoHydra checkpoints (pre-disjoint-bases) used a single
    (out, r) ``P_basis`` shared across experts. The save pipeline must still
    bake these into hydra moe form so old artifacts remain convertible.
    """
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.S_p": torch.randn(E, r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_basis": torch.randn(3 * out_dim, r),  # legacy 2-D
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
    }
    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_hydra_to_hydra")
    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)


# ---------------------------------------------------------------------------
# Metadata stamp
# ---------------------------------------------------------------------------


def _load_metadata(path: Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return f.metadata() or {}


def test_metadata_stamps_ss_network_spec(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = _make_std_lora_sd(prefix, r, in_dim, out_dim)

    out = tmp_path / "out.safetensors"
    lora_save.save_network_weights(
        dict(sd),
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_network_spec": "lora"},
        save_variant="standard",
    )
    meta = _load_metadata(out)
    assert meta.get("ss_network_spec") == "lora"
