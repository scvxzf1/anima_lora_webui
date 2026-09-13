"""Exercise the actual adapter writer and loader on an NF4 Krea block."""

import copy
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from library.models.krea2_raw.dit import SingleStreamBlock
from library.models.krea2_raw.quantize import quantize_dit_to_nf4
from networks import NETWORK_REGISTRY
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.factory import create_network_from_weights
from networks.lora_anima.network import LoRANetwork


@pytest.mark.parametrize(
    "variant,reft_alpha",
    [
        ("dora", None),
        ("ortho", None),
        ("reft", None),
        ("reft", 0.5),
        ("reft", 8.0),
    ],
)
def test_nf4_krea_export_reload_keeps_adapter_identity_and_output(
    tmp_path,
    variant,
    reft_alpha,
):
    torch.manual_seed(9)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([SingleStreamBlock(32, 2, 2)])
    model.config = SimpleNamespace(features=32)
    model.bfloat16()
    quantize_dit_to_nf4(model, torch.device("cpu"))
    model.requires_grad_(False)
    reloaded_model = copy.deepcopy(model)
    spec = NETWORK_REGISTRY["lora" if variant == "reft" else variant]
    cfg = LoRANetworkCfg(
        module_class=spec.module_class,
        lora_dim=4,
        alpha=4,
        unet_target_replace_modules=["SingleStreamBlock"],
        use_ortho=variant == "ortho",
        add_reft=variant == "reft",
        reft_dim=4,
        reft_layers="all",
        reft_alpha=reft_alpha,
    )
    network = LoRANetwork([], model, cfg)
    network._network_spec = spec
    network.apply_to([], model, apply_text_encoder=False)
    network.bfloat16()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    x = torch.randn(1, 7, 32, dtype=torch.bfloat16)
    vec = torch.randn(1, 1, 192, dtype=torch.bfloat16) * 0.1
    mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.blocks[0](x, vec, None, mask).float().square().mean()
        loss.backward()
        optimizer.step()
    network.eval()
    model.eval()
    expected = model.blocks[0](x, vec, None, mask)
    filename = tmp_path / f"{variant}.safetensors"
    network.save_weights(str(filename), torch.bfloat16, {})
    with safe_open(filename, framework="pt") as saved:
        keys = list(saved.keys())
        if variant == "dora":
            assert any("magnitude" in key or "dora_scale" in key for key in keys)
        elif variant == "ortho":
            assert any("lora_up.weight" in key for key in keys)
            assert not any(".S_p" in key for key in keys)
        else:
            assert "reft_unet_blocks_0.learned_source.weight" in keys
    loaded, weights = create_network_from_weights(
        1.0, str(filename), None, [], reloaded_model, for_inference=True
    )
    loaded.apply_to([], reloaded_model, apply_text_encoder=False)
    result = loaded.load_state_dict(weights, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    loaded.bfloat16().eval()
    reloaded_model.eval()
    actual = reloaded_model.blocks[0](x, vec, None, mask)
    # Ortho export refactors a BF16 matrix product; it is not bitwise identical.
    tolerance = (
        dict(rtol=2e-2, atol=1e-2) if variant == "ortho" else dict(rtol=0, atol=0)
    )
    torch.testing.assert_close(actual, expected, **tolerance)
    if variant == "reft":
        assert len(loaded.unet_refts) == 1
        assert loaded.unet_refts[0].scale == network.unet_refts[0].scale
        assert reloaded_model.blocks[0]._forward == loaded.unet_refts[0].forward
    loaded.set_multiplier(0)
    baseline = reloaded_model.blocks[0](x, vec, None, mask)
    loaded.set_multiplier(1)
    assert not torch.equal(actual, baseline)
