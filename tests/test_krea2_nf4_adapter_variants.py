"""Focused NF4 compatibility tests for the Krea adapter repair work."""

from types import SimpleNamespace

import pytest
import torch

from library.models.krea2_raw.quantize import quantize_dit_to_nf4
from networks.lora_anima.builders import _resolve_reft_embed_dim, create_reft_modules
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_modules import LoRAModule
from networks.lora_modules.dora import DoRALoRAModule
from networks.lora_modules.ortho import OrthoLoRAModule
from networks.lora_modules.reft import ReFTModule
from networks.lora_modules.weight_access import (
    is_nf4_weight,
    materialize_module_weight,
)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4, bias=True)

    def forward(self, x):
        return self.fc(x)


def _nf4_tiny() -> _TinyModel:
    model = _TinyModel().to(torch.bfloat16)
    quantize_dit_to_nf4(model, torch.device("cpu"))
    return model


def test_nf4_materialization_restores_logical_shape_without_mutating_packed_data():
    model = _nf4_tiny()
    weight = model.fc.weight
    packed_before = weight.data.detach().clone()

    assert is_nf4_weight(weight)
    assert tuple(weight.data.shape) != (4, 8)
    dense = materialize_module_weight(model.fc)

    assert dense.shape == (4, 8)
    assert dense.dtype == torch.float32
    assert torch.isfinite(dense).all()
    torch.testing.assert_close(weight.data, packed_before)


def test_dora_nf4_forward_backward_and_row_norm_match_dense_reference():
    torch.manual_seed(7)
    model = _nf4_tiny()
    dora = DoRALoRAModule("lora_fc", model.fc, lora_dim=2, alpha=2)
    dora.to(torch.bfloat16)
    dora.apply_to()
    model.train()
    with torch.no_grad():
        dora.lora_up.weight.normal_(std=0.1)

    x = torch.randn(3, 8, dtype=torch.bfloat16)
    output = model(x)
    assert output.shape == (3, 4)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert dora.lora_up.weight.grad is not None
    assert dora.lora_down.weight.grad is not None
    assert dora.magnitude.grad is not None

    probe = torch.zeros(1, 4, dtype=torch.bfloat16)
    fast_norm = dora._nf4_merged_norm(probe).detach()
    base = materialize_module_weight(model.fc)
    down = dora.lora_down.weight.detach().float()
    up = dora.lora_up.weight.detach().float()
    dense_norm = (base + dora.scale * (up @ down)).norm(dim=1).clamp_min(1e-6)
    torch.testing.assert_close(fast_norm, dense_norm, rtol=3e-3, atol=3e-3)


def test_dora_nf4_merge_and_fuse_fail_closed():
    model = _nf4_tiny()
    dora = DoRALoRAModule("lora_fc", model.fc, lora_dim=2, alpha=2)
    dora.apply_to()
    with torch.no_grad():
        dora.lora_up.weight.normal_(std=0.02)

    with pytest.raises(RuntimeError, match="not supported for NF4"):
        dora.fuse_weight()


def test_ortho_nf4_forward_and_backward_uses_dequantized_svd_init():
    torch.manual_seed(13)
    model = _nf4_tiny()
    ortho = OrthoLoRAModule("lora_fc", model.fc, lora_dim=2, alpha=2)
    ortho.to(torch.bfloat16)
    ortho.apply_to()
    model.train()

    output = model(torch.randn(3, 8, dtype=torch.bfloat16))
    assert output.shape == (3, 4)
    assert torch.isfinite(output).all()
    output.float().mean().backward()
    assert ortho.S_p.grad is not None
    assert ortho.S_q.grad is not None
    assert ortho.lambda_layer.grad is not None


class _ReFTBlock(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def test_reft_krea_dimension_falls_back_to_model_config():
    unet = torch.nn.Module()
    unet.blocks = torch.nn.ModuleList([_ReFTBlock(), _ReFTBlock()])
    unet.config = SimpleNamespace(features=8)
    cfg = LoRANetworkCfg(
        module_class=LoRAModule,
        lora_dim=2,
        alpha=2,
        add_reft=True,
        reft_dim=2,
        reft_layers="last_1",
    )

    assert _resolve_reft_embed_dim(unet, unet.blocks[1]) == 8
    _, refts = create_reft_modules(
        unet,
        cfg=cfg,
        multiplier=1.0,
        logger=__import__("logging").getLogger(__name__),
    )
    assert len(refts) == 1
    assert isinstance(refts[0], ReFTModule)
    assert refts[0].rotate_layer.in_features == 8
    assert refts[0].original_name == "blocks.1"


def test_reft_krea_zero_init_is_identity_and_trains():
    block = _ReFTBlock()
    reft = ReFTModule("reft", block, embed_dim=8, reft_dim=2, alpha=2)
    reft.apply_to()
    x = torch.randn(2, 5, 8, requires_grad=True)
    output = block(x)
    torch.testing.assert_close(output, x)
    output.square().mean().backward()
    assert reft.learned_source.weight.grad is not None
