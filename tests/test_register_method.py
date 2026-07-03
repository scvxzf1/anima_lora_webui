"""Register-token adapter invariants."""

from __future__ import annotations

import torch

from library.anima.models import Anima
from networks.methods.register import RegisterNetwork, create_network_from_weights


def _tiny_anima(num_blocks: int = 4) -> Anima:
    return Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    ).eval().requires_grad_(False)


def _inputs(latent_h: int = 32, latent_w: int = 32):
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, latent_h, latent_w)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)
    return x, timesteps, crossattn_emb


def _net(unet, **kwargs) -> RegisterNetwork:
    defaults = dict(
        num_registers=4,
        arm="B",
        qkv_mode="unfrozen",
        target_blocks="2-3",
        insert_block=2,
    )
    defaults.update(kwargs)
    return RegisterNetwork(unet, **defaults)


@torch.no_grad()
def test_mid_stack_insertion_and_strip():
    model = _tiny_anima()
    inputs = _inputs()
    base = model.forward_mini_train_dit(*inputs)

    net = _net(model, insert_block=2)
    net.apply_to(None, model)
    seq_seen = {}
    hooks = [
        block.register_forward_hook(
            lambda module, args, out, block_idx=block_idx: seq_seen.__setitem__(
                block_idx, out.shape[2]
            )
        )
        for block_idx, block in enumerate(model.blocks)
    ]
    out = model.forward_mini_train_dit(*inputs)
    for handle in hooks:
        handle.remove()

    seq_len = (32 // 2) * (32 // 2)
    assert seq_seen == {0: seq_len, 1: seq_len, 2: seq_len + 4, 3: seq_len + 4}
    assert out.shape == base.shape
    assert net.last_reg_ratio is not None
    assert net.last_patch_sink_ratio is not None

    net.remove()
    assert torch.equal(model.forward_mini_train_dit(*inputs), base)


@torch.no_grad()
def test_entry_insertion_geometry():
    model = _tiny_anima()
    inputs = _inputs()
    base = model.forward_mini_train_dit(*inputs)

    net = _net(model, insert_block=0)
    net.apply_to(None, model)
    seq_seen = {}
    hooks = [
        block.register_forward_hook(
            lambda module, args, out, block_idx=block_idx: seq_seen.__setitem__(
                block_idx, out.shape[2]
            )
        )
        for block_idx, block in enumerate(model.blocks)
    ]
    out = model.forward_mini_train_dit(*inputs)
    for handle in hooks:
        handle.remove()
    net.remove()

    seq_len = (32 // 2) * (32 // 2)
    assert seq_seen == {block_idx: seq_len + 4 for block_idx in range(4)}
    assert out.shape == base.shape


@torch.no_grad()
def test_arm_l_is_step0_noop():
    model = _tiny_anima()
    inputs = _inputs()
    base = model.forward_mini_train_dit(*inputs)
    net = _net(model, num_registers=0, insert_block=2)
    net.apply_to(None, model)
    assert torch.equal(model.forward_mini_train_dit(*inputs), base)
    net.remove()


def test_gradients_reach_registers_through_frozen_prefix():
    model = _tiny_anima()
    inputs = _inputs()
    net = _net(model, insert_block=2)
    net.apply_to(None, model)
    model.forward_mini_train_dit(*inputs).sum().backward()
    assert net.register.grad is not None
    assert net.register.grad.abs().sum() > 0
    for surface in net.qkv.values():
        assert all(param.grad is not None for param in surface.parameters())
    net.remove()


def test_down_init_weight_svd():
    import pytest

    model = _tiny_anima()
    inputs = _inputs()
    with torch.no_grad():
        base = model.forward_mini_train_dit(*inputs)

    net = _net(
        model, qkv_mode="lora", lora_rank=4, down_init="weight_svd", num_registers=0
    )
    for block_idx in net.target_blocks:
        down = net.qkv[str(block_idx)].down
        gram = down @ down.T
        assert torch.allclose(gram, torch.eye(4), atol=1e-4)
        weight = model.blocks[block_idx].self_attn.qkv_proj.weight.float()
        svd_energy = (weight @ down.T).norm()
        rand = torch.linalg.qr(torch.randn(weight.shape[1], 4)).Q.T
        assert svd_energy > (weight @ rand.T).norm()
    net.apply_to(None, model)
    with torch.no_grad():
        assert torch.equal(model.forward_mini_train_dit(*inputs), base)
    net.remove()

    with pytest.raises(ValueError, match="down_init"):
        _net(model, qkv_mode="lora", down_init="nope")
    with pytest.raises(ValueError, match="only applies"):
        _net(model, qkv_mode="unfrozen", down_init="weight_svd")


def test_from_weights_restores_scale_and_insert_block(tmp_path):
    from safetensors.torch import save_file

    model = _tiny_anima()
    net = _net(model, qkv_mode="lora", lora_rank=4, lora_alpha=2.0, insert_block=1)
    assert net.scale == 0.5
    path = str(tmp_path / "adapter.safetensors")
    save_file(net.state_dict(), path, metadata=net.metadata_fields())

    net2, _ = create_network_from_weights(1.0, path, None, None, _tiny_anima())
    assert net2.scale == 0.5
    assert net2.insert_block == 1
    assert net2.qkv_mode == "lora"
    assert net2.K == 4

    legacy_meta = {
        key: value for key, value in net.metadata_fields().items() if key != "ss_insert_block"
    }
    legacy = str(tmp_path / "legacy.safetensors")
    save_file(net.state_dict(), legacy, metadata=legacy_meta)
    net3, _ = create_network_from_weights(1.0, legacy, None, None, _tiny_anima())
    assert net3.insert_block == 0
