"""LoRA-family register-token invariants."""

from __future__ import annotations

import torch

from library.anima.models import Anima
from networks.lora_anima.factory import create_network, create_network_from_weights


def _tiny_anima(num_blocks: int = 4) -> Anima:
    torch.manual_seed(42)
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


def _net(model, **kw):
    defaults = {
        "num_registers": "4",
        "register_insert_block": "2",
    }
    defaults.update(kw)
    return create_network(1.0, 4, 4.0, None, [], model, **defaults)


def test_mid_stack_insertion_strip_and_flags():
    model = _tiny_anima()
    inp = _inputs()
    with torch.no_grad():
        base = model.forward_mini_train_dit(*inp)

    net = _net(model)
    assert net.extra_seq_tokens == 4
    assert not net.is_mergeable()
    net.apply_to([], model, apply_text_encoder=False, apply_unet=True)

    seq_seen = {}
    hooks = [
        block.register_forward_hook(
            lambda module, args, out, block_idx=block_idx: seq_seen.__setitem__(
                block_idx, out.shape[2]
            )
        )
        for block_idx, block in enumerate(model.blocks)
    ]
    with torch.no_grad():
        out = model.forward_mini_train_dit(*inp)
    for handle in hooks:
        handle.remove()

    seq_len = (32 // 2) * (32 // 2)
    assert seq_seen == {0: seq_len, 1: seq_len, 2: seq_len + 4, 3: seq_len + 4}
    assert out.shape == base.shape
    assert net.register_injector is not None
    assert net.register_injector.last_reg_ratio is not None
    assert net.register_injector.last_patch_sink_ratio is not None


def test_no_registers_is_plain_lora():
    model = _tiny_anima()
    net = create_network(1.0, 4, 4.0, None, [], model)
    assert net.extra_seq_tokens == 0
    assert net.register_injector is None
    assert net.is_mergeable()
    assert "register_tokens" not in net.state_dict()


def test_gradients_and_optimizer_group():
    model = _tiny_anima()
    inp = _inputs()
    net = _net(model, register_lr_scale="50")
    net.apply_to([], model, apply_text_encoder=False, apply_unet=True)

    params, descs = net.prepare_optimizer_params_with_multiple_te_lrs(
        None, 1e-4, 1e-4
    )
    assert "register tokens" in descs
    reg_group = params[descs.index("register tokens")]
    assert abs(reg_group["lr"] - 1e-4 * 50.0) < 1e-15
    assert list(reg_group["params"]) == [net.register_tokens]

    model.forward_mini_train_dit(*inp).sum().backward()
    assert net.register_tokens.grad is not None
    assert net.register_tokens.grad.abs().sum() > 0


def test_save_load_roundtrip_bit_exact(tmp_path):
    model = _tiny_anima()
    inp = _inputs()
    net = _net(model)
    net.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        out = model.forward_mini_train_dit(*inp)

    path = str(tmp_path / "lora_reg.safetensors")
    net.save_weights(path, torch.float32, {"ss_output_name": "test"})

    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        meta = f.metadata()
    assert "register_tokens" in keys
    assert meta["ss_num_registers"] == "4"
    assert meta["ss_register_insert_block"] == "2"

    model2 = _tiny_anima()
    net2, sd2 = create_network_from_weights(1.0, path, None, [], model2)
    assert net2.cfg.num_registers == 4
    assert net2.cfg.register_insert_block == 2
    assert not net2.is_mergeable()
    net2.apply_to([], model2, apply_text_encoder=False, apply_unet=True)
    info = net2.load_state_dict(sd2, strict=False)
    assert "register_tokens" not in info.missing_keys
    with torch.no_grad():
        out2 = model2.forward_mini_train_dit(*inp)
    assert torch.equal(out2, out)


def test_merge_refuses_register_tokens():
    from scripts.merge_to_dit import scan_non_bakeable_keys

    found = scan_non_bakeable_keys({"register_tokens": torch.zeros(4, 64)})
    assert any("register" in kind for kind in found)


def test_repa_pool_trims_register_tail():
    from library.training.repa import pool_dit_tokens_to_grid
    import pytest

    seq_len = 16 * 16
    reg_k = 4
    dim = 64
    captured = torch.randn(2, 1, seq_len + reg_k, 1, dim)
    pooled = pool_dit_tokens_to_grid(captured, (32, 32), 2, 8, 8, trim_tail=reg_k)
    assert pooled.shape == (2, 64, dim)

    with pytest.raises(RuntimeError, match="Block/patch-grid mismatch"):
        pool_dit_tokens_to_grid(captured, (32, 32), 2, 8, 8)
