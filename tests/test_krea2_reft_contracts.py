"""Real Krea blocks exercise the ReFT checkpoint and compile boundaries."""

import copy
from types import SimpleNamespace

import pytest
import torch

from library.models.krea2_raw.dit import SingleStreamBlock, SingleStreamDiT
from library.models.krea2_raw.quantize import quantize_dit_to_nf4
from networks.lora_anima.application import set_enabled
from networks.lora_anima.builders import _resolve_reft_embed_dim
from networks.lora_modules.reft import ReFTModule


def _block(nf4=False):
    torch.manual_seed(19)
    block = SingleStreamBlock(features=32, heads=2, multiplier=2)
    if nf4:
        block.bfloat16()
        quantize_dit_to_nf4(block, torch.device("cpu"))
    block.requires_grad_(False)
    return block


def _reft(block, **kwargs):
    alpha = kwargs.pop("alpha", 4)
    reft = ReFTModule(
        "reft_unet_blocks_0", block, embed_dim=32, reft_dim=4, alpha=alpha, **kwargs
    )
    reft.apply_to()
    with torch.no_grad():
        reft.learned_source.weight.normal_(std=0.01)
    return reft


def _inputs(dtype=torch.float32):
    return (
        torch.randn(1, 7, 32, dtype=dtype, requires_grad=True),
        torch.randn(1, 1, 192, dtype=dtype) * 0.1,
        None,
        torch.ones(1, 7, dtype=torch.bool),
    )


def _model(blocks, swap=0):
    model = SingleStreamDiT.__new__(SingleStreamDiT)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList(blocks)
    model.blocks_to_swap = swap
    return model


def test_krea_reft_width_comes_from_block_not_unrelated_config():
    assert (
        _resolve_reft_embed_dim(
            SimpleNamespace(config=SimpleNamespace(features=999)), _block()
        )
        == 32
    )


def test_reft_disabled_by_network_and_zero_strength_is_identity():
    block = _block()
    base = copy.deepcopy(block)
    reft = _reft(block)
    network = SimpleNamespace(
        text_encoder_loras=[], unet_loras=[], text_encoder_refts=[], unet_refts=[reft]
    )
    inputs = _inputs()
    enabled = block(*inputs)
    assert not torch.equal(enabled, base(*inputs))
    set_enabled(network, False)
    torch.testing.assert_close(block(*inputs), base(*inputs), rtol=0, atol=0)
    set_enabled(network, True)
    torch.testing.assert_close(block(*inputs), enabled, rtol=0, atol=0)
    reft.multiplier = 0
    torch.testing.assert_close(block(*inputs), base(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("nf4", [False, True])
def test_reft_checkpoint_recomputes_intervention_with_matching_gradients(nf4):
    block = _block(nf4)
    ckpt_block = copy.deepcopy(block)
    reft = _reft(block, dropout=0.1)
    ckpt_reft = _reft(ckpt_block, dropout=0.1)
    if nf4:
        reft.bfloat16()
        ckpt_reft.bfloat16()
    ckpt_reft.load_state_dict(reft.state_dict(), strict=True)
    ckpt_block.enable_gradient_checkpointing()
    inputs = _inputs(torch.bfloat16 if nf4 else torch.float32)
    ckpt_inputs = tuple(
        t.detach().clone().requires_grad_(t.requires_grad)
        if isinstance(t, torch.Tensor)
        else t
        for t in inputs
    )
    torch.manual_seed(41)
    output = block(*inputs)
    output.float().square().mean().backward()
    saved = []
    torch.manual_seed(41)
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(tuple(t.shape)) or t, lambda t: t
    ):
        ckpt_output = ckpt_block(*ckpt_inputs)
    ckpt_output.float().square().mean().backward()
    torch.testing.assert_close(ckpt_output, output, rtol=0, atol=0)
    torch.testing.assert_close(ckpt_inputs[0].grad, inputs[0].grad, rtol=0, atol=0)
    assert (7, 32) not in saved  # ReFT's full-width Linear input is not retained.
    for actual, expected in zip(ckpt_reft.parameters(), reft.parameters()):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_reft_apply_is_idempotent_and_rejects_duplicate_adapter():
    block = _block()
    public_forward = block.forward
    reft = _reft(block)
    reft.apply_to()
    assert block.forward == public_forward
    assert block._forward == reft.forward
    second = ReFTModule("second", block, embed_dim=32)
    with pytest.raises(RuntimeError, match="already installed"):
        second.apply_to()


def test_reft_rejects_installation_after_compile(monkeypatch):
    block = _block()
    model = _model([block])
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    model.compile_blocks()
    reft = ReFTModule("late", block, embed_dim=32)
    with pytest.raises(RuntimeError, match="before compile_blocks"):
        reft.apply_to()


def test_reft_recompile_keeps_adapter_and_leaves_swapped_tail_eager(monkeypatch):
    blocks = [_block() for _ in range(4)]
    refts = [_reft(block) for block in blocks]
    model = _model(blocks, swap=2)
    targets = []

    def compile_spy(fn, **kwargs):
        targets.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", compile_spy)
    model.compile_blocks()
    model.compile_blocks()
    assert targets == [refts[0].forward, refts[1].forward] * 2
    assert not hasattr(blocks[2], "_krea_compile_base_forward")


def test_reft_aot_compile_checkpoint_and_reload_keep_gradients(monkeypatch):
    # CPU AOT cannot trace the CUDA-only cuDNN probe/fallback. Exercise the
    # real block with native CPU SDPA; Flash/Inductor is checked on hardware.
    def cpu_attention(q, k, v, *, mask, gqa, mode):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=gqa
        )
        return out.transpose(1, 2).flatten(2)

    monkeypatch.setattr(
        "library.models.krea2_raw.dit.run_krea2_attention", cpu_attention
    )
    block = _block()
    reference_block = copy.deepcopy(block)
    reft = _reft(block)
    reference = _reft(reference_block)
    reference.load_state_dict(reft.state_dict(), strict=True)
    block.enable_gradient_checkpointing()
    model = _model([block])
    model.compile_blocks(backend="aot_eager")
    inputs = _inputs()
    ref_inputs = tuple(
        t.detach().clone().requires_grad_(t.requires_grad)
        if isinstance(t, torch.Tensor)
        else t
        for t in inputs
    )
    actual = block(*inputs)
    expected = reference_block(*ref_inputs)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    for p, q in zip(reft.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    with torch.no_grad():
        reference.learned_source.weight.add_(0.01)
    state = reference.state_dict()
    state["alpha"] = torch.tensor(0.5)
    reference.load_state_dict(state, strict=True)
    reft.load_state_dict(reference.state_dict(), strict=True)
    assert reft.scale == reference.scale == 0.125
    torch.testing.assert_close(block(*inputs), reference_block(*ref_inputs))


@pytest.mark.parametrize("rank", [0, 33])
def test_reft_rejects_rank_outside_embedding_width(rank):
    with pytest.raises(ValueError, match="rank must be"):
        ReFTModule("bad", _block(), embed_dim=32, reft_dim=rank)


def test_reft_network_reload_restores_each_blocks_alpha():
    saved = torch.nn.ModuleList()
    restored = torch.nn.ModuleList()
    inputs = _inputs()
    for alpha in (0.5, 8.0):
        block = _block()
        restored_block = copy.deepcopy(block)
        source = ReFTModule("source", block, embed_dim=32, alpha=alpha)
        target = ReFTModule("target", restored_block, embed_dim=32, alpha=1)
        source.apply_to()
        target.apply_to()
        with torch.no_grad():
            source.learned_source.weight.normal_(std=0.1)
        saved.append(source)
        restored.append(target)
    restored.load_state_dict(saved.state_dict(), strict=True)
    for source, target in zip(saved, restored):
        assert target.scale == source.scale
        torch.testing.assert_close(target(*inputs), source(*inputs), rtol=0, atol=0)


def test_nf4_reft_native_state_and_optimizer_resume():
    block = _block(nf4=True)
    resumed_block = copy.deepcopy(block)
    reft = _reft(block, alpha=0.5).bfloat16()
    resumed = _reft(resumed_block).bfloat16()
    optimizer = torch.optim.AdamW(reft.parameters(), lr=1e-3)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    inputs = _inputs(torch.bfloat16)

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        loss = model(*inputs).float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        return loss.detach()

    for _ in range(3):
        step(block, optimizer)
    resumed.load_state_dict(copy.deepcopy(reft.state_dict()), strict=True)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    torch.testing.assert_close(block(*inputs), resumed_block(*inputs), rtol=0, atol=0)
    torch.testing.assert_close(
        step(block, optimizer),
        step(resumed_block, resumed_optimizer),
        rtol=0,
        atol=0,
    )
    for key, value in reft.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
