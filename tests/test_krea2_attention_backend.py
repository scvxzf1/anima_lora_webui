from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library.models.krea2_raw import attention_backend
from library.models.krea2_raw.attention_backend import (
    normalize_krea2_attention_mode,
    prepare_krea2_attention,
    run_krea2_attention,
    validate_krea2_attention_mode,
)
from library.models.krea2_raw.dit import Attention, SingleStreamDiT
from networks import attention_dispatch


class _AttentionTarget:
    def __init__(self) -> None:
        self.mode = None

    def set_attention_mode(self, mode: str) -> None:
        self.mode = mode


@pytest.mark.parametrize("value", [None, "", "torch", "TORCH"])
def test_normalize_krea2_attention_defaults_to_torch(value) -> None:
    assert normalize_krea2_attention_mode(value) == "torch"


@pytest.mark.parametrize("value", ["sdpa", "cudnn", "cudnn-sdpa"])
def test_normalize_krea2_attention_accepts_torch_aliases(value) -> None:
    assert normalize_krea2_attention_mode(value) == "torch"


def test_normalize_krea2_attention_rejects_other_backends() -> None:
    with pytest.raises(ValueError, match="supports only"):
        normalize_krea2_attention_mode("xformers")


def test_prepare_krea2_flash_validates_provider_and_dtype(monkeypatch) -> None:
    target = _AttentionTarget()
    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_available_for_dtype",
        lambda dtype: dtype == torch.bfloat16,
    )

    assert prepare_krea2_attention(target, "flash", dtype=torch.bfloat16) == "flash"
    assert target.mode == "flash"

    with pytest.raises(RuntimeError, match="fp16 or bf16"):
        validate_krea2_attention_mode("flash", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="requires"):
        validate_krea2_attention_mode("flash", dtype=torch.float16)


def test_prepare_krea2_flash_enables_dynamic_output_capture(monkeypatch) -> None:
    target = _AttentionTarget()
    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_available_for_dtype",
        lambda _dtype: True,
    )
    monkeypatch.setattr(
        torch._dynamo.config,
        "capture_dynamic_output_shape_ops",
        False,
    )

    prepare_krea2_attention(
        target,
        "flash",
        dtype=torch.float16,
        compile_enabled=True,
    )

    assert torch._dynamo.config.capture_dynamic_output_shape_ops is True


def test_prepare_krea2_attention_requires_model_setter() -> None:
    with pytest.raises(TypeError, match="set_attention_mode"):
        prepare_krea2_attention(object(), "torch", dtype=torch.bfloat16)


def test_flash_varlen_packs_batch_gqa_and_zeros_padding(monkeypatch) -> None:
    calls = SimpleNamespace()

    def fake_flash(q, k, v, cu_q, cu_k, max_q, max_k, *, softmax_scale=None):
        calls.q_shape = tuple(q.shape)
        calls.k_shape = tuple(k.shape)
        calls.cu_q = cu_q.detach().cpu().tolist()
        calls.cu_k = cu_k.detach().cpu().tolist()
        calls.max_q = max_q
        calls.max_k = max_k
        calls.scale = softmax_scale
        repeat = q.shape[1] // k.shape[1]
        return q + k.repeat_interleave(repeat, dim=1) + v.repeat_interleave(repeat, dim=1)

    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_available_for_dtype",
        lambda _dtype: True,
    )
    monkeypatch.setattr(attention_dispatch, "flash_attn_varlen_func", fake_flash)

    q = torch.randn(2, 4, 5, 3, dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 2, 5, 3, dtype=torch.float16, requires_grad=True)
    v = torch.randn(2, 2, 5, 3, dtype=torch.float16, requires_grad=True)
    valid = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    mask = valid[:, None, :, None] & valid[:, None, None, :]

    output = run_krea2_attention(
        q,
        k,
        v,
        mask=mask,
        scale=0.25,
        gqa=True,
        mode="flash",
    )
    output_4d = output.view(2, 5, 4, 3)

    assert output.shape == (2, 5, 12)
    assert calls.q_shape == (7, 4, 3)
    assert calls.k_shape == (7, 2, 3)
    assert calls.cu_q == calls.cu_k == [0, 3, 7]
    assert calls.max_q == calls.max_k == 5
    assert calls.scale == 0.25
    assert torch.count_nonzero(output_4d[~valid]) == 0

    output.sum().backward()
    for grad in (q.grad, k.grad, v.grad):
        grad_by_token = grad.permute(0, 2, 1, 3)
        assert torch.count_nonzero(grad_by_token[~valid]) == 0
        assert torch.count_nonzero(grad_by_token[valid]) > 0


def test_flash_varlen_stays_eager_inside_compiled_checkpoint(monkeypatch) -> None:
    compiling_states = []

    def fake_flash(q, k, v, *_args, **_kwargs):
        compiling_states.append(torch.compiler.is_compiling())
        repeat = q.shape[1] // k.shape[1]
        return q + k.repeat_interleave(repeat, dim=1) + v.repeat_interleave(
            repeat, dim=1
        )

    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_available_for_dtype",
        lambda _dtype: True,
    )
    monkeypatch.setattr(attention_dispatch, "flash_attn_varlen_func", fake_flash)

    valid = torch.tensor([[True, True, False, True, False]])
    mask = valid[:, None, :, None] & valid[:, None, None, :]

    def run(q, k, v, attention_mask):
        return run_krea2_attention(
            q,
            k,
            v,
            mask=attention_mask,
            gqa=True,
            mode="flash",
        )

    compiled = torch.compile(run, backend="eager", dynamic=False)
    q = torch.randn(1, 4, 5, 3, dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 2, 5, 3, dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 2, 5, 3, dtype=torch.float16, requires_grad=True)

    output = torch_checkpoint(
        compiled,
        q,
        k,
        v,
        mask,
        use_reentrant=False,
        preserve_rng_state=False,
    )
    output.float().square().mean().backward()

    assert compiling_states
    assert not any(compiling_states)
    for tensor in (output, q.grad, k.grad, v.grad):
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, 5, 5, dtype=torch.bool),
        torch.ones(2, 2, 5, 5, dtype=torch.bool),
        torch.ones(2, 1, 5, 5, dtype=torch.float32),
    ],
)
def test_flash_varlen_rejects_non_boolean_outer_product_mask(monkeypatch, mask) -> None:
    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_available_for_dtype",
        lambda _dtype: True,
    )
    q = torch.randn(2, 4, 5, 3, dtype=torch.float16)
    k = torch.randn(2, 2, 5, 3, dtype=torch.float16)

    with pytest.raises(ValueError, match="boolean"):
        run_krea2_attention(q, k, k, mask=mask, gqa=True, mode="flash")


def test_flash_varlen_rejects_invalid_gqa_head_ratio(monkeypatch) -> None:
    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_available_for_dtype",
        lambda _dtype: True,
    )
    q = torch.randn(1, 3, 4, 2, dtype=torch.float16)
    k = torch.randn(1, 2, 4, 2, dtype=torch.float16)

    with pytest.raises(ValueError, match="GQA"):
        run_krea2_attention(q, k, k, mode="flash")


def test_torch_attention_forwards_gqa_to_sdpa(monkeypatch) -> None:
    calls = {}

    def fake_sdpa(q, k, v, **kwargs):
        calls.update(kwargs)
        return q

    monkeypatch.setattr(attention_backend.F, "scaled_dot_product_attention", fake_sdpa)
    q = torch.randn(1, 4, 3, 2)
    k = torch.randn(1, 2, 3, 2)

    output = run_krea2_attention(q, k, k, gqa=True, mode="torch")

    assert output.shape == (1, 3, 8)
    assert calls["enable_gqa"] is True


def test_single_stream_dit_switches_all_attention_modules() -> None:
    model = SingleStreamDiT.__new__(SingleStreamDiT)
    torch.nn.Module.__init__(model)
    model.first_attention = Attention(dim=8, heads=2, kvheads=1)
    model.nested = torch.nn.Sequential(Attention(dim=8, heads=2, kvheads=2))

    model.set_attention_mode("flash")

    attention_modes = [
        module.attn_mode for module in model.modules() if isinstance(module, Attention)
    ]
    assert model.attn_mode == "flash"
    assert attention_modes == ["flash", "flash"]
