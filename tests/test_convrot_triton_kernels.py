"""Contract checks; CUDA suite is explicitly opt-in, never a CPU-run claim.

Run on the selected device only after exclusive GPU access is established:
CONVROT_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=<UUID> .venv/bin/python -m pytest
    tests/test_convrot_triton_kernels.py
"""

import os
import subprocess
import sys

import pytest
import torch

from library.runtime.convrot import triton_kernels as kernels


def test_public_import_does_not_load_triton():
    code = """
import sys
import torch
before = {key for key in sys.modules if key == 'triton' or key.startswith('triton.')}
from library.runtime.convrot import triton_kernels
after = {key for key in sys.modules if key == 'triton' or key.startswith('triton.')}
assert after == before, after - before
assert 'library.runtime.convrot._triton_impl' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_missing_triton_is_explicit(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "library.runtime.convrot" and "_triton_impl" in kwargs.get(
            "fromlist", args[2] if len(args) > 2 else ()
        ):
            raise ImportError("simulated missing Triton")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    kernels._load_kernels.cache_clear()
    with pytest.raises(RuntimeError, match="require an installed CUDA Triton"):
        kernels._load_kernels()


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (2, 65537)])
def test_quant_invalid_shape(shape):
    with pytest.raises(ValueError, match="last dimension"):
        kernels.quantize_rowwise(torch.empty(shape))


def test_quant_invalid_dtype_and_cpu():
    with pytest.raises(TypeError, match="dtype"):
        kernels.quantize_rowwise(torch.ones(2, 3, dtype=torch.int8))
    with pytest.raises(ValueError, match="CUDA tensor"):
        kernels.quantize_rowwise(torch.ones(2, 3))


@pytest.mark.parametrize("layout", ["", "wrong", None])
def test_gemm_invalid_layout(layout):
    with pytest.raises(ValueError, match="weight_layout"):
        kernels.debug_int8_matmul_accumulator(
            torch.ones(2, 3, dtype=torch.int8),
            torch.ones(3, 4, dtype=torch.int8),
            weight_layout=layout,
        )


@pytest.mark.parametrize(
    "x_shape,w_shape", [((2, 0), (0, 3)), ((2, 3), (4, 2)), ((2, 3), (3, 0))]
)
def test_gemm_invalid_k_n(x_shape, w_shape):
    with pytest.raises(ValueError, match="K/N|K mismatch"):
        kernels.debug_int8_matmul_accumulator(
            torch.empty(x_shape, dtype=torch.int8),
            torch.empty(w_shape, dtype=torch.int8),
        )


def test_gemm_invalid_dtype_rank_and_cpu():
    x = torch.ones(2, 3, dtype=torch.int8)
    w = torch.ones(3, 4, dtype=torch.int8)
    with pytest.raises(TypeError, match="torch.int8"):
        kernels.debug_int8_matmul_accumulator(x.float(), w)
    with pytest.raises(ValueError, match="must be 2D"):
        kernels.debug_int8_matmul_accumulator(x, w.unsqueeze(0))
    with pytest.raises(ValueError, match="CUDA tensor"):
        kernels.debug_int8_matmul_accumulator(x, w)


def test_scale_contract():
    with pytest.raises(ValueError, match="shape"):
        kernels._scale(torch.ones(2), "x_scale", (2, 1), torch.device("cpu"))
    with pytest.raises(TypeError, match="dtype"):
        kernels._scale(
            torch.ones(2, dtype=torch.int32), "w_scale", (2,), torch.device("cpu")
        )
    with pytest.raises(ValueError, match="same CUDA device"):
        kernels._scale(torch.ones(2), "w_scale", (2,), torch.device("meta"))


@pytest.fixture
def cuda_device():
    if os.environ.get("CONVROT_RUN_CUDA_TESTS") != "1":
        pytest.skip("CUDA contract tests require explicit CONVROT_RUN_CUDA_TESTS=1")
    if not torch.cuda.is_available():
        pytest.fail("CUDA tests were requested but CUDA is unavailable")
    pytest.importorskip("triton")
    return torch.device("cuda:0")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [1, 65, 256, 8192])
def test_cuda_quant_matches_reference(cuda_device, dtype, k):
    from library.runtime.convrot.gemm import quantize_activation_absmax_int8

    gen = torch.Generator().manual_seed(19)
    x = torch.randn(2, 5, k * 2, generator=gen).to(device=cuda_device, dtype=dtype)[
        ..., ::2
    ]
    x[0, 0] = 0
    x[0, 1] = 1.0e-20
    x[0, 2, 0] = -100
    q, scale = kernels.quantize_rowwise(x)
    q_ref, scale_ref = quantize_activation_absmax_int8(x)
    assert torch.equal(scale, scale_ref)
    assert torch.equal(q, q_ref)
    assert q.shape == x.shape and scale.shape == (2, 5, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_quant_round_ties(cuda_device, dtype):
    x = torch.tensor(
        [[-127, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 127]],
        device=cuda_device,
        dtype=dtype,
    )
    q, scale = kernels.quantize_rowwise(x)
    assert torch.equal(scale, torch.ones_like(scale))
    assert q.cpu().tolist() == [[-127, -4, -2, -2, 0, 0, 2, 2, 4, 127]]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cuda_nonfinite_diagnostic(cuda_device, value):
    with pytest.raises(ValueError, match="NaN or Inf"):
        kernels.quantize_rowwise(
            torch.tensor([[value]], device=cuda_device), check_finite=True
        )


@pytest.mark.parametrize("m", [1, 16, 17, 4032, 4200])
@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_gemm_integer_and_scale_tails(cuda_device, m, layout, dtype):
    gen = torch.Generator().manual_seed(31)
    k, n = 65, 67
    x_cpu = torch.randint(-128, 128, (m, k), dtype=torch.int8, generator=gen)
    w_cpu = torch.randint(-128, 128, (k, n), dtype=torch.int8, generator=gen)
    expected_acc = (x_cpu.to(torch.int64) @ w_cpu.to(torch.int64)).to(torch.int32)
    x = torch.empty(m, k * 2, dtype=torch.int8, device=cuda_device)
    x[:, ::2] = x_cpu.to(cuda_device)
    x = x[:, ::2]
    w = w_cpu.to(cuda_device)
    if layout == "nk":
        w = w.t()  # Preserve noncontiguous weight layout.
    sx = (torch.rand(m, 1, generator=gen) / 127).to(cuda_device)
    sw = (torch.rand(n, generator=gen) / 127).to(
        device=cuda_device, dtype=torch.bfloat16
    )
    acc = kernels.debug_int8_matmul_accumulator(x, w, weight_layout=layout)
    assert acc.dtype == torch.int32
    assert torch.equal(acc.cpu(), expected_acc)
    expected = expected_acc.to(cuda_device).float().mul(sx).mul(sw.float()).to(dtype)
    result = kernels.int8_matmul_scaled(
        x, sx, w, sw, weight_layout=layout, out_dtype=dtype
    )
    assert torch.equal(result, expected)


def test_cuda_leading_dims_bias_and_empty(cuda_device):
    x = torch.ones(2, 17, 33, dtype=torch.int8, device=cuda_device)
    w = torch.ones(33, 35, dtype=torch.int8, device=cuda_device)
    sx = torch.full((2, 17, 1), 0.3, device=cuda_device)
    sw = torch.full((35,), 0.7, device=cuda_device)
    bias = torch.linspace(-0.5, 0.5, 35, device=cuda_device)
    expected = (
        (torch.full((2, 17, 35), 33.0, device=cuda_device) * sx) * sw + bias
    ).bfloat16()
    result = kernels.int8_matmul_scaled(x, sx, w, sw, bias=bias)
    assert torch.equal(result, expected)
    q, scale = kernels.quantize_rowwise(torch.empty(0, 33, device=cuda_device))
    assert q.shape == (0, 33) and scale.shape == (0, 1)
    result = kernels.int8_matmul_scaled(q, scale, w, sw)
    assert result.shape == (0, 35)
