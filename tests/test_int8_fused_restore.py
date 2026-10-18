"""int8 fused block-swap restore: numerics, staging and mode plumbing."""

import pytest
import torch

from library.runtime.block_swap_config import normalize_block_swap_int8_restore_mode
from library.runtime.block_swap_masters import (
    Int8BlockSwapCpuMaster,
    _restore_int8_cpu_master_fused,
)


def _master(rows, cols, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator).mul_(0.02)
    scale = (weight.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    quantized = (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return Int8BlockSwapCpuMaster(
        quantized=quantized.contiguous(), scale=scale.float().contiguous(),
        shape=(rows, cols),
    )


def _reference(master, dtype):
    rows = master.quantized.to(torch.float32)
    return (rows * master.scale.float()[:, None]).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused dequant")
def test_fused_restore_matches_fp32_reference(dtype):
    master = _master(128, 96)
    dst = torch.empty(master.shape, dtype=dtype, device="cuda")
    _restore_int8_cpu_master_fused(master, dst, device=torch.device("cuda"), dtype=dtype)
    torch.cuda.synchronize()
    reference = _reference(master, dtype).to("cuda")
    assert torch.equal(dst, reference), "fused dequant must be bitwise identical"
    assert torch.isfinite(dst).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused dequant")
def test_fused_restore_with_staging_is_equivalent_and_allocates_nothing():
    master = _master(256, 128, seed=3)
    device = torch.device("cuda")
    dst = torch.empty(master.shape, dtype=torch.float16, device=device)
    staging = torch.empty(master.quantized.numel(), dtype=torch.int8, device=device)
    scale_staging = torch.empty(master.shape[0], dtype=torch.float32, device=device)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    _restore_int8_cpu_master_fused(
        master, dst, device=device, dtype=torch.float16,
        staging=staging, scale_staging=scale_staging,
    )
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - baseline <= 0, (
        "staged fused restore must not allocate new device memory"
    )
    assert torch.equal(dst, _reference(master, torch.float16).to(device))


def test_fused_restore_rejects_shape_mismatch():
    master = _master(32, 16)
    dst = torch.empty((16, 32), dtype=torch.float16)
    with pytest.raises(ValueError):
        _restore_int8_cpu_master_fused(master, dst, device=torch.device("cpu"),
                                       dtype=torch.float16)


def test_normalize_accepts_fused_and_alias():
    assert normalize_block_swap_int8_restore_mode("fused") == "fused"
    assert normalize_block_swap_int8_restore_mode("single_kernel") == "fused"
    assert normalize_block_swap_int8_restore_mode(None) == "fused"
