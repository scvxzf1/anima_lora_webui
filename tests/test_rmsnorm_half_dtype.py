"""RMSNorm must restore input half dtype before affine (E: QKV source dtype)."""

from __future__ import annotations

import torch

from library.anima.models import RMSNorm


def test_rmsnorm_preserves_half_dtype() -> None:
    norm = RMSNorm(16)
    for dtype in (torch.bfloat16, torch.float16):
        x = torch.randn(2, 4, 16, dtype=dtype)
        y = norm(x)
        assert y.dtype == dtype
        assert y.shape == x.shape
        assert torch.isfinite(y.float()).all()


def test_rmsnorm_matches_legacy_fp32_affine_up_to_half_rounding() -> None:
    """New path casts before weight mul; legacy did (fp32_norm * w).to(x).

    Difference is only half rounding on the final mul — stay within a tight band.
    """
    torch.manual_seed(0)
    norm = RMSNorm(32)
    x = torch.randn(3, 8, 32, dtype=torch.bfloat16)
    y_new = norm(x)

    # Legacy formula from pre-E models.RMSNorm.forward
    output = norm._norm(x.float())
    y_legacy = (output * norm.weight).to(x.dtype)

    rel = (y_new.float() - y_legacy.float()).norm() / y_legacy.float().norm().clamp_min(
        1e-8
    )
    assert rel.item() < 1e-3
