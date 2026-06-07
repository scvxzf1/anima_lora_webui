from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cuda_track_stays_bitsandbytes_compatible():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cu130" in pyproject
    assert "https://download.pytorch.org/whl/cu132" not in pyproject
    assert "bitsandbytes>=0.49.2" in pyproject

    assert 'name = "bitsandbytes"' in lock
    assert 'name = "cuda-toolkit"\nversion = "13.0.2"' in lock
    assert "flash_attn-2.8.3+cu130torch2.12" in lock
    assert "flash_attn-2.8.3+cu132torch2.12" not in lock
