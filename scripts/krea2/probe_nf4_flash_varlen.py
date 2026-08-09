#!/usr/bin/env python3
"""Run the unified NF4 probe with the Krea-2 FlashAttention varlen backend.

This wrapper selects the production ``attn_mode=flash`` path while retaining the
unified NF4 ablation harness and its metrics.

Example::

    K2_ABL_GPU=1 K2_ABL_IMG=1024 K2_ABL_STEPS=20 \
      K2_ABL_NF4=1 K2_ABL_SWAP=0 K2_ABL_GRAD_CKPT=full \
      K2_ABL_COMPILE=1 K2_ABL_TE_CPU=1 \
      .venv/bin/python scripts/krea2/probe_nf4_flash_varlen.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_ABL_GPU", "1"))

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["K2_ABL_ATTN_MODE"] = "flash"

from probe_nf4_ablation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
