"""Probe imports must preserve the supervisor's explicit device selection."""

import os
from pathlib import Path
import subprocess
import sys


def test_krea_ablation_import_preserves_external_gpu_uuid():
    root = Path(__file__).resolve().parents[1]
    selected = "GPU-supervisor-selected-device"
    result = subprocess.run(
        [sys.executable, "-c",
         "import os; import scripts.krea2.probe_nf4_ablation; "
         f"assert os.environ['CUDA_VISIBLE_DEVICES'] == {selected!r}; "
         "assert os.environ['CUDA_DEVICE_ORDER'] == 'FASTEST_FIRST'"],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": selected,
             "CUDA_DEVICE_ORDER": "FASTEST_FIRST", "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
