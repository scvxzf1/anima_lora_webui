from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_measure_bias_help_imports_adapter_loader() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/dcw/measure_bias.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "--lora_weight" in result.stdout
    assert "--dcw_sweep" in result.stdout
