from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.experimental_tasks import inference


def test_resolve_ref_image_prefers_env_without_consuming_extra(monkeypatch):
    monkeypatch.setenv("REF_IMAGE", "env/ref.png")
    monkeypatch.setattr(inference, "_random_ref_image", lambda directory: None)

    ref_image, extra = inference._resolve_ref_image(
        ["pos/ref.png", "--seed", "1"],
        Path("/fallback"),
        usage="USAGE",
    )

    assert ref_image == "env/ref.png"
    assert extra == ["pos/ref.png", "--seed", "1"]


def test_resolve_ref_image_consumes_first_positional(monkeypatch):
    monkeypatch.delenv("REF_IMAGE", raising=False)
    monkeypatch.setattr(inference, "_random_ref_image", lambda directory: None)

    ref_image, extra = inference._resolve_ref_image(
        ["pos/ref.png", "--seed", "1"],
        Path("/fallback"),
        usage="USAGE",
    )

    assert ref_image == "pos/ref.png"
    assert extra == ["--seed", "1"]


def test_resolve_ref_image_keeps_flag_first_extra_for_fallback(monkeypatch):
    calls: list[Path] = []

    def fake_random(directory: Path) -> str:
        calls.append(directory)
        return "fallback/ref.png"

    monkeypatch.delenv("REF_IMAGE", raising=False)
    monkeypatch.setattr(inference, "_random_ref_image", fake_random)

    ref_image, extra = inference._resolve_ref_image(
        ["--seed", "1"],
        Path("/fallback"),
        usage="USAGE",
    )

    assert ref_image == "fallback/ref.png"
    assert extra == ["--seed", "1"]
    assert calls == [Path("/fallback")]


def test_resolve_ref_image_exits_when_no_source(monkeypatch, capsys):
    monkeypatch.delenv("REF_IMAGE", raising=False)
    monkeypatch.setattr(inference, "_random_ref_image", lambda directory: None)

    with pytest.raises(SystemExit) as exc:
        inference._resolve_ref_image([], Path("/empty"), usage="USAGE")

    assert exc.value.code == 1
    assert capsys.readouterr().err == f"USAGE{os.linesep}"


def test_copy_latest_png_sidecar_uses_latest_non_sidecar(tmp_path, capsys):
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    source = tmp_path / "refs" / "source.png"
    source.parent.mkdir()
    source.write_bytes(b"source")
    older = save_dir / "older.png"
    older.write_bytes(b"older")
    latest = save_dir / "latest.png"
    latest.write_bytes(b"latest")
    ignored_sidecar = save_dir / "ignored_ref.png"
    ignored_sidecar.write_bytes(b"ignored")
    os.utime(older, (10, 10))
    os.utime(latest, (20, 20))
    os.utime(ignored_sidecar, (30, 30))

    copied = inference._copy_latest_png_sidecar(
        save_dir,
        str(source),
        suffix="_ref.png",
        message="Ref pasted",
    )

    assert copied == save_dir / "latest_ref.png"
    assert copied.read_bytes() == b"source"
    assert capsys.readouterr().out == f"  > Ref pasted: {copied}{os.linesep}"


def test_copy_latest_png_sidecar_noops_without_generated_png(tmp_path, capsys):
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    source = tmp_path / "refs" / "source.png"
    source.parent.mkdir()
    source.write_bytes(b"source")
    existing_sidecar = save_dir / "only_src.png"
    existing_sidecar.write_bytes(b"old")

    copied = inference._copy_latest_png_sidecar(
        save_dir,
        str(source),
        suffix="_src.png",
        message="Source pasted",
    )

    assert copied is None
    assert capsys.readouterr().out == ""


def test_measure_bias_help_imports_adapter_loader() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/dcw/measure_bias.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "--lora_weight" in result.stdout
    assert "--dcw_sweep" in result.stdout
