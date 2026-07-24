"""Regression: sample_prompts path resolve under externalized configs_root."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from library.training.train_bootstrap import (
    normalize_sample_args,
    resolve_sample_prompts_path,
)


def test_resolve_sample_prompts_path_via_configs_root(
    tmp_path: Path, monkeypatch
) -> None:
    configs = tmp_path / "anima-配置"
    prompt = configs / "sample-prompts" / "imported" / "demo.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("1girl --w 1024 --h 1024\n", encoding="utf-8")
    monkeypatch.setattr(
        "library.env.get_configs_root",
        lambda: configs,
    )

    # Portable configs/ relative path must not require cwd-relative existence.
    assert not Path("configs/sample-prompts/imported/demo.txt").is_file()
    resolved = resolve_sample_prompts_path(
        "configs/sample-prompts/imported/demo.txt"
    )
    assert resolved is not None
    assert Path(resolved).resolve() == prompt.resolve()


def test_normalize_sample_args_keeps_external_configs_path(
    tmp_path: Path, monkeypatch
) -> None:
    configs = tmp_path / "anima-配置"
    prompt = configs / "sample-prompts" / "imported" / "demo.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("1girl --w 1024 --h 1024\n", encoding="utf-8")
    monkeypatch.setattr(
        "library.env.get_configs_root",
        lambda: configs,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = SimpleNamespace(
        sample_prompts="configs/sample-prompts/imported/demo.txt",
        sample_every_n_steps=100,
        sample_every_n_epochs=None,
        sample_at_first=False,
        output_dir=str(out_dir),
    )
    normalize_sample_args(args)
    assert Path(args.sample_prompts).resolve() == prompt.resolve()
    # Must NOT treat the path string as an inline prompt.
    bogus = out_dir / "sample_prompts.txt"
    assert not bogus.exists()


def test_normalize_sample_args_still_writes_inline_prompts(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    args = SimpleNamespace(
        sample_prompts="1girl, solo --w 512 --h 512",
        sample_every_n_steps=50,
        sample_every_n_epochs=None,
        sample_at_first=False,
        output_dir=str(out_dir),
    )
    normalize_sample_args(args)
    written = Path(args.sample_prompts)
    assert written.name == "sample_prompts.txt"
    assert "1girl, solo" in written.read_text(encoding="utf-8")
