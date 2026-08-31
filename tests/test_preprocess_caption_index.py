from __future__ import annotations

import subprocess

import pytest

from scripts.tasks import downloads, preprocess


def test_preprocess_skips_caption_index_without_implicit_download(
    monkeypatch, tmp_path, capsys
):
    vocab = tmp_path / "missing-vocab.json"
    download_called = False
    caption_index_called = False

    def fail_download(_extra):
        nonlocal download_called
        download_called = True

    def fail_caption_index(_extra):
        nonlocal caption_index_called
        caption_index_called = True

    monkeypatch.setattr(downloads, "cmd_download_tagger", fail_download)
    monkeypatch.setattr(preprocess, "_path", lambda _key, _default: str(vocab))
    monkeypatch.setattr(preprocess, "cmd_caption_index", fail_caption_index)

    preprocess._build_caption_index_best_effort()

    assert download_called is False
    assert caption_index_called is False
    output = capsys.readouterr().out
    assert "skipping caption-index" in output
    assert "python tasks.py download-tagger" in output


def test_preprocess_builds_caption_index_when_vocab_exists(monkeypatch, tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(preprocess, "_path", lambda _key, _default: str(vocab))
    monkeypatch.setattr(preprocess, "cmd_caption_index", calls.append)

    preprocess._build_caption_index_best_effort()

    assert calls == [[]]


def test_download_tagger_applies_request_and_total_timeouts(monkeypatch, tmp_path):
    calls: list[tuple[list[str], dict]] = []

    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(
        downloads,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    downloads.cmd_download_tagger([])

    command, kwargs = calls[0]
    assert command[:4] == [
        "hf",
        "download",
        "sorryhyun/anima-tagger",
        "v2/vocab.json",
    ]
    assert kwargs["timeout"] == downloads._TAGGER_DOWNLOAD_TIMEOUT_SECONDS
    assert kwargs["env"]["HF_HUB_ETAG_TIMEOUT"] == "30"
    assert kwargs["env"]["HF_HUB_DOWNLOAD_TIMEOUT"] == "30"


def test_download_tagger_reports_total_timeout(monkeypatch, tmp_path, capsys):
    def time_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", time_out)

    with pytest.raises(SystemExit, match="124"):
        downloads.cmd_download_tagger([])

    assert "download timed out after 120s" in capsys.readouterr().out
