"""Focused contracts for Anima per-band dynamic-sequence compilation."""

from __future__ import annotations

import pytest
import torch

from library.datasets.buckets import (
    band_for_seq,
    cluster_token_bands,
    widen_bands,
)


def test_cluster_token_bands_is_data_driven() -> None:
    assert cluster_token_bands([]) == []
    assert cluster_token_bands([4200, 4032, 4200, 4032]) == [(4032, 4200)]
    assert cluster_token_bands([3000, 3024, 3600, 4032, 4200]) == [
        (3000, 3024),
        (3600, 3600),
        (4032, 4200),
    ]


def test_cluster_token_bands_splits_only_above_relative_gap() -> None:
    # 10% exactly stays in the same band; a larger gap starts a new one.
    assert cluster_token_bands([100, 110], rel_gap=0.10) == [(100, 110)]
    assert cluster_token_bands([100, 111], rel_gap=0.10) == [(100, 100), (111, 111)]


def test_band_for_seq_handles_edges_and_gaps() -> None:
    bands = [(3000, 3024), (4032, 4200)]
    assert band_for_seq(bands, 3000) == (3000, 3024)
    assert band_for_seq(bands, 3024) == (3000, 3024)
    assert band_for_seq(bands, 4032) == (4032, 4200)
    assert band_for_seq(bands, 4200) == (4032, 4200)
    assert band_for_seq(bands, 3500) is None
    assert band_for_seq(bands, 2000) is None
    assert band_for_seq(bands, 5000) is None
    assert band_for_seq([], 4032) is None


def test_widen_bands_only_grows_upper_bounds_and_rejects_touching() -> None:
    bands = [(3000, 3024), (4032, 4200)]
    assert widen_bands(bands, 16) == [(3000, 3040), (4032, 4216)]
    assert widen_bands(bands, 0) == bands
    assert widen_bands(bands, -4) == bands
    with pytest.raises(ValueError, match="inter-band gap"):
        widen_bands(bands, 4032 - 3024)


def test_compile_signature_is_stable_and_bands_change_identity() -> None:
    from library.runtime.harness import compile_signature

    common = dict(
        n_token_families=2,
        seq_range=(3000, 4200),
        dynamic_seq=True,
        backend="inductor",
    )
    classic = compile_signature(**common, mode=None)
    assert classic == (
        "families=2;seq_range=(3000, 4200);dynamic_seq=True;backend=inductor;mode=None"
    )
    assert compile_signature(**common, mode="", seq_bands=None) == classic
    per_band = compile_signature(
        **common,
        mode=None,
        seq_bands=[(4032, 4200), (3000, 3024)],
    )
    assert per_band.endswith(";seq_bands=[(3000, 3024), (4032, 4200)]")
    assert per_band != classic


def test_anima_compat_disables_bands_without_dynamic_sequence() -> None:
    from library.training.compat_matrix import check_training_compat

    result = check_training_compat(
        {
            "model_family": "anima",
            "compile_dynamic_seq": False,
            "compile_seq_bands": True,
        }
    )

    assert result.ok
    assert {item.code for item in result.warnings} == {
        "compile_seq_bands_requires_dynamic_seq"
    }
    assert [(item.key, item.value) for item in result.mutations] == [
        ("compile_seq_bands", False)
    ]


class _BandFakeUnet:
    patch_spatial = 2
    vae_spatial_compression = 8

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def compile_blocks(self, backend, **kwargs):
        self.calls.append({"backend": backend, **kwargs})
        self._dynamic_seq = bool(kwargs["dynamic_seq"])
        self._dynamic_seq_range = kwargs["seq_range"]
        self._dynamic_seq_bands = kwargs.get("seq_bands")


class _BandFakeNetwork:
    extra_seq_tokens = 0


def test_training_harness_forwards_and_stamps_bands(tmp_path, monkeypatch) -> None:
    from library.runtime import harness

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_compile_cache_base", None)
    unet = _BandFakeUnet()
    bands = [(3000, 3024), (4032, 4200)]

    harness.compile_blocks_for_training(
        unet,
        _BandFakeNetwork(),
        backend="eager",
        bucket_resolutions=[(896, 1152), (960, 1120)],
        n_token_families=2,
        seq_range=(3000, 4200),
        seq_bands=bands,
        dynamic_seq=True,
    )

    assert unet.calls[0]["seq_bands"] == bands
    assert unet._training_compile_config["seq_bands"] == bands
    assert unet._training_compile_config["seq_range"] == (3000, 4200)


def test_training_harness_adds_new_sample_as_singleton_band(
    tmp_path, monkeypatch
) -> None:
    from library.runtime import harness

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(harness, "_compile_cache_base", None)
    unet = _BandFakeUnet()
    network = _BandFakeNetwork()
    harness.compile_blocks_for_training(
        unet,
        network,
        backend="eager",
        n_token_families=2,
        seq_range=(3000, 4200),
        seq_bands=[(3000, 3024), (4032, 4200)],
        dynamic_seq=True,
    )

    assert harness.ensure_training_compile_seq_range(unet, network, {3600}) is True
    assert unet.calls[-1]["seq_bands"] == [
        (3000, 3024),
        (3600, 3600),
        (4032, 4200),
    ]
    assert unet._training_compile_config["seq_bands"] == unet.calls[-1]["seq_bands"]
    assert harness.ensure_training_compile_seq_range(unet, network, {3600}) is False


def test_dynamic_seq_wrapper_dispatches_to_containing_band(monkeypatch) -> None:
    from library.anima import models

    marks: list[tuple[int, int, int]] = []

    def fake_mark(tensor, axis, lo, hi, *, strict=False):
        del tensor
        marks.append((axis, lo, hi))
        assert strict is True

    monkeypatch.setattr(models, "_mark_seq_axis_dynamic", fake_mark)
    calls: list[int] = []

    def compiled(x, *args):
        calls.append(int(x.shape[2]))
        return x

    wrapped = models._make_dynamic_seq_forward(
        compiled,
        [(100, 110), (200, 210)],
        strict_marks=True,
    )
    x = torch.zeros(1, 1, 105, 1, 4)
    rope = (torch.zeros(105, 1, 1, 1), torch.zeros(105, 1, 1, 1))
    assert wrapped(x, None, None, None, rope) is x
    assert calls == [105]
    assert marks == [(2, 100, 110), (0, 100, 110), (0, 100, 110)]

    # A gap is intentionally left unmarked in strict per-band mode.
    marks.clear()
    x_gap = torch.zeros(1, 1, 150, 1, 4)
    rope_gap = (torch.zeros(150, 1, 1, 1), torch.zeros(150, 1, 1, 1))
    wrapped(x_gap, None, None, None, rope_gap)
    assert marks == []


def test_dynamic_seq_wrapper_keeps_legacy_union_call_form() -> None:
    from library.anima.models import _make_dynamic_seq_forward

    wrapped = _make_dynamic_seq_forward(
        lambda x, *args: x,
        100,
        200,
    )
    x = torch.zeros(1, 1, 150, 1, 4)
    assert wrapped(x, None, None, None, None) is x


def test_dynamic_seq_wrapper_rejects_invalid_bands() -> None:
    from library.anima.models import _make_dynamic_seq_forward

    with pytest.raises(ValueError, match="invalid dynamic-seq token band"):
        _make_dynamic_seq_forward(lambda x, *args: x, [(100, 99)])
    with pytest.raises(ValueError, match="non-overlapping"):
        _make_dynamic_seq_forward(lambda x, *args: x, [(100, 150), (140, 200)])
