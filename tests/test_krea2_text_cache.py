"""Krea-2 text-cache dropout and caption-variant contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library.datasets.dataset_getitem_sample import load_text_fields
from library.models.krea2_raw import strategy as krea_strategy_module
from library.models.krea2_raw.family import prepare_text_embedding_for_training
from library.models.krea2_raw.strategy import (
    Krea2TextEncoderOutputsCachingStrategy,
    Krea2TextEncodingStrategy,
)
from library.preprocess.caption_cache_settings import (
    DEFAULT_CAPTION_SHUFFLE_VARIANTS,
    DEFAULT_CAPTION_TAG_DROPOUT_RATE,
    resolve_caption_cache_settings,
)
from library.training.anima_strategies import (
    get_text_encoder_outputs_caching_strategy,
)
from scripts.krea2.preprocess_te_cache import _iter_caption_sources


def _write_variant_cache(
    path: Path,
    *,
    multi_source: bool = False,
    v0_intact: bool = True,
) -> None:
    tensors = {
        "num_variants": torch.tensor(3, dtype=torch.int64),
        "caption_shuffle_variants": torch.tensor(3, dtype=torch.int64),
        "caption_tag_dropout_rate": torch.tensor(0.25, dtype=torch.float32),
        "caption_dropout_rate": torch.tensor(0.0, dtype=torch.float32),
    }
    if v0_intact:
        tensors["v0_intact"] = torch.tensor(1, dtype=torch.int8)
    if multi_source:
        tensors["caption_multi_source"] = torch.tensor(1, dtype=torch.int8)
    for index in range(3):
        tensors[f"hiddens_v{index}"] = torch.full(
            (2, 1, 1), float(index), dtype=torch.bfloat16
        )
        tensors[f"mask_v{index}"] = torch.full((2,), True, dtype=torch.bool)
    save_file(tensors, str(path))


def test_legacy_single_variant_cache_remains_readable(tmp_path: Path) -> None:
    cache_path = tmp_path / "legacy.safetensors"
    save_file(
        {
            "hiddens": torch.ones(2, 1, 1, dtype=torch.bfloat16),
            "mask": torch.ones(2, dtype=torch.bool),
            "caption_dropout_rate": torch.tensor(0.0),
        },
        str(cache_path),
    )
    strategy = Krea2TextEncoderOutputsCachingStrategy()

    hiddens, mask, rate = strategy.load_outputs_npz(str(cache_path))

    assert strategy.is_disk_cached_outputs_expected(str(cache_path))
    assert torch.equal(hiddens, torch.ones_like(hiddens))
    assert mask.all()
    assert float(rate) == 0.0


def test_shuffle_enabled_rejects_single_variant_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "single.safetensors"
    save_file(
        {
            "hiddens": torch.ones(2, 1, 1, dtype=torch.bfloat16),
            "mask": torch.ones(2, dtype=torch.bool),
            "caption_dropout_rate": torch.tensor(0.0),
        },
        str(cache_path),
    )
    strategy = Krea2TextEncoderOutputsCachingStrategy(
        use_shuffled_caption_variants=True
    )

    assert not strategy.is_disk_cached_outputs_expected(str(cache_path))


def test_krea_cache_factory_receives_variant_selection_flag() -> None:
    args = SimpleNamespace(
        model_family="krea2_raw",
        cache_text_encoder_outputs=True,
        cache_text_encoder_outputs_to_disk=True,
        text_encoder_batch_size=2,
        skip_cache_check=False,
        use_shuffled_caption_variants=True,
    )

    strategy = get_text_encoder_outputs_caching_strategy(args, torch.bfloat16)

    assert isinstance(strategy, Krea2TextEncoderOutputsCachingStrategy)
    assert strategy.use_shuffled_caption_variants is True


def test_shuffled_variant_selection_pins_v0_when_disabled(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "variants.safetensors"
    _write_variant_cache(cache_path)
    strategy = Krea2TextEncoderOutputsCachingStrategy(
        use_shuffled_caption_variants=False
    )

    hiddens, _, _ = strategy.load_outputs_npz(str(cache_path))

    assert torch.equal(hiddens, torch.zeros_like(hiddens))


def test_shuffled_variant_selection_uses_nonzero_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "variants.safetensors"
    _write_variant_cache(cache_path)
    monkeypatch.setattr(krea_strategy_module.random, "random", lambda: 0.9)
    monkeypatch.setattr(krea_strategy_module.random, "randint", lambda _a, b: b)
    strategy = Krea2TextEncoderOutputsCachingStrategy(
        use_shuffled_caption_variants=True
    )

    hiddens, _, _ = strategy.load_outputs_npz(str(cache_path))

    assert torch.equal(hiddens, torch.full_like(hiddens, 2.0))


def test_multi_source_selection_is_uniform_even_when_shuffle_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "multi.safetensors"
    _write_variant_cache(cache_path, multi_source=True)
    monkeypatch.setattr(krea_strategy_module.random, "randint", lambda _a, b: b)
    strategy = Krea2TextEncoderOutputsCachingStrategy(
        use_shuffled_caption_variants=False
    )

    hiddens, _, _ = strategy.load_outputs_npz(str(cache_path))

    assert torch.equal(hiddens, torch.full_like(hiddens, 2.0))


class _FakeTokenizeStrategy:
    def tokenize(self, captions):
        return [list(captions)]


class _FakeEncodingStrategy:
    def encode_tokens(self, _tokenize_strategy, _models, tokens):
        captions = tokens[0]
        hiddens = torch.tensor(
            [float(len(caption)) for caption in captions],
            dtype=torch.bfloat16,
        ).reshape(-1, 1, 1, 1)
        return [hiddens, torch.ones(len(captions), 1, dtype=torch.bool)]


def test_variant_cache_writer_stamps_generation_settings(tmp_path: Path) -> None:
    cache_path = tmp_path / "written.safetensors"
    info = SimpleNamespace(
        caption="a",
        caption_variants=["a", "longer"],
        cache_caption_variants=True,
        caption_multi_source=False,
        caption_shuffle_variants=2,
        caption_tag_dropout_rate=0.25,
        caption_dropout_rate=0.0,
        text_encoder_outputs_npz=str(cache_path),
    )
    strategy = Krea2TextEncoderOutputsCachingStrategy(batch_size=1)

    strategy.cache_batch_outputs(
        _FakeTokenizeStrategy(),
        [object()],
        _FakeEncodingStrategy(),
        [info],
    )

    with safe_open(str(cache_path), framework="pt") as handle:
        keys = set(handle.keys())
        assert int(handle.get_tensor("num_variants")) == 2
        assert int(handle.get_tensor("caption_shuffle_variants")) == 2
        assert float(handle.get_tensor("caption_tag_dropout_rate")) == pytest.approx(
            0.25
        )
    assert {"v0_intact", "hiddens_v0", "hiddens_v1", "mask_v0", "mask_v1"} <= keys
    assert strategy.is_disk_cached_outputs_expected(
        str(cache_path),
        expected_num_variants=2,
        expected_caption_shuffle_variants=2,
        expected_caption_tag_dropout_rate=0.25,
        expected_multi_source=False,
    )
    assert not strategy.is_disk_cached_outputs_expected(
        str(cache_path),
        expected_num_variants=2,
        expected_caption_shuffle_variants=2,
        expected_caption_tag_dropout_rate=0.5,
        expected_multi_source=False,
    )


def test_in_memory_variant_cache_selects_on_each_sample_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = SimpleNamespace(
        caption="a",
        caption_variants=["a", "longer"],
        cache_caption_variants=True,
        caption_multi_source=False,
        caption_shuffle_variants=2,
        caption_tag_dropout_rate=0.0,
        caption_dropout_rate=0.25,
        text_encoder_outputs_npz=None,
        text_encoder_outputs=None,
    )
    strategy = Krea2TextEncoderOutputsCachingStrategy(
        cache_to_disk=False,
        batch_size=2,
        use_shuffled_caption_variants=True,
    )
    strategy.cache_batch_outputs(
        _FakeTokenizeStrategy(),
        [object()],
        _FakeEncodingStrategy(),
        [info],
    )
    monkeypatch.setattr(krea_strategy_module.random, "random", lambda: 0.9)
    monkeypatch.setattr(krea_strategy_module.random, "randint", lambda _a, b: b)
    dataset = SimpleNamespace(text_encoder_output_caching_strategy=strategy)

    fields = load_text_fields(
        dataset,
        info,
        SimpleNamespace(),
        sampler=None,
        target_stem="sample",
        strip_identity=False,
    )

    hiddens, _, rate = fields["text_encoder_outputs"]
    assert torch.equal(hiddens, torch.full_like(hiddens, 6.0))
    assert float(rate) == pytest.approx(0.25)


def test_loaded_cache_uses_current_subset_dropout_rate() -> None:
    cached = (
        torch.ones(2, 1, 1),
        torch.ones(2, dtype=torch.bool),
        torch.tensor(0.0),
    )
    image_info = SimpleNamespace(
        caption="caption",
        caption_dropout_rate=0.35,
        text_encoder_outputs=cached,
        text_encoder_outputs_npz=None,
    )
    dataset = SimpleNamespace(
        text_encoder_output_caching_strategy=SimpleNamespace(is_partial=False),
    )

    fields = load_text_fields(
        dataset,
        image_info,
        SimpleNamespace(),
        sampler=None,
        target_stem="sample",
        strip_identity=False,
    )

    assert float(fields["text_encoder_outputs"][-1]) == pytest.approx(0.35)
    assert float(cached[-1]) == 0.0


def test_krea_training_prep_applies_dropout_without_mutating_cache() -> None:
    cached_hiddens = torch.ones(2, 3, 1, 2)
    cached_mask = torch.ones(2, 3, dtype=torch.bool)

    hiddens, mask = prepare_text_embedding_for_training(
        [cached_hiddens, cached_mask],
        {"caption_dropout_rates": torch.tensor([1.0, 0.0])},
        Krea2TextEncodingStrategy(),
        device=torch.device("cpu"),
        weight_dtype=torch.float32,
    )

    assert torch.count_nonzero(hiddens[0]) == 0
    assert not mask[0].any()
    assert torch.equal(hiddens[1], torch.ones_like(hiddens[1]))
    assert mask[1].all()
    assert cached_hiddens.bool().all()
    assert cached_mask.all()


def test_caption_cache_settings_use_env_then_config_then_defaults() -> None:
    assert resolve_caption_cache_settings({}, {}) == (
        DEFAULT_CAPTION_SHUFFLE_VARIANTS,
        DEFAULT_CAPTION_TAG_DROPOUT_RATE,
    )
    assert resolve_caption_cache_settings(
        {"caption_shuffle_variants": 2, "caption_tag_dropout_rate": 0.3},
        {},
    ) == (2, 0.3)
    assert resolve_caption_cache_settings(
        {"caption_shuffle_variants": 2, "caption_tag_dropout_rate": 0.3},
        {
            "CAPTION_SHUFFLE_VARIANTS": "6",
            "CAPTION_TAG_DROPOUT_RATE": "0.05",
        },
    ) == (6, 0.05)


def test_krea_caption_source_preserves_captions_json_variants(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.png").write_bytes(b"not-decoded-when-min-pixels-zero")
    (tmp_path / "captions.json").write_text(
        '{"sample.png": ["first", "second"]}',
        encoding="utf-8",
    )

    items = list(
        _iter_caption_sources(
            tmp_path,
            recursive=False,
            path_pattern="*",
            min_pixels=0,
            prefer_json_caption=False,
            caption_source_mode="captions_json",
            caption_extension=".txt",
        )
    )

    assert len(items) == 1
    assert items[0][1].caption_texts() == ["first", "second"]
