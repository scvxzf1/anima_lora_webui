"""CPU-only regression tests for Anima vocab-pack TE cache identity."""

import os

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from library import cache_identity
from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy
from library.preprocess import text as preprocess_text


def _cache_tensors():
    return {
        "prompt_embeds": torch.zeros(2, 3, dtype=torch.bfloat16),
        "attn_mask": torch.ones(2, dtype=torch.int32),
        "t5_input_ids": torch.zeros(2, dtype=torch.int64),
        "t5_attn_mask": torch.ones(2, dtype=torch.int32),
        "caption_dropout_rate": torch.tensor(0.0, dtype=torch.float32),
    }


def test_vocab_pack_digest_detects_same_size_same_mtime_edit(tmp_path):
    pack = tmp_path / "jp_v1"
    pack.mkdir()
    manifest = pack / "tokenizer.json"
    manifest.write_text('{"token":"日"}', encoding="utf-8")
    original_stat = manifest.stat()
    before = cache_identity.vocab_pack_identity(pack)

    manifest.write_text('{"token":"本"}', encoding="utf-8")
    os.utime(manifest, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    after = cache_identity.vocab_pack_identity(pack)

    assert before["sha256"] != after["sha256"]


def test_anima_strategy_rejects_legacy_cache_when_vocab_pack_selected(tmp_path):
    pack = tmp_path / "jp_v1.json"
    pack.write_text('{"name":"jp_v1"}', encoding="utf-8")
    cache_path = tmp_path / "sample_anima_te.safetensors"
    save_file(
        _cache_tensors(),
        cache_path,
        metadata={"model_family": "anima", "cache_schema": "anima_te_v1"},
    )

    strategy = AnimaTextEncoderOutputsCachingStrategy(
        True,
        1,
        False,
        identity_inputs={"vocab_pack": str(pack)},
        require_cache_identity=True,
    )
    assert not strategy.is_disk_cached_outputs_expected(str(cache_path))


def test_anima_strategy_rejects_changed_vocab_pack_digest(tmp_path):
    pack = tmp_path / "jp_v1.json"
    pack.write_text('{"token":"日"}', encoding="utf-8")
    strategy = AnimaTextEncoderOutputsCachingStrategy(
        True,
        1,
        False,
        identity_inputs={"vocab_pack": str(pack)},
        require_cache_identity=True,
    )
    cache_path = tmp_path / "sample_anima_te.safetensors"
    save_file(
        _cache_tensors(),
        cache_path,
        metadata=strategy.stamp_metadata(
            {"model_family": "anima", "cache_schema": "anima_te_v1"}
        ),
    )
    assert strategy.is_disk_cached_outputs_expected(str(cache_path))

    original_stat = pack.stat()
    pack.write_text('{"token":"本"}', encoding="utf-8")
    os.utime(pack, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert not strategy.is_disk_cached_outputs_expected(str(cache_path))


def test_preprocess_reencodes_after_vocab_pack_change(tmp_path, monkeypatch):
    image = tmp_path / "cjk.png"
    Image.new("RGB", (800, 800)).save(image)
    image.with_suffix(".txt").write_text("日本語", encoding="utf-8")
    pack = tmp_path / "jp_v1.json"
    pack.write_text('{"token":"日"}', encoding="utf-8")
    calls = []

    def fake_encode(captions, *_args):
        calls.append(list(captions))
        count = len(captions)
        return (
            torch.zeros(count, 2, 3, dtype=torch.bfloat16),
            torch.ones(count, 2, dtype=torch.int32),
            torch.zeros(count, 2, dtype=torch.int64),
            torch.ones(count, 2, dtype=torch.int32),
            None,
        )

    monkeypatch.setattr(preprocess_text, "_encode_batch", fake_encode)
    kwargs = dict(
        device=torch.device("cpu"), cache_dir=tmp_path / "cache", batch_size=1,
        min_pixels=0, verbose=False, identity_inputs={"vocab_pack": str(pack)},
        require_cache_identity=True,
    )
    first = preprocess_text.cache_text_embeddings(tmp_path, object(), object(), object(), **kwargs)
    second = preprocess_text.cache_text_embeddings(tmp_path, object(), object(), object(), **kwargs)
    assert (first.written, second.skipped) == (1, 1)

    original_stat = pack.stat()
    pack.write_text('{"token":"本"}', encoding="utf-8")
    os.utime(pack, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    third = preprocess_text.cache_text_embeddings(tmp_path, object(), object(), object(), **kwargs)
    assert third.written == 1
    assert len(calls) == 2

    with safe_open(tmp_path / "cache" / "cjk_anima_te.safetensors", framework="pt") as handle:
        identity = cache_identity.read_stamp(handle.metadata())
    assert identity["vocab_pack"]["sha256"] == cache_identity.vocab_pack_identity(pack)["sha256"]
