"""Cache identity, variants and retry integration using tiny CPU encoders."""

from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from library.preprocess import adaptive_batch, latents, text
from library.preprocess.batch_policy import AutoBatchPolicy


@pytest.fixture
def simulated_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    original = adaptive_batch.AutoBatcher
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *_: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 200)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def factory(device, **kwargs):
        runner = original(device, **kwargs)
        runner.cuda = True
        runner.policy = AutoBatchPolicy(16, predict=False, throughput=False)
        runner._budget = lambda: 10000
        return runner

    monkeypatch.setattr(adaptive_batch, "AutoBatcher", factory)
    monkeypatch.setattr(latents, "AutoBatcher", factory)
    monkeypatch.setattr(text, "AutoBatcher", factory)


class TinyVAE:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, fail_above=100):
        self.fail_above = fail_above
        self.calls = []

    def encode_pixels_to_latents(self, images):
        self.calls.append((len(images), tuple(images.shape[-2:])))
        if len(images) > self.fail_above:
            raise torch.cuda.OutOfMemoryError("injected VAE OOM")
        return images[:, :, ::8, ::8]


def images_in(path, count=24, size=(16, 16)):
    path.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", size, (index, index, index)).save(path / f"{index:03}.png")
        (path / f"{index:03}.txt").write_text(str(index))


def test_vae_auto_matches_fixed_and_skips_existing(tmp_path, simulated_cuda):
    source = tmp_path / "src"
    images_in(source, 40)
    images_in(source / "wide", 12, (32, 16))
    fixed = tmp_path / "fixed"
    auto = tmp_path / "auto"
    fixed.mkdir()
    auto.mkdir()
    latents.cache_latents(
        source, TinyVAE(), cache_dir=fixed, recursive=True, batch_size=2
    )
    vae = TinyVAE(fail_above=2)
    stats = latents.cache_latents(
        source, vae, cache_dir=auto, recursive=True, batch_size="auto"
    )
    assert stats.written == 52
    assert max(batch for batch, _ in vae.calls) > 2
    for expected in fixed.rglob("*.npz"):
        with np.load(expected) as a, np.load(auto / expected.relative_to(fixed)) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    calls = len(vae.calls)
    skipped = latents.cache_latents(
        source, vae, cache_dir=auto, recursive=True, batch_size="auto"
    )
    assert skipped.skipped == 52 and len(vae.calls) == calls
    wide = [batch for batch, shape in vae.calls if shape == (16, 32)]
    assert wide[0] == 1


def test_explicit_vae_batch_does_not_retry(tmp_path):
    images_in(tmp_path, 4)
    vae = TinyVAE(fail_above=2)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        latents.cache_latents(tmp_path, vae, batch_size=4)
    assert len(vae.calls) == 1


def test_anima_auto_variants_and_prior_keep_padding_and_values(
    tmp_path, simulated_cuda, monkeypatch
):
    images_in(tmp_path / "src", 20)
    fixed, auto = tmp_path / "fixed", tmp_path / "auto"
    fixed.mkdir()
    auto.mkdir()
    calls = []
    fail = False

    def encode(captions, *args):
        calls.append(len(captions))
        if fail and len(captions) > 2:
            raise torch.cuda.OutOfMemoryError("injected TE OOM")
        values = (
            torch.tensor([float(caption) for caption in captions])
            .view(-1, 1)
            .repeat(1, 512)
        )
        return (
            values,
            torch.ones_like(values).int(),
            values.long(),
            torch.ones_like(values).int(),
            values.clone(),
        )

    monkeypatch.setattr(text, "_encode_batch", encode)
    monkeypatch.setattr(
        text, "build_diff_output_prior_caption", lambda caption, **_: caption
    )
    monkeypatch.setattr(
        text, "_caption_variants_for_cache", lambda caption, *_: [caption.render()] * 9
    )
    kwargs = dict(
        device="cpu",
        min_pixels=0,
        caption_shuffle_variants=9,
        llm_adapter=object(),
        diff_output_preservation_class="class",
    )
    text.cache_text_embeddings(
        tmp_path / "src",
        None,
        None,
        SimpleNamespace(dtype=torch.bfloat16),
        cache_dir=fixed,
        batch_size=2,
        **kwargs,
    )
    fail = True
    stats = text.cache_text_embeddings(
        tmp_path / "src",
        None,
        None,
        SimpleNamespace(dtype=torch.bfloat16),
        cache_dir=auto,
        batch_size="auto",
        **kwargs,
    )
    assert stats.written == 20
    for expected in fixed.glob("*.safetensors"):
        a, b = load_file(expected), load_file(auto / expected.name)
        assert set(a) == set(b)
        for key in a:
            assert torch.equal(a[key], b[key])
        assert b["prompt_embeds_v8"].shape == (512,)


@pytest.mark.parametrize("family", ["krea2", "z_image"])
def test_krea_z_auto_variants_and_cache_metadata(tmp_path, simulated_cuda, family):
    from library.models.krea2_raw.strategy import Krea2TextEncoderOutputsCachingStrategy
    from library.models.z_image.strategy import ZImageTextEncoderOutputsCachingStrategy
    from library.preprocess.captions import CaptionSource
    from scripts.krea2.preprocess_te_cache import _cache_items

    cls = (
        Krea2TextEncoderOutputsCachingStrategy
        if family == "krea2"
        else ZImageTextEncoderOutputsCachingStrategy
    )
    calls = []
    model = torch.nn.Linear(1, 1)
    tokenizer = SimpleNamespace(tokenize=lambda captions: captions)

    def encode(tokenizer, models, captions):
        calls.append(len(captions))
        if len(captions) > 2:
            raise torch.cuda.OutOfMemoryError("injected")
        shape = (-1, 1, 1, 1) if family == "krea2" else (-1, 1, 1)
        hiddens = torch.tensor([int(caption) for caption in captions]).reshape(shape)
        return hiddens.expand(*hiddens.shape[:-1], 2560).contiguous(), torch.ones(
            len(captions), 1
        )

    strategy = cls(batch_size="auto")
    items = [
        (
            tmp_path / f"{i}.png",
            CaptionSource(captions=[str(i), str(i + 100), str(i + 200)]),
        )
        for i in range(20)
    ]
    kwargs = dict(
        data_dir=tmp_path,
        cache_dir=tmp_path,
        batch_size="auto",
        caption_shuffle_variants=0,
        caption_tag_dropout_rate=0.0,
        overwrite=False,
        caching_strategy=strategy,
        tokenize_strategy=tokenizer,
        encoding_strategy=SimpleNamespace(encode_tokens=encode),
        text_encoder=model,
    )
    assert _cache_items(items, **kwargs) == (20, 0)
    assert any(batch > 2 for batch in calls)
    for i in range(20):
        path = strategy.get_outputs_npz_path(str(tmp_path / f"{i}.png"))
        tensors = load_file(path)
        assert int(tensors["num_variants"]) == 3
        assert tensors["hiddens_v2"].eq(i + 200).all()
        assert strategy.is_disk_cached_outputs_expected(path)
    count = len(calls)
    assert _cache_items(items, **kwargs) == (0, 20)
    assert len(calls) == count
