from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from library.models.qwen_image_2_1 import preview_conditions, preview_decode, sampling, training_preview
from library.models.qwen_image_2_1.strategy import encode_edit_prompt
from web.services.preview.edit_metadata import merge_edit_preview_metadata


def test_snapshot_freezes_reference_and_disambiguates_identical_prompts(tmp_path):
    records = []
    for index, color in enumerate(("red", "blue")):
        source = tmp_path / f"source {index}.png"
        Image.new("RGB", (64, 64), color).save(source)
        records.append({"prompt": "same instruction", "sample_task": "edit", "reference_image": str(source),
                        "width": 64, "height": 64})
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    snapshots = preview_conditions.snapshot_prompts(SimpleNamespace(sample_prompts=str(path), output_dir=str(tmp_path), sample_sampler="euler"))
    frozen = [Path(prompt["reference_image"]).read_bytes() for prompt in snapshots]
    assert frozen[0] != frozen[1]
    assert preview_conditions.condition_key(snapshots[0]) != preview_conditions.condition_key(snapshots[1])
    assert preview_conditions.reference_key(snapshots[0]) != preview_conditions.reference_key(snapshots[1])
    for record in records:
        Path(record["reference_image"]).unlink()
    path.write_text("[]", encoding="utf-8")
    assert [Path(prompt["reference_image"]).read_bytes() for prompt in snapshots] == frozen


def test_snapshot_freezes_ordered_references_and_keeps_single_legacy_field(tmp_path):
    paths = []
    for index, (size, color) in enumerate((((64, 32), "red"), ((32, 64), "blue"))):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", size, color).save(path)
        paths.append(str(path))
    records = [
        {"prompt": "edit", "sample_task": "edit", "reference_images": paths, "width": 64, "height": 64},
        {"prompt": "edit", "sample_task": "edit", "reference_images": paths[:1], "width": 64, "height": 64},
    ]
    source = tmp_path / "prompts.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    snapshots = preview_conditions.snapshot_prompts(SimpleNamespace(sample_prompts=str(source), output_dir=str(tmp_path), sample_sampler="euler"))
    frozen = snapshots[0]["reference_images"]
    assert len(frozen) == 2 and frozen[0] != frozen[1]
    assert "reference_image" not in snapshots[0]
    assert "reference_images" not in snapshots[1]
    assert snapshots[1]["reference_image"] == frozen[0]
    for path in paths:
        Path(path).unlink()
    assert [Image.open(path).getpixel((0, 0)) for path in frozen] == [(255, 0, 0), (0, 0, 255)]


def test_encode_edit_prompt_multi_image_order_and_legacy_prefix():
    class Inputs:
        input_ids = torch.tensor([[42, 7, 42, 8]])
        attention_mask = torch.ones(1, 4, dtype=torch.long)

        def to(self, device):
            return self

    class Processor:
        def __call__(self, **kwargs):
            self.text = kwargs["text"][0]
            self.images = kwargs["images"]
            return Inputs()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.model = SimpleNamespace(language_model=SimpleNamespace(norm=torch.nn.Identity()))

        def forward(self, **kwargs):
            hidden = self.model.language_model.norm(torch.zeros(1, 4, 4096))
            return SimpleNamespace(hidden_states=(hidden,))

    processor = Processor()
    tokenizer = SimpleNamespace(processor=processor, drop_idx=0, image_token_id=42)
    images = [Image.new("RGB", (32, 16), "red"), Image.new("RGB", (16, 32), "blue")]
    hidden, mask, slots = encode_edit_prompt(Model(), tokenizer, "change", images)
    assert processor.images == images
    assert "<image1><|vision_start|>" in processor.text
    assert "<image2><|vision_start|>" in processor.text
    assert hidden.shape == (4, 4096) and mask.all() and slots.tolist() == [True, False, True, False]
    encode_edit_prompt(Model(), tokenizer, "change", images[0])
    assert "<image1>" not in processor.text


def test_text_cache_same_instruction_preserves_image_condition(tmp_path, monkeypatch):
    prompts = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (64, 64), color).save(path)
        prompts.append({"prompt": "same instruction", "negative_prompt": "", "scale": 3,
                        "reference_image": str(path), "enum": index})
    monkeypatch.setattr(preview_conditions, "snapshot_prompts", lambda args: prompts)

    class Tokenizer:
        pass

    monkeypatch.setattr(preview_conditions, "QwenImage21EditTokenizeStrategy", Tokenizer)

    def encode(model, tokenizer, text, image):
        value = float(image.getpixel((0, 0))[0]) + bool(text)
        return torch.full((2, 3), value), torch.ones(2), torch.ones(2)

    monkeypatch.setattr(preview_conditions, "encode_edit_prompt", encode)
    trainer = SimpleNamespace()
    preview_conditions.cache_preview_text(trainer, SimpleNamespace(), object(), Tokenizer(), None)
    outputs = trainer.sample_prompts_te_outputs
    assert len(outputs) == 4
    positive = [outputs[preview_conditions.condition_key(prompt)][0] for prompt in prompts]
    assert not torch.equal(*positive)
    for prompt in prompts:
        positive = outputs[preview_conditions.condition_key(prompt)][0]
        negative = outputs[preview_conditions.condition_key(prompt, negative=True)][0]
        torch.testing.assert_close(positive, negative + 1)
    assert trainer.sample_prompts_snapshot is prompts


def test_multi_reference_text_and_vae_cache_preserve_order(tmp_path, monkeypatch):
    paths = []
    for index, (size, color) in enumerate((((64, 32), "red"), ((32, 64), "blue"))):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", size, color).save(path)
        paths.append(str(path))
    prompt = {"prompt": "change", "negative_prompt": "avoid", "scale": 3,
              "reference_images": paths, "enum": 0}
    monkeypatch.setattr(preview_conditions, "snapshot_prompts", lambda args: [prompt])
    class Tokenizer:
        pass
    monkeypatch.setattr(preview_conditions, "QwenImage21EditTokenizeStrategy", Tokenizer)
    seen = []

    def encode(model, tokenizer, text, images):
        seen.append((text, [image.getpixel((0, 0)) for image in images]))
        return torch.zeros(6, 4096), torch.ones(6), torch.tensor([1, 1, 1, 1, 0, 0])

    monkeypatch.setattr(preview_conditions, "encode_edit_prompt", encode)
    monkeypatch.setattr(preview_conditions, "encode_qwen_image_2_1_latents", lambda vae, pixels: torch.full((1, 64, pixels.shape[-2] // 16, pixels.shape[-1] // 16), float(pixels[0, 0, 0, 0])))
    monkeypatch.setattr(preview_conditions, "clean_memory_on_device", lambda device: None)
    trainer = SimpleNamespace()
    preview_conditions.cache_preview_text(trainer, SimpleNamespace(), object(), Tokenizer(), None)
    assert [entry[0] for entry in seen] == ["change", "avoid"]
    assert all(entry[1] == [(255, 0, 0), (0, 0, 255)] for entry in seen)
    vae = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32, to=lambda device: vae)
    accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1)
    preview_conditions.cache_preview_references(trainer, SimpleNamespace(), accelerator, vae)
    latents = trainer.sample_prompts_te_outputs[preview_conditions.reference_key(prompt)]
    assert len(latents) == 2 and [tuple(latent.shape[-2:]) for latent in latents] == [(2, 4), (4, 2)]
    assert latents[0][0, 0, 0, 0] > latents[1][0, 0, 0, 0]


@pytest.mark.parametrize("negative_prompt,present,scale,expected_negative", [
    (None, False, 3.0, False), (None, True, 3.0, False),
    ("", True, 3.0, True), ("avoid", True, 1.0, False),
])
def test_edit_cache_negative_only_when_cfg_needed(tmp_path, monkeypatch, negative_prompt, present, scale, expected_negative):
    path = tmp_path / "reference.png"
    Image.new("RGB", (64, 64), "red").save(path)
    prompt = {"prompt": "edit", "reference_image": str(path), "enum": 0, "scale": scale}
    if present:
        prompt["negative_prompt"] = negative_prompt
    monkeypatch.setattr(preview_conditions, "snapshot_prompts", lambda args: [prompt])

    class Tokenizer:
        pass

    monkeypatch.setattr(preview_conditions, "QwenImage21EditTokenizeStrategy", Tokenizer)
    seen = []

    def encode(model, tokenizer, text, image):
        seen.append(text)
        return torch.zeros(4, 3), torch.ones(4), torch.ones(4)

    monkeypatch.setattr(preview_conditions, "encode_edit_prompt", encode)
    monkeypatch.setattr(preview_conditions, "encode_qwen_image_2_1_latents", lambda vae, pixels: torch.zeros(1, 64, 4, 4))
    monkeypatch.setattr(preview_conditions, "clean_memory_on_device", lambda device: None)
    trainer = SimpleNamespace()
    preview_conditions.cache_preview_text(trainer, SimpleNamespace(), object(), Tokenizer(), None)
    outputs = trainer.sample_prompts_te_outputs
    assert seen == (["edit", negative_prompt] if expected_negative else ["edit"])
    assert (preview_conditions.condition_key(prompt, negative=True) in outputs) is expected_negative
    vae = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32, to=lambda device: vae)
    accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1)
    preview_conditions.cache_preview_references(trainer, SimpleNamespace(), accelerator, vae)
    assert len(outputs[preview_conditions.reference_key(prompt)]) == 1


@pytest.mark.parametrize("scale", [1.0, 3.0])
def test_official_euler_seed_cfg_and_reference_contract(scale):
    calls = []
    positive = (torch.ones(1, 2, 3), torch.ones(1, 2), torch.ones(1, 2))
    negative = (torch.zeros(1, 2, 3), torch.ones(1, 2), torch.ones(1, 2))
    reference = torch.ones(1, 64, 4, 4)
    dit = SimpleNamespace(prepare_block_swap_before_forward=lambda: None)

    def forward(model, latent, text, mask, sigma, **kwargs):
        assert model is dit and mask.dtype == torch.bool
        assert kwargs["image_slot_mask"].dtype == torch.bool
        assert torch.equal(kwargs["reference_latents"], reference)
        calls.append(float(text.mean()))
        return torch.ones_like(latent) * text.mean()

    prompt = {"width": 64, "height": 64, "sample_steps": 3, "guidance_scale": scale,
              "negative_prompt": "avoid"}
    kwargs = dict(device=torch.device("cpu"), dtype=torch.float32, seed=12, forward=forward)
    first = sampling.generate_preview(dit, prompt, positive, negative, reference, **kwargs)
    assert calls == ([1.0] * 3 if scale == 1 else [1.0, 0.0] * 3)
    again = sampling.generate_preview(dit, prompt, positive, negative, reference, **kwargs)
    assert torch.equal(first, again)
    initial = torch.randn(first.shape, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(first, initial - scale)
    different = sampling.generate_preview(dit, prompt, positive, negative, reference, **{**kwargs, "seed": 13})
    assert not torch.equal(first, different)


@pytest.mark.parametrize("negative_prompt,present,expected_calls", [
    (None, False, [1.0] * 3), (None, True, [1.0] * 3),
    ("", True, [1.0, 0.0] * 3),
])
def test_euler_cfg_requires_explicit_negative_prompt(negative_prompt, present, expected_calls):
    positive = (torch.ones(1, 2, 3), torch.ones(1, 2), torch.zeros(1, 2))
    negative = (torch.zeros(1, 2, 3), torch.ones(1, 2), torch.zeros(1, 2))
    prompt = {"width": 64, "height": 64, "sample_steps": 3, "scale": 3.0}
    if present:
        prompt["negative_prompt"] = negative_prompt
    calls = []

    def forward(dit, latent, text, mask, sigma, **kwargs):
        calls.append(float(text.mean()))
        return torch.ones_like(latent) * text.mean()

    sampling.generate_preview(None, prompt, positive, negative if negative_prompt is not None else None,
                              None, device=torch.device("cpu"), dtype=torch.float32, seed=12,
                              forward=forward)
    assert calls == expected_calls


def test_euler_passes_ordered_reference_tuple_to_forward():
    references = (torch.ones(1, 64, 2, 4), torch.full((1, 64, 4, 2), 2.0))
    conditions = (torch.zeros(1, 4, 4096), torch.ones(1, 4), torch.ones(1, 4))
    seen = []

    def forward(dit, latent, hidden, mask, sigma, **kwargs):
        seen.append(kwargs["reference_latents"])
        return torch.zeros_like(latent)

    sampling.generate_preview(None, {"width": 64, "height": 64, "sample_steps": 2}, conditions,
                              conditions, references, device=torch.device("cpu"), dtype=torch.float32,
                              seed=1, forward=forward)
    assert len(seen) == 2
    assert all(isinstance(item, tuple) and len(item) == 2 for item in seen)
    for item in seen:
        torch.testing.assert_close(item[0], references[0])
        torch.testing.assert_close(item[1], references[1])


class SwapModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.switches = []

    def switch_block_swap_for_inference(self):
        self.switches.append("inference")

    def switch_block_swap_for_training(self):
        self.switches.append("training")


@pytest.mark.parametrize("fail", [False, True])
def test_local_sample_restores_modes_swap_and_rng(tmp_path, monkeypatch, fail):
    dit, network = SwapModel(), torch.nn.Linear(1, 1)
    network.eval()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    args = SimpleNamespace(output_dir=str(tmp_path), seed=123, output_name="test")
    monkeypatch.setattr(training_preview, "prepare_dtype", lambda args: (torch.float32, None))

    def generate(*args, **kwargs):
        assert not dit.training and not network.training
        torch.randn(10)
        if fail:
            raise RuntimeError("sample failed")
        return torch.zeros(1, 64, 4, 4)

    monkeypatch.setattr(training_preview, "generate_preview", generate)
    prompt = {"prompt": "test", "enum": 0}
    outputs = {preview_conditions.condition_key(prompt): (), preview_conditions.condition_key(prompt, negative=True): ()}
    state = torch.random.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="sample failed"):
            training_preview._sample_local(accelerator, args, dit, network, [prompt], outputs, None, 5)
    else:
        training_preview._sample_local(accelerator, args, dit, network, [prompt], outputs, None, 5)
        files = list((tmp_path / "sample" / "latents").glob("*.pt"))
        record = torch.load(files[0], weights_only=True)
        assert record["metadata"]["seed"] == 123 and record["metadata"]["step"] == 5
    assert dit.training and not network.training
    assert dit.switches == ["inference", "training"]
    assert torch.equal(state, torch.random.get_rng_state())


def test_local_sample_passes_all_cached_references(tmp_path, monkeypatch):
    dit = SwapModel()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    args = SimpleNamespace(output_dir=str(tmp_path), seed=1, output_name="test")
    prompt = {"prompt": "edit", "enum": 0, "reference_images": ["first.png", "second.png"]}
    references = (torch.zeros(1, 64, 2, 4), torch.ones(1, 64, 4, 2))
    outputs = {
        preview_conditions.condition_key(prompt): (),
        preview_conditions.condition_key(prompt, negative=True): (),
        preview_conditions.reference_key(prompt): references,
    }
    seen = []
    monkeypatch.setattr(training_preview, "prepare_dtype", lambda args: (torch.float32, None))

    def generate(dit, prompt, positive, negative, reference, **kwargs):
        seen.append(reference)
        return torch.zeros(1, 64, 4, 4)

    monkeypatch.setattr(training_preview, "generate_preview", generate)
    training_preview._sample_local(accelerator, args, dit, None, [prompt], outputs, None, 1)
    assert len(seen) == 1 and seen[0] is references


def test_local_sample_allows_missing_negative_cache(tmp_path, monkeypatch):
    dit = SwapModel()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    args = SimpleNamespace(output_dir=str(tmp_path), seed=1, output_name="test")
    prompt = {"prompt": "edit", "enum": 0, "scale": 3.0}
    outputs = {preview_conditions.condition_key(prompt): ()}
    monkeypatch.setattr(training_preview, "prepare_dtype", lambda args: (torch.float32, None))

    def generate(dit, prompt, positive, negative, reference, **kwargs):
        assert positive == () and negative is None and reference is None
        return torch.zeros(1, 64, 4, 4)

    monkeypatch.setattr(training_preview, "generate_preview", generate)
    training_preview._sample_local(accelerator, args, dit, None, [prompt], outputs, None, 1)


class FakeVAE:
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(latents_mean=[2.0] * 64, latents_std=[3.0] * 64)

    def to(self, device):
        self.device = device
        return self

    def decode(self, latent, return_dict):
        self.decoded = latent
        return (torch.zeros(1, 4, 1, 16, 16),)


def test_decode_reverses_latent_normalization():
    vae = FakeVAE()
    image = preview_decode.decode_latent(vae, torch.ones(1, 64, 4, 4))
    assert vae.decoded.shape == (1, 64, 1, 4, 4)
    assert torch.all(vae.decoded == 5)
    assert image.size == (16, 16) and image.getpixel((0, 0)) == (127, 127, 127)
    with pytest.raises(ValueError):
        preview_decode.decode_latent(vae, torch.zeros(1, 16, 4, 4))


def test_decode_edit_comparison_and_frozen_metadata(tmp_path, monkeypatch):
    root = tmp_path / "sample"
    (root / "latents").mkdir(parents=True)
    (root / "references").mkdir()
    reference = root / "references" / "source.png"
    Image.new("RGB", (16, 16), "red").save(reference)
    metadata = {"prompt": "frozen prompt", "negative_prompt": "bad", "sample_task": "edit",
                "reference_image": str(reference), "seed": 12, "step": 4,
                "width": 64, "height": 64, "sample_steps": 3, "guidance_scale": 2.0}
    latent_path = root / "latents" / "sample.pt"
    torch.save({"latents": torch.ones(1, 64, 4, 4), "metadata": metadata}, latent_path)
    monkeypatch.setattr(preview_decode, "clean_memory_on_device", lambda device: None)
    preview_decode.decode_pending_samples(SimpleNamespace(device=torch.device("cpu")), SimpleNamespace(output_dir=str(tmp_path)), FakeVAE())
    assert not latent_path.exists()
    assert (root / "results" / "sample.png").is_file()
    with Image.open(root / "sample.png") as image:
        assert image.size == (32, 44)
        assert image.getpixel((0, 28)) == (255, 0, 0)
        merged = merge_edit_preview_metadata({"prompt": "mutated", "step": 999}, image.info)
    assert merged["prompt"] == "frozen prompt" and merged["step"] == 4
    assert merged["reference_file"] == "sample/references/source.png"
    assert merged["result_file"] == "sample/results/sample.png"
    assert merged["parameters"]["guidance_scale"] == 2.0


def test_decode_failure_retains_latent_for_retry(tmp_path, monkeypatch):
    root = tmp_path / "sample" / "latents"
    root.mkdir(parents=True)
    path = root / "bad.pt"
    torch.save({"latents": torch.zeros(1, 16, 4, 4), "metadata": {}}, path)
    monkeypatch.setattr(preview_decode, "clean_memory_on_device", lambda device: None)
    vae = FakeVAE()
    preview_decode.decode_pending_samples(SimpleNamespace(device=torch.device("cpu")), SimpleNamespace(output_dir=str(tmp_path)), vae)
    assert path.exists() and vae.device == torch.device("cpu")
