import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch

from library.models.qwen_image_2_1 import preview_decode
from web.services.preview.edit_metadata import merge_edit_preview_metadata


class FakeVAE:
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(latents_mean=[0.0] * 64, latents_std=[1.0] * 64)

    def to(self, device):
        self.device = device
        return self

    def decode(self, latent, return_dict):
        return (torch.zeros(1, 4, 1, 16, 16),)


def test_multi_comparison_has_bounded_equal_columns():
    generated = Image.new("RGB", (2048, 1024), "white")
    references = [Image.new("RGB", (32, 64), color) for color in ("red", "green", "blue", "yellow")]
    image = preview_decode.multi_comparison_image(references, generated)
    assert image.size == (5 * 512, 256 + 28)
    assert image.getpixel((256, 156)) == (255, 0, 0)
    assert image.getpixel((4 * 512 + 256, 156)) == (255, 255, 255)


def test_decode_multi_reference_metadata_and_frozen_merge(tmp_path, monkeypatch):
    root = tmp_path / "sample"
    (root / "latents").mkdir(parents=True)
    (root / "references").mkdir()
    references = []
    for index, color in enumerate(("red", "blue")):
        path = root / "references" / f"{index}.png"
        Image.new("RGB", (16, 16), color).save(path)
        references.append(str(path))
    metadata = {"prompt": "frozen", "sample_task": "edit", "reference_images": references,
                "reference_image": references[0], "step": 3}
    torch.save({"latents": torch.ones(1, 64, 4, 4), "metadata": metadata}, root / "latents" / "sample.pt")
    monkeypatch.setattr(preview_decode, "clean_memory_on_device", lambda device: None)
    preview_decode.decode_pending_samples(SimpleNamespace(device=torch.device("cpu")),
                                          SimpleNamespace(output_dir=str(tmp_path)), FakeVAE())
    with Image.open(root / "sample.png") as image:
        assert image.size == (48, 44)
        assert image.getpixel((0, 28)) == (255, 0, 0)
        frozen = json.loads(image.info["qwen_preview"])
        merged = merge_edit_preview_metadata({"prompt": "changed", "step": 999}, image.info)
    assert frozen["reference_files"] == ["sample/references/0.png", "sample/references/1.png"]
    assert merged["reference_files"] == frozen["reference_files"]
    assert merged["reference_file"] == frozen["reference_files"][0]
    assert merged["result_file"] == "sample/results/sample.png"
    assert merged["sample_task"] == "edit" and merged["prompt"] == "frozen" and merged["step"] == 3
    assert (root / "results" / "sample.png").is_file()
    assert not (root / "latents" / "sample.pt").exists()
