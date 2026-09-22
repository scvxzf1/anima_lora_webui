import hashlib
import json

import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import save_file
import toml
import torch
import pytest

from bench.adaptive_runtime.training_fixture import export_fixture, smoke_config
from library.anima.strategy import AnimaLatentsCachingStrategy
from library.models.krea2_raw.strategy import Krea2TextEncoderOutputsCachingStrategy


def make_source(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (32, 40), "red").save(image)
    image.with_suffix(".txt").write_text("test caption")
    source = tmp_path / "inputs.safetensors"
    tensors = {"latents": torch.ones(1, 16, 1, 4, 4), "noise": torch.zeros(1, 16, 1, 4, 4),
               "hidden": torch.ones(1, 512, 12, 2560), "mask": torch.ones(1, 512, dtype=torch.bool)}
    save_file(tensors, str(source), metadata={"schema": "adaptive_krea_inputs_v1"})
    report = {"schema": "adaptive_krea_inputs_v1", "status": "ok", "resolution": "32",
              "image": str(image), "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
              "caption_sha256": hashlib.sha256(image.with_suffix(".txt").read_bytes()).hexdigest(),
              "cache_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    source.with_suffix(".json").write_text(json.dumps(report))
    return source, tensors


def test_export_uses_validated_production_sidecars(tmp_path):
    source, tensors = make_source(tmp_path)
    destination = export_fixture(source, tmp_path / "fixture")
    latent = next((destination / "cache").glob("*.npz"))
    text = next((destination / "cache").glob("*.safetensors"))
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    cached, *_ = strategy.load_latents_from_disk(str(latent), (32, 32))
    assert torch.equal(torch.from_numpy(cached), tensors["latents"][0, :, 0])
    hidden, mask, rate = Krea2TextEncoderOutputsCachingStrategy().load_outputs_npz(str(text))
    assert torch.equal(hidden.float(), tensors["hidden"][0]) and mask.all() and rate == 0
    config = smoke_config(destination, tmp_path / "run", tmp_path / "weights")
    assert config["max_train_steps"] == 3 and config["adaptive_oom_retry"]
    assert config["blocks_to_swap"] == 26 and not config["torch_compile"]
    assert toml.load(destination / "dataset.toml")["datasets"][0]["resolution"] == 32
    with Image.open(tmp_path / "source.png") as original, Image.open(destination / "images/sample.png") as exported:
        expected = ImageOps.fit(ImageOps.exif_transpose(original).convert("RGB"), (32, 32),
                                method=Image.Resampling.LANCZOS)
        assert np.array_equal(np.asarray(exported), np.asarray(expected))
    with pytest.raises(FileExistsError):
        export_fixture(source, destination)
    with pytest.raises(FileExistsError):
        smoke_config(destination, destination, tmp_path / "weights")


def test_export_rejects_source_drift_before_writing(tmp_path):
    source, _ = make_source(tmp_path)
    (tmp_path / "source.txt").write_text("changed caption")
    with pytest.raises(ValueError, match="hash changed"):
        export_fixture(source, tmp_path / "fixture")
    assert not (tmp_path / "fixture").exists()


def test_export_refuses_lossy_text_conversion(tmp_path):
    source, tensors = make_source(tmp_path)
    tensors["hidden"][0, 0, 0, 0] = 1.001
    save_file(tensors, str(source), metadata={"schema": "adaptive_krea_inputs_v1"})
    report = json.loads(source.with_suffix(".json").read_text())
    report["cache_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_suffix(".json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="lose source precision"):
        export_fixture(source, tmp_path / "fixture")
    assert not (tmp_path / "fixture").exists()
