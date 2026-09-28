from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from library.anima.text_strategies import LatentsCachingStrategy
from library.datasets.dataset_cache import DatasetCacheMixin
from library.datasets.image_utils import glob_images
from library.datasets.qwen_image_edit import (
    edit_condition_fingerprint,
    inspect_edit_pairs,
    require_complete_edit_pairs,
)
from library.models.qwen_image_2_1.strategy import (
    QwenImage21EditTextCache,
    QwenImage21EditTokenizeStrategy,
    QwenImage21LatentCache,
)


def _image(path: Path, size: tuple[int, int] = (64, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


def test_edit_pairs_match_casefolded_stems_with_relative_subdirectories(tmp_path: Path) -> None:
    target_dir = tmp_path / "targets"
    reference_dir = tmp_path / "references"
    _image(target_dir / "sub" / "Hero.PNG")
    _image(reference_dir / "sub" / "hero.jpg")
    _image(reference_dir / "unused.png")

    report = inspect_edit_pairs(
        glob_images(str(target_dir), recursive=True),
        target_dir=str(target_dir),
        reference_dir=str(reference_dir),
        recursive=True,
    )

    require_complete_edit_pairs(report)
    assert report.pairs[str(target_dir / "sub" / "Hero.PNG")] == str(reference_dir / "sub" / "hero.jpg")
    assert report.unused_references == (str(reference_dir / "unused.png"),)


def test_edit_pairs_reject_missing_and_ambiguous_references(tmp_path: Path) -> None:
    target_dir = tmp_path / "targets"
    reference_dir = tmp_path / "references"
    _image(target_dir / "missing.png")
    _image(target_dir / "ambiguous.png")
    _image(reference_dir / "ambiguous.png")
    _image(reference_dir / "ambiguous.jpg")

    report = inspect_edit_pairs(
        glob_images(str(target_dir)),
        target_dir=str(target_dir),
        reference_dir=str(reference_dir),
    )
    with pytest.raises(ValueError, match="no paired reference.*ambiguous pair key"):
        require_complete_edit_pairs(report)


def _edit_info(target: Path, reference: Path):
    return SimpleNamespace(
        absolute_path=str(target),
        reference_image_path=str(reference),
        caption="make the background blue",
        bucket_reso=(64, 32),
    )


def test_reference_latent_cache_requires_exact_expected_grid(tmp_path: Path) -> None:
    target = tmp_path / "targets" / "target.png"
    reference = tmp_path / "references" / "target.png"
    _image(target)
    _image(reference)
    info = _edit_info(target, reference)
    subset = SimpleNamespace(cache_dir=str(tmp_path / "cache"), image_dir=str(target.parent))
    strategy = QwenImage21LatentCache(True, 1, False)
    path = strategy.get_edit_reference_latent_path(info, subset)
    info.edit_reference_latent_path = path
    fingerprint = edit_condition_fingerprint(reference, info.caption, info.bucket_reso)
    metadata = {
        "edit_cache_schema": strategy.EDIT_LATENT_CACHE_SCHEMA,
        "edit_condition_fingerprint": fingerprint,
    }

    save_file({"latent": torch.zeros(64, 2, 4)}, path, metadata=metadata)
    assert strategy.is_edit_reference_cache_expected(info, subset)
    assert strategy.load_edit_reference_latent(info, subset).shape == (64, 2, 4)

    save_file({"latent": torch.zeros(64, 3, 4)}, path, metadata=metadata)
    assert not strategy.is_edit_reference_cache_expected(info, subset)
    with pytest.raises(ValueError, match="shape mismatch"):
        strategy.load_edit_reference_latent(info, subset)


def test_edit_text_cache_requires_reference_slot_count_and_fingerprint(tmp_path: Path) -> None:
    target = tmp_path / "targets" / "target.png"
    reference = tmp_path / "references" / "target.png"
    _image(target)
    _image(reference)
    info = _edit_info(target, reference)
    subset = SimpleNamespace(cache_dir=str(tmp_path / "cache"), image_dir=str(target.parent))
    strategy = QwenImage21EditTextCache(True, 1, False)
    path = strategy.get_outputs_npz_path_for_info(info, subset)
    fingerprint = edit_condition_fingerprint(reference, info.caption, info.bucket_reso)
    metadata = {
        "edit_cache_schema": strategy.EDIT_CACHE_SCHEMA,
        "edit_condition_fingerprint": fingerprint,
    }
    hiddens = torch.zeros(5, 4096, dtype=torch.bfloat16)
    slots = torch.tensor([False, True, False, True, False])
    tensors = {
        "hiddens": hiddens,
        "mask": torch.ones(5, dtype=torch.bool),
        "image_slots": slots,
        "caption_dropout_rate": torch.tensor(0.0),
    }

    save_file(tensors, path, metadata=metadata)
    assert strategy.is_expected_for_info(path, info)
    assert strategy.load_outputs_npz(path)[2].sum().item() == 2

    tensors["image_slots"] = torch.tensor([False, True, False, False, False])
    save_file(tensors, path, metadata=metadata)
    assert not strategy.is_expected_for_info(path, info)


def test_edit_tokenizer_strategy_provides_required_qwen3_video_processor(monkeypatch) -> None:
    import transformers
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    from library.models.qwen_image_2_1 import strategy as qwen_strategy

    class FakeTokenizer:
        padding_side = "right"
        pad_token_id = 151643

        def encode(self, *_args, **_kwargs):
            return [1, 2]

        def convert_tokens_to_ids(self, _token):
            return 151655

    captured = {}

    def capture_processor_init(_self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(qwen_strategy, "resolve_tokenizer_path", lambda _path: "local-tokenizer")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: FakeTokenizer())
    monkeypatch.setattr(Qwen3VLProcessor, "__init__", capture_processor_init)

    QwenImage21EditTokenizeStrategy("qwen3.safetensors")

    assert isinstance(captured["video_processor"], Qwen3VLVideoProcessor)


def test_edit_target_cache_path_is_rechecked_and_rebuilt(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    cache = tmp_path / "target_cache.npz"
    _image(target)
    _image(reference)
    info = SimpleNamespace(
        image_key="target",
        absolute_path=str(target),
        image_size=(64, 32),
        bucket_reso=(64, 32),
        reference_image_path=str(reference),
        latents_npz=str(cache),
        image=None,
    )
    subset = SimpleNamespace(
        image_dir=str(tmp_path), cache_dir=str(tmp_path),
        flip_aug=False, alpha_mask=False, random_crop=False,
    )
    dataset = DatasetCacheMixin()
    dataset.image_data = {info.image_key: info}
    dataset.image_to_subset = {info.image_key: subset}

    class FakeStrategy:
        cache_to_disk = True
        batch_size = 1

        def is_disk_cached_latents_expected(self, _reso, path, _flip, _mask):
            return Path(path).exists()

        def get_edit_reference_latent_path(self, _info, _subset):
            return str(tmp_path / "reference_cache.safetensors")

        def is_edit_reference_cache_expected(self, _info, _subset):
            return True

        def cache_batch_latents(self, _model, batch, _flip, _mask, _crop):
            assert batch == [info]
            cache.touch()

    monkeypatch.setattr(LatentsCachingStrategy, "get_strategy", lambda: FakeStrategy())
    assert not dataset.is_latents_cache_complete()

    accelerator = SimpleNamespace(num_processes=1, process_index=0, device=torch.device("cpu"))
    dataset.new_cache_latents(None, accelerator)

    assert cache.exists()
    assert dataset.is_latents_cache_complete()


@pytest.mark.parametrize("task,subset,match", [
    ("edit", {"flip_aug": True}, "deterministic reference/target"),
    ("t2i", {"reference_image_dir": "reference"}, "训练任务为 t2i"),
])
def test_loaded_qwen_dataset_config_rechecks_task_contract(task, subset, match) -> None:
    from library.training.bootstrap import TrainingBootstrap

    args = SimpleNamespace(
        model_family="qwen_image_2_1", qwen_image_2_1_task=task,
        cache_latents=True, cache_text_encoder_outputs=True,
    )
    dataset_config = {
        "datasets": [{"batch_size": 1, "subsets": [{"image_dir": "target", **subset}]}],
    }
    if task == "edit":
        dataset_config["datasets"][0]["subsets"][0]["reference_image_dir"] = "reference"

    with pytest.raises(ValueError, match=match):
        TrainingBootstrap.validate_qwen_dataset_config(args, dataset_config)


@pytest.mark.parametrize("source", ["training", "dataset_general"])
def test_qwen_edit_rejects_inherited_augmentation(source) -> None:
    from library.training.bootstrap import TrainingBootstrap

    args = SimpleNamespace(
        model_family="qwen_image_2_1", qwen_image_2_1_task="edit",
        cache_latents=True, cache_text_encoder_outputs=True,
        flip_aug=source == "training",
    )
    dataset_config = {
        "general": {"color_aug": source == "dataset_general"},
        "datasets": [{"batch_size": 1, "subsets": [{
            "image_dir": "target", "reference_image_dir": "reference",
        }]}],
    }
    with pytest.raises(ValueError, match="deterministic reference/target"):
        TrainingBootstrap.validate_qwen_dataset_config(args, dataset_config)
