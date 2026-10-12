import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from tests.test_training_recovery_cursor import group, train_steps
from library.training.adaptive_runtime.resume_data import make_cursor_dataloader
from library.training.adaptive_runtime.resume_rng import capture_rng, restore_rng, validate_rng
from library.training.adaptive_runtime.resume_checkpoint import (
    save_recovery_state, validate_recovery_state, restore_recovery_auxiliary,
)
from library.training.adaptive_runtime.resume_identity import hash_file


def setup():
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    acc = Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(use_stateful_dataloader=True))
    raw = make_cursor_dataloader(group(), seed=18)
    raw.dataset.data_identity = {"schema": "fixture", "sha256": "fixture"}
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.95)
    model, opt, loader, sched = acc.prepare(model, opt, raw, sched)
    state = SimpleNamespace(accelerator=acc, train_dataloader=loader, global_step=0)
    def hook(models, weights, directory):
        from pathlib import Path
        (Path(directory) / "train_state.json").write_text(json.dumps({"current_step": state.global_step}))
    acc.register_save_state_pre_hook(hook)
    return state, model, opt, sched


@pytest.mark.parametrize("cut", [5, 12])
def test_accelerate_committed_resume_cpu(tmp_path, cut):
    state, model, opt, sched = setup()
    loader = state.train_dataloader
    _, iterator = train_steps(loader, model, opt, sched, cut)
    state.global_step = cut
    before = capture_rng()
    path = save_recovery_state(state, tmp_path, precision_contract_id="native", precision_manifest_id="p1",
                               counters={"global_step": cut, "completed_steps": cut},
                               precision_metadata={"mode": "fp16_fp32", "base": None, "execution": None})
    torch.testing.assert_close(before["cpu"], capture_rng()["cpu"], rtol=0, atol=0)
    expected, _ = train_steps(loader, model, opt, sched, 18-cut, iterator=iterator)
    final_rng = capture_rng()
    resumed, model2, opt2, sched2 = setup()
    validate_recovery_state(path, owner_root=tmp_path, precision_contract_id="native", precision_manifest_id="p1")
    resumed.accelerator.load_state(path)
    restore_recovery_auxiliary(resumed.accelerator, resumed.train_dataloader, path,
                               owner_root=tmp_path, precision_contract_id="native", precision_manifest_id="p1")
    actual, _ = train_steps(resumed.train_dataloader, model2, opt2, sched2, 18-cut)
    for a, b in zip(actual, expected):
        assert a["key"] == b["key"]
        for k in ("value", "target", "loss"):
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), model2.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(opt.state_dict(), opt2.state_dict(), rtol=0, atol=0)
    assert sched.state_dict() == sched2.state_dict()
    torch.testing.assert_close(final_rng["cpu"], capture_rng()["cpu"], rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["hash", "cursor", "rng", "identity", "external", "partial"])
def test_snapshot_rejects_faults(tmp_path, fault):
    state, model, opt, sched = setup()
    train_steps(state.train_dataloader, model, opt, sched, 1)
    state.global_step = 1
    path = save_recovery_state(state, tmp_path, precision_contract_id="native", precision_manifest_id="p1",
                               counters={"global_step": 1, "completed_steps": 1},
                               precision_metadata={"mode": "fp16_fp32", "base": None, "execution": None})
    kwargs = dict(owner_root=tmp_path, precision_contract_id="native")
    if fault == "hash":
        (path / "optimizer.bin").write_bytes(b"corrupt")
    if fault in ("cursor", "rng"):
        name = "cursor.pt" if fault == "cursor" else "strict_rng.pt"
        torch.save({}, path / name)
        meta = json.loads((path / "snapshot.json").read_text())
        meta["files"][name] = hash_file(path / name)
        (path / "snapshot.json").write_text(json.dumps(meta))
    if fault == "identity":
        kwargs["data_identity"] = {"sha256": "changed"}
    if fault == "external":
        kwargs["owner_root"] = tmp_path / "other"
    if fault == "partial":
        path = path.rename(tmp_path / ".partial")
    with pytest.raises(ValueError):
        validate_recovery_state(path, **kwargs)


def test_rng_validation_is_pure_and_strict():
    before = capture_rng()
    validate_rng(before)
    torch.testing.assert_close(before["cpu"], capture_rng()["cpu"], rtol=0, atol=0)
    bad = {**before, "cpu": torch.zeros(3, dtype=torch.uint8)}
    with pytest.raises(ValueError):
        restore_rng(bad)
    torch.testing.assert_close(before["cpu"], capture_rng()["cpu"], rtol=0, atol=0)


def test_content_identity_same_size_change(tmp_path):
    p = tmp_path / "cached"
    p.write_bytes(b"abcd")
    before = hash_file(p)
    p.write_bytes(b"abce")
    assert before != hash_file(p)


def test_data_symlink_identity_tracks_target_and_content(tmp_path):
    from library.training.adaptive_runtime.resume_identity import hash_data_file
    first, second, alias = (tmp_path / name for name in ("first", "second", "alias"))
    first.write_bytes(b"abcd")
    second.write_bytes(b"abcd")
    alias.symlink_to(first)
    initial = hash_data_file(alias)
    assert initial["resolved_path"] == str(first)
    with pytest.raises(ValueError, match="regular identity"):
        hash_file(alias)
    first.write_bytes(b"abce")
    assert hash_data_file(alias) != initial
    first.write_bytes(b"abcd")
    alias.unlink()
    alias.symlink_to(second)
    assert hash_data_file(alias) != initial


def test_standard_identity_tracks_cache_caption_config_and_rejects_custom(tmp_path):
    from library.datasets.dreambooth import DreamBoothDataset
    from library.datasets.group import DatasetGroup
    from library.datasets.subsets import ImageInfo
    from library.training.adaptive_runtime.resume_identity import build_data_identity
    from tests.test_training_recovery_cursor import BucketSamples
    sample = BucketSamples()
    dataset = DreamBoothDataset.__new__(DreamBoothDataset)
    dataset.__dict__.update(sample.__dict__)
    dataset.subsets = []
    source, latent, text = (tmp_path / n for n in ("image.png", "latent.npz", "text.safetensors"))
    for p in (source, latent, text):
        p.write_bytes(b"abcd")
    info = ImageInfo("a", 1, "caption", False, str(source))
    info.latents_npz, info.text_encoder_outputs_npz = str(latent), str(text)
    dataset.image_data = {"a": info}
    data = DatasetGroup([dataset])
    identity = build_data_identity(data, config={"family": "anima"})
    latent.write_bytes(b"abce")
    assert build_data_identity(data, config={"family": "anima"}) != identity
    latent.write_bytes(b"abcd")
    info.caption = "new caption"
    assert build_data_identity(data, config={"family": "anima"}) != identity
    info.caption = "caption"
    assert build_data_identity(data, config={"family": "krea2"}) != identity
    with pytest.raises(ValueError, match="custom"):
        build_data_identity(group())


@pytest.mark.parametrize("fault", ["traversal", "root_symlink", "child_symlink"])
def test_owned_path_rejects_escape_before_pickle(tmp_path, monkeypatch, fault):
    from library.training.adaptive_runtime.resume_checkpoint import _owned
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if fault == "traversal":
        path = root / ".." / "outside"
    elif fault == "root_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        path, root = alias / "snapshot", alias
    else:
        path = root / "snapshot"
        path.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle must not be loaded"))
    with pytest.raises(ValueError):
        _owned(path, root)


@pytest.mark.parametrize("fault", ["mode", "binding", "completed", "native_missing", "native_counts"])
def test_precision_and_counter_tampering_rejected_before_pickle(tmp_path, monkeypatch, fault):
    state, model, opt, sched = setup()
    train_steps(state.train_dataloader, model, opt, sched, 1)
    state.global_step = 1
    path = save_recovery_state(state, tmp_path, precision_contract_id="native", precision_manifest_id="p1",
                               counters={"global_step": 1, "completed_steps": 1},
                               precision_metadata={"mode": "fp16_fp32", "base": None, "execution": None})
    metadata = json.loads((path / "snapshot.json").read_text())
    precision = json.loads((path / "precision.json").read_text())
    if fault == "mode":
        del precision["mode"]
    elif fault == "binding":
        precision["precision_manifest_id"] = "different"
    elif fault == "completed":
        metadata["counters"]["completed_steps"] = 0
    else:
        precision["mode"] = "native_fp16"
        if fault == "native_counts":
            metadata["counters"].update(attempted_steps=1, effective_steps=1, skipped_steps=1)
    (path / "precision.json").write_text(json.dumps(precision))
    metadata["files"]["precision.json"] = hash_file(path / "precision.json")
    (path / "snapshot.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("pickle before metadata validation"))
    with pytest.raises(ValueError):
        validate_recovery_state(path, owner_root=tmp_path, precision_contract_id="native")
