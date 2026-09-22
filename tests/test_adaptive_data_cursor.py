from copy import deepcopy
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bench.adaptive_runtime.bucket_state import capture_buckets, restore_buckets
from bench.adaptive_runtime.stateful_dataset import (
    StatefulDatasetView, capture_loader, prepare_cursor_loader, restore_loader,
)
from library.datasets.base import BaseDataset
from library.datasets.buckets import BucketBatchIndex, BucketManager
from library.datasets.group import DatasetGroup


class BucketSamples(BaseDataset):
    """Use actual epoch/shuffle/length behavior with bounded random CPU samples."""

    def __init__(self, offset=0):
        self.seed = 42 + offset
        self.current_epoch = self.current_step = 0
        self.max_train_steps = 20
        self.caching_mode = None
        self._length = 6
        self._largest_bucket_index = None
        self.bucket_manager = BucketManager((256, 256), 128, 512, 64)
        self.bucket_manager.resos = [(256, 256), (256, 320)]
        self.bucket_manager.reso_to_id = {(256, 256): 0, (256, 320): 1}
        self.bucket_manager.buckets = [[f"{offset}:a", f"{offset}:b", f"{offset}:a"],
                                       [f"{offset}:c", f"{offset}:d", f"{offset}:c"]]
        self.buckets_indices = [BucketBatchIndex(bucket, 1, i) for bucket in range(2) for i in range(3)]
        self.image_data = {key: SimpleNamespace() for bucket in self.bucket_manager.buckets for key in bucket}
        self.num_train_images, self.num_reg_images = 6, 0
        self.shuffle_buckets()

    def __getitem__(self, index):
        entry = self.buckets_indices[index]
        key = self.bucket_manager.buckets[entry.bucket_index][entry.batch_index]
        return {"key": key, "bucket": entry.bucket_index, "step": self.current_step,
                "value": torch.tensor([random.random(), np.random.rand(), torch.rand(()).item()])}


def group():
    return DatasetGroup([BucketSamples(), BucketSamples(1)])


def test_bucket_order_roundtrip_preserves_repeats_and_future_epoch_shuffle():
    original = group()
    original.set_current_epoch(3)
    snapshot = capture_buckets(original)
    restored = group()
    restore_buckets(restored, snapshot)
    assert capture_buckets(restored) == snapshot
    original.set_current_epoch(4)
    restored.set_current_epoch(4)
    assert capture_buckets(restored) == capture_buckets(original)


def test_bucket_restore_rejects_drift_before_mutating_first_dataset():
    current = group()
    before = capture_buckets(current)
    bad = deepcopy(before)
    bad["datasets"][0]["current_epoch"] = 9
    bad["datasets"][1]["buckets"][0].pop()
    with pytest.raises(ValueError, match="membership"):
        restore_buckets(current, bad)
    assert capture_buckets(current) == before


def test_data_rng_isolated_from_training_rng():
    data = StatefulDatasetView(group(), seed=5)
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)
    expected = (random.random(), np.random.rand(), torch.rand(()))
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)
    data.set_epoch(0)
    data[0]
    data[1]
    actual = (random.random(), np.random.rand(), torch.rand(()))
    assert actual == expected


def make_loader():
    pytest.importorskip("torchdata.stateful_dataloader")
    return prepare_cursor_loader(group(), seed=18)


@pytest.mark.parametrize("cut", [1, 5, 11, 12])
def test_accelerate_prefetch_and_next_epoch_replay(tmp_path, cut):
    loader = make_loader()
    iterator = iter(loader)
    for _ in range(cut):
        next(iterator)
        torch.rand(7)
        random.random()
        np.random.rand()
    captured = capture_loader(loader)
    data_state = captured["loader"]["dataset_state"]
    assert data_state["fetched"] == cut and data_state["epoch"] == 0
    assert loader.dataset.fetched == min(cut + 1, len(loader))
    assert data_state["buckets"]["datasets"][0]["current_epoch"] == 1
    # Real serialization catches mutable-reference snapshots and tensor states.
    path = tmp_path / "cursor.pt"
    torch.save(captured, path)
    expected = list(iterator) + list(loader)
    # Disturb all training RNG streams; data replay must remain independent.
    random.seed(999)
    np.random.seed(998)
    torch.manual_seed(997)
    restored = make_loader()
    saved = torch.load(path, weights_only=False)
    restore_loader(restored, saved)
    assert restored.dataset.fetched == cut and restored.dataset.epoch == 0
    actual = list(restored)
    if cut < 12:
        actual += list(restored)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a["key"] == b["key"]
        torch.testing.assert_close(a["bucket"], b["bucket"], rtol=0, atol=0)
        torch.testing.assert_close(a["step"], b["step"], rtol=0, atol=0)
        torch.testing.assert_close(a["value"], b["value"], rtol=0, atol=0)


def test_invalid_loader_snapshot_is_rejected_before_restore():
    loader = make_loader()
    with pytest.raises(ValueError, match="snapshot"):
        restore_loader(loader, {"iteration": -1})


def training_setup():
    loader = make_loader()
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    return loader, model, optimizer, scheduler


def train_steps(loader, model, optimizer, scheduler, count, *, iterator=None):
    if iterator is None:
        iterator = iter(loader)
    trace = []
    for _ in range(count):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        target = torch.randn(1, 1) + random.random() + np.random.rand()
        loss = (model(batch["value"]) - target).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        trace.append({**batch, "target": target, "loss": loss.detach()})
    return trace, iterator


def resume_in_fresh_process(directory):
    saved = torch.load(directory / "training.pt", weights_only=False)
    loader, model, optimizer, scheduler = training_setup()
    restore_loader(loader, saved["loader"])
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    random.setstate(saved["rng"][0])
    np.random.set_state(saved["rng"][1])
    torch.set_rng_state(saved["rng"][2])
    trace, _ = train_steps(loader, model, optimizer, scheduler, saved["remaining"])
    torch.save({"trace": trace, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}, directory / "resumed.pt")


@pytest.mark.parametrize("cut", [5, 12])
def test_new_process_training_trajectory_with_data_and_rng(tmp_path, cut):
    loader, model, optimizer, scheduler = training_setup()
    _, iterator = train_steps(loader, model, optimizer, scheduler, cut)
    torch.save({"loader": capture_loader(loader), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state()),
                "remaining": 18 - cut}, tmp_path / "training.pt")
    trace, _ = train_steps(loader, model, optimizer, scheduler, 18 - cut, iterator=iterator)
    script = ("import runpy,sys; from pathlib import Path; "
              f"ns=runpy.run_path({str(Path(__file__).resolve())!r}); "
              "ns['resume_in_fresh_process'](Path(sys.argv[1]))")
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    resumed = torch.load(tmp_path / "resumed.pt", weights_only=False)
    assert len(resumed["trace"]) == len(trace)
    for expected, actual in zip(trace, resumed["trace"]):
        assert expected["key"] == actual["key"]
        for name in ("value", "bucket", "step", "target", "loss"):
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    torch.testing.assert_close(resumed["model"], model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(resumed["optimizer"], optimizer.state_dict(), rtol=0, atol=0)
    assert resumed["scheduler"] == scheduler.state_dict()
