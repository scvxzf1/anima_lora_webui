"""Worker-zero data RNG prototype for torchdata + Accelerate look-ahead.

Not wired into training: cache identity, overflow and full resume contracts are
still required. Only CPU dataset code on the main thread is supported here.
"""

from contextlib import contextmanager
from copy import deepcopy
import random
import threading

import numpy as np
import torch

from .bucket_state import capture_buckets, restore_buckets


def _rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state())


def _set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def prepare_cursor_loader(group, *, seed):
    from accelerate.data_loader import prepare_data_loader
    from torchdata.stateful_dataloader.sampler import BatchSampler, RandomSampler

    data = StatefulDatasetView(group, seed=seed)
    sampler = RandomSampler(data, generator=torch.Generator().manual_seed(seed + 1))
    raw = torch.utils.data.DataLoader(
        data, batch_sampler=BatchSampler(sampler, 1, False), num_workers=0,
        generator=torch.Generator().manual_seed(seed + 2))
    return prepare_data_loader(raw, device=torch.device("cpu"), num_processes=1, process_index=0,
                               put_on_device=False, use_stateful_dataloader=True)


def capture_loader(loader):
    return deepcopy({"schema": "stateful_loader_prototype_v1",
                     "loader": loader.state_dict(), "iteration": loader.iteration})


def restore_loader(loader, snapshot):
    snapshot = deepcopy(snapshot)
    if (snapshot.get("schema") != "stateful_loader_prototype_v1"
            or type(snapshot.get("iteration")) is not int or snapshot["iteration"] < 0
            or not isinstance(snapshot.get("loader"), dict)
            or type(snapshot["loader"].get("_iterator_finished")) is not bool):
        raise ValueError("Invalid stateful loader snapshot")
    loader.iteration = snapshot["iteration"]
    loader.load_state_dict(snapshot["loader"])
    # Materialize torchdata's lazy restore BEFORE Accelerate sets the epoch.
    # Otherwise an exhausted-state restore overwrites the next epoch's shuffle.
    loader.base_dataloader.state_dict()
    if snapshot["loader"]["_iterator_finished"]:
        loader.iteration += 1


class StatefulDatasetView(torch.utils.data.Dataset):
    def __init__(self, group, *, seed):
        self.group = group
        self.epoch = -1
        self.fetched = 0
        self._rng = (random.Random(seed).getstate(), np.random.RandomState(seed % 2**32).get_state(),
                     torch.Generator().manual_seed(seed).get_state())

    @contextmanager
    def _data_rng(self):
        if (threading.current_thread() is not threading.main_thread()
                or torch.utils.data.get_worker_info() is not None):
            raise ValueError("Data RNG prototype supports main-thread worker=0 only")
        previous = _rng_state()
        try:
            _set_rng(self._rng)
            yield
        finally:
            self._rng = _rng_state()
            _set_rng(previous)

    def __len__(self):
        return len(self.group)

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            with self._data_rng():
                self.group.set_current_epoch(epoch + 1)
            self.epoch = epoch

    def __getitem__(self, index):
        with self._data_rng():
            self.group.set_current_step(self.fetched)
            item = self.group[index]
            self.fetched += 1
            return item

    def state_dict(self):
        return deepcopy({"schema": "isolated_data_rng_prototype_v1", "epoch": self.epoch,
                         "fetched": self.fetched, "rng": self._rng,
                         "buckets": capture_buckets(self.group)})

    def load_state_dict(self, state):
        state = deepcopy(state)
        if state.get("schema") != "isolated_data_rng_prototype_v1":
            raise ValueError("Unknown data RNG snapshot")
        if type(state["fetched"]) is not int or state["fetched"] < 0:
            raise ValueError("Invalid fetched count")
        if type(state["epoch"]) is not int or state["epoch"] < -1:
            raise ValueError("Invalid data epoch")
        previous = _rng_state()
        try:
            _set_rng(state["rng"])
        finally:
            _set_rng(previous)
        restore_buckets(self.group, state["buckets"])
        self.epoch, self.fetched, self._rng = state["epoch"], state["fetched"], state["rng"]
