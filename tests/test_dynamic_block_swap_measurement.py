import json
from types import SimpleNamespace

import torch

from scripts.experiments import dynamic_block_swap_measurement as measurement


def test_measurement_times_full_accumulated_update(monkeypatch, tmp_path):
    times = iter([10, 30])
    monkeypatch.setattr(measurement.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 50)
    recorder = measurement.UpdateRecorder(
        SimpleNamespace(output=tmp_path, memory_history=False),
        SimpleNamespace(blocks_to_swap=26),
        lambda *_: torch.tensor(1.0),
        None,
    )
    state = SimpleNamespace(
        global_step=2, accelerator=SimpleNamespace(sync_gradients=False)
    )
    recorder(None, state, {"latents": torch.zeros(1, 16, 8, 8)})
    assert not (tmp_path / "updates.jsonl").exists()
    state.accelerator.sync_gradients = True
    recorder(None, state, {"latents": torch.zeros(1, 16, 16, 8)})
    row = json.loads((tmp_path / "updates.jsonl").read_text())
    assert row["seconds"] == 20 and row["microsteps"] == 2
    assert row["shapes"] == [[1, 16, 8, 8], [1, 16, 16, 8]]
    assert not recorder.shapes
