from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from web.services.tagging.local_worker_client import (
    LocalTaggingWorkerClient,
    LocalWorkerError,
    build_worker_environment,
)
from web.services.tagging.local_worker_protocol import sanitize_result


def _write_worker_script(path: Path, *, blocking: bool = False) -> None:
    body = """
import json
import os
import sys
import time

request = json.loads(sys.stdin.readline())
base = {"version": 1, "job_id": request["job_id"]}
def emit(sequence, event_type, **payload):
    print(json.dumps({**base, "seq": sequence, "type": event_type, **payload}), flush=True)

emit(1, "ready", provider=request["provider"], runtime_warning="")
"""
    if blocking:
        body += "time.sleep(60)\n"
    else:
        body += """
emit(2, "result", index=0, result={"image": request["images"][0], "tags": ["fixture"], "caption": os.environ.get("CUDA_VISIBLE_DEVICES", "")})
emit(3, "progress", done=1, total=1)
emit(4, "complete", results=1, total=1)
"""
    path.write_text(body, encoding="utf-8")


def test_worker_environment_maps_one_gpu_without_mutating_parent() -> None:
    base = {"CUDA_VISIBLE_DEVICES": "7", "PATH": "/bin"}
    selected = build_worker_environment({"device": "cuda", "gpu_index": 2}, base)
    assert selected["CUDA_VISIBLE_DEVICES"] == "2"
    assert selected["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert base["CUDA_VISIBLE_DEVICES"] == "7"

    inherited = build_worker_environment({"device": "cpu", "gpu_index": 2}, base)
    assert inherited["CUDA_VISIBLE_DEVICES"] == "7"


def test_worker_result_sanitizer_keeps_missing_image_for_positional_fallback() -> None:
    assert "image" not in sanitize_result({"image": "", "tags": ["fixture"]})


def test_worker_client_completes_and_reaps_process(tmp_path: Path) -> None:
    script = tmp_path / "fixture_worker.py"
    _write_worker_script(script)

    async def run() -> None:
        client = LocalTaggingWorkerClient(
            provider="wd14",
            settings={"device": "cuda", "gpu_index": 3},
            job_id="job-complete",
            command=[sys.executable, str(script)],
            root=tmp_path,
        )
        results = await client.run([tmp_path / "sample.png"])
        assert results[0]["tags"] == ["fixture"]
        assert results[0]["caption"] == "3"
        assert client.last_pid is not None and client.last_pid != os.getpid()
        assert client.last_returncode == 0
        assert client.pid is None
        assert client.last_progress == (1, 1)

    asyncio.run(run())


def test_worker_client_cancel_terminates_and_reaps_process(tmp_path: Path) -> None:
    script = tmp_path / "blocking_worker.py"
    _write_worker_script(script, blocking=True)

    async def run() -> None:
        client = LocalTaggingWorkerClient(
            provider="wd14",
            settings={"device": "cpu"},
            job_id="job-cancel",
            command=[sys.executable, str(script)],
            root=tmp_path,
        )
        task = asyncio.create_task(client.run([tmp_path / "sample.png"]))
        for _ in range(100):
            if client.pid is not None:
                break
            await asyncio.sleep(0.01)
        assert client.pid is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.pid is None
        assert client.last_returncode is not None

    asyncio.run(run())


def test_worker_client_reaps_process_when_cancel_races_with_spawn(tmp_path: Path) -> None:
    script = tmp_path / "spawn_race_worker.py"
    _write_worker_script(script, blocking=True)

    async def run() -> None:
        client = LocalTaggingWorkerClient(
            provider="wd14",
            settings={"device": "cpu"},
            job_id="job-spawn-race",
            command=[sys.executable, str(script)],
            root=tmp_path,
        )
        create_process = client._create_process

        async def delayed_create_process():
            await asyncio.sleep(0.05)
            return await create_process()

        client._create_process = delayed_create_process
        task = asyncio.create_task(client.run([tmp_path / "sample.png"]))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.pid is None
        assert client.last_pid is not None
        assert client.last_returncode is not None

    asyncio.run(run())


def test_worker_client_rejects_inconsistent_complete_event(tmp_path: Path) -> None:
    script = tmp_path / "bad_complete_worker.py"
    script.write_text(
        """
import json
import sys
request = json.loads(sys.stdin.readline())
base = {"version": 1, "job_id": request["job_id"]}
print(json.dumps({**base, "seq": 1, "type": "ready"}), flush=True)
print(json.dumps({**base, "seq": 2, "type": "complete", "results": 1, "total": 1}), flush=True)
""",
        encoding="utf-8",
    )

    async def run() -> None:
        client = LocalTaggingWorkerClient(
            provider="wd14",
            settings={"device": "cpu"},
            job_id="job-bad-complete",
            command=[sys.executable, str(script)],
            root=tmp_path,
        )
        with pytest.raises(LocalWorkerError, match="结果数量不一致"):
            await client.run([tmp_path / "sample.png"])
        assert client.pid is None

    asyncio.run(run())
