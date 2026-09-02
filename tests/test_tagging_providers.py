from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from web.services.tagging import jobs
from web.services.tagging.local_worker_client import LocalWorkerError
from web.services.tagging.providers.base import LocalTaggingError
from web.services.tagging.providers.cltagger import CLTagger
from web.services.tagging.providers.factory import get_tagger
from web.services.tagging.providers.onnx_base import OnnxTaggerBase, _normalize_output
from web.services.tagging.providers.wd14 import WD14Tagger


class _DummyInput:
    name = "input"
    shape = [None, 3, 2, 2]


class _DummyOutput:
    name = "scores"


class _DummySession:
    def __init__(self, providers: list[str]):
        self._providers = providers
        self.batches: list[tuple[int, ...]] = []
        self.closed = False

    def get_providers(self):
        return list(self._providers)

    def get_inputs(self):
        return [_DummyInput()]

    def get_outputs(self):
        return [_DummyOutput()]

    def run(self, _outputs, feed):
        batch = next(iter(feed.values()))
        self.batches.append(tuple(batch.shape))
        return [np.ones((batch.shape[0], 2), dtype=np.float32)]

    def close(self):
        self.closed = True


class _CudaFailingSession(_DummySession):
    def run(self, _outputs, _feed):
        raise RuntimeError("CUDA out of memory")


class _DummyTagger(OnnxTaggerBase):
    name = "dummy"

    def prepare(self):
        raise AssertionError("the fixture should provide a prepared session")

    def _preprocess(self, _image):
        return np.zeros((3, 2, 2), dtype=np.float32)

    def _postprocess_one(self, scores):
        return ([str(float(scores[0]))], {"score": float(scores[0])})


def test_normalize_output_rejects_ambiguous_batches() -> None:
    assert _normalize_output(np.array([1.0, 2.0]), 1).shape == (1, 2)
    assert _normalize_output(np.array([1.0, 2.0]), 2).shape == (2, 1)
    assert _normalize_output(np.array([[1.0, 2.0]]), 1).shape == (1, 2)
    with np.testing.assert_raises(ValueError):
        _normalize_output(np.arange(6, dtype=np.float32), 2)
    with np.testing.assert_raises(ValueError):
        _normalize_output(np.array(1.0), 2)
    with np.testing.assert_raises(ValueError):
        _normalize_output(np.array([], dtype=np.float32), 1)


def test_cpu_and_directml_sessions_force_single_image_batches(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2), color=(index, 20, 30)).save(path)
        paths.append(path)

    cpu = _DummyTagger({"batch_size": 8})
    cpu._session = _DummySession(["CPUExecutionProvider"])
    cpu._input_name = "input"
    assert len(list(cpu.tag(paths))) == 2
    assert cpu._session.batches == [(1, 3, 2, 2), (1, 3, 2, 2)]

    dml = _DummyTagger({"batch_size": 8})
    dml._session = _DummySession(["DmlExecutionProvider", "CPUExecutionProvider"])
    dml._input_name = "input"
    assert len(list(dml.tag(paths))) == 2
    assert dml._session.batches == [(1, 3, 2, 2), (1, 3, 2, 2)]


def test_cuda_inference_retries_with_cpu_session(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2)).save(path)
    cpu_session = _DummySession(["CPUExecutionProvider"])
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=lambda _path, providers: cpu_session,
        ),
    )
    tagger = _DummyTagger({"batch_size": 4})
    gpu_session = _CudaFailingSession(["CUDAExecutionProvider", "CPUExecutionProvider"])
    tagger._session = gpu_session
    tagger._input_name = "input"
    tagger._model_path = tmp_path / "model.onnx"
    results = list(tagger.tag([path]))
    assert results[0]["tags"]
    assert tagger._session is cpu_session
    assert gpu_session.closed is True
    assert tagger.runtime_warning == "CUDA 推理失败，已回退 CPU"


def test_session_metadata_failure_closes_new_session(tmp_path: Path, monkeypatch) -> None:
    class NoInputSession(_DummySession):
        def get_inputs(self):
            return []

    session = NoInputSession(["CPUExecutionProvider"])
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=lambda _path, providers: session,
        ),
    )
    tagger = _DummyTagger({"device": "cpu"})
    with pytest.raises(LocalTaggingError, match="没有输入节点"):
        tagger._create_session(tmp_path / "model.onnx")
    assert session.closed is True
    assert tagger._session is None


def test_explicit_gpu_selection_fails_closed_without_cuda_provider(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=lambda path, providers: calls.append((path, providers)),
        ),
    )
    tagger = _DummyTagger({"device": "cuda", "gpu_index": 2})
    with pytest.raises(LocalTaggingError, match="GPU 2"):
        tagger._create_session(tmp_path / "model.onnx")
    assert calls == []


def test_explicit_gpu_selection_uses_logical_cuda_device_zero(tmp_path: Path, monkeypatch) -> None:
    captured = []
    session = _DummySession(["CUDAExecutionProvider", "CPUExecutionProvider"])

    def create_session(_path, providers):
        captured.append(providers)
        return session

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            InferenceSession=create_session,
        ),
    )
    tagger = _DummyTagger({"device": "cuda", "gpu_index": 2})
    tagger._create_session(tmp_path / "model.onnx")
    assert captured == [[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]]
    assert tagger._using_gpu is True


def test_wd14_postprocess_applies_categories_thresholds_and_blacklist() -> None:
    tagger = WD14Tagger(
        {
            "general_threshold": 0.4,
            "character_threshold": 0.8,
            "blacklist": ["blue_hair"],
        }
    )
    tagger._tags = ["low", "character_tag", "rating", "blue_hair", "high"]
    tagger._categories = [0, 4, 9, 0, 0]
    tags, scores = tagger._postprocess_one(np.array([0.5, 0.7, 0.99, 0.99, 0.9]))
    assert tags == ["high", "low"]
    assert scores["high"] == 0.9


def test_cltagger_mapping_sigmoid_and_category_gates(tmp_path: Path) -> None:
    mapping = {
        "idx_to_tag": ["artist_tag", "character_tag", "copyright_tag"],
        "tag_to_category": {
            "artist_tag": "Artist",
            "character_tag": "Character",
            "copyright_tag": "Copyright",
        },
    }
    path = tmp_path / "tag_mapping.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    tagger = CLTagger(
        {
            "general_threshold": 0.4,
            "character_threshold": 0.95,
            "add_artist_tag": False,
            "add_copyright_tag": True,
        }
    )
    tagger._labels = tagger._load_tag_mapping(path)
    tags, scores = tagger._postprocess_one(np.array([0.0, 2.0, 2.0]))
    assert tags == ["copyright tag"]
    assert 0.88 < scores["copyright tag"] < 0.89


def test_provider_factory_fails_closed() -> None:
    assert get_tagger("wd14", {}).name == "wd14"
    assert get_tagger("cltagger", {}).name == "cltagger"
    try:
        get_tagger("unsupported", {})
    except ValueError as exc:
        assert "不支持的本地打标 provider" in str(exc)
    else:
        raise AssertionError("unknown providers must fail closed")


def _patch_job_images(tmp_path: Path, monkeypatch, count: int = 1) -> list[Path]:
    paths = []
    for index in range(count):
        path = tmp_path / "images" / f"image-{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color=(index, 30, 40)).save(path)
        paths.append(path)

    def resolve(_dataset, _index, image_file, *, source):
        path = Path(image_file)
        return {
            "path": path,
            "file": path.as_posix(),
            "name": path.name,
            "url": "",
            "thumbnail_url": "",
            "caption": {"text": ""},
            "source": source,
        }

    monkeypatch.setattr(jobs, "resolve_tagging_image", resolve)
    return paths


def _local_settings() -> dict:
    return {
        "provider": "wd14",
        "asset_id": "fixture",
        "device": "cpu",
        "batch_size": 4,
        "general_threshold": 0.35,
        "character_threshold": 0.85,
        "blacklist": [],
    }


class _FakeLocalWorker:
    runtime_warning = ""

    def __init__(self, tagger):
        self.tagger = tagger
        self.stopped = False

    async def run(self, paths):
        available, reason = self.tagger.is_available()
        if not available:
            raise LocalWorkerError(reason, phase="initialization")
        return [dict(result) for result in self.tagger.tag(paths)]

    async def stop(self):
        self.stopped = True


def _fake_worker_factory(tagger):
    return lambda **_kwargs: _FakeLocalWorker(tagger)


def test_local_job_accepts_empty_prompts_and_converts_tags(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_job_images(tmp_path, monkeypatch)
    monkeypatch.setattr(jobs, "load_settings", lambda: _local_settings())

    class FakeTagger:
        def is_available(self):
            return True, "fixture ready"

        def tag(self, values):
            yield {"image": values[0], "tags": ["1girl", "blue hair"]}

    tagger = FakeTagger()

    async def run() -> None:
        manager = jobs.TaggingJobManager(local_worker_factory=_fake_worker_factory(tagger))
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "items": [paths[0].as_posix()],
            }
        )
        job_id = created["job"]["id"]
        await manager._tasks[job_id]
        snapshot = manager.snapshot(job_id)["job"]
        assert snapshot["state"] == "completed"
        assert snapshot["items"][0]["state"] == "ready"
        assert snapshot["items"][0]["proposed_caption"] == "1girl, blue hair"
        assert snapshot["prompt"] == ""
        assert any(line["event"] == "item_succeeded" for line in manager.get_logs(job_id=job_id)["lines"])

    asyncio.run(run())


def test_local_job_reports_unavailable_provider(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_job_images(tmp_path, monkeypatch)
    monkeypatch.setattr(jobs, "load_settings", lambda: _local_settings())

    class FakeTagger:
        def is_available(self):
            return False, "runtime missing"

    tagger = FakeTagger()

    async def run() -> None:
        manager = jobs.TaggingJobManager(local_worker_factory=_fake_worker_factory(tagger))
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "items": [paths[0].as_posix()],
            }
        )
        job_id = created["job"]["id"]
        await manager._tasks[job_id]
        snapshot = manager.snapshot(job_id)["job"]
        assert snapshot["state"] == "failed"
        assert snapshot["items"][0]["error"] == "runtime missing"
        assert any(line["event"] == "provider_failed" for line in manager.get_logs(job_id=job_id)["lines"])

    asyncio.run(run())


def test_local_job_preserves_path_mapping_and_partial_states(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_job_images(tmp_path, monkeypatch, count=3)
    monkeypatch.setattr(jobs, "load_settings", lambda: _local_settings())

    class FakeTagger:
        def is_available(self):
            return True, "ready"

        def tag(self, values):
            yield {"image": values[1], "tags": ["second"]}
            yield {"image": values[0], "error": "bad image"}
            yield {"image": values[2], "tags": []}

    tagger = FakeTagger()

    async def run() -> None:
        manager = jobs.TaggingJobManager(local_worker_factory=_fake_worker_factory(tagger))
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "items": [path.as_posix() for path in paths],
            }
        )
        job_id = created["job"]["id"]
        await manager._tasks[job_id]
        items = manager.snapshot(job_id)["job"]["items"]
        assert manager.snapshot(job_id)["job"]["state"] == "partial"
        assert [item["state"] for item in items] == ["failed", "ready", "empty"]
        assert items[1]["proposed_caption"] == "second"

    asyncio.run(run())


def test_local_job_in_place_rerun_only_sends_selected_images(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_job_images(tmp_path, monkeypatch, count=3)
    local_settings = {
        **_local_settings(),
        "_profile_id": "local-profile",
        "_profile_name": "本地模型",
    }
    monkeypatch.setattr(jobs, "load_settings", lambda: dict(local_settings))
    monkeypatch.setattr(jobs, "get_effective_settings", lambda _profile_id: dict(local_settings))
    calls: list[list[str]] = []

    class FakeTagger:
        def is_available(self):
            return True, "ready"

        def tag(self, values):
            calls.append([str(value) for value in values])
            for value in values:
                yield {"image": value, "tags": [Path(value).stem]}

    async def run() -> None:
        manager = jobs.TaggingJobManager(local_worker_factory=_fake_worker_factory(FakeTagger()))
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "items": [path.as_posix() for path in paths],
            }
        )
        job_id = created["job"]["id"]
        await manager._tasks[job_id]
        first = manager.snapshot(job_id)["job"]
        item_id = first["items"][1]["id"]
        rerun = await manager.rerun(job_id, item_ids=[item_id])
        assert rerun["job"]["id"] == job_id
        assert len(manager.jobs) == 1
        await manager._tasks[job_id]
        final = manager.snapshot(job_id)["job"]
        assert calls == [[path.as_posix() for path in paths], [paths[1].as_posix()]]
        assert final["items"][0]["proposed_caption"] == paths[0].stem
        assert final["items"][1]["proposed_caption"] == paths[1].stem

    asyncio.run(run())


def test_local_job_does_not_positionally_reuse_mismatched_image_result(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    unrelated = tmp_path / "unrelated.png"
    for path in (first, second, unrelated):
        path.write_bytes(b"fixture")

    def resolve(_dataset_file, _dataset_index, image_file, *, source):
        path = Path(image_file)
        return {
            "path": path,
            "file": path.as_posix(),
            "name": path.name,
            "url": "",
            "thumbnail_url": "",
            "caption": {"text": ""},
            "source": source,
        }

    monkeypatch.setattr(jobs, "resolve_tagging_image", resolve)
    monkeypatch.setattr(jobs, "load_settings", lambda: _local_settings())

    class FakeTagger:
        def is_available(self):
            return True, "ready"

        def tag(self, _values):
            return [
                {"image": unrelated.as_posix(), "caption": "wrong image"},
                {"image": second.as_posix(), "caption": "second caption"},
            ]

    tagger = FakeTagger()

    async def run() -> None:
        manager = jobs.TaggingJobManager(local_worker_factory=_fake_worker_factory(tagger))
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "items": [first.as_posix(), second.as_posix()],
            }
        )
        job_id = created["job"]["id"]
        await manager._tasks[job_id]
        items = manager.snapshot(job_id)["job"]["items"]
        assert [item["state"] for item in items] == ["failed", "ready"]
        assert items[0]["proposed_caption"] == ""
        assert items[0]["error"] == "本地模型没有返回结果"
        assert items[1]["proposed_caption"] == "second caption"

    asyncio.run(run())


def test_local_job_normalizes_unexpected_inference_exception(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_job_images(tmp_path, monkeypatch)
    monkeypatch.setattr(jobs, "load_settings", lambda: _local_settings())

    class FakeTagger:
        def is_available(self):
            return True, "ready"

        def tag(self, _values):
            raise RuntimeError("session exploded")
            yield  # pragma: no cover

    tagger = FakeTagger()

    async def run() -> None:
        manager = jobs.TaggingJobManager(local_worker_factory=_fake_worker_factory(tagger))
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "items": [paths[0].as_posix()],
            }
        )
        job_id = created["job"]["id"]
        await manager._tasks[job_id]
        snapshot = manager.snapshot(job_id)["job"]
        assert snapshot["state"] == "failed"
        assert snapshot["items"][0]["error"] == "session exploded"
        assert any(line["event"] == "inference_failed" for line in manager.get_logs(job_id=job_id)["lines"])

    asyncio.run(run())


def test_local_job_cancel_waits_for_worker_cleanup(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_job_images(tmp_path, monkeypatch)
    monkeypatch.setattr(jobs, "load_settings", lambda: _local_settings())
    started = asyncio.Event()
    released = asyncio.Event()

    class BlockingWorker:
        runtime_warning = ""

        async def run(self, _paths):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                released.set()
                raise

        async def stop(self):
            released.set()

    async def run() -> None:
        manager = jobs.TaggingJobManager(local_worker_factory=lambda **_kwargs: BlockingWorker())
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "items": [paths[0].as_posix()],
            }
        )
        job_id = created["job"]["id"]
        await asyncio.wait_for(started.wait(), timeout=1)
        snapshot = await manager.cancel(job_id)
        assert snapshot["job"]["state"] == "canceled"
        assert released.is_set()
        assert job_id not in manager._tasks
        assert job_id not in manager._local_workers

    asyncio.run(run())
