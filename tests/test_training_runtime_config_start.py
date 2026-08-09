"""Web runtime start/preflight wiring tests"""

from __future__ import annotations

from types import SimpleNamespace

from library.training.stage_schedule import active_subset_indices_for_step
from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_handle_start_uses_runtime_config_for_direct_training(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    run_dir = tmp_path / "output" / "runs" / "522-20260523-114514"
    run_dir.mkdir(parents=True)
    (run_dir / "model_cache").mkdir()
    (run_dir / "dataset_cache").mkdir()
    (run_dir / "training_output").mkdir()
    (run_dir / "config.runtime.toml").write_text(
        'output_dir = "output/runs/522-20260523-114514/training_output"\n',
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    svc = FakeService()
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(training_routes, "is_web_runtime_config", lambda value: value.endswith("config.runtime.toml"))
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "output/runs/522-20260523-114514/config.runtime.toml",
            "extra_args": ["--foo"],
            "confirmed": True,
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["job"] == "training"
    assert payload["train_after"] is False
    assert svc.preprocess_calls == []
    assert len(svc.start_calls) == 1
    args, kwargs = svc.start_calls[0]
    assert args[:4] == ("522", "default", ["--foo"], "imported")
    assert kwargs["config_file"] == "output/runs/522-20260523-114514/config.runtime.toml"
    assert kwargs["use_runtime_dir"] is False

def test_saved_form_values_are_frozen_into_runtime_config(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    prompts = config_service.save_sample_prompts_file(
        "masterpiece, character test\n",
        train_config_file="configs/imported/522.toml",
    )
    ok, msg, content, changed, _warnings = config_service.patch_raw_file_values(
        "configs/imported/522.toml",
        {
            "network_dim": 32,
            "sample_every_n_steps": 50,
            "blocks_to_swap": 12,
            "block_swap_restore_mode": "slab",
            "sample_prompts": prompts["file"],
        },
    )

    assert ok is True, msg
    assert set(changed) == {
        "blocks_to_swap",
        "block_swap_restore_mode",
        "network_dim",
        "sample_every_n_steps",
        "sample_prompts",
    }
    saved_cfg = toml.loads(content)
    assert saved_cfg["network_dim"] == 32
    assert saved_cfg["sample_every_n_steps"] == 50
    assert saved_cfg["blocks_to_swap"] == 12
    assert saved_cfg["block_swap_restore_mode"] == "slab"
    assert saved_cfg["sample_prompts"] == "configs/sample-prompts/imported/522.txt"

    reloaded_cfg = config_service.load_merged_config("522", "default", "imported")
    assert reloaded_cfg["network_dim"] == 32
    assert reloaded_cfg["sample_every_n_steps"] == 50
    assert reloaded_cfg["blocks_to_swap"] == 12
    assert reloaded_cfg["block_swap_restore_mode"] == "slab"
    assert reloaded_cfg["sample_prompts"] == "configs/sample-prompts/imported/522.txt"

    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )
    runtime_cfg = toml.load(tmp_path / runtime["runtime_config_file"])
    dataset_cfg = toml.load(tmp_path / runtime["dataset_config_file"])

    assert runtime_cfg["network_dim"] == 32
    assert runtime_cfg["sample_every_n_steps"] == 50
    assert runtime_cfg["blocks_to_swap"] == 12
    assert runtime_cfg["block_swap_restore_mode"] == "slab"
    assert runtime_cfg["sample_prompts"] == "configs/sample-prompts/imported/522.txt"
    assert runtime_cfg["dataset_config"].endswith("dataset.runtime.toml")
    assert runtime_cfg["source_image_dir"] == "image_dataset/a"
    assert runtime_cfg["resized_image_dir"].endswith("/dataset_cache/dataset-01/resized")
    assert runtime_cfg["lora_cache_dir"].endswith("/dataset_cache/dataset-01/lora")

    first_subset = dataset_cfg["datasets"][0]["subsets"][0]
    assert first_subset["custom_attributes"]["source_dir"] == "image_dataset/a"
    assert first_subset["image_dir"].endswith("/dataset_cache/dataset-01/resized")
    assert first_subset["cache_dir"].endswith("/dataset_cache/dataset-01/lora")

def test_handle_start_requires_explicit_confirmation_for_runtime_config(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    svc = FakeService()
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(training_routes, "is_web_runtime_config", lambda value: value.endswith("config.runtime.toml"))
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "output/runs/522-20260523-114514/config.runtime.toml",
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 409
    payload = json.loads(response.text)
    assert payload["requires_confirmation"] is True
    assert payload["requires_preprocess_confirmation"] is False
    assert svc.preprocess_calls == []
    assert svc.start_calls == []

def test_handle_start_uses_runtime_config_from_absolute_output_root(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path / "external-runs"
    run_dir = output_root / "522-20260523-114514"
    model_cache = run_dir / "model_cache"
    dataset_cache = run_dir / "dataset_cache" / "dataset-01"
    training_output = run_dir / "training_output"
    source_dir = tmp_path / "image_dataset" / "a"
    for path in (model_cache, dataset_cache / "resized", dataset_cache / "lora", training_output, source_dir):
        path.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(dataset_cache / "resized" / "a.png")
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(source_dir / "a.png")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    runtime_config = run_dir / "config.runtime.toml"
    dataset_config = run_dir / "dataset.runtime.toml"
    runtime_config.write_text(
        "\n".join(
            [
                f'dataset_config = "{dataset_config.as_posix()}"',
                f'output_dir = "{training_output.as_posix()}"',
                f'logging_dir = "{(model_cache / "logs").as_posix()}"',
                'source_image_dir = "image_dataset/a"',
                f'resized_image_dir = "{(dataset_cache / "resized").as_posix()}"',
                f'lora_cache_dir = "{(dataset_cache / "lora").as_posix()}"',
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    dataset_config.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                f'image_dir = "{(dataset_cache / "resized").as_posix()}"',
                f'cache_dir = "{(dataset_cache / "lora").as_posix()}"',
                'custom_attributes = {source_dir = "image_dataset/a"}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: output_root.resolve())

    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": str(runtime_config),
            "confirmed": True,
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["job"] == "training"
    assert svc.preprocess_calls == []
    assert len(svc.start_calls) == 1
    assert svc.start_calls[0][1]["config_file"] == str(runtime_config)
    assert svc.start_calls[0][1]["use_runtime_dir"] is False

def test_start_after_preprocess_uses_runtime_config_for_preflight(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    captured = {}

    def fake_preflight(*args, **kwargs):
        captured["preflight_args"] = args
        captured["preflight_kwargs"] = kwargs
        return {"ok": True}

    svc = TrainingService(web.Application())
    monkeypatch.setattr(training_service, "preflight_training_config", fake_preflight)

    async def fake_start(*args, **kwargs):
        captured["start_args"] = args
        captured["start_kwargs"] = kwargs

    svc.start = fake_start

    asyncio.run(
        svc._start_pending_training(
            {
                "variant": "522",
                "preset": "default",
                "methods_subdir": "imported",
                "extra_args": [],
                "config_file": "output/runs/522-20260523-114514/config.runtime.toml",
                "source_config_file": "configs/imported/522.toml",
                "gpu_whitelist": [0],
            }
        )
    )

    assert captured["preflight_kwargs"]["config_file"] == "output/runs/522-20260523-114514/config.runtime.toml"
    assert captured["start_kwargs"]["config_file"] == "output/runs/522-20260523-114514/config.runtime.toml"
    assert captured["start_kwargs"]["source_config_file"] == "configs/imported/522.toml"
    assert captured["start_kwargs"]["use_runtime_dir"] is False


def test_second_prepare_reuses_pool_without_full_private_copy(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    for name, caption in (("a", "1girl"), ("b", "1boy")):
        source = tmp_path / "image_dataset" / name
        source.mkdir(parents=True, exist_ok=True)
        (source / f"{name}.png").write_bytes(b"\x89PNG\r\nfake")
        (source / f"{name}.txt").write_text(caption, encoding="utf-8")

    class FirstRunDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 11, 45, 14)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FirstRunDatetime)
    training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )
    first_run = tmp_path / "output" / "runs" / "522-20260523-114514"
    metadata = json.loads((first_run / "run.meta.json").read_text(encoding="utf-8"))
    bindings = metadata.get("dataset_cache_bindings") or []
    assert bindings
    assert metadata.get("cache_pool_root")

    pool_path = Path(bindings[0]["pool_path"])
    if not pool_path.is_absolute():
        pool_path = tmp_path / pool_path
    sentinel = pool_path / "resized" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("shared", encoding="utf-8")

    class SecondRunDatetime(FirstRunDatetime):
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 0, 0)

    monkeypatch.setattr(training_service, "datetime", SecondRunDatetime)
    training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    second_cache = (
        tmp_path
        / "output"
        / "runs"
        / "522-20260523-120000"
        / "dataset_cache"
        / "dataset-01"
        / "resized"
    )
    assert (second_cache / "sentinel.txt").read_text(encoding="utf-8") == "shared"
    pool_root = tmp_path / "output" / "cache_pool"
    entries = [
        path
        for path in pool_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert entries


def test_web_runtime_trigger_clone_materializes_extra_subset(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    source_root = tmp_path / "image_dataset" / "mixed"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_root / "top.png")
    Image.new("RGB", (8, 8), color=(60, 40, 20)).save(nested / "sample.png")
    captions = {
        "top.png": ["1girl, solo, silver hair, purple eyes, white dress"],
        "nested/sample.png": ["1girl, solo, forest, moonlight"],
    }
    (source_root / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False), encoding="utf-8"
    )
    second_source = tmp_path / "image_dataset" / "second"
    second_source.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 80, 40)).save(second_source / "second.png")
    (second_source / "second.txt").write_text("second character", encoding="utf-8")

    (tmp_path / "configs" / "datasets" / "522.toml").write_text(
        "\n".join(
            [
                "stage_schedule_enabled = true",
                'stage_schedule = [{name = "mixed", subset_index = 0, start_pct = 0.0, end_pct = 0.5}, {name = "second", subset_index = 1, start_pct = 0.5, end_pct = 1.0}]',
                "",
                "[[datasets]]",
                'caption_source_mode = "captions_json"',
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/mixed_resized"',
                'cache_dir = "old/mixed_lora"',
                "recursive = true",
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 1.0}, trigger_clone = {enabled = true, prompt = "my_character", num_repeats = 3}}',
                "num_repeats = 5",
                "",
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/second_resized"',
                'cache_dir = "old/second_lora"',
                'custom_attributes = {source_dir = "image_dataset/second"}',
                "num_repeats = 2",
            ]
        ),
        encoding="utf-8",
    )

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 8, 0)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)
    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    run_dir = tmp_path / "output" / "runs" / "522-20260523-120800"
    clone_source = run_dir / "dataset_cache" / "dataset-01" / "trigger-clone-source"
    assert (clone_source / "top.png").is_file()
    assert (clone_source / "nested" / "sample.png").is_file()
    materialized = json.loads(
        (clone_source / "captions.json").read_text(encoding="utf-8")
    )
    assert materialized == {
        "nested/sample.png": ["my_character"],
        "top.png": ["my_character"],
    }
    manifest = json.loads(
        (clone_source / "trigger-clone-results.json").read_text(encoding="utf-8")
    )
    assert manifest["source_dir"].endswith("dataset-01/source")
    assert manifest["num_repeats"] == 3
    assert manifest["total"] == 2

    dataset_cfg = toml.loads(
        (run_dir / "dataset.runtime.toml").read_text(encoding="utf-8")
    )
    assert len(dataset_cfg["datasets"]) == 3
    original_subset = dataset_cfg["datasets"][0]["subsets"][0]
    clone_dataset = dataset_cfg["datasets"][1]
    clone_subset = clone_dataset["subsets"][0]
    assert original_subset["num_repeats"] == 5
    assert clone_subset["num_repeats"] == 3
    assert clone_subset["image_dir"].endswith("dataset-01/trigger-clone-resized")
    assert clone_subset["cache_dir"].endswith("dataset-01/trigger-clone-lora")
    assert clone_subset["custom_attributes"]["source_dir"].endswith(
        "dataset-01/trigger-clone-source"
    )
    assert clone_dataset["caption_source_mode"] == "captions_json"
    assert clone_dataset["prefer_json_caption"] is False
    assert "trigger_clone" not in original_subset["custom_attributes"]
    assert runtime["dataset_dirs"][1]["source_dir"].endswith(
        "dataset-01/trigger-clone-source"
    )
    assert runtime["dataset_dirs"][2]["source_dir"] == "image_dataset/second"

    runtime_cfg = toml.loads(
        (run_dir / "config.runtime.toml").read_text(encoding="utf-8")
    )
    assert runtime_cfg["stage_schedule_target_groups"] == [[0, 1], [2]]
    runtime_cfg["max_train_steps"] = 100
    args = SimpleNamespace(**runtime_cfg)
    assert active_subset_indices_for_step(args, 0) == {0, 1}
    assert active_subset_indices_for_step(args, 50) == {2}


def test_clone_runtime_dataset_rows_preserves_trigger_clone_materialized_dirs(
    tmp_path,
):
    old_group = tmp_path / "old" / "dataset_cache" / "dataset-01"
    source = old_group / "trigger-clone-source"
    resized = old_group / "trigger-clone-resized"
    lora = old_group / "trigger-clone-lora"
    for path in (source, resized, lora):
        path.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source / "sample.png")
    (source / "captions.json").write_text(
        json.dumps({"sample.png": ["my_character"]}), encoding="utf-8"
    )
    (resized / "sample.png").write_bytes(b"resized")
    (lora / "sample_anima_te.safetensors").write_bytes(b"cache")

    cloned = training_service._clone_runtime_dataset_rows(
        [
            {
                "source_dir": source.as_posix(),
                "image_dir": resized.as_posix(),
                "cache_dir": lora.as_posix(),
                "num_repeats": 3,
                "settings": {"caption_source_mode": "captions_json"},
            }
        ],
        tmp_path / "new" / "dataset_cache",
        copy_existing=True,
    )

    new_group = tmp_path / "new" / "dataset_cache" / "dataset-01"
    assert (new_group / "trigger-clone-source" / "captions.json").is_file()
    assert (new_group / "trigger-clone-resized" / "sample.png").is_file()
    assert (
        new_group / "trigger-clone-lora" / "sample_anima_te.safetensors"
    ).is_file()
    assert cloned[0]["source_dir"].endswith("dataset-01/trigger-clone-source")
    assert cloned[0]["image_dir"].endswith("dataset-01/trigger-clone-resized")
    assert cloned[0]["cache_dir"].endswith("dataset-01/trigger-clone-lora")
