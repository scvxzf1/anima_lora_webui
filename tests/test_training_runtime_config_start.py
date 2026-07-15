"""Web runtime start/preflight wiring tests"""

from __future__ import annotations

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

