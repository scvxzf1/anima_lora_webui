"""Web runtime config core path / precision / sample tests"""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_web_runtime_config_creates_run_directory_and_overrides_paths(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 11, 45, 14)

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

    run_dir = tmp_path / "output" / "runs" / "522-20260523-114514"
    assert runtime["run_dir"] == "output/runs/522-20260523-114514"
    assert (run_dir / "config.original.toml").read_text(encoding="utf-8") == (
        tmp_path / "configs" / "imported" / "522.toml"
    ).read_text(encoding="utf-8")
    assert (run_dir / "model_cache" / "logs").is_dir()
    assert (run_dir / "training_output" / "sample").is_dir()
    assert (run_dir / "dataset_cache" / "dataset-01" / "resized").is_dir()
    assert (run_dir / "dataset_cache" / "dataset-02" / "lora").is_dir()
    run_meta = json.loads((run_dir / "run.meta.json").read_text(encoding="utf-8"))
    assert run_meta["history_source_config_file"] == "configs/imported/522.toml"
    assert run_meta["runtime_config_file"] == "output/runs/522-20260523-114514/config.runtime.toml"

    runtime_cfg = toml.loads((run_dir / "config.runtime.toml").read_text(encoding="utf-8"))
    assert runtime_cfg["output_dir"] == "output/runs/522-20260523-114514/training_output"
    assert runtime_cfg["logging_dir"] == "output/runs/522-20260523-114514/model_cache/logs"
    assert runtime_cfg["dataset_config"] == "output/runs/522-20260523-114514/dataset.runtime.toml"

    assert runtime_cfg["source_image_dir"] == "image_dataset/a"
    assert runtime_cfg["resized_image_dir"] == "output/runs/522-20260523-114514/dataset_cache/dataset-01/resized"
    assert runtime_cfg["lora_cache_dir"] == "output/runs/522-20260523-114514/dataset_cache/dataset-01/lora"

    dataset_cfg = toml.loads((run_dir / "dataset.runtime.toml").read_text(encoding="utf-8"))
    assert dataset_cfg["datasets"][0]["batch_size"] == 2
    assert dataset_cfg["datasets"][1]["batch_size"] == 2
    assert "resolution" not in dataset_cfg["datasets"][0]
    assert "bucket_reso_steps" not in dataset_cfg["datasets"][0]
    first_subset = dataset_cfg["datasets"][0]["subsets"][0]
    second_subset = dataset_cfg["datasets"][1]["subsets"][0]
    assert first_subset["custom_attributes"]["source_dir"] == "image_dataset/a"
    assert first_subset["custom_attributes"]["preprocess"]["resolution"] == 768
    assert first_subset["image_dir"].endswith("dataset-01/resized")
    assert first_subset["cache_dir"].endswith("dataset-01/lora")
    assert second_subset["custom_attributes"]["source_dir"] == "image_dataset/b"
    assert second_subset["custom_attributes"]["preprocess"]["resolution"] == 1024
    assert second_subset["image_dir"].endswith("dataset-02/resized")
    assert second_subset["cache_dir"].endswith("dataset-02/lora")

    env = {}
    training_service._apply_runtime_env(env, runtime)
    assert env["ANIMA_RUNTIME_CONFIG"] == "output/runs/522-20260523-114514/config.runtime.toml"
    assert env["TORCHINDUCTOR_CACHE_DIR"].endswith("model_cache/torchinductor")
    assert env["TRITON_CACHE_DIR"].endswith("model_cache/triton")

def test_runtime_config_main_paths_do_not_depend_on_bind_legacy(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    assert not hasattr(training_runtime_config, "_bind_legacy")

    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    recovered = training_service._runtime_from_config_file(runtime["runtime_config_file"])
    assert recovered is not None
    assert recovered["runtime_config_file"] == runtime["runtime_config_file"]
    assert Path(runtime["runtime_config_file"]).name == "config.runtime.toml"

def test_saved_web_form_values_reach_runtime_config_and_train_loader(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    train_file = "configs/imported/522.toml"
    train_path = tmp_path / train_file
    saved_values = {
        "learning_rate": 0.000321,
        "sample_every_n_steps": 9,
        "save_last_n_epochs": 4,
        "checkpointing_last_n_epochs": 3,
        "block_swap_transfer_dtype": "fp8_e4m3",
        "block_swap_restore_mode": "slab",
        "memory_probe_jsonl": "auto",
        "preprocess_precision_preference": "fp16",
    }

    ok, msg, _content, changed = config_service.patch_raw_file_values(
        train_file,
        saved_values,
        content=train_path.read_text(encoding="utf-8"),
    )
    assert ok, msg
    assert set(saved_values) <= set(changed)

    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file=train_file,
    )
    runtime_path = tmp_path / runtime["runtime_config_file"]
    runtime_cfg = toml.loads(runtime_path.read_text(encoding="utf-8"))
    for key, value in saved_values.items():
        assert runtime_cfg[key] == value
    assert runtime_cfg["output_dir"].endswith("/training_output")

    import train
    from library.config.io import read_config_from_file

    parser = train.setup_parser()
    argv = ["--config_file", str(runtime_path), "--no-config-snapshot"]
    args = read_config_from_file(parser.parse_args(argv), parser, argv=argv)
    assert args.learning_rate == saved_values["learning_rate"]
    assert args.sample_every_n_steps == saved_values["sample_every_n_steps"]
    assert args.save_last_n_epochs == saved_values["save_last_n_epochs"]
    assert args.checkpointing_last_n_epochs == saved_values["checkpointing_last_n_epochs"]
    assert args.block_swap_transfer_dtype == saved_values["block_swap_transfer_dtype"]
    assert args.block_swap_restore_mode == saved_values["block_swap_restore_mode"]
    assert args.memory_probe_jsonl == saved_values["memory_probe_jsonl"]
    assert args.output_dir == runtime_cfg["output_dir"]

def test_runtime_config_defaults_preprocess_precision_from_mixed_precision(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    train_file = "configs/imported/522.toml"
    train_path = tmp_path / train_file
    train_path.write_text(
        train_path.read_text(encoding="utf-8") + '\nmixed_precision = "fp16"\n',
        encoding="utf-8",
    )

    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file=train_file,
    )
    runtime_path = tmp_path / runtime["runtime_config_file"]
    runtime_cfg = toml.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime_cfg["mixed_precision"] == "fp16"
    assert runtime_cfg["preprocess_precision_preference"] == "fp16"

def test_training_sample_config_reports_effective_sampler() -> None:
    sample = training_service._sample_config_from_cfg(
        {
            "sample_prompts": "configs/sample_prompts.txt",
            "sample_every_n_epochs": 1,
            "sample_sampler": "ddim",
        },
        ["--sample_sampler", "dpmsolver++"],
    )

    assert sample["sample_sampler"] == "euler"
    assert sample["sample_sampler_raw"] == "dpmsolver++"
    assert sample["sample_sampler_status"] == "legacy"

    supported = training_service._sample_config_from_cfg(
        {
            "sample_prompts": "configs/sample_prompts.txt",
            "sample_every_n_steps": 50,
            "sample_sampler": "lcm",
        },
        [],
    )
    assert supported["sample_sampler"] == "lcm"
    assert supported["sample_sampler_status"] == "supported"

def test_runtime_config_recovers_source_group_from_run_meta(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 11, 45, 14)

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

    recovered = training_service._runtime_from_config_file(runtime["runtime_config_file"])
    assert recovered is not None
    assert recovered["history_source_config_file"] == "configs/imported/522.toml"

    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())
    svc._start_history_task(
        job="training",
        variant="522",
        preset="default",
        methods_subdir="imported",
        output_dir=recovered["output_dir"],
        sample_dir=recovered["sample_dir"],
        data_dirs=recovered["data_dirs"],
        sample_config={},
        command=["python", "train.py"],
        config_file=recovered["runtime_config_file"],
        runtime_info=recovered,
    )

    task = svc.list_history_tasks(include_archived=True)[0]
    assert task["history_group_key"] == "source:configs/imported/522.toml"
    assert task["history_source_config_file"] == "configs/imported/522.toml"
    assert task["history_run_label"] == "522-20260523-114514"

def test_absolute_output_root_runtime_config_allowed_in_preflight(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path.parent / "absolute-output-root"
    monkeypatch.setattr(settings_service, "resolve_output_root", lambda value=None: output_root.resolve())
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root.resolve())
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: output_root.resolve())
    monkeypatch.setattr(
        training_service,
        "_display_settings_path",
        lambda path: str(Path(path).resolve()),
    )

    for rel in ("image_dataset/a", "image_dataset/b"):
        image_dir = tmp_path / rel
        image_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_dir / "sample.png")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    (tmp_path / "configs" / "imported" / "522.toml").write_text(
        "\n".join(
            [
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
                'dataset_config = "configs/datasets/522.toml"',
            ]
        ),
        encoding="utf-8",
    )

    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    runtime_config = runtime["runtime_config_file"]
    assert Path(runtime_config).is_absolute()
    assert config_service.is_web_runtime_config(runtime_config) is True
    result = config_service.preflight_training_config(
        "522",
        "default",
        "imported",
        config_file=runtime_config,
    )
    assert not any("项目目录内" in item["message"] for item in result["errors"])

def test_status_snapshot_includes_runtime_info():
    svc = TrainingService(web.Application())
    svc.current_runtime_info = {
        "run_dir": "output/runs/522-20260523-114514",
        "runtime_config_file": "output/runs/522-20260523-114514/config.runtime.toml",
        "original_config_file": "output/runs/522-20260523-114514/config.original.toml",
        "dataset_config_file": "output/runs/522-20260523-114514/dataset.runtime.toml",
        "model_cache_dir": "output/runs/522-20260523-114514/model_cache",
        "dataset_cache_dir": "output/runs/522-20260523-114514/dataset_cache",
        "training_output_dir": "output/runs/522-20260523-114514/training_output",
        "logs_dir": "output/runs/522-20260523-114514/model_cache/logs",
    }

    snapshot = svc.get_status_snapshot()

    assert snapshot["run_dir"] == "output/runs/522-20260523-114514"
    assert snapshot["runtime_config_file"].endswith("config.runtime.toml")
    assert snapshot["dataset_cache_dir"].endswith("dataset_cache")

