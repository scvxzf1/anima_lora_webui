from __future__ import annotations

from pathlib import Path
import os

from PIL import Image
import toml

from web.routes import preview as preview_routes
from web.services import preview_service, settings_service


def test_global_settings_default_save_and_resolve(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    _write_base_model_defaults(settings_file.parent)
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)

    defaults = settings_service.get_global_settings()
    assert defaults["output_root"] == "output/runs"
    assert defaults["pretrained_model_name_or_path"] == "models/base.safetensors"
    assert defaults["qwen3"] == "models/qwen.safetensors"
    assert defaults["vae"] == "models/vae.safetensors"
    assert defaults["defaults"]["pretrained_model_name_or_path"] == "models/base.safetensors"

    saved = settings_service.save_global_settings({
        "output_root": "custom/runs",
        "pretrained_model_name_or_path": "${ANIMA_DIT_MODEL}",
        "qwen3": "/abs/qwen.safetensors",
        "vae": "models/custom_vae.safetensors",
    })
    assert saved["output_root"] == "custom/runs"
    assert saved["pretrained_model_name_or_path"] == "${ANIMA_DIT_MODEL}"
    assert saved["qwen3"] == "/abs/qwen.safetensors"
    assert saved["vae"] == "models/custom_vae.safetensors"
    assert settings_service.resolve_output_root() == (tmp_path / "custom/runs").resolve()

    blank_saved = settings_service.save_global_settings({
        "pretrained_model_name_or_path": "",
    })
    assert blank_saved["pretrained_model_name_or_path"] == "${ANIMA_DIT_MODEL}"

    absolute_root = tmp_path / "absolute-runs"
    saved_abs = settings_service.save_global_settings({"output_root": str(absolute_root)})
    assert saved_abs["output_root"] == absolute_root.resolve().as_posix()
    assert saved_abs["pretrained_model_name_or_path"] == "${ANIMA_DIT_MODEL}"
    assert settings_service.resolve_output_root() == absolute_root.resolve()

    data = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert data["global"]["output_root"] == absolute_root.resolve().as_posix()
    assert data["global"]["pretrained_model_name_or_path"] == "${ANIMA_DIT_MODEL}"
    assert data["global"]["qwen3"] == "/abs/qwen.safetensors"
    assert data["global"]["vae"] == "models/custom_vae.safetensors"


def _write_base_model_defaults(configs: Path) -> None:
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "base.toml").write_text(
        "\n".join(
            [
                'pretrained_model_name_or_path = "models/base.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )


def _patch_preview_settings_file(monkeypatch, settings_file: Path, *, root: Path | None = None) -> None:
    monkeypatch.setattr(preview_service, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    if root is not None:
        monkeypatch.setattr(preview_service, "ROOT", root)
        monkeypatch.setattr(settings_service, "ROOT", root)


def test_preview_settings_preserve_global_section(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "custom/runs"\n', encoding="utf-8")
    _patch_preview_settings_file(monkeypatch, settings_file)

    preview_service.save_preview_settings(
        {
            "training_dir": "output/ckpt/sample",
            "inference_dir": "output/tests",
            "custom_dir": "",
        }
    )

    data = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert data["global"]["output_root"] == "custom/runs"


def test_preview_settings_allow_absolute_inference_and_custom_dirs(tmp_path, monkeypatch):
    settings_file = tmp_path / "web-ui-settings.toml"
    inference_dir = tmp_path / "inference"
    custom_dir = tmp_path / "custom"

    _patch_preview_settings_file(monkeypatch, settings_file)

    payload = preview_service.save_preview_settings(
        {
            "training_dir": "output/ckpt/sample",
            "inference_dir": str(inference_dir),
            "custom_dir": str(custom_dir),
        }
    )

    assert payload["inference_dir"] == inference_dir.resolve().as_posix()
    assert payload["custom_dir"] == custom_dir.resolve().as_posix()
    assert preview_service.get_preview_settings()["inference_dir"] == inference_dir.resolve().as_posix()
    assert preview_service.get_preview_settings()["custom_dir"] == custom_dir.resolve().as_posix()


def test_training_preview_defaults_to_latest_runtime_run(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")

    _patch_preview_settings_file(monkeypatch, settings_file, root=tmp_path)

    old_sample = tmp_path / "output" / "runs" / "522-20260523-114514" / "training_output" / "sample"
    new_sample = tmp_path / "output" / "runs" / "522-20260523-114515" / "training_output" / "sample"
    old_sample.mkdir(parents=True)
    new_sample.mkdir(parents=True)

    old_image = old_sample / "old_e000001_00_20260523114514_1.png"
    new_image = new_sample / "new_e000001_00_20260523114515_2.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(old_image)
    Image.new("RGB", (8, 8), color=(56, 34, 12)).save(new_image)

    for ts, path in ((100.0, old_sample.parent.parent.parent), (100.0, old_sample.parent.parent), (100.0, old_sample), (100.0, old_image), (200.0, new_sample.parent.parent.parent), (200.0, new_sample.parent.parent), (200.0, new_sample), (200.0, new_image)):
        os.utime(path, (ts, ts))

    payload = preview_service.list_preview_images("training")

    assert payload["directory"] == "output/runs/522-20260523-114515/training_output/sample"
    assert payload["preview_settings"]["effective_training_source"] == "latest_run"
    assert payload["preview_settings"]["latest_run_dir"] == "output/runs/522-20260523-114515"
    assert payload["images"][0]["name"] == "new_e000001_00_20260523114515_2.png"


def test_selected_history_task_without_sample_dir_does_not_fallback_to_latest_run(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    _patch_preview_settings_file(monkeypatch, settings_file, root=tmp_path)
    latest_sample = tmp_path / "output" / "runs" / "522-20260523-114515" / "training_output" / "sample"
    latest_sample.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(56, 34, 12)).save(latest_sample / "latest_e000001_00_20260523114515_2.png")

    payload = preview_service.list_preview_images(
        "training",
        current_task_sample_dir="",
        task={"id": "task-old", "job": "training"},
        task_id="task-old",
        task_label="历史任务",
        allow_latest_fallback=False,
    )

    assert payload["count"] == 0
    assert payload["directory"] == ""
    assert payload["message"] == "这个历史训练任务没有记录样张目录"
    assert payload["preview_settings"]["effective_training_source"] == "selected_task_missing"


def test_preview_image_absolute_file_must_be_under_saved_preview_dir(tmp_path, monkeypatch):
    settings_file = tmp_path / "web-ui-settings.toml"
    custom_dir = tmp_path / "custom"
    other_dir = tmp_path / "other"
    custom_dir.mkdir()
    other_dir.mkdir()
    allowed_image = custom_dir / "allowed.png"
    blocked_image = other_dir / "blocked.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(allowed_image)
    Image.new("RGB", (8, 8), color=(56, 34, 12)).save(blocked_image)

    _patch_preview_settings_file(monkeypatch, settings_file)
    preview_service.save_preview_settings(
        {
            "training_dir": "output/ckpt/sample",
            "inference_dir": "output/tests",
            "custom_dir": str(custom_dir),
        }
    )

    assert preview_service.resolve_preview_image(str(allowed_image)) == allowed_image.resolve()
    try:
        preview_service.resolve_preview_image(str(blocked_image))
    except ValueError as exc:
        assert "已保存的预览目录" in str(exc)
    else:
        raise AssertionError("项目外且不在已保存预览目录内的图片不应允许读取")


def test_preview_image_absolute_file_allowed_under_global_output_root(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    abs_root = tmp_path / "absolute-runs"
    sample_dir = abs_root / "522-20260523-114514" / "training_output" / "sample"
    sample_dir.mkdir(parents=True)
    image_path = sample_dir / "allowed.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(image_path)
    settings_file.write_text(f'[global]\noutput_root = "{abs_root.as_posix()}"\n', encoding="utf-8")

    _patch_preview_settings_file(monkeypatch, settings_file, root=tmp_path)

    assert preview_service.resolve_preview_image(str(image_path)) == image_path.resolve()


def test_training_preview_images_include_sample_details(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    image_path = sample_dir / "rokkotsu_goddess_5_14_e000004_01_20260516114757_1234.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(image_path)

    prompt_file = tmp_path / "sample_prompts.txt"
    prompt_file.write_text(
        "\n".join(
            [
                "first prompt --w 512 --h 512 --s 16 --g 3.0 --d 42",
                "second prompt --w 1024 --h 1024 --s 28 --g 4.0 --d 1234",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(preview_service, "_resolve_display_path", lambda _value: prompt_file)
    monkeypatch.setattr(preview_service, "_training_step_index", lambda _task: {4: 1120})

    payload = preview_service.list_preview_images(
        "training",
        current_task_sample_dir=str(sample_dir),
        sample_config={
            "sample_prompts": "configs/sample_prompts.txt",
            "sample_sampler": "euler",
        },
        task={"output_dir": str(tmp_path), "variant": "rokkotsu_goddess_5_14"},
        task_id="task-1",
        task_label="imported / rokkotsu_goddess_5_14",
    )

    assert payload["label"] == "训练过程中采样结果 · imported / rokkotsu_goddess_5_14"
    assert payload["count"] == 1

    sample = payload["images"][0]["sample"]
    assert sample["epoch"] == 4
    assert sample["step"] == 1120
    assert sample["prompt_index"] == 1
    assert sample["seed"] == 1234
    assert sample["sampler"] == "euler"
    assert sample["prompt"] == "second prompt"
    assert sample["parameters"] == {
        "width": 1024,
        "height": 1024,
        "sample_steps": 28,
        "guidance_scale": 4.0,
        "seed": 1234,
        "sample_sampler": "euler",
    }


def test_training_weights_default_to_latest_runtime_run(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")

    _patch_preview_settings_file(monkeypatch, settings_file, root=tmp_path)

    sample_dir = tmp_path / "output" / "runs" / "522-20260523-114515" / "training_output" / "sample"
    sample_dir.mkdir(parents=True)
    weight_dir = sample_dir.parent
    weight_path = weight_dir / "demo.safetensors"
    weight_path.write_bytes(b"stub")

    payload = preview_service.list_training_weights()

    assert payload["directory"] == "output/runs/522-20260523-114515/training_output"
    assert payload["count"] == 1
    assert payload["weights"][0]["name"] == "demo.safetensors"
    assert payload["weights"][0]["download_url"].startswith("/api/preview/weight?file=")


def test_training_weight_download_resolves_allowed_paths(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    abs_root = tmp_path / "absolute-runs"
    weight_dir = abs_root / "522-20260523-114514" / "training_output"
    weight_dir.mkdir(parents=True)
    weight = weight_dir / "demo.safetensors"
    weight.write_bytes(b"stub")
    settings_file.write_text(f'[global]\noutput_root = "{abs_root.as_posix()}"\n', encoding="utf-8")
    _patch_preview_settings_file(monkeypatch, settings_file, root=tmp_path)

    assert preview_service.resolve_training_weight(str(weight)) == weight.resolve()
    assert preview_service.resolve_training_weight(weight.relative_to(tmp_path).as_posix()) == weight.resolve()


def test_training_weight_download_resolves_selected_task_output(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    output_dir = tmp_path / "legacy-output"
    output_dir.mkdir()
    weight = output_dir / "demo.safetensors"
    weight.write_bytes(b"stub")
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    _patch_preview_settings_file(monkeypatch, settings_file, root=tmp_path)

    task = {"id": "task-a", "job": "training", "variant": "demo", "output_dir": str(output_dir)}
    listing = preview_service.list_training_weights(task, allow_latest_fallback=False)

    assert listing["weights"][0]["download_url"].endswith("&task_id=task-a")
    assert preview_service.resolve_training_weight(str(weight), task=task) == weight.resolve()


def test_training_weight_download_rejects_non_weight_and_outside_paths(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    output_root = tmp_path / "output" / "runs"
    allowed_dir = output_root / "522-20260523-114514" / "training_output"
    allowed_dir.mkdir(parents=True)
    bad_ext = allowed_dir / "demo.txt"
    bad_ext.write_text("not a weight", encoding="utf-8")
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"stub")
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    _patch_preview_settings_file(monkeypatch, settings_file, root=tmp_path)

    try:
        preview_service.resolve_training_weight(str(bad_ext))
    except ValueError as exc:
        assert "权重" in str(exc)
    else:
        raise AssertionError("非 safetensors 文件不应允许下载")

    try:
        preview_service.resolve_training_weight(str(outside))
    except ValueError as exc:
        assert "训练输出目录" in str(exc)
    else:
        raise AssertionError("全局输出目录外的权重不应允许下载")


def test_selected_history_task_without_output_dir_does_not_fallback_to_latest_run(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    _patch_preview_settings_file(monkeypatch, settings_file, root=tmp_path)
    sample_dir = tmp_path / "output" / "runs" / "522-20260523-114515" / "training_output" / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir.parent / "demo.safetensors").write_bytes(b"stub")

    payload = preview_service.list_training_weights(
        {"id": "task-old", "job": "training"},
        allow_latest_fallback=False,
    )

    assert payload["count"] == 0
    assert payload["directory"] == ""
    assert payload["message"] == "这个历史训练任务没有记录输出目录"


def test_step_sample_filename_uses_step_not_epoch():
    path = preview_service.ROOT / "output/ckpt/sample/model_000120_00_20260516114757_7.png"

    parsed = preview_service._parse_sample_image_name(path)

    assert parsed is not None
    assert parsed["epoch"] is None
    assert parsed["step"] == 120
    assert parsed["prompt_index"] == 0
    assert parsed["seed"] == 7


def test_config_group_preview_images_merge_training_tasks(tmp_path, monkeypatch):
    sample_a = tmp_path / "task-a" / "sample"
    sample_b = tmp_path / "task-b" / "sample"
    sample_other = tmp_path / "task-other" / "sample"
    sample_a.mkdir(parents=True)
    sample_b.mkdir(parents=True)
    sample_other.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(
        sample_a / "demo_e000001_00_20260517100000_1.png"
    )
    Image.new("RGB", (8, 8), color=(56, 34, 12)).save(
        sample_b / "demo_e000002_00_20260517110000_2.png"
    )
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(
        sample_other / "demo_e000003_00_20260517120000_3.png"
    )

    monkeypatch.setattr(preview_service, "_training_step_index", lambda _task: {})
    tasks = [
        {
            "id": "task-a",
            "job": "training",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "sample_dir": str(sample_a),
            "started_at_text": "2026-05-17 10:00:00",
        },
        {
            "id": "task-b",
            "job": "training",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "sample_dir": str(sample_b),
            "started_at_text": "2026-05-17 11:00:00",
        },
        {
            "id": "task-b-copy",
            "job": "training",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "sample_dir": str(sample_b),
            "started_at_text": "2026-05-17 11:30:00",
        },
    ]

    payload = preview_service.list_config_group_preview_images(
        tasks,
        methods_subdir="imported",
        variant="demo",
        preset="default",
    )

    assert payload["mode"] == "config_group"
    assert payload["count"] == 2
    assert payload["task_count"] == 3
    assert {item["source_task"]["id"] for item in payload["images"]} == {"task-a", "task-b"}
    assert all("task_id=" in item["url"] for item in payload["images"])


def test_config_group_training_weights_merge_and_dedupe(tmp_path, monkeypatch):
    output_a = tmp_path / "task-a" / "output"
    output_b = tmp_path / "task-b" / "output"
    output_a.mkdir(parents=True)
    output_b.mkdir(parents=True)
    weight_a = output_a / "demo-000001.safetensors"
    weight_b = output_b / "demo-000002.safetensors"
    weight_a.write_bytes(b"stub")
    weight_b.write_bytes(b"stub")

    def fake_metadata(path):
        if path.name.endswith("000001.safetensors"):
            return {
                "ss_epoch": "1",
                "ss_steps": "100",
                "ss_num_epochs": "4",
                "ss_max_train_steps": "400",
                "ss_output_name": "demo",
                "ss_training_started_at": "100",
            }
        return {
            "ss_epoch": "2",
            "ss_steps": "200",
            "ss_num_epochs": "4",
            "ss_max_train_steps": "400",
            "ss_output_name": "demo",
            "ss_training_started_at": "300",
        }

    monkeypatch.setattr(preview_service, "_read_safetensors_metadata", fake_metadata)
    tasks = [
        {
            "id": "task-a",
            "job": "training",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "output_dir": str(output_a),
            "started_at": 100,
            "finished_at": 200,
            "started_at_text": "2026-05-17 10:00:00",
        },
        {
            "id": "task-b",
            "job": "training",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "output_dir": str(output_b),
            "started_at": 300,
            "finished_at": 400,
            "started_at_text": "2026-05-17 11:00:00",
        },
        {
            "id": "task-b-copy",
            "job": "training",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "output_dir": str(output_b),
            "started_at": 500,
            "finished_at": 600,
            "started_at_text": "2026-05-17 12:00:00",
        },
    ]

    payload = preview_service.list_config_group_training_weights(
        tasks,
        methods_subdir="imported",
        variant="demo",
        preset="default",
    )

    assert payload["mode"] == "config_group"
    assert payload["count"] == 2
    assert payload["group_task_count"] == 3
    assert payload["task_count"] == 2
    assert {item["source_task"]["id"] for item in payload["weights"]} == {"task-a", "task-b"}
    assert all("source_task" in item for item in payload["weights"])


def test_preview_route_config_group_prefers_history_group_key():
    tasks = [
        {
            "id": "task-a",
            "job": "training",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "history_group_key": "source:configs/imported/a.toml",
        },
        {
            "id": "task-b",
            "job": "training",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "history_group_key": "source:configs/imported/b.toml",
        },
    ]
    request = _PreviewRequest(
        {
            "mode": "config_group",
            "methods_subdir": "imported",
            "variant": "demo",
            "preset": "default",
            "group_key": "source:configs/imported/a.toml",
        },
        _PreviewHistoryService(tasks),
    )

    selected = preview_routes._selected_config_group_tasks(request)

    assert [task["id"] for task in selected] == ["task-a"]


class _PreviewRequest:
    def __init__(self, query: dict[str, str], service: object) -> None:
        self.query = query
        self.app = {"training_service": service}


class _PreviewHistoryService:
    def __init__(self, tasks: list[dict]) -> None:
        self._tasks = tasks

    def list_history_tasks(self) -> list[dict]:
        return list(self._tasks)

    def get_history_task(self, task_id: str) -> dict:
        for task in self._tasks:
            if task.get("id") == task_id:
                return {"ok": True, "task": task}
        raise FileNotFoundError(task_id)
