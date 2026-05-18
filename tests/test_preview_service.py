from __future__ import annotations

from PIL import Image

from web.services import preview_service


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
