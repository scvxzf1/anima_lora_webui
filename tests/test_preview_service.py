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
