import json

from PIL import Image
import pytest

from library.training import preview_spec
from library.models.family_registry import model_family_capability_catalog


def test_reference_reader_accepts_jpeg_and_rejects_animation(tmp_path):
    source = tmp_path / "reference.jpg"
    Image.new("RGB", (64, 64), "red").save(source)
    image = preview_spec.read_reference(source)
    assert image.mode == "RGB" and image.size == (64, 64)
    animation = tmp_path / "animated.gif"
    Image.new("RGB", (64, 64), "red").save(
        animation, save_all=True, append_images=[Image.new("RGB", (64, 64), "blue")]
    )
    with pytest.raises(ValueError, match="单帧"):
        preview_spec.read_reference(animation)


def test_txt_keeps_braces_and_structured_paths(tmp_path):
    from library.train_util import load_prompts

    reference = tmp_path / "参考 图 -- 原图.png"
    Image.new("RGB", (64, 64)).save(reference)
    record = {"prompt": "edit -- preserve", "reference_image": str(reference),
              "sample_task": "edit", "width": 64, "height": 64}
    path = tmp_path / "prompts.txt"
    path.write_text("{soft focus} portrait --w 512\n" + json.dumps(record, ensure_ascii=False), encoding="utf-8")
    prompts = load_prompts(str(path))
    assert prompts[0]["prompt"] == "{soft focus} portrait"
    assert prompts[0]["width"] == 512
    assert prompts[1] == {**record, "enum": 1}
    preview_spec.validate_preview_prompt(prompts[1], "qwen_image_2_1")


@pytest.mark.parametrize("family", ["anima", "krea2_raw", "z_image"])
def test_other_families_reject_edit(family):
    with pytest.raises(ValueError, match="不支持"):
        preview_spec.validate_preview_prompt({"sample_task": "edit", "reference_image": "/missing.png"}, family)


@pytest.mark.parametrize("override", [
    {"width": 63}, {"height": 65}, {"width": 2080}, {"width": 64.5},
    {"width": True}, {"height": "bad"}, {"sample_sampler": "heun"},
    {"flow_shift": 1.0}, {"sample_steps": 0}, {"sample_steps": 1001},
    {"guidance_scale": 0.5}, {"guidance_scale": float("nan")},
    {"guidance_scale": float("inf")},
])
def test_qwen_rejects_invalid_preview_parameters(override):
    with pytest.raises(ValueError):
        preview_spec.validate_preview_prompt({"prompt": "test", **override}, "qwen_image_2_1")


@pytest.mark.parametrize("steps", [True, 1.5])
def test_qwen_steps_must_be_integer(steps):
    with pytest.raises(ValueError):
        preview_spec.validate_preview_prompt({"prompt": "test", "sample_steps": steps}, "qwen_image_2_1")


def test_structured_prompt_rejects_bad_json_without_breaking_legacy():
    assert preview_spec.parse_structured_prompt("{soft focus} portrait") is None
    with pytest.raises(ValueError):
        preview_spec.parse_structured_prompt('{"prompt":')
    with pytest.raises(ValueError):
        preview_spec.parse_structured_prompt('{"prompt": 12}')


def test_reference_path_rejects_traversal_and_missing(tmp_path):
    for path in ("", str(tmp_path / ".." / "outside.png"), str(tmp_path / "missing.png")):
        with pytest.raises(ValueError):
            preview_spec.reference_path(path)


def test_multi_reference_schema_and_legacy_compatibility(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"reference-{index}.png"
        Image.new("RGB", (64, 64)).save(path)
        paths.append(str(path))
    prompt = {"prompt": "edit", "sample_task": "edit", "reference_images": paths,
              "width": 64, "height": 64}
    assert preview_spec.preview_references(prompt) == paths
    preview_spec.validate_preview_prompt(prompt, "qwen_image_2_1")
    assert preview_spec.preview_references({"reference_image": paths[0]}) == paths[:1]
    preview_spec.validate_preview_prompt({"prompt": "edit", "reference_image": paths[0]}, "qwen21")
    assert preview_spec.preview_task({"reference_images": paths}) == "edit"
    catalog = {item["name"]: item for item in model_family_capability_catalog()}
    assert catalog["qwen_image_2_1"]["max_preview_references"] == 4
    assert all(item["max_preview_references"] == 0 for name, item in catalog.items() if name != "qwen_image_2_1")


@pytest.mark.parametrize("references", [[], "image.png", [1], [""], ["image.png", None]])
def test_multi_reference_schema_rejects_invalid_lists(references):
    with pytest.raises(ValueError, match="reference_images"):
        preview_spec.validate_preview_prompt({"prompt": "edit", "reference_images": references}, "qwen_image_2_1")


def test_multi_reference_conflict_t2i_and_limits(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGB", (64, 64)).save(path)
    reference = str(path)
    with pytest.raises(ValueError, match="不能同时使用"):
        preview_spec.validate_preview_prompt({"sample_task": "edit", "reference_image": reference,
                                              "reference_images": [reference]}, "qwen_image_2_1")
    with pytest.raises(ValueError, match="文生图"):
        preview_spec.validate_preview_prompt({"sample_task": "t2i", "reference_images": [reference]}, "qwen_image_2_1")
    with pytest.raises(ValueError, match="最多支持 4"):
        preview_spec.validate_preview_prompt({"reference_images": [reference] * 5}, "qwen_image_2_1")
    with pytest.raises(ValueError, match="总像素"):
        preview_spec.validate_preview_prompt({"reference_images": [reference] * 2,
                                              "width": 2048, "height": 2048}, "qwen_image_2_1")


@pytest.mark.parametrize("field,count,target", [
    ("reference_image", 1, 2048),
    ("reference_images", 4, 1024),
])
def test_preview_file_reference_budget_boundary_and_single_read(tmp_path, monkeypatch, field, count, target):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (64, 64)).save(reference)
    value = str(reference) if field == "reference_image" else [str(reference)] * count
    prompt = {"prompt": "edit", "sample_task": "edit", field: value,
              "width": target, "height": target}
    prompts_path = tmp_path / "prompts.json"
    config = {"sample_at_first": True, "sample_prompts": str(prompts_path),
              "model_family": "qwen_image_2_1"}
    original_read = preview_spec.read_reference
    reads = []

    def tracked_read(source):
        reads.append(source)
        return original_read(source)

    monkeypatch.setattr(preview_spec, "read_reference", tracked_read)
    prompts_path.write_text(json.dumps([prompt] * 4), encoding="utf-8")
    preview_spec.validate_preview_file(config)
    assert len(reads) == 4 * count

    reads.clear()
    prompts_path.write_text(json.dumps([prompt] * 5), encoding="utf-8")
    with pytest.raises(ValueError, match="样张 5: .*16 Mi"):
        preview_spec.validate_preview_file(config)
    assert len(reads) == 5 * count
