"""Frozen edit-preview metadata takes precedence over mutable prompt files."""

import json


def merge_edit_preview_metadata(base: dict, png_metadata: dict) -> dict:
    try:
        record = json.loads(png_metadata.get("qwen_preview", "null"))
    except (ValueError, TypeError):
        return base
    if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
        return base
    parameters = {key: record[key] for key in (
        "width", "height", "sample_steps", "guidance_scale", "seed", "sample_sampler",
    ) if key in record}
    return {**base, "prompt": record["prompt"], "negative_prompt": record.get("negative_prompt", ""),
            "raw_prompt": json.dumps(record, ensure_ascii=False), "step": record.get("step", base.get("step")),
            "seed": record.get("seed"), "sampler": "euler", "parameters": parameters,
            "sample_task": record.get("sample_task", "edit" if record.get("reference_images") or record.get("reference_image") else "t2i"),
            "reference_files": record.get("reference_files", [record["reference_file"]] if record.get("reference_file") else []),
            "reference_file": record.get("reference_file", ""), "result_file": record.get("result_file", ""),
            "source": {"from_png": True}}
