"""Live training previews preserve B unless a sample explicitly requests C."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from library.models.qwen_image_2_1 import preview_conditions, training_preview


@pytest.mark.parametrize("choice", [None, False, True])
def test_training_preview_cache_opt_in(tmp_path, monkeypatch, choice):
    prompt = {"prompt": "test", "enum": 0}
    if choice is not None:
        prompt["use_kv_cache"] = choice
    model = torch.nn.Linear(1, 1)
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    args = SimpleNamespace(output_dir=str(tmp_path), seed=1, output_name="test")
    seen = []

    def generate(*values, **kwargs):
        seen.append(kwargs["use_kv_cache"])
        return torch.zeros(1, 64, 4, 4)

    monkeypatch.setattr(training_preview, "prepare_dtype", lambda args: (torch.float32, None))
    monkeypatch.setattr(training_preview, "generate_preview", generate)
    training_preview._sample_local(accelerator, args, model, None, [prompt],
                                   {preview_conditions.condition_key(prompt): ()}, None, 1)
    assert seen == [choice is True]
    assert model.training


@pytest.mark.parametrize("value", ["false", 1, None])
def test_snapshot_rejects_non_boolean_cache_option_before_model_load(tmp_path, value):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([{"prompt": "test", "use_kv_cache": value}]))
    args = SimpleNamespace(sample_prompts=str(path), output_dir=str(tmp_path), sample_sampler="euler")
    with pytest.raises(ValueError, match="use_kv_cache must be a boolean"):
        preview_conditions.snapshot_prompts(args)
