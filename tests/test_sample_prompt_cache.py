from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from library.training.sample_prompt_cache import (
    load_sample_prompt_cache,
    restore_sample_prompt_cache,
    save_sample_prompt_cache,
)


def _args(tmp_path: Path):
    model = tmp_path / "qwen.safetensors"
    model.write_bytes(b"model-v1")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "test prompt --w 1024 --h 1024 --s 28 --g 4 --d 114\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        model_family="krea2_raw",
        qwen3=str(model),
        sample_prompts=str(prompts),
        mixed_precision="bf16",
    )


def _outputs():
    return {
        "test prompt": (
            torch.arange(24, dtype=torch.bfloat16).reshape(1, 2, 12, 1),
            torch.ones(1, 2, dtype=torch.bool),
        ),
        "": (
            torch.zeros(1, 2, 12, 1, dtype=torch.bfloat16),
            torch.zeros(1, 2, dtype=torch.bool),
        ),
    }


def test_sample_prompt_cache_round_trip_and_restore(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cache_root = tmp_path / "cache"
    prompts = [{"prompt": "test prompt", "negative_prompt": "", "enum": 0}]

    path = save_sample_prompt_cache(args, prompts, _outputs(), cache_root=cache_root)
    cached = load_sample_prompt_cache(args, cache_root=cache_root)

    assert path.is_file()
    assert cached is not None
    snapshot, outputs = cached
    assert snapshot[0]["prompt"] == "test prompt"
    assert torch.equal(outputs["test prompt"][0], _outputs()["test prompt"][0])
    assert outputs[""][1].dtype == torch.bool

    trainer = SimpleNamespace(
        sample_prompts_snapshot=None,
        sample_prompts_te_outputs=None,
    )
    assert restore_sample_prompt_cache(trainer, args, cache_root=cache_root)
    assert trainer.sample_prompts_te_outputs["test prompt"][0].device.type == "cpu"


def test_sample_prompt_cache_invalidates_when_prompts_change(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cache_root = tmp_path / "cache"
    save_sample_prompt_cache(
        args,
        [{"prompt": "test prompt", "negative_prompt": ""}],
        _outputs(),
        cache_root=cache_root,
    )

    Path(args.sample_prompts).write_text("changed prompt\n", encoding="utf-8")

    assert load_sample_prompt_cache(args, cache_root=cache_root) is None


def test_sample_prompt_cache_invalidates_when_model_changes(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cache_root = tmp_path / "cache"
    save_sample_prompt_cache(
        args,
        [{"prompt": "test prompt", "negative_prompt": ""}],
        _outputs(),
        cache_root=cache_root,
    )

    Path(args.qwen3).write_bytes(b"model-v2-with-different-size")

    assert load_sample_prompt_cache(args, cache_root=cache_root) is None


def test_sample_prompt_cache_supports_anima_four_tensor_outputs(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.model_family = "anima"
    args.qwen3_max_token_length = 512
    args.t5_max_token_length = 512
    args.t5_tokenizer_path = str(tmp_path / "t5-tokenizer")
    prompts = [{"prompt": "test prompt", "negative_prompt": ""}]
    outputs = {
        prompt: tuple(
            torch.full((1, 2), index, dtype=torch.int64) for index in range(4)
        )
        for prompt in ("test prompt", "")
    }

    save_sample_prompt_cache(args, prompts, outputs, cache_root=tmp_path / "cache")
    cached = load_sample_prompt_cache(args, cache_root=tmp_path / "cache")

    assert cached is not None
    assert len(cached[1]["test prompt"]) == 4
    assert torch.equal(cached[1]["test prompt"][3], outputs["test prompt"][3])


def test_sample_prompt_cache_ignores_corrupt_file(tmp_path: Path) -> None:
    args = _args(tmp_path)
    cache_root = tmp_path / "cache"
    path = save_sample_prompt_cache(
        args,
        [{"prompt": "test prompt", "negative_prompt": ""}],
        _outputs(),
        cache_root=cache_root,
    )
    path.write_bytes(b"not-a-safetensors-file")

    assert load_sample_prompt_cache(args, cache_root=cache_root) is None
