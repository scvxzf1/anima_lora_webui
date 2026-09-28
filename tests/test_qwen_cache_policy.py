"""Execution policy must survive configuration and stay TE-only."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from library.models.qwen_image_2_1.cache_policy import resolve_cache_policy
from library.models.qwen_image_2_1 import text_encoder_runtime as runtime
from scripts.tasks import preprocess
from scripts.tasks.qwen_edit_preprocess import run_edit_preprocess


@pytest.mark.parametrize("policy,expected", [
    ("auto", ("auto", "auto")), ("cpu_offload", ("cuda", "on")),
    ("gpu", ("cuda", "off")), ("cpu", ("cpu", "off")),
])
def test_policy_routes_both_cache_commands(tmp_path, monkeypatch, policy, expected):
    assert resolve_cache_policy(policy) == expected
    config = tmp_path / "dataset.toml"
    config.write_text("")
    overrides = {"dataset_config": str(config), "qwen_text_encoder_cache_policy": policy}
    calls = []
    run_edit_preprocess(overrides, run=calls.append, python="python", path=lambda k, d: d,
                        dtype="bfloat16", extra=[])
    monkeypatch.setattr(preprocess, "_path_overrides_value", lambda: overrides)
    monkeypatch.setattr(preprocess, "run", calls.append)
    monkeypatch.setattr(preprocess, "_path", lambda k, d: d)
    monkeypatch.setattr(preprocess, "_preprocess_precision_dtype", lambda: "bfloat16")
    preprocess._run_preprocess_te_qwen_image_2_1({"source_image_dir": "source", "lora_cache_dir": "cache"}, [])
    assert len(calls) == 2
    for command in calls:
        assert command[command.index("--cache_policy") + 1] == policy


def test_invalid_policy_and_conflicting_overrides_rejected():
    with pytest.raises(ValueError, match="Invalid"):
        resolve_cache_policy("guess")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_cache_policy("gpu", offload="on")
    assert resolve_cache_policy("auto", device="cpu", offload="off") == ("cpu", "off")
    assert resolve_cache_policy("cpu_offload", device="cuda:1") == ("cuda:1", "on")
    assert resolve_cache_policy("gpu", device="cuda:2") == ("cuda:2", "off")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_cache_policy("cpu", device="cuda:1")


def test_auto_threshold_and_explicit_modes(monkeypatch):
    model = torch.nn.Linear(4, 4, bias=False)  # 64 bytes
    device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (100, 100))
    assert runtime._should_offload(model, device, "auto") is True
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (200, 200))
    assert runtime._should_offload(model, device, "auto") is False
    assert runtime._should_offload(model, device, "on") is True
    assert runtime._should_offload(model, device, "off") is False


def test_cpu_policy_does_not_move_edit_vae_to_cpu(monkeypatch):
    from scripts.qwen_image_2_1 import preprocess_edit_cache as edit
    from library.models.qwen_image_2_1 import weights

    for strategy_class in (edit.LatentsCachingStrategy, edit.TextEncoderOutputsCachingStrategy,
                           edit.TokenizeStrategy, edit.TextEncodingStrategy):
        monkeypatch.setattr(strategy_class, "_strategy", None)

    events = []

    class Group:
        image_data = {"one": object()}
        text_done = False
        latent_done = False

        def is_text_encoder_outputs_cache_complete(self):
            return self.text_done

        def is_latents_cache_complete(self):
            return self.latent_done

        def new_cache_text_encoder_outputs(self, models, accelerator):
            assert accelerator.device.type == "cpu"
            self.text_done = True
            events.append("text")

        def new_cache_latents(self, vae, accelerator):
            assert accelerator.device.type == "cuda"
            self.latent_done = True
            events.append("latents")

    @contextmanager
    def encoder(*args, **kwargs):
        assert kwargs["cache_policy"] == "cpu"
        yield SimpleNamespace(_qwen_execution_device=torch.device("cpu"))
        events.append("text_freed")

    class Vae:
        def to(self, *args, **kwargs):
            return self

        def requires_grad_(self, value):
            return self

        def eval(self):
            return self

    def load_vae(*args, **kwargs):
        assert events == ["text", "text_freed"]
        return Vae()

    monkeypatch.setattr(edit, "build_edit_datasets", lambda _: [Group()])
    monkeypatch.setattr(edit, "QwenImage21EditTokenizeStrategy", lambda _: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runtime, "text_encoder_for_cache", encoder)
    monkeypatch.setattr(weights, "load_qwen_image_2_1_vae", load_vae)
    edit.cache_edit("unused", vae_path="unused", qwen3_path="unused", cache_policy="cpu")
    assert events == ["text", "text_freed", "latents"]


def test_cli_schema_recognizes_policy():
    from library.training.cli_args import add_sd_models_arguments
    import argparse

    parser = argparse.ArgumentParser()
    add_sd_models_arguments(parser)
    assert parser.parse_args([]).qwen_text_encoder_cache_policy == "auto"
    assert parser.parse_args(["--qwen_text_encoder_cache_policy", "cpu_offload"]).qwen_text_encoder_cache_policy == "cpu_offload"
    with pytest.raises(SystemExit):
        parser.parse_args(["--qwen_text_encoder_cache_policy", "invalid"])


def test_policy_conflict_is_rejected_before_loading_weights(monkeypatch):
    from library.models.qwen_image_2_1 import weights

    def forbidden(*args, **kwargs):
        raise AssertionError("Weights must not load for conflicting policy")

    monkeypatch.setattr(weights, "load_qwen_image_2_1_text_encoder", forbidden)
    with pytest.raises(ValueError, match="conflicts"):
        with runtime.text_encoder_for_cache("unused", dtype=torch.bfloat16,
                                            cache_policy="gpu", offload="on"):
            pass


def test_t2i_all_cache_hits_skip_encoder_load(tmp_path, monkeypatch, capsys):
    from scripts.qwen_image_2_1 import preprocess_te_cache as script
    import sys

    source = SimpleNamespace(captions=None, render=lambda: "edit")
    monkeypatch.setattr(script, "_iter_caption_sources", lambda *a, **k: [(tmp_path / "one.png", source)])
    monkeypatch.setattr(script.QwenImage21TextCache, "is_disk_cached_outputs_expected", lambda *a, **k: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Cache hits must not load encoder")

    monkeypatch.setattr(script, "text_encoder_for_cache", forbidden)
    monkeypatch.setattr(sys, "argv", ["cache", "--dir", str(tmp_path), "--cache_dir", str(tmp_path),
                                      "--qwen3", "unused", "--cache_policy", "gpu"])
    script.main()
    assert "0 written, 1 reused" in capsys.readouterr().out
