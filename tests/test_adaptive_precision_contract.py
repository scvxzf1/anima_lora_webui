from types import SimpleNamespace

import pytest
import torch

from library.training.adaptive_runtime.contract import precision_contract_id, precision_request


def config(**changes):
    values = dict(
        adaptive_precision="fp16_fp32",
        adaptive_resolved_mode="fp16_fp32",
        adaptive_candidate="fp16",
        mixed_precision="fp16",
        model_family="krea2_raw",
        adaptive_fp32_modules=["blocks.0.*"],
        adaptive_loss_scale=1024.0,
        base_compute="bf16",
        attn_mode="torch",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_memory_only_changes_do_not_change_precision_contract():
    first = config(blocks_to_swap=20, adaptive_oom_retry_max_swap=24)
    second = config(blocks_to_swap=24, adaptive_oom_retry_max_swap=26)
    assert precision_contract_id(first) == precision_contract_id(second)


def test_precision_changes_change_contract_identity():
    baseline = precision_contract_id(config())
    assert precision_contract_id(config(adaptive_loss_scale=2048.0)) != baseline
    assert precision_contract_id(config(adaptive_fp32_modules=["blocks.1.*"])) != baseline
    assert precision_contract_id(config(attn_mode="sdpa")) != baseline


def test_contract_is_canonical_and_keeps_ordered_explicit_patterns():
    request = precision_request(config(adaptive_fp32_modules=("b", "a")))
    assert request["schema"] == "adaptive_precision_request_v1"
    assert request["adaptive_fp32_modules"] == ["b", "a"]
    assert precision_contract_id(config(adaptive_fp32_modules=["b", "a"])) == precision_contract_id(
        config(adaptive_fp32_modules=("b", "a"))
    )


def test_available_cuda_with_unreadable_identity_fails_closed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "current_device",
        lambda: (_ for _ in ()).throw(RuntimeError("device probe failed")),
    )
    with pytest.raises(ValueError, match="precision_contract_mismatch.*device identity"):
        precision_contract_id(config())
