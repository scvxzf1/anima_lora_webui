from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from library.runtime.peak_probe import PeakProbe
from networks.plugins.lokr import autograd as lokr_autograd
from networks.plugins.lokr.autograd import (
    lokr_add_grouped_delta_,
    lokr_project,
    lokr_project_factor,
    lokr_project_factor_group,
)
from networks.plugins.lokr.module import LoKrModule


def test_lokr_eval_forward_ignores_stale_timestep_mask():
    base = torch.nn.Linear(4, 4, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
    )
    lokr.apply_to()

    with torch.no_grad():
        lokr.org_module_ref[0].weight.zero_()
        lokr.lokr_w1.fill_(1.0)
        lokr.lokr_w2.fill_(1.0)
        lokr._timestep_mask.zero_()

    x = torch.ones(1, 4)

    lokr.train()
    train_out = lokr.org_module_ref[0](x)

    lokr.eval()
    eval_out = lokr.org_module_ref[0](x)

    torch.testing.assert_close(train_out, torch.zeros_like(train_out))
    assert torch.count_nonzero(eval_out).item() == eval_out.numel()


def test_lokr_project_matches_kron_linear_forward_and_backward():
    torch.manual_seed(1)
    factor = 2
    in_dim = 3
    out_dim = 4
    x = torch.randn(2, 5, factor * in_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)
    grad = torch.randn(2, 5, factor * out_dim)

    y = lokr_project(x, w1, w2, factor, in_dim, out_dim)
    y.backward(grad)
    grads = [x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = F.linear(x_ref, torch.kron(w1_ref, w2_ref))
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_project_does_not_materialize_kron(monkeypatch):
    factor = 2
    in_dim = 3
    out_dim = 4
    x = torch.randn(2, factor * in_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)

    def _fail_kron(*_args, **_kwargs):
        raise AssertionError("lokr_project must not materialize torch.kron")

    monkeypatch.setattr(torch, "kron", _fail_kron)

    y = lokr_project(x, w1, w2, factor, in_dim, out_dim)
    y.sum().backward()

    assert y.shape == (2, factor * out_dim)
    assert x.grad is not None
    assert w1.grad is not None
    assert w2.grad is not None


def test_lokr_project_factor_chunked_matches_kron_slice(monkeypatch):
    torch.manual_seed(3)
    factor = 2
    in_dim = 3
    out_dim = 4
    out_factor = 1
    x = torch.randn(5, factor * in_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)
    grad = torch.randn(5, out_dim)

    monkeypatch.setattr(lokr_autograd, "_LOKR_PROJECT_CHUNK_BYTES", 1)

    y = lokr_project_factor(x, w1, w2, out_factor, factor, in_dim, out_dim)
    y.backward(grad)
    grads = [x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref_full = F.linear(x_ref, torch.kron(w1_ref, w2_ref))
    start = out_factor * out_dim
    y_ref = y_ref_full[:, start : start + out_dim]
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_project_factor_group_matches_kron_slice(monkeypatch):
    torch.manual_seed(5)
    factor = 4
    in_dim = 3
    out_dim = 4
    out_start = 1
    out_count = 2
    x = torch.randn(2, 5, factor * in_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)
    grad = torch.randn(2, 5, out_count * out_dim)

    monkeypatch.setattr(lokr_autograd, "_LOKR_PROJECT_CHUNK_BYTES", 1)

    y = lokr_project_factor_group(
        x, w1, w2, out_start, out_count, factor, in_dim, out_dim
    )
    y.backward(grad)
    grads = [x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref_full = F.linear(x_ref, torch.kron(w1_ref, w2_ref))
    start = out_start * out_dim
    y_ref = y_ref_full[..., start : start + out_count * out_dim]
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_add_grouped_delta_matches_kron_forward_and_backward(monkeypatch):
    torch.manual_seed(7)
    factor = 4
    in_dim = 3
    out_dim = 4
    x = torch.randn(2, 5, factor * in_dim, requires_grad=True)
    base_leaf = torch.randn(2, 5, factor * out_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)
    gate = torch.tensor([[0.75]], dtype=torch.float32)
    grad = torch.randn(2, 5, factor * out_dim)

    monkeypatch.setattr(lokr_autograd, "_LOKR_PROJECT_CHUNK_BYTES", 1)

    y = lokr_add_grouped_delta_(
        base_leaf + 0,
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
    )
    y.backward(grad)
    grads = [base_leaf.grad.clone(), x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    base_ref = base_leaf.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = base_ref + F.linear(x_ref.float(), torch.kron(w1_ref, w2_ref)) * gate
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], base_ref.grad)
    torch.testing.assert_close(grads[1], x_ref.grad)
    torch.testing.assert_close(grads[2], w1_ref.grad)
    torch.testing.assert_close(grads[3], w2_ref.grad)


def test_lokr_module_custom_forward_matches_kron_path():
    torch.manual_seed(2)
    x = torch.randn(3, 6, requires_grad=True)
    grad = torch.randn(3, 8)

    base = torch.nn.Linear(6, 8, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
    )
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True
    with torch.no_grad():
        lokr.org_module_ref[0].weight.copy_(torch.randn_like(lokr.org_module_ref[0].weight))
        lokr.lokr_w1.copy_(torch.randn_like(lokr.lokr_w1))
        lokr.lokr_w2.copy_(torch.randn_like(lokr.lokr_w2))

    y = lokr.org_module_ref[0](x)
    y.backward(grad)
    grads = [x.grad.clone(), lokr.lokr_w1.grad.clone(), lokr.lokr_w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = lokr.lokr_w1.detach().clone().requires_grad_()
    w2_ref = lokr.lokr_w2.detach().clone().requires_grad_()
    org_weight = lokr.org_module_ref[0].weight.detach()
    y_ref = F.linear(x_ref, org_weight) + F.linear(x_ref.float(), torch.kron(w1_ref, w2_ref))
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_module_custom_forward_reuses_base_output(monkeypatch):
    torch.manual_seed(4)
    x = torch.randn(3, 6, requires_grad=True)

    base = torch.nn.Linear(6, 8, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
    )
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True

    def _fail_empty_like(*_args, **_kwargs):
        raise AssertionError("custom LoKr forward must not allocate a full result")

    monkeypatch.setattr(torch, "empty_like", _fail_empty_like)

    y = lokr.org_module_ref[0](x)
    y.sum().backward()

    assert y.shape == (3, 8)
    assert x.grad is not None
    assert lokr.lokr_w1.grad is not None
    assert lokr.lokr_w2.grad is not None


def test_lokr_module_custom_forward_uses_configured_factor_group(monkeypatch):
    torch.manual_seed(6)
    x = torch.randn(3, 16, requires_grad=True)

    base = torch.nn.Linear(16, 16, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=4,
        lokr_factor_group_size=2,
        lokr_project_chunk_bytes=1234,
    )
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True

    calls = []
    original = lokr_autograd.lokr_add_grouped_delta_

    def _tracking_project(*args, **kwargs):
        calls.append((args[8], args[9]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "networks.plugins.lokr.module.lokr_add_grouped_delta_",
        _tracking_project,
    )

    y = lokr.org_module_ref[0](x)
    y.sum().backward()

    assert calls == [(2, 1234)]


def test_lokr_peak_probe_records_delta_events(tmp_path):
    x = torch.randn(2, 4)
    base = torch.nn.Linear(4, 4, bias=False)
    lokr = LoKrModule(
        "lora_unet_blocks_3_mlp_layer1",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
    )
    lokr.original_name = "blocks.3.mlp.layer1"
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True
    lokr._peak_probe = PeakProbe(str(tmp_path / "peak.jsonl"), max_steps=0, level="full")

    lokr._peak_probe.begin_step(0)
    _ = lokr.org_module_ref[0](x)
    lokr._peak_probe.end_step()

    rows = [
        json.loads(line)
        for line in (tmp_path / "peak.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = [row.get("label") for row in rows]
    assert "lokr_after_base" in labels
    assert "lokr_before_delta_apply" in labels
    assert "lokr_after_delta_apply" in labels
    delta = next(row for row in rows if row.get("label") == "lokr_after_delta_apply")
    assert delta["block_idx"] == 3
    assert delta["block_phase"] == "mlp"
    assert delta["op_name"] == "mlp_layer1"
