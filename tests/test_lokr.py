from __future__ import annotations

import torch
import torch.nn.functional as F

from networks.lora_modules.custom_autograd import lokr_project
from networks.lora_modules.lokr import LoKrModule


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
