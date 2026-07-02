"""unsloth reentrant-checkpoint grad-flow invariant."""

from __future__ import annotations

import torch


def test_unsloth_ckpt_propagates_param_grads_via_input():
    import torch.nn as nn

    from library.anima.models import unsloth_checkpoint

    lin = nn.Linear(4, 4)
    x = torch.randn(2, 4, requires_grad=True)
    unsloth_checkpoint(lin, x).sum().backward()
    assert lin.weight.grad is not None
    assert lin.weight.grad.abs().sum() > 0
