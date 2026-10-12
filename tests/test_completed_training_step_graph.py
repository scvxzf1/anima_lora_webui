"""Completed step returns preserve diagnostics without retaining frozen ctx."""

from contextlib import nullcontext
from types import SimpleNamespace
import weakref

import torch


def test_run_step_callbacks_get_live_loss_outer_result_releases_ctx(monkeypatch):
    from library.training import loop
    from library.training.auto_block_swap import online

    references, callbacks = [], []
    class FrozenConstant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, parameter, constant):
            ctx.constant = constant
            return parameter.sum() * constant.sum()
        @staticmethod
        def backward(ctx, grad):
            return grad.expand(1, 1) * ctx.constant.sum(), None
    network = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(network.parameters(), lr=.01)
    def process(*args, **kwargs):
        constant = torch.ones(5)
        references.append(weakref.ref(constant))
        return FrozenConstant.apply(network.weight, constant)
    def after(state, loss):
        callbacks.append((loss.requires_grad, loss.grad_fn is not None, loss.item()))
    monkeypatch.setattr(online, "before_microstep", lambda *args: None)
    monkeypatch.setattr(online, "after_microstep", after)
    monkeypatch.setattr(loop, "adaptive_notify", lambda *args: None)
    monkeypatch.setattr(loop, "precision_observe", lambda *args, **kwargs: None)
    monkeypatch.setattr(loop, "synchronize_optimizer_gradients", lambda *args: None)
    monkeypatch.setattr(loop, "debug_finite_enabled", lambda *args: False)
    accelerator = SimpleNamespace(
        sync_gradients=True, device=torch.device("cpu"),
        unwrap_model=lambda model: model, accumulate=lambda model: nullcontext(),
        backward=lambda loss: loss.backward(),
    )
    state = SimpleNamespace(
        args=SimpleNamespace(log_every_n_steps=1, max_train_steps=2, max_grad_norm=0),
        accelerator=accelerator, network=network, optimizer=optimizer,
        training_model=network, text_encoder=None, unet=network,
        on_step_start_for_network=lambda *args: None, train_ctx=None,
        global_step=0, profile_started=False, is_tracking=False,
        lr_scheduler=SimpleNamespace(step=lambda: None),
    )
    trainer = SimpleNamespace(
        _cudagraph_mark_step=False, _state=SimpleNamespace(personalization_observer={}),
        process_batch=process, on_step_start=lambda *args, **kwargs: None,
        run_after_backward=lambda *args: None,
    )
    result = loop._run_step(trainer, state, {})
    assert callbacks == [(True, True, result.item())]
    assert not result.requires_grad and result.grad_fn is None
    assert references[0]() is None
    assert all(parameter.grad is None for parameter in network.parameters())
