"""Actual Diffusers blocks and ModelOffloader: temporary preview and return to training."""
import pytest
import torch

from library.models.qwen_image_2_1.block_swap import enable_qwen_image_2_1_block_swap
from library.models.qwen_image_2_1.preview_memory import preview_state_scope, preview_swap_scope
from tests.test_qwen_image_2_1_block_swap import _tiny_model, _forward


@pytest.mark.parametrize("old,new", [(0, 1), (1, 0)])
def test_actual_offloader_zero_boundary_keeps_weights_and_backward(old, new):
    model = _tiny_model().requires_grad_(False)
    projection = model.transformer_blocks[0].attn.to_q
    adapter = torch.nn.Linear(projection.in_features, projection.out_features, bias=False)
    original = projection.forward
    projection.forward = lambda x: original(x) + adapter(x)
    model.enable_gradient_checkpointing()
    with preview_state_scope(model, adapter, torch.device("cpu")), torch.no_grad():
        torch.manual_seed(123)
        expected = _forward(model)
    swap = enable_qwen_image_2_1_block_swap(model, old, torch.device("cpu"),
                                           transfer_dtype="bf16", restore_mode="slab")
    params = tuple(adapter.parameters())
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    config = dict(model_family="qwen_image_2_1", blocks_to_swap=old,
                  qwen_sample_swap_mode="custom", qwen_sample_blocks_to_swap=new)
    try:
        with preview_state_scope(model, adapter, torch.device("cpu")):
            with preview_swap_scope(model, config), torch.no_grad():
                torch.manual_seed(123)
                assert torch.equal(_forward(model), expected)
        assert model.blocks_to_swap == swap.offloader.blocks_to_swap == old
        assert not swap.offloader.forward_only
        assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
        assert all(left is right for left, right in zip(params, adapter.parameters()))
        _forward(model).float().square().mean().backward()
        assert adapter.weight.grad is not None and torch.isfinite(adapter.weight.grad).all()
        assert adapter.weight.grad.count_nonzero() > 0
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        swap.offloader.set_forward_only(False)
        swap.offloader.thread_pool.shutdown(wait=True)
        for handle in swap.offloader.remove_handles:
            handle.remove()
