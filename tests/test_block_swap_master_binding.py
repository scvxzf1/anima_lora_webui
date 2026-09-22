import pytest
import torch

from library.runtime.block_swap_masters import _bind_captured_cpu_weights


def test_native_cpu_master_replaces_duplicate_storage_without_changing_values():
    block = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).requires_grad_(False)
    parameter = block[0].weight
    expected = parameter.detach().clone()
    master = expected.clone()
    assert parameter.data_ptr() != master.data_ptr()
    _bind_captured_cpu_weights(block, {"0": master})
    assert block[0].weight is parameter
    assert parameter.data_ptr() == master.data_ptr()
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["trainable", "dtype", "shape", "container", "subclass"])
def test_non_native_or_trainable_payload_is_not_rebound(case):
    block = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)).requires_grad_(False)
    master = block[0].weight.detach().clone()
    if case == "trainable":
        block.requires_grad_(True)
    elif case == "dtype":
        master = master.half()
    elif case == "shape":
        master = master.flatten()
    elif case == "container":
        master = {"quantized": master}
    else:
        class CustomParameter(torch.nn.Parameter):
            pass
        block[0].weight = CustomParameter(block[0].weight.detach(), requires_grad=False)
    original = block[0].weight.data_ptr()
    _bind_captured_cpu_weights(block, {"0": master})
    assert block[0].weight.data_ptr() == original
