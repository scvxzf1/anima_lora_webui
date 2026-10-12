"""Real tiny packed fixtures; CUDA cases can be selected independently."""
import contextlib
import copy

import pytest
import torch
from bitsandbytes.functional import dequantize_4bit, quantize_4bit
from bitsandbytes.nn import Linear4bit, Params4bit
from torch import nn
from torch.utils.checkpoint import checkpoint

from library.anima.mixed_nf4.linear import AnimaMixedNF4Linear
from library.anima.mixed_nf4.runtime import restore_linear


def packed_pair(device="cpu", *, bias=True, in_features=64, out_features=64):
    generator = torch.Generator().manual_seed(18)
    weight = torch.randn(out_features, in_features, generator=generator, dtype=torch.bfloat16) * .02
    data, state = quantize_4bit(weight, compress_statistics=True, quant_type="nf4")
    bias_tensor = torch.randn(out_features, generator=generator, dtype=torch.bfloat16) * .01 if bias else None
    custom = restore_linear(data.clone(), state.as_dict(packed=True),
                            in_features=in_features, out_features=out_features, bias=bias_tensor)
    stock = Linear4bit(in_features, out_features, bias=bias, compute_dtype=torch.bfloat16,
                      compress_statistics=True, quant_type="nf4", device="meta")
    stock.weight = Params4bit.from_prequantized(
        data.clone(), state.as_dict(packed=True), requires_grad=False, device="cpu", module=stock)
    if bias:
        stock.bias = nn.Parameter(bias_tensor.clone(), requires_grad=False)
    return custom.to(device), stock.to(device)


def input_tensor(device, dtype, noncontiguous=False):
    x = torch.randn(2, 3, 64, device=device, dtype=dtype)
    if noncontiguous:
        x = x.transpose(0, 1)
        assert not x.is_contiguous()
    return x.detach().requires_grad_()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("autocast", [False, True])
def test_cpu_matches_stock_outputs_and_input_grad(dtype, autocast):
    custom, stock = packed_pair()
    x = input_tensor("cpu", dtype, noncontiguous=True)
    reference_input = x.detach().clone().requires_grad_()
    weight, state = custom.weight, custom.weight.quant_state
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual, expected = custom(x), stock(reference_input)
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, reference_input.grad, rtol=0, atol=0)
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.count_nonzero() > 0
    assert custom.weight is weight and custom.weight.quant_state is state
    assert custom.weight.grad is None


def test_restored_subclass_preserves_matching_and_state_dict():
    from networks.lora_anima.targeting import collect_lora_target_candidates

    custom, stock = packed_pair()
    assert isinstance(custom, AnimaMixedNF4Linear)
    assert isinstance(custom, Linear4bit)
    assert isinstance(custom, nn.Linear)
    root = nn.Module()
    root.projection = custom
    candidates = collect_lora_target_candidates(
        root_module=root, prefix="lora_unet", target_replace_modules=None,
        exclude_patterns=[], include_patterns=[], is_unet=True,
        layer_start=None, layer_end=None, modules_dim=None, modules_alpha=None,
        reg_dims=None, default_dim=None, lora_dim=4, alpha=4)
    assert len(candidates) == 1 and candidates[0].child_module is custom
    assert not candidates[0].skipped
    actual, expected = custom.state_dict(), stock.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 NF4 autocast")


@cuda
@pytest.mark.parametrize("shape", [(1, 1, 4032, 1, 256), (1, 1, 4200, 1, 256)])
def test_cuda_native_bucket_shape_exact_stock_oracle(shape):
    # Native flattened Anima layout and production token budgets; width is reduced.
    custom, stock = packed_pair("cuda", bias=False, in_features=256, out_features=1024)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    packed = custom.weight.data.clone()
    dequant = dequantize_4bit(custom.weight.data, custom.weight.quant_state).clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = checkpoint(custom, x, use_reentrant=False)
        expected = checkpoint(lambda value: stock(value.to(torch.bfloat16)), xr, use_reentrant=False)
    assert actual.shape == (*shape[:-1], 1024) and actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.count_nonzero() > 0
    torch.testing.assert_close(custom.weight.data, packed, rtol=0, atol=0)
    torch.testing.assert_close(dequantize_4bit(custom.weight.data, custom.weight.quant_state),
                               dequant, rtol=0, atol=0)


@cuda
def test_cuda_anima_fp32_5d_input_checkpoint_and_real_lora():
    from library.anima.models import Anima
    from networks.lora_modules.lora import LoRAModule

    torch.manual_seed(29)
    model = Anima(max_img_h=256, max_img_w=256, max_frames=4,
                  in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
                  concat_padding_mask=False, model_channels=64, num_blocks=2,
                  num_heads=4, mlp_ratio=2, crossattn_emb_channels=64,
                  use_adaln_lora=True, adaln_lora_dim=16, use_llm_adapter=False,
                  attn_mode="torch").bfloat16().cuda().train()
    for block in model.blocks:
        for name in ("layer1", "layer2"):
            source = getattr(block.mlp, name)
            packed, state = quantize_4bit(source.weight.detach(), compress_statistics=True, quant_type="nf4")
            setattr(block.mlp, name, restore_linear(
                packed, state.as_dict(packed=True), in_features=source.in_features,
                out_features=source.out_features, bias=source.bias).cuda())
    model.requires_grad_(False)
    reference = copy.deepcopy(model)
    adapters = []
    for index, block in enumerate(model.blocks):
        for name in ("layer1", "layer2"):
            base, ref = getattr(block.mlp, name), getattr(reference.blocks[index].mlp, name)
            dense = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                              dtype=torch.bfloat16, device="cuda")
            dense.weight = nn.Parameter(dequantize_4bit(ref.weight.data, ref.weight.quant_state), requires_grad=False)
            dense.bias = ref.bias
            setattr(reference.blocks[index].mlp, name, dense)
            pair = [LoRAModule(name, module, lora_dim=4, alpha=4).cuda() for module in (base, dense)]
            for adapter in pair:
                adapter.apply_to()
            with torch.no_grad():
                pair[0].lora_up.weight.normal_(std=.02)
            pair[1].load_state_dict(pair[0].state_dict())
            adapters.extend(pair)
    model.enable_gradient_checkpointing()
    reference.enable_gradient_checkpointing()
    block_input_dtypes = []
    handles = [block.register_forward_pre_hook(lambda module, args: block_input_dtypes.append(args[0].dtype))
               for block in model.blocks]
    x = torch.randn(1, 16, 1, 8, 8, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    context = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
    # Anima's AdaLN keeps the timestep dtype outside autocast.
    t = torch.tensor([.5], device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual, expected = model(x, t, context), reference(xr, t, context)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    assert block_input_dtypes and set(block_input_dtypes) == {torch.bfloat16}
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.count_nonzero() > 0
    torch.testing.assert_close(x.grad, xr.grad, rtol=.03, atol=1e-5)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() > 0
               for adapter in adapters for p in adapter.parameters())
    for handle in handles:
        handle.remove()


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("with_lora", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_cuda_autocast_matches_dequantized_linear(dtype, noncontiguous, checkpointed, with_lora, bias):
    custom, stock = packed_pair("cuda", bias=bias)
    reference = nn.Linear(64, 64, bias=bias, dtype=torch.bfloat16, device="cuda")
    reference.weight = nn.Parameter(dequantize_4bit(
        custom.weight.data, custom.weight.quant_state).to(torch.bfloat16), requires_grad=False)
    if bias:
        reference.bias = nn.Parameter(custom.bias.detach().clone(), requires_grad=False)
    stock_forward = stock.forward
    stock.forward = lambda x: stock_forward(x.to(torch.bfloat16))
    adapters = []
    if with_lora:
        from networks.lora_modules.lora import LoRAModule
        for base in (custom, reference, stock):
            adapter = LoRAModule("tiny", base, lora_dim=4, alpha=4).cuda()
            adapters.append(adapter)
        with torch.no_grad():
            adapters[0].lora_up.weight.normal_(std=.02)
        for adapter in adapters:
            adapter.apply_to()
        # apply_to removes org_module from state_dict; copy adapter weights only.
        for adapter in adapters[1:]:
            adapter.load_state_dict(adapters[0].state_dict())
    x = input_tensor("cuda", dtype, noncontiguous)
    reference_input = x.detach().clone().requires_grad_()
    stock_input = x.detach().clone().requires_grad_()
    weight, state = custom.weight, custom.weight.quant_state
    packed_before = custom.weight.data.clone()
    dequant_before = dequantize_4bit(custom.weight.data, state).clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = checkpoint(custom, x, use_reentrant=False) if checkpointed else custom(x)
        expected = checkpoint(reference, reference_input, use_reentrant=False) if checkpointed else reference(reference_input)
        stock_expected = checkpoint(stock, stock_input, use_reentrant=False) if checkpointed else stock(stock_input)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, stock_expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    stock_expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, stock_input.grad, rtol=0, atol=0)
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.count_nonzero() > 0
    torch.testing.assert_close(x.grad, reference_input.grad, rtol=.03, atol=1e-5)
    assert custom.weight is weight and custom.weight.quant_state is state
    assert custom.weight.grad is None
    torch.testing.assert_close(custom.weight.data, packed_before, rtol=0, atol=0)
    torch.testing.assert_close(dequantize_4bit(custom.weight.data, state), dequant_before, rtol=0, atol=0)
    if with_lora:
        assert set(dict(adapters[0].named_parameters())) == {"lora_down.weight", "lora_up.weight"}
        for name, parameter in adapters[0].named_parameters():
            expected_grad = dict(adapters[1].named_parameters())[name].grad
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero() > 0
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=.03, atol=1e-5)
            torch.testing.assert_close(parameter.grad, dict(adapters[2].named_parameters())[name].grad,
                                       rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("context", ["none", "nested_disabled", "fp16", "bf16"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float32])
def test_cuda_non_target_contract_matches_stock(context, dtype, compute_dtype):
    if context == "bf16" and dtype == torch.float32 and compute_dtype == torch.bfloat16:
        pytest.skip("Target contract covered against dense Linear above")
    custom, stock = packed_pair("cuda", bias=False)
    custom.compute_dtype = stock.compute_dtype = compute_dtype
    x = input_tensor("cuda", dtype)
    xr = x.detach().clone().requires_grad_()
    with contextlib.ExitStack() as stack:
        if context != "none":
            stack.enter_context(torch.autocast("cuda", dtype=torch.float16 if context == "fp16" else torch.bfloat16))
        if context == "nested_disabled":
            stack.enter_context(torch.autocast("cuda", enabled=False))
        actual, expected = custom(x), stock(xr)
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
