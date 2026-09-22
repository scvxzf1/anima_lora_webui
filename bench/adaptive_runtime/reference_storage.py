"""Experimental exact BF16-source storage with FP32 execution, after placement."""

import torch


def validate_reference_storage(args):
    if not getattr(args, "reference_bf16_storage", False):
        return
    if (args.precision != "fp32-reference" or args.swap <= 0
            or not args.disposable_probe or not args.capture_training
            or args.resume or args.checkpoint_every_step or args.capture_linear):
        raise ValueError("BF16 reference storage requires a swapped disposable FP32 training capture")


class Fp32ExecutionGuard:
    """Check actual module execution, including checkpoint recomputation."""

    def __init__(self, model, report):
        self.stats = {"module_calls_checked": 0, "compute_dtype": "float32",
                      "cpu_master_dtype": "bfloat16", "cpu_master_bytes": 0}
        report["reference_storage"] = self.stats
        self.handles = [module.register_forward_pre_hook(self.check)
                        for module in model.modules()]

    def check(self, module, args):
        if torch.is_autocast_enabled("cuda"):
            raise RuntimeError("FP32 reference cannot execute under CUDA autocast")
        for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False)):
            if tensor.is_floating_point() and (
                    tensor.dtype != torch.float32 or tensor.device.type != "cuda"):
                raise RuntimeError("FP32 reference encountered a non-FP32 CUDA execution tensor")
        self.stats["module_calls_checked"] += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def promote_placed_reference(model, report):
    """Call once after ordinary BF16 placement, before any forward or transfer.

    Parked frozen weights retain exact BF16 source values. All other floating
    tensors are already on CUDA and promoted once; only frozen weights swap.
    This bench helper intentionally rejects unsupported private-API states.
    """
    offloader = model.offloader
    masters = offloader._cpu_weight_masters
    if (offloader.restore_mode != "foreach" or offloader.futures
            or offloader._swap_gpu_slab_cache or not masters
            or len(masters) != len(model.blocks)):
        raise ValueError("Reference storage requires initialized idle foreach masters")
    parked = set()
    total_bytes = 0
    for block, table in zip(model.blocks, masters, strict=True):
        for name, master in table.items():
            weight = block.get_submodule(name).weight
            if (type(master) is not torch.Tensor or master.device.type != "cpu"
                    or master.dtype != torch.bfloat16 or type(weight) is not torch.nn.Parameter
                    or weight.requires_grad or weight.shape != master.shape):
                raise ValueError("Reference storage requires frozen unquantized BF16 source masters")
            total_bytes += master.numel() * master.element_size()
            if weight.device.type == "cpu":
                if weight.data_ptr() != master.data_ptr():
                    raise ValueError("Parked reference weight does not alias its immutable master")
                parked.add(id(weight))
    # Validate everything before mutation; no CPU floating payload may escape
    # the master inventory and accidentally remain low precision at execution.
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.is_floating_point() and id(tensor) not in parked and tensor.device.type != "cuda":
            raise ValueError("Reference has an unexpected nonresident floating tensor")
    for parameter in model.parameters():
        if parameter.is_floating_point() and id(parameter) not in parked:
            parameter.data = parameter.data.float()
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_floating_point():
                module._buffers[name] = buffer.float()
    offloader._cpu_weight_master_dtypes = [
        {name: torch.float32 for name in table} for table in masters]
    offloader._swap_plan_cache.clear()
    offloader._swap_slab_plan_cache.clear()
    guard = Fp32ExecutionGuard(model, report)
    guard.stats["cpu_master_bytes"] = total_bytes
    return guard
