"""CPU masters for training's actual exchange pairs, not the fixed middle."""

from .block_swap_masters import _can_swap_frozen_weight_to_cpu, _capture_cpu_master


def participating_blocks(total, count):
    # Forward retires [0,count), backward restores them from [total-count,total).
    return set(range(count)) | set(range(total - count, total))


def uses_sparse_masters(offloader):
    return getattr(offloader, "master_scope", "all") == "participating"


def capture_missing(offloader, blocks, count):
    """Capture newly participating GPU blocks before any residency migration."""
    if not uses_sparse_masters(offloader):
        return
    for index in sorted(participating_blocks(len(blocks), count)):
        if offloader._cpu_weight_masters[index]:
            continue
        masters, dtypes = {}, {}
        for name, module in blocks[index].named_modules():
            if not _can_swap_frozen_weight_to_cpu(module):
                continue
            weight = module.weight
            master, _ = _capture_cpu_master(
                weight, module_name=name, pin_memory=offloader.cuda_available,
                transfer_dtype="bf16",
            )
            masters[name], dtypes[name] = master, weight.dtype
        masters, slab, plan = offloader._pack_cpu_master_block(
            masters, pin_memory=offloader.cuda_available,
        )
        offloader._cpu_weight_masters[index] = masters
        offloader._cpu_weight_master_dtypes[index] = dtypes
        offloader._cpu_weight_master_slabs[index] = slab
        offloader._cpu_weight_master_slab_plans[index] = plan


def trim_masters(offloader, blocks, count):
    if not uses_sparse_masters(offloader):
        return
    active = participating_blocks(len(blocks), count)
    # Cached plans own strong master references, so invalidate them before dropping data.
    offloader._swap_plan_cache.clear()
    offloader._swap_slab_plan_cache.clear()
    for index in range(len(blocks)):
        if index not in active:
            if offloader.cuda_available and any(
                module.weight.device.type != "cuda"
                for module in blocks[index].modules()
                if _can_swap_frozen_weight_to_cpu(module)
            ):
                raise RuntimeError("Cannot release a CPU master before restoring its GPU weight")
            offloader._cpu_weight_masters[index] = {}
            offloader._cpu_weight_master_dtypes[index] = {}
            offloader._cpu_weight_master_slabs[index] = None
            offloader._cpu_weight_master_slab_plans[index] = None
    sizes = [sum(
        master.stored_nbytes() if hasattr(master, "stored_nbytes")
        else master.numel() * master.element_size()
        for master in masters.values()
    ) for masters in offloader._cpu_weight_masters]
    offloader._frozen_weight_bytes_by_block = sizes
    offloader._frozen_weight_master_bytes = sum(sizes)
