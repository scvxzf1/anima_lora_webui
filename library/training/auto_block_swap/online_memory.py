"""Bound prospective CPU master growth before a sparse-residency migration."""

from library.runtime.block_swap_sparse_masters import participating_blocks
from .resources import host_memory


def check_master_growth(controller, decision):
    host = host_memory()
    masters = controller.offloader._cpu_weight_masters
    needed = sum(
        controller.block_bytes[index]
        for index in participating_blocks(len(controller.block_bytes), decision.blocks)
        if not masters[index]
    )
    if not needed or needed + 2 * max(controller.block_bytes) < host.available - host.reserve:
        return True
    if decision.reason in ("gpu_pressure", "new_shape"):
        raise RuntimeError("Dynamic AUTO cannot satisfy both host RAM and GPU reserve")
    return False
