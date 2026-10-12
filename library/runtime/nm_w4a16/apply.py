"""Explicit reversible plain-LoRA MLP patch, separate from train.py/bootstrap.

Original BF16 weights remain resident for restoration. This API cannot claim
whole-model memory savings and is not compatible with merge/compile/swap yet.
"""

from dataclasses import dataclass
import re

from torch import nn

from .config import NMConfig
from .linear import FrozenNMLinear


_MLP = re.compile(r"^(?:net\.)?blocks\.\d+\.(?:mlp\.(?:layer1|layer2|gate|up|down)|img_mlp\.(?:gate_layer|proj))$")
_CHILD = "_nm_w4a16_experiment"


@dataclass
class LoRABasePatch:
    lora: nn.Module
    base: nn.Linear
    original_forward: object
    runtime: FrozenNMLinear

    def restore(self):
        if self.lora.org_forward != self.runtime.forward:
            raise RuntimeError("base forward was changed after installing the NM experiment")
        self.lora.org_forward = self.original_forward
        delattr(self.base, _CHILD)


def patch_plain_lora_mlp(lora, *, config=NMConfig(), input_second_moment=None):
    """Install on ONE explicitly selected, already-applied plain LoRA module.

    No implicit model traversal, base release, checkpoint integration, compile,
    or production configuration change. Caller must restore before merge/save.
    """
    from networks.lora_modules.lora import LoRAModule

    if type(lora) is not LoRAModule or getattr(lora, "_fused", False):
        raise TypeError("NM experiment only supports unfused plain LoRAModule")
    name = str(getattr(lora, "original_name", ""))
    if not _MLP.fullmatch(name):
        raise ValueError("NM experiment only supports explicitly named DiT block MLPs")
    original = getattr(lora, "org_forward", None)
    base = getattr(original, "__self__", None)
    if type(base) is not nn.Linear or getattr(original, "__func__", None) is not nn.Linear.forward:
        raise ValueError("apply LoRA first; base forward must not already be replaced")
    if hasattr(base, _CHILD) or getattr(base, "_compiled_call_impl", None) is not None:
        raise ValueError("base is already patched or compiled")
    runtime = FrozenNMLinear.from_linear(base, config=config, input_second_moment=input_second_moment)
    if config.cache_sparse:
        runtime.prepare_sparse()
    base.add_module(_CHILD, runtime)
    lora.org_forward = runtime.forward
    return LoRABasePatch(lora, base, original, runtime)
