# LoKr (Low-Rank Kronecker Product) module.
#
# The saved tensor layout follows LyCORIS/ComfyUI convention:
# ``{lora_name}.lokr_w1`` and ``{lora_name}.lokr_w2``.

import logging
import math

import torch
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


class LoKrModule(BaseLoRAModule):
    """LyCORIS-style LoKr adapter for Linear layers.

    Delta weight is materialized as ``kron(w1, w2)`` where
    ``w1`` is ``factor x factor`` and ``w2`` is
    ``(out_features / factor) x (in_features / factor)``.
    """

    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
        factor=8,
    ):
        if not isinstance(org_module, torch.nn.Linear):
            raise ValueError("LoKrModule only supports torch.nn.Linear modules")
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

        in_features = int(org_module.in_features)
        out_features = int(org_module.out_features)
        self.factor = self._find_factor(in_features, out_features, int(factor))
        self.in_dim = in_features // self.factor
        self.out_dim = out_features // self.factor

        self.lokr_w1 = torch.nn.Parameter(torch.empty(self.factor, self.factor))
        self.lokr_w2 = torch.nn.Parameter(torch.empty(self.out_dim, self.in_dim))

        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lokr_w2)

        if channel_scale is not None:
            logger.warning(
                "LoKr module %s: channel_scale ignored because Kronecker "
                "weights cannot absorb a per-input-column scale safely",
                lora_name,
            )
        if rank_dropout is not None:
            logger.warning(
                "LoKr module %s: rank_dropout ignored because LoKr has no "
                "explicit low-rank bottleneck",
                lora_name,
            )

        self.org_module_ref = [org_module]
        self._fused = False

    @staticmethod
    def _find_factor(in_features: int, out_features: int, target_factor: int) -> int:
        target_factor = max(1, int(target_factor))
        candidates = [target_factor]
        candidates.extend(f for f in (16, 8, 4, 2, 1) if f < target_factor)
        for factor in dict.fromkeys(candidates):
            if factor > 0 and in_features % factor == 0 and out_features % factor == 0:
                return factor
        return 1

    def _compute_weight(self) -> torch.Tensor:
        return torch.kron(self.lokr_w1, self.lokr_w2)

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)
        if self._skip_module():
            return org_forwarded

        if self.training and self.dropout is not None:
            x_lokr = F.dropout(x, p=self.dropout)
        else:
            x_lokr = x

        weight = self._compute_weight()
        if self.training:
            lx = F.linear(x_lokr.float(), weight.float())
            # LoKr has no rank axis; T-LoRA masks reduce to a scalar gate.
            lx = lx * self._timestep_mask[:, :1]
            lx = lx.to(org_forwarded.dtype)
        else:
            lx = F.linear(x_lokr, weight.to(x_lokr.dtype))

        return org_forwarded + lx * self.multiplier * self.scale

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        return self._compute_weight().float() * multiplier * self.scale

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w1 = sd["lokr_w1"].to(torch.float).to(device)
            w2 = sd["lokr_w2"].to(torch.float).to(device)
            delta = torch.kron(w1, w2)
            weight.data.add_((delta * self.multiplier * self.scale).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False
