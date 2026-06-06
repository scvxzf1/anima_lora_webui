"""GLoRA (Generalized LoRA) plugin module."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


class GLoRAModule(BaseLoRAModule):
    """LyCORIS-style GLoRA adapter for Linear layers.

    GLoRA changes a Linear layer as:

        y = W(x + A(x)) + B(x)

    where A and B are both low-rank. The A path depends on the base weight W,
    so this cannot be represented losslessly as a regular LoRA up/down pair.
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
    ):
        if not isinstance(org_module, torch.nn.Linear):
            raise ValueError("GLoRAModule only supports torch.nn.Linear modules")
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

        self.a2 = torch.nn.Linear(in_features, self.lora_dim, bias=False)
        self.a1 = torch.nn.Linear(self.lora_dim, in_features, bias=False)
        self.b2 = torch.nn.Linear(in_features, self.lora_dim, bias=False)
        self.b1 = torch.nn.Linear(self.lora_dim, out_features, bias=False)

        torch.nn.init.kaiming_uniform_(self.a1.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.b1.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.a2.weight)
        torch.nn.init.zeros_(self.b2.weight)

        if channel_scale is not None:
            logger.warning(
                "GLoRA module %s: channel_scale ignored because the A path "
                "modifies the base Linear input before W is applied",
                lora_name,
            )

        self.org_module_ref = [org_module]
        self._fused = False
        self._fused_delta: torch.Tensor | None = None

    def _rank_gate(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._timestep_mask.to(device=x.device, dtype=x.dtype)
        while gate.dim() < x.dim():
            gate = gate.unsqueeze(1)
        return x * gate

    def _rank_dropout(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        p = self.rank_dropout
        if p is None or not self.training:
            return x, 1.0
        if p <= 0.0:
            return x, 1.0
        if p >= 1.0:
            return torch.zeros_like(x), 1.0

        keep = (
            torch.rand((x.size(0), self.lora_dim), device=x.device) > float(p)
        ).to(x.dtype)
        if x.dim() == 3:
            keep = keep.unsqueeze(1)
        elif x.dim() == 4:
            keep = keep.unsqueeze(-1).unsqueeze(-1)
        return x * keep, 1.0 / (1.0 - float(p))

    def _paths(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_work = x.float()

        a_mid = F.linear(x_work, self.a2.weight.float())
        b_mid = F.linear(x_work, self.b2.weight.float())

        if self.training:
            a_mid = self._rank_gate(a_mid)
            b_mid = self._rank_gate(b_mid)
            a_mid, a_drop_scale = self._rank_dropout(a_mid)
            b_mid, b_drop_scale = self._rank_dropout(b_mid)
        else:
            a_drop_scale = 1.0
            b_drop_scale = 1.0

        a_out = F.linear(a_mid, self.a1.weight.float()) * a_drop_scale
        b_out = F.linear(b_mid, self.b1.weight.float()) * b_drop_scale

        if self.dropout is not None and self.training:
            a_out = F.dropout(a_out, p=self.dropout)
            b_out = F.dropout(b_out, p=self.dropout)

        scale = float(self.multiplier) * float(self.scale)
        return a_out * scale, b_out * scale

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self._skip_module():
            return self.org_forward(x)

        a_out, b_out = self._paths(x)
        base = self.org_forward(x + a_out.to(dtype=x.dtype))
        return base + b_out.to(device=base.device, dtype=base.dtype)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        org_weight = self.org_module_ref[0].weight.detach().float()
        a1 = self.a1.weight.float()
        a2 = self.a2.weight.float()
        b1 = self.b1.weight.float()
        b2 = self.b2.weight.float()
        delta = (org_weight @ a1) @ a2
        delta = delta + (b1 @ b2)
        return delta * float(multiplier) * float(self.scale)

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module_ref[0].weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            base = weight.data.float().to(device)
            a1 = sd["a1.weight"].float().to(device)
            a2 = sd["a2.weight"].float().to(device)
            b1 = sd["b1.weight"].float().to(device)
            b2 = sd["b2.weight"].float().to(device)
            delta = ((base @ a1) @ a2) + (b1 @ b2)
            weight.data.add_((delta * self.multiplier * self.scale).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype).detach()
        org_module.weight.data += delta
        self._fused_delta = delta
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        if self._fused_delta is not None:
            org_module.weight.data -= self._fused_delta.to(org_module.weight.dtype)
        self._fused_delta = None
        self._fused = False
