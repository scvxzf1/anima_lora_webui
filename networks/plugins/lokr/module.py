"""LoKr (Low-Rank Kronecker Product) plugin module."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from library.runtime.peak_probe import record_peak_probe_event
from networks.lora_modules.base import BaseLoRAModule
from networks.plugins.lokr.autograd import (
    DEFAULT_LOKR_PROJECT_CHUNK_BYTES,
    lokr_add_grouped_delta_,
    lokr_project_factor_group,
)

logger = logging.getLogger(__name__)


class LoKrModule(BaseLoRAModule):
    """LyCORIS-style LoKr adapter for Linear layers."""

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
        lokr_factor_group_size=8,
        lokr_project_chunk_bytes=DEFAULT_LOKR_PROJECT_CHUNK_BYTES,
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
                "weights cannot safely absorb per-input-column scale",
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
        self.use_custom_lokr_autograd = False
        self.lokr_factor_group_size = max(1, int(lokr_factor_group_size))
        self.lokr_project_chunk_bytes = max(1, int(lokr_project_chunk_bytes))
        self._peak_probe = None

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
        original_name = getattr(self, "original_name", None)
        peak_probe = self._peak_probe
        record_lokr = bool(getattr(peak_probe, "record_lokr", False))
        if record_lokr:
            record_peak_probe_event(
                peak_probe,
                "lokr_after_base",
                tensor=org_forwarded,
                module_type="lokr",
                lora_name=self.lora_name,
                original_name=original_name,
                factor=self.factor,
                group_size=self.lokr_factor_group_size,
                chunk_bytes=self.lokr_project_chunk_bytes,
            )
        if self._skip_module():
            return org_forwarded

        x_lokr = F.dropout(x, p=self.dropout) if self.training and self.dropout else x
        if record_lokr:
            record_peak_probe_event(
                peak_probe,
                "lokr_before_delta_apply",
                tensor=x_lokr,
                module_type="lokr",
                lora_name=self.lora_name,
                original_name=original_name,
                factor=self.factor,
                group_size=self.lokr_factor_group_size,
                chunk_bytes=self.lokr_project_chunk_bytes,
            )
        if self.training:
            if self.use_custom_lokr_autograd:
                gate_scale = self._timestep_mask[:, :1] * self.multiplier * self.scale
                group_size = max(1, min(int(self.lokr_factor_group_size), self.factor))
                if gate_scale.numel() == 1 and org_forwarded.is_contiguous():
                    result = lokr_add_grouped_delta_(
                        org_forwarded,
                        x_lokr,
                        self.lokr_w1,
                        self.lokr_w2,
                        gate_scale,
                        self.factor,
                        self.in_dim,
                        self.out_dim,
                        group_size,
                        self.lokr_project_chunk_bytes,
                    )
                else:
                    # Preserve the existing broadcast semantics for rare
                    # non-scalar T-LoRA masks; the fused path intentionally
                    # optimizes the standard scalar LoKr gate used by Anima.
                    result = org_forwarded
                    for out_start in range(0, self.factor, group_size):
                        out_count = min(group_size, self.factor - out_start)
                        start = out_start * self.out_dim
                        end = start + out_count * self.out_dim
                        lx_slice = lokr_project_factor_group(
                            x_lokr,
                            self.lokr_w1,
                            self.lokr_w2,
                            out_start,
                            out_count,
                            self.factor,
                            self.in_dim,
                            self.out_dim,
                            self.lokr_project_chunk_bytes,
                        )
                        result[..., start:end].add_(
                            (lx_slice * gate_scale).to(org_forwarded.dtype)
                        )
                if record_lokr:
                    record_peak_probe_event(
                        peak_probe,
                        "lokr_after_delta_apply",
                        tensor=result,
                        module_type="lokr",
                        lora_name=self.lora_name,
                        original_name=original_name,
                        factor=self.factor,
                        group_size=self.lokr_factor_group_size,
                        chunk_bytes=self.lokr_project_chunk_bytes,
                    )
                return result
            else:
                lx = F.linear(x_lokr.float(), self._compute_weight().float())
            # LoKr has no rank axis; T-LoRA masks reduce to a scalar gate.
            lx = lx * self._timestep_mask[:, :1]
            lx = lx.to(org_forwarded.dtype)
        else:
            weight = self._compute_weight()
            lx = F.linear(x_lokr, weight.to(x_lokr.dtype))

        result = org_forwarded + lx * self.multiplier * self.scale
        if record_lokr:
            record_peak_probe_event(
                peak_probe,
                "lokr_after_delta_apply",
                tensor=result,
                module_type="lokr",
                lora_name=self.lora_name,
                original_name=original_name,
                factor=self.factor,
                group_size=self.lokr_factor_group_size,
                chunk_bytes=self.lokr_project_chunk_bytes,
            )
        return result

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
