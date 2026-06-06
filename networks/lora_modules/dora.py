"""DoRA extension for the standard Anima LoRA module.

DoRA keeps the LoRA low-rank direction but trains a per-output magnitude
parameter. At forward/merge time the adapted weight ``W + ΔW`` is normalized
by its row norm and rescaled by the learned magnitude.
"""

from __future__ import annotations

from typing import Optional

import torch

from networks.lora_modules.lora import LoRAModule


class DoRALoRAModule(LoRAModule):
    supports_conv2d = True

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
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            channel_scale=channel_scale,
        )
        magnitude = self._compute_weight_norm(
            org_module.weight.detach().to(dtype=torch.float32)
        )
        self.magnitude = torch.nn.Parameter(magnitude.contiguous())
        self._fused_delta: Optional[torch.Tensor] = None

    @staticmethod
    def _compute_weight_norm(weight: torch.Tensor) -> torch.Tensor:
        flat = weight.reshape(weight.shape[0], -1)
        return torch.linalg.norm(flat, dim=1).clamp_min(1e-6)

    def _delta_weight(self, multiplier: Optional[float] = None) -> torch.Tensor:
        return super().get_weight(multiplier=multiplier)

    def _merged_weight(
        self,
        *,
        multiplier: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        org_weight = self.org_module_ref[0].weight.to(device=device, dtype=dtype)
        delta = self._delta_weight(multiplier=multiplier).to(
            device=org_weight.device, dtype=dtype
        )
        merged = org_weight + delta
        row_scale = (
            self.magnitude.to(device=org_weight.device, dtype=dtype)
            / self._compute_weight_norm(merged).detach()
        ).clamp_min(1e-6)
        view_shape = [merged.shape[0]] + [1] * (merged.ndim - 1)
        return merged * row_scale.view(*view_shape)

    @staticmethod
    def _view_output_channel_param(
        value: torch.Tensor,
        out: torch.Tensor,
        *,
        is_conv2d: bool,
    ) -> torch.Tensor:
        if out.ndim < 2:
            raise ValueError(f"Unsupported DoRA output ndim: {out.ndim}")
        if is_conv2d:
            return value.view(1, -1, *([1] * (out.ndim - 2)))
        return value.view(*([1] * (out.ndim - 1)), -1)

    def get_weight(self, multiplier=None):
        org_weight = self.org_module_ref[0].weight.to(torch.float32)
        merged = self._merged_weight(
            multiplier=multiplier,
            device=org_weight.device,
            dtype=torch.float32,
        )
        return merged - org_weight

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            down_weight = sd["lora_down.weight"].to(torch.float32).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float32).to(device)
            magnitude_tensor = sd.get(
                "magnitude", sd.get("dora_scale", sd.get("dora_magnitude"))
            )
            if magnitude_tensor is None:
                raise KeyError("DoRA checkpoint is missing magnitude/dora_scale")
            magnitude = magnitude_tensor.to(torch.float32).to(device)

            if "inv_scale" in sd and down_weight.dim() == 2:
                inv_scale = sd["inv_scale"].to(torch.float32).to(device)
                down_weight = down_weight * inv_scale.unsqueeze(0)

            if len(down_weight.size()) == 2:
                delta = up_weight @ down_weight
            elif down_weight.size()[2:4] == (1, 1):
                delta = (
                    up_weight.squeeze(3).squeeze(2)
                    @ down_weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                delta = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)

            merged = weight.data.to(torch.float32).to(device) + (
                delta * self.scale * self.multiplier
            )
            row_scale = (magnitude / self._compute_weight_norm(merged).detach()).clamp_min(
                1e-6
            )
            view_shape = [merged.shape[0]] + [1] * (merged.ndim - 1)
            weight.data.copy_((merged * row_scale.view(*view_shape)).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused_delta = delta.detach().clone()
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        if self._fused_delta is not None:
            org_module.weight.data -= self._fused_delta.to(org_module.weight.dtype)
        self._fused_delta = None
        self._fused = False

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if self.training and self._skip_module():
            return org_forwarded

        x_lora = self._rebalance(x)
        lx = self.lora_down(x_lora)
        lx = lx * self._timestep_mask

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)
        delta_out = self.lora_up(lx) * self.multiplier * scale

        org_module = self.org_module_ref[0]
        bias = getattr(org_module, "bias", None)
        is_conv2d = org_module.__class__.__name__ == "Conv2d"
        bias_view = None
        if bias is not None:
            bias_view = self._view_output_channel_param(
                bias.to(device=org_forwarded.device, dtype=org_forwarded.dtype),
                org_forwarded,
                is_conv2d=is_conv2d,
            )

        base_without_bias = (
            org_forwarded if bias_view is None else org_forwarded - bias_view
        )
        merged_norm = self._compute_weight_norm(
            self.org_module_ref[0].weight.to(
                device=org_forwarded.device, dtype=torch.float32
            )
            + self._delta_weight().to(device=org_forwarded.device, dtype=torch.float32)
        ).detach()
        row_scale = (
            self.magnitude.to(device=org_forwarded.device, dtype=org_forwarded.dtype)
            / merged_norm.to(device=org_forwarded.device, dtype=org_forwarded.dtype)
        ).clamp_min(1e-6)
        row_scale = self._view_output_channel_param(
            row_scale,
            org_forwarded,
            is_conv2d=is_conv2d,
        )

        adapted = (base_without_bias + delta_out.to(org_forwarded.dtype)) * row_scale
        if bias_view is not None:
            adapted = adapted + bias_view
        return adapted
