# Step-expert LoRA: shared down-projection + K up-heads selected by the
# diffusion step index (no learned router — the step counter is known at call
# time). Used by the turbo DP-DMD student when ``per_step_expert`` is on so
# head A serves step 0 (diversity) and head B serves step 1 (quality) without
# the two conflicting gradients fighting over one set of up-weights.
#
# Layout mirrors a plain ``LoRAModule`` (Linear-only) but with the up-projection
# replaced by a ``ModuleList`` of K heads off one shared ``lora_down``. Only the
# head at ``self._step`` contributes per forward, so per-step inference compute
# is identical to a single-head LoRA. Selection is a plain Python int attribute
# (not a tensor read) so ``torch.compile`` guards on it and specializes one
# graph per step value (K graphs) instead of forcing an ``.item()`` graph break.

import logging
import math

import torch

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.custom_autograd import lora_down_project

logger = logging.getLogger(__name__)


class StepExpertLoRAModule(BaseLoRAModule):
    """Shared-A, dual(-K)-B-head LoRA with hard step-index head selection.

    ``forward`` reads ``self._step`` (set by the network's
    ``set_step_index`` / coordinator's ``set_student_step``) and routes the
    bottleneck through ``self.lora_ups[self._step]``. ``lora_down`` is shared
    across heads — the common subspace both objectives train; only the up-heads
    specialize.

    Linear-only (the turbo student targets fused attention / MLP projections,
    all ``nn.Linear``). ``merge_to`` / ``fuse_weight`` are intentionally absent:
    K per-step heads cannot fold into one static DiT weight, so ``make merge``
    refuses per-step-expert turbo (see ``scripts/merge_to_dit.py``).
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
        step_expert_K: int = 2,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
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

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("StepExpertLoRAModule supports Linear only")

        self.K = int(step_expert_K)
        if self.K < 1:
            raise ValueError(f"step_expert_K must be >= 1, got {self.K}")

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        # K up-heads off the single shared down-proj. zero-init every head so
        # ΔW = 0 at start (same invariant LoRAModule relies on); the head a step
        # routes to is the only one that ever receives gradient for that step.
        self.lora_ups = torch.nn.ModuleList(
            [torch.nn.Linear(self.lora_dim, out_dim, bias=False) for _ in range(self.K)]
        )

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        for up in self.lora_ups:
            torch.nn.init.zeros_(up.weight)

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        # Opt-in (set by the network factory): save bf16 x instead of fp32
        # x_lora for the down-proj backward.
        self.use_custom_down_autograd = False

        # Hard step-index selection. Plain Python int (not a buffer): the LoRA
        # forward is monkey-patched into the compiled DiT block, so reading a
        # tensor + ``.item()`` here would force a graph break. As a guarded int,
        # dynamo specializes one graph per distinct value — exactly the K we
        # cycle through (mirrors the 2-graphs-by-token-count compile story).
        self._step = 0

    def set_step(self, k: int) -> None:
        """Select the active up-head for subsequent forwards (0 <= k < K)."""
        if not (0 <= k < self.K):
            raise IndexError(
                f"step index {k} out of range for {self.K} heads ({self.lora_name})"
            )
        self._step = int(k)

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)
        up = self.lora_ups[self._step]

        if not self.training:
            x_lora = self._rebalance(x)
            lx = up(self.lora_down(x_lora))
            return org_forwarded + lx * self.multiplier * self.scale

        # Training: bf16 storage, fp32 bottleneck matmuls (mirrors LoRAModule).
        if self._skip_module():
            return org_forwarded

        if self.use_custom_down_autograd and isinstance(
            self.lora_down, torch.nn.Linear
        ):
            inv_scale = self.inv_scale if self._has_channel_scale else None
            lx = lora_down_project(x, self.lora_down.weight, inv_scale)
        else:
            x_lora = self._rebalance(x)
            lx = torch.nn.functional.linear(
                x_lora.float(), self.lora_down.weight.float()
            )

        lx = lx * self._timestep_mask

        if self.dropout is not None:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        lx = torch.nn.functional.linear(lx, up.weight.float())
        return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)
