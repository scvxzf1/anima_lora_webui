# LoReFT: Low-Rank Representation Fine-Tuning.
# Wu et al., "ReFT: Representation Finetuning for Language Models" (NeurIPS 2024)

from typing import Optional

import torch


class ReFTModule(torch.nn.Module):
    """LoReFT: low-rank residual-stream intervention.

        h_new = h + R^T(ΔW·h + b) * scale * multiplier

    R is an orthogonal rotation; ΔW (``learned_source``) is the learned delta
    in that subspace. Direct ΔW parameterisation (vs the paper's ``Wh + b − Rh``
    form) avoids activation-level cancellation, so the module runs in bf16
    without fp32 upcasts.

    ``org_module`` is normally a DiT Block — wrapping at the residual-stream
    level matches the paper (Wu et al., NeurIPS 2024 §3.3). Zero-init keeps
    delta=0 at step 0.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        embed_dim: Optional[int] = None,
        multiplier=1.0,
        reft_dim=4,
        alpha=1,
        dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if embed_dim is None:
            if hasattr(org_module, "out_features"):
                embed_dim = org_module.out_features
            else:
                raise ValueError(
                    "embed_dim must be provided when wrapping a non-Linear module "
                    f"(got {type(org_module).__name__})"
                )
        if not 1 <= reft_dim <= embed_dim:
            raise ValueError(f"ReFT rank must be in [1, {embed_dim}], got {reft_dim}")
        self.reft_dim = reft_dim
        self.enabled = True
        self._applied = False
        self._forward_method = getattr(org_module, "reft_forward_method", "forward")
        if self._forward_method not in {"forward", "_forward"}:
            raise ValueError(f"Unsupported ReFT target: {self._forward_method!r}")

        # R: orthogonal rotation into the intervention subspace.
        self.rotate_layer = torch.nn.Linear(embed_dim, reft_dim, bias=False)
        r_rand = torch.randn(
            embed_dim, reft_dim, device=self.rotate_layer.weight.device
        )
        r_orth, _ = torch.linalg.qr(r_rand)
        with torch.no_grad():
            self.rotate_layer.weight.copy_(r_orth.T)
        del r_rand, r_orth

        # ΔW within R's subspace; zero-init → delta=0 at step 0.
        self.learned_source = torch.nn.Linear(embed_dim, reft_dim)
        torch.nn.init.zeros_(self.learned_source.weight)
        torch.nn.init.zeros_(self.learned_source.bias)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().item()
        alpha = reft_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / reft_dim
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.module_dropout = module_dropout

        # See BaseLoRAModule._timestep_mask: all-ones default → identity, no
        # None-vs-Tensor guard. T-LoRA rebinds via set_reft_timestep_mask.
        self.register_buffer(
            "_timestep_mask",
            torch.ones(1, reft_dim, dtype=torch.float32),
            persistent=False,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # The factory cannot infer per-block alpha from its placeholder config.
        # Keep the cached scalar in sync on both network and native-state loads.
        self.scale = self.alpha.item() / self.reft_dim

    def apply_to(self):
        if self._applied:
            return
        if self._forward_method == "_forward" and hasattr(
            self.org_module, "_krea_compile_base_forward"
        ):
            raise RuntimeError(
                "Apply ReFT before compile_blocks(); rebuild the model first"
            )
        original = getattr(self.org_module, self._forward_method)
        if isinstance(getattr(original, "__self__", None), ReFTModule):
            raise RuntimeError("A ReFT adapter is already installed on this block")
        self.org_forward = original
        setattr(self.org_module, self._forward_method, self.forward)
        self._applied = True
        del self.org_module

    def forward(self, *args, **kwargs):
        # Works for wrapped Linear (x) and wrapped DiT Block (multi-arg).
        h = self.org_forward(*args, **kwargs)

        if not self.enabled or self.multiplier == 0:
            return h

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return h

        delta = torch.nn.functional.linear(
            h, self.learned_source.weight, self.learned_source.bias
        )

        if self.training:
            delta = delta * self._timestep_mask.to(delta.dtype)

        if self.dropout is not None and self.training:
            delta = torch.nn.functional.dropout(delta, p=self.dropout)

        edit = torch.nn.functional.linear(delta, self.rotate_layer.weight.T)
        return h + edit * (self.multiplier * self.scale)

    def regularization(self):
        """||R R^T - I||^2."""
        R = self.rotate_layer.weight
        reg = torch.sum((R @ R.T - torch.eye(self.reft_dim, device=R.device)) ** 2)
        return reg
