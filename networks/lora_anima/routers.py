"""Network-level router modules for the LoRA-family facade."""

from typing import Optional

import torch


# Post-LLM-adapter crossattn_emb width. Fixed by the Anima DiT
# (``crossattn_emb_channels = 1024`` in ``library/anima/models.py``) -- the
# T5-compatible cross-attention input dim. Threaded into ContentRouter as
# a hard constant rather than a cfg knob; if Anima ever ships a model with
# a different cross-attn width, surface this through the DiT config and
# update both call sites.
CROSSATTN_EMB_DIM: int = 1024


class GlobalRouter(torch.nn.Module):
    """Single network-level router feeding every routing-aware module.

    Two-layer MLP -> softmax/tau -- same parameterization as FeRA's
    ``SoftFrequencyRouter``. Final layer is zero-init so step-0 gates
    are uniform across experts. Combined with zero-init expert ups (free
    mode) or zero-init ``lambda_layer`` (ortho mode) this guarantees
    dW=0 at the first optimizer step (clean residual baseline).

    Owned by ``LoRANetwork`` when ``cfg.route_per_layer=False`` and
    ``cfg.use_moe_style`` selects an MoE layout. Reads the per-step
    routing signal (FEI simplex / sinusoidal sigma features) supplied by
    the train loop via ``set_fei`` / ``set_sigma``, and broadcasts the
    resulting gates ``(B, E)`` to every routing-aware module's
    ``_routing_weights`` buffer via ``LoRANetwork.set_routing_weights``.

    Exposes ``_last_gates`` / ``_last_input`` for the metrics layer
    (``LoRANetwork.metrics`` and the future FECL handler in task #5);
    both are detached and overwritten per forward.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        hidden_dim: int = 64,
        tau: float = 0.7,
        apply_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(
                f"GlobalRouter: input_dim must be > 0, got {input_dim}"
            )
        if num_experts <= 1:
            raise ValueError(
                f"GlobalRouter: num_experts must be > 1, got {num_experts}"
            )
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.tau = float(tau)
        # Parameterless input LN -- used by the ``crossattn_emb`` source, where
        # the pooled T5-space text vector has a wide per-channel variance
        # budget (the first Linear's effective input scale would otherwise
        # track caption length / padding ratio). Same trick as ContentRouter;
        # ``elementwise_affine=False`` keeps the state_dict free of ln_* keys
        # and the on/off state is deterministic from ``router_source`` so no
        # metadata stamp is needed. No-op for the sigma / FEI sources.
        self.apply_layer_norm = bool(apply_layer_norm)
        self.ln_in: Optional[torch.nn.LayerNorm] = (
            torch.nn.LayerNorm(self.input_dim, elementwise_affine=False)
            if self.apply_layer_norm
            else None
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_experts),
        )
        # Uniform-at-init: zero the output layer so softmax(0/tau) = 1/E.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

        # Per-step diagnostics. Overwritten on every forward; readable by
        # ``LoRANetwork.metrics`` and the FECL loss handler. Detached at
        # write so holding the reference across the step boundary doesn't
        # pin autograd state. ``_last_fei`` is an alias of ``_last_input``
        # under the FEI router source -- wired in ``forward``.
        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None
        self._last_fei: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim). Promote to fp32 for the matmul + softmax --
        # bf16 logits + softmax(tau<1) underflow at low energies. Inference
        # casts the parent LoRANetwork to bf16, which would otherwise drag
        # the router weights along; re-pin to fp32 on first forward so the
        # matmul dtype matches the upcast input.
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
            if self.ln_in is not None:
                self.ln_in.float()
        x32 = x.float()
        # ``crossattn_emb`` source hands a raw ``(B, L, D)`` text tensor; pool
        # to ``(B, D)`` with RMS over the sequence axis (matches ContentRouter
        # / chimera per-Linear pooling). sigma / FEI sources already arrive as
        # ``(B, input_dim)`` and skip this branch.
        if x32.dim() == 3:
            x32 = x32.pow(2).mean(dim=1).sqrt()
        if self.ln_in is not None:
            x32 = self.ln_in(x32)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        # ``_last_input`` is the raw routing-signal tensor that fed this
        # forward -- FEI simplex (router_source="fei") or sinusoidal-sigma
        # features ("sigma"). Aliased as ``_last_fei`` for the FECL handler
        # / plan2 task #5 -- keeps the diagnostic surface stable across
        # router-source variants.
        self._last_input = x32.detach()
        self._last_fei = self._last_input
        return gates


class FreqRouter(torch.nn.Module):
    """ChimeraHydra freq-pool router (one per network).

    Two-layer MLP feeding softmax/tau over the ``K_f`` freq experts. Input is
    ``concat(FEI(z_t), sinusoidal-sigma-features)`` -- both functions of the
    per-step sigma/z_t. The router lives at network top level and broadcasts
    ``pi_f`` to every chimera module's ``_freq_routing_weights`` buffer; the
    broadcast preserves grad_fn so ``dL_denoise/dpi_f`` reaches the router's
    parameters along the same path FeRA's GlobalRouter uses (eq. 6-7, 11).

    Critical: the output layer uses NON-zero init (small N(0, std)). Unlike
    GlobalRouter (which zero-inits to guarantee dW=0 at step 0), a
    zero-init freq router would be a fixed point of the additive
    composition -- the freq pool would receive uniform gates that fail to
    differentiate the experts and the gradient `dL/dW_router` would never
    leave zero. The chimera proposal mandates non-zero output init for
    exactly this reason (see proposal §"Init").

    Per-modality LayerNorm (``apply_layer_norm=True``): when both
    ``fei_dim`` and ``sigma_dim`` are > 0, each modality's slice of the
    concat input is passed through a parameterless ``LayerNorm`` before
    the MLP. The 2-D FEI simplex and the 16/32-D sinusoidal-sigma block have
    different per-channel variance budgets at init (variance contribution
    scales as ``n_channels``), so without LN the higher-dim sigma block can
    fan-in-overpower FEI ~``sigma_dim/fei_dim``x at init. LN is
    intentionally parameterless (``elementwise_affine=False``) -- keeps the
    save/load surface unchanged, no metadata stamp needed for the LN
    weights themselves (only for the on/off flag).
    """

    def __init__(
        self,
        input_dim: int,
        num_freq_experts: int,
        *,
        hidden_dim: int = 32,
        tau: float = 1.0,
        init_std: float = 0.1,
        fei_dim: int = 0,
        sigma_dim: int = 0,
        apply_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"FreqRouter: input_dim must be > 0, got {input_dim}")
        if num_freq_experts <= 1:
            raise ValueError(
                f"FreqRouter: num_freq_experts must be > 1, got {num_freq_experts}"
            )
        self.input_dim = int(input_dim)
        self.num_freq_experts = int(num_freq_experts)
        self.tau = float(tau)
        self.fei_dim = int(fei_dim)
        self.sigma_dim = int(sigma_dim)
        # LN only fires when both modalities are present -- its job is
        # variance balance across the concat, which is a no-op (or worse,
        # destructive on the 2-D simplex) when only one modality is in
        # play. The dim-sum check guards against the rebuild path where
        # fei_dim+sigma_dim wasn't threaded through; in that case LN stays
        # off and the router behaves like the pre-LN build.
        self.apply_layer_norm = bool(apply_layer_norm) and (
            self.fei_dim > 0
            and self.sigma_dim > 0
            and self.fei_dim + self.sigma_dim == self.input_dim
        )
        # SiLU (proposal §Routers): smoother than ReLU on small-input MLPs
        # and consistent with the DiT's own activation choice.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_freq_experts),
        )
        with torch.no_grad():
            # Hidden layer keeps default Linear init. Only the output layer
            # gets the small-std non-zero init that breaks the freq-pool
            # cold-start fixed point -- see class docstring.
            torch.nn.init.normal_(self.net[-1].weight, std=float(init_std))
            torch.nn.init.zeros_(self.net[-1].bias)

        # Parameterless per-modality LN. elementwise_affine=False keeps the
        # state_dict free of ln_* keys, so old (LN-off) checkpoints stay
        # load-compatible -- the on/off semantics are carried by the
        # ``apply_layer_norm`` flag (stamped to metadata), not by tensor
        # presence in the state_dict.
        self.ln_fei: Optional[torch.nn.LayerNorm] = None
        self.ln_sigma: Optional[torch.nn.LayerNorm] = None
        if self.apply_layer_norm:
            self.ln_fei = torch.nn.LayerNorm(self.fei_dim, elementwise_affine=False)
            self.ln_sigma = torch.nn.LayerNorm(self.sigma_dim, elementwise_affine=False)

        # Per-step diagnostics, parallel to GlobalRouter.
        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # See GlobalRouter.forward -- fp32 compute is load-bearing for the
        # softmax(logits / tau) precision at small tau. Inference casts the
        # parent LoRANetwork to bf16; re-pin router weights to fp32.
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
        x32 = x.float()
        if self.apply_layer_norm:
            fei_part = self.ln_fei(x32[..., : self.fei_dim])
            sigma_part = self.ln_sigma(
                x32[..., self.fei_dim : self.fei_dim + self.sigma_dim]
            )
            x32 = torch.cat([fei_part, sigma_part], dim=-1)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        self._last_input = x32.detach()
        return gates


class ContentRouter(torch.nn.Module):
    """ChimeraHydra content-pool router, network-level variant (one per network).

    Same MLP shape as FreqRouter -- ``Linear -> SiLU -> Linear -> softmax/tau`` --
    but the input is a pooled ``crossattn_emb`` (per-sample text features,
    the same vector flowing into the DiT's cross-attention). Output ``pi_c``
    is broadcast to every chimera module's ``_content_routing_weights``
    buffer (slot-assign, grad_fn preserved) and replaces the per-Linear
    softmax over pooled ``lx_c``.

    Built only when ``cfg.content_router_source != "input"``. The per-Linear
    ``self.router`` is then skipped at construction time on each chimera
    module -- the content pool sees only this network-level gate.

    Init rationale: same as FreqRouter (small non-zero output init via
    ``init_std``). Uniform gates would be a fixed point under the additive
    pool composition -- ``dL/dW_router`` would never leave zero. The freq
    router's ``0.1`` default is the cell that already works in this stack.
    """

    def __init__(
        self,
        input_dim: int,
        num_content_experts: int,
        *,
        hidden_dim: int = 64,
        tau: float = 1.0,
        init_std: float = 0.1,
        apply_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"ContentRouter: input_dim must be > 0, got {input_dim}")
        if num_content_experts <= 1:
            raise ValueError(
                f"ContentRouter: num_content_experts must be > 1, got {num_content_experts}"
            )
        self.input_dim = int(input_dim)
        self.num_content_experts = int(num_content_experts)
        self.tau = float(tau)
        self.apply_layer_norm = bool(apply_layer_norm)
        # Parameterless LN on the pooled cross-attn vector. Pooled T5-space
        # features have a wide per-channel variance budget -- without LN the
        # first Linear's effective input scale tracks caption length /
        # padding ratio. ``elementwise_affine=False`` keeps the state_dict
        # free of ln_* keys (same trick as FreqRouter).
        self.ln_in: Optional[torch.nn.LayerNorm] = (
            torch.nn.LayerNorm(self.input_dim, elementwise_affine=False)
            if self.apply_layer_norm
            else None
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_content_experts),
        )
        with torch.no_grad():
            torch.nn.init.normal_(self.net[-1].weight, std=float(init_std))
            torch.nn.init.zeros_(self.net[-1].bias)

        # Per-step diagnostics, parallel to GlobalRouter / FreqRouter.
        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same fp32 pin as GlobalRouter / FreqRouter -- softmax/tau at small tau
        # underflows in bf16. Caller may pass an already-pooled (B, D) tensor
        # or a raw (B, L, D) crossattn_emb; pool to (B, D) here so the
        # network entry point can stay shape-agnostic.
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
            if self.ln_in is not None:
                self.ln_in.float()
        x32 = x.float()
        if x32.dim() == 3:
            x32 = x32.pow(2).mean(dim=1).sqrt()
        if self.ln_in is not None:
            x32 = self.ln_in(x32)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        self._last_input = x32.detach()
        return gates
