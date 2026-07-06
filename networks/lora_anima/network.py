# LoRANetwork: the module-assembly / training-orchestration core of the LoRA
# adapter stack for Anima. Targets DiT blocks (and optionally text-encoder
# attention) with pluggable per-module classes supplied by a NetworkSpec.

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import torch

from library.log import setup_logging
from library.training.metrics import MetricContext
from networks import ModuleCreationContext, NETWORK_REGISTRY
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import _parse_reft_layers
from networks.lora_anima.persistence import (
    load_lora_network_weights,
    reabsorb_baked_inv_scale,
    save_lora_network_weights,
    strip_orig_mod_keys,
)
from networks.lora_anima import optimizer_groups, router_stats
from networks.lora_anima.targeting import (
    collect_lora_target_candidates,
    compile_lora_target_patterns,
)
from networks.register_injection import RegisterInjector
from networks.lora_modules import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
    ReFTModule,
    StackedExpertsLoRAModule,
    _sigma_sinusoidal_features,
)

setup_logging()
logger = logging.getLogger(__name__)

# Post-LLM-adapter crossattn_emb width. Fixed by the Anima DiT
# (``crossattn_emb_channels = 1024`` in ``library/anima/models.py``) — the
# T5-compatible cross-attention input dim. Threaded into ContentRouter as
# a hard constant rather than a cfg knob; if Anima ever ships a model with
# a different cross-attn width, surface this through the DiT config and
# update both call sites.
CROSSATTN_EMB_DIM: int = 1024


class GlobalRouter(torch.nn.Module):
    """Single network-level router feeding every routing-aware module.

    Two-layer MLP → softmax/τ — same parameterization as FeRA's
    ``SoftFrequencyRouter``. Final layer is zero-init so step-0 gates
    are uniform across experts. Combined with zero-init expert ups (free
    mode) or zero-init ``lambda_layer`` (ortho mode) this guarantees
    ΔW=0 at the first optimizer step (clean residual baseline).

    Owned by ``LoRANetwork`` when ``cfg.route_per_layer=False`` and
    ``cfg.use_moe_style`` selects an MoE layout. Reads the per-step
    routing signal (FEI simplex / sinusoidal σ features) supplied by
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
        # Parameterless input LN — used by the ``crossattn_emb`` source, where
        # the pooled T5-space text vector has a wide per-channel variance
        # budget (the first Linear's effective input scale would otherwise
        # track caption length / padding ratio). Same trick as ContentRouter;
        # ``elementwise_affine=False`` keeps the state_dict free of ln_* keys
        # and the on/off state is deterministic from ``router_source`` so no
        # metadata stamp is needed. No-op for the σ / FEI sources.
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
        # Uniform-at-init: zero the output layer so softmax(0/τ) = 1/E.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

        # Per-step diagnostics. Overwritten on every forward; readable by
        # ``LoRANetwork.metrics`` and the FECL loss handler. Detached at
        # write so holding the reference across the step boundary doesn't
        # pin autograd state. ``_last_fei`` is an alias of ``_last_input``
        # under the FEI router source — wired in ``forward``.
        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None
        self._last_fei: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim). Promote to fp32 for the matmul + softmax —
        # bf16 logits + softmax(τ<1) underflow at low energies. Inference
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
        # / chimera per-Linear pooling). σ / FEI sources already arrive as
        # ``(B, input_dim)`` and skip this branch.
        if x32.dim() == 3:
            x32 = x32.pow(2).mean(dim=1).sqrt()
        if self.ln_in is not None:
            x32 = self.ln_in(x32)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        # ``_last_input`` is the raw routing-signal tensor that fed this
        # forward — FEI simplex (router_source="fei") or sinusoidal-σ
        # features ("sigma"). Aliased as ``_last_fei`` for the FECL handler
        # / plan2 task #5 — keeps the diagnostic surface stable across
        # router-source variants.
        self._last_input = x32.detach()
        self._last_fei = self._last_input
        return gates


class FreqRouter(torch.nn.Module):
    """ChimeraHydra freq-pool router (one per network).

    Two-layer MLP feeding softmax/τ over the ``K_f`` freq experts. Input is
    ``concat(FEI(z_t), sinusoidal-σ-features)`` — both functions of the
    per-step σ/z_t. The router lives at network top level and broadcasts
    ``π_f`` to every chimera module's ``_freq_routing_weights`` buffer; the
    broadcast preserves grad_fn so ``∂L_denoise/∂π_f`` reaches the router's
    parameters along the same path FeRA's GlobalRouter uses (eq. 6-7, 11).

    Critical: the output layer uses NON-zero init (small N(0, std)). Unlike
    GlobalRouter (which zero-inits to guarantee ΔW=0 at step 0), a
    zero-init freq router would be a fixed point of the additive
    composition — the freq pool would receive uniform gates that fail to
    differentiate the experts and the gradient `∂L/∂W_router` would never
    leave zero. The chimera proposal mandates non-zero output init for
    exactly this reason (see proposal §"Init").

    Per-modality LayerNorm (``apply_layer_norm=True``): when both
    ``fei_dim`` and ``sigma_dim`` are > 0, each modality's slice of the
    concat input is passed through a parameterless ``LayerNorm`` before
    the MLP. The 2-D FEI simplex and the 16/32-D sinusoidal-σ block have
    different per-channel variance budgets at init (variance contribution
    scales as ``n_channels``), so without LN the higher-dim σ block can
    fan-in-overpower FEI ~``sigma_dim/fei_dim``× at init. LN is
    intentionally parameterless (``elementwise_affine=False``) — keeps the
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
        # LN only fires when both modalities are present — its job is
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
            # cold-start fixed point — see class docstring.
            torch.nn.init.normal_(self.net[-1].weight, std=float(init_std))
            torch.nn.init.zeros_(self.net[-1].bias)

        # Parameterless per-modality LN. elementwise_affine=False keeps the
        # state_dict free of ln_* keys, so old (LN-off) checkpoints stay
        # load-compatible — the on/off semantics are carried by the
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
        # See GlobalRouter.forward — fp32 compute is load-bearing for the
        # softmax(logits / τ) precision at small τ. Inference casts the
        # parent LoRANetwork to bf16; re-pin router weights to fp32.
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
        x32 = x.float()
        if self.apply_layer_norm:
            fei_part = self.ln_fei(x32[..., : self.fei_dim])
            sigma_part = self.ln_sigma(x32[..., self.fei_dim : self.fei_dim + self.sigma_dim])
            x32 = torch.cat([fei_part, sigma_part], dim=-1)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        self._last_input = x32.detach()
        return gates


class ContentRouter(torch.nn.Module):
    """ChimeraHydra content-pool router, network-level variant (one per network).

    Same MLP shape as FreqRouter — ``Linear → SiLU → Linear → softmax/τ`` —
    but the input is a pooled ``crossattn_emb`` (per-sample text features,
    the same vector flowing into the DiT's cross-attention). Output ``π_c``
    is broadcast to every chimera module's ``_content_routing_weights``
    buffer (slot-assign, grad_fn preserved) and replaces the per-Linear
    softmax over pooled ``lx_c``.

    Built only when ``cfg.content_router_source != "input"``. The per-Linear
    ``self.router`` is then skipped at construction time on each chimera
    module — the content pool sees only this network-level gate.

    Init rationale: same as FreqRouter (small non-zero output init via
    ``init_std``). Uniform gates would be a fixed point under the additive
    pool composition — ``∂L/∂W_router`` would never leave zero. The freq
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
        # features have a wide per-channel variance budget — without LN the
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
        # Same fp32 pin as GlobalRouter / FreqRouter — softmax/τ at small τ
        # underflows in bf16. Caller may pass an already-pooled (B, D) tensor
        # or a raw (B, L, D) crossattn_emb; pool to (B, D) here so the
        # network entry point can stay shape-agnostic.
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
            if self.ln_in is not None:
                self.ln_in.float()
        x32 = x.float()
        if x32.dim() == 3:
            x32 = x32.pow(2).mean(dim=1).sqrt()  # RMS over seq, matches chimera per-Linear pool
        if self.ln_in is not None:
            x32 = self.ln_in(x32)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        self._last_input = x32.detach()
        return gates


class LoRANetwork(torch.nn.Module):
    # Target modules: DiT blocks, embedders, final layer. embedders and final layer are excluded by default.
    ANIMA_TARGET_REPLACE_MODULE = [
        "Block",
        "PatchEmbed",
        "TimestepEmbedding",
        "FinalLayer",
    ]
    # Target modules: LLM Adapter blocks
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    # Target modules for text encoder (Qwen3)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Qwen3Attention",
        "Qwen3MLP",
        "Qwen3SdpaAttention",
        "Qwen3FlashAttention2",
    ]

    LORA_PREFIX_ANIMA = "lora_unet"  # ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER = "lora_te"  # Qwen3

    def __init__(
        self,
        text_encoders: list,
        unet,
        cfg: LoRANetworkCfg,
        *,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Mutable runtime state — explicitly NOT in cfg. ``set_multiplier`` and
        # ``set_loraplus_lr_ratio`` write these post-construction; per-step
        # diagnostics (hit counters, σ caches) accumulate during training.
        self.multiplier = multiplier
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None
        self._channel_scale_misses: List[str] = []
        self._channel_scale_hits: int = 0
        self._sigma_router_hits: int = 0
        self._hydra_router_hits: int = 0
        self._hydra_router_misses: int = 0
        self._last_sigma: Optional[torch.Tensor] = None
        # Hydra up-weight grad-norm snapshot (T-LoRA / σ-bucket conflict
        # diagnostic). Filled by ``capture_up_grad_stats`` between backward
        # and ``optimizer.zero_grad``; consumed by the ``hydra_up_grad``
        # metric. Values stay on-device until ``get_up_grad_stats`` runs the
        # D2H — capture happens every sync step but the metric only reads on
        # log steps, so the sync was the per-step bottleneck.
        self._last_up_grad_stats: Dict[str, object] = {}
        # Per-step cache for ``get_router_stats`` — both the progress-bar
        # postfix and the metrics layer call it on log steps. Cleared in
        # ``clear_step_caches`` so the next forward recomputes.
        self._router_stats_cache: Optional[Dict[str, object]] = None
        # Separate cache for the chimera dual-pool router stats — different
        # reduction (mean gates per pool, not argmax-histogram) and different
        # entropy normalization (per-pool log(K_pool)). Same lifecycle.
        self._chimera_router_stats_cache: Optional[Dict[str, object]] = None

        # Local aliases for the closure body and the post-closure ReFT block.
        # Reading via `cfg.foo` works too; aliases just keep the diff small.
        module_class = cfg.module_class
        nominal_spec = next(
            (
                spec
                for spec in NETWORK_REGISTRY.values()
                if spec.module_class is module_class
            ),
            None,
        )
        modules_dim = cfg.modules_dim
        modules_alpha = cfg.modules_alpha
        dropout = cfg.dropout
        rank_dropout = cfg.rank_dropout
        module_dropout = cfg.module_dropout
        verbose = cfg.verbose
        alpha = cfg.alpha
        lora_dim = cfg.lora_dim
        train_llm_adapter = cfg.train_llm_adapter
        add_reft = cfg.add_reft
        reft_dim = cfg.reft_dim
        reft_alpha = cfg.reft_alpha
        reft_layers = cfg.reft_layers

        # Unified routing scope. ``cfg.router_targets`` is the single regex
        # that governs which Linears participate in routed adaptation (Hydra
        # MoE leaves + σ-feature concat + FEI-feature concat all share it).
        # From-weights path supplies an explicit name set per router family
        # (different families may have different module memberships in older
        # checkpoints); when present, the explicit set wins over the regex.
        _router_re = (
            re.compile(cfg.router_targets) if cfg.router_targets else None
        )

        self._sigma_router_names = (
            set(cfg.sigma_router_names) if cfg.sigma_router_names else None
        )
        self._sigma_router_re = (
            _router_re
            if (
                cfg.router_source == "sigma"
                and _router_re is not None
                and self._sigma_router_names is None
            )
            else None
        )

        self._fei_router_names = (
            set(cfg.fei_router_names) if cfg.fei_router_names else None
        )
        self._fei_router_re = (
            _router_re
            if (
                cfg.router_source == "fei"
                and _router_re is not None
                and self._fei_router_names is None
            )
            else None
        )
        self._fei_router_hits = 0
        # Modules built with ``use_global_router=True`` (shared_A +
        # ``route_per_layer=False``): the per-layer router is skipped and gates
        # arrive via the network-level ``GlobalRouter``. Counted separately
        # from ``_fei_router_hits`` because the per-layer FEI cat is bypassed.
        self._global_router_hits = 0
        # Retained as a network attr (library/inference/adapters.py reads it
        # via getattr); derived from cfg.router_source.
        self.use_fei_router = cfg.router_source == "fei"
        self.use_sigma_router = cfg.router_source == "sigma"
        # Shared-A Hydra layout + network-level router (FEI-on-Hydra global).
        # Toggle for the per-module construction loop below; lets Hydra /
        # OrthoHydra modules skip ``self.router`` and consume gates from the
        # ``GlobalRouter`` instead. Mirrors the FeRA (independent_A) routing
        # location without changing the underlying Hydra parameter layout.
        self._use_global_router_for_hydra = (
            cfg.use_moe_style == "shared_A"
            and not cfg.route_per_layer
            and cfg.router_source != "none"
        )

        # Per-module HydraLoRA gating. Matching modules get the Hydra class;
        # non-matching modules fall back to plain LoRA / OrthoLoRA so MoE
        # capacity is concentrated where specialization is actually learnable.
        # Fresh path: regex over `original_name`. From-weights path: explicit
        # name set detected from checkpoint keys. Explicit set wins. None on
        # both = apply MoE everywhere (legacy).
        self._hydra_router_names = (
            set(cfg.hydra_router_names) if cfg.hydra_router_names else None
        )
        self._hydra_router_re = (
            _router_re if (_router_re is not None and self._hydra_router_names is None)
            else None
        )

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(
                f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}"
            )
            logger.info(
                f"neuron dropout: p={dropout}, rank dropout: p={rank_dropout}, module dropout: p={module_dropout}"
            )

        exclude_re_patterns = compile_lora_target_patterns(
            cfg.exclude_patterns,
            logger=logger,
        )
        include_re_patterns = compile_lora_target_patterns(
            cfg.include_patterns,
            logger=logger,
        )

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            default_dim: Optional[int] = None,
        ) -> Tuple[List[LoRAModule], List[str]]:
            prefix = (
                self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER
            )

            # First pass: collect candidate modules. Class selection, router
            # counters, and constructor kwargs stay below in the second pass.
            candidates = collect_lora_target_candidates(
                root_module=root_module,
                prefix=prefix,
                target_replace_modules=target_replace_modules,
                exclude_patterns=exclude_re_patterns,
                include_patterns=include_re_patterns,
                is_unet=is_unet,
                layer_start=cfg.layer_start,
                layer_end=cfg.layer_end,
                modules_dim=modules_dim,
                modules_alpha=modules_alpha,
                reg_dims=cfg.reg_dims,
                default_dim=default_dim,
                lora_dim=lora_dim,
                alpha=alpha,
                verbose=verbose,
                logger=logger,
            )

            # Second pass: create LoRA modules with progress bar
            from tqdm import tqdm

            loras = []
            skipped = []
            non_skipped = [
                (item.lora_name, item.child_module, item.dim, item.alpha, item.original_name)
                for item in candidates
                if not item.skipped
            ]
            skipped = [item.lora_name for item in candidates if item.skipped]

            # VeRA uses one frozen random projection bank (A/B) and slices it
            # per adapted Linear.  The plugin owns the module implementation;
            # the network builder only supplies the largest shape visible in
            # this create scope so all VeRA leaves share one deterministic
            # bank instead of allocating layer-local random matrices.
            if nominal_spec is not None and nominal_spec.name == "vera":
                linear_candidates = [
                    cm for _ln, cm, _d, _a, _on in non_skipped
                    if isinstance(cm, torch.nn.Linear)
                ]
                if linear_candidates:
                    cfg.plugin_args["_vera_max_in_features"] = max(
                        int(m.in_features) for m in linear_candidates
                    )
                    cfg.plugin_args["_vera_max_out_features"] = max(
                        int(m.out_features) for m in linear_candidates
                    )

            label = (
                "DiT"
                if is_unet
                else f"TE{text_encoder_idx + 1}"
                if text_encoder_idx is not None
                else "model"
            )
            for lora_name, child_module, dim, alpha_val, original_name in tqdm(
                non_skipped, desc=f"Creating {label} LoRA", leave=False
            ):
                # Per-module class resolution: when the network's nominal class
                # is Hydra (MoE), narrow it to only the layers in the hydra
                # filter. Non-matching layers fall back to plain LoRA /
                # OrthoLoRA so router overhead + balance-loss pressure are
                # concentrated on sites where specialization is learnable.
                effective_module_class = module_class
                if (
                    module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                        ChimeraHydraLoRAModule,
                        ChimeraHydraInferenceModule,
                    )
                    and is_unet
                ):
                    if self._hydra_router_names is not None:
                        hydra_on = lora_name in self._hydra_router_names
                    elif self._hydra_router_re is not None:
                        hydra_on = bool(self._hydra_router_re.search(original_name))
                    else:
                        hydra_on = True
                    if hydra_on:
                        self._hydra_router_hits += 1
                    else:
                        self._hydra_router_misses += 1
                        if module_class is HydraLoRAModule:
                            effective_module_class = LoRAModule
                        elif module_class is ChimeraHydraInferenceModule:
                            # Load path. Unrouted leg was saved as plain LoRA
                            # (OrthoLoRA distilled to ``.lora_down.weight`` +
                            # ``.lora_up.weight`` at save time — see
                            # ``_convert_ortho_to_lora``).
                            effective_module_class = LoRAModule
                        else:
                            # Train path (ChimeraHydraLoRAModule) and
                            # OrthoHydra: unrouted leg uses the OrthoLoRA
                            # Cayley parameterization.
                            effective_module_class = OrthoLoRAModule

                extra_kwargs = {}
                if effective_module_class == OrthoLoRAModule:
                    pass  # no extra kwargs — SVD init reads from org_module directly
                elif effective_module_class == ChimeraHydraLoRAModule:
                    # Pool split is the chimera's only constructor surface;
                    # σ/FEI feature dims are 0 by design (the network-level
                    # FreqRouter owns those axes — see chimera.py module
                    # docstring). The pool sum must equal cfg.num_experts
                    # by ``LoRANetworkCfg.from_kwargs`` invariant.
                    extra_kwargs["num_experts_content"] = cfg.num_experts_content
                    extra_kwargs["num_experts_freq"] = cfg.num_experts_freq
                    extra_kwargs["centered_gate"] = cfg.chimera_centered_gate
                    extra_kwargs["lambda_init"] = cfg.chimera_lambda_init
                    if cfg.content_router_source == "crossattn_emb":
                        extra_kwargs["use_global_content_router"] = True
                elif effective_module_class == ChimeraHydraInferenceModule:
                    # Inference (free-form) twin of the chimera training
                    # class. Same constructor surface — both pool sizes
                    # arrive from the chimera-stamped metadata via
                    # ``cfg.from_weights``.
                    extra_kwargs["num_experts_content"] = cfg.num_experts_content
                    extra_kwargs["num_experts_freq"] = cfg.num_experts_freq
                    extra_kwargs["centered_gate"] = cfg.chimera_centered_gate
                    if cfg.content_router_source == "crossattn_emb":
                        extra_kwargs["use_global_content_router"] = True
                elif effective_module_class == OrthoHydraLoRAModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                    extra_kwargs["centered_gate"] = cfg.ortho_centered_gate
                    extra_kwargs["lambda_init"] = cfg.ortho_lambda_init
                    if self._use_global_router_for_hydra:
                        extra_kwargs["use_global_router"] = True
                        self._global_router_hits += 1
                elif effective_module_class == HydraLoRAModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                    extra_kwargs["centered_gate"] = cfg.ortho_centered_gate
                    if cfg.expert_init_std > 0.0:
                        extra_kwargs["expert_init_std"] = cfg.expert_init_std
                    if self._use_global_router_for_hydra:
                        extra_kwargs["use_global_router"] = True
                        self._global_router_hits += 1
                    if cfg.use_chimera_hydra:
                        # Dual-pool runtime form (load path from a distilled
                        # chimera checkpoint — see factory.py is_chimera_hydra
                        # branch). HydraLoRAModule narrows its router to K_c
                        # outputs and registers _freq_routing_weights for the
                        # network-level FreqRouter broadcast. σ/FEI feature
                        # dims must stay 0 here — FreqRouter owns those axes.
                        extra_kwargs["num_experts_content"] = cfg.num_experts_content
                        if cfg.content_router_source == "crossattn_emb":
                            extra_kwargs["use_global_content_router"] = True
                elif effective_module_class == StackedExpertsLoRAModule:
                    # Independent-A (FeRA). Gates arrive via the network-level
                    # ``GlobalRouter`` through the shared ``_routing_weights``
                    # buffer — no per-Linear router knob to set. ``num_experts``
                    # must match ``cfg.num_experts`` (and therefore the
                    # GlobalRouter's output width) or the routing-weight
                    # broadcast inside ``forward`` shape-mismatches.
                    extra_kwargs["num_experts"] = cfg.num_experts
                    extra_kwargs["ortho"] = cfg.use_ortho
                    if cfg.use_ortho:
                        extra_kwargs["ortho_init_std"] = cfg.ortho_init_std

                if cfg.down_init != "kaiming" and effective_module_class is LoRAModule:
                    extra_kwargs["down_init"] = cfg.down_init

                effective_spec = (
                    nominal_spec
                    if nominal_spec is not None
                    and effective_module_class is nominal_spec.module_class
                    else next(
                        (
                            spec
                            for spec in NETWORK_REGISTRY.values()
                            if spec.module_class is effective_module_class
                        ),
                        None,
                    )
                )
                if (
                    effective_spec is not None
                    and effective_spec.module_kwargs is not None
                ):
                    extra_kwargs.update(
                        effective_spec.module_kwargs(
                            ModuleCreationContext(
                                cfg=cfg,
                                is_unet=is_unet,
                                lora_name=lora_name,
                                original_name=original_name,
                                child_module=child_module,
                                module_class=effective_module_class,
                            )
                        )
                    )

                # Hard σ-band expert partition: applied to every Hydra/
                # OrthoHydra module (independent of the σ-feature router
                # regex). Each module owns the partition; the network-level
                # ``set_sigma`` propagates ``_sigma`` to enable per-step band
                # selection. Validation (E % N == 0) lives in cfg parsing.
                if (
                    cfg.specialize_experts_by_sigma_buckets
                    and effective_module_class
                    in (HydraLoRAModule, OrthoHydraLoRAModule)
                    and is_unet
                ):
                    extra_kwargs["specialize_experts_by_sigma_buckets"] = True
                    extra_kwargs["num_sigma_buckets"] = cfg.num_sigma_buckets
                    if cfg.sigma_bucket_boundaries is not None:
                        extra_kwargs["sigma_bucket_boundaries"] = (
                            cfg.sigma_bucket_boundaries
                        )

                # σ-conditional router: only widen the router input with
                # sinusoidal(σ) features on modules whose name matches the
                # layer filter (cross_attn.q / self_attn.qkv by default — see
                # B0 pre-analysis in timestep-hydra.md). From-weights path uses
                # an explicit name set; fresh-from-kwargs path uses a regex
                # over original_name. Gated on the effective class so a
                # hydra-excluded module can't pick up σ either. Skipped under
                # ``use_global_router`` — the network-level router consumes
                # the routing signal once and the per-Linear cat is dead.
                if (
                    cfg.router_source == "sigma"
                    and effective_module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                    )
                    and is_unet
                    and not self._use_global_router_for_hydra
                ):
                    if self._sigma_router_names is not None:
                        enable = lora_name in self._sigma_router_names
                    elif self._sigma_router_re is not None:
                        enable = bool(self._sigma_router_re.search(original_name))
                    else:
                        enable = True
                    if enable:
                        extra_kwargs["sigma_feature_dim"] = cfg.sigma_feature_dim
                        self._sigma_router_hits += 1

                # FEI-conditional router (FeRA-style). Same gating as σ —
                # widen the router input with the per-sample FEI simplex on
                # modules whose name matches the layer filter. The FEI tensor
                # itself is computed once per step in the train/inference loop
                # and propagated via ``LoRANetwork.set_fei``. Skipped under
                # ``use_global_router`` — the GlobalRouter reads FEI directly
                # at the network level and per-Linear cat is dead.
                if (
                    cfg.router_source == "fei"
                    and effective_module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                    )
                    and is_unet
                    and not self._use_global_router_for_hydra
                ):
                    if self._fei_router_names is not None:
                        enable_fei = lora_name in self._fei_router_names
                    elif self._fei_router_re is not None:
                        enable_fei = bool(self._fei_router_re.search(original_name))
                    else:
                        enable_fei = True
                    if enable_fei:
                        extra_kwargs["fei_feature_dim"] = cfg.fei_feature_dim
                        self._fei_router_hits += 1

                # Per-channel scaling is DiT-only. LoKr is excluded because a
                # full input-channel scale cannot be represented by its
                # Kronecker factors or native LoKr checkpoint format.
                if (
                    cfg.channel_scales_dict is not None
                    and is_unet
                    and not (
                        effective_spec is not None and effective_spec.name == "lokr"
                    )
                ):
                    _cs = cfg.channel_scales_dict.get(lora_name)
                    if _cs is not None:
                        extra_kwargs["channel_scale"] = _cs
                        self._channel_scale_hits += 1
                    else:
                        self._channel_scale_misses.append(lora_name)

                lora = effective_module_class(
                    lora_name,
                    child_module,
                    self.multiplier,
                    dim,
                    alpha_val,
                    dropout=dropout,
                    rank_dropout=rank_dropout,
                    module_dropout=module_dropout,
                    **extra_kwargs,
                )
                lora.fp32_compute = bool(cfg.lora_fp32_compute)
                lora.original_name = original_name
                loras.append(lora)

            return loras, skipped

        # Create LoRA for text encoders (Qwen3 - typically not trained for Anima)
        # Skip for OrthoLoRA since SVD init is expensive and TE modules are discarded in apply_to anyway
        self.text_encoder_loras: List[LoRAModule] = []
        skipped_te = []
        if text_encoders is not None and module_class not in (
            OrthoLoRAModule,
            OrthoHydraLoRAModule,
            ChimeraHydraLoRAModule,
            ChimeraHydraInferenceModule,
        ):
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create LoRA for Text Encoder {i + 1}:")
                te_loras, te_skipped = create_modules(
                    False,
                    i,
                    text_encoder,
                    LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                )
                logger.info(
                    f"create LoRA for Text Encoder {i + 1}: {len(te_loras)} modules."
                )
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create LoRA for DiT blocks
        target_modules = list(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras: List[LoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)

        logger.info(f"create LoRA for Anima DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} LoRA modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        if cfg.channel_scales_dict is not None:
            logger.info(
                f"channel_scaling: {self._channel_scale_hits} DiT modules "
                f"received calibration-based input scaling"
            )
            if self._channel_scale_misses:
                logger.warning(
                    f"channel_scaling: {len(self._channel_scale_misses)} DiT modules "
                    f"have no calibration stats (first: {self._channel_scale_misses[:3]}). "
                    f"These will train without input rebalancing — regenerate the vendored "
                    f"calibration with `python bench/channel_stats/analyze_lora_input_channels.py "
                    f"--per_artist --dump_channel_stats networks/calibration/channel_stats.safetensors` "
                    f"if this is unexpected."
                )

        # Create ReFT modules on the DiT residual stream (block outputs), following
        # Wu et al. (2024) §3.3 — one intervention per selected block, not per
        # internal Linear. Selection is controlled by ``reft_layers``.
        self.unet_refts: List[ReFTModule] = []
        self.text_encoder_refts: List[ReFTModule] = []
        if add_reft:
            dit_blocks = getattr(unet, "blocks", None)
            if dit_blocks is None or len(dit_blocks) == 0:
                raise ValueError(
                    "add_reft=True but DiT has no .blocks attribute to wrap. "
                    "Block-level ReFT requires a transformer with a `blocks` ModuleList."
                )
            num_blocks = len(dit_blocks)
            selected_indices = _parse_reft_layers(reft_layers, num_blocks)

            reft_alpha_value = reft_alpha if reft_alpha is not None else alpha
            for idx in selected_indices:
                block = dit_blocks[idx]
                block_embed_dim = getattr(block, "x_dim", None)
                if block_embed_dim is None:
                    raise ValueError(
                        f"Block {idx} ({type(block).__name__}) has no `x_dim`; "
                        "cannot infer embed_dim for ReFT."
                    )
                reft_name = f"reft_unet_blocks_{idx}"
                reft = ReFTModule(
                    reft_name,
                    block,
                    embed_dim=block_embed_dim,
                    multiplier=multiplier,
                    reft_dim=reft_dim,
                    alpha=reft_alpha_value,
                    dropout=dropout,
                    module_dropout=module_dropout,
                )
                reft.original_name = f"blocks.{idx}"
                self.unet_refts.append(reft)
            logger.info(
                f"create ReFT for Anima DiT: {len(self.unet_refts)}/{num_blocks} "
                f"blocks (reft_dim={reft_dim}, layers={reft_layers!r})"
            )

        # assertion: no duplicate names
        names = set()
        for lora in (
            self.text_encoder_loras
            + self.unet_loras
            + self.text_encoder_refts
            + self.unet_refts
        ):
            assert lora.lora_name not in names, (
                f"duplicated lora name: {lora.lora_name}"
            )
            names.add(lora.lora_name)

        # Alias each sigma-aware module's ``_sigma`` / ``_sigma_features``
        # buffer to a single network-level shared tensor. ``set_sigma`` then
        # updates the shared tensor in place once and every aliased module
        # buffer sees the new value through shared storage — instead of
        # ~56 per-module ``copy_`` calls per training step.
        self._wire_shared_sigma_buffers()
        self._wire_shared_fei_buffers()
        self._wire_shared_routing_buffers()
        self._wire_shared_freq_routing_buffers()
        self._wire_shared_content_routing_buffers()

        # Build the network-level GlobalRouter when the cfg selects MoE
        # without per-Linear routers. The input dim is derived from the
        # routing signal: ``"fei"`` → ``fei_feature_dim`` simplex,
        # ``"sigma"`` → ``sigma_feature_dim`` sinusoidal features.
        # Routing-aware modules: ``independent_A`` (StackedExperts) always
        # consume the broadcast gates; ``shared_A`` (Hydra / OrthoHydra)
        # consumes them when built with ``use_global_router=True``.
        self.global_router: Optional[GlobalRouter] = None
        # ``use_crossattn_router`` advertises to the train / inference call
        # sites that they must fire ``set_crossattn_routing`` with the pooled
        # text tensor each forward (parallel to chimera's ``use_content_router``
        # but broadcasting to the standard ``_routing_weights`` slot).
        self.use_crossattn_router: bool = False
        if cfg.use_moe_style is not False and not cfg.route_per_layer:
            router_layer_norm = False
            if cfg.router_source == "fei":
                router_input_dim = int(cfg.fei_feature_dim)
            elif cfg.router_source == "sigma":
                router_input_dim = int(cfg.sigma_feature_dim)
            elif cfg.router_source == "crossattn_emb":
                # Pooled post-LLM-adapter text feature (the DiT's cross-attn
                # K/V). LN on by default — wide T5-space variance budget.
                router_input_dim = CROSSATTN_EMB_DIM
                router_layer_norm = True
            else:
                router_input_dim = 0
            if router_input_dim > 0 and cfg.num_experts > 1:
                self.global_router = GlobalRouter(
                    input_dim=router_input_dim,
                    num_experts=int(cfg.num_experts),
                    hidden_dim=int(cfg.router_hidden_dim),
                    tau=float(cfg.router_tau),
                    apply_layer_norm=router_layer_norm,
                )
                self.use_crossattn_router = cfg.router_source == "crossattn_emb"
                logger.info(
                    f"GlobalRouter: source={cfg.router_source!r}, "
                    f"input_dim={router_input_dim}, "
                    f"num_experts={cfg.num_experts}, "
                    f"hidden={cfg.router_hidden_dim}, τ={cfg.router_tau:.2f}, "
                    f"LN={router_layer_norm}, "
                    f"routing-aware modules={len(self._routing_aware_loras)}"
                )

        # ChimeraHydra FreqRouter: one per network, broadcasts ``π_f`` over
        # the freq pool of every chimera module. Input is
        # ``concat(FEI, sinusoidal-σ-features)`` — owned by the freq router
        # exclusively (the per-layer content router never sees σ/FEI). Built
        # only when at least one chimera module was actually constructed; the
        # router_targets regex can narrow the chimera class to a subset of
        # layers (others fall back to OrthoLoRA).
        self.freq_router: Optional[FreqRouter] = None
        if cfg.use_chimera_hydra and self._chimera_aware_loras:
            freq_input_dim = int(cfg.fei_feature_dim) + int(cfg.sigma_feature_dim)
            if freq_input_dim <= 0:
                raise ValueError(
                    "use_chimera_hydra=True requires fei_feature_dim + "
                    f"sigma_feature_dim > 0 for the FreqRouter input (got "
                    f"FEI={cfg.fei_feature_dim}, σ={cfg.sigma_feature_dim})."
                )
            self.freq_router = FreqRouter(
                input_dim=freq_input_dim,
                num_freq_experts=int(cfg.num_experts_freq),
                hidden_dim=int(cfg.router_hidden_dim),
                tau=float(cfg.router_tau),
                init_std=float(cfg.freq_router_init_std),
                fei_dim=int(cfg.fei_feature_dim),
                sigma_dim=int(cfg.sigma_feature_dim),
                apply_layer_norm=bool(cfg.freq_router_layer_norm),
            )
            # Force the per-step conditioning hook to fire set_fei every
            # step (router_conditioning.py reads this flag). Chimera ties
            # σ + FEI together for the freq router input, so the set_fei
            # path is where we re-fire FreqRouter.
            self.use_fei_router = True
            logger.info(
                f"ChimeraHydra FreqRouter: input_dim={freq_input_dim} "
                f"(FEI={cfg.fei_feature_dim} + σ={cfg.sigma_feature_dim}), "
                f"K_f={cfg.num_experts_freq}, hidden={cfg.router_hidden_dim}, "
                f"τ={cfg.router_tau:.2f}, init_std={cfg.freq_router_init_std}, "
                f"LN={self.freq_router.apply_layer_norm}, "
                f"chimera modules={len(self._chimera_aware_loras)}"
            )

        # ChimeraHydra ContentRouter: network-level twin of FreqRouter for
        # the content pool. Built only when ``content_router_source ==
        # "crossattn_emb"`` AND at least one chimera module exists. Per-Linear
        # ``self.router`` is None on those modules in that case — π_c flows
        # exclusively through the broadcast ``_content_routing_weights``
        # slot. ``use_content_router=True`` advertises to the train /
        # inference call sites that they must thread ``crossattn_emb``
        # through ``set_content`` (no-op otherwise).
        self.content_router: Optional[ContentRouter] = None
        self.use_content_router: bool = False
        if (
            cfg.use_chimera_hydra
            and cfg.content_router_source == "crossattn_emb"
            and self._chimera_aware_loras
        ):
            self.content_router = ContentRouter(
                input_dim=CROSSATTN_EMB_DIM,
                num_content_experts=int(cfg.num_experts_content),
                hidden_dim=int(cfg.router_hidden_dim),
                tau=float(cfg.router_tau),
                init_std=float(cfg.content_router_init_std),
                apply_layer_norm=bool(cfg.content_router_layer_norm),
            )
            self.use_content_router = True
            logger.info(
                f"ChimeraHydra ContentRouter: input_dim={CROSSATTN_EMB_DIM} "
                f"(pooled crossattn_emb), K_c={cfg.num_experts_content}, "
                f"hidden={cfg.router_hidden_dim}, τ={cfg.router_tau:.2f}, "
                f"init_std={cfg.content_router_init_std}, "
                f"LN={cfg.content_router_layer_norm}, "
                f"chimera modules={len(self._chimera_aware_loras)} "
                "— per-Linear content router disabled"
            )

        self.register_injector: Optional[RegisterInjector] = None
        self.extra_seq_tokens = int(cfg.num_registers)
        if cfg.num_registers > 0:
            n_blocks = len(unet.blocks)
            if not (0 <= cfg.register_insert_block < n_blocks):
                raise ValueError(
                    f"register_insert_block must be in [0, {n_blocks}), "
                    f"got {cfg.register_insert_block}"
                )
            self.register_tokens = torch.nn.Parameter(
                torch.randn(cfg.num_registers, int(unet.model_channels))
                * cfg.register_init_std
            )
            self.register_injector = RegisterInjector(
                num_registers=cfg.num_registers,
                insert_block=cfg.register_insert_block,
                get_scaled_tokens=lambda: self.register_tokens * self.multiplier,
            )
            logger.info(
                f"Register tokens: K={cfg.num_registers}, "
                f"insert_block={cfg.register_insert_block}, "
                f"lr scale x{cfg.register_lr_scale:g}, "
                f"init_std={cfg.register_init_std:g}. "
                "Checkpoint stays kept-live at inference."
            )

    def _wire_shared_sigma_buffers(self) -> None:
        """Replace each HydraLoRA / OrthoHydraLoRA module's ``_sigma`` and
        ``_sigma_features`` buffers with references to a single network-level
        tensor (per sigma_feature_dim for the features). Modules then read the
        same tensor object as their own attribute, so an in-place ``copy_`` on
        the network's shared buffer flows to every module without a Python
        propagation loop.

        Run once at the end of ``__init__`` — before any forward fires, so
        Dynamo / cudagraphs capture the aliased data pointer on first compile
        and never see a per-module pointer-mismatch event.
        """
        sigma_loras: List[torch.nn.Module] = []
        by_dim: Dict[int, List[torch.nn.Module]] = {}
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_sigma" not in lora._buffers:
                continue
            sigma_loras.append(lora)
            d = int(getattr(lora, "sigma_feature_dim", 0))
            if d > 0 and "_sigma_features" in lora._buffers:
                by_dim.setdefault(d, []).append(lora)
        self._sigma_aware_loras = sigma_loras
        self._sigma_aware_loras_by_dim = by_dim
        if not sigma_loras:
            self._shared_sigma = None
            self._shared_sigma_features: Dict[int, torch.Tensor] = {}
            return

        # Pick the first module's placeholder buffer as the canonical shared
        # tensor; rebind every other module's buffer to the same object. The
        # placeholder is shape (1,) / (1, dim) — set_sigma replaces it with a
        # full-shape tensor on the first call (and re-aliases at the same time).
        shared_sigma = sigma_loras[0]._buffers["_sigma"]
        for lora in sigma_loras:
            lora._buffers["_sigma"] = shared_sigma
        self._shared_sigma = shared_sigma

        self._shared_sigma_features = {}
        for dim, loras in by_dim.items():
            shared_feat = loras[0]._buffers["_sigma_features"]
            for lora in loras:
                lora._buffers["_sigma_features"] = shared_feat
            self._shared_sigma_features[dim] = shared_feat

    def _wire_shared_fei_buffers(self) -> None:
        """Replace each FEI-aware module's ``_fei`` buffer with a single
        network-level shared tensor (per FEI feature dim).

        Mirrors ``_wire_shared_sigma_buffers``. ``set_fei`` writes to one
        shared buffer per dim; aliased module ``_fei`` buffers see the
        update through shared storage. The aliasing-recovery dance from
        ``set_sigma`` (rebind whenever shape or device drift breaks the
        identity) applies here too — ``Module._apply`` (``.to(device)``)
        independently reallocates buffers and silently breaks the link if
        we don't identity-check. See ``[[project_set_sigma_aliasing_bug]]``.
        """
        fei_loras: List[torch.nn.Module] = []
        by_dim: Dict[int, List[torch.nn.Module]] = {}
        for lora in self.unet_loras + self.text_encoder_loras:
            d = int(getattr(lora, "fei_feature_dim", 0))
            if d <= 0:
                continue
            if "_fei" not in lora._buffers:
                continue
            fei_loras.append(lora)
            by_dim.setdefault(d, []).append(lora)
        self._fei_aware_loras = fei_loras
        self._fei_aware_loras_by_dim = by_dim
        if not fei_loras:
            self._shared_fei: Dict[int, torch.Tensor] = {}
            return

        # One shared placeholder per dim — ``set_fei`` rebinds to full-shape
        # ``(B, dim)`` on first call.
        self._shared_fei = {}
        for dim, loras in by_dim.items():
            shared_feat = loras[0]._buffers["_fei"]
            for lora in loras:
                lora._buffers["_fei"] = shared_feat
            self._shared_fei[dim] = shared_feat

    def _wire_shared_routing_buffers(self) -> None:
        """Alias every routing-aware module's ``_routing_weights`` buffer to
        one network-level shared tensor.

        Mirrors ``_wire_shared_sigma_buffers`` / ``_wire_shared_fei_buffers``.
        ``StackedExpertsLoRAModule.__init__`` registers a ``(1, E)`` uniform
        placeholder; this pass picks the first module's buffer as canonical
        and rebinds every other module to the same object. ``set_routing_weights``
        then updates one shared tensor per step; aliased module buffers
        see the new gates through shared storage.

        All routing-aware modules in our build share the same ``num_experts``
        by construction (driven by ``cfg.num_experts``), so a single shared
        tensor is enough — no per-dim split like ``_shared_fei``.
        """
        routing_loras: List[torch.nn.Module] = []
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_routing_weights" not in lora._buffers:
                continue
            routing_loras.append(lora)
        self._routing_aware_loras = routing_loras
        if not routing_loras:
            self._shared_routing_weights: Optional[torch.Tensor] = None
            return

        canonical = routing_loras[0]._buffers["_routing_weights"]
        for lora in routing_loras:
            lora._buffers["_routing_weights"] = canonical
        self._shared_routing_weights = canonical

    def _wire_shared_content_routing_buffers(self) -> None:
        """Alias every chimera module's ``_content_routing_weights`` buffer to
        one shared tensor.

        Parallel to :meth:`_wire_shared_freq_routing_buffers`. ContentRouter
        broadcasts ``π_c`` once per step via direct slot assignment; aliased
        buffers on every chimera module see the new gates through shared
        storage. The buffer must carry the router's grad_fn (NO .detach(),
        NO .copy_()) so ``∂L_denoise/∂π_c`` reaches the ContentRouter.

        Identifies chimera modules by buffer presence (every chimera module
        registers ``_content_routing_weights`` in ``__init__``, regardless
        of router_source — the buffer is just dead under per-Linear mode).
        """
        content_loras: List[torch.nn.Module] = []
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_content_routing_weights" not in lora._buffers:
                continue
            content_loras.append(lora)
        self._content_aware_loras = content_loras
        if not content_loras:
            self._shared_content_routing_weights: Optional[torch.Tensor] = None
            return

        canonical = content_loras[0]._buffers["_content_routing_weights"]
        for lora in content_loras:
            lora._buffers["_content_routing_weights"] = canonical
        self._shared_content_routing_weights = canonical

    def _wire_shared_freq_routing_buffers(self) -> None:
        """Alias every chimera module's ``_freq_routing_weights`` buffer to one
        shared tensor.

        Parallel to ``_wire_shared_routing_buffers`` but on the chimera-
        specific buffer name. FreqRouter broadcasts ``π_f`` once per step via
        direct slot assignment; aliased buffers on every chimera module see
        the new gates through shared storage. The buffer must carry the
        router's grad_fn (NO .detach(), NO .copy_()) so ``∂L_denoise/∂π_f``
        reaches the FreqRouter — same contract as
        ``router_state._set_routing_weights``.
        """
        freq_loras: List[torch.nn.Module] = []
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_freq_routing_weights" not in lora._buffers:
                continue
            freq_loras.append(lora)
        self._chimera_aware_loras = freq_loras
        if not freq_loras:
            self._shared_freq_routing_weights: Optional[torch.Tensor] = None
            return

        canonical = freq_loras[0]._buffers["_freq_routing_weights"]
        for lora in freq_loras:
            lora._buffers["_freq_routing_weights"] = canonical
        self._shared_freq_routing_weights = canonical

    def prepare_network(self, args):
        if getattr(args, "lora_fp32_accumulation", False):
            logger.warning(
                "--lora_fp32_accumulation is deprecated and has no effect; "
                "fp32 accumulation is now unconditional in LoRA/Hydra/ReFT "
                "bottleneck matmuls. Remove the flag from your config."
            )

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def fuse_weights(self):
        """Merge all LoRA deltas into base model weights for zero-overhead inference."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.fuse_weight()

    def unfuse_weights(self):
        """Remove all LoRA deltas from base model weights."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.unfuse_weight()

    def set_timestep_mask(self, timesteps: torch.Tensor, max_timestep: float = 1.0):
        """Compute and set timestep-dependent rank mask on all modules."""
        if not self.cfg.use_timestep_mask:
            return

        max_rank = self.cfg.lora_dim
        # Reuse a single GPU-resident mask to avoid ~200 CPU→GPU transfers per step
        mask = getattr(self, "_shared_timestep_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, max_rank, device=timesteps.device)
            self._shared_timestep_mask = mask
            self._timestep_mask_arange = torch.arange(max_rank, device=timesteps.device)
            for lora in self.text_encoder_loras + self.unet_loras:
                lora._timestep_mask = mask

        # Compute threshold r entirely on device — avoids GPU→CPU .item() sync and
        # keeps the effective rank as a tensor so the mask build stays static-shape.
        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = (
            frac.pow(self.cfg.alpha_rank_scale) * (max_rank - self.cfg.min_rank)
            + self.cfg.min_rank
        )
        r = r.clamp(max=float(max_rank))
        mask.copy_((self._timestep_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def set_step_index(self, step_index: int) -> None:
        """Broadcast a hard denoising-step index to step-expert modules."""
        k = int(step_index)
        for lora in self.text_encoder_loras + self.unet_loras:
            set_step = getattr(lora, "set_step", None)
            if set_step is not None:
                set_step(k)

    def set_reft_timestep_mask(
        self, timesteps: torch.Tensor, max_timestep: float = 1.0
    ):
        """Compute and set timestep-dependent mask on ReFT modules."""
        if not self.cfg.use_timestep_mask:
            return
        refts = self.text_encoder_refts + self.unet_refts
        if not refts:
            return
        reft_dim = self.cfg.reft_dim

        mask = getattr(self, "_shared_reft_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, reft_dim, device=timesteps.device)
            self._shared_reft_mask = mask
            self._reft_mask_arange = torch.arange(reft_dim, device=timesteps.device)
            for reft in refts:
                reft._timestep_mask = mask

        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = frac.pow(self.cfg.alpha_rank_scale) * (reft_dim - 1) + 1
        r = r.clamp(max=float(reft_dim))
        mask.copy_((self._reft_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def clear_timestep_mask(self):
        """Restore full-rank masks on every LoRA / ReFT module.

        Each module's ``_timestep_mask`` is a Tensor by construction (default
        all-ones buffer at init, rebound to the shared live-updated mask when
        ``set_timestep_mask`` runs). Clearing fills the shared masks with ones
        in place — modules that were rebound immediately see the neutral mask
        via the shared reference; modules with local defaults are already
        neutral. Never set to None: the always-a-Tensor invariant is what
        keeps the adapter forward free of a None-vs-Tensor guard under
        ``torch.compile``.
        """
        shared = getattr(self, "_shared_timestep_mask", None)
        if shared is not None:
            shared.fill_(1.0)
        shared_reft = getattr(self, "_shared_reft_mask", None)
        if shared_reft is not None:
            shared_reft.fill_(1.0)

    def set_sigma(self, sigmas: torch.Tensor) -> None:
        """Stash per-sample σ on every HydraLoRA module whose router accepts σ.

        Mirrors ``set_timestep_mask`` — one call per training step. σ and the
        sinusoidal-features tensor are stored in network-level shared buffers
        whose storage is aliased into every sigma-aware module's ``_sigma`` /
        ``_sigma_features`` (see ``_wire_shared_sigma_buffers``), so the
        update is one in-place ``copy_`` per shared tensor instead of a
        per-module Python loop.

        IMPORTANT: write in place rather than rebinding. Inductor captures
        the buffers as static cudagraph inputs and re-records the whole graph
        if the data pointer changes — rebinding every step caused per-step
        re-record under ``compile_inductor_mode=reduce-overhead``
        (cudagraph_trees log: "static input data pointer changed"). Pointer
        only changes on the first call (placeholder → full-shape) and on a
        rare batch-shape change; both re-alias every module to the new tensor.

        Aliasing-recovery: ``Module._apply`` (i.e. ``network.to(device)``)
        reallocates each registered buffer independently, breaking the
        identity established by ``_wire_shared_sigma_buffers``. The
        ``self._shared_sigma`` Python attribute is *not* touched by
        ``Module._apply`` (it isn't a registered buffer), so post-``.to(...)``
        we may have a stale CPU shared tensor while the modules' ``_sigma``
        buffers all live on GPU and are no longer aliased to anything. Detect
        this on every call (cheap identity check against the canonical module's
        live buffer) and force the rebind path to re-establish aliasing —
        otherwise the in-place ``copy_`` writes to the orphaned CPU tensor
        and every module silently keeps reading its own zero-initialized
        ``_sigma``. This bug only manifests at B=1 (placeholder shape (1,)
        matches runtime shape so the historical rebind path was skipped),
        which is why σ-band partition and σ-feature router were both dead at
        ``batch_size=1`` despite the unit tests passing in eager mode.
        """
        sigmas = sigmas.detach()
        self._last_sigma = sigmas
        # Either path needs per-module ``_sigma``: σ-feature concat router
        # (sigma_feature_dim>0) and hard σ-band expert partition. Skip the
        # propagation entirely when neither is configured.
        if not (
            self.cfg.router_source == "sigma"
            or self.cfg.specialize_experts_by_sigma_buckets
        ):
            return
        sigma_loras = self._sigma_aware_loras
        if not sigma_loras:
            return

        # Canonical = the live buffer on the first sigma-aware module. After
        # ``network.to(device)`` this is the GPU-allocated tensor; before any
        # device move it's still the CPU placeholder from
        # ``_wire_shared_sigma_buffers``.
        canonical = sigma_loras[0]._buffers["_sigma"]
        cast = sigmas.to(dtype=canonical.dtype, device=canonical.device)
        # Rebind whenever (a) the shared attribute lost identity with the
        # canonical (e.g. ``.to()`` rebinding broke aliasing) or (b) the
        # shape changed (placeholder → full batch). Both branches need to
        # re-alias every module so the next call's fast path actually
        # propagates.
        needs_rebind = (
            self._shared_sigma is not canonical
            or canonical.shape != cast.shape
        )
        if needs_rebind:
            new_sigma = cast.detach().clone()
            for lora in sigma_loras:
                lora._buffers["_sigma"] = new_sigma
            self._shared_sigma = new_sigma
            shared_sigma = new_sigma
        else:
            canonical.copy_(cast)
            shared_sigma = canonical

        for dim, loras in self._sigma_aware_loras_by_dim.items():
            canonical_feat = loras[0]._buffers["_sigma_features"]
            feat = _sigma_sinusoidal_features(shared_sigma, dim).detach()
            cast_feat = feat.to(
                dtype=canonical_feat.dtype, device=canonical_feat.device
            )
            feat_needs_rebind = (
                self._shared_sigma_features.get(dim) is not canonical_feat
                or canonical_feat.shape != cast_feat.shape
            )
            if feat_needs_rebind:
                new_feat = cast_feat.clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat
            else:
                canonical_feat.copy_(cast_feat)

    def clear_sigma(self) -> None:
        """Reset cached σ to zeros.

        Never set to None: ``_sigma`` stays a Tensor so the unconditional
        sinusoidal path in ``_compute_gate`` has no None-vs-Tensor guard to
        recompile on under ``torch.compile``. Used in eval / validation
        and by inference teardown (``clear_hydra_sigma``). Zero in place to
        keep the cudagraph data pointer stable (see ``set_sigma`` note).

        Like ``set_sigma``, must operate on the *live* per-module buffer —
        ``Module._apply`` (``.to(device)``) breaks the init-time aliasing,
        and ``self._shared_sigma`` may then point at an orphaned CPU tensor
        whose zeroing wouldn't reach any module. Zero the canonical module
        buffer instead and re-establish aliasing if it was broken.
        """
        self._last_sigma = None
        if not self._sigma_aware_loras:
            return
        sigma_loras = self._sigma_aware_loras
        canonical = sigma_loras[0]._buffers["_sigma"]
        if self._shared_sigma is not canonical:
            for lora in sigma_loras:
                lora._buffers["_sigma"] = canonical
            self._shared_sigma = canonical
        canonical.zero_()
        for dim, loras in self._sigma_aware_loras_by_dim.items():
            canonical_feat = loras[0]._buffers["_sigma_features"]
            if self._shared_sigma_features.get(dim) is not canonical_feat:
                for lora in loras:
                    lora._buffers["_sigma_features"] = canonical_feat
                self._shared_sigma_features[dim] = canonical_feat
            zero_feat = _sigma_sinusoidal_features(canonical, dim)
            cast_feat = zero_feat.to(
                dtype=canonical_feat.dtype, device=canonical_feat.device
            )
            if canonical_feat.shape == cast_feat.shape:
                canonical_feat.copy_(cast_feat)
            else:
                new_feat = cast_feat.detach().clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat

    def set_fei(self, fei: torch.Tensor) -> None:
        """Stash per-sample FEI ``[B, fei_dim]`` on every FEI-aware module.

        Parallel to ``set_sigma`` — one call per training/inference step.
        Same shared-buffer aliasing recovery: identity-check ``self._shared_fei``
        against the canonical module's live buffer, rebind on shape change
        or after ``Module._apply`` orphans the link
        (``[[project_set_sigma_aliasing_bug]]``).

        ``fei`` must be ``(B, fei_feature_dim)`` matching
        ``cfg.fei_feature_dim`` (default 2 for the simplex). Caller is the
        train/inference loop running ``library.runtime.fei.compute_fei_2band``
        on ``z_t`` once per step.

        When ``cfg.route_per_layer=False`` and a ``GlobalRouter`` is wired,
        the router fires on the fresh FEI and its gates are broadcast to
        every routing-aware module via ``set_routing_weights`` in the same
        call — one entry point for the FeRA-style global-router path.
        """
        fei = fei.detach()
        # Fast-path: if there are no per-Linear FEI consumers, no global
        # router, and no chimera FreqRouter needing FEI, nothing to do.
        has_per_layer_fei = bool(getattr(self, "_fei_aware_loras", None))
        global_fei_router = (
            self.global_router
            if (
                self.global_router is not None
                and self.cfg.router_source == "fei"
                and not self.cfg.route_per_layer
            )
            else None
        )
        chimera_freq_router = (
            self.freq_router
            if (
                getattr(self, "freq_router", None) is not None
                and getattr(self, "_chimera_aware_loras", None)
            )
            else None
        )
        if not (
            has_per_layer_fei
            or global_fei_router is not None
            or chimera_freq_router is not None
        ):
            return
        if not (
            self.use_fei_router
            or global_fei_router is not None
            or chimera_freq_router is not None
        ):
            return

        # Per-layer FEI broadcast (legacy path — FEI-on-Hydra Phase 1).
        if has_per_layer_fei:
            # Group loras by their feature dim — every fei-aware module
            # currently in our network shares the same dim (cfg-level), but
            # the loop is robust to a future per-layer dim override.
            for dim, loras in self._fei_aware_loras_by_dim.items():
                canonical = loras[0]._buffers["_fei"]
                cast = fei.to(dtype=canonical.dtype, device=canonical.device)
                if cast.dim() == 1:
                    cast = cast.unsqueeze(0)
                if cast.shape[-1] != dim:
                    raise ValueError(
                        f"set_fei: fei.shape[-1]={cast.shape[-1]} != fei_feature_dim={dim}"
                    )
                current_shared = self._shared_fei.get(dim)
                needs_rebind = (
                    current_shared is not canonical
                    or canonical.shape != cast.shape
                )
                if needs_rebind:
                    new_fei = cast.detach().clone()
                    for lora in loras:
                        lora._buffers["_fei"] = new_fei
                    self._shared_fei[dim] = new_fei
                else:
                    canonical.copy_(cast)

        # Global router (FeRA-style): fire on fresh FEI and broadcast gates.
        # Router runs WITH grad so the autograd path ``L_denoise → y_t →
        # α_{t,m} → g_φ`` (FeRA eq. 6-7, 11) reaches the GlobalRouter params.
        # ``set_routing_weights`` reassigns each expert module's buffer slot
        # to the live ``gates`` tensor (no detach, no in-place copy).
        if global_fei_router is not None:
            gates = global_fei_router(fei)
            self.set_routing_weights(gates)

        # ChimeraHydra FreqRouter: input is concat(FEI, sinusoidal-σ-features).
        # σ already arrived through ``set_sigma`` (which fires before
        # ``set_fei`` in ``apply_router_conditioning``); the freq router lives
        # at network level and computes its features fresh each step rather
        # than relying on per-module shared σ-feature buffers (chimera modules
        # are built with ``sigma_feature_dim=0`` since the freq router owns
        # the σ axis exclusively).
        if chimera_freq_router is not None:
            sigma = self._last_sigma
            if sigma is None:
                raise RuntimeError(
                    "ChimeraHydra FreqRouter requires set_sigma to fire before "
                    "set_fei within the same step (apply_router_conditioning "
                    "preserves this order — check custom call sites)."
                )
            sigma_dim = int(self.cfg.sigma_feature_dim)
            sigma_feat = _sigma_sinusoidal_features(sigma, sigma_dim)
            # Match the FEI tensor's device/dtype and batch axis. Both should
            # share the same B by construction (one σ per sample, one FEI per
            # sample), so a straight cat is correct.
            fei_cast = fei.to(device=sigma_feat.device, dtype=sigma_feat.dtype)
            if fei_cast.dim() == 1:
                fei_cast = fei_cast.unsqueeze(0)
            router_in = torch.cat([fei_cast, sigma_feat], dim=-1)
            freq_gates = chimera_freq_router(router_in)
            self.set_freq_routing_weights(freq_gates)

    def clear_fei(self) -> None:
        """Reset cached FEI to zeros without rebinding pointers.

        Same in-place-zero pattern as ``clear_sigma`` — keeps cudagraph
        data pointers stable. Re-establishes aliasing if ``Module._apply``
        broke it since the last call.
        """
        if not getattr(self, "_fei_aware_loras", None):
            return
        for dim, loras in self._fei_aware_loras_by_dim.items():
            canonical = loras[0]._buffers["_fei"]
            current_shared = self._shared_fei.get(dim)
            if current_shared is not canonical:
                for lora in loras:
                    lora._buffers["_fei"] = canonical
                self._shared_fei[dim] = canonical
            canonical.zero_()

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast a ``(B, E)`` gate tensor to every routing-aware module.

        Called either:
          * Internally by ``set_fei`` when ``cfg.route_per_layer=False`` and
            ``cfg.router_source="fei"`` — the GlobalRouter fires on the
            fresh FEI and its output is broadcast here.
          * Externally by future ``"sigma"`` global-router paths /
            inference callers needing to push pre-computed gates.

        Assigns the SAME live ``weights`` tensor reference to every routing-
        aware module's ``_routing_weights`` buffer slot (no detach, no in-
        place copy). This is what gives the GlobalRouter its gradient path:
        ``L_denoise`` backprop flows through ``y_t = Σ α_{t,m} E_m(z_t)``
        (FeRA eq. 7) into ``α``, then through the assigned buffer reads
        into ``g_φ``'s parameters. The cudagraph-pointer-stability story is
        intentionally traded away here — gates are a tiny ``(B, E)`` tensor
        and the autograd path is what makes the router train at all.
        """
        if not getattr(self, "_routing_aware_loras", None):
            return
        routing_loras = self._routing_aware_loras
        canonical_buf = routing_loras[0]._buffers["_routing_weights"]
        w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        for lora in routing_loras:
            lora._routing_weights = w
        self._shared_routing_weights = w

    def clear_routing_weights(self) -> None:
        """Reset gates to uniform ``1/E`` in place.

        Called between training steps (or by inference teardown). Pointer
        stays stable for cudagraph capture; re-aliases if ``Module._apply``
        broke the link.
        """
        if not getattr(self, "_routing_aware_loras", None):
            return
        routing_loras = self._routing_aware_loras
        canonical = routing_loras[0]._buffers["_routing_weights"]
        if self._shared_routing_weights is not canonical:
            for lora in routing_loras:
                lora._buffers["_routing_weights"] = canonical
            self._shared_routing_weights = canonical
        E = int(canonical.shape[-1])
        canonical.fill_(1.0 / max(E, 1))

    def set_crossattn_routing(self, crossattn_emb: torch.Tensor) -> None:
        """Fire the network-level GlobalRouter on a pooled text vector.

        Used when ``cfg.router_source="crossattn_emb"`` (route_per_layer=False).
        ``crossattn_emb`` is the post-LLM-adapter text feature tensor — either
        ``(B, L, D)`` (raw, the GlobalRouter pools) or ``(B, D)`` (pre-pooled).
        No-op when no crossattn GlobalRouter is wired.

        Router runs WITH grad so ``L_denoise → y_t → α → GlobalRouter params``
        is intact; broadcast through :meth:`set_routing_weights` (the same
        ``_routing_weights`` slot the σ/FEI global router writes — the Hydra /
        stacked-experts modules need no crossattn-specific buffer).

        Call BEFORE each forward, separately for cond / uncond branches at
        inference — gates depend on the caption, so the two branches route
        differently (parallel to chimera's ``set_content``).
        """
        if self.global_router is None or not getattr(
            self, "use_crossattn_router", False
        ):
            return
        gates = self.global_router(crossattn_emb)
        self.set_routing_weights(gates)

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast ``π_f`` from the FreqRouter to every chimera module.

        Direct slot assignment (NO .detach(), NO .copy_()) so the buffer
        carries the router's grad_fn — same contract as
        ``set_routing_weights`` for the GlobalRouter. The chimera module's
        ``_compute_gate`` reads ``_freq_routing_weights`` to build the
        ``[π_c | π_f]`` concatenation, so the autograd path
        ``L_denoise → out_f → π_f → FreqRouter params`` is intact.
        """
        if not getattr(self, "_chimera_aware_loras", None):
            return
        freq_loras = self._chimera_aware_loras
        canonical_buf = freq_loras[0]._buffers["_freq_routing_weights"]
        w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        for lora in freq_loras:
            lora._freq_routing_weights = w
        self._shared_freq_routing_weights = w

    def clear_freq_routing_weights(self) -> None:
        """Reset chimera freq gates to uniform ``1/K_f`` in place."""
        if not getattr(self, "_chimera_aware_loras", None):
            return
        freq_loras = self._chimera_aware_loras
        canonical = freq_loras[0]._buffers["_freq_routing_weights"]
        if self._shared_freq_routing_weights is not canonical:
            for lora in freq_loras:
                lora._buffers["_freq_routing_weights"] = canonical
            self._shared_freq_routing_weights = canonical
        K_f = int(canonical.shape[-1])
        canonical.fill_(1.0 / max(K_f, 1))

    def set_content(self, crossattn_emb: torch.Tensor) -> None:
        """Fire the network-level ContentRouter on a pooled text vector.

        ``crossattn_emb`` is the post-LLM-adapter text feature tensor —
        either ``(B, L, D)`` (raw, this method pools) or ``(B, D)``
        (pre-pooled by the caller). No-op when the network has no
        ContentRouter (chimera off, or ``content_router_source="input"``).

        Router runs WITH grad so ``L_denoise → out_c → π_c → ContentRouter
        params`` is intact. Slot-assigned through
        :meth:`set_content_routing_weights`, same broadcast contract as
        ``set_freq_routing_weights`` / ``set_routing_weights``.
        """
        if self.content_router is None:
            return
        if not getattr(self, "_content_aware_loras", None):
            return
        gates = self.content_router(crossattn_emb)
        self.set_content_routing_weights(gates)

    def set_content_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast ``π_c`` from the ContentRouter to every chimera module.

        Direct slot assignment (NO .detach(), NO .copy_()) so the buffer
        carries the router's grad_fn — same contract as
        :meth:`set_freq_routing_weights`. Externally callable for inference
        paths that pre-compute gates (e.g. fixed per-prompt content slot
        debugging) without firing the MLP every step.
        """
        if not getattr(self, "_content_aware_loras", None):
            return
        content_loras = self._content_aware_loras
        canonical_buf = content_loras[0]._buffers["_content_routing_weights"]
        w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        for lora in content_loras:
            lora._content_routing_weights = w
        self._shared_content_routing_weights = w

    def clear_content_routing_weights(self) -> None:
        """Reset chimera content gates to uniform ``1/K_c`` in place."""
        if not getattr(self, "_content_aware_loras", None):
            return
        content_loras = self._content_aware_loras
        canonical = content_loras[0]._buffers["_content_routing_weights"]
        if self._shared_content_routing_weights is not canonical:
            for lora in content_loras:
                lora._buffers["_content_routing_weights"] = canonical
            self._shared_content_routing_weights = canonical
        K_c = int(canonical.shape[-1])
        canonical.fill_(1.0 / max(K_c, 1))

    def clear_step_caches(self) -> None:
        """Drop per-step tensor references (``_last_gate``) and invalidate
        memoized router-stats caches between training steps.

        Called unconditionally from the training loop before each forward,
        for two reasons:

        (1) ``_last_gate`` caches a tensor produced inside the compiled
        forward — under ``torch.compile(mode='reduce-overhead')`` that tensor
        lives in the inductor cudagraph memory pool. Holding a Python
        reference across the step boundary prevents ``cudagraph_trees`` from
        reclaiming pool memory and silently demotes the run to the eager
        fallback path. Call must precede ``cudagraph_mark_step_begin()``.

        (2) ``_router_stats_cache`` / ``_chimera_router_stats_cache`` memoize
        per-step router diagnostics so the progress-bar postfix and the TB
        logging layer share one D2H sync. Without per-step invalidation
        these freeze at their first computed values — and on runs without
        cudagraph mode (``_cudagraph_mark_step=False``) the invalidation has
        no other trigger, so TB shows the same usage/entropy on every log
        step.

        ``_sigma`` is intentionally *not* cleared: it's rebound by
        ``set_sigma`` before every forward, the caller passes a tensor from
        outside the compiled region (the flow-matching sampler's ``timesteps``,
        not a pool-allocated intermediate), and keeping it a Tensor at all
        times is what lets the adapter ``_compute_gate`` drop the None-vs-
        Tensor guard under ``torch.compile``.

        Safe to call unconditionally — consumers (balance loss, router stats)
        read ``_last_gate`` only within the step that wrote it.
        """
        self._last_sigma = None
        self._router_stats_cache = None
        self._chimera_router_stats_cache = None
        for lora in self.unet_loras + self.text_encoder_loras:
            if hasattr(lora, "_last_gate"):
                lora._last_gate = None
        # Drop the GlobalRouter's per-step transients for the same reason —
        # ``_last_gates`` / ``_last_input`` are detached tensors that may live
        # in the inductor cudagraph memory pool; holding a Python reference
        # across the step boundary blocks pool reclamation.
        if self.global_router is not None:
            self.global_router._last_gates = None
            self.global_router._last_input = None
            self.global_router._last_fei = None
        # Same treatment for the chimera FreqRouter.
        if getattr(self, "freq_router", None) is not None:
            self.freq_router._last_gates = None
            self.freq_router._last_input = None
        # …and the chimera ContentRouter (network-level content-pool variant).
        if getattr(self, "content_router", None) is not None:
            self.content_router._last_gates = None
            self.content_router._last_input = None

    def step_balance_loss_warmup(self, global_step: int, max_train_steps: int) -> None:
        return router_stats.step_balance_loss_warmup(self, global_step, max_train_steps)

    @staticmethod
    def _switch_balance(gate: torch.Tensor) -> torch.Tensor:
        return router_stats.switch_balance(gate)

    def get_balance_loss(self) -> torch.Tensor:
        return router_stats.get_balance_loss(self)

    def _get_chimera_balance_loss(self) -> torch.Tensor:
        return router_stats.get_chimera_balance_loss(self)

    def get_router_entropy(self) -> Optional[float]:
        return router_stats.get_router_entropy(self)

    def get_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float], List[List[float]], List[int]]]:
        return router_stats.get_router_stats(self)

    def get_chimera_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float]]]:
        return router_stats.get_chimera_router_stats(self)

    def capture_up_grad_stats(self) -> None:
        return router_stats.capture_up_grad_stats(self)

    def get_up_grad_stats(self) -> Dict[str, List[float]]:
        return router_stats.get_up_grad_stats(self)

    def get_ortho_regularization(self) -> torch.Tensor:
        return router_stats.get_ortho_regularization(self)

    def metrics(self, ctx: MetricContext) -> dict[str, float]:
        return router_stats.metrics(self, ctx)

    @staticmethod
    def _strip_orig_mod_keys(state_dict):
        return strip_orig_mod_keys(state_dict)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        state_dict = strip_orig_mod_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def load_weights(self, file):
        return load_lora_network_weights(self, file)

    def _reabsorb_baked_inv_scale(self, weights_sd: Dict[str, torch.Tensor]) -> None:
        return reabsorb_baked_inv_scale(self, weights_sd)

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(
                f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules"
            )
        else:
            self.text_encoder_loras = []
            self.text_encoder_refts = []

        if apply_unet:
            logger.info(f"enable LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []
            self.unet_refts = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        # ReFT wraps each selected DiT Block's forward, so the chain is:
        #   Block.__call__ -> ReFT.forward -> original Block.forward
        #   (inside which LoRA-wrapped Linears still fire normally).
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.apply_to()
            self.add_module(reft.lora_name, reft)

        if apply_unet and self.register_injector is not None:
            self.register_injector.apply(unet)

    def is_mergeable(self):
        return self.cfg.num_registers == 0

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_ANIMA):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for DiT")
        else:
            self.unet_loras = []

        # Pre-group checkpoint keys by LoRA module prefix (avoid O(modules * keys) scan)
        # Keys are "{module_name}.{param}" where module_name has no dots (dots → underscores)
        grouped_sd: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in weights_sd.items():
            prefix, dot, suffix = key.partition(".")
            if not dot:
                continue
            if prefix not in grouped_sd:
                grouped_sd[prefix] = {}
            grouped_sd[prefix][suffix] = value

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = grouped_sd.get(lora.lora_name, {})
            if sd_for_lora:
                lora.merge_to(sd_for_lora, dtype, device)

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(
        self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
    ):
        return optimizer_groups.set_loraplus_lr_ratio(
            self,
            loraplus_lr_ratio,
            loraplus_unet_lr_ratio,
            loraplus_text_encoder_lr_ratio,
        )

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        return optimizer_groups.prepare_lora_optimizer_params(
            self, text_encoder_lr, unet_lr, default_lr
        )

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        return save_lora_network_weights(self, file, dtype, metadata)

    def backup_weights(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                org_module._lora_org_weight = org_module.weight.detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                if not org_module._lora_restored:
                    org_module.weight.data.copy_(org_module._lora_org_weight)
                    org_module._lora_restored = True

    def pre_calculation(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                lora_weight = lora.get_weight().to(
                    org_module.weight.device, dtype=org_module.weight.dtype
                )
                org_module.weight.data.add_(lora_weight)

                org_module._lora_restored = False
                lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        if getattr(self.cfg, "use_dora", False):
            return 0, 0, 0

        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (
                    (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(
                    down.permute(1, 0, 2, 3), up
                ).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
