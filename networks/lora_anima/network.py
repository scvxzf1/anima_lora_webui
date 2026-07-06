# LoRANetwork: the module-assembly / training-orchestration core of the LoRA
# adapter stack for Anima. Targets DiT blocks (and optionally text-encoder
# attention) with pluggable per-module classes supplied by a NetworkSpec.

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import torch

from library.log import setup_logging
from library.training.metrics import MetricContext
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.persistence import (
    load_lora_network_weights,
    reabsorb_baked_inv_scale,
    save_lora_network_weights,
    strip_orig_mod_keys,
)
from networks.lora_anima import builders, optimizer_groups, router_stats, routing_state
from networks.lora_anima.targeting import compile_lora_target_patterns
from networks.lora_modules import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
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
        modules_dim = cfg.modules_dim
        dropout = cfg.dropout
        rank_dropout = cfg.rank_dropout
        module_dropout = cfg.module_dropout
        verbose = cfg.verbose
        alpha = cfg.alpha
        lora_dim = cfg.lora_dim
        train_llm_adapter = cfg.train_llm_adapter

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
            return builders.create_lora_modules(
                self,
                is_unet=is_unet,
                text_encoder_idx=text_encoder_idx,
                root_module=root_module,
                target_replace_modules=target_replace_modules,
                default_dim=default_dim,
                exclude_patterns=exclude_re_patterns,
                include_patterns=include_re_patterns,
                logger=logger,
            )

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
        self.text_encoder_refts, self.unet_refts = builders.create_reft_modules(
            unet,
            cfg=cfg,
            multiplier=multiplier,
            logger=logger,
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
        self.global_router: Optional[GlobalRouter]
        # ``use_crossattn_router`` advertises to the train / inference call
        # sites that they must fire ``set_crossattn_routing`` with the pooled
        # text tensor each forward (parallel to chimera's ``use_content_router``
        # but broadcasting to the standard ``_routing_weights`` slot).
        self.global_router, self.use_crossattn_router = builders.create_global_router(
            cfg,
            router_class=GlobalRouter,
            crossattn_emb_dim=CROSSATTN_EMB_DIM,
            routing_aware_count=len(self._routing_aware_loras),
            logger=logger,
        )

        # ChimeraHydra FreqRouter: one per network, broadcasts ``π_f`` over
        # the freq pool of every chimera module. Input is
        # ``concat(FEI, sinusoidal-σ-features)`` — owned by the freq router
        # exclusively (the per-layer content router never sees σ/FEI). Built
        # only when at least one chimera module was actually constructed; the
        # router_targets regex can narrow the chimera class to a subset of
        # layers (others fall back to OrthoLoRA).
        self.freq_router: Optional[FreqRouter]

        # ChimeraHydra ContentRouter: network-level twin of FreqRouter for
        # the content pool. Built only when ``content_router_source ==
        # "crossattn_emb"`` AND at least one chimera module exists. Per-Linear
        # ``self.router`` is None on those modules in that case — π_c flows
        # exclusively through the broadcast ``_content_routing_weights``
        # slot. ``use_content_router=True`` advertises to the train /
        # inference call sites that they must thread ``crossattn_emb``
        # through ``set_content`` (no-op otherwise).
        self.content_router: Optional[ContentRouter]
        (
            self.freq_router,
            self.content_router,
            chimera_uses_fei_router,
            self.use_content_router,
        ) = builders.create_chimera_routers(
            cfg,
            freq_router_class=FreqRouter,
            content_router_class=ContentRouter,
            crossattn_emb_dim=CROSSATTN_EMB_DIM,
            chimera_count=len(self._chimera_aware_loras),
            logger=logger,
        )
        if chimera_uses_fei_router:
            # Force the per-step conditioning hook to fire set_fei every step.
            self.use_fei_router = True

        self.register_tokens, self.register_injector, self.extra_seq_tokens = (
            builders.create_register_injector(self, unet, logger=logger)
        )

    def _wire_shared_sigma_buffers(self) -> None:
        return routing_state.wire_shared_sigma_buffers(self)

    def _wire_shared_fei_buffers(self) -> None:
        return routing_state.wire_shared_fei_buffers(self)

    def _wire_shared_routing_buffers(self) -> None:
        return routing_state.wire_shared_routing_buffers(self)

    def _wire_shared_content_routing_buffers(self) -> None:
        return routing_state.wire_shared_content_routing_buffers(self)

    def _wire_shared_freq_routing_buffers(self) -> None:
        return routing_state.wire_shared_freq_routing_buffers(self)

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
        return routing_state.set_timestep_mask(self, timesteps, max_timestep)

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
        return routing_state.set_reft_timestep_mask(self, timesteps, max_timestep)

    def clear_timestep_mask(self):
        return routing_state.clear_timestep_mask(self)

    def set_sigma(self, sigmas: torch.Tensor) -> None:
        return routing_state.set_sigma(self, sigmas)

    def clear_sigma(self) -> None:
        return routing_state.clear_sigma(self)

    def set_fei(self, fei: torch.Tensor) -> None:
        return routing_state.set_fei(self, fei)

    def clear_fei(self) -> None:
        return routing_state.clear_fei(self)

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        return routing_state.set_routing_weights(self, weights)

    def clear_routing_weights(self) -> None:
        return routing_state.clear_routing_weights(self)

    def set_crossattn_routing(self, crossattn_emb: torch.Tensor) -> None:
        return routing_state.set_crossattn_routing(self, crossattn_emb)

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        return routing_state.set_freq_routing_weights(self, weights)

    def clear_freq_routing_weights(self) -> None:
        return routing_state.clear_freq_routing_weights(self)

    def set_content(self, crossattn_emb: torch.Tensor) -> None:
        return routing_state.set_content(self, crossattn_emb)

    def set_content_routing_weights(self, weights: torch.Tensor) -> None:
        return routing_state.set_content_routing_weights(self, weights)

    def clear_content_routing_weights(self) -> None:
        return routing_state.clear_content_routing_weights(self)

    def clear_step_caches(self) -> None:
        return routing_state.clear_step_caches(self)

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
