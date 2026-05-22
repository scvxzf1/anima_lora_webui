# Per-layer time-indexed soft tokens — SoftREPA-style parameterization (without
# the contrastive loss). DiT is frozen; trains a small bank of K continuous
# vectors per (layer, t-bucket) pair, prepended (via end-of-sequence overwrite)
# to crossattn_emb at each block independently.
#
# Reference: Lee et al., "Aligning Text to Image in Diffusion Models is Easier
# Than You Think" (NeurIPS 2025) — arXiv:2503.08250. We adopt only their
# parameterization (per-layer × per-t soft tokens), trained under plain flow-
# matching loss; the contrastive InfoNCE objective is intentionally skipped.
# In Anima the DiT is cross-attention (not joint-stream MM-DiT like SD3), so
# crossattn_emb does not evolve through blocks — each block independently
# receives crossattn_emb extended with its own layer tokens. No strip/re-prepend
# dance is needed.
#
# Splice strategy: end-of-sequence overwrite of zero-padding (K tail slots),
# preserving static crossattn_emb shape so `_run_blocks` torch.compile stays
# happy. Zero-padded positions act as cross-attention sinks (see Anima's text-
# encoder padding invariant), so writing tokens into them gives them attention
# mass without changing seqlen.
#
# Why a separate module from postfix.py: postfix splices once at the cached
# adapter output (training-time and inference-time). Soft tokens splice per-
# block via monkey-patched Block.forward (ReFT-pattern), a fundamentally
# different surface.
#
# Inference path: the per-step splice runs from inside the denoising loop —
# library/inference/generation.py + networks/spectrum.py call append_postfix(...,
# timesteps=t) per CFG branch before each forward, mirroring the training-side
# trainer hook. On Spectrum cached steps the blocks don't fire, so soft tokens
# silently no-op for those steps (composes freely with --spectrum).

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from library.log import setup_logging
from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    StepCtx,
)
from networks.methods.base import AdapterNetworkBase

import logging

setup_logging()
logger = logging.getLogger(__name__)

# Anima cached crossattn_emb dimension (Qwen3 hidden size, post LLM-adapter).
DEFAULT_EMBED_DIM = 1024

# Contrastive negative-sourcing modes (docs/proposal/soft_tokens_contrastive.md).
# ``shuffled`` draws an unrelated cached-TE negative; ``jaccard`` keeps shuffled
# sourcing but down-weights each negative's logit by its caption tag-overlap;
# ``hard`` draws a same-artist / different-character sibling (falls back to
# shuffled for orphan artists).
CONTRASTIVE_MODES = ("shuffled", "jaccard", "hard")


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    num_tokens = network_dim if network_dim is not None else 4
    embed_dim = int(kwargs.get("embed_dim", DEFAULT_EMBED_DIM))
    n_layers = int(kwargs.get("n_layers", 10))
    n_t_buckets = int(kwargs.get("n_t_buckets", 100))
    init_std = float(kwargs.get("init_std", 0.02))
    splice_position = kwargs.get("splice_position", "end_of_sequence")
    contrastive_weight = float(kwargs.get("contrastive_weight", 0.0))
    contrastive_k = int(kwargs.get("contrastive_k", 1))
    contrastive_negative_mode = str(kwargs.get("contrastive_negative_mode", "shuffled"))
    contrastive_tau = float(kwargs.get("contrastive_tau", 0.5))
    contrastive_warmup_ratio = float(kwargs.get("contrastive_warmup_ratio", 0.1))
    contrastive_jaccard_alpha = float(kwargs.get("contrastive_jaccard_alpha", 1.0))
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=init_std,
        splice_position=splice_position,
        contrastive_weight=contrastive_weight,
        contrastive_k=contrastive_k,
        contrastive_negative_mode=contrastive_negative_mode,
        contrastive_tau=contrastive_tau,
        contrastive_warmup_ratio=contrastive_warmup_ratio,
        contrastive_jaccard_alpha=contrastive_jaccard_alpha,
        multiplier=multiplier,
    )
    return network


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    tokens = weights_sd.get("tokens")
    t_offsets = weights_sd.get("t_offsets.weight")
    if tokens is None or t_offsets is None:
        raise ValueError(
            f"soft_tokens weight file must contain 'tokens' and 't_offsets.weight' "
            f"(got keys: {list(weights_sd.keys())[:8]})"
        )
    n_layers, num_tokens, embed_dim = tokens.shape
    n_t_buckets = t_offsets.shape[0]
    # Splice position is a runtime knob, not learned — read from metadata if
    # present, otherwise default. CLI kwargs win for post-hoc overrides.
    metadata_splice = None
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = f.metadata() or {}
            metadata_splice = meta.get("ss_splice_position")
    splice_position = kwargs.get(
        "splice_position", metadata_splice or "end_of_sequence"
    )
    network = SoftTokensNetwork(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_t_buckets=n_t_buckets,
        init_std=0.0,  # weights are loaded; init_std doesn't matter
        splice_position=splice_position,
        # Contrastive is a training-only objective (extra forwards) — it leaves
        # no learned parameters, so the loaded checkpoint is bit-identical
        # whether or not it trained. Keep it off on the inference path.
        contrastive_weight=0.0,
        multiplier=multiplier,
    )
    return network, weights_sd


class SoftTokensNetwork(AdapterNetworkBase):
    """Per-layer time-indexed soft tokens.

    Parameters:
      - tokens: (n_layers, K, D) — base per-layer tokens, small-std init.
      - t_offsets: Embedding(n_t_buckets, n_layers * D) — per-(t_bucket, layer)
        broadcast offset (one D-vector applied to every token in the layer).
        Zero-init so step 0 reproduces the un-time-conditioned base tokens.

    Param count: n_layers·K·D + n_t_buckets·n_layers·D
    With defaults (n_layers=10, K=4, D=1024, n_t_buckets=100): 40k + 1.0M ≈ 1.05M.
    """

    network_module = "networks.methods.soft_tokens"
    network_spec = "soft_tokens"

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        n_layers: int = 10,
        n_t_buckets: int = 100,
        init_std: float = 0.02,
        splice_position: str = "end_of_sequence",
        contrastive_weight: float = 0.0,
        contrastive_k: int = 1,
        contrastive_negative_mode: str = "shuffled",
        contrastive_tau: float = 0.5,
        contrastive_warmup_ratio: float = 0.1,
        contrastive_jaccard_alpha: float = 1.0,
        multiplier: float = 1.0,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        # Upper-bound check against actual block count happens in apply_to().
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if n_t_buckets <= 0:
            raise ValueError(f"n_t_buckets must be positive, got {n_t_buckets}")
        if splice_position not in ("front_of_padding", "end_of_sequence"):
            raise ValueError(
                f"splice_position must be 'front_of_padding' or 'end_of_sequence', "
                f"got {splice_position!r}"
            )
        if contrastive_negative_mode not in CONTRASTIVE_MODES:
            raise ValueError(
                f"contrastive_negative_mode must be one of {CONTRASTIVE_MODES}, "
                f"got {contrastive_negative_mode!r}"
            )
        if contrastive_weight > 0.0:
            if contrastive_k < 1:
                raise ValueError(f"contrastive_k must be >= 1, got {contrastive_k}")
            if contrastive_tau <= 0.0:
                raise ValueError(
                    f"contrastive_tau must be positive, got {contrastive_tau}"
                )

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_t_buckets = n_t_buckets
        self.splice_position = splice_position
        self.multiplier = multiplier

        # Contrastive objective (SoftREPA-style InfoNCE, B=1-adapted via
        # cached-TE hard negatives — docs/proposal/soft_tokens_contrastive.md).
        # Pure training-time objective: the extra forwards live in
        # ``SoftTokensMethodAdapter.extra_forwards`` and the warmup-gated weight
        # is applied composer-side. ``_contrastive_target_weight`` gates composer
        # activation,
        # ``_contrastive_weight`` is the live (warmup-held) value the loss
        # handler multiplies by.
        self._contrastive_target_weight = float(contrastive_weight)
        self._contrastive_warmup_ratio = float(contrastive_warmup_ratio)
        self._contrastive_tau = float(contrastive_tau)
        self.contrastive_k = int(contrastive_k)
        self.contrastive_negative_mode = str(contrastive_negative_mode)
        self.contrastive_jaccard_alpha = float(contrastive_jaccard_alpha)
        self._contrastive_weight = (
            0.0
            if self._contrastive_warmup_ratio > 0.0
            else self._contrastive_target_weight
        )

        self.tokens = nn.Parameter(
            torch.randn(n_layers, num_tokens, embed_dim) * init_std
        )
        # Per-(bucket, layer) D-vector offset. Broadcast across the K-token axis
        # at lookup so the bucket only has to learn one D-vector per layer per
        # bucket (not K). Zero-init = identity perturbation at step 0.
        self.t_offsets = nn.Embedding(n_t_buckets, n_layers * embed_dim)
        nn.init.zeros_(self.t_offsets.weight)

        # Step-scoped state set by append_postfix() once per forward pass and
        # consumed by the per-block hooks installed in apply_to(). Kept as a
        # plain attribute (not a buffer) — recreated each step, no need to
        # persist or move with .to(). _step_seqlens is only populated for
        # front_of_padding splice; end_of_sequence ignores it.
        self._step_layer_tokens: Optional[torch.Tensor] = None  # (n_layers, B, K, D)
        self._step_seqlens: Optional[torch.Tensor] = None  # (B,) int

        # Reverse-bookkeeping for apply_to(): keep references so we could
        # un-monkey-patch later (currently unused but cheap to track).
        self._block_refs: list[nn.Module] = []
        self._original_forwards: list = []

        n_token_params = self.tokens.numel()
        n_offset_params = self.t_offsets.weight.numel()
        logger.info(
            f"SoftTokensNetwork: {n_layers} layers × {num_tokens} tokens × dim {embed_dim}, "
            f"{n_t_buckets} t-buckets, splice={splice_position} → "
            f"{n_token_params + n_offset_params} params "
            f"({n_token_params} base + {n_offset_params} t-offset)"
        )

    # Sentinel attribute so train.py's ``hasattr(network, "append_postfix")``
    # branch picks us up: train.py will then call append_postfix(..., timesteps=...)
    # at the right point in the forward loop, which we use only to compute the
    # step-scoped per-layer tokens. The crossattn_emb passes through unchanged
    # — splicing happens inside the per-block hooks below.
    @property
    def num_postfix_tokens(self) -> int:
        return self.num_tokens

    def _bucketize(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map sigma in [0, 1] (Anima convention) to integer buckets [0, n_t_buckets).

        Outside-range values are clamped, so callers don't need to pre-clamp.
        """
        t = timesteps.detach().float().flatten()
        idx = torch.floor(t * self.n_t_buckets).long()
        return idx.clamp(min=0, max=self.n_t_buckets - 1)

    def append_postfix(
        self,
        crossattn_emb: torch.Tensor,
        crossattn_seqlens: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-step layer tokens and cache them for the block hooks.

        Returns crossattn_emb unchanged — splice happens per-block in the hooks
        installed by ``apply_to()``. We just piggyback on train.py's existing
        per-step trainer hook to receive timesteps.
        """
        if timesteps is None:
            raise ValueError(
                "soft_tokens requires timesteps (per-step) — train.py passes "
                "this automatically; inference path is not yet wired up"
            )
        B = crossattn_emb.shape[0]
        bucket_idx = self._bucketize(timesteps)  # (B,)
        # (B, n_layers * D) → (B, n_layers, D) → (B, n_layers, 1, D)
        offsets = self.t_offsets(bucket_idx).view(B, self.n_layers, self.embed_dim)
        # (n_layers, K, D) → (1, n_layers, K, D), broadcast over batch.
        base = self.tokens.unsqueeze(0)
        # (B, n_layers, K, D); offset broadcasts across K.
        per_step = base + offsets.unsqueeze(2)
        # Transpose to (n_layers, B, K, D) for cheap per-layer indexing in the
        # block hook closure.
        self._step_layer_tokens = per_step.transpose(0, 1).contiguous()
        # front_of_padding needs per-sample seqlens at hook time; end_of_sequence
        # ignores them. Cache regardless so the hook doesn't have to know which
        # mode is active (the splice branch reads or skips).
        self._step_seqlens = (
            crossattn_seqlens.detach().to(torch.long)
            if crossattn_seqlens is not None
            else None
        )
        return crossattn_emb

    def _make_block_hook(self, layer_idx: int, org_forward):
        """Closure that splices layer_idx's tokens into crossattn_emb tail.

        Block.forward signature (from library/anima/models.py:1179):
          forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params,
                  rope_cos_sin=None, adaln_lora_B_T_3D=None)
        """
        K = self.num_tokens
        splice_position = self.splice_position
        net = self  # capture network for state lookup

        def hook(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            *args,
            **kwargs,
        ):
            step_tokens = net._step_layer_tokens
            if step_tokens is not None:
                # (B, K, D) for this layer. Cast to crossattn dtype/device.
                layer_tok = step_tokens[layer_idx].to(
                    dtype=crossattn_emb.dtype, device=crossattn_emb.device
                )
                S = crossattn_emb.shape[1]
                if S < K:
                    raise RuntimeError(
                        f"crossattn_emb seqlen {S} < num_tokens {K}; cannot splice"
                    )
                if splice_position == "end_of_sequence":
                    # Overwrite the K tail (zero-padding) slots. torch.cat
                    # preserves autograd through both branches.
                    crossattn_emb = torch.cat(
                        [crossattn_emb[:, : S - K, :], layer_tok], dim=1
                    )
                else:  # front_of_padding
                    # Place K tokens at [seqlens[i], seqlens[i]+K) per sample —
                    # displaces the strongest sinks. scatter() preserves grad
                    # on the written values.
                    seqlens = net._step_seqlens
                    if seqlens is None:
                        raise RuntimeError(
                            "front_of_padding splice requires crossattn_seqlens; "
                            "trainer must pass it to append_postfix()"
                        )
                    offsets = seqlens.to(crossattn_emb.device).unsqueeze(
                        1
                    ) + torch.arange(K, device=crossattn_emb.device)  # (B, K)
                    D = crossattn_emb.shape[-1]
                    idx = offsets.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
                    crossattn_emb = crossattn_emb.scatter(1, idx, layer_tok)
            return org_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                *args,
                **kwargs,
            )

        return hook

    def apply_to(
        self,
        text_encoders,
        unet,
        apply_text_encoder=True,
        apply_unet=True,
    ):
        """Monkey-patch the first n_layers DiT blocks with the splice hook."""
        blocks = getattr(unet, "blocks", None)
        if blocks is None:
            raise RuntimeError("unet has no .blocks attribute — not an Anima DiT?")
        if len(blocks) < self.n_layers:
            raise RuntimeError(
                f"unet has {len(blocks)} blocks but n_layers={self.n_layers}"
            )
        self._block_refs = []
        self._original_forwards = []
        for k in range(self.n_layers):
            block = blocks[k]
            org_forward = block.forward
            block.forward = self._make_block_hook(k, org_forward)
            self._block_refs.append(block)
            self._original_forwards.append(org_forward)
        logger.info(
            f"soft_tokens: monkey-patched first {self.n_layers} of {len(blocks)} "
            f"DiT blocks (end-of-sequence splice, K={self.num_tokens})"
        )

    # ── Standard adapter API ────────────────────────────────────────────

    def get_trainable_params(self):
        return [self.tokens, self.t_offsets.weight]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        params = [{"params": self.get_trainable_params(), "lr": lr}]
        descriptions = ["soft_tokens(tokens+t_offsets)"]
        return params, descriptions

    def state_dict_for_save(self, dtype):
        return {
            "tokens": self.tokens.detach().clone().cpu().to(dtype),
            "t_offsets.weight": self.t_offsets.weight.detach().clone().cpu().to(dtype),
        }

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_num_tokens": str(self.num_tokens),
            "ss_embed_dim": str(self.embed_dim),
            "ss_n_layers": str(self.n_layers),
            "ss_n_t_buckets": str(self.n_t_buckets),
            "ss_splice_position": self.splice_position,
            # Contrastive objective is training-only (no learned params), but
            # stamp the config for run provenance.
            "ss_contrastive_weight": str(self._contrastive_target_weight),
            "ss_contrastive_k": str(self.contrastive_k),
            "ss_contrastive_negative_mode": self.contrastive_negative_mode,
            "ss_contrastive_tau": str(self._contrastive_tau),
            "ss_contrastive_warmup_ratio": str(self._contrastive_warmup_ratio),
        }

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        if "tokens" not in weights_sd or "t_offsets.weight" not in weights_sd:
            raise ValueError(
                f"Missing required keys in soft_tokens checkpoint "
                f"(got: {list(weights_sd.keys())[:8]})"
            )
        self.tokens.data.copy_(weights_sd["tokens"])
        self.t_offsets.weight.data.copy_(weights_sd["t_offsets.weight"])
        logger.info(
            f"Loaded soft_tokens weights: tokens={tuple(self.tokens.shape)}, "
            f"t_offsets={tuple(self.t_offsets.weight.shape)}"
        )

    def metrics(self, ctx) -> dict[str, float]:
        """TensorBoard bank-state diagnostics.

        ``soft_tokens/*`` — bank-state diagnostics, computed on the K base
        tokens averaged over layers. Read these as a collapse / divergence
        detector:
          - ``tokens_mean_cos`` near 0  → bank is orthogonal-ish (good).
          - ``tokens_mean_cos`` near 1  → slot collapse (slots redundant).
          - ``tokens_mean_norm`` blowing up → bank magnitude diverging.
          - ``offset_mean_norm`` staying ~0 → t-offset buckets aren't training
            (FM gradient isn't reaching them; check LR).
        """
        del ctx
        out: dict[str, float] = {}

        # Bank-state diagnostics always logged when there are ≥ 2 tokens per
        # layer to take pairs from — cheap collapse/divergence signal.
        if self.num_tokens >= 2 and self.n_layers > 0:
            tokens = self.tokens.detach()
            # Pairwise cos / d² per layer, averaged across layers.
            cos_sum = 0.0
            d_min = float("inf")
            for k in range(self.n_layers):
                z = tokens[k]
                zn = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
                gram = zn @ zn.t()
                n = gram.shape[0]
                iu = torch.triu_indices(n, n, offset=1, device=gram.device)
                cos_sum += float(gram[iu[0], iu[1]].mean().item())
                d_sq = torch.pdist(z, p=2).pow(2)
                if d_sq.numel():
                    d_min = min(d_min, float(d_sq.min().item()))
            out["soft_tokens/tokens_mean_cos"] = cos_sum / self.n_layers
            out["soft_tokens/tokens_min_d_sq"] = d_min if d_min != float("inf") else 0.0
            out["soft_tokens/tokens_mean_norm"] = float(
                tokens.flatten(1).norm(dim=-1).mean().item()
            )
        out["soft_tokens/offset_mean_norm"] = float(
            self.t_offsets.weight.detach()
            .view(self.n_t_buckets, self.n_layers, self.embed_dim)
            .permute(1, 0, 2)
            .flatten(1)
            .norm(dim=-1)
            .mean()
            .item()
        )
        return out

    def step_contrastive_warmup(self, global_step: int, max_train_steps: int) -> None:
        """Activate the contrastive objective once training crosses its warmup
        window. Step function: ``_contrastive_weight`` holds at 0 for the first
        ``_contrastive_warmup_ratio`` of steps, then flips to
        ``_contrastive_target_weight``. No-op when target is 0.

        Rationale: let plain FM gradient shape a non-degenerate bank before the
        contrastive term starts pulling the soft tokens toward text
        discrimination, so the contrast sharpens an existing signal rather than
        fighting a near-random init.
        """
        target = float(self._contrastive_target_weight)
        ratio = float(self._contrastive_warmup_ratio)
        if target <= 0.0:
            return
        if ratio <= 0.0 or max_train_steps <= 0:
            self._contrastive_weight = target
            return
        warmup_steps = int(max_train_steps * ratio)
        self._contrastive_weight = 0.0 if global_step < warmup_steps else target

    def contrastive_loss(
        self,
        v_pos: torch.Tensor,
        v_neg: torch.Tensor,
        v_target: torch.Tensor,
        neg_penalty: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """SoftREPA-style InfoNCE over cached-TE negatives (B=1-adapted).

        Each forward shares the same ``(x_t, ε, t)`` and spliced soft tokens;
        only ``crossattn_emb`` differs (matched vs. mismatched text). The logit
        for a forward is the negative flow-matching error against the shared
        velocity target, scaled by τ::

            ℓ_* = -‖v_* − v_target‖² / τ            (mean over C·H·W per sample)
            L   = -log( exp(ℓ_pos) / Σ_{pos,neg} exp(ℓ_*) )

        Gradient flows to the soft tokens (via every ``v_*``) to make the
        matched text explain the anchor's latent better than mismatched text —
        i.e. to sharpen the cross-attention's text discrimination.

        ``neg_penalty`` (the ``jaccard`` mode's ``α·s``, shape ``(B, k)``) is
        subtracted from each negative logit before the softmax — a negative that
        shares tags with the anchor becomes a less-surprising mismatch and pulls
        less gradient. ``None`` ⇒ plain InfoNCE (``shuffled`` / ``hard``).

        Shapes
        ------
        v_pos, v_target : ``(B, C, H, W)``
        v_neg           : ``(B, k, C, H, W)``
        neg_penalty     : ``(B, k)`` or None

        Returns
        -------
        (loss_scalar, diagnostics) where diagnostics carries the contrastive
        accuracy (pos beats every negative) and the mean pos−neg logit gap for
        TensorBoard.
        """
        tau = float(self._contrastive_tau)
        vp = v_pos.float()
        vt = v_target.float()
        vn = v_neg.float()
        B = vp.shape[0]
        k = vn.shape[1]
        # Per-sample mean-squared FM error → logit. Reduce over all non-batch
        # dims so τ has a stable scale across resolutions.
        pos_err = (vp - vt).pow(2).reshape(B, -1).mean(dim=1)  # (B,)
        logit_pos = -pos_err / tau  # (B,)
        vt_exp = vt.unsqueeze(1)  # (B, 1, C, H, W)
        neg_err = (vn - vt_exp).pow(2).reshape(B, k, -1).mean(dim=2)  # (B, k)
        logits_neg = -neg_err / tau  # (B, k)
        if neg_penalty is not None:
            # jaccard mode: down-weight tag-overlapping negatives.
            logits_neg = logits_neg - neg_penalty.to(logits_neg.dtype)

        # InfoNCE: -log softmax of the positive over {pos, neg_1..k}.
        all_logits = torch.cat([logit_pos.unsqueeze(1), logits_neg], dim=1)  # (B, 1+k)
        loss = (-logit_pos + torch.logsumexp(all_logits, dim=1)).mean()

        with torch.no_grad():
            acc = (logit_pos.unsqueeze(1) > logits_neg).all(dim=1).float().mean()
            gap = (logit_pos - logits_neg.mean(dim=1)).mean()
        diagnostics = {
            "contrastive_acc": float(acc.item()),
            "contrastive_logit_gap": float(gap.item()),
        }
        return loss, diagnostics


class SoftTokensMethodAdapter(MethodAdapter):
    """Runs the contrastive negative forwards for soft tokens.

    The contrastive objective (docs/proposal/soft_tokens_contrastive.md) needs
    one extra DiT forward per negative, sharing the anchor's ``(x_t, ε, t)`` and
    spliced soft tokens but swapping ``crossattn_emb`` for a cached negative
    text embedding. This is exactly the ``extra_forwards`` contract.

    Wiring:
      - ``prime_for_forward`` stashes ``batch["neg_crossattn_emb"]`` (the
        dataset surfaces ``(B, k, S, D)`` on train steps only).
      - ``extra_forwards`` runs the ``k`` negative forwards and returns the raw
        InfoNCE scalar under ``"soft_tokens_contrastive"``; the composer applies
        the warmup-gated ``_contrastive_weight``.
      - ``metrics`` surfaces the term + accuracy + logit gap each log step.

    Training-only: the negatives are absent (and forwards skipped) outside
    training, so validation FM-MSE stays a clean per-token regression metric.
    """

    name = "soft_tokens_contrastive"

    def __init__(self) -> None:
        self._neg_crossattn: Optional[torch.Tensor] = None
        self._neg_jaccard: Optional[torch.Tensor] = None
        self._last_metrics: dict[str, float] = {}

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        del ctx, latents
        if not is_train or not isinstance(batch, dict):
            self._neg_crossattn = None
            self._neg_jaccard = None
            return
        self._neg_crossattn = batch.get("neg_crossattn_emb")
        # Per-negative tag-overlap Jaccard (B, k), present only in jaccard mode.
        self._neg_jaccard = batch.get("neg_jaccard")

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts) -> Optional[dict]:
        if not primary.is_train:
            return None
        neg = self._neg_crossattn
        if neg is None:
            return None
        net = ctx.accelerator.unwrap_model(ctx.network)
        if float(getattr(net, "_contrastive_target_weight", 0.0) or 0.0) <= 0.0:
            return None

        device = primary.noisy_model_input.device
        ce_dtype = primary.crossattn_emb.dtype
        neg = neg.to(device)  # (B, k, S, D)
        k = neg.shape[1]

        v_pos = primary.model_pred.squeeze(2)  # (B, C, H, W) — live anchor graph
        # Rectified-flow velocity target — same as train.py's primary target.
        v_target = primary.noise - primary.latents  # (B, C, H, W)
        timesteps = primary.timesteps
        base_kw = dict(primary.forward_kwargs)

        # Snapshot the anchor's per-step splice state. The eager (value) pass of
        # each checkpointed negative below mutates the per-step buffers, so we
        # restore the anchor's afterwards. (The anchor's own autograd graph holds
        # tensor *references*, so it is unaffected by these attribute writes —
        # this restore just keeps the live network state coherent.)
        anchor_tokens = net._step_layer_tokens
        anchor_seqlens = net._step_seqlens

        # ── Gradient caching (your "grad-accum across the contrastive forwards")
        # The InfoNCE coupling is a tiny head over (B,C,H,W) velocities, but each
        # negative's velocity comes from a full frozen-DiT forward. Holding all
        # k+1 activation graphs at once is (k+1)× peak and OOMs a 16GB card even
        # at k=1. So we never build the negative graphs eagerly: each negative
        # forward is wrapped in activation checkpointing, so its graph is
        # *recomputed one at a time during backward* — peak stays at a single DiT
        # forward regardless of k. (Plain LoRA at any rank fits one forward, which
        # is why rank never OOMs — rank touches params, not activations.)
        #
        # The splice is mutable network state (`_step_layer_tokens`, set by
        # `append_postfix`, read by the per-block hooks), so the re-prime MUST run
        # *inside* the checkpointed callable: otherwise the backward recompute
        # would read whatever negative/anchor state is live at backward time and
        # splice the wrong tokens. Re-priming inside also keeps the param→token
        # graph (`self.tokens` → `_step_layer_tokens`) intact through recompute,
        # so gradient still reaches the soft tokens.
        def _neg_velocity(neg_j: torch.Tensor) -> torch.Tensor:
            def run(nmi, ts, neg_emb, pm):
                # Cached crossattn_emb is zero-padded past the real text length
                # (the LLM adapter zeroes padding positions), so non-zero rows
                # give the per-sample seqlen the front_of_padding splice needs.
                seqlens = (neg_emb.abs().sum(dim=-1) > 0).sum(dim=-1).to(torch.int32)
                # Returns crossattn_emb unchanged — the per-block hooks splice
                # during the forward off the buffer this call primes.
                ce = net.append_postfix(neg_emb, seqlens, timesteps=ts)
                kw_j = dict(base_kw)
                if "pooled_text_override" in kw_j:
                    kw_j["pooled_text_override"] = neg_emb.max(dim=1).values
                return primary.anima_call(
                    nmi, ts, ce, padding_mask=pm, **kw_j
                ).squeeze(2)

            return checkpoint(
                run,
                primary.noisy_model_input,
                timesteps,
                neg_j,
                primary.padding_mask,
                use_reentrant=False,
            )

        v_neg = torch.stack(
            [_neg_velocity(neg[:, j].to(dtype=ce_dtype)) for j in range(k)], dim=1
        )  # (B, k, C, H, W)

        # Restore anchor splice state (checkpoint ran each negative forward
        # eagerly to produce its value, mutating the per-step buffers).
        net._step_layer_tokens = anchor_tokens
        net._step_seqlens = anchor_seqlens

        # jaccard mode: subtract α·s from each negative logit (s = caption
        # tag-overlap surfaced by the dataset). No-op for shuffled / hard.
        neg_penalty = None
        if (
            getattr(net, "contrastive_negative_mode", "shuffled") == "jaccard"
            and self._neg_jaccard is not None
        ):
            alpha = float(getattr(net, "contrastive_jaccard_alpha", 1.0) or 0.0)
            neg_penalty = alpha * self._neg_jaccard.to(device).float()

        loss, diag = net.contrastive_loss(
            v_pos, v_neg, v_target, neg_penalty=neg_penalty
        )

        live = float(getattr(net, "_contrastive_weight", 0.0) or 0.0)
        loss_val = float(loss.detach().item())
        self._last_metrics = {
            "reg/soft_tokens_contrastive": loss_val,
            "reg/soft_tokens_contrastive_weighted": live * loss_val,
            "reg/soft_tokens_contrastive_lambda_live": live,
            "soft_tokens/contrastive_acc": diag["contrastive_acc"],
            "soft_tokens/contrastive_logit_gap": diag["contrastive_logit_gap"],
        }
        return {"soft_tokens_contrastive": loss}

    def metrics(self, ctx) -> dict:
        del ctx
        return dict(self._last_metrics)
