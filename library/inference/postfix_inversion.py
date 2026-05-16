"""Per-image inversion of the orthogonal postfix tail.

Probe instrument for ``docs/proposal/postfix_residual_per_image_inversion.md``.
Optimizes a K-dim scale vector ``s`` so that

    ψ = concat(T5(tags), Q @ diag(s))      (S=512, D, last K rows replaced)

minimizes flow-matching loss against the frozen DiT for one image, where
``Q ∈ R^{K×D}`` is a fixed row-orthonormal basis (typically the top-K right
singular vectors of the cached T5 corpus). The K trainable parameters live in
``s``; ``Q`` is frozen.

The body of the loop is lifted from ``archive/inversion/invert_reference.py``
with three changes:

1. Prefix is the actual cached ``{stem}_anima_te.safetensors`` (the tags-
   conditioned T5 output for this image), not an encoded ``--template``.
2. The K trainable slots become ``Q @ diag(s)`` (K params per image) instead
   of a free ``(K, D)`` tensor (K·D params).
3. Splice position is ``end_of_sequence`` — same as the default postfix path.

Not a deployable adapter. The output is the K-vector ``s`` and a loss CSV,
which downstream analysis (PCA / clustering / lane-discipline / multi-seed
functional cosine — see proposal §Probe metrics) consumes.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from networks.methods.postfix import _build_svd_te_basis, _make_orthonormal_basis

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 512


# region Basis


def load_or_build_basis(
    K: int,
    D: int,
    *,
    kind: str = "svd_te",
    te_cache_dir: Optional[str] = None,
    basis_path: Optional[str] = None,
    svd_num_files: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Return a frozen ``(K, D)`` row-orthonormal basis ``Q``.

    Caching contract: if ``basis_path`` exists AND has the right shape AND was
    built for the same ``kind``/``seed`` metadata, load from disk. Otherwise
    compute fresh and save to ``basis_path`` (when set). The SVD over a 256-
    file sample is the expensive piece — caching avoids redoing it across
    sweep modes and seeds.
    """
    if basis_path and os.path.exists(basis_path):
        cached = torch.load(basis_path, map_location="cpu", weights_only=False)
        if isinstance(cached, dict):
            Q = cached.get("basis")
            meta_kind = cached.get("kind")
            meta_seed = cached.get("seed")
        else:
            Q = cached
            meta_kind, meta_seed = None, None
        if Q is not None and Q.shape == (K, D):
            if meta_kind is not None and meta_kind != kind:
                logger.warning(
                    f"basis cache at {basis_path} was built with kind={meta_kind!r} "
                    f"but caller requested {kind!r}; rebuilding"
                )
            elif meta_seed is not None and int(meta_seed) != int(seed):
                logger.warning(
                    f"basis cache at {basis_path} was built with seed={meta_seed} "
                    f"but caller requested {seed}; rebuilding"
                )
            else:
                logger.info(
                    f"Loaded cached basis: {basis_path} (K={K}, D={D}, kind={meta_kind or kind})"
                )
                return Q.float().contiguous()

    if kind == "svd_te":
        if te_cache_dir is None:
            raise ValueError(
                "basis kind 'svd_te' requires te_cache_dir (a directory of cached "
                "_anima_te.safetensors files)"
            )
        Q = _build_svd_te_basis(te_cache_dir, K, D, num_files=svd_num_files, seed=seed)
        logger.info(
            f"Built SVD-of-cached-TE basis: K={K} D={D} from {te_cache_dir} "
            f"(num_files={svd_num_files}, seed={seed})"
        )
    elif kind == "random":
        gen = torch.Generator().manual_seed(int(seed))
        Q = _make_orthonormal_basis(K, D, kind="random", generator=gen)
        logger.info(f"Built random orthonormal basis: K={K} D={D} (seed={seed})")
    else:
        raise ValueError(f"unknown basis kind {kind!r}: expected 'svd_te' or 'random'")

    if basis_path:
        os.makedirs(os.path.dirname(basis_path) or ".", exist_ok=True)
        torch.save(
            {"basis": Q.float().contiguous(), "kind": kind, "seed": int(seed)},
            basis_path,
        )
        logger.info(f"Cached basis to {basis_path}")
    return Q.float().contiguous()


# endregion


# region Prefix


def load_cached_prefix(te_path: str, *, device: torch.device) -> torch.Tensor:
    """Load ``T5(tags)`` prefix from ``{stem}_anima_te.safetensors``.

    Returns a ``(1, MAX_SEQ_LEN, D)`` bf16 tensor, zero-padded to MAX_SEQ_LEN.
    Padding rows (where ``attn_mask_v0`` is False) are zeroed explicitly so the
    splice region is guaranteed-clean.
    """
    sd = load_file(te_path)
    if "crossattn_emb_v0" in sd:
        emb = sd["crossattn_emb_v0"].float()
    elif "crossattn_emb" in sd:
        emb = sd["crossattn_emb"].float()
    else:
        raise KeyError(
            f"{te_path}: missing 'crossattn_emb_v0' / 'crossattn_emb' "
            f"(keys: {list(sd.keys())[:8]}...)"
        )

    if "attn_mask_v0" in sd:
        mask = sd["attn_mask_v0"].bool()
        if mask.shape[0] == emb.shape[0]:
            emb = emb.clone()
            emb[~mask] = 0.0
    if emb.shape[0] < MAX_SEQ_LEN:
        emb = F.pad(emb, (0, 0, 0, MAX_SEQ_LEN - emb.shape[0]))
    elif emb.shape[0] > MAX_SEQ_LEN:
        emb = emb[:MAX_SEQ_LEN]

    return emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16)


def assemble_emb(prefix_emb: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """Concat ``[prefix[:, :S-K], tail]`` — end_of_sequence splice.

    Mirrors ``PostfixNetwork.append_postfix`` for ``splice_position='end_of_sequence'``.
    Autograd flows through ``tail``; prefix is constant.
    """
    K = tail.shape[-2]
    S = prefix_emb.shape[1]
    return torch.cat(
        [prefix_emb[:, : S - K, :], tail.to(prefix_emb.dtype)],
        dim=1,
    )


# endregion


# region Optimization core


def sample_sigmas(
    batch_size: int,
    device: torch.device,
    *,
    sigma_sampling: str = "uniform",
    sigmoid_scale: float = 1.0,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    """Sigma sampler — uniform or sigmoid, with optional P-GRAFT-style low-σ skip.

    Re-uses the rescale-to-[sigma_min, 1.0] convention from
    ``archive/inversion/invert_embedding.py:sample_sigmas``.
    """
    if sigma_sampling == "sigmoid":
        sigmas = torch.sigmoid(sigmoid_scale * torch.randn(batch_size, device=device))
    elif sigma_sampling == "uniform":
        sigmas = torch.rand(batch_size, device=device)
    else:
        raise ValueError(f"unknown sigma_sampling {sigma_sampling!r}")
    lo = max(0.0, min(1.0, sigma_min))
    if lo > 0.0:
        sigmas = lo + (1.0 - lo) * sigmas
    return sigmas


def fm_loss_step(
    anima,
    latents: torch.Tensor,
    emb_full: torch.Tensor,
    sigmas: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """One flow-matching loss eval — identical math to invert_reference.py."""
    n_t = sigmas.shape[0]
    lat = latents.expand(n_t, -1, -1, -1)
    noise = torch.randn_like(lat)
    sv = sigmas.view(-1, 1, 1, 1)
    noisy = (1.0 - sv) * lat + sv * noise
    noisy_5d = noisy.to(torch.bfloat16).unsqueeze(2)
    emb = emb_full.expand(n_t, -1, -1)
    pm = padding_mask.expand(n_t, -1, -1, -1)
    timesteps = sigmas.to(torch.bfloat16)
    pred = anima(noisy_5d, timesteps, emb, padding_mask=pm).squeeze(2)
    target = noise - lat
    return F.mse_loss(pred.float(), target.float())


# endregion


# region Public entrypoint


@dataclass
class TailInversionConfig:
    """Per-image optimization knobs. Defaults match the proposal."""

    K: int = 48
    steps: int = 100
    lr: float = 0.01
    lr_schedule: str = "cosine"  # "cosine" or "constant"
    grad_accum: int = 4
    timesteps_per_step: int = 1
    sigma_sampling: str = "uniform"  # "uniform" or "sigmoid"
    sigmoid_scale: float = 1.0
    sigma_min: float = 0.0
    lambda_zero: float = 0.0  # ‖s‖² regularization weight
    init_std: float = 0.0  # 0 → zero-init; >0 → N(0, init_std²)
    log_every: int = 10
    metadata: dict = field(default_factory=dict)


@dataclass
class TailInversionResult:
    """Output of one inversion run — the things the analyzer reads."""

    s: torch.Tensor  # (K,) float32, on CPU
    best_loss: float  # best (fm_loss + reg) across optimization steps
    best_fm_loss: float  # fm component at the best-loss step
    best_step: int
    final_s_l2: float
    history: list[dict] = field(default_factory=list)


def invert_tail(
    anima,
    *,
    prefix_emb: torch.Tensor,
    latents: torch.Tensor,
    basis_Q: torch.Tensor,
    config: TailInversionConfig,
    device: torch.device,
    seed: int = 0,
    log_path: Optional[str] = None,
) -> TailInversionResult:
    """Optimize ``s`` for one image. Returns the best-loss ``s`` and diagnostics.

    Args:
        anima: frozen DiT (already ``requires_grad_(False)``-ed; ``torch.compile``
            on the outside is fine — shapes are static for fixed K + image size).
        prefix_emb: ``(1, MAX_SEQ_LEN, D)`` bf16 cached T5 output for this image.
        latents: ``(B=1, C, H/8, W/8)`` bf16 cached VAE latents for this image.
        basis_Q: ``(K, D)`` row-orthonormal fp32 basis (frozen). Caller is
            responsible for matching ``K == config.K``.
        config: optimization knobs.
        device: cuda or cpu.
        seed: RNG seed for sigma sampling + ``s`` init.
        log_path: optional CSV path for per-step loss / s‖²/ grad norm.
    """
    cfg = config
    K, D = basis_Q.shape
    if K != cfg.K:
        raise ValueError(f"basis_Q rows ({K}) != config.K ({cfg.K})")
    if prefix_emb.shape[-1] != D:
        raise ValueError(
            f"prefix_emb dim ({prefix_emb.shape[-1]}) != basis_Q dim ({D})"
        )

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    if cfg.init_std <= 0.0:
        s_init = torch.zeros(K, dtype=torch.float32, device=device)
    else:
        s_init = torch.randn(K, dtype=torch.float32, device=device) * cfg.init_std
    s = torch.nn.Parameter(s_init)

    optimizer = torch.optim.AdamW([s], lr=cfg.lr, weight_decay=0.0)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.01
        )
        if cfg.lr_schedule == "cosine"
        else None
    )

    Q = basis_Q.to(device=device, dtype=torch.float32).contiguous()

    h_lat, w_lat = latents.shape[-2], latents.shape[-1]
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step",
                "loss",
                "fm_loss",
                "reg",
                "best_loss",
                "lr",
                "grad_norm",
                "s_l2",
            ],
        )
        csv_writer.writeheader()

    best_loss = float("inf")
    best_fm_loss = float("inf")
    best_s: Optional[torch.Tensor] = None
    best_step = 0
    history: list[dict] = []

    pbar = tqdm(range(cfg.steps), desc=f"Inverting tail K={K}", leave=False)
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_fm = 0.0
        accum_reg = 0.0
        for _ in range(cfg.grad_accum):
            sigmas = sample_sigmas(
                cfg.timesteps_per_step,
                device,
                sigma_sampling=cfg.sigma_sampling,
                sigmoid_scale=cfg.sigmoid_scale,
                sigma_min=cfg.sigma_min,
            )
            # tail = Q · diag(s) computed in fp32, cast to bf16 inside assemble_emb
            tail = (Q * s.unsqueeze(-1)).unsqueeze(0)  # (1, K, D) fp32
            emb_full = assemble_emb(prefix_emb, tail)
            fm_loss = fm_loss_step(anima, latents, emb_full, sigmas, padding_mask)
            if cfg.lambda_zero > 0.0:
                reg = cfg.lambda_zero * s.pow(2).sum()
            else:
                reg = torch.zeros((), device=device, dtype=fm_loss.dtype)
            loss = fm_loss + reg
            (loss / cfg.grad_accum).backward()
            accum_loss += loss.item()
            accum_fm += fm_loss.item()
            accum_reg += float(reg.detach().item())

        grad_norm = torch.nn.utils.clip_grad_norm_([s], max_norm=1.0).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = accum_loss / cfg.grad_accum
        fm_val = accum_fm / cfg.grad_accum
        reg_val = accum_reg / cfg.grad_accum
        s_l2 = float(s.detach().norm().item())
        lr_now = optimizer.param_groups[0]["lr"]

        if loss_val < best_loss:
            best_loss = loss_val
            best_fm_loss = fm_val
            best_s = s.detach().clone().cpu()
            best_step = step

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            pbar.set_postfix(
                loss=f"{loss_val:.6f}",
                best=f"{best_loss:.6f}",
                s_l2=f"{s_l2:.3f}",
                lr=f"{lr_now:.2e}",
            )

        history.append(
            {
                "step": step,
                "loss": loss_val,
                "fm_loss": fm_val,
                "reg": reg_val,
                "best_loss": best_loss,
                "lr": lr_now,
                "grad_norm": grad_norm,
                "s_l2": s_l2,
            }
        )
        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "step": step,
                    "loss": f"{loss_val:.6f}",
                    "fm_loss": f"{fm_val:.6f}",
                    "reg": f"{reg_val:.6f}",
                    "best_loss": f"{best_loss:.6f}",
                    "lr": f"{lr_now:.2e}",
                    "grad_norm": f"{grad_norm:.6f}",
                    "s_l2": f"{s_l2:.6f}",
                }
            )

    if csv_file is not None:
        csv_file.close()

    assert best_s is not None, "optimization produced no best_s"
    return TailInversionResult(
        s=best_s,
        best_loss=best_loss,
        best_fm_loss=best_fm_loss,
        best_step=best_step,
        final_s_l2=float(s.detach().norm().item()),
        history=history,
    )


# endregion


# region IO


def save_tail_s(
    save_path: str,
    s: torch.Tensor,
    *,
    K: int,
    D: int,
    basis_kind: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save a single per-image ``s`` vector as a safetensors file.

    Schema is intentionally minimal — analysis works on ``s`` alone, and the
    basis is reconstructed from the cached file at analyzer time. Metadata
    pins the run config so a downstream consumer can verify (K, basis kind,
    optimization budget, etc.) match expectations.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    state = {"s": s.detach().clone().cpu().float().contiguous()}
    meta = {
        "ss_artifact": "postfix_tail_s",
        "ss_K": str(K),
        "ss_D": str(D),
        "ss_basis_kind": basis_kind,
    }
    if metadata:
        for k, v in metadata.items():
            meta[str(k)] = str(v)
    save_file(state, save_path, metadata=meta)


def load_tail_s(s_path: str) -> tuple[torch.Tensor, dict]:
    """Inverse of ``save_tail_s``: returns ``(s: (K,) fp32, metadata: dict)``."""
    from safetensors import safe_open

    with safe_open(s_path, framework="pt") as f:
        s = f.get_tensor("s").float().contiguous()
        meta = f.metadata() or {}
    return s, dict(meta)


def splice_tail_into_te_cache(
    te_in_path: str,
    te_out_path: str,
    *,
    s: torch.Tensor,
    Q: torch.Tensor,
    variant_index: int = 0,
) -> None:
    """Bake ``Q @ diag(s)`` into the last K rows of a cached T5 prefix.

    Reads ``{te_in_path}``'s ``crossattn_emb_v{variant_index}`` (or
    ``crossattn_emb`` for single-variant caches), overwrites positions
    ``[S-K, S)`` with the postfix tail, and writes a minimal single-variant
    cache to ``{te_out_path}``. The output schema is just ``crossattn_emb``
    — exactly what ``edit.py``'s ``--cached_embed`` single-variant branch
    expects, so dry-mode reconstruction picks up the spliced prefix without
    any downstream code changes.

    Args:
        te_in_path: Source ``{stem}_anima_te.safetensors`` cache.
        te_out_path: Destination — created fresh, single-variant.
        s: ``(K,)`` scales, fp32 or bf16, on any device.
        Q: ``(K, D)`` row-orthonormal basis matching the original inversion.
        variant_index: Which cached variant to splice into. v0 (pristine
            caption) is what dry-mode normally uses.
    """
    from safetensors import safe_open

    K, D = Q.shape
    if s.shape != (K,):
        raise ValueError(f"s shape {tuple(s.shape)} != (K={K},)")

    with safe_open(te_in_path, framework="pt") as f:
        keys = set(f.keys())
        has_variants = "num_variants" in keys
        if has_variants:
            suf = f"_v{variant_index}"
            crossattn_key = f"crossattn_emb{suf}"
        else:
            crossattn_key = "crossattn_emb"
        if crossattn_key not in keys:
            raise KeyError(
                f"{te_in_path}: missing {crossattn_key!r} "
                f"(keys: {sorted(keys)[:8]}...)"
            )
        emb = f.get_tensor(crossattn_key).float()  # (S, D)

    S = emb.shape[0]
    if emb.shape[-1] != D:
        raise ValueError(
            f"cached emb dim {emb.shape[-1]} != basis Q dim {D} (file: {te_in_path})"
        )
    if S < K:
        raise ValueError(f"cached emb seq len {S} < K={K} (file: {te_in_path})")

    tail = (Q.float().cpu() * s.float().cpu().unsqueeze(-1))  # (K, D)
    spliced = emb.clone()
    spliced[S - K : S, :] = tail
    spliced_bf16 = spliced.to(torch.bfloat16).contiguous()

    os.makedirs(os.path.dirname(te_out_path) or ".", exist_ok=True)
    save_file(
        {"crossattn_emb": spliced_bf16},
        te_out_path,
        metadata={
            "ss_artifact": "postfix_tail_spliced_te_cache",
            "ss_source_te": os.path.basename(te_in_path),
            "ss_variant_index": str(variant_index),
            "ss_K": str(K),
            "ss_D": str(D),
            "ss_splice_position": "end_of_sequence",
        },
    )


# endregion
