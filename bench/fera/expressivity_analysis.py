"""Analytical FeRA vs OrthoHydra expressivity comparison — training-free.

WHAT THIS SCRIPT MEASURES AND WHY
=================================

The question
------------
"Is OrthoHydra structurally bottlenecked compared to FeRA's independent-A
architecture for the styles we already care about?" — answered without
training a single new model. We treat existing trained Anima LoRAs as
*proxies for what the dataset asks the model to learn* and ask, for each
parameterization, how close it can get to those targets within its
reachable manifold.

The three handles
-----------------

H1. SUBSPACE PROJECTION ERROR  — load-bearing test
    For each LoRA module, compute the Frobenius residual after projecting
    the trained ΔW = (alpha/r)·B·A onto each parameterization's reachable
    set:

      r_FeRA(R_F)     = ‖ΔW − SVD_{R_F}(ΔW)‖_F / ‖ΔW‖_F

         FeRA's reachable manifold is Σ_k g_k · B_k A_k with free
         (A_k, B_k), so the best rank-R_F approximation is the Eckart–
         Young truncation of ΔW. Default R_F = R (matched-rank to OH,
         so the only thing left between r_FeRA and r_OH is OH's subspace
         constraint). Pass --fera-rank-multiplier M_F to set R_F = M_F·R
         and recover the paper's "total FeRA reachable rank = M·R" budget
         — but note: trained-LoRA proxies have rank ≤ R_train, so any
         M_F ≥ 1 with R_F ≥ R_train forces r_FeRA to numerical zero
         (degenerate comparison).

      r_OH(M, R_OH)   = ‖ΔW − Π_OH(ΔW)‖_F / ‖ΔW‖_F

         OrthoHydra(`OrthoHydraLoRAModule`) reachable operators are
         constrained to:
           - col space ⊂ span(U_base[:, :M·R_OH])    (disjoint per-expert
             slices of the pretrained W's SVD — see ortho.py:240-269)
           - row space ⊂ span(V_base[:, :R_OH])      (shared Q basis)
           - rank ≤ R_OH                              (shared diag λ)
         The row-space constraint alone already enforces rank ≤ R_OH,
         so Π_OH(ΔW) = U_top·U_topᵀ · ΔW · V_top·V_topᵀ has rank ≤ R
         by construction and no separate rank truncation is needed.

    Read the gap r_OH − r_FeRA per layer class at matched rank:
      - r_OH ≈ r_FeRA       → pretrained W's SVD already spans the
                              directions trained styles use within rank R;
                              OrthoHydra's frozen-basis bet is justified.
      - r_OH ≫ r_FeRA       → trained styles want directions outside
                              the pretrained W's column/row space;
                              OrthoHydra is bottlenecking.

    Per-layer-class aggregation matters — a global mean hides "attention
    needs FeRA, MLP is fine with OrthoHydra" patterns.

H2. CROSS-STYLE PRINCIPAL ANGLES  — Tian et al.'s shared-A assumption
    HydraLoRA (Tian et al. NeurIPS'24) shares lora_down across experts
    on the empirical claim that during multi-task LoRA training, A's
    cluster across tasks while B's diverge. Test it on your existing
    LoRA zoo.

    For each pair (i, j) of LoRAs that share a module name and rank:
      angle_A_deg = mean principal angle between row-space of A_i and
                    row-space of A_j  (each A: R × D_in)
      angle_B_deg = mean principal angle between col-space of B_i and
                    col-space of B_j  (each B: D_out × R)

    Read:
      - mean(angle_A) ≪ mean(angle_B)  → shared-A is empirically valid;
                                         the HydraLoRA architecture matches
                                         Anima's substyle variance pattern.
      - mean(angle_A) ≈ mean(angle_B)  → Tian's clustering doesn't hold;
                                         FeRA's independent-A captures
                                         real per-style input structure.

H3. INTRINSIC EFFECTIVE RANK  — sanity-check denominator
    Participation ratio of ΔW's singular spectrum, plus top-R singular
    mass fraction. If a trained LoRA is heavily over-parameterized
    (effective rank ≪ R_train), both r_FeRA and r_OH will be near zero
    and the H1 gap loses signal. Report alongside to know when to trust
    H1.

What this script CANNOT tell you
--------------------------------
- Routing dynamics. A larger reachable manifold is only useful if SGD
  finds the right point in it. OrthoHydra's smaller manifold may train
  *faster* to a worse-but-acceptable solution.
- Generalization. More expressive families also overfit more easily.
- Composability with T-LoRA's rank mask or ReFT's residual-stream
  intervention — those further constrain the search space.
- Whether the styles in your *training data* are harder than the existing
  LoRA targets used as proxies. If your zoo is too narrow, H1's verdict
  doesn't transfer to the broader use case.

Interpretation cheat sheet
--------------------------
| Observation                                | Implication                                            |
|--------------------------------------------|--------------------------------------------------------|
| r_OH / r_FeRA ≈ 1 across layers            | Ship FEI router on shared-A; don't touch the experts.  |
| r_OH / r_FeRA > 2 in attention, ≈ 1 in MLP | Mixed: independent-A for attention, shared-A elsewhere.|
| r_OH / r_FeRA ≫ 1 everywhere               | OrthoHydra bottlenecking; FeRA+FECL adoption is on.    |
| mean_angle(A) ≪ mean_angle(B)              | Tian et al. holds → shared-A is right architecturally. |
| mean_angle(A) ≈ mean_angle(B)              | Tian fails → independent-A captures real signal.       |

Usage
-----

    python -m bench.fera.expressivity_analysis \\
        --lora output/ckpt/anima-hydra-0511-4812.safetensors \\
        --lora output/ckpt/anima-tlora-0509-12.safetensors \\
        --base-dit models/diffusion_models/anima-base-v1.0.safetensors \\
        --num-experts 4 \\
        --label hydra-vs-tlora

Outputs land in ``bench/fera/results/<YYYYMMDD-HHMM>[-<label>]/``:

    result.json           — standard bench envelope (bench/_common.py)
    per_module.csv        — one row per (lora, module): r_FeRA, r_OH, ratio,
                            effective_rank, top-R singular mass
    principal_angles.csv  — pairwise (lora_i, lora_j, module): angle_A_deg,
                            angle_B_deg  (only if ≥2 LoRAs given)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

# Make bench._common importable when invoked via `python -m` or directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench._common import make_run_dir, write_result  # noqa: E402


LORA_PREFIX = "lora_unet_"
DOWN_SUFFIX = ".lora_down.weight"
UP_SUFFIX = ".lora_up.weight"
ALPHA_SUFFIX = ".alpha"


@dataclass
class LoRAModule:
    down: torch.Tensor   # (R, D_in)
    up: torch.Tensor     # (D_out, R)
    alpha: float         # scalar; LoRA scale = alpha / R


@dataclass
class LoadedLoRA:
    path: Path
    modules: dict[str, LoRAModule]   # underscored module key → tensors


def _build_base_index(base_state: dict[str, torch.Tensor]) -> dict[str, str]:
    """Map underscored module path → original (dotted) key in base state dict.

    Anima's DiT keys are prefixed with ``net.`` (e.g. ``net.blocks.0.cross_attn.q_proj``);
    sd-scripts LoRA keys drop the prefix and replace dots with underscores
    (``lora_unet_blocks_0_cross_attn_q_proj``). We strip ``net.`` before
    underscore-encoding so the round-trip resolves.
    """
    index: dict[str, str] = {}
    for key in base_state:
        if not key.endswith(".weight"):
            continue
        mod_path = key[: -len(".weight")]
        stripped = mod_path[len("net."):] if mod_path.startswith("net.") else mod_path
        index[stripped.replace(".", "_")] = mod_path
    return index


def _load_lora(path: Path) -> LoadedLoRA:
    """Group keys into ``{module_name: LoRAModule}``."""
    raw = load_file(str(path))
    by_module: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in raw.items():
        if not k.startswith(LORA_PREFIX):
            continue
        body = k[len(LORA_PREFIX):]
        if body.endswith(DOWN_SUFFIX):
            name = body[: -len(DOWN_SUFFIX)]
            by_module.setdefault(name, {})["down"] = v
        elif body.endswith(UP_SUFFIX):
            name = body[: -len(UP_SUFFIX)]
            by_module.setdefault(name, {})["up"] = v
        elif body.endswith(ALPHA_SUFFIX):
            name = body[: -len(ALPHA_SUFFIX)]
            by_module.setdefault(name, {})["alpha"] = v

    modules: dict[str, LoRAModule] = {}
    for name, parts in by_module.items():
        if "down" not in parts or "up" not in parts:
            continue   # MoE up-stack and other non-standard keys are skipped
        alpha_t = parts.get("alpha")
        alpha = float(alpha_t.item()) if alpha_t is not None else float(parts["down"].shape[0])
        modules[name] = LoRAModule(down=parts["down"], up=parts["up"], alpha=alpha)
    return LoadedLoRA(path=path, modules=modules)


def _classify_module(dotted_key: str) -> str:
    """Coarse layer-class buckets for aggregation. Matches Anima DiT naming."""
    if "cross_attn" in dotted_key:
        return "cross_attn"
    if "self_attn" in dotted_key:
        return "self_attn"
    if re.search(r"mlp\.layer[12]", dotted_key):
        return "mlp"
    if "adaln_modulation" in dotted_key:
        return "adaln"
    return "other"


def _effective_rank(singular_values: torch.Tensor) -> float:
    """Participation ratio of the singular spectrum: (Σ σ)² / Σ σ².

    A rank-1 operator has effective_rank=1; a fully-uniform spectrum of length
    R has effective_rank=R. Robust to scaling. Used as a sanity-check
    denominator — if effective_rank ≪ R_train, the trained LoRA was
    over-parameterized and H1 gaps lose signal.
    """
    s = singular_values.double()
    num = s.sum().pow(2)
    den = s.pow(2).sum().clamp_min(1e-30)
    return (num / den).item()


def _principal_angles_rad(M1: torch.Tensor, M2: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians) between the row-spaces of two (r, d) matrices.

    Standard QR-then-SVD construction (Björck & Golub 1973). Returns
    ``min(r1, r2)`` angles sorted ascending in [0, π/2] — works for mismatched
    rank inputs, in which case the angles characterize how well the smaller
    row-space fits inside the larger. NaN-safe: numerical drift past 1.0 is
    clamped.
    """
    Q1, _ = torch.linalg.qr(M1.T)   # (d, r1)
    Q2, _ = torch.linalg.qr(M2.T)   # (d, r2)
    cosines = torch.linalg.svdvals(Q1.T @ Q2).clamp(-1.0, 1.0)
    return torch.arccos(cosines)


def base_svd_topk(base_w: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k left/right singular vectors of ``base_w`` via randomized SVD.

    Returns (U_top: (out, k), V_top: (in, k)). We only need the top M·R
    columns of U and top R columns of V for the OrthoHydra projection, so
    ``svd_lowrank`` (q=k+oversample, 2 power iters) is ~10–100× faster than
    a full ``torch.linalg.svd`` at the layer sizes we hit (~1k–3k).
    Accuracy on the top-k factors is near machine precision.
    """
    q = min(k + 10, min(base_w.shape))
    U, _S, V = torch.svd_lowrank(base_w, q=q, niter=2)
    return U[:, :k].contiguous(), V[:, :k].contiguous()


def analyze_module(
    delta_w: torch.Tensor,
    U_top: torch.Tensor,
    V_top: torch.Tensor,
    num_experts: int,
    rank: int,
    fera_rank_multiplier: int = 1,
) -> dict[str, float | int]:
    """Compute H1 + H3 metrics for one (LoRA module, base weight) pair.

    Math summary (see top-of-file docstring for context):
      R_F = fera_rank_multiplier * rank        # default = rank (matched-rank)
      r_FeRA = ‖ΔW − rank-R_F-trunc(ΔW)‖_F / ‖ΔW‖_F
      r_OH  = ‖ΔW − U_top·U_topᵀ·ΔW·V_top·V_topᵀ‖_F / ‖ΔW‖_F
        with U_top = U_base[:, :M·R_OH], V_top = V_base[:, :R_OH].

    Π_OH already has rank ≤ rank(V_top) ≤ R_OH, so no separate rank
    truncation is applied — V_top's column count enforces the cap.

    ``U_top`` and ``V_top`` are passed in (computed once per base module,
    reused across LoRAs); see ``base_svd_topk``.
    """
    out_dim, in_dim = delta_w.shape
    norm_d = torch.linalg.norm(delta_w).clamp_min(1e-30)
    fera_rank = fera_rank_multiplier * rank
    oh_basis_width = num_experts * rank  # U_top column budget for OH

    # SVD of the trained delta — thin SVD on a low-rank operator is cheap.
    s_delta = torch.linalg.svdvals(delta_w)
    delta_rank = int((s_delta > s_delta[0] * 1e-6).sum().item())

    # H1a: FeRA reachable = best rank-R_F approximation. By Eckart-Young the
    # residual ‖ΔW − ΔW_k‖_F² = Σ_{i>k} σ_i². With the default
    # matched-rank setting (R_F = rank), this captures the rank-R truncation
    # penalty an unconstrained low-rank approximation would pay — the floor
    # OrthoHydra needs to clear to look "as good as" rank-R LoRA.
    if fera_rank >= s_delta.numel():
        r_fera = torch.tensor(0.0, device=delta_w.device)
    else:
        r_fera = s_delta[fera_rank:].pow(2).sum().sqrt() / norm_d

    # H1b: OrthoHydra reachable. Π(ΔW) = U_top (U_topᵀ ΔW V_top) V_topᵀ.
    # Since V_top has only R columns, Π has rank ≤ R automatically — no
    # extra SVD-truncation step is needed.
    proj = U_top @ (U_top.T @ delta_w @ V_top) @ V_top.T
    r_oh = torch.linalg.norm(delta_w - proj) / norm_d

    eff_rank = _effective_rank(s_delta)
    top_R_mass = (s_delta[:rank].sum() / s_delta.sum().clamp_min(1e-30)).item()

    return {
        "out_dim": int(out_dim),
        "in_dim": int(in_dim),
        "R": int(rank),
        "M": int(num_experts),
        "fera_rank": int(fera_rank),
        "oh_basis_width": int(oh_basis_width),
        "delta_rank": delta_rank,
        "r_FeRA": float(r_fera.item()),
        "r_OH": float(r_oh.item()),
        "ratio_OH_over_FeRA": float((r_oh / r_fera.clamp_min(1e-12)).item()),
        "effective_rank": eff_rank,
        "top_R_singular_mass": top_R_mass,
    }


def _aggregate_by_class(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean of r_FeRA / r_OH / ratio per module class across all LoRAs."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["module_class"], []).append(row)
    out: dict[str, dict[str, float]] = {}
    for cls, items in buckets.items():
        out[cls] = {
            "n_modules": len(items),
            "r_FeRA_mean": sum(r["r_FeRA"] for r in items) / len(items),
            "r_OH_mean": sum(r["r_OH"] for r in items) / len(items),
            "ratio_OH_over_FeRA_mean": sum(r["ratio_OH_over_FeRA"] for r in items) / len(items),
            "effective_rank_mean": sum(r["effective_rank"] for r in items) / len(items),
        }
    return out


def _aggregate_angles(rows: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["module_class"], []).append(row)
    out: dict[str, dict[str, float]] = {}
    for cls, items in buckets.items():
        out[cls] = {
            "n_pairs": len(items),
            "angle_A_deg_mean": sum(r["angle_A_deg"] for r in items) / len(items),
            "angle_B_deg_mean": sum(r["angle_B_deg"] for r in items) / len(items),
            "ratio_A_over_B": (
                (sum(r["angle_A_deg"] for r in items) / len(items))
                / max(sum(r["angle_B_deg"] for r in items) / len(items), 1e-9)
            ),
        }
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analytical FeRA vs OrthoHydra expressivity comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--lora",
        action="append",
        required=True,
        type=Path,
        help="Path to a trained LoRA safetensors (repeatable; ≥2 enables H2).",
    )
    p.add_argument(
        "--base-dit",
        required=True,
        type=Path,
        help="Path to the base DiT safetensors (e.g. models/diffusion_models/anima-base-v1.0.safetensors).",
    )
    p.add_argument(
        "--num-experts",
        type=int,
        default=4,
        help="M in OrthoHydra's M·R col-subspace width (default: 4, matches fera.toml).",
    )
    p.add_argument(
        "--rank-override",
        type=int,
        default=None,
        help="If set, force R=rank for all modules instead of using each LoRA's intrinsic rank.",
    )
    p.add_argument(
        "--fera-rank-multiplier",
        type=int,
        default=1,
        help=(
            "M_F so FeRA reachable rank R_F = M_F·R. Default 1 = matched-rank "
            "comparison to OrthoHydra (the only meaningful setting on trained-LoRA "
            "proxies — see top-of-file H1). M_F=num_experts recovers the paper's "
            "M·R FeRA budget but forces r_FeRA→0 for any LoRA with R_train ≤ M·R."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--label", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Lazy load base weights once. State dict is bf16 on disk; we promote to
    # fp32 per-tensor inside analyze_module to keep peak memory low.
    base_state = load_file(str(args.base_dit))
    base_index = _build_base_index(base_state)
    print(f"Base DiT: {len(base_index)} weight modules indexed.")

    loras = [_load_lora(p) for p in args.lora]
    for lora in loras:
        print(f"LoRA {lora.path.name}: {len(lora.modules)} modules.")

    # H1: per-(LoRA, module) projection error.
    # Base-W SVD is cached per module across LoRAs — each (D_out × D_in) SVD
    # is the dominant cost, and at 280 modules × N LoRAs the redundant recompute
    # was the timeout source. Cache key is (base_path, k) where k = max(M·R)
    # we'll ever ask for at that module — usually constant.
    per_module_rows: list[dict] = []
    skipped_no_base = 0
    svd_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    total_pairs = sum(len(l.modules) for l in loras)
    pbar = tqdm(total=total_pairs, desc="H1 projection", unit="mod")
    for lora_idx, lora in enumerate(loras):
        pbar.set_postfix_str(lora.path.name)
        for mod_key, mod in lora.modules.items():
            pbar.update(1)
            base_path = base_index.get(mod_key)
            if base_path is None:
                skipped_no_base += 1
                continue
            down = mod.down.to(device=device, dtype=torch.float32)
            up = mod.up.to(device=device, dtype=torch.float32)
            R = down.shape[0]
            R_use = args.rank_override or R
            scale = mod.alpha / R
            delta_w = (up @ down) * scale            # (out, in)

            # Cache hit: U_top has at least M·R cols, V_top has at least R cols.
            cached = svd_cache.get(base_path)
            need_k = args.num_experts * R_use
            if cached is None or cached[0].shape[1] < need_k or cached[1].shape[1] < R_use:
                base_w = base_state[base_path + ".weight"].to(device=device, dtype=torch.float32)
                if delta_w.shape != base_w.shape:
                    skipped_no_base += 1
                    continue
                # Always compute enough cols for max(M·R, R) — same as need_k since M≥1.
                U_top_full, V_top_full = base_svd_topk(base_w, k=need_k)
                svd_cache[base_path] = (U_top_full, V_top_full)
                cached = svd_cache[base_path]

            U_top = cached[0][:, :need_k]
            V_top = cached[1][:, :R_use]

            metrics = analyze_module(
                delta_w=delta_w,
                U_top=U_top,
                V_top=V_top,
                num_experts=args.num_experts,
                rank=R_use,
                fera_rank_multiplier=args.fera_rank_multiplier,
            )
            metrics["lora"] = lora.path.name
            metrics["lora_idx"] = lora_idx
            metrics["module"] = base_path
            metrics["module_class"] = _classify_module(base_path)
            per_module_rows.append(metrics)

    pbar.close()
    print(f"H1: {len(per_module_rows)} (LoRA × module) rows; skipped {skipped_no_base}.")

    # H2: cross-LoRA principal angles (only for shared modules with matching rank).
    angle_rows: list[dict] = []
    if len(loras) >= 2:
        pairs = list(itertools.combinations(range(len(loras)), 2))
        total_shared = sum(
            len(loras[i].modules.keys() & loras[j].modules.keys()) for i, j in pairs
        )
        pbar2 = tqdm(total=total_shared, desc="H2 principal angles", unit="pair")
        for i, j in pairs:
            mods_i, mods_j = loras[i].modules, loras[j].modules
            for mod_key in mods_i.keys() & mods_j.keys():
                pbar2.update(1)
                mi, mj = mods_i[mod_key], mods_j[mod_key]
                # Require same in/out dims (same module in pretrained model);
                # ranks may differ — principal angles are well-defined between
                # subspaces of different dimension (min(R_i, R_j) angles).
                if mi.down.shape[1] != mj.down.shape[1] or mi.up.shape[0] != mj.up.shape[0]:
                    continue
                base_path = base_index.get(mod_key, mod_key)
                Ai = mi.down.to(device=device, dtype=torch.float32)
                Aj = mj.down.to(device=device, dtype=torch.float32)
                Bi = mi.up.T.to(device=device, dtype=torch.float32)   # (R, D_out)
                Bj = mj.up.T.to(device=device, dtype=torch.float32)
                ang_A = _principal_angles_rad(Ai, Aj)
                ang_B = _principal_angles_rad(Bi, Bj)
                R_i = int(Ai.shape[0])
                R_j = int(Aj.shape[0])
                angle_rows.append(
                    {
                        "lora_i": loras[i].path.name,
                        "lora_j": loras[j].path.name,
                        "module": base_path,
                        "module_class": _classify_module(base_path),
                        "rank_i": R_i,
                        "rank_j": R_j,
                        "rank_min": min(R_i, R_j),
                        "angle_A_deg": float(ang_A.mean().item() * 180.0 / math.pi),
                        "angle_B_deg": float(ang_B.mean().item() * 180.0 / math.pi),
                    }
                )
        pbar2.close()
        print(f"H2: {len(angle_rows)} cross-LoRA module pairs compared.")

    # Aggregate and write outputs
    aggregate_h1 = _aggregate_by_class(per_module_rows)
    aggregate_h2 = _aggregate_angles(angle_rows) if angle_rows else {}

    run_dir = make_run_dir("fera", label=args.label)
    _write_csv(run_dir / "per_module.csv", per_module_rows)
    artifacts = ["per_module.csv"]
    if angle_rows:
        _write_csv(run_dir / "principal_angles.csv", angle_rows)
        artifacts.append("principal_angles.csv")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={
            "n_loras": len(loras),
            "n_modules_per_lora": [len(l.modules) for l in loras],
            "h1_aggregate_by_class": aggregate_h1,
            "h2_aggregate_by_class": aggregate_h2,
        },
        artifacts=artifacts,
        device=device,
    )

    # Console summary for quick read.
    print(f"\nResults: {run_dir}")
    print("\n=== H1: r_OH / r_FeRA by module class ===")
    print(f"{'class':<14} {'n':>6} {'r_FeRA':>10} {'r_OH':>10} {'ratio':>8} {'eff_rank':>10}")
    for cls, m in sorted(aggregate_h1.items()):
        print(
            f"{cls:<14} {m['n_modules']:>6} {m['r_FeRA_mean']:>10.4f} "
            f"{m['r_OH_mean']:>10.4f} {m['ratio_OH_over_FeRA_mean']:>8.2f} "
            f"{m['effective_rank_mean']:>10.2f}"
        )
    if aggregate_h2:
        print("\n=== H2: principal angles by module class (degrees) ===")
        print(f"{'class':<14} {'n_pairs':>8} {'∠A':>8} {'∠B':>8} {'A/B':>8}")
        for cls, m in sorted(aggregate_h2.items()):
            print(
                f"{cls:<14} {m['n_pairs']:>8} {m['angle_A_deg_mean']:>8.2f} "
                f"{m['angle_B_deg_mean']:>8.2f} {m['ratio_A_over_B']:>8.2f}"
            )


if __name__ == "__main__":
    main()
