"""Router diagnostics and metrics helpers for the LoRA-family network."""

import math
from typing import Dict, List, Optional, Union

import torch

from library.training.metrics import MetricContext


def step_balance_loss_warmup(network, global_step: int, max_train_steps: int) -> None:
    target = float(getattr(network, "_balance_loss_target_weight", 0.0) or 0.0)
    ratio = float(getattr(network, "_balance_loss_warmup_ratio", 0.0) or 0.0)
    if ratio <= 0.0 or max_train_steps <= 0 or target <= 0.0:
        return
    warmup_steps = int(max_train_steps * ratio)
    network._balance_loss_weight = 0.0 if global_step < warmup_steps else target


def switch_balance(gate: torch.Tensor) -> torch.Tensor:
    num_experts = gate.shape[-1]
    expert_idx = gate.argmax(dim=-1)
    frac = torch.zeros(num_experts, device=gate.device, dtype=gate.dtype)
    frac.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=gate.dtype))
    frac = frac / gate.shape[0]
    gate_mean = gate.mean(dim=0)
    return num_experts * (frac * gate_mean).sum()


def get_balance_loss(network) -> torch.Tensor:
    if getattr(network, "_use_chimera_hydra", False):
        return get_chimera_balance_loss(network)

    total = None
    per_bucket_total = None
    count = 0
    per_bucket_count = 0

    sigma = network._last_sigma
    num_buckets = network.cfg.num_sigma_buckets
    bucket_w = float(network.cfg.per_bucket_balance_weight or 0.0)
    want_per_bucket = (
        network.cfg.router_source == "sigma"
        and sigma is not None
        and num_buckets > 1
        and bucket_w > 0.0
    )
    if want_per_bucket:
        thresholds = torch.linspace(0.0, 1.0, num_buckets + 1, device=sigma.device)[
            1:-1
        ]
        bucket_ids = torch.bucketize(sigma.float(), thresholds)

    for lora in network.unet_loras + network.text_encoder_loras:
        gate = getattr(lora, "_last_gate", None)
        if gate is None:
            continue
        term = switch_balance(gate)
        total = term if total is None else total + term
        count += 1

        if want_per_bucket and getattr(lora, "sigma_feature_dim", 0) > 0:
            module_bucket_sum = None
            module_bucket_count = 0
            for b in range(num_buckets):
                mask = bucket_ids == b
                if int(mask.sum()) < 2:
                    continue
                bterm = switch_balance(gate[mask])
                module_bucket_sum = (
                    bterm if module_bucket_sum is None else module_bucket_sum + bterm
                )
                module_bucket_count += 1
            if module_bucket_sum is not None:
                per_bucket_total = (
                    module_bucket_sum / module_bucket_count
                    if per_bucket_total is None
                    else per_bucket_total + module_bucket_sum / module_bucket_count
                )
                per_bucket_count += 1

    if total is None:
        return torch.tensor(0.0)
    out = total / count
    if per_bucket_total is not None and per_bucket_count > 0:
        out = out + bucket_w * (per_bucket_total / per_bucket_count)
    return out


def get_chimera_balance_loss(network) -> torch.Tensor:
    K_c_default = int(getattr(network.cfg, "num_experts_content", 0))
    total_c = None
    total_f = None
    count_c = 0
    count_f = 0
    for lora in network.unet_loras + network.text_encoder_loras:
        gate = getattr(lora, "_last_gate", None)
        if gate is None:
            continue
        K_c = int(getattr(lora, "num_experts_content", K_c_default))
        if K_c <= 0:
            continue
        gate_c = gate[..., :K_c]
        gate_f = gate[..., K_c:]
        if gate_c.shape[-1] > 1:
            term_c = switch_balance(gate_c)
            total_c = term_c if total_c is None else total_c + term_c
            count_c += 1
        if gate_f.shape[-1] > 1:
            term_f = switch_balance(gate_f)
            total_f = term_f if total_f is None else total_f + term_f
            count_f += 1

    if total_c is None and total_f is None:
        return torch.tensor(0.0)
    w_c = float(getattr(network, "_balance_w_content", 0.0) or 0.0)
    w_f = float(getattr(network, "_balance_w_freq", 0.0) or 0.0)
    outer = float(getattr(network, "_balance_loss_weight", 0.0) or 0.0)
    out = torch.tensor(0.0)
    if total_c is not None and count_c > 0:
        out = out + outer * w_c * (total_c / count_c)
    if total_f is not None and count_f > 0:
        out = out + w_f * (total_f / count_f)
    return out


def get_router_entropy(network) -> Optional[float]:
    if getattr(network, "_use_chimera_hydra", False):
        cstats = get_chimera_router_stats(network)
        if not cstats:
            return None
        parts = [
            cstats[k] for k in ("content_entropy", "freq_entropy") if k in cstats
        ]
        if not parts:
            return None
        return sum(parts) / len(parts)
    stats = get_router_stats(network)
    return stats.get("entropy_mean") if stats else None


def get_router_stats(
    network,
) -> Dict[str, Union[float, List[float], List[List[float]], List[int]]]:
    if network._router_stats_cache is not None:
        return network._router_stats_cache

    gates: List[torch.Tensor] = []
    E_ref: Optional[int] = None
    for lora in network.unet_loras + network.text_encoder_loras:
        gate = getattr(lora, "_last_gate", None)
        if gate is None:
            continue
        E = gate.shape[-1]
        if E <= 1:
            continue
        if E_ref is None:
            E_ref = E
        elif E != E_ref:
            continue
        gates.append(gate)

    if not gates:
        return {}

    g = torch.stack(gates, dim=0)
    M, B, E = g.shape

    sigma = network._last_sigma
    num_buckets = int(network.cfg.num_sigma_buckets)
    want_per_bucket = sigma is not None and num_buckets > 1
    band_partition_active = bool(
        network.cfg.specialize_experts_by_sigma_buckets and num_buckets > 1
    )
    effective_E = (E // num_buckets) if band_partition_active else E
    norm = math.log(effective_E) if effective_E > 1 else 1.0

    p = g.float().clamp_min(1e-12)
    H_per_module = -(p * p.log()).sum(dim=-1).mean(dim=-1) / norm
    top2 = p.topk(2, dim=-1).values
    margin_per_module = (top2[..., 0] - top2[..., 1]).mean(dim=-1)
    expert_idx = g.argmax(dim=-1)
    usage_per_module = torch.nn.functional.one_hot(expert_idx, num_classes=E).to(
        g.dtype
    ).sum(dim=1) / float(B)

    H_per_module = H_per_module.detach()
    q_probs = torch.tensor(
        [0.05, 0.5, 0.95],
        device=H_per_module.device,
        dtype=H_per_module.dtype,
    )
    q = torch.quantile(H_per_module, q_probs)
    summary = torch.stack(
        [H_per_module.mean(), q[0], q[1], q[2], margin_per_module.detach().mean()]
    ).cpu()
    usage_mean = usage_per_module.detach().mean(dim=0).cpu().tolist()
    out: Dict[str, Union[float, List[float], List[List[float]], List[int]]] = {
        "entropy_mean": float(summary[0]),
        "entropy_p05": float(summary[1]),
        "entropy_p50": float(summary[2]),
        "entropy_p95": float(summary[3]),
        "margin_mean": float(summary[4]),
        "expert_usage": usage_mean,
    }

    if want_per_bucket and sigma is not None:
        thresholds = torch.linspace(0.0, 1.0, num_buckets + 1, device=sigma.device)[
            1:-1
        ]
        bucket_ids = torch.bucketize(sigma.float(), thresholds).clamp(
            0, num_buckets - 1
        )
        bucket_counts_t = torch.zeros(
            num_buckets, device=sigma.device, dtype=torch.long
        )
        bucket_counts_t.scatter_add_(
            0, bucket_ids, torch.ones_like(bucket_ids, dtype=torch.long)
        )
        bucket_ids_dev = bucket_ids.to(expert_idx.device)
        flat_idx = bucket_ids_dev[None, :] * E + expert_idx
        bu = torch.zeros(M, num_buckets * E, device=g.device, dtype=g.dtype)
        bu.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=g.dtype))
        bu = bu.view(M, num_buckets, E)
        bc = bucket_counts_t.to(g.dtype).clamp_min(1).view(1, num_buckets, 1)
        out["expert_usage_per_bucket"] = (bu / bc).detach().mean(dim=0).cpu().tolist()
        out["bucket_counts"] = bucket_counts_t.cpu().tolist()

    network._router_stats_cache = out
    return out


def get_chimera_router_stats(
    network,
) -> Dict[str, Union[float, List[float]]]:
    if not getattr(network, "_use_chimera_hydra", False):
        return {}
    if network._chimera_router_stats_cache is not None:
        return network._chimera_router_stats_cache

    out: Dict[str, Union[float, List[float]]] = {}
    K_c_default = int(getattr(network.cfg, "num_experts_content", 0))

    pi_c_list: List[torch.Tensor] = []
    K_c_ref: Optional[int] = None
    for lora in network.unet_loras + network.text_encoder_loras:
        gate = getattr(lora, "_last_gate", None)
        if gate is None:
            continue
        K_c = int(getattr(lora, "num_experts_content", K_c_default))
        if K_c <= 0:
            continue
        if K_c_ref is None:
            K_c_ref = K_c
        elif K_c != K_c_ref:
            continue
        pi_c_list.append(gate[..., :K_c])

    if pi_c_list and K_c_ref is not None and K_c_ref > 1:
        pi_c = torch.stack(pi_c_list, dim=0).float().clamp_min(1e-12)
        norm_c = math.log(K_c_ref)
        H_c_per_mod = -(pi_c * pi_c.log()).sum(dim=-1).mean(dim=-1) / norm_c
        top2_c = pi_c.topk(2, dim=-1).values
        margin_c_per_mod = (top2_c[..., 0] - top2_c[..., 1]).mean(dim=-1)
        usage_c = pi_c.mean(dim=(0, 1))
        summary_c = torch.stack(
            [H_c_per_mod.mean().detach(), margin_c_per_mod.mean().detach()]
        ).cpu()
        out["content_entropy"] = float(summary_c[0])
        out["content_margin"] = float(summary_c[1])
        out["content_usage"] = usage_c.detach().cpu().tolist()

    fr = getattr(network, "freq_router", None)
    pi_f = fr._last_gates if fr is not None else None
    if pi_f is not None and pi_f.dim() == 2 and pi_f.shape[-1] > 1:
        K_f = int(pi_f.shape[-1])
        pf = pi_f.float().clamp_min(1e-12)
        norm_f = math.log(K_f)
        H_f = (-(pf * pf.log()).sum(dim=-1).mean() / norm_f).detach()
        top2_f = pf.topk(2, dim=-1).values
        margin_f = (top2_f[..., 0] - top2_f[..., 1]).mean().detach()
        usage_f = pf.mean(dim=0).detach()
        summary_f = torch.stack([H_f, margin_f]).cpu()
        out["freq_entropy"] = float(summary_f[0])
        out["freq_margin"] = float(summary_f[1])
        out["freq_usage"] = usage_f.cpu().tolist()

    network._chimera_router_stats_cache = out
    return out


def capture_up_grad_stats(network) -> None:
    if not getattr(network, "_use_hydra", False):
        network._last_up_grad_stats = {}
        return

    use_tlora = bool(network.cfg.use_timestep_mask)
    min_rank = int(network.cfg.min_rank) if use_tlora else 0
    max_rank = int(network.cfg.lora_dim)
    min_rank = max(0, min(min_rank, max_rank))
    has_tlora_split = use_tlora and 0 < min_rank < max_rank

    up_grads: List[torch.Tensor] = []
    sp_grads: List[torch.Tensor] = []
    expert_band_ref: Optional[torch.Tensor] = None

    for lora in network.unet_loras + network.text_encoder_loras:
        up = getattr(lora, "lora_up_weight", None)
        sp = getattr(lora, "S_p", None)
        up_grad = up.grad if isinstance(up, torch.nn.Parameter) else None
        sp_grad = sp.grad if isinstance(sp, torch.nn.Parameter) else None
        if up_grad is not None:
            up_grads.append(up_grad.detach())
        if sp_grad is not None and sp_grad.dim() == 3:
            sp_grads.append(sp_grad.detach())
        if expert_band_ref is None:
            band = getattr(lora, "_expert_band", None)
            if band is not None:
                expert_band_ref = band.detach()

    if not up_grads and not sp_grads:
        network._last_up_grad_stats = {}
        return

    total_per_exp: Optional[torch.Tensor] = None
    below_per_exp: Optional[torch.Tensor] = None
    above_per_exp: Optional[torch.Tensor] = None
    sp_total_per_exp: Optional[torch.Tensor] = None
    device_ref: Optional[torch.device] = None

    if up_grads:
        big_up = torch.cat(up_grads, dim=1).float()
        sq_up = big_up.square()
        total_per_exp = sq_up.sum(dim=(1, 2))
        device_ref = total_per_exp.device
        if has_tlora_split:
            below_per_exp = sq_up[:, :, :min_rank].sum(dim=(1, 2))
            above_per_exp = sq_up[:, :, min_rank:].sum(dim=(1, 2))

    if sp_grads:
        big_sp = torch.stack(sp_grads, dim=0).float()
        sp_total_per_exp = big_sp.square().sum(dim=(0, 2, 3))
        if device_ref is None:
            device_ref = sp_total_per_exp.device

    out: Dict[str, object] = {
        "min_rank": [float(min_rank)],
        "num_buckets": [float(network.cfg.num_sigma_buckets)],
    }
    if total_per_exp is not None:
        out["total"] = total_per_exp
    if below_per_exp is not None and above_per_exp is not None:
        out["below"] = below_per_exp
        out["above"] = above_per_exp
    if sp_total_per_exp is not None:
        out["sp_total"] = sp_total_per_exp

    if (
        expert_band_ref is not None
        and bool(network.cfg.specialize_experts_by_sigma_buckets)
        and int(network.cfg.num_sigma_buckets) > 1
    ):
        B = int(network.cfg.num_sigma_buckets)
        band = expert_band_ref.to(device_ref)

        def _scatter_to_band(per_exp: torch.Tensor) -> torch.Tensor:
            buf = torch.zeros(B, device=per_exp.device, dtype=per_exp.dtype)
            buf.scatter_add_(0, band, per_exp)
            return buf

        if total_per_exp is not None:
            out["total_band"] = _scatter_to_band(total_per_exp)
        if below_per_exp is not None and above_per_exp is not None:
            out["below_band"] = _scatter_to_band(below_per_exp)
            out["above_band"] = _scatter_to_band(above_per_exp)
        if sp_total_per_exp is not None:
            out["sp_total_band"] = _scatter_to_band(sp_total_per_exp)

    network._last_up_grad_stats = out


def get_up_grad_stats(network) -> Dict[str, List[float]]:
    raw = network._last_up_grad_stats
    if not raw:
        return {}
    materialized: Dict[str, List[float]] = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            materialized[k] = v.detach().cpu().tolist()
        else:
            materialized[k] = list(v)  # type: ignore[arg-type]
    return materialized


def get_ortho_regularization(network) -> torch.Tensor:
    total_reg = torch.tensor(0.0, device=next(network.parameters()).device)
    count = 0
    for lora in network.text_encoder_loras + network.unet_loras:
        if hasattr(lora, "regularization"):
            p_reg, q_reg = lora.regularization()
            total_reg = total_reg + p_reg + q_reg
            count += 1
    for reft in network.text_encoder_refts + network.unet_refts:
        total_reg = total_reg + reft.regularization()
        count += 1
    return total_reg / max(count, 1)


def metrics(network, ctx: MetricContext) -> dict[str, float]:
    out: dict[str, float] = {}

    ortho_w = float(getattr(network, "_ortho_reg_weight", 0.0) or 0.0)
    if ortho_w > 0.0:
        v = get_ortho_regularization(network)
        if torch.is_tensor(v):
            v = v.detach().item()
        out["reg/ortho"] = float(v)
        out["reg/ortho_weighted"] = float(ortho_w * v)

    bal_w = float(getattr(network, "_balance_loss_weight", 0.0) or 0.0)
    if bal_w > 0.0:
        v = get_balance_loss(network)
        if torch.is_tensor(v):
            v = v.detach().item()
        out["reg/balance"] = float(v)
        out["reg/balance_weighted"] = float(bal_w * v)

    if not getattr(network, "_use_hydra", False):
        return out

    if getattr(network, "_use_chimera_hydra", False):
        cstats = get_chimera_router_stats(network)
        if cstats:
            if "content_entropy" in cstats:
                out["chimera/content_entropy"] = float(cstats["content_entropy"])
                out["chimera/content_margin"] = float(cstats["content_margin"])
                for i, v in enumerate(cstats.get("content_usage", [])):
                    out[f"chimera/content_usage/{i}"] = float(v)
            if "freq_entropy" in cstats:
                out["chimera/freq_entropy"] = float(cstats["freq_entropy"])
                out["chimera/freq_margin"] = float(cstats["freq_margin"])
                for i, v in enumerate(cstats.get("freq_usage", [])):
                    out[f"chimera/freq_usage/{i}"] = float(v)
    else:
        stats = get_router_stats(network)
        if stats:
            out["hydra/router_entropy"] = float(stats["entropy_mean"])
            out["hydra/router_entropy_p05"] = float(stats["entropy_p05"])
            out["hydra/router_entropy_p50"] = float(stats["entropy_p50"])
            out["hydra/router_entropy_p95"] = float(stats["entropy_p95"])
            out["hydra/router_margin"] = float(stats["margin_mean"])
            for i, v in enumerate(stats.get("expert_usage", [])):
                out[f"hydra/expert_usage/{i}"] = float(v)
            for b, row in enumerate(stats.get("expert_usage_per_bucket", [])):
                for i, v in enumerate(row):
                    out[f"hydra/expert_usage_b{b}/{i}"] = float(v)
            for b, c in enumerate(stats.get("bucket_counts", [])):
                out[f"hydra/bucket_count/{b}"] = float(c)

    up = get_up_grad_stats(network)
    if up:
        eps = 1e-12

        def _emit_per_expert(prefix: str, sq: list[float]) -> None:
            for i, v in enumerate(sq):
                out[f"hydra/up_grad/{prefix}/exp{i}"] = float(v) ** 0.5

        def _emit_per_band(prefix: str, sq: list[float]) -> None:
            for b, v in enumerate(sq):
                out[f"hydra/up_grad/{prefix}/band{b}"] = float(v) ** 0.5

        if "total" in up:
            _emit_per_expert("total", up["total"])
        if "below" in up and "above" in up:
            _emit_per_expert("below", up["below"])
            _emit_per_expert("above", up["above"])
            for i, (b_, a_) in enumerate(zip(up["below"], up["above"])):
                out[f"hydra/up_grad/above_below_ratio/exp{i}"] = float(
                    a_
                ) ** 0.5 / (float(b_) ** 0.5 + eps)
        if "sp_total" in up:
            _emit_per_expert("sp_total", up["sp_total"])
        if "total_band" in up:
            _emit_per_band("total", up["total_band"])
        if "below_band" in up and "above_band" in up:
            _emit_per_band("below", up["below_band"])
            _emit_per_band("above", up["above_band"])
            for b, (bv, av) in enumerate(zip(up["below_band"], up["above_band"])):
                out[f"hydra/up_grad/above_below_ratio/band{b}"] = float(
                    av
                ) ** 0.5 / (float(bv) ** 0.5 + eps)
        if "sp_total_band" in up:
            _emit_per_band("sp_total", up["sp_total_band"])

    if (
        network.global_router is not None
        and network.global_router._last_gates is not None
    ):
        gates = network.global_router._last_gates
        if gates.dim() == 2 and gates.shape[1] > 1:
            g = gates.float().clamp_min(1e-12)
            E = int(g.shape[-1])
            norm = math.log(E)
            H = -(g * g.log()).sum(dim=-1).mean() / norm
            top2 = g.topk(2, dim=-1).values
            margin = (top2[..., 0] - top2[..., 1]).mean()
            usage = g.mean(dim=0)
            summary = torch.stack([H.detach(), margin.detach()]).cpu()
            out["fera/router_entropy"] = float(summary[0])
            out["fera/router_margin"] = float(summary[1])
            for i, v in enumerate(usage.detach().cpu().tolist()):
                out[f"fera/expert_usage/{i}"] = float(v)

    return out
