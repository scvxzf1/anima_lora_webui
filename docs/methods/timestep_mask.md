# Timestep Rank Masking (T-LoRA)

Timestep-dependent rank masking for LoRA training. Effective rank varies with the denoising step — **full at high noise, reduced toward `min_rank` at low noise** (Anima convention: `t=0` pure noise, `t=1` clean).

> **For the structural walkthrough** (rank schedule math, mask application inside the LoRA bottleneck, training-only semantics, shared GPU-resident tensor), see **`docs/structure/timestep-mask.md`**. This doc is the usage / ops reference.

## Quick start

T-LoRA variants live in `configs/gui-methods/` (one file per variant, no toggle blocks):

```bash
make lora-gui GUI_PRESETS=tlora              # OrthoLoRA + timestep masking
make lora-gui GUI_PRESETS=tlora_ortho_reft   # OrthoLoRA + T-LoRA + ReFT stack
```

Or enable the T-LoRA block in `configs/methods/lora.toml` and run `make lora`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_timestep_mask` | false | Enable timestep rank masking |
| `min_rank` | 1 | Minimum active rank (floor at the clean end) |
| `alpha_rank_scale` | 1.0 | Power-law exponent (1.0 = linear, >1 = steeper toward noise end, <1 = flatter) |
| `network_dim` | — | Maximum rank (R_max), set by the method config |

## Compatibility

Timestep masking composes with every adapter module type. The mask is applied at the bottleneck (after down-projection), so it is orthogonal to the module's outer parameterization:

| Module | Where mask is applied |
|--------|----------------------|
| **LoRA** | After `lora_down`, before dropout and `lora_up` |
| **OrthoLoRA (Cayley)** | After `Q_eff` projection, multiplied with `lambda_layer` |
| **HydraLoRA** | After shared `lora_down`; per-expert `lora_up` heads unaffected |
| **ReFT** | Separate mask with its own `reft_dim` and floor of 1 |

`configs/gui-methods/tlora_ortho_reft.toml` and the default block in `configs/methods/lora.toml` stack LoRA + OrthoLoRA + T-LoRA + ReFT together.

## Configs

`configs/methods/lora.toml` (T-LoRA toggle block) — OrthoLoRA (Cayley) + timestep masking, rank 64:

```toml
use_ortho = true
use_timestep_mask = true
min_rank = 1
alpha_rank_scale = 1.0
network_dim = 64
```

## Implementation

| File | Role |
|------|------|
| `networks/lora_anima/routing_state.py` | `set_timestep_mask()` — computes rank, writes shared mask |
| `networks/lora_anima/routing_state.py` | `set_reft_timestep_mask()` — same for ReFT modules |
| `networks/lora_anima/routing_state.py` | `clear_timestep_mask()` — fills shared mask with ones (full rank) |
| `networks/lora_anima/network.py` | Facade forwarding the three helpers above |
| `networks/lora_modules/*` | Per-module training-only mask multiply in bottleneck |
| `library/training/forward/router_conditioning.py` | Calls set/clear each step after noise sampling |

## Training-only semantics

- Training (`is_train=True`): write the scheduled mask once per step.
- Validation / sample / inference: full rank. Modules gate the multiply with `self.training`; the conditioning hook clears on `is_train=False`; sample/validation also clear after `network.eval()`.
- Checkpoint metadata stamps `ss_use_timestep_mask` / `ss_min_rank` / `ss_alpha_rank_scale` (mask itself is not saved). Warm-start / inference load keeps T-LoRA off by design — continue training still takes the schedule from the active TOML / form, not from these stamps alone.
- Merge is supported for plain/Ortho T-LoRA because inference already runs full rank.
